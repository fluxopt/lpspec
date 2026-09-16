"""One build: every declaration into rows of the model frames a sink drains.

Declarations build one at a time and concatenate at the end; their rows are
independent. The two registries that fill *during* a build — the variable and
constraint label frames — are here because a declaration built later has to see
what earlier ones produced; everything attaching produced is frozen by contrast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, get_args

import numpy as np
import polars as pl
from math_spec import program

from lpspec.errors import DataError, null_bounds_message
from lpspec.relational import sinks
from lpspec.relational.collect import polars_engine
from lpspec.relational.engines.polars import coverage, labels
from lpspec.relational.engines.polars.compiler import PolarsCompiler
from lpspec.relational.engines.polars.fragments import TermFragment, absence_restrictions, join_on
from lpspec.relational.sinks.tables import SENSE

if TYPE_CHECKING:
    import numpy.typing as npt
    from math_spec.program import ObjectiveSense
    from polars._typing import MaintainOrderJoin

    from lpspec.relational.engines.polars.attaching import AttachedSources


#: The frames a sink reads, as schemas.
_COLS = ('lb', 'ub', 'vtype')
_OBJ = ('col', 'coeff')
_QUAD = ('col_l', 'col_r', 'coeff')
_ROWS = ('row', 'sense', 'rhs')
_MATRIX = ('row', 'col', 'coeff')
_QMATRIX = ('row', 'col_l', 'col_r', 'coeff')
_SOS = ('set', 'type', 'col', 'weight', 'big_m')

#: The dtype of each of those columns. ``vtype`` is an ``Enum`` over the
#: variable types the plan declares, so a type added upstream and not reaching
#: here fails where the column is built. ``col``, ``set`` and ``weight`` are
#: ``Int32``, the solver's own index width. A *label* stays ``Int64``: it is a
#: position in the full pre-mask coordinate product, which can pass 2^31 while
#: every survivor fits.
_DTYPES = {
    'col': pl.Int32, 'row': pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, 'coeff': pl.Float64,
    'sense': SENSE, 'vtype': pl.Enum(get_args(program.VariableDomain)),
    'set': pl.Int32, 'type': pl.UInt8, 'weight': pl.Int32, 'big_m': pl.Float64,
    'col_l': pl.Int32, 'col_r': pl.Int32,
}  # fmt: skip


@dataclass
class Measured:
    """What one build measured about itself, and a rebuild replaces wholesale.

    It outlives :class:`BuiltModel`: ``close()`` releases the frames and
    diagnostics still answer, so everything here is a count or a small frame
    rather than a read of the model. Most of it is taken as it is measured, so a
    build that raises still reports what it got to; the three sizes are written
    once the build has finished.
    """

    #: ``name -> (coordinates, rows)`` for each parameter attached short of the
    #: coordinates its dims reach.
    sparse: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: ``name -> rows not built``, because every term they had vanished.
    omitted: dict[str, int] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` coefficient magnitude, per constraint
    #: block, taken as each share is built.
    coefficients: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` bound magnitude, per variable block.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` right-hand-side magnitude, per constraint
    #: block, over the rows that survived.
    rhs: dict[str, tuple[float, float]] = field(default_factory=dict)
    objective_range: tuple[float, float] | None = None
    columns: int = 0
    rows: int = 0
    nonzeros: int = 0


@dataclass(frozen=True)
class BuiltModel:
    """One build's product: the frames a sink drains, and what reads them back.

    The compiler holds the same ``variables`` dict rather than a copy.
    """

    program: program.Program
    attached: AttachedSources
    compiler: PolarsCompiler
    #: One :class:`~lpspec.relational.engines.polars.labels.Labelled` per
    #: declaration, one map per label space: columns and rows are numbered
    #: independently, and a model may name a variable and a constraint alike.
    variables: dict[str, labels.Labelled]
    constraints: dict[str, labels.Labelled]

    cols: pl.DataFrame
    obj: pl.DataFrame
    #: The objective's quadratic part, one row per *unordered pair* of columns:
    #: ``coeff`` is the coefficient of ``x[col_l] · x[col_r]`` in the objective
    #: as written, and never half of it. Each sink converts into its own
    #: spelling (:class:`~lpspec.relational.sinks.tables.Tables`).
    quad: pl.DataFrame
    #: The quadratic part of every quadratic constraint row, one row per
    #: ``(row, unordered pair)``. Quadratic constraints are built last, so
    #: these rows are the **tail** of the label space and a sink takes them as
    #: a slice.
    qmatrix: pl.DataFrame
    rows: pl.DataFrame
    matrix: pl.DataFrame
    sos: pl.DataFrame
    matrix_starts: npt.NDArray[np.int64]

    column_count: int
    row_count: int
    objective_constant: float
    objective_sense: ObjectiveSense | None

    def tables(self) -> sinks.Tables:
        """What every sink reads, and no more."""
        return sinks.Tables(
            cols=self.cols,
            obj=self.obj,
            quad=self.quad,
            qmatrix=self.qmatrix,
            rows=self.rows,
            matrix=self.matrix,
            sos=self.sos,
            row_starts=self.matrix_starts,
            column_count=self.column_count,
            row_count=self.row_count,
            objective_sense=self.objective_sense,
            objective_constant=self.objective_constant,
        )


class Assembly:
    """One build in progress: the mutable half, discarded once it has frozen.

    Every counter here is one a declaration *advances* — a variable claims the
    next run of columns, a constraint the next run of rows — so they cannot be
    on the frozen product. :meth:`run` turns the lot into a
    :class:`BuiltModel`, and nothing outside this class writes to any of it.
    """

    def __init__(self, program: program.Program, attached: AttachedSources, measured: Measured) -> None:
        self.program = program
        self.attached = attached
        self.measured = measured
        self.variables: dict[str, labels.Labelled] = {}
        self.constraints: dict[str, labels.Labelled] = {}
        self.compiler = PolarsCompiler(program, attached, self.variables)
        self.n_cols = 0
        self.n_rows = 0
        #: How many special-ordered sets have been numbered. Sets are dense
        #: ``0..n-1`` across the model, like columns and rows, so a sink names
        #: one by its number and two builds agree on which.
        self.n_sets = 0
        self.quad: pl.DataFrame | None = None
        self.obj_const = 0.0
        self.obj_sense: ObjectiveSense | None = None

    def run(self) -> BuiltModel:
        """Build every declaration, then freeze what they produced.

        Quadratic constraints are built last, so their rows are a contiguous
        tail of the label space and every sink downstream takes them as a
        slice; the sort is stable, so file order survives inside each half.

        The matrix and ``rows`` leave in ``(row, col)`` order, as ``Tables``
        promises its sinks. The stack already has it — each share leaves
        sorted and owns the next run of rows — so the order is *checked* with
        one linear scan rather than sorted. :func:`_row_starts` reads the CSR
        index off that order, after which ``row`` is dropped from the matrix.
        """
        cols = [self._build_variable(name, v) for name, v in self.program.variables.items()]
        sets = [self._build_sos(s, self.program.variable(s.variable)) for s in self.program.sos.values()]
        ordered = sorted(self.program.constraints.items(), key=lambda item: declares_quadratic(item[1]))  # pyrefly: ignore[implicit-any-lambda]  — a (name, declaration) pair
        built = [self._build_constraint(name, c) for name, c in ordered]
        objective = self._build_objective(self.program.objective)

        stacked = labels.in_position_order(_stack([m for _, m, _ in built if m is not None], _MATRIX), 'row')
        matrix_starts = _row_starts(stacked, self.n_rows)
        matrix = stacked.select('col', 'coeff').rechunk()

        self.measured.columns = self.n_cols
        self.measured.rows = self.n_rows
        self.measured.nonzeros = matrix.height
        return BuiltModel(
            program=self.program,
            attached=self.attached,
            compiler=self.compiler,
            variables=self.variables,
            constraints=self.constraints,
            cols=_stack(cols, _COLS),
            obj=_stack([objective] if objective is not None else [], _OBJ),
            quad=_stack([] if self.quad is None else [self.quad], _QUAD),
            qmatrix=labels.in_position_order(_stack([q for _, _, q in built if q is not None], _QMATRIX), 'row'),
            rows=labels.in_position_order(_stack([r for r, _, _ in built], _ROWS), 'row'),
            matrix=matrix,
            sos=_stack(sets, _SOS),
            matrix_starts=matrix_starts,
            column_count=self.n_cols,
            row_count=self.n_rows,
            objective_constant=self.obj_const,
            objective_sense=self.obj_sense,
        )

    def _matrix_share(
        self, pieces: list[pl.LazyFrame], name: str, *expressions: program.ExpressionNode
    ) -> tuple[pl.DataFrame, pl.Series]:
        """One constraint's share: in ``(row, col)`` order, repeated cells summed.

        Nothing runs unconditionally except three linear probes — the null
        count, whether the stack arrives in order, whether any cell repeats —
        so the sort and the aggregate run only when a probe says they would
        change something. The share is rechunked first: a streaming collect
        returns morsels as chunks, and ``shift(1)`` crosses chunk boundaries
        that ``is_sorted`` does not.

        Zeros go before any of that, so the probes read them and the sort
        orders them no longer — pruned behind a probe too, so a share with
        nothing to drop skips the rechunk. A cancelling pair survives to the
        aggregate and only becomes a zero there, so the prune runs again on
        the path that aggregated, and only on it.

        Returns:
            The share, and the rows that had *any* term — read off the frame
            before a prune takes the answer away: a row whose every
            coefficient is zero owns no entries and is not thereby a row with
            no terms.
        """
        stacked = pl.concat(pieces).collect(engine=polars_engine()).rechunk()
        coverage.refuse_null_coefficients(stacked, name, *expressions)
        pruned = _pruned(stacked)
        term_rows = stacked.get_column('row').unique() if pruned.height != stacked.height else None
        stacked = pruned
        row, col = pl.col('row'), pl.col('col')
        tied, ahead = row == row.shift(1), row > row.shift(1)
        repeat = tied & (col == col.shift(1))
        probes = stacked.select(
            (ahead | (tied & (col >= col.shift(1)))).all().alias('#ordered'),
            repeat.any().alias('#repeated'),
        )
        ordered, repeated = probes.row(0)
        if not ordered:
            stacked = stacked.sort('row', 'col')
            repeated = stacked.select(repeat.any()).item()
        if not repeated:
            return stacked, term_rows if term_rows is not None else _ordered_rows(stacked)
        aggregated = (
            stacked.lazy()
            .group_by('row', 'col')
            .agg(pl.col('coeff').sum())
            .sort('row', 'col')
            .collect(engine=polars_engine())
        )
        return _pruned(aggregated), term_rows if term_rows is not None else _ordered_rows(aggregated)

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------

    def _build_variable(self, name: str, v: program.VariableDeclaration) -> pl.DataFrame:
        """One variable's labelled frame, and its share of ``cols``.

        The share leaves in label order, ``cols`` carrying no ``col`` of its
        own: a row's *position* is its solver column index. The bounds joins
        usually keep that order, so it is verified with one linear scan and
        re-established only when a join lost it.

        Only the label and the two bounds are collected, keeping the dim
        columns and joined parameters inside the lazy pipeline. A null bound
        is a bound parameter with no value where the variable has a column; it
        is probed on the two columns and counted only on the model that has
        one.
        """
        start = self.n_cols
        labelled = labels.frame(self.compiler, v.dims, v.where, 'var_label', start)
        self.n_cols = start + labelled.height
        self.variables[name] = labels.Labelled(labelled.lazy(), start, labelled.height)

        bounded = labels.in_position_order(
            self.compiler.bounds(labelled.lazy(), name, v)
            .select('var_label', pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64))
            .collect(engine=polars_engine()),
            'var_label',
        )
        cols = bounded.select('lb', 'ub', pl.lit(v.domain, dtype=_DTYPES['vtype']).alias('vtype'))

        if bounded.get_column('lb').null_count() or bounded.get_column('ub').null_count():
            bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
            raise DataError(null_bounds_message(name, bad))

        if (spread := _magnitude_range(bounded, 'lb', 'ub')) is not None:
            self.measured.bounds[name] = spread
        return cols

    def _build_sos(self, s: program.SosDeclaration, v: program.VariableDeclaration) -> pl.DataFrame:
        """One declaration's sets as ``(set, type, col, weight, big_m)``, over *v*.

        Builds no column and no row: a set names columns the variable already
        made, so it runs after every variable and before any constraint.

        **A set and a weight are the two halves of a coordinate's row-major
        position**, split at the ``over`` dim, by two divisions rather than by
        reading a dim's ordinal per member. A position is the label itself
        where the variable dropped nothing; a masked one reads the ordinals
        and renumbers the sets densely. A member's weight is its
        coordinate's position in the declared order, so a masked-out
        coordinate leaves its neighbours adjacent rather than leaving a hole.

        **The stream leaves grouped by set and ascending in weight**, which is
        what lets a sink read a set's edges off the neighbouring row. The
        order is verified and the sort runs only where members interleave — on
        **both** columns, because a sort that reordered ties would be a set
        whose members arrive out of weight order.

        ``big_m`` rides along per member, at ``inf`` where the block declared
        none — the one thing here no sink taking a set natively reads.
        """
        held = self.variables[s.variable]
        cardinality = self.compiler.data.cardinality
        stride = math.prod(cardinality[d] for d in v.dims[v.dims.index(s.over) + 1 :])
        span = cardinality[s.over] * stride

        if v.where is None:
            frame = pl.select(pl.int_range(held.height, dtype=pl.Int64).alias('#position')).lazy()
            place, col = pl.col('#position'), (pl.col('#position') + held.start)
        else:
            frame = held.frame
            place, col = self.compiler.row_major(v.dims, self.compiler.ordinal_of), pl.col('var_label')
        placed = frame.select(
            ((place // span) * stride + place % stride).alias('#set position'),
            ((place // stride) % cardinality[s.over] + 1).cast(_DTYPES['weight']).alias('weight'),
            col.cast(_DTYPES['col']).alias('col'),
        ).collect(engine=polars_engine())

        position = pl.col('#set position')
        grouped = placed if placed.get_column('#set position').is_sorted() else placed.sort('#set position', 'weight')
        dense = position if v.where is None else (position != position.shift(1)).fill_null(True).cum_sum() - 1
        built = grouped.select(
            (dense + self.n_sets).cast(_DTYPES['set']).alias('set'),
            pl.lit(s.sos_type, dtype=_DTYPES['type']).alias('type'),
            'col',
            'weight',
            pl.lit(float('inf') if s.big_m is None else s.big_m, dtype=_DTYPES['big_m']).alias('big_m'),
        )
        if built.height:
            self.n_sets = built.item(-1, 'set') + 1
        return built

    def _build_constraint(
        self, name: str, c: program.ConstraintDeclaration
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
        """One constraint as its ``rows``, its share of the matrix, and its quadratic share.

        Terms normalise to the left, constants to the right. Whether the data
        is there where the row reads it is :mod:`coverage`'s to answer, in the
        order its table gives: the two questions an aggregation would hide are
        asked of the pieces and the parameters first, and the rows pass then
        carries the third.

        Duplicates from ``Sum`` and ``GroupSum`` — which project rather than
        aggregate — and from ``x + 2 * x`` collapse in :meth:`_matrix_share`'s
        terminal aggregate, read off the data rather than reasoned from how
        the fragments were reshaped.

        The labelled frame is kept for the dual read-back, and its block
        narrows when rows go termless: the run of labels a declaration owns is
        what survived, not what it declared. A purely quadratic row has no
        linear entries at all, so what decides whether a row is built is
        whether *either* matrix has a term.
        """
        quadratic = declares_quadratic(c)
        lhs = self.compiler.expression(c.lhs, f"constraint '{name}' lhs", quadratic=quadratic)
        rhs = self.compiler.expression(c.rhs, f"constraint '{name}' rhs", quadratic=quadratic)
        terms = [(p, 1.0) for p in lhs.terms] + [(p, -1.0) for p in rhs.terms]
        quads = [(p, 1.0) for p in lhs.quads] + [(p, -1.0) for p in rhs.quads]
        consts = [(p, 1.0) for p in rhs.consts] + [(p, -1.0) for p in lhs.consts]
        restrictions = absence_restrictions([p for p, _ in (*terms, *quads)])
        start = self.n_rows
        declared = labels.declared_height(self.compiler, c.dims, c.where) if restrictions else None
        labelled = labels.frame(self.compiler, c.dims, c.where, 'row', start, restrictions)
        if declared is not None and declared > labelled.height:
            self.measured.omitted[name] = self.measured.omitted.get(name, 0) + declared - labelled.height
        self.n_rows = start + labelled.height
        frame = labelled.lazy()
        self.constraints[name] = labels.Labelled(frame, start, labelled.height)

        subject = f"constraint '{name}'"
        pieces = [p for p, _ in consts]
        coverage.refuse_null_constants(
            coverage.narrowed_to_rows(frame, pieces), program.divisor_parameters(c.lhs, c.rhs), subject
        )
        coverage.refuse_short_constants(self.compiler, frame, pieces, c, subject, self.measured.sparse)
        rows = coverage.constant_side(self.compiler, frame, consts, c, subject)

        if not terms and not quads:
            none = pl.Series('row', [], dtype=_DTYPES['row'])
            rows, _, self.n_rows = self._drop_termless_rows(name, rows, _stack([], _MATRIX), none, start)
            return rows, None, None

        pieces = []
        carried_order: MaintainOrderJoin | None = 'left_right' if len(terms) == 1 else None
        for p, sign in terms:
            placed = join_on(frame, p.frame, p.dims, 'inner', maintain_order=carried_order)
            pieces.append(
                placed.select(
                    'row',
                    pl.col('var_label').cast(_DTYPES['col']).alias('col'),
                    (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
                )
            )
        matrix, term_rows = (
            self._matrix_share(pieces, subject, c.lhs, c.rhs)
            if pieces
            else (_stack([], _MATRIX), pl.Series('row', [], dtype=_DTYPES['row']))
        )
        qmatrix = self._quadratic_share(frame, quads, name, c)
        if qmatrix is not None:
            term_rows = pl.concat([term_rows, qmatrix.get_column('row').unique()]).unique()
        rows, matrix, self.n_rows = self._drop_termless_rows(name, rows, matrix, term_rows, start)
        spread = _magnitude_range(matrix, 'coeff')
        if spread is not None:
            self.measured.coefficients[name] = spread
        if (sides := _magnitude_range(rows, 'rhs')) is not None:
            self.measured.rhs[name] = sides
        if qmatrix is not None:
            qmatrix = qmatrix.filter(pl.col('row').is_in(rows.get_column('row')))
        return rows, matrix, qmatrix

    def _quadratic_share(
        self, frame: pl.LazyFrame, quads: list[tuple[TermFragment, float]], name: str, c: program.ConstraintDeclaration
    ) -> pl.DataFrame | None:
        """One constraint's quadratic entries as ``(row, col_l, col_r, coeff)``.

        It sorts and aggregates unconditionally, where :meth:`_matrix_share`
        probes first. Pairs are ordered by column index for
        :meth:`_objective_quadratic`'s reason.
        """
        if not quads:
            return None
        pieces = [
            join_on(frame, p.frame, p.dims, 'inner').select(
                'row',
                *_ordered_pair(),
                (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
            )
            for p, sign in quads
        ]
        stacked = pl.concat(pieces).collect(engine=polars_engine())
        coverage.refuse_null_coefficients(stacked, f"constraint '{name}'", c.lhs, c.rhs)
        return _without_zeros(
            stacked.lazy()
            .group_by('row', 'col_l', 'col_r')
            .agg(pl.col('coeff').sum())
            .sort('row', 'col_l', 'col_r')
            .collect(engine=polars_engine())
        )

    def _drop_termless_rows(
        self, name: str, rows: pl.DataFrame, matrix: pl.DataFrame, kept: pl.Series, start: int
    ) -> tuple[pl.DataFrame, pl.DataFrame, int]:
        """Rows that kept no variable term are not built, and the block closes up.

        A row with no variables is not a constraint — it asserts something
        about constants, which the solver cannot act on. Three provenances
        reach that shape — an absent variable, an empty reduction, a missing
        coefficient — and all three drop the row.

        *kept* is the row set the share had terms for, which is
        :meth:`_matrix_share`'s to answer: the share it returns has been pruned
        of zero coefficients, so a row missing from it may have had every term
        and every one of them zero. That row stays — ``0 >= 10`` is infeasible
        — where a row that never had a term goes.

        Labels are dense and the dual read-back reads a block by position, so a
        dropped row may not leave a gap: survivors renumber from *start* and
        the row counter rewinds.
        """
        if kept.len() == rows.height:
            return rows, matrix, start + rows.height

        surviving = rows.filter(pl.col('row').is_in(kept)).sort('row')
        renumber = surviving.select('row').with_row_index('__new__', offset=start)
        self.measured.omitted[name] = rows.height - surviving.height
        remap = dict(zip(renumber.get_column('row'), renumber.get_column('__new__'), strict=True))
        rows = surviving.with_columns(pl.col('row').replace_strict(remap))
        matrix = matrix.with_columns(pl.col('row').replace_strict(remap))
        kept_frame = (
            self.constraints[name]
            .frame.filter(pl.col('row').is_in(kept))
            .with_columns(pl.col('row').replace_strict(remap))
        )
        self.constraints[name] = labels.Labelled(kept_frame, start, surviving.height)
        return rows, matrix, start + surviving.height

    def _build_objective(self, o: program.ObjectiveDeclaration | None) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        ``None`` in is the file that declares no objective at all, and it takes
        the same path out: the sense stays ``min`` and the constant ``0``, so
        the sink is handed a zero objective and answers whether the constraints
        can be met.

        This projection drops the dims, so a dim that arrived by broadcast puts
        several rows on one column and their **sum** is the coefficient — the
        hand-off scatters with ``dense[at] = values``, which keeps the *last*
        write. The aggregate runs only when a column repeats, probed by
        :func:`_repeats_a_label`: the stack arrives unordered, so adjacency
        proves nothing.
        """
        if o is None:
            return None
        comp = self.compiler.expression(o.expression, 'objective', quadratic=True)
        for p in comp.consts:
            assert not p.dims, (
                f'objective constant part has dims {list(p.dims)} — the language refuses a '
                f'variable-free part of an objective that carries any'
            )
            self.obj_const += p.frame.select(pl.col('cval').sum()).collect().item() or 0.0
        self.obj_sense = o.sense
        self.quad = self._objective_quadratic(comp.quads, o.expression)
        if not comp.terms:
            return None
        pieces = [
            p.frame.select(pl.col('var_label').cast(_DTYPES['col']).alias('col'), pl.col('coeff')) for p in comp.terms
        ]
        stacked = pl.concat(pieces).collect(engine=polars_engine())
        coverage.refuse_null_coefficients(stacked, 'objective', o.expression)
        if _repeats_a_label(stacked.get_column('col'), self.n_cols):
            stacked = stacked.lazy().group_by('col').agg(pl.col('coeff').sum()).collect(engine=polars_engine())
        objective = _without_zeros(stacked)
        self.measured.objective_range = _magnitude_range(objective, 'coeff')
        return objective

    def _objective_quadratic(
        self, quads: tuple[TermFragment, ...], expression: program.ExpressionNode
    ) -> pl.DataFrame | None:
        r"""The objective's quadratic part as ``(col_l, col_r, coeff)``, or ``None``.

        **One row per unordered pair**, at the coefficient the file wrote:
        ``coeff · x[col_l] · x[col_r]``, whole and not halved. Each sink spells
        that differently — a Hessian is :math:`\frac12 x^\top Q x`, the LP
        section is divided by two, Gurobi takes :math:`x^\top Q x` — so what
        leaves here is the algebra and the conversion is theirs. The aggregate
        runs only when a pair repeats, probed like the linear half.

        **It leaves sorted, and that is a contract**:
        :attr:`~lpspec.relational.sinks.tables.Tables.structure` hashes it, and
        the join hands pairs back in whatever order the data made.
        """
        if not quads:
            return None
        pieces = [p.frame.select(*_ordered_pair(), pl.col('coeff')) for p in quads]
        stacked = pl.concat(pieces).collect(engine=polars_engine())
        coverage.refuse_null_coefficients(stacked, 'objective', expression)
        if stacked.select(pl.struct('col_l', 'col_r').n_unique()).item() != stacked.height:
            stacked = (
                stacked.lazy().group_by('col_l', 'col_r').agg(pl.col('coeff').sum()).collect(engine=polars_engine())
            )
        return _without_zeros(stacked.sort('col_l', 'col_r'))


def short_parameters(program: program.Program, attached: AttachedSources) -> dict[str, tuple[int, int]]:
    """Which parameters arrived short, and by how much: ``name -> (reach, rows)``.

    Arithmetic over two dicts attaching already filled — a dimension's height
    and a parameter's. The door has refused duplicates and strangers, so the
    height *is* the number of coordinates covered.
    """
    short: dict[str, tuple[int, int]] = {}
    for name, p in program.parameters.items():
        if not p.dims:
            continue
        reach = math.prod(attached.cardinality[d] for d in p.dims)
        rows = attached.parameter_rows[name]
        if rows < reach:
            short[name] = (reach, rows)
    return short


def declares_quadratic(c: program.ConstraintDeclaration) -> bool:
    """Whether constraint *c* multiplies two variable-carrying operands, either side."""
    return program.is_quadratic(c.lhs) or program.is_quadratic(c.rhs)


def _ordered_pair() -> tuple[pl.Expr, pl.Expr]:
    """A quadratic pair canonicalised by column index, so ``x·y`` and ``y·x`` land in one row."""
    return (
        pl.min_horizontal('var_label', 'var_label_2').cast(_DTYPES['col_l']).alias('col_l'),
        pl.max_horizontal('var_label', 'var_label_2').cast(_DTYPES['col_r']).alias('col_r'),
    )


def _magnitude_range(frame: pl.DataFrame, *columns: str) -> tuple[float, float] | None:
    """The smallest and largest magnitude across *columns*, or ``None`` where none has one.

    Magnitudes rather than signed extremes: a row scaled by ``-1e9`` is as
    badly scaled as one scaled by ``1e9``. Zero and infinity are dropped,
    matching the ``Bound`` and ``RHS`` lines a solver prints, which exclude the
    same two.

    Each sign is reduced where it lies, so ``|x|`` is never built. Both signs
    are asked because the smallest magnitude can be interior to either.
    """
    sides: list[pl.Expr] = []
    for i, column in enumerate(columns):
        value = pl.col(column)
        finite, up, down = value.is_finite(), value > 0, value < 0
        sides += [
            value.filter(finite & up).min().alias(f'#low+{i}'),
            value.filter(finite & up).max().alias(f'#high+{i}'),
            value.filter(finite & down).max().alias(f'#low-{i}'),
            value.filter(finite & down).min().alias(f'#high-{i}'),
        ]
    answered = frame.select(sides).row(0, named=True)
    lows = [abs(bound) for name, bound in answered.items() if name.startswith('#low') and bound is not None]
    highs = [abs(bound) for name, bound in answered.items() if name.startswith('#high') and bound is not None]
    return (min(lows), max(highs)) if lows else None


def _without_zeros(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* with the entries that cannot reach the answer removed.

    A coefficient of exactly zero states that a variable is not in a row, which
    is what an absent row already states.

    **A pruned share can no longer say which rows had terms**, and a row whose
    every coefficient is zero still asserts something — ``0 >= 10`` is
    infeasible — so :meth:`Assembly._matrix_share` reads that row set off each
    frame before pruning it. Nulls cannot be here: a null coefficient is an
    undefined divisor, refused before this runs.
    """
    return matrix.filter(pl.col('coeff') != 0)


def _ordered_rows(matrix: pl.DataFrame) -> pl.Series:
    """The distinct ``row`` labels of a matrix already ordered by ``row``.

    Only ever called where the caller has *established* that order — the
    ``#ordered`` probe returned true, or the sort ran, or the aggregate ended
    on ``sort('row', 'col')``.

    ``set_sorted`` is an assertion, not a check: on a column that is not
    ascending it returns whichever labels the walk happens to see, which is a
    model missing rows rather than an error. A caller that moves the probe, or
    reaches here on a path that never ran one, breaks this silently — which is
    why every call site is inside the branch that just proved it.
    """
    return matrix.get_column('row').set_sorted().unique()


def _repeats_a_label(labels: pl.Series, count: int) -> bool:
    """Whether any of *labels* occurs twice, over a dense ``0..count-1`` space.

    Labels are the solver's own indices, so they index a scratch bitmap
    directly.

    *count* must exceed every label — it is the declaration counter the labels
    were drawn from, so a caller passing a stale one indexes out of bounds and
    raises rather than reporting a wrong answer.
    """
    seen = np.zeros(count, dtype=bool)
    seen[labels.to_numpy()] = True
    return int(np.count_nonzero(seen)) != labels.len()


def _pruned(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* without its zeros — unchanged, and not rechunked, when it has none.

    Filtering leaves a chunked frame that the ``shift(1)`` probes downstream
    read across every boundary, so a share with no zero to drop is returned as
    it is.
    """
    if not matrix.select(pl.col('coeff').eq(0).any()).item():
        return matrix
    return _without_zeros(matrix).rechunk()


def _row_starts(ordered: pl.DataFrame, row_count: int) -> npt.NDArray[np.int64]:
    """Each row's first entry in the row-ordered *ordered* — CSR's own index.

    Run-length, scatter, cumulative sum. *ordered* must ascend in ``row``: a
    row whose entries arrived in two runs would have the first run overwritten
    and the spans silently wrong.
    """
    runs = ordered['row'].rle()
    starts = np.zeros(row_count + 1, dtype=np.int64)
    starts[runs.struct.field('value').to_numpy() + 1] = runs.struct.field('len').to_numpy()
    return np.cumsum(starts, out=starts)


def _stack(frames: list[pl.DataFrame], columns: tuple[str, ...]) -> pl.DataFrame:
    """Concatenate *frames*, or an empty frame of *columns* when there are none."""
    if frames:
        return pl.concat(frames)
    return pl.DataFrame(schema={name: _DTYPES[name] for name in columns})
