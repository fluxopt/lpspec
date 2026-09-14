"""What a caller reads back — a solve's :class:`Result`, a build's :class:`Diagnostics`.

The objects ``lps.solve`` and ``model.diagnostics()`` hand back, so they are
the pieces of this subpackage a reader meets without going looking. A
:class:`Result` holds one finished frame per declaration, its values already
laid out over the build's coordinates, so no reader ever goes back to the
engine.

Named for linopy's envelope (``Result`` = status + solution + report).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field, replace
from datetime import datetime  # noqa: TC003  — a Record annotation this module writes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from lpspec.errors import (
    LpspecError,
    NoSolutionError,
    already_readable_message,
    no_model_behind_this_answer_message,
    unknown_name_message,
)
from lpspec.relational.parquet import (
    RECORD_FILE,
    RECORD_SCHEMA,
    Metrics,
    Record,
    clear_the_answer,
    reader_kind,
    write_format,
    write_reasons,
    write_whole,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd
    import polars as pl
    import xarray as xr

    from lpspec.relational.status import SolveStatus


#: How much of the session a solve keeps, as a request to
#: :meth:`lpspec.api.Model.solve` and as the report in
#: :attr:`Result.kept`. The two things a session holds — the solver with the
#: model on it, and the work that solver did — can only be dropped in that
#: order: there is no carrying on from a solver that was closed, so the fourth
#: combination does not exist.
Keep = Literal['nothing', 'solver', 'progress']

#: What each word keeps, in the order of how much that is. About provenance
#: and not mechanism: whether *progress* is a basis, an incumbent or a sink's
#: own notion is the sink's business.
KEEPS: Mapping[Keep, str] = {
    'nothing': 'the model is handed to a fresh solver, which has nothing to begin from',
    'solver': 'the solver already holding the model is reused, and the work the last solve did is discarded',
    'progress': 'the solver is reused and carries on from where the last solve got to',
}


def unknown_keep_message(keep: object) -> str:
    """The message for a *keep* outside the three."""
    options = '\n'.join(f'  {name}: {what}' for name, what in KEEPS.items())
    return f'unknown keep {keep!r}. A solve may keep:\n{options}'


#: What the bridges out say when the environment cannot serve them, ``{module}``
#: being pandas or xarray.
_NEEDS_THE_EXTRA = (
    '{module} ships with the [linopy] extra rather than with the engine, so this build cannot bridge out '
    'to it: pip install "lpspec[linopy]". A result needs nothing added to be read as it stands — primal() '
    'and dual() return polars frames, and save() writes one file per declaration.'
)


def tidy_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    """A tidy polars frame as pandas, column by column.

    Raises:
        ModuleNotFoundError: On an install without pandas, naming the extra.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_NEEDS_THE_EXTRA.format(module='pandas')) from exc

    return pd.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def tidy_to_dataarray(frame: pd.DataFrame, name: str) -> xr.DataArray:
    """The same, labelled by its non-``value`` columns.

    A scalar declaration has none and comes back 0-dimensional.

    Probed rather than imported: an ``import xarray`` anywhere under
    ``relational/`` is what hard rule 2 forbids (``tests/test_architecture.py``).

    Raises:
        ModuleNotFoundError: On an install without xarray, naming the extra.
    """
    if importlib.util.find_spec('xarray') is None:
        raise ModuleNotFoundError(_NEEDS_THE_EXTRA.format(module='xarray'))

    dims = [column for column in frame.columns if column != 'value']
    if not dims:
        return frame['value'].to_xarray().rename(name)
    return frame.set_index(dims).to_xarray()['value'].rename(name)


def tidy_to_dataset(names: Sequence[str], one: Callable[[str], xr.DataArray]) -> xr.Dataset:
    """*names* as one dataset, each array built by *one*."""
    first, *rest = names
    dataset = one(first).to_dataset(name=first)
    for name in rest:
        dataset[name] = one(name)
    return dataset


def _number(value: float, *, sign: bool = False) -> str:
    """*value* as the shortest string that reads back as itself.

    Shortest round-trip, never rounded. A trailing ``.0`` is dropped, as linopy
    prints ``+50``.
    """
    text = repr(float(value))
    text = text.removesuffix('.0')
    return f'+{text}' if sign and not text.startswith('-') else text


def _bracket(labels: str) -> str:
    """``[1, wind]``, or nothing at all for a declaration over no dims.

    A scalar is ``z``, not ``z[]`` — linopy's spelling.
    """
    return f'[{labels}]' if labels else ''


@dataclass(frozen=True)
class ConstraintRow:
    """One built constraint row, spelled back out — what :meth:`~lpspec.api.Model.row` returns.

    The row a model actually built at one coordinate: every term with its
    coefficient, and the comparison and right-hand side it was built against.
    Read off the built model, so it needs no solve — and it is the *built*
    row, after ``where`` masking, after any term whose variable was absent
    dropped out, and after a coefficient the data made exactly zero stopped
    being a term at all
    (:func:`~lpspec.relational.engines.polars.assembly._without_zeros`). Those
    three are why a row can be shorter than the file suggests, and why reading
    one is worth it when a model says something other than what its author
    wrote.

    Printing it gives the row as one line of math, which is what reading a row
    usually means; :attr:`terms` is the same content as a frame, for the row
    too wide to read and for anything that filters or joins.

    Attributes:
        name: The constraint this row belongs to.
        coordinate: Where in that declaration it sits.
        terms: ``(variable, coordinate, coefficient)``, one row per term, in
            the solver's own column order. ``coordinate`` is the term's labels
            in its variable's dim order — what goes in the brackets — rendered
            rather than spread across dim columns, since two terms of one row
            may come from variables with different dims and so cannot share
            them.
        sense: ``<=``, ``>=`` or ``==``.
        rhs: What the left-hand side is compared against.
    """

    name: str
    coordinate: Mapping[str, object]
    terms: pl.DataFrame
    sense: str
    rhs: float

    #: How many terms a line spells out before it summarises instead.
    display_terms = 12

    def __str__(self) -> str:
        """The row as one line: ``balance[snapshot=1]: +1 p[…] +50 p[…] >= 60``.

        linopy's shape for the same job. A row wider than :attr:`display_terms`
        summarises rather than truncating.
        """
        return f'{self.name}{_bracket(self._where())}: {self._body()} {self.sense} {_number(self.rhs)}'

    #: The line, not the field-by-field dataclass dump.
    __repr__ = __str__

    def _where(self) -> str:
        """``snapshot=1, g=gas`` — the coordinate, in the declaration's dim order."""
        return ', '.join(f'{dim}={label}' for dim, label in self.coordinate.items())

    def _body(self) -> str:
        """The terms, spelled out or summarised — the part that depends on width."""
        if not self.terms.height:
            return '(no terms)'
        if self.terms.height <= self.display_terms:
            return ' '.join(
                f'{_number(coefficient, sign=True)} {variable}{_bracket(coordinate)}'
                for variable, coordinate, coefficient in self.terms.iter_rows()
            )
        return f'{self.terms.height} terms — {self._per_declaration()}'

    def _per_declaration(self) -> str:
        """``p: 300 (|coef| 1…50)`` per variable, in the row's own term order.

        The two things a row too wide to read shows: how much of it each
        declaration contributes, and whether its coefficients span an order of
        magnitude.
        """
        import polars as pl

        grouped = self.terms.group_by('variable', maintain_order=True).agg(
            pl.len().alias('terms'),
            pl.col('coefficient').abs().min().alias('low'),
            pl.col('coefficient').abs().max().alias('high'),
        )
        return ', '.join(
            f'{variable}: {terms} (|coef| {_number(low)})'
            if low == high
            else f'{variable}: {terms} (|coef| {_number(low)}…{_number(high)})'
            for variable, terms, low, high in grouped.iter_rows()
        )


@dataclass(frozen=True)
class Diagnostics:
    """What a build and its solves did that the answer does not show.

    Advisory, all of it: no answer depends on any field. Read them when a loop
    is slower or smaller than it should be.
    """

    #: The shape the build produced: columns, rows, and matrix entries. The
    #: thing to report when a model is bigger than its author expected — a
    #: broadcast that multiplied rows shows up here first.
    columns: int
    rows: int
    nonzeros: int

    #: What the **last solve's sink** had to add on top of those to take the
    #: model, and zero for every sink that took it as built. A sink with no
    #: SOS concept is handed the sets as binaries and linking rows
    #: (:mod:`lpspec.relational.sinks.sos`). Zero until something has been
    #: solved: a *writer* is handed the model as built, and reports nothing.
    added_columns: int
    added_rows: int

    #: ``(constraint, rows_not_built)`` — every declared row that did not reach
    #: the solver (the absence rules), by either route: one emptied of all its
    #: terms, and one a **propagated absence** deleted while its other terms were
    #: still live. Empty for a model whose every declared row was built — a
    #: recurrence's first coordinate counting as a row it declared and did not
    #: get, so a ``shift`` against the horizon's edge reports here and is the
    #: boundary rather than a fault. Counts rather than coordinates: the label
    #: of an unbuilt row does not exist.
    omissions: pl.DataFrame

    #: ``(parameter, coordinates, rows, missing)`` — one row per parameter whose
    #: source is short of the coordinates its dims reach, in declaration order,
    #: and **empty where every one is complete**. Sparsity is the ordinary case
    #: here — absence is how a model masks — so this reports it rather than
    #: judging it: what a missing row means is the absence rules', and whether
    #: it was meant is the caller's to say.
    #:
    #: A parameter over no dims has one coordinate and attaching already refuses
    #: a source that does not carry exactly one row for it, so it is never
    #: here.
    sparse_parameters: pl.DataFrame

    #: ``(constraint, smallest, largest)`` — the coefficient **magnitudes** each
    #: constraint block put in the matrix, one row per block that kept a term,
    #: in build order. A solver's own ``Matrix range`` line answers this for the
    #: whole model; what it cannot say, and what a caller can act on, is which
    #: *declaration* holds the outlier. ``largest / smallest`` over the frame is
    #: the conditioning to compare against the solver's. A block whose every row
    #: went (the absence rules) has no entry, the same way it has no rows.
    coefficient_range: pl.DataFrame

    #: ``(variable, smallest, largest)`` — the **bound** magnitudes each variable
    #: block put on its columns, one row per block that declared a finite one.
    #: The axis a solver reports and does not repair: HiGHS prints a ``Bound``
    #: range beside its ``Matrix`` one, equilibrates the matrix automatically,
    #: and answers the bounds with ``Consider scaling the bounds by …`` — so a
    #: model can be clean on :attr:`coefficient_range` and still be the one the
    #: solver is complaining about. Zero and infinity are excluded, an
    #: unbounded side and a ``lower: 0`` being nothing the solver represents. A
    #: large ``largest`` is usually a big number standing in for "uncapped", and
    #: wants no upper bound at all rather than a rounder one.
    bound_range: pl.DataFrame

    #: ``(constraint, smallest, largest)`` — the same for each block's
    #: right-hand sides, over the rows that survived. The fourth of the four
    #: ranges a solver reports, and the last of them this can answer per
    #: declaration rather than per model.
    rhs_range: pl.DataFrame

    #: The same pair for the objective's coefficients, or ``None`` where the
    #: model declares no objective and where every term of one cancelled.
    objective_range: tuple[float, float] | None

    #: How many times this model has been solved, and how many of those solves
    #: loaded the solver from scratch instead of pushing values onto one that
    #: already held it. Read together: ``loads == 1`` is a driver on the fast
    #: path — the first solve had nothing to keep — and ``loads == solves`` on
    #: an iterating driver is the difference between "lpspec is slow" and
    #: "this model masks on a parameter that varies", unless the driver asked
    #: for ``keep='nothing'``, which loads by construction. ``loads`` ticks on
    #: exactly the solves that report :attr:`Result.kept` of ``nothing`` —
    #: the same event, counted here and named there.
    solves: int
    loads: int

    #: Cumulative wall-clock seconds per phase, keyed by the phase's name:
    #: ``attach`` (the caller's sources onto the plan), ``build`` (declarations
    #: into the model frames), ``handoff`` (the built model into a solver),
    #: ``solve`` (the solver's own run), ``write`` (the built model to a
    #: file). A phase that never ran has no key; one that ran again holds the
    #: sum — an update's attach and build land on top of the first's, the way
    #: ``solves`` keeps counting. Clocks rather than a profile: enough to say
    #: which phase a slow loop spends its time in, not why.
    seconds: Mapping[str, float]

    def metrics(self) -> Metrics:
        """The sizes, counters and clocks as one value — the row an archive records.

        What ``archive=`` records beside the answer, and what a caller feeding
        its own store reads off a model it solved. Which fields reach it and
        what it means cumulatively are
        :class:`~lpspec.relational.parquet.Metrics`'s to say; a phase this
        build never entered reads zero there. ``run`` is null: the name is the
        publisher's, and nothing has published this yet.
        """
        clocks = self.seconds
        return Metrics(
            columns=self.columns,
            rows=self.rows,
            nonzeros=self.nonzeros,
            added_columns=self.added_columns,
            added_rows=self.added_rows,
            solves=self.solves,
            loads=self.loads,
            attach_seconds=clocks.get('attach', 0.0),
            build_seconds=clocks.get('build', 0.0),
            handoff_seconds=clocks.get('handoff', 0.0),
            solve_seconds=clocks.get('solve', 0.0),
            write_seconds=clocks.get('write', 0.0),
        )


def _named(frames: Mapping[str, pl.LazyFrame], name: str, kind: str) -> pl.LazyFrame:
    try:
        return frames[name]
    except KeyError:
        raise KeyError(unknown_name_message(kind, name, frames)) from None


@dataclass
class Result:
    """What a solve returned — the outcome, and access to any values.

    Returned whatever the solve concluded: test :attr:`has_primal` before
    reading values, or catch :class:`~lpspec.errors.NoSolutionError`. The
    values are this result's own, so a later solve on the same model does not
    rewrite them, and there is no lifetime to manage — :meth:`close` releases
    what this result holds early, and nothing breaks without it.

    An update is no exception. A result owns everything it reads — one finished
    frame per declaration, its own values already laid out over the label frames
    of the build it answered — so it outlives anything done to the model
    afterwards: an update, another solve, ``model.close()``. What retaining one
    costs is those label frames staying alive, which matters once a caller keeps
    several, as a sweep, a rolling horizon and Benders all do.
    """

    _status: SolveStatus
    _objective: float
    #: One ``(dims…, value)`` frame per declaration, lazy and in label order —
    #: a read is a collect. ``None`` is what :meth:`close` leaves behind, and
    #: the primal's absence is what "closed" means: both go together, and an
    #: empty mapping is a solve that left nothing, which the status reports.
    _primals: Mapping[str, pl.LazyFrame] | None
    _duals: Mapping[str, pl.LazyFrame] | None
    #: The constraints' left-hand sides at the solution, laid out exactly as
    #: :attr:`_duals` — same frames, same row order — and present whenever the
    #: primals are: unlike a dual, an activity exists at any incumbent.
    _activities: Mapping[str, pl.LazyFrame] | None
    #: How much of the session this solve kept, read off what actually ran —
    #: never off what was asked for.
    _kept: Keep
    #: One deferred reader per declared named expression. Nothing about an
    #: expression is lowered or compiled until its reader is called. Released
    #: with the primals by :meth:`close`, since each holds this build's frames
    #: and values.
    _expressions: Mapping[str, Callable[[], pl.DataFrame]] | None = None
    #: :meth:`evaluate`'s implementation, over the same snapshot the readers
    #: above close on, composed above the lane so this one need not read the
    #: model as written (hard rule 2). ``None`` where the model was built from
    #: an already-lowered ``Program``. Released with the primals.
    _evaluate: Callable[[str | Mapping[str, Any]], pl.DataFrame] | None = None
    #: :meth:`extend`'s half of the same: earlier entries and a new block in,
    #: one deferred reader per new entry plus what a later extend needs to read
    #: them out. Opaque here — the model as written, which this lane may not
    #: read (hard rule 2). ``None`` with :attr:`_evaluate`. Released with the
    #: primals.
    _extend: (
        Callable[
            [Mapping[str, Any], Mapping[str, Any]],
            tuple[Mapping[str, Callable[[], pl.DataFrame]], Mapping[str, Any]],
        ]
        | None
    ) = None
    #: What :attr:`_extend` carried out of the last extend, to hand back to the
    #: next. Empty on a result no extend has been through.
    _added: Mapping[str, Any] = field(default_factory=dict)
    #: Why there are no duals, when a solve that left values still has none.
    #: ``None`` whenever :attr:`_duals` holds them.
    _no_duals: str | None = None
    #: Which spec this answered, as :func:`~lpspec.relational.parquet.digest_of`
    #: names it. Attached by the model that solved, so a solve run off a
    #: lowered program — which has no document — leaves it ``None``.
    _spec_digest: str | None = None
    #: When the solver returned, in UTC. Attached by the model that solved.
    _solved_at: datetime | None = None

    @property
    def status(self) -> str:
        """Coarse outcome: ``ok`` / ``warning`` / ``error`` / ``aborted`` / ``unknown``."""
        return self._status.status

    @property
    def termination_condition(self) -> str:
        """What the solver said — ``optimal``, ``infeasible``, ``time_limit`` and so on."""
        return self._status.termination_condition

    @property
    def is_ok(self) -> bool:
        """The linopy rollup: not an error, an abort or a refusal."""
        return self._status.is_ok

    @property
    def has_primal(self) -> bool:
        """Whether there are values to read — what the accessors gate on.

        Narrower than :attr:`is_ok`: a run stopped at a time limit before any
        incumbent is ``ok`` with nothing to read.
        """
        return self._status.is_readable

    @property
    def objective(self) -> float:
        """The objective value, or ``nan`` when there is no solution."""
        return self._objective

    @property
    def spec_digest(self) -> str | None:
        """Which spec this answered — a digest of the file, not its name.

        Two answers carrying one digest answered the same document, so a table
        of saved cases says whether it is comparing like with like. The *data*
        may differ entirely: two scenarios of one spec share this. ``None``
        where the solve ran off a lowered program, which has no document.
        """
        return self._spec_digest

    @property
    def solved_at(self) -> datetime | None:
        """When the solver returned, in UTC — ``None`` where the solve carried no clock.

        What orders a table concatenated from runs solved apart, so that a
        comparison is not left reading the timestamps of the files.
        """
        return self._solved_at

    @property
    def kept(self) -> Keep:
        """How much of the session this solve kept — one of :data:`KEEPS`.

        What *happened*, not what was asked: ``keep=`` is a preference, and a
        first solve or a structure that moved keeps ``nothing`` whatever it
        requested, the solver having been loaded again. So a driver that asked
        to keep ``progress`` and reads ``nothing`` back is being told its
        labels moved. Advisory, like :class:`Diagnostics`: no answer depends
        on it.
        """
        return self._kept

    def _unclosed(self, what: str) -> Mapping[str, pl.LazyFrame]:
        """The primals, or why nothing here can be read: this result was closed.

        The closed check is read off the primals whichever mapping the caller
        wants: :meth:`close` releases them together.
        """
        if self._primals is None:
            raise LpspecError(
                f'cannot read {what}: this result was closed, and closing releases its values and its '
                f'hold on the coordinates they lay out over. Frames already read stay valid — they are '
                f'their own data — so read what you need before close(), or drop the `with` and close '
                f'when you are done.'
            )
        return self._primals

    def _readable(self, frames: Mapping[str, pl.LazyFrame] | None, what: str) -> Mapping[str, pl.LazyFrame]:
        """*frames*, or why they cannot be read — closed first, then the status."""
        self._unclosed(what)
        if not self._status.is_readable:
            wording = f' ({self._status.solver_wording})' if self._status.solver_wording else ''
            raise NoSolutionError(
                f'cannot read {what}: the solve terminated {self.termination_condition!r}'
                f'{wording}, so there are no values to read. Test '
                f'`has_primal` first. This raises rather than returning, because the solver '
                f'hands back a full-length vector of zeros either way and it is '
                f'indistinguishable from an answer.'
            )
        assert frames is not None, 'close() releases the primal, dual and activity frames together'
        return frames

    def primal(self, name: str) -> pl.DataFrame:
        """The tidy solution of variable *name* — ``(dims…, value)``.

        Rows come back in label order, row-major over the variable's coordinate
        product, so two reads and two runs agree.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed.
            KeyError: No variable is called *name*.
        """
        frames = self._readable(self._primals, f"the primal of '{name}'")
        return _named(frames, name, 'variable').collect(engine='streaming')

    def dual(self, name: str) -> pl.DataFrame:
        """Shadow prices of constraint *name* — ``(dims…, value)``.

        :meth:`primal`'s shape and order, over constraint rows.

        Raises:
            NoSolutionError: The solve left no values at all.
            LpspecError: This result was closed, or it left primals but no
                duals — an integer variable makes them undefined.
            KeyError: No constraint is called *name*.
        """
        frames = self._readable(self._duals, f"the dual of '{name}'")
        if self._no_duals is not None:
            raise LpspecError(self._no_duals)
        return _named(frames, name, 'constraint').collect(engine='streaming')

    def activity(self, name: str) -> pl.DataFrame:
        """The left-hand side of constraint *name* at the solution — ``(dims…, value)``.

        :meth:`dual`'s shape and order, and the other half of a row's story:
        how far each row's ``Σ aᵢxᵢ`` sits from its bound. The solver's own
        number, not a recomputation. Readable whenever there is a solution —
        unlike :meth:`dual` it is well-defined on a mixed-integer model. On an
        ``==`` row it equals the right-hand side up to solver tolerance by
        construction.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed.
            KeyError: No constraint is called *name*.
        """
        frames = self._readable(self._activities, f"the activity of '{name}'")
        return _named(frames, name, 'constraint').collect(engine='streaming')

    def expression(self, name: str) -> pl.DataFrame:
        """The value of named expression *name* at this solution — ``(dims…, value)``.

        Evaluated at the solve's primal values and aggregated to the
        expression's own dims, in declaration order. Lowered and compiled on
        this call, so a model that reads no expression pays for none. Takes a
        declared name, never an expression string.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed.
            DataError: A divisor with no value where the expression divides.
            KeyError: No named expression is called *name*.
        """
        self._readable(self._primals, f"expression '{name}'")
        readers = self._expressions or {}
        try:
            reader = readers[name]
        except KeyError:
            raise KeyError(
                unknown_name_message('named expression', name, readers)
                + ' expression() takes a name declared under expressions:, never an expression string.'
            ) from None
        return reader()

    def evaluate(self, expression: str | Mapping[str, Any]) -> pl.DataFrame:
        """The value of *expression* at this solution — ``(dims…, value)``.

        :meth:`expression` for a quantity the file never named, written the way
        ``expressions:`` writes one (a string, or the mapping carrying
        ``cases:`` with ``foreach:`` and ``otherwise:``) and answered in the
        same shape. It may use every name the solved model declares and only
        those. A declared name is served by its own reader rather than lowered
        again.

        Names nothing, so it is not a *kind*: not written by :meth:`to_parquet`,
        not spilled by a sweep, not reachable through ``kind='expression'``.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed, the model was built from an
                already-lowered ``Program``, or the expression reads a dual
                and the solve left none.
            DataError: A divisor with no value where the expression divides.
            LanguageError: A construct outside the language, or a name the
                model does not declare.
        """
        self._readable(self._primals, 'an expression')
        readers = self._expressions or {}
        if isinstance(expression, str) and expression in readers:
            return readers[expression]()
        if self._evaluate is None:
            raise LpspecError(no_model_behind_this_answer_message())
        return self._evaluate(expression)

    def extend(self, added: Mapping[str, Any]) -> Result:
        """This solve, with the expressions *added* declares also readable.

        Status, objective and the frames :meth:`primal`, :meth:`dual` and
        :meth:`activity` hand back are carried over unchanged and uncopied, so
        the result is this solve rather than a second one::

            report = result.extend({'expressions': {'co2': 'sum(p * rate, over=generator)'}})
            report.expression('co2')  # the added one
            report.expression('total_gen')  # the model's own, still there
            report.to_dataset(kind='expression')

        *added* is a model fragment carrying ``expressions:`` and nothing else,
        each entry as :meth:`evaluate` takes one. Unlike :meth:`evaluate` the
        entries are named, so they read through :meth:`expression`, ride every
        bridge, and are written by :meth:`to_parquet` beside the declared ones.

        Adds, never replaces: a name this result already reads is refused, not
        shadowed. Nothing is mutated — closing one of the two leaves the other
        whole. The block is lowered once when handed in, so a bad expression
        fails here; each entry compiles only when read.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed, the model was built from an
                already-lowered ``Program``, or a name it already reads.
            LanguageError: A construct outside the language, or a name the
                model does not declare.
            SchemaError: A fragment carrying any section but ``expressions:``.
        """
        self._readable(self._primals, 'an expression')
        if self._extend is None:
            raise LpspecError(no_model_behind_this_answer_message())
        readers = dict(self._expressions or {})
        added_readers, carried = self._extend(self._added, added)
        if clash := sorted(set(added_readers) & set(readers)):
            raise LpspecError(already_readable_message(clash))
        return replace(self, _expressions=readers | dict(added_readers), _added=carried)

    def _frame(self, name: str, kind: str) -> pl.DataFrame:
        """*name* through the reader *kind* names — the dispatch every bridge shares."""
        reader = {'primal': self.primal, 'dual': self.dual, 'expression': self.expression}[reader_kind(kind)]
        return reader(name)

    def _names(self, kind: str) -> tuple[str, ...]:
        """Every name of *kind* this result can read — what a bridge takes by default.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed, or *kind* is ``dual`` and
                the duals are undefined.
        """
        if reader_kind(kind) == 'primal':
            return tuple(self._readable(self._primals, 'the solution'))
        if kind == 'dual':
            frames = self._readable(self._duals, 'the duals')
            if self._no_duals is not None:
                raise LpspecError(self._no_duals)
            return tuple(frames)
        self._readable(self._primals, 'the expressions')
        return tuple(self._expressions or {})

    def to_pandas(self, name: str, kind: str = 'primal') -> pd.DataFrame:
        """One name's values as a tidy :class:`pandas.DataFrame`.

        Args:
            name: A variable, a constraint or a named expression, as *kind*
                says.
            kind: ``primal``, ``dual`` or ``expression`` — the reader this
                stands in for.
        """
        return tidy_to_pandas(self._frame(name, kind))

    def to_dataarray(self, name: str, kind: str = 'primal') -> xr.DataArray:
        """One name's values as a labelled :class:`xarray.DataArray`, :meth:`to_pandas`'s arguments.

        Dense over the name's dims: a masked coordinate comes back NaN.
        """
        return tidy_to_dataarray(self.to_pandas(name, kind), name)

    def to_dataset(self, *names: str, kind: str = 'primal') -> xr.Dataset:
        """The named values of one *kind* as one :class:`xarray.Dataset`; all of that kind by default.

        One kind per call: a dual and a variable of the same name would
        collide, and mean something else per row. Each arrives dense over its
        own dims, all at once — on a large model name the few you need, or use
        :meth:`save`, which writes every kind.

        Args:
            names: What to include; none means every name of *kind*.
            kind: ``primal``, ``dual`` or ``expression``.
        """
        return tidy_to_dataset(names or self._names(kind), lambda name: self.to_dataarray(name, kind))

    def save(self, directory: str | Path) -> Path:
        """Every kind this solve answered with, one file per name, into *directory*.

        ``objective.parquet`` holds the
        :class:`~lpspec.relational.parquet.Record` — how the solve terminated
        and what it reached, in the columns a sweep keys and folds. A solve
        that reached no objective writes null there rather than ``nan``, so a
        directory per case is a table an aggregate reads. Then
        ``primal/<name>.parquet`` for every variable, ``dual/<name>.parquet``
        for every constraint where the duals are defined, and
        ``expression/<name>.parquet`` for every named expression this data
        can evaluate — an integer variable leaves the duals out, and an
        expression that fails on this data is left out, :meth:`expression`
        still saying why. The primals are streamed to disk in
        :meth:`primal`'s order, so the same model and data write the same
        bytes.

        ``activity/<name>.parquet`` goes beside them for every constraint,
        which no ``kind=`` names — a sweep folds three kinds and never holds
        these, so a saved result carries them under a name of their own.

        ``reasons.parquet`` holds ``(kind, name, reason)`` for whatever is
        deliberately not here, and is absent when everything is: one row per
        expression that failed, and one with an empty *name* for the duals,
        whose absence is never per-constraint. Written because a directory
        that simply lacks a file cannot tell "there is none, and here is why"
        from "no such name", which is the one thing :meth:`dual` and
        :meth:`expression` do say.

        A solve that left no values writes the record and nothing else. A run
        that came back infeasible is an answer a set of saved cases needs on
        disk, rather than a directory that does not exist.

        **The directory holds this answer and no other.** Whatever a previous
        save left there is removed first, so a re-run cannot leave one model's
        frames beside another's record. Files that are not part of the layout
        are left alone.

        Returns:
            The directory.

        Raises:
            LpspecError: This result was closed.
        """
        import polars as pl

        primals = self._unclosed('the solution')
        out = Path(directory)
        clear_the_answer(out)
        write_format(out)
        record = Record.of(
            self.termination_condition,
            self.objective,
            has_primal=self.has_primal,
            spec_digest=self._spec_digest,
            solved_at=self._solved_at,
        )
        write_whole(pl.DataFrame([record._asdict()], schema_overrides=RECORD_SCHEMA), out / RECORD_FILE)
        if not self._status.is_readable:
            return out
        for name, frame in primals.items():
            write_whole(frame, out / 'primal' / f'{name}.parquet')
        for name, frame in (self._duals or {}).items():
            write_whole(frame, out / 'dual' / f'{name}.parquet')
        for name, frame in (self._activities or {}).items():
            write_whole(frame, out / 'activity' / f'{name}.parquet')
        no_expressions: dict[str, str] = {}
        for name, reader in (self._expressions or {}).items():
            try:
                evaluated = reader()
            except LpspecError as absent:
                no_expressions[name] = str(absent)
                continue
            write_whole(evaluated, out / 'expression' / f'{name}.parquet')
        write_reasons(out, self._no_duals, no_expressions)
        return out

    def close(self) -> None:
        """Release what this result holds early. Optional.

        Its frames, which carry both its own values and its hold on the label
        frames of the build it answered. Frames already read stay valid. Never
        the model or the solver, which are the
        :class:`~lpspec.api.Model`'s to close.
        """
        self._primals = self._duals = self._activities = self._expressions = None
        self._evaluate = self._extend = None

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
