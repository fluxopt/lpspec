"""A ``where:`` mask as a query: which rows of a coordinate product survive.

The plan's predicate nodes in, a boolean expression out — and the frame the
walk had to join parameters onto to build it, since a mask reads values the
product does not carry. The joins happen *during* the walk: the condition is
built first and the frame read after.

A closed vocabulary of its own — comparisons against a parameter, a dimension
label, a position along a dimension, a relation, arithmetic over parameters,
and the three connectives. It
takes the :class:`~lpspec.relational.engines.polars.scope.Scope` as an
argument and holds nothing. :func:`masked` is the product a declaration is
instantiated over, cut by its mask: the one place the two meet.

:class:`Carrier` lives here too, and the bounds walk imports it: both walks
that read parameters build an expression over columns they are joining on as
they go.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import polars as pl
from math_spec import program

from lpspec.errors import DataError, position_out_of_range_message, short_groups_message
from lpspec.relational.engines.polars.fragments import join_on
from lpspec.relational.engines.polars.relations import GROUP_RANK, GROUP_SIZE, Grouping

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable

    from polars._typing import JoinStrategy

    from lpspec.relational.engines.polars.scope import Scope


class Carrier:
    """A frame a walk joins onto, each attachment made at most once.

    Both walks that read parameters — the mask (:func:`compile_predicate`)
    and the bounds (:meth:`~lpspec.relational.engines.polars.compiler.PolarsCompiler.bounds`)
    — build an expression over columns they are joining on as they go, so the
    frame and the set of aliases already attached travel together.
    """

    def __init__(self, frame: pl.LazyFrame) -> None:
        self.frame = frame
        self._attached: set[str] = set()

    def once(self, alias: str, attach: Callable[[pl.LazyFrame, str], pl.LazyFrame]) -> str:
        """Join *attach* onto the frame under *alias*, unless it already is.

        Returns:
            *alias*, so a caller reads the column it just made sure of.
        """
        if alias not in self._attached:
            self.frame = attach(self.frame, alias)
            self._attached.add(alias)
        return alias


def _defined(col: pl.Expr, dtype: program.ParameterDtype) -> pl.Expr:
    """What a bare parameter name in a ``where`` asks of *col*.

    Three readings, and the declaration picks: a ``bool`` is its own answer, a
    ``str`` is defined wherever the table has a row, and a number has to be
    finite as well. Read off the declaration rather than the column, which the
    door has already held to it.
    """
    if dtype == 'bool':
        return col.is_not_null() & col.cast(pl.Boolean)
    if dtype == 'str':
        return col.is_not_null()
    return col.is_not_null() & col.is_finite()


def masked(scope: Scope, dims: tuple[str, ...], where: program.Mask | None) -> pl.LazyFrame:
    """The masked coordinate product over *dims*.

    Labels, plus the ordinals a caller sorts by so labels follow declaration
    order.

    A mask that has to join restricts by semi-join: the predicate reads only
    its own dims, so it is evaluated over *their* product and the full product
    is semi-joined against the truth set, which leaves the left side's row
    order intact.

    Four shapes stay on the direct filter path, which is pointwise and keeps
    order too: a predicate that joins nothing, one reading no frame dim, one
    reading dims outside the frame (so errors name the full frame), and one
    reading **every** frame dim.
    """
    out = scope.product(dims)
    if where is None:
        return out
    on = tuple(d for d in dims if d in where.dims)
    if on and len(on) < len(dims) and where.dims <= set(dims):
        keyed = scope.product(on)
        carrier, condition = compile_predicate(scope, keyed, where, on)
        if carrier is keyed:
            return out.filter(falsy_if_null(condition))
        surviving = carrier.filter(falsy_if_null(condition)).select(*on)
        return out.join(surviving, on=list(on), how='semi')
    carrier, condition = compile_predicate(scope, out, where, dims)
    return carrier.filter(falsy_if_null(condition))


def compile_predicate(
    scope: Scope, frame: pl.LazyFrame, mask: program.Mask, dims: tuple[str, ...]
) -> tuple[pl.LazyFrame, pl.Expr]:
    """``(frame with the mask's parameters joined, boolean expression)``.

    Walking joins the parameters, so the condition is built first and the
    frame read after — one expression would return the pre-walk frame.

    **A name the mask is certain of is joined rather than left-joined**,
    and a certain variable is semi-joined and never read
    (:func:`_certain_names`). An atom over a missing value reads as false
    either way, so the strategies differ only in *where* the row is dropped.

    ``VariableDefined`` is the one atom answered by a join rather than a
    column test — existence lives in the variable's own frame — keyed by
    dims the dim rule has already checked are inside this frame.

    No join here maintains order: consumers verify where they read
    (:func:`labels.in_position_order`), so a shuffle costs a sort
    downstream at worst, never a wrong label.
    """
    certain = _certain_names(mask)
    carrier = Carrier(frame)

    def join_param(param: str) -> str:
        how: JoinStrategy = 'inner' if param in certain else 'left'
        return carrier.once(
            f'__where {param}__',
            lambda f, alias: scope.parameter_join(f, param, dims, alias, f"where-parameter '{param}'", how),
        )

    def refuse_outside_frame(reading: str, dimension: str) -> None:
        """A mask reading a dim the frame does not span — the plan's refusal, asserted here.

        Reducing a mask over an unlisted dim would admit a row wherever *any*
        coordinate of it satisfied the mask. The language refuses it at load,
        before a plan exists to carry it, so the frame planner states it as the
        invariant it now is.
        """
        assert dimension in dims, f'where-comparison on {reading} is outside the frame dims {list(dims)}'

    def join_ordinal(dimension: str) -> str:
        refuse_outside_frame(f"dimension '{dimension}'", dimension)
        return carrier.once(
            f'__where ord {dimension}__',
            lambda f, alias: f.join(
                scope.data.dimensions[dimension].select(pl.col('val').alias(dimension), pl.col('ord').alias(alias)),
                on=dimension,
                how='left',
            ),
        )

    def join_group_offset(p: program.DimensionPosition) -> str:
        """One column: the row's ordinal minus its own group's target ordinal.

        Joined on the dimension and the partition's joined dimensions, which
        the frame carries: a group is read at the rest of its key.
        """
        assert p.partition is not None, 'an ungrouped position counts along the whole dimension and asks for no table'
        grouping = Grouping.of(scope.data, p.partition)
        for dim in grouping.keys:
            refuse_outside_frame(f"dimension '{dim}'", dim)
        _refuse_short_groups(p, grouping)
        target = pl.lit(p.position) if p.position >= 0 else pl.col(GROUP_SIZE) + p.position
        offset = pl.col(GROUP_RANK) - target
        return carrier.once(
            f'__where ord {p.name} by {p.partition.name}__',
            lambda f, alias: f.join(
                grouping.table.select(pl.col('val').alias(p.name), *grouping.joined, offset.alias(alias)),
                on=list(grouping.keys),
                how='left',
            ),
        )

    def join_relation(relation: str, dims: tuple[str, ...], column: str | None) -> str:
        """*relation* read at *dims* — one value column, or with ``None`` whether a row is there at all.

        The frame supplies the key's dimensions for a keyed relation and every
        column's for a bare one, which is what the plan stamped on the leaf;
        the columns read at are the relation's own, matched to *dims* in order.
        """
        for dim in dims:
            refuse_outside_frame(f"relation '{relation}' reading dimension '{dim}'", dim)
        shape = scope.program.relations[relation]
        roles = shape.key or shape.roles
        assert tuple(shape.dim(role) for role in roles) == dims, f"relation '{relation}' is read at its key"
        read = pl.col(column) if column is not None else pl.lit(value=True)
        return carrier.once(
            f'__where relation {relation}.{column or ""}__',
            lambda f, alias: f.join(
                scope.data.relations[relation].select(
                    *(pl.col(role).alias(dim) for role, dim in zip(roles, dims, strict=True)), read.alias(alias)
                ),
                on=list(dims),
                how='left',
            ),
        )

    def _side(expression: program.Expression, dims: tuple[str, ...]) -> str:
        """*expression* joined onto the carrier as one value column, and that column's name.

        A side of a comparison is arithmetic over parameters — the language
        refuses a variable and a ``dual()`` there — so it compiles to constant
        fragments alone, and they are added over the comparison's dims the way
        a constant side of a constraint is. A coordinate no fragment reaches
        stays null so the comparison reads false, which is what every other
        atom over a missing value does.
        """
        from lpspec.relational.engines.polars.compiler import PolarsCompiler  # mask ↔ expression recursion

        def attach(frame: pl.LazyFrame, alias: str) -> pl.LazyFrame:
            compiled = PolarsCompiler(scope).expression(expression, 'a where clause')
            assert not compiled.terms and not compiled.quads, (
                'a where side is arithmetic over parameters, which compiles to constants alone'
            )
            product = frame.select(*dims).unique() if dims else frame.select(pl.lit(0).alias('__one__')).head(1)
            added = PolarsCompiler(scope).added(compiled.consts, product, fill=False)
            return join_on(frame, added.rename({'cval': alias}), dims, 'left')

        return carrier.once(f'__where expression {expression!r}__', attach)

    def walk(p: program.Predicate) -> pl.Expr:
        if isinstance(p, program.ExpressionComparison):
            left, right = (_side(p.left, p.dims), _side(p.right, p.dims))
            return falsy_if_null(_COLUMN_COMPARISONS[p.op](pl.col(left), pl.col(right)))
        if isinstance(p, program.ParameterComparison):
            return _compare(pl.col(join_param(p.name)), p.op, p.value)
        if isinstance(p, program.DimensionComparison):
            refuse_outside_frame(f"dimension '{p.name}'", p.name)
            return _compare(_dimension_column(p.name, p.value), p.op, p.value)
        if isinstance(p, program.DimensionPosition):
            if p.partition is not None:
                return falsy_if_null(_COLUMN_COMPARISONS[p.op](pl.col(join_group_offset(p)), pl.lit(0)))
            at = _position_ordinal(p, scope.data.cardinality[p.name])
            return _COLUMN_COMPARISONS[p.op](pl.col(join_ordinal(p.name)), pl.lit(at))
        if isinstance(p, program.RelationComparison):
            column = pl.col(join_relation(p.name, p.dims, p.column))
            if isinstance(p.value, str):
                column = column.cast(pl.String)
            return _compare(column, p.op, p.value)
        if isinstance(p, program.RelationPairComparison):
            left = pl.col(join_relation(p.name, p.dims, p.column))
            right = pl.col(join_relation(p.other, p.dims, p.other_column))
            return _COLUMN_COMPARISONS[p.op](left, right)
        if isinstance(p, program.RelationDefined):
            return pl.col(join_relation(p.name, p.dims, None)).is_not_null()
        if isinstance(p, program.ParameterDefined):
            return _defined(pl.col(join_param(p.name)), scope.program.parameters[p.name].dtype)
        if isinstance(p, program.VariableDefined):
            on = list(scope.program.variables[p.name].dims)
            coordinates = scope.variables[p.name].frame.select(*on)
            if p.name in certain:
                carrier.once(f'__where defined {p.name}__', lambda f, _: f.join(coordinates, on=on, how='semi'))
                return pl.lit(value=True)
            flag = carrier.once(
                f'__where defined {p.name}__',
                lambda f, alias: f.join(
                    coordinates.unique().with_columns(pl.lit(value=True).alias(alias)), on=on, how='left'
                ),
            )
            return falsy_if_null(pl.col(flag))
        if isinstance(p, program.BooleanLiteral):
            return pl.lit(value=p.value)
        if isinstance(p, program.And):
            return walk(p.left) & walk(p.right)
        if isinstance(p, program.Or):
            return walk(p.left) | walk(p.right)
        if isinstance(p, program.Not):
            return ~falsy_if_null(walk(p.operand))
        assert_never(p)  # pyrefly: ignore[bad-argument-type]  — ArithmeticComparison is in the union and lowering always replaces it (NEVER_LOWERED)

    condition = walk(mask.root)
    return carrier.frame, condition


def _certain_names(mask: program.Mask) -> frozenset[str]:
    """Parameter and variable names whose absence alone makes the whole mask false.

    A row those names have no value for is one the filter would drop anyway, so
    the join may drop it first. Only the ``AND`` spine counts: under ``OR`` or
    ``NOT`` an absent value can still leave the mask true, and dropping the
    row there is a wrong model rather than a slow one.
    """
    atoms = (program.ParameterComparison, program.ParameterDefined, program.VariableDefined)
    return frozenset(a.name for a in mask.conjuncts if isinstance(a, atoms))


def _refuse_short_groups(p: program.DimensionPosition, grouping: Grouping) -> None:
    """Refuse a position no coordinate of some group occupies.

    The ungrouped counterpart is :func:`_position_ordinal`, and the reason is
    the same one construct-wide: a boundary clause that silently seeds no row
    leaves that group's recurrence unanchored. Grouping only multiplies the
    chance — one short period is enough — so it is checked per group.

    A coordinate in no group is not in the grouping's table, so no group of
    ``None`` can be counted short. A group is named by its value, or by the
    tuple of its joined coordinates and values where the partition's key is
    wider than the dimension it walks.
    """
    assert p.partition is not None
    needed = p.position + 1 if p.position >= 0 else -p.position
    sizes = grouping.table.select(*grouping.key, GROUP_SIZE).unique().collect()
    named = (row[0] if len(grouping.key) == 1 else row[:-1] for row in sizes.iter_rows() if row[-1] < needed)
    if short := sorted(str(group) for group in named):
        raise DataError(short_groups_message(p.name, p.partition.name, p.op, p.position, short))


def falsy_if_null(condition: pl.Expr) -> pl.Expr:
    """*condition* with null read as false.

    A missing parameter row must exclude the coordinate rather than
    propagate. Masks are row absence.
    """
    return condition.fill_null(value=False)


def _position_ordinal(p: program.DimensionPosition, cardinality: int) -> int:
    """*p*'s position as an ordinal into a dimension of *cardinality* labels.

    A negative position counts from the end. Out of range is an error rather
    than a predicate matching nothing: a boundary clause that silently seeds
    no row leaves the recurrence unanchored.
    """
    at = p.position + cardinality if p.position < 0 else p.position
    if not 0 <= at < cardinality:
        raise DataError(position_out_of_range_message(p.name, p.op, p.position, at, cardinality))
    return at


def _dimension_column(dimension: str, value: float | str | datetime.date) -> pl.Expr:
    """The column a where-comparison on *dimension* reads.

    A string label is compared in ``String`` scope, undoing attaching's ``Enum``:
    The where-string rules order labels bytewise and read an unknown label as
    matching nothing,
    where an ``Enum`` orders by declaration and refuses strangers.
    """
    column = pl.col(dimension)
    return column.cast(pl.String) if isinstance(value, str) else column


#: The comparison operators, evaluated column against column.
_COLUMN_COMPARISONS: dict[program.PredicateOperator, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    '==': lambda left, right: left == right,
    '!=': lambda left, right: left != right,
    '<': lambda left, right: left < right,
    '<=': lambda left, right: left <= right,
    '>': lambda left, right: left > right,
    '>=': lambda left, right: left >= right,
}


def _compare(column: pl.Expr, op: program.PredicateOperator, value: float | str | datetime.date) -> pl.Expr:
    """One where-comparison. A string, a float and a date are all literals here."""
    return _COLUMN_COMPARISONS[op](column, pl.lit(value))
