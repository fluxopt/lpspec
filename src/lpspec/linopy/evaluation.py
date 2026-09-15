"""The plan evaluated: an expression as its value, a ``where:`` as a boolean array over the coordinates it masks.

One module for both walks because each holds the other: a cased expression
holds a mask per region, and a mask may compare two expressions. An
:class:`~math_spec.program.ExpressionNode` in, a linopy term, an array or a
number out (:func:`evaluate_expression`); a :class:`~math_spec.program.WhereNode` in, one
``xr.DataArray`` of booleans out (:func:`evaluate_where`), and
:func:`as_linopy_mask` puts it in the shape linopy's ``mask=`` takes. What a
plan is *built into* is ``builder.py``'s. Both lanes read the same node kinds:
``relational/engines/polars/compiler.py`` and ``predicates.py`` answer each
with a polars query where this one answers with an array.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import xarray as xr
from math_spec import program

from lpspec.errors import DataError, LpspecError, position_out_of_range_message, short_groups_message
from lpspec.linopy import absence
from lpspec.linopy.operators import (
    _grouped,
    operator_at,
    operator_grouped_sum,
    operator_shift,
    operator_sum,
    operator_sum_back,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import linopy
    import pandas as pd


#: Where-comparison operators, evaluated element-wise on a DataArray.
_PREDICATE_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
}


@dataclass(frozen=True)
class EvaluationContext:
    """Everything evaluating a plan needs beyond the node: the data, the axes, the model, the lookups, the program.

    ``dim_coords`` carries the attached lookup columns, which a predicate on a
    lookup and a grouped operator both read instead of the parameter dataset.
    """

    dataset: xr.Dataset
    master_coords: Mapping[str, pd.Index]
    model: linopy.Model
    dim_coords: Mapping[str, Mapping[str, xr.DataArray]]
    program: program.Program
    #: Whether *model* is solved and the plan is read at its solution — a
    #: variable is then its ``.solution`` and ``dual(c)`` the constraint's
    #: ``.dual`` — rather than built into it.
    solved: bool = False


def evaluate_where(mask: program.Mask | None, ctx: EvaluationContext) -> xr.DataArray:
    """Evaluate a lowered mask against a parameter dataset.

    Always a boolean DataArray. The no-mask case comes back 0-dimensional, so
    callers combine with ``&``/``|`` without case analysis.
    """
    if mask is None:
        return xr.DataArray(True)

    return _eval_node(mask.root, ctx)


def _eval_node(node: program.WhereNode, ctx: EvaluationContext) -> xr.DataArray:
    """One predicate node as a boolean DataArray.

    Two absences read as exclusion rather than as an answer: a variable's
    masked-out coordinate is absent (:func:`absence.present`), and a
    comparison over NaN comes back false. A comparison of expressions asks
    each side where it has a value (:func:`constant_side`) rather than reading
    its NaNs, since a build fills those before they reach the arithmetic.

    **A null lookup value is excluded explicitly rather than by ``fillna``.** A
    partial lookup arrives as an object array holding ``None``, and numpy
    answers ``None != 'north'`` with *True* rather than with null — so a ``!=``
    would keep exactly the labels that map nowhere.
    """
    dataset, master_coords = ctx.dataset, ctx.master_coords

    def evaluate(child: program.WhereNode) -> xr.DataArray:
        return _eval_node(child, ctx)

    if isinstance(node, program.BooleanLiteralNode):
        return xr.DataArray(node.value)

    if isinstance(node, program.ParameterDefinedNode):
        return _defined(dataset[node.name], ctx.program.parameter(node.name).dtype)

    if isinstance(node, program.VariableDefinedNode):
        return absence.present(ctx.model, node.name)

    if isinstance(node, (program.ParameterComparisonNode, program.DimensionComparisonNode)):
        if isinstance(node, program.ParameterComparisonNode):
            arr = dataset[node.name]
        else:
            arr = xr.DataArray(
                master_coords[node.name],
                coords={node.name: master_coords[node.name]},
                dims=[node.name],
            )

        result = _PREDICATE_OPS[node.op](arr, _as_the_axis_spells_it(arr, node.value))
        return result.fillna(False).astype(bool)

    if isinstance(node, program.DimensionPositionNode):
        labels = master_coords[node.name]
        if node.by is not None:
            arr = _group_offsets(node, bound_lookup(node.by, node.name, ctx.dim_coords), np.asarray(labels))
            return (_PREDICATE_OPS[node.op](arr, 0) & arr.notnull()).fillna(value=False).astype(bool)
        at = node.position + len(labels) if node.position < 0 else node.position
        if not 0 <= at < len(labels):
            raise DataError(position_out_of_range_message(node.name, node.op, node.position, at, len(labels)))
        arr = xr.DataArray(np.arange(len(labels)), coords={node.name: labels}, dims=[node.name])
        return _PREDICATE_OPS[node.op](arr, at).astype(bool)

    if isinstance(node, program.LookupComparisonNode):
        arr = bound_lookup(node.name, node.over, ctx.dim_coords)
        return (_PREDICATE_OPS[node.op](arr, node.value) & arr.notnull()).fillna(value=False).astype(bool)

    if isinstance(node, program.LookupPairComparisonNode):
        left = bound_lookup(node.name, node.over, ctx.dim_coords)
        right = bound_lookup(node.other, node.over, ctx.dim_coords)
        defined = left.notnull() & right.notnull()
        return (_PREDICATE_OPS[node.op](left, right) & defined).fillna(value=False).astype(bool)

    if isinstance(node, program.LookupDefinedNode):
        return bound_lookup(node.name, node.over, ctx.dim_coords).notnull()

    if isinstance(node, program.ExpressionComparisonNode):
        (left, left_defined), (right, right_defined) = constant_side(node.left, ctx), constant_side(node.right, ctx)
        compared = _PREDICATE_OPS[node.op](left, right) & left_defined & right_defined
        return xr.DataArray(compared).fillna(value=False).astype(bool)

    if isinstance(node, program.NotNode):
        return ~evaluate(node.operand)

    if isinstance(node, program.AndNode):
        return evaluate(node.left) & evaluate(node.right)

    if isinstance(node, program.OrNode):
        return evaluate(node.left) | evaluate(node.right)

    assert not isinstance(node, program.ArithmeticComparisonNode), 'lowering rebuilds every mask a program carries'
    assert_never(node)


def _defined(arr: xr.DataArray, dtype: str) -> xr.DataArray:
    """What a bare parameter name in a ``where`` asks: the declaration picks the reading.

    A ``bool`` is its own answer — a slot the data has no row for is false,
    the array having widened to float to hold the NaN — a ``str`` is defined
    wherever the data has a row, and a number has to be finite as well.
    """
    if dtype == 'bool':
        return arr.fillna(False).astype(bool)
    if dtype == 'str':
        return arr.notnull()
    return arr.notnull() & np.isfinite(arr)


def _group_offsets(node: program.DimensionPositionNode, groups: xr.DataArray, labels: np.ndarray) -> xr.DataArray:
    """Each coordinate's distance from the boundary of *its own* group.

    Zero marks the coordinate the position names, so every comparator reads the
    same as it does ungrouped. ``nan`` where the lookup sends a coordinate
    nowhere: in no group, so no group's boundary.

    Raises:
        DataError: If any group is shorter than the position names, which would
            leave that group's rows unseeded and the model quietly unanchored.
    """
    partition = _grouped(node.name, labels, groups)
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(str(g) for g, n in zip(partition.names, partition.counts, strict=True) if n < needed)
    if short:
        raise DataError(short_groups_message(node.name, str(node.by), node.op, node.position, short))
    target = node.position if node.position >= 0 else partition.size + node.position
    return partition.within.where(partition.grouped) - target


def unbound_lookup_message(name: str, over: str) -> str:
    """A declared lookup read with no attached map."""
    return (
        f"lookup '{name}' over dimension '{over}' has no attached values. "
        f"Pass it under key '{name}' as a table with columns ['{over}', '{name}']."
    )


def bound_lookup(name: str, over: str, dim_coords: Mapping[str, Mapping[str, xr.DataArray]]) -> xr.DataArray:
    """A lookup's attached values as an array over the dim it is over."""
    try:
        return dim_coords[over][name]
    except KeyError:
        raise DataError(unbound_lookup_message(name, over)) from None


def as_linopy_mask(mask: xr.DataArray) -> xr.DataArray | None:
    """Convert an evaluated where mask to linopy's ``mask=`` argument.

    linopy expects ``None`` for "no mask"; a 0-d True mask means exactly
    that. Everything else (including 0-d False) passes through.
    """
    if mask.ndim == 0 and bool(mask):
        return None
    return mask


def _as_the_axis_spells_it(arr: Any, value: Any) -> Any:
    """A where literal in the spelling the axis it is compared against uses.

    A quoted ISO date resolves to a ``datetime.date`` (the where rules), and a
    temporal axis arrives as ``datetime64`` — numpy compares the two by
    raising, so the axis decides, a literal carrying no dtype of its own.
    """
    if getattr(arr, 'dtype', None) is not None and arr.dtype.kind == 'M':
        return np.datetime64(value)
    return value


# ---------------------------------------------------------------------------
# Plan evaluation
# ---------------------------------------------------------------------------


def evaluate_expression(node: program.ExpressionNode, ctx: EvaluationContext) -> Any:
    """One plan node as a linopy term, an array, or a number.

    One node kind per branch: a variable is its linopy term, a parameter its
    filled array, arithmetic the Python operator linopy overloads, and an
    operator its function in ``operators.py``.
    """
    if isinstance(node, program.Constant):
        return node.value

    if isinstance(node, program.Variable):
        variable, declared = ctx.model.variables[node.name], ctx.program.variable(node.name).absence
        return absence.variable_value(variable, declared) if ctx.solved else absence.variable_term(variable, declared)

    if isinstance(node, program.Dual):
        assert ctx.solved, 'a dual reached a build — the language keeps one out of the math'
        return _dual(node.constraint, ctx)

    if isinstance(node, program.Parameter):
        return absence.coefficient(ctx.dataset[node.name])

    if isinstance(node, program.Negate):
        return -evaluate_expression(node.operand, ctx)

    if isinstance(node, program.Add):
        return evaluate_expression(node.left, ctx) + evaluate_expression(node.right, ctx)

    if isinstance(node, program.Multiply):
        return evaluate_expression(node.left, ctx) * evaluate_expression(node.right, ctx)

    if isinstance(node, program.Divide):
        return evaluate_expression(node.numerator, ctx) / evaluate_expression(node.divisor, ctx)

    if isinstance(node, program.Power):
        return evaluate_expression(node.base, ctx) ** evaluate_expression(node.exponent, ctx)

    if isinstance(node, program.Sum):
        summed = evaluate_expression(node.operand, ctx)
        for dimension in node.over:
            summed = operator_sum(summed, dimension)
        return summed

    if isinstance(node, program.GroupSum):
        return operator_grouped_sum(
            evaluate_expression(node.operand, ctx),
            _lookup_arrays(node.over, node.coordinate, ctx),
            into=node.into,
            labels=ctx.master_coords,
        )

    if isinstance(node, program.At):
        return operator_at(
            evaluate_expression(node.operand, ctx), _lookup_arrays(node.over, node.coordinate, ctx), into=node.into
        )

    if isinstance(node, program.Translate):
        return operator_shift(
            evaluate_expression(node.operand, ctx),
            over=node.dimension,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=node.fill,
            by=_partition(node, ctx),
        )

    if isinstance(node, program.Window):
        return operator_sum_back(
            evaluate_expression(node.operand, ctx),
            over=node.dimension,
            within=_amount(node.width, ctx),
            wrap=node.wrap,
            by=_partition(node, ctx),
        )

    if isinstance(node, program.Cases):
        return _cases(node, ctx)

    assert_never(node)


def constant_side(node: program.ExpressionNode, ctx: EvaluationContext) -> tuple[Any, Any]:
    """One side of a ``where`` comparing expressions: its value, and where it has one.

    The value is :func:`evaluate_expression`'s, each absent parameter row a zero, which is
    the right reading for every piece that adds — under a ``+``, a sum or a
    window an absent term is one fewer, and a zero is one fewer. What the
    fill cannot say is that a side with no value at all compares *false*
    rather than as zero, so where the side has one is decided beside it by
    structure (:func:`_has_value`) and the caller masks the comparison by it.
    """
    return evaluate_expression(node, ctx), _has_value(node, ctx)


def _has_value(node: program.ExpressionNode, ctx: EvaluationContext) -> Any:
    """Where *node* has a value, as a boolean array over its dims — the absence rules, read off the plan.

    A sum has a value where any term does, a product or a quotient where both
    factors do, and an operator wherever it gathers a present slot: the
    indicator is put through the same operator as the value, so a shift's
    edge counts as a value where the model gave it one and a window's
    unreachable lag does not. A variable cannot arrive: the language keeps a
    mask variable-free.
    """
    if isinstance(node, program.Constant):
        return True
    if isinstance(node, program.Parameter):
        return ctx.dataset[node.name].notnull()
    if isinstance(node, program.Negate):
        return _has_value(node.operand, ctx)
    if isinstance(node, program.Add):
        return _has_value(node.left, ctx) | _has_value(node.right, ctx)
    if isinstance(node, program.Multiply):
        return _has_value(node.left, ctx) & _has_value(node.right, ctx)
    if isinstance(node, program.Divide):
        return _has_value(node.numerator, ctx) & _has_value(node.divisor, ctx)
    if isinstance(node, program.Power):
        return _has_value(node.base, ctx) & _has_value(node.exponent, ctx)
    if isinstance(node, program.Cases):
        regions = (
            _in_region(_present_slots(region.value, ctx), evaluate_where(region.when, ctx)) for region in node.regions
        )
        return functools.reduce(operator.add, regions) > 0
    if isinstance(node, (program.Sum, program.GroupSum, program.At, program.Translate, program.Window)):
        return _gathered(node, _present_slots(node.operand, ctx), ctx) > 0
    assert not isinstance(node, (program.Variable, program.Dual)), 'a where compares variable-free expressions'
    assert_never(node)


def _present_slots(node: program.ExpressionNode, ctx: EvaluationContext) -> Any:
    """:func:`_has_value` as ones and zeros, the shape an operator sums."""
    return _has_value(node, ctx) * 1.0


def _gathered(
    node: program.Sum | program.GroupSum | program.At | program.Translate | program.Window,
    slots: Any,
    ctx: EvaluationContext,
) -> Any:
    """*slots* through the operator *node* applies to its operand — how many present slots each output gathers.

    A shift's edge is a present slot where the model filled it, so the
    indicator is shifted with a fill of one there and with none where the
    value has none.
    """
    if isinstance(node, program.Sum):
        for dimension in node.over:
            slots = operator_sum(slots, dimension)
        return slots
    if isinstance(node, program.GroupSum):
        mappings = _lookup_arrays(node.over, node.coordinate, ctx)
        return operator_grouped_sum(slots, mappings, into=node.into, labels=ctx.master_coords)
    if isinstance(node, program.At):
        return operator_at(slots, _lookup_arrays(node.over, node.coordinate, ctx), into=node.into)
    if isinstance(node, program.Translate):
        return operator_shift(
            slots,
            over=node.dimension,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=None if node.fill is None else 1.0,
            by=_partition(node, ctx),
        )
    return operator_sum_back(
        slots, over=node.dimension, within=_amount(node.width, ctx), wrap=node.wrap, by=_partition(node, ctx)
    )


def _dual(name: str, ctx: EvaluationContext) -> xr.DataArray:
    """``dual(name)`` at the solve — linopy's own ``.dual`` on the constraint.

    Refused on a model declaring integrality before linopy is asked: HiGHS
    hands a MIP back with a dual of zero on every row, and linopy stores it,
    so the number would be read rather than the absence the other lane
    reports.

    Raises:
        LpspecError: A variable declares integrality, so the duals are
            undefined; or the solver stored none.
    """
    discrete = sorted(n for n, v in ctx.program.variables.items() if v.domain != 'continuous')
    if discrete:
        raise LpspecError(
            f'named expression reads dual({name}), and duals are undefined for a mixed-integer model: '
            f'{", ".join(discrete)} declare integrality. Read it off a continuous model.'
        )
    try:
        return ctx.model.constraints[name].dual
    except AttributeError:
        raise LpspecError(
            f'named expression reads dual({name}), and this solve stored no duals — the solver returned none.'
        ) from None


def _in_region(value: Any, mask: xr.DataArray) -> Any:
    """*value* where the region holds, and a hard zero everywhere else.

    A **fill**, not a multiplication: inside the mask absence still stands, and
    outside it the value is a hard zero. A bare number has no absence to
    protect, so there the mask multiplies.
    """
    if hasattr(value, 'to_linexpr'):
        value = value.to_linexpr()
    if hasattr(value, 'where'):
        return value.where(mask, 0)
    return mask * value


def _cases(node: program.Cases, ctx: EvaluationContext) -> Any:
    """A value defined by region, as the regions added.

    The regions are disjoint and total — the language proved that before any
    data attached — so each one filled with zero outside itself and the lot
    added gives every coordinate exactly one region's value.
    """
    filled = (
        _in_region(evaluate_expression(region.value, ctx), evaluate_where(region.when, ctx)) for region in node.regions
    )
    return functools.reduce(operator.add, filled)


def _amount(amount: int | str, ctx: EvaluationContext) -> Any:
    """An offset or a width: the number, or the integer parameter naming it.

    Read through :func:`absence.coefficient` like any other parameter — a step
    nobody supplied is a step of nothing, which is what a zero offset means.
    """
    return absence.coefficient(ctx.dataset[amount]) if isinstance(amount, str) else amount


def _partition(node: program.Translate | program.Window, ctx: EvaluationContext) -> Any:
    """The lookup a windowed operator may not reach across, as its values.

    **Named for the dimension its values are labels of**, not for itself: an
    amount declared over the group's own dim is read through this array by
    :func:`~lpspec.linopy.operators._per_group`, which pairs the two by that
    name.
    """
    if node.partition is None:
        return None
    array = bound_lookup(node.partition, node.dimension, ctx.dim_coords)
    return array.rename(ctx.program.dimension(node.dimension).targets[node.partition])


def _lookup_arrays(over: str, names: tuple[str, ...], ctx: EvaluationContext) -> tuple[Any, ...]:
    """The declared lookups *names* as arrays over *over*, in the order the plan wrote them."""
    return tuple(bound_lookup(name, over, ctx.dim_coords) for name in names)
