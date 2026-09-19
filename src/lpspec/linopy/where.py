"""A ``where:`` predicate as a boolean array over the coordinates it masks.

The other half of what a declaration says: ``builder.py`` builds the thing,
this decides where it exists. A :class:`~math_spec.program.WhereNode` in, one
``xr.DataArray`` of booleans out, and :func:`as_linopy_mask` puts it in the
shape linopy's ``mask=`` takes. Both lanes read the same node kinds, and
``relational/engines/polars/predicates.py`` answers each with a polars
expression where this one answers with an array.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import xarray as xr
from math_spec import program

from lpspec.errors import DataError, position_out_of_range_message, short_groups_message
from lpspec.linopy import absence
from lpspec.linopy.operators import _grouped

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
    """Everything evaluating a plan needs beyond the node: the data, the axes, the model, the relations, the program.

    ``relations`` carries one array per value column, keyed by its relation's
    name and its own, over the dimensions the key names — what a predicate on a
    relation and a grouped operator both read instead of the parameter dataset.
    """

    dataset: xr.Dataset
    master_coords: Mapping[str, pd.Index]
    model: linopy.Model
    relations: Mapping[tuple[str, str], xr.DataArray]
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
    comparison over NaN comes back false.

    **A null relation value is excluded explicitly rather than by ``fillna``.**
    A partial map arrives as an object array holding ``None``, and numpy
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

    if isinstance(node, program.ExpressionComparisonNode):
        from lpspec.linopy.builder import _eval  # mask ↔ expression recursion

        left, right = _eval(node.left, ctx), _eval(node.right, ctx)
        compared = xr.DataArray(_PREDICATE_OPS[node.op](left, right))
        return compared.fillna(value=False).astype(bool)

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
        if node.partition is not None:
            by = node.partition.name
            (column,) = node.partition.group
            arr = _group_offsets(node, by, bound_relation(by, column, ctx.relations), np.asarray(labels))
            return (_PREDICATE_OPS[node.op](arr, 0) & arr.notnull()).fillna(value=False).astype(bool)
        at = node.position + len(labels) if node.position < 0 else node.position
        if not 0 <= at < len(labels):
            raise DataError(position_out_of_range_message(node.name, node.op, node.position, at, len(labels)))
        arr = xr.DataArray(np.arange(len(labels)), coords={node.name: labels}, dims=[node.name])
        return _PREDICATE_OPS[node.op](arr, at).astype(bool)

    if isinstance(node, program.RelationComparisonNode):
        arr = bound_relation(node.name, node.column, ctx.relations)
        return (_PREDICATE_OPS[node.op](arr, node.value) & arr.notnull()).fillna(value=False).astype(bool)

    if isinstance(node, program.RelationPairComparisonNode):
        left = bound_relation(node.name, node.column, ctx.relations)
        right = bound_relation(node.other, node.other_column, ctx.relations)
        defined = left.notnull() & right.notnull()
        return (_PREDICATE_OPS[node.op](left, right) & defined).fillna(value=False).astype(bool)

    if isinstance(node, program.RelationDefinedNode):
        return _relation_has_a_row(node, ctx)

    if isinstance(node, program.NotNode):
        return ~evaluate(node.operand)

    if isinstance(node, program.AndNode):
        return evaluate(node.left) & evaluate(node.right)

    if isinstance(node, program.OrNode):
        return evaluate(node.left) | evaluate(node.right)

    assert_never(node)  # pyrefly: ignore[bad-argument-type]  — ArithmeticComparisonNode is in the union and lowering always replaces it (NEVER_LOWERED)


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


def _group_offsets(
    node: program.DimensionPositionNode, by: str, groups: xr.DataArray, labels: np.ndarray
) -> xr.DataArray:
    """Each coordinate's distance from the boundary of *its own* group.

    Zero marks the coordinate the position names, so every comparator reads the
    same as it does ungrouped. ``nan`` where the relation sends a coordinate
    nowhere: in no group, so no group's boundary.

    Raises:
        DataError: If any group is shorter than the position names, which would
            leave that group's rows unseeded and the model quietly unanchored.
    """
    partition = _grouped(node.name, labels, groups)
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(str(g) for g, n in zip(partition.names, partition.counts, strict=True) if n < needed)
    if short:
        raise DataError(short_groups_message(node.name, by, node.op, node.position, short))
    target = node.position if node.position >= 0 else partition.size + node.position
    return partition.within.where(partition.grouped) - target


def _relation_has_a_row(node: program.RelationDefinedNode, ctx: EvaluationContext) -> xr.DataArray:
    """Where the relation holds a row at the frame's coordinates.

    The arrays are padded to the key's whole product, so a key the table leaves
    out is null in every column; a row it does hold carries at least one value,
    a column left null in it being a value the model reads as absent.
    """
    columns = ctx.program.relations[node.name].values
    return reduce(operator.or_, (bound_relation(node.name, column, ctx.relations).notnull() for column in columns))


def unbound_relation_message(name: str) -> str:
    """A declared relation read with no attached map."""
    return f"relation '{name}' has no attached values. Pass it under key '{name}' as a table of the rows it holds."


def bound_relation(name: str, column: str, relations: Mapping[tuple[str, str], xr.DataArray]) -> xr.DataArray:
    """One value column of a map as an array over the dimensions its key names."""
    try:
        return relations[name, column]
    except KeyError:
        raise DataError(unbound_relation_message(name)) from None


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
