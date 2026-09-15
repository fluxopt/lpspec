"""Model builder: logical plan + data → linopy Model.

**One section per kind of declaration**, in the order a build performs them:
``Variables``, ``Special-ordered sets``, ``Constraints``, ``Objectives``, each
ending in the ``model.add_*`` call it exists to make.

What an expression is worth and where a declaration exists are
``evaluation.py``'s two walks, ``operators.py`` evaluates a built-in once its
operands are values, and the positions an absent value is spelled differently
in are ``absence.py``. *Which* linopy call each construct becomes is the table
in ``docs/about/linopy.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from math_spec import program

from lpspec.errors import DataError, LaneError, null_bounds_message
from lpspec.lanes import LANES
from lpspec.linopy._notes import note
from lpspec.linopy.coverage import check_constant_side_covers, check_divisors_cover, gaps_under
from lpspec.linopy.evaluation import EvaluationContext, as_linopy_mask, evaluate_expression, evaluate_where
from lpspec.relational.sinks.capabilities import lane_cannot_build_message, required

if TYPE_CHECKING:
    import linopy
    import pandas as pd
    import xarray as xr

_SIGN_MAP = {'==': '=', '<=': '<=', '>=': '>='}


def build_model(
    model: linopy.Model,
    program: program.Program,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    dim_coords: dict[str, dict[str, xr.DataArray]],
) -> None:
    """Populate a linopy Model from a lowered program and loaded parameters.

    This mutates *model* in-place, adding variables, constraints and the
    objective as declared in *program*. Nothing is re-checked here: a program
    is trusted by construction, ``to_program`` having decided every rule the
    language can decide without data.
    """
    ctx = EvaluationContext(dataset, master_coords, model, dim_coords, program)
    _build_variables(ctx)
    _build_sos(ctx)
    _build_constraints(ctx)
    _build_objective(ctx)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def _build_variables(ctx: EvaluationContext) -> None:
    for name, vdef in ctx.program.variables.items():
        with note(f"while building variable '{name}'"):
            coords = {d: ctx.master_coords[d] for d in vdef.dims}
            mask = evaluate_where(vdef.where, ctx)

            _check_bounds_are_defined(name, vdef, ctx.dataset, mask)

            ctx.model.add_variables(
                lower=_bound(vdef.lower, ctx.dataset),
                upper=_bound(vdef.upper, ctx.dataset),
                coords=coords,
                name=name,
                mask=as_linopy_mask(mask),
                binary=vdef.domain == 'binary',
                integer=vdef.domain == 'integer',
            )


def _check_bounds_are_defined(name: str, vdef: program.VariableDeclaration, dataset: xr.Dataset, mask: Any) -> None:
    """Refuse a bound with no value at build, before the NaN reaches linopy's IO layer.

    Checked against the variable's own mask: a coordinate the variable does not
    occupy needs no bound.
    """
    missing = sum(gaps_under(dataset[name], mask) for name in sorted(program.parameters_of(vdef.lower, vdef.upper)))
    if missing:
        raise DataError(null_bounds_message(name, missing))


def _bound(bound: program.ExpressionNode, dataset: xr.Dataset) -> Any:
    """A bound as linopy takes it: the literal, or the named parameter's array.

    A gap is not filled here: absence's zero is a coefficient and never a
    bound, so a gap survives to :func:`_check_bounds_are_defined`.
    """
    if isinstance(bound, program.Constant):
        return bound.value
    if isinstance(bound, program.Parameter):
        return dataset[bound.name]
    msg = f'bounds accept a number or a parameter, and lowering builds nothing else — got {type(bound).__name__}'
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Special-ordered sets
# ---------------------------------------------------------------------------


def _build_sos(ctx: EvaluationContext) -> None:
    """Attach every ``sos:`` block to the variable it names."""
    for name, sos in ctx.program.sos.items():
        with note(f"while building sos '{name}'"):
            ctx.model.add_sos_constraints(
                ctx.model.variables[sos.variable],
                sos_type=sos.sos_type,
                sos_dim=sos.over,
                big_m=sos.big_m,
            )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _refuse_what_the_lane_cannot_build(p: program.Program) -> None:
    """Refuse a construct the language accepts and this lane cannot build, before linopy is asked.

    What the lane lacks is :data:`lpspec.lanes.LANES`'s to say.
    """
    if missing := LANES['linopy'].missing(required(p)):
        raise LaneError(lane_cannot_build_message('linopy', missing))


def _build_constraints(ctx: EvaluationContext) -> None:
    _refuse_what_the_lane_cannot_build(ctx.program)
    for name, row in ctx.program.constraints.items():
        with note(f"while building constraint '{name}'"):
            mask = evaluate_where(row.where, ctx)
            context = f"constraint '{name}'"

            check_divisors_cover(context, (row.lhs, row.rhs), ctx, mask)
            check_constant_side_covers(context, row, ctx, mask)

            lhs = evaluate_expression(row.lhs, ctx)
            rhs = evaluate_expression(row.rhs, ctx)
            if _term_free(lhs) and _term_free(rhs):
                continue

            term, other, sense = _sides(lhs, rhs, row.sense)
            ctx.model.add_constraints(term, _SIGN_MAP[sense], other, name=name, mask=as_linopy_mask(mask))


#: What reading a comparison from its other side does to it.
_FLIPPED: dict[program.ConstraintSense, program.ConstraintSense] = {'==': '==', '<=': '>=', '>=': '<='}


def _sides(lhs: Any, rhs: Any, sense: program.ConstraintSense) -> tuple[Any, Any, program.ConstraintSense]:
    """The comparison with a term on the left, which is the only side linopy takes one on.

    Either side may carry the terms — ``cap >= p`` and ``p <= cap`` both build
    — and ``add_constraints`` accepts an expression as its ``lhs`` alone,
    answering anything else with a ``TypeError`` naming a linopy type. Reading
    the row from the other side reverses the comparison.

    Reached only once a side is known to carry a term, so the ``rhs`` returned
    where the ``lhs`` is term-free is the one that does.
    """
    if not _term_free(lhs):
        return lhs, rhs, sense
    return rhs, lhs, _FLIPPED[sense]


def _term_free(side: Any) -> bool:
    """Whether *side* has nowhere for a variable term to sit.

    A bare ``Variable`` is a term; a ``LinearExpression`` over an empty axis has
    a term dimension of length zero; anything else is data. Both sides
    term-free is a constraint the *data* emptied — a dimension with no members
    reduces away to a number — which the absence rules say is not a row. An
    expression naming no variable to begin with is refused at load.
    """
    if hasattr(side, 'to_linexpr'):
        return False
    return getattr(side, 'nterm', 0) == 0


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _build_objective(ctx: EvaluationContext) -> None:
    """Build the declared objective, if any, onto the model.

    An objective has no ``where``, so its divisor check runs with no row mask.
    The expression is scalar by the time it gets here, the language having
    refused one carrying dims, so it is evaluated like any other; linopy's
    ``*`` answers a product of two variables with a ``QuadraticExpression`` on
    its own.
    """
    odef = ctx.program.objective
    if odef is None:
        return
    with note('while building the objective'):
        check_divisors_cover('the objective', (odef.expression,), ctx, None)

        expr = evaluate_expression(odef.expression, ctx)
        _refuse_an_objective_constant(expr)

        ctx.model.add_objective(expr, overwrite=True, sense=_LINOPY_SENSE[odef.sense])


#: The objective sense as linopy spells it.
_LINOPY_SENSE: dict[program.ObjectiveSense, str] = {'minimize': 'min', 'maximize': 'max'}


#: The one construct this lane accepts and cannot build, as the sentence a user reads.
OBJECTIVE_CONSTANT_IS_A_LANE_GAP = (
    "the objective carries a constant term, and this lane cannot build one: linopy's objective "
    'takes no constant. The relational lane builds it and returns the right number, so the spec '
    'is sayable and only this lane is short: run it with `lpspec.solve` / `lpspec.build`. '
    'Dropping the constant here is refused deliberately — it would answer a different model.'
)


def _refuse_an_objective_constant(expr: Any) -> None:
    """Refuse an objective this lane cannot build, before linopy is asked."""
    const = getattr(expr, 'const', None)
    if const is not None and bool(np.any(np.asarray(const) != 0)):
        raise LaneError(OBJECTIVE_CONSTANT_IS_A_LANE_GAP)
