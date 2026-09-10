"""Model builder: logical plan + data → linopy Model.

**One section per kind of translation**, in the order a build performs them:
the four declarations (``Variables``, ``Special-ordered sets``,
``Constraints``, ``Objectives``) each ending in the ``model.add_*`` call they
exist to make, then ``Plan evaluation`` for what an expression is worth.

Two questions a build asks are answered beside it: ``operators.py`` evaluates
a built-in once its operands are values, and ``where.py`` turns a predicate
into the boolean array a declaration is masked by. The positions an absent
value is spelled differently in are ``absence.py``. *Which* linopy call each
construct becomes is the table in ``docs/about/linopy.md``.
"""

from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
from math_spec import program

from lpspec.errors import DataError, LaneError, LpspecError, null_bounds_message
from lpspec.lanes import LANES
from lpspec.linopy import absence
from lpspec.linopy._notes import note
from lpspec.linopy.coverage import check_constant_side_covers, check_divisors_cover, gaps_under
from lpspec.linopy.operators import operator_at, operator_grouped_sum, operator_shift, operator_sum, operator_sum_back
from lpspec.linopy.where import EvaluationContext, as_linopy_mask, bound_lookup, evaluate_where
from lpspec.relational.sinks.capabilities import lane_cannot_build_message, required

if TYPE_CHECKING:
    from collections.abc import Iterator

    import linopy
    import pandas as pd
    import xarray as xr

_SIGN_MAP = {'==': '=', '<=': '<=', '>=': '>='}


def build_model(
    model: linopy.Model,
    program: program.Program,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    lookups: dict[str, dict[str, xr.DataArray]],
) -> None:
    """Populate a linopy Model from a lowered program and loaded parameters.

    This mutates *model* in-place, adding variables, constraints and the
    objective as declared in *program*. Nothing is re-checked here: a program
    is trusted by construction, ``to_program`` having decided every rule the
    language can decide without data.
    """
    ctx = EvaluationContext(dataset, master_coords, model, lookups, program)
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

    Read raw rather than through :func:`absence.coefficient`: the absence
    rules' zero is a coefficient and never a bound, so a gap has to survive to
    :func:`_check_bounds_are_defined` instead of being filled in.
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
    """Attach every ``sos:`` block to the variable it names.

    linopy holds a set the same way the language declares one — a variable, a
    dimension of it, a type — so this is the block handed over, not a
    formulation rebuilt.
    """
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

    What the lane lacks is :data:`lpspec.lanes.LANES`'s to say; refused in the
    language's own words rather than as linopy's ``NotImplementedError``.
    """
    if missing := LANES['linopy'].missing(required(p)):
        raise LaneError(lane_cannot_build_message('linopy', missing))
    for walk in _every_walk(p):
        if not _reads_as_a_function(walk):
            raise LaneError(_walk_the_lane_cannot_read_message(walk))


def _every_walk(p: program.Program) -> Iterator[program.Walk]:
    """Every walk the program takes, whether an operator's or a partition's."""
    for node in program.walk(*p.expressions):
        if isinstance(node, program.GroupSum | program.At):
            yield from node.walks
        elif isinstance(node, program.Translate | program.Window) and node.partition is not None:
            yield node.partition


def _reads_as_a_function(walk: program.Walk) -> bool:
    """Whether this lane can read the walk as arrays indexed by one dimension.

    This lane holds a lookup as a dense array over the dimension its key is
    over, so a walk it can take reads that array and nothing else: one key
    column, nothing joined on beside it, and the two ends over different
    dimensions — a self-map's group would otherwise replace the axis it is
    grouped along.
    """
    return len(walk.key) == 1 and not walk.joined and not set(walk.consumed_dims) & set(walk.produced_dims)


def _walk_the_lane_cannot_read_message(walk: program.Walk) -> str:
    """A walk the relational engine takes and this lane does not, named by what makes it one."""
    if not walk.key:
        why = 'declares no key, so it is a relation this lane has no array for'
    elif len(walk.key) > 1:
        why = f'is keyed by {list(walk.key)}, and this lane reads a lookup at one dimension'
    elif walk.joined:
        why = f'is walked joining on {list(walk.joined)}, which this lane cannot key an array by'
    else:
        why = f'relates {walk.consumed_dims[0]!r} to itself, so the walk would replace the axis it reads along'
    return (
        f"the linopy lane cannot walk lookup '{walk.name}': it {why}. "
        f'Build this model on the relational engine — lps.solve() and lps.build() take it as it stands.'
    )


def _build_constraints(ctx: EvaluationContext) -> None:
    _refuse_what_the_lane_cannot_build(ctx.program)
    for name, row in ctx.program.constraints.items():
        with note(f"while building constraint '{name}'"):
            mask = evaluate_where(row.where, ctx)
            context = f"constraint '{name}'"

            check_divisors_cover(context, (row.lhs, row.rhs), ctx, mask)
            check_constant_side_covers(context, row, ctx, mask)

            lhs = _eval(row.lhs, ctx)
            rhs = _eval(row.rhs, ctx)
            if _term_free(lhs) and _term_free(rhs):
                continue

            term, other, sense = _sides(lhs, rhs, row.sense)
            ctx.model.add_constraints(term, _SIGN_MAP[sense], other, name=name, mask=as_linopy_mask(mask))


#: What reading a comparison from its other side does to it.
_FLIPPED: dict[program.ConstraintSense, program.ConstraintSense] = {'==': '==', '<=': '>=', '>=': '<='}


def _sides(lhs: Any, rhs: Any, sense: program.ConstraintSense) -> tuple[Any, Any, program.ConstraintSense]:
    """The comparison with a term on the left, which is the only side linopy takes one on.

    The language puts the terms on neither side — either may carry them — so a
    file writing ``cap >= p`` says what ``p <= cap`` says, and both have to build.
    ``add_constraints`` accepts an expression as its ``lhs`` alone and answers
    anything else with a ``TypeError`` naming a linopy type, so the swap
    happens here; reading the row from the other side reverses the comparison,
    which is the whole of what it costs.

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

        expr = _eval(odef.expression, ctx)
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
    """Refuse an objective this lane cannot build, before linopy is asked.

    linopy's own refusal names neither the file nor the other lane. A check
    rather than a `try`, because the upstream message is not a contract and a
    nonzero constant is the whole of what it means.
    """
    const = getattr(expr, 'const', None)
    if const is not None and bool(np.any(np.asarray(const) != 0)):
        raise LaneError(OBJECTIVE_CONSTANT_IS_A_LANE_GAP)


# ---------------------------------------------------------------------------
# Plan evaluation
# ---------------------------------------------------------------------------


def _eval(node: program.ExpressionNode, ctx: EvaluationContext) -> Any:
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
        return -_eval(node.operand, ctx)

    if isinstance(node, program.Add):
        return _eval(node.left, ctx) + _eval(node.right, ctx)

    if isinstance(node, program.Multiply):
        return _eval(node.left, ctx) * _eval(node.right, ctx)

    if isinstance(node, program.Divide):
        return _eval(node.numerator, ctx) / _eval(node.divisor, ctx)

    if isinstance(node, program.Power):
        return _eval(node.base, ctx) ** _eval(node.exponent, ctx)

    if isinstance(node, program.Sum):
        summed = _eval(node.operand, ctx)
        for dimension in node.over:
            summed = operator_sum(summed, dimension)
        return summed

    if isinstance(node, program.GroupSum):
        return operator_grouped_sum(
            _eval(node.operand, ctx),
            _walk_arrays(node.walks, ctx),
            into=node.into,
            labels=ctx.master_coords,
        )

    if isinstance(node, program.At):
        return operator_at(_eval(node.operand, ctx), _walk_arrays(node.walks, ctx), into=node.into)

    if isinstance(node, program.Translate):
        return operator_shift(
            _eval(node.operand, ctx),
            over=node.dimension,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=node.fill,
            by=_partition(node, ctx),
        )

    if isinstance(node, program.Window):
        return operator_sum_back(
            _eval(node.operand, ctx),
            over=node.dimension,
            within=_amount(node.width, ctx),
            wrap=node.wrap,
            by=_partition(node, ctx),
        )

    if isinstance(node, program.Cases):
        return _cases(node, ctx)

    assert_never(node)


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

    A **fill**, not a multiplication. Multiplying would carry the value's own
    absence out of the region that owns it: the ``otherwise`` of a commitment
    file shifts with no fill and so has nothing at the first snapshot, which
    times a false mask is still nothing rather than zero, and the row the
    other regions do cover would be unmade by a region that does not claim it.
    Inside the mask absence still stands. A bare number is the one value with
    no absence to protect, so there the mask multiplies.
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
    filled = (_in_region(_eval(region.value, ctx), evaluate_where(region.when, ctx)) for region in node.regions)
    return functools.reduce(operator.add, filled)


def _amount(amount: int | str, ctx: EvaluationContext) -> Any:
    """An offset or a width: the number, or the integer parameter naming it.

    Read through :func:`absence.coefficient` like any other parameter — a step
    nobody supplied is a step of nothing, which is what a zero offset means.
    """
    return absence.coefficient(ctx.dataset[amount]) if isinstance(amount, str) else amount


def _partition(node: program.Translate | program.Window, ctx: EvaluationContext) -> Any:
    """The group a windowed operator may not reach across, as the lookup column that makes it.

    **Named for the dimension the group's labels are of**, not for the lookup:
    an amount declared over that dimension is read through this array by
    :func:`~lpspec.linopy.operators._per_group`, which pairs the two by that
    name.
    """
    if node.partition is None:
        return None
    walk = node.partition
    array = bound_lookup(walk.name, walk.produced[0], ctx.lookups)
    return array.rename(walk.dim(walk.produced[0]))


def _walk_arrays(walks: tuple[program.Walk, ...], ctx: EvaluationContext) -> tuple[Any, ...]:
    """The value columns each walk reads, as arrays over the dimension its key is over.

    One array per dimension the walk lands the operand on, in the order the
    walks wrote them: a sum reads the columns it produces and a read the ones
    it consumes, which are the same value columns either way.
    """
    return tuple(
        bound_lookup(walk.name, role, ctx.lookups)
        for walk in walks
        for role in (walk.produced if set(walk.consumed) == set(walk.key) else walk.consumed)
    )
