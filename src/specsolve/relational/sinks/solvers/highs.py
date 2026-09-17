"""The ``highs`` solver: the whole model straight into HiGHS, in one call.

The default, and the only one whose dependency ships with the package. Every
vector crosses as a numpy buffer, with no float→text→parse round trip.

**Nothing textual crosses into numpy**: a row's ``'<='`` becomes a
:data:`~specsolve.relational.sinks.tables.SENSE_CODES` byte before it is read
here.

``highspy`` is imported inside the function, being optional: importing this
module stays free for callers that only write LP files.

:class:`Highs` is the same hand-off held open — what a driver that re-solves
one model with new numbers uses, and where the warm basis lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.solvers.base import SolveAnswer, Solver, WarmStart
from specsolve.relational.sinks.tables import SENSE_CODES, solver_vector
from specsolve.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    from specsolve.relational.sinks.tables import RowVectors, Tables


#: HiGHS model status -> termination condition. Copied from linopy's own
#: ``Highs.CONDITION_MAP``; ``tests/test_solve_status.py`` asserts it still
#: matches.
_CONDITION_OF_HIGHS_STATUS = {
    'kNotset': 'unknown',
    'kLoadError': 'internal_solver_error',
    'kModelError': 'internal_solver_error',
    'kPresolveError': 'internal_solver_error',
    'kSolveError': 'internal_solver_error',
    'kPostsolveError': 'internal_solver_error',
    'kModelEmpty': 'unknown',
    'kMemoryLimit': 'resource_interrupt',
    'kOptimal': 'optimal',
    'kInfeasible': 'infeasible',
    'kUnboundedOrInfeasible': 'infeasible_or_unbounded',
    'kUnbounded': 'unbounded',
    'kObjectiveBound': 'terminated_by_limit',
    'kObjectiveTarget': 'terminated_by_limit',
    'kTimeLimit': 'time_limit',
    'kIterationLimit': 'iteration_limit',
    'kSolutionLimit': 'terminated_by_limit',
    'kInterrupt': 'user_interrupt',
    'kUnknown': 'unknown',
}


def build_highs(
    tables: Tables,
    solver_options: Mapping[str, Any] | None = None,
) -> Highs:
    """Load the model into a :class:`highspy.Highs` and stop there.

    The hand-off without the simplex. `bench/` ends here, as linopy's
    ``Model.to_highspy()`` does on that side.

    Returns:
        The :class:`Highs` holding the model, at ``.handle``.
    """
    return Highs(tables, None, solver_options)


def _built(tables: Tables, solver_options: Mapping[str, Any] | None) -> Any:
    """The populated :class:`highspy.Highs`.

    One ``passModel`` loads the whole model at once — the scalars, the five
    dense vectors, and the matrix as row-wise CSR. Every array crosses as a
    numpy buffer.

    The integrality vector spans the whole index even where no column is
    integer. HiGHS reads it either way, and an empty one is read as whatever
    the memory held (1.15.1).
    """
    import highspy
    import numpy as np

    if tables.qmatrix.height:
        raise SpecsolveError(
            'HiGHS has no quadratic-constraint concept at all — no entry point takes one — and '
            f'this model has {tables.row_count - tables.linear_row_count} such rows. Solving through '
            'sps.solve() '
            'refuses this earlier and names the sinks that do take it; reaching build_highs '
            'directly skips that, and loading the rows without their quadratic part would be a '
            'different model that solves.'
        )

    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    for option, value in (solver_options or {}).items():
        h.setOptionValue(option, value)

    cols = tables.dense_columns(inf)
    rlb, rub = _row_bounds(tables.dense_rows(inf), inf)
    sense = highspy.ObjSense.kMaximize if tables.objective_sense == 'maximize' else highspy.ObjSense.kMinimize
    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    _loaded(
        h,
        h.passModel(
            tables.column_count,
            tables.row_count,
            tables.matrix.height,
            0,
            int(highspy.MatrixFormat.kRowwise),
            int(highspy.HessianFormat.kTriangular),
            int(sense),
            0.0,
            cols.cost,
            cols.lb,
            cols.ub,
            rlb,
            rub,
            tables.row_starts.astype(np.int32),
            tables.matrix['col'].to_numpy(),
            tables.matrix['coeff'].to_numpy(),
            empty_i,
            empty_i,
            empty_f,
            _integrality(cols),
        ),
        'the model',
    )
    _pass_hessian(h, tables)
    return h


def _integrality(cols: Any) -> Any:
    """The per-column integrality vector HiGHS reads: 0 continuous, 1 integer."""
    import numpy as np

    return cols.integral.astype(np.int32)


def _pass_hessian(h: Any, tables: Tables) -> None:
    r"""The objective's quadratic part, as the Hessian HiGHS reads.

    ``passHessian`` takes :math:`Q` in :math:`\frac12 x^\top Q x`, lower
    triangle only, in column-major (CSC) order — so the conversion from the
    unordered-pair form the engine hands over is two rules, and they differ:

    * a **diagonal** pair states :math:`q\,x_i^2`, and :math:`\frac12 Q_{ii}
      x_i^2 = q\,x_i^2` needs :math:`Q_{ii} = 2q`;
    * an **off-diagonal** pair states :math:`q\,x_i x_j` once, where the
      symmetric matrix holds it twice — :math:`\frac12 (Q_{ij} + Q_{ji}) = q`
      — so the stored value is :math:`q` itself.

    The whole part goes over at once — there is no incremental Hessian API —
    but onto the model already loaded, which is what lets :meth:`Highs.push`
    replace it without a reload.
    """
    import highspy
    import numpy as np

    if not tables.quad.height:
        return
    lower = tables.quad['col_r'].to_numpy().astype(np.int32, copy=False)
    upper = tables.quad['col_l'].to_numpy().astype(np.int32, copy=False)
    diagonal = lower == upper
    values = np.where(diagonal, tables.quad['coeff'].to_numpy() * 2.0, tables.quad['coeff'].to_numpy())

    order = np.lexsort((lower, upper))
    starts = np.zeros(tables.column_count + 1, dtype=np.int32)
    np.add.at(starts, upper + 1, 1)
    _loaded(
        h,
        h.passHessian(
            tables.column_count,
            len(order),
            int(highspy.HessianFormat.kTriangular),
            np.cumsum(starts, out=starts),
            lower[order],
            values[order],
        ),
        'the quadratic objective',
    )


class Highs(Solver):
    """HiGHS, holding one model — :class:`Solver`'s member for the default sink.

    The second solve of an updated model changes bounds, costs and right-hand
    sides on the model HiGHS already holds and starts from the basis the last
    solve ended on, unless the caller carries the basis across with
    :meth:`warm_start` and
    :meth:`~specsolve.relational.sinks.solvers.base.Solver.warm`.
    """

    #: The loaded model. ``close`` drops it.
    _handle: Any

    requires = ('highspy',)
    unavailable_message = 'highspy ships with specsolve, so a build without it is broken rather than missing an extra'

    #: No SOS concept at all, so a set arrives already written as binaries and
    #: linking rows. A *convex* Hessian goes in through ``passHessian``, and
    #: the pair is probed in ``test_sink_capability_probes.py``. A set cannot
    #: stand beside a Hessian: the rewrite that gets one in here *is* binaries.
    capabilities = Capabilities(
        supports={
            'integrality': 'native',
            'sos': 'reformulated',
            'quadratic_objective': 'native',
        },
        excludes=(
            frozenset({'quadratic_objective', 'integrality'}),
            frozenset({'quadratic_objective', 'sos'}),
        ),
    )

    def _load(self, tables: Tables, batch_rows: int | None) -> None:
        """Load in one call — *batch_rows* is the family's parameter and this member has no batches."""
        del batch_rows
        self._handle = _built(tables, self._options)

    @property
    def handle(self) -> Any:
        return self._handle

    def push(self, tables: Tables) -> None:
        """The index vectors are built here rather than held."""
        import highspy
        import numpy as np

        inf = highspy.kHighsInf
        cols = tables.dense_columns(inf)
        columns = np.arange(tables.column_count, dtype=np.int32)
        _loaded(self._handle, self._handle.changeColsCost(tables.column_count, columns, cols.cost), 'new costs')
        _loaded(
            self._handle, self._handle.changeColsBounds(tables.column_count, columns, cols.lb, cols.ub), 'new bounds'
        )

        rows = np.arange(tables.row_count, dtype=np.int32)
        rlb, rub = _row_bounds(tables.dense_rows(inf), inf)
        _loaded(self._handle, self._handle.changeRowsBounds(tables.row_count, rows, rlb, rub), 'new right-hand sides')
        _pass_hessian(self._handle, tables)

    def warm_start(self) -> WarmStart | None:
        """The basis the last solve left, or its incumbent where none is valid.

        A solved MIP is the model that holds an answer but no valid basis —
        ``getBasis().valid`` is false — so what crosses is ``col_value`` as an
        incumbent. A model not yet solved holds neither.
        """
        import numpy as np

        basis = self._handle.getBasis()
        if basis.valid:
            return WarmStart(
                solver='highs',
                column_statuses=np.fromiter((int(status) for status in basis.col_status), dtype=np.int8),
                row_statuses=np.fromiter((int(status) for status in basis.row_status), dtype=np.int8),
                column_values=None,
            )
        if _has_primal(self._handle):
            values = np.asarray(self._handle.getSolution().col_value, dtype=np.float64)
            return WarmStart(solver='highs', column_statuses=None, row_statuses=None, column_values=values)
        return None

    def _warm(self, ws: WarmStart) -> None:
        """``setBasis`` for a basis, ``setSolution`` for an incumbent.

        Both report a refusal by return value, like every hand-off here, so
        both go through :func:`_took`.
        """
        import highspy

        if (statuses := ws.basis()) is not None:
            column_statuses, row_statuses = statuses
            basis = highspy.HighsBasis()
            basis.col_status = [highspy.HighsBasisStatus(int(status)) for status in column_statuses]
            basis.row_status = [highspy.HighsBasisStatus(int(status)) for status in row_statuses]
            basis.valid = True
            _took(self._handle.setBasis(basis), 'the carried basis')
        else:
            assert ws.column_values is not None, (
                'a warm start with no basis carries an incumbent — it holds nothing else'
            )
            solution = highspy.HighsSolution()
            solution.col_value = [float(value) for value in ws.column_values]
            _took(self._handle.setSolution(solution), 'the carried incumbent')

    def _run(self, tables: Tables) -> SolveAnswer:
        """Solve, and read the one error HiGHS reports as a refusal to start.

        A ``kError`` from ``run()`` leaves the model status unset — there is no
        solve to read back — so a quadratic model that gets one is refused
        explicitly. The pair a Hessian is otherwise refused for, integrality
        beside it, is declared on the descriptor and never reaches a load.
        """
        import highspy

        if self._handle.run() == highspy.HighsStatus.kError and tables.quad.height:
            raise SpecsolveError(
                'the highs sink refused to run this quadratic objective, and a Hessian that is not '
                'positive semidefinite is why it refuses one: it solves convex QPs only. Convexity is a '
                'property of the coefficients rather than of the model, so nothing could refuse it '
                "before the data was attached — the sink's other quadratic refusal, a Hessian standing "
                'beside integrality, is declared and caught before the build.\n'
                'Solve with a sink whose capabilities list a nonconvex quadratic objective as native — '
                'check(spec, sink=...) names them — or write the model to an .lp file for a solver that '
                'takes one. A convex reformulation — the curve as a piecewise: block with '
                'method: convex — keeps the LP, and with it the duals and the warm start a quadratic '
                'objective gives up.'
            )
        status = _status_of(self._handle)
        if not status.is_readable:
            return SolveAnswer.unreadable(status)

        objective = self._handle.getInfo().objective_function_value + tables.objective_constant
        solution = self._handle.getSolution()
        primal = solver_vector(solution.col_value)
        dual = solver_vector(solution.row_dual) if solution.dual_valid else None
        activity = solver_vector(solution.row_value)
        return SolveAnswer(status, objective, primal, dual, activity)

    def forget(self) -> None:
        """``clearSolver``: the basis and the solution go, the model stays.

        HiGHS skips presolve for a run that starts from a basis.
        """
        self._handle.clearSolver()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.clear()
        self._handle = None


def _row_bounds(rows: RowVectors, inf: float) -> tuple[Any, Any]:
    """HiGHS's ``(lower, upper)`` spelling of a sense code and right-hand side.

    An inequality is open on the side its sense does not bound.
    """
    import numpy as np

    return (
        np.where(rows.sense == SENSE_CODES['<='], -inf, rows.rhs),
        np.where(rows.sense == SENSE_CODES['>='], inf, rows.rhs),
    )


def _loaded(h: Any, status: Any, what: str) -> None:
    """Raise unless the solver accepted the hand-off.

    HiGHS reports a rejected call by return value and carries on with whatever
    it had.

    Raises:
        SpecsolveError: If the batch was refused.
    """
    import highspy

    if status == highspy.HighsStatus.kError:
        raise SpecsolveError(
            f'the solver refused {what}: {h.modelStatusToString(h.getModelStatus())!r}. '
            f'The model it holds is not the one handed over, so any answer would describe a '
            f'different one. This is an engine bug rather than a problem with the model — '
            f'please report it.'
        )


def _took(status: Any, what: str) -> None:
    """Raise unless the solver accepted a warm-start hint.

    HiGHS reports a refusal by return value and carries on.

    Raises:
        SpecsolveError: If the hint was refused.
    """
    import highspy

    if status == highspy.HighsStatus.kError:
        raise SpecsolveError(
            f'HiGHS refused {what} even though it spans the loaded model, so the solve would '
            f'silently start cold instead of warm. This is an engine bug rather than a problem '
            f'with the model — please report it.'
        )


def _status_of(h: Any) -> SolveStatus:
    """What the solve concluded, on both axes.

    ``has_primal`` is the solver's own answer to "is there anything here",
    which the termination condition does not give: a run stopped at a time
    limit may or may not have found an incumbent.
    """
    model_status = h.getModelStatus()
    return SolveStatus(
        termination_condition=_CONDITION_OF_HIGHS_STATUS.get(str(model_status).rsplit('.', 1)[-1], 'unknown'),
        solver_wording=h.modelStatusToString(model_status),
        has_primal=_has_primal(h),
    )


def _has_primal(h: Any) -> bool:
    """Whether HiGHS holds a feasible primal — the one question both the status and a warm start ask."""
    import highspy

    return h.getInfo().primal_solution_status == int(highspy.SolutionStatus.kSolutionStatusFeasible)
