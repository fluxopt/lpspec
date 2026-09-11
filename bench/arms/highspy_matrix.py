"""The floor under the `highs` sink: a matrix, hand-written, into `highspy`.

`gurobipy-matrix` is the denominator the `gurobi` tables have — raw solver API,
no modelling layer, so a reader can tell *"1.15x faster than linopy"* from
*"1.15x, of which 94% is Gurobi"*. The `highs` tables had no such arm, and HiGHS
is the sink the page leads with, so every ratio there was a number nobody could
put a floor under. This is that floor.

It hands the model over in one `addCols` and one `addRows`, where our own sink
chunks: bounding residency is the engine's discipline, and the floor exists to
have none. What it costs is the irreducible price of emitting the coefficients.

**It is not a fifth opinion about the model.** It builds the same `Lp` as
`gurobipy-matrix`, out of the same `bench/models/<case>/matrix.py`, so the two
floors differ in the solver they load rather than in the matrix they load it
with. `bench/floor.py` answers the same question for `transport` alone and by
hand, with a faster tiled CSR of its own; this is the arm that puts the answer
in the published tables for every case.

`highspy` needs no guard in `REQUIRES`: it is a hard dependency of the package
under test, so an environment that can run the `lpspec` arm can run this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

#: Where this arm can hand a model over. One sink, and it is the point of the
#: arm: an LP file would measure a writer rather than a load.
SINKS = ('highs',)

#: Nothing to check — see the module docstring.
REQUIRES = ()

#: Which formulation module in `bench/models/<case>/` this arm builds from.
DIALECT = 'highspy-matrix'


class Prepared(NamedTuple):
    """What the timed verbs need. The parquet is *not* read here — reading it is
    this arm's own cost, exactly as it is every other arm's."""

    case_name: str
    paths: dict[str, str]


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    del size, options
    return Prepared(case_name, dict(paths))


def _row_bounds(lp: Any) -> tuple[Any, Any]:
    """A sense per row as the pair of bounds HiGHS takes.

    An equality pins both bounds to the right-hand side and a `'<'` leaves the
    lower one open. There is no third sense in any case here, and a fourth
    spelling arriving silently would be a different model measured under this
    one's name, so it raises.
    """
    import highspy
    import numpy as np

    unknown = set(np.unique(lp.senses)) - {'=', '<'}
    if unknown:
        raise ValueError(f'{DIALECT} has no row bounds for sense(s) {sorted(unknown)} — only "=" and "<"')
    return np.where(lp.senses == '=', lp.rhs, -highspy.kHighsInf), lp.rhs


def _built(prepared: Prepared) -> Any:
    """A populated `highspy.Highs`, `run()` never called — the whole timed body."""
    import highspy
    import numpy as np
    import polars as pl

    from bench.models import formulation

    tables = {name: pl.read_parquet(path) for name, path in prepared.paths.items()}
    lp = formulation(prepared.case_name, DIALECT).build(tables)
    matrix = lp.matrix.tocsr()
    lower, upper = _row_bounds(lp)

    empty_index = np.empty(0, dtype=np.int32)
    empty_value = np.empty(0, dtype=np.float64)
    highs = highspy.Highs()
    highs.setOptionValue('output_flag', False)
    highs.addCols(len(lp.obj), lp.obj, lp.lower, lp.upper, 0, empty_index, empty_index, empty_value)
    highs.addRows(
        len(lp.rhs),
        lower,
        upper,
        matrix.nnz,
        matrix.indptr[:-1].astype(np.int32),
        matrix.indices.astype(np.int32),
        matrix.data,
    )
    return highs


def _counts(highs: Any) -> Counts:
    return {'columns': highs.getNumCol(), 'rows': highs.getNumRow(), 'nonzeros': highs.getNumNz()}


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Build the matrix and load it — for this arm those are one act.

    There is no second hand-off to time: a populated `highspy.Highs` is what
    `addCols` and `addRows` produce directly, where our own arm reaches it by
    handing a built matrix to `build_highs`. That difference is the
    measurement.
    """
    del sink
    highs = _built(prepared)
    try:
        return _counts(highs)
    finally:
        highs.clear()


def build_only(prepared: Prepared) -> Counts:
    """The same work: this arm has no sink to leave out."""
    return build_and_emit('highs', prepared)


def objective(prepared: Prepared) -> float:
    """Solve, and return what the parity gate compares.

    HiGHS carries no size-limited licence, so unlike the gurobipy arms this one
    solves the model it just built rather than writing it out for a second
    solver to read.
    """
    highs = _built(prepared)
    try:
        highs.run()
        status = highs.modelStatusToString(highs.getModelStatus())
        if status != 'Optimal':
            raise RuntimeError(f'HiGHS finished {status!r} on the {DIALECT} model, not optimal')
        return float(highs.getInfo().objective_function_value)
    finally:
        highs.clear()
