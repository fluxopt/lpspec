"""The solve outcome, and linopy as its oracle.

`relational/status.py` copies linopy's status vocabulary spelling for
spelling, and each solver sink copies linopy's own mapping for its solver.
Copies rot. These tests import linopy and compare, so a divergence — ours
drifting, or a linopy release moving — fails here instead of being discovered
by a user who knows one vocabulary and is handed another.

The engine itself never imports linopy (docs/about/architecture.md, hard
rule 2). Tests may — the same oracle arrangement the differential tests use
for the math.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, NamedTuple

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import NoSolutionError
from lpspec.relational.parquet import Record, _column_types
from lpspec.relational.sinks.solvers.gurobi import _CONDITION_OF_GUROBI_STATUS, _LINOPY_DIVERGENCES
from lpspec.relational.sinks.solvers.highs import _CONDITION_OF_HIGHS_STATUS
from lpspec.relational.sinks.solvers.xpress import _CONDITION_OF_SOL_STATUS
from lpspec.relational.status import STATUS_TO_TERMINATION_CONDITIONS, SolveStatus
from tests.conftest import CASES

# ---------------------------------------------------------------------------
# linopy as the oracle for the vocabulary
# ---------------------------------------------------------------------------


def test_the_status_rollup_matches_linopy():
    constants = pytest.importorskip('linopy.constants')
    theirs = {
        status.value: {condition.value for condition in conditions}
        for status, conditions in constants.STATUS_TO_TERMINATION_CONDITION_MAP.items()
    }
    assert {k: set(v) for k, v in STATUS_TO_TERMINATION_CONDITIONS.items()} == theirs


def test_the_highs_mapping_matches_linopy():
    assert _linopy_condition_map('Highs', ast.Attribute, 'attr') == _CONDITION_OF_HIGHS_STATUS


def test_the_gurobi_mapping_matches_linopy_where_it_claims_to():
    """The copy, minus three declared exceptions.

    linopy's Gurobi map contradicts Gurobi's own documented status codes in
    three places, each listed in ``_LINOPY_DIVERGENCES`` with its reason.
    Asserted in both directions: everything else still matches, and every
    declared divergence still diverges — so if linopy fixes one, the entry
    has to go.
    """
    theirs = _linopy_condition_map('Gurobi', ast.Constant, 'value')
    assert set(theirs) == set(_CONDITION_OF_GUROBI_STATUS), (
        'linopy and this package no longer cover the same Gurobi status codes'
    )
    for code, condition in theirs.items():
        if code in _LINOPY_DIVERGENCES:
            assert _CONDITION_OF_GUROBI_STATUS[code] != condition, (
                f'linopy now agrees with us on status {code} — drop the entry from _LINOPY_DIVERGENCES'
            )
        else:
            assert _CONDITION_OF_GUROBI_STATUS[code] == condition


def test_the_xpress_mapping_matches_linopy():
    """Copied entry for entry, with nothing claimed as an exception.

    The map is keyed by ``SolStatus`` *value* here and by the enum member
    there, the sink not being allowed to import xpress at module level — so
    the enum is what the two are compared through, and a member renamed
    upstream fails here rather than silently dropping an entry.
    """
    xpress = pytest.importorskip('xpress', reason='the xpress sink needs the [xpress] extra')
    theirs = _linopy_condition_map('Xpress', ast.Attribute, 'attr', ast.Constant, 'value')
    assert theirs, 'linopy no longer spells its Xpress map as SolStatus attributes against strings'
    assert {int(xpress.SolStatus[name]): condition for name, condition in theirs.items()} == _CONDITION_OF_SOL_STATUS


def test_the_xpress_sink_adds_to_linopys_answer_rather_than_contradicting_it():
    """The xpress sink reads a second axis linopy never looks at, so there is
    nothing to disagree with — and the word it reports still has to be one
    linopy defines.
    """
    every = set().union(*STATUS_TO_TERMINATION_CONDITIONS.values())
    assert set(_CONDITION_OF_SOL_STATUS.values()) <= every
    assert 'internal_solver_error' in every, 'the condition the second axis reports is linopy vocabulary'


def test_every_gurobi_divergence_stays_inside_linopys_vocabulary():
    """Diverging on a verdict is not licence to invent a word for it. Every
    condition this package reports is one linopy also defines, which is what
    keeps `status`, `is_ok` and the rollup meaningful across both."""
    assert set(_CONDITION_OF_GUROBI_STATUS.values()) <= set().union(*STATUS_TO_TERMINATION_CONDITIONS.values())


def _linopy_condition_map(
    solver: str,
    node: type[ast.expr],
    attribute: str,
    value_node: type[ast.expr] | None = None,
    value_attribute: str | None = None,
) -> dict[Any, Any]:
    """linopy's ``CONDITION_MAP`` for *solver*, read out of its source.

    Each solver spells the map differently — HiGHS keys it by
    ``HighsModelStatus`` attributes, Gurobi by integer literals, Xpress by
    ``SolStatus`` attributes against plain strings — so the node type and the
    attribute holding the value are arguments, and the two sides may differ.

    Brittle to a linopy refactor, deliberately: the map is a local inside a
    method, so there is nothing to import, and a copy nobody checks is a copy
    that rots. If linopy moves it, the assertions say so rather than passing
    vacuously.
    """
    solvers = pytest.importorskip('linopy.solvers')
    tree = ast.parse(inspect.getsource(solvers))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == solver), None)
    assert cls is not None, f'linopy no longer has a {solver} solver class — re-verify the copy by hand'
    literals = [
        n.value
        for n in ast.walk(cls)
        if isinstance(n, (ast.AnnAssign, ast.Assign)) and 'CONDITION_MAP' in ast.dump(n)
        if isinstance(n.value, ast.Dict)
    ]
    assert literals, f'linopy no longer defines {solver}.CONDITION_MAP as a dict literal — re-verify by hand'
    values, value_of = value_node or node, value_attribute or attribute
    return {
        getattr(key, attribute): getattr(value, value_of)
        for key, value in zip(literals[0].keys, literals[0].values, strict=True)
        if isinstance(key, node) and isinstance(value, values)
    }


# ---------------------------------------------------------------------------
# what the two axes mean here
# ---------------------------------------------------------------------------


def test_ok_means_values_worth_reading_not_optimality():
    """A run stopped at a time limit still has an incumbent."""
    assert SolveStatus('optimal').is_ok
    assert SolveStatus('time_limit').is_ok
    assert SolveStatus('suboptimal').is_ok
    assert not SolveStatus('infeasible').is_ok
    assert not SolveStatus('unbounded').is_ok


def test_an_infeasible_solve_reports_both_axes_and_a_nan_objective():
    with lps.solve(*CASES['INFEASIBLE']) as solution:
        assert solution.status == 'warning'
        assert solution.termination_condition == 'infeasible'
        assert not solution.is_ok
        assert solution.objective != solution.objective, 'nan, not 0.0'


def test_reading_results_without_a_solution_raises():
    """HiGHS returns a full-length vector of zeros whatever the status, so
    handing it back would be indistinguishable from an answer."""
    with lps.solve(*CASES['INFEASIBLE']) as solution:
        with pytest.raises(NoSolutionError, match='infeasible'):
            solution.primal('p')
        with pytest.raises(NoSolutionError, match='infeasible'):
            solution.dual('meet')


def test_a_solve_that_left_no_values_writes_the_record_and_no_frames(tmp_path):
    """An export of a run that did not solve is the record alone.

    Was: it raised, so a variant that came back infeasible left nothing on
    disk and could not be told apart from one nobody ran. Reading a value
    still raises — there is none — and that is the test above.
    """
    with lps.solve(*CASES['INFEASIBLE']) as solution:
        out = solution.save(tmp_path / 'infeasible')
    assert sorted(entry.name for entry in out.iterdir()) == ['format.json', 'objective.parquet'], (
        'the record and the layout it is in; no values, so no primal/, dual/ or expression/'
    )
    record = pl.read_parquet(out / 'objective.parquet')
    assert record.row(0, named=True)['termination_condition'] == 'infeasible'
    assert record['objective'].to_list() == [None], 'no objective was reached, so the column holds none'


def test_a_case_that_reached_no_objective_does_not_poison_the_others(tmp_path):
    """A directory per case is a table, and in a table an absent number is null.

    nan is a *number* to every aggregate that meets it: one infeasible case
    among a hundred turns the mean of the hundred into nan, in polars and in
    any SQL engine reading the same files. `has_primal` already says which
    rows reached an objective, so the column has nothing to spend a sentinel
    on.
    """
    for name, case in (('solved', 'LP'), ('unsolved', 'INFEASIBLE')):
        with lps.solve(*CASES[case]) as solution:
            solution.save(tmp_path / name)

    table = pl.read_parquet(tmp_path / '*' / 'objective.parquet')
    assert table['objective'].null_count() == 1, 'one of the two cases reached no objective'
    assert table['objective'].is_nan().sum() == 0, 'and it is written as no value rather than as nan'
    assert table['objective'].mean() == table.filter('has_primal')['objective'].item(), (
        'so the mean over the cases is the mean over the ones that solved'
    )


def test_a_record_column_that_names_no_written_type_is_refused_at_import():
    """The schema is derived from `Record`, so a column added to it cannot skip declaring one.

    Restated by hand it could: the next nullable column would go back to the
    type polars infers from a single row — the defect the schema exists to
    close, reintroduced with a green suite and nothing to show it.
    """

    class Unwritable(NamedTuple):
        when: bytes

    with pytest.raises(lps.LpspecError, match='_WRITTEN_AS'):
        _column_types(Unwritable)

    assert tuple(_column_types(Record)) == Record._fields, 'and Record itself derives all of its own'


def test_a_case_with_no_spec_digest_concatenates_with_one_that_has_it(tmp_path):
    """The same claim on the other nullable column, which is the record's own.

    A solve run off a lowered program has no document to digest, so its
    record's `spec_digest` is absent. Inferred from the row it would be a
    `Null` column rather than an empty `String` one, and concatenating cases
    solved apart is what the record is written for: `Null` first refuses the
    string that follows it, and string first widens. An order the reader
    happens to pick is not a schema. The columns are declared instead, so an
    absence is that column's own type holding none.
    """
    spec, sources = CASES['LP']
    for name, model in (('document', spec), ('lowered', lps.check(spec))):
        with lps.solve(model, sources) as solution:
            solution.save(tmp_path / name)

    each = {name: pl.read_parquet(tmp_path / name / 'objective.parquet') for name in ('document', 'lowered')}
    assert [frame.schema['spec_digest'] for frame in each.values()] == [pl.String, pl.String], (
        'a string column wherever it is written, whether or not this solve named a document'
    )
    assert each['lowered']['spec_digest'].to_list() == [None], 'a solve off a lowered program names none'
    both = pl.concat([each['lowered'], each['document']])
    assert both['spec_digest'].null_count() == 1, 'and the two concatenate whichever is read first'


# ---------------------------------------------------------------------------
# solver options, and the incumbent question they make reachable
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def knapsack():
    """A MIP big enough that HiGHS does not finish it instantly."""
    import random

    random.seed(0)
    n = 60
    weights = [random.randint(10**6, 2 * 10**6) for _ in range(n)]
    spec = {
        'dimensions': {'i': {'dtype': 'int'}, 'one': {'dtype': 'int'}},
        'parameters': {'w': {'dims': ['i']}, 'cap': {'dims': ['one']}},
        'variables': {'x': {'foreach': ['i'], 'domain': 'binary'}},
        'constraints': {'budget': {'foreach': ['one'], 'expression': 'sum(x * w, over=i) <= cap'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x * w, over=i)'},
    }
    sources = {
        'i': list(range(n)),
        'one': [0],
        'w': pl.DataFrame({'i': list(range(n)), 'value': [float(v) for v in weights]}),
        'cap': pl.DataFrame({'one': [0], 'value': [float(sum(weights) // 2)]}),
    }
    return spec, sources


def test_solver_options_reach_the_solver(knapsack):
    """Forwarded verbatim, the way linopy's are. `time_limit=0` is the cheapest
    proof: without it this model solves to optimality."""
    spec, sources = knapsack
    with lps.solve(spec, sources, solver_options={'time_limit': 0.0}) as result:
        assert result.termination_condition == 'time_limit'
    with lps.solve(spec, sources) as result:
        assert result.termination_condition == 'optimal'


def test_a_time_limit_with_no_incumbent_is_ok_but_unreadable(knapsack):
    """The gap `is_ok` alone cannot see, and where we go beyond linopy.

    A MIP stopped before it found any feasible point rolls up to `ok` —
    linopy's `safe_get_solution` would read its zero-filled `col_value` as an
    answer. `has_primal` carries the solver's own verdict instead.
    """
    spec, sources = knapsack
    with lps.solve(spec, sources, solver_options={'time_limit': 0.0}) as result:
        assert result.is_ok, "linopy's rollup says the run was not an error"
        assert not result.has_primal, 'but nothing was found'
        assert result.objective != result.objective, 'nan, not 0.0'
        with pytest.raises(NoSolutionError, match='time_limit'):
            result.primal('x')


def test_an_optimal_solve_is_both_ok_and_readable(knapsack):
    spec, sources = knapsack
    with lps.solve(spec, sources) as result:
        assert result.is_ok
        assert result.has_primal
        assert result.primal('x')['value'].sum() > 0
