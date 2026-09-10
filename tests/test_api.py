"""Native API: YAML → streaming engine → solver, with linopy never imported.

The linopy-free guarantee is asserted in a subprocess so the suite's own
oracle imports cannot pollute the check.

This module is deliberately **pandas-free**: it is the bare install's proof
that the native path — frames in, build, solve, frames out — needs no
dataframe library beyond the engine's own. The tests that exercise the bridges
*out* (``to_pandas``, ``to_dataarray``) say so with an ``importorskip``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace
from unittest import mock

import numpy as np
import polars as pl
import pytest
from math_spec import Spec, to_program, to_spec

import lpspec as lps
from lpspec.errors import DimensionError
from tests.conftest import (
    CASES,
    DISPATCH_COST,
    DISPATCH_GENERATORS,
    DISPATCH_P_MAX,
    EXAMPLES_DIR,
    _dispatch_load,
    override,
    raw_of,
    solve_written_file,
)


@pytest.fixture
def dispatch_solution(dispatch_yaml, dispatch_frame_inputs):
    """The dispatch model solved on the native lane, closed after the test."""
    sources = dispatch_frame_inputs
    with lps.solve(dispatch_yaml, sources) as result:
        yield result


def test_solve(dispatch_solution, dispatch_frame_inputs):
    sources = dispatch_frame_inputs
    assert dispatch_solution.is_ok
    assert np.isfinite(dispatch_solution.objective)
    balance = dispatch_solution.primal('p').group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
    assert np.allclose(balance['value'], sources['load']['value'])


def test_build_context_manager_and_write(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    with lps.build(dispatch_yaml, sources) as model:
        result = model.solve()
        assert result.is_ok
        objective_direct = result.objective

    lp = lps.write(dispatch_yaml, sources, tmp_path / 'm.lp')
    assert solve_written_file(lp) == pytest.approx(objective_direct, rel=1e-9)


def test_a_points_parameter_supplied_as_a_parquet_path_keeps_its_own_curve_length(tmp_path):
    """The mask a ``points:`` parameter derives was read off the caller's object before any path
    was opened, so a curve supplied as a file was held to the full breakpoint grid and refused for
    the rows a shorter curve does not have.
    """
    from tests.conftest import port_sources, port_spec

    frames = port_sources('piecewise_ragged')
    paths = {}
    for name, frame in frames.items():
        frame.write_parquet(tmp_path / f'{name}.parquet')
        paths[name] = str(tmp_path / f'{name}.parquet')

    with (
        lps.solve(port_spec('piecewise_ragged'), paths) as from_paths,
        lps.solve(port_spec('piecewise_ragged'), frames) as from_frames,
    ):
        assert from_paths.objective == pytest.approx(from_frames.objective, rel=1e-9), (
            'a curve read from a file is the curve read from a frame'
        )


def test_a_string_is_a_parquet_path_at_every_door(tmp_path):
    """One fact with one home: `as_frame` is what every reader of a source goes
    through, so a path attaches the same at a parameter, an index, a lookup,
    a curve and a sweep's axis — the `points:` curve that was refused from a
    path while accepted from a frame was the fifth reader lacking it."""
    from lpspec.frames import as_frame

    frame = pl.DataFrame({'snapshot': [0, 1], 'value': [1.0, 2.0]})
    frame.write_parquet(tmp_path / 'load.parquet')
    assert as_frame(str(tmp_path / 'load.parquet')).collect().equals(frame), 'a str is scanned as parquet'
    assert as_frame(tmp_path / 'load.parquet').collect().equals(frame), 'and so is a Path'
    assert as_frame(frame).collect().equals(frame), 'a table is normalised as before'
    assert as_frame(2.0) is None, 'a number is not a table, and the caller says what it is'


def test_parquet_path_sources(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    paths = {}
    for name, frame in sources.items():
        p = tmp_path / f'{name}.parquet'
        frame.write_parquet(p)
        paths[name] = str(p)

    with lps.solve(dispatch_yaml, paths) as result:
        assert result.is_ok
        objective = result.objective

    with lps.solve(dispatch_yaml, sources) as ref:
        assert objective == pytest.approx(ref.objective, rel=1e-9)


#: ``examples/dispatch.yaml``'s numbers written out in Python rather than
#: handed over as tables — the shapes a hand-written model reaches for.
_PLAIN = {
    'dict': {
        'p_max': dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True)),
        'cost': dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True)),
        'load': dict(enumerate(_dispatch_load())),
    },
    'sequence': {
        'p_max': list(DISPATCH_P_MAX),
        'cost': list(DISPATCH_COST),
        'load': _dispatch_load(),
    },
}


@pytest.mark.parametrize('shape', sorted(_PLAIN), ids=sorted(_PLAIN))
def test_plain_python_sources_reach_the_same_answer_as_tables(dispatch_yaml, dispatch_frame_inputs, shape):
    """A dict and a sequence are sources, and mean what the tables mean.

    A dict carries its own labels; a sequence is positional against the index,
    which is why the dimensions are resolved before any parameter is read.
    """
    frames = dispatch_frame_inputs
    with lps.solve(dispatch_yaml, frames) as tables:
        expected = tables.objective

    index = {'snapshot': frames['snapshot'], 'generator': frames['generator']}
    with lps.solve(dispatch_yaml, _PLAIN[shape] | index) as plain:
        assert plain.objective == pytest.approx(expected, rel=1e-9)


def test_one_number_stands_for_every_coordinate(dispatch_yaml, dispatch_frame_inputs):
    """A scalar covers the dims the parameter declares, not just a 0-D one.

    Dense by construction, and materialised here — which is the cost of saying
    it this way rather than declaring the parameter ``dims: []``.
    """
    frames = dispatch_frame_inputs
    flat = {**frames, 'cost': 7.0}
    spelled = {**frames, 'cost': pl.DataFrame({'generator': list(DISPATCH_GENERATORS), 'value': [7.0] * 3})}

    with (
        lps.solve(dispatch_yaml, flat) as broadcast,
        lps.solve(dispatch_yaml, spelled) as written,
    ):
        assert broadcast.objective == pytest.approx(written.objective, rel=1e-9)


@pytest.mark.parametrize(
    ('sources', 'match'),
    [
        pytest.param({'p_max': [100.0, 60.0]}, 'one entry per label', id='a-sequence-of-the-wrong-length'),
        pytest.param({'p_max': object()}, 'cannot adapt', id='nothing-table-shaped-at-all'),
        pytest.param({'snapshot': 3.0}, 'cannot read labels out of float', id='an-index-that-is-one-number'),
    ],
)
def test_a_plain_python_source_that_does_not_fit_is_refused(dispatch_yaml, dispatch_frame_inputs, sources, match):
    frames = dispatch_frame_inputs
    with pytest.raises(lps.DataError, match=match):
        lps.build(dispatch_yaml, {**frames, **sources}).close()


#: One parameter over two dims — what a dict and a sequence cannot cover.
_TWO_DIMS = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['g', 't']}},
    'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


@pytest.mark.parametrize(
    ('source', 'match'),
    [
        pytest.param({('wind', 0): 1.0}, 'a dict maps one label to one value', id='a-dict'),
        pytest.param([1.0, 2.0, 3.0, 4.0], 'a sequence runs along one dimension', id='a-sequence'),
    ],
)
def test_a_flat_shape_cannot_cover_two_dimensions(source, match):
    """Both carry one axis, and the rewrite is the table that carries both."""
    with pytest.raises(lps.DataError, match=match):
        lps.build(_TWO_DIMS, {'cap': source}).close()


def test_a_one_level_series_cannot_cover_two_dimensions():
    """A pandas Series is a sequence with its index along: one axis, declined the same way."""
    pandas = pytest.importorskip('pandas')
    series = pandas.Series([1.0, 2.0], index=pandas.Index(['wind', 'gas'], name='g'))
    with pytest.raises(lps.DataError, match='a sequence runs along one dimension'):
        lps.build(_TWO_DIMS, {'cap': series}).close()


def test_a_positional_source_needs_the_labels_it_is_written_against():
    """A sequence says what the values are and not what they are labelled, and no
    lane reads labels off the parameters."""
    spec = {
        'dimensions': {'g': {}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x, over=g)'},
    }
    with pytest.raises(lps.DataError, match='nothing else supplies an index'):
        lps.build(spec, {'cap': [1.0, 2.0]}).close()


def test_runtime_is_linopy_free(dispatch_yaml):
    """Import the package, build and solve on Arrow sources — linopy never loads.

    pandas and pyarrow are on the list too, and that is newer than it looks:
    on the duckdb engine they could not be, because duckdb imported pandas
    opportunistically when registering any Python object, so "not in
    ``sys.modules``" was not a claim this package could keep. polars imports
    neither until asked, so the stronger claim is now available and is pinned
    here — a bridge out (``to_pandas``, ``to_dataarray``) must stay a bridge
    and never become something the build path walks over on its own.

    Distinct from, and weaker than, the claim that they need not be
    *installed*: the bare-install CI job is what proves that, running this
    suite with no dataframe library beyond polars present at all.
    """
    absent = ('linopy', 'xarray', 'pandas', 'pyarrow')
    script = textwrap.dedent(f"""
        import sys
        assert "linopy" not in sys.modules

        import polars as pl
        import lpspec as lps
        for lib in {absent!r}:
            assert lib not in sys.modules, f"package import pulled in {{lib}}"

        result = lps.solve(
            {str(dispatch_yaml)!r},
            {{
                "p_max": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                       "value": [100.0, 60.0, 200.0]}}),
                "cost": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                      "value": [1.0, 2.0, 50.0]}}),
                "load": pl.DataFrame({{"snapshot": [0, 1, 2],
                                      "value": [80.0, 120.0, 150.0]}}),
                "snapshot": range(3),
                "generator": ["wind", "solar", "gas"],
            }},
        )
        assert result.is_ok
        assert isinstance(result.primal("p"), pl.DataFrame), "no dataframe on either side"
        assert result.primal("p").height == 9
        result.close()
        for lib in {absent!r}:
            assert lib not in sys.modules, f"solve pulled in {{lib}}"
        print("LINOPY_FREE_OK")
    """)
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert 'LINOPY_FREE_OK' in out.stdout


@pytest.mark.parametrize(
    'form',
    ['path', 'str', 'dict', 'spec', 'program'],
)
def test_every_verb_opens_a_model_the_way_the_language_does(dispatch_yaml, dispatch_frame_inputs, tmp_path, form):
    """One first argument across the five verbs, and it is `to_program`'s own.

    A caller who has already read the file — `to_spec` for the math, `check`
    for the plan — hands that back rather than the path, and every verb takes
    it. Asserted per verb rather than on `check` alone: each annotates
    `Buildable` and each has its own door, so one that forgot to pass the
    model through would only show up here.
    """
    spec = {
        'path': dispatch_yaml,
        'str': str(dispatch_yaml),
        'dict': to_spec(dispatch_yaml).to_dict(),
        'spec': to_spec(dispatch_yaml),
        'program': lps.check(dispatch_yaml),
    }[form]
    with lps.solve(dispatch_yaml, dispatch_frame_inputs) as reference:
        expected = reference.objective

    assert lps.check(spec).variables['p'].dims == ('snapshot', 'generator'), (
        'check lowers it, and the plan is the same one whichever form the model arrived as'
    )
    with lps.build(spec, dispatch_frame_inputs) as model:
        assert model.solve('highs').objective == pytest.approx(expected, rel=1e-9), 'build takes it'
    with lps.solve(spec, dispatch_frame_inputs) as result:
        assert result.objective == pytest.approx(expected, rel=1e-9), 'and so does solve'
    assert lps.write(spec, dispatch_frame_inputs, tmp_path / f'{form}.lp').exists(), 'and write'

    runs = lps.solve_over(spec, dispatch_frame_inputs, [(0, dict(dispatch_frame_inputs))], key_name='draw')
    assert runs.keys == [0], 'a hand-built axis of one slice still runs, whatever the model arrived as'
    assert runs.objective['objective'].to_list() == pytest.approx([expected], rel=1e-9), (
        'and the sweep reaches the answer the one-shot verbs do'
    )


def test_check_and_to_program_need_no_data(dispatch_yaml):
    """The model stands for itself: the plan is read from the file when
    wanted, never carried on a built model."""
    for program in (lps.check(dispatch_yaml), to_program(dispatch_yaml)):
        assert program.variables['p'].dims == ('snapshot', 'generator')
        assert program.parameters['load'].dims == ('snapshot',)


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param('sum(p ** 2)', 'over variables', id='a-power-over-a-variable'),
        pytest.param(
            'sum(p) * sum(p)',
            'sums of more than one term',
            id='two-reductions-multiplied-caught-with-no-data-bound',
        ),
    ],
)
def test_check_reports_language_errors_before_any_data_is_bound(
    dispatch_yaml, dispatch_frame_inputs, expression, match
):
    """The CI verb enforces the ceiling with no data attached (math-spec's docs/about/limits.md).

    The refusal is the language's, at load (math-spec's ``test_degree.py``);
    what is asserted here is that both verbs surface it, ``build`` saying the
    same thing rather than deferring it to the solver. The raw file is
    assembled by hand because validating it is the refusal.
    """
    raw = {**to_spec(dispatch_yaml).model_dump(), 'objective': {'sense': 'minimize', 'expression': expression}}

    with pytest.raises(lps.LanguageError, match=match):
        lps.check(raw)
    sources = dispatch_frame_inputs
    with pytest.raises(lps.LanguageError, match=match):
        lps.build(raw, sources)


def test_error_hierarchy_is_one_catchable_tree():
    """One ``except`` covers the package, and the model/run split is real."""
    for cls in (lps.LanguageError, lps.DataError):
        assert issubclass(cls, lps.LpspecError)
    for cls in (lps.SchemaError, lps.DimensionError, lps.PiecewiseExpansionError):
        assert issubclass(cls, lps.LanguageError)
    assert not issubclass(lps.DataError, lps.LanguageError)
    assert issubclass(lps.LpspecError, ValueError)


def test_an_unknown_solver_is_refused_with_the_alternatives(dispatch_yaml, dispatch_frame_inputs):
    """The set of solvers is closed, and a name outside it never falls back to
    the default — solving with a solver other than the one asked for is the one
    answer that cannot be right. Here rather than in ``test_gurobi_sink.py``,
    which skips without the extra: the closed set is a property of the package,
    not of gurobi. Refused before the build, as an unwritable suffix is."""
    from lpspec.relational.sinks import SOLVERS

    sources = dispatch_frame_inputs
    with pytest.raises(lps.LpspecError, match='unknown solver'):
        lps.solve(dispatch_yaml, sources, solver_name='cplex')
    assert set(SOLVERS) == {'highs', 'gurobi', 'xpress'}


def test_a_solver_this_environment_cannot_run_is_refused_before_the_build(
    dispatch_yaml, dispatch_frame_inputs, monkeypatch
):
    """A name in the closed set is not a promise the package is installed.

    `gurobi` is a name lpspec knows on an install that never took the extra, so
    the two mistakes are different and get different sentences. Both refuse
    where the sink is resolved, which is before the build: resolving it there is
    what makes naming a sink nothing can serve cost no model, and that was only
    half true while a known name always resolved.

    Faked by naming a package nothing has rather than by uninstalling gurobipy,
    so the check runs wherever the suite does and still goes through the real
    probe.
    """
    from lpspec import api
    from lpspec.relational.sinks import SOLVERS

    sources = dispatch_frame_inputs
    monkeypatch.setattr(SOLVERS['gurobi'], 'requires', ('a_package_no_environment_has',))
    monkeypatch.setattr(
        api.PolarsEngine, 'build', lambda *_a, **_k: pytest.fail('the model was built before the refusal')
    )

    with pytest.raises(ModuleNotFoundError, match=r'not installed here.*\[gurobi\] extra'):
        lps.solve(dispatch_yaml, sources, solver_name='gurobi')


def test_a_list_of_models_is_refused(dispatch_yaml):
    """Composition is merging declarations, not passing several models.

    The message points at the dict, because a caller holding two files has
    somewhere to go — #30 declined the native merge rather than deferring it.
    """
    with pytest.raises(lps.LanguageError, match='merge the declarations'):
        lps.check([dispatch_yaml, dispatch_yaml])


def test_write_suffix_dispatch(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    out = lps.write(dispatch_yaml, sources, tmp_path / 'm.lp')
    assert out.stat().st_size > 0
    with pytest.raises(ValueError, match='unknown output format'):
        lps.write(dispatch_yaml, sources, tmp_path / 'm.nc')


def test_a_solution_saves_every_kind_it_answered_with(dispatch_solution, dispatch_yaml, tmp_path):
    """Every kind the solve answered with, tidy, streamed straight to disk.

    `<kind>/<name>.parquet`, because the language lets a constraint carry a
    variable's name and a flat directory could not hold both.
    """
    assert dispatch_solution.is_ok
    out = dispatch_solution.save(tmp_path / 'solution')
    assert out == tmp_path / 'solution'
    frame = pl.read_parquet(out / 'primal' / 'p.parquet')
    assert set(frame.columns) == {'snapshot', 'generator', 'value'}
    assert frame.height == dispatch_solution.primal('p').height
    assert {p.stem for p in (out / 'dual').iterdir()} == set(lps.check(dispatch_yaml).constraints), (
        'one dual file per constraint'
    )


def test_a_saved_solution_says_how_it_terminated(dispatch_solution, tmp_path):
    """The record beside the frames: what the numbers themselves cannot carry.

    Without it a directory holds every value the solve produced and cannot say
    what the solve concluded, so a set of saved cases answers neither which
    one was cheapest nor which one did not solve. Written as the row a sweep
    writes per slice, so a directory per case concatenates.
    """
    out = dispatch_solution.save(tmp_path / 'solution')
    record = pl.read_parquet(out / 'objective.parquet')
    assert record.columns == ['status', 'termination_condition', 'objective', 'has_primal', 'model'], (
        'the columns a sweep keys and folds, minus the key'
    )
    assert record.height == 1, 'one solve, one row'
    assert record.row(0, named=True) == {
        'status': dispatch_solution.status,
        'termination_condition': dispatch_solution.termination_condition,
        'objective': dispatch_solution.objective,
        'has_primal': dispatch_solution.has_primal,
        'model': dispatch_solution.model,
    }, 'the row carries what the result itself reports, not a second reading of the solve'


def test_an_export_writes_the_kinds_the_solve_answered_with(tmp_path):
    """An integer variable leaves the duals undefined and the export leaves
    them out; an expression that cannot be evaluated on this data is left out
    the same way, and `expression()` still says why."""
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['t']}, 'scale': {'dims': ['t']}},
        'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0}, 'domain': 'integer'}},
        'constraints': {'meet': {'foreach': ['t'], 'expression': 'p >= load'}},
        'expressions': {'twice': '2 * p', 'ratio': 'p / scale'},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    sources = {'t': range(2), 'load': [1.5, 2.5], 'scale': pl.DataFrame({'t': [0], 'value': [2.0]})}
    with lps.solve(spec, sources) as result:
        out = result.save(tmp_path)
        with pytest.raises(lps.LpspecError):
            result.expression('ratio')
        with pytest.raises(lps.LpspecError, match='integer'):
            result.to_dataset(kind='dual')
    assert sorted(p.name for p in out.iterdir()) == [
        'activity',
        'expression',
        'objective.parquet',
        'primal',
        'reasons.parquet',
    ], 'no dual/ — there are none to write, and reasons.parquet is where that is said'
    assert [p.name for p in (out / 'expression').iterdir()] == ['twice.parquet'], 'the one that evaluated'
    assert pl.read_parquet(out / 'expression' / 'twice.parquet')['value'].to_list() == [4.0, 6.0], (
        'twice the integer dispatch that meets 1.5 and 2.5'
    )


def test_a_saved_solution_carries_the_activities(dispatch_solution, dispatch_yaml, tmp_path):
    """The fourth reader a result has, and the one the export left behind.

    `activity` is not a `kind=` any bridge takes — a sweep folds three kinds
    and never holds these — so it needs naming separately or a saved answer
    cannot answer what a row's left-hand side reached.
    """
    out = dispatch_solution.save(tmp_path / 'solution')
    constraints = set(lps.check(dispatch_yaml).constraints)
    assert {p.stem for p in (out / 'activity').iterdir()} == constraints, 'one activity file per constraint'
    for name in constraints:
        assert pl.read_parquet(out / 'activity' / f'{name}.parquet').equals(dispatch_solution.activity(name))


def test_a_saved_solution_says_why_a_kind_is_absent(tmp_path):
    """An absence is a fact about the answer, so it is written down.

    Skipping a dual an integer variable made undefined, and an expression this
    data cannot evaluate, leaves a directory that cannot tell "there is none,
    and here is why" from "no such name". `dual` and `expression` say why in
    the process that solved; the file has to say it too.
    """
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['t']}, 'scale': {'dims': ['t']}},
        'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0}, 'domain': 'integer'}},
        'constraints': {'meet': {'foreach': ['t'], 'expression': 'p >= load'}},
        'expressions': {'twice': '2 * p', 'ratio': 'p / scale'},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    sources = {'t': range(2), 'load': [1.5, 2.5], 'scale': pl.DataFrame({'t': [0], 'value': [2.0]})}
    with lps.solve(spec, sources) as result:
        out = result.save(tmp_path)
        with pytest.raises(lps.LpspecError) as no_dual:
            result.dual('meet')
        with pytest.raises(lps.LpspecError) as no_ratio:
            result.expression('ratio')

    absent = pl.read_parquet(out / 'reasons.parquet')
    assert absent.columns == ['kind', 'name', 'reason'], 'the kind, what is missing under it, and why'
    assert absent.sort('kind', 'name').rows() == [
        ('dual', '', str(no_dual.value)),
        ('expression', 'ratio', str(no_ratio.value)),
    ], 'the whole kind for the duals, one name for the expression, each with the sentence the reader gives'


def test_a_saved_solution_loads_back_as_the_result_it_was(dispatch_solution, dispatch_yaml, tmp_path):
    """Every reader answers what it answered, off the directory rather than a session.

    The point of writing the record and the fourth kind: a `Result` is frames
    and a handful of scalars, so nothing about it needs the build that made
    it, the solver that filled it, or the process either ran in.
    """
    loaded = lps.load_result(dispatch_solution.save(tmp_path / 'solution'))

    assert (loaded.status, loaded.termination_condition) == (
        dispatch_solution.status,
        dispatch_solution.termination_condition,
    ), 'the outcome as recorded, on both axes'
    assert loaded.objective == dispatch_solution.objective
    assert loaded.has_primal
    program = lps.check(dispatch_yaml)
    for name in program.variables:
        assert loaded.primal(name).equals(dispatch_solution.primal(name))
    for name in program.constraints:
        assert loaded.dual(name).equals(dispatch_solution.dual(name))
        assert loaded.activity(name).equals(dispatch_solution.activity(name))


def test_a_loaded_result_gives_the_reason_the_solve_gave(tmp_path):
    """An absence loads back as the sentence, not as an unknown name."""
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['t']}, 'scale': {'dims': ['t']}},
        'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0}, 'domain': 'integer'}},
        'constraints': {'meet': {'foreach': ['t'], 'expression': 'p >= load'}},
        'expressions': {'twice': '2 * p', 'ratio': 'p / scale'},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    sources = {'t': range(2), 'load': [1.5, 2.5], 'scale': pl.DataFrame({'t': [0], 'value': [2.0]})}
    with lps.solve(spec, sources) as result:
        loaded = lps.load_result(result.save(tmp_path))
        with pytest.raises(lps.LpspecError) as no_dual:
            result.dual('meet')
        with pytest.raises(lps.LpspecError) as no_ratio:
            result.expression('ratio')

    assert loaded.expression('twice').equals(pl.DataFrame({'t': [0, 1], 'value': [4.0, 6.0]}))
    with pytest.raises(lps.LpspecError, match='integer'):
        loaded.dual('meet')
    with pytest.raises(lps.LpspecError) as loaded_no_ratio:
        loaded.expression('ratio')
    assert (str(loaded_no_ratio.value), str(no_ratio.value)) == (str(no_ratio.value), str(no_ratio.value)), (
        'the expression names the same reason it named in the process that solved'
    )
    assert 'integer' in str(no_dual.value), 'and the dual refuses for the reason it refused there'


def test_a_solve_that_left_no_values_loads_back_and_still_has_none(tmp_path):
    """A run that did not solve is an answer, and reads back as that answer."""
    with lps.solve(*CASES['INFEASIBLE']) as solution:
        loaded = lps.load_result(solution.save(tmp_path / 'infeasible'))
    assert loaded.termination_condition == 'infeasible'
    assert not loaded.has_primal, 'the record says the solve produced none, so no reader is offered any'
    assert loaded.objective != loaded.objective, 'nan, as the solve reported it'
    with pytest.raises(lps.NoSolutionError, match='infeasible'):
        loaded.primal('p')


def test_a_directory_that_is_not_a_saved_answer_is_refused(tmp_path):
    empty = tmp_path / 'nothing'
    empty.mkdir()
    with pytest.raises(lps.DataError, match=r'objective\.parquet'):
        lps.load_result(empty)


def test_read_back_is_in_label_order_and_stays_there(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """A read is a join, and a join settles no order — so the read states one.

    Was: every call came back in whatever order the hash join finished in, so
    two reads of one unchanged result disagreed and five writes of one solution
    produced five different files. Nothing was wrong with the numbers, which is
    what made it worth stating rather than leaving to the planner.

    Label order is row-major over the coordinate product, so it is checkable
    against the coordinates themselves: `snapshot` varies slowest, and within
    it `generator` follows the order the file declares.
    """
    sources = dispatch_frame_inputs
    generators = list(sources['p_max']['generator'])
    with lps.solve(dispatch_yaml, sources) as result:
        first = result.primal('p')
        assert first.equals(result.primal('p')), 'a second read agrees, to the row'

        by_declaration = first.with_columns(
            pl.col('generator').replace_strict(generators, range(len(generators))).alias('ord')
        )
        assert by_declaration.equals(by_declaration.sort('snapshot', 'ord'))

        written = [(result.save(tmp_path / f'solution{i}') / 'primal' / 'p.parquet').read_bytes() for i in range(3)]
        assert len(set(written)) == 1, 'the same solution writes the same bytes'


def test_a_result_stays_readable_until_it_is_closed(dispatch_yaml, dispatch_frame_inputs):
    """No lifetime to manage: reading is valid until you say otherwise.

    A result owns its read-back, so nothing expires it from outside and a
    caller who never closes loses nothing but memory. `close()` is there to
    release the label frames it pins early, and it means what it says — after
    it, there is nothing left to read.
    """
    sources = dispatch_frame_inputs
    result = lps.solve(dispatch_yaml, sources)
    height = result.primal('p').height
    assert height > 0
    assert result.primal('p').height == height, 'still readable, with no close in sight'

    result.close()
    with pytest.raises(lps.LpspecError, match='this result was closed'):
        result.primal('p')


def test_a_second_solve_does_not_rewrite_the_first_result(dispatch_yaml, dispatch_frame_inputs):
    """A result reports its own solve, not the engine's latest.

    Was: the values lived on the engine and every reader went back to them,
    so `objective` was a snapshot while `primal` was live — one result
    disagreeing with itself after a second solve, silently and with plausible
    numbers. Nothing supported re-binds data yet, so the bound has to be moved
    the way the planned in-place update will (#382: `changeColsBounds`
    against labels that are already solver indices).
    """
    key = ['snapshot', 'generator']  # a read is a join, so compare on coordinates
    sources = dispatch_frame_inputs
    with lps.build(dispatch_yaml, sources) as model:
        first = model.solve()
        before = first.primal('p').sort(key)
        assert first.is_ok

        built = model._engine._model
        model._engine._built = replace(built, obj=built.obj.with_columns(-pl.col('coeff')))
        second = model.solve()

        assert not second.primal('p').sort(key).equals(before), 'the second solve really moved'
        assert first.primal('p').sort(key).equals(before), 'and the first still reports its own'
        assert first.objective != pytest.approx(second.objective)


def test_primal_is_a_frame_and_to_pandas_is_the_bridge(dispatch_solution):
    """A frame is the shape results come in; pandas is an exit, not a shape.

    The two must describe the same table — the bridge is a conversion, not a
    second query with its own opinion about column order or dtypes.
    """
    frame = dispatch_solution.primal('p')
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ['snapshot', 'generator', 'value']

    pandas = pytest.importorskip('pandas')
    converted = dispatch_solution.to_pandas('p')
    assert isinstance(converted, pandas.DataFrame)
    assert list(converted.columns) == frame.columns
    assert len(converted) == frame.height
    assert frame['value'].sum() == pytest.approx(converted['value'].sum())


@pytest.mark.parametrize(
    ('absent', 'bridge'),
    [
        pytest.param('pandas', 'to_pandas', id='to_pandas-without-pandas'),
        pytest.param('xarray', 'to_dataarray', id='to_dataarray-without-xarray'),
        pytest.param('xarray', 'to_dataset', id='to_dataset-without-xarray'),
    ],
)
def test_a_bridge_out_names_the_extra_that_carries_it(dispatch_solution, absent, bridge):
    """A bridge out of a bare install says which extra to add.

    pandas and xarray ship with ``[linopy]`` rather than with the engine, so
    the bare `No module named 'pandas'` names a package no install instruction
    mentions and leaves the reader to guess. The gurobi sink already answers
    the same question with the extra; these three did not.

    The assertion is the extra, not the missing package: on an install that
    has neither, `to_dataarray` fails at the pandas half and reports that one.
    """
    with (
        mock.patch.dict(sys.modules, {absent: None}),
        pytest.raises(ModuleNotFoundError, match=r'pip install "lpspec\[linopy\]"'),
    ):
        getattr(dispatch_solution, bridge)('p')


def test_no_operator_registry_on_this_package():
    """The operator set is closed — there is no way to register more (#38's
    ``escape:`` island replaces the idea).

    This is what makes the two lanes accept the same language, and hence what
    makes the differential tests an oracle rather than a comparison of
    dialects (docs/about/architecture.md, "The expressive ceiling"). What
    ``math_spec`` exports is pinned name by name in math-spec's own suite, so
    the surface asserted here is this package's.
    """
    assert not hasattr(lps, 'register')


def test_solution_to_dataarray(dispatch_solution):
    """Long tables are right for joining, wrong for the array math that
    post-processing is mostly made of. `to_dataarray` is the bridge."""
    pytest.importorskip('xarray')
    arr = dispatch_solution.to_dataarray('p')
    tidy = dispatch_solution.to_pandas('p')

    assert arr.name == 'p', "named for the variable, not 'value' — the tidy column it came from"
    assert sorted(arr.dims) == ['generator', 'snapshot']
    assert arr.sizes['generator'] == 3
    wind_0 = tidy.query("generator == 'wind' and snapshot == 0")['value'].iloc[0]
    assert float(arr.sel(generator='wind', snapshot=0)) == pytest.approx(wind_0), (
        'the labelled form is the tidy form, indexed'
    )


def test_solution_to_dataset(dispatch_solution):
    """Several variables at once, each keeping its own dims."""
    pytest.importorskip('xarray')
    ds = dispatch_solution.to_dataset('p')
    tidy = dispatch_solution.to_pandas('p')

    assert list(ds.data_vars) == ['p']
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    first = tidy.iloc[0]
    assert float(ds['p'].sel(snapshot=first['snapshot'], generator=first['generator'])) == pytest.approx(first['value'])


def test_every_bridge_takes_a_kind(dispatch_solution, dispatch_yaml):
    """`to_pandas`, `to_dataarray` and `to_dataset` take `kind=` the way `scan`
    does — one kind per call, `primal` by default — so a price is a
    `DataArray` without going through `dual` and the bridge by hand, and a
    dataset of every dual has no name to collide with."""
    pytest.importorskip('xarray')
    constraint = next(iter(lps.check(dispatch_yaml).constraints))
    tidy = dispatch_solution.to_pandas(constraint, 'dual')
    assert tidy['value'].tolist() == dispatch_solution.dual(constraint)['value'].to_list()
    array = dispatch_solution.to_dataarray(constraint, 'dual')
    assert array.name == constraint
    assert set(dispatch_solution.to_dataset(kind='dual').data_vars) == set(lps.check(dispatch_yaml).constraints), (
        'all of one kind by default, as to_dataset() is all of the variables'
    )
    with pytest.raises(lps.LpspecError, match='primal, dual, expression'):
        dispatch_solution.to_pandas('p', 'objective')


def test_a_dataset_of_expressions_holds_every_one_this_data_evaluates():
    """`to_dataset(kind='expression')` is every declared expression, each over
    its own dims, and one that fails on this data fails the call the way
    `expression` does rather than being left out silently."""
    pytest.importorskip('xarray')
    spec = {**TWO_VARIABLE_SPEC, 'expressions': {'shed_twice': '2 * shed', 'total': 'sum(p, over=generator)'}}
    n = 4
    sources = {
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [100.0, 200.0]}),
        'load': pl.DataFrame({'snapshot': list(range(n)), 'value': np.full(n, 90.0)}),
        'snapshot': range(n),
        'generator': ['wind', 'gas'],
    }
    with lps.solve(spec, sources) as result:
        ds = result.to_dataset(kind='expression')
        assert set(ds.data_vars) == {'shed_twice', 'total'}, 'every declared expression, none named'
        assert list(ds['total'].dims) == ['snapshot'], 'each over its own dims'
        assert set(result.to_dataset('total', kind='expression').data_vars) == {'total'}, 'named ones only'


TWO_VARIABLE_SPEC = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {'p_max': {'dims': ['generator']}, 'load': {'dims': ['snapshot']}},
    'variables': {
        'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'shed': {'foreach': ['snapshot'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'balance': {
            'foreach': ['snapshot'],
            'expression': 'sum(p, over=generator) + shed == load',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(shed)'},
}


def test_to_dataset_defaults_to_every_variable():
    """A small model wants all of them at once, as linopy's model.solution
    gives you — naming them would be busywork."""
    pytest.importorskip('xarray')
    n = 4
    sources = {
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [100.0, 200.0]}),
        'load': pl.DataFrame({'snapshot': list(range(n)), 'value': np.full(n, 90.0)}),
    }

    with lps.solve(TWO_VARIABLE_SPEC, sources | {'snapshot': range(n), 'generator': ['wind', 'gas']}) as result:
        ds = result.to_dataset()
        subset = result.to_dataset('shed')

    assert set(ds.data_vars) == {'p', 'shed'}
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    assert list(ds['shed'].dims) == ['snapshot'], 'each variable keeps its own dims'
    assert set(subset.data_vars) == {'shed'}


@pytest.mark.parametrize(
    'raw',
    [
        pytest.param({'dimensionz': {}}, id='unknown-key'),
        pytest.param({'dimensions': {'g': {'dtype': 'complex'}}}, id='bad-dtype'),
        pytest.param({'version': 99}, id='unknown-version'),
        pytest.param(
            {
                'dimensions': {'g': {'dtype': 'str'}},
                'constraints': {'c': {'foreach': ['g'], 'expression': 'nope <= 1'}},
            },
            id='undeclared-name',
        ),
    ],
)
def test_a_wrong_model_raises_one_tree(raw: dict[str, object], tmp_path):
    """Every documented door answers with `LpspecError` (#527).

    Spec checking happens in two places — pydantic's validators and the
    language checkers — and they failed differently, so `except LpspecError`,
    the thing `docs/reference/api.md` tells a caller to write, missed the majority of
    model mistakes and a caller had no way to know which.

    `Spec.__init__` is *not* in this list, and cannot be: defining one makes
    pydantic route validation through it, which runs every after-validator
    twice and the first time with no context, breaking `extend()`.
    """
    doors = {
        'to_spec': lambda: to_spec(raw),
        'lps.check': lambda: lps.check(raw),
        'lps.solve': lambda: lps.solve(raw, {}),
        'lps.write': lambda: lps.write(raw, {}, str(tmp_path / 'm.lp')),
        'Spec.model_validate': lambda: Spec.model_validate(raw),
    }
    for door, call in doors.items():
        with pytest.raises(lps.LpspecError) as ei:
            call()
        assert 'errors.pydantic.dev' not in str(ei.value), f"{door} leaks pydantic's envelope"


def test_a_closed_result_says_it_was_closed(dispatch_yaml, dispatch_frame_inputs):
    """`close` releases the read-back the readers lay values over, and they say so.

    The status gate cannot notice: closing releases the coordinates, not the
    solve, so `is_readable` stays true and the reader used to fall through to
    a bare `AssertionError`. Frames read before the close are their own data
    and stay valid, which is the half worth stating in the message.
    """
    sources = dispatch_frame_inputs
    sol = lps.solve(dispatch_yaml, sources)
    frame = sol.primal('p')
    objective = sol.objective
    sol.close()

    assert frame.height > 0, 'a frame read before the close is its own data'
    assert sol.objective == objective, 'and the outcome needs no model to report'
    for read in (lambda: sol.primal('p'), lambda: sol.dual('power_balance')):
        with pytest.raises(lps.LpspecError, match='this result was closed'):
            read()


def test_check_catches_a_dim_error_with_no_sources_bound():
    """`check` is a CI verb, and this is what makes it one.

    Every dim rule is decided from declarations alone — math-spec's own suite
    is the whole set (#1150) — so the claim worth making *here* is
    not that the rule exists but that the runner reaches it without a byte of
    data. Kept on this side of the split for that reason: it is an assertion
    about `check`, not about dims.
    """
    raw = override(
        raw_of(EXAMPLES_DIR / 'dispatch.yaml'),
        **{'constraints.stray': {'foreach': ['snapshot'], 'expression': 'p <= p_max'}},
    )
    with pytest.raises(DimensionError):
        lps.check(raw)
