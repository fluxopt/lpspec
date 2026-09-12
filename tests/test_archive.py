"""``archive=``: a model, its data and its answer as one file, and back.

The property is the one ``tidy_sources`` sees: what attaches from the archive
is what attached from the caller's own tables, frame for frame, over every
ported instance — the corpus is where every source shape and every declared
dtype already lives, so a shape the archive cannot carry fails here by name.
"""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

import polars as pl
import pytest
import yaml as pyyaml
from math_spec import to_program, to_spec

import lpspec as lps
from lpspec.layout import ANSWER_DIR, DIGESTS_MEMBER, _staging_for
from lpspec.relational.parquet import COST_FILE, Cost, digest_of_file
from lpspec.sources import attachable, tidy_sources
from tests.conftest import (
    DISPATCH_COST,
    DISPATCH_GENERATORS,
    DISPATCH_P_MAX,
    DISPATCH_SNAPSHOTS,
    PORT_REFERENCES,
    _dispatch_load,
    override,
    port_sources,
    port_spec,
    raw_of,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from math_spec import Spec


def _question(artifact: lps.SolveArchive | lps.SweepArchive) -> tuple[Spec, Mapping[str, object]]:
    """The pair every verb takes, read off an artifact."""
    return artifact.spec, artifact.sources


def _archived(spec, sources, out: Path) -> Path:
    """The archive a solve writes, which is the only way one is made."""
    with lps.solve(spec, sources, archive=out):
        return out


@pytest.mark.parametrize('name', sorted(PORT_REFERENCES), ids=str)
def test_what_attaches_from_the_archive_is_what_attached_from_the_tables(name: str, tmp_path: Path) -> None:
    program = to_program(port_spec(name))
    sources = port_sources(name)
    archive = _archived(port_spec(name), sources, tmp_path / 'model.zip')
    spec, unpacked = _question(lps.load_archive(archive, tmp_path / 'out'))

    assert set(unpacked) == set(attachable(program)), (
        'the archive holds one member per attachable key — every declared parameter, dimension and lookup, '
        'and nothing a piecewise block derives'
    )
    before = tidy_sources(program, sources)
    after = tidy_sources(to_program(spec), unpacked)
    differing = [key for key in before if not before[key].collect().equals(after[key].collect())]
    assert not differing, (
        f'frames that came back changed: {differing} — the archive carries the labels, values and dtypes'
    )


def test_the_round_trip_solves_to_the_same_objective(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    with (
        lps.solve(dispatch_yaml, dispatch_frame_inputs) as direct,
        lps.solve(*_question(lps.load_archive(archive, tmp_path / 'out'))) as unpacked,
    ):
        assert unpacked.objective == pytest.approx(direct.objective, rel=1e-9), (
            'the archive builds the model the tables did'
        )


def test_plain_python_shapes_are_written_as_the_tables_they_stand_for(dispatch_yaml: Path, tmp_path: Path) -> None:
    """A dict, a positional sequence and a bare label range all come back as tidy frames."""
    sources = {
        'p_max': dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True)),
        'cost': list(DISPATCH_COST),
        'load': pl.DataFrame({'snapshot': range(DISPATCH_SNAPSHOTS), 'value': _dispatch_load()}),
        'snapshot': range(DISPATCH_SNAPSHOTS),
        'generator': list(DISPATCH_GENERATORS),
    }
    archive = _archived(dispatch_yaml, sources, tmp_path / 'dispatch.zip')
    _, unpacked = _question(lps.load_archive(archive, tmp_path / 'out'))
    cost = pl.read_parquet(unpacked['cost'])
    snapshot = pl.read_parquet(unpacked['snapshot'])

    assert cost.columns == ['generator', 'value'], 'a positional sequence is spread over its labels'
    assert cost['value'].to_list() == list(DISPATCH_COST), 'in the order the index declares them'
    assert snapshot.columns == ['snapshot'], 'a bare label range is written as an index table'
    assert snapshot.height == DISPATCH_SNAPSHOTS, 'one row per label'


def test_the_archive_is_the_file_and_stored_parquet(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    with zipfile.ZipFile(archive) as zipped:
        members = {info.filename: info.compress_type for info in zipped.infolist()}
        beside_the_answer = {name for name in members if not name.startswith('answer/')}
        assert beside_the_answer == {
            'model.yaml',
            'sources.parquet',
            *(f'sources/{k}.parquet' for k in dispatch_frame_inputs),
        }, (
            'the layout is model.yaml, one parquet member per source key, the table digesting them, and the '
            'answer under its own'
        )
        assert any(name.startswith('answer/') for name in members), 'every archive carries the answer that made it'
        assert set(members.values()) == {zipfile.ZIP_STORED}, 'members are stored — parquet is already compressed'
        assert to_spec(pyyaml.safe_load(zipped.read('model.yaml'))) == to_spec(dispatch_yaml), (
            'model.yaml is the model the source file declares'
        )


def test_a_parquet_path_is_copied_as_its_own_bytes(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    """Decoding and re-encoding parquet is byte-identical output for the CPU of a full read (#459)."""
    load = dispatch_frame_inputs['load'].with_columns(pl.lit('a stray column').alias('note'))
    path = tmp_path / 'load.parquet'
    load.write_parquet(path)
    archive = _archived(dispatch_yaml, {**dispatch_frame_inputs, 'load': str(path)}, tmp_path / 'dispatch.zip')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.read('sources/load.parquet') == path.read_bytes(), (
            'the file travels untouched, stray column included — it is filtered where it attaches, as a path is'
        )


def test_unpack_lays_the_archive_out_in_the_directory(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    spec, sources = _question(lps.load_archive(archive, tmp_path / 'out'))

    assert sources == {k: tmp_path / 'out' / 'sources' / f'{k}.parquet' for k in dispatch_frame_inputs}, (
        'every source comes back as the path it was extracted to, one per key'
    )
    assert all(p.is_file() for p in sources.values()), 'and each path is a file on disk'
    assert (tmp_path / 'out' / 'model.yaml').is_file(), 'the file lands beside them, as the archive holds it'
    assert to_spec(tmp_path / 'out' / 'model.yaml') == spec, 'and is the model handed back'


def test_a_refused_model_writes_nothing(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    out = tmp_path / 'dispatch.zip'
    with pytest.raises(lps.DataError, match="no data provided for parameter 'cost'"):
        lps.solve(dispatch_yaml, {k: v for k, v in dispatch_frame_inputs.items() if k != 'cost'}, archive=out)
    assert not out.exists(), 'a model that cannot be built writes no archive'


@pytest.mark.parametrize(
    ('members', 'says'),
    [
        pytest.param({'sources/load.parquet': b''}, "has no 'model.yaml'", id='no-model'),
        pytest.param({'model.yaml': b'', 'load.parquet': b''}, "holds ['load.parquet']", id='member-outside-sources'),
        pytest.param({'model.yaml': b'', 'sources/load.csv': b''}, "holds ['sources/load.csv']", id='not-parquet'),
    ],
)
def test_a_zip_outside_the_layout_is_refused(members: dict[str, bytes], says: str, tmp_path: Path) -> None:
    path = tmp_path / 'other.zip'
    with zipfile.ZipFile(path, 'w') as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    with pytest.raises(lps.LayoutError) as excinfo:
        _question(lps.load_archive(path, tmp_path / 'out'))
    assert says in str(excinfo.value), 'the message names what was found, and the layout archive= writes'
    assert not (tmp_path / 'out').exists(), 'nothing is extracted from a zip that is not an archive'


def test_a_directory_archive_holds_what_the_zip_holds_and_is_read_where_it_lies(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The suffix picks the container, and the container is all that differs.

    A zip is the directory packed. The members are the same either way, so the
    only thing that changes is whether anything has to be unpacked before a
    scan can see the parquet files.
    """
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip')
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')

    with zipfile.ZipFile(tmp_path / 'case.zip') as zipped:
        packed = sorted(zipped.namelist())
    spread = sorted(f.relative_to(tmp_path / 'case').as_posix() for f in (tmp_path / 'case').rglob('*') if f.is_file())
    assert spread == packed, 'the same members, laid out instead of packed'

    loose = lps.load_archive(tmp_path / 'case')
    unpacked = lps.load_archive(tmp_path / 'case.zip', tmp_path / 'out')
    assert loose.answer.objective == unpacked.answer.objective
    assert loose.sources['load'].parent.parent == tmp_path / 'case', (
        'a directory archive hands back paths into itself, so there is no second copy to keep alive'
    )


@pytest.mark.parametrize(
    ('suffix', 'into', 'says'),
    [
        pytest.param('.zip', None, 'needs somewhere to unpack', id='a-zip-with-no-into'),
        pytest.param('', 'anywhere', 'read where it lies', id='a-directory-with-an-into'),
    ],
)
def test_into_is_asked_for_exactly_where_something_must_be_unpacked(
    suffix: str, into: str | None, says: str, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """`into` is not a convention to remember — each shape says which it is.

    A zip has no paths a scan can reach, so it needs somewhere writable, and
    only the caller knows one: an archive often lives where it is only read.
    A directory already has them, so an `into` would have nothing to do.
    """
    out = tmp_path / f'case{suffix}'
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=out)
    with pytest.raises(lps.LayoutError, match=says):
        lps.load_archive(out, None if into is None else tmp_path / into)


def test_a_directory_that_already_holds_something_is_refused(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An archive is written whole, so it never merges into what is there.

    Overwriting a directory cannot be one rename the way replacing a file can,
    so what would be left is a mix of two archives that reads as one.
    """
    out = tmp_path / 'case'
    out.mkdir()
    (out / 'mine.txt').write_text('not an archive')
    with pytest.raises(lps.LayoutError, match='already holds something'):
        lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=out)
    assert sorted(f.name for f in out.iterdir()) == ['mine.txt'], 'and what was there is untouched'


@pytest.mark.parametrize(
    'verb',
    [
        pytest.param(lambda spec, sources: lps.build(spec, sources), id='build'),
        pytest.param(lambda spec, sources: lps.solve(spec, sources), id='solve'),
        pytest.param(
            lambda spec, sources: lps.solve_over(spec, sources, lps.EachCoordinate('generator')), id='solve_over'
        ),
    ],
)
def test_a_lowered_program_is_not_a_model_any_verb_takes(verb, dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """Lowering has no inverse, so a Program is refused at the door, not at the archive.

    An answer from one could not name the document it came back from, and
    nothing built from one could be archived. Every verb reads a model through
    one function, so the sentence is written once and arrives before anything
    is built.
    """
    with pytest.raises(lps.LpspecError, match='lowered Program is not a model this takes'):
        verb(lps.check(dispatch_yaml), dispatch_frame_inputs)


def test_the_archive_lands_whole(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path, monkeypatch) -> None:
    """The archive is written beside its name and renamed into place, so a
    reader that finds it finds all of it: a parent directory that does not
    exist is made, and a failure after the archive is open leaves nothing
    under either name."""
    from math_spec import Spec

    out = tmp_path / 'nested' / 'dispatch.zip'
    _archived(dispatch_yaml, dispatch_frame_inputs, out)
    assert sorted(p.name for p in out.parent.iterdir()) == ['dispatch.zip'], 'the archive alone, no .part beside it'

    def fails(self):
        raise RuntimeError('the box went away')

    monkeypatch.setattr(Spec, 'to_yaml', fails)
    later = tmp_path / 'later.zip'
    with pytest.raises(RuntimeError, match='went away'):
        lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=later)
    assert not list(tmp_path.glob('later*')), 'a write that did not finish leaves nothing under either name'


def test_an_archive_carries_the_answer_beside_the_question(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The full artifact: what was asked, the data it was asked of, and what came back.

    A saved answer alone cannot say which model produced it, and an archived
    model alone has to be re-solved to be read. One file holds both, and the
    two cannot drift apart or be paired up wrongly.
    """
    with lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip') as solved:
        loaded = lps.load_archive(tmp_path / 'case.zip', tmp_path / 'case')

        assert loaded.answer.objective == solved.objective
        for name in to_program(to_spec(dispatch_yaml)).variables:
            assert loaded.answer.primal(name).equals(solved.primal(name))

    with lps.solve(*_question(loaded)) as resolved:
        assert resolved.objective == pytest.approx(loaded.answer.objective, rel=1e-9), (
            'the question in the archive is the one its answer answered'
        )


def test_two_archives_of_one_spec_over_different_numbers_are_told_apart(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """What `spec_digest` cannot say, and the reason the data is digested too.

    A spec digest is of the document. Two runs of one model over different
    numbers carry the same one, so on that column alone they read as the same
    question asked twice.
    """
    halved = pl.DataFrame(
        {'snapshot': dispatch_frame_inputs['load']['snapshot'], 'value': dispatch_frame_inputs['load']['value'] * 0.5}
    )
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'base').close()
    lps.solve(dispatch_yaml, {**dispatch_frame_inputs, 'load': halved}, archive=tmp_path / 'halved').close()
    base, other = lps.load_archive(tmp_path / 'base'), lps.load_archive(tmp_path / 'halved')

    assert base.answer.spec_digest == other.answer.spec_digest, 'one document, so the spec digest cannot separate them'
    moved = (
        base.source_digests.join(other.source_digests, on='source', suffix='_other')
        .filter(pl.col('digest') != pl.col('digest_other'))['source']
        .to_list()
    )
    assert moved == ['load'], 'and the digests name the one input that moved, not merely that something did'


def test_the_digest_table_names_every_source_the_archive_holds(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A digest per member of `sources/`, so nothing is silently unattested.

    A table short of one key would leave that input outside the claim while
    reading as a complete one.
    """
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case').close()
    case = lps.load_archive(tmp_path / 'case')

    assert case.source_digests.columns == ['source', 'digest'], 'the table is the key and what its bytes digest to'
    assert case.source_digests['source'].to_list() == sorted(case.sources), (
        'one row per archived source, in source order rather than the order the caller happened to pass them'
    )
    held = tmp_path / 'case' / 'sources'
    recomputed = {file.stem: digest_of_file(file) for file in held.glob('*.parquet')}
    assert dict(case.source_digests.iter_rows()) == recomputed, (
        'and each digest is of the bytes the archive holds, so a reader can check it against the archive alone'
    )


def test_a_sweep_archive_digests_the_sources_it_was_cut_from(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A sweep archives its sources whole, so the digests are of the whole.

    One digest per slice would name data the archive does not hold: the point
    of archiving a sweep's sources whole is that one copy carries every
    slice's rows.
    """
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    lps.solve_over(dispatch_yaml, sources, lps.EachCoordinate('scenario'), archive=tmp_path / 'study')
    study = lps.load_archive(tmp_path / 'study')

    assert isinstance(study, lps.SweepArchive), 'the archive carries an axis, or this is testing the other type'
    assert study.source_digests['source'].to_list() == sorted(study.sources), 'one row per source, as for one solve'
    held = tmp_path / 'study' / 'sources'
    assert dict(study.source_digests.iter_rows()) == {
        file.stem: digest_of_file(file) for file in held.glob('*.parquet')
    }, 'each of the whole sources the sweep was cut from'


def test_an_archive_holding_no_digest_table_is_refused_by_name(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An archive written before the digests reads as short of a member.

    Nothing else would catch it: the answer's layout stamp says nothing about
    the archive around it, and the sources are all still there.
    """
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case').close()
    (tmp_path / 'case' / DIGESTS_MEMBER).unlink()

    with pytest.raises(lps.LayoutError, match=DIGESTS_MEMBER) as excinfo:
        lps.load_archive(tmp_path / 'case')
    assert 'Solving the model it holds again' in str(excinfo.value), 'and the message names the way out'


def test_an_archive_records_what_reaching_its_answer_cost(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The one part of an archive that re-solving cannot recover.

    Everything else it holds is reproducible by construction — that is what
    `lps.solve(case.spec, case.sources)` is for. The clocks are of the machine
    that ran them, so unless the solve writes them down nothing does.
    """
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')
    cost = lps.load_archive(tmp_path / 'case').diagnostics

    assert cost.columns == [
        'columns',
        'rows',
        'nonzeros',
        'sink_columns',
        'sink_rows',
        'solves',
        'loads',
        'attach',
        'build',
        'handoff',
        'solve',
        'write',
    ], 'the sizes, then the counters, then one clock per phase in the order the phases run'
    assert cost.height == 1, 'one solve writes one row'
    row = cost.row(0, named=True)
    assert (row['solves'], row['loads']) == (1, 1), (
        'lps.solve builds the model it solves, so the row covers that one solve and its one load'
    )
    assert row['build'] > 0.0, 'the build ran, so its clock is not the zero that says a phase did not'


def test_the_cost_row_says_how_many_solves_its_clocks_cover(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """Why `solves` is a column rather than something the reader assumes.

    The diagnostics a model reports are its whole life, and an archive is
    written from inside one solve of it. So a row off a model that has solved
    before carries clocks covering every one of those solves, and the column
    saying so is what stops it from reading as this answer's own.
    """
    with lps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        model.solve()
        model.solve(archive=tmp_path / 'second')

    assert lps.load_archive(tmp_path / 'second').diagnostics['solves'].item() == 2, (
        'two solves ran before the archive was written, and the row it carries counts both'
    )


def test_a_case_that_wrote_a_file_and_one_that_did_not_are_still_one_table(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """Why a phase that never ran writes zero instead of no column.

    The point of the row is that a directory of archives is a table. Two cases
    that entered different phases would otherwise write different schemas, and
    a concat over the directory would fail on whichever one differed.
    """
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'solved')
    with lps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        model.write(tmp_path / 'model.lp')
        model.solve(archive=tmp_path / 'written')

    rows = [lps.load_archive(tmp_path / case).diagnostics for case in ('solved', 'written')]
    assert [row['write'].item() == 0.0 for row in rows] == [True, False], (
        'the first case never wrote a file and the second did, or the two schemas were never in question'
    )
    assert pl.concat(rows).height == 2, 'and the two rows are one table'


def test_an_archive_holding_no_cost_row_is_refused_by_name(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The layout stamp cannot say this one, so the missing member has to.

    `format.json` is held at 0 while the layout moves, so an archive written
    before the cost row reads as current and is short of a member instead.
    """
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')
    (tmp_path / 'case' / ANSWER_DIR / COST_FILE).unlink()

    with pytest.raises(lps.LayoutError, match=COST_FILE) as excinfo:
        lps.load_archive(tmp_path / 'case')
    assert 'solving the model it holds again' in str(excinfo.value), 'and the message names the way out'


def test_every_phase_a_build_clocks_has_a_column_to_travel_in(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A phase added to the engine and not to `Cost` would be dropped in silence.

    The row is written off `timings`, whose keys are whatever the engine
    clocked, and a key with no column of its own simply does not travel.
    """
    with lps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        model.write(tmp_path / 'model.lp')
        model.solve()
        clocked = set(model.diagnostics().timings)

    assert clocked == {'attach', 'build', 'write', 'handoff', 'solve'}, (
        'this model entered every phase a build clocks, or the check below passes on the ones it missed'
    )
    assert not clocked - set(Cost._fields), f'every phase the engine clocks is a Cost column, and {clocked} is not'


def test_an_updated_model_archives_the_data_it_actually_answered(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An update moves the question, so the model is the only thing that knows it.

    `update` puts new numbers on a built model, and what it answers from then
    on is the merge — not the mapping the caller passed `build`. Nothing
    outside the call could tell those two questions apart, the spec being
    unchanged, so the archive is written from inside it.
    """
    halved = pl.DataFrame(
        {'snapshot': dispatch_frame_inputs['load']['snapshot'], 'value': dispatch_frame_inputs['load']['value'] * 0.5}
    )
    with lps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        before = model.solve().objective
        after = model.update({'load': halved}).solve(archive=tmp_path / 'updated.zip')
        assert after.objective != pytest.approx(before, rel=1e-9), 'the update moved the answer, or this proves nothing'

    loaded = lps.load_archive(tmp_path / 'updated.zip', tmp_path / 'updated')
    with lps.solve(*_question(loaded)) as resolved:
        assert resolved.objective == pytest.approx(after.objective, rel=1e-9), (
            'the archived question is the updated one, so it re-solves to the answer it carries'
        )


def test_a_scenario_sweep_is_an_artifact_and_runs_again(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The axis is what makes a sweep's sources legible, so it travels with them.

    They carry the column it slices on, which the model does not declare, and
    the archive holds them **whole** — one copy, not one per slice.
    """
    axis = lps.EachCoordinate('scenario')
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    runs = lps.solve_over(dispatch_yaml, sources, axis, archive=tmp_path / 'study.zip')
    study = lps.load_archive(tmp_path / 'study.zip', tmp_path / 'study')

    assert study.axis == axis, 'the axis comes back as the value it went in as'
    assert study.answer.objective.equals(runs.objective)
    assert pl.read_parquet(study.sources['load']).equals(sources['load']), (
        'the sliced source is archived whole, the column the axis cuts on included'
    )
    again = lps.solve_over(study.spec, study.sources, study.axis)
    assert again.objective['objective'].to_list() == pytest.approx(runs.objective['objective'].to_list()), (
        'the archive re-runs to the sweep it recorded, slice for slice'
    )


def test_a_rolling_horizon_keeps_the_way_back_to_the_dimension_it_sliced(tmp_path: Path) -> None:
    """`original_index` is the one thing a windowed sweep cannot rebuild from its frames.

    The dimension a window sliced and the coordinates each one owns go in the
    manifest beside them, so a stitched read off the archive is the stitched
    read off the sweep.
    """
    from tests.test_strategy import WINDOW, horizon_sources

    axis = lps.EachWindow('snapshot', steps=4, lookahead=2, into='t')
    sources = horizon_sources(12)
    runs = lps.solve_over(WINDOW, sources, axis, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll.zip')
    stitched = runs.primal('soc', original_index=True)
    loaded = lps.load_archive(tmp_path / 'roll.zip', tmp_path / 'roll')

    assert loaded.axis == axis
    assert loaded.answer.scan('soc', original_index=True).collect().equals(stitched), (
        'the lookahead rows are dropped on the way out of the archive as they were in the process'
    )


def test_an_archive_whose_answer_names_another_model_is_refused(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The one thing the archive asserts that its members do not: they belong together.

    A solve writes both together, so this cannot fire on one this package
    wrote — it stands between a hand-edited zip and a reader who would take it
    at its word and re-solve to something else.
    """
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'case.zip')
    other = override(raw_of(dispatch_yaml), **{'variables.p.bounds.upper': 1.0})
    tampered = tmp_path / 'tampered.zip'
    with zipfile.ZipFile(archive) as held, zipfile.ZipFile(tampered, 'w') as edited:
        for name in held.namelist():
            edited.writestr(name, pyyaml.safe_dump(other) if name == 'model.yaml' else held.read(name))

    with pytest.raises(lps.LpspecError, match='came back from a different model'):
        lps.load_archive(tampered, tmp_path / 'out')
    assert lps.load_archive(archive, tmp_path / 'fine').spec == to_spec(dispatch_yaml), (
        'and the archive as written reads back as the model it holds'
    )


def test_every_slice_of_a_sweep_names_the_model_it_answered(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A sweep's digest is the sweep's, not each slice's.

    Every slice is built off the one lowered program, which has no document to
    digest. Left to the slice, every row would carry null: the comparison
    could not run, and the pairing an archive checks on the way in would have
    nothing to compare.
    """
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    runs = lps.solve_over(dispatch_yaml, sources, lps.EachCoordinate('scenario'))
    alone = lps.solve(dispatch_yaml, dispatch_frame_inputs)

    assert runs.objective['spec_digest'].null_count() == 0, 'no slice is left without the document it answered'
    assert runs.objective['spec_digest'].unique().to_list() == [alone.spec_digest], (
        'and it is the same digest one solve of the same file carries'
    )


def test_a_sweep_archive_whose_answer_names_another_model_is_refused(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The sibling of the one-solve check, and the one that was vacuous.

    A sweep's guard reads every slice's digest. While those were null it could
    not fire at all, so a swapped `model.yaml` loaded happily.
    """
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    lps.solve_over(dispatch_yaml, sources, lps.EachCoordinate('scenario'), archive=tmp_path / 'study')
    other = override(raw_of(dispatch_yaml), **{'variables.p.bounds.upper': 1.0})
    (tmp_path / 'study' / 'model.yaml').write_text(pyyaml.safe_dump(other))

    with pytest.raises(lps.LpspecError, match='came back from a different model'):
        lps.load_archive(tmp_path / 'study')


def test_saving_an_answer_twice_leaves_only_the_second(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A directory holds one answer, so the first one's frames do not survive.

    Without this a second save leaves both models' frames side by side, and
    `load_result` reports the second model's digest while answering for a
    variable only the first declared.
    """
    renamed = raw_of(dispatch_yaml)
    renamed['variables']['q'] = renamed['variables'].pop('p')
    renamed['constraints']['power_balance']['expression'] = 'sum(q, over=generator) == load'
    renamed['objective']['expression'] = 'sum(q * cost)'

    lps.solve(dispatch_yaml, dispatch_frame_inputs).save(tmp_path / 'shared')
    out = lps.solve(renamed, dispatch_frame_inputs).save(tmp_path / 'shared')

    assert sorted(f.stem for f in (out / 'primal').glob('*.parquet')) == ['q'], (
        "only the second model's variable is left"
    )
    with pytest.raises(KeyError, match='unknown variable'):
        lps.load_result(out).primal('p')


def test_saving_an_answer_into_an_unpacked_archive_takes_its_cost_row_with_it(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A cost row belongs to the answer beside it, so a second save removes it.

    Unpacking an archive gives a directory `result.save` will take, and what a
    save leaves is one answer. A row saying what a different solve on a
    different machine spent would otherwise sit beside it, readable through a
    reader that reports this answer's digest.
    """
    lps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')
    answer = tmp_path / 'case' / ANSWER_DIR
    assert (answer / COST_FILE).is_file(), 'the archive wrote one, or this proves nothing'

    lps.solve(dispatch_yaml, dispatch_frame_inputs).save(answer)

    assert not (answer / COST_FILE).exists(), 'and a saved answer carries no cost row, so none is left behind'


def test_saved_cases_say_whether_they_are_comparable(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path) -> None:
    """Why the digest is written rather than only checked.

    Concatenating the records of cases solved apart gives a comparison table,
    and one distinct `spec_digest` in it is the claim that the table compares like
    with like. Nothing else on disk says so.
    """
    other = override(raw_of(dispatch_yaml), **{'variables.p.bounds.upper': 1000.0})
    records = []
    for name, spec in (('base', dispatch_yaml), ('capped', other)):
        with lps.solve(spec, dispatch_frame_inputs) as solved:
            out = solved.save(tmp_path / name)
        records.append(pl.read_parquet(out / 'objective.parquet').select(pl.lit(name).alias('case'), pl.all()))

    table = pl.concat(records)
    assert table['spec_digest'].n_unique() == 2, 'two models, so the table is not comparing like with like'
    assert lps.load_result(tmp_path / 'base').spec_digest == table.filter(pl.col('case') == 'base')['spec_digest'][0], (
        'and a loaded answer carries the digest its record holds'
    )


def test_an_answer_in_another_layout_is_refused_by_name(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The layout moves while the package is on 0.0.1aN, so a stale one says so.

    Nothing reads another layout back — there is no migration and there will
    not be one — so the stamp exists to turn a missing column into a sentence
    naming what to do instead. The stamp stays zero while the layout moves, so
    what it catches is an answer written before there was one; a number is
    written here to reach the sentence from the other side too.
    """
    with lps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        out = solved.save(tmp_path / 'solution')
    (out / 'format.json').write_text(json.dumps({'answer': 1}))

    with pytest.raises(lps.LayoutError, match='solve the model again and save it'):
        lps.load_result(out)

    (out / 'format.json').unlink()
    with pytest.raises(lps.LayoutError, match='layout None'):
        lps.load_result(out)


def test_a_hand_built_axis_is_refused(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """A list of `(key, sources)` is a set of sources per slice.

    Nothing serialises it but a copy of every slice's data, and the slices are
    unrelated questions anyway — so the refusal sends them to one artifact
    each rather than inventing a layout for them.
    """
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    with pytest.raises(lps.LpspecError, match='archive one solve each'):
        lps.solve_over(dispatch_yaml, sources, [('a', sources)], key_name='case', archive='never.zip')


def _by_scenario(names: list[str]) -> pl.DataFrame:
    return pl.concat(
        [
            pl.DataFrame({'snapshot': range(DISPATCH_SNAPSHOTS), 'value': _dispatch_load()}).with_columns(
                pl.lit(name).alias('scenario')
            )
            for name in names
        ]
    )


def test_a_sweep_refuses_a_lowered_program_before_it_solves_a_slice(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """A sweep refuses it at the same door, and before the first slice is taken."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    with pytest.raises(lps.LpspecError, match='was lowered from'):
        lps.solve_over(lps.check(dispatch_yaml), sources, lps.EachCoordinate('scenario'), archive='never.zip')


def test_two_writers_to_one_target_stage_in_separate_places(tmp_path: Path) -> None:
    """The staging name is not a function of the target, because one name is shared.

    Two archives written to one path at once met in it: the second to open a
    staging area cleared the first's members out from under it, and the first
    went on to rename a torn directory into place and report it written. The
    tear surfaced only at load, as an archive with no ``model.yaml``.
    """
    out = tmp_path / 'case'

    assert _staging_for(out) != _staging_for(out), 'two writers to one target stage in separate places'


def test_a_written_archive_leaves_no_staging_beside_it(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path) -> None:
    """What a writer stages is its own to remove, in both containers."""
    for name in ('case', 'case.zip'):
        _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / name)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['case', 'case.zip'], (
        'the staging each write allocated is gone, leaving the two archives alone'
    )
