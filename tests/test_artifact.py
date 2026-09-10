"""``Artifact``: a model, its data and its answer as one file, and back.

The property is the one ``tidy_sources`` sees: what attaches from the archive
is what attached from the caller's own tables, frame for frame, over every
ported instance — the corpus is where every source shape and every declared
dtype already lives, so a shape the archive cannot carry fails here by name.
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import polars as pl
import pytest
import yaml as pyyaml
from math_spec import to_program, to_spec

import lpspec as lps
from lpspec.sources import attachable, tidy_sources
from tests.conftest import (
    DISPATCH_COST,
    DISPATCH_GENERATORS,
    DISPATCH_P_MAX,
    DISPATCH_SNAPSHOTS,
    PORT_REFERENCES,
    _dispatch_load,
    port_sources,
    port_spec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from math_spec import Spec


def _question(artifact: lps.Artifact) -> tuple[Spec, Mapping[str, object]]:
    """The pair every verb takes, read off an artifact."""
    return artifact.spec, artifact.sources


@pytest.mark.parametrize('name', sorted(PORT_REFERENCES), ids=str)
def test_what_attaches_from_the_archive_is_what_attached_from_the_tables(name: str, tmp_path: Path) -> None:
    program = to_program(port_spec(name))
    sources = port_sources(name)
    archive = lps.Artifact(port_spec(name), sources).save(tmp_path / 'model.zip')
    spec, unpacked = _question(lps.load_artifact(archive, tmp_path / 'out'))

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
    archive = lps.Artifact(dispatch_yaml, dispatch_frame_inputs).save(tmp_path / 'dispatch.zip')
    with (
        lps.solve(dispatch_yaml, dispatch_frame_inputs) as direct,
        lps.solve(*_question(lps.load_artifact(archive, tmp_path / 'out'))) as unpacked,
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
    archive = lps.Artifact(dispatch_yaml, sources).save(tmp_path / 'dispatch.zip')
    _, unpacked = _question(lps.load_artifact(archive, tmp_path / 'out'))
    cost = pl.read_parquet(unpacked['cost'])
    snapshot = pl.read_parquet(unpacked['snapshot'])

    assert cost.columns == ['generator', 'value'], 'a positional sequence is spread over its labels'
    assert cost['value'].to_list() == list(DISPATCH_COST), 'in the order the index declares them'
    assert snapshot.columns == ['snapshot'], 'a bare label range is written as an index table'
    assert snapshot.height == DISPATCH_SNAPSHOTS, 'one row per label'


def test_the_archive_is_the_file_and_stored_parquet(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    archive = lps.Artifact(dispatch_yaml, dispatch_frame_inputs).save(tmp_path / 'dispatch.zip')
    with zipfile.ZipFile(archive) as zipped:
        members = {info.filename: info.compress_type for info in zipped.infolist()}
        assert set(members) == {'model.yaml', *(f'sources/{k}.parquet' for k in dispatch_frame_inputs)}, (
            'the layout is model.yaml plus one parquet member per source key, nothing else'
        )
        assert set(members.values()) == {zipfile.ZIP_STORED}, 'members are stored — parquet is already compressed'
        assert to_spec(pyyaml.safe_load(zipped.read('model.yaml'))) == to_spec(dispatch_yaml), (
            'model.yaml is the model the source file declares'
        )


def test_a_parquet_path_is_copied_as_its_own_bytes(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    """Decoding and re-encoding parquet is byte-identical output for the CPU of a full read (#459)."""
    load = dispatch_frame_inputs['load'].with_columns(pl.lit('a stray column').alias('note'))
    path = tmp_path / 'load.parquet'
    load.write_parquet(path)
    archive = lps.Artifact(dispatch_yaml, {**dispatch_frame_inputs, 'load': str(path)}).save(tmp_path / 'dispatch.zip')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.read('sources/load.parquet') == path.read_bytes(), (
            'the file travels untouched, stray column included — it is filtered where it attaches, as a path is'
        )


def test_unpack_lays_the_archive_out_in_the_directory(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    archive = lps.Artifact(dispatch_yaml, dispatch_frame_inputs).save(tmp_path / 'dispatch.zip')
    spec, sources = _question(lps.load_artifact(archive, tmp_path / 'out'))

    assert sources == {k: tmp_path / 'out' / 'sources' / f'{k}.parquet' for k in dispatch_frame_inputs}, (
        'every source comes back as the path it was extracted to, one per key'
    )
    assert all(p.is_file() for p in sources.values()), 'and each path is a file on disk'
    assert (tmp_path / 'out' / 'model.yaml').is_file(), 'the file lands beside them, as the archive holds it'
    assert to_spec(tmp_path / 'out' / 'model.yaml') == spec, 'and is the model handed back'


def test_a_refused_model_writes_nothing(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    out = tmp_path / 'dispatch.zip'
    with pytest.raises(lps.DataError, match="no data provided for parameter 'cost'"):
        lps.Artifact(dispatch_yaml, {k: v for k, v in dispatch_frame_inputs.items() if k != 'cost'}).save(out)
    assert not out.exists(), 'the sources are checked before the archive is opened'


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
    with pytest.raises(lps.DataError) as excinfo:
        _question(lps.load_artifact(path, tmp_path / 'out'))
    assert says in str(excinfo.value), 'the message names what was found, and the layout one Artifact.save() writes'
    assert not (tmp_path / 'out').exists(), 'nothing is extracted from a zip that is not an archive'


def test_a_lowered_program_is_refused_by_name(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    """A lowered program has no file to write, and the docstring says so; the refusal says it too."""
    out = tmp_path / 'dispatch.zip'
    with pytest.raises(lps.LpspecError, match='a lowered Program has no file to write'):
        lps.Artifact(lps.check(dispatch_yaml), dispatch_frame_inputs).save(out)
    assert not out.exists(), 'nothing is written'


def test_the_archive_lands_whole(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path, monkeypatch) -> None:
    """The archive is written beside its name and renamed into place, so a
    reader that finds it finds all of it: a parent directory that does not
    exist is made, and a failure after the archive is open leaves nothing
    under either name."""
    from math_spec import Spec

    out = tmp_path / 'nested' / 'dispatch.zip'
    assert lps.Artifact(dispatch_yaml, dispatch_frame_inputs).save(out) == out
    assert sorted(p.name for p in out.parent.iterdir()) == ['dispatch.zip'], 'the archive alone, no .part beside it'

    def fails(self):
        raise RuntimeError('the box went away')

    monkeypatch.setattr(Spec, 'to_yaml', fails)
    later = tmp_path / 'later.zip'
    with pytest.raises(RuntimeError, match='went away'):
        lps.Artifact(dispatch_yaml, dispatch_frame_inputs).save(later)
    assert not list(tmp_path.glob('later*')), 'a write that did not finish leaves nothing under either name'


def test_an_archive_carries_the_answer_beside_the_question(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The full artifact: what was asked, the data it was asked of, and what came back.

    A saved answer alone cannot say which model produced it, and an archived
    model alone has to be re-solved to be read. One file holds both, and the
    two cannot drift apart or be paired up wrongly.
    """
    with lps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        archive = lps.Artifact(dispatch_yaml, dispatch_frame_inputs, solved).save(tmp_path / 'case.zip')
        loaded = lps.load_artifact(archive, tmp_path / 'case')

        assert loaded.answer is not None, 'the archive was given an answer, so it comes back with one'
        assert loaded.answer.objective == solved.objective
        for name in to_program(to_spec(dispatch_yaml)).variables:
            assert loaded.answer.primal(name).equals(solved.primal(name))

    with lps.solve(*_question(loaded)) as resolved:
        assert resolved.objective == pytest.approx(loaded.answer.objective, rel=1e-9), (
            'the question in the archive is the one its answer answered'
        )


def test_an_archive_of_the_question_alone_comes_back_with_no_answer(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The answer is optional, and its absence is a value rather than a failure."""
    archive = lps.Artifact(dispatch_yaml, dispatch_frame_inputs).save(tmp_path / 'question.zip')
    loaded = lps.load_artifact(archive, tmp_path / 'out')
    assert loaded.answer is None, 'nothing was given one, so nothing comes back'
    assert not (tmp_path / 'out' / 'answer').exists(), 'and no answer/ was written to extract'


def test_a_sweep_is_refused_by_name(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    """A sweep's sources are not this model's sources, and the refusal says where to put one.

    `EachCoordinate` slices on a column the spec does not declare, so the
    whole sources carry more than one row per coordinate and the door that
    checks one solve's data refuses them. `Runs.save` is where a sweep goes.
    """
    scenarios = ['low', 'high']
    load = pl.concat(
        [
            pl.DataFrame({'snapshot': range(DISPATCH_SNAPSHOTS), 'value': _dispatch_load()}).with_columns(
                pl.lit(name).alias('scenario')
            )
            for name in scenarios
        ]
    )
    sources = {**dispatch_frame_inputs, 'load': load}
    runs = lps.solve_over(dispatch_yaml, sources, lps.EachCoordinate('scenario'))
    with pytest.raises(lps.LpspecError, match=r'runs\.save'):
        lps.Artifact(dispatch_yaml, sources, runs)  # pyrefly: ignore[bad-argument-type]  — what the test asserts

    out = runs.save(tmp_path / 'sweep')
    assert lps.load_runs(out).objective.equals(runs.objective), 'a sweep round-trips on its own'


def test_a_lowered_program_is_refused_by_name_too(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """A lowered program has no file to write, and the refusal says so."""
    with pytest.raises(lps.LpspecError, match='a lowered Program has no file to write'):
        lps.Artifact(lps.check(dispatch_yaml), dispatch_frame_inputs)
