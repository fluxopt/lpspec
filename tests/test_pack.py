"""``pack`` and ``unpack``: a model and its data as one file, and back.

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
    from pathlib import Path


@pytest.mark.parametrize('name', sorted(PORT_REFERENCES), ids=str)
def test_what_attaches_from_the_archive_is_what_attached_from_the_tables(name: str, tmp_path: Path) -> None:
    program = to_program(port_spec(name))
    sources = port_sources(name)
    spec, unpacked = lps.unpack(lps.pack(port_spec(name), sources, tmp_path / 'model.zip'))

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
    archive = lps.pack(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    with lps.solve(dispatch_yaml, dispatch_frame_inputs) as direct, lps.solve(*lps.unpack(archive)) as unpacked:
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
    _, unpacked = lps.unpack(lps.pack(dispatch_yaml, sources, tmp_path / 'dispatch.zip'))

    assert unpacked['cost'].columns == ['generator', 'value'], 'a positional sequence is spread over its labels'
    assert unpacked['cost']['value'].to_list() == list(DISPATCH_COST), 'in the order the index declares them'
    assert unpacked['snapshot'].columns == ['snapshot'], 'a bare label range is written as an index table'
    assert unpacked['snapshot'].height == DISPATCH_SNAPSHOTS, 'one row per label'


def test_the_archive_is_the_file_and_stored_parquet(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    archive = lps.pack(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    with zipfile.ZipFile(archive) as zipped:
        members = {info.filename: info.compress_type for info in zipped.infolist()}
        assert set(members) == {'model.yaml', *(f'sources/{k}.parquet' for k in dispatch_frame_inputs)}, (
            'the layout is model.yaml plus one parquet member per source key, nothing else'
        )
        assert set(members.values()) == {zipfile.ZIP_STORED}, 'members are stored — parquet is already compressed'
        assert to_spec(pyyaml.safe_load(zipped.read('model.yaml'))) == to_spec(dispatch_yaml), (
            'model.yaml is the model the source file declares'
        )


def test_a_refused_model_writes_nothing(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    out = tmp_path / 'dispatch.zip'
    with pytest.raises(lps.DataError, match="no data provided for parameter 'cost'"):
        lps.pack(dispatch_yaml, {k: v for k, v in dispatch_frame_inputs.items() if k != 'cost'}, out)
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
        lps.unpack(path)
    assert says in str(excinfo.value), 'the message names what was found, and the layout one pack() writes'
