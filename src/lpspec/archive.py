"""The archive's file format: what a zip holding a model, its data and its answer contains.

``model.yaml``, one ``sources/<key>.parquet`` per key the file declares,
``answer/`` in the layout both answers already save, and ``axis.json`` where
the sources are cut.

Below :mod:`lpspec.api` and :mod:`lpspec.strategy`, because both write one:
the verb that solves is the only place that holds the model, the data and the
answer at once, which is why nothing assembles the three after the fact.
:mod:`lpspec.artifact` sits above all three and reads what is written here.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from math_spec import to_program

from lpspec.errors import LayoutError
from lpspec.sources import supplied, tidy_sources

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import polars as pl
    from math_spec import Spec

    from lpspec.lanes import Source

#: The archive's one layout. ``axis.json`` is also what says which kind an
#: archive holds, a sweep being the one whose sources are cut.
MODEL_MEMBER = 'model.yaml'
AXIS_MEMBER = 'axis.json'
SOURCES_DIR = PurePosixPath('sources')
ANSWER_DIR = PurePosixPath('answer')


@contextmanager
def beside(out: Path) -> Iterator[Path]:
    """A scratch directory on the filesystem *out* will land on, gone when the block ends.

    Where a caller has to lay an answer out before it is packed. The archive's
    own directory is made first, so the scratch and the file it feeds share a
    filesystem and the pack is a copy rather than a move across devices.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        yield Path(scratch)


def write_archive(
    out: Path,
    spec: Spec,
    sources: Mapping[str, Source],
    *,
    checked: Mapping[str, Source],
    whole: Mapping[str, pl.LazyFrame],
    axis: Mapping[str, Any] | None,
    answer: Path | None,
) -> Path:
    """Pack a model, its data and its answer into one zip at *out*.

    Args:
        out: Where to write. The directory is made if it does not exist.
        spec: The model as written, packed as ``model.yaml``.
        sources: What was attached, keyed as the file declares. A parquet path
            is copied as its own bytes; anything else is written as the tidy
            table it stands for.
        checked: The sources the declarations are checked against — all of
            them for one solve, one slice for a sweep, whose whole sources
            carry a column the model does not declare.
        whole: What to write instead of the checked frame, which is how a
            sliced source reaches the archive carrying every slice's rows.
        axis: The axis manifest, or ``None`` where the sources are not cut.
        answer: A directory already holding the answer's own layout — a spill,
            or one :func:`beside` handed the caller — or ``None``.

    Returns:
        *out*, which the archive lands at whole or not at all: it is written
        under a neighbouring name and renamed into place, so a write that does
        not finish leaves nothing under either.

    Raises:
        LanguageError: A model the language does not accept.
        DataError: A source that is missing, unreadable or the wrong shape.
    """
    program = to_program(spec)
    frames = supplied(program, tidy_sources(program, checked))
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + '.part')
    try:
        with zipfile.ZipFile(part, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(MODEL_MEMBER, spec.to_yaml())
            for name, frame in frames.items():
                member = str(SOURCES_DIR / f'{name}.parquet')
                given = sources.get(name)
                if isinstance(given, (str, Path)):
                    archive.write(given, member)
                else:
                    buffer = io.BytesIO()
                    whole.get(name, frame).collect().write_parquet(buffer, compression='zstd')
                    archive.writestr(member, buffer.getvalue())
            if axis is not None:
                archive.writestr(AXIS_MEMBER, json.dumps(axis))
            if answer is not None:
                for file in sorted(answer.rglob('*')):
                    if file.is_file():
                        archive.write(file, str(ANSWER_DIR / file.relative_to(answer).as_posix()))
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, out)
    return out


def extract(path: str | Path, into: Path) -> list[PurePosixPath]:
    """Unpack an archive :func:`write_archive` wrote, and give back what it held.

    Refuses before extracting anything, so a file that is not an archive
    leaves *into* as it found it.

    Raises:
        LayoutError: A member outside the layout, or no ``model.yaml``.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    with zipfile.ZipFile(path) as archive:
        members = [PurePosixPath(name) for name in archive.namelist() if not name.endswith('/')]
        strays = [str(m) for m in members if not _in_the_layout(m)]
        if strays or PurePosixPath(MODEL_MEMBER) not in members:
            raise LayoutError(_not_an_archive_message(path, strays))
        archive.extractall(into)
    return members


def is_source_member(member: PurePosixPath) -> bool:
    """Whether *member* is one of the ``sources/`` tables."""
    return member.parent == SOURCES_DIR and member.suffix == '.parquet'


def _in_the_layout(member: PurePosixPath) -> bool:
    return (
        member in (PurePosixPath(MODEL_MEMBER), PurePosixPath(AXIS_MEMBER))
        or is_source_member(member)
        or ANSWER_DIR in member.parents
    )


def _not_an_archive_message(path: str | Path, strays: list[str]) -> str:
    found = f'holds {strays}' if strays else "has no 'model.yaml'"
    return (
        f'{path} is not an archive: it {found}. One that archive= writes holds exactly '
        f"'model.yaml', one 'sources/<key>.parquet' per key the file declares, 'answer/' holding what the "
        f"solve returned, and 'axis.json' where its sources are sliced."
    )
