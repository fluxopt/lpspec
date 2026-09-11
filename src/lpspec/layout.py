"""The archive's layout: what holds a model, its data and its answer, as a zip or a directory.

``model.yaml``, one ``sources/<key>.parquet`` per key the file declares,
``answer/`` in the layout both answers already save, and ``axis.json`` where
the sources are cut. The same members either way — a zip is the directory
packed, which is why one of them is read where it lies and the other has to be
unpacked first.

Below :mod:`lpspec.api` and :mod:`lpspec.strategy`, because both write one:
the verb that solves is the only place that holds the model, the data and the
answer at once, which is why nothing assembles the three after the fact.
:mod:`lpspec.archive` sits above all three and reads what is written here.
"""

from __future__ import annotations

import io
import json
import shutil
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


def check_the_target(out: Path) -> None:
    """Refuse a directory target that already holds something, before anything is solved.

    Called where the archive is asked for as well as where it is written, so
    an hour of solving does not end in a refusal the call already implied.

    A ``.zip`` target is replaced, because renaming one file over another is
    one atomic step. A directory cannot be replaced that way, so merging into
    what is there would leave two archives readable as one.
    """
    if out.suffix != '.zip' and out.is_dir() and any(out.iterdir()):
        raise LayoutError(
            f'{str(out)!r} already holds something, and a directory archive is written whole rather than '
            f'merged into what is there. Name a directory that does not exist, or delete this one. A .zip '
            f'target is replaced instead, one file over another being a single step.'
        )


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
    """Write a model, its data and its answer to *out* — one zip, or a directory.

    **The suffix decides**, as :func:`~lpspec.api.write`'s does: ``.zip`` packs
    the layout into one file, and anything else lays the same members out in a
    directory. A directory is read where it lies, because the parquet files are
    already where a scan needs them; a zip has to be unpacked first.

    Args:
        out: Where to write, ``.zip`` or a directory. Its parent is made if it
            does not exist.
        spec: The model as written, held as ``model.yaml``.
        sources: What was attached, keyed as the file declares, and the whole
            of what the archive holds. A parquet path is copied as its own
            bytes; anything else is written as the tidy table it stands for.
            A name *checked* carries and this does not is one the axis made,
            such as a window's local index, and is not written: the axis in
            the archive makes it again.
        checked: The sources the declarations are checked against — all of
            them for one solve, one slice for a sweep, whose whole sources
            carry a column the model does not declare.
        whole: What to write instead of the checked frame, which is how a
            sliced source reaches the archive carrying every slice's rows.
        axis: The axis manifest, or ``None`` where the sources are not cut.
        answer: A directory already holding the answer's own layout — a spill,
            or one :func:`beside` handed the caller — or ``None``.

    Returns:
        *out*, which lands whole or not at all: it is built under a
        neighbouring name and renamed into place, so a write that does not
        finish leaves nothing under either.

    Raises:
        LanguageError: A model the language does not accept.
        DataError: A source that is missing, unreadable or the wrong shape.
        LayoutError: A directory target that already holds something.
    """
    check_the_target(out)
    program = to_program(spec)
    frames = supplied(program, tidy_sources(program, checked))
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + '.part')
    members = _Members.under(part, zipped=out.suffix == '.zip')
    try:
        members.put(MODEL_MEMBER, spec.to_yaml().encode())
        for name, frame in frames.items():
            if name not in sources:
                continue
            member = str(SOURCES_DIR / f'{name}.parquet')
            given = sources[name]
            if isinstance(given, (str, Path)):
                members.copy(Path(given), member)
            else:
                buffer = io.BytesIO()
                whole.get(name, frame).collect().write_parquet(buffer, compression='zstd')
                members.put(member, buffer.getvalue())
        if axis is not None:
            members.put(AXIS_MEMBER, json.dumps(axis).encode())
        if answer is not None:
            for file in sorted(answer.rglob('*')):
                if file.is_file():
                    members.copy(file, str(ANSWER_DIR / file.relative_to(answer).as_posix()))
        members.close()
    except BaseException:
        members.discard()
        raise
    if out.suffix != '.zip' and out.is_dir():
        out.rmdir()
    part.replace(out)
    return out


class _Members:
    """Somewhere to put the layout's members, whether that is a zip or a directory.

    One writer for both shapes, because the layout is the same either way and
    only the container differs. Everything lands under a neighbouring ``.part``
    name so the real one appears whole.
    """

    def __init__(self, part: Path, archive: zipfile.ZipFile | None) -> None:
        self._part = part
        self._archive = archive

    @classmethod
    def under(cls, part: Path, *, zipped: bool) -> _Members:
        """A part-file zip, or a part directory ready to be filled."""
        if zipped:
            return cls(part, zipfile.ZipFile(part, 'w', compression=zipfile.ZIP_STORED))
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        return cls(part, None)

    def put(self, member: str, data: bytes) -> None:
        """Write *data* as *member*."""
        if self._archive is not None:
            self._archive.writestr(member, data)
            return
        self._at(member).write_bytes(data)

    def copy(self, file: Path, member: str) -> None:
        """Copy *file*'s own bytes in as *member*, decoding nothing."""
        if self._archive is not None:
            self._archive.write(file, member)
            return
        shutil.copyfile(file, self._at(member))

    def close(self) -> None:
        """Finish the container, leaving the part ready to be renamed."""
        if self._archive is not None:
            self._archive.close()

    def discard(self) -> None:
        """Leave nothing behind, whichever shape was being written."""
        if self._archive is not None:
            self._archive.close()
            self._part.unlink(missing_ok=True)
            return
        shutil.rmtree(self._part, ignore_errors=True)

    def _at(self, member: str) -> Path:
        under = self._part / member
        under.parent.mkdir(parents=True, exist_ok=True)
        return under


def opened(path: str | Path, into: Path | None) -> Path:
    """Where an archive's members are on disk, unpacking it first if it is one file.

    A directory archive is read where it lies: its parquet files are already
    where a scan needs them, so there is nothing to unpack and nowhere to put
    it. A zip is not, so it needs somewhere writable — which only the caller
    knows, an archive often living where it is only read.

    Args:
        path: The archive, a ``.zip`` or a directory.
        into: Where to unpack a zip, made if it does not exist. Refused for a
            directory archive, which needs none.

    Returns:
        The directory the members are in — *into* for a zip, *path* itself for
        a directory.

    Raises:
        LayoutError: A member outside the layout, no ``model.yaml``, a zip
            with no *into*, or an *into* given for a directory. Nothing is
            unpacked.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    held = Path(path)
    if held.is_dir():
        if into is not None:
            raise LayoutError(
                f'{str(held)!r} is a directory archive, so it is read where it lies and into= has nothing to '
                f'do. Drop into=; it is for unpacking a .zip.'
            )
        _check_the_layout(held, _members_under(held))
        return held
    if into is None:
        raise LayoutError(
            f'{str(held)!r} is one file, so reading it needs somewhere to unpack: pass into=. A directory '
            f'archive is read where it lies and needs none.'
        )
    with zipfile.ZipFile(held) as archive:
        _check_the_layout(held, [PurePosixPath(name) for name in archive.namelist() if not name.endswith('/')])
        archive.extractall(into)
    return into


def members_of(under: Path) -> list[PurePosixPath]:
    """Every member the layout holds, relative to the directory they are in."""
    return _members_under(under)


def _members_under(under: Path) -> list[PurePosixPath]:
    return [PurePosixPath(file.relative_to(under).as_posix()) for file in sorted(under.rglob('*')) if file.is_file()]


def _check_the_layout(named: Path, members: list[PurePosixPath]) -> None:
    """Refuse anything that is not this layout, before a byte is unpacked."""
    strays = [str(m) for m in members if not _in_the_layout(m)]
    if strays or PurePosixPath(MODEL_MEMBER) not in members:
        raise LayoutError(_not_an_archive_message(named, strays))


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
