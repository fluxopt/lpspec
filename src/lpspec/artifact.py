"""The whole thing in one file: a model, the data it was solved with, and what came back.

An answer alone cannot say which model produced it, and a model alone has to
be solved again to be read. :class:`Artifact` holds both, and :meth:`save`
writes them as one zip so they cannot drift apart or be paired up wrongly.

Above ``api`` and ``strategy`` rather than beside them: an artifact carries
either kind of answer, so it is the one place that knows about both a
``Result`` and a ``Runs``. Neither of them knows about it.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from math_spec import Spec, to_program, to_spec
from math_spec.program import Program

from lpspec.api import load_result
from lpspec.errors import DataError, LpspecError
from lpspec.sources import supplied, tidy_sources
from lpspec.strategy import Runs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.lanes import Source
    from lpspec.relational.result import Result

__all__ = ['Artifact', 'load_artifact']

#: The archive's one layout: the file, one parquet member per source key, and
#: the answer under a directory of its own, in the shape
#: :meth:`~lpspec.relational.result.Result.save` writes.
_MODEL_MEMBER = 'model.yaml'
_SOURCES_DIR = PurePosixPath('sources')
_ANSWER_DIR = PurePosixPath('answer')


@dataclass(frozen=True)
class Artifact:
    """A model, the data it was solved with, and what came back.

    The full artifact, and the unit an archived study travels as. Construct
    one and :meth:`save` it; :func:`load_artifact` gives it back.

    A sweep is not one of these. Its sources carry the column its axis slices
    on, which the model does not declare, so they are not sources this model
    accepts and neither the check below nor the write could make sense of
    them without the axis that cuts them. A :class:`~lpspec.strategy.Runs` is
    refused by name; :meth:`~lpspec.strategy.Runs.save` and
    :func:`~lpspec.strategy.load_runs` carry a sweep's answer on its own.

    Attributes:
        spec: The model as written. A path or a mapping is loaded on the way
            in, so this is always a ``Spec``.
        sources: What was attached to it. Going in, anything
            :func:`~lpspec.api.build` takes; coming back, the parquet files
            the archive holds — which is the same type, ``Path`` being a
            source like any other.
        answer: What came back, or ``None``.
    """

    spec: Spec
    sources: Mapping[str, Source]
    answer: Result | None = None
    #: Where an archive was extracted, for the answer that reads lazily off
    #: it. Kept so the directory is named by the object that depends on it.
    directory: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Load *spec* if it was given as a path or a mapping, and refuse a lowered program.

        Frozen, so the normalised value goes back through ``object``: a field
        that is sometimes a path and sometimes a ``Spec`` would put the same
        branch in every reader.
        """
        if isinstance(self.answer, Runs):
            raise LpspecError(
                "a sweep's sources carry the column its axis slices on, which the model does not declare, "
                "so an artifact cannot check or write them the way it checks and writes one solve's. Save "
                'the sweep on its own with runs.save(directory), which load_runs reads back.'
            )
        if isinstance(self.spec, Program):
            raise LpspecError(
                'an artifact holds the model as written — a path, a mapping or a Spec — and a lowered '
                'Program has no file to write. Pass what it was lowered from.'
            )
        object.__setattr__(self, 'spec', to_spec(self.spec))

    def save(self, out: str | Path) -> Path:
        """Write the model, its data and its answer as one zip file.

        The archive holds ``model.yaml``, ``sources/<key>.parquet`` for every
        key the file declares, and ``answer/`` holding what
        :meth:`~lpspec.relational.result.Result.save` or
        :meth:`~lpspec.strategy.Runs.save` writes. The sources go in through
        the same door :func:`~lpspec.api.build` reads them, so what is refused
        there is refused here and nothing is written. A parquet path is copied
        as its own bytes; a table is written as parquet; a plain-Python shape
        — a number, a label sequence, a ``{label: value}`` map — as the tidy
        table it stands for. Members are stored uncompressed, because parquet
        already is.

        The archive lands whole: written beside *out* and renamed into place,
        its directory made if it does not exist, and nothing left under either
        name by a write that did not finish.

        Args:
            out: Where to write; a ``.zip`` suffix is the convention.

        Returns:
            The path written.

        Raises:
            LanguageError: A file the language does not accept.
            DataError: A source that is missing, unreadable, or the wrong
                shape.
            LpspecError: An answer that was closed, or a sweep that holds no
                values.
        """
        program = to_program(self.spec)
        frames = supplied(program, tidy_sources(program, self.sources))
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + '.part')
        try:
            with zipfile.ZipFile(part, 'w', compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(_MODEL_MEMBER, self.spec.to_yaml())
                for name, frame in frames.items():
                    member = str(_SOURCES_DIR / f'{name}.parquet')
                    given = self.sources.get(name)
                    if isinstance(given, (str, Path)):
                        archive.write(given, member)
                    else:
                        buffer = io.BytesIO()
                        frame.collect().write_parquet(buffer, compression='zstd')
                        archive.writestr(member, buffer.getvalue())
                if self.answer is not None:
                    _write_answer(archive, self.answer, beside=out.parent)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, out)
        return out


def _write_answer(archive: zipfile.ZipFile, answer: Result, beside: Path) -> None:
    """*answer*'s own layout into *archive*, under ``answer/``.

    Through a directory rather than into memory, because that is the layout
    ``save`` already writes and because a large primal is streamed to disk
    rather than passed through this process. The scratch directory is made
    beside the archive, so it lands on the filesystem the caller chose.
    """
    with tempfile.TemporaryDirectory(dir=beside) as scratch:
        saved = answer.save(scratch)
        for file in sorted(saved.rglob('*')):
            if file.is_file():
                archive.write(file, str(_ANSWER_DIR / file.relative_to(saved).as_posix()))


def load_artifact(path: str | Path, into: str | Path) -> Artifact:
    """Read back an archive :meth:`Artifact.save` wrote.

    ``lps.solve(artifact.spec, artifact.sources)`` asks the question again;
    ``artifact.answer`` is what it answered the first time.

    Args:
        path: The zip file.
        into: The directory to extract to, created if it does not exist. The
            members land in it as the archive holds them, and the answer reads
            lazily off them, so it has to outlive what is read.

    Returns:
        The model as written, its sources keyed as the file declares them, and
        the answer — ``None`` where the archive holds none.

    Raises:
        LanguageError: A ``model.yaml`` the language does not accept.
        DataError: A member outside the layout. Nothing is extracted.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    into = Path(into)
    with zipfile.ZipFile(path) as archive:
        members = [PurePosixPath(name) for name in archive.namelist() if not name.endswith('/')]
        strays = [str(m) for m in members if not _in_the_layout(m)]
        if strays or PurePosixPath(_MODEL_MEMBER) not in members:
            raise DataError(_not_an_archive_message(path, strays))
        archive.extractall(into)
    carried = any(_ANSWER_DIR in m.parents for m in members)
    return Artifact(
        to_spec(into / _MODEL_MEMBER),
        {m.stem: into / m for m in members if _is_source_member(m)},
        load_result(into / _ANSWER_DIR) if carried else None,
        into,
    )


def _in_the_layout(member: PurePosixPath) -> bool:
    return member == PurePosixPath(_MODEL_MEMBER) or _is_source_member(member) or _ANSWER_DIR in member.parents


def _is_source_member(member: PurePosixPath) -> bool:
    return member.parent == _SOURCES_DIR and member.suffix == '.parquet'


def _not_an_archive_message(path: str | Path, strays: list[str]) -> str:
    found = f'holds {strays}' if strays else "has no 'model.yaml'"
    return (
        f'{path} is not an artifact: it {found}. One Artifact.save() writes holds exactly '
        f"'model.yaml', one 'sources/<key>.parquet' per key the file declares, and 'answer/' where it "
        f'was given one.'
    )
