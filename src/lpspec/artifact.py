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
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from math_spec import Spec, to_program, to_spec
from math_spec.program import Program

from lpspec.api import load_result
from lpspec.errors import DataError, LpspecError
from lpspec.relational.result import Result
from lpspec.sources import supplied, tidy_sources
from lpspec.strategy import EachCoordinate, EachWindow, Runs, carries, load_runs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.lanes import Source

__all__ = ['Artifact', 'load_artifact']

#: The archive's one layout: the file, one parquet member per source key, the
#: answer under a directory of its own in the shape ``save`` writes, and the
#: axis where the sources are sliced. A sweep's answer carries ``sweep.json``
#: of its own, which is what tells the two answers apart on the way back in.
_MODEL_MEMBER = 'model.yaml'
_AXIS_MEMBER = 'axis.json'
_SOURCES_DIR = PurePosixPath('sources')
_ANSWER_DIR = PurePosixPath('answer')
_SWEEP_MANIFEST = 'sweep.json'


@dataclass(frozen=True)
class Artifact:
    """A model, the data it was solved with, and what came back.

    The full artifact, and the unit an archived study travels as. Construct
    one and :meth:`save` it; :func:`load_artifact` gives it back.

    A sweep is one of these, and ``axis`` is what makes it one: its sources
    carry the column the axis slices on, which the model does not declare, so
    they are legible only beside the axis that cuts them. **The axis is
    present exactly when the sources are sliced** — a
    :class:`~lpspec.strategy.Runs` without one is refused, and so is an axis
    beside a single solve's answer. A hand-built list of ``(key, sources)``
    is refused too: those are unrelated questions, so they are one artifact
    each.

    ``answer`` is whichever the solve produced. The two disagree about one
    name — ``Result.objective`` is a number and ``Runs.objective`` a frame,
    one row per slice — so code that does not know which it holds asks
    ``isinstance(artifact.answer, Runs)``.

    Attributes:
        spec: The model as written. A path or a mapping is loaded on the way
            in, so this is always a ``Spec``.
        sources: What was attached to it. Going in, anything
            :func:`~lpspec.api.build` takes; coming back, the parquet files
            the archive holds — which is the same type, ``Path`` being a
            source like any other.
        answer: What came back, or ``None`` for the question alone.
        axis: How the sources were cut, where they were. ``None`` for a
            single solve.
    """

    spec: Spec
    sources: Mapping[str, Source]
    answer: Result | Runs | None = None
    axis: EachCoordinate | EachWindow | None = None
    #: Where an archive was extracted, for the answer that reads lazily off
    #: it. Kept so the directory is named by the object that depends on it.
    directory: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Hold the three to their invariant, and load *spec* if it was given as a path or a mapping.

        The axis is present exactly when the sources are sliced, which is what
        makes them legible: without it a sweep's carry a column the model does
        not declare, and with it beside one solve's answer it claims a cut
        that did not happen. Frozen, so the loaded spec goes back through
        ``object``: a field that is sometimes a path and sometimes a ``Spec``
        would put the same branch in every reader.
        """
        if self.axis is not None and not isinstance(self.axis, (EachCoordinate, EachWindow)):
            raise LpspecError(
                'an artifact takes EachCoordinate or EachWindow, which say how one set of sources was cut. '
                'A hand-built list is a set of sources per slice, which are unrelated questions — archive '
                'one artifact each.'
            )
        if isinstance(self.answer, Runs) and self.axis is None:
            raise LpspecError(
                "a sweep's answer needs the axis that produced it: its sources carry the column the axis "
                'slices on, which the model does not declare, so nothing can check or write them without '
                'it. Pass the axis solve_over was given.'
            )
        if self.axis is not None and isinstance(self.answer, Result):
            raise LpspecError(
                'an axis says the sources are cut into slices, and this answer came back from one solve of '
                'all of them. Drop the axis, or pass the sweep that ran it.'
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
        key the file declares, ``answer/`` holding what
        :meth:`~lpspec.relational.result.Result.save` or
        :meth:`~lpspec.strategy.Runs.save` writes, and ``axis.json`` where the
        sources are sliced. A parquet path is copied as its own bytes; a table
        is written as parquet; a plain-Python shape — a number, a label
        sequence, a ``{label: value}`` map — as the tidy table it stands for.
        Members are stored uncompressed, because parquet already is.

        The sources go in through the same door that reads them, so what is
        refused there is refused here and nothing is written: :func:`build`'s
        for a single solve, and for a sweep the door
        :func:`~lpspec.strategy.solve_over` uses, which is one slice of them.
        A source the axis cuts is written **whole** — the archive holds the
        sources as the sweep was given them, not one copy per slice. Whether
        the *model* can be cut this way stays ``solve_over``'s question, asked
        when it is run.

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
        frames = supplied(program, tidy_sources(program, self._as_the_model_sees_them()))
        whole = {} if self.axis is None else carries(self.sources, self.axis.dim)
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
                        whole.get(name, frame).collect().write_parquet(buffer, compression='zstd')
                        archive.writestr(member, buffer.getvalue())
                if self.axis is not None:
                    archive.writestr(_AXIS_MEMBER, json.dumps(_axis_manifest(self.axis)))
                if self.answer is not None:
                    _write_answer(archive, self.answer, beside=out.parent)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, out)
        return out

    def _as_the_model_sees_them(self) -> Mapping[str, Source]:
        """The sources with the axis column gone — one slice of a sweep's, or all of one solve's.

        A sweep's whole sources carry a column the model does not declare, so
        the door that checks one solve's data would refuse them for having
        more than one row per coordinate. One slice is what the model is
        actually built from, so it is what the check has to see.
        """
        if self.axis is None:
            return self.sources
        cut = self.axis.slices(self.sources)
        if not cut:
            raise DataError('the axis produced no slices, so there is nothing the model would be built from')
        return cut[0][1]


def _axis_manifest(axis: EachCoordinate | EachWindow) -> dict[str, Any]:
    """*axis* as the JSON the archive carries — the one home for that shape, with :func:`_axis_from`."""
    if isinstance(axis, EachCoordinate):
        return {'each': 'coordinate', 'dim': axis.dim}
    steps = axis.steps if isinstance(axis.steps, int) else list(axis.steps)
    return {'each': 'window', 'dim': axis.dim, 'steps': steps, 'lookahead': axis.lookahead, 'into': axis.into}


def _axis_from(manifest: Mapping[str, Any]) -> EachCoordinate | EachWindow:
    """The axis :func:`_axis_manifest` wrote."""
    if manifest['each'] == 'coordinate':
        return EachCoordinate(manifest['dim'])
    return EachWindow(manifest['dim'], steps=manifest['steps'], lookahead=manifest['lookahead'], into=manifest['into'])


def _write_answer(archive: zipfile.ZipFile, answer: Result | Runs, beside: Path) -> None:
    """*answer*'s own layout into *archive*, under ``answer/``.

    Through a directory rather than into memory, because that is the layout
    both answers already write and because a large primal is streamed to disk
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
    axis_member = into / _AXIS_MEMBER
    axis = _axis_from(json.loads(axis_member.read_text())) if axis_member.is_file() else None
    return Artifact(
        to_spec(into / _MODEL_MEMBER),
        {m.stem: into / m for m in members if _is_source_member(m)},
        _read_answer(into / _ANSWER_DIR) if any(_ANSWER_DIR in m.parents for m in members) else None,
        axis,
        into,
    )


def _read_answer(under: Path) -> Result | Runs:
    """Whichever answer is under *under*, read as itself.

    A sweep writes a manifest of its own and a single solve does not, so the
    archive says which without being told.
    """
    return load_runs(under) if (under / _SWEEP_MANIFEST).is_file() else load_result(under)


def _in_the_layout(member: PurePosixPath) -> bool:
    return (
        member in (PurePosixPath(_MODEL_MEMBER), PurePosixPath(_AXIS_MEMBER))
        or _is_source_member(member)
        or _ANSWER_DIR in member.parents
    )


def _is_source_member(member: PurePosixPath) -> bool:
    return member.parent == _SOURCES_DIR and member.suffix == '.parquet'


def _not_an_archive_message(path: str | Path, strays: list[str]) -> str:
    found = f'holds {strays}' if strays else "has no 'model.yaml'"
    return (
        f'{path} is not an artifact: it {found}. One Artifact.save() writes holds exactly '
        f"'model.yaml', one 'sources/<key>.parquet' per key the file declares, 'answer/' where it was "
        f"given one, and 'axis.json' where its sources are sliced."
    )
