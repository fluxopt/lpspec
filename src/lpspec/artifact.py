"""The whole thing in one file: a model, the data it was solved with, and what came back.

An answer alone cannot say which model produced it, and a model alone has to
be solved again to be read. An artifact holds both, and :meth:`save` writes
them as one zip so they cannot drift apart or be paired up wrongly.

Two of them, because a sweep's sources are cut and one solve's are not:
:class:`SolveArtifact` and :class:`SweepArtifact`. They share the archive's
layout and nothing else; :func:`load_artifact` reads either.

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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from math_spec import Spec, to_spec
from math_spec.program import Program

from lpspec.api import load_result
from lpspec.errors import DataError, LayoutError, LpspecError
from lpspec.lanes import lowered
from lpspec.relational.parquet import digest_of
from lpspec.sources import supplied, tidy_sources
from lpspec.strategy import EachCoordinate, EachWindow, Runs, carries, load_runs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

    from lpspec.lanes import Source
    from lpspec.relational.result import Result

__all__ = ['SolveArtifact', 'SweepArtifact', 'load_artifact']

#: The archive's one layout: the file, one parquet member per source key, the
#: answer under a directory of its own in the shape ``save`` writes, and the
#: axis where the sources are sliced. ``axis.json`` is also what says which
#: artifact an archive holds, a sweep being the one whose sources are cut.
_MODEL_MEMBER = 'model.yaml'
_AXIS_MEMBER = 'axis.json'
_SOURCES_DIR = PurePosixPath('sources')
_ANSWER_DIR = PurePosixPath('answer')


@dataclass(frozen=True)
class SolveArtifact:
    """A model, the data it was solved with, and what one solve of it returned.

    The unit an archived case travels as. Construct one and :meth:`save` it;
    :func:`load_artifact` gives it back. A sweep is its sibling,
    :class:`SweepArtifact`.

    Attributes:
        spec: The model as written. A path or a mapping is loaded on the way
            in, so this is always a ``Spec``.
        sources: What was attached to it. Going in, anything
            :func:`~lpspec.api.build` takes; coming back, the parquet files
            the archive holds — which is the same type, ``Path`` being a
            source like any other.
        answer: What came back, or ``None`` for the question alone.
    """

    spec: Spec
    sources: Mapping[str, Source]
    answer: Result | None = None

    def __post_init__(self) -> None:
        """Load *spec*, and refuse a lowered program or an answer to a different spec."""
        _load_the_spec(self)
        _check_the_pairing(self.spec, () if self.answer is None else (self.answer.spec_digest,))

    def save(self, out: str | Path) -> Path:
        """Write the model, its data and its answer as one zip file.

        The archive holds ``model.yaml``, ``sources/<key>.parquet`` for every
        key the file declares, and ``answer/`` holding what
        :meth:`~lpspec.relational.result.Result.save` writes. A parquet path
        is copied as its own bytes; a table is written as parquet; a
        plain-Python shape — a number, a label sequence, a ``{label: value}``
        map — as the tidy table it stands for. Members are stored
        uncompressed, because parquet already is.

        The sources go in through the same door :func:`~lpspec.api.build`
        reads them, so what is refused there is refused here and nothing is
        written. The archive lands whole: written beside *out* and renamed
        into place, its directory made if it does not exist, and nothing left
        under either name by a write that did not finish.

        Args:
            out: Where to write; a ``.zip`` suffix is the convention.

        Returns:
            The path written.

        Raises:
            LanguageError: A file the language does not accept.
            DataError: A source that is missing, unreadable, or the wrong
                shape.
            LpspecError: An answer that was closed.
        """
        return _write(self, Path(out), self.sources, whole={}, axis=None)


@dataclass(frozen=True)
class SweepArtifact:
    """A model, the data a sweep was solved over, the axis that cut it, and what came back.

    :class:`SolveArtifact`'s sibling, and the axis is what separates them: a
    sweep's sources carry the column it slices on, which the model does not
    declare, so they are legible only beside it. That is why the axis is a
    field and not an argument — without it nothing could check or write them.

    ``lps.solve_over(sweep.spec, sweep.sources, sweep.axis)`` runs it again.

    Attributes:
        spec: The model as written, as :class:`SolveArtifact` holds it.
        sources: What the sweep was given — **whole**, carrying every slice's
            rows, because one copy per slice is what a sweep exists not to
            write.
        axis: :class:`~lpspec.strategy.EachCoordinate` or
            :class:`~lpspec.strategy.EachWindow`. A hand-built list of
            ``(key, sources)`` is refused: those are unrelated questions, so
            they are one :class:`SolveArtifact` each.
        answer: What came back, or ``None`` for the question alone.
    """

    spec: Spec
    sources: Mapping[str, Source]
    axis: EachCoordinate | EachWindow
    answer: Runs | None = None

    def __post_init__(self) -> None:
        """Refuse an axis nothing can serialise, then load *spec* as the sibling does."""
        if not isinstance(self.axis, (EachCoordinate, EachWindow)):
            raise LpspecError(
                'a sweep artifact takes EachCoordinate or EachWindow, which say how one set of sources was '
                'cut. A hand-built list is a set of sources per slice, which are unrelated questions — '
                'archive one SolveArtifact each.'
            )
        _load_the_spec(self)
        _check_the_pairing(self.spec, () if self.answer is None else self.answer.objective['spec_digest'].to_list())

    def save(self, out: str | Path) -> Path:
        """Write the model, its data, its axis and its answer as one zip file.

        :meth:`SolveArtifact.save`'s layout with ``axis.json`` beside it, and
        ``answer/`` holding what :meth:`~lpspec.strategy.Runs.save` writes.

        Two things differ, both because the sources are cut. **What the check
        sees is one slice of them** — the door
        :func:`~lpspec.strategy.solve_over` uses, so what that refuses this
        refuses. **What is written is all of them**, the column the axis cuts
        on included. Whether the *model* can be cut this way stays
        ``solve_over``'s question, asked when the sweep is run.

        Args:
            out: Where to write; a ``.zip`` suffix is the convention.

        Returns:
            The path written.

        Raises:
            LanguageError: A file the language does not accept.
            DataError: A source that is missing, unreadable or the wrong
                shape, or an axis that produced no slices.
            LpspecError: A sweep that holds no values.
        """
        return _write(self, Path(out), self._one_slice(), carries(self.sources, self.axis.dim), self.axis)

    def _one_slice(self) -> Mapping[str, Source]:
        """The sources as the model sees them, which is what the check has to see.

        The whole sources carry a column the model does not declare, so the
        door that checks one solve's data would refuse them for having more
        than one row per coordinate.
        """
        cut = self.axis.slices(self.sources)
        if not cut:
            raise DataError('the axis produced no slices, so there is nothing the model would be built from')
        return cut[0][1]


def _check_the_pairing(spec: Spec, answered: Sequence[str | None]) -> None:
    """Refuse an answer that came back from a different spec than this one.

    The one thing an artifact asserts that its three fields do not: that they
    belong together. Without it a mispaired triple archives cleanly and the
    file re-solves to an answer other than the one it carries. An answer
    solved off a lowered program digests to ``None`` and is taken on trust —
    there is no document to compare it against.
    """
    mine = digest_of(spec.to_yaml())
    if others := sorted({other for other in answered if other is not None and other != mine}):
        raise LpspecError(
            f'this answer came back from a different spec: it carries {others} and the one given here digests '
            f'to {mine}. An artifact is what was asked and what came back, so a mispaired one would re-solve '
            f'to an answer other than the one it holds. Pass the spec that was solved.'
        )


def _load_the_spec(artifact: SolveArtifact | SweepArtifact) -> None:
    """Normalise ``spec`` in place, and refuse a lowered program.

    Frozen dataclasses, so the loaded value goes back through ``object``: a
    field that is sometimes a path and sometimes a ``Spec`` would put the same
    branch in every reader.
    """
    if isinstance(artifact.spec, Program):
        raise LpspecError(
            'an artifact holds the model as written — a path, a mapping or a Spec — and a lowered '
            'Program has no file to write. Pass what it was lowered from.'
        )
    object.__setattr__(artifact, 'spec', to_spec(artifact.spec))


def _write(
    artifact: SolveArtifact | SweepArtifact,
    out: Path,
    checked: Mapping[str, Source],
    whole: Mapping[str, pl.LazyFrame],
    axis: EachCoordinate | EachWindow | None,
) -> Path:
    """The archive both artifacts write, differing only in *checked*, *whole* and *axis*.

    *checked* is the sources the declarations are checked against — all of
    them, or one slice. *whole* is what to write instead of the checked frame,
    which is how a sliced source reaches the archive carrying every slice's
    rows. The file lands whole or not at all.
    """
    program = lowered(artifact.spec)
    frames = supplied(program, tidy_sources(program, checked))
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + '.part')
    try:
        with zipfile.ZipFile(part, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(_MODEL_MEMBER, artifact.spec.to_yaml())
            for name, frame in frames.items():
                member = str(_SOURCES_DIR / f'{name}.parquet')
                given = artifact.sources.get(name)
                if isinstance(given, (str, Path)):
                    archive.write(given, member)
                else:
                    buffer = io.BytesIO()
                    whole.get(name, frame).collect().write_parquet(buffer, compression='zstd')
                    archive.writestr(member, buffer.getvalue())
            if axis is not None:
                archive.writestr(_AXIS_MEMBER, json.dumps(_axis_manifest(axis)))
            if artifact.answer is not None:
                _write_answer(archive, artifact.answer, beside=out.parent)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, out)
    return out


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


def load_artifact(path: str | Path, into: str | Path) -> SolveArtifact | SweepArtifact:
    """Read back an archive either artifact's ``save`` wrote.

    Which comes back is read off the archive, not asked for: it carries an
    axis or it does not. ``lps.solve(artifact.spec, artifact.sources)`` asks
    the question again — ``lps.solve_over(…, artifact.axis)`` for a sweep —
    and ``artifact.answer`` is what it answered the first time.

    Args:
        path: The zip file.
        into: The directory to extract to, created if it does not exist. The
            members land in it as the archive holds them, and the answer reads
            lazily off them, so it has to outlive what is read.

    Returns:
        A :class:`SweepArtifact` where the archive carries an axis and a
        :class:`SolveArtifact` where it does not, holding the model as
        written, its sources keyed as the file declares them, and the answer —
        ``None`` where the archive holds none.

    Raises:
        LanguageError: A ``model.yaml`` the language does not accept.
        LayoutError: A member outside the layout, or an answer whose layout
            has moved since it was written. Nothing is extracted.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    into = Path(into)
    with zipfile.ZipFile(path) as archive:
        members = [PurePosixPath(name) for name in archive.namelist() if not name.endswith('/')]
        strays = [str(m) for m in members if not _in_the_layout(m)]
        if strays or PurePosixPath(_MODEL_MEMBER) not in members:
            raise LayoutError(_not_an_archive_message(path, strays))
        archive.extractall(into)
    spec = to_spec(into / _MODEL_MEMBER)
    sources = {m.stem: into / m for m in members if _is_source_member(m)}
    carried = any(_ANSWER_DIR in m.parents for m in members)
    axis_member = into / _AXIS_MEMBER
    if not axis_member.is_file():
        answer = load_result(into / _ANSWER_DIR) if carried else None
        return SolveArtifact(spec, sources, answer)
    axis = _axis_from(json.loads(axis_member.read_text()))
    return SweepArtifact(spec, sources, axis, load_runs(into / _ANSWER_DIR) if carried else None)


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
