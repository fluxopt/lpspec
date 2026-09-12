"""What comes back out of an archive: a model, the data it was solved with, and what came back.

An answer alone cannot say which model produced it, and a model alone has to
be solved again to be read. An archive holds both, and :func:`load_archive`
gives them back as one of two values — :class:`SolveArchive` for one solve,
:class:`SweepArchive` where the sources were cut.

**Reading only.** Nothing here writes an archive: the verb that solves does,
through :mod:`lpspec.layout`, because a solve is the one moment the model, the
data and the answer exist together. Assembling the three after the fact is
what let a mispaired case be filed as a matching one.

Above ``api`` and ``strategy`` rather than beside them: an artifact carries
either kind of answer, so it is the one place that knows about both a
``Result`` and a ``Runs``. Neither of them knows about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from math_spec import to_spec

from lpspec.api import load_result
from lpspec.errors import LayoutError, LpspecError
from lpspec.layout import ANSWER_DIR, AXIS_MEMBER, MODEL_MEMBER, is_source_member, members_of, opened
from lpspec.relational.parquet import COST_FILE, digest_of
from lpspec.strategy import EachCoordinate, EachWindow, Runs, axis_from, load_runs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from math_spec import Spec

    from lpspec.lanes import Source
    from lpspec.relational.result import Result

__all__ = ['SolveArchive', 'SweepArchive', 'load_archive']


@dataclass(frozen=True)
class SolveArchive:
    """A model, the data it was solved with, and what one solve of it returned.

    What :func:`load_archive` gives back for an archive whose sources were
    not cut. ``lps.solve(artifact.spec, artifact.sources)`` asks the question
    again, and :attr:`answer` is what it answered the first time.

    Attributes:
        spec: The model as written.
        sources: What was attached to it, as the parquet files the archive
            holds — a ``Path`` being a source like any other, so attaching
            streams them from disk.
        answer: What came back.
        diagnostics: One :class:`~lpspec.relational.parquet.Cost` row — what
            the build and its solves spent reaching that answer. Beside
            :attr:`answer` rather than on it, which is the asymmetry with
            :class:`SweepArchive`, where ``answer.diagnostics`` carries the
            same columns one per slice: a :class:`~lpspec.strategy.Runs` is a
            fold and knows each slice's share of a cumulative reading, where a
            :class:`~lpspec.relational.result.Result` is one solve of a model
            that may have had many and could only hold a number it has no way
            to attribute.
    """

    spec: Spec
    sources: Mapping[str, Source]
    answer: Result
    diagnostics: pl.DataFrame

    def __post_init__(self) -> None:
        """Refuse an archive whose answer names a different model than its own."""
        _check_the_pairing(self.spec, (self.answer.spec_digest,))


@dataclass(frozen=True)
class SweepArchive:
    """A model, the data a sweep was solved over, the axis that cut it, and what came back.

    :class:`SolveArchive`'s sibling, and the axis is what separates them: a
    sweep's sources carry the column it slices on, which the model does not
    declare, so they are legible only beside it.

    ``lps.solve_over(sweep.spec, sweep.sources, sweep.axis)`` runs it again.

    Attributes:
        spec: The model as written, as :class:`SolveArchive` holds it.
        sources: What the sweep was given — **whole**, carrying every slice's
            rows, because one copy per slice is what a sweep exists not to
            write.
        axis: :class:`~lpspec.strategy.EachCoordinate` or
            :class:`~lpspec.strategy.EachWindow`, the axis that cut them.
        answer: Every slice's answers, keyed by slice, and **spilled**: the
            frames stay in the extracted directory and
            :meth:`~lpspec.strategy.Runs.scan` reads them.
    """

    spec: Spec
    sources: Mapping[str, Source]
    axis: EachCoordinate | EachWindow
    answer: Runs

    def __post_init__(self) -> None:
        """Refuse an archive whose slices name a different model than its own."""
        _check_the_pairing(self.spec, self.answer.objective['spec_digest'].to_list())


def _check_the_pairing(spec: Spec, answered: Sequence[str | None]) -> None:
    """Refuse an archive whose answer came back from a different model than the one beside it.

    A solve writes both members together, so this cannot fire on an archive
    this package wrote — it is what stands between a hand-assembled or edited
    zip and a reader who would take it at its word and re-solve to something
    else. An answer solved off a lowered program digests to ``None`` and is
    taken on trust; there is no document to compare it against.
    """
    mine = digest_of(spec.to_yaml())
    if others := sorted({other for other in answered if other is not None and other != mine}):
        raise LpspecError(
            f'this archive holds an answer that came back from a different model: the answer carries '
            f'{others} and the model.yaml beside it digests to {mine}. Re-solving it would give an answer '
            f'other than the one it holds, so it is not read.'
        )


def _cost_in(answer: Path) -> pl.DataFrame:
    """The cost row *answer* holds, read whole — it is one row.

    Raises:
        LayoutError: An answer with no cost row, which is every archive
            written before one was recorded. The stamp beside it cannot say
            so: the layout number is held at zero while the layout moves.
    """
    file = answer / COST_FILE
    if not file.is_file():
        raise LayoutError(
            f'{str(answer)!r} holds no {COST_FILE!r}, so this archive was written before one recorded what '
            f'its solve cost. What a build and its solves spent is of the machine that ran them and cannot '
            f'be recovered by re-solving here; everything else in the archive can, and solving the model it '
            f'holds again writes an archive that carries both.'
        )
    return pl.read_parquet(file)


def load_archive(path: str | Path, into: str | Path | None = None) -> SolveArchive | SweepArchive:
    """Read back an archive an ``archive=`` wrote.

    Which comes back is read off the archive, not asked for: it carries an
    axis or it does not.

    Args:
        path: The archive — a ``.zip``, or the directory one was written to.
        into: Where to unpack a zip, created if it does not exist. The members
            land in it as the archive holds them and the answer reads lazily
            off them, so it has to outlive what is read. A directory archive
            needs none, being read where it lies; passing one is refused.

    Returns:
        A :class:`SweepArchive` where the archive carries an axis and a
        :class:`SolveArchive` where it does not, holding the model as written,
        its sources keyed as the file declares them, the answer, and what
        reaching it cost.

    Raises:
        LanguageError: A ``model.yaml`` the language does not accept.
        LayoutError: A member outside the layout, a zip with no *into*, an
            *into* given for a directory, an answer whose layout has moved
            since it was written, or one holding no cost row. Nothing is
            unpacked.
        LpspecError: An answer that names a different model than the one
            beside it.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    under = opened(path, None if into is None else Path(into))
    spec = to_spec(under / MODEL_MEMBER)
    sources = {m.stem: under / m for m in members_of(under) if is_source_member(m)}
    axis_member = under / AXIS_MEMBER
    if not axis_member.is_file():
        return SolveArchive(spec, sources, load_result(under / ANSWER_DIR), _cost_in(under / ANSWER_DIR))
    axis = axis_from(json.loads(axis_member.read_text()))
    return SweepArchive(spec, sources, axis, load_runs(under / ANSWER_DIR))
