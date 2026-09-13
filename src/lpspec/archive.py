"""What comes back out of an archive: a spec, the data it was solved with, and what came back.

An answer alone cannot say which model produced it, and a model alone has to
be solved again to be read. An archive holds both, and :func:`load_archive`
gives them back as one of two values — :class:`SolveArchive` for one solve,
:class:`SweepArchive` where the sources were cut.

**Two readers, one pair of values.** :func:`load_archive` reads it whole, so
what comes back is in memory and owes the archive nothing. :func:`scan_archive`
leaves the frames where they lie and reads each at the call that asks for it,
so the members have to outlive it; that is the reader for an archive larger
than memory, or one most of whose names go unread. Which one ran shows in two
places and nowhere else: a source is a table or the path to one, and the answer
collects or scans.

**Reading only.** Nothing here writes an archive: the verb that solves does,
through :mod:`lpspec.layout`, because a solve is the one moment the spec, the
data and the answer exist together. Assembling the three after the fact is
what let a mispaired case be filed as a matching one.

Above ``api`` and ``strategy`` rather than beside them: an archive carries
either kind of answer, so it is the one place that knows about both a
``Result`` and a ``Runs``. Neither of them knows about it.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from math_spec import to_spec

from lpspec.api import load_result, scan_result
from lpspec.errors import LayoutError, LpspecError
from lpspec.layout import (
    ANSWER_DIR,
    AXIS_MEMBER,
    DIGESTS_MEMBER,
    MODEL_MEMBER,
    is_source_member,
    members_of,
    opened,
)
from lpspec.relational.parquet import COST_FILE, digest_of
from lpspec.strategy import EachCoordinate, EachWindow, Runs, axis_from, load_runs, scan_runs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from math_spec import Spec

    from lpspec.lanes import Source
    from lpspec.relational.result import Result

__all__ = ['SolveArchive', 'SweepArchive', 'load_archive', 'scan_archive']


@dataclass(frozen=True)
class SolveArchive:
    """A spec, the data it was solved with, and what one solve of it returned.

    What :func:`load_archive` and :func:`scan_archive` give back for an archive
    whose sources were not cut. ``lps.solve(archive.spec, archive.sources)``
    asks the question again, and :attr:`answer` is what it answered the first
    time.

    Attributes:
        spec: The spec as written.
        sources: What was attached to it, keyed as the file declares it — the
            table each parquet member holds from :func:`load_archive`, and the
            path to it from :func:`scan_archive`, a ``Path`` being a source
            like any other, so attaching streams it from disk instead.
        answer: What came back — read whole or read as its readers are called,
            as the two verbs differ.
        source_digests: ``(run, source, digest)``, one row per member of
            ``sources/``, in source order. What
            :attr:`~lpspec.relational.result.Result.spec_digest` cannot say:
            two archives of one spec over different numbers carry the same
            spec digest and differ here, and *which* rows differ names the
            input that moved. ``run`` is the archive's own name, as it is on
            the record and the cost row, so a table read across a directory of
            archives attributes its rows without parsing paths.
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
    source_digests: pl.DataFrame
    diagnostics: pl.DataFrame

    def __post_init__(self) -> None:
        """Refuse an archive whose answer names a different model than its own."""
        _check_the_pairing(self.spec, (self.answer.spec_digest,))


@dataclass(frozen=True)
class SweepArchive:
    """A spec, the data a sweep was solved over, the axis that cut it, and what came back.

    :class:`SolveArchive`'s sibling, and the axis is what separates them: a
    sweep's sources carry the column it slices on, which the model does not
    declare, so they are legible only beside it.

    ``lps.solve_over(sweep.spec, sweep.sources, sweep.axis)`` runs it again.

    Attributes:
        spec: The spec as written, as :class:`SolveArchive` holds it.
        sources: What the sweep was given — **uncut**, carrying every slice's
            rows, because one copy per slice is what a sweep exists not to
            write. A table or a path, as :class:`SolveArchive` holds them.
        axis: :class:`~lpspec.strategy.EachCoordinate` or
            :class:`~lpspec.strategy.EachWindow`, the axis that cut them.
        answer: Every slice's answers, keyed by slice — **held** from
            :func:`load_archive`, so :meth:`~lpspec.strategy.Runs.primal` and
            its siblings answer, and **spilled** from :func:`scan_archive`,
            the frames staying in the extracted directory for
            :meth:`~lpspec.strategy.Runs.scan` to read.
        source_digests: ``(run, source, digest)``, as :class:`SolveArchive`
            holds it. Of the sources **uncut**, which is how the archive holds
            them, so it names the data the sweep was cut from rather than any
            slice's share of it.
    """

    spec: Spec
    sources: Mapping[str, Source]
    axis: EachCoordinate | EachWindow
    answer: Runs
    source_digests: pl.DataFrame

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


def _digests_in(under: Path) -> pl.DataFrame:
    """The ``(run, source, digest)`` table *under* holds.

    Read, never re-computed: verifying it means hashing every source, which is
    a pass over all the data an archive holds and is the caller's to ask for
    on the occasion they want it checked rather than this package's to spend
    on every load.

    Raises:
        LayoutError: An archive with no digest table, which is every one
            written before there was one.
    """
    file = under / DIGESTS_MEMBER
    if not file.is_file():
        raise LayoutError(
            f'{str(under)!r} holds no {DIGESTS_MEMBER!r}, so this archive was written before one digested '
            f'the data beside the model. Solving the model it holds again writes an archive that carries '
            f'both, and the sources to do that with are in this one.'
        )
    return pl.read_parquet(file)


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
    """Read back an archive an ``archive=`` wrote, whole.

    Which comes back is read off the archive, not asked for: it carries an
    axis or it does not.

    **Everything is in memory when this returns** — the sources as the tables
    they hold, the answer as the frames it holds — so what comes back owes the
    archive nothing afterwards. A zip needs somewhere to unpack all the same,
    but only for the duration: with no *into* it goes to a scratch directory
    that is gone by the time the value is handed back. An archive larger than
    memory is :func:`scan_archive`.

    Args:
        path: The archive — a ``.zip``, or the directory one was written to.
        into: Where to unpack a zip, created if it does not exist, for a
            caller who wants the extracted tree as well as the value — to
            query with an engine that reads parquet, say. A directory archive
            needs none, being read where it lies; passing one is refused.

    Returns:
        A :class:`SweepArchive` where the archive carries an axis and a
        :class:`SolveArchive` where it does not, holding the spec as written,
        its sources keyed as the file declares them, a digest of each, the
        answer, and what reaching it cost.

    Raises:
        LanguageError: A ``model.yaml`` the language does not accept.
        LayoutError: A member outside the layout or an *into* given for a
            directory, neither of which unpacks anything, and — once it is —
            an archive holding no digest table, or an answer whose layout has
            moved since it was written or that holds no cost row.
        LpspecError: An answer that names a different model than the one
            beside it.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    held = Path(path)
    if into is not None or held.is_dir():
        return _archive_under(opened(held, None if into is None else Path(into)), whole=True)
    with tempfile.TemporaryDirectory() as scratch:
        return _archive_under(opened(held, Path(scratch)), whole=True)


def scan_archive(path: str | Path, into: str | Path | None = None) -> SolveArchive | SweepArchive:
    """The same archive, read as its readers are called rather than now.

    :func:`load_archive`'s other half, and the same two types. What differs is
    that nothing but the spec, the axis, the digest table and the cost row is
    read: the sources come back as the parquet paths they now are — a ``Path``
    being a source like any other, so attaching streams them from disk — and
    the answer reads each frame at the call that asks for it
    (:func:`~lpspec.api.scan_result`, :func:`~lpspec.strategy.scan_runs`).

    So **the members have to outlive what was read off them**: *into* is
    required for a zip here, and kept.

    Args:
        path: As :func:`load_archive` takes it.
        into: Where to unpack a zip, created if it does not exist and **kept**.
            Required for a zip, there being nothing to read off once a scratch
            directory is gone; refused for a directory archive, as above.

    Raises:
        LayoutError: As :func:`load_archive` raises it, and a zip with no
            *into*.
        LpspecError: As :func:`load_archive` raises it.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    return _archive_under(opened(path, None if into is None else Path(into)), whole=False)


def _archive_under(under: Path, *, whole: bool) -> SolveArchive | SweepArchive:
    """The archive whose members are in *under*, read whole or left on disk.

    The one body behind :func:`load_archive` and :func:`scan_archive`: what an
    archive holds does not depend on when its bytes move, so only the reading
    differs. Both halves of that reading — a source, and the answer — turn on
    the one flag, there being no archive whose sources are held and whose
    answer is not.
    """
    spec = to_spec(under / MODEL_MEMBER)
    sources: dict[str, Source] = {
        m.stem: pl.read_parquet(under / m) if whole else under / m for m in members_of(under) if is_source_member(m)
    }
    digests = _digests_in(under)
    axis_member = under / AXIS_MEMBER
    if not axis_member.is_file():
        saved = under / ANSWER_DIR
        answer = (load_result if whole else scan_result)(saved)
        return SolveArchive(spec, sources, answer, digests, _cost_in(saved))
    axis = axis_from(json.loads(axis_member.read_text()))
    slices = (load_runs if whole else scan_runs)(under / ANSWER_DIR)
    return SweepArchive(spec, sources, axis, slices, digests)
