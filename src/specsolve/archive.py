"""Reading an archive back: the spec, the data it was solved with, and what came back.

:func:`load_archive` reads it whole; :func:`scan_archive` leaves the frames on
disk and reads each at the call that asks for it. Either gives back a
:class:`SolveArchive` for one solve, or a :class:`SweepArchive` where the
sources were cut. Nothing here writes one: ``archive=`` on the verbs that
solve does, through :mod:`specsolve.layout`.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from math_spec import to_spec

from specsolve.api import attach_readers, load_result, scan_result
from specsolve.errors import SpecsolveError
from specsolve.layout import ANSWER_DIR, AXIS_MEMBER, DIGESTS_MEMBER, MODEL_MEMBER, SOURCES_DIR, opened
from specsolve.relational.parquet import METRICS_FILE, Metrics, digest_of, row_of
from specsolve.strategy import (
    EachCoordinate,
    EachWindow,
    Runs,
    attach_sweep_readers,
    axis_from,
    load_runs,
    scan_runs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from math_spec import Spec

    from specsolve.lanes import Source
    from specsolve.relational.result import Result

__all__ = ['SolveArchive', 'SweepArchive', 'load_archive', 'scan_archive']


@dataclass(frozen=True)
class SolveArchive:
    """A spec, the data it was solved with, and what one solve of it returned.

    ``sps.solve(archive.spec, archive.sources)`` asks the question again.

    Attributes:
        spec: The spec as written.
        sources: What was attached, keyed as the file declares it: a table
            from :func:`load_archive`, the path to one from :func:`scan_archive`.
        answer: What came back.
        source_digests: ``(run, source, digest)``, one row per source, so two
            archives of one spec over different numbers name the input that
            moved.
        metrics: What reaching the answer took, as one
            :class:`~specsolve.relational.parquet.Metrics`.
    """

    spec: Spec
    sources: Mapping[str, Source]
    answer: Result
    source_digests: pl.DataFrame
    metrics: Metrics


@dataclass(frozen=True)
class SweepArchive:
    """A spec, the data a sweep was solved over, the axis that cut it, and what came back.

    ``sps.solve_over(sweep.spec, sweep.sources, sweep.axis, carry=sweep.carry)``
    runs it again.

    Attributes:
        spec: The spec as written.
        sources: What the sweep was given, uncut. A table or a path, as
            :class:`SolveArchive` holds them.
        axis: What cut them.
        carry: ``{parameter: variable}`` the slices were chained with, empty
            where they were not.
        answer: Every slice's answer, keyed by slice. Held from
            :func:`load_archive`, spilled from :func:`scan_archive`.
        source_digests: As :class:`SolveArchive` holds it, of the uncut
            sources.
    """

    spec: Spec
    sources: Mapping[str, Source]
    axis: EachCoordinate | EachWindow
    carry: Mapping[str, str]
    answer: Runs
    source_digests: pl.DataFrame


def load_archive(path: str | Path, into: str | Path | None = None) -> SolveArchive | SweepArchive:
    """Read an archive back whole: the sources as tables, the answer's frames in memory.

    Args:
        path: The archive, a ``.zip`` or the directory one was written to.
        into: Where to unpack a zip, kept afterwards, for a caller who wants
            the extracted tree as well. Without it a zip unpacks to a scratch
            directory that is gone when this returns. Refused for a directory
            archive, which is read where it lies.

    Returns:
        A :class:`SweepArchive` where the archive carries an axis, a
        :class:`SolveArchive` where it does not.

    Raises:
        LanguageError: A ``model.yaml`` the language does not accept.
        LayoutError: A member outside the layout, an *into* given for a
            directory, or an answer whose layout has moved since it was
            written.
        SpecsolveError: An answer that names a different model than the one
            beside it.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    held = Path(path)
    if into is not None or held.is_dir():
        return _read(opened(held, into), whole=True)
    with tempfile.TemporaryDirectory() as scratch:
        return _read(opened(held, scratch), whole=True)


def scan_archive(path: str | Path, into: str | Path | None = None) -> SolveArchive | SweepArchive:
    """Read an archive back off disk: the sources as paths, each frame read at the call that asks for it.

    The members have to outlive the value, so *into* is required for a zip
    and kept. The same values and the same errors as :func:`load_archive`,
    and ``LayoutError`` for a zip with no *into*.
    """
    return _read(opened(path, into), whole=False)


def _read(under: Path, *, whole: bool) -> SolveArchive | SweepArchive:
    spec = to_spec(under / MODEL_MEMBER)
    sources: dict[str, Source] = {
        member.stem: pl.read_parquet(member) if whole else member
        for member in sorted((under / SOURCES_DIR).glob('*.parquet'))
    }
    digests = pl.read_parquet(under / DIGESTS_MEMBER)
    saved = under / ANSWER_DIR
    axis_member = under / AXIS_MEMBER
    if not axis_member.is_file():
        answer = attach_readers((load_result if whole else scan_result)(saved), spec, sources)
        _check_the_pairing(spec, [answer.spec_digest])
        metrics = row_of(Metrics, pl.read_parquet(saved / METRICS_FILE).row(0, named=True), saved / METRICS_FILE)
        return SolveArchive(spec, sources, answer, digests, metrics)
    manifest = json.loads(axis_member.read_text())
    axis, carry = axis_from(manifest), manifest.get('carry', {})
    answer = attach_sweep_readers((load_runs if whole else scan_runs)(saved), spec, sources, axis, carry)
    _check_the_pairing(spec, answer.objective['spec_digest'].to_list())
    return SweepArchive(spec, sources, axis, carry, answer, digests)


def _check_the_pairing(spec: Spec, answered: Sequence[str | None]) -> None:
    """Refuse an archive whose answer came back from a different model than the one beside it.

    A solve writes both together, so this catches a hand-edited archive. A
    ``None`` digest is an answer solved off a lowered program and is not
    compared.
    """
    mine = digest_of(spec.to_yaml())
    if others := sorted({other for other in answered if other is not None and other != mine}):
        raise SpecsolveError(
            f'this archive holds an answer that came back from a different model: the answer carries '
            f'{others} and the model.yaml beside it digests to {mine}, so re-solving it would give another answer.'
        )
