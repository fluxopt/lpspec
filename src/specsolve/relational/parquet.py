"""Answers on disk as parquet: the layout a result and a sweep both write, and the writer that lands a file whole.

Under a directory, ``<kind>/<name>`` for each of the three kinds a solve
answers with — the primals, the duals, the named expressions — so a
constraint carrying a variable's name never collides with it. A result
writes one file under each name; a sweep one per slice, and reads them back
as one. Beside them is the :class:`Record`, which says how the solve
terminated: a result writes one row, a sweep one per slice.

A saved result holds two things a sweep does not: ``activity/<name>`` for
every constraint, which no ``kind=`` names because no fold carries it, and
``reasons.parquet`` saying why a kind or a name is deliberately not there.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple, get_args, get_type_hints

import polars as pl

from specsolve.errors import LayoutError, SpecsolveError
from specsolve.relational.status import SolveStatus, status_of

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

#: The three kinds of frame a solve answers with, named after the reader each
#: comes back through, and what each is a frame of.
KINDS = ('primal', 'dual', 'expression')
LABELS = {'primal': 'variable', 'dual': 'constraint', 'expression': 'named expression'}


#: What the layout under a directory looks like. **Zero while the layout is
#: still moving**, and it starts counting at one the day that stops. So an
#: answer another zero-era build wrote reads as current, and what the stamp
#: catches is one written before there was a stamp. **Compared, never
#: branched on.**
ANSWER_FORMAT = 0
FORMAT_FILE = 'format.json'


def write_format(directory: Path) -> None:
    """Stamp *directory* with the layout its contents are in."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / FORMAT_FILE).write_text(json.dumps({'answer': ANSWER_FORMAT}))


def check_format(directory: Path) -> None:
    """Refuse a saved answer whose layout is not the one this package reads.

    Called after whatever identifies the directory as an answer at all, so a
    directory that is simply not one gets that message rather than this.

    Raises:
        LayoutError: A stamp that is not this package's, which is every
            answer written before there was one.
    """
    file = directory / FORMAT_FILE
    found = json.loads(file.read_text())['answer'] if file.is_file() else None
    if found != ANSWER_FORMAT:
        raise LayoutError(
            f'{str(directory)!r} holds a saved answer in layout {found}, and this package reads '
            f'{ANSWER_FORMAT}. The layout moves while the package is on 0.0.1aN and nothing reads an '
            f'older one back: solve the model again and save it. An archive that archive= wrote still '
            f'holds the model and the data to do that with.'
        )


#: How much of a sha256 a digest here keeps. Sixteen hex characters is 64
#: bits.
_DIGEST_WIDTH = 16


def digest_of_bytes(data: bytes) -> str:
    """A short, stable name for *data* — what the digests here are made with."""
    return hashlib.sha256(data).hexdigest()[:_DIGEST_WIDTH]


def digest_of_file(path: Path) -> str:
    """The same name for a file's bytes, read a chunk at a time."""
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
    return sha.hexdigest()[:_DIGEST_WIDTH]


def digest_of(yaml: str) -> str:
    """A short, stable name for a spec — what two answers must share to be comparable.

    Over the YAML a ``Spec`` round-trips to, which is exactly what an archive
    writes as ``model.yaml``: two answers carrying one digest answered the
    same document, byte for byte. Not the same *model* — that is the document
    with its data, and two scenarios of one spec share this and share nothing
    else. What an archive holds beside it says whether the data agreed too:
    :func:`digest_of_file` over each member of ``sources/``.
    """
    return digest_of_bytes(yaml.encode())


class Record(NamedTuple):
    """How a solve terminated, what it reached, and which spec it answered.

    One row per solve, and the same columns whoever wrote them: a result
    writes one, a sweep one per slice keyed by its own key. The only part of
    an answer the frames themselves cannot carry — a run that left no values
    writes this and nothing else.
    """

    status: str
    termination_condition: str
    #: What the solve reached, or ``None`` where it reached nothing. Null
    #: rather than ``nan``: nan is a *number* to every aggregate that meets it.
    #: :attr:`Result.objective` is a float and reads it back as ``nan``, having
    #: no null to return.
    objective: float | None
    #: Whether the solve produced values, which the condition alone does not
    #: say: a run stopped at a limit before any incumbent is ``ok`` with
    #: nothing to read.
    has_primal: bool
    #: :func:`digest_of` the spec this answered, or ``None`` where the solve
    #: was run off a lowered program and there was no document to digest. Null
    #: on disk, never an empty string.
    spec_digest: str | None
    #: When the solver returned, in UTC. ``None`` for a solve that carried no
    #: clock — a result built by hand, or read back from a record written
    #: before this column.
    solved_at: datetime | None = None
    #: What the archive holding this answer was called — its file name without
    #: a ``.zip``, so ``runs/nightly-2026-09-10.zip`` writes
    #: ``nightly-2026-09-10`` and a directory called ``case.v2`` keeps both
    #: halves of its name. Stamped when the archive is written and null until
    #: then.
    run: str | None = None
    #: :attr:`~specsolve.relational.sinks.handoff.Handoff.contents` of the model this
    #: answered — the spec *and* its data, where :attr:`spec_digest` is the
    #: document alone. ``None`` for an answer written before this column, and
    #: for one whose result was never asked for it.
    model_digest: str | None = None

    @classmethod
    def of(
        cls,
        termination_condition: str,
        objective: float,
        *,
        has_primal: bool,
        spec_digest: str | None,
        solved_at: datetime | None,
        model_digest: str | None = None,
    ) -> Record:
        """The row a solve that terminated this way writes.

        ``status`` is derived here rather than passed, and an objective is
        dropped to null here rather than at each writer.

        Args:
            termination_condition: What the solver said.
            objective: What the solve reached. Written only where there are
                values to read — ``nan`` is a *number* to every aggregate.
            has_primal: Whether there are values, which the condition alone
                does not say.
            spec_digest: :func:`digest_of` the spec answered, or ``None``.
            solved_at: When the solver returned, in UTC. ``None`` where the
                solve carried no clock.
            model_digest: The built model's digest, or ``None`` where this
                answer never held one.
        """
        return cls(
            status_of(termination_condition),
            termination_condition,
            objective if has_primal else None,
            has_primal,
            spec_digest,
            solved_at,
            model_digest=model_digest,
        )

    @property
    def solve_status(self) -> SolveStatus:
        """The status this row records — the way back from columns.

        The solver's own wording is gone, and ``status`` is derived again
        rather than read off the row.
        """
        return SolveStatus(self.termination_condition, has_primal=self.has_primal)


#: What each Python type a record column is annotated with is written as.
#: A column whose annotation is not here fails at import rather than at the
#: write.
_WRITTEN_AS: Mapping[type, pl.DataType | type[pl.DataType]] = {
    str: pl.String,
    float: pl.Float64,
    bool: pl.Boolean,
    int: pl.Int64,
    #: Carried with its zone, not as a naive column claiming to be UTC.
    datetime: pl.Datetime(time_zone='UTC'),
}


def _column_types(record: type[NamedTuple]) -> dict[str, pl.DataType | type[pl.DataType]]:
    """*record*'s columns as they are written, off its own annotations.

    ``X | None`` is written as ``X`` holding null.
    """
    written: dict[str, pl.DataType | type[pl.DataType]] = {}
    for name, hint in get_type_hints(record).items():
        declared = next((arg for arg in get_args(hint) if arg is not type(None)), hint)
        if declared not in _WRITTEN_AS:
            raise SpecsolveError(
                f'{record.__name__}.{name} is annotated {declared!r}, which nothing here writes. A record '
                f'column has to say what type it is written as: add it to _WRITTEN_AS.'
            )
        written[name] = _WRITTEN_AS[declared]
    return written


#: :class:`Record`'s columns as they are written, so a row whose ``objective``
#: or ``spec_digest`` is absent writes that column's own type holding null
#: rather than the ``Null`` one polars would infer from a single row. Passed
#: as ``schema_overrides``, so a sweep's key column beside them keeps the type
#: its own value infers to.
RECORD_SCHEMA = _column_types(Record)


class Metrics(NamedTuple):
    """What a build and its solves took, as the row an archive records beside the answer.

    :class:`Record`'s sibling — one says how the solve terminated, this is the
    measure of what it took — and the same columns whoever writes them, so
    rows written by runs that never met concatenate into one table.

    The scalars of :class:`~specsolve.relational.result.Diagnostics` and none of
    its frames: a coefficient range is a table per declaration, which does not
    fold into a row beside a count.

    **Cumulative over the model's life**, as every counter it is read off is.
    :attr:`solves` says how many solves the clocks cover; it reads ``1`` for
    the archive :func:`specsolve.solve` writes, that verb building the model it
    solves.
    """

    #: The shape the build produced, in the solver's own vocabulary.
    columns: int
    rows: int
    nonzeros: int
    #: How many solves the row covers, and how many of those loaded the solver
    #: from scratch. Read together with the clocks, which are cumulative over
    #: exactly these solves.
    solves: int
    loads: int
    #: Wall-clock seconds in each phase a build clocks, in the order they run:
    #: the caller's sources onto the plan, the declarations into the model
    #: frames, the built model into a solver, the solver's own run, and the
    #: built model streamed to an LP or MPS file. A phase that never ran writes
    #: zero rather than no column.
    #:
    #: So :attr:`write_seconds` reads zero on an archive whose caller never
    #: asked for a file, which is most of them: it is
    #: :meth:`~specsolve.api.Model.write`'s clock rather than the archive's own. **What writing the archive cost is
    #: not here and is not anywhere**: a caller who wants that number times the
    #: call.
    attach_seconds: float
    build_seconds: float
    handoff_seconds: float
    solve_seconds: float
    write_seconds: float
    #: What the archive holding this row was called, as :attr:`Record.run` is
    #: stamped onto the record beside it: the archive's file name without a
    #: ``.zip``. Null until one is written.
    run: str | None = None


#: :class:`Metrics`'s columns as they are written, as :data:`RECORD_SCHEMA`.
METRICS_SCHEMA = _column_types(Metrics)


class SliceMetrics(NamedTuple):
    """What one slice of a sweep took — :class:`Metrics` one dimension in.

    Not the same columns, and the fold is what separates them. A slice's clocks
    are its own share rather than a cumulative total; ``loaded`` says whether
    the solver took this slice from scratch, where a whole model counts its
    loads; and what a sink added, how many solves ran and what a file write
    took are facts about a model's life that one slice of a sweep has no share
    of.

    Written per slice by the spill and read back as one table, so a sweep's
    every slice concatenates the way a directory of archives does.
    """

    #: The shape this slice built, as :class:`Metrics` reports a whole model's.
    columns: int
    rows: int
    nonzeros: int
    #: Whether the solver took this slice's model from scratch instead of
    #: having values pushed onto one it already held. Under a serial fold the
    #: first slice does and the rest do not, so a later ``True`` is a slice
    #: whose data moved a mask; under an executor every slice loads.
    loaded: bool
    #: This slice's own seconds per phase, so a slow sweep says which slice and
    #: which phase of it. A whole model's ``write`` has no per-slice meaning —
    #: a sweep writes no file per slice — and there is no column for it.
    attach_seconds: float
    build_seconds: float
    handoff_seconds: float
    solve_seconds: float


def row_of[R](row_type: Callable[..., R], columns: Mapping[str, object], found: Path) -> R:
    """One row read off disk as the type that declares its columns.

    A file short of a column or carrying one nothing declares is a sentence
    naming it rather than a frame of the wrong shape folded into whatever reads
    it next.

    Args:
        row_type: :class:`Record`, :class:`Metrics` or :class:`SliceMetrics`.
        columns: The row as read, ``name: value``.
        found: What to name in the message — the file or directory it came from.

    Raises:
        LayoutError: The columns are not the ones *row_type* declares.
    """
    declared = set(row_type._fields)  # pyrefly: ignore[missing-attribute] — every caller passes a NamedTuple
    if (missing := sorted(declared - set(columns))) or (stray := sorted(set(columns) - declared)):
        short = f'is short of {missing}' if missing else f'holds {stray}'
        raise LayoutError(
            f'{str(found)!r} holds a saved {row_type.__name__} row that {short}, so it was written in a '
            f'layout this package does not read. The layout moves while the package is on 0.0.1aN and '
            f'nothing reads an older one back: solve the model again and save it.'
        )
    return row_type(**columns)


#: The three files that sit beside the frames — a result and a sweep both
#: write them, and :func:`specsolve.archive.load_archive` reads back whichever
#: wrote: the record of how the solve terminated, what reaching it cost, and
#: the reasons behind whatever is deliberately not there.
RECORD_FILE = 'record.parquet'
METRICS_FILE = 'metrics.parquet'
REASONS_FILE = 'reasons.parquet'


def consolidated(under: Path, file: str) -> pl.DataFrame:
    """The table *file* names under *under*, whichever shape wrote it, as one frame.

    A spill writes it one file per slice under a directory named for what the
    file holds — ``record.parquet`` beside ``record/`` — so the name of
    one gives the other and only the file is passed.

    Reads both shapes, so one reader serves an archive and the spill it was
    packed from. Rows stay in slice order, the files being named by position.

    Raises:
        LayoutError: Neither shape is under *under*. Every record here is
            written as the fold goes, whether or not a slice produced values,
            so a directory holding neither was not written by this package.
    """
    apart = file.removesuffix('.parquet')
    if (single := under / file).is_file():
        frames = [pl.read_parquet(single)]
    elif (many := under / apart).is_dir():
        frames = [pl.read_parquet(path) for path in sorted(many.glob('*.parquet'))]
    else:
        raise LayoutError(
            f'{str(under)!r} holds no {file!r} and no {apart!r} beside it, so it is not a sweep or a saved '
            f'answer this package wrote. Every record here is written as the fold goes — one file, or one '
            f'per slice — whether or not a slice produced values.'
        )
    return pl.concat(frames)


def clear_the_answer(directory: Path) -> None:
    """Remove what a saved answer holds, leaving anything else in *directory* alone.

    Only the layout's own members go, so a directory the caller also keeps other
    files in survives.
    """
    import shutil

    for kind in (*KINDS, 'activity'):
        shutil.rmtree(directory / kind, ignore_errors=True)
    for member in (RECORD_FILE, METRICS_FILE, REASONS_FILE, FORMAT_FILE):
        (directory / member).unlink(missing_ok=True)


def write_reasons(directory: Path, no_duals: str | None, no_expressions: Mapping[str, str]) -> None:
    """``(kind, name, reason)`` for what a solve could not produce, or no file at all.

    An empty *name* is the whole kind, which is how the duals are absent —
    an integer variable makes every one of them undefined, never one
    constraint's. Written only when there is something to say, the way a kind
    with no values writes no directory.
    """
    rows = [] if no_duals is None else [{'kind': 'dual', 'name': '', 'reason': no_duals}]
    rows += [{'kind': 'expression', 'name': name, 'reason': why} for name, why in no_expressions.items()]
    if rows:
        write_whole(pl.DataFrame(rows), directory / REASONS_FILE)


def read_reasons(directory: Path) -> tuple[str | None, dict[str, str]]:
    """What :func:`write_reasons` wrote: the duals' reason, and one per named expression."""
    file = directory / REASONS_FILE
    rows: list[tuple[str, str, str]] = pl.read_parquet(file).rows() if file.is_file() else []
    return (
        next((why for kind, _, why in rows if kind == 'dual'), None),
        {name: why for kind, name, why in rows if kind == 'expression'},
    )


def reader_kind(kind: str) -> str:
    """*kind* as one of :data:`KINDS`, which every reader that takes a name takes beside it.

    Raises:
        SpecsolveError: A *kind* that names no reader.
    """
    if kind not in KINDS:
        raise SpecsolveError(f'kind is one of {", ".join(KINDS)}, not {kind!r}')
    return kind


def write_whole(frame: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
    """*frame* at *path*, arriving whole: written beside it and renamed into place.

    A lazy frame is sunk, so it streams to disk without passing through this
    process; a reader that finds *path* finds all of it, and one that finds
    only the ``.part`` beside it finds a write that did not finish.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    if isinstance(frame, pl.LazyFrame):
        frame.sink_parquet(part)
    else:
        frame.write_parquet(part)
    part.replace(path)
