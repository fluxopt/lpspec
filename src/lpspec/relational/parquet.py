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

from lpspec.errors import LayoutError, LpspecError
from lpspec.relational.status import SolveStatus, status_of

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: The three kinds of frame a solve answers with, named after the reader each
#: comes back through, and what each is a frame of.
KINDS = ('primal', 'dual', 'expression')
LABELS = {'primal': 'variable', 'dual': 'constraint', 'expression': 'named expression'}


#: What the layout under a directory looks like. **Zero while the layout is
#: still moving**, and it starts counting at one the day that stops: a number
#: spent on every move — a column added, a column's type changed, a kind that
#: writes a new directory — says only that something changed, and the
#: ``0.0.1aN`` stream says that already. So an answer another zero-era build
#: wrote reads as current, and what the stamp catches is one written before
#: there was a stamp. **Compared, never branched on.** The project holds no
#: compatibility promise, so there is no version this reads an old layout
#: back through; the number exists to turn a missing column into a sentence
#: naming the recovery. The day something writes ``if version == 0`` it has
#: become the deprecation path this project refuses.
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


def digest_of(yaml: str) -> str:
    """A short, stable name for a spec — what two answers must share to be comparable.

    Over the YAML a ``Spec`` round-trips to, which is exactly what an archive
    writes as ``model.yaml``: two answers carrying one digest answered the
    same document, byte for byte. Not the same *model* — that is the document
    with its data, and two scenarios of one spec share this and share nothing
    else. Sixteen hex characters, because this is read by a person scanning a
    comparison table beside four other columns and sixty-four would push the
    numbers off the line.
    """
    return hashlib.sha256(yaml.encode()).hexdigest()[:16]


class Record(NamedTuple):
    """How a solve terminated, what it reached, and which spec it answered.

    One row per solve, and the same columns whoever wrote them: a result
    writes one, a sweep one per slice keyed by its own key. That is what lets
    cases solved apart concatenate into the table a sweep folds, and it is the
    only part of an answer the frames themselves cannot carry — a run that
    left no values writes this and nothing else.
    """

    status: str
    termination_condition: str
    #: What the solve reached, or ``None`` where it reached nothing. Null
    #: rather than ``nan`` because this is a table column: nan is a *number*
    #: to every aggregate that meets it, so one infeasible case among a
    #: hundred would make the mean of the hundred nan, here and in any engine
    #: reading the same files. :attr:`Result.objective` is a float and reads
    #: it back as ``nan``, having no null to return.
    objective: float | None
    #: Whether the solve produced values, which the condition alone does not
    #: say: a run stopped at a limit before any incumbent is ``ok`` with
    #: nothing to read.
    has_primal: bool
    #: :func:`digest_of` the spec this answered, or ``None`` where the solve
    #: was run off a lowered program and there was no document to digest. Null
    #: on disk, never an empty string, so a comparison table counts the
    #: answers that named a document rather than one more distinct value.
    #: Carried so that answers written apart can be *told* to be comparable:
    #: one distinct value across a concatenated table means one spec.
    spec_digest: str | None
    #: When the solver returned, in UTC. Written so that a table concatenated
    #: from runs solved apart can be ordered without reading the paths they
    #: came from. ``None`` for a solve that carried no clock — a result built
    #: by hand, or read back from a record written before this column.
    solved_at: datetime | None = None
    #: What the archive holding this answer was called — its file name without
    #: a ``.zip``, so ``runs/nightly-2026-09-10.zip`` writes
    #: ``nightly-2026-09-10`` and a directory called ``case.v2`` keeps both
    #: halves of its name. Stamped
    #: when the archive is written and null until then, because the name is
    #: the publisher's rather than the solve's. It is the column a warehouse
    #: of runs joins on, which is why it is here and not left to whoever
    #: parses the file paths.
    run: str | None = None

    @classmethod
    def of(
        cls,
        termination_condition: str,
        objective: float,
        *,
        has_primal: bool,
        spec_digest: str | None,
        solved_at: datetime | None = None,
    ) -> Record:
        """The row a solve that terminated this way writes.

        The one home for how an answer becomes columns: ``status`` is derived
        here rather than passed, and an objective is dropped to null here
        rather than at each writer. Keyword-only past the condition because
        ``status`` and ``termination_condition`` are both strings, so a
        positional call is one field order away from writing a wrong file that
        no type checker and no test would object to.

        Args:
            termination_condition: What the solver said.
            objective: What the solve reached. Written only where there are
                values to read — ``nan`` is a *number* to every aggregate.
            has_primal: Whether there are values, which the condition alone
                does not say.
            spec_digest: :func:`digest_of` the spec answered, or ``None``.
            solved_at: When the solver returned, in UTC. ``None`` where the
                solve carried no clock.
        """
        return cls(
            status_of(termination_condition),
            termination_condition,
            objective if has_primal else None,
            has_primal,
            spec_digest,
            solved_at,
        )

    @property
    def solve_status(self) -> SolveStatus:
        """The status this row records — the way back from columns.

        The solver's own wording is gone, being a sentence to read rather than
        a column to group by, and ``status`` is derived again rather than read
        off the row: a file whose two columns disagree is answered by the
        table that owns the rollup.
        """
        return SolveStatus(self.termination_condition, has_primal=self.has_primal)


#: What each Python type a record column is annotated with is written as.
#: A column whose annotation is not here fails at import rather than at the
#: write, which is the moment its author is choosing the type.
_WRITTEN_AS: Mapping[type, pl.DataType | type[pl.DataType]] = {
    str: pl.String,
    float: pl.Float64,
    bool: pl.Boolean,
    int: pl.Int64,
    #: Carried with its zone rather than as a naive column claiming to be UTC,
    #: which is the reading every consumer would have to be told.
    datetime: pl.Datetime(time_zone='UTC'),
}


def _column_types(record: type[NamedTuple]) -> dict[str, pl.DataType | type[pl.DataType]]:
    """*record*'s columns as they are written, off its own annotations.

    Derived rather than restated: a column added to :class:`Record` and not
    here would silently go back to the type polars infers from a single row,
    which is the defect this schema exists to close and one a green suite
    would not show. ``X | None`` is written as ``X`` holding null.
    """
    written: dict[str, pl.DataType | type[pl.DataType]] = {}
    for name, hint in get_type_hints(record).items():
        declared = next((arg for arg in get_args(hint) if arg is not type(None)), hint)
        if declared not in _WRITTEN_AS:
            raise LpspecError(
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


class Cost(NamedTuple):
    """What a build and its solves spent, as the row an archive records beside the answer.

    :class:`Record`'s sibling — one says how the solve terminated, this says
    what reaching that cost — and the same columns whoever writes them, so
    rows written by runs that never met concatenate into one table.

    The scalars of :class:`~lpspec.relational.result.Diagnostics` and none of
    its frames: a coefficient range is a table per declaration, which does not
    fold into a row beside a count.

    **Cumulative over the model's life**, as every counter it is read off is.
    :attr:`solves` is what says how many solves the clocks cover, so a row can
    never quietly mean something other than what it holds — it reads ``1`` for
    the archive :func:`lpspec.solve` writes, that verb building the model it
    solves.
    """

    columns: int
    rows: int
    nonzeros: int
    sink_columns: int
    sink_rows: int
    solves: int
    loads: int
    #: Wall-clock seconds in each phase a build clocks, in the order they run.
    #: A phase that never ran writes zero rather than no column: the point of
    #: the row is that a directory of them is a table.
    attach: float
    build: float
    handoff: float
    solve: float
    write: float


#: :class:`Cost`'s columns as they are written, for :data:`RECORD_SCHEMA`'s
#: reason: a clock that happened to be zero would otherwise infer to the type
#: its own single row suggests.
COST_SCHEMA = _column_types(Cost)


#: The three files that sit beside the frames, named here because a result and
#: a sweep both write them and :func:`lpspec.archive.load_archive` reads back
#: whichever wrote: the record of how the solve terminated, what reaching it
#: cost, and the reasons behind whatever is deliberately not there.
RECORD_FILE = 'objective.parquet'
COST_FILE = 'diagnostics.parquet'
REASONS_FILE = 'reasons.parquet'


def consolidated(under: Path, file: str) -> pl.DataFrame | None:
    """The table *file* names under *under*, whichever shape wrote it, as one frame.

    A spill writes it one file per slice under a directory named for what the
    file holds — ``objective.parquet`` beside ``objective/`` — so the name of
    one gives the other and only the file is passed.

    What makes an archive's record one file where a spill's is one per slice.
    The spill writes them apart because the objective file's *existence* is
    how a resumed sweep knows a slice finished; an archive is finished by
    definition, and one file is what a reader globbing a warehouse of them
    needs — a directory beside a file means no single glob finds both, and the
    one that finds half finds it silently.

    Reads both shapes, so one reader serves an archive and the spill it was
    packed from. Rows stay in slice order, the files being named by position.
    ``None`` where there is nothing under either name — a directory that is
    not a sweep this package wrote, or an answer saved without the cost row
    only the model that holds the build can record.
    """
    if (single := under / file).is_file():
        frames = [pl.read_parquet(single)]
    elif (many := under / file.removesuffix('.parquet')).is_dir():
        frames = [pl.read_parquet(path) for path in sorted(many.glob('*.parquet'))]
    else:
        return None
    return pl.concat(frames)


def clear_the_answer(directory: Path) -> None:
    """Remove what a saved answer holds, leaving anything else in *directory* alone.

    A second save into one directory would otherwise leave the first answer's
    frames beside the second's: a name the new model never declared, readable
    through a reader that reports the new model's digest. Only the layout's own
    members go, so a directory the caller also keeps other files in survives.
    """
    import shutil

    for kind in (*KINDS, 'activity'):
        shutil.rmtree(directory / kind, ignore_errors=True)
    for member in (RECORD_FILE, COST_FILE, REASONS_FILE, FORMAT_FILE):
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
        LpspecError: A *kind* that names no reader.
    """
    if kind not in KINDS:
        raise LpspecError(f'kind is one of {", ".join(KINDS)}, not {kind!r}')
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
