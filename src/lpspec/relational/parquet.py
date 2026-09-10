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
import os
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

from lpspec.errors import LpspecError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: The three kinds of frame a solve answers with, named after the reader each
#: comes back through, and what each is a frame of.
KINDS = ('primal', 'dual', 'expression')
LABELS = {'primal': 'variable', 'dual': 'constraint', 'expression': 'named expression'}


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
    objective: float
    #: Whether the solve produced values, which the condition alone does not
    #: say: a run stopped at a limit before any incumbent is ``ok`` with
    #: nothing to read.
    has_primal: bool
    #: :func:`digest_of` the spec this answered, or ``None`` where the solve
    #: was run off a lowered program and there was no document to digest.
    #: Carried so that answers written apart can be *told* to be comparable:
    #: one distinct value across a concatenated table means one spec.
    spec_digest: str | None


#: The two files that sit beside the frames, named here because a result and a
#: sweep both write them and :func:`lpspec.artifact.load_artifact` reads back
#: whichever wrote: the record of how the solve terminated, and the reasons
#: behind whatever is deliberately not there.
RECORD_FILE = 'objective.parquet'
REASONS_FILE = 'reasons.parquet'


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
    os.replace(part, path)
