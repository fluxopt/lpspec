"""Answers on disk as parquet: the layout a result and a sweep both write, and the writer that lands a file whole.

Under a directory, ``<kind>/<name>`` for each of the three kinds a solve
answers with — the primals, the duals, the named expressions — so a
constraint carrying a variable's name never collides with it. A result
writes one file under each name; a sweep one per slice, and reads them back
as one.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl

from lpspec.errors import LpspecError

if TYPE_CHECKING:
    from pathlib import Path

#: The three kinds of frame a solve answers with, named after the reader each
#: comes back through, and what each is a frame of.
KINDS = ('primal', 'dual', 'expression')
LABELS = {'primal': 'variable', 'dual': 'constraint', 'expression': 'named expression'}


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
