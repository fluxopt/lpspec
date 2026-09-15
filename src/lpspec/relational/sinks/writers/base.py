"""What a writer **is**: three decisions every format makes the same way.

How a section is appended, how a float is rendered, and how an index is. One
home each.

The family base; it renders no format of its own.
"""

from __future__ import annotations

from typing import IO

import polars as pl

__all__ = ['chunk_key', 'digits', 'number', 'sink']


def sink(frame: pl.LazyFrame, f: IO[bytes]) -> None:
    """Append a one-column frame to *f*, one raw line per row.

    A CSV writer with the CSV switched off, straight into the handle the caller
    holds: polars writes through its buffer, so an ``f.write()`` between two
    sinks lands between them and no concatenation pass rereads the file.

    ``maintain_order`` is stated rather than inherited: the parameter is
    documented as unstable.
    """
    frame.sink_csv(f, include_header=False, quote_style='never', maintain_order=True)


def chunk_key(axis: pl.Expr, lo: int, slots: int, within: pl.Expr) -> pl.Expr:
    """The one sort key of a chunked section: ``slots`` consecutive keys per *axis* value.

    Chunk-relative, so the product is bounded by a chunk's height rather than
    the model's.
    """
    return ((axis - lo) * slots + within).alias('key')


def number(value: pl.Expr) -> pl.Expr:
    """A float as text.

    Polars' cast is shortest-*round-trip* rather than merely shortest, so the
    double a solver reads back is the double the engine computed.
    """
    return value.cast(pl.String)


def digits(value: pl.Expr) -> pl.Expr:
    """An index as text — never in scientific notation, whatever its size."""
    return value.cast(pl.Int64).cast(pl.String)
