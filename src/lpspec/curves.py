"""Filling the parameters a ``piecewise:`` expansion emitted.

A block lowers into constraints over parameters the file never wrote, so no
caller can supply them. Each says how it is filled on its own
:attr:`~math_spec.program.ParameterDeclaration.derivation`, and this fills it
from the curve's own data.

What a block *assumes* of its numbers is not here: the language states that as
an ordinary assumption and :mod:`lpspec.assumptions` checks it.

Called from :func:`~lpspec.sources.tidy_sources`, so both lanes pass through it
by entering the one door.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars as pl
from math_spec.program import FirstOf, LastOf, MaskOf

from lpspec.frames import as_frame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from math_spec.program import Program

    from lpspec.lanes import Source


def derive_curve_sources(
    program: Program, sources: dict[str, pl.LazyFrame], data: Mapping[str, Source]
) -> dict[str, pl.LazyFrame]:
    """Fill every parameter a ``piecewise:`` expansion emitted, which no caller can have.

    Three kinds, each named by the parameter's own
    :attr:`~math_spec.program.ParameterDeclaration.derivation`, so which
    parameters these are and how each is filled is read off the program rather
    than inferred from a block:

    :class:`~math_spec.program.MaskOf` is the mask a ``points: bp_x`` asked for
    by naming one of the block's own values — the curve runs as far as ``bp_x``
    does, so its length stays where it already is rather than being asked for a
    second time, and the other links are still checked against the one named.
    :class:`~math_spec.program.FirstOf` and :class:`~math_spec.program.LastOf`
    mark where each masked curve begins and ends: ``lp`` states a curve as its
    segment lines, so it needs two rows holding the linked expression inside
    the curve's *own* range, and a ``where:`` has no operator to find them
    with.

    Every frame is true-only and sparse, since a missing row reads as false in
    a ``where``. **The masks are filled first**, because the two edge flags are
    read off one. Called from :func:`tidy_sources`, before the loop that asks
    for data the caller does have.

    Returns:
        *sources* with a frame under each emitted parameter whose own source
        can be read, and nothing under the rest — attaching refuses those in the
        message that knows what the declaration wanted.
    """
    for name, declared in program.parameters.items():
        if isinstance(mask := declared.derivation, MaskOf):
            rows = _coordinates(data.get(mask.values), list(program.parameters[mask.values].dims))
            if rows is not None:
                sources[name] = rows.with_columns(value=pl.lit(True))

    for name, declared in program.parameters.items():
        match declared.derivation:
            case FirstOf(block, mask):
                edge = pl.col('_ord').min()
            case LastOf(block, mask):
                edge = pl.col('_ord').max()
            case _:
                continue
        over = program.piecewise[block].over
        dims = list(program.parameters[mask].dims)
        table = _coordinates(sources.get(mask, data.get(mask)), dims, keep_value=True)
        if table is None:
            continue
        order = _label_frame(over, sources, table).with_row_index('_ord')
        marked = table.filter(pl.col('value').cast(pl.Boolean)).join(order, on=over, how='inner')
        frame_dims = [d for d in dims if d != over]
        at = edge.over(frame_dims) if frame_dims else edge
        sources[name] = marked.filter(pl.col('_ord') == at).select([*dims, 'value'])
    return sources


def _coordinates(source: Source | None, dims: Sequence[str], keep_value: bool = False) -> pl.LazyFrame | None:
    """The coordinates *source* carries, or ``None`` where it carries all of them.

    ``None`` covers "nothing supplied", "dense by construction" and "not
    readable here" alike — a source attaching refuses is refused there, with
    the message that knows what the declaration wanted. *keep_value* keeps the value column too, which the
    ``points:`` mask is read from rather than merely counted. A parquet path is
    read here as it is at attaching, or a ``points:`` parameter supplied as one
    would derive no mask and its curve would be held to the full grid.
    """
    if source is None:
        return None
    if isinstance(source, Mapping) and len(dims) == 1:
        keys = pl.LazyFrame({dims[0]: list(source.keys())})
        return keys.with_columns(pl.Series('value', list(source.values())).implode().explode()) if keep_value else keys
    table = as_frame(source, tuple(dims))
    if table is None or not set(dims) <= set(table.collect_schema().names()):
        return None
    columns = [*dims, 'value'] if keep_value else list(dims)
    if keep_value and 'value' not in table.collect_schema().names():
        return None
    return table.select(columns)


def _label_frame(dim: str, sources: Mapping[str, pl.LazyFrame], present: pl.LazyFrame) -> pl.LazyFrame:
    """*dim*'s labels as one column, from wherever the model's index comes from.

    *sources* is what :func:`tidy_sources` returned, so a dimension with an
    index — declared by the file or passed by the caller — is already a frame
    here, and which of the two it was stopped mattering at that door.

    Falls back to the labels the curve itself carries, which is what a
    dimension with no index of its own is attached against.

    **Order is kept.** The two edge flags number these labels to say where each
    curve begins and ends, and an unordered unique hands them a permutation —
    which marks the wrong breakpoint, or does not, run to run.
    """
    source = sources.get(dim)
    if source is None:
        return present.select(dim).unique(maintain_order=True)
    return source.select(dim).unique(maintain_order=True)
