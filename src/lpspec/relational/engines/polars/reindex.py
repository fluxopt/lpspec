"""Re-indexing along one dimension's own order: ``shift`` and ``sum_back``.

The two operators that move a fragment's rows *along* a dimension rather than
across dims. ``shift`` is a pointwise remap of the dim through its ordinal —
one output row per input row — and ``sum_back`` a one-to-many one, a row at
*o* contributing at every ``o + lag`` inside the window. They share the
ordinal arithmetic, the scratch columns below, and the question no other
operator has to answer: what happens at the edge, where the walk runs out of
dimension.

Both take the :class:`~lpspec.relational.engines.polars.compiler.PolarsCompiler`
and hold nothing. They read three things off it — ``data``, ``program`` and
``widen`` — and everything else here is their own, built once per operator
as an :class:`_Axis`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import polars as pl

from lpspec.relational.engines.polars.fragments import Presence, TermFragment, refuse_a_fragment_without_the_dims
from lpspec.relational.engines.polars.relations import GROUP_RANK, GROUP_SIZE, Grouping

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from math_spec import program

    from lpspec.relational.engines.polars.compiler import PolarsCompiler


#: Scratch columns. The spaces make them unrepresentable as declared names, so
#: they cannot collide with a dimension or relation the model already has.
_OFFSET = '__offset'
_LAG = '__lag'
_WIDTH = '__width'
_ORD_IN = '__ord in__'
_ORD_OUT = '__ord out__'


@dataclass(frozen=True)
class _Axis:
    """One dimension as an operator walks along it: its ranked table, and the two keyed sides of the remap.

    ``incoming`` and ``outgoing`` read the same position column, so a change to
    how a partitioned walk ranks cannot move one side without the other — the
    pair every remap and every edge computation here reads. Built once per
    operator; everything below takes it rather than rebuilding it.
    """

    dimension: str
    #: The groups the walk stays inside, or ``None`` along the whole axis.
    grouping: Grouping | None
    #: The dimension table — the grouping's, ranked in-group, when partitioned.
    table: pl.LazyFrame
    #: The walked position: within-group rank under a partition, axis-wide ``ord`` otherwise.
    position: pl.Expr
    #: What a wrap closes on: the group's size, or the axis cardinality.
    span: pl.Expr
    incoming: pl.LazyFrame
    outgoing: pl.LazyFrame
    landing: list[str]
    card: int

    @classmethod
    def of(cls, compiler: PolarsCompiler, dimension: str, partition: program.Walk | None) -> _Axis:
        """Rank *dimension* inside each group where the walk is partitioned.

        Unpartitioned, the axis-wide ``ord`` is the position and there is no
        span for a wrap to close on. Under a partition both are read per group, so
        a neighbour is decided by position within *that* group — and a
        coordinate the map places nowhere is not in this table at all and joins
        to nothing, which is what it reaches everywhere else.
        """
        card = compiler.data.cardinality[dimension]
        grouping = None if partition is None else Grouping.of(compiler.data, partition)
        if grouping is None:
            table = compiler.data.dimensions[dimension]
            position, span = pl.col('ord'), pl.lit(card, dtype=pl.Int64)
            key: tuple[str, ...] = ()
        else:
            table = grouping.table
            position, span = pl.col(GROUP_RANK), pl.col(GROUP_SIZE)
            key = grouping.key
        incoming = table.select(
            pl.col('val').alias(dimension),
            position.alias(_ORD_IN),
            *key,
            *([pl.col(GROUP_SIZE)] if grouping is not None else []),
        )
        outgoing = table.select(pl.col('val').alias(dimension), position.alias(_ORD_OUT), *key)
        return cls(dimension, grouping, table, position, span, incoming, outgoing, [_ORD_OUT, *key], card)

    @property
    def joined(self) -> tuple[str, ...]:
        """The partition's other key dimensions, which the operand carries and every keyed side reads at."""
        return self.grouping.joined if self.grouping is not None else ()

    @property
    def keys(self) -> tuple[str, ...]:
        """The columns a walked frame is keyed by: the dimension, and the partition's joined dimensions."""
        return (self.dimension, *self.joined)

    def placed(self) -> pl.LazyFrame:
        """The coordinates the partition places in some group — the rows a partitioned walk can reach."""
        assert self.grouping is not None, 'an unpartitioned walk places every label, so nothing asks which it grouped'
        return self.grouping.placed()

    def group_column(self, dim: str) -> str | None:
        """The group column carrying *dim*'s labels, where the partition groups into it."""
        return self.grouping.column_of(dim) if self.grouping is not None else None

    def remap(
        self,
        source: pl.LazyFrame,
        carried: Sequence[str],
        dims: tuple[str, ...],
        *,
        moved: pl.Expr,
        prepared: Callable[[pl.LazyFrame], pl.LazyFrame],
    ) -> pl.LazyFrame:
        """*source* with the walked dimension moved by *moved*.

        *dims* is the caller's, because a presence frame need not carry the
        fragment's: an acyclic shift's presence speaks only about the dim it
        vacated, so projecting the fragment's dims onto it asks for columns it
        never had. *prepared* splices in whatever extra join the operator needs
        — a lag table, a named offset — between the two keyed sides.
        """
        kept = [d for d in dims if d != self.dimension]
        walked = prepared(source.join(self.incoming, on=list(self.keys), how='inner').drop(self.dimension))
        return (
            walked.with_columns(moved.alias(_ORD_OUT))
            .join(self.outgoing, on=self.landing, how='inner')
            .select(*kept, self.dimension, *carried)
        )


def window_fragment(compiler: PolarsCompiler, p: TermFragment, s: program.Window, context: str) -> TermFragment:
    """A one-to-many remap of the dim through its ord.

    A row at *o* contributes at every ``o + lag`` for ``lag`` inside the
    window, so the terms land on each output position that can see them and
    the terminal ``sum(coeff)`` at assembly adds them up.

    The lag table is built to the widest window the data asks for; a named
    width then keeps only the lags that entity reaches. Every join is still
    on a dim-table key or the width's own dims, so the reach stays a relation
    and the locality class is the one :meth:`translate_fragment` has.

    Unlike a shift this vacates nothing: the window at the first position
    is short rather than empty, since it always contains that position
    itself. So an operand with no presence gains none — unless a partition
    makes one: a coordinate the map places nowhere is in no group, so the
    window reaches nothing for it, itself included, and that is the one way
    a window loses a row it would otherwise keep.

    Under ``by=`` the walk is inside the group: positions are the within-group
    rank rather than the axis-wide ``ord``, and a wrap closes on the group's
    own size, exactly as :func:`translate_fragment` walks a partitioned shift.
    """
    if s.dimension not in p.dims:
        refuse_a_fragment_without_the_dims(p, [s.dimension], context, f'sum_back(along={s.dimension!r})')
    axis = _Axis.of(compiler, s.dimension, s.partition)

    width_name = s.width if isinstance(s.width, str) else None
    if width_name is not None:
        widest = int(compiler.data.parameters[width_name].select(pl.col('value').max()).collect().item() or 0)
    else:
        assert not isinstance(s.width, str)
        widest = s.width
    lags = pl.LazyFrame({_LAG: pl.Series(range(min(widest, axis.card)), dtype=pl.Int64)})

    moved = pl.col(_ORD_IN) + pl.col(_LAG)
    if s.wrap:
        moved = moved % axis.span

    def lagged(frame: pl.LazyFrame) -> pl.LazyFrame:
        """Every reachable lag beside each row — a named width keeps only the lags its entity reaches."""
        frame = frame.join(lags, how='cross')
        if width_name is None:
            return frame
        widths, keys = _named_amount(compiler, axis, width_name, _WIDTH)
        return frame.join(widths, on=keys, how='inner').filter(pl.col(_LAG) < pl.col(_WIDTH))

    remap = partial(axis.remap, moved=moved, prepared=lagged)

    def travelled(presence: Presence) -> Presence:
        keyed_by, source = presence.keyed_by, presence.frame
        if keyed_by is not None and not set(axis.keys).issubset(keyed_by):
            source, keyed_by = compiler.widen(source, keyed_by, p.dims), None
        return Presence(remap(source, [], p.dims if keyed_by is None else keyed_by).unique(), keyed_by)

    frame = remap(p.frame, p.carried, p.dims)
    if not p.presences and axis.grouping is not None:
        return TermFragment(p.dims, frame, p.kind, presences=(Presence(axis.placed(), axis.keys),))
    return TermFragment(p.dims, frame, p.kind, presences=tuple(travelled(x) for x in p.presences))


def translate_fragment(compiler: PolarsCompiler, p: TermFragment, s: program.Translate, context: str) -> TermFragment:
    """A pointwise remap of the dim through its ord.

    A row at *o* contributes at ``(o + by) % card``.

    Both joins are on a dim-table key, so the row count is unchanged and an
    out-of-range ordinal does not join. No window function; bounded-halo
    locality. The operand's *presences* are :func:`travelled_presences` below.

    Every fill over a *constant* is written, ``0`` included: the
    arithmetic is unchanged, but the slot now has a value, so asking for
    zero stops being indistinguishable from having nothing. Over a *term*
    there is nothing to write — ``edge=0`` on a variable means the vacated
    slot contributes no term at all (the operator rules), where a zero-coefficient
    entry would be a matrix nonzero standing for a term that is not there.
    Lowering refuses every other numeric edge over a variable.
    """
    if s.dimension not in p.dims:
        refuse_a_fragment_without_the_dims(p, [s.dimension], context, f'shift(along={s.dimension!r})')
    others = [d for d in p.dims if d != s.dimension]
    axis = _Axis.of(compiler, s.dimension, s.partition)
    edge = _Edge.of(compiler, axis, s)

    named_offset = isinstance(s.offset, str)
    if named_offset:
        moved = pl.col(_ORD_IN) + pl.col(_OFFSET)
    else:
        assert not isinstance(s.offset, str)
        moved = pl.col(_ORD_IN) + s.offset
    if s.wrap:
        moved = (moved % axis.span + axis.span) % axis.span

    def offsetted(frame: pl.LazyFrame) -> pl.LazyFrame:
        """A per-entity offset is one more equi-join, on keys the frame already carries."""
        if edge.offsets is None:
            return frame
        offsets, keys = edge.offsets
        return frame.join(offsets, on=keys, how='inner')

    remap = partial(axis.remap, moved=moved, prepared=offsetted)

    def travelled_presences() -> tuple[Presence, ...]:
        """Where the variable exists after the shift, and what keys it.

        An existing presence **travels**: the coordinate set goes through
        the same map the rows did, and the inner join drops whatever the
        edge vacated. Under a fill the vacated positions go back in
        (:meth:`_vacated`) — a filled slot counts as present. A narrow
        presence is widened first when the shift moves a dim it is silent
        about, since there is no column to remap otherwise.

        An operand with **no** presence gets one: nothing was absent before
        and the acyclic edge now is, where without this the vacated slot
        would merely fail to join and the row would survive with its term
        quietly gone. It is keyed by the one dimension it speaks about. Under
        a wrap or a fill a policy speaks about a group's edge, and a
        coordinate in no group has none: it is absent under every policy.
        """
        if not p.presences:
            if s.wrap or s.fill is not None:
                return () if axis.grouping is None else (Presence(axis.placed(), axis.keys),)
            return (Presence(edge.coordinates(vacated=False), edge.keys),)
        return tuple(travelled(x) for x in p.presences)

    def travelled(presence: Presence) -> Presence:
        source, keyed_by = presence.frame, presence.keyed_by
        if keyed_by is not None and not set(edge.keys).issubset(keyed_by):
            source, keyed_by = compiler.widen(source, keyed_by, p.dims), None
        moved_presence = remap(source, [], p.dims if keyed_by is None else keyed_by)
        if s.wrap or s.fill is None:
            return Presence(moved_presence, keyed_by)
        vacated = edge.vacated_of(compiler, presence, p.dims)
        return Presence(pl.concat([moved_presence, vacated], how='vertical_relaxed').unique())

    frame = remap(p.frame, p.carried, p.dims)
    if not s.wrap and s.fill is not None and p.kind == 'const':
        frame = pl.concat([frame, edge.filled(compiler, others, s.fill)], how='vertical_relaxed')
    return replace(p, frame=frame, presences=travelled_presences())


@dataclass(frozen=True)
class _Edge:
    """The edge of an acyclic shift along an :class:`_Axis`: which coordinates it vacates, and what keys them.

    Keyed by the translated dimension, a named offset's own dims and the
    partition's joined dims, each once. How far back a row reaches decides
    which rows have nothing to reach, so under a named offset the two
    entities of one coordinate need not agree about whether it is the edge;
    and under a partition keyed on more than the dimension it walks, which
    coordinate is a group's edge depends on the rest of the key — a
    generator's own season. A grouped dimension is not among the keys: a lag
    per group varies along the translated dimension itself.
    """

    axis: _Axis
    shift: program.Translate
    #: The dims a per-entity offset varies over — empty where it is a number.
    offset_dims: tuple[str, ...]
    #: A named offset's values and the keys the axis table reads them by, or ``None`` for a number.
    offsets: tuple[pl.LazyFrame, list[str]] | None

    @classmethod
    def of(cls, compiler: PolarsCompiler, axis: _Axis, s: program.Translate) -> _Edge:
        if not isinstance(s.offset, str):
            return cls(axis, s, (), None)
        dims = compiler.program.parameter(s.offset).dims
        offset_dims = tuple(d for d in dims if axis.group_column(d) is None)
        return cls(axis, s, offset_dims, _named_amount(compiler, axis, s.offset, _OFFSET))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.axis.dimension, *self.offset_dims, *self.axis.joined)))

    def coordinates(self, *, vacated: bool) -> pl.LazyFrame:
        """The coordinates the shift vacates, or keeps, under :attr:`keys`.

        Exact complements: a fill and the presence set it implies must not
        disagree about which coordinates the edge is. Under a partition the
        edge is **each group's**, counted along the same within-group rank
        the translation itself walks: a coordinate reaches outside its own
        group exactly where it would have reached outside the axis. A
        coordinate in no group is neither — it is absent, the reading
        :meth:`_Axis.placed` gives it, so it is not in the table at all. A
        per-group offset reaches it by the group column rather than by a
        cross join, one lag standing for the whole group.
        """
        axis, s = self.axis, self.shift
        table, position, span = axis.table, axis.position, axis.span
        if self.offsets is not None:
            offsets, keys = self.offsets
            on = [key for key in keys if axis.grouping is not None and key in axis.grouping.key]
            table = table.join(offsets, on=on, how='inner') if on else table.join(offsets, how='cross')
            offset = pl.col(_OFFSET)
        else:
            assert not isinstance(s.offset, str)
            offset = pl.lit(s.offset, dtype=pl.Int64)
        source = position - offset
        reaches = (source % span + span) % span if s.wrap else source
        outside = (reaches < 0) | (reaches >= span)
        keyed = [d for d in self.keys if d != s.dimension]
        return table.filter(outside if vacated else ~outside).select(pl.col('val').alias(s.dimension), *keyed)

    def filled(self, compiler: PolarsCompiler, others: list[str], fill: float) -> pl.LazyFrame:
        """``(dims…, cval=fill)`` at every coordinate the shift vacated.

        Dense over *others*, not over the rows the operand happened to carry.

        Only a *truthy* fill gets here, ``fill=0`` needing no rows at all. A
        nonzero fill reaches a translation only over a variable-free operand,
        so this is always the const branch and never invents a ``var_label``.
        """
        edge = self.coordinates(vacated=True)
        for d in others:
            if d in self.keys:
                continue
            edge = edge.join(compiler.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return edge.with_columns(pl.lit(fill, dtype=pl.Float64).alias('cval')).select(
            *others, self.shift.dimension, 'cval'
        )

    def vacated_of(self, compiler: PolarsCompiler, presence: Presence, dims: tuple[str, ...]) -> pl.LazyFrame:
        """The edge positions ``shift`` leaves with nothing to move in, for one presence.

        Reached only under ``fill=0``, which is the whole of what ``fill`` does
        here: back in the presence set they are present-with-no-term, a zero
        contribution and a surviving row. Left out, absence propagates and the
        row drops.

        Only the ``shift`` edge qualifies; a coordinate the variable's own mask
        removed is genuinely absent and remapping already dropped it. So the
        edge is crossed with the other-dim combinations the variable actually
        has, one vacated row each. The incoming presence is widened to the other
        dims first, since a narrowly keyed one — a pullback's, an earlier
        shift's — is silent about the columns this reads.
        """
        others = [d for d in dims if d != self.shift.dimension]
        edge = self.coordinates(vacated=True)
        if not others:
            return edge
        have = presence.keys(dims)
        source = presence.frame if all(d in have for d in others) else compiler.widen(presence.frame, have, dims)
        keys = [d for d in self.keys if d in others]
        rows = source.select(*others).unique()
        return rows.join(edge, on=keys, how='inner') if keys else rows.join(edge, how='cross')


def _named_amount(compiler: PolarsCompiler, axis: _Axis, name: str, alias: str) -> tuple[pl.LazyFrame, list[str]]:
    """A named offset's or width's values, and the keys a frame reads them by.

    A **per-group** amount is declared over a dimension the partition groups
    into, and no frame carries a column of it: what travels with a coordinate is
    the relation's own value, so the amount is read under the group column and one
    equi-join lands each group its own. A coordinate the map places
    nowhere is in no partitioned table and joins to nothing, which is what it
    reaches everywhere else.
    """
    dims = compiler.program.parameter(name).dims
    keys = [axis.group_column(d) or d for d in dims]
    frame = compiler.data.parameters[name].select(
        *(pl.col(d).alias(key) for d, key in zip(dims, keys, strict=True)),
        pl.col('value').cast(pl.Int64).alias(alias),
    )
    return frame, keys
