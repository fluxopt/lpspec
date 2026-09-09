"""Eager evaluation of the language's built-in operators, on xarray and linopy.

Each entry point takes an operand that is already a value — an ``xr.DataArray``
for a parameter, a linopy ``Variable`` or ``LinearExpression`` for anything
carrying one — and returns the same. Nothing here reads the schema or the
model: ``builder.py`` evaluates the operands and the keywords, and calls in
with a lookup as the plan walks it (:class:`~math_spec.program.Walk`) beside
the relation the door bound (:class:`~lpspec.linopy.loader.BoundLookup`).
Each entry point comes first and its own machinery after.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from lpspec.linopy import absence

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from math_spec.program import Walk

    from lpspec.linopy.loader import BoundLookup


def operator_sum(array: Any, over: str) -> Any:
    """Sum *array* over dimension *over*, which lowering has checked it carries.

    The one sum linopy cannot take — a term carrying a dimension the data left
    empty — is built directly (:func:`_empty_sum`).
    """
    if not isinstance(array, xr.DataArray) and any(not array.sizes[dim] for dim in array.coord_dims if dim != over):
        return _empty_sum(array, over)
    return array.sum(over)


def _empty_sum(array: Any, over: str) -> Any:
    """*over* reduced while a dimension beside it is empty — the sum linopy refuses to take.

    linopy's own ``sum`` stacks the summed dim into its term axis, and once any
    remaining dimension is zero-sized the reshape cannot infer the term count
    (``cannot reshape array of size 0``, PyPSA/linopy#906). Every coordinate of
    the result is empty, so it is built as the constant zero over the
    coordinates linopy's sum would keep.
    """
    from linopy.expressions import LinearExpression

    kept = [dim for dim in array.coord_dims if dim != over]
    zeros = xr.DataArray(
        np.zeros([array.sizes[dim] for dim in kept]),
        coords={dim: array.indexes[dim] for dim in kept},
        dims=kept,
    )
    return LinearExpression.from_constant(array.model, zeros)


def operator_grouped_sum(
    array: Any, walks: Sequence[tuple[Walk, BoundLookup]], *, labels: Mapping[str, pd.Index]
) -> Any:
    """Sum *array* through lookups, consuming the dims the walks consume and producing the ones they produce.

    YAML: ``sum(p, by=gen_bus)``, ``sum(p, by=[gen_bus, gen_tech])`` or
    ``sum(p, by=zone_of, from=generator, into=zone)``. Each walk is a lookup
    and how it is walked, with the relation the door bound. Three shapes,
    decided by what the walk's key licenses:

    * **a function read** — the key lies inside the produced and joined
      columns, so the consumed dims are read at it: a pullback
      (:func:`_read`), which is what a sum with one term per coordinate is.
    * **a grouping** — the key is exactly the consumed and joined columns,
      so each member has one group: linopy's own ``groupby``
      (:func:`_grouped_sum`).
    * **a relation** — anything else, a bare relation among them: the
      incidence of the relation masks the operand and the consumed dims are
      summed out (:func:`_related_sum`).

    *labels* holds each dimension's declared index, and the result carries
    exactly those coordinates.
    """
    kinds = {_kind(walk) for walk, _ in walks}
    carried = tuple(d for d in _dims(array) if d not in _consumed(walks))
    if kinds == {'read'}:
        return _read(array, walks)
    if kinds == {'group'} and not any(d in carried for walk, _ in walks for d in walk.produced_dims):
        return _grouped_sum(array, walks, labels=labels)
    return _related_sum(array, walks)


def operator_at(array: Any, walks: Sequence[tuple[Walk, BoundLookup]]) -> Any:
    """Read *array* through lookups — the adjoint of a group.

    YAML: ``at(on, by=component)``. Grouping sums *along* a lookup, this
    indexes *through* it: every walk reads value columns at a key the operand
    fixes, the language having refused any other, so the operand carries the
    consumed dims and the result the produced ones.
    """
    return _read(array, walks)


def _kind(walk: Walk) -> Literal['read', 'group', 'relation']:
    """Which shape a walk takes on this lane — see :func:`operator_grouped_sum`."""
    if walk.is_function_read:
        return 'read'
    if walk.key and set(walk.key) == {*walk.consumed, *walk.joined}:
        return 'group'
    return 'relation'


def _read(array: Any, walks: Sequence[tuple[Walk, BoundLookup]]) -> Any:
    """*array* at each key tuple's value: xarray's vectorised selection is the pullback exactly.

    Every consumed dim is indexed by the value column over it, an array over
    the key's dimensions, so the result carries those — and one the operand
    already carries is read pointwise, at the row's own coordinate. A
    self-map's consumed dim is a key dim too, and comes back labelled by the
    key rather than by what was read at it.

    A key the relation leaves out reads nothing and its row is absent, the
    reading ``sum`` gives a member in no group. It cannot be selected, so the
    indexer reads any label there and the result is masked at those
    positions, which then hold the operand's own **absence** rather than a
    zero: absence propagates and takes the row with it, where a zero would
    leave a row asserting ``x <= 0`` at a coordinate the model said nothing
    about.
    """
    indexers = {walk.dim(role): bound.value(role) for walk, bound in walks for role in walk.consumed}
    present = reduce(operator.and_, (m.notnull() for m in indexers.values()))
    keyed = {dim: m[dim].values for dim, m in indexers.items() if dim in m.dims}
    if bool(present.all()):
        return array.sel(indexers).assign_coords(keyed)
    filled = {dim: m.fillna(_any_label(array, dim, m)) for dim, m in indexers.items()}
    return array.sel(filled).assign_coords(keyed).where(present)


def _any_label(array: Any, target: str, mapping: Any) -> object:
    """A label of *target* to read at a position the map sends nowhere, before the result is masked there.

    The first the map does point at, and the first of *array*'s own where it
    points nowhere at all — the operand carries the dim, and a map with no
    row against an empty target reads nothing, which the mask says.
    """
    pointed = mapping.to_numpy()[mapping.notnull().to_numpy()]
    return pointed[0] if len(pointed) else array.indexes[target][0]


def _grouped_sum(array: Any, walks: Sequence[tuple[Walk, BoundLookup]], *, labels: Mapping[str, pd.Index]) -> Any:
    """Each produced dim as a coordinate along the key, then one ``groupby`` over the produced and joined dims.

    A null lookup value says the key belongs to no group, so its terms
    contribute nowhere — and with several lookups a member missing *any* of
    them belongs to no group at all. Those members are masked absent and
    gathered under a label no dimension has (:func:`_no_label`), since linopy
    refuses a null key, and that group goes with the reindex onto the
    declared labels.

    The result is reindexed onto *labels*: a groupby yields only the labels
    some member actually points at, in xarray's sort order, and linopy v1
    aligns on membership *and* order. Lookup values are validated against
    their dimension's labels when they are bound, so this only ever adds a
    label, never drops a term. A self-map's produced dim is the consumed one,
    so its coordinate rides under a scratch name until the consumed dim is
    grouped away.
    """
    mappings = {walk.dim(role): bound.value(role) for walk, bound in walks for role in walk.produced}
    present = reduce(operator.and_, (m.notnull() for m in mappings.values()))
    if not bool(present.all()):
        array = array.where(present)
        mappings = {d: m.fillna(_no_label(labels[d])) for d, m in mappings.items()}
    consumed = _consumed(walks)
    names = {d: f'__produced {d}__' if d in consumed else d for d in mappings}
    attached = array.assign_coords({names[d]: (m.dims, m.values) for d, m in mappings.items()})
    joined = dict.fromkeys(d for walk, _ in walks for d in walk.joined_dims)
    summed = attached.groupby([*names.values(), *joined]).sum()
    renamed = {name: d for d, name in names.items() if name != d}
    return _reindexed(summed.rename(renamed) if renamed else summed, into=tuple(mappings), labels=labels)


def _no_label(labels: pd.Index) -> object:
    """A group label no member of *labels* is: what an unmapped member is gathered under before the reindex drops it."""
    if labels.dtype.kind in 'iufb':
        return np.inf
    if labels.dtype.kind == 'M':
        return pd.Timestamp.max
    return '\x00 unmapped'


def _related_sum(array: Any, walks: Sequence[tuple[Walk, BoundLookup]]) -> Any:
    """*array* masked by the relations' incidence and summed over the consumed dims.

    The reading every walk has, taken literally: a term contributes at each
    produced tuple the relation holds it beside, and nowhere else. The
    incidence is dense, so this is the shape the other two exist to avoid
    paying for where a key licenses them. A produced dim the operand carries
    already is joined on by the mask's alignment — a masked sum — and one
    over the consumed dimension itself, a self-map's, is carried under a
    scratch name until the consumed dim is summed out.
    """
    consumed = _consumed(walks)
    renamed: dict[str, str] = {}
    incidence = None
    for walk, bound in walks:
        named = {role: walk.dim(role) for role in (*walk.consumed, *walk.joined)}
        for role in walk.produced:
            dim = walk.dim(role)
            named[role] = f'__produced {dim}__' if dim in consumed else dim
            if named[role] != dim:
                renamed[named[role]] = dim
        held = bound.incidence(named)
        incidence = held if incidence is None else incidence & held
    summed = array.where(incidence).sum(list(consumed))
    return summed.rename(renamed) if renamed else summed


def _consumed(walks: Sequence[tuple[Walk, BoundLookup]]) -> tuple[str, ...]:
    """The dims a node's walks consume, each once — one set for a grouping, one per walk for a pullback through several."""
    return tuple(dict.fromkeys(d for w, _ in walks for d in w.consumed_dims))


def _dims(array: Any) -> tuple[str, ...]:
    """The dims of a value on this lane, an expression naming its coordinate dims where an array names its dims."""
    return tuple(str(d) for d in (array.coord_dims if not isinstance(array, xr.DataArray) else array.dims))


@dataclass(frozen=True)
class _Edge:
    """One ``edge=`` policy: cyclic, a fill, or absent (``fill=None``, no wrap)."""

    wrap: bool
    fill: float | None


@dataclass(frozen=True)
class Partition:
    """A lookup as ``shift``, ``sum_back`` and ``position`` walk it: the group at each coordinate, and what a group reads.

    Attributes:
        groups: Each coordinate's group — the lookup's value tuple, or its one
            value — over the walked dimension and the dims the lookup joins
            on, null where the coordinate is in no group.
        values: Each value column over the same key, by the dimension it is
            over: what an amount declared over a group's dimension is read
            through (:func:`_per_group`).
    """

    groups: xr.DataArray
    values: Mapping[str, xr.DataArray]


def operator_shift(
    array: Any, *, over: str, offset: Any, wrap: bool, fill: float | None, by: Partition | None = None
) -> Any:
    """Translate *array* along one dimension — the value at *t - offset*.

    YAML: ``shift(soc, over=snapshot, offset=1)``. *wrap* is cyclic and vacates
    nothing; *fill* is what the vacated positions contribute; neither leaves
    them **absent**, which propagates and drops the row — what linopy v1's own
    ``shift`` already answers.

    A DataArray shift always fills, absence not being representable in data, so
    lowering refuses a bare shift over a variable-free operand and that branch
    is only reached under a numeric fill.

    *offset* arrives as an array where the model named a parameter — an offset
    that differs per entity, which is a gather rather than a shift
    (:func:`_gather_by_offset`).
    """
    edge = _Edge(wrap, fill)
    if by is not None:
        groups = _grouped(over, np.asarray(array.indexes[over]), by.groups)
        return _gather_in_groups(array, over, _per_group(offset, by), groups=groups, edge=edge)
    if isinstance(offset, xr.DataArray) and offset.ndim:
        return _gather_by_offset(array, over, offset, edge=edge)
    amount: dict[Hashable, int] = {over: int(offset)}
    if wrap:
        return array.roll(amount, roll_coords=False) if isinstance(array, xr.DataArray) else array.roll(amount)
    if isinstance(array, xr.DataArray):
        return array.shift(amount, fill_value=fill if fill is not None else np.nan)
    shifted = array.shift(amount)
    return shifted if fill is None else absence.vacated(shifted, array, over, _off_the_axis(array, over, offset), fill)


def operator_sum_back(array: Any, *, over: str, within: Any, wrap: bool, by: Partition | None = None) -> Any:
    """Sum *array* over a trailing window along one dimension.

    YAML: ``sum_back(started, over=snapshot, within=min_up)``. The result at
    *t* is the sum from *t - within + 1* through *t*, so a width of 1 is the
    operand itself and *wrap* lets the window reach around the axis.

    Written as a sum of scalar gathers, one per position of the widest window
    the data asks for — a bound read from data, sound because it decides how
    many *terms* are added rather than what the plan does.

    A position the window cannot reach contributes a **zero**, never an
    absence: absence propagates, so an unreachable lag would annihilate the
    whole row, and a window at the first position is short, not empty. The
    window that reaches **nothing** is the exception — a zero there would
    build a row about constants alone — so the fill is paired with the
    positions any lag actually reached, and a window that reached none keeps
    no row. A width of zero everywhere is that window at every position.

    ``by=`` stops the window at each group's edge. A width declared over the
    group's own dim is read through the lookup first (:func:`_per_group`).
    """
    within = _per_group(within, by) if by is not None else within
    asked = _widest(within)
    widest = max(1, min(asked, int(array.sizes[over])))
    probe = _Edge(wrap=wrap, fill=None)
    groups = _grouped(over, np.asarray(array.indexes[over]), by.groups) if by is not None else None
    terms: list[Any] = []
    reached: list[Any] = []
    for lag in range(widest):
        lagged = (
            _gather_in_groups(array, over, lag, groups=groups, edge=probe)
            if groups is not None
            else _gather_by_offset(array, over, lag, edge=probe)
        )
        live, term = ~lagged.isnull(), absence.filled(lagged, 0.0)
        if isinstance(within, xr.DataArray):
            live, term = live & (within > lag), term * (within > lag).astype(float)
        terms.append(term)
        reached.append(live)
    return _merged(terms).where(reduce(operator.or_, reached))


def _widest(within: Any) -> int:
    """The widest window the data asks for, which bounds how many lags are gathered.

    A coordinate the lookup maps nowhere asks for no window: a width read
    through one carries the operand's own absence there (:func:`_per_group`),
    so the maximum skips the holes rather than coming back absent itself, and
    a width no coordinate carries at all is a window of nothing. The lag loop
    takes it from there — ``within > lag`` is false at a hole, so the
    coordinate contributes no term and keeps no row.
    """
    if not isinstance(within, xr.DataArray):
        return int(within)
    widths = np.asarray(within, dtype=float)
    return 0 if bool(np.isnan(widths).all()) else int(np.nanmax(widths))


def _merged(terms: list[Any]) -> Any:
    """The sum of *terms*, which all share one set of coordinates.

    Merged in one step rather than added one at a time: a running sum
    re-concatenates the term axis once per lag it has already absorbed.
    """
    if isinstance(terms[0], xr.DataArray):
        return reduce(operator.add, terms)
    from linopy import merge

    return merge(terms)


def _reindexed(summed: Any, *, into: tuple[str, ...], labels: Mapping[str, pd.Index]) -> Any:
    """*summed* over exactly the declared labels, empty groups filled with an empty sum.

    Reindexing onto a label no member reached creates an **absent** slot, and
    the unstacked multi-key groupby invents the combinations no member lands on
    the same way. Neither is an absence in this language: a group with no
    members holds the empty sum, which is 0. ``fillna`` reaches the constant of
    a ``LinearExpression`` and the value of the ``DataArray`` a grouped
    parameter is, so one call serves both, and it comes after the reindex
    because the invented combinations sit at *present* labels.
    """
    return summed.reindex({d: labels[d] for d in into}).fillna(0.0)


def _gather_by_offset(array: Any, over: str, offset: Any, *, edge: _Edge) -> Any:
    """Translate *array* along *over* by an offset that differs per entity.

    A scalar shift is one call; a per-entity one is a **gather**: every output
    position reads a source position of its own, so the index is an array over
    the offset's dims and *over* rather than a number.

    Selection is by *label* rather than by ordinal, because that is what linopy
    passes through to its own labels — which also keeps a non-integer axis (a
    datetime snapshot) working for free.

    Out-of-range positions are clipped so the gather stays on the axis, then
    emptied again by ``where``, so an edge means the same thing it does for a
    scalar shift: absent by default, and :func:`~lpspec.linopy.absence.vacated`
    fills it where the model asked. Under ``wrap`` nothing is out of range and
    the modulo is the whole of it.
    """
    card = int(array.sizes[over])
    labels = np.asarray(array.indexes[over])
    ordinal = xr.DataArray(np.arange(card), coords={over: labels}, dims=[over])
    source = (ordinal - offset).astype(int)

    def gathered(ordinals: Any) -> Any:
        """Pick each position's source, then put the original labels back — the indexer carries no coordinate."""
        picked = array.sel({over: _labelled(labels, ordinals, over)})
        return picked.assign_coords({over: labels})

    if edge.wrap:
        return gathered(source % card)
    inside = ((source >= 0) & (source < card)).assign_coords({over: labels})
    moved = gathered(source.clip(0, card - 1)).where(inside)
    return moved if edge.fill is None else absence.vacated(moved, array, over, ~inside, edge.fill)


def _per_group(offset: Any, partition: Partition) -> Any:
    """*offset* at every coordinate, where it is declared over a group's own dim.

    One lag per group — a lead time that differs by period, which a
    ``(period, timestep)`` model writes as an offset over ``period`` because
    ``period`` is not the axis it walks. The group *is* the lookup's value, so
    the lag a coordinate moves by is its group's, read through that value
    column: the pullback ``at()`` already is, absent where the coordinate is
    in no group. Every other offset is returned as it came.

    The group's label is dropped rather than ridden along: a pullback leaves
    what it read through as a coordinate, and a constraint built from one is
    then reported as carrying a dimension the language says a shift does not
    have.
    """
    if not isinstance(offset, xr.DataArray):
        return offset
    for dim, values in partition.values.items():
        if dim not in offset.dims:
            continue
        present = values.notnull()
        read = offset.sel({dim: values.fillna(_any_label(offset, dim, values))})
        offset = read.where(present).drop_vars(dim)
    return offset


@dataclass(frozen=True)
class _Groups:
    """How one lookup partitions an axis, as the arrays every in-group gather reads.

    Computed once per operand and shared across a window's lags: the partition
    does not depend on the lag, and the roster is the one Python loop over the
    axis in this lane. A lookup joined on other dims partitions the axis once
    per coordinate of them, so the per-coordinate arrays carry those dims
    beside the axis and a group is one value at one such coordinate.

    Attributes:
        labels: The axis's own labels, in order.
        grouped: Whether each coordinate belongs to any group.
        belongs: The group ordinal of each coordinate, 0 where it has none.
        within: Each coordinate's position inside its group.
        size: Each coordinate's group size, 1 where it has none.
        roster: The label ordinal at each ``(group, position)``.
        names: Each group's key, by group ordinal — the lookup's value, with
            the joined coordinate beside it where the lookup has one.
        counts: Each group's member count, by group ordinal.
    """

    labels: np.ndarray
    grouped: xr.DataArray
    belongs: xr.DataArray
    within: xr.DataArray
    size: xr.DataArray
    roster: np.ndarray
    names: tuple[object, ...]
    counts: tuple[int, ...]


def _grouped(over: str, labels: np.ndarray, groups: Any) -> _Groups:
    """The partition *groups* makes of the axis *over* carrying *labels*.

    A coordinate the lookup sends nowhere belongs to no group: its ``within``
    is 0, its ``size`` 1 and its ``grouped`` False, and every gather reads the
    last of those first. A lookup joined on other dims partitions the axis
    once per coordinate of them, so the per-coordinate arrays carry those
    dims beside the axis and a group is one value at one such coordinate.
    """
    joined = [str(d) for d in groups.dims if d != over]
    aligned = groups.sel({over: labels}).transpose(over, *joined)
    keys = np.asarray(aligned.values, dtype=object).reshape(len(labels), -1)
    axes = [aligned.indexes[d].tolist() for d in joined]
    coordinates = [tuple(axis[i] for axis, i in zip(axes, at, strict=True)) for at in np.ndindex(*aligned.shape[1:])]

    peers: dict[object, list[tuple[int, int]]] = {}
    within = np.zeros(keys.shape, dtype=int)
    grouped = np.zeros(keys.shape, dtype=bool)
    for j, coordinate in enumerate(coordinates):
        for k, key in enumerate(keys[:, j]):
            if absence.unmapped(key):
                continue
            grouped[k, j] = True
            beside = peers.setdefault(key if not joined else (key, *coordinate), [])
            within[k, j] = len(beside)
            beside.append((k, j))

    order = {key: g for g, key in enumerate(peers)}
    widest = max((len(beside) for beside in peers.values()), default=1)
    roster = np.zeros((max(len(peers), 1), widest), dtype=int)
    for key, beside in peers.items():
        roster[order[key], : len(beside)] = [k for k, _ in beside]
    belongs = np.zeros(keys.shape, dtype=int)
    span = np.ones(keys.shape, dtype=int)
    for key, beside in peers.items():
        for k, j in beside:
            belongs[k, j], span[k, j] = order[key], len(beside)

    def on_axis(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values.reshape(aligned.shape), coords=aligned.coords, dims=aligned.dims)

    return _Groups(
        labels,
        on_axis(grouped),
        on_axis(belongs),
        on_axis(within),
        on_axis(span),
        roster,
        tuple(peers),
        tuple(len(beside) for beside in peers.values()),
    )


def _gather_in_groups(array: Any, over: str, offset: Any, *, groups: _Groups, edge: _Edge) -> Any:
    """Translate *array* inside each group *groups* makes, not along the axis.

    The neighbour of a coordinate is the one *offset* back among the coordinates
    sharing its group, so the gather is by a source ordinal computed per group:
    a position past the group's start is vacated where the axis edge would
    vacate, and under *wrap* it comes round to that group's own last.

    *offset* is a number, or an array where the model named a parameter — per
    entity, per group (:func:`_per_group`), or both — so the source ordinal is
    computed from the within-group position rather than looked up member by
    member, and carries the offset's own dims.

    A coordinate in no group reaches nothing — the null reading a partial
    lookup gets everywhere else. That is not the same as reaching *off* a
    group's edge, which is what a policy speaks for, so the two are tracked
    apart and only the second is filled.
    """
    reached = groups.within - offset
    if edge.wrap:
        reached = reached % groups.size
    inside = groups.grouped & (reached >= 0) & (reached < groups.size)

    def peer(group: Any, position: Any) -> Any:
        """The label ordinal sitting at *position* of the group at *group*."""
        return groups.roster[group, position]

    source = xr.apply_ufunc(peer, groups.belongs, reached.where(inside, 0).astype(int))
    labels = groups.labels
    gathered = array.sel({over: _labelled(labels, source, over)}).assign_coords({over: labels}).where(inside)
    if edge.fill is None:
        return gathered
    return absence.vacated(gathered, array, over, groups.grouped & ~inside, edge.fill)


def _off_the_axis(array: Any, over: str, offset: float) -> Any:
    """Which positions along *over* a scalar shift of *offset* leaves vacated.

    The source a position reads, off both ends, so one expression covers a
    shift in either direction — the same verdict :func:`_gather_by_offset`
    reaches with ``inside`` negated.
    """
    labels = np.asarray(array.indexes[over])
    source = xr.DataArray(np.arange(len(labels)), coords={over: labels}, dims=[over]) - int(offset)
    return (source < 0) | (source >= len(labels))


def _labelled(labels: Any, ordinals: Any, over: str) -> Any:
    """*ordinals* as the coordinate labels they stand for, keeping their dims.

    Carries no coordinates of its own: an indexer that keeps the axis's own
    coordinate asserts the values it holds *are* that axis, and after a gather
    they are not — which xarray reports as a size conflict rather than a
    mislabelling.
    """
    return xr.DataArray(labels[ordinals.transpose(*ordinals.dims).values], dims=ordinals.dims)
