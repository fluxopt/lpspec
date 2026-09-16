"""The crossing into pandas and xarray: the door's frames as this lane's arrays.

The language's relation is a table over any number of dimensions, keyed by
any number of its columns. What this lane builds of it is the **single-valued
map**: a keyed relation with one value column, so the map is one dense array
over its key's dimensions and a walk is an ``assign_coords`` and a ``groupby``,
or a vectorised ``sel``.
:func:`refuse_relations_the_lane_does_not_build` turns the rest away at the
lane's door, naming the relational lane, which builds every shape the language
admits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from math_spec import program as _program

from lpspec.errors import LaneError
from lpspec.frames import to_pandas

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import polars as pl
    from math_spec import program


def refuse_relations_the_lane_does_not_build(program: program.Program) -> None:
    """Refuse a relation shape this lane does not build, before any data is read.

    Raises:
        LaneError: A bare relation, one whose key determines several columns,
            or a partition grouped by a map keyed on more than the dimension
            it walks.
    """
    for name, relation in program.relations.items():
        if not relation.values:
            raise LaneError(
                _relation_shape_message(f"relation '{name}' is bare — every one of its columns is in its key")
            )
        if len(relation.values) > 1:
            raise LaneError(
                _relation_shape_message(
                    f"relation '{name}' has {len(relation.values)} columns its key does not determine "
                    f'({list(relation.values)})'
                )
            )
    for node in _partitioning(program):
        assert node.partition is not None, '_partitioning yields only the nodes that carry one'
        if node.partition.joined:
            raise LaneError(
                _relation_shape_message(
                    f"a partition by '{node.partition.name}' groups by a map keyed by {list(node.partition.key)}, "
                    f'and this lane groups a shift, sum_back or position by a map keyed by the dimension it '
                    f'walks alone'
                )
            )


def _relation_shape_message(what: str) -> str:
    return (
        f'the linopy lane builds the single-valued map — a keyed relation with one value column — '
        f'and {what}. The language accepts it and the relational '
        f'lane builds it, so this is a limit of the lane rather than of the spec. Build it with '
        f'lps.build()/lps.solve() instead.'
    )


def _partitioning(
    program: program.Program,
) -> Iterator[program.Translate | program.Window | program.DimensionPositionNode]:
    """Every node that groups by a relation: an operator in an expression, and a ``position(by=)`` in a mask."""
    bodies = (*program.expressions, *(e.expression for e in program.named_expressions.values()))
    for node in _program.walk(*bodies):
        if isinstance(node, (_program.Translate, _program.Window)) and node.partition is not None:
            yield node
        if isinstance(node, _program.Cases):
            for region in node.regions:
                yield from _positions(region.when)
    for declaration in (*program.variables.values(), *program.constraints.values()):
        yield from _positions(declaration.where)


def _positions(mask: program.Mask | None) -> Iterator[program.DimensionPositionNode]:
    """The grouped positions one mask tests, or nothing."""
    if mask is None:
        return
    for atom in mask.atoms:
        if isinstance(atom, _program.DimensionPositionNode) and atom.partition is not None:
            yield atom


def dimension_coords(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
) -> tuple[dict[str, pd.Index], dict[str, xr.DataArray]]:
    """Every dimension's labels, and each declared relation as an array over the dimensions it is keyed by.

    *tidy* is :func:`~lpspec.sources.tidy_sources`' output, so every index and
    map has been read and checked; what happens here is the conversion.

    Returns:
        The master coordinates by dimension, and one array per declared
        relation, by name.
    """
    master = {d: pd.Index(pd.unique(to_pandas(tidy[d].select(d).collect())[d]), name=d) for d in program.dimensions}
    return master, _relation_arrays(program, tidy, master)


def _relation_arrays(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
    master: Mapping[str, pd.Index],
) -> dict[str, xr.DataArray]:
    """Each map as an array over the dimensions its key names, holding its one value column.

    A map arrives as the table it declares, holding rows only where it is
    defined. **The padding happens here**: an array is dense by construction,
    and linopy's ``groupby`` wants one aligned to the dimensions' coordinates
    — so a key the relation leaves out becomes a null, which every reader on
    this lane treats as "in no group". A key of several columns pads to their
    product the same way.
    """
    out: dict[str, xr.DataArray] = {}
    for name, relation in program.relations.items():
        dims = [relation.dim(role) for role in relation.key]
        (value,) = relation.values
        frame = to_pandas(tidy[name].collect())
        keys = [frame[role].to_numpy() for role in relation.key]
        keyed = pd.Index(keys[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(keys, names=dims)
        index = pd.MultiIndex.from_product([master[d] for d in dims], names=dims) if len(dims) > 1 else master[dims[0]]
        padded = pd.Series(frame[value].to_numpy(), index=keyed).reindex(index)
        shape = tuple(len(master[d]) for d in dims)
        coords = {d: master[d] for d in dims}
        out[name] = xr.DataArray(padded.to_numpy().reshape(shape), dims=dims, coords=coords, name=name)
    return out


def load_parameters(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
    master_coords: Mapping[str, pd.Index],
) -> xr.Dataset:
    """Every declared parameter as the dataset this lane builds against.

    *tidy* is :func:`~lpspec.sources.tidy_sources`' output — one checked
    ``(dims…, value)`` frame per parameter — so what is left is the crossing
    into xarray.

    Returns:
        One DataArray per parameter, reindexed onto the master coordinates.
    """
    arrays: dict[str, xr.DataArray] = {}
    for pname, pdef in program.parameters.items():
        arr = _from_tidy(tidy[pname].collect(), pdef.dims)
        if pdef.dims:
            onto = {d: master_coords[d] for d in pdef.dims}
            arr = arr.reindex(onto, fill_value=False) if arr.dtype == bool else arr.reindex(onto)
        arrays[pname] = arr
    return xr.Dataset(arrays)


def _from_tidy(frame: pl.DataFrame, dims: Sequence[str]) -> xr.DataArray:
    """A tidy ``(dims…, value)`` frame as an array.

    A dims-less value keeps the dtype it arrived with, as a column does — a
    ``bool`` cast to ``0.0`` would read as *defined* under a bare ``where``.
    """
    if not dims:
        return xr.DataArray(frame['value'].to_numpy()[0])
    columns = [frame[d].to_numpy() for d in dims]
    index = (
        pd.Index(columns[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(columns, names=list(dims))
    )
    return xr.DataArray.from_series(pd.Series(frame['value'].to_numpy(), index=index))
