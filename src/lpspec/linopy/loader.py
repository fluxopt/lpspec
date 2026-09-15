"""The crossing into pandas and xarray: the door's frames as this lane's arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from lpspec.frames import to_pandas
from lpspec.relations import key_dims

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl
    from math_spec import program


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
    """Each map as an array over the dimensions its key names.

    A map arrives as its own ``(key dims…, relation)`` table holding rows only
    where it is defined. **The padding happens here**: an array is dense by
    construction, and linopy's ``groupby`` wants one aligned to the
    dimensions' coordinates — so a key the relation leaves out becomes a
    null, which every reader on this lane treats as "in no group". A key of
    several columns pads to their product the same way.
    """
    out: dict[str, xr.DataArray] = {}
    for name, relation in program.relations.items():
        dims = list(key_dims(relation))
        frame = to_pandas(tidy[name].collect()).set_index(dims)[name]
        index = pd.MultiIndex.from_product([master[d] for d in dims], names=dims) if len(dims) > 1 else master[dims[0]]
        padded = frame.reindex(index)
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
