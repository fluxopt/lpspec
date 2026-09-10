"""The crossing into pandas and xarray: the door's frames as this lane's arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from lpspec.frames import to_pandas

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl
    from math_spec import program


def dimension_coords(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
) -> tuple[dict[str, pd.Index], dict[str, dict[str, xr.DataArray]]]:
    """Every dimension's labels, and each keyed lookup's value columns as arrays over its key.

    *tidy* is :func:`~lpspec.sources.tidy_sources`' output, so every index and
    relation has been read and checked; what happens here is the conversion.

    Returns:
        The master coordinates by dimension, and by lookup name the array each
        of its value columns carries over the key's dimensions. A bare relation
        has no key to index by and is absent from the second — this lane reads
        a lookup as a function, and :func:`~lpspec.linopy.builder.build_model`
        refuses the walks that are not one.
    """
    master = {d: pd.Index(pd.unique(to_pandas(tidy[d].select(d).collect())[d]), name=d) for d in program.dimensions}
    return master, _lookup_arrays(program, tidy, master)


def _lookup_arrays(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
    master: Mapping[str, pd.Index],
) -> dict[str, dict[str, xr.DataArray]]:
    """Each keyed lookup's value columns as arrays over the dimensions of its key.

    A relation arrives holding rows only where it has them. **The padding
    happens here**: an array is dense by construction, and linopy's ``groupby``
    wants one aligned to the dimension's coordinates — so a key tuple the
    relation leaves out becomes a null, which every reader on this lane treats
    as "in no group". Under the default ``coverage: total`` there are none, the
    door having refused the gap already.
    """
    out: dict[str, dict[str, xr.DataArray]] = {}
    for name, lk in program.lookups.items():
        if not lk.key:
            continue
        rows = to_pandas(tidy[name].collect())
        keys = [lk.dim(role) for role in lk.key]
        index = (
            pd.Index(rows[lk.key[0]].to_numpy(), name=keys[0])
            if len(keys) == 1
            else pd.MultiIndex.from_arrays([rows[role].to_numpy() for role in lk.key], names=keys)
        )
        onto = {d: master[d] for d in keys}
        for role in lk.values:
            column = xr.DataArray.from_series(pd.Series(rows[role].to_numpy(), index=index))
            out.setdefault(name, {})[role] = column.reindex(onto).rename(f'{name}.{role}')
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

    Read through numpy rather than ``to_pandas()``, which wants pyarrow. A
    dims-less value keeps the dtype it arrived with, as a column does — a
    ``bool`` cast to ``0.0`` would read as *defined* under a bare ``where``.
    """
    if not dims:
        return xr.DataArray(frame['value'].to_numpy()[0])
    columns = [frame[d].to_numpy() for d in dims]
    index = (
        pd.Index(columns[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(columns, names=list(dims))
    )
    return xr.DataArray.from_series(pd.Series(frame['value'].to_numpy(), index=index))
