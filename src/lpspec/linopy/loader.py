"""The crossing into pandas and xarray: the door's frames as this lane's arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from lpspec.frames import to_pandas

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl
    from math_spec import program
    from math_spec.program import LookupDeclaration, Walk


def dimension_coords(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
) -> tuple[dict[str, pd.Index], dict[str, BoundLookup]]:
    """Every dimension's labels, and each declared lookup as this lane reads it.

    *tidy* is :func:`~lpspec.sources.tidy_sources`' output, so every index and
    relation has been read and checked; what happens here is the conversion.

    Returns:
        The master coordinates by dimension, and each lookup by name.
    """
    master = {d: pd.Index(pd.unique(to_pandas(tidy[d].select(d).collect())[d]), name=d) for d in program.dimensions}
    lookups = {name: BoundLookup(lk, to_pandas(tidy[name].collect()), master) for name, lk in program.lookups.items()}
    return master, lookups


@dataclass(frozen=True)
class BoundLookup:
    """One lookup's relation as this lane reads it: the rows, and the arrays read off them.

    ``rows`` is the relation as the door checked it, one column per declared
    column. A keyed lookup's value columns are read as arrays over the key's
    dimensions (:meth:`value`), and any lookup as a boolean incidence over the
    dimensions of the columns asked for (:meth:`incidence`). **The padding
    happens here**: an array is dense by construction, and linopy's
    ``groupby`` and xarray's vectorised selection both want one aligned to
    the dimensions' coordinates — so a key the relation leaves out becomes a
    null, which every reader on this lane treats as "in no group", and a
    tuple it leaves out becomes false.
    """

    declaration: LookupDeclaration
    rows: pd.DataFrame
    master: Mapping[str, pd.Index]
    _values: dict[str, xr.DataArray] = field(default_factory=dict, compare=False, repr=False)

    def value(self, role: str) -> xr.DataArray:
        """Value column *role* at every key tuple, over the key's dimensions, null where the key maps nowhere."""
        if role not in self._values:
            lk = self.declaration
            assert role in lk.values, f"'{role}' is a key column of '{lk.name}', which is read at rather than read"
            self._values[role] = self._over_key(self.rows[role].to_numpy())
        return self._values[role]

    def groups(self, walk: Walk) -> xr.DataArray:
        """The group each key tuple is in under *walk*: the tuple of the value columns it produces, or the one value."""
        values = list(walk.produced)
        if len(values) == 1:
            return self.value(values[0])
        keys = list(zip(*(self.rows[v].to_numpy() for v in values), strict=True))
        return self._over_key(np.array([*keys, None], dtype=object)[:-1])

    def _over_key(self, values: np.ndarray) -> xr.DataArray:
        """*values*, one per row, as an array over the key's dimensions, reindexed onto every coordinate."""
        lk = self.declaration
        dims = [lk.dim(k) for k in lk.key]
        assert len(set(dims)) == len(dims), f"'{lk.name}' is keyed by two columns over one dimension"
        columns = [self.rows[k].to_numpy() for k in lk.key]
        index = pd.Index(columns[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(columns, names=dims)
        array = xr.DataArray.from_series(pd.Series(values, index=index))
        return array.reindex({d: self.master[d] for d in dims})

    def incidence(self, named: Mapping[str, str]) -> xr.DataArray:
        """Whether the relation holds a row at each tuple, over one dimension per column in *named*.

        *named* maps each column read to the dimension name the array carries
        it under — its own dimension, or a scratch name where two columns are
        over one dimension.
        """
        lk = self.declaration
        columns = [self.rows[role].to_numpy() for role in named]
        index = pd.MultiIndex.from_arrays(columns, names=list(named.values()))
        held = xr.DataArray.from_series(pd.Series(True, index=index))
        onto = {name: self.master[lk.dim(role)].rename(name) for role, name in named.items()}
        return held.reindex(onto).fillna(value=False).astype(bool)


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
