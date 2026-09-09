# Preparing the data

This page turns the files an instance arrives in into the `sources` mapping
the verbs take, one frame per parameter. It assumes you have the files and
know polars or pandas. What that mapping may contain, and what attaching
refuses, is [the data contract](../reference/data.md).

## The files you start from

Real instances arrive as entity tables, with attributes side by side in the
shape a PyPSA-style CSV folder holds, and as tidy time series. The committed
instance for [dispatch](dispatch.md) is in that shape:

`examples/ports/data/dispatch/generators.csv`

```csv
generator,p_max,cost
wind,80.0,10.0
solar,0.0,25.0
gas,200.0,50.0
```

`examples/ports/data/dispatch/load.csv`

```csv
snapshot,value
0,60.0
1,120.0
2,180.0
3,90.0
```

## One frame per parameter

Split the entity table's columns out, one `select` per parameter. A frame
(a table in memory) for a parameter carries its dimension columns and a
`value` column. The time series passes through untouched:

```python
import polars as pl

generators = pl.read_csv('examples/ports/data/dispatch/generators.csv')

load = pl.read_csv('examples/ports/data/dispatch/load.csv')

sources = {
    'snapshot': load.select('snapshot').unique(maintain_order=True),
    'generator': generators.select('generator'),
    'p_max': generators.select('generator', pl.col('p_max').alias('value')),
    'cost': generators.select('generator', pl.col('cost').alias('value')),
    'load': load,
}
```

That `sources` is what every call on the model pages attaches. With data
curated as one parquet file per parameter, pass the paths instead,
`sources = {'p_max': 'p_max.parquet', ...}`, and the engine scans the files
itself.

## From linopy's shapes

Pass an indexed pandas Series as it is. Its index levels attach to dimensions
by name, so there is nothing to convert. Turn a `DataArray` into a Series with
`.to_series()`: lpspec reads tables and hands arrays back, never the other way.
The [dispatch](dispatch.md) instance, linopy-style:

```python
import pandas as pd

p_max = pd.Series({'wind': 80.0, 'solar': 0.0, 'gas': 200.0}).rename_axis('generator')
cost = pd.Series({'wind': 10.0, 'solar': 25.0, 'gas': 50.0}).rename_axis('generator')
load = pd.Series([60.0, 120.0, 180.0, 90.0]).rename_axis('snapshot')

sources = {'snapshot': load.index, 'generator': p_max.index, 'p_max': p_max, 'cost': cost, 'load': load}
```

## From PyPSA's shapes

A static attribute over one dimension is an indexed Series already, so rename
its index and pass it. A wide time series needs `stack()` back to tidy and
`reset_index()` after it, because a parameter over two dimensions arrives as a
frame carrying both as columns. Here the load is mapped from load names onto
buses on the way, the shape [transport](transport.md) attaches:

```python
load = (
    n.loads_t.p_set.rename(columns=n.loads.bus)
    .rename_axis(index='snapshot', columns='bus')
    .stack()
    .rename('value')
    .reset_index()
)

sources = {
    'snapshot': load['snapshot'].unique(),
    'bus': load['bus'].unique(),
    'generator': n.generators.index.rename('generator'),
    'p_max': n.generators['p_nom'].rename_axis('generator'),
    'cost': n.generators['marginal_cost'].rename_axis('generator'),
    'gen_bus': n.generators['bus'].rename_axis('generator').reset_index(),
    'load': load,
}
```

`gen_bus` is not a parameter but a
[lookup](https://math-spec.readthedocs.io/en/latest/reference/language/dimensions/#lookups),
so it arrives under its own name as the relation it is. Pass PyPSA's `bus`
column across as it stands rather than merging it into an index.
