# Sweep a model

One call, [`solve_over`](reference/sweeps.md), solves a model once per slice
of its data and reads the answers back as one table. This page runs it once per
scenario, and then window by window with state carried between windows.

Every block on this page runs when the site is built, and what you see under it
is what it printed on this commit. A block that raises fails the build.

## 1. One solve per scenario

The dispatch model of [Run a model](guide.md), with two load levels. The `load`
table carries a `scenario` column, which the model does not declare.
[`EachCoordinate('scenario')`](reference/sweeps.md#the-axes) solves the model
once per label of that column:

```python exec="true" source="material-block" result="text" session="sweep"
import polars as pl

import specsolve as sps

GENERATORS = ['wind', 'solar', 'gas']
LOW = [60.0, 110.0, 170.0, 90.0]

sources = {
    'snapshot': range(4),
    'generator': GENERATORS,
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 40.0, 200.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [10.0, 25.0, 50.0]}),
    'load': pl.DataFrame(
        {
            'scenario': ['low'] * 4 + ['high'] * 4,
            'snapshot': [0, 1, 2, 3] * 2,
            'value': LOW + [1.5 * v for v in LOW],
        }
    ),
}

sweep = sps.solve_over('examples/dispatch.yaml', sources, sps.EachCoordinate('scenario'))

print(sweep.record.select('scenario', 'termination_condition', 'objective'))
```

`sweep.record` has one row per scenario. The readers of a
[result](reference/api.md#reading-a-result) read a sweep too, with the
scenario column in front:

```python exec="true" source="material-block" result="text" session="sweep"
print(sweep.primal('p').pivot(on='generator', index=['scenario', 'snapshot'], values='value'))
```

Gas covers what wind and solar cannot, and the high scenario needs it at three
snapshots of four.

## 2. Window by window

A rolling horizon solves consecutive windows of a time dimension, and hands the
state of a store from one window to the next. The data below covers eight hours.
Wind blows in the first two hours of every four, and the load is higher when it
does not.

```python exec="true" source="material-block" session="sweep"
HOURS = list(range(8))
WIND = [70.0, 70.0, 0.0, 0.0] * 2
LOAD = [25.0, 25.0, 85.0, 85.0] * 2

hourly = {
    'generator': ['wind', 'gas'],
    'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [0.0, 40.0]}),
    'p_max': pl.DataFrame({'hour': HOURS * 2, 'generator': ['wind'] * 8 + ['gas'] * 8, 'value': WIND + [200.0] * 8}),
    'load': pl.DataFrame({'hour': HOURS, 'value': LOAD}),
    'soc_initial': 0.0,
}
```

[`EachWindow`](reference/sweeps.md#the-axes) cuts `hour` into windows and
numbers each one from zero, into the dimension that `into` names. Before it
cuts, it asks the model how its rows couple along that dimension.
[`examples/storage.yaml`](examples/storage.md) closes its store into a cycle,
so its first snapshot reads its last. `solve_over` refuses it before a window
is solved:

```python exec="true" source="material-block" result="text" session="sweep"
try:
    sps.solve_over('examples/storage.yaml', hourly, sps.EachWindow('hour', steps=3, lookahead=3, into='snapshot'))
except sps.SpecsolveError as exc:
    print(exc)
```

[`examples/rolling/horizon.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/rolling/horizon.yaml)
is a store written for one window, over a dimension `t`. Its first row starts
from the parameter `soc_initial` instead of the last row. Each window below
keeps three hours and sees three more.
[`carry`](reference/sweeps.md#carrying-state-between-slices) copies the last
level each window keeps into the `soc_initial` of the next window. The first
window takes `soc_initial` from the sources:

```python exec="true" source="material-block" result="text" session="sweep"
rolling = sps.solve_over(
    'examples/rolling/horizon.yaml',
    hourly,
    sps.EachWindow('hour', steps=3, lookahead=3, into='t'),
    carry={'soc_initial': 'soc'},
)

print(rolling.record.select('hour_start', 'termination_condition', 'objective'))
```

Each window is keyed by the hour it starts at. `original_index=True` keeps the
hours each window owns, and reads them back over `hour` as one schedule:

```python exec="true" source="material-block" result="text" session="sweep"
print(rolling.primal('soc', original_index=True))
```

The store charges in the windy hours and discharges in the calm ones. The
window that starts at hour 3 opens from the level the first window left at
hour 2.

## 3. Myopic pathways

A myopic pathway is the same call over investment periods:
`EachCoordinate('year')` with `carry={'existing': 'total'}` hands each period
the fleet the one before it left.
[`examples/myopic/run.py`](https://github.com/fluxopt/specsolve/blob/main/examples/myopic/run.py)
runs one.

## Where next

| | |
|---|---|
| [Sweeps and rolling horizons](reference/sweeps.md) | every axis, `carry`, `keep` and `spill_to=`, and how a sweep is read |
| [Running a sweep in parallel](howto/parallel.md) | one slice per worker |
| [Archiving a solve](howto/archiving.md) | the model, its data and every slice kept as one archive |
