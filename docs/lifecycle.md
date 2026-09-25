# Fix, relax, remove

Three verbs a linopy reader reaches for first, spelled as the loops of
[the previous page](interactive.md). None is a method here, which is
[hard rule 5](https://github.com/fluxopt/specsolve/blob/main/docs/about/architecture.md#hard-rules).

| linopy | here | loop |
|---|---|---|
| `x.fix(v)` | both bounds read a parameter; write the same number into both | **1**, data, no rebuild |
| `x.relax()` | `domain:` in the declaration | 3 |
| `remove_constraints` | drop the key from the spec | 3 |

The model is `examples/dispatch.yaml`, as before. Every block runs when the
site is built, and a block that raises fails the build.

```python exec="true" source="material-block" session="lifecycle"
import polars as pl
from math_spec import to_spec

import specsolve as sps

MODEL = 'examples/dispatch.yaml'
GENERATORS = ['wind', 'solar', 'gas']

sources = {
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 40.0, 200.0]}),
    'snapshot': pl.DataFrame({'snapshot': range(6)}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 60.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [90.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
}
```

## Fix

A fix is two bound parameters, each total over the variable's coordinates: a
frame holding only the pinned rows is a load error. `p_max` keeps its job as
the `where` mask, so a pin does not renumber the labels.

```python exec="true" source="material-block" result="text" session="lifecycle"
pinnable = to_spec(MODEL).to_dict()
pinnable['parameters']['p_lo'] = {'dims': ['snapshot', 'generator']}
pinnable['parameters']['p_hi'] = {'dims': ['snapshot', 'generator']}
pinnable['variables']['p']['bounds'] = {'lower': 'p_lo', 'upper': 'p_hi'}

grid = pl.DataFrame({'snapshot': range(6)}).join(pl.DataFrame({'generator': GENERATORS}), how='cross')
p_lo = grid.with_columns(value=pl.lit(0.0))
p_hi = grid.join(sources['p_max'], on='generator')

pinned = sps.build(pinnable, sources | {'p_lo': p_lo, 'p_hi': p_hi})
unpinned = pinned.solve().objective

hold = pl.when(pl.col('generator') == 'gas').then(60.0).otherwise(pl.col('value'))
held = pinned.update({'p_lo': p_lo.with_columns(value=hold), 'p_hi': p_hi.with_columns(value=hold)}).solve().objective

pinning = pinned.diagnostics()
print(f'{pinning.loads} loads over {pinning.solves} solves — a pin moves bounds, not labels')
print(pl.DataFrame({'gas': ['free to dispatch', 'held at 60'], 'objective': [unpinned, held]}))
```

One load for both answers. A `p == p_pin` constraint would do the same at a
row per pinned variable, and put the information in a shadow price instead of
a reduced cost. Write a row for a combination, `sum(p, over=generator) ==
target`, which is not a bound.

## Relax

Integrality is what the column is, so changing it is loop 3: patch `domain:`
and build again. An integer variable makes duals undefined, and asking for
one says so.

```python exec="true" source="material-block" result="text" session="lifecycle"
integral = to_spec(MODEL).to_dict()
integral['variables']['p']['domain'] = 'integer'

milp = sps.solve(integral, sources)
print(f'integer objective {milp.objective:,.1f}, has_primal {milp.has_primal}')

try:
    milp.dual('power_balance')
except sps.SpecsolveError as exc:
    print(exc)

relaxed = sps.solve(MODEL, sources)  # the same file, continuous as declared
print(relaxed.dual('power_balance'))
```

## Remove

A constraint family is a key in a mapping, so removing it is `pop`. Below,
the ramp limit from the previous page, added and taken away.

```python exec="true" source="material-block" result="text" session="lifecycle"
ramped = to_spec(MODEL).to_dict()
ramped['parameters']['ramp_max'] = {'dims': ['generator']}
ramped['constraints']['ramp_up'] = {
    'dims': ['snapshot', 'generator'],
    'expression': 'p - shift(p, along=snapshot, offset=1) <= ramp_max',
}
data = sources | {'ramp_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 100.0, 20.0]})}

with_ramp = sps.solve(ramped, data).objective
ramped['constraints'].pop('ramp_up')
without_ramp = sps.solve(ramped, data).objective

print(pl.DataFrame({'model': ['with ramp_up', 'ramp_up removed'], 'objective': [with_ramp, without_ramp]}))
```

The data-shaped alternative is a `where` on the constraint: the declaration
stays and builds no rows where the mask is false. That is loop 1 in spelling
and loop 2 in cost, since a mask that changes membership renumbers labels.
`diagnostics().loads` says which one you got, and `omissions` counts the rows
not built.

## What is still missing

An IIS (irreducible infeasible subsystem) on an infeasible model.
`model.row(name, **coordinate)` gives one row's terms, comparison and
right-hand side without a solve, and
[debugging a wrong answer](howto/debug.md) is the recipe. The whole
relationship is
[relationship to linopy](https://github.com/fluxopt/specsolve/blob/main/docs/about/linopy.md).

## Where next

[Sweep a model](sweep.md) solves one model once per scenario, and then
window by window.
