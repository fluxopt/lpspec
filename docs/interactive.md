# Change a model

The dispatch model from [Run a model](guide.md), changed three ways, cheapest
first. Every block is a function of a **spec** (the YAML, as a `dict`) and its
**sources**, so blocks re-run in any order mean the same thing.

1. **New numbers**: `update`, and the solver keeps the model it has loaded.
2. **More rows**: the same math over a longer axis.
3. **New math**: patch the `dict` and re-run.

A fourth section reads a built row back, for when the answer is wrong and the
file looks right.

Every block on this page runs when the site is built, and what you see under it
is what it printed on this commit. A block that raises fails the build.

```python exec="true" source="material-block" session="loops"
import polars as pl
from mathspec import to_markdown, to_spec

import specsolve as sps

SPEC = 'examples/dispatch.yaml'
GENERATORS = ['wind', 'solar', 'gas']

sources = {
    'snapshot': pl.DataFrame({'snapshot': range(6)}),
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 40.0, 200.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 60.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [90.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
}

print(to_markdown(SPEC))
```

## 1. New numbers

Build once, then `update` per run. New costs go onto the model HiGHS holds,
and the matrix is never handed over twice.

```python exec="true" source="material-block" result="text" session="loops"
model = sps.build(SPEC, sources)

rows = []
for gas_cost in (40.0, 60.0, 90.0):
    costs = pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, gas_cost]})
    rows.append({'gas_cost': gas_cost, 'objective': model.update({'cost': costs}).solve().objective})

sweep = pl.DataFrame(rows)
reused = model.diagnostics()

print(f'{reused.loads} model loaded, {reused.solves} solves')
print(sweep)
```

`loads` is 1 against `solves` of 3: three answers, one load.
`model.update(x).solve()` gives what `sps.solve(SPEC, sources | x)` gives,
always. The next block solves the last cost from scratch to show it.

```python exec="true" source="material-block" result="text" session="loops"
fresh = sps.solve(SPEC, sources | {'cost': costs}).objective
updated = sweep.filter(pl.col('gas_cost') == 90.0).item(0, 'objective')

print(f'updated {updated:,.1f} — fresh build {fresh:,.1f}')
```

## 2. More rows

A longer horizon is a longer table plus the index to match.

```python exec="true" source="material-block" result="text" session="loops"
horizon = pl.DataFrame(
    {
        'snapshot': range(12),
        'value': [90.0, 120.0, 150.0, 180.0, 140.0, 100.0, 95.0, 130.0, 160.0, 190.0, 150.0, 110.0],
    }
)

index = pl.DataFrame({'snapshot': range(12)})
schedule = model.update({'snapshot': index, 'load': horizon}).solve().primal('p')
grown = model.diagnostics()

print(f'{schedule.height} rows of p now, and {grown.loads} loads over {grown.solves} solves')
print(schedule.head())
```

`loads` is 2 now: new coordinates renumber the columns, so this model was
loaded from scratch. The answer is the same either way.
`schedule.pivot(on='generator', index='snapshot', values='value')` is the
wide view.

## 3. New math

`to_dict()` is the spec as data, and every verb takes a `dict`. An edit is a
key, and `to_spec` validates it again. Below, a ramp limit on gas, which
needs a parameter as well as a constraint.

```python exec="true" source="material-block" result="text" session="loops"
spec = to_spec(SPEC).to_dict()
spec['parameters']['ramp_max'] = {'dims': ['generator']}
spec['constraints']['ramp_up'] = {
    'dims': ['snapshot', 'generator'],
    'expression': 'p - shift(p, along=snapshot, offset=1) <= ramp_max',
}

ramp_max = pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 100.0, 20.0]})
base = sps.solve(SPEC, sources).objective
ramped = sps.solve(spec, sources | {'ramp_max': ramp_max}).objective

print(pl.DataFrame({'model': ['dispatch', 'dispatch + ramp limit'], 'objective': [base, ramped]}))
```

The limit binds: gas starts climbing early, and free wind is curtailed to
make room. The math re-renders from the patched spec:

```python exec="true" source="material-block" session="loops"
print(to_markdown(spec, legend=False, numbered=False))
```

An edit the language refuses is refused before any data is attached:

```python exec="true" source="material-block" result="text" session="loops"
typo = {
    **spec,
    'constraints': {
        **spec['constraints'],
        'peak': {'dims': ['snapshot'], 'expression': 'sum(p, over=generators) <= load'},
    },
}

try:
    sps.check(typo)
except sps.LanguageError as exc:
    print(exc)
```

## 4. When the answer is wrong

`p` is declared `where: "p_max > 0"`, so a capacity of zero does not park a
generator at zero. It deletes the column and every term that referenced it.
Below, gas is retired with one number:

```python exec="true" source="material-block" result="text" session="loops"
retired = sources | {'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 40.0, 0.0]})}
short = sps.build(SPEC, retired)
answer = short.solve()

print(f'{answer.status} / {answer.termination_condition}')
try:
    answer.primal('p')
except sps.NoSolutionError as exc:
    print(exc)
```

Infeasible, while the file still reads `sum(p, over=generator) == load` over
all three generators. `row` gives the row the build produced, with no solve:

```python exec="true" source="material-block" result="text" session="loops"
fleet = sps.build(SPEC, sources)

print(fleet.row('power_balance', snapshot=3))
print(short.row('power_balance', snapshot=3))
print(f'{fleet.diagnostics().columns} columns became {short.diagnostics().columns}')
```

`p[3, gas]` is missing from the second row: six columns went with the
`where`, and `power_balance` asks two generators for a load of 180. The line
is [linopy's shape](reference/api.md#specsolve.relational.result.ConstraintRow). No solver output names
this fault. Where a mask takes every row of a declaration,
`diagnostics().omissions` counts them.

## What leaves the session

The spec, as a file you can diff and commit:

```python exec="true" source="material-block" result="yaml" session="loops"
print(to_spec(spec).to_yaml())
```

## Where next

[Fix, relax, remove](lifecycle.md) spells `fix`, `relax` and "remove that
constraint" as these three loops.
[`keep=`](reference/api.md#specsolve.Model.solve) is the one choice made for
you here: `'solver'`.
