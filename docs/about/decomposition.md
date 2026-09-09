# Decomposition, as evidence

This page shows that the language can express a Benders decomposition and reach
the right answer, for anyone decomposing a model in lpspec or asking for a
driver that does it.

**lpspec ships no decomposition driver.** Whether it should is
[#596](https://github.com/fluxopt/lpspec/issues/596). Every block below is
validated against
[`examples/benders/`](https://github.com/fluxopt/lpspec/blob/main/examples/benders/run.py).

## Why anyone wants it

A model too large to solve is usually made smaller: representative days instead
of a year, forty nodes instead of three hundred, one weather year instead of
many. Each answers a *different question*, and none bounds how wrong it is for
the one you asked. Decomposition answers the question you asked, with a
**gap**: stop at 1% and you are within 1%.

## The problem, whole

Choose generator capacity, then dispatch it, with investment and operation
decided together. The capacity choice is small and the dispatch is large, which
is what makes it worth decomposing.

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  invest: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
variables:
  cap:
    foreach: [generator]
    bounds: {lower: 0, upper: 100}
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0}
constraints:
  capacity:
    foreach: [snapshot, generator]
    expression: p <= cap * avail
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) >= load
objective:
  sense: minimize
  expression: sum(cap * invest) + sum(p * cost)
```

## The split is one substitution

The subproblem is the same dispatch at a capacity someone else chose: **`cap`
stops being a variable and becomes a parameter.**

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
  cap_hat: {dims: [generator]}          # was `cap`, a variable
variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0}
constraints:
  capacity:
    foreach: [snapshot, generator]
    expression: p <= cap_hat * avail
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) >= load
objective:
  sense: minimize
  expression: sum(p * cost)
```

Three things are *not* here: no `invest`, no `cap` bounds, no investment term.
The subproblem does not know it is part of anything.

## The master, where a cut is data

The master keeps the capacity decision. It stands in `theta` for the dispatch
it can no longer see: one variable holding what operating that capacity will
cost. Cuts teach the master what `theta` is:

```yaml
dimensions:
  generator: {dtype: str}
  cut: {dtype: int}
  fcut: {dtype: int}
parameters:
  invest: {dims: [generator]}
  cut_const: {dims: [cut]}
  cut_slope: {dims: [cut, generator]}
  fcut_const: {dims: [fcut]}
  fcut_slope: {dims: [fcut, generator]}
variables:
  cap:
    foreach: [generator]
    bounds: {lower: 0, upper: 100}
  theta:
    foreach: []
    bounds: {lower: 0}
constraints:
  optimality_cut:
    foreach: [cut]
    expression: theta >= cut_const + sum(cut_slope * cap, over=generator)
  feasibility_cut:
    foreach: [fcut]
    expression: sum(fcut_slope * cap, over=generator) <= fcut_const
objective:
  sense: minimize
  expression: sum(cap * invest) + theta
```

**`cut` and `fcut` take their members from data**
([the data contract](../reference/data.md)). An iteration appends rows to their
parameter tables and generates no YAML, so the model a reviewer reads is the
model that runs.

`theta` is a scalar variable, `foreach: []`, and its `lower: 0` is the only
thing keeping the first master bounded before any cut exists.

## Reading a cut out of an answer

A cut is the value and the slope of the subproblem at the capacity that was
tried. The slope is the shadow price of the capacity constraint, weighted by
availability and summed over snapshots. `sources` is the data the model
attaches ([glossary](../reference/glossary.md#how-it-runs)):

```python
import lpspec as lps
import polars as pl

with lps.solve('examples/benders/sub.yaml', sources) as sub:
    slope = (
        sub.dual('capacity')
        .join(avail, on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
```

The whole interface with the [engine](../reference/glossary.md#how-it-runs) is
`dual` and a join against the model's own `avail` table. Appending the cut is
two `pl.concat` calls onto the parameter tables the master already declares.

## When the subproblem is infeasible

Below some capacity there is no dispatch at all, and the subproblem is
infeasible. lpspec hands back **no Farkas ray**. An infeasible solve has no
readable status, so `dual()` raises rather than returning a vector of zeros
that looks like an answer.

The cut comes instead from a fourth model, the subproblem with a slack and an
objective that asks *how far from dispatchable* this capacity is:

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
  cap_hat: {dims: [generator]}
variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0}
  short:
    foreach: [snapshot]
    bounds: {lower: 0}
constraints:
  capacity:
    foreach: [snapshot, generator]
    expression: p <= cap_hat * avail
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) + short >= load
objective:
  sense: minimize
  expression: sum(short)
```

Its optimum is zero exactly when the subproblem is feasible, and its capacity
duals are the slope the feasibility cut needs. It is a separate file because a
model declares one objective.

## The loop

```python
sub_model, feasibility_model, master_model = (lps.check(path) for path in paths)

for step in range(25):
    with lps.solve(sub_model, {**dispatch, 'cap_hat': capacity}) as sub:
        dispatchable = sub.has_primal
        if dispatchable:
            slope, here_value = slope_at(sub, capacity)
            upper = min(upper, spent(capacity) + sub.objective)
            appended(tables, 'cut', sub.objective - here_value, slope)

    if not dispatchable:
        with lps.solve(feasibility_model, {**dispatch, 'cap_hat': capacity}) as short:
            slope, here_value = slope_at(short, capacity)
            appended(tables, 'fcut', here_value - short.objective, slope)

    with lps.solve(master_model, {**master_sources, **coordinates}) as master:
        lower = master.objective
        capacity = master.primal('cap').select('generator', 'value')

    if upper < float('inf') and upper - lower <= 1e-6 * abs(upper):
        break
```

Twenty lines, three `lps.solve` calls, and a growing pair of tables. **A reader
could write this**, which is the observation that matters most for
[#596](https://github.com/fluxopt/lpspec/issues/596).

The models are loaded once above the loop, because a cut is a row in a
parameter table rather than an edit to a file. `lps.solve` accepts what
`lps.check` returns, a lowered program
([glossary](../reference/glossary.md#the-chain)), anywhere it accepts a path.
So parse, validation and lowering are paid once for the run instead of three
times an iteration. Any driver over a fixed model does the same, and
`solve_over` already does.

## Running it

```bash
pixi run python examples/benders/run.py
```

```text
the whole problem, in one plan: 9600.00

  step 0  feasibility  lower  2025.00   upper none yet
  step 1  feasibility  lower  2625.00   upper none yet
  step 2  feasibility  lower  2850.00   upper none yet
  step 3  optimality   lower  9600.00   upper 9600.00

decomposed: 9600.00 in 4 steps
monolithic: 9600.00
difference: 0.0e+00
cuts: 1 optimality, 3 feasibility
```

Three capacities are excluded as undispatchable before one proves feasible, and
the first optimality cut then closes the gap exactly.

## The check is the algorithm's own

lpspec can always build the monolith from the same sources, so the example
solves both and prints the difference: `0.0e+00` above, asserted in
`tests/test_benders_example.py`. That is the two-lane differential test aimed
at an algorithm instead of an engine. It is always available because the
undecomposed form is another file over the same data.

## What is deliberately absent

Missing is everything that makes a decomposition survive a real model: cut
management as the master grows, stabilisation, multi-cut, tolerances that hold
when duals are degenerate, and an answer for when convergence does not happen.
That is the surface [#596](https://github.com/fluxopt/lpspec/issues/596) asks
whether to own. This page settles only that the *language* is not the obstacle.
