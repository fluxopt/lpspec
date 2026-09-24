# How the benchmarks were taken

This page is the method behind the numbers on the
[benchmark page](benchmarks-scaling.html), for anyone deciding how far to
trust a cell there. That page holds the results: five libraries over three
models, with the numbers under each chart.

**Every published number is the median of a measurement's rounds, and every
band is the first to the third quartile of the same rounds.** Every measurement
gets the same nine rounds, pinned rather than calibrated by duration, so no
cell is a best-of-nine beside a neighbour's best-of-forty. The median beats
the fastest round because a cell whose nine rounds all ran slow has no clean
round to pick. It beats the mean because one slow round moves a mean and
leaves a median where it was. On this run four published cells have a mean
above 1.10x their median, the worst at 1.21x: `fleet/s` and `fleet/xs` on
highspy-matrix, `transport/xs` and `dispatch/xs` on gurobipy-matrix. All four
are the two smallest rungs of a matrix arm, where the whole measurement is
milliseconds and one scheduler hiccup is the gap. No specsolve, linopy or pyomo
cell on this run reaches 1.05x.

The median flipped one cell against specsolve on this run: `dispatch/s` on the
`gurobi` [sink](../reference/glossary.md#how-it-runs), against linopy. On
`gurobi` our build alternates between a fast and a slow state round after
round, and no other library's build does
([#1288](https://github.com/fluxopt/specsolve/issues/1288)).

## How to reproduce it

```bash
uv run --locked bench/reproduce.py
```

`bench/reproduce.py.lock` freezes every version, git commits included: two of
the five libraries install from git and one of those is a branch. `--locked`
refuses to start if the resolution has drifted.

Everything the tables are drawn from is in
[`bench/results`](https://github.com/fluxopt/specsolve/blob/main/bench/results):
one file per sink and case. Each carries the machine, the versions, the commit
and every round of every measurement. A case the box could not finish leaves
no file behind. `pixi run table` prints the directory as one long CSV and
commits nothing; the JSON stays the archive because it keeps the rounds.
`pixi run refresh` re-takes the numbers and writes the tables into their fences
and the chart's data literal into its own.

## First model against every model after it

<!-- bench:marginal -->

### Marginal cost per model

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +13.3 ms on specsolve and +1.9 ms on linopy and +3.4 ms on pyomo and +1.8 ms on gurobipy-loop and +8.2 ms on gurobipy-matrix and +1.0 ms on highspy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | specsolve: first | specsolve: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady | highspy-matrix: first | highspy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dispatch | 10k | 26.4 ms | **23.7 ms** | 28.8 ms | 27.1 ms | 31.2 ms | 28.0 ms | 24.3 ms | 23.3 ms | 23.6 ms | 14.9 ms | 5.5 ms | 4.9 ms |
| fleet | 12k | 115.1 ms | **92.0 ms** | 174.9 ms | 172.7 ms | 35.9 ms | 32.2 ms | 66.7 ms | 64.9 ms | 24.5 ms | 23.1 ms | 6.6 ms | 5.9 ms |
| dispatch | 100k | 33.9 ms | **29.9 ms** | 29.4 ms | 27.9 ms | 227.8 ms | 225.7 ms | 204.6 ms | 210.1 ms | 74.4 ms | 81.8 ms | 7.1 ms | 7.2 ms |
| fleet | 120k | 133.1 ms | **100.9 ms** | 180.8 ms | 172.2 ms | 831.7 ms | 832.8 ms | 839.4 ms | 833.0 ms | 135.2 ms | 140.3 ms | 11.2 ms | 11.0 ms |
| dispatch | 1M | 76.2 ms | **73.4 ms** | 37.0 ms | 35.7 ms | 4619.7 ms | 4511.0 ms | 2487.3 ms | 2519.4 ms | 705.8 ms | 698.0 ms | 35.9 ms | 27.9 ms |
| fleet | 1.2M | 211.8 ms | **184.5 ms** | 198.7 ms | 197.9 ms | 5920.6 ms | 5884.0 ms | 7732.6 ms | 7707.5 ms | 1415.7 ms | 1399.6 ms | 107.0 ms | 105.6 ms |
| dispatch | 10M | 465.0 ms | **442.6 ms** | 162.5 ms | 157.0 ms | — | — | 26636.5 ms | 26538.7 ms | 7901.1 ms | 7807.2 ms | 805.5 ms | 790.6 ms |
| fleet | 12M | 1320.4 ms | **1320.3 ms** | 471.7 ms | 467.1 ms | — | — | — | — | 14693.1 ms | 14670.9 ms | 1424.9 ms | 1422.5 ms |

<!-- bench:/marginal -->

## The same size, reached by widening

This table renders from `highs` rungs of `storage` and `transport`, and no run
has published it: both cases are killed on that sink before they write a file,
for the reason [below](#not-measured-yet). The committed results hold
`transport` through `gurobi` only, on [the chart page](benchmarks-scaling.html),
and `storage` through neither sink. `bench.report` drops a fragment it cannot
render rather than blanking the fence, so the fence stays empty until
`pixi run ladder` brings those rungs back.

<!-- bench:sweeps -->
<!-- bench:/sweeps -->

## Not measured yet

Listed so that a claim with no table under it is visible as one.

- **Sparsity, which is the engine's whole premise.** Every published cell is
  100% live: the `where` in `dispatch` removes nothing, and `transport` and
  `fleet` carry no mask at all. A dense coordinate product is the shape an
  array engine is built for. The published ladder therefore compares
  the two lanes only where the relational one has the least to win. `nodal` and
  `sector` are the sparse cases, at 25% and 8.3% of their product. `nodal` has
  a linopy formulation now, so `pixi run density` sweeps it at four densities;
  no published run has taken it.
- **Solve time.** Every number stops at the hand-off. The simplex is the
  solver's work whoever filled the model.
- **The LP-file round trip.** The tables price writing a file, never reading
  one back.
- **Sizes past `l`.** `xl` and `2xl` exist in the harness and no run
  publishes them.
- **`storage` is absent from every table, and `transport` from the `highs`
  ones.** The memory budget projects the next rung off a `w10` or `w100` cell
  that took under a gigabyte. Where the projection comes in under 16 GB the run
  starts a rung that does not fit, and `bench/memory-watchdog.sh` kills the
  case to save the box. Three cases died that way on this run:
  `transport/w100` on `highs` at 23.7 GB, `storage/w1000` on `highs` at
  25.3 GB, `storage/w1000` on `gurobi` at 26.3 GB. A killed case writes no
  file, so each loses every rung it had already measured, its rows in the
  marginal table above included
  ([#1498](https://github.com/fluxopt/specsolve/issues/1498)). `storage` survived
  on `gurobi` until this run only because the projection stopped
  `gurobipy-matrix` one rung earlier: 0.858 GB at `storage/w100` projects to
  17.2 GB, just over the budget. The last `highs` numbers past `w10` are in
  [#1285](https://github.com/fluxopt/specsolve/pull/1285), taken on a machine that
  could hold them. There specsolve took 0.11 s and 0.59 GB at `transport/w100`,
  against linopy's 53.53 s and 14.26 GB.
- **Anything about expressiveness.** Four models say nothing about a fifth.

## Method

Each measurement runs in a process of its own. Peak memory is `ru_maxrss`
rather than a tracker. Import is excluded from the timing and teardown is
included. A run refuses to start on a machine that is already working. The
rest is in
[`bench/README.md`](https://github.com/fluxopt/specsolve/blob/main/bench/README.md):
every flag, every default switched off and what it costs.

**Peak carries an allocator cost that only the polars arms pay.** polars
ships its own jemalloc settings, so a peak measured through it holds pages
freed and not yet returned. An arm on the system allocator never enters
jemalloc. Our peak moves 12–27% with the decay clock on and off, where
linopy's does not move at three digits
([#896](https://github.com/fluxopt/specsolve/issues/896)). It runs against us and
is left in.

**memray never times anything.** Its tracker slows an allocation-heavy
engine several-fold and overcounts reserved arenas. Peak RSS is the metric;
memray is for attribution.
