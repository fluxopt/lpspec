# How the benchmarks were taken

This page is the method behind the numbers on the
[benchmark page](benchmarks-scaling.html), for anyone deciding how far to
trust a cell there. That page holds the results: five libraries over four
models, with the numbers under each chart.

**Every published number is the median of a measurement's rounds, and every
band is the first to the third quartile of the same rounds.** Every measurement
gets the same nine rounds, pinned rather than calibrated by duration, so no
cell is a best-of-nine beside a neighbour's best-of-forty. The median rather
than the fastest round, because a cell whose nine rounds all ran slow has no
clean round to pick. The median rather than the mean, because one slow round
moves a mean and leaves a median where it was. No published cell on this run
has a mean above 1.10x its median.

The median flipped two cells against lpspec on this run: `dispatch/xs` on the
`gurobi` [sink](../reference/glossary.md#how-it-runs) and `dispatch/s` on
`highs`, both against linopy. On `gurobi` our build alternates between a fast
and a slow state round after round, and no other library's build does
([#1288](https://github.com/fluxopt/lpspec/issues/1288)).

## How to reproduce it

```bash
uv run --locked bench/reproduce.py
```

`bench/reproduce.py.lock` freezes every version, git commits included: two of
the five libraries install from git and one of those is a branch. `--locked`
refuses to start if the resolution has drifted.

Everything the tables are drawn from is in
[`bench/results`](https://github.com/fluxopt/lpspec/blob/main/bench/results):
one file per sink and case. Each carries the machine, the versions, the commit
and every round of every measurement. A case the box could not finish leaves
no file behind. `pixi run table` prints the directory as one long CSV and
commits nothing; the JSON stays the archive because it keeps the rounds.
`pixi run refresh` re-takes the numbers and writes the tables into their fences
and the chart's data literal into its own.

## First model against every model after it

<!-- bench:marginal -->

### Marginal cost per model

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +29.6 ms on lpspec and +2.0 ms on linopy and +6.8 ms on pyomo and +11.8 ms on gurobipy-loop and +14.5 ms on gurobipy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | lpspec: first | lpspec: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dispatch | 10k | 28.4 ms | **20.9 ms** | 27.9 ms | 27.0 ms | 34.0 ms | 32.6 ms | 38.2 ms | 26.4 ms | 26.0 ms | 15.2 ms |
| fleet | 12k | 106.8 ms | **99.2 ms** | 183.5 ms | 181.3 ms | 38.4 ms | 37.2 ms | 84.2 ms | 72.1 ms | 37.4 ms | 23.1 ms |
| dispatch | 100k | 40.5 ms | **33.4 ms** | 29.9 ms | 28.0 ms | 282.1 ms | 277.8 ms | 206.3 ms | 209.5 ms | 94.5 ms | 79.8 ms |
| fleet | 120k | 117.5 ms | **86.8 ms** | 189.1 ms | 186.7 ms | 727.9 ms | 718.6 ms | 834.6 ms | 820.5 ms | 154.2 ms | 152.8 ms |
| dispatch | 1M | 124.9 ms | **91.7 ms** | 38.3 ms | 36.6 ms | 4589.6 ms | 4575.7 ms | 2587.0 ms | 2625.7 ms | 852.6 ms | 830.4 ms |
| fleet | 1.2M | 230.0 ms | **201.5 ms** | 213.3 ms | 209.6 ms | 5792.9 ms | 5769.9 ms | 8058.4 ms | 8016.8 ms | 1606.9 ms | 1607.5 ms |
| dispatch | 10M | 677.7 ms | **602.3 ms** | 158.0 ms | 157.1 ms | — | — | 27268.8 ms | 27273.8 ms | 8781.2 ms | 8672.3 ms |
| fleet | 12M | 1561.1 ms | **1471.2 ms** | 495.6 ms | 490.5 ms | — | — | — | — | 16624.4 ms | 16586.2 ms |

<!-- bench:/marginal -->

## The same size, reached by widening

This table is drawn through the `highs` sink, and no run has published it. The
two cases that carry width rungs are killed there before they write a file, for
the reason [below](#not-measured-yet). [The chart page](benchmarks-scaling.html)
carries the same ladder through `gurobi`.

<!-- bench:sweeps -->
<!-- bench:/sweeps -->

**No file in `bench/results` carries the table above.** It renders from `highs`
rungs of `storage` and `transport`, and the committed results hold those two
cases through the `gurobi` sink only. `bench.report` drops a fragment it cannot
render rather than blanking the fence, so the numbers stand here from a run
whose file is gone. Treat them as unsourced until `pixi run ladder` refills
them.

## Not measured yet

Listed so that a claim with no table under it is visible as one.

- **Sparsity, which is the engine's whole premise.** Every published cell is
  100% live: the `where` in `dispatch` removes nothing, and `transport`,
  `storage` and `fleet` carry no mask at all. A dense coordinate product is the
  shape an array engine is built for. The published ladder therefore compares
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
- **`transport` and `storage` are absent on the `highs` sink, and the width
  ladder with them.** The published run takes `w100` and `w1000`, and the
  `highs` job also measures the repeated build. That budget projects the next
  rung off a `w10` cell that took under a gigabyte, so the run starts
  `transport/w100` and `storage/w1000`, where `bench/memory-watchdog.sh` kills
  the case to save the box. A killed case writes no file, so both models lose
  every rung they had already measured, their rows in the marginal table above
  included ([#1498](https://github.com/fluxopt/lpspec/issues/1498)). The
  `gurobi` job keeps both models, because it leaves the repeated build to
  `highs`. The last `highs` numbers past `w10` are in
  [#1285](https://github.com/fluxopt/lpspec/pull/1285), taken on a machine that
  could hold them. There lpspec took 0.11 s and 0.59 GB at `transport/w100`,
  against linopy's 53.53 s and 14.26 GB.
- **Anything about expressiveness.** Four models say nothing about a fifth.

## Method

Each measurement runs in a process of its own. Peak memory is `ru_maxrss`
rather than a tracker. Import is excluded from the timing and teardown is
included. A run refuses to start on a machine that is already working. The
rest is in
[`bench/README.md`](https://github.com/fluxopt/lpspec/blob/main/bench/README.md):
every flag, every default switched off and what it costs.

**Peak carries an allocator cost that only the polars arms pay.** polars
ships its own jemalloc settings, so a peak measured through it holds pages
freed and not yet returned. An arm on the system allocator never enters
jemalloc. Our peak moves 12–27% with the decay clock on and off, where
linopy's does not move at three digits
([#896](https://github.com/fluxopt/lpspec/issues/896)). It runs against us and
is left in.

**memray never times anything.** Its tracker slows an allocation-heavy
engine several-fold and overcounts reserved arenas. Peak RSS is the metric;
memray is for attribution.
