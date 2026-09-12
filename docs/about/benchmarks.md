# How the benchmarks were taken

This page is the method behind the numbers on the
[benchmark page](benchmarks-scaling.html), for anyone deciding how far to
trust a cell there. That page holds the results: five libraries over four
models, with the numbers under each chart.

**Every published number is the median of a measurement's rounds, and every
band is the first to the third quartile of the same rounds.** Nine rounds is
the floor. The fastest round would be a best-of-n with unequal n, because the
harness calibrates by duration. A mean would be 2.9x the median here, because
one round in forty of a 20 ms measurement took 1.5 s.

The median flipped nine cells against lpspec, all on the `gurobi`
[sink](../reference/glossary.md#how-it-runs). There our build alternates
between a fast and a slow state round after round, and no other library's
build does ([#1288](https://github.com/fluxopt/lpspec/issues/1288)).

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

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +8.7 ms on lpspec and +1.5 ms on linopy and +5.7 ms on pyomo and +17.2 ms on gurobipy-loop and +29.4 ms on gurobipy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | lpspec: first | lpspec: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dispatch | 10k | 35.2 ms | **28.1 ms** | 27.4 ms | 26.1 ms | 36.8 ms | 35.0 ms | 27.0 ms | 26.9 ms | 28.4 ms | 28.6 ms |
| fleet | 12k | 138.0 ms | **137.6 ms** | 164.9 ms | 164.2 ms | 42.1 ms | 38.9 ms | 100.6 ms | 73.6 ms | 25.9 ms | 24.6 ms |
| dispatch | 100k | 45.1 ms | **34.7 ms** | 55.5 ms | 26.9 ms | 290.7 ms | 282.5 ms | 230.1 ms | 213.0 ms | 112.3 ms | 86.3 ms |
| fleet | 120k | 155.4 ms | **162.6 ms** | 167.9 ms | 166.7 ms | 783.1 ms | 771.6 ms | 861.8 ms | 868.0 ms | 186.6 ms | 158.5 ms |
| dispatch | 1M | 141.9 ms | **136.9 ms** | 37.4 ms | 35.3 ms | 4914.9 ms | 4874.3 ms | 2678.5 ms | 2709.3 ms | 919.3 ms | 852.7 ms |
| fleet | 1.2M | 350.9 ms | **322.4 ms** | 189.2 ms | 188.3 ms | 6118.4 ms | 6135.4 ms | 8697.9 ms | 8308.3 ms | 1697.9 ms | 1667.3 ms |
| dispatch | 10M | 652.8 ms | **569.6 ms** | 164.5 ms | 158.6 ms | — | — | 28775.9 ms | 28746.3 ms | 9084.8 ms | 9012.6 ms |
| fleet | 12M | 1590.7 ms | **1519.8 ms** | 470.2 ms | 468.3 ms | — | — | — | — | 17277.6 ms | 17067.0 ms |

<!-- bench:/marginal -->

## The same size, reached by widening

This table is drawn through the `highs` sink, and no run has published it. The
two cases that carry width rungs are killed there before they write a file, for
the reason [below](#not-measured-yet). [The chart page](benchmarks-scaling.html)
carries the same ladder through `gurobi`.

<!-- bench:sweeps -->
<!-- bench:/sweeps -->

## Not measured yet

Listed so that a claim with no table under it is visible as one.

- **Solve time.** Every number stops at the hand-off. The simplex is the
  solver's work whoever filled the model.
- **The LP-file round trip.** The tables price writing a file, never reading
  one back.
- **Sizes past `l`.** `xl` and `2xl` exist in the harness and no run
  publishes them.
- **`transport` and `storage` are absent on the `highs` sink, and the width
  ladder with them.** The published run takes `w100` and `w1000`. The `highs`
  job also measures the repeated build, and that budget projects the next rung
  off a `w10` cell that took under a gigabyte, so the run starts
  `transport/w100` and `storage/w1000`. `bench/memory-watchdog.sh` kills the
  case there to save the box. A killed case writes no file, so both models lose
  every rung they had already measured, their rows in the marginal table above
  included ([#1498](https://github.com/fluxopt/lpspec/issues/1498)). The
  `gurobi` job keeps both models, because it leaves the repeated build to
  `highs`. The last `highs` numbers past `w10` are in
  [#1285](https://github.com/fluxopt/lpspec/pull/1285), taken on a machine that
  could hold them: lpspec 0.11 s and 0.59 GB against linopy 53.53 s and
  14.26 GB at `transport/w100`.
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
