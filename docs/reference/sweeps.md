# Sweeps and rolling horizons

This page is the reference for `solve_over`: the axes it takes, the `Runs` it
returns, and the `carry`, `executor` and `to=` keywords. Read it when one
[model](glossary.md#the-chain) (a spec with your data attached) has to solve
once per scenario, window or period.

`solve_over` runs the same model once per slice and folds the answers together.
A slice is one set of [sources](glossary.md#how-it-runs) (the data you attach,
one table or value per name) under one key. Scenarios, rolling horizons and
myopic pathways are all the same fold. A plan cannot contain a loop, but a
process may loop over plans.

```python
import lpspec as lps

runs = lps.solve_over('spec.yaml', sources, lps.EachCoordinate('scenario'))
runs.objective  # (scenario, status, termination_condition, objective)
runs.primal('p')  # (scenario, snapshot, generator, value)
```

## The axes

An axis says how the sources split into slices. `solve_over` accepts three.

| | |
|---|---|
| `lps.EachCoordinate(dim)` | One slice per coordinate of `dim` (one label along that dimension): scenarios, draws, investment periods. A source carrying `dim` is filtered to one coordinate and the column is dropped, so the model never mentions `dim`. Every other source passes through untouched. A spec that declares `dim` is refused, since no slice would supply it. The slices run in the sorted order of the coordinates, which is the order a `carry` chains them in. |
| `lps.EachWindow(dim, length, step, into)` | One slice per window of consecutive coordinates of `dim`. `length` is what the solver sees and `step` is what the window keeps, so `length > step` is overlap. The dimension is re-indexed rather than dropped, into a dense `0..n-1` column named `into`. The spec has to declare `into`, and the model addresses the window through it. |
| a sequence of `(key, sources)` pairs | A hand-built axis. Each slice says what the *whole* model attaches, and the call must pass `key_name=`. |

```python
runs = lps.solve_over(
    'window.yaml',
    sources,
    lps.EachWindow('snapshot', length=48, step=24, into='t'),
    carry={'soc_initial': ('soc', 23)},
)
runs.primal('soc')  # (snapshot_start, t, value) — the window, and the index inside it
```

**A window spans coordinates, not values.** `length=48` is forty-eight
snapshots however they are numbered. The dimension only has to be orderable:
datetimes, strings and gapped integers all work. `into` is the dense local
index, and it keeps a seam's `where: "t == 0"` matching. It has no default,
because the name belongs to the model.

**"Each calendar month" is a precomputed column plus `EachCoordinate`.** Its
groups are unequal, so a window cannot express them. Overlap is what only
`EachWindow` offers.

**`axis.slices(sources)` is the list the axis would run.** It returns the
`(key, sources)` pairs a hand-built axis takes, so you can build one slice
alone:

```python
slices = lps.EachWindow('snapshot', 48, 24, into='t').slices(sources)
lps.build('window.yaml', slices[37][1]).write('window-37.lp')  # the one that was infeasible
```

Solved as a list, the slices key by `key_name=` and nothing is stitched:
`original_index` belongs to the axis, not to the list. Two axes compose the
same way: a comprehension over the slices of one axis, each sliced again by the
other.

**Sources cross a slice in every shape `build` takes.** A table carrying the
axis, whether a frame (an in-memory table) or a parquet path, is filtered. A
number, a `{label: value}` map or a bare sequence carries no column and passes
through as it is. A table carrying the axis that is short of a coordinate
another table has raises an `LpspecWarning` before a slice is taken, naming
both tables. That slice builds the source empty. An absent row reads as absent,
which is how a model masks, so the gap is reported rather than refused.

## Reading a sweep

**`Runs` reads like [`Result`](api.md#reading-a-result), one dimension wider.**
`primal`, `dual`, `expression`, `to_pandas`, `to_dataarray`, `to_dataset` and
`to_parquet` keep their names, and every frame has the slice key prepended.

**You name the extra dimension, not the library.** `EachCoordinate('scenario')`
keys on `scenario`, a window keys on `<dim>_start`, and `key_name=` overrides
either. So `runs.to_dataarray('p')` on a scenario sweep is
`(scenario, snapshot, generator)`: `.sel` one scenario, take a spread across
them, plot the band.

**`original_index=` asks for the answer over real coordinates.** It is a
keyword on the readers, not a reader of its own:

```python
runs.primal('soc')  # (snapshot_start, t, value) — keyed by slice
runs.primal('soc', original_index=True)  # (snapshot, value) — the answer
runs.dual('balance', original_index=True)  # the same, for a price
runs.expression('spend', original_index=True)  # the model's own quantity, over real coordinates
```

For `EachWindow` this is the stitched answer. The overlap is dropped, the
global coordinate is restored, and each window contributes the `step`
coordinates it owns. The final window can hold no more, so it keeps all of its
rows. For `EachCoordinate` and a hand-built axis nothing was re-indexed, so the
frame comes back unchanged.

**Keyed is the default, because stitching is lossy.** Stitching keeps only what
each window owns and drops the lookahead rows the sweep solved. For the same
reason `to_dataset` and `to_parquet` have no `original_index`: a bulk export
keeps every row the sweep holds.

**Every bridge takes `kind=`, the way `scan` does.** `to_pandas(name, kind)`,
`to_dataarray(name, kind)` and `to_dataset(*names, kind)` read `primal`, `dual`
or `expression`, with `primal` the default. `original_index` sits beside it
where the reader has one. So `runs.to_dataarray('balance', 'dual', original_index=True)`
is the stitched price over time, and `runs.to_dataset(kind='expression')` is
every expression the slices evaluated. One call reads one kind: a dual and a
variable of the same name would collide in one dataset.

**`to_parquet` writes every kind.** `runs.to_parquet('runs/')` writes what
`to=` would have written: every primal, dual and expression, one file per slice
and name, with the record and the manifest. The directory is a spilled sweep.
`scan` reads it, and the call that made the sweep, pointed at it with `to=`,
reads it back without solving.

**There is no per-slice reader.** One slice is a partition of a frame you
already hold: `runs.primal('p').partition_by(runs.key_name, as_dict=True)`.

| Rule | |
|---|---|
| **everything a slice produced is kept** | Every variable's primals and every constraint's duals come back through `runs.primal(name)` and `runs.dual(name)`. Each slice's *model* is released as the loop goes, so build peak stays at one slice however many there are. What accumulates is the answer. |
| **duals are keyed, never combined** | `runs.dual(name)` has the shape of `runs.primal(name)`. Averaging window prices, taking the last, and reading one slice alone are all defensible, so the reduction is yours. A slice whose model had an integer variable contributes no duals, and `runs.objective` says which slice. |
| **expressions are evaluated per slice** | Every declared `expressions:` name is evaluated at each slice's solution and read through `runs.expression(name)`. Under `original_index=True` only the rows each window owns survive, so summing the stitched frame cannot double-count the lookahead. A quantity *reduced over* the sliced dimension has no way back and is refused there. The error names the per-slice read as the alternative. |
| **no aggregate objective** | `objective` is a frame keyed by slice. Scenarios are a distribution, not a sum, and summing window objectives double-counts whatever the overlap discards. |
| **the lookahead is `t >= step`** | Overlapping windows return every row they solved, including the tail the next window recomputes. Keeping only what each window owns is one clause and no special case: `runs.primal('soc').filter(pl.col('t') < step)`. |
| **a slice that did not solve contributes no rows** | So a `primal` frame can be shorter than the sweep. `objective` is always one row per slice, and it records which slices did not solve. |
| **a window keys as `<dim>_start`** | `EachWindow('snapshot', …)` drops `snapshot` and re-indexes to `into`, so the key column is `snapshot_start` and holds where each window began. |
| **a hand-built axis names its own key** | A plain list of slices cannot say what its keys are coordinates *of*, so it must pass `key_name='draw'`. `key_name` overrides the derived name on any axis. It is refused only when it collides with a column the frames already carry: a dimension the spec declares, or `value`, `status`, `termination_condition`, `objective`. |
| **`runs.diagnostics` says what each slice cost** | One row per slice, `(key, columns, rows, nonzeros, loaded, attach, build, handoff, solve)`: `model.diagnostics()` one dimension wider, its counts and clocks only. `loaded` says the solver took the model from scratch. A serial sweep loads once and pushes values after, so a later `True` is a slice whose data moved a mask. Under `executor=` every slice builds alone and every one loads. The clocks are that slice's own seconds, so a slow sweep says which slice and which phase. |
| **a slice that fails says which slice** | The error is the engine's own, untouched, with a note on it: `in slice 'bad' (3 of 3)`. A fifty-window traceback names the window. |
| **a sweep's memory grows with its answer, unless it is spilled** | The models are released as the fold goes; the extracted frames accumulate. `to=` writes them out instead ([below](#spilling-a-sweep-to-disk)), and `to_parquet` writes a held sweep out the same way, after the fact. |

## Spilling a sweep to disk

`to=` names a directory. Each slice's frames are written there as the fold
goes rather than held, so the sweep's memory stays at one slice however many
there are:

```python
runs = lps.solve_over('window.yaml', sources, lps.EachWindow('snapshot', 48, 24, into='t'), to='runs/')
runs.scan('soc')  # a LazyFrame: (snapshot_start, t, value), every window, in order
runs.scan('balance', 'dual', original_index=True).collect()  # the same readers, the same keywords
```

| Rule | |
|---|---|
| **`scan` is the reader** | `runs.scan(name, kind='primal')` returns `primal`, `dual` or `expression` as a `LazyFrame` over the files, `original_index=` included. On a sweep held in memory it is the same reader made lazy, so a line written for a spilled sweep runs unchanged on one that fit. The eager readers and the exports refuse a spilled sweep and name `scan`: `primal` returns a frame in memory or raises, never one it would have to load first. |
| **one file per slice and name** | `<kind>/<name>/<position>.parquet`, with the slice key a column of each. `objective/` and `diagnostics/` hold the record, one row per slice. `runs.objective` and `runs.diagnostics` stay in memory; they are small. |
| **every file lands whole** | A file is written beside its final name and renamed into place. The objective file is written last and marks a slice done, so a slice interrupted part way is solved again rather than read back short. |
| **an interrupted sweep resumes** | Run the same call at the same directory. A slice already there is read back, not solved, and under a `carry` the state is read off its file. Only the slices that had not finished are built. |
| **a directory holds one sweep** | `sweep.json` records the key name and the keys. A different sweep pointed at the directory is refused. The same sweep over changed data or a changed model is not detected, so delete the directory to solve again. |
| **the parent writes** | Under `executor=` a worker's answer crosses back to the parent, and the process that owns the directory writes it. |

## Carrying state between slices

`carry` copies one slice's answer into the next slice's data. It is a mapping
`{parameter: (variable, index)}`.

```python
runs = lps.solve_over(
    'window.yaml',
    sources,
    lps.EachWindow('snapshot', length=48, step=24, into='t'),
    carry={'soc_initial': ('soc', 23)},
)
```

| Rule | |
|---|---|
| **a carry is a copy, never arithmetic** | Accumulation (`existing += built`) is a derived variable in the YAML, where the maths is reviewable. |
| **the two declarations say what is copied** | The carry collapses the one dimension the *variable* has and the *parameter* does not, and `index` names a coordinate of it. Every other dimension rides along. So `soc` over `(t, storage)` into `soc_initial` over `(storage)` drops `t` and hands both stores forward. `total` over `(generator)` into `existing` over `(generator)` drops nothing and needs no index; pass `None`. |
| **the index is explicit, and it is a kept coordinate** | With `EachWindow(…, 48, 24, …)` the state to carry sits at coordinate 23 of `into`, not 47. The rows from `step` on are the lookahead the next window solves again, so an index there is refused, and the error names 23. |
| **the first slice needs a seed** | `carry` supplies the parameter from the second slice on. The first slice takes it from `sources`, and a sweep whose sources lack it is refused before a slice is taken. |
| **a carry is checked before anything is read** | The dims come from the YAML, so a carry that cannot line up raises before the axis has scanned a single source: collapsing two dimensions at once, a parameter over more dimensions than the variable, an index where the sides already match, no seed, an index in the lookahead. `check` cannot answer this for you, because `carry` is an argument to the call, not part of the model. |
| **the last slice carries nothing** | There is no next slice to read it. |
| **a slice that leaves nothing to carry stops the sweep** | An infeasible window has no level to hand forward, so the next window cannot start. The error names the slice, how it terminated, and the slice left waiting. A sweep without a carry records the slice in `objective` and goes on. |
| **`carry` excludes `executor`** | A carried value makes slice *i+1* depend on slice *i*, so the slices cannot run concurrently. The call is refused. |

## Running slices in parallel

`executor` is any
[`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#executor-objects):
a `submit` that returns a `Future`, and nothing else. The package ships no
remote transport and no vendor integration, so the executor is the extension
point.

| | Use it when | Notes |
|---|---|---|
| `None` *(default)* | always, until measurement says otherwise | Sequential. Nothing is serialised, because nothing crosses a boundary. |
| `ThreadPoolExecutor` | rarely | Works, and sources are **not** encoded. Polars is already multithreaded, so slices contend with its pool. Threads share an address space, so peak memory is additive rather than per-worker. |
| `ProcessPoolExecutor` | genuine local parallelism | **Must not use `fork`**; see below. Sources cross as parquet. |
| anything remote | a cluster you already run | dask's `Client`, ray's wrappers, loky. The workers are assumed not to share your filesystem, so paths travel as bytes. Pass `workers_share_fs=True` if the workers do mount it. |

**A forked worker hangs.** The polars thread pool does not survive `fork`, and
the failure is a hang rather than an error, so it looks like a slow solve.
`solve_over` cannot enforce the start method, because a remote executor has
none to inspect. Pass the context yourself:

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def main():
    ctx = multiprocessing.get_context('spawn')  # or 'forkserver'
    with ProcessPoolExecutor(4, mp_context=ctx) as pool:
        runs = lps.solve_over('spec.yaml', sources, lps.EachCoordinate('scenario'), executor=pool)


if __name__ == '__main__':  # spawn re-imports your module; without this it recurses
    main()
```

**Parallel is N × peak.** Each worker holds its own slice's model, so a
four-way pool wants four times the memory of one slice. That is a machine
decision, which is why `None` is the default rather than a pool sized for you.

**Pass paths or frames, whichever you already have.** Sources cross a process
boundary as parquet, never as pickled frames. A path the workers can reach
stays a path. One they cannot reach travels as its own bytes, untouched. Either
way a source no slice rewrote is encoded once for the whole sweep rather than
once per slice. There is nothing to tune. `df.lazy()` is not an optimisation:
an eager frame is embedded in the plan, so it pickles *larger* than the frame.
Only `scan_parquet` is a reference.

## How a sweep runs

| | |
|---|---|
| **a partition is a filter on the sources** | Not a narrower index: the containment check refuses parameter rows outside the declared coordinates. The axis rewrites the rows and the index they are over in one mapping. |
| **one model, updated per slice** | Every slice is the same maths over different numbers, so a serial sweep builds once and [updates](api.md#re-solving-with-new-numbers). The YAML is parsed once and the plan lowered once, and a slice whose structure matches the last keeps the loaded solver. A sweep under `executor=` builds per slice, because a built model is the one thing that does not cross a process. |
| **`keep=` reaches every slice, and the fold chooses none of them** | It defaults to `'solver'`, exactly as [`solve`](api.md#how-much-of-the-session-a-solve-keeps) does. In a fold `keep='progress'` has something to carry, since consecutive slices differ by one step. Whether carrying pays is a fact about the *model*, so the driver does not decide for you. Under `executor=` it cannot apply: a pooled sweep builds per slice, so every slice is a first solve and keeps `'nothing'`. |
| **the model is asked before it is sliced** | The plan says what each axis can bear ([`separability`](https://math-spec.readthedocs.io/en/latest/reference/language/reading/#asking-whether-an-axis-can-be-cut)). `EachWindow` needs `into` *windowable*. It refuses a coupling by naming the declaration and the change that would lift it. It reads an offset the data decides off the data. It requires `length - step` to cover what the rows read ahead, and it warns where a `position()` restarts per window. What the rows read *behind* is what a window's first rows meet the edge policy with, the rolling-horizon seed, and it is not refused. `EachCoordinate` is not asked: the column it slices is one the spec must not declare, and a spec that declares it is refused where the key is named, so the model never sees the axis. |
| **the model is parsed once** | `solve_over` validates it up front, so a model outside the language fails before the data is touched. Every worker is handed the lowered [program](glossary.md#the-chain) (the spec after parsing, ready to build), in this process or across one, and none reads the YAML or lowers it again. |
| **a slice is total** | A slice says what the *whole* model attaches, not what changed since the one before it. The class axes always do; a hand-built list has to keep the rule. |
