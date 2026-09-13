# Glossary

The one definition of each name this project uses. The rest hang off one
distinction:

> A **spec** is the math you write. A **model** is that spec with your data on
> it. A **result** is one answer read back.

```
check ──▶ Program ──▶ build ──▶ Model ──▶ solve ──▶ Result
 (spec)   (lowered)   (+data)             (answer)
```

## The chain

**Spec**
: The math before any data: a YAML file, a mapping, or a `Spec` from
  `math_spec.to_spec`. It carries no numbers, and every verb takes it first.
  What it may contain is
  [the language](https://math-spec.readthedocs.io/en/latest/reference/language/).

**Program**
: The spec lowered to the plan a build reads its rows off: what [`check`](api.md)
  returns, still with no data. The two states are the language's
  ([reading a loaded model](https://math-spec.readthedocs.io/en/latest/reference/language/reading/#two-states-and-the-difference-between-them)).

**Model**
: A spec with data attached: what [`build`](api.md) returns (`lpspec.Model`).
  One model feeds any sink: `solve()`, `write(path)`, `row(...)`,
  `diagnostics()`. `update(...)` puts new numbers on it in place.

**Result**
: One answer read back from a solve: `objective`, `primal(name)`,
  `dual(name)`, `expression(name)`. It owns its tables, so it outlives its
  model.

**Archive**
: A spec, the data it was solved with and what came back, written together as
  one zip or one directory by `archive=` ([archiving](api.md#archiving-a-model)).
  It reads back as a `SolveArchive`, or a `SweepArchive` where the sources were
  cut. Never "artifact".

## The verbs

**check** · **build** · **solve** · **write**
: `check(spec)` validates and lowers. `build(spec, sources)` returns a
  [Model](#the-chain). `solve` and `write` build and then solve or stream in
  one call. There is no Python API for constructing a spec.

**update**
: `model.update(sources)` puts new numbers on a built model in place, naming
  only what changed. A change that moves a mask rebuilds and solves cold.

**load** · **scan**
: The two ways a saved answer is read back, differing in when the bytes move.
  `load_result`, `load_runs` and `load_archive` read **whole**: the frames are
  in memory when the call returns, and the directory is free afterwards.
  `scan_result`, `scan_runs` and `scan_archive` read each frame at the call
  that asks for it, and the files have to outlive what was read off them
  ([loading or scanning](api.md#loading-or-scanning)). Each pair takes the same
  arguments and hands back the same type. Never "open".

**Buildable**
: The type alias for a spec argument: `str | Path | dict | Spec | Program`.

**Source**
: The type alias for one value of `sources`. The shapes it covers are
  [the data contract](data.md#what-a-parameter-accepts).

**Label**
: One member of a dimension, `wind` say, and its type alias:
  `int | float | str | datetime`. A sweep's slice key is a label too, and
  `EachCoordinate(dim)` slices on one label of `dim` at a time.

## The data

**Index**
: A dimension's labels in order, supplied under the dimension's own key in
  `sources`. `shift` reads that order positionally
  ([the data contract](data.md#where-coordinates-come-from)).

**Coordinate**
: One point of a declaration's dimensions: one snapshot for one generator. A
  parameter has a value at each coordinate it covers, or no row there. The
  language calls the dimensions themselves the declaration's *frame*
  ([named expressions](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#named-expressions)).

**Table**
: A polars `DataFrame` with one column per dimension, a `value` column and one
  row per coordinate: what a parameter arrives as, and what `primal` hands
  back. The code calls one a **frame** and means the same thing. The plural
  [Tables](#the-built-form) is a different noun: the built model as a sink sees
  it.

**Mask**
: The `where:` on a declaration. What an excluded coordinate means is
  [absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/).

## How it runs

**Lane**
: One of two ways a spec is executed. The **relational lane** (the default)
  validates at load time, lowers to the plan and streams on polars. The
  **linopy lane** (`lpspec.linopy`, the `[linopy]` extra) builds the same spec
  as a `linopy.Model`. Both accept the same language
  ([relationship to linopy](../about/linopy.md#2-it-is-the-oracle)).

**eager**
: The linopy lane, and nothing else — the **eager lane** in the differential
  suite and the benchmark harness. Never a reader: how a saved answer is read
  is [load or scan](#the-verbs).

**Engine**
: The relational lane's builder: it fills the model's tables from the attached
  data and hands them to a sink.

**Sink**
: Where the built [tables](#the-built-form) land: a solver (`highs`, `gurobi`,
  `xpress`) or a file writer (`.lp`, `.mps`). `linopy` is a lane, not a sink.

**Sources**
: The data you attach: parameter, dimension and lookup names to tables, and
  dimension names to their labels.

**attach**
: Fitting sources onto a spec to make a [Model](#the-chain); what `build` does
  and `update` does again. Never "bind", so that `bound` means one thing.

## The built form

**Tables**
: The built model as a sink sees it: `cols` (bounds, type), `obj`, `rows`,
  `matrix` (CSR), `quad` and `sos`.

**keep**
: How much of a session `model.solve` carries to the next solve: `solver`
  (default), `progress` (its work too) or `nothing` ([the verbs](api.md)).

**solve_over** (a sweep)
: Solve one spec once per slice of an axis and fold the answers into a
  `Runs`, releasing each slice's model as it goes. A sweep, never a "study".

**held** · **spilled**
: Where a sweep's frames are. A **held** sweep carries them in memory, and
  `runs.primal(name)` and the exports — the **frame readers**, the ones that
  hand back a table — answer off them. A **spilled** sweep left them in a
  directory, which is what `spill_to=` writes and what `scan_runs` reads: there
  `runs.scan(name)` is the reader and the frame readers refuse
  ([spilling](sweeps.md#spilling-a-sweep-to-disk)).

## Row types

**Record** · **Metrics** · **SliceMetrics**
: The three saved rows, each a `NamedTuple` that names its own columns. Where
  a column is nullable, the type also derives the schema it is written with,
  so an all-null column keeps its own type instead of the one a single row
  infers. **Record** is how a solve terminated, one per solve. **Metrics** is
  what it took — the sizes, what the sink added to them, the counters and the
  clocks, every clock naming its unit — and is what `archive.metrics` hands
  back ([the attributes](api.md#diagnostics)). **SliceMetrics** is one slice of
  a sweep's share of that, in its own columns, and is the row behind
  `runs.metrics`.

  A **row** is a value and gets a type; a **table** stays a
  [Table](#the-data). So `Record` and `SliceMetrics` are the rows behind
  `runs.objective` and `runs.metrics` rather than what those hand back, and
  a reader that wants one row of a table asks the frame for it.

## `bound` means one thing

**bound**
: A lower or upper limit on a variable or a constraint row: the `bounds:` of
  a declaration, the `BOUNDS` section of an `.mps` file, an absent bound the
  solver reads as infinity. Nothing else; data is
  [attached](#how-it-runs), never bound.
