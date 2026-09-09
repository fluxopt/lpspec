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

## The verbs

**check** · **build** · **solve** · **write**
: `check(spec)` validates and lowers. `build(spec, sources)` returns a
  [Model](#the-chain). `solve` and `write` build and then solve or stream in
  one call. There is no Python API for constructing a spec.

**update**
: `model.update(sources)` puts new numbers on a built model in place, naming
  only what changed. A change that moves a mask rebuilds and solves cold.

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
  back.

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
  `Runs`, releasing each slice's model as it goes.

## `bound` means one thing

**bound**
: A lower or upper limit on a variable or a constraint row: the `bounds:` of
  a declaration, the `BOUNDS` section of an `.mps` file, an absent bound the
  solver reads as infinity. Nothing else; data is
  [attached](#how-it-runs), never bound.
