# Glossary

This page holds the one definition of each name this project uses, for anyone
who meets a term on another page. The rest hang off one distinction:

> A **spec** is the math you write. A **model** is that spec with your data on
> it. A **result** is one answer read back.

```
check ──▶ Program ──▶ build ──▶ Model ──▶ solve ──▶ Result
 (spec)   (lowered)   (+data)             (answer)
```

## The chain

**Spec**
: The math, before any data: a YAML file, a mapping, or an object the language
  has already read (a `Spec` from `math_spec.to_spec`). It declares the
  dimensions, parameters, variables, constraints and objective, and it is what
  is checked for being *sayable*. It carries no numbers. Every verb takes it
  first, spelled `spec` in the signatures.

**Program**
: A spec after the language has parsed, expanded, validated and *lowered* it to
  the internal plan. [`check`](api.md) returns one, and a build reads its rows
  off one. Still no data. It is `math_spec`'s own type, so typeset it or read
  its declarations through that package.

**Model**
: A spec with your data attached to it, which is what [`build`](api.md) returns
  (the class `lpspec.Model`). One model feeds any number of sinks:
  `model.solve()`, `model.write(path)`, `model.row(...)`, `model.diagnostics()`.
  `model.update(...)` puts new numbers on it in place. Instances are named
  `model` in the code.

**Result**
: One answer, read back from a solve: `result.objective`, `result.primal(name)`,
  `result.dual(name)`, `result.expression(name)`. It owns the frames it reads,
  so it outlives the model it came from.

## The verbs

**check** · **build** · **solve** · **write**
: The four things you can do, all on a spec plus, for the last three, sources.
  `check(spec)` validates and lowers. `build(spec, sources)` returns a
  [Model](#the-chain). `solve` and `write` build and then solve or stream in a
  single call. There is no Python API for constructing a spec: the math is
  written in YAML.

**update**
: `model.update(sources)` puts new numbers on the same model, in place, without
  re-reading the YAML or re-lowering the plan. Only what changed is named. When
  the change moves a mask, by renumbering labels, the model is rebuilt and
  solved cold rather than pushed onto a loaded solver.

**Buildable**
: The type alias for anything the verbs accept as the spec: `str | Path |
  dict | Spec | Program`.

**Source**
: The type alias for anything the verbs accept under one name of `sources`: a
  parquet path, a table (polars, pandas, or any Arrow-capsule table), or the
  plain-Python shapes, a `{label: value}` map, a sequence, one number.

**Label**
: One member of a dimension, `wind` say, and the type alias for it:
  `int | float | str | datetime`, the four dtypes an index may declare. A
  sweep's slice key is a label too.

## The data

**Index**
: A dimension's labels, in order. It is supplied under the dimension's own key
  in `sources`, the first occurrence of each label is its position, and that
  order is what `shift` reads positionally
  ([the data contract](data.md#where-coordinates-come-from)).

**Coordinate**
: One point of a declaration's dimensions: for a variable over
  `[snapshot, generator]`, one snapshot for one generator. A parameter has a
  value at each coordinate it covers, or no row there.

**Frame**
: A table in memory, polars by default, with one column per dimension and a
  `value` column, one row per coordinate. Sources carry frames in, and results
  hand frames back.

**Mask**
: The `where:` on a declaration. A coordinate the mask excludes has no row and
  no column: it is absent, not zero, and the built model is smaller for it
  ([absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/)).

## How it runs

**Lane**
: One of the two ways a spec is executed. The **relational lane** (the default,
  `lpspec.build`/`solve`) validates at load time, lowers to the plan, and
  streams relationally on polars. The **linopy lane** (`lpspec.linopy`, needs
  the `[linopy]` extra) constructs the same spec as a `linopy.Model`. Both
  accept exactly the same language, which is why the differential tests are an
  oracle rather than a comparison of dialects.

**Engine**
: The relational lane's builder. It fills the model's frames from the attached
  data and hands them to a sink.

**Sink**
: Where the built [tables](#the-built-form) land: a solver (`highs`, `gurobi`,
  `xpress`) or a file writer (`.lp`, `.mps`). `linopy` is a lane, not a sink.

**Sources**
: The data you attach: a mapping of parameter, dimension and lookup names to
  tables (parquet paths or in-memory frames), and dimension names to their
  labels.

**attach**
: Fitting your sources onto a spec to make a [Model](#the-chain). `build` does
  it, and `update` does it again with new numbers. There is no separate public
  verb for it. The data operation is called *attach*, never "bind", so that
  `bound` is free to mean one thing only (below).

## The built form

**Tables** (a `tables` value)
: The built model as a sink sees it: the numeric problem in frames,
  `cols` (bounds, type), `obj`, `rows`, `matrix` (CSR), plus `quad` and `sos`.
  The class name is `Tables`, and the variables that hold one are named
  `tables`. It is the built form of a model, not the spec.

**keep**
: How much of a solve session `model.solve` may carry to the next solve:
  `solver` (reuse the loaded solver, default), `progress` (keep its work too),
  or `nothing` (a cold baseline). See [the verbs](api.md).

**solve_over** (a sweep)
: Solve one spec once per slice of an axis (scenarios, windows, periods) and
  fold the answers together into a `Runs`. It is a fold: the previous slice's
  model is released as the loop goes.

## `bound` means one thing

**bound**
: A lower or upper limit on a variable or a constraint row: the `bounds:` of a
  declaration, the `BOUNDS` section of an `.mps` file, an absent bound the
  solver reads as infinity. Nothing else. Attaching data to a spec is
  [**attach**](#how-it-runs), never "bind", so that `bound` carries no second
  meaning.
