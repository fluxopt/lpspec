# Python API

This page describes what each verb takes, returns, guarantees and refuses, for
anyone who runs a spec from Python. A *spec* is the math as you wrote it: the
YAML file. What that file may contain is
[the language](https://math-spec.readthedocs.io/en/latest/reference/language/);
this page is what loads, checks, builds, solves and reads one back.

```python
import lpspec as lps

lps.check('spec.yaml')  # compiles? no data needed

result = lps.solve('spec.yaml', sources)
result.objective
result.primal('p')  # a polars.DataFrame
result.dual('power_balance')
```

## The verbs

Every verb takes the spec first and, except `check`, the *sources* second: the
tables that carry its numbers. A *model* is the spec with those tables attached.
A *result* is one answer read back from a solve. A *sink* is where a built model
lands: a solver, or a file writer. The [glossary](glossary.md) holds the one
definition of each.

| | |
|---|---|
| `lps.check(spec, sink=None)` | parse, expand, validate and lower; attach no data. With a `sink`, also say whether that sink will take it. Returns the lowered `Program`, which every verb here takes back |
| `math_spec.to_spec(spec)` | the file as written, for editing and typesetting it. This is the language's own verb, from the package that owns it |
| `lps.build(spec, sources)` | attach data and build; returns a `Model` |
| `lps.solve(spec, sources, solver_name='highs', solver_options=None)` | build and solve in one call; returns a `Result` |
| `lps.solve_over(spec, sources, axis, ...)` | solve once per slice and fold the answers: [sweeps](sweeps.md) |
| `lps.write(spec, sources, out)` | build and stream to a file; the suffix picks the format |
| `lps.pack(spec, sources, out)` | the file and its data as one zip, to archive or send: [Archiving a model](#archiving-a-model) |
| `lps.unpack(path, into)` | extract it: the `Spec` and the sources as parquet paths, in the shape every verb takes |
| `model.row(name, **coordinate)` | what one built constraint row says: terms, comparison, right-hand side |
| `math_spec.to_latex` / `to_typst` / `to_markdown` | the math as a document: [typeset](https://math-spec.readthedocs.io/en/latest/reference/typeset/) |
| `lps.Model` / `lps.Result` / `lps.Runs` | the types the verbs hand back, importable so a wrapper can annotate its own signature. The spec going *in* is the language's type, `math_spec.Spec` or `math_spec.program.Program`, from the package a caller already called to get one |

## Errors and warnings

**Every error is one tree, rooted at `LpspecError`.** `LanguageError` (with
`SchemaError`, `DimensionError`, `PiecewiseExpansionError`) is a fault in the
spec. `DataError` is a fault in the data attached to it. `LaneError` is a spec
one lane cannot build. `NoSolutionError` is a solve that left nothing to read.
Which one you get:
[errors](https://math-spec.readthedocs.io/en/latest/reference/language/errors/#which-error-you-get).

**`LpspecWarning` is the one warning category**, and carries `check`'s advice.
`warnings.simplefilter('error', lps.LpspecWarning)` makes a spec repository
fail CI on it.

## The spec argument

**Every verb takes the spec as a path, a `str`, a `dict`, a `Spec` or a
`Program`.** That is exactly what `math_spec.to_program` takes, and it is what
opens the spec. `check`, `build`, `solve`, `write`, `solve_over`, `Model` and
both linopy-lane verbs share that first argument, so a framework that emits
declarations never writes a temporary file to run them:

```python
spec = {'dimensions': ..., 'variables': ..., 'constraints': ..., 'objective': ...}

lps.solve(spec, sources)  # a dict runs like a file
checked = lps.check(spec)  # ...or lower once and keep the plan
lps.solve(checked, sources)  # a Program is passed through, not re-lowered

to_spec(spec).to_yaml()  # the review copy — a dict-built spec still gets a file
```

**A framework emits data, not YAML text, and never merges files.** That is the
supported path. The last line is its condition: a generated spec must be able
to show you a file. Hand-written math still starts as a file.

**A `Spec` goes back out two ways, and they agree.** `to_dict()` is the spec as
data. `to_yaml()` is that dict as the file you review and diff. Loading,
dumping and loading again is stable for both forms, and dumping twice gives
the same bytes.

**Every value is written; only what is absent is dropped.** Absent is a null,
an infinite bound, or a mapping that declares nothing. An infinite bound is the
unbounded side, which omitting the bound already means. An empty list stays:
`foreach: []` is a scalar declaration.

## The sources argument

`sources` maps each declared name to its data: a parquet path, or any table
that exposes the Arrow PyCapsule protocol (polars, pandas, pyarrow). A
dimension's own key supplies labels that neither the sources nor the YAML
carries. The exact rules are [the data contract](data.md). The whole set of
accepted shapes is the type `lpspec.lanes.Source`, which every verb annotates
`sources` with.

```python
result = lps.solve(
    'dispatch.yaml',
    {'load': 'load.parquet', 'cost': cost_frame, 'p_max': p_max_frame},
)
```

**`sources` is the whole of the build's input**: parameters and dimension
indexes in one mapping. **`solver_options` is not a build knob.** It is
forwarded to the solver verbatim.

## Checking a spec

**`check` is the CI verb.** It parses, expands, resolves and lowers the spec
and attaches nothing, so a spec repository can validate every commit without
shipping the data. It returns the *program*: the spec lowered to the plan a
build reads its rows off.

### Checking against a sink

Whether a spec is *sayable* does not depend on the solver. Where it can *land*
is a separate question
([what a sink can ingest](https://math-spec.readthedocs.io/en/latest/about/ceiling/#capability-is-not-the-ceiling)),
and `sink=` asks it:

```python
lps.check('spec.yaml')  # sayable?
lps.check('spec.yaml', sink='highs')  # ...and will HiGHS take it?
lps.check('spec.yaml', sink='.lp')  # ...will the LP writer?
```

`sink` is a solver name (`highs`, `gurobi`) or an output suffix (`.lp`). **It
is optional and silent by default.** With a sink named, you get back one of:

- **A refusal (`LpspecError`)** if the sink has no such concept, or refuses the
  combination. The message names the construct, the sink, *and* the sinks that
  do take it. Degree 2 is what reaches one: only Gurobi and the LP writer take
  a quadratic row, and HiGHS refuses a quadratic objective *beside*
  integrality while taking either alone.
- **A warning** if the sink takes it only by rewriting. `sos:` on HiGHS is the
  one case: the set arrives as binaries, so a spec that declared no
  integrality comes back mixed-integer and without duals.

**`check` answers off a declared table, with no data and no installed
solver.** `check(m, sink='gurobi')` answers on a machine that has never had
gurobipy.

**`solve` and `write` read the same table**, so a refusal comes whether or not
you asked. `lps.write(m, sources, 'model.mps')` on a model carrying a
quadratic term is refused by name rather than handed back as a file whose
quadratic rows are missing. `sink=` gives the same sentence before the build.

### What each sink takes

The four quadratic rows, and the two sections HiGHS writes but will not read
back, are probed against the shipped solvers by
`tests/test_sink_capability_probes.py` and
`tests/test_gurobi_capability_probes.py`. The rest are read off the APIs.

| | `lp_file` | `mps_file` | HiGHS direct | Gurobi direct | Xpress direct |
|---|---|---|---|---|---|
| affine rows, COO, integrality | text | text, `MARKER` | native | native | native |
| semi-continuous | text | **not written** — no `SC` bound | `kSemiContinuous` | native | native |
| SOS1 / SOS2 | text section | `SOS` section | **no concept** — rewritten to binaries | `addSOS` | native |
| indicator | text section | **not written** | **no concept** | `addGenConstrIndicator` | native |
| convex quadratic objective | text section | **not written** | `passHessian` | `setMObjective` | **no path here** |
| nonconvex quadratic objective | text section | **not written** | **refused** | native, at default parameters | **no path here** |
| quadratic objective **and** integrality | text section | **not written** | **refused** | native (MIQP) | **no path here** |
| quadratic constraint | text section, unreadable | **not written** | **no concept** | `addQConstr` | **no path here** |

- **HiGHS excludes quadratic twice**: by convexity, and by conjunction with
  integrality. Neither is a set membership.
- **The `lp_file` column says what can be written, not what reads back.** The
  same HiGHS parser takes the quadratic-objective section and refuses both
  the `sos` and the quadratic-constraint sections.
- **"No path here" describes this package, not Xpress.** The Optimizer takes
  a Hessian; the sink in `solvers/xpress.py` never hands it one. A descriptor
  says what the sink ingests, not what the library could.

## Building a model

`lps.build` returns a `Model`: the math with your data on it. Build once when
one model should feed more than one sink, or be solved more than once:

```python
model = lps.build('spec.yaml', sources)
model.write('model.lp')
result = model.solve()
model.diagnostics()  # what the build and its solves did that the answer does not show
model.row('balance', snapshot=17)  # what one row actually says
```

**Questions about the model are `build`'s, not `solve`'s.** `solve` hands
back an answer and `write` a path. How big the model is, what it did not
build, what one row says and how its re-solves went are all answered by the
`Model`.

### Reading one row

`row` says what one constraint, at one *coordinate*, says once the data is on
it. One point of a constraint's dimensions, one snapshot for one generator, is
a coordinate. `to_latex` and its siblings render the model as math before any
data, and `result.dual('balance')` gives a row's number without its terms;
`row` is the third question, and the one a wrong model is debugged by.

```python
print(model.row('balance', snapshot=1))
# balance[snapshot=1]: +1 p[1, wind] +50 p[1, gas] +30 p[1, coal] >= 60
```

**The line is linopy's format.** Their `Constraint.print()` renders a row the
same way. The row's own identity is added on the same line, where linopy
prints it as a header.

The same content is a frame, for a row too wide to read and for anything that
filters or joins:

```python
row = model.row('balance', snapshot=17)
row.terms  # (variable, coordinate, coefficient), one row per term
row.sense  # '=='
row.rhs  # 80.0
```

**A row too wide to spell out is summarised, not truncated.** Twelve terms of
three hundred would be twelve arbitrary ones:

```python
print(model.row('balance', t=0))
# balance[t=0]: 301 terms — p: 300 (|coef| 0.001…0.3), slack: 1 (|coef| 1000) >= 5
```

The line says how much of the row each declaration contributes, and whether
its coefficients span an order of magnitude. `diagnostics().coefficient_range`
reports that spread per *declaration*; nothing reports it per row.
`display_terms` sets where a line stops spelling terms out.

**`row` reads the built row.**

- A coefficient is the number the *data* produced, where the file shows a
  parameter name, and every digit of it.
- A term whose variable a `where` masked out is **not there**, so a row
  shorter than the file suggests says so.
- A term whose coefficient the data made **exactly zero** is not there
  either. The build prunes it, so the row reads the matrix the solver was
  handed.
- A row a `where` removed raises, and the message names the three things that
  cause it.

**`row` needs no solve.** A model too wrong to solve is the one whose rows need
reading.

**The coordinate names every dimension of the declaration.** A partial one
names a set of rows rather than one. The constraint is positional, so a
dimension may be called `name` and still be named in the coordinate. A label
the dimension cannot hold (a string against an integer dimension, a stranger
against a declared label set) is refused naming the dimension, not the dtypes.

**There is no verb for a column.** A variable's bounds are in `to_yaml()`. Its
coefficients are the transpose of `row`, which nothing exposes.

## Reading a result

```python
result.status, result.termination_condition, result.objective
result.is_ok  # rolled-up verdict: not an error, abort or refusal
result.has_primal  # narrower: are there values to read
result.kept  # how much of the session this solve kept: 'nothing', 'solver' or 'progress'

result.primal('p')  # tidy frame (dims…, value) in label order — the native shape
result.dual('power_balance')  # shadow prices, same shape, same join
result.activity('power_balance')  # each row's left-hand side at the solution
result.expression('co2')  # a named expression at the solution, over its own dims

result.to_pandas('p')  # the same, as a DataFrame
result.to_dataarray('p')  # the same, labelled: .sel / resample / plot
result.to_dataarray('power_balance', 'dual')  # a price, labelled — every bridge takes kind=
result.to_dataset()  # every variable by default; names for a subset
result.to_dataset(kind='dual')  # every dual; one kind per dataset
result.to_parquet(
    directory
)  # every kind, primal/ dual/ expression/, one file per name; primals streamed, never through this process
```

**`primal` returns a `polars.DataFrame`**, one row per coordinate: a *frame*.
It is Arrow-backed, so it exports the same protocol the loader recognises.
`to_pandas` and `to_dataarray` are the bridges out. They need pandas and
xarray, which ship with the `[linopy]` extra.

| Rule | |
|---|---|
| **`is_ok` is not `has_primal`** | `is_ok` rolls up the termination condition. `has_primal` adds the solver's verdict on whether an incumbent exists, and every reader gates on it. A MIP that hits `time_limit` before finding a feasible point is `ok` with nothing to read |
| **reading with no primal raises** | `NoSolutionError`; `objective` is `nan` |
| **`expression` takes a declared name** | the value of a [named expression](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#named-expressions) at the solution, aggregated to its own dimensions. Never an expression string. An unknown name is a `KeyError` listing what is declared. It is compiled at the read, so a build with fifty declared expressions that reads none pays for none |
| **`dual` raises rather than zero-filling** | no values at all is `NoSolutionError`. Values but no duals is `LpspecError`, and any integer or binary variable makes duals undefined |
| **a solver can make a model mixed-integer** | an [`sos:`](https://math-spec.readthedocs.io/en/latest/reference/language/piecewise/#sos) set reaches a solver with no SOS concept as binaries, so an otherwise continuous model solved on `highs` has no duals and says so. `gurobi` and `xpress` branch on the set itself and keep them |
| **duals exist only where a solver ran** | a model written to LP and solved elsewhere never passes back through here. Reduced costs and slacks are not exposed |
| **`to_dataset` costs what it says** | each variable arrives dense over its own dimensions. Name a subset, or use `to_parquet` |
| **every bridge takes `kind=`** | `to_pandas(name, kind)`, `to_dataarray(name, kind)` and `to_dataset(*names, kind)` read `primal`, `dual` or `expression`, `primal` by default. One kind per call, so a dataset of every dual has no variable of the same name to collide with |
| **`to_parquet` writes every kind** | `primal/<name>.parquet`, `dual/<name>.parquet`, `expression/<name>.parquet`. The kind directory keeps a constraint's name apart from a variable's. A dual an integer variable made undefined, and an expression this data cannot evaluate, are left out; `dual` and `expression` still say why |

**Nothing has to be released.** The built model is frames this process owns,
so `primal` and the `to_*` readers stay valid for as long as the `Result`
does. `close()` and the context-manager protocol hand a large model back
early.

## Writing a file instead of solving

```python
lps.write('spec.yaml', sources, 'model.lp')
```

**The suffix picks the writer**: `.lp` or `.mps`. Anything else is a
`ValueError` listing what can be written. The suffix is checked before the
build, so a format nothing can write costs no model.

**The two formats describe one model**, and name their columns and rows the
same way. LP is the one a person diffs; MPS is the one a decade-old toolchain
accepts.

## Re-solving with new numbers

`update` puts new data on a model that is already built. A loop that solves
the same math over and over pays for the YAML, the plan and the build once:

```python
model = lps.build('sub.yaml', sources)
for capacity in search:
    result = model.update({'cap_hat': capacity}).solve()
    price = result.dual('capacity')
```

| | |
|---|---|
| **it names what changed** | everything else keeps what `build` attached. A change is a parameter, or a dimension index under its own key; a coordinate set grows by handing over a longer table |
| **the answer is the reference build's** | `model.update(x)` solves what `build(spec, sources \| x)` solves, always |
| **it never refuses** | there is no capability to query and no shape of data it rejects. New values can cost the *fast path*, never the answer |
| **the solver stays loaded where it can** | new bounds, costs and right-hand sides go onto the model the solver already holds, so the matrix is never handed over twice. Whether the next solve also carries on from the *work* the last one did is [`keep=`](#how-much-of-the-session-a-solve-keeps). An update that moves a *mask* (a parameter a `where` compares against) renumbers labels, so that model is loaded again and keeps nothing |
| **earlier results keep reading** | a `Result` owns its values and the label frames of the build it answered, so an old answer stays an answer over its own coordinates. Retaining one keeps those frames alive until it is dropped or closed |
| **an update that raises releases the model** | the same rule as `build`. Half a model would answer the next `solve` with a mixture of two |
| **a name the spec does not declare raises `DataError`** | an update that named nothing would silently re-solve the numbers already attached |

**For a sweep, a rolling horizon or a myopic pathway, [`solve_over`](sweeps.md)
is this loop written for you.** `update` is the primitive underneath, for when
the next set of numbers depends on the last answer. Where the next set depends
on *you*, [Change a model](../interactive.ipynb) is the notebook loop.

### How much of the session a solve keeps

A session holds two things: the solver with the model on it, and the work that
solver did. An update keeps the first, so a second solve never hands the matrix
over again. `keep=` says whether it keeps the second. The two can only be
dropped in that order: there is no carrying on from a solver that was closed.

```python
result = model.update({'load': load}).solve()
result.kept  # 'solver' — reused, and the work it did discarded

again = model.update({'load': more}).solve(keep='progress')
again.kept  # 'progress' — it carried on from where the last solve got to

baseline = model.solve(keep='nothing')  # whatever the session held, gone
baseline.kept  # 'nothing'
```

| | What it asks for | Ask for it when |
|---|---|---|
| `keep='nothing'` | the model handed over again, into a solver that has never seen it; `diagnostics().loads` ticks with it | you are **measuring**. The held solver is discarded *before* the load, so cold is structural rather than scrubbed: no basis, no incumbent, no solver-internal state. A benchmark needs that, and so does comparing two sets of `solver_options`, so the first run cannot flatter the second |
| `keep='solver'` *(default)* | the hand-off skipped, and the solver asked to run as though the model were new | **until you have measured otherwise.** The solver gets the run it would have had on a fresh load, without paying for the load. Every ordinary update loop wants this and nothing else |
| `keep='progress'` | that, and the solver left holding what its last run reached | the model is **hard for its solver's preprocessing** *and* consecutive solves differ by a small step: a rolling horizon, a myopic pathway, a search that inches |

**`keep='progress'` can lose by an order of magnitude and win by a factor of
two.** Over six updates on HiGHS
([#815](https://github.com/fluxopt/lpspec/pull/815)), carrying the solver's
work cost **76.6 s against 4.3 s** on a dispatch model whose presolve cracks
the problem outright, an 18× loss. On a storage model whose cyclic recurrence
presolve cannot crack, it cost **111.2 s against 213.9 s**, a 1.9× win.

**Measure which one your model wants.** Run the loop each way and read the
clock the package already keeps. `kept` confirms the request was honoured
rather than quietly downgraded:

```python
for keep in ('solver', 'progress'):
    model = lps.build('spec.yaml', sources)
    for numbers in walk:
        assert model.update(numbers).solve(keep=keep).kept in {keep, 'nothing'}
    print(keep, model.diagnostics().timings['solve'])
```

Take the faster one. **The answer does not change either way.** Across both
models above the objectives agreed to 2e-15 relative, so this is a timing
question only.

**`result.kept` reports what happened, not what was asked.** An update that
had to rebuild reports `'nothing'`, whatever it asked for. `'nothing'` on every
iteration means the session is being rebuilt away, and `loads` ticks on
exactly those solves.

**What progress is made of stays the solver's business.** A basis, an
incumbent or the solver's own notion: `kept` says how much was kept, not what
it was. No solver option reaches the same thing; on both solvers that ship, an
option asking for it did not produce it
([#815](https://github.com/fluxopt/lpspec/pull/815)).

**A rebuild carries no progress.** A cutting-plane master re-solved after
gaining a cut has gained a *row*, and a basis spans the model it was read
from. [#382](https://github.com/fluxopt/lpspec/issues/382) tracks that case.

## Archiving a model

```python
lps.pack('spec.yaml', sources, 'model.zip')
result = lps.solve(*lps.unpack('model.zip', 'model/'))

spec, paths = lps.unpack('model.zip', 'model/')
frames = {name: pl.read_parquet(path) for name, path in paths.items()}  # in memory, when you want them
```

**`pack` writes a model as one zip**: `model.yaml`, and
`sources/<key>.parquet` for every key the file declares. The sources go in
through the same door `build` reads them, so a model `build` refuses is refused
here and nothing is written. A parquet path is copied as its own bytes. A
table is written as parquet. A bare label range, a `{label: value}` map or a
single number is written as the tidy table it stands for. Parquet rather than
text keeps the dtypes [the contract](data.md) checks; `datetime` labels and an
`int` column would not survive JSON. Members are stored uncompressed.

**`unpack` extracts the archive into a directory** and returns the `Spec` and
a `{key: Path}`: the parquet files where they now are, so attaching streams
them from disk and holds nothing here. They are checked where they attach, so
an archive edited by hand gets the same sentence any other source would.
Anything in the zip outside that layout is refused as not an archive `pack`
wrote, and nothing is extracted.

## Diagnostics

`model.diagnostics()` reports what a build and its solves did that the answer
does not show. **Every field is advisory.** Nothing about an answer depends on
any of them.

| Field | |
|---|---|
| `columns`, `rows`, `nonzeros` | the shape the build produced. `check` cannot answer this, having no data. A broadcast that multiplied rows shows up here first |
| `sink_columns`, `sink_rows` | what the last solve's solver had to *add* to that shape. Zero unless it had no concept of a set the spec declares; then it is the binaries and linking rows it was handed instead |
| `omissions` | rows a constraint declared but did not build ([absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/#a-row-with-no-variable-terms-is-not-built)) |
| `sparse_parameters` | `(parameter, coordinates, rows, missing)`, one row per parameter whose source is short of the coordinates its dimensions reach; empty where every one is complete. Sparsity is how a model masks, so this reports rather than judges. A table that lost a row and a `where:` that removed one build the same model, and nothing else says which parameters could be either |
| `coefficient_range` | `(constraint, smallest, largest)`, the coefficient **magnitudes** each block put in the matrix. A solver prints one range for the whole model; this says which declaration holds the outlier. `largest / smallest` over the frame is the conditioning to compare against the solver's own |
| `bound_range` | `(variable, smallest, largest)`, the **bound** magnitudes each variable block put on its columns. A solver reports this axis and does not repair it: HiGHS equilibrates the matrix by itself and answers the bounds with `Consider scaling the bounds by …`, so a model can be clean on `coefficient_range` and still be the one it complains about. Zero and infinity are excluded, since a `lower: 0` and an unbounded side are nothing the solver represents; that is what makes the pair comparable with the line HiGHS prints. A large `largest` is usually a big number standing in for "uncapped", and wants no upper bound rather than a rounder one |
| `rhs_range` | `(constraint, smallest, largest)`, the same for each block's right-hand sides, over the rows that survived. The fourth of the four ranges a solver prints |
| `objective_range` | the same pair for the costs, or `None` where the spec declares no objective. It sits beside the frame rather than in it: badly scaled costs and a badly scaled matrix are different faults with different repairs |
| `solves`, `loads` | how many solves ran, and how many of them loaded the model from scratch. A driver on the fast path leaves `loads` at one however many times it goes round. `loads == solves` is the difference between "lpspec is slow" and "this model masks on a parameter that varies" |
| `timings` | cumulative wall seconds per phase: `attach`, `build`, `handoff`, `solve`, `write` |

**`diagnostics()` answers after `close()` too.** Every field is a count, a
clock or a small frame the model keeps rather than a read of what it releases.
A sweep's diagnostics are `runs.diagnostics`, the counts and clocks one row per
slice ([sweeps](sweeps.md#reading-a-sweep)).

## Choosing a solver

**The caller chooses the solver, not the file.** `solver_name` is `highs`
(ships with the package), `gurobi` (the `[gurobi]` extra) or `xpress` (the
`[xpress]` extra). Nothing in the YAML names one: the same file means the same
model whichever solver takes it. A name outside the three is an error listing
them, never a quiet fallback.

**Options travel in the chosen solver's own vocabulary**, forwarded verbatim.
A time limit is three different words:

```python
lps.solve('spec.yaml', sources, solver_options={'time_limit': 60})
lps.solve('spec.yaml', sources, solver_name='gurobi', solver_options={'TimeLimit': 60})
lps.solve('spec.yaml', sources, solver_name='xpress', solver_options={'timelimit': 60})
```

**Gurobi's remote and licensing options travel the same way**, so Compute
Server, Instant Cloud and WLS need nothing from this package:

```python
options = {'ComputeServer': 'srv:61000', 'ServerPassword': '…'}
lps.solve('spec.yaml', sources, solver_name='gurobi', solver_options=options)
```

The options are applied when Gurobi's environment is created, which
`ComputeServer`, `TokenServer` and `WLSAccessID` require.

## The linopy lane

A *lane* is one of the two ways a spec is executed; the verbs above are the
relational lane. `lpspec.linopy.build` and `lpspec.linopy.expression` (the
`[linopy]` extra) build the same YAML as a `linopy.Model` instead of attaching
it relationally, and read a named expression back off a solved one.
[Relationship to linopy](../about/linopy.md#3-it-is-a-lane) documents them.
