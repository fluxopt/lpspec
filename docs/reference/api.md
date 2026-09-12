# Python API

This page describes what each verb takes, returns, guarantees and refuses, for
anyone who runs a spec from Python. A *spec* is the YAML file; what it may
contain is
[the language](https://math-spec.readthedocs.io/en/latest/reference/language/).

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
tables that carry its numbers. The [glossary](glossary.md) defines *model*,
*result*, *sink* and the other house terms this page uses.

| | |
|---|---|
| `lps.check(spec, sink=None)` | parse, expand, validate and lower; attach no data. With a `sink`, also say whether that sink takes it. Returns the lowered `Program`, for reading the plan — no verb takes one back |
| `math_spec.to_spec(spec)` | the file as written, for editing and typesetting; the language's own verb |
| `lps.build(spec, sources)` | attach data and build; returns a `Model` |
| `lps.solve(spec, sources, solver_name='highs', solver_options=None)` | build and solve in one call; returns a `Result` |
| `lps.solve_over(spec, sources, axis, ...)` | solve once per slice and fold the answers: [sweeps](sweeps.md) |
| `lps.write(spec, sources, out)` | build and stream to a file; the suffix picks the format |
| `archive=` on `lps.solve`, `model.solve`, `lps.solve_over` | write the model, its data and this answer as one zip: [Archiving a model](#archiving-a-model) |
| `lps.load_archive(path, into)` | an archive back as a `SolveArchive`, or a `SweepArchive` where its sources were cut |
| `lps.load_result(directory)` | an answer `result.save(dir)` wrote, back as a `Result` |
| `lps.load_runs(directory)` | a sweep `runs.save(dir)` or `solve_over(spill_to=)` wrote, back as a `Runs` |
| `model.row(name, **coordinate)` | one built constraint row: terms, comparison, right-hand side |
| `math_spec.to_latex` / `to_typst` / `to_markdown` | the math as a document: [typeset](https://math-spec.readthedocs.io/en/latest/reference/typeset/) |
| `lps.Model` / `lps.Result` / `lps.Runs` | the types the verbs hand back, importable so a wrapper can annotate its signature. The spec going *in* is `math_spec.Spec` |

## Errors and warnings

**Every error is one tree, rooted at `LpspecError`.** `LanguageError` (with
`SchemaError`, `DimensionError`, `PiecewiseExpansionError`) is a fault in the
spec. `DataError` is a fault in the data attached to it. `LayoutError` is a
directory or an archive that is not a layout this package reads. `LaneError`
is a spec one lane cannot build. `NoSolutionError` is a solve that left
nothing to read.
Which one you get:
[errors](https://math-spec.readthedocs.io/en/latest/reference/language/errors/#which-error-you-get).

**`LpspecWarning` is the one warning category**, and carries `check`'s advice.
`warnings.simplefilter('error', lps.LpspecWarning)` makes a spec repository
fail CI on it.

## The spec argument

**Every verb takes the spec as a path, a `str`, a `dict` or a `Spec`**: what
`math_spec.to_program` takes, less the lowered `Program` it returns. So a
framework that emits declarations never writes a temporary file to run them:

```python
spec = {'dimensions': ..., 'variables': ..., 'constraints': ..., 'objective': ...}

lps.solve(spec, sources)  # a dict runs like a file
kept = to_spec(spec)  # ...or read once and keep the document
lps.solve(kept, sources)  # a Spec is not read again

to_spec(spec).to_yaml()  # the review copy — a dict-built spec still gets a file
```

**Keep the `Spec`, not the `Program`.** `lps.check` hands back a lowered
`Program` for reading the plan, and no verb takes one: lowering has no
inverse, so an answer built from one could not name the document it came from
and nothing built from one could be archived. Keeping the `Spec` is also the
faster half — reading a file costs about ten times what lowering it does, and
a `Spec` handed back to a verb is not read again
([#1579](https://github.com/fluxopt/lpspec/pull/1579)).

**A framework emits data, not YAML text, and never merges files.** A generated
spec must be able to show you a file. Hand-written math still starts as one.

**A dict-built spec still gets a file.** `to_dict()` and `to_yaml()` are the
language's, and what they write is
[its page](https://math-spec.readthedocs.io/en/latest/reference/language/reading/#writing-a-spec-back-out).

## The sources argument

`sources` maps each declared name to its data, and a dimension's own key
supplies its labels. What each value may be, and what attaching refuses, is
[the data contract](data.md); the type is `lpspec.lanes.Source`, which every
verb annotates `sources` with.

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
the data. It returns the *program*: the spec lowered to the plan a build reads
its rows off.

### Names that differ only by case

**Two declarations of one namespace whose names differ only by case are
refused**, whichever verb lowers the spec. The language takes them and the
mathematics wants them: `p` beside `P` is power beside rated power. An answer
on disk cannot hold both. Every declaration is written as a file named after
it, and a case-insensitive filesystem folds the two into one. A stock macOS
volume is one, and so is a stock Windows one. The second overwrites the first
and keeps its name, so the surviving name reads back carrying the other's
values.

```
variable 'P' and variable 'p' differ only by case, and one answer on disk
cannot hold both: ... Tell them apart by a suffix rather than a capital:
'p_rated' beside 'p'.
```

The namespaces are the language's own: one flat namespace holding dimensions,
lookups, parameters, variables and named expressions, and constraints beside
it. A constraint may carry a variable's name already, so a constraint `P`
beside a variable `p` is accepted. The two are written under `dual/` and
`primal/`, which nothing folds together.

Refused at every door and not only where the archive is written, so a solve
worth archiving is not found to be unarchivable after it has run. Both lanes lower through the
same function, so neither accepts a file the other refuses.

### Checking against a sink

Whether a spec is *sayable* does not depend on the solver. Where it can *land*
is
[a separate question](https://math-spec.readthedocs.io/en/latest/about/limits/#solver-capability),
and `sink=` asks it:

```python
lps.check('spec.yaml')  # sayable?
lps.check('spec.yaml', sink='highs')  # ...and will HiGHS take it?
lps.check('spec.yaml', sink='.lp')  # ...will the LP writer?
```

`sink` is a solver name (`highs`, `gurobi`) or an output suffix (`.lp`). **It
is optional and silent by default.** With a sink named, you get back one of:

- **A refusal (`LpspecError`)** if the sink has no such concept, or refuses the
  combination. The message names the construct, the sink, and the sinks that
  do take it. Only Gurobi and the LP writer take a quadratic row, and HiGHS
  refuses a quadratic objective *beside* integrality while taking either
  alone.
- **A warning** if the sink takes it only by rewriting. `sos:` on HiGHS is the
  one case: the set arrives as binaries, so a spec that declared no
  integrality comes back mixed-integer and without duals.

**`check` answers off a declared table, with no data and no installed
solver.** `check(m, sink='gurobi')` answers on a machine that has never had
gurobipy.

**`solve` and `write` read the same table**, so a refusal comes whether or not
you asked. `lps.write(m, sources, 'model.mps')` on a model carrying a
quadratic term is refused by name rather than written with its quadratic rows
missing.

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
  integrality.
- **The `lp_file` column says what can be written, not what reads back.** The
  same HiGHS parser takes the quadratic-objective section and refuses the
  `sos` and quadratic-constraint sections.
- **"No path here" describes this package, not Xpress.** The Optimizer takes
  a Hessian; the sink in `solvers/xpress.py` never hands it one.

## Building a model

`lps.build` returns a `Model`: the math with your data on it. Build once when
one model feeds more than one sink, or is solved more than once:

```python
model = lps.build('spec.yaml', sources)
model.write('model.lp')
result = model.solve()
model.diagnostics()  # what the build and its solves did that the answer does not show
model.row('balance', snapshot=17)  # what one row actually says
```

**Questions about the model are `build`'s, not `solve`'s.** How big the model
is, what it did not build, what one row says and how its re-solves went are
the `Model`'s to answer.

### Reading one row

`row` says what one constraint says at one *coordinate*, once the data is on
it. `to_latex` renders the model before any data, and `result.dual('balance')`
gives a row's number without its terms; `row` is the third question, and the
one a wrong model is debugged by.

```python
print(model.row('balance', snapshot=1))
# balance[snapshot=1]: +1 p[1, wind] +50 p[1, gas] +30 p[1, coal] >= 60
```

**The line is linopy's format**, as `Constraint.print()` renders it, with the
row's identity on the same line where linopy prints a header.

The same content is a table, for a row too wide to read and for anything that
filters or joins:

```python
row = model.row('balance', snapshot=17)
row.terms  # (variable, coordinate, coefficient), one row per term
row.sense  # '=='
row.rhs  # 80.0
```

**A row too wide to spell out is summarised, not truncated**:

```python
print(model.row('balance', t=0))
# balance[t=0]: 301 terms — p: 300 (|coef| 0.001…0.3), slack: 1 (|coef| 1000) >= 5
```

The line says how much of the row each declaration contributes, and whether
its coefficients span an order of magnitude. `diagnostics().coefficient_range`
reports that spread per *declaration*; nothing reports it per row.
`display_terms` sets where a line stops spelling terms out.

**`row` reads the built row.**

- A coefficient is the number the *data* produced, every digit of it.
- A term whose variable a `where` masked out is **not there**.
- A term whose coefficient the data made **exactly zero** is not there
  either: the build prunes it.
- A row a `where` removed raises, and the message names the three things that
  cause it.

**`row` needs no solve.**

**The coordinate names every dimension of the declaration.** A partial one
names a set of rows rather than one. The constraint is positional, so a
dimension may be called `name` and still be named in the coordinate. A label
the dimension cannot hold (a string against an integer dimension, a stranger
against a declared label set) is refused naming the dimension, not the dtypes.

**There is no verb for a column.** A variable's bounds are in `to_yaml()`; its
coefficients are the transpose of `row`, which nothing exposes.

## Reading a result

```python
result.status, result.termination_condition, result.objective
result.spec_digest  # a digest of the spec this answered — None off a lowered Program
result.is_ok  # rolled-up verdict: not an error, abort or refusal
result.has_primal  # narrower: are there values to read
result.kept  # how much of the session this solve kept: 'nothing', 'solver' or 'progress'

result.primal('p')  # tidy table (dims…, value) in label order — the native shape
result.dual('power_balance')  # shadow prices, same shape, same join
result.activity('power_balance')  # each row's left-hand side at the solution
result.expression('co2')  # a named expression at the solution, over its own dims

result.to_pandas('p')  # the same, as a DataFrame
result.to_dataarray('p')  # the same, labelled: .sel / resample / plot
result.to_dataarray('power_balance', 'dual')  # a price, labelled — every bridge takes kind=
result.to_dataset()  # every variable by default; names for a subset
result.to_dataset(kind='dual')  # every dual; one kind per dataset
result.save(
    directory
)  # the whole answer to disk: objective.parquet, primal/ dual/ activity/ expression/, reasons.parquet
lps.load_result(directory)  # and back, every reader answering what it answered
```

**`primal` returns a `polars.DataFrame`**, one row per coordinate: a *frame*.
It is Arrow-backed, so it exports the protocol the loader recognises.
`to_pandas` and `to_dataarray` are the bridges out; they need pandas and
xarray, from the `[linopy]` extra.

| Rule | |
|---|---|
| **`is_ok` is not `has_primal`** | `is_ok` rolls up the termination condition. `has_primal` adds the solver's verdict on whether an incumbent exists, and every reader gates on it. A MIP that hits `time_limit` before a feasible point is `ok` with nothing to read |
| **reading with no primal raises** | `NoSolutionError`; `objective` is `nan`. `save` is the exception: it writes the record and no frames, an infeasible run being an answer a set of saved cases needs on disk |
| **`expression` takes a declared name** | the value of a [named expression](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#named-expressions) at the solution, aggregated to its own dimensions; never an expression string. An unknown name is a `KeyError` listing what is declared. It is compiled at the read, so unread expressions cost nothing |
| **`dual` raises rather than zero-filling** | no values at all is `NoSolutionError`; values but no duals is `LpspecError`. Any integer or binary variable makes duals undefined |
| **a solver can make a model mixed-integer** | an [`sos:`](https://math-spec.readthedocs.io/en/latest/reference/language/piecewise/#sos) set reaches a solver with no SOS concept as binaries, so an otherwise continuous model solved on `highs` has no duals and says so. `gurobi` and `xpress` branch on the set itself and keep them |
| **duals exist only where a solver ran** | a model written to LP and solved elsewhere never passes back through here. Reduced costs and slacks are not exposed |
| **`to_dataset` costs what it says** | each variable arrives dense over its own dimensions. Name a subset, or use `save` |
| **every bridge takes `kind=`** | `to_pandas(name, kind)`, `to_dataarray(name, kind)` and `to_dataset(*names, kind)` read `primal`, `dual` or `expression`, `primal` by default. One kind per call |
| **`save` writes the whole answer** | `objective.parquet` says how the solve terminated — `status`, `termination_condition`, `objective`, `has_primal`, `spec_digest` — in the columns a sweep keys per slice, so cases solved apart concatenate. A solve that reached no objective writes null there rather than `nan`, so a mean over a set of cases is the mean over the ones that solved. Then `primal/<name>.parquet`, `dual/<name>.parquet`, `activity/<name>.parquet` and `expression/<name>.parquet`. A dual an integer variable made undefined, and an expression this data cannot evaluate, are left out, and `reasons.parquet` says why |
| **`load_result` reads it back whole** | every reader answers what it answered, and an absence raises the sentence the solve gave. Two session facts do not survive: `kept` reads `nothing`, and a refusal carries the termination condition rather than the solver's verbatim wording. The frames are read lazily, so the directory has to outlive the result |

**Nothing has to be released.** `primal` and the `to_*` readers stay valid for
as long as the `Result` does. `close()` and the context-manager protocol hand a
large model back early.

## Writing a file instead of solving

```python
lps.write('spec.yaml', sources, 'model.lp')
```

**The suffix picks the writer**: `.lp` or `.mps`. Anything else is a
`ValueError` listing what can be written, raised before the build.

**The two formats describe one model**, and name their columns and rows the
same way. LP is the one a person diffs; MPS is the one a decade-old toolchain
accepts.

## Re-solving with new numbers

`update` puts new data on a model that is already built, so a loop over the
same math pays for the YAML, the plan and the build once:

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
| **the solver stays loaded where it can** | new bounds, costs and right-hand sides go onto the model the solver already holds. Whether the next solve also carries on from the *work* the last one did is [`keep=`](#how-much-of-the-session-a-solve-keeps). An update that moves a *mask* (a parameter a `where` compares against) renumbers labels, so that model is loaded again and keeps nothing |
| **earlier results keep reading** | a `Result` owns its values and the label tables of the build it answered. Retaining one keeps those tables alive until it is dropped or closed |
| **an update that raises releases the model** | the same rule as `build` |
| **a name the spec does not declare raises `DataError`** | an update that named nothing would silently re-solve the numbers already attached |

**For a sweep, a rolling horizon or a myopic pathway, [`solve_over`](sweeps.md)
is this loop written for you.** `update` is the primitive underneath, for when
the next set of numbers depends on the last answer. Where it depends on *you*,
[Change a model](../interactive.ipynb) is the notebook loop.

### How much of the session a solve keeps

A session holds two things: the solver with the model on it, and the work that
solver did. An update keeps the first. `keep=` says whether it keeps the
second. The two can only be dropped in that order.

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
| `keep='nothing'` | the model handed over again, into a solver that has never seen it; `diagnostics().loads` ticks with it | you are **measuring**. The held solver is discarded *before* the load, so cold is structural: no basis, no incumbent, no solver-internal state. A benchmark needs that, and so does comparing two sets of `solver_options` |
| `keep='solver'` *(default)* | the hand-off skipped, and the solver asked to run as though the model were new | **until you have measured otherwise.** Every ordinary update loop wants this and nothing else |
| `keep='progress'` | that, and the solver left holding what its last run reached | the model is **hard for its solver's preprocessing** *and* consecutive solves differ by a small step: a rolling horizon, a myopic pathway, a search that inches |

**`keep='progress'` can lose by an order of magnitude and win by a factor of
two.** Over six updates on HiGHS
([#815](https://github.com/fluxopt/lpspec/pull/815)), carrying the solver's
work cost **76.6 s against 4.3 s** on a dispatch model whose presolve cracks
the problem outright, an 18× loss, and **111.2 s against 213.9 s** on a
storage model whose cyclic recurrence presolve cannot crack, a 1.9× win.

**Which one a model wants is measured**
([timing a loop](../howto/debug.md#6-when-a-loop-of-re-solves-is-slow)).
**The answer does not change either way**: across both models above the
objectives agreed to 2e-15 relative.

**`result.kept` reports what happened, not what was asked.** An update that
had to rebuild reports `'nothing'`, whatever it asked for, and `loads` ticks on
exactly those solves. `'nothing'` on every iteration means the session is
being rebuilt away.

**What progress is made of stays the solver's business.** `kept` says how much
was kept, not what it was. No solver option reaches the same thing; on both
solvers that ship, an option asking for it did not produce it
([#815](https://github.com/fluxopt/lpspec/pull/815)).

**A rebuild carries no progress.** A cutting-plane master re-solved after
gaining a cut has gained a *row*, and a basis spans the model it was read
from. [#382](https://github.com/fluxopt/lpspec/issues/382) tracks that case.

## Archiving a model

```python
lps.solve('spec.yaml', sources, archive='case.zip')

case = lps.load_archive('case.zip', 'case/')
case.answer.primal('p')  # what came back
lps.solve(case.spec, case.sources)  # the same question, asked again
```

**An archive is the model, its data and its answer**: `model.yaml`,
`sources/<key>.parquet` for every key the file declares, `sources.parquet`
digesting those members, `answer/` holding what `result.save` or `runs.save`
writes, and `axis.json` for a sweep. Beside the answer is
`answer/diagnostics.parquet`, one row saying what the build and its solves
spent, which the verb writes rather than `save`.

**The suffix decides the container**, as `lps.write`'s does. `.zip` packs those
members into one file, to send or to store; anything else lays them out in a
directory. The two hold the same thing, and only reading them differs:

```python
lps.solve('spec.yaml', sources, archive='case/')  # a directory
lps.load_archive('case/')  # read where it lies — no into=
```

**A directory archive needs no `into`, and a zip requires one.** Nothing is read
at load: the sources come back as paths and every frame is a `scan_parquet`, so
the parquet files have to be on disk. A directory's already are. A zip's are
not, and only you know somewhere writable — an archive often lives where it is
only read — so there is no default, and passing none is refused by name.

**Every verb that solves takes `archive=`, and nothing else writes one.**
`lps.solve`, `model.solve` and `lps.solve_over` each hold the model, the data
and the answer at the moment they are asked for, so the three are written
together and cannot be paired up wrongly. There is no way to assemble them
afterwards: an answer records the spec it came back from and not the data it
was solved over, so nothing in a hand-assembled archive could show that its
answer is the one those sources produce. The digests say which data an archive
*holds*, which is a different claim.

**A sweep's archive carries its axis**, as `axis.json`, because its sources
are cut: they hold the column the axis slices on, which the model does not
declare. `spill_to=` and `archive=` are different destinations and compose —
the spill is what the archive packs.

The recipes are [archiving a solve](../howto/archiving.md): keeping the answer
an update produced, archiving a sweep too large to hold, comparing cases solved
apart, and querying an archive from a database.

The sources go in through the same door that reads them, so what is refused
there is refused here and nothing is written: `build`'s for one solve, and for
a sweep the door `solve_over` uses, which is one slice of them. A parquet path is copied as its
own bytes; a table, a bare label range, a `{label: value}` map or a single
number is written as the tidy parquet table it stands for. Parquet keeps the
dtypes [the contract](data.md) checks. Members are stored uncompressed.

**`load_archive` reads the answer lazily**, so the files it reads off have to
outlive it: the archive itself for a directory, the `into` directory for a
zip. Its `sources` come back as the parquet paths they now are — the same type
they went in as, `Path` being a source like any other — so attaching streams
them from disk. Anything outside the layout is refused, and a zip is refused
before it is unpacked.

**An archive is a parquet tree.** Every frame is tidy: the model's own
dimension columns, and a `value` column. An answer therefore joins to the
sources it was solved from, on the coordinates both carry.

```sql
-- generation priced by the load it met, answer joined to source
select p.scenario, p.snapshot, p.generator, p.value, load.value as load
from 'study/answer/primal/p/*.parquet' p
join 'study/sources/load.parquet' load using (scenario, snapshot);
```

A sweep keys every file it writes with one column of one type. The files under
a kind are one table, and the kinds join to each other on that key. What a file
holds is named by its path, not by a column. Read a kind with a glob, and add
the engine's own filename column where the declaration has to travel with the
rows.

| Rule | |
|---|---|
| **the model is held as written** | `model.yaml` is what the file said, so `archive.spec` reads back as one `Spec` whatever went in. A lowered `Program` is refused: it has no file to write |
| **a saved answer is stamped with its layout** | `format.json` beside the frames, `0` while the layout is still moving and counting from `1` the day it settles. Nothing reads an older layout back, so the stamp turns a missing column into a sentence: solve the model again and save it. An archive still holds the model and the data to do that with |
| **`spec_digest` says whether a comparison compares like with like** | a digest of the spec every answer carries, written into the record and checked when an archive is read back. Concatenate the records of cases solved apart and one distinct `spec_digest` is the claim that they answered the same document; an archive whose answer names another model is refused rather than read. A solve run off a lowered `Program` has no document and carries `None`, which counts as its own value — so one null among real digests breaks the comparison, and a table where *every* digest is null counts one distinct value while having checked nothing. Ask for the digests to be present as well as to agree: `n_unique() == 1 and null_count() == 0` |
| **the sources are digested, one row each** | `archive.source_digests` is `(source, digest)` for every member of `sources/`, held as `sources.parquet` beside that directory — inside it, a table about the sources would be read as one of them. It answers what `spec_digest` cannot: two archives of one document over different numbers agree on the spec digest and differ here, and the rows that differ name the input that moved. The digest is of the bytes the archive holds, so a reader can recompute it from the archive alone; two archives of the same data written by different polars versions can still differ, parquet being what is hashed rather than the table's meaning. Reading an archive does not verify them — that is a pass over every byte it holds, and it is the caller's to ask for |
| **the cost row is written by the solve, not by `save`** | `archive.diagnostics` is one row of `model.diagnostics()`'s sizes, counters and clocks, and `answer/diagnostics.parquet` is where it sits. A `Result` is one solve and those counters are the model's whole life, so a result has no share of them to carry and `result.save` writes none; the verb that archives holds the model and can. `solves` says how many solves the clocks cover — `1` for `lps.solve`, which builds the model it solves. A phase that never ran writes zero, so cases that entered different phases are one table. A sweep's are `answer.diagnostics` instead, one row per slice, a fold knowing each slice's share |
| **the two are separate types because the axis is not optional** | a sweep's sources carry the column the axis cuts on, which the model does not declare, so they are legible only beside it. A `SweepArchive` has it and a `SolveArchive` has no such field, so nothing downstream meets `Result \| Runs`. `load_archive` returns whichever the archive holds |
| **a sliced source is archived whole** | one copy carrying every slice's rows, not one copy per slice. What the check sees is one slice of them, which is what the model is built from |
| **a hand-built axis is refused** | a list of `(key, sources)` is a set of sources per slice, which are unrelated questions. Archive one solve each. Refused before the first slice is taken, as a lowered `Program` is |
| **the model's own fitness for slicing stays `solve_over`'s** | whether a window can carry this model's coupling and reach is asked when the sweep is run, not when it is archived |
| **a sweep's answer reads back spilled** | its frames stay in the extracted directory and `runs.scan(name)` reads them, which is what `solve_over(spill_to=)` already produces. `original_index` works: the dimension a window sliced and the coordinates each owns are in the manifest |

## Diagnostics

`model.diagnostics()` reports what a build and its solves did that the answer
does not show. **Every field is advisory.** Nothing about an answer depends on
any of them.

| Field | |
|---|---|
| `columns`, `rows`, `nonzeros` | the shape the build produced; `check` cannot answer this, having no data |
| `sink_columns`, `sink_rows` | what the last solve's solver *added* to that shape: zero, or the binaries and linking rows that replaced a set it has no concept of |
| `omissions` | rows a constraint declared but did not build ([absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/#a-row-with-no-variable-terms-is-not-built)) |
| `sparse_parameters` | `(parameter, coordinates, rows, missing)`, one row per parameter whose source is short of the coordinates its dimensions reach. Sparsity is how a model masks, so this reports rather than judges: a table that lost a row and a `where:` that removed one build the same model, and nothing else says which |
| `coefficient_range` | `(constraint, smallest, largest)`, the coefficient **magnitudes** each block put in the matrix. `largest / smallest` over the table is the conditioning to compare against the solver's own |
| `bound_range` | `(variable, smallest, largest)`, the **bound** magnitudes each variable block put on its columns, zero and infinity excluded. HiGHS reports this axis (`Consider scaling the bounds by …`) and does not repair it. A large `largest` is usually a big number standing in for "uncapped", and wants no upper bound rather than a rounder one |
| `rhs_range` | `(constraint, smallest, largest)`, the same for each block's right-hand sides, over the rows that survived |
| `objective_range` | the same pair for the costs, or `None` where the spec declares no objective |
| `solves`, `loads` | how many solves ran, and how many of them loaded the model from scratch. `loads == solves` means the model masks on a parameter that varies |
| `timings` | cumulative wall seconds per phase: `attach`, `build`, `handoff`, `solve`, `write` |

**`diagnostics()` answers after `close()` too.** A sweep's diagnostics are
`runs.diagnostics`, one row per slice ([sweeps](sweeps.md#reading-a-sweep)).

**`as_row()` is the scalars as one row**, in the columns every writer of one
uses: the sizes, `solves`, `loads`, and a clock per phase. The frames are not
in it. A range is a table per declaration, which does not fold into a row
beside a count. This is what `archive=` records, and what a caller feeding its
own store reads off a model it solved.

## Choosing a solver

**The caller chooses the solver, not the file.** `solver_name` is `highs`
(ships with the package), `gurobi` (the `[gurobi]` extra) or `xpress` (the
`[xpress]` extra). Nothing in the YAML names one. A name outside the three is
an error listing them, never a quiet fallback.

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
`[linopy]` extra) build the same YAML as a `linopy.Model`, and read a named
expression back off a solved one.
[Relationship to linopy](../about/linopy.md#3-it-is-a-lane) documents them.
