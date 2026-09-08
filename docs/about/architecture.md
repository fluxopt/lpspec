# Architecture

Brief, current, precise. A PR that changes the structure described here updates
this file in the same PR. The language is [the language reference](https://math-spec.readthedocs.io/en/latest/reference/language/); what may
enter it is [the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/); plans and refusals
are [the roadmap](roadmap.md); measured results are
[the benchmarks](benchmarks.md), produced by the harness in
[bench/](https://github.com/fluxopt/lpspec/blob/main/bench/README.md) — which is
also how a claim here gets falsified.

`python examples/walkthrough.py` executes the pipeline below stage by stage
and prints what each one produces — the same public calls `lps.solve` makes,
so the demonstration cannot drift from the code. Its output is committed as
[examples/walkthrough.out](https://github.com/fluxopt/lpspec/blob/main/examples/walkthrough.out) and asserted line for line
(`tests/test_walkthrough.py`), so reading it is the same as running it — and a
stage that starts telling a different story shows up as a diff in that file.

## Thesis

A YAML math spec is a **closed AST known before any data is touched**. That one
property makes everything else legal: the whole model can be compiled — to a
logical plan executed relationally and streamed to a sink, or, in another
package, to eager xarray/linopy calls — with both paths provably meaning the
same thing. Every rule below protects
it. (A *declared* memory ceiling is not something we have; see
[the memory axis](roadmap.md#where-it-is-going).)

**The producer of the AST is a different package.** `math_spec` parses,
expands, resolves and judges a file, and this repository consumes what comes
out — so the widest fence in the drawing is not a directory rule at all, it is
`pyproject.toml`, and it is the amber box below: everything in it, the
typesetter included, is that one package, which depends on nothing here and
cannot import anything here. **Its passes are named in the box and not drawn**:
how a file becomes an AST is math-spec's architecture, documented and tested
there, and a second copy of it here would be one more thing to drift. What
crosses is the waist, and the waist is the whole of what this drawing needs of
it.

**The waist has a second consumer, and it is not in this drawing either.**
linopy builds a `linopy.Model` from the same `Program`
([PyPSA/linopy#922](https://github.com/PyPSA/linopy/pull/922)), which is what
makes "one contract, many consumers" a fact about the language rather than a
description of this repository — and it is what the differential suite measures
this engine against. There used to be a second lane in here doing that job; it
was deleted in favour of the package that owns it
([linopy.md](linopy.md#3-it-is-the-other-consumer-of-the-language)).

**The dashed box is outside every fence, and that is the point.**
`sources.py` is the seam: it turns a caller's tables into the frames a plan is
executed against, and it reads the schema to do it. Drawing it inside
`relational/` would be a lie about the fence — the engine imports nothing from
the package, while this does.

**Data enters below the seam through one door.** `sources.tidy_sources` reads
every shape a caller may pass — a frame of any library, a dict, a sequence, a
bare number, a parquet path — into tidy polars frames, and the engine executes
its plan against those frames directly. Polars is therefore the one
representation, and pandas a bridge at the edge, on the way *out* — which is
also what the dependency set says, pandas being declared with `[linopy]` rather
than as a runtime dependency.

**One reader, because two disagreed.** Two readers under one roof used to read
the caller's object each in its own library, and the same instant then had two
spellings — a `datetime.date` out of pandas, a `pl.Date` out of polars —
costing a reconciling guard at every place the two met, one of which was always
missing. A conversion cannot disagree with itself: past `tidy_sources` nothing
about the library a caller reached for survives (#1076). linopy has its own
reader and its own answer to the same question, which is now something the
differential measures rather than something this file decides.

The `method: convex` curvature guard sits below the seam for the neighbouring
reason — it needs values rather than a schema — in `curves.py`, which the door
calls so that nothing can enter without it. What matters for the waist is
the direction: data goes no further **up** than here, so nothing above the seam
has ever seen a value — which is what makes `show it` and `check it` free.

```mermaid
flowchart TB
    Y[YAML file] --> SPEC
    DATA[("your data<br/>parquet · polars · any Arrow table")] --> SRC

    subgraph MS["math-spec — another package, pinned in pyproject.toml: read · expand · resolve · judge · lower"]
        SPEC["<b>Spec</b> — what the file says<br/>fully resolved: names typed, dims checked, degree judged"]
        SPEC --> TS["typesetting/<br/>latex · typst · markdown<br/><i>a consumer, not a stage</i>"]
        SPEC -->|"to_program"| PLAN["<b>Program</b> — what it means, the narrow waist<br/>the plan every consumer builds from<br/>closed from both sides"]
    end

    SPEC -->|"the declarations to attach against"| SRC
    SRC["<b>sources.py</b> — flat<br/>data → the tidy frames, by name<br/><i>the one door data enters by</i>"]

    PLAN -->|"outside the plan:<br/>LanguageError naming the construct"| ERR["load error<br/>(no fallback)"]
    PLAN -->|"the plan"| COMP
    SRC --> ATTACH
    PLAN -->|"the same plan, in another package"| LIN

    subgraph REL["relational/ — the streaming lane"]
        direction TB
        subgraph ENG["engines/polars/ — the only part a second engine replaces"]
            direction TB
            COMP["compiler.py<br/>plan → lazy frames · reads nothing"] --> ENGINE
            ATTACH["attaching.py<br/>→ AttachedSources, frozen"] --> ENGINE["assembly.py + labels.py<br/>assemble the model frames"]
        end
        ENG --> TABLES["sinks/tables.py<br/>cols · obj · rows · A · sos"]
        TABLES --> LPS["sinks/writers/<br/>a file, chosen by suffix<br/>lp_file · mps_file"]
        TABLES --> DIRECT["sinks/solvers/<br/>CSR batches → the solver, chosen by name<br/>highs (ships) · gurobi · xpress (extras)"]
        DIRECT --> SOL["result.py<br/>label join, never dense"]
    end

    SOL --> ANS["<b>Result</b> — this lane runs to the answer<br/>objective · primal · dual · activity · expression<br/>polars frames you can join"]

    LIN["<b>linopy</b> — another package<br/>Model.from_spec · linopy.spec.attach<br/><i>its own reader, its own build</i>"]
    DATA --> LIN
    LIN --> MODEL["<b>a linopy.Model</b><br/>yours to solve, read back and typeset, with linopy"]

    classDef laneL fill:#fdf6ec,stroke:#b7791f,stroke-width:2px,color:#111
    classDef laneR fill:#f0f7f0,stroke:#3a7d44,stroke-width:2px,color:#111
    classDef laneE fill:#eef1fb,stroke:#4a5fc1,stroke-width:2px,color:#111
    classDef laneT fill:#f7f0f7,stroke:#8b3a7d,stroke-width:2px,color:#111
    classDef waist fill:#e9edfa,stroke:#4a5fc1,stroke-width:3px,color:#111
    classDef flat fill:#fffdf5,stroke:#8a8578,stroke-width:2px,stroke-dasharray:4 3,color:#111
    classDef data fill:#fdf4e8,stroke:#b7791f,stroke-width:1.5px,color:#111
    classDef out fill:#eef6ee,stroke:#3a7d44,stroke-width:2px,color:#111
    class MS laneL
    class REL laneR
    class LIN laneE
    class TS laneT
    class PLAN waist
    class SRC flat
    class DATA data
    class ANS,MODEL out
```
**The two consumers are peers in what they take, not in what they hand back.**
Both accept the same file and refuse the same constructs — and there the
symmetry ends. `relational/` runs to an answer: it assembles the model frames,
hands them to a sink and reads the solution back as a `Result`. linopy hands
back a `linopy.Model`, which its own API solves, reads and typesets. That is
not a gap waiting to be closed on either side: a caller who asks for a
`linopy.Model` is asking for linopy's API on the far side of it, and a wrapper
here would be one nobody wanted.

Five modules sit outside the fence: the seam drawn above, plus `curves.py`,
the one guard that needs numbers, `api.py`, which runs the lot and declares
what a verb takes, `strategy.py`, which drives it a slice at a time,
`frames.py`, the table boundary, and `errors.py`, the leaf every fence points
at. That is a category, not a leftovers bin. See
[What counts as language](#what-counts-as-language).

Eligibility is decided by **attempting the lowering** — `to_program` returns
a `Program` or raises `lps.LanguageError` — so it cannot drift from what the
engine supports. Both consumers call it, which is what makes "neither accepts a
file the other refuses" mechanical rather than maintained: it is one gate in
one package, and its sentence is the sentence both raise. Errors split model
from run: everything under `LanguageError` is decidable without data,
`DataError` is what a source failed to supply, and both are `LpspecError`
(`errors.py`). `LaneError` is the third thing that can be wrong and the one
hard rule 3 does not forbid — **accepting is not building**, so a model both
accept may still meet a wall inside one of them, and this one says so in its
own words rather than passing an upstream exception through. Expansion precedes
validation, because a formulation emits declarations and those are language too
— a stray dim in generated math is the same error as a stray dim in a written
one.

## One contract, many consumers

The AST is a **narrow waist**. Everything upstream emits it, everything
downstream reads it, and nothing else has to agree on anything — so the model
you write once is the same model that gets checked, solved, typeset and read
back.

```mermaid
flowchart LR
    Y(["your math, written once<br/>one YAML file"]) --> AST
    AST["<b>the whole model</b> — <code>Spec</code>, and the <code>Program</code> it lowers to<br/>names typed, dims checked, degree judged<br/><i>before a byte of data is read</i>"]
    AST --> SHOW["<b>show it</b><br/>math_spec.typesetting · its CLI<br/><i>no data, no solver</i>"]
    AST --> CHECK["<b>check it</b><br/>parse → expand → validate → lower<br/><i>no data, no solver</i>"]
    AST --> RUN["<b>run it</b><br/>solver · LP/MPS file · linopy"]
    DATA[("your data<br/>parquet · polars · any Arrow table")] --> RUN
    RUN --> ANS(["<b>your answers</b><br/>tables you can join"])
    classDef built fill:#eef6ee,stroke:#3a7d44,stroke-width:1.5px,color:#111
    classDef waist fill:#e9edfa,stroke:#4a5fc1,stroke-width:3px,color:#111
    classDef data fill:#fdf4e8,stroke:#b7791f,stroke-width:1.5px,color:#111
    class Y,SHOW,CHECK,RUN,ANS built
    class AST waist
    class DATA data
```

**Only one arrow carries data, and it arrives after the model is already
judged.** That is the contract the waist is: a `Spec` is complete —
names typed, dims checked, degree decided — before a source is attached, so
`show it` and `check it` are not cut-down versions of a build, they are the
same model with the data arrow missing. `check` is the build's own front half
run to completion and stopped before attaching, which is why it is a CI verb,
costs seconds, and needs nothing but the file.

**Each box is a family, and [the table below](#the-python-surface) is the
members of the ones this package answers** — `show it` is answered upstream
now, and that is the same point from the other side: none of them is a
rewrite. Each reads the same AST the engine reads, so a renderer is a
tree walk, a check is a pass with no data attached, and a new output format is one
module in `relational/sinks/writers/`.

**The renderer is that claim cashed, and it is not here.**
`math_spec.typesetting` typesets any model either consumer can build, in one
walk of the resolved AST, holding no opinion they do not already hold: a
`piecewise:` block prints as the λ-formulation it expands to, not as the sugar
it was written as. It lives in the same package as the language, and this one does not
depend on it — which is the strongest form the "a new consumer is free" claim
can take.

That is also the honest test of the waist: a consumer that reads the AST and
nothing else needs no part of this repository to run. What is here is what
genuinely touches data or a plan. Two properties carry the rest — **data enters at
exactly one place**, which is why checking a model costs seconds and needs
nothing but the file, and the waist is **closed**, which is what
[the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/)
protects: a new consumer is free, a new primitive is taxed. What is planned,
and why, is [the roadmap](roadmap.md).

### The Python surface

**Twenty names, and the count is the feature.** The model is the YAML file;
Python is how you *run* it — so the whole surface is the diagram above written
out, with nothing that constructs math and nothing that reaches the plan. Names
are `lpspec.` unless shown otherwise, and what each one *does* is
[the Python API](../reference/api.md). **Data?** is the column that matters: a verb
that says *no* needs nothing but the file, which is what makes it a CI verb.
*Italic rows are the ones the shape makes cheap and nobody has built.*

**Loading a file and rendering one are not on this list.** `to_spec`,
`SymbolTable`, the three `to_…` renderers and the shell
front that runs them are `math_spec.`'s, counted in its own `__all__`, and a
caller that wants them imports that package rather than a re-export here: one
name, one home. What this package
exports is what it does, which is attach, build, solve and read back.

**The errors are the exception, and they are the only one.** A name is
re-exported here when a caller meets it *without choosing to*: a
`LanguageError` arrives unbidden out of `lps.solve`, and no call reaches it.
The language's own nouns are not like that — `check` hands back a `Program`
and every verb accepts a `Spec`, but obtaining either means calling
`math_spec`, so a caller annotating one is already in the package that owns
it. Re-exporting them would be a second home for a name, which is the rule
above.

**Nothing here reads a `Spec`.** Binding, the guards and the engine take the
`Program`; a `Spec` reaching a verb is passed straight to
`math_spec.to_program` and never looked at. The model *as written* — editing
it, dumping it, typesetting it — is `math_spec`'s side of the line.

**What a verb hands back is part of that verb's signature**, which is why
`Model`, `Result` and `Runs` are named here and not only reached off a
call. A caller that *wraps* this package — a framework whose own function
returns a solve — writes the type down, and a type it cannot import is a type
it cannot write. The same argument runs the errors one step further than the
language half: `NoSolutionError` is what every reader on a `Result` raises and
`LpspecWarning` is what `check` emits, so a sweep that records an infeasible
scenario rather than dying on it needs both by name. None of the five
constructs math or reaches the plan — each is what a verb already handed over,
which is the line the count is drawn on.

| | you want to | the call | data? |
|---|---|---|---|
| **check it** | will this build, is the math sayable, do the dims line up | `check` — parse → expand → validate → lower, one pass, every answer | no |
| | *will that solver take it* | | |
| **run it** | stream it straight into a solver | `solve`, or `build` → `Model` to drive several sinks off one build | **yes** |
| | re-solve one built model with new numbers | `model.update(...)` — the label contract, spent | **yes** |
| | how big is it, how is it scaled, what did the build and its solves do, and where did the time go | `model.diagnostics()` → `columns` · `rows` · `nonzeros` · `sink_columns` · `sink_rows` · `omissions` · `coefficient_range` · `bound_range` · `rhs_range` · `objective_range` · `solves` · `loads` · `timings`, all advisory | **yes** |
| | write an LP or MPS file for anything else | `write` | **yes** |
| | solve it once per scenario, window or period | `solve_over` over a `EachCoordinate` / `EachWindow` axis | **yes** |
| | build the same math as a `linopy.Model` | `linopy.Model.from_spec` — [another package's verb](linopy.md#3-it-is-the-other-consumer-of-the-language) | **yes** |
| **read it** | values, shadow prices, the objective | `result.objective` · `.primal` · `.dual`, plus the status pair | — |
| | the quantity the model named | `result.expression(name)` — lowered on demand at the read, never at build; `model.spec.expressions[name]` on linopy's side | — |
| | bridge out to another library | `.to_pandas` · `.to_dataarray` · `.to_parquet` | — |
| | name it in your own signature | `Model` · `Result` · `Runs`, what `build`, `solve` and `solve_over` hand back; `Spec` re-exported for the model as written, and `math_spec.program.Program` for what `check` hands back | — |
| **catch it** | tell a bad model from bad data | `LpspecError` ⊃ `LanguageError` · `DataError` · `DimensionError` · `SchemaError` · `PiecewiseExpansionError` · `LaneError` | — |
| | record an infeasible run instead of dying on it | `NoSolutionError`, raised by every reader on a `Result` | — |
| | fail CI on advice, not just on errors | `LpspecWarning`, what `check` emits | no |

**Flat, and there is no namespace left.** `lpspec.linopy` was the one, and it
earned it by being a different lane with its own dependencies; it is gone, and
what replaced it is another package's verb. `strategy.py` was never a lane, so
`solve_over` and its axes sit at the top level beside `solve`.

That is a rule with teeth rather than a taste: the surface test exempts
submodules (`not inspect.ismodule`), so moving names under `lpspec.something`
moves them out from under the list a reviewer reads. **Grouping trades an
enforced surface for a tidier one**, which is the opposite of what the count is
for.

**A return type is not a name.** `build` returns a `Model`, `solve` a
`Result` and `solve_over` a `Runs`, and none is exported — you reach them by
calling, and import them from their module only to write an annotation. What
the objects themselves carry (`Result` alone has twelve readers) is documented
in [the Python API](../reference/api.md) rather than counted here, which is why capability grows
much faster than this table does.

The discipline that keeps that from being a way to dodge the count: **a handle's
methods answer "what do I do with this", never "what is this"**. `solve`,
`write`, `close` and `update` pass; anything that changed a declaration would be
a language feature wearing a method, and hard rule 5 refuses it wherever it is
spelled. It is also why these are named for what they *are* rather than for what
built them — a second engine must not change a top-level verb's return type.

**What the data arrow carries** is [the data contract](../reference/data.md) and is not
restated here. The one structural fact: **attaching is by name at both levels** —
a mapping keyed by declared parameter, and inside each table, columns named for
that parameter's declared dims. The single positional fallback (an *unnamed*
pandas index) is narrow on purpose, because renaming a named level would
transpose the data silently whenever two dims share a label space.

`tests/test_architecture.py` pins all of it: `__all__` must match the table,
**and** no public non-module attribute may exist outside it. Both directions,
because either alone rots — the first catches a name documented and never
exported, the second a helper that leaked into the namespace by being imported
at the top of `__init__.py`. That check found one the day it was written.

There is deliberately no Python API for *constructing* a model, no way to hand
in a plan, and no registry to populate. That is hard rule 5 below, and it is
what makes a `.yaml` file the thing you review, diff and cite — rather than the
serialisation of a Python object you would have to run to understand.

## Hard rules

*Enforced, not aspirational: `tests/test_architecture.py` encodes these as
static checks and CI's bare-install job proves the dependency claims.*

**These rules constrain the language.** What a construct may say, which layer
may know what, and what a file means on its own — each survives any engine, and
each decides what can enter the language reference. How much a build *costs* is a
property of the engine, measured in [the benchmarks](benchmarks.md), and
deliberately not a rule: a cost phrased as a rule makes one implementation's
choice load-bearing in the language's rulebook.

0. **The layers are ordered, and imports prove it.** Every module imports only
   downward, at module level, with **no exception at all**:
   `DELIBERATE_LAZY_IMPORTS` in `tests/test_architecture.py` is empty, and an
   undeclared in-function import fails the build. A lazy import here is only
   ever a leftover — a cycle to remove, not to defer.
1. **Core AST is the whole language, and the language is upstream.** Both
   backends consume only core AST — macros and `piecewise:` are expanded away
   before dispatch, and so is a named expression unless it states `cases:`,
   which arrives as a node of its own because a mask in a value position is
   not something a substitution can carry — and the plan/query/xarray are
   backend-private. The AST crossing that seam is **fully resolved**, names
   typed `Variable`/`Parameter`/`Dimension`, so a backend cannot hold its own
   opinion about what a name refers to. The waist is closed from the front by
   construction rather than by a test: what a model *means* cannot depend on
   what is done with it, because the package that decides the meaning does not
   depend on this one and cannot import it — a line in `pyproject.toml` rather
   than an allowlist a test in this repository could hold. **Our half of it is
   still checked**: every `math_spec` import under `src/lpspec` names the
   package and never a module inside it
   (`test_the_language_is_imported_as_one_package`), so what this repository
   depends on is the one `__all__` math-spec pins rather than the union of
   whatever its submodules happen to expose. A submodule path would be a
   contract nobody agreed to — it can carry a private name, and it cannot be
   counted.
2. **The engine knows nothing about linopy, xarray or YAML.** `relational/` goes
   plan → engine → a solver sink → solver, with linopy's semantics as a spec to match
   rather than code to share; it never sees the schema or the AST. **The engine is a directory, not a convention:** `engines/polars/`
   is one implementation, and everything above it — `sinks/`, `status.py`,
   and the plan vocabulary itself, which is
   `math_spec.program`'s — is what any implementation answers to. An engine
   package is named for its engine; nothing *inside* one is.
   Enforced *more* strictly than stated — it imports nothing from the
   package at all, bar one declared leaf (`errors.py`, in
   `ENGINE_MAY_IMPORT`), because a near-zero import surface is what keeps the
   subpackage extractable. Widening
   that list is a decision, not an accident.
   **`errors.py` is a leaf by name and not by cost**: it re-exports the
   language's half of the hierarchy, so importing it loads the language. That
   is the price of the root class living upstream of everything that extends
   it. What the engine still raises through it is `DataError` and `LaneError`
   — a verdict about the *data* or about this engine's reach; a verdict about
   what the file may **say** is the language's, made upstream on the spec
   before a program exists at all, and what is left here asserts rather than
   refuses.
3. **One language, two consumers — not fast-vs-slow versions of each other.**
   Both build the models a file declares: this engine attaches and solves
   relationally, linopy constructs a `linopy.Model` the caller owns.
   **Both accept exactly the same language**, and that is structural rather
   than careful — both run the same `to_program` gate, in the package that owns
   it, so a construct the streaming subset refuses is refused there in the same
   sentence, and no operator registry exists that could create a divergence.
   That equality is what makes the differential tests an oracle rather than a
   comparison of dialects. A construct outside the language is a load error
   naming the construct and its rewrite, never a redirection to the other.

   **Accepting is not building, and constructs separate them in both
   directions.** Either may accept what it cannot construct:
   `linopy.Model.add_constraints` refuses a `QuadraticExpression`, and this
   engine refuses an operator acting along a dimension a constant part does not
   carry. Both are named in
   [linopy.md](linopy.md#where-they-part-accepting-is-not-building).
   **What each costs is the oracle**: a construct only one of them builds is
   checked by one of them, and the differential test is replaced by weaker ones
   — two independent encodings reaching one optimum, a residual at the returned
   primal — that no shared misreading fails. Every name added to that gap is a
   construct fewer eyes have seen.

   Since the oracle became another package, that gap has a second kind of
   entry: a construct both build and read *differently*. Those are strict
   `xfail`s naming the divergence, listed in
   [linopy.md](linopy.md#where-they-part-today-by-accident), and each one is a
   differential the suite is not currently running.
4. **Backend-visible YAML files are self-contained.** No Python-side state
   (registries, session objects) may change what a file means.
5. **The public interface is a declared model, not a Python API.** YAML is what
   we ship and document; the contract underneath is the language's two
   states — a `Spec` and the `Program` it lowers to — and whether
   that seam is ever blessed is open
   ([#381](https://github.com/fluxopt/lpspec/issues/381)). The Python surface is
   the runner (`api.py`) and the driver over it (`strategy.py`); the plan is
   internal. The whole of it is
   [nineteen names](#the-python-surface), pinned by a test — so the surface grows
   through a list a reviewer reads, like every other fence here.

## The plan, node for node

**The plan is the vocabulary every consumer speaks**, so each node has exactly
one meaning and this table is the whole of that mapping for this one — what the
file writes, and what the query does with it. What linopy makes of the same
node is linopy's to document, and a copy of it here would be one more thing to
drift. `tests/test_docs_site.py` holds it to
`math_spec.program.Expression`'s own subclasses: a node with no row here is a
node whose two readings nobody wrote down.

**The plan decides what is sayable; the engine only builds.** Every refusal
about the shape of a file — a reduction over a dimension its operand does not
span, a mask wider than what it masks, a bound reaching past its variable, a
degree no position takes — is the language's, made upstream when the spec is
validated and never re-decided on this side of the pin. Where the engine used
to re-decide those after compiling, in fragment vocabulary, it now asserts:
reaching one is a program that was never a valid spec, not a file that said
something wrong. The two verdicts it still *raises* are its own — `DataError`
about the data, and the `LaneError` for the one construct the language accepts
and this engine cannot build (#1137).

**Fan-in** is the column a consumer *acts* on rather than merely documents. It
says how an output row's slots relate to the input's, and the language answers
it for every node — `math_spec.program.fan_in`, total, so a consumer asks
rather than keeping its own list of which kinds reshape anything. Anything but
one-to-one mixes several input slots into one output row, so absence has to be
pushed into the operand before the rewrite consumes it. `Window` missing from
the list that used to hold this is how the two came to disagree about a
constant at a masked slot
([#1142](https://github.com/fluxopt/lpspec/issues/1142)).

| plan node | the file writes | fan-in | the relational query |
| --- | --- | --- | --- |
| `Constant` | a number | one-to-one | a one-row const fragment |
| `Parameter` | a declared name | one-to-one | its table as `(dims…, cval)` |
| `Variable` | a declared name | one-to-one | `(dims…, var_label, coeff=1)`, plus where it exists; at a read, its primal as a const fragment with the same presence, a zero at every absent slot under `absence: zero` |
| `Dual` | `dual(c)` | one-to-one | at a read only: the constraint's rows beside its share of the dual vector, a const fragment present exactly where a row stands |
| `Negate` | `-x` | one-to-one | the value column negated |
| `Add` | `x + y`, `x - y` | one-to-one | the two fragment lists concatenated |
| `Multiply` | `x * y` | one-to-one | a join on the shared dims; two variable factors pair into a quadratic fragment |
| `Divide` | `x / p` | one-to-one | a **left** join, so a divisor with no value leaves a null to report |
| `Power` | `p ** q` | one-to-one | an inner join and `pow` |
| `Sum` | `sum(x)`, `sum(x, over=d)` | many-to-one | the summed dims projected away — no aggregate |
| `GroupSum` | `sum(x, by=lk)` | many-to-one | one inner join with the lookup's table, the grouped dim traded for its targets |
| `At` | `at(x, by=lk)` | one-to-one | the same table joined the other way, fanning out |
| `Translate` | `shift(x, over=d, offset=n)` | one-to-one | a remap through the dimension's `ord`, modulo its size under `wrap` |
| `Window` | `sum_back(x, over=d, within=w)` | one-to-many | a row lands at every position whose window reaches it — no aggregate |
| `Cases` | a named expression's `cases:` block | one-to-one | each region's value cut to its own mask and the fragment lists concatenated |

A `Cases` is the one node carrying a **mask in a value position**, and the one
whose several values are alternatives rather than slots summed together: the
language proves the regions disjoint and total before any data attaches, so an
output row reads exactly one of them and neither consumer ranks them. What
neither may do is let a region speak outside itself — a region's data is
owed only where the region applies, and a region empty at a coordinate it does
not claim must leave the row that the other regions do cover.

**A read is the one walk where every leaf is a number.** A named expression
is evaluated after the solve, never built: this engine compiles a variable to
its primal and `dual(c)` to the constraint's row duals as const fragments, and
linopy reads `.solution` and `.dual` and does xarray arithmetic. That is why
the language holds an entry the math never reads to no degree and neither
needs one — a product of two variables, a variable
under a power, a division by one are arithmetic over values — and why `Dual`
is the one node a build refuses on sight: the language keeps it out of the
math, and only a read can answer it. A solve that left no duals refuses the
read with the sentence `result.dual` gives, and every other entry still reads.

**Neither reduction aggregates**, which is the theme the table repeats: a
`Sum` drops columns, a `GroupSum` swaps them and a `Window` replicates rows,
and every duplicate collapses once in the terminal `SUM(coeff) GROUP BY row,
col` at assembly. The per-construct detail lives where it is acted on: the
polars column conventions in `compiler.py` and `fragments.py`.

## The relational lane

**The spine is one module per box above.** `attaching.py` takes the tidy frames
`sources.py` handed over the seam and freezes them into what every query is
written against; `compiler.py` turns plan nodes into
lazy frames and reads nothing; `assembly.py` fills the model frames; `sinks/`
drains them; `engine.py` runs that lifecycle and holds the solver between
solves. Three more sit beside the engine rather than inside it, because each
answers a question the engine merely *uses*: `labels.py` decides which
coordinate gets which solver index, `readback.py` spells a row or a solve back
out in the model's own names, and `result.py` is what a caller reads a solve
back through. The remaining four are not on the spine and the diagram does not
draw them — `fragments.py` is the vocabulary a compiled expression is *in* (the
one the spine itself speaks is upstream), `predicates.py` the one a `where:` is,
and `reindex.py` the two operators that walk a dimension's own order;
`status.py` is the boundary a solver's verdict comes back over. The other
boundary, a caller's table on the way in, is `frames.py` — top level rather
than in this lane, because all three consumers read it. The map below is the
full list.

That split is what makes the ceiling's admissibility test something you can
*perform* rather than reason about: build a `PolarsCompiler`, hand it a node,
read `.explain()` — `tests/test_compiler.py` does exactly that over empty
frames, since a schema is all it takes to compile a query. It is also why a new
sink is a module in one of two families rather than another method on the
engine.

**What attaching produces is a value.** `AttachedSources` is frozen — parameters,
dimensions, their cardinalities, and which parameters are boolean — because a
query is written against data that has stopped changing. The variable frames
are passed *beside* it and stay mutable, since a variable frame appears as its
declaration is built and a constraint compiled afterwards has to see it. That
is the one live registry in the lane, and keeping it out of the carrier is
what makes it visible in a signature rather than only in a docstring.

**What a build produces is a value too, for the same reason.** `BuiltModel` is
frozen — the model frames, the label frames, the per-declaration blocks and the
compiler that made them — because a build is finished when it exists. What
fills during assembly lives on `_Assembly`, which is discarded once it has
frozen, so the engine holds one field where it used to hold a frame each: "has
this engine got a model" is one question rather than seven, `close()` is one
assignment, and a build that raises leaves no model rather than half of one.
What survives that release is `_Measured` — the counts `diagnostics()` reports,
which are measurements about the build rather than parts of it.

**Tidy tables.** Parameters are `(dims…, value)`; a variable frame is
`(dims…, var_label)`, one row per *existing* variable; a linear expression is
`(frame dims…, var_label, coeff)` plus a constant part; constraint rows are
`(row, sense, rhs)`; the coefficient matrix is COO `(row, col, coeff)` while
declarations build, and lands as CSR at assembly — `(col, coeff)` in row-major
order plus a `row_starts` offset array, the same three arrays a solver takes,
at 12 bytes per entry. Masks
are **row absence** — no NaN sentinels, no `-1` labels. Broadcasting is a join,
`sum` drops coordinate columns, `sum(by=)` joins the dim table and projects a
declared lookup in place of the grouped dim. Neither aggregates: both
rewrite a fragment's dim tuple, and duplicates collapse in the terminal
`SUM(coeff) GROUP BY row, col` at assembly.

**The label contract**, and the one place order is load-bearing. Everything else
in the lane is order-free, which is what lets the query planner rearrange it.

- Labels are dense `0..n-1` by construction, so `var_label` **is** the solver
  column index and `row` the solver row index — no remapping. That is what
  `update` spends: new bounds, costs and right-hand sides go onto a loaded
  solver by position, and appending rows moves no column and renumbers no
  existing row. Structural editing stays out of scope; an update that *does*
  move a label is a rebuild, and the answer is the same either way.
- They are **row-major over the masked coordinate product**, sorted on the
  dimensions' declared ordinals. A contract, not a side effect: it is what makes
  a build reproducible run to run.
- Variables and constraint rows are the same operation over different frames and
  it is written once (`labels.frame`): number the surviving coordinates by their
  row-major position in the declared product. A mask that cannot see the leading
  dims leaves the survivors a *rectangle*, so only the masked suffix is
  materialised — a guarded shortcut inside that one function, which must reach
  the integers the general path would have. That is why labelling is a module
  with stated inputs rather than a method among twenty: nothing else about a
  build can move an index.
- The same order comes **back**: `primal` / `dual` / `to_parquet` read the
  label frame, which was numbered in that order, and the LP sink writes it.

**The plan is affine-by-design.** No node introduces variables or constraints as
a side effect of an expression; formulations are model *transformations*.
Variable *types* are not formulations — binary/integer are a `vtype` column, LP
`binary`/`general` sections and HiGHS integrality, which keeps basic MILP inside
the streaming lane. **`sos:` is the same shape**: a `SosDeclaration` naming
columns the variable already made, one more stream out of the engine, and no
expression node — which is why a set can be carried whole to a sink that has
the concept. Reimplementing linopy's reformulation passes inside the plan is
explicitly rejected: that duplicates the library this package consumes. Where
one is unavoidable — a sink with no SOS at all — it happens at the *sink*
boundary, on the built tables, and never in the plan.

**A frame is the boundary in both directions.** `frames.py`
recognises a caller's table through the Arrow PyCapsule protocol without
importing any dataframe library, and `Result.primal` hands back a
`polars.DataFrame`, which exports the same protocol. That symmetry is what
keeps pandas and pyarrow off the dependency list: they are bridges *out*
(`to_pandas`, `to_dataarray`), shipped with the `[linopy]` extra, not shapes
the engine holds. The bare-install CI job runs the suite with neither present.

**Sinks are capped, explicitly.** Four streams and no more: `cols` (bounds,
objective coefficients, integrality), `rows`, `A` in CSR, and `sos` — the
special-ordered sets, `(set, type, col, weight, big_m)`. The upgrade path from
here is `genconstr`, plus a semi-continuous threshold on `cols`.

**The fourth is the one that lands unevenly**, because its destination differs
per sink (see "Capability is not the ceiling"). So a solver **declares** how it
satisfies one — `native` or `reformulated` — and the *family* acts on the
answer (`solvers.ingestible`), handing a sink that cannot take a set the same
feasible region as binaries and linking rows (`sinks/sos.py`, whose README
carries the per-sink table). Declared rather than discovered at the hand-off is
what [Track 3](https://github.com/fluxopt/lpspec/issues/472) asked for, and this
is its first two entries.

What the rewrite adds goes **after** the model, which is the label contract
spent rather than bent: an appended column moves none of the model's own and an
appended row renumbers none of its rows, so a solve reads its answer back by
the same slice either way.

**A sink is one of two things, and the directory says which.** A **solver**
runs the tables and returns an answer, chosen by **name** at the call
(`solver_name='gurobi'`); a **writer** renders them to a file, chosen by the
output's **suffix** — because a file's format is a property of the file, while
which solver runs is a property of nothing but the call. Both sets are closed
dict literals (`SOLVERS`, `WRITERS`): no YAML key names a solver, and nothing
installed may change what either resolves to.

The split is a directory rather than a convention for the reason `engines/` is:
**how many solvers there are will change, and what a solver has to answer will
not.** A new one is a module named for it and a line in `SOLVERS` — no method
on the engine, no branch in `api.py`, no name on the Python surface. Members
share the projection of `cols` and `obj` onto the solver's column index, which
lives on `Tables` so two solvers cannot drift into loading different
models; they never share hand-off code, because the currencies differ (HiGHS and
Xpress take the three CSR arrays, gurobipy a matrix object) and because an
optional package must stay off the import path of a caller who does not use
it.

### What a quadratic objective costs the sink

Neither direct API has a per-coefficient counterpart to `changeCoeff`:
`passHessian` and `setMObjective` take the quadratic part whole. Under the
aligned-only scope (`variable × variable` at the same coordinates) `Q` is
**diagonal**, so it costs 16 bytes per quadratic column — 0.16 GB at 10⁷
columns, 1.60 GB at 10⁸ — against a direct-sink peak already dominated by the
solver's own model. HiGHS accepts `dim_ < num_col` (verified), so ordering the
quadratic variables first bounds the Hessian to that block.

**The diagonal argument dies as soon as the product is not aligned**, and the
language does not restrict it to aligned: `x[i] * y[i, j]` broadcasts and
`x[i] * y[j] * a[i, j]` joins through a table. The replacement bound is one
entry per pair the expression states — the `nnz` of whatever couples the
factors — which is still a declared-shape quantity. What is not is the cross
join of two reductions, and that is the shape the language refuses
(`math_spec.degree`).

**Whole is not the same as reloading.** A second `passHessian` lands on the
model already loaded, replacing `Q` and leaving the LP standing — so a moved
quadratic *coefficient* is pushed like a cost, and only the sparsity *pattern*
is structure.

## Module map

| Module | Role |
|---|---|
| `math_spec` (a dependency) | the whole language: the file is read, expanded, resolved, judged and lowered there, and what crosses into this repository is its two public states — a `Spec`, what the file says, and the `Program` it lowers to — [its own reference](https://math-spec.readthedocs.io/en/latest/reference/language/) |
| `api.py` | the runner: `check` / `build` / `solve` / `write`, linopy-free; and `Buildable`, what every verb takes as the spec |
| `sources.py` | the one door: a caller's data (parquet paths, in-memory tables, plain-Python shapes) read into tidy frames and checked against the declarations — one row per coordinate, labels that exist, values present and of the declared type |
| `curves.py` | the one guard that needs numbers rather than a schema: is a `piecewise:` curve supplied everywhere it is built, monotone, and of the curvature its method is exact for |
| `frames.py` | the boundary — caller tables in, via the Arrow PyCapsule protocol; read by the front door and the driver |
| `errors.py` | the run half, and the whole re-exported — what a caller catches off `lps.`; a wording lives here only where two modules raise it |
| `strategy.py` | the driver above the runner: one plan per slice, folded — scenarios, rolling horizon, myopic pathways |
| `relational/engines/polars/compiler.py` | plan → lazy frames; pure, reads nothing |
| `relational/engines/polars/reindex.py` | `shift` and `sum_back`: moving a fragment's rows along one dimension's own order, and what happens at the edge |
| `relational/engines/polars/predicates.py` | a `where:` mask as a boolean query over the coordinate product; the plan's predicate nodes, and nothing else |
| `relational/engines/polars/fragments.py` | what an expression compiles *to*: the additive pieces and the arithmetic over them; holds no state and reads no data |
| `relational/status.py` | solve outcome on two axes; linopy's vocabulary, copied not imported |
| `relational/engines/polars/labels.py` | which coordinate gets which solver index; one rule, one guarded shortcut that must agree with it |
| `relational/engines/polars/attaching.py` | the door's frames → `AttachedSources`, the frozen, `Enum`-encoded frames every query is written against |
| `relational/engines/polars/assembly.py` | one build: every declaration into rows of the model frames, quadratic constraints last |
| `relational/engines/polars/readback.py` | a built row, a solve's frames and a named expression, spelled back out in the model's own labels |
| `relational/engines/polars/engine.py` | the lifecycle: build, hand to a sink, read back, the counters and clocks `diagnostics()` reports |
| `relational/result.py` | what a solve returned: status, objective, and the label joins that read values back |
| `relational/sinks/tables.py` | what every sink reads and no more — the five frames plus the batching scalars, and their projection onto the solver's column index; what an engine produces |
| `relational/sinks/capabilities.py` | what a sink can ingest — hard rule 3's *accepts ≠ builds* axis; `lanes.py` declares each **lane** against the same vocabulary |
| `relational/sinks/sos.py` | the one stream a sink may not be able to ingest, written as two it can: sets → binaries and linking rows |
| `relational/sinks/` | how a built model leaves, in two families: `solvers/` (one module per solver, chosen by name) and `writers/` (one per format, chosen by suffix) — [README](https://github.com/fluxopt/lpspec/blob/main/src/lpspec/relational/sinks/README.md) |
| `linopy/__init__.py` | the lane's two verbs: `build` constructing a `linopy.Model`, and `expression` reading a named quantity off a solved one |
| `linopy/loader.py` | the crossing into pandas and xarray: `tidy_sources`' frames as master coords and an `xr.Dataset` |
| `linopy/coverage.py` | the two positions an absent row has no reading for: a divisor and a constant side |
| `linopy/absence.py` | the four positions an absent value is spelled differently in — absence is positional in this lane |
| `linopy/builder.py` | eager backend: core AST → `linopy.Model` |
| `linopy/operators.py` | the eager evaluation of every built-in, on xarray and linopy |
| `linopy/where.py` | a resolved `where:` as a boolean array, and the shape linopy's `mask=` takes |
| `linopy/_notes.py` | attach context to an exception on the way out; no package imports, no opinions |

**Two subpackages, and the directory *is* the rule in both cases.** Everything
under `relational/` is the relational lane and imports nothing else from the
package, with a second boundary inside it — `engines/` holds implementations,
the rest of `relational/` is what they implement; everything under `linopy/` is
the eager lane and is the only code allowed to import linopy or xarray.
`tests/test_architecture.py` reads membership off the path in both cases, so
neither fence can be stepped over by naming a file differently.

There used to be four, and the two that left are the same claim twice.
`language/` was fenced to import nothing from this package — an empty
allowlist, kept empty so the directory could be lifted out without an edit —
and `typeset/` was fenced to read the AST and nothing else. Both were lifted
out. **A fence whose allowlist is empty is a package waiting to happen**, and
that is the one prediction in this file that has since been paid: what the two
fences were protecting is now protected by them being somewhere else.

What remains points one way. `relational/`'s fence points outward at one
declared leaf, `errors.py`; the language's points nowhere at
all, because it is not here. `errors.py` is the seam that survived in the other
direction: the root class lives upstream, so importing this package's errors
imports the language, and the run half extends the model half rather than
paralleling it.

### What counts as language

The rule is [its own page](https://math-spec.readthedocs.io/en/latest/about/what-counts-as-language/), because it decides what
may live here rather than how this package is arranged:

> **A rule is language iff two consumers answering it separately would be a
> bug.**

Every "one implementation each" rule in this file is that test applied, and the
implementations are now upstream: names resolve once
(`math_spec.resolution`), the operator set is closed (`math_spec.operators`)
and lowering turns it into the closed plan-node set both lanes dispatch on —
which is the axis a test here holds them to, neither lane keeping a table of
operator names — an operator's dim rule lives only in `math_spec.dimensions`
with lowering **asking** for the verdict rather than deciding again, and degree
lives only in `math_spec.degree`.
`math_spec.piecewise` is upstream by the same test: a formulation emits
declarations, and declarations are language. The rule decided where the cut
fell — everything it called language went, and everything it did not stayed.

The test also says what cannot follow the rest upstream. `curves.py` answers a
question two consumers answering separately *would* be a bug — is this curve
monotone, is its curvature the one the declared method is exact for — so by the
rule it is language; it is here because the answer needs numbers, and the
language has never seen one. The half that does not need them is upstream: a
block's `assumptions` name each condition and `assumption_message` words the
refusal, and the caller holding the values does the checking. A rule is only
ours when data is what decides it.

The corollary is what the top level is *for*. A module stays flat when it is
legitimately **both** halves: `sources.py` attaches data to a validated schema,
`curves.py` judges the numbers it attached, `api.py` runs the lot. That is a real
category and a small one — a flat module should be arguable.

### Naming across the layers

The same construct passes through three layers, and each names it in full —
no abbreviations, so a name never has to be decoded. The **layer is the
suffix**, which is what keeps the three vocabularies from colliding:

| Layer | Suffix | Example |
|---|---|---|
| YAML block (`math_spec.model`) | `Block` | `VariableBlock`, `PiecewiseBlock` |
| Core AST (`math_spec.*_parser`) | `Node` | `VariableNode`, `DimensionComparisonNode` |
| Program (`math_spec.program`) | none / `Declaration` | `Variable`, `VariableDeclaration` |

All three rows are another package's now, which is exactly why the table
stays: every name a lane dispatches on is spelled against a vocabulary this
repository does not control, and a rename upstream that collides here is a
thing to notice.

Two rules follow from that table, and a PR that adds a construct keeps them:

- **A node names the coordinate map, not a surface spelling.** The translation
  node is `Translate`, and it stayed that way when the surface collapsed to a
  single `shift(…, edge=)`: the node is named for what it does to coordinates,
  so which keyword the language happens to expose does not reach it.
- **Nothing is abbreviated.** `Cmp` became `ParameterComparison`, `vtype`
  became `variable_type`. The one place abbreviation survives is frame column
  names inside the engine, which are not Python identifiers.

### Where a concept is already linopy's, use linopy's name

For anything this package shares with linopy — solve statuses, result shapes,
solver metrics, duals — adopt **linopy's primitive**: its spelling, its field
names, its decomposition. `status` / `termination_condition` are two axes and
`is_ok` is the rollup because that is linopy's model. Our audience arrives from
linopy/PyPSA, and a second vocabulary for one fact is a tax on all of them; it
also keeps the oracle honest, since the lanes can then be compared exactly.

**Copy it; do not import it.** The engine may not import linopy (rule 2), so the
tables live here and a test imports linopy to assert the copy still matches
(`tests/test_solve_status.py`) — a copy nobody checks is a copy that rots.

This applies to vocabulary we *share*. Where the design genuinely differs it
stays ours: there is no `Solution` of dense arrays to hold, because values are
read back by joining labels to coordinates.

## Extension checklists

**Add a macro or named expression:** edit YAML. Nothing else.

**Add a sink:** a module in `relational/sinks/solvers/` named for the solver
(`solve_<name>`, `build_<name>`, one line in `SOLVERS`, its dependency behind an
extra and imported inside the function), or one in `writers/` keyed by suffix in
`WRITERS`. Either way it declares what it can ingest — a `Capabilities`
descriptor beside the code that knows, since a sink declaring nothing reads as
taking nothing. Nothing above it changes — no method on the engine, no branch in
`api.py`, no name on the Python surface. The
[README](https://github.com/fluxopt/lpspec/blob/main/src/lpspec/relational/sinks/README.md)
is the full list, and `tests/test_architecture.py` checks the shape off the path.

**Add a consumer of the AST** (a renderer, a checker, a report): a package of
its own, depending on `math-spec` and not on this one. It reads
`math_spec.to_spec` and stops there — if it needs the plan it is a lane, not
a consumer, and the ceiling doc is the conversation to have first. The
renderer is the worked example: it was a fenced directory here until the fence
turned out to be a package boundary.

**Add an operator:** two repositories, in this order. **In math-spec:** grammar
(usually free — `f(x, k=v)` already parses) → signature in `operators.BUILTINS`
(arity and which arguments name dimensions — resolution, validation and
lowering all read it from there, so the shape is declared once) → its dim rule
and its degree verdict → the plan node it lowers to → the language reference.
Then **here**, against a released tag: eager implementation → compiler case →
engine → differential test through a solver *and* the LP writer, and this file
if structural. The pin is what sequences them: nothing in this
repository can lower an operator the pinned language does not parse, so the
upstream half lands and is tagged first, and
[the nightly canary](https://github.com/fluxopt/lpspec/blob/main/.github/workflows/canary.yml)
is what says the two halves have not drifted since.

Three things are deliberately *not* per-operator work, because they are one
implementation each: an operator's dim rule lives only in `math_spec.dimensions` —
both its dim *set* and its verdict on an operand that lacks the dim being
reduced along, which lowering asks for rather than deciding again — its degree
verdict lives only in `math_spec.degree`, which every consumer asks; and the
dense-label assignment that gives a coordinate its solver index lives only in
`relational/engines/polars/labels.py`, shared by variables and constraint
rows. What a consumer still owns is what is about *building*: here, the
fragment rewrite the compiler performs.
