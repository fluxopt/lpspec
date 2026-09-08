# Relationship to linopy

Everything about [linopy](https://github.com/PyPSA/linopy) in one place, because
it is otherwise the kind of thing that gets mentioned everywhere and explained
nowhere. Three separate relationships, and conflating them is what made the rest
of the docs noisy:

| | What | Where it matters |
|---|---|---|
| **Not a dependency** | solving a model never imports it | packaging |
| **The oracle** | how we know the answers are right | testing |
| **The other consumer** | it reads the same YAML, natively | where to go for a `linopy.Model` |

## 1. It is not a runtime dependency

`lps.solve`, `lps.build`, `lps.write` and `lps.check` go YAML → polars → HiGHS
or file, and import nothing from linopy, xarray or pandas. CI proves it: the
bare-install job runs the whole suite with none of them present.

`pip install "lpspec[linopy]"` adds linopy, xarray and pandas, which buys two
things and nothing else — the differential oracle below, and the `to_pandas` /
`to_dataarray` bridges out of a result. A bare install is a complete one.

**Nothing a bare install can reach names linopy, including in a traceback.** The
public exception tree is rooted at `LpspecError`, with no alias
([#389](https://github.com/fluxopt/lpspec/issues/389)) — a name from this
extra has no business reaching a caller who never installed it.

## 2. It is the oracle

Correctness here is not "the tests pass"; it is **the same YAML, built both
ways, produces the same model**. The differential suite builds a model through
the relational engine and through `linopy.Model.from_spec`, and compares.

That is only meaningful because both paths consume the *same resolved AST* and
neither may hold its own opinion about what a name means — the narrow waist in
[the architecture notes](architecture.md#one-contract-many-consumers). If they resolved
names independently, the suite would be comparing two dialects rather than
checking one language. It is also why the two pins move together: the oracle
lowering with a different math-spec than this package is two languages again,
by the back door.

It also has a known blind spot, which is why the model gallery exists: a
**shared misreading** passes the differential suite green. Only an outside
published optimum catches that, and
[docs/examples/index.md](../examples/index.md) is where those live.

**Where a concept is already linopy's, we copy its name** — solve statuses,
`status` / `termination_condition` as two axes with `is_ok` as the rollup, the
shape of a result. Our audience arrives from linopy and PyPSA, and a second
vocabulary for one fact is a tax on all of them. But **copy it, do not import
it**: the engine may not import linopy, so the tables live here and a test
imports linopy to assert the copy still matches. A copy nobody checks is a copy
that rots.

## 3. It is the other consumer of the language

**The same file, built as a `linopy.Model` — by linopy.** `Model.from_spec`
reads a math-spec program natively
([PyPSA/linopy#922](https://github.com/PyPSA/linopy/pull/922)), so the way to
get one is to ask linopy for it:

```python
import linopy

m = linopy.Model.from_spec('spec.yaml', sources={...})
m.solve()
m.spec.expressions['co2'].solution  # a named quantity, read back
```

There used to be a second lane in *this* package doing the same job —
`lpspec.linopy.build`, a peer of `lps.build` picked by an import. It is gone.
Keeping it meant maintaining a smaller implementation of something linopy owns:
its own reads back through `model.spec`, survives `to_netcdf` and `Model.copy`,
and typesets the model it built, none of which the lane did.

What this package takes on instead is that the two agree. The differential
suite builds every model here and there and compares them, which is section 2
— and it is a stronger claim than it was, because the two no longer share a
reader: `linopy.spec.attach` enforces the language's binding rules in its own
code, where the lane went through this package's `sources.py` and could only
ever have agreed with itself.

**What was lost with the lane, and is worth knowing:**

- `check(spec, sink='linopy')` — asking, before any data, whether linopy will
  take a file. Capability now covers sinks only.
- The two constructs below used to raise a `LaneError` naming the wall *and*
  the route around it. One of them is linopy's to say now, and it says it as a
  library exception.

### Where they part: accepting is not building

**Neither is a language limit**: both files pass `check`, and each is built by
the one the other cannot.

**The first is linopy's: an objective carrying a constant.** `linopy.Objective`'s
expression setter rejects any expression whose `const` is nonzero — *"Constant
values in objective function not supported."* — and there is no slot to put one
in, which is why PyPSA carries `n.objective_constant` out of band. So a model
like `examples/ports/osemosys_utopia.yaml`, whose objective owes a fixed cost
on capacity that already stood in 1990, builds here and raises there
([#894](https://github.com/fluxopt/lpspec/issues/894)).

**The second is a quadratic constraint**, which `add_constraints` refuses
outright — a `QuadraticExpression` is not something it takes, and no
reformulation of it is exact. This engine builds one and gurobi or an `.lp`
file carries it.

**The third is this engine's, and it is the mirror: an operator acting along a
dimension a constant part does not carry**, beside a term that does — say
`sum(x * k + d, over=t)` where `d` is a scalar. A constant part compiles to its
own frame here, so a fragment with no rows for `t` has no slots for the
operator to act on, and under a mask which slots those are is known only to the
rows. linopy has no such split — the operand is one masked expression — so it
builds the file as written.

It is **one wall, reached by all four operators that act along a dimension**
(`sum(over=)`, `sum(by=)`, `shift`, `sum_back`), which is why they share a
refusal rather than each wording its own. The message names the rewrite that
reaches the same number — declare the parameter over the dimension and supply
it there ([#1137](https://github.com/fluxopt/lpspec/issues/1137)).

Finding that wall is what turned up a real disagreement behind it: `sum_back`
read a constant at a slot the variable was absent from, where every other
operator drops it, so the two answered 2.5 and 3.0 on a file **neither**
refused. A reduction consumes its operand before any row exists, so absence has
to be pushed into the operand first — `sum` and `sum(by=)` did that and the
window did not. Fixed by giving the window the same pass, with a differential
test over every operator that moves along a dimension
([#1142](https://github.com/fluxopt/lpspec/issues/1142)).

### Where they part today by accident

Two differences are not deliberate on either side: they are places the oracle
reads the language differently from the reference, and every one of them is a
strict `xfail` in the suite naming it, so the day it is fixed upstream the
suite goes red and the marker comes out.

| What | Which is right |
|---|---|
| a **sparse coefficient** table — rows only where the value is nonzero | this engine. [Rule 8](https://math-spec.readthedocs.io/en/latest/reference/language/absence/) reads a missing row as the value that contributes nothing, which for a coefficient is `0`; the oracle refuses it wherever a parameter is used. Nine of the ten ported models are refused on this alone. |
| a **`dtype: datetime` dimension** whose labels arrive as `datetime.date` | this engine. Labels are canonicalised against the declared dtype ([#1076](https://github.com/fluxopt/lpspec/issues/1076)), so one instant is one label whichever library spelled it; the oracle compares the objects it was handed. |

### The same language, and the same data

Both accept **exactly the same language**, and structurally: both run the same
`to_program` gate, so a construct one refuses the other refuses in the same
sentence — it is math-spec's sentence, raised before either attaches a source.

The **data** is where they differ by design, and the difference is containers
rather than rules. This package reads *tables*: a parquet path, any table
exporting the Arrow PyCapsule protocol, a `pd.Series` carrying its dims in an
index, a `dict` or a sequence over one dimension, or one number. linopy reads
pandas and xarray, and takes a dimension's labels as a sequence where this
takes a one-column table. Neither reads the other's spelling of a dimension
index, and neither takes an `xr.DataArray` as a *source* here — this package
reads tables and hands arrays back.

The rules underneath are the language's and are the same on both sides: labels
come from `sources` or from what the file declares and never from a parameter,
a dimension the file declares *and* the caller supplies is refused, and a
dimension with neither has no index and is refused rather than derived.

## What we deliberately do not take

Array operations (`merge`, `reindex`, `stack`), the Python modeling API, and the
solver layer. The first is data prep
([the limits](https://math-spec.readthedocs.io/en/latest/reference/language/errors/#what-the-language-will-not-say)), the second is
[hard rule 5](architecture.md#hard-rules) — the model is the file you review
and diff — and the third is
[#106](https://github.com/fluxopt/lpspec/issues/106), where we adopt linopy's
*design* for declared solver capabilities without adopting its code.

The modeling API is the one a reader arriving from linopy misses first, and
what replaces it is two notebook pages: [Change a model](../interactive.ipynb)
for the loops — `update` for new numbers, a longer table for more rows, a
patched `dict` for new math — and [Fix, relax, remove](../lifecycle.ipynb) for
the verbs, which are the same loops aimed at `fix`, `relax` and
`remove_constraints`. What neither replaces is the *debugging*: an IIS. Both
pages say so — a built row is read with
[`row`](../reference/api.md#reading-one-row), in linopy's own form.

Where linopy is genuinely ahead, and why none of it is a ceiling question, is the
honest snapshot in [the roadmap](roadmap.md#honest-snapshot).

What is *owed* to linopy rather than merely true of it — and the same for
Calliope, whose math language this surface is derived from — is
[prior art and credit](prior-art.md).
