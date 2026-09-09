# Relationship to linopy

Everything about [linopy](https://github.com/PyPSA/linopy) in one place, for a
reader who arrives from linopy or PyPSA. There are three separate relationships.
Keeping them apart is what keeps the rest of the docs quiet:

| | What | Where it matters |
|---|---|---|
| **Not a dependency** | solving a model never imports it | packaging |
| **The oracle** | how we know the answers are right | testing |
| **The lane** | the second thing a file can be built as | what a caller chooses |

## 1. It is not a runtime dependency

`lps.solve`, `lps.build`, `lps.write` and `lps.check` go YAML → polars → HiGHS
or file. They import nothing from linopy, xarray or pandas. CI proves it: the
bare-install job runs the whole suite with none of the three present.

`pip install "lpspec[linopy]"` adds linopy, xarray and pandas. The extra buys
two things and nothing else: the lane below, and the `to_pandas` /
`to_dataarray` bridges out of a
[result](../reference/glossary.md#the-chain) (one answer, read back from a
solve). The lane is a peer, not a fallback. Nothing routes to it, and a bare
install is a complete one.

**Nothing a bare install can reach names linopy, including a traceback.** The
public exception tree is rooted at `LpspecError`, with no alias
([#389](https://github.com/fluxopt/lpspec/issues/389)). A name from this extra
must not reach a caller who never installed it.

## 2. It is the oracle

Correctness here is not "the tests pass". It is **the same YAML, built both
ways, produces the same model**. The differential suite builds a model through
the relational engine and through linopy, and compares the two.

That comparison means something only because both paths consume the *same
resolved AST*. Neither may hold its own opinion about what a name means. That is
the narrow waist in
[the architecture notes](architecture.md#one-contract-many-consumers). If each
path resolved names on its own, the suite would compare two dialects rather than
check one language.

The oracle has one known blind spot, and the model gallery exists to cover it. A
**shared misreading** passes the differential suite green. Only a published
optimum from outside catches it, and
[docs/examples/index.md](../examples/index.md) is where those live.

**Where a concept is already linopy's, lpspec copies its name.** Solve statuses;
`status` and `termination_condition` as two axes with `is_ok` as the rollup; the
shape of a result. The audience arrives from linopy and PyPSA, and a second
vocabulary for one fact taxes all of them. But **copy it, do not import it.**
The engine may not import linopy, so the tables live here. A test imports linopy
to assert the copy still matches. A copy nobody checks is a copy that rots.

## 3. It is a lane

A [lane](../reference/glossary.md#how-it-runs) is one of the two ways a spec
(the YAML file: the math, before any data) is executed. This one builds the same
file as a `linopy.Model` instead of attaching data relationally. The caller
picks the lane by an import. The call is the one `lps.build` takes: the same
first argument (a path, a mapping, or a spec the language has already read),
the same `sources` (the mapping of names to the tables you attach), the same
index sources.

```python
from lpspec import linopy as lpspec_linopy

m = lpspec_linopy.build('spec.yaml', {...})  # -> linopy.Model
m.solve(...)
lpspec_linopy.expression(m, 'spec.yaml', 'co2', {...})  # a named quantity, read back
```

Both calls are *pure*: YAML in, a model or a value out, nothing retained.
`build` returns a plain `linopy.Model` with no accessor, no attached schema and
no patched attributes, so nothing is lost across `pickle`, `deepcopy` or
`to_netcdf`. To inspect the math, re-read the file with `to_spec`. `expression`
is the reader, and the same purity forces it to take `sources` again. It
evaluates a declared named expression
([named expressions](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#named-expressions))
on the solved model and hands back linopy's native `.solution`. That is the
eager half of `result.expression(name)`, so the differential suite can hold the
two lanes to one answer.

**This lane constructs; it does not attach.** Math for a `linopy.Model` that
something else built, a PyPSA network say, had a verb here and no longer does
([#845](https://github.com/fluxopt/lpspec/issues/845)). That verb was the one
file allowed to reference names it did not declare. Paying for that exception
across the whole language layer bought one use case. Build a second model and
merge it.

### What a construct becomes

This is the whole translation in one place: what `lpspec.linopy.build` calls for
each thing a file can say. Each row lives in `linopy/builder.py`, one section
per group below.

| Declaration | linopy |
|---|---|
| `variables:` | `Model.add_variables(lower, upper, coords, name, mask, binary, integer)` |
| `sos:` | `Model.add_sos_constraints(variable, sos_type, sos_dim, big_m)` — the block handed over, not a formulation rebuilt |
| `constraints:` | `Model.add_constraints(lhs, sign, rhs, name, mask)`, one rule per declaration |
| `objective:` | `Model.add_objective(expr, sense)`, each additive term summed over the dims it carries |
| `expressions:` | evaluated at the solution — every variable its `.solution`, every `dual(c)` the constraint's `.dual` — as xarray arithmetic, so an entry the math never reads is read at whatever degree it was written |

| In an expression | linopy or xarray |
|---|---|
| `x` — a variable | `Model.variables['x']`, `.fillna(0)` under `absence: zero` |
| `p` — a parameter | its `xr.DataArray`, `.fillna(0.0)` where it stands as a coefficient |
| `+` `-` `*` `/` | the Python operators linopy overloads |
| `sum(x, over=t)` | `.sum('t')` |
| `sum(x, by=lk)` | the lookup attached as a coordinate, then `.groupby()`, reindexed onto the target dimension's declared labels — one key per lookup, so `by=[lk1, lk2]` groups by both at once |
| `at(p, by=lk)` | `.sel({into: lookup})` — xarray's vectorised selection *is* the pullback, and one entry per lookup reads a tuple of labels at once |
| `shift(x, over=t, offset=n)` | `.shift({t: n})`; `.roll({t: n})` under `edge: wrap`; a `.sel()` gather where the offset differs per entity or `by=` groups it |
| `sum_back(x, over=t, within=w)` | a sum of `w` scalar gathers, each unreachable position contributing zero; under `by=` each gather reads inside the group, so the window stops at its edge |
| `dual(c)` | `Model.constraints['c'].dual`, at a read only — the language keeps a dual out of the math, and a solve that stored none refuses the read |

| A `where:` | linopy |
|---|---|
| on a declaration | the `mask=` argument — a mask that excludes nothing is passed as `None` |
| `defined(x)` | `Model.variables['x'].labels != -1`, linopy's own marker for an absent slot |
| a comparison | the Python comparison operators element-wise, absence reading as false |

Absence is the one thing with no single row. It is positional: a missing
parameter row is zero in a coefficient, an error in `bounds:`, and false in a
`where` operand. `linopy/absence.py` holds all four spellings together. The
builder calls them qualified, as `absence.coefficient(...)`, so a reader meets
the name at the call rather than only at the definition.

### The same language, and the same data

**The lane accepts exactly the same language.** That equality is what makes the
oracle an oracle, and it is structural: both lanes run the same `to_program`
gate. A construct one lane refuses, the other refuses in the same sentence,
never with a redirection to the other lane.

**Accepting is not building, and two constructs part the lanes, one in each
direction.** Neither is a language limit. Both files pass `check`, and each is
built by the lane the other cannot.

**The first is this lane's wall: an objective carrying a constant.** The
expression setter of `linopy.Objective` rejects any expression whose `const` is
nonzero: *"Constant values in objective function not supported."* There is no
slot to put one in. That is why PyPSA carries `n.objective_constant` out of
band. So a model like `examples/ports/osemosys_utopia.yaml`, whose objective
owes a fixed cost on capacity that already stood in 1990, builds relationally
and raises linopy's `ValueError` on this lane. **Dropping the constant is the
one repair that must not happen.** The lane is the oracle, so a quietly
shortened objective would recalibrate every differential test on such a model
to the wrong number. Adding the constant back as a variable pinned to `[1, 1]`
reaches the right answer, and was refused too. It puts a column on the caller's
model that the other lane does not have. So the lane says it in its own words.
`builder.py` checks for a constant before linopy is asked and raises
`LaneError`, naming the wall and the route that does build the model.
`tests/test_corpus_parity.py` carries the strict xfail, typed to that error
rather than to any `ValueError`. The day linopy grows a slot, the test XPASSes
and the check comes out with it
([#894](https://github.com/fluxopt/lpspec/issues/894)).

**The second is the relational lane's wall, and it is the mirror: an operator
acting along a dimension that a constant part does not carry.** The constant
stands beside a term that does carry it. Take `sum(x * k + d, over=t)` where `d` is a scalar. The relational
lane compiles a constant part as its own frame (one table of rows). A fragment
with no rows for `t` has no slots for the operator to act on. Under a mask (a
`where:` that excludes some rows), only the rows know which slots those are.
This lane has no such split. The operand is one masked expression, so the
constant is dropped wherever the term is, and the lane builds the file as
written.

**It is one wall, reached by all four operators that act along a dimension**
(`sum(over=)`, `sum(by=)`, `shift`, `sum_back`). That is why they share one
refusal rather than each wording its own: a fix for one that left the others
would fix a symptom. The relational lane names the rewrite that reaches the
same number: declare the parameter over the dimension and supply it there
([#1137](https://github.com/fluxopt/lpspec/issues/1137)).

Finding that wall turned up a real disagreement behind it. `sum_back` read a
constant at a slot the variable was absent from, where every other operator
drops it. So the two lanes answered 2.5 and 3.0 on a file **neither** refused. A
reduction consumes its operand before any row exists, so absence has to be
pushed into the operand first. `sum` and `sum(by=)` did that and the window did
not. The fix gave the window the same pass, with a differential test over every
operator that moves along a dimension
([#1142](https://github.com/fluxopt/lpspec/issues/1142)).

Both walls are worth reading twice, because the shape is easy to mistake for a
language limit and is not one. A `LaneError` names the wall *and* the route
around it. That is the difference between the two classes.

**The lane takes the same data too.** It did not always
([#60](https://github.com/fluxopt/lpspec/issues/60)). A parameter is a parquet
path, any table exporting the Arrow PyCapsule protocol, a `pd.Series` carrying
its dimensions in an index, a `dict` or a sequence over one dimension, or one
number spread over every coordinate it covers (a coordinate is one point of the
dimensions a parameter spans). Neither lane reads an `xr.DataArray`: this
package reads tables and hands arrays back. A dimension index is any of those
tables too, under the dimension's own key in `sources`. Labels come from
`sources` or from what the file declares, and from exactly one of the two. A
dimension the file declares and the caller also supplies is refused by both
lanes in the same sentence. A dimension with none of the three has no index.
Both lanes refuse it in the same sentence again rather than derive it from the
parameters that span it. A parameter carries a label, never the set of labels
that exist, nor what a label maps to. The index also fixes the *order*, so pass
one wherever order matters.

So one `sources` mapping goes to either lane, and an import alone decides which
lane builds a file.

## Parts of linopy not taken

lpspec does not take array operations (`merge`, `reindex`, `stack`), the Python
modeling API, or the solver layer. The first is data prep
([the limits](https://math-spec.readthedocs.io/en/latest/reference/language/errors/#what-the-language-will-not-say)).
The second is [hard rule 5](architecture.md#hard-rules): the model is the file
you review and diff. The third is
[#106](https://github.com/fluxopt/lpspec/issues/106), where lpspec adopts
linopy's *design* for declared solver capabilities without adopting its code.

The modeling API is the one a reader arriving from linopy misses first. Two
notebook pages replace it. [Change a model](../interactive.ipynb) covers the
loops: `update` for new numbers, a longer table for more rows, a patched `dict`
for new math. [Fix, relax, remove](../lifecycle.ipynb) covers the verbs, which
are the same loops aimed at `fix`, `relax` and `remove_constraints`. Neither
replaces the *debugging*: an IIS. Both pages say so. A built row is read with
[`row`](../reference/api.md#reading-one-row), in linopy's own form.

Where linopy is ahead, and why none of it is a ceiling question, is the honest
snapshot in [the roadmap](roadmap.md#honest-snapshot).

What is *owed* to linopy rather than merely true of it is in
[prior art and credit](prior-art.md). The same page credits Calliope, whose
math language this surface is derived from.
