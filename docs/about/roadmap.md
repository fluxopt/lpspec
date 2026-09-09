# Roadmap

This page says why lpspec exists, where it is going and what it will not
become. Read it before you propose a feature. **No work items live here.** The
work is issues, grouped under three parents:

- [Track 1 — primitives](https://github.com/fluxopt/lpspec/issues/470)
- [Track 2 — the operational surface](https://github.com/fluxopt/lpspec/issues/471)
- [Track 3 — capabilities, and the degree line](https://github.com/fluxopt/lpspec/issues/472)

An index maintained by hand beside an issue tracker is a second copy that
drifts, so there is none here. The issues are the list, and this page is the
argument for what the list is *for*.

## Why

An optimisation model is math, and math is worth reading. It usually arrives
as Python that *builds* math. The equations are entangled with the loops, the
frames (tables) and the library that assembled them. A diff then shows
scaffolding rather than constraints, and nothing can read the model except the
program that wrote it. Reviewing such a model means reviewing a program, which
is a different and much harder job.

lpspec makes the math the artifact. A YAML file says what the variables,
constraints and objective *are*. The file is validated at load time and built
at runtime. Someone who understands the math can review it without
understanding the builder. That is the whole thesis, and every rule below
follows from it.

## Where it is going

**One language, more than one place to run it.** The same file builds natively
on the relational engine or onto a `linopy.Model` that already exists in
memory. That is not a fallback and not a dialect. It is one language, so a
differential test between the two lanes (the two ways a file is executed,
[glossary](../reference/glossary.md#how-it-runs)) is an oracle rather than a
comparison.

**A build that streams, with a ceiling you can declare.** The model is frames
and the build is relational, so nothing dense is ever materialised. Peak memory
tracks the model rather than a number someone guessed. What is missing is the
*declaration*: there is no way to say "build this within N gigabytes or fail".
The honest version is partition-wise execution, which the locality closure
already guarantees is safe.

**A solve that explains itself.** A solved model should tell you why it is
infeasible, what a row costs, and what changed since the last solve. It should
do so without opening a file no editor can hold. Most of that is a query over
frames that already exist.

**Component libraries, composed rather than generated.** A fixed set of
parametrised templates agree on a port/flow convention and merge into one
program (a lowered spec, [glossary](../reference/glossary.md#the-chain))
before a single build pass. **Topology is data.** Wiring a system is rows in a
connectivity table, never generated YAML. So structure stays bounded by the
number of component *types*, while cardinality lives entirely in data.

## What it will not become

**Two durable losses, and they are the price of the closed AST (abstract
syntax tree).** One is structure that needs the solver's *answer* to decide
the next row, inside one plan. The other is imperative modelling of any kind.
That price buys load-time validation, two lanes on one language, and a build
that streams. Everything else is scheduling.

The specific refusals are in
[the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/#deliberate-non-primitives),
with the reason and the rewrite for each: data prep, arbitrary array ops,
domain helpers, normalisation, in-plan conditionals, a Python modelling API.
Read them before proposing a feature. Parity with another tool is not by
itself a reason to add anything.

## Honest snapshot

**Cheaper here, because the model is tables:** IIS (irreducible infeasible
subsystem) read-back, a join rather than a scatter; serialisation to parquet;
elastic relaxation; dualisation, since transposing a COO matrix is swapping two
column names. Model statistics and coefficient ranges were the first of these
and already ship. `diagnostics()` gives the range *per declaration*, taken as
each block is built. That is what naming the badly scaled one costs when the
matrix arrives a declaration at a time.

**Ahead of comparable declarative layers:** a sparse-by-construction build with
no dense intermediate, and a hand-off straight to the solver rather than
through a file; parametrised `macros:` ([Calliope](prior-art.md)'s
sub-expressions take no arguments); binary and integer variables; piecewise as
N links with per-link signs, convex mode and `active` gating; load-time
validation of every expression, `where` string and *uncalled* macro template.

**Behind linopy**, and none of it a ceiling question: the post-solve object
(labelled DataArrays vs tidy tables; `to_dataarray` bridges), debugging (an
IIS via Gurobi), lifecycle (mutate, re-solve, warm start, `relax`/`fix`),
solver breadth (ten backends and four handoffs vs three direct sinks plus
files, a sink being where a built model lands), and the variable types and
constraint kinds the capability model still gates.

**The ranking this implies:** indexed access blocks whole model classes today;
the operational verbs block using the engine at 3am; solver breadth blocks
arrival from linopy at all; semi-continuous and `cumsum`-over-data are cheap,
unblocked and unscheduled.
