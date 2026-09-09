# Roadmap

This page says why lpspec exists, where it is going and what it will not
become, for anyone about to propose a feature. **No work items live here.** A
hand-maintained index beside an issue tracker is a second copy that drifts, so
the issues are the list, grouped under three parents, and this page is the
argument for what the list is *for*:

- [Track 1 — primitives](https://github.com/fluxopt/lpspec/issues/470)
- [Track 2 — the operational surface](https://github.com/fluxopt/lpspec/issues/471)
- [Track 3 — capabilities, and the degree line](https://github.com/fluxopt/lpspec/issues/472)

## Why

An optimisation model is math worth reading, and it usually arrives as Python
that *builds* math. The equations are entangled with the loops, the
[frames](../reference/glossary.md#the-data) and the library that assembled
them. A diff then shows scaffolding rather than constraints, nothing can read
the model except the program that wrote it, and reviewing the model means
reviewing a program.

lpspec makes the math the artifact: a YAML file says what the variables,
constraints and objective *are*, validated at load time and built at runtime.
Someone who understands the math can review it without understanding the
builder. Every rule below follows from that.

## Where it is going

**One language, more than one place to run it.** The same file builds natively
on the relational engine or onto a `linopy.Model` that already exists in
memory, neither a fallback nor a dialect. So a differential test between the
two [lanes](../reference/glossary.md#how-it-runs) is an oracle rather than a
comparison.

**A build that streams, with a ceiling you can declare.** The model is frames
and the build is relational, so nothing dense is ever materialised and peak
memory tracks the model rather than a number someone guessed. What is missing
is the *declaration*: there is no way to say "build this within N gigabytes or
fail". The honest version is partition-wise execution, which the locality
closure already guarantees is safe.

**A solve that explains itself.** A solved model should tell you why it is
infeasible, what a row costs and what changed since the last solve, without
opening a file no editor can hold. Most of that is a query over frames that
already exist.

**Component libraries, composed rather than generated.** A fixed set of
parametrised templates agree on a port/flow convention and merge into one
[program](../reference/glossary.md#the-chain) before a single build pass.
**Topology is data.** Wiring a system is rows in a connectivity table, never
generated YAML, so structure stays bounded by the number of component *types*
while cardinality lives in data.

## What it will not become

**Two durable losses, and they are the price of the closed AST (abstract
syntax tree):** structure that needs the solver's *answer* to decide the next
row inside one plan, and imperative modelling of any kind. That price buys
load-time validation, two lanes on one language, and a build that streams.
Everything else is scheduling.

The specific refusals, each with its reason and its rewrite, are in
[the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/#deliberate-non-primitives):
data prep, arbitrary array ops, domain helpers, normalisation, in-plan
conditionals, a Python modelling API. Parity with another tool is not by
itself a reason to add anything.

## Honest snapshot

**Cheaper here, because the model is tables:** IIS (irreducible infeasible
subsystem) read-back, a join rather than a scatter; serialisation to parquet;
elastic relaxation; dualisation, since transposing a COO matrix is swapping two
column names. Model statistics and coefficient ranges were the first of these
and already ship: `diagnostics()` gives the range *per declaration*, taken as
each block is built, which is what naming the badly scaled one costs when the
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
solver breadth (ten backends and four handoffs vs three direct
[sinks](../reference/glossary.md#how-it-runs) plus files), and the variable
types and constraint kinds the capability model still gates.

**The ranking this implies:** indexed access blocks whole model classes today;
the operational verbs block using the engine at 3am; solver breadth blocks
arrival from linopy at all; semi-continuous and `cumsum`-over-data are cheap,
unblocked and unscheduled.
