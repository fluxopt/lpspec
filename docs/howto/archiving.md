# Archiving a solve

How to keep a solve: the model, the data it was solved with, and what came
back. Later you can read the answer, ask the question again, or hand both to
someone else. Every recipe here is one argument on a verb you already call.

## Archive as you solve

```python
import lpspec as lps

lps.solve('dispatch.yaml', sources, archive='case/')
```

That writes `model.yaml`, one `sources/<key>.parquet` per key the file
declares, and `answer/` holding everything the solve produced:

```text
case/
    model.yaml
    sources/cost.parquet
    sources/load.parquet
    …
    sources.parquet               (source, digest) — what each of them is
    answer/objective.parquet      how it terminated, what it reached, when, and under what name
    answer/diagnostics.parquet    what the build and its solves spent
    answer/primal/p.parquet       one file per variable
    answer/dual/power_balance.parquet
```

**The suffix picks the container.** Anything without a `.zip` suffix is a
directory, as above. `.zip` packs the same members into one file, which is
what you send or store:

```python
lps.solve('dispatch.yaml', sources, archive='case.zip')
```

**Three verbs take `archive=`, and nothing else writes one**: `lps.solve`,
`model.solve` and `lps.solve_over`. Each holds the model, the data and the
answer at the moment you ask, so the three are written together and cannot be
paired up wrongly ([the verbs](../reference/api.md#archiving-a-model)).

## Read it back

```python
case = lps.load_archive('case/')

case.answer.objective  # what it reached
case.answer.primal('p')  # the values it came back with
lps.solve(case.spec, case.sources)  # the same question, asked again
```

`case.spec` and `case.sources` are the pair every verb takes, so asking again
is the call you made the first time.

## Read what a solve cost

`diagnostics` is one row: how big the model was, how many solves the archive
covers, and wall-clock seconds in each phase.

```python
case = lps.load_archive('case/')
# columns, rows, nonzeros, sink_columns, sink_rows, solves, loads, attach, build, handoff, solve, write, run
case.diagnostics
```

This is the one part of an archive that re-solving cannot give back. The
clocks are of the machine that ran them, so nothing recovers them later.

**The row covers the model's whole life, and `solves` says how long that is.**
`lps.solve` builds the model it solves, so its archive reads `solves` of 1 and
the clocks are that answer's own. A model solved more than once before it was
archived carries the sum:

```python
with lps.build('dispatch.yaml', sources) as model:
    model.solve()
    model.solve(archive='second/')  # solves: 2, and the clocks cover both
```

**A phase the build never entered writes zero**, so cases that ran different
phases still concatenate into one table.

**A sweep records the same columns per slice**, read as `runs.diagnostics` and
archived in one `answer/diagnostics.parquet` as a solve's is. A fold knows
where one slice's share of the clocks begins. A single `Result` does not: it is
one solve of a model that may have had many, so it carries no such number.

**A zip needs somewhere to unpack.** Nothing is read at load. The sources come
back as paths, and each frame is read off disk only when you ask for it. A
directory's files are already where a read needs them; a zip's are not.

```python
case = lps.load_archive('case.zip', 'case/')
```

Name a directory you can write to and will keep, because the answer reads off
it as you use it. A directory archive needs none, and passing one is refused.

## Keep the answer an update produced

`update` puts new numbers on a built model, and what the model answers from
then on is the merge. The file it was built from has not changed, so nothing
outside the call can tell the new question from the old one. Archive it where
it is answered:

```python
with lps.build('dispatch.yaml', sources) as model:
    model.update({'p_max': doubled}).solve(archive='case/')
```

## Archive a sweep too large to hold

A sweep holds every slice's answers until it is done, unless you spill it.
`spill_to=` writes each slice's frames as the fold goes, so the sweep holds
one slice at a time. `archive=` packs the whole study. Pass both and the spill
is what the archive packs, so a sweep too large to hold is archived without
ever being held:

```python
axis = lps.EachCoordinate('scenario')
lps.solve_over('dispatch.yaml', sources, axis, spill_to='work/', archive='study/')
```

The archive carries the axis as well, so the study runs again from the file
alone:

```python
study = lps.load_archive('study/')

study.answer.scan('p')  # keyed by scenario
lps.solve_over(study.spec, study.sources, study.axis)
```

A sweep's frames are keyed one file per slice — `answer/primal/p/000000.parquet`
and so on — which is the layout [`spill_to=`](../reference/sweeps.md) writes.
Its record is not: `answer/objective.parquet` holds every slice's row, one
file, as a solve's does. A spill writes that record per slice for a reason —
the objective file's existence is how a resumed sweep knows a slice finished.
An archive has no resume to serve. So one glob finds every run in a directory
of them, whether a solve or a sweep wrote it.

## Compare cases solved apart

Every archive's `answer/objective.parquet` holds the same columns whoever
wrote it, and two of them say which run each row came from. `run` is the
archive's own name. `solved_at` is when the solver returned. The rows carry
their own labels, so nothing has to read the paths:

```python
import polars as pl

table = pl.read_parquet('runs/*/answer/objective.parquet')
table.sort('solved_at').select('run', 'status', 'objective')
```

A sweep's archive lands in the same table, one row per slice, with its key
column beside `run`. `answer/diagnostics.parquet` carries `run` the same way,
so what each slice cost is attributable across a warehouse too. Read the two
together with `pl.concat(..., how='diagonal')` where a warehouse holds both.

**Check the digests before you read the numbers.** `spec_digest` is a digest of
the model file an answer came back from. Every answer carries one, so a single
distinct value across the table is the claim that every row answered the same
document:

```python
assert table['spec_digest'].n_unique() == 1, 'one model, or this compares nothing'
```

## Find which input changed between two runs

`spec_digest` says the two answered the same document. It says nothing about
the numbers, so two runs of one model over different data carry the same one.
`sources.parquet` is what separates them: `(source, digest)`, one row per
member of `sources/`.

```python
base = lps.load_archive('base/')
other = lps.load_archive('halved/')

moved = base.source_digests.join(other.source_digests, on='source', suffix='_other').filter(
    pl.col('digest') != pl.col('digest_other')
)
moved['source'].to_list()  # ['load']
```

The answer is the name of the input, not merely that something moved. That is
what per-source rows buy over one digest of everything.

**The digest is of the bytes the archive holds**, so a reader can check it
against the archive alone: hash `sources/load.parquet` and you get the row
back. Two archives of the same data written by different versions of polars
can still differ, because what is digested is the parquet, not the meaning of
the table.

**Reading an archive does not verify them.** That would be a pass over every
byte of data the archive holds, on every load. Hash the members yourself on
the occasion you want it checked.

## Query an archive from a database

A directory archive is a tree of parquet files, so a query engine reads it
where it lands. DuckDB, on the study above:

```sql
select scenario, objective from 'study/answer/objective.parquet'
where has_primal order by objective;
```

```text
┌──────────┬───────────┐
│ scenario │ objective │
├──────────┼───────────┤
│ lo       │       7.0 │
│ hi       │      17.0 │
└──────────┴───────────┘
```

Every frame is tidy: the model's own dimension columns, and a `value` column.
So `study/answer/primal/p/*.parquet` is the variable `p` over every slice, and
joins to the rest on those columns. A zip has to go through `load_archive`
first, because no query engine reads inside one.

The cost rows read the same way, and carry `run` as the record does. Over a
directory of archives they are the table that says which cases are growing,
and where the time goes:

```sql
select run, rows, nonzeros, build, solve from 'runs/*/answer/diagnostics.parquet'
order by build + solve desc;
```

## What an archive will not take

**A sweep cut by a hand-built axis.** A list of `(key, sources)` is a set of
sources per slice, which are unrelated questions:

```text
archive= takes a sweep cut by EachCoordinate or EachWindow, which say how one
set of sources was cut and so how the archive can be re-run. A hand-built list
is a set of sources per slice, which are unrelated questions — archive one
solve each.
```

Refused before the first slice is solved, so nothing is written and no solver
time is spent on an archive you cannot have. A directory that already holds
something is refused the same way, rather than merged into.

Nothing else is out of reach. Every model a verb accepts can be archived,
because every verb reads a model through one door and that door takes only
what has a document behind it — a path, a mapping or a `Spec`, never the
lowered `Program` `lps.check` hands back.
