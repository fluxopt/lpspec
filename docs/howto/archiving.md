# Archiving a solve

How to keep a solve: the spec, the data it was solved with, and what came
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
    sources.parquet               (run, source, digest) — what each of them is
    answer/objective.parquet      how it terminated, what it reached, when, and under what name
    answer/metrics.parquet        what the build and its solves took
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
`model.solve` and `lps.solve_over`. Each holds the spec, the data and the
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

**`load_archive` reads it whole.** The sources come back as the tables the
members hold, and the answer's frames are in memory. Nothing has to be kept
alive afterwards, and a zip needs nowhere to unpack: it goes to a scratch
directory that is gone by the time you get the value.

```python
case = lps.load_archive('case.zip')
```

Pass `into=` anyway when you want the extracted tree as well, to query with an
engine that reads parquet. A directory archive is read where it lies, so it
takes no `into=` and passing one is refused.

## Read one too big to hold

`scan_archive` is the same two values with nothing read. The sources come back
as the paths they now are, and each frame is read off disk at the call that
asks for it.

```python
sweep = lps.scan_archive('sweep.zip', 'sweep/')
sweep.answer.scan('p')  # read at the collect, one name at a time
```

**Scan the archive you will not read most of**, as well as the one that does
not fit: a load reads every name, a scan only the ones you ask for.

**What is scanned has to outlive what it reads off.** A zip needs an `into=`
you will keep.

| | `load_archive` | `scan_archive` |
|---|---|---|
| a source | the table the member holds | the path to it |
| the answer | frames in memory | read at the call that asks |
| a sweep's answer | held: `runs.primal('p')` | spilled: `runs.scan('p')` |
| a zip's `into=` | optional, and scratch without one | required, and kept |

`lps.load_result` / `lps.scan_result` and `lps.load_runs` / `lps.scan_runs` are
the same pair one level down, for an answer `result.save` or `runs.save` wrote.

## Read what a solve cost

`metrics` is a `Metrics`: how big the model was, how many solves the clocks
cover, and wall-clock seconds in each phase, as one value
([the attributes](../reference/api.md#diagnostics)).

```python
case = lps.load_archive('case/')

case.metrics.rows  # how big the model was
case.metrics.solves  # how many solves the clocks cover
case.metrics.build_seconds  # turning declarations into frames
```

This is the one part of an archive that re-solving cannot give back: the
clocks are of the machine that ran them.

**The row covers the model's whole life, and `solves` says how long that is.**
`lps.solve` builds the model it solves, so its archive reads `solves` of 1. A
model solved more than once before it was archived carries the sum:

```python
with lps.build('dispatch.yaml', sources) as model:
    model.solve()
    model.solve(archive='second/')  # solves: 2, and the clocks cover both
```

A sweep records each slice's own metrics as `runs.metrics` instead
([sweeps](../reference/sweeps.md)).

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
one slice at a time. `archive=` packs the whole sweep. Pass both and the spill
is what the archive packs, so a sweep too large to hold is archived without
ever being held:

```python
axis = lps.EachCoordinate('scenario')
lps.solve_over('dispatch.yaml', sources, axis, spill_to='work/', archive='sweep/')
```

The archive carries the axis as well, so the sweep runs again from the file
alone:

```python
sweep = lps.scan_archive('sweep/')

sweep.answer.scan('p')  # keyed by scenario, read at the collect
lps.solve_over(sweep.spec, sweep.sources, sweep.axis)
```

`scan_archive` reads a sweep back spilled, as `spill_to=` left it.
`load_archive` gives the held sweep where it fits, and `runs.primal('p')`
answers on that one.

A sweep's frames are keyed one file per slice — `answer/primal/p/000000.parquet`
and so on — which is the layout [`spill_to=`](../reference/sweeps.md) writes.
Its record is not: `answer/objective.parquet` holds every slice's row, one
file, as a solve's does. A spill writes that record per slice for a reason —
the objective file's existence is how a resumed sweep knows a slice finished.
An archive has no resume to serve. So one glob finds every run in a directory
of them, whether a solve or a sweep wrote it.

## Read a directory of them

Runs pile up, and a directory of archives is a table per glob: which runs
terminated how, what each cost, and which input changed between them. Every row
carries `run`, the archive's own name, so nothing reads the paths.

```python
import polars as pl

pl.read_parquet('runs/*/answer/objective.parquet').sort('solved_at')
```

The recipes are [reading a directory of runs](warehouse.md): comparing cases
solved apart, finding the input that moved, and reading the whole thing from a
query engine.

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
