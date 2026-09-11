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
    answer/objective.parquet      how it terminated, and what it reached
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

A sweep's `answer/` is keyed one file per slice — `answer/objective/000000.parquet`
and so on — which is the layout [`spill_to=`](../reference/sweeps.md) already
writes.

## Compare cases solved apart

Every archive's `answer/objective.parquet` is one row, in the same columns
whoever wrote it. Concatenate them, labelling each row by the directory it came
from:

```python
import polars as pl
from pathlib import Path

table = pl.concat(
    [
        pl.scan_parquet(case / 'answer' / 'objective.parquet').select(pl.lit(case.name).alias('case'), pl.all())
        for case in sorted(Path('runs').iterdir())
    ]
).collect()
```

**Check the digests before you read the numbers.** `spec_digest` is a digest of
the model file an answer came back from. One distinct value across the table is
the claim that every row answered the same document:

```python
digests = table['spec_digest']
assert digests.null_count() == 0 and digests.n_unique() == 1, 'one model, or this compares nothing'
```

Ask that the digests are there as well as equal. A solve run off a lowered
`Program` has no document to digest, so it writes null. A table of nothing but
nulls has one distinct value while having checked nothing.

## Query an archive from a database

A directory archive is a tree of parquet files, so a query engine reads it
where it lands. DuckDB, on the study above:

```sql
select scenario, objective from 'study/answer/objective/*.parquet'
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

## What an archive will not take

**A model built from a lowered `Program`.** `lps.check` hands one back, and
lowering has no inverse, so there is no file to write:

```python
lps.solve(lps.check('dispatch.yaml'), sources, archive='case/')
```

```text
archive= holds the model as written — a path, a mapping or a Spec — and this
model was built from a lowered Program, which cannot be written back out as
one. Build it from what it was lowered from: whatever was handed to
lps.check() or math_spec.to_program(). The Program stays the argument that
solves.
```

**A sweep cut by a hand-built axis.** A list of `(key, sources)` is a set of
sources per slice, which are unrelated questions:

```text
archive= takes a sweep cut by EachCoordinate or EachWindow, which say how one
set of sources was cut and so how the archive can be re-run. A hand-built list
is a set of sources per slice, which are unrelated questions — archive one
solve each.
```

Both are refused before the first slice is solved, so nothing is written and
no solver time is spent on an archive you cannot have.
