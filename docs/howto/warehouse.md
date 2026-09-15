# Reading a directory of runs

How to read many archives at once: which runs terminated how, what each cost,
and which input changed between them. One archive is
[archiving a solve](archiving.md); this is the directory they pile up in.

## The three tables

An archive is a tree of parquet files, so a directory of them is a table per
glob. Nothing is loaded and no schema is maintained. Over a directory of
solves:

```python
import polars as pl

answers = pl.read_parquet('runs/*/answer/objective.parquet')
metrics = pl.read_parquet('runs/*/answer/metrics.parquet')
inputs = pl.read_parquet('runs/*/sources.parquet')
```

| glob | one row per | says |
|---|---|---|
| `answer/objective.parquet` | solve, or sweep slice | how it terminated, what it reached, when, under what name |
| `answer/metrics.parquet` | the same | what the build and its solves spent, and how big the model was |
| `sources.parquet` | source per archive | what each input's bytes digest to |

**Every row says which archive it came from.** `run` is the archive's own name
— `runs/nightly-2026-09-10.zip` writes `nightly-2026-09-10` — and it is on all
three, so nothing has to read the paths.

**A directory holding both solves and sweeps does not glob.** A sweep's record
carries the dimension its axis cut on, and its metrics carry a different set of
columns from a solve's — `loaded` in place of `solves`, `loads`,
`added_columns`, `added_rows` and `write_seconds`. polars refuses the mismatch
rather than guessing:

```text
SchemaError: extra column in file outside of expected schema: scenario
```

Union by name instead. The columns one side lacks come back null:

```python
from glob import glob

answers = pl.concat(
    [pl.read_parquet(file) for file in sorted(glob('runs/*/answer/objective.parquet'))],
    how='diagonal',
)
```

`sources.parquet` is the same three columns whoever wrote it, so that one globs
either way.

**Use the readers for one archive, not for many.** `load_archive` and
`scan_archive` give back a spec, its sources and an answer, which is what
re-running a case needs. A warehouse question is a query over the parquet, and
these globs never build a model.

## Compare cases solved apart

`solved_at` is when the solver returned, so a table concatenated from runs
solved on different machines still orders:

```python
table = pl.read_parquet('runs/*/answer/objective.parquet')
table.sort('solved_at').select('run', 'status', 'objective')
```

**Check the digests before you read the numbers.** `spec_digest` is a digest of
the spec an answer came back from. One distinct value across the table is the
claim that every row answered the same document:

```python
assert table['spec_digest'].n_unique() == 1, 'one model, or this compares nothing'
```

A solve run off a lowered `Program` has no document and carries `None`, which
counts as its own value. A table where *every* digest is null therefore counts
one distinct value while having checked nothing, so ask for them to be present
as well as to agree:

```python
assert table['spec_digest'].null_count() == 0, 'and every row named the document it answered'
```

## Find which input changed between two runs

`spec_digest` says two runs answered the same document. It says nothing about
the numbers, so two runs of one model over different data carry the same one.
`sources.parquet` is what separates them:

```python
import lpspec as lps

base = lps.load_archive('runs/base/')
other = lps.load_archive('runs/halved/')

moved = base.source_digests.join(other.source_digests, on='source', suffix='_other').filter(
    pl.col('digest') != pl.col('digest_other')
)
moved['source'].to_list()  # ['load']
```

The answer is the name of the input, not merely that something moved. That is
what per-source rows buy over one digest of everything.

**Across a whole directory it is a window rather than a join**, `run` being on
every row:

```python
inputs = pl.read_parquet('runs/*/sources.parquet')
inputs.sort('run').with_columns(before=pl.col('digest').shift().over('source')).filter(
    pl.col('before').is_not_null() & (pl.col('before') != pl.col('digest'))
)
```

**The digest is of the bytes the archive holds**, so a reader can check it
against the archive alone: hash `sources/load.parquet` and you get the row
back. Two archives of the same data written by different versions of polars can
still differ, because what is digested is the parquet, not the meaning of the
table.

**Reading an archive does not verify them.** That would be a pass over every
byte of data the archive holds, on every read. Hash the members yourself on the
occasion you want it checked.

## See what the runs cost

`answer/metrics.parquet` carries the model's size beside the clocks, so one
query says which cases are growing and where the time goes:

```python
metrics.select('run', 'rows', 'nonzeros', 'build_seconds', 'solve_seconds').sort(
    pl.col('build_seconds') + pl.col('solve_seconds'), descending=True
)
```

This is the one part of an archive that re-solving cannot give back: the clocks
are of the machine that ran them. The thirteen columns are
[the metrics](../reference/api.md#diagnostics).

**A sweep records a subset of them**, one row per slice, keyed by the axis and
stamped with `run` like any other: its metrics are a `SliceMetrics`
([reading a sweep](../reference/sweeps.md#reading-a-sweep)). A fold knows each
slice's share of the clocks, which is why it reports `loaded` where a solve's
row reports `solves` and `loads`.

## Query it from a database

A directory archive is parquet where it lies, so a query engine reads it
without polars in the way. DuckDB, and `union_by_name` is what takes the two
kinds of archive together:

```sql
select run, rows, nonzeros, build_seconds, solve_seconds
from read_parquet('runs/*/answer/metrics.parquet', union_by_name = true)
order by build_seconds + solve_seconds desc;
```

**Only the three tables above carry `run`.** The value frames do not:
`answer/primal/p.parquet` is the model's own dimension columns and a `value`
column, and nothing in it says which archive it came from. Across a directory
they need the engine's own filename column, and a join that leaves it out
crosses the runs without complaining:

```sql
select run, snapshot, generator, value
from (select *, regexp_extract(filename, 'runs/([^/]+)/', 1) as run
      from read_parquet('runs/*/answer/primal/p.parquet', filename = true))
order by run, snapshot;
```

Inside one archive there is nothing to attribute. Every frame is tidy, so the
values join to the sources they were solved from on the coordinates both carry:

```sql
select p.snapshot, p.generator, p.value, load.value as load
from 'runs/base/answer/primal/p.parquet' p
join 'runs/base/sources/load.parquet' load using (snapshot);
```

**A zip has to be unpacked first**, because no query engine reads inside one.
Give either reader an `into=` and query what lands there.
