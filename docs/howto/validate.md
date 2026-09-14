# Validate data before attaching it

Catch a wrong number in your entity tables before it reaches a solve. This is
for anyone whose data arrives from a spreadsheet, another tool or a pipeline.
The tables you validate here are the ones [preparing the data](data.md) turns
into the `sources` mapping the verbs take.

## What attach already refuses

Attach checks the *structure* of every source, so a data model in front of it
does not repeat these. The [data contract](../reference/data.md) lists them all.
Three matter at scale:

- **A stray label is refused, not absorbed.** A misspelled `generator` raises.
  The index says which labels exist, so a typo cannot pass as a new unit.
- **A parameter is one value per [coordinate](../reference/glossary.md).** Two
  rows for one generator raise. So do a null value, a `NaN`, and a column of the
  wrong dtype.
- **A [lookup](https://math-spec.readthedocs.io/en/latest/reference/language/dimensions/#lookups)
  value must be a real label of its target.** This is the one referential check
  attach makes, and the topology section below builds on it.

What attach cannot know is whether a readable number is *right*. A negative
capacity, an efficiency above one, or a minimum above a maximum each attaches
and solves. A schema over your entity tables catches those.

## Check the entity tables

Put a schema on the wide entity table, where the attributes still sit side by
side. Validate it before the projection to one table per parameter.
[patito](https://patito.readthedocs.io) is polars-native, so a schema is column
types plus the domain rules attach leaves to you:

```python
import lpspec as lps
import patito as pt
import polars as pl


class Generators(pt.Model):
    generator: str = pt.Field(unique=True)
    p_max: float = pt.Field(ge=0)
    cost: float = pt.Field(ge=0)


generators = pl.read_csv('examples/ports/data/dispatch/generators.csv')
load = pl.read_csv('examples/ports/data/dispatch/load.csv')

Generators.validate(generators)  # raises on a negative capacity or a repeated name
```

A failure names the column and the offending rows, pointing at the file you fix.
Then project the validated tables and solve. The projection is
[preparing the data](data.md):

```python
sources = {
    'snapshot': load.select('snapshot'),
    'generator': generators.select('generator'),
    'p_max': generators.select('generator', pl.col('p_max').alias('value')),
    'cost': generators.select('generator', pl.col('cost').alias('value')),
    'load': load,
}
result = lps.solve('dispatch.yaml', sources)
```

## Refuse related columns that disagree

Some rules span two columns of one row: a minimum below a maximum, a ramp within
a capacity. Write them as a field constraint — a polars expression the whole
column must satisfy:

```python
class Generators(pt.Model):
    generator: str = pt.Field(unique=True)
    p_min: float = pt.Field(ge=0)
    p_max: float = pt.Field(ge=0, constraints=pl.col('p_max') >= pl.col('p_min'))
```

Attach cannot make this check. The projection splits the entity into one table
per parameter, and `p_min` and `p_max` only meet again at the solve. The wide
table is where they still share a row.

## Let lpspec check topology, by making it a lookup

Membership is referential: the bus a generator sits on, the two buses a line
joins. Attach already checks a lookup against the labels that exist. So declare
the relation as a lookup, not a plain column:

```yaml
<!-- doctest: wrap=lookups -->
gen_bus:
  description: the bus a generator sits on
  over: generator
  into: bus
```

Supply it under its own key, as a table of the rows it maps.
[Transport](../examples/transport.md) is the model this comes from. A value that
is not a declared `bus` label is refused:

```python
gen_bus = pl.DataFrame({
    'generator': ['wind_n', 'gas_s'],
    'bus': ['north', 'nrth'],  # 'south' misspelled
})
lps.solve('transport.yaml', sources | {'gen_bus': gen_bus})
```

```
dimension 'generator' lookup 'gen_bus' has value(s) that are not 'bus' labels:
'nrth'. Every value must be a declared 'bus' label — otherwise sum(by=gen_bus)
drops those terms in the join that places them, and the model builds and solves
without them.
```

Keep the bus as a column instead, and attach never reads it as a reference. Then
the check is yours to make:

```python
buses = pl.read_csv('buses.csv')
assert generators['bus'].is_in(buses['bus']).all(), 'every generator sits on a declared bus'
```

The lookup is the shorter path, and its check runs on both
[lanes](../reference/glossary.md).

## Where each check runs

- **Domain rules run upstream, on the wide entity table.** Bounds, ranges, units
  and cross-column agreement need the numbers and the whole entity. A schema
  library states them once and reports every violation together.
- **Structural and referential rules run at attach.** Real labels, one row per
  coordinate, dtypes, and lookups against known labels — the
  [data contract](../reference/data.md), on both lanes.
- **Sparsity is reported, not refused.** A table that lost rows builds a smaller
  model. Read `diagnostics().sparse_parameters` after a
  [build](../reference/api.md#diagnostics) to see which parameters arrived short
  of their dimensions:

```python
model = lps.build('dispatch.yaml', sources)
model.diagnostics().sparse_parameters
```
