# The data contract

This page lists what `sources` may hold and what attaching refuses and accepts,
for anyone putting data on a [spec](glossary.md). `sources` supplies the
numbers, keyed by the names the spec declares:

```python
import lpspec as lps

result = lps.solve(
    'dispatch.yaml',
    {'load': 'load.parquet', 'cost': cost_frame, 'p_max': p_max_frame},
)
```

Getting from the files an instance arrives in to these shapes is
[preparing the data](../howto/data.md).

## What a parameter accepts

For a parameter declared `dims: [d1, d2]`, the value under its key is one of:

- a **parquet path**;
- **a table exposing the Arrow PyCapsule protocol** (polars, pandas, pyarrow,
  duckdb) with columns `d1, d2, value`;
- an **`int` or `float`**, one value for every
  [coordinate](glossary.md#the-data) the parameter covers;
- a **`dict`** of label to value, for a parameter over one dimension;
- a **sequence** (list, tuple, `np.ndarray`) for a parameter over one
  dimension, positional against that dimension's index.

**The last three shapes serve models written out in Python.** Each is dense,
and attach materialises it: one number over `(snapshot, generator)` becomes one
row per pair. Declare a constant as `dims: []` instead. A sequence carries no
labels, so its dimension's index comes from one of the three sources under
[where coordinates come from](#where-coordinates-come-from).

**A `pd.Series` is unwrapped first.** Its one dimension is its index rather
than a column. Attach unwraps it only if pandas is already imported. An unnamed
index attaches to the declared dimension. A named index attaches by that name,
and a name outside the declared dimensions raises.

**A `MultiIndex` is refused.** A parameter over two dimensions arrives as a
table with both as columns. `series.reset_index()` is the whole change.

**An `xr.DataArray` is refused.** Pass `array.to_series().reset_index()`.
`Result.to_dataarray()` is the way back out.

**The [linopy lane](../about/linopy.md#3-it-is-a-lane) reads every shape on
this list**, so one `sources` mapping goes to either
[lane](glossary.md#how-it-runs).

**Nothing on this path imports pandas, xarray or linopy on your behalf.**

## Where coordinates come from

**Each dimension's index is resolved before any parameter loads**, from a key
in `sources` named after the dimension. That key holds a table with a column of
that name, a parquet path, or a bare sequence of the labels. The first
occurrence of each label is its position, and that order is what
[`shift`](https://math-spec.readthedocs.io/en/latest/reference/language/operators/#shift)
reads positionally.

**A dimension nothing supplies raises.** Attach never reads labels out of the
parameters. Which labels an axis has is data's to say, and that rule is
[the language's](https://math-spec.readthedocs.io/en/latest/reference/language/dimensions/).

**A map goes under
[the lookup's own name](https://math-spec.readthedocs.io/en/latest/reference/language/dimensions/#the-map-is-supplied-under-the-lookups-own-name)**,
as a table of the rows it has. Attach reads it against the labels the index
supplied: a label no row mentions is unmapped, and a key matching no label is
refused as a typo.

## What attaching refuses and accepts

**A coordinate has a value, or it has no row.** Attach refuses a row whose
value is null or NaN. Polars and parquet write a hole as a null, pandas has
only NaN, and `None` in a pandas column is NaN by the time either lane sees it.

### Refused

| What arrives | What the message says |
|---|---|
| a declared parameter with no data | names the parameter |
| a source nothing can be read as a table from | names the shapes that are read |
| an `xr.DataArray` | names `to_series().reset_index()` |
| a `pd.Series` with a `MultiIndex` | names the table and the `reset_index()` that gets there |
| a `dims: []` parameter whose source has more than one row | one value broadcast everywhere has one row |
| a dict or a sequence for a parameter over more than one dimension | each runs along one dimension |
| a sequence whose length is not the dimension's | positional, so one entry per label |
| a sequence for a dimension nothing else supplies labels for | names the three ways to supply them |
| a key naming neither a parameter, a dimension nor a lookup | names the near miss |
| a lookup relation short of either column | names the pair, and what each is |
| a lookup relation with a null in its value column | a map is partial by omitting a row |
| a lookup relation mapping one label twice | a lookup is single-valued |
| a map with both authors, or neither | names them, and says which way out |
| an index carrying a column named after a lookup over it | names the key it belongs under |
| a table missing a declared dimension column, or `value` | names the columns needed |
| a `value` column carrying a null or a NaN | names the parameter and the coordinates |
| a label outside the dimension's index | names the parameter and the strays |
| two rows for one coordinate | |
| a lookup with two values for one label | |
| a lookup value that is not a label of its target | one wording, checked once for both lanes |
| a dimension carrying lookups with no index | |
| a dimension nothing can supply labels for | names both ways to fix it |
| a dimension the spec declares and the caller also supplies | names the declaration and the colliding key |
| a lookup whose map the spec declares and the caller also supplies | names the map and the colliding column |
| a declared map whose labels nothing supplies | names the map, and asks only for the labels |
| a declared map keyed by something the labels do not carry | names the lookup and the strays |
| a column that is not the declared `dtype` | names both, and the declaration the data would satisfy |
| a divisor with no value where the model divides by it | names the parameter and how many rows ([absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/)) |
| a comparison's whole constant side with no value where the row is built | the same, naming the constraint |
| a bound parameter with no value where the variable exists | names both models the two repairs build |

### Accepted

| What arrives | What happens |
|---|---|
| an undeclared column in a table | ignored |
| a coordinate with no row | sparse variables; what a missing row means where it is read is [absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/). `diagnostics().sparse_parameters` names the parameters that arrived short of their dims ([api](api.md#diagnostics)) |
| a value that is readable and wrong | bound as given |

### Stray labels

**The index is what makes a stray label a stray.**

```python
cost = {'wind': 1.0, 'gsa': 2.0}  # 'gas' misspelled — refused by name
```

A dimension whose labels came from the parameters would read `gsa` as a third
generator.

## Growing or replacing the data

**A built model takes new numbers with
[`update`](api.md#re-solving-with-new-numbers).** A sweep over slices of one
dimension is [`solve_over`](sweeps.md). Both attach through the rules above.

**The [linopy lane](../about/linopy.md#the-same-language-and-the-same-data)
attaches by these same rules, refusals included**, held to them by
`tests/test_data_parity.py`. The same malformed source gets the same verdict,
and where one defect has one repair, the same message.
