# Run a model

Five steps from `pip install` to an answer read back, on the dispatch model
of [the home page](index.md): three generators meet a load over four
snapshots at least cost. What a model file may contain is
[the language's](https://math-spec.readthedocs.io/en/latest/reference/language/)
to say.

## 1. Install

```bash
pip install specsolve
```

That brings polars, HiGHS and the language.

## 2. Save the model

Copy the YAML from [the home page](index.md#the-whole-thing-in-one-model)
into `dispatch.yaml`. It is also
[`examples/dispatch.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/dispatch.yaml)
in the repository.

## 3. Check the file

```python
import specsolve as sps

program = sps.check('dispatch.yaml')
```

`check` raises if the file uses something outside the language, and needs no
data to do it.

## 4. Attach the numbers and solve

The file declares three parameters and two dimensions. `sources` supplies
each by name. A parameter over one dimension is a table with that dimension
and a `value` column; a bare sequence supplies a dimension's labels:

```python
import polars as pl

generators = ['wind', 'solar', 'gas']
sources = {
    'p_max': pl.DataFrame({'generator': generators, 'value': [80.0, 0.0, 200.0]}),
    'cost': pl.DataFrame({'generator': generators, 'value': [10.0, 25.0, 50.0]}),
    'load': pl.DataFrame({'snapshot': range(4), 'value': [60.0, 120.0, 180.0, 90.0]}),
    'snapshot': range(4),
    'generator': generators,
}

result = sps.solve('dispatch.yaml', sources)
print(result.objective)  # 10500.0
```

Wind at 10 runs first, and gas at 50 covers what is left. Solar has no
capacity, so the `where: "p_max > 0"` on `p` built no column for it.

## 5. Read the answer back

```python
print(result.primal('p'))  # (snapshot, generator, value): eight rows, wind and gas at each snapshot
print(result.dual('power_balance'))  # (snapshot, value): the price of one more unit of load
```

Each answer is a polars table keyed by the declaration's labels. The dual is
the cost of the last generator on: 10 at snapshot 0, where wind alone covers
the load, and 50 at the other three.

To hand the model to another tool instead, write it. The suffix picks the
format:

```python
sps.write('dispatch.yaml', sources, 'dispatch.lp')
```

## Where next

| | |
|---|---|
| [Change a model](interactive.ipynb) | the next lesson: new numbers, more rows, new math |
| [Preparing the data](howto/data.md) | from files to the tables above |
| [The verbs](reference/api.md) · [The data contract](reference/data.md) | what every call takes, returns and refuses |
| [Language reference](https://math-spec.readthedocs.io/en/latest/reference/language/) · [the limits of the language](https://math-spec.readthedocs.io/en/latest/about/limits/) | what a file may contain, and where it stops |
| [Debug a wrong answer](howto/debug.md) | when it solves and the number is wrong, or it does not solve |
| [Examples](examples/index.md) | every model in the repository |
| [Roadmap](about/roadmap.md) | what is refused on purpose |
