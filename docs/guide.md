# Run a model

Five steps from `pip install` to an answer read back, on the dispatch model
of [the home page](index.md): three generators meet a load over six
snapshots at least cost. What a model file may contain is
[the language's](https://math-spec.readthedocs.io/en/latest/reference/language/)
to say.

## 1. Install

```bash
pip install lpspec
```

That brings polars, HiGHS and the language.

## 2. Save the model

Copy the YAML from [the home page](index.md#the-whole-thing-in-one-model)
into `dispatch.yaml`. It is also
[`examples/dispatch.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/dispatch.yaml)
in the repository.

## 3. Check the file

```python
import lpspec as lps

program = lps.check('dispatch.yaml')
```

`check` lowers the file to a plan with no data attached, and raises if the
file uses something outside the language.

## 4. Attach the numbers and solve

The file declares three parameters and two dimensions. `sources` supplies
each by name. A parameter over one dimension is a frame with that dimension
and a `value` column; a bare sequence supplies a dimension's coordinates:

```python
import polars as pl

generators = ['wind', 'solar', 'gas']
sources = {
    'p_max': pl.DataFrame({'generator': generators, 'value': [100.0, 60.0, 200.0]}),
    'cost': pl.DataFrame({'generator': generators, 'value': [1.0, 2.0, 50.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [80.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
    'snapshot': range(6),
    'generator': generators,
}

result = lps.solve('dispatch.yaml', sources)
print(result.objective)  # 1920.0
```

Wind at 1 and solar at 2 run first. Gas at 50 runs only at snapshot 3, where
the load of 180 exceeds the 160 the other two can give.

## 5. Read the answer back

```python
print(result.primal('p'))  # (snapshot, generator, value), one row per generator and snapshot
print(result.dual('power_balance'))  # (snapshot, value): the price of one more unit of load
```

Each answer is a polars frame keyed by the declaration's coordinates. The dual
is the cost of the last generator on: 50 at snapshot 3, 1 or 2 elsewhere.

To hand the model to another tool instead, write it. The suffix picks the
format:

```python
lps.write('dispatch.yaml', sources, 'dispatch.lp')
```

## Where next

| | |
|---|---|
| [Change a model](interactive.ipynb) | the next lesson: new numbers, more rows, new math |
| [Preparing the data](howto/data.md) | from files to the frames above |
| [The verbs](reference/api.md) · [The data contract](reference/data.md) | what every call takes, returns and refuses |
| [Language reference](https://math-spec.readthedocs.io/en/latest/reference/language/) · [the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/) | what a file may contain, and where it stops |
| [Examples](examples/index.md) | every model in the repository |
| [Roadmap](about/roadmap.md) | what is refused on purpose |
