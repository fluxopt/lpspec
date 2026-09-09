# Run a model

This page takes you from an installed package to a solved model and its answer
read back, in five steps. It assumes a terminal, Python 3.12 and nothing about
this project.

The model is the dispatch problem on [the home page](index.md): three
generators meet a load over six snapshots at least cost. What a model file may
contain is the language's to say, and the language is
[documented with itself](https://math-spec.readthedocs.io/en/latest/reference/language/).
This page uses the file as it stands.

## 1. Install

```bash
pip install lpspec
```

That brings polars, the HiGHS solver and the language. No other solver and no
dataframe library beyond polars is needed for this page.

## 2. Save the model

Copy the YAML from [the home page](index.md#the-whole-thing-in-one-model) into
a file called `dispatch.yaml` in your working directory. It is also the file
[`examples/dispatch.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/dispatch.yaml)
in the repository, and [its gallery page](examples/dispatch.md) shows the same
model written in linopy.

## 3. Check the file

```python
import lpspec as lps

program = lps.check('dispatch.yaml')
```

`check` reads the file, expands it and lowers it to a plan, with no data
attached. It returns a program, the lowered model, and raises if the file uses
something outside the language. A file that passes here builds with any data
that fits its declarations.

## 4. Attach the numbers and solve

The file declares three parameters, `p_max`, `cost` and `load`, and two
dimensions, `snapshot` and `generator`. `sources` supplies each by name, and a
parameter over one dimension is a frame with that dimension and a `value`
column:

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

The two bare sequences supply the coordinates of each dimension. Wind at 1 and
solar at 2 run first, and gas at 50 runs only at snapshot 3, where the load of 180
exceeds the 160 the other two can give. That is 20 units of gas at
50, and the rest at 1 or 2, so the total is 1920.

## 5. Read the answer back

```python
print(result.primal('p'))  # (snapshot, generator, value), one row per generator and snapshot
print(result.dual('power_balance'))  # (snapshot, value): the price of one more unit of load
```

Each answer is a polars frame keyed by the coordinates the declaration named.
The dual of `power_balance` is the marginal cost at each snapshot: the cost of
the last generator on, so 50 at snapshot 3 and 1 or 2 everywhere else.

To hand the same model to another tool instead of solving it, write it as a
file. The suffix picks the format:

```python
lps.write('dispatch.yaml', sources, 'dispatch.lp')
```

You now have a model that checks, solves and reads back. Everything after this
point is a choice, which is why it is on another page.

## Where next

| | |
|---|---|
| [Change a model](interactive.ipynb) | the next lesson: new numbers, more rows and new math on the model you just solved |
| [Preparing the data](examples/data.md) | your data arrives as files, not as the frames above; this is the recipe from one to the other |
| [The verbs](reference/api.md) · [The data contract](reference/data.md) | what every call takes and returns, and what attaching refuses |
| [Language reference](https://math-spec.readthedocs.io/en/latest/reference/language/) | what a file may contain, exactly, and [the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/) it stops at |
| [Examples](examples/index.md) | every model in the repository, and which constructs each exercises |
| [Roadmap](about/roadmap.md) | what is refused on purpose, with reasons |
