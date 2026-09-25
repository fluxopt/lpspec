# specsolve

<!-- --8<-- [start:badges] -->

[![CI](https://img.shields.io/github/actions/workflow/status/fluxopt/specsolve/ci.yml?style=flat-square&branch=main)](https://github.com/fluxopt/specsolve/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/specsolve.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/specsolve/)
[![Python](https://img.shields.io/pypi/pyversions/specsolve?logo=python&logoColor=white&style=flat-square)](https://pypi.org/project/specsolve/)
[![Docs](https://readthedocs.org/projects/specsolve/badge/?version=latest&style=flat-square)](https://specsolve.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat-square)](https://github.com/fluxopt/specsolve/blob/main/LICENSE)

<!-- --8<-- [end:badges] -->

**Solve an optimisation model written in YAML. Attach your data as tables, and
keep the solver loaded for quick updates and warm starts.**

<!-- --8<-- [start:intro] -->

specsolve builds and solves [math-spec](https://github.com/energy-models/math-spec)
models. The file states the math, and math-spec checks it before any data
exists. specsolve attaches your tables, builds the model on polars, and hands it
to HiGHS, Gurobi or Xpress. The same file can also build a `linopy.Model`
([linopy](https://specsolve.readthedocs.io/en/latest/about/linopy/)).

<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:benefits] -->

- **Tables in, tables out.** Pass any Arrow table, such as polars, pandas or
  DuckDB, or a parquet path. Results come back as tables, and an archive keeps
  the model, its data and its results as parquet, ready for queries, plots or
  BI. [Tables in, tables out →](https://specsolve.readthedocs.io/en/latest/tables/)
- **Sweeps and rolling horizons built in.** One call runs scenario sweeps,
  rolling horizons and myopic pathways over the same model. Each window is
  checked against how the model couples before it runs. [Sweep a model →](https://specsolve.readthedocs.io/en/latest/sweep/)
- **Fast, and hard to get wrong.** Tables hold only the rows that exist, so a
  model's topology does not change its cost. The solver stays loaded:
  `update()` puts new numbers on it, and `keep='progress'` warm-starts from the
  last run. The API is a handful of verbs, with nothing to tune.
  [Benchmarks →](https://specsolve.readthedocs.io/en/latest/about/benchmarks-scaling.html)
- **Validated against PyPSA.** PyPSA's model is one file here, grown rung by
  rung through storage, unit commitment, multi-period and stochastic runs. All
  16 rungs match PyPSA's objective, and 12 match its duals row for row.
  [The PyPSA ladder →](https://specsolve.readthedocs.io/en/latest/examples/pypsa_ladder/)

<!-- --8<-- [end:benefits] -->

## Example

<!-- --8<-- [start:model] -->
```yaml
# dispatch.yaml
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  p_max: {dims: [generator]}
  load:  {dims: [snapshot]}
  cost:  {dims: [generator]}
variables:
  p:
    dims: [snapshot, generator]
    where: "p_max > 0"
    bounds: {lower: 0, upper: p_max}
constraints:
  power_balance:
    dims: [snapshot]
    expression: sum(p, over=generator) == load
objective:
  sense: minimize
  expression: sum(p * cost)
```
<!-- --8<-- [end:model] -->

<!-- --8<-- [start:solve] -->

```python
import specsolve as sps, polars as pl

generators = ['wind', 'solar', 'gas']
sources = {
    'p_max': pl.DataFrame({'generator': generators, 'value': [100.0, 60.0, 200.0]}),
    'cost': pl.DataFrame({'generator': generators, 'value': [1.0, 2.0, 50.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [80.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
    'snapshot': range(6),
    'generator': generators,
}

result = sps.solve('dispatch.yaml', sources)
print(result.objective)  # 1920.0
print(result.primal('p'))
print(result.dual('power_balance'))
```

<!-- --8<-- [end:solve] -->

## Documentation

The documentation is at <https://specsolve.readthedocs.io>. What a file may
contain is math-spec's
[language reference](https://math-spec.readthedocs.io/en/latest/reference/language/).

## Installation

```bash
pip install specsolve
```

That brings polars, HiGHS and the language. Add the `[linopy]` extra for the
linopy lane and the pandas and xarray bridges, and `[gurobi]` or `[xpress]` for
those solvers. To work on specsolve, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Prior art

The YAML surface comes from [Calliope](https://github.com/calliope-project/calliope),
and [linopy](https://github.com/PyPSA/linopy) supplies the vocabulary, the
oracle and every benchmark denominator.
[Prior art and credit](docs/about/prior-art.md) says what came from each.

## Status

Alpha, pre-1.0.

<!-- --8<-- [start:status] -->

**Breaking changes land without a deprecation cycle.** Pin an exact version if
you depend on this, and read the
[changelog](https://github.com/fluxopt/specsolve/blob/main/CHANGELOG.md) before
upgrading. A retired spelling fails at load and names its rewrite. Real models
round-trip through solve and are tested against linopy. The accepted surface
is not yet frozen.

<!-- --8<-- [end:status] -->

## Licence

[MIT](LICENSE).
