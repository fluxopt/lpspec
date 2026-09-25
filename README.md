# specsolve

<!-- --8<-- [start:badges] -->

[![CI](https://img.shields.io/github/actions/workflow/status/fluxopt/specsolve/ci.yml?style=flat-square&branch=main)](https://github.com/fluxopt/specsolve/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/specsolve.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/specsolve/)
[![Python](https://img.shields.io/pypi/pyversions/specsolve?logo=python&logoColor=white&style=flat-square)](https://pypi.org/project/specsolve/)
[![Docs](https://readthedocs.org/projects/specsolve/badge/?version=latest&style=flat-square)](https://specsolve.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat-square)](https://github.com/fluxopt/specsolve/blob/main/LICENSE)

<!-- --8<-- [end:badges] -->

**Solve an optimisation model written in YAML. Attach your data as tables, and
get a loaded solver with no LP file in between.**

<!-- --8<-- [start:intro] -->

specsolve builds and solves [math-spec](https://github.com/energy-models/math-spec)
models. The file states the math, and math-spec checks it before any data
exists. specsolve attaches your tables, builds the model on polars, and hands it
to HiGHS, Gurobi or Xpress. The same file can also build a `linopy.Model`
([linopy](https://specsolve.readthedocs.io/en/latest/about/linopy/)).

<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:benefits] -->

- **Straight to the solver.** YAML and tables in, a loaded solver out. At 10M
  variables it loads HiGHS faster than linopy does, with less peak memory.
  [Benchmarks →](https://specsolve.readthedocs.io/en/latest/about/benchmarks/)
- **Tidy, sparse tables.** Every parameter, variable and constraint is a tidy
  table, with one row per coordinate that exists. A mask is an absent row, not
  a NaN in a dense array.
  [Architecture →](https://specsolve.readthedocs.io/en/latest/about/architecture/)
- **Checked against somebody else.** Every ported model matches an optimum from
  GAMS, PyPSA, OSeMOSYS, OR-Library or TSPLIB, and its duals where the
  reference records them.
  [The models →](https://specsolve.readthedocs.io/en/latest/examples/)
- **Your tables in, tables out.** Pass polars, pandas or pyarrow objects, or
  parquet paths. Results come back as tables, with bridges to pandas and xarray.
  [Your data →](https://specsolve.readthedocs.io/en/latest/howto/data/)

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
