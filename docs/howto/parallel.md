# Running a sweep in parallel

One slice of a [sweep](../reference/sweeps.md) per worker, on a pool you
start. The reference for `executor=` is
[running slices in parallel](../reference/sweeps.md#running-slices-in-parallel).

## Start the pool with `spawn`

A forked worker hangs, so pass a `spawn` or `forkserver` context and guard
the entry point:

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import lpspec as lps


def main():
    ctx = multiprocessing.get_context('spawn')  # or 'forkserver'
    with ProcessPoolExecutor(4, mp_context=ctx) as pool:
        runs = lps.solve_over('spec.yaml', sources, lps.EachCoordinate('scenario'), executor=pool)
    print(runs.objective)


if __name__ == '__main__':  # spawn re-imports your module; without this it recurses
    main()
```

Four workers hold four slices at once, so the pool wants four times the
memory of one slice.

## Hand the workers paths

A parquet path the workers can reach stays a path. A path they cannot reach,
and every in-memory table, crosses as parquet bytes. So for a cluster, write
the sources to a filesystem the workers mount and pass the paths:

```python
sources = {'load': '/shared/load.parquet', 'cost': '/shared/cost.parquet', 'p_max': '/shared/p_max.parquet'}
runs = lps.solve_over('spec.yaml', sources, lps.EachCoordinate('scenario'), executor=client, workers_share_fs=True)
```

`workers_share_fs=True` says the mount is there; without it the paths are
read here and shipped.

## Read the sweep back

`runs` reads as it does for a serial sweep, one column wider:

```python
runs.objective  # (scenario, status, termination_condition, objective, has_primal, model)
runs.primal('p')  # (scenario, snapshot, generator, value)
runs.diagnostics  # one row per slice; every slice loaded its own solver
```

A sweep with a `carry` cannot run in parallel, because each slice reads the
one before it.
