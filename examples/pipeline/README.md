# A dispatch pipeline

An end-to-end study on lpspec, from CSV files to a report — the shape a real
model is run in, small enough to read in one sitting and meant to be copied.

```bash
pip install lpspec patito
python examples/pipeline/run.py
```

## The five stages

`run.py` is the driver; each stage is a function in it.

1. **Load** — read the entity tables from `data/`.
2. **Validate** — hold them to domain rules in `schema.py` before a number
   reaches lpspec: capacities and costs non-negative, names unique, and every
   scenario naming a generator that exists. This is the layer lpspec leaves to
   you — it checks structure, not whether a readable number is right.
3. **Solve** — attach the data to `model.yaml` and solve once per scenario in
   `data/scenarios.csv`. Each scenario is a set of overrides: a load scale, a
   unit taken offline, a fuel price shock.
4. **Export** — write the dispatch, prices and a per-scenario summary to `out/`.
5. **Report** — render a self-contained `out/report.html` dashboard.

## The files

| File | What it is |
|---|---|
| `model.yaml` | the model — dispatch with a carbon read-out. The math, no data. |
| `data/generators.csv` | one row per unit: capacity, marginal cost, carbon intensity |
| `data/load.csv` | demand at each snapshot |
| `data/scenarios.csv` | one row per solve; blank `outage`/`price_gen` means none |
| `schema.py` | the patito schemas and the referential checks |
| `run.py` | the pipeline, and the report generator |
| `run.out` | the committed output, so a change that breaks the run shows in the diff |

## Make it yours

- **Change the numbers** — edit the CSVs and re-run. Break a rule (a negative
  capacity, a scenario naming a generator that does not exist) to see the
  validation stop it.
- **Add a scenario** — one row in `data/scenarios.csv`.
- **Change the model** — edit `model.yaml`. Adding a parameter means adding its
  column and a line in `schema.py` and `base_sources`.
- **Swap the validator** — `schema.py` is the only patito-specific file; the rest
  is lpspec and polars.

The workflow behind this, and where the boundary between your data model and
lpspec's own checks falls, is
[validating data before attaching it](../../docs/howto/validate.md).
