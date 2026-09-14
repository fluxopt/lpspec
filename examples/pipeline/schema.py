"""Validate the entity tables before they are attached to the model.

lpspec checks the *structure* of what you attach — real labels, one row per
coordinate, dtypes, no null values (``docs/reference/data.md``). It cannot know
whether a readable number is *right*: a negative capacity, an efficiency above
one, or a scenario naming a generator that does not exist all attach and solve.

This module is the front line that catches those, in the caller's own terms,
with `patito <https://patito.readthedocs.io>`_ — polars-native schemas plus the
referential checks patito cannot express. ``validate`` raises before a single
number reaches lpspec.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class Generator(pt.Model):
    """One generating unit: what it can produce, what it costs, what it emits."""

    generator: str = pt.Field(unique=True)
    p_max: float = pt.Field(ge=0)
    cost: float = pt.Field(ge=0)
    co2: float = pt.Field(ge=0)


class Snapshot(pt.Model):
    """Demand at one dispatch period."""

    snapshot: int = pt.Field(unique=True)
    load: float = pt.Field(ge=0)


class Scenario(pt.Model):
    """One solve: a named set of overrides on the base data.

    ``outage`` and ``price_gen`` are blank in the CSV for "none"; they are read
    as null and checked against the generator names in :func:`validate`.
    """

    name: str = pt.Field(unique=True)
    load_scale: float = pt.Field(gt=0)
    outage: str | None = pt.Field(default=None)
    price_gen: str | None = pt.Field(default=None)
    price_scale: float = pt.Field(gt=0)


def load_tables(data_dir: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read the three input CSVs, coercing integer columns a schema wants as float.

    A blank cell is read as null rather than ``''``, and columns a spreadsheet
    left as whole numbers (``90``, not ``90.0``) are cast to float, so the schema
    below checks values rather than how the CSV happened to type them.
    """
    generators = pl.read_csv(f'{data_dir}/generators.csv').cast(
        {'p_max': pl.Float64, 'cost': pl.Float64, 'co2': pl.Float64}
    )
    snapshots = pl.read_csv(f'{data_dir}/load.csv').cast({'load': pl.Float64})
    scenarios = pl.read_csv(f'{data_dir}/scenarios.csv', null_values=['']).cast(
        {'load_scale': pl.Float64, 'price_scale': pl.Float64}
    )
    return generators, snapshots, scenarios


def validate(
    generators: pl.DataFrame, snapshots: pl.DataFrame, scenarios: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Hold the entity tables to their domain rules, and stop on the first table that fails.

    Args:
        generators: The generator table, one row per unit.
        snapshots: The load table, one row per dispatch period.
        scenarios: The scenario table, one row per solve.

    Returns:
        The same three tables, unchanged, once every rule passes.

    Raises:
        patito.exceptions.DataFrameValidationError: A column-level rule failed —
            a negative capacity, a repeated name, a non-positive load scale.
        ValueError: A scenario names an ``outage`` or ``price_gen`` that is not a
            generator — a reference patito cannot check on its own.
    """
    Generator.validate(generators)
    Snapshot.validate(snapshots)
    Scenario.validate(scenarios)

    known = set(generators['generator'])
    strays = []
    for row in scenarios.iter_rows(named=True):
        for field in ('outage', 'price_gen'):
            name = row[field]
            if name is not None and name not in known:
                strays.append(f"scenario '{row['name']}' {field}='{name}'")
    if strays:
        raise ValueError(
            'scenario(s) name a generator that does not exist: '
            + '; '.join(strays)
            + f'. Known generators: {sorted(known)}.'
        )
    return generators, snapshots, scenarios
