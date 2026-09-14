"""A typed, validated data model for a power network, feeding lpspec's `transport.yaml`.

    pixi run python examples/energy_model/run.py

**This is a pattern, not a feature.** lpspec deliberately holds no
domain vocabulary: a model is YAML over dimensions and parameters, and its
door (`sources.py`) checks the *shape* of the data — every parameter present,
one value per coordinate, real labels, no nulls, the declared dtype, lookups
single-valued and pointing at labels that exist. What it does not check, and
[says it will not](https://github.com/fluxopt/lpspec/blob/main/docs/about/roadmap.md),
is anything about *energy systems*: that a capacity is non-negative, that a
reverse limit is signed the way the model reads it, that a bus carrying load
can actually be reached over the lines. Those are domain facts, and the roadmap
files them under "domain helpers" — a thing that lives above lpspec, in a layer
of your own.

This is that layer, in one file and one dependency (polars, which lpspec brings
already). `Network` holds the components as tables, `validate()` refuses the
domain mistakes lpspec structurally cannot see, and `to_sources()` lowers the
validated components into the `sources` mapping `lps.solve` takes — including
completing the load grid, because lpspec refuses a constraint's constant side
that is present at one bus and absent at another.

**Three refusals carry the point**, each a mistake lpspec would solve around
silently — a confident answer to the wrong problem, never an error:

- a **range** error — a negative cost, bound as given and quietly wrong;
- a **sign** error — a reverse transmission limit passed positive, which forces
  flow instead of bounding it;
- an **islanded** bus — load with no path to any generator, whose balance row
  lpspec never builds, so the demand is silently left unserved.

For a richer schema half — column types, ranges and foreign keys as
declarations rather than hand-written checks — a table-first validator drops in
where `_check_schema` sits: `patito` and `pandera` both speak polars, and
`pydantic` speaks rows. The domain half below stays yours either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

import lpspec as lps

HERE = Path(__file__).parent
MODEL = HERE.parent / 'transport.yaml'


class DomainError(ValueError):
    """A network that is well-formed as tables but wrong as a power system."""


#: What each component table must carry, and the polars dtype each column is.
#: The dimension columns are strings (`snapshot` an int), matching the
#: `transport.yaml` declarations the sources attach against.
SCHEMA: dict[str, dict[str, pl.DataType]] = {
    'buses': {'bus': pl.String()},
    'generators': {'generator': pl.String(), 'bus': pl.String(), 'p_max': pl.Float64(), 'cost': pl.Float64()},
    'lines': {
        'line': pl.String(),
        'from_bus': pl.String(),
        'to_bus': pl.String(),
        'cap': pl.Float64(),
        'neg_cap': pl.Float64(),
    },
    'load': {'snapshot': pl.Int64(), 'bus': pl.String(), 'value': pl.Float64()},
}


@dataclass(frozen=True)
class Network:
    """A power network as four component tables, validated before it becomes a model.

    Attributes:
        buses: One row per node — `(bus,)`.
        generators: `(generator, bus, p_max, cost)`, a unit on a bus.
        lines: `(line, from_bus, to_bus, cap, neg_cap)`, a link between two buses.
        load: `(snapshot, bus, value)`, demand where it is non-zero — the rest
            of the snapshot-by-bus grid is completed with zero on the way out.
    """

    buses: pl.DataFrame
    generators: pl.DataFrame
    lines: pl.DataFrame
    load: pl.DataFrame

    def validate(self) -> Network:
        """Refuse a network that is malformed as tables or wrong as a system.

        Returns self, so `Network(...).validate()` reads as one expression.

        Raises:
            DomainError: A missing or mistyped column, a duplicate id, a value
                out of range or wrongly signed, a component referencing a bus
                that does not exist, a self-looping line, or a bus that carries
                load but no generator can reach over the lines.
        """
        tables = {'buses': self.buses, 'generators': self.generators, 'lines': self.lines, 'load': self.load}
        for name, table in tables.items():
            _check_schema(name, table)

        _check_unique('buses', self.buses, 'bus')
        _check_unique('generators', self.generators, 'generator')
        _check_unique('lines', self.lines, 'line')

        _check_nonneg('generators', self.generators, ('p_max', 'cost'))
        _check_nonneg('lines', self.lines, ('cap',))
        _check_nonpos('lines', self.lines, ('neg_cap',))

        buses = set(self.buses['bus'])
        _check_references('generators', self.generators, {'bus': buses})
        _check_references('lines', self.lines, {'from_bus': buses, 'to_bus': buses})
        _check_references('load', self.load, {'bus': buses})

        loops = self.lines.filter(pl.col('from_bus') == pl.col('to_bus'))['line'].to_list()
        if loops:
            raise DomainError(f'line(s) {loops} start and end on the same bus — a line joins two buses')

        self._check_connected()
        return self

    def _check_connected(self) -> None:
        """Every bus with load can reach a generator over the lines.

        A bus nothing reaches contributes no term to any balance, so lpspec
        builds no balance row for it at all: the solve succeeds and the load
        there is simply never served — no error, no infeasibility, an answer
        that looks fine and quietly meets less demand than was asked. Lines are
        undirected here: reachability is the same both ways.
        """
        reachable = _reachable_from_generators(self.buses['bus'].to_list(), self.lines, self.generators)
        stranded = self.load.filter((pl.col('value') > 0) & ~pl.col('bus').is_in(reachable))['bus'].unique().to_list()
        if stranded:
            raise DomainError(
                f'bus(es) {sorted(stranded)} carry load but no generator can reach them over the lines. '
                f'lpspec builds no balance row for a bus nothing reaches, so this load is silently left '
                f'unserved and the solve looks fine.'
            )

    def to_sources(self) -> dict[str, pl.DataFrame]:
        """Lower the validated components into the `sources` mapping `transport.yaml` takes.

        Dimensions, the three lookups (`gen_bus`, `line_from`, `line_to`) as
        the relations lpspec reads under their own names, one tidy `(dims…,
        value)` table per parameter, and the load completed to the full
        snapshot-by-bus grid with zero where none was given.
        """
        snapshots = self.load.select('snapshot').unique().sort('snapshot')
        grid = snapshots.join(self.buses.select('bus'), how='cross')
        load = grid.join(self.load, on=['snapshot', 'bus'], how='left').with_columns(pl.col('value').fill_null(0.0))
        return {
            'snapshot': snapshots,
            'bus': self.buses.select('bus'),
            'generator': self.generators.select('generator'),
            'line': self.lines.select('line'),
            'gen_bus': self.generators.select('generator', 'bus'),
            'line_from': self.lines.select('line', pl.col('from_bus').alias('bus')),
            'line_to': self.lines.select('line', pl.col('to_bus').alias('bus')),
            'p_max': self.generators.select('generator', pl.col('p_max').alias('value')),
            'cost': self.generators.select('generator', pl.col('cost').alias('value')),
            'cap': self.lines.select('line', pl.col('cap').alias('value')),
            'neg_cap': self.lines.select('line', pl.col('neg_cap').alias('value')),
            'load': load,
        }

    def solve(self) -> lps.Result:
        """Validate, then hand the lowered sources to lpspec's relational lane."""
        return lps.solve(str(MODEL), self.validate().to_sources())


def _check_schema(name: str, table: pl.DataFrame) -> None:
    """Every declared column present and of its declared dtype."""
    want = SCHEMA[name]
    if missing := [c for c in want if c not in table.columns]:
        raise DomainError(f"'{name}' is missing column(s) {missing}; it carries {table.columns}")
    if wrong := {c: str(table.schema[c]) for c, dt in want.items() if table.schema[c] != dt}:
        wanted = {c: str(dt) for c, dt in want.items() if c in wrong}
        raise DomainError(f"'{name}' has column(s) of the wrong type: {wrong} — wanted {wanted}")


def _check_unique(name: str, table: pl.DataFrame, key: str) -> None:
    """An id column names each row once."""
    dupes = table.group_by(key).len().filter(pl.col('len') > 1)[key].to_list()
    if dupes:
        raise DomainError(f"'{name}' names {sorted(dupes)} more than once in '{key}' — an id is one row")


def _check_nonneg(name: str, table: pl.DataFrame, columns: tuple[str, ...]) -> None:
    """A quantity that has no physical negative — a capacity, a cost."""
    for column in columns:
        offenders = table.filter(pl.col(column) < 0)
        if offenders.height:
            first = offenders.row(0, named=True)
            raise DomainError(f"'{name}' has {column} < 0 (e.g. {column}={first[column]}) — it must be >= 0")


def _check_nonpos(name: str, table: pl.DataFrame, columns: tuple[str, ...]) -> None:
    """A reverse limit the model reads as a lower bound, so it is signed <= 0."""
    for column in columns:
        offenders = table.filter(pl.col(column) > 0)
        if offenders.height:
            first = offenders.row(0, named=True)
            raise DomainError(
                f"'{name}' has {column} > 0 (e.g. {column}={first[column]}) — a reverse limit is a lower "
                f'bound on flow, so it must be <= 0; passing it positive forces flow rather than bounding it'
            )


def _check_references(name: str, table: pl.DataFrame, columns: dict[str, set[str]]) -> None:
    """Every value in a foreign-key column is a bus that exists."""
    for column, allowed in columns.items():
        strays = sorted(set(table[column].to_list()) - allowed)
        if strays:
            raise DomainError(f"'{name}'.{column} references bus(es) {strays} that no row of 'buses' declares")


def _reachable_from_generators(buses: list[str], lines: pl.DataFrame, generators: pl.DataFrame) -> set[str]:
    """The buses in the same connected component as some generator, lines undirected."""
    parent = {b: b for b in buses}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in lines.iter_rows(named=True):
        parent[find(row['from_bus'])] = find(row['to_bus'])
    roots = {find(b) for b in generators['bus']}
    return {b for b in buses if find(b) in roots}


#: A small two-bus grid: cheap and peaking plant on bus A, free wind on bus B,
#: one line between them, three snapshots of rising demand.
VALID = Network(
    buses=pl.DataFrame({'bus': ['A', 'B']}),
    generators=pl.DataFrame(
        {
            'generator': ['cheap', 'peak', 'wind'],
            'bus': ['A', 'A', 'B'],
            'p_max': [100.0, 100.0, 40.0],
            'cost': [10.0, 80.0, 0.0],
        }
    ),
    lines=pl.DataFrame({'line': ['L1'], 'from_bus': ['A'], 'to_bus': ['B'], 'cap': [100.0], 'neg_cap': [-100.0]}),
    load=pl.DataFrame({'snapshot': [0, 1, 2], 'bus': ['B', 'B', 'B'], 'value': [50.0, 90.0, 130.0]}),
)


def _with(base: Network, **tables: pl.DataFrame) -> Network:
    """A copy of *base* with some component tables swapped — for the broken cases."""
    return Network(**{**base.__dict__, **tables})


#: Three well-formed-as-tables networks that are wrong as power systems, each
#: labelled by the class of mistake the domain layer catches.
BROKEN: dict[str, Network] = {
    'range': _with(
        VALID,
        generators=VALID.generators.with_columns(
            pl.when(pl.col('generator') == 'peak').then(-5.0).otherwise(pl.col('cost')).alias('cost')
        ),
    ),
    'sign': _with(VALID, lines=VALID.lines.with_columns(neg_cap=pl.lit(30.0))),
    'islanded': _with(
        VALID,
        buses=pl.DataFrame({'bus': ['A', 'B', 'C']}),
        load=pl.concat([VALID.load, pl.DataFrame({'snapshot': [0], 'bus': ['C'], 'value': [20.0]})]),
    ),
}


def _rejection(net: Network) -> str:
    """The domain layer's verdict on a network, as one line."""
    try:
        net.validate()
    except DomainError as exc:
        return str(exc)
    return 'NOT REFUSED — the check missed it'


def main() -> None:
    result = VALID.solve()

    generation = result.primal('p').group_by('generator').agg(pl.col('value').sum()).sort('generator')
    flow = result.primal('f').sort('snapshot', 'line')
    total_gen = float(generation['value'].sum())
    total_load = float(VALID.to_sources()['load']['value'].sum())
    assert abs(total_gen - total_load) < 1e-6, 'generation must meet load — the network has no losses'

    print('A two-bus grid: three generators, one line, three snapshots of rising demand.')
    print(
        f'validated: {VALID.buses.height} buses, {VALID.generators.height} generators, '
        f'{VALID.lines.height} line, {VALID.load.height} load rows given'
    )
    print()
    print(f'solved: objective {result.objective:.2f}, generation {total_gen:.1f} meets load {total_load:.1f}')
    print()
    print('generation by unit')
    for row in generation.iter_rows(named=True):
        print(f'  {row["generator"]:<6} {row["value"]:>7.1f}')
    print("line flow towards its 'to' bus")
    for row in flow.iter_rows(named=True):
        print(f'  s{row["snapshot"]} {row["line"]:<3} {row["value"]:>7.1f}')
    print()

    print('Three mistakes lpspec would not catch, refused here before it builds:')
    for label, net in BROKEN.items():
        print(f'  {label:<9} {_rejection(net)}')


if __name__ == '__main__':
    main()
