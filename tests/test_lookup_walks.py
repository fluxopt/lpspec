"""A lookup is a relation, walked in the direction each call names — on both lanes.

The declaration keeps one claim, the ``key:`` the table is single-valued per,
and every call says which columns it consumes and produces. The models here
are the ones the language reference writes the forms for: a generator's zone
that changes by period, walked from either key; one table with two value
columns landed on a product or read as a pair; a value column consumed, so
a coarse quantity spreads onto its keys; a produced dimension the operand
already carries, which is a masked sum; two ends of a line in one table with
roles; a self-map; a bare relation; and a partition grouped per scenario.

Every construct is checked differentially: the relational lane joins the
relation on the columns the walk names where the eager lane groups, selects
or masks by the relation's incidence, and nothing but the agreement shows
that the three readings are one.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from tests.conftest import by_coord
from tests.differential import both_lanes_refuse, differential

GENERATORS = ['g1', 'g2']
ZONES = ['z1', 'z2']
PERIODS = [2030, 2050]

#: The model math-spec#161 was filed for: ``zone_balance`` groups ``p`` by a
#: zone that is read at the row's own period.
ZONAL = {
    'dimensions': {'generator': {'dtype': 'str'}, 'zone': {'dtype': 'str'}, 'period': {'dtype': 'int'}},
    'lookups': {'zone_of': {'over': ['generator', 'period', 'zone'], 'key': ['generator', 'period']}},
    'parameters': {'cost': {'dims': ['generator']}, 'demand': {'dims': ['zone', 'period']}},
    'variables': {'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0}}},
    'constraints': {
        'zone_balance': {'foreach': ['zone', 'period'], 'expression': 'sum(p, by=zone_of, from=generator) >= demand'}
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

#: ``g1`` is the cheap generator. In 2030 it serves ``z1``; in 2050 it moves
#: to ``z2``, and ``z1``'s demand triples.
HONEST = [('g1', 2030, 'z1'), ('g2', 2030, 'z2'), ('g1', 2050, 'z2'), ('g2', 2050, 'z1')]
FROZEN_AT_2030 = [('g1', 2030, 'z1'), ('g2', 2030, 'z2'), ('g1', 2050, 'z1'), ('g2', 2050, 'z2')]


def _zone_of(rows: list[tuple[str, int, str]]) -> pl.DataFrame:
    """The relation as it is supplied: one column per declared column, named after it."""
    return pl.DataFrame(rows, schema=['generator', 'period', 'zone'], orient='row')


def _zonal_sources(zone_of: list[tuple[str, int, str]]) -> dict:
    return {
        'generator': GENERATORS,
        'zone': ZONES,
        'period': PERIODS,
        'zone_of': _zone_of(zone_of),
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 5.0]}),
        'demand': pl.DataFrame(
            {'zone': ['z1', 'z2', 'z1', 'z2'], 'period': [2030, 2030, 2050, 2050], 'value': [10.0, 10.0, 30.0, 10.0]}
        ),
    }


def _fixed(spec: dict, values: dict[str, list[float]]) -> tuple[dict, dict]:
    """*spec* with ``p`` pinned by its bounds to *values* per generator over the periods, and the sources that pin it.

    A walk is easiest to read off a quantity nothing else moves, so the
    constraints under test equate a fresh variable to the walk and the
    assertion is on that variable.
    """
    pinned = {
        **spec,
        'parameters': {**spec['parameters'], 'fixed': {'dims': ['generator', 'period']}},
        'variables': {
            **spec['variables'],
            'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 'fixed', 'upper': 'fixed'}},
        },
    }
    rows = [(g, e, v) for g, per in values.items() for e, v in zip(PERIODS, per, strict=True)]
    fixed = pl.DataFrame(rows, schema=['generator', 'period', 'value'], orient='row')
    return pinned, {'fixed': fixed}


# ---------------------------------------------------------------------------
# a two-key table, walked from either key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('zone_of', 'objective'),
    [
        pytest.param(HONEST, 220.0, id='the-zone-moves-with-the-period'),
        pytest.param(FROZEN_AT_2030, 140.0, id='the-map-frozen-at-2030'),
    ],
)
def test_a_zone_that_changes_by_period_is_read_at_the_row_s_own_period(zone_of, objective):
    """The two numbers that tell a two-key table from a map frozen at one period.

    Frozen, the cheap generator serves ``z1``'s tripled 2050 demand and the
    model costs 140; honestly mapped, it has moved to ``z2`` and the dear one
    serves the 30 units at 5 apiece. ``period`` is the key column not walked,
    so it is joined on: the group at 2050 reads the 2050 row.
    """
    with differential(ZONAL, _zonal_sources(zone_of)) as run:
        assert run.result.objective == pytest.approx(objective), (
            'the group at 2050 has to read the 2050 map, not the 2030 one — both lanes agree, on the wrong number'
        )


def test_the_same_table_is_walked_from_its_other_key():
    """``sum(p, by=zone_of, from=period)`` consumes ``period``, joins on ``generator`` and lands on ``zone``.

    A generator's history per zone: what it produced over the periods it sat
    in that zone. The values are distinct, so a walk that consumed the wrong
    key or joined on nothing lands on a number this table does not hold.
    """
    spec, pins = _fixed(ZONAL, {'g1': [1.0, 2.0], 'g2': [3.0, 4.0]})
    spec = {
        **spec,
        'variables': {**spec['variables'], 'h': {'foreach': ['generator', 'zone'], 'bounds': {'lower': 0}}},
        'constraints': {
            'history': {'foreach': ['generator', 'zone'], 'expression': 'h == sum(p, by=zone_of, from=period)'}
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(h)'},
    }
    with differential(spec, _zonal_sources(HONEST) | pins) as run:
        held = by_coord(run.result, 'h', 'generator', 'zone')
    assert held == {('g1', 'z1'): 1.0, ('g1', 'z2'): 2.0, ('g2', 'z1'): 4.0, ('g2', 'z2'): 3.0}, (
        "each generator's output summed over the periods it spent in each zone"
    )


def test_both_key_columns_are_consumed_at_once_by_a_from_list():
    """``from=[generator, period]`` is ``sum(sum(p, by=zone_of, from=generator), over=period)`` said once."""
    spec, pins = _fixed(ZONAL, {'g1': [1.0, 2.0], 'g2': [3.0, 5.0]})
    spec = {
        **spec,
        'variables': {**spec['variables'], 't': {'foreach': ['zone'], 'bounds': {'lower': 0}}},
        'constraints': {
            'total': {'foreach': ['zone'], 'expression': 't == sum(p, by=zone_of, from=[generator, period])'}
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(t)'},
    }
    with differential(spec, _zonal_sources(HONEST) | pins) as run:
        held = by_coord(run.result, 't', 'zone')
    assert held == {'z1': 6.0, 'z2': 5.0}, 'z1 holds g1 in 2030 and g2 in 2050; z2 the other two'


def test_a_generator_in_two_zones_in_one_period_is_refused_at_bind():
    """Single-valued per ``(generator, period)`` — the claim the key makes, held at bind.

    The refusal names the key tuple rather than a label: ``g1`` is mapped
    once per period everywhere else.
    """
    message = both_lanes_refuse(ZONAL, _zonal_sources([*HONEST, ('g1', 2030, 'z2')]), 'more than once')
    assert "('g1', 2030)" in message, 'the refusal names the (generator, period) key that maps twice'
    assert 'single-valued per (generator, period)' in message, 'and says what the table is single-valued per'


def test_a_group_no_member_reaches_in_one_period_is_an_empty_sum_there():
    """A zone with no generator at some period holds the empty sum on a constant side.

    ``need`` is grouped through the same table ``p`` is, and ``z3`` has a
    member in 2030 only — so at 2050 its right-hand side is 0 rather than a
    hole. ``spare`` keeps a term in every row, so the row stands to be
    counted: one per ``(zone, period)`` on both lanes, the empty groups
    included.
    """
    spec = {
        **ZONAL,
        'parameters': {**ZONAL['parameters'], 'need': {'dims': ['generator', 'period']}},
        'variables': {**ZONAL['variables'], 'spare': {'foreach': ['zone', 'period'], 'bounds': {'lower': 0}}},
        'constraints': {
            'zone_balance': {
                'foreach': ['zone', 'period'],
                'expression': 'sum(p, by=zone_of, from=generator) + spare >= sum(need, by=zone_of, from=generator)',
            }
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(p * cost) + 100 * sum(spare)'},
    }
    sources = {
        **_zonal_sources([('g1', 2030, 'z3'), ('g2', 2030, 'z2'), ('g1', 2050, 'z2'), ('g2', 2050, 'z1')]),
        'zone': [*ZONES, 'z3'],
        'need': pl.DataFrame(
            {'generator': ['g1', 'g2', 'g1', 'g2'], 'period': [2030, 2030, 2050, 2050], 'value': [4.0, 6.0, 8.0, 2.0]}
        ),
    }
    with differential(spec, sources) as run:
        assert run.result.objective == pytest.approx(4.0 + 30.0 + 8.0 + 10.0), (
            'each generator covers its own need at its own zone, and the empty groups bind nothing'
        )
        assert run.engine.diagnostics().rows == 6, 'one row per (zone, period), the empty 2050 z3 group included'


# ---------------------------------------------------------------------------
# a value column consumed: at, and the sum that is the same read
# ---------------------------------------------------------------------------


PRICE = pl.DataFrame(
    {'zone': ['z1', 'z2', 'z1', 'z2'], 'period': [2030, 2030, 2050, 2050], 'value': [1.0, 2.0, 3.0, 4.0]}
)


def test_at_reads_the_zone_this_generator_sat_in_that_period():
    """``at(price, by=zone_of, into=generator)`` consumes ``zone``, joins ``period`` and produces ``generator``.

    Every value is distinct, so a pullback that read the wrong period — the
    map's first, say, or a broadcast over every period — lands on a number
    this table does not hold at that coordinate.
    """
    spec = {
        **ZONAL,
        'parameters': {**ZONAL['parameters'], 'price': {'dims': ['zone', 'period']}},
        'constraints': {
            'at_least': {'foreach': ['generator', 'period'], 'expression': 'p >= at(price, by=zone_of, into=generator)'}
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    with differential(spec, {**_zonal_sources(HONEST), 'price': PRICE}) as run:
        held = by_coord(run.result, 'p', 'generator', 'period')
    assert held == {('g1', 2030): 1.0, ('g2', 2030): 2.0, ('g1', 2050): 4.0, ('g2', 2050): 3.0}, (
        'g1 reads z1 in 2030 and z2 in 2050, g2 the other way round — the price of its own zone at its own period'
    )


def test_a_sum_consuming_a_value_column_is_the_same_read_and_a_carried_dimension_is_read_pointwise():
    """Two spellings of one number: the fan-out sum, and a pullback whose operand already carries ``generator``.

    ``sum(price, by=zone_of, from=zone, into=generator)`` lands ``price`` on
    every generator in the zone — one term per coordinate, since the key
    determines the zone, so it is the read ``at`` makes. ``at(price * p, ...)``
    consumes ``zone`` from an operand that carries ``generator`` already: the
    produced dimension is joined on too, so each row reads the price of its
    own generator's zone rather than fanning out over every generator.
    """
    spec, pins = _fixed(ZONAL, {'g1': [1.0, 2.0], 'g2': [3.0, 5.0]})
    spec = {
        **spec,
        'parameters': {**spec['parameters'], 'price': {'dims': ['zone', 'period']}},
        'variables': {
            **spec['variables'],
            'r': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0}},
            'm': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0}},
        },
        'constraints': {
            'spread': {
                'foreach': ['generator', 'period'],
                'expression': 'r == sum(price, by=zone_of, from=zone, into=generator) * p',
            },
            'masked': {
                'foreach': ['generator', 'period'],
                'expression': 'm == at(price * p, by=zone_of, into=generator)',
            },
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(r) + sum(m)'},
    }
    with differential(spec, {**_zonal_sources(HONEST), 'price': PRICE, **pins}) as run:
        spread = by_coord(run.result, 'r', 'generator', 'period')
        masked = by_coord(run.result, 'm', 'generator', 'period')
    expected = {('g1', 2030): 1.0, ('g2', 2030): 6.0, ('g1', 2050): 8.0, ('g2', 2050): 15.0}
    assert spread == expected, "the zone's price at the row's period, times the generator's own output"
    assert masked == expected, 'and the same number through a pullback whose operand carried generator already'


def test_at_through_a_key_the_relation_leaves_out_is_absent_at_that_coordinate():
    """A generator in no zone at one period reads nothing there, and the row is not built.

    ``g2`` is mapped in 2030 only, so no ceiling reaches it in 2050 and it
    runs to its own bound — on both lanes, the eager one masking the position
    it could not select rather than dropping a whole label.
    """
    spec = {
        **ZONAL,
        'parameters': {**ZONAL['parameters'], 'floor': {'dims': ['zone', 'period']}},
        'variables': {
            'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0, 'upper': 10}},
            'cap': {'foreach': ['zone', 'period'], 'bounds': {'lower': 0, 'upper': 'floor'}},
        },
        'constraints': {
            'at_most': {'foreach': ['generator', 'period'], 'expression': 'p <= at(cap, by=zone_of, into=generator)'}
        },
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }
    sources = {**_zonal_sources([('g1', 2030, 'z1'), ('g2', 2030, 'z2'), ('g1', 2050, 'z2')]), 'floor': PRICE}
    with differential(spec, sources) as run:
        held = by_coord(run.result, 'p', 'generator', 'period')
    assert held == {('g1', 2030): 1.0, ('g2', 2030): 2.0, ('g1', 2050): 4.0, ('g2', 2050): 10.0}, (
        "g2 has no zone in 2050, so no zone's cap reaches it there and only its own bound holds it"
    )


# ---------------------------------------------------------------------------
# one table with two value columns: landed on a product, read as a pair
# ---------------------------------------------------------------------------


BUSES = ['b1', 'b2']
TECHNOLOGIES = ['wind', 'gas']

TWO_VALUES = {
    'dimensions': {
        'generator': {'dtype': 'str'},
        'bus': {'dtype': 'str'},
        'technology': {'dtype': 'str'},
        'period': {'dtype': 'int'},
    },
    'lookups': {'gen_bt': {'over': ['generator', 'bus', 'technology'], 'key': 'generator'}},
    'parameters': {'tech_cap': {'dims': ['bus', 'technology']}},
    'variables': {
        'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0}},
        'q': {'foreach': ['bus', 'technology', 'period'], 'bounds': {'lower': 0}},
        'c': {'foreach': ['generator'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'landed': {
            'foreach': ['bus', 'technology', 'period'],
            'expression': 'q == sum(p, by=gen_bt, into=[bus, technology])',
        },
        'read': {'foreach': ['generator'], 'expression': 'c == at(tech_cap, by=gen_bt, from=[bus, technology])'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(q) + sum(c)'},
}


def test_one_table_lands_on_a_product_and_reads_a_two_column_slot():
    """``into=[bus, technology]`` is one join onto two value columns; ``from=[bus, technology]`` reads the pair back.

    ``g2`` and ``g3`` share a bus and ``g1`` and ``g3`` a technology, so a
    walk that landed on one column alone would merge groups this one keeps
    apart; the capacity table is distinct everywhere, so a read of the wrong
    pair lands on a number that is not its generator's.
    """
    generators = ['g1', 'g2', 'g3']
    spec, pins = _fixed(TWO_VALUES, {'g1': [1.0, 1.0], 'g2': [2.0, 2.0], 'g3': [3.0, 3.0]})
    sources = {
        'generator': generators,
        'bus': BUSES,
        'technology': TECHNOLOGIES,
        'period': PERIODS,
        'gen_bt': pl.DataFrame(
            {'generator': generators, 'bus': ['b1', 'b2', 'b2'], 'technology': ['wind', 'gas', 'wind']}
        ),
        'tech_cap': pl.DataFrame(
            {
                'bus': ['b1', 'b1', 'b2', 'b2'],
                'technology': ['wind', 'gas', 'wind', 'gas'],
                'value': [10.0, 20.0, 30.0, 40.0],
            }
        ),
        **pins,
    }
    with differential(spec, sources) as run:
        landed = by_coord(run.result, 'q', 'bus', 'technology', 'period')
        read = by_coord(run.result, 'c', 'generator')
    assert {k: v for k, v in landed.items() if k[2] == 2030} == {
        ('b1', 'wind', 2030): 1.0,
        ('b1', 'gas', 2030): 0.0,
        ('b2', 'wind', 2030): 3.0,
        ('b2', 'gas', 2030): 2.0,
    }, 'each (bus, technology) holds exactly the generators the table places there, and an unreached pair holds nothing'
    assert read == {'g1': 10.0, 'g2': 40.0, 'g3': 30.0}, (
        'each generator reads the capacity of its own (bus, technology)'
    )


# ---------------------------------------------------------------------------
# a produced dimension the operand already carries: a masked sum
# ---------------------------------------------------------------------------


def test_a_sum_producing_a_dimension_the_operand_carries_is_masked_by_it():
    """``sum(load * p, by=gen_bus)`` with ``load[bus, period]`` keeps each term where the generator's bus is the row's bus.

    The product carries ``bus`` before the walk produces it, so the walk
    joins on it: ``load`` at ``b1`` scales the generators on ``b1`` and
    nobody else, which the numbers below tell from a sum that fanned every
    load onto every group.
    """
    spec = {
        'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}, 'period': {'dtype': 'int'}},
        'lookups': {'gen_bus': {'over': ['generator', 'bus'], 'key': 'generator'}},
        'parameters': {'load': {'dims': ['bus', 'period']}},
        'variables': {
            'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0}},
            's': {'foreach': ['bus', 'period'], 'bounds': {'lower': 0}},
        },
        'constraints': {'scaled': {'foreach': ['bus', 'period'], 'expression': 's == sum(load * p, by=gen_bus)'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(s)'},
    }
    generators = ['g1', 'g2', 'g3']
    spec, pins = _fixed(spec, {'g1': [1.0, 1.0], 'g2': [2.0, 2.0], 'g3': [4.0, 4.0]})
    sources = {
        'generator': generators,
        'bus': BUSES,
        'period': PERIODS,
        'gen_bus': pl.DataFrame({'generator': generators, 'bus': ['b1', 'b2', 'b2']}),
        'load': pl.DataFrame(
            {'bus': ['b1', 'b2', 'b1', 'b2'], 'period': [2030, 2030, 2050, 2050], 'value': [10.0, 100.0, 1.0, 2.0]}
        ),
        **pins,
    }
    with differential(spec, sources) as run:
        held = by_coord(run.result, 's', 'bus', 'period')
    assert held == {('b1', 2030): 10.0, ('b2', 2030): 600.0, ('b1', 2050): 1.0, ('b2', 2050): 12.0}, (
        "each bus's load times the output of the generators on that bus alone"
    )


# ---------------------------------------------------------------------------
# roles: two ends of a line in one table, and a self-map
# ---------------------------------------------------------------------------


LINES = ['l1', 'l2']

ENDS = {
    'dimensions': {'line': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'lookups': {'ends': {'over': {'line': 'line', 'bus0': 'bus', 'bus1': 'bus'}, 'key': 'line'}},
    'parameters': {'flow': {'dims': ['line']}},
    'variables': {
        'f': {'foreach': ['line'], 'bounds': {'lower': 'flow', 'upper': 'flow'}},
        'n': {'foreach': ['bus'], 'bounds': {'lower': -100, 'upper': 100}},
        'y': {'foreach': ['line'], 'where': 'ends.bus0 != ends.bus1', 'bounds': {'lower': 1, 'upper': 1}},
    },
    'constraints': {
        'nodal': {'foreach': ['bus'], 'expression': 'n == sum(f, by=ends, into=bus1) - sum(f, by=ends, into=bus0)'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(n) + sum(y)'},
}

#: ``l1`` runs from ``b1`` to ``b2``; ``l2`` is a loop on ``b2``.
ENDS_SOURCES = {
    'line': LINES,
    'bus': BUSES,
    'ends': pl.DataFrame({'line': LINES, 'bus0': ['b1', 'b2'], 'bus1': ['b2', 'b2']}),
    'flow': pl.DataFrame({'line': LINES, 'value': [3.0, 5.0]}),
}


def test_a_lines_two_ends_are_one_table_walked_to_either_column():
    """``into=bus1`` and ``into=bus0`` walk one relation to each of its columns over ``bus``.

    The loop's flow arrives and leaves at ``b2``, so it nets to nothing
    there; a walk that read ``bus0`` where ``bus1`` was named would move
    ``l1``'s flow the wrong way. ``where: ends.bus0 != ends.bus1`` compares
    the two columns of one table and keeps only the line that is not a loop.
    """
    with differential(ENDS, ENDS_SOURCES) as run:
        injection = by_coord(run.result, 'n', 'bus')
        built = sorted(row['line'] for row in run.result.primal('y').to_dicts())
    assert injection == {'b1': -3.0, 'b2': 3.0}, "l1's flow leaves b1 and arrives at b2; the loop nets to nothing"
    assert built == ['l1'], 'only the line whose two ends differ survives ends.bus0 != ends.bus1'


SNAPSHOTS = [0, 1, 2, 3]

REPRESENTATIVE = {
    'dimensions': {'snapshot': {'dtype': 'int'}},
    'lookups': {'rep_of': {'over': {'snapshot': 'snapshot', 'rep': 'snapshot'}, 'key': 'snapshot'}},
    'parameters': {'base': {'dims': ['snapshot']}},
    'variables': {
        'x': {'foreach': ['snapshot'], 'bounds': {'lower': 0}},
        'w': {'foreach': ['snapshot'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'representative': {'foreach': ['snapshot'], 'expression': 'x == at(base, by=rep_of)'},
        'weighted': {'foreach': ['snapshot'], 'expression': 'w == sum(x, by=rep_of)'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(x) + sum(w)'},
}


def test_a_self_map_reads_along_its_arrow_and_sums_against_it():
    """A table with two columns over one dimension: ``at`` reads the representative's value, ``sum`` collects onto it.

    ``snapshot`` is consumed and ``rep`` produced, both over one dimension,
    so the frame is unchanged through both walks and only the values move:
    every snapshot takes its representative's base, and each representative
    gathers the snapshots it stands for — nothing at the ones that stand for
    no one.
    """
    sources = {
        'snapshot': SNAPSHOTS,
        'rep_of': pl.DataFrame({'snapshot': SNAPSHOTS, 'rep': [0, 0, 2, 2]}),
        'base': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 40.0]}),
    }
    with differential(REPRESENTATIVE, sources) as run:
        read = by_coord(run.result, 'x', 'snapshot')
        gathered = by_coord(run.result, 'w', 'snapshot')
    assert read == {0: 10.0, 1: 10.0, 2: 30.0, 3: 30.0}, "each snapshot reads its representative's base"
    assert gathered == {0: 20.0, 1: 0.0, 2: 60.0, 3: 0.0}, 'each representative gathers the snapshots it stands for'


# ---------------------------------------------------------------------------
# a bare relation: walked with both ends named, tested by a bare where
# ---------------------------------------------------------------------------


CONNECTED = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}, 'period': {'dtype': 'int'}},
    'lookups': {'connection': {'over': ['generator', 'bus']}},
    'parameters': {},
    'variables': {
        'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0}},
        'c': {'foreach': ['bus', 'period'], 'bounds': {'lower': 0}},
        'k': {'foreach': ['generator', 'bus'], 'where': 'connection', 'bounds': {'lower': 1, 'upper': 1}},
    },
    'constraints': {
        'reachable': {
            'foreach': ['bus', 'period'],
            'expression': 'c == sum(p, by=connection, from=generator, into=bus)',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(c) + sum(k)'},
}


def test_a_bare_relation_is_summed_through_with_both_ends_named():
    """A generator connected to two buses lands its output on both, and a bare ``where`` keeps the pairs the relation holds.

    No key, so nothing is single-valued: ``g1`` reaches ``b1`` and ``b2``
    alike, which no keyed lookup could say. The eager lane reads the
    relation as a dense incidence and the relational one joins it — the
    same numbers, and the same three ``(generator, bus)`` rows for ``k``.
    """
    spec, pins = _fixed(CONNECTED, {'g1': [1.0, 2.0], 'g2': [10.0, 20.0]})
    sources = {
        'generator': GENERATORS,
        'bus': BUSES,
        'period': PERIODS,
        'connection': pl.DataFrame({'generator': ['g1', 'g1', 'g2'], 'bus': ['b1', 'b2', 'b2']}),
        **pins,
    }
    with differential(spec, sources) as run:
        held = by_coord(run.result, 'c', 'bus', 'period')
        pairs = sorted((row['generator'], row['bus']) for row in run.result.primal('k').to_dicts())
    assert held == {('b1', 2030): 1.0, ('b2', 2030): 11.0, ('b1', 2050): 2.0, ('b2', 2050): 22.0}, (
        'g1 counts at both buses it connects to, g2 at b2 alone'
    )
    assert pairs == [('g1', 'b1'), ('g1', 'b2'), ('g2', 'b2')], (
        'the bare where keeps exactly the rows the relation holds'
    )


def test_a_bare_relation_holding_a_row_twice_is_refused_at_bind():
    """A relation is a set of rows: a pair held twice would count a term twice in every sum through it."""
    doubled = pl.DataFrame({'generator': ['g1', 'g1', 'g2'], 'bus': ['b1', 'b1', 'b2']})
    spec, pins = _fixed(CONNECTED, {'g1': [1.0, 2.0], 'g2': [10.0, 20.0]})
    sources = {'generator': GENERATORS, 'bus': BUSES, 'period': PERIODS, 'connection': doubled, **pins}
    message = both_lanes_refuse(spec, sources, r'holds 1 row\(s\) more than once')
    assert "('g1', 'b1')" in message, 'the refusal names the row held twice'


# ---------------------------------------------------------------------------
# shift(by=), sum_back(by=) and position(by=) inside a group that differs per scenario
# ---------------------------------------------------------------------------


BLOCK_SNAPSHOTS = [0, 1, 2, 3, 4]
SCENARIOS = ['base', 'alt']

#: The blocks partition the axis differently per scenario: ``base`` is one
#: block of three and one of two, ``alt`` one of two and one of three with
#: snapshot 4 in no block at all.
BLOCK_OF = [
    ('base', 0, 'a'),
    ('base', 1, 'a'),
    ('base', 2, 'a'),
    ('base', 3, 'b'),
    ('base', 4, 'b'),
    ('alt', 0, 'a'),
    ('alt', 1, 'a'),
    ('alt', 2, 'b'),
    ('alt', 3, 'b'),
]

BLOCKED = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'block': {'dtype': 'str'}, 'scenario': {'dtype': 'str'}},
    'lookups': {'block_of': {'over': ['snapshot', 'scenario', 'block'], 'key': ['snapshot', 'scenario']}},
    'variables': {
        'x': {'foreach': ['snapshot', 'scenario'], 'bounds': {'lower': 0, 'upper': 100}},
        'w': {'foreach': ['snapshot', 'scenario'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'seed': {
            'foreach': ['snapshot', 'scenario'],
            'where': 'position(snapshot, by=block_of) == 0',
            'expression': 'x == 5',
        },
        'step': {
            'foreach': ['snapshot', 'scenario'],
            'where': 'position(snapshot, by=block_of) > 0',
            'expression': 'x == shift(x, over=snapshot, offset=1, by=block_of) + 1',
        },
        'window': {
            'foreach': ['snapshot', 'scenario'],
            'expression': 'w == sum_back(x, over=snapshot, within=2, by=block_of)',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(x) + sum(w)'},
}


def _blocked_sources() -> dict:
    return {
        'snapshot': BLOCK_SNAPSHOTS,
        'block': ['a', 'b'],
        'scenario': SCENARIOS,
        'block_of': pl.DataFrame(BLOCK_OF, schema=['scenario', 'snapshot', 'block'], orient='row'),
    }


def test_a_walk_inside_a_group_is_taken_within_each_scenario_s_own_blocks():
    """``position``, ``shift`` and ``sum_back`` all walk ``snapshot``, join on ``scenario`` and group by ``block``.

    ``x`` counts up from 5 at each block's first snapshot, so its value *is*
    the within-group position, and the position of snapshot 2 differs by
    scenario: third of ``a`` under ``base``, first of ``b`` under ``alt``. A
    walk that ranked the axis once for both scenarios could not tell them
    apart. ``w`` is the trailing pair, short at each block's start rather
    than reaching into the block before it, and snapshot 4 under ``alt`` is
    in no block: no position, no neighbour, no window, so both stay at zero.
    """
    with differential(BLOCKED, _blocked_sources()) as run:
        x = by_coord(run.result, 'x', 'scenario', 'snapshot')
        w = by_coord(run.result, 'w', 'scenario', 'snapshot')
    assert x == {
        ('base', 0): 5.0,
        ('base', 1): 6.0,
        ('base', 2): 7.0,
        ('base', 3): 5.0,
        ('base', 4): 6.0,
        ('alt', 0): 5.0,
        ('alt', 1): 6.0,
        ('alt', 2): 5.0,
        ('alt', 3): 6.0,
        ('alt', 4): 0.0,
    }, "5 plus the within-group position, counted inside each scenario's own blocks"
    assert w == {
        ('base', 0): 5.0,
        ('base', 1): 11.0,
        ('base', 2): 13.0,
        ('base', 3): 5.0,
        ('base', 4): 11.0,
        ('alt', 0): 5.0,
        ('alt', 1): 11.0,
        ('alt', 2): 5.0,
        ('alt', 3): 11.0,
        ('alt', 4): 0.0,
    }, "the trailing pair inside each block, short at the block's first snapshot"


def test_a_block_too_short_for_the_position_is_named_with_its_scenario():
    """The short-group refusal names the group as ``(block, scenario)``, on both lanes with one sentence.

    Position 2 exists in ``base``'s block ``a`` and ``alt``'s block ``b``, and
    in neither scenario's other block — and which block is short depends on
    the scenario, so the name has to carry it.
    """
    spec = {
        **BLOCKED,
        'constraints': {'seed': {**BLOCKED['constraints']['seed'], 'where': 'position(snapshot, by=block_of) == 2'}},
    }
    message = both_lanes_refuse(spec, _blocked_sources(), 'shorter than that')
    assert "('b', 'base')" in message and "('a', 'alt')" in message, (
        "each short group is named with the scenario it is short in — base's b and alt's a"
    )


# ---------------------------------------------------------------------------
# where: a two-key lookup is read at both keys
# ---------------------------------------------------------------------------


HOME_OF = [('g1', 2030, 'z1'), ('g2', 2030, 'z1'), ('g1', 2050, 'z1'), ('g2', 2050, 'z1')]


def _where_spec(where: str) -> dict:
    return {
        **ZONAL,
        'lookups': {
            **ZONAL['lookups'],
            'home_of': {'over': ['generator', 'period', 'zone'], 'key': ['generator', 'period']},
        },
        'variables': {'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 1, 'upper': 1}, 'where': where}},
        'constraints': {},
        'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
    }


@pytest.mark.parametrize(
    ('where', 'built'),
    [
        pytest.param("zone_of == 'z1'", [('g1', 2030)], id='against-a-label'),
        pytest.param('zone_of != home_of', [('g1', 2050), ('g2', 2030)], id='two-lookups-keyed-alike'),
        pytest.param('zone_of', [('g1', 2030), ('g1', 2050), ('g2', 2030)], id='a-bare-lookup-is-the-partial-case'),
        pytest.param("NOT zone_of == 'z1'", [('g1', 2050), ('g2', 2030), ('g2', 2050)], id='negated'),
    ],
)
def test_a_where_reads_a_two_key_lookup_at_the_row_s_own_period(where, built):
    """The mask is a join on ``(generator, period)`` in one lane and a rank-two array in the other.

    ``g1`` is in ``z1`` in 2030 and ``z2`` in 2050, ``g2`` the other way, and
    everyone's home is ``z1`` — so which rows survive changes with the period
    under every predicate here. ``g2`` is unmapped in 2050 for the bare form,
    where a null compares false and the negation keeps it.
    """
    sources = {
        **_zonal_sources([('g1', 2030, 'z1'), ('g2', 2030, 'z2'), ('g1', 2050, 'z2')]),
        'home_of': _zone_of(HOME_OF),
    }
    with differential(_where_spec(where), sources) as run:
        rows = run.result.primal('p')
        assert sorted(zip(rows['generator'], rows['period'], strict=True)) == sorted(built), (
            f'where: {where!r} kept the wrong (generator, period) rows'
        )


# ---------------------------------------------------------------------------
# the relation's contract at the door
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('zone_of', 'match'),
    [
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g2'], 'zone': ['z1', 'z2']}),
            r"must carry columns \['generator', 'period', 'zone'\] .*\['period'\] missing",
            id='a-relation-short-of-a-key-column',
        ),
        pytest.param(
            _zone_of([*HONEST[:3], ('g2', 2070, 'z1')]),
            r"column 'period' holds 2070, which are not labels of 'period'",
            id='a-key-that-is-not-a-label',
        ),
        pytest.param(
            _zone_of([*HONEST[:3], ('g2', 2050, 'z9')]),
            r"column 'zone' has value\(s\) that are not 'zone' labels: 'z9'",
            id='a-value-that-is-not-a-label',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g2'], 'period': [2030, None], 'zone': ['z1', 'z2']}),
            r"a null in 'period': generator='g2', period=None, zone='z2'",
            id='a-null-in-a-key-column',
        ),
    ],
)
def test_the_relation_is_held_to_its_columns(zone_of, match):
    """Every column is required, checked against its dimension's labels, and never null."""
    with pytest.raises(lps.DataError, match=match):
        lps.build(ZONAL, {**_zonal_sources(HONEST), 'zone_of': zone_of}).close()


def test_an_unsupplied_lookup_names_every_column_it_takes():
    sources = {k: v for k, v in _zonal_sources(HONEST).items() if k != 'zone_of'}
    with pytest.raises(lps.DataError, match=r"columns \['generator', 'period', 'zone'\]") as caught:
        lps.build(ZONAL, sources).close()
    assert 'one row per (generator, period) tuple it holds' in str(caught.value), (
        'the refusal says what a row of the relation is'
    )


def test_a_relation_with_roles_is_supplied_under_the_role_names():
    """The columns are the declared roles, so a table naming a dimension where a role was declared is short of a column."""
    unrolled = pl.DataFrame({'line': LINES, 'bus': ['b1', 'b2']})
    with pytest.raises(lps.DataError, match=r"must carry columns \['line', 'bus0', 'bus1'\]"):
        lps.build(ENDS, {**ENDS_SOURCES, 'ends': unrolled}).close()
