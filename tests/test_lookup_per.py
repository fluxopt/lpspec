"""``per:`` — a lookup conditioned on a second dimension, on both lanes.

A map keyed by ``over:`` alone gives every generator one zone for the whole
model. ``per: [period]`` keys it by ``(generator, period)`` as well: ``over``
is consumed, ``into`` is produced, and ``per`` is joined on and passed through
by every operator that takes a lookup. The models here are the ones the
language reference writes the keyword for (math-spec#161): a generator whose
bidding zone changes by period, and the three data cases that told the
composite ``over: [generator, period]`` apart from this — the honest map, the
map frozen at 2030, and a generator in two zones at once, which used to be a
legal membership parameter and is now refused at bind.

Every operator is checked differentially: the relational lane joins the
relation on ``(over, *per)`` where the eager lane reads a rank-two array, and
nothing but the agreement shows that the two readings are one.
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
    'lookups': {'zone_of': {'over': 'generator', 'into': 'zone', 'per': ['period']}},
    'parameters': {'cost': {'dims': ['generator']}, 'demand': {'dims': ['zone', 'period']}},
    'variables': {'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0}}},
    'constraints': {'zone_balance': {'foreach': ['zone', 'period'], 'expression': 'sum(p, by=zone_of) >= demand'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

#: ``g1`` is the cheap generator. In 2030 it serves ``z1``; in 2050 it moves
#: to ``z2``, and ``z1``'s demand triples.
HONEST = [('g1', 2030, 'z1'), ('g2', 2030, 'z2'), ('g1', 2050, 'z2'), ('g2', 2050, 'z1')]
FROZEN_AT_2030 = [('g1', 2030, 'z1'), ('g2', 2030, 'z2'), ('g1', 2050, 'z1'), ('g2', 2050, 'z2')]


def _zone_of(rows: list[tuple[str, int, str]]) -> pl.DataFrame:
    """The relation as it is supplied: one key column per dimension, the value under the target's name."""
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


@pytest.mark.parametrize(
    ('zone_of', 'objective'),
    [
        pytest.param(HONEST, 220.0, id='the-zone-moves-with-the-period'),
        pytest.param(FROZEN_AT_2030, 140.0, id='the-map-frozen-at-2030'),
    ],
)
def test_a_zone_that_changes_by_period_is_read_at_the_row_s_own_period(zone_of, objective):
    """The two numbers that tell a conditioned map from a frozen one.

    Frozen, the cheap generator serves ``z1``'s tripled 2050 demand and the
    model costs 140; honestly mapped, it has moved to ``z2`` and the dear one
    serves the 30 units at 5 apiece. A composite ``over: [generator, period]``
    could not say either — it would contract ``period`` inside the sum.
    """
    with differential(ZONAL, _zonal_sources(zone_of)) as run:
        assert run.result.objective == pytest.approx(objective), (
            'the group at 2050 has to read the 2050 map, not the 2030 one — both lanes agree, on the wrong number'
        )


def test_a_generator_in_two_zones_in_one_period_is_refused_at_bind():
    """Single-valued per ``(generator, period)`` — the guard a membership parameter never had.

    Under ``over: [generator, period]`` with a ``0/1`` parameter this was a
    legal model at 170.0, ``g1`` serving both zones in 2030. As a lookup the
    key maps twice, and the refusal names the key rather than the label: ``g1``
    is mapped once per period everywhere else.
    """
    message = both_lanes_refuse(ZONAL, _zonal_sources([*HONEST, ('g1', 2030, 'z2')]), 'more than once')
    assert "('g1', 2030)" in message, 'the refusal names the (generator, period) key that maps twice'
    assert 'single-valued per (generator, period)' in message, 'and says what the map is single-valued per'


def test_a_group_no_member_reaches_in_one_period_is_an_empty_sum_there():
    """A zone with no generator at some period holds the empty sum on a constant side.

    ``need`` is grouped through the same conditioned map ``p`` is, and ``z3``
    has a member in 2030 only — so at 2050 its right-hand side is 0 rather
    than a hole. ``spare`` keeps a term in every row, so the row stands to be
    counted: one per ``(zone, period)`` on both lanes, the empty groups
    included, where a hole would have dropped it on one lane or the other.
    """
    spec = {
        **ZONAL,
        'parameters': {**ZONAL['parameters'], 'need': {'dims': ['generator', 'period']}},
        'variables': {**ZONAL['variables'], 'spare': {'foreach': ['zone', 'period'], 'bounds': {'lower': 0}}},
        'constraints': {
            'zone_balance': {
                'foreach': ['zone', 'period'],
                'expression': 'sum(p, by=zone_of) + spare >= sum(need, by=zone_of)',
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
# at(by=): the coarse value at the row's own coordinate of the per dim
# ---------------------------------------------------------------------------


FLOORED = {
    **ZONAL,
    'parameters': {**ZONAL['parameters'], 'floor': {'dims': ['zone', 'period']}},
    'constraints': {'at_least': {'foreach': ['generator', 'period'], 'expression': 'p >= at(floor, by=zone_of)'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

FLOOR = pl.DataFrame(
    {'zone': ['z1', 'z2', 'z1', 'z2'], 'period': [2030, 2030, 2050, 2050], 'value': [1.0, 2.0, 3.0, 4.0]}
)


def test_at_reads_the_zone_this_generator_sat_in_that_period():
    """The adjoint: ``floor[zone, period]`` back at ``(generator, period)``, pointwise on ``period``.

    Every value is distinct, so a pullback that read the wrong period — the
    map's first, say, or a broadcast over every period — lands on a number
    this table does not hold at that coordinate.
    """
    sources = {**_zonal_sources(HONEST), 'floor': FLOOR}
    with differential(FLOORED, sources) as run:
        held = by_coord(run.result, 'p', 'generator', 'period')
    assert held == {('g1', 2030): 1.0, ('g2', 2030): 2.0, ('g1', 2050): 4.0, ('g2', 2050): 3.0}, (
        'g1 reads z1 in 2030 and z2 in 2050, g2 the other way round — the floor of its own zone at its own period'
    )


def test_at_through_a_key_the_map_leaves_out_is_absent_at_that_coordinate():
    """A generator in no zone at one period reads nothing there, and the row is not built.

    The same reading a partial map gets unconditioned, now per key: ``g2`` is
    mapped in 2030 only, so no ceiling reaches it in 2050 and it runs to its
    own bound — on both lanes, the eager one masking the position it could
    not select rather than dropping a whole label. A variable is what is read
    through the map, absence being representable on a term where a constant
    side would report a hole.
    """
    spec = {
        **FLOORED,
        'variables': {
            'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0, 'upper': 10}},
            'cap': {'foreach': ['zone', 'period'], 'bounds': {'lower': 0, 'upper': 'floor'}},
        },
        'constraints': {'at_most': {'foreach': ['generator', 'period'], 'expression': 'p <= at(cap, by=zone_of)'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }
    sources = {**_zonal_sources([('g1', 2030, 'z1'), ('g2', 2030, 'z2'), ('g1', 2050, 'z2')]), 'floor': FLOOR}
    with differential(spec, sources) as run:
        held = by_coord(run.result, 'p', 'generator', 'period')
    assert held == {('g1', 2030): 1.0, ('g2', 2030): 2.0, ('g1', 2050): 4.0, ('g2', 2050): 10.0}, (
        "g2 has no zone in 2050, so no zone's cap reaches it there and only its own bound holds it"
    )


# ---------------------------------------------------------------------------
# shift(by=), sum_back(by=) and position(by=) inside a group that differs per scenario
# ---------------------------------------------------------------------------


SNAPSHOTS = [0, 1, 2, 3, 4]
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
    'lookups': {'block_of': {'over': 'snapshot', 'into': 'block', 'per': ['scenario']}},
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
        'snapshot': SNAPSHOTS,
        'scenario': SCENARIOS,
        'block': ['a', 'b'],
        'block_of': pl.DataFrame(BLOCK_OF, schema=['scenario', 'snapshot', 'block'], orient='row'),
    }


def test_a_walk_inside_a_group_is_taken_within_each_scenario_s_own_blocks():
    """``position``, ``shift`` and ``sum_back`` all group within ``(block, scenario)``.

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
# where: a conditioned lookup is read at (over, *per)
# ---------------------------------------------------------------------------


HOME_OF = [('g1', 2030, 'z1'), ('g2', 2030, 'z1'), ('g1', 2050, 'z1'), ('g2', 2050, 'z1')]


def _where_spec(where: str) -> dict:
    return {
        **ZONAL,
        'lookups': {**ZONAL['lookups'], 'home_of': {'over': 'generator', 'into': 'zone', 'per': ['period']}},
        'variables': {'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 1, 'upper': 1}, 'where': where}},
        'constraints': {},
        'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
    }


@pytest.mark.parametrize(
    ('where', 'built'),
    [
        pytest.param("zone_of == 'z1'", [('g1', 2030)], id='against-a-label'),
        pytest.param('zone_of != home_of', [('g1', 2050), ('g2', 2030)], id='two-lookups-sharing-their-per'),
        pytest.param('zone_of', [('g1', 2030), ('g1', 2050), ('g2', 2030)], id='a-bare-lookup-is-the-partial-case'),
        pytest.param("NOT zone_of == 'z1'", [('g1', 2050), ('g2', 2030), ('g2', 2050)], id='negated'),
    ],
)
def test_a_where_reads_a_conditioned_lookup_at_the_row_s_own_period(where, built):
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
            r"must carry columns \['generator', 'period', 'zone'\]",
            id='a-relation-short-of-its-per-column',
        ),
        pytest.param(
            _zone_of([*HONEST[:3], ('g2', 2070, 'z1')]),
            r"maps 2070, which are not labels of 'period'",
            id='a-per-key-that-is-not-a-label',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g2'], 'period': [2030, None], 'zone': ['z1', 'z2']}),
            r"a null in 'period': generator='g2', period=None",
            id='a-null-in-the-per-column',
        ),
    ],
)
def test_the_relation_is_held_to_its_key(zone_of, match):
    """Each ``per`` column is a key: required, checked against its dimension's labels, and never null."""
    with pytest.raises(lps.DataError, match=match):
        lps.build(ZONAL, {**_zonal_sources(HONEST), 'zone_of': zone_of}).close()


def test_an_unsupplied_conditioned_lookup_names_every_column_it_takes():
    sources = {k: v for k, v in _zonal_sources(HONEST).items() if k != 'zone_of'}
    with pytest.raises(lps.DataError, match=r"columns \['generator', 'period', 'zone'\]") as caught:
        lps.build(ZONAL, sources).close()
    assert 'one row per (generator, period) key it maps' in str(caught.value), (
        'the refusal says what a row of the relation is'
    )
