"""Lookups: a relation between dimensions, and the claims a declaration makes about its data.

A lookup is a table with one column per dimension it relates; `key:` says a row
is identified by those columns, and `coverage:` says whether every key tuple has
one. What these tests hold still: a lookup name joins the flat namespace, the
relation arrives under its own key with a column per column declared, every
claim the declaration makes is checked at bind and named in the refusal, a
`where` reads a value column at the key on both lanes, and each walk the
language admits — a composite key, several value columns, a bare relation, a
self-map, a masked sum — builds and reaches the answer worked out by hand.

The linopy lane holds a lookup as an array over the dimension its key is over,
so the walks that are not a function of one dimension are refused there and
built on the relational engine alone; those refusals are pinned here too.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest
from math_spec import to_spec

import lpspec as lps
from lpspec.errors import DataError, LaneError, LpspecError, LpspecWarning
from tests.conftest import by_coord


def _spec(objective: str = 'sum(x, over=snapshot)') -> dict:
    return {
        'dimensions': {'snapshot': {'dtype': 'int'}, 'period': {'dtype': 'int'}},
        'lookups': {'period_of': {'over': ['snapshot', 'period'], 'key': 'snapshot'}},
        'parameters': {'load': {'dims': ['snapshot']}},
        'variables': {'x': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'foreach': ['snapshot'], 'expression': 'x >= load'}},
        'objective': {'sense': 'minimize', 'expression': objective},
    }


def _index() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2]})


def _period_of() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})


def _load() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})


def _sources() -> dict:
    return {'load': _load(), 'snapshot': _index(), 'period': [1, 2], 'period_of': _period_of()}


# ---------------------------------------------------------------------------
# the flat namespace, and the advice a declaration draws
# ---------------------------------------------------------------------------


def test_a_lookup_joins_the_flat_namespace():
    spec = _spec()
    spec['parameters']['period_of'] = {'dims': ['snapshot']}
    with pytest.raises(LpspecError, match="Parameter 'period_of' collides with the lookup"):
        to_spec(spec)


def test_a_lookup_cannot_take_a_dimensions_name():
    spec = _spec()
    spec['lookups']['snapshot'] = {'over': ['snapshot', 'period'], 'key': 'snapshot'}
    with pytest.raises(LpspecError, match="Lookup 'snapshot' collides with the dimension"):
        to_spec(spec)


def test_a_by_typo_is_offered_the_lookups_it_could_have_meant():
    spec = _spec()
    spec['dimensions']['bus'] = {'dtype': 'str'}
    spec['lookups']['bus_of'] = {'over': ['snapshot', 'bus'], 'key': 'snapshot'}
    spec['constraints']['c'] = {'foreach': ['bus'], 'expression': 'sum(x, by=bus_ov) >= load'}
    with pytest.raises(LpspecError, match=r'by=bus_ov\) does not name a lookup') as caught:
        lps.check(spec)
    assert "'bus_of'" in str(caught.value), 'the listing offers the lookup the typo is one letter from'


def test_check_advises_an_unused_dimension():
    spec = _spec()
    spec['dimensions']['scenario'] = {'dtype': 'str'}
    with pytest.warns(LpspecWarning, match="'scenario' is never used"):
        lps.check(spec)


def test_a_dimension_a_lookup_has_a_column_over_draws_no_advice():
    """Nothing is indexed by `period` here — a `where` selects on it and that is use enough.

    The members of a dimension a lookup relates are the labels that column is
    checked against, so declaring one no constraint spans is the ordinary way
    to make a selection safe rather than an oversight to warn about.
    """
    spec = _spec()
    spec['constraints']['c']['where'] = 'period_of == 1'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lps.check(spec)


def test_a_clean_model_checks_silently():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lps.check(_spec('sum(sum(x, by=period_of), over=period)'))


# ---------------------------------------------------------------------------
# the relation, supplied under the lookup's own key
# ---------------------------------------------------------------------------


GENERATORS = ['g1', 'g2', 'g3']
COST = [1.0, 2.0, 0.5]
LOAD = [5.0, 4.0]

BASE = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'lookups': {'gen_bus': {'over': ['generator', 'bus'], 'key': 'generator', 'coverage': 'masked'}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['bus']}},
    'variables': {'p': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'balance': {'foreach': ['bus'], 'expression': 'sum(p, by=gen_bus) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

_RELATION = pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']})
_BASE_SOURCES = {
    'bus': ['north', 'south'],
    'generator': GENERATORS,
    'gen_bus': _RELATION,
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': COST}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': LOAD}),
}


def test_a_supplied_relation_reaches_the_declared_lookup_without_touching_the_index():
    """The point: a caller adds a relation to a dimension whose index is not theirs.

    `g3` is the cheapest generator and contributes nothing, which is what a key
    with no row means under `masked`. The index goes in as a bare label list, so
    nothing here rewrites a table someone else generated.
    """
    with lps.solve(BASE, _BASE_SOURCES) as result:
        built = by_coord(result, 'p', 'generator')
        assert result.objective == pytest.approx(13.0), (
            'g1 covers the north at 1 and g2 the south at 2, and g3 — on no bus — cannot help'
        )
    assert built['g3'] == pytest.approx(0.0), 'a generator in no row of the relation is a generator on no bus'


def test_a_supplied_relation_agrees_with_the_oracle():
    """Both lanes read the relation through the one front door, so both see it."""
    from tests.differential import differential
    from tests.oracle import pd

    data = {
        'cost': pd.Series(COST, index=GENERATORS),
        'load': pd.Series(LOAD, index=['north', 'south']),
        'bus': ['north', 'south'],
        'generator': GENERATORS,
        'gen_bus': _RELATION,
    }
    with differential(BASE, data) as run:
        assert run.result.objective == pytest.approx(13.0), (
            'both lanes place the same terms, so both reach the optimum the relational lane does'
        )


def test_a_relation_is_not_a_column_of_an_index_it_has_a_column_over():
    """The one stray column that is refused rather than filtered away.

    An index may carry anything — attributes, other frameworks' fields — and
    the extras are dropped. A column named after a lookup over that dimension
    is not an extra: it is a relation somebody meant to supply, and dropping it
    would build the model they did not write.
    """
    carried = pl.DataFrame({'generator': GENERATORS, 'gen_bus': ['south', 'north', None]})
    with pytest.raises(DataError, match=re.escape("index for dimension 'generator' carries a 'gen_bus' column")):
        lps.solve(BASE, {**_BASE_SOURCES, 'generator': carried})


def test_a_relation_alone_does_not_say_which_labels_exist():
    """A relation between dimensions is not a dimension, whoever holds it.

    A column of it may name a label twice or leave one out, and its row order is
    whatever someone typed — so reading the label set out of it would let an
    added row create a member and a reordered one re-order the axis that
    ``shift`` reads positionally. `g3` — in no row and still a generator — is
    exactly the member that would vanish.
    """
    with pytest.raises(DataError, match=re.escape("has lookups with a column over it (sources['gen_bus'])")):
        lps.solve(BASE, {k: v for k, v in _BASE_SOURCES.items() if k != 'generator'})


def test_a_lookup_with_no_author_at_all_is_refused():
    """Neither is not a spelling of empty: a lookup nothing supplies is missing data."""
    with pytest.raises(DataError, match="no data provided for lookup 'gen_bus'"):
        lps.solve(BASE, {k: v for k, v in _BASE_SOURCES.items() if k != 'gen_bus'})


@pytest.mark.parametrize(
    ('relation', 'match'),
    [
        pytest.param(
            pl.DataFrame({'generator': ['g1'], 'gen_bus': ['north']}),
            r"must carry columns \['generator', 'bus'\]",
            id='a-column-named-after-the-lookup-and-not-after-the-column',
        ),
        pytest.param(
            pl.DataFrame({'bus': ['north']}),
            r"short of \['generator'\]",
            id='no-key-column',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']}),
            r"holds 1 'generator'\(s\) more than once: generator='g1'",
            id='one-key-tuple-twice',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g9'], 'bus': ['north']}),
            r"column 'generator' holds 1 value\(s\) that are not labels of 'generator'",
            id='a-key-that-is-not-a-label',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1'], 'bus': ['atlantis']}),
            r"column 'bus' holds 1 value\(s\) that are not labels of 'bus'",
            id='a-value-that-is-not-a-label',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', None]}),
            r"null in one of \['generator', 'bus'\]",
            id='a-row-naming-nothing',
        ),
    ],
)
def test_a_supplied_relation_is_held_to_what_the_declaration_claims(relation, match):
    """Every column a label of its dimension, one row per key tuple, and no null.

    A stray label on either side is the refusal the column form got for free: a
    relation that rides an index cannot name a label the index lacks. Dropping
    it instead would place that generator's terms nowhere while the model built
    and solved.
    """
    with pytest.raises(DataError, match=match):
        lps.solve(BASE, {**_BASE_SOURCES, 'gen_bus': relation})


@pytest.mark.parametrize('lane', ['relational', 'eager'])
def test_a_supplied_relation_is_refused_the_same_way_on_both_lanes(lane):
    """One defect, one sentence: the checks live in the door both lanes enter."""
    from tests.oracle import lpspec_linopy

    build = lps.solve if lane == 'relational' else lpspec_linopy.build
    twice = pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']})
    with pytest.raises(DataError, match=r"holds 1 'generator'\(s\) more than once"):
        build(BASE, {**_BASE_SOURCES, 'gen_bus': twice})


# ---------------------------------------------------------------------------
# coverage: whether a key with no row was meant
# ---------------------------------------------------------------------------


def _total() -> dict:
    """`BASE` as the file writes it by default — every generator on a bus."""
    return {**BASE, 'lookups': {'gen_bus': {'over': ['generator', 'bus'], 'key': 'generator'}}}


def test_a_total_lookup_short_of_a_key_is_refused_naming_the_key():
    """The claim `coverage:` makes, checked where the data is.

    `g3` is on no bus. Under the default the file says every generator is, so
    this is a hole in the data rather than a generator that belongs nowhere —
    and the difference is an answer: masked, `g3` contributes nothing and the
    optimum is 13.0; total, nobody finds out at all, because a row that lands
    in no group costs nothing and changes no number.
    """
    with pytest.raises(DataError) as caught:
        lps.solve(_total(), _BASE_SOURCES)
    message = str(caught.value)
    assert "lookup 'gen_bus' is total" in message, 'the refusal names the claim that failed'
    assert "generator='g3'" in message, 'and the key that has no row'
    assert 'coverage: masked' in message, 'and the declaration that would make the gap deliberate'


def test_a_total_lookup_with_every_key_is_the_ordinary_case():
    whole = pl.DataFrame({'generator': GENERATORS, 'bus': ['north', 'south', 'south']})
    with lps.solve(_total(), {**_BASE_SOURCES, 'gen_bus': whole}) as result:
        assert result.objective == pytest.approx(7.0), (
            'g1 covers the north at 1, and g3 — now placed — the south at 0.5'
        )


@pytest.mark.parametrize('lane', ['relational', 'eager'])
def test_totality_is_checked_at_the_one_door_both_lanes_enter(lane):
    from tests.oracle import lpspec_linopy

    build = lps.build if lane == 'relational' else lpspec_linopy.build
    with pytest.raises(DataError, match=r"lookup 'gen_bus' is total"):
        build(_total(), _BASE_SOURCES)


def test_a_composite_key_is_total_over_the_product_of_its_dimensions():
    """A row for every generator in every period, and the refusal names the pair that lacks one."""
    short = pl.DataFrame(
        {
            'generator': ['g1', 'g1', 'g2'],
            'period': [2030, 2040, 2030],
            'zone': ['north', 'south', 'south'],
        }
    )
    with pytest.raises(DataError) as caught:
        lps.build(COMPOSITE, {**_COMPOSITE_SOURCES, 'zone_of': short})
    message = str(caught.value)
    assert "generator='g2'" in message and 'period=2040' in message, (
        'a composite key is total over the product, so the missing tuple is named as a tuple'
    )


def test_a_bare_relation_is_the_rows_it_has():
    """No key, so nothing for coverage to be total over, and a short table is no defect."""
    with lps.solve(BARE, _BARE_SOURCES) as result:
        assert result.objective == pytest.approx(4.0), (
            'a bare relation is the rows it has, and a short table is no defect'
        )


# ---------------------------------------------------------------------------
# a where reading a lookup
# ---------------------------------------------------------------------------


#: A three-line network whose structure is entirely lookups: each line has two
#: endpoints in one table, and `spur` deliberately has an open end so the
#: partial case is reachable.
NETWORK = {
    'dimensions': {'bus': {'dtype': 'str'}, 'line': {'dtype': 'str'}, 'grid': {'dtype': 'int'}},
    'lookups': {
        'send': {'over': ['line', 'bus'], 'key': 'line'},
        'recv': {'over': ['line', 'bus'], 'key': 'line', 'coverage': 'masked'},
        'voltage': {'over': ['line', 'grid'], 'key': 'line'},
    },
    'parameters': {'cap': {'dims': ['line']}, 'price': {'dims': ['line']}},
    'variables': {'f': {'foreach': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'ceiling': {'foreach': ['line'], 'expression': 'f <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(f * price)'},
}

LINES = ['ring_a', 'ring_b', 'loop', 'spur']
SEND = ['north', 'south', 'north', 'north']
RECV = ['south', 'north', 'north']
VOLTAGE = [220, 380, 220, 380]
CAP = [10.0, 20.0, 30.0, 40.0]
PRICE = [1.0, 1.0, 1.0, 1.0]

#: `loop` starts and ends on the same bus; `spur` has no receiving end at all,
#: which `recv` says by having no row for it — and by declaring `masked`.
NETWORK_SOURCES = {
    'bus': pl.DataFrame({'bus': ['north', 'south']}),
    'line': pl.DataFrame({'line': LINES}),
    'grid': [220, 380],
    'send': pl.DataFrame({'line': LINES, 'bus': SEND}),
    'recv': pl.DataFrame({'line': LINES[:3], 'bus': RECV}),
    'voltage': pl.DataFrame({'line': LINES, 'grid': VOLTAGE}),
    'cap': pl.DataFrame({'line': LINES, 'value': CAP}),
    'price': pl.DataFrame({'line': LINES, 'value': PRICE}),
}


@pytest.mark.parametrize(
    ('where', 'kept'),
    [
        pytest.param('voltage == 220', ['loop', 'ring_a'], id='a-value-column-against-a-literal'),
        pytest.param("send == 'north'", ['loop', 'ring_a', 'spur'], id='a-value-column-against-a-label'),
        pytest.param('send != recv', ['ring_a', 'ring_b'], id='two-lookups-over-one-dimension'),
        pytest.param('recv', ['loop', 'ring_a', 'ring_b'], id='a-bare-name-is-the-masked-case'),
        pytest.param('NOT voltage == 220', ['ring_b', 'spur'], id='negated'),
        pytest.param('voltage == 380 AND send != recv', ['ring_b'], id='conjoined-with-a-pair-comparison'),
    ],
)
def test_a_where_reads_a_lookup(where, kept):
    """`kept` is asserted rather than a count: a predicate that inverted its
    sense would keep the complement, which is the same size on a symmetric case
    and a different model everywhere.

    `spur`'s missing `recv` row is the reading a masked lookup gets — a
    comparison over it is false, so the bare name keeps exactly the lines that
    have one.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    with lps.solve(spec, NETWORK_SOURCES) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == sorted(kept), f'where: {where!r} built the wrong set of variables'


@pytest.mark.parametrize(
    ('where', 'objective'),
    [
        pytest.param('send != recv', 30.0, id='the-two-ring-lines-survive'),
        pytest.param('NOT send != recv', 70.0, id='negated-over-a-masked-lookup'),
        # The two probes for the eager lane's explicit null exclusion. Only a
        # `!=` reaches it: numpy answers `None != 'north'` with True, so
        # without it the eager lane keeps exactly `spur` — the line with no
        # `recv` row — where the relational lane drops it.
        pytest.param("recv != 'north'", 10.0, id='not-equal-over-a-missing-row'),
        pytest.param('recv != send', 30.0, id='not-equal-between-two-lookups'),
    ],
)
def test_a_lookup_where_agrees_with_the_oracle(where, objective):
    """A mask reading a lookup is a join on the relation in the relational lane
    and an array read in the eager one; nothing but this shows they agree on
    which rows survive, since a wrong mask still solves."""
    from tests.differential import differential
    from tests.oracle import pd

    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    data = {'cap': pd.Series(CAP, index=LINES), 'price': pd.Series(PRICE, index=LINES)}
    index = {
        'bus': pd.Index(['north', 'south'], name='bus'),
        'line': pd.DataFrame({'line': LINES}),
        'grid': [220, 380],
        'send': pd.DataFrame({'line': LINES, 'bus': SEND}),
        'recv': pd.DataFrame({'line': LINES[:3], 'bus': RECV}),
        'voltage': pd.DataFrame({'line': LINES, 'grid': VOLTAGE}),
    }
    with differential(spec, data | index) as run:
        assert run.result.objective == pytest.approx(objective), (
            f'where: {where!r} — the two lanes agree on the objective but not on this one'
        )


def test_a_where_on_a_lookup_outside_the_frame_is_refused():
    """A keyed lookup is read at its key, so the key's dimensions have to be in
    the frame — otherwise the mask would silently reduce over an unlisted dim."""
    spec = {
        **NETWORK,
        'constraints': {
            'ceiling': {'foreach': ['bus'], 'where': 'voltage == 220', 'expression': 'sum(f, by=send) <= 100'}
        },
    }
    with pytest.raises(LpspecError, match=r"where-lookup 'voltage' reads dims \['line'\] outside the frame"):
        to_spec(spec)


def test_a_lookup_comparison_is_checked_against_the_dtype_of_the_column_it_reads():
    """`grid` is an int dimension, so a quoted right-hand side would match
    nothing rather than erroring at run time."""
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "voltage == 'high'"}}}
    with pytest.raises(LpspecError, match=r"has dtype 'int'"):
        to_spec(spec)


def test_a_comparison_against_a_label_the_dimension_lacks_masks_everything_out():
    """A stranger label masks everything out; it does not raise.

    The where-string rules' reading for every other comparison, and the reason
    the lookup column is compared as a string: attaching casts it to the
    dimension's `Enum`, which orders by declaration and *refuses* a label
    outside it — so without the cast back this is a polars error rather than an
    empty mask.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send == 'atlantis'"}}}
    with lps.build(spec, NETWORK_SOURCES) as model:
        surviving = model._engine._model.variables['f'].frame.select(pl.len()).collect().item()
    assert surviving == 0, "a label no bus carries matches nothing, so no 'f' is built"


def test_a_lookup_orders_bytewise_not_by_declaration():
    """Binding casts a lookup column to the dimension's `Enum`, which orders by
    *declaration*, so an ordering comparison read off it would answer a
    different question — and silently, since both readings return a mask.
    `south` is declared first here precisely so the two disagree."""
    sources = {**NETWORK_SOURCES, 'bus': pl.DataFrame({'bus': ['south', 'north']})}
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send >= 'south'"}}}
    with lps.solve(spec, sources) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == ['ring_b'], "only ring_b sends from 'south'; declaration order would keep the 'north' lines too"


# ---------------------------------------------------------------------------
# the walks a relation admits
# ---------------------------------------------------------------------------


#: A generator's zone, per period: a key of two columns, walked from either.
COMPOSITE = {
    'dimensions': {'generator': {'dtype': 'str'}, 'period': {'dtype': 'int'}, 'zone': {'dtype': 'str'}},
    'lookups': {'zone_of': {'over': ['generator', 'period', 'zone'], 'key': ['generator', 'period']}},
    'parameters': {'demand': {'dims': ['zone', 'period']}},
    'variables': {'p': {'foreach': ['generator', 'period'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'balance': {
            'foreach': ['zone', 'period'],
            'expression': 'sum(p, by=zone_of, from=generator, into=zone) >= demand',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

_COMPOSITE_SOURCES = {
    'generator': ['g1', 'g2'],
    'period': [2030, 2040],
    'zone': ['north', 'south'],
    'zone_of': pl.DataFrame(
        {
            'generator': ['g1', 'g1', 'g2', 'g2'],
            'period': [2030, 2040, 2030, 2040],
            'zone': ['north', 'south', 'south', 'south'],
        }
    ),
    'demand': pl.DataFrame(
        {
            'zone': ['north', 'south', 'north', 'south'],
            'period': [2030, 2030, 2040, 2040],
            'value': [3.0, 4.0, 0.0, 5.0],
        }
    ),
}


def test_a_key_of_two_columns_is_walked_from_the_one_the_call_names():
    """`zone_of` is a function of the pair, and `from=generator` walks it there.

    2030 asks 3 of the north zone, which only `g1` is in, and 4 of the south,
    which only `g2` is in; 2040 puts both in the south and asks 5 of it. So the
    optimum is 3 + 4 + 5 = 12, and no other split of the periods reaches it.
    """
    with lps.solve(COMPOSITE, _COMPOSITE_SOURCES) as result:
        assert result.objective == pytest.approx(12.0), (
            '3 in the north and 4 in the south in 2030, and 5 in the south in 2040'
        )
        built = by_coord(result, 'p', 'generator', 'period')
    assert built[('g1', 2030)] == pytest.approx(3.0), 'the north zone in 2030 holds g1 alone'


def test_the_same_table_walked_from_its_other_key_column():
    """`from=period` consumes the period and joins on the generator, so the
    result is per generator and zone — the same table, walked its other way.

    `g1` spends one period in each zone, so a cap of 6 per (generator, zone)
    pair holds each of its periods to 6 on its own; `g2` spends both in the
    south, so the pair holds those two together. The optimum is 6 + 6 + 6.
    """
    spec = {
        **COMPOSITE,
        'parameters': {},
        'constraints': {
            'history': {
                'foreach': ['generator', 'zone'],
                'expression': 'sum(p, by=zone_of, from=period, into=zone) <= 6',
            }
        },
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }
    sources = {k: v for k, v in _COMPOSITE_SOURCES.items() if k != 'demand'}
    with lps.solve(spec, sources) as result:
        assert result.objective == pytest.approx(18.0), "each of g1's periods reaches 6, and g2's two share one"


#: Two value columns of one table, landed on their product in one join.
PRODUCT = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}, 'technology': {'dtype': 'str'}},
    'lookups': {'gen_bt': {'over': ['generator', 'bus', 'technology'], 'key': 'generator'}},
    'parameters': {'tech_cap': {'dims': ['bus', 'technology']}},
    'variables': {'p': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'capped': {
            'foreach': ['bus', 'technology'],
            'expression': 'sum(p, by=gen_bt, into=[bus, technology]) <= tech_cap',
        }
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(p, over=generator)'},
}

_PRODUCT_SOURCES = {
    'generator': ['g1', 'g2', 'g3'],
    'bus': ['north', 'south'],
    'technology': ['wind', 'gas'],
    'gen_bt': pl.DataFrame(
        {
            'generator': ['g1', 'g2', 'g3'],
            'bus': ['north', 'north', 'south'],
            'technology': ['wind', 'gas', 'wind'],
        }
    ),
    'tech_cap': pl.DataFrame(
        {
            'bus': ['north', 'north', 'south', 'south'],
            'technology': ['wind', 'gas', 'wind', 'gas'],
            'value': [2.0, 3.0, 4.0, 5.0],
        }
    ),
}


def test_one_table_lands_on_a_product_of_two_value_columns():
    """Each generator sits at one (bus, technology) slot and takes that slot's
    cap: 2 + 3 + 4 = 9, with the south's gas cap of 5 reaching nobody."""
    with lps.solve(PRODUCT, _PRODUCT_SOURCES) as result:
        assert result.objective == pytest.approx(9.0), 'each generator takes the cap of its own slot: 2 + 3 + 4'


#: No key at all: a generator may connect to several buses.
BARE = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'lookups': {'connection': {'over': ['generator', 'bus']}},
    'parameters': {'load': {'dims': ['bus']}},
    'variables': {'p': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'serve': {'foreach': ['bus'], 'expression': 'sum(p, by=connection, from=generator, into=bus) >= load'}
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p, over=generator)'},
}

_BARE_SOURCES = {
    'generator': ['g1', 'g2'],
    'bus': ['north', 'south'],
    'connection': pl.DataFrame({'generator': ['g1', 'g1', 'g2'], 'bus': ['north', 'south', 'south']}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [3.0, 4.0]}),
}


def test_a_bare_relation_is_summed_through_with_both_ends_named():
    """`g1` reaches both buses and `g2` only the south, so `g1` alone covers the
    north's 3 and the south's 4 at once: the cheapest total is 4, not 7."""
    with lps.solve(BARE, _BARE_SOURCES) as result:
        assert result.objective == pytest.approx(4.0), (
            'g1 reaches both buses, so it covers the north 3 and the south 4 at once'
        )
        built = by_coord(result, 'p', 'generator')
    assert built['g2'] == pytest.approx(0.0), 'a bare relation adds every row it finds, so one generator serves both'


#: A dimension related to itself: every snapshot names the one that stands for it.
SELF_MAP = {
    'dimensions': {'snapshot': {'dtype': 'int'}},
    'lookups': {'rep_of': {'over': {'snapshot': 'snapshot', 'rep': 'snapshot'}, 'key': 'snapshot'}},
    'parameters': {'cap': {'dims': ['snapshot']}},
    'variables': {'p': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'follow': {'foreach': ['snapshot'], 'expression': 'p == at(p, by=rep_of)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p, over=snapshot)'},
}

_SELF_SOURCES = {
    'snapshot': [0, 1, 2, 3],
    'rep_of': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'rep': [0, 0, 2, 2]}),
    'cap': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [4.0, 9.0, 1.0, 9.0]}),
}


def test_a_self_map_reads_along_its_own_arrow():
    """Two columns over one dimension, so the walk consumes `snapshot` and
    produces `snapshot` — and the frame is unchanged through it.

    Snapshots 0 and 1 take representative 0 and 2 and 3 take representative 2,
    so each group is held to its representative's cap: 4 twice and 1 twice, and
    the caps of 9 on the followers never bind.
    """
    with lps.solve(SELF_MAP, _SELF_SOURCES) as result:
        assert result.objective == pytest.approx(10.0), 'each group is held to its representative: 4 twice and 1 twice'
        held = by_coord(result, 'p', 'snapshot')
    assert held[1] == pytest.approx(4.0), "a follower takes its representative's value, not its own cap"


#: The produced dimension the operand already carries: a masked sum.
MASKED_SUM = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}, 'snapshot': {'dtype': 'int'}},
    'lookups': {'gen_bus': {'over': ['generator', 'bus'], 'key': 'generator'}},
    'parameters': {'load': {'dims': ['snapshot', 'bus']}},
    'variables': {'p': {'foreach': ['generator', 'snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'balance': {'foreach': ['snapshot', 'bus'], 'expression': 'sum(load * p, by=gen_bus) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

_MASKED_SUM_SOURCES = {
    'generator': ['g1', 'g2'],
    'bus': ['north', 'south'],
    'snapshot': [0, 1],
    'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']}),
    'load': pl.DataFrame(
        {'snapshot': [0, 0, 1, 1], 'bus': ['north', 'south', 'north', 'south'], 'value': [1.0, 2.0, 3.0, 4.0]}
    ),
}


def test_a_produced_dimension_the_operand_carries_is_joined_on():
    """`load` already spans `bus`, so the sum restricts each term to the row
    where the generator's bus is the row's bus.

    Every row then reads `load * p >= load` for its own generator alone, so
    each of the four `p` sits at 1 and the objective is 4. Summed without the
    mask each row would see both generators and the total would be 2.
    """
    with lps.solve(MASKED_SUM, _MASKED_SUM_SOURCES) as result:
        assert result.objective == pytest.approx(4.0), (
            'each row reads load * p >= load for its own generator alone, so every p sits at 1'
        )


# ---------------------------------------------------------------------------
# what the linopy lane cannot walk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('spec', 'sources', 'match'),
    [
        pytest.param(COMPOSITE, _COMPOSITE_SOURCES, r"keyed by \['generator', 'period'\]", id='a-composite-key'),
        pytest.param(BARE, _BARE_SOURCES, 'declares no key', id='a-bare-relation'),
        pytest.param(SELF_MAP, _SELF_SOURCES, "relates 'snapshot' to itself", id='a-self-map'),
    ],
)
def test_a_walk_the_linopy_lane_cannot_read_is_refused_before_linopy_is_asked(spec, sources, match):
    """This lane holds a lookup as a dense array over the dimension its key is
    over, so a walk that is not a function of one dimension has no array to
    read — refused in the language's own words rather than as an xarray error
    from inside a half-built model, and pointing at the lane that does build it.
    """
    from tests.oracle import lpspec_linopy

    with pytest.raises(LaneError, match=match) as caught:
        lpspec_linopy.build(spec, sources)
    assert 'relational engine' in str(caught.value), 'the refusal names the lane that takes the model as it stands'


#: A calendar keyed by the pair: each plant keeps its own days, so a partition
#: walking `t` joins on `plant` rather than ranking every plant together.
PER_PLANT_CALENDAR = {
    'dimensions': {'t': {'dtype': 'int'}, 'plant': {'dtype': 'str'}, 'day': {'dtype': 'str'}},
    'lookups': {'day_of': {'over': ['t', 'plant', 'day'], 'key': ['t', 'plant']}},
    'parameters': {'price': {'dims': ['plant', 't']}},
    'variables': {'x': {'foreach': ['plant', 't'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'ramp': {'foreach': ['plant', 't'], 'expression': 'x <= shift(x, over=t, offset=1, edge=0, by=day_of) + 1'}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x * price)'},
}

_CALENDAR_SOURCES = {
    't': [0, 1, 2, 3],
    'plant': ['p1', 'p2'],
    'day': ['mon', 'tue'],
    'day_of': pl.DataFrame(
        {
            't': [0, 1, 2, 3, 0, 1, 2, 3],
            'plant': ['p1'] * 4 + ['p2'] * 4,
            'day': ['mon', 'mon', 'tue', 'tue', 'mon', 'tue', 'tue', 'tue'],
        }
    ),
    'price': pl.DataFrame({'plant': ['p1'] * 4 + ['p2'] * 4, 't': [0, 1, 2, 3] * 2, 'value': [1.0] * 8}),
}


def test_a_partition_joins_on_the_key_columns_it_does_not_walk():
    """A ramp of 1 per step, restarting at each plant's own day boundary.

    `p1` runs mon-mon-tue-tue and `p2` mon-tue-tue-tue, so the two see different
    boundaries at the same coordinates: `p1` climbs 1, 2 and restarts at 1, 2,
    while `p2` restarts once and climbs 1, 2, 3 — 13 in all. Ranked without the
    join on `plant`, both plants would sit in one group per day and the
    within-group positions would be somebody else's.
    """
    with lps.solve(PER_PLANT_CALENDAR, _CALENDAR_SOURCES) as result:
        assert result.objective == pytest.approx(13.0), "each plant's days are its own, so the two restart apart"
        climbed = by_coord(result, 'x', 'plant', 't')
    assert climbed[('p2', 3)] == pytest.approx(3.0), 'p2 spends three steps in one day and reaches 3'
    assert climbed[('p1', 3)] == pytest.approx(2.0), 'p1 spends two, and a new day put it back to 1'


def test_a_bare_relation_in_a_where_is_read_relationally_and_refused_on_the_lane():
    """`where: connection` tests that a row exists, at every column of a bare relation.

    So the frame carries both, and the three related pairs are the three rows
    the constraint builds out of four. The relational engine meets that as a
    semi-join on the pair; the linopy lane holds a lookup as an array indexed
    by a key, and a relation with no key has none, so it says so rather than
    failing inside the mask.
    """
    from tests.oracle import lpspec_linopy

    spec = {
        **BARE,
        'constraints': {'related': {'foreach': ['generator', 'bus'], 'where': 'connection', 'expression': 'p <= load'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(p, over=generator)'},
    }
    with lps.build(spec, _BARE_SOURCES) as model:
        assert model.diagnostics().rows == 3, 'g1 is related to both buses and g2 to one, so four pairs make three rows'

    with pytest.raises(LaneError, match="cannot read lookup 'connection' in a where"):
        lpspec_linopy.build(spec, _BARE_SOURCES)


def test_a_composite_key_read_in_a_where_agrees_between_the_lanes():
    """A keyed lookup is read at its key, and a key of two columns is read at both.

    Nothing about that is the relational engine's alone — the eager lane holds
    the value column as an array over the pair — so this is the one shape of
    `where` that both lanes take and only this test walks.
    """
    from tests.differential import differential
    from tests.oracle import pd

    spec = {
        **COMPOSITE,
        'parameters': {},
        'constraints': {
            'northern': {
                'foreach': ['generator', 'period'],
                'where': "zone_of == 'north'",
                'expression': 'p <= 2',
            }
        },
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }
    sources = {
        'generator': ['g1', 'g2'],
        'period': [2030, 2040],
        'zone': ['north', 'south'],
        'zone_of': pd.DataFrame(
            {
                'generator': ['g1', 'g1', 'g2', 'g2'],
                'period': [2030, 2040, 2030, 2040],
                'zone': ['north', 'south', 'south', 'south'],
            }
        ),
    }
    with differential(spec, sources) as run:
        assert run.result.objective == pytest.approx(32.0), 'only (g1, 2030) is northern, and it alone is capped at 2'
