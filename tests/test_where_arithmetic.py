"""A ``where`` comparing expressions, read the same on both lanes and by the absence rules.

The language reads either side of a comparison as an expression
(math-spec's expressions page, *arithmetic in a comparison*), and says two
things about a value that is not there: a side with no value at a coordinate
compares false, and under a summing operator the absent term is one fewer.
Each lane answers with its own machinery — a polars join per side, an xarray
value beside a structural "has a value" — so the table below pins both to the
rows the rules name rather than only to each other.
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from tests.conftest import DISPATCH_SPEC, dispatch_spec_path, override
from tests.oracle import lpspec_linopy, pd  # skips the module without the [linopy] extra

#: The dispatch model with one more parameter, ``extra``, supplied for ``gas`` only.
SPEC = override(DISPATCH_SPEC, **{'parameters.extra': {'dims': ['generator']}})

WIND, GAS = 'wind', 'gas'
EVERYWHERE = [(t, g) for t in range(4) for g in (WIND, GAS)]


@pytest.fixture
def inputs():
    return {
        'p_max': pd.Series({WIND: 100.0, GAS: 200.0}),
        'cost': pd.Series({WIND: 0.0, GAS: 50.0}),
        'extra': pd.Series({GAS: 1.0}),
        'load': pd.Series([80.0, 90.0, 70.0, 100.0], index=pd.RangeIndex(4, name='snapshot')),
        'snapshot': pd.RangeIndex(4, name='snapshot'),
        'generator': [WIND, GAS],
    }


@pytest.mark.parametrize(
    ('where', 'rows'),
    [
        pytest.param('2 * cost + 50 <= p_max', EVERYWHERE, id='arithmetic-on-both-sides'),
        pytest.param('p_max + extra > 50', EVERYWHERE, id='an-absent-term-under-plus-is-one-fewer'),
        pytest.param('p_max + extra > 100', [(t, GAS) for t in range(4)], id='the-present-term-alone-is-compared'),
        pytest.param('extra * 2 == 2', [(t, GAS) for t in range(4)], id='an-absent-factor-compares-false'),
        pytest.param('extra * 2 != 1', [(t, GAS) for t in range(4)], id='an-absent-side-is-false-even-under-not-equal'),
        pytest.param('load / extra > 50', [(t, GAS) for t in range(4)], id='an-absent-divisor-compares-false'),
        pytest.param('p_max ** 2 > 20000', [(t, GAS) for t in range(4)], id='a-power'),
        pytest.param('sum(extra, over=generator) > 0', EVERYWHERE, id='an-absent-term-under-a-sum-is-one-fewer'),
        pytest.param(
            'sum(p_max, over=generator) - 250 > load', [], id='a-reduction-beside-a-parameter-over-another-dim'
        ),
        pytest.param(
            'shift(load, along=snapshot, offset=1, edge=0) < 85',
            [(t, g) for t in (0, 1, 3) for g in (WIND, GAS)],
            id='a-bare-shift-has-a-value-at-the-edge-it-names',
        ),
        pytest.param(
            'load - shift(load, along=snapshot, offset=1, edge=0) > 0',
            [(t, g) for t in (0, 1, 3) for g in (WIND, GAS)],
            id='a-shift-with-an-edge-has-a-value-at-the-first-row',
        ),
        pytest.param(
            'load - shift(load, along=snapshot, offset=1, edge=0) > 0 AND position(snapshot) > 0',
            [(t, g) for t in (1, 3) for g in (WIND, GAS)],
            id='beside-a-position-predicate',
        ),
    ],
)
def test_both_lanes_keep_the_rows_the_absence_rules_name(tmp_path, inputs, where, rows):
    """The rows ``p`` is built at, on each lane, against the rows the rules name.

    ``extra`` is the sparse one: ``p_max + extra`` has a value at ``wind``
    (one term fewer), ``extra * 2`` does not (a factor with no value), and
    ``!=`` is the comparison that would read a missing value as *true* on
    a lane that compared NaN rather than asking where the side has a value.
    """
    path = dispatch_spec_path(tmp_path, **{'parameters.extra': SPEC['parameters']['extra'], 'variables.p.where': where})
    expected = sorted(rows)

    built = lpspec_linopy.build(path, dict(inputs))
    labels = built.variables['p'].labels.to_dataframe('label').reset_index()
    eager = sorted(map(tuple, labels[labels['label'] != -1][['snapshot', 'generator']].itertuples(index=False)))
    assert eager == expected, f'{where}: the linopy lane built p at {eager}'

    with lps.build(path, dict(inputs)) as model:
        frame = model._engine._model.variables['p'].frame.select('snapshot', 'generator').collect()
    relational = sorted(map(tuple, frame.iter_rows()))
    assert relational == expected, f'{where}: the relational lane built p at {relational}'


def test_a_named_expression_expands_inside_a_comparison(tmp_path, inputs):
    """A side is read as an expression is, so a named one stands in it and the relational lane reads it through the same door."""
    path = dispatch_spec_path(
        tmp_path,
        **{
            'parameters.extra': SPEC['parameters']['extra'],
            'expressions.headroom': {'expression': 'p_max - cost'},
            'variables.p.where': 'headroom > 120',
        },
    )
    with lps.build(path, dict(inputs)) as model:
        frame = model._engine._model.variables['p'].frame.select('generator').unique().collect()
    assert frame['generator'].to_list() == [GAS], 'p_max - cost is 100 for wind and 150 for gas'
    built = lpspec_linopy.build(path, dict(inputs))
    assert int((built.variables['p'].labels != -1).sum()) == 4, 'the linopy lane keeps the same four rows'


def test_the_polars_side_is_null_where_no_piece_has_a_value():
    """The relational lane's reading of a whole side, at the join: null, not zero, where nothing adds up.

    `p_max + extra` is a sum of two pieces, so it has a value wherever either
    does; `extra * 2` is one piece with a hole in it. A lane that filled the
    hole with zero on the way in would read `extra * 2 == 0` as true at
    ``wind``.
    """
    from math_spec import to_program

    from lpspec.relational.engines.polars.attaching import attach
    from lpspec.relational.engines.polars.compiler import PolarsCompiler
    from lpspec.relational.engines.polars.predicates import compile_predicate
    from lpspec.sources import tidy_sources

    spec = override(SPEC, **{'variables.p.where': 'extra * 2 == 0 OR p_max + extra > 0'})
    plan = to_program(spec)
    sources = {
        'p_max': pl.DataFrame({'generator': [WIND, GAS], 'value': [100.0, 200.0]}),
        'cost': pl.DataFrame({'generator': [WIND, GAS], 'value': [0.0, 50.0]}),
        'extra': pl.DataFrame({'generator': [GAS], 'value': [1.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [80.0] * 4}),
        'snapshot': pl.DataFrame({'snapshot': [0, 1, 2, 3]}),
        'generator': pl.DataFrame({'generator': [WIND, GAS]}),
    }
    compiler = PolarsCompiler(plan, attach(plan, tidy_sources(plan, sources)), {})
    mask = plan.variables['p'].where
    assert mask is not None
    product = compiler.frame(('generator',), None)
    carrier, condition = compile_predicate(compiler, product, mask, ('generator',))
    sides = sorted(c for c in carrier.collect_schema().names() if c.startswith('__where side'))
    assert len(sides) == 4, 'each of the four sides is its own column'
    read = carrier.select('generator', *sides, condition.alias('holds')).sort('generator').collect()
    assert read.filter(pl.col('generator') == WIND).select(sides[0]).item() is None, 'extra * 2 is null at wind'
    assert read.filter(pl.col('generator') == WIND).select(sides[2]).item() == 100.0, 'p_max + extra is 100 at wind'
    assert read['holds'].to_list() == [True, True], 'both survive on the second disjunct'


@pytest.mark.parametrize(
    ('where', 'kept'),
    [
        pytest.param('cap <= at(bus_cap, by=send)', ['loop', 'ring_a'], id='a-pullback-with-no-value-at-south'),
        pytest.param('at(bus_cap, by=recv) > 0', ['loop', 'ring_b'], id='a-pullback-through-a-partial-lookup'),
        pytest.param('at(sum(cap, by=send), by=send) > 50', ['loop', 'ring_a', 'spur'], id='a-grouped-sum-read-back'),
        pytest.param('sum_back(cap, along=line, window=2) > 25', ['loop', 'ring_b', 'spur'], id='a-window'),
    ],
)
def test_both_lanes_read_a_lookup_operator_inside_a_side(tmp_path, where, kept):
    """The operators that read a lookup, inside a side: `bus_cap` is supplied for `north` only, and `spur` has no `recv`."""
    from tests.test_label_coords import NETWORK, NETWORK_SOURCES

    spec = override(NETWORK, **{'parameters.bus_cap': {'dims': ['bus']}, 'variables.f.where': where})
    sources = dict(NETWORK_SOURCES, bus_cap=pl.DataFrame({'bus': ['north'], 'value': [35.0]}))
    path = tmp_path / 'network.yaml'
    path.write_text(pyyaml.safe_dump(spec))

    with lps.build(path, dict(sources)) as model:
        relational = sorted(model._engine._model.variables['f'].frame.select('line').collect()['line'].to_list())
    assert relational == kept, f'{where}: the relational lane built f at {relational}'

    built = lpspec_linopy.build(path, {name: table.to_pandas() for name, table in sources.items()})
    labels = built.variables['f'].labels.to_dataframe('label').reset_index()
    eager = sorted(labels[labels['label'] != -1]['line'])
    assert eager == kept, f'{where}: the linopy lane built f at {eager}'


def test_a_cased_expression_inside_a_side_is_read_region_by_region(tmp_path, inputs):
    """A `cases:` expression in a side: each coordinate is worth the one region that claims it."""
    tier = {
        'dims': ['generator'],
        'cases': {'cheap': {'when': 'cost <= 10', 'expression': 'p_max'}},
        'otherwise': 'p_max / 4',
    }
    path = dispatch_spec_path(
        tmp_path,
        **{'parameters.extra': SPEC['parameters']['extra'], 'expressions.tier': tier, 'variables.p.where': 'tier > 90'},
    )
    with lps.build(path, dict(inputs)) as model:
        frame = model._engine._model.variables['p'].frame.select('generator').unique().collect()
    assert frame['generator'].to_list() == [WIND], 'wind is worth 100 in its region and gas 50 in the other'
    built = lpspec_linopy.build(path, dict(inputs))
    assert int((built.variables['p'].labels != -1).sum()) == 4, 'the linopy lane keeps the same four rows'
