"""`lps.evaluate(spec, sources)`: a spec's named expressions as arithmetic.

A spec with no variables is a calculation, not an optimisation — dimensions,
parameters, lookups and ``expressions:`` — so each expression has a value with
no solver and no chosen point. What is pinned here: the value is the arithmetic
the data implies, the frame's dims are the ones the expression survives over, a
grouped sum relabels through a lookup, laziness (nothing compiles until a read),
the unknown-name refusal, the closed refusal, and the refusal that names
`solve` for a spec that declares a decision.

The engine underneath is the same one a solve reads named expressions through
(``test_expression_reader.py``); evaluate hands it a compiler carrying no
solution, so the value tests here and the at-a-solution tests there share it.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LpspecError
from lpspec.relational.engines.polars.compiler import PolarsCompiler

SPEC = {
    'dimensions': {
        'snapshot': {'dtype': 'int'},
        'generator': {'dtype': 'str'},
        'node': {'dtype': 'str'},
    },
    'parameters': {
        'dispatch': {'dims': ['snapshot', 'generator']},
        'cost': {'dims': ['generator']},
    },
    'lookups': {'bus': {'over': 'generator', 'into': 'node'}},
    'expressions': {
        'cost_by_gen': 'dispatch * cost',
        'total_cost': 'sum(dispatch * cost)',
        'served': 'sum(dispatch, over=generator)',
        'by_node': 'sum(dispatch, by=bus)',
    },
}

GENERATORS = ['wind', 'gas']


def sources() -> dict[str, object]:
    return {
        'snapshot': [0, 1, 2],
        'generator': GENERATORS,
        'node': ['n1', 'n2'],
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 50.0]}),
        'dispatch': pl.DataFrame(
            {
                'snapshot': [0, 0, 1, 1, 2, 2],
                'generator': GENERATORS * 3,
                'value': [10.0, 2.0, 20.0, 3.0, 5.0, 7.0],
            }
        ),
        'bus': pl.DataFrame({'generator': GENERATORS, 'node': ['n1', 'n2']}),
    }


@pytest.fixture
def ev():
    return lps.evaluate(SPEC, sources())


def test_a_scalar_expression_is_the_arithmetic_the_data_implies(ev):
    frame = ev.expression('total_cost')
    assert frame.columns == ['value'] and frame.height == 1, 'an expression with no dims is a single value row'
    assert frame.item() == pytest.approx(635.0), 'sum(dispatch * cost) is 10·1 + 2·50 + 20·1 + 3·50 + 5·1 + 7·50 = 635'


def test_a_sum_over_one_of_two_dims_is_the_per_coordinate_total(ev):
    got = dict(zip(*ev.expression('served').sort('snapshot'), strict=True))
    assert got == pytest.approx({0: 12.0, 1: 23.0, 2: 12.0}), (
        'sum(dispatch, over=generator) adds the two generators at each snapshot'
    )


def test_a_grouped_sum_relabels_through_a_lookup(ev):
    frame = ev.expression('by_node').sort('snapshot', 'node')
    assert frame.columns == ['snapshot', 'node', 'value'], 'a group over generator lands on node, keeping snapshot'
    assert frame['value'].to_list() == pytest.approx([10.0, 2.0, 20.0, 3.0, 5.0, 7.0]), (
        'each generator maps to its own node, so the group is a relabel rather than a reduction here'
    )


@pytest.mark.parametrize(
    ('name', 'dims'),
    [
        pytest.param('cost_by_gen', {'snapshot', 'generator'}, id='a-product-over-both'),
        pytest.param('served', {'snapshot'}, id='summed-over-one-of-two'),
        pytest.param('total_cost', set(), id='summed-over-both'),
        pytest.param('by_node', {'snapshot', 'node'}, id='grouped-to-node'),
    ],
)
def test_the_frame_carries_exactly_the_dims_the_expression_survives_over(ev, name, dims):
    frame = ev.expression(name)
    assert set(frame.columns) - {'value'} == dims, (
        'the returned frame answers over the dims the expression still ranges over after its sums'
    )


def test_expressions_lists_the_declared_names_in_declaration_order(ev):
    assert ev.expressions == ('cost_by_gen', 'total_cost', 'served', 'by_node'), (
        'expressions names what the file declares, in the order it declares them'
    )


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('nope', id='a-typo'),
        pytest.param('sum(dispatch, over=generator)', id='an-expression-string'),
    ],
)
def test_expression_lists_the_declared_names_and_points_a_string_at_evaluate(ev, name):
    with pytest.raises(KeyError, match=r'by_node, cost_by_gen, served, total_cost') as caught:
        ev.expression(name)
    assert 'evaluate() takes an expression string' in str(caught.value), (
        'expression() takes declared names only; the refusal points an ad-hoc string at evaluate()'
    )


def test_evaluate_values_an_expression_the_file_never_named(ev):
    got = dict(zip(*ev.evaluate('sum(dispatch, over=generator)').sort('snapshot'), strict=True))
    assert got == pytest.approx({0: 12.0, 1: 23.0, 2: 12.0}), (
        'evaluate lowers an ad-hoc expression against the var-free spec and reads it as arithmetic'
    )


def test_evaluate_serves_a_declared_name_from_its_own_reader(ev):
    assert ev.evaluate('total_cost').item() == pytest.approx(635.0), (
        'a declared name handed to evaluate is served by its reader rather than lowered again'
    )


@pytest.mark.parametrize(
    ('decision', 'names'),
    [
        pytest.param(
            {'variables': {'x': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 'cost'}}}},
            'variables (x)',
            id='a-variable',
        ),
        pytest.param(
            {
                'variables': {'x': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 'cost'}}},
                'objective': {'sense': 'minimize', 'expression': 'sum(x * cost)'},
            },
            'an objective',
            id='a-variable-and-objective',
        ),
    ],
)
def test_a_spec_that_declares_a_decision_is_refused_and_names_solve(decision, names):
    spec = {**SPEC, **decision}
    with pytest.raises(LpspecError, match=r'lps\.solve') as caught:
        lps.evaluate(spec, sources())
    assert names in str(caught.value), 'the refusal names the decision it found, so the author sees what to drop'


def test_a_closed_evaluation_refuses_a_read(ev):
    ev.close()
    with pytest.raises(LpspecError, match='closed'):
        ev.expression('total_cost')


def test_the_context_manager_closes_on_exit():
    with lps.evaluate(SPEC, sources()) as ev:
        assert ev.expression('total_cost').item() == pytest.approx(635.0), 'readable inside the block'
    with pytest.raises(LpspecError, match='closed'):
        ev.expression('total_cost')


def test_nothing_is_compiled_until_an_expression_is_read(monkeypatch):
    compiled = []
    original = PolarsCompiler.expression

    def counting(self, expr, context, **kwargs):
        compiled.append(context)
        return original(self, expr, context, **kwargs)

    monkeypatch.setattr(PolarsCompiler, 'expression', counting)
    ev = lps.evaluate(SPEC, sources())
    assert compiled == [], 'evaluate defers every expression — four declared and none read must compile none'
    ev.expression('served')
    assert compiled == ["named expression 'served'"], 'reading one expression compiles that one expression'


def test_to_pandas_bridges_out(ev):
    pytest.importorskip('pandas')
    frame = ev.to_pandas('served')
    assert list(frame.columns) == ['snapshot', 'value'], 'the pandas frame keeps the tidy columns'
    assert frame['value'].tolist() == pytest.approx([12.0, 23.0, 12.0]), 'and the same values'
