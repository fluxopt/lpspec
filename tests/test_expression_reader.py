"""`Result.expression(name)`: a named expression readable after a solve (#562).

The relational lane only — the differential half, both lanes agreeing on the
same values, lives in ``test_linopy_lane.py`` with the rest of the oracle
comparisons. What is pinned here: the value is the one the primal implies, an
expression no constraint references still reads, the frame's dims are
the ones it survives over, laziness (a build compiles no expression; a read compiles that
one), and the unknown-name refusal.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DataError, LpspecError
from lpspec.relational.engines.polars.compiler import PolarsCompiler
from tests.fixtures import override

SPEC = {
    'dimensions': {
        'snapshot': {'dtype': 'int'},
        'generator': {'dtype': 'str'},
    },
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {
        'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
    },
    'expressions': {
        'total_gen': 'sum(p, over=generator)',
        'spend': 'sum(p * cost, over=generator)',
        'answer': '21 * 2',
        'total_cost': 'sum(sum(p * cost, over=generator), over=snapshot)',
        'squared': 'sum(p * p, over=generator)',
        'price': 'dual(balance)',
        'weighted': 'dual(balance) * total_gen',
        'rational': '1 / (1 + total_gen)',
    },
    'constraints': {
        'balance': {'foreach': ['snapshot'], 'expression': 'total_gen == load'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(sum(p * cost, over=generator), over=snapshot)'},
}


def sources() -> dict[str, pl.DataFrame]:
    return {
        'snapshot': [0, 1, 2],
        'generator': ['g1', 'g2'],
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [100.0, 100.0]}),
        'cost': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [10.0, 20.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [50.0, 120.0, 80.0]}),
    }


@pytest.fixture(scope='module')
def result():
    """One solve for the whole module — `lps.solve` closes the model, so
    every read below also proves the readers outlive it."""
    return lps.solve(SPEC, sources())


def test_a_referenced_expression_reads_the_value_its_constraint_pinned(result):
    frame = result.expression('total_gen')
    assert frame.columns == ['snapshot', 'value'], 'an expression frame is (dims…, value), dims in declaration order'
    got = dict(zip(frame['snapshot'], frame['value'], strict=True))
    assert got == pytest.approx({0: 50.0, 1: 120.0, 2: 80.0}), (
        'balance pins total_gen to load, so the reader must hand back exactly the load values'
    )


def test_an_expression_nothing_references_reads_the_value_the_primal_implies(result):
    external = (
        result.primal('p')
        .join(sources()['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost')).sum().alias('value'))
        .sort('snapshot')
    )
    frame = result.expression('spend')
    assert frame.sort('snapshot').equals(external), (
        'spend is referenced by nothing, and must still equal sum(p * cost) computed from the primal by hand'
    )


def test_a_scalar_expression_is_one_row_matching_the_objective(result):
    frame = result.expression('total_cost')
    assert frame.columns == ['value'] and frame.height == 1, 'an expression with no dims is a single value row'
    assert frame.item() == pytest.approx(result.objective), (
        'total_cost restates the objective, so the two numbers must agree'
    )


def test_a_variable_free_expression_is_legal_and_reads_its_constant(result):
    assert result.expression('answer').item() == pytest.approx(42.0), (
        'the grammar admits a variable-free named expression, and its value is the constant it spells'
    )


@pytest.mark.parametrize(
    ('name', 'dims'),
    [
        pytest.param('total_gen', {'snapshot'}, id='summed-over-one-of-two'),
        pytest.param('spend', {'snapshot'}, id='a-product-summed-over-one-of-two'),
        pytest.param('answer', set(), id='a-constant'),
        pytest.param('total_cost', set(), id='summed-over-both'),
    ],
)
def test_the_frame_carries_exactly_the_dims_the_expression_survives_over(result, name, dims):
    frame = result.expression(name)
    assert set(frame.columns) - {'value'} == dims, (
        'the returned frame answers over the dims the expression still ranges over after its sums'
    )


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('nope', id='a-typo'),
        pytest.param('sum(p, over=generator)', id='an-expression-string'),
    ],
)
def test_an_unknown_name_lists_the_declared_names_and_refuses_strings(result, name):
    with pytest.raises(
        KeyError, match=r'answer, price, rational, spend, squared, total_cost, total_gen, weighted'
    ) as caught:
        result.expression(name)
    assert 'never an expression string' in str(caught.value), (
        'the refusal must say expression() takes declared names only, not arbitrary expression strings'
    )


def test_a_masked_coordinate_has_no_row():
    masked = {
        **SPEC,
        'variables': {
            'p': {
                'foreach': ['snapshot', 'generator'],
                'bounds': {'lower': 0, 'upper': 'p_max'},
                'where': 'p_max > 0',
            }
        },
        'expressions': {'scaled': 'p * cost'},
        'constraints': {'balance': {'foreach': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    }
    data = sources() | {
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [200.0, 0.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [50.0, 120.0, 80.0]}),
    }
    frame = lps.solve(masked, data).expression('scaled')
    assert frame['generator'].unique().to_list() == ['g1'], (
        'absence propagates into a reader the way it does into a constraint (the operator rules): the masked-out '
        "generator's coordinates have no rows rather than zeros"
    )
    assert frame.height == 3, 'the surviving generator keeps one row per snapshot'


def test_an_entry_of_degree_two_reads_the_primal_squared(result):
    """The language holds an entry the math never reads to no degree, and a read needs none: every variable is a number by then."""
    primal = result.primal('p')
    want = primal.with_columns(pl.col('value') ** 2).group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
    got = result.expression('squared')
    assert got['snapshot'].to_list() == want['snapshot'].to_list(), 'one row per snapshot, in label order'
    assert got['value'].to_list() == pytest.approx(want['value'].to_list()), (
        'p * p at the solution is each primal squared, summed over generators'
    )


def test_an_entry_reads_a_constraints_dual(result):
    assert result.expression('price').equals(result.dual('balance')), (
        "dual(balance) is the constraint's own dual frame, row for row"
    )


def test_a_dual_multiplies_like_any_number(result):
    dual = result.dual('balance')
    load = sources()['load']
    want = [d * v for d, v in zip(dual['value'], load['value'], strict=True)]
    assert result.expression('weighted')['value'].to_list() == pytest.approx(want), (
        'balance pins total_gen to load, so dual(balance) * total_gen is the dual times the load'
    )


def test_a_divisor_that_adds_is_added_up_before_it_divides(result):
    want = [1 / (1 + v) for v in sources()['load']['value']]
    assert result.expression('rational')['value'].to_list() == pytest.approx(want), (
        'no degree rule holds an entry the math never reads, so 1 / (1 + total_gen) is one value per snapshot'
    )


def test_a_dual_on_a_solve_that_left_none_is_refused_by_name():
    """An integer variable makes duals undefined; the entry reading one is refused with `Result.dual`'s own sentence, and every other entry still reads."""
    result = lps.solve(override(SPEC, **{'variables.p.domain': 'integer'}), sources())
    with pytest.raises(LpspecError, match='duals are undefined'):
        result.expression('price')
    assert result.expression('spend').height == 3, 'the refusal is per entry, not per result'


def test_a_divisor_that_adds_keeps_its_hole():
    """Adding up a divisor must not invent a zero: a coordinate no piece covers stays null, so the division reports it rather than dividing by it."""
    spec = override(
        SPEC,
        **{
            'parameters.scale': {'dims': ['snapshot']},
            'parameters.other': {'dims': ['snapshot']},
            'expressions.holed': '1 / (scale + other)',
        },
    )
    covered = pl.DataFrame({'snapshot': [0, 1], 'value': [2.0, 3.0]})
    result = lps.solve(spec, sources() | {'scale': covered, 'other': covered})
    with pytest.raises(DataError, match='used as a divisor but covers 1 fewer'):
        result.expression('holed')


@pytest.mark.parametrize('crossed', [pytest.param('p * r', id='a-product'), pytest.param('p ** r', id='a-power')])
def test_a_product_is_absent_where_either_factor_is(crossed):
    """Presence travels out of a product, and a power, from both sides — as it does out of a quadratic term at a build.

    `r` is masked out at `g2` and is 1 where it exists, so `p * r` and `p ** r`
    both read `p`; `bonus`, a constant over the same dims, is owed only where
    the crossed term exists, so the sum over generators reads it at `g1` alone.
    Losing `r`'s presence would add `bonus` at `g2` back in.
    """
    spec = override(
        SPEC,
        **{
            'parameters.r_max': {'dims': ['generator']},
            'parameters.bonus': {'dims': ['snapshot', 'generator']},
            'variables.r': {
                'foreach': ['snapshot', 'generator'],
                'bounds': {'lower': 1, 'upper': 1},
                'where': 'r_max > 0',
            },
            'expressions.summed_with': f'sum({crossed} + bonus, over=generator)',
        },
    )
    bonus = pl.DataFrame({'snapshot': [0, 0, 1, 1, 2, 2], 'generator': ['g1', 'g2'] * 3, 'value': [10.0] * 6})
    data = sources() | {'r_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [1.0, 0.0]}), 'bonus': bonus}
    result = lps.solve(spec, data)
    p = result.primal('p').filter(pl.col('generator') == 'g1').sort('snapshot')
    want = [v * 1.0 + 10.0 for v in p['value']]
    assert result.expression('summed_with')['value'].to_list() == pytest.approx(want), (
        f'the sum reads {crossed} and bonus at g1 only, since r is absent at g2'
    )


def test_a_build_compiles_no_expression_and_a_read_compiles_exactly_one(monkeypatch):
    compiled = []
    original = PolarsCompiler.expression

    def counting(self, expr, context, **kwargs):
        compiled.append(context)
        return original(self, expr, context, **kwargs)

    monkeypatch.setattr(PolarsCompiler, 'expression', counting)
    with lps.build(SPEC, sources()) as model:
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == [], 'a build lowers no named expression — fifty declared and none read must cost none'
        assert len(compiled) == 2 * len(SPEC['constraints']) + 1, (
            'a model declaring expressions compiles exactly what one without them compiles: '
            'each constraint side, and the objective'
        )
        outcome = model.solve()
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == [], 'a solve lowers none either — the readers it hands out are thunks'
        outcome.expression('spend')
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == ["named expression 'spend'"], 'reading one expression compiles that one expression'


def test_a_closed_result_refuses_an_expression_read():
    with lps.build(SPEC, sources()) as model:
        outcome = model.solve()
    outcome.close()
    with pytest.raises(LpspecError, match='closed'):
        outcome.expression('spend')
