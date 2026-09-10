"""The two expression readers on a solved model: `expression(name)` (#562), and `evaluate`.

The relational lane only — the differential half, both lanes agreeing on the
same values, lives in ``test_linopy_lane.py`` with the rest of the oracle
comparisons. What is pinned here for `expression`: the value is the one the
primal implies, an expression no constraint references still reads, the
frame's dims are the ones it survives over, laziness (a build compiles no
expression; a read compiles that one), and the unknown-name refusal. For
`evaluate`, below: a name and the body it stands for read one value, both
written forms are taken, and what it refuses — a name the model does not
declare, and a model that arrived already lowered.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec import expressions
from lpspec.errors import DataError, LanguageError, LpspecError, SchemaError
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


def test_a_variable_declared_zero_is_zero_under_a_nonlinear_read():
    """`absence: zero` says the quantity *is* zero where the variable has no row.

    Affine arithmetic cannot tell no row from a zero, and neither can a sum a
    constant reaches — `1 + p` lands the 1 on every coordinate. `0.5 ** p`
    can: the absent generator contributes `0.5 ** 0`, which is 1, rather
    than nothing, and the present one a value near zero.
    """
    spec = override(
        SPEC,
        **{
            'variables.p.where': 'p_max > 0',
            'variables.p.absence': 'zero',
            'expressions.grown': 'sum(0.5 ** p, over=generator)',
        },
    )
    data = sources() | {'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [200.0, 0.0]})}
    result = lps.solve(spec, data)
    p = result.primal('p').sort('snapshot')
    assert p['generator'].unique().to_list() == ['g1'], 'g2 is masked out, so only g1 has a primal'
    want = [0.5**v + 1.0 for v in p['value']]
    assert result.expression('grown')['value'].to_list() == pytest.approx(want), (
        'the absent generator is a zero under absence: zero, so 0.5 ** 0 counts as 1 in the sum'
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


# ---------------------------------------------------------------------------
# evaluate: an expression the file never named
# ---------------------------------------------------------------------------


def test_a_declared_name_and_the_body_it_stands_for_read_one_value(result):
    """`evaluate` takes a name because the language takes one: it substitutes a declared name where it stands, so the two spellings are one expression."""
    declared = result.evaluate('total_gen')
    written = result.evaluate('sum(p, over=generator)')
    assert declared.equals(result.expression('total_gen')), 'a declared name is served by the reader that holds it'
    assert written.equals(declared), "the body reads what the name reads, the name being the body's own spelling"


def test_an_expression_the_file_never_declared_reads_what_the_primal_implies(result):
    external = (
        result.primal('p')
        .join(sources()['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost') * 2).sum().alias('value'))
        .sort('snapshot')
    )
    frame = result.evaluate('sum(p * cost * 2, over=generator)')
    assert frame.columns == ['snapshot', 'value'], 'an evaluated frame is (dims…, value), like a declared one'
    assert frame.sort('snapshot').equals(external), (
        'an expression nothing declared is evaluated at the same primal a declared one is'
    )


def test_a_mapping_is_the_other_form_the_language_writes_an_expression_in(result):
    """A bare string and a mapping are `ExpressionBlock`'s two written forms, so `evaluate` takes both."""
    assert result.evaluate({'expression': 'sum(p, over=generator)'}).equals(result.expression('total_gen'))


def test_a_mapping_carries_the_cases_a_string_cannot_say():
    """`cases:` is why the mapping form is not sugar: a region-varying quantity has no spelling as one string."""
    spec = {
        **SPEC,
        'parameters': {**SPEC['parameters'], 'peak': {'dims': ['snapshot'], 'dtype': 'bool'}},
    }
    data = sources() | {'peak': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [False, True, False]})}
    frame = lps.solve(spec, data).evaluate(
        {
            'foreach': ['snapshot'],
            'cases': {'busy': {'when': 'peak', 'expression': 'total_gen'}},
            'otherwise': 0,
        }
    )
    got = dict(zip(frame['snapshot'], frame['value'], strict=True))
    assert got == pytest.approx({0: 0.0, 1: 120.0, 2: 0.0}), (
        'the case holds only where peak does, and otherwise supplies the rest'
    )


def test_an_expression_may_read_a_dual_the_file_never_priced(result):
    assert result.evaluate('dual(balance) * 2')['value'].to_list() == pytest.approx(
        [v * 2 for v in result.dual('balance')['value']]
    ), 'the math reads nothing evaluated, so a dual stands in it exactly as it stands in a declared entry'


def test_a_name_the_model_does_not_declare_is_refused_rather_than_read_as_a_gap(result):
    """A read reaches the solved model's own declarations and no further, so a new parameter is named as missing rather than read as an absence — supplying one is a build."""
    with pytest.raises(LanguageError, match='co2_rate'):
        result.evaluate('sum(p * co2_rate, over=generator)')


def test_the_splice_steps_over_a_declaration_of_its_own_name():
    """Names share one flat namespace, so the spliced entry must not land on a declared one and shadow what the expression reads.

    Read through an expression that *references* the collision rather than
    naming it: naming it is served by the declared reader, and never splices.
    """
    spec = {**SPEC, 'expressions': {**SPEC['expressions'], '_evaluated': 'sum(p, over=generator) * 3'}}
    assert lps.solve(spec, sources()).evaluate('_evaluated * 2')['value'].to_list() == pytest.approx(
        [300.0, 720.0, 480.0]
    ), "the splice lands beside the declaration, so the expression still reads the model's own entry"


def test_a_model_built_from_a_lowered_program_says_why_it_cannot_evaluate():
    """`check` hands back a Program, and a Program is what a model lowered to — reading an expression needs the model as written."""
    result = lps.solve(lps.check(SPEC), sources())
    with pytest.raises(LpspecError, match='lowered Program'):
        result.evaluate('sum(p, over=generator)')
    assert result.evaluate('total_gen').equals(result.expression('total_gen')), (
        'a declared name is readable either way — it needs no lowering, being already lowered'
    )


def test_a_closed_result_refuses_to_evaluate():
    result = lps.solve(SPEC, sources())
    result.close()
    with pytest.raises(LpspecError, match='was closed'):
        result.evaluate('sum(p, over=generator)')


def test_an_evaluated_expression_names_nothing_and_so_is_not_a_kind(result, tmp_path):
    """It is not written, spilled or enumerated: a quantity worth keeping across runs is worth declaring."""
    written = {p.stem for p in (result.to_parquet(tmp_path) / 'expression').glob('*.parquet')}
    assert written == set(SPEC['expressions']), 'to_parquet writes the declared names, and evaluate adds none'


# ---------------------------------------------------------------------------
# extend: this solve, asked more questions
# ---------------------------------------------------------------------------


REPORT = {'expressions': {'burn': 'sum(p * cost, over=generator)', 'shadow': 'dual(balance)'}}


@pytest.fixture(scope='module')
def report(result):
    return result.extend(REPORT)


def test_extending_answers_with_this_solve_rather_than_a_second_one(result, report):
    assert report.objective == result.objective, 'the solve is not re-run, so its objective is the one it reached'
    assert report.status == result.status
    assert report.primal('p').equals(result.primal('p')), 'the primal frames are carried over, not recomputed'


def test_an_added_quantity_reads_what_the_primal_implies(result, report):
    external = (
        result.primal('p')
        .join(sources()['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost')).sum().alias('value'))
        .sort('snapshot')
    )
    assert report.expression('burn').sort('snapshot').equals(external)
    assert report.expression('shadow').equals(result.dual('balance')), (
        'an added entry may read a dual like a declared one'
    )


def test_an_added_quantity_sits_beside_the_models_own(report):
    assert report.expression('total_gen').height == 3, "the model's declared entries are still readable"
    assert set(report.to_dataset(kind='expression').data_vars) == set(SPEC['expressions']) | {'burn', 'shadow'}, (
        'every bridge reads the added names, the added ones being named'
    )


def test_an_added_quantity_is_written_like_a_declared_one(report, tmp_path):
    """Named, so it is a *kind*: unlike `evaluate`, this is spilled and written."""
    written = {p.stem for p in (report.to_parquet(tmp_path) / 'expression').glob('*.parquet')}
    assert written == set(SPEC['expressions']) | {'burn', 'shadow'}, (
        'to_parquet writes the added names beside the declared'
    )


def test_extending_leaves_the_result_it_extended_alone(result, report):
    assert 'burn' not in result._names('expression'), 'extend adds to a new result and mutates nothing'
    with pytest.raises(KeyError, match='burn'):
        result.expression('burn')


def test_a_later_block_reads_what_an_earlier_one_added(report):
    """`expression('burn')` works on this result, so an expression written against it has to as well."""
    twice = report.extend({'expressions': {'double_burn': 'burn * 2'}})
    assert twice.expression('double_burn')['value'].to_list() == pytest.approx(
        [v * 2 for v in report.expression('burn')['value']]
    ), 'an added name resolves in a later block the way a declared one does'


@pytest.mark.parametrize(
    ('added', 'match'),
    [
        pytest.param({'expressions': {'total_gen': '1'}}, 'flat namespace', id='a-declared-expression'),
        pytest.param({'expressions': {'p': '1'}}, 'flat namespace', id='a-variable'),
        pytest.param({'expressions': {'cost': '1'}}, 'flat namespace', id='a-parameter'),
        pytest.param({'expressions': {'snapshot': '1'}}, 'flat namespace', id='a-dimension'),
    ],
)
def test_a_name_the_model_already_declares_is_refused(report, added, match):
    with pytest.raises(LanguageError, match=match):
        report.extend(added)


@pytest.mark.parametrize(
    ('added', 'match'),
    [
        pytest.param({'parameters': {'x': {'dims': []}}}, r"carries \['parameters'\]", id='a-parameter-section'),
        pytest.param(
            {'variables': {'q': {'foreach': []}}, 'expressions': {'e': '1'}},
            r"carries \['variables'\]",
            id='a-variable-section-beside-a-good-one',
        ),
        pytest.param({'expressions': {}}, 'reads nothing', id='an-empty-block'),
        pytest.param({}, 'reads nothing', id='no-block-at-all'),
    ],
)
def test_a_fragment_that_would_build_rather_than_read_is_refused(report, added, match):
    with pytest.raises(SchemaError, match=match):
        report.extend(added)


def test_a_name_already_added_is_refused_rather_than_replaced(report):
    with pytest.raises(LpspecError, match='already readable'):
        report.extend({'expressions': {'burn': 'sum(p)'}})


def test_a_block_is_lowered_once_however_many_entries_it_has(result, monkeypatch):
    """The cost the docstring claims: handing in a block beats handing in its entries one at a time."""
    lowerings = []
    real = expressions.to_program
    monkeypatch.setattr(expressions, 'to_program', lambda spec: lowerings.append(1) or real(spec))
    result.extend({'expressions': {f'q{i}': f'sum(p) * {i}' for i in range(8)}})
    assert len(lowerings) == 1, 'eight entries, one lowering of the model'


def test_extending_a_model_built_from_a_lowered_program_says_why_it_cannot():
    with pytest.raises(LpspecError, match='lowered Program'):
        lps.solve(lps.check(SPEC), sources()).extend(REPORT)


def test_a_closed_result_refuses_to_extend():
    result = lps.solve(SPEC, sources())
    result.close()
    with pytest.raises(LpspecError, match='was closed'):
        result.extend(REPORT)
