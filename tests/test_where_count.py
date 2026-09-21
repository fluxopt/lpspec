"""A ``where`` that counts what a predicate admits, and one that reads a predicate at a neighbour.

Two calls read a predicate where every other operator reads arithmetic
(math-spec's expressions page, *a predicate is an operand*):
``count(<predicate>, over=<dim>)`` answers a number, and
``shift(<predicate>, along=, offset=)`` answers a predicate. Both lanes build
the same model from either, which is what these assert — the coordinates built,
not just the count of them, because a predicate that inverted its sense would
keep the complement.
"""

from __future__ import annotations

import polars as pl
import pytest

from tests.conftest import DISPATCH_SPEC, override
from tests.differential import differential
from tests.oracle import pd  # skips the module without the [linopy] extra


def _spec(where: str) -> dict:
    return override(
        DISPATCH_SPEC,
        **{'parameters.offer': {'dims': ['snapshot', 'generator']}, 'variables.p.where': where},
    )


def _sources(offer: pl.DataFrame) -> dict:
    return {
        'p_max': pd.Series({'wind': 100.0, 'gas': 200.0}),
        'cost': pd.Series({'wind': 0.0, 'gas': 50.0}),
        'load': pd.Series([80.0] * 4, index=pd.RangeIndex(4, name='snapshot')),
        'snapshot': pd.RangeIndex(4, name='snapshot'),
        'generator': ['wind', 'gas'],
        'offer': offer,
    }


def _offer(**per_generator: list[float | None]) -> pl.DataFrame:
    """One tidy frame from ``generator=[value per snapshot]``, a ``None`` standing for no row at all."""
    rows = [
        (t, name, value)
        for name, values in per_generator.items()
        for t, value in enumerate(values)
        if value is not None
    ]
    return pl.DataFrame(
        {'snapshot': [t for t, _, _ in rows], 'generator': [g for _, g, _ in rows], 'value': [v for _, _, v in rows]},
        schema={'snapshot': pl.Int64, 'generator': pl.String, 'value': pl.Float64},
    )


def _built(where: str, offer: pl.DataFrame) -> list[tuple[int, str]]:
    """``p``'s coordinates under *where*, both lanes agreeing on the model first."""
    with differential(_spec(where), _sources(offer)) as agreed:
        rows = agreed.result.primal('p').to_dicts()
    return sorted((row['snapshot'], row['generator']) for row in rows)


def test_a_count_is_one_number_per_coordinate_the_predicate_keeps():
    """``over`` is reduced away, so the count speaks about each generator without the file saying "each"."""
    offer = _offer(wind=[10.0, 10.0, 10.0, 10.0], gas=[10.0, 0.0, 0.0, 0.0])
    built = _built('count(offer > 0, over=snapshot) >= 3', offer)

    assert built == [(0, 'wind'), (1, 'wind'), (2, 'wind'), (3, 'wind')], (
        'wind offers at three snapshots or more and gas at one, so the mask keeps wind at every snapshot'
    )


def test_a_coordinate_the_predicate_admits_nowhere_counts_zero():
    """A generator with no row at all is a count of zero, which is the answer rather than a gap that drops the row."""
    offer = _offer(wind=[None, None, None, None], gas=[None, None, None, None])
    built = _built('count(offer > 0, over=snapshot) == 0', offer)

    assert built == [(t, g) for t in range(4) for g in ('gas', 'wind')], (
        'neither generator offers anywhere, so both count zero and the mask keeps every coordinate'
    )


def test_a_shift_over_a_predicate_is_false_where_the_translation_vacates():
    """``offset=1`` reads one coordinate back, and the first coordinate reads false rather than wrapping or filling."""
    offer = _offer(wind=[150.0, 50.0, 50.0, 50.0], gas=[10.0, 10.0, 10.0, 10.0])
    built = _built('NOT shift(offer > 100, along=snapshot, offset=1)', offer)

    assert built == [(0, 'gas'), (0, 'wind'), (1, 'gas'), (2, 'gas'), (2, 'wind'), (3, 'gas'), (3, 'wind')], (
        'only wind at snapshot 1 is dropped: the predicate holds at snapshot 0, and snapshot 0 itself '
        'reads false because the translation vacates there'
    )


def test_a_count_over_a_translated_predicate_counts_the_runs_a_mask_starts():
    """The shape #582 needs a curve to state: a coordinate the predicate admits whose neighbour it does not."""
    offer = _offer(wind=[10.0, 10.0, 10.0, 10.0], gas=[0.0, 10.0, 0.0, 10.0])
    built = _built('count(offer > 0 AND NOT shift(offer > 0, along=snapshot, offset=1), over=snapshot) == 1', offer)

    assert built == [(0, 'wind'), (1, 'wind'), (2, 'wind'), (3, 'wind')], (
        "wind's offers are one consecutive run and gas's are two, so only wind admits exactly one run start"
    )


def test_a_count_reads_a_dimension_the_frame_does_not_span():
    """The count reduces its own dim away, so a declaration need not span it — it is the frame's other dims that must match."""
    offer = _offer(wind=[10.0, 10.0, 10.0, 10.0], gas=[10.0, 10.0, 10.0, 10.0])
    spec = override(
        DISPATCH_SPEC,
        **{
            'parameters.offer': {'dims': ['snapshot', 'generator']},
            'constraints.balance.where': 'count(offer > 0, over=generator) >= 2',
        },
    )
    with differential(spec, _sources(offer)) as agreed:
        rows = agreed.result.dual('balance').to_dicts()

    assert sorted(row['snapshot'] for row in rows) == [0, 1, 2, 3], (
        'both generators offer at every snapshot, so the balance row stands at all four — the count '
        'over generator asks nothing of the row it masks'
    )


@pytest.mark.parametrize(
    ('where', 'kept'),
    [
        pytest.param('count(offer > 0, over=snapshot) > 2', ['wind'], id='strictly-more-than'),
        pytest.param('count(offer > 0, over=snapshot) == 1', ['gas'], id='exactly-one'),
        pytest.param('count(offer > 0, over=snapshot) != 1', ['wind'], id='not-one'),
        pytest.param('count(offer > 0, over=snapshot) <= 1', ['gas'], id='at-most-one'),
    ],
)
def test_a_count_is_compared_by_the_operator_the_file_wrote(where, kept):
    """Every relation a where may write, over the same two counts — four and one."""
    offer = _offer(wind=[10.0, 10.0, 10.0, 10.0], gas=[10.0, 0.0, 0.0, 0.0])
    built = _built(where, offer)

    assert sorted({g for _, g in built}) == kept, f'where: {where!r} kept the wrong generators'
    assert len(built) == 4, 'whichever generator the count keeps, it is kept at every snapshot'
