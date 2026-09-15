"""An ``assumptions:`` block, checked at the door on both lanes.

The language states what a file assumes of its data and decides nothing about
the numbers (math-spec's declarations page, *assumptions*); this package binds
the data, so it is the one that refuses. Both lanes read sources through
`tidy_sources`, so one check serves both and one sentence comes back from
either — the language's own, the failing coordinates appended.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DataError
from tests.conftest import DISPATCH_SPEC, override
from tests.differential import both_lanes_refuse, differential
from tests.oracle import pd  # skips the module without the [linopy] extra

GENERATORS = ['wind', 'solar', 'gas']


def _spec(holds: str, where: str | None = None, **patch: object) -> dict:
    assumption: dict[str, object] = {'holds': holds}
    if where is not None:
        assumption['where'] = where
    return override(
        DISPATCH_SPEC, **{'parameters.p_min': {'dims': ['generator']}, 'assumptions.floor': assumption, **patch}
    )


def _sources(p_min: dict[str, float]) -> dict:
    return {
        'generator': GENERATORS,
        'snapshot': pd.RangeIndex(2, name='snapshot'),
        'p_max': pd.Series({'wind': 100.0, 'solar': 60.0, 'gas': 200.0}),
        'p_min': pd.Series(p_min),
        'cost': pd.Series({'wind': 1.0, 'solar': 2.0, 'gas': 50.0}),
        'load': pd.Series([80.0, 90.0], index=pd.RangeIndex(2, name='snapshot')),
    }


def test_data_that_holds_the_assumption_builds_on_both_lanes():
    with differential(_spec('p_min <= p_max'), _sources({'wind': 0.0, 'solar': 0.0, 'gas': 50.0})):
        pass


def test_data_that_fails_it_is_refused_in_the_languages_words_with_the_coordinates():
    """`p_min` exceeds `p_max` at two generators; the sentence names the assumption, what it reads, and where."""
    sentence = both_lanes_refuse(
        _spec('p_min <= p_max'),
        _sources({'wind': 150.0, 'solar': 0.0, 'gas': 250.0}),
        match="assumption 'floor' does not hold for the data bound to 'p_max', 'p_min'",
    )
    assert sentence.endswith('at 2 coordinates: (generator=wind), (generator=gas)'), (
        "the failing coordinates follow in the dimension's own label order, the count first"
    )


def test_a_missing_row_reads_as_false_and_fails_the_assumption():
    """`p_min` has no row for solar: a comparison over a missing value is false, as in any mask, so the assumption fails there."""
    sentence = both_lanes_refuse(
        _spec('p_min <= p_max'), _sources({'wind': 0.0, 'gas': 50.0}), match="assumption 'floor' does not hold"
    )
    assert sentence.endswith('at 1 coordinates: (generator=solar)')


def test_where_narrows_the_coordinates_the_assumption_is_held_at():
    """The same failing data, but `where` admits only the generators that cost something to run — wind is not asked."""
    with differential(_spec('p_min <= p_max', where='cost > 10'), _sources({'wind': 150.0, 'solar': 0.0, 'gas': 50.0})):
        pass


def test_an_assumption_comparing_arithmetic_is_read_by_the_same_rules(tmp_path):
    """`holds` is a where string, so it takes what a `where` takes — here an expression against a parameter over another dim."""
    spec = _spec('load <= sum(p_max, over=generator)', **{'parameters.load': {'dims': ['snapshot']}})
    with differential(spec, _sources({'wind': 0.0, 'solar': 0.0, 'gas': 50.0})):
        pass
    sources = _sources({'wind': 0.0, 'solar': 0.0, 'gas': 50.0})
    sources['load'] = pd.Series([80.0, 900.0], index=pd.RangeIndex(2, name='snapshot'))
    sentence = both_lanes_refuse(
        spec, sources, match="assumption 'floor' does not hold for the data bound to 'load', 'p_max'"
    )
    assert sentence.endswith('at 1 coordinates: (snapshot=1)')


def test_an_assumption_over_no_dimension_names_no_coordinate():
    spec = override(DISPATCH_SPEC, **{'parameters.budget': {'dims': []}, 'assumptions.funded': {'holds': 'budget > 0'}})
    sources = {**_sources({'wind': 0.0, 'solar': 0.0, 'gas': 50.0}), 'budget': 0.0}
    del sources['p_min']
    sentence = both_lanes_refuse(
        spec, sources, match="assumption 'funded' does not hold for the data bound to 'budget'"
    )
    assert sentence == "assumption 'funded' does not hold for the data bound to 'budget'", (
        'nothing to point at, so nothing appended'
    )


def test_more_failing_coordinates_than_are_spelled_out_are_counted():
    """Six of the eight coordinates fail; five are spelled out and the sixth is counted."""
    spec = override(
        DISPATCH_SPEC,
        **{'parameters.p_min': {'dims': ['snapshot', 'generator']}, 'assumptions.floor': {'holds': 'p_min <= p_max'}},
    )
    p_min = pl.DataFrame(
        {
            'snapshot': [0, 0, 0, 1, 1, 1],
            'generator': ['wind', 'solar', 'gas', 'wind', 'solar', 'gas'],
            'value': [500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        }
    )
    sources = {**_sources({}), 'snapshot': pd.RangeIndex(2, name='snapshot'), 'p_min': p_min}
    with pytest.raises(DataError, match=r'at 6 coordinates: .*\(snapshot=1, generator=solar\), and 1 more$'):
        lps.build(spec, sources).close()
