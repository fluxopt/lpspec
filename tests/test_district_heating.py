"""The district-heating plant must keep splitting the day across its whole fleet.

`examples/district_heating.py` is the claim that lpspec expresses a committed,
multi-carrier heating plant — a boiler, a CHP unit, a heat pump and an electric
boiler on shared curves, each switched on or off with a minimum load and a
minimum time up and down — and that on a plausible winter day the merit order
puts every one of them to work. The script asserts its own heat balance, fleet
use and recosted objective internally, so `run_example` running it is already a
check; these are the two claims stated apart from it.

No committed golden, unlike the other example tests: this is a mixed-integer
program with alternative optima, so the *objective* is unique but the *schedule*
that achieves it need not be, and a byte-for-byte transcript would drift with
the solver rather than with the model. The invariants below are what the example
actually promises, and each is stable across those optima.
"""

from __future__ import annotations

import pytest

import lpspec as lps
from tests.conftest import EXAMPLES_DIR, run_example

EXAMPLE = EXAMPLES_DIR / 'district_heating.py'


@pytest.fixture(scope='module')
def output() -> str:
    return run_example(EXAMPLE, 'district_heating_example')


def test_the_whole_fleet_is_committed(output: str) -> None:
    """Every unit runs in some period — the split is the model's, not the script's.

    Stated apart from the objective because it is the *point*: a fleet that
    left the boiler or the CHP unit idle all day would still solve, at a cost
    the number below would happily report.
    """
    line = next(line for line in output.splitlines() if line.startswith('periods committed'))
    counts = [int(word) for word in line.split() if word.isdigit()]
    assert len(counts) == 4, f'four units must report a committed count: {line!r}'
    assert all(count > 0 for count in counts), f'every unit must run in some period: {line!r}'


def test_the_plant_reaches_its_optimum() -> None:
    """The cost the page cites, solved from the example's own sources."""
    import examples.district_heating as dh

    result = lps.solve(dh.MODEL, dh.sources())
    assert result.termination_condition == 'optimal', result.termination_condition
    assert result.objective == pytest.approx(14235.04, abs=0.5), result.objective
