"""The district-heating plant must keep splitting the day across its whole fleet.

`examples/district_heating.py` is the claim that lpspec expresses a
multi-carrier heating plant — a boiler, a CHP unit, a heat pump and an electric
boiler on shared curves — where only the two burners are committed on or off,
the electric units dispatching freely, and that on a plausible winter day the
merit order puts every one of them to work. The script asserts its own heat
balance, fleet use and recosted objective internally, so `run_example` running
it is already a check; these are the claims stated apart from it.

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


def test_only_the_burners_carry_a_status() -> None:
    """Two committable units have a status; the electric units are masked out of it.

    Stated apart from the objective because it is the *point* of `where:
    committable`: a status for the heat pump or electric boiler would be a
    binary the model never needed, and the count of status rows is how the mask
    shows in the built program.
    """
    import examples.district_heating as dh

    status = lps.solve(dh.MODEL, dh.sources()).primal('status')
    assert sorted(status['unit'].unique().to_list()) == ['boiler', 'chp'], (
        'only the boiler and CHP unit are committable, so only they carry a status'
    )


def test_the_whole_fleet_is_used(output: str) -> None:
    """Every unit works in some period — the split is the model's, not the script's.

    A fleet that left the boiler or the CHP unit idle all day, or never called
    on the heat pump, would still solve at a cost the number below would happily
    report; the example asserts the split internally and this checks it ran.
    """
    assert 'the whole fleet is used' in output, output


def test_the_plant_reaches_its_optimum() -> None:
    """The cost the page cites, solved from the example's own sources."""
    import examples.district_heating as dh

    result = lps.solve(dh.MODEL, dh.sources())
    assert result.termination_condition == 'optimal', result.termination_condition
    assert result.objective == pytest.approx(14228.04, abs=0.5), result.objective
