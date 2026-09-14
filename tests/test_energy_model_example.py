"""The energy-model example must keep catching what lpspec does not, or it is not evidence.

`examples/energy_model/run.py` is the claim that a domain data model — typed
components, validated, lowered to `sources` — is a layer you write *above*
lpspec rather than a feature lpspec should grow. A claim like that is worth
what the check behind it is worth, and the check is the gap itself: the same
network that the domain layer refuses is one lpspec builds and solves without
complaint, returning a confident answer to the wrong problem.

So the substantive test solves the broken networks past the validator and
watches lpspec accept them, then watches the validator refuse them. If lpspec
ever grew a check of its own that caught one of these, the example would still
pass its golden while quietly ceasing to make its point — that is the drift
this guards.

Committed output for the rest, for the reason `test_benders_example.py` gives:
a page that shows output is promising what a reader will see, and a diff is how
that promise stays true. Regenerate with ``--update-golden``.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

import lpspec as lps
from tests.conftest import EXAMPLES_DIR, assert_golden, run_example

if TYPE_CHECKING:
    from types import ModuleType

EXAMPLE = EXAMPLES_DIR / 'energy_model' / 'run.py'
GOLDEN = EXAMPLE.with_name('run.out')

#: The broken networks lpspec solves rather than refuses. `islanded` is left
#: out of the *solve* leg only because solving it trips an unrelated polars
#: deprecation in lpspec's own attaching path, which `filterwarnings = error`
#: would surface as a failure of the wrong thing; the validator's refusal of it
#: is asserted with the other two.
SOLVED_ANYWAY = ['range', 'sign']


@pytest.fixture(scope='module')
def module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('energy_model_run', EXAMPLE)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope='module')
def output() -> str:
    return run_example(EXAMPLE, 'energy_model_example')


def test_the_validator_refuses_every_broken_network(module: ModuleType) -> None:
    """The domain layer catches all three mistakes — the promise it makes."""
    for net in module.BROKEN.values():
        with pytest.raises(module.DomainError):
            net.validate()


@pytest.mark.parametrize('label', SOLVED_ANYWAY)
def test_lpspec_solves_what_the_validator_refuses(module: ModuleType, label: str) -> None:
    """The gap the example exists to fill: lpspec builds and solves the broken network.

    Asserted apart from the golden because it *is* the point — a golden alone
    would keep passing if lpspec started rejecting these, and the example would
    then guard a gap that had closed.
    """
    net = module.BROKEN[label]
    result = lps.solve(str(module.MODEL), net.to_sources())
    assert result.status == 'ok', f'lpspec should solve {label!r} unmoved, not report {result.status!r}'


def test_the_valid_network_conserves_energy(output: str) -> None:
    """The example asserts this internally; this is the check that it ran at all."""
    assert 'meets load 270.0' in output, output


def test_the_example_matches_its_committed_output(output: str, pytestconfig: pytest.Config) -> None:
    assert_golden(
        output,
        GOLDEN,
        pytestconfig,
        drifted='the energy-model example no longer prints what the docs show:',
    )
