"""The differential harness: one model, two implementations, one answer.

The differential test is this project's central claim — the same YAML must
mean the same thing here and in linopy, which reads math-spec natively
(docs/about/architecture.md, hard rule 3). Twelve tests made that claim by hand, in
seven files, each rebuilding the same fifteen lines: build the oracle, solve,
take the objective, re-parse the schema, lower it, attach sources, execute,
compare.

What the repetition cost was not correctness but *evenness*. Every copy
compared objectives and checked the status, but only five of the twelve also
wrote the LP file and re-solved it, and nothing recorded why the other seven
skipped that third opinion — so the strength of the claim varied with which
file you happened to be reading. Here it is one ``lp=True``, and a test that
does not ask for it is visibly choosing not to.

Importing this module is the ``[linopy]`` guard: it reaches the oracle
through ``tests.oracle``, so a bare install skips every module that uses the
harness at collection time, with no filename list to maintain.

Usage — the engine stays open for the length of the ``with`` block, so
per-variable primal checks live inside it::

    with differential(NONCONVEX_YAML, sources, lp=True) as run:
        assert run.result.to_pandas('op_cost') ...
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from math_spec import to_program

import lpspec as lps
from lpspec.errors import DataError
from lpspec.relational.engines.polars.engine import PolarsEngine
from lpspec.sources import tidy_sources
from tests.conftest import raw_of, schema_of, solve_written_file
from tests.oracle import linopy, spec_oracle

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lpspec.relational.engines.polars.engine import Result

#: Both hand the same numbers to the same solver, so they must agree to
#: solver precision, not to a fudge factor. One tolerance, one place.
RTOL = 1e-9


class NoFiniteAnswerError(AssertionError):
    """The fixture admits no finite optimum, so neither is on trial.

    An ``AssertionError`` because that is what it was and what every caller
    that does not catch it still wants: a failure naming the fixture. A class
    of its own because a caller generating its models — ``test_expression_sweep``
    — must tell "this data has no answer" from "the two disagree", and was
    doing it by matching the message text.
    """


@dataclass
class Agreement:
    """What the two produced, for tests that assert past the objective."""

    oracle: float
    """The oracle's objective — the number both implementations had to reach."""

    model: linopy.Model
    """The oracle's model, for structural assertions (labels, masks, solution)."""

    result: Result
    """The relational solution; live until the ``with`` block exits."""

    engine: PolarsEngine
    lp: Path | None = None
    """The written LP file, when ``lp=True`` — already checked to agree."""


@contextmanager
def differential(
    spec: str | Path | dict[str, Any],
    sources: Mapping[str, Any],
    *,
    lp: bool = False,
) -> Iterator[Agreement]:
    """Build ``spec`` on both with the same inputs; assert they agree.

    ``spec`` is a ``Path`` to a file, the YAML text itself, or a raw dict —
    the oracle keeps the spec text on the model, so text and dicts are written
    to a temporary file here rather than in every caller.

    **Duals are not compared here, and cannot be.** An LP with alternative
    optima has many optimal dual solutions, and the two hand HiGHS the
    same rows in a different order, so it lands on a different basis:
    ``genx_piecewise_fuel`` agrees on the objective to nine decimals, differs in
    2 of 72 entries of one primal, and in 12 of 48 entries of one dual. A
    lane-to-lane dual assertion would therefore be false rather than merely
    strict. What *is* checkable is a dual against a recording made from an
    instance designed to have a unique one, which is ``test_ports`` and
    ``test_corpus_parity``'s job rather than this harness's.

    Set ``lp=True`` to also write and re-solve the LP file, the third opinion.
    HiGHS reads that file, so a model carrying ``sos:`` must not ask for it:
    HiGHS has no SOS concept and its parser refuses the section outright, which
    is the same fact ``reformulate_sos='auto'`` handles on the oracle's side —
    a no-op for every model that declares no set, and what lets the oracle solve
    one that does.
    """
    schema = schema_of(spec)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        path = spec if isinstance(spec, Path) else _write(work / 'model.yaml', spec)

        m = spec_oracle.build(path, dict(sources))
        m.solve(solver_name='highs', output_flag=False, reformulate_sos='auto')
        oracle = float(m.objective.value)
        if not np.isfinite(oracle):
            raise NoFiniteAnswerError('the oracle is infeasible or unbounded — fix the data, not the tolerance')

        program = to_program(schema)
        with PolarsEngine() as engine:
            engine.build(program, tidy_sources(program, dict(sources)))
            result = engine.solve()
            assert result.is_ok, f'the relational lane reached no solution: {result.status}'
            assert result.objective == pytest.approx(oracle, rel=RTOL), (
                f'the two disagree on the objective — this engine {result.objective}, linopy {oracle}'
            )
            _same_shape(engine.diagnostics(), m)

            lp_path = None
            if lp:
                lp_path = work / 'model.lp'
                engine.write(lp_path)
                assert solve_written_file(lp_path) == pytest.approx(oracle, rel=RTOL), (
                    f'the written {lp_path.name} re-solves to a different objective than both reached'
                )

            yield Agreement(oracle=oracle, model=m, result=result, engine=engine, lp=lp_path)


def both_refuse(spec: str | Path | dict[str, Any], sources: Mapping[str, Any], match: str) -> str:
    """This engine and linopy both refuse *sources*; ours is returned for the cases that pin more of it.

    Not a ``pytest.raises`` around :func:`differential`: the oracle builds
    first there and satisfies the raises on its own, so a reader that let the
    data through would go unnoticed — which is the divergence a data check is
    most likely to have. The two are built apart.

    **The sentences are no longer compared.** They were while both read
    the caller's tables through one ``tidy_sources``: one defect, one sentence,
    and a wording that had drifted was a reader that had drifted. linopy has
    its own reader now — ``linopy.spec.attach``, written against the same
    binding rules in another package — so the checkable claim is that both
    refuse the defect and that neither mistakes it for a language error. Ours
    is matched against *match*; theirs need only be a ``SpecDataError``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = spec if isinstance(spec, Path) else _write(Path(tmp) / 'model.yaml', spec)
        with pytest.raises(DataError, match=match) as refusal:
            lps.build(path, dict(sources)).close()
        with pytest.raises(spec_oracle.SpecDataError):
            spec_oracle.build(path, dict(sources))
    return str(refusal.value)


def _same_shape(diagnostics: Any, oracle: Any) -> None:
    """The two built the same *model*, not merely the same answer.

    An objective, a dual vector and a re-solved LP file are all invariant to a
    column that cannot move: a variable pinned to ``[0, 0]``, or a row that is
    true whatever the solver does. So a build could materialise either and
    every other assertion here would still pass — which is not hypothetical, it
    is how a first draft of ``absence: zero`` shipped an extra column per absent
    coordinate with the whole suite green.

    Counts rather than a set comparison: the two name their columns differently
    by design (labels against a ``(name, coordinate)`` index), and the claim
    worth making is that the same declarations produced the same number of
    them.
    """
    assert diagnostics.columns == oracle.nvars, (
        f'the two disagree on how many columns this model has — this engine {diagnostics.columns}, '
        f'linopy {oracle.nvars}'
    )
    assert diagnostics.rows == oracle.ncons, (
        f'the two disagree on how many rows this model has — this engine {diagnostics.rows}, linopy {oracle.ncons}'
    )


def _write(path: Path, spec: str | dict[str, Any]) -> Path:
    import yaml as pyyaml

    path.write_text(spec if isinstance(spec, str) else pyyaml.safe_dump(raw_of(spec)))
    return path
