"""The data-time guard on what a model assumes of its data.

Everything decidable without data is decided at load, and an ``assumptions:``
entry is what is left over: a predicate only the numbers can answer. The
language states each one — the predicate, the coordinates it is checked at,
and the sentence `assumption_message` refuses in — and
every condition a ``piecewise:`` method puts on its breakpoints arrives the
same way. What is decided here is whether the data holds it, and where not.

The mask walk answers, which is the relational engine's: a second reading of a
predicate at the door would drift from the one the rows are built with. Called
from [`tidy_sources`][specsolve.sources.tidy_sources], so both lanes pass through it by
entering the one door.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mathspec.program import assumption_message

from specsolve.errors import DataError
from specsolve.relational.engines.polars.attaching import attach
from specsolve.relational.engines.polars.predicates import masked
from specsolve.relational.engines.polars.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl
    from mathspec.program import Assumption, Program


def validate_assumptions(program: Program, sources: Mapping[str, pl.LazyFrame]) -> None:
    """Refuse data that does not hold what the model assumes of it.

    A model stating nothing pays nothing: the frames are shaped for the walk
    only where there is an assumption to walk. Nothing is assumed of a
    parameter the door left unfilled — a ``piecewise:`` mask whose own source
    cannot be read — which attaching refuses on its own terms.

    Args:
        program: The lowered spec — every assumption by the name a refusal
            quotes.
        sources: What [`tidy_sources`][specsolve.sources.tidy_sources] holds once every
            parameter, relation and index is read.

    Raises:
        DataError: An assumption the data does not hold, in the language's own
            words, with one coordinate it fails at.
    """
    if not program.assumptions or any(name not in sources for name in program.parameters):
        return
    scope = Scope(program, attach(program, sources), {})
    for name, assumption in program.assumptions.items():
        failing = _a_coordinate_it_fails_at(scope, assumption)
        if failing is not None:
            raise DataError(f'{assumption_message(name, assumption)}{failing}')


def _a_coordinate_it_fails_at(scope: Scope, assumption: Assumption) -> str | None:
    """One coordinate the assumption does not hold at, or ``None`` where it holds everywhere.

    The search is the assumption negated — the coordinates its ``where``
    admits and its predicate does not — so one filter over one product answers
    it however many conditions the predicate joins. A predicate with no value
    at a coordinate does not hold there, a mask reading a missing row as
    false, so the negation admits that coordinate and the refusal names it.

    The empty string is the answer for an assumption over no dims: it fails,
    and there is no coordinate to name.
    """
    failing = ~assumption.predicate
    if assumption.where is not None:
        failing = failing & assumption.where
    dims = scope.in_declaration_order(failing.dims)
    offending = masked(scope, dims, failing).head(1).collect()
    if not offending.height:
        return None
    row = offending.row(0, named=True)
    at = ', '.join(f'{d}={row[d]!r}' for d in dims)
    return f'\n  Not so at {at}' if at else ''
