"""An ``assumptions:`` entry, checked at the door: what a file assumes of its data, refused where the data fails it.

Called from :func:`~lpspec.sources.tidy_sources` once every parameter is read,
so both lanes pass through it and refuse in the language's own words
(:func:`math_spec.program.assumption_message`), the failing coordinates
appended. Only the file's own entries are checked here: a
:data:`~math_spec.program.Check` a ``piecewise:`` method implies is one number
sequence per curve rather than one predicate per coordinate, and
:mod:`lpspec.curves` holds it.

The predicate is read by the relational lane's mask compiler — the one reader
of a mask this package has that needs no model built first, and the same
reader every ``where:`` goes through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_spec.program import Holds, assumption_message

from lpspec.errors import DataError
from lpspec.relational.engines.polars.attaching import attach
from lpspec.relational.engines.polars.predicates import compile_predicate, falsy_if_null, masked
from lpspec.relational.engines.polars.scope import UNIT, Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl
    from math_spec.program import Program

#: How many failing coordinates the refusal spells out. The count is always given.
SHOWN = 5


def validate_assumptions(program: Program, sources: Mapping[str, pl.LazyFrame]) -> None:
    """Refuse *sources* at the first ``assumptions:`` entry whose predicate is false somewhere its ``where`` admits.

    The failing coordinates are the admitted product less the rows the
    predicate keeps, rather than the rows it drops: a mask reading a parameter
    certain of it inner-joins, so a coordinate with no row is gone from the
    carrier before the predicate is read, and it is exactly the coordinate a
    missing row has to count as failing.

    Args:
        program: The lowered spec.
        sources: What :func:`~lpspec.sources.tidy_sources` holds once every
            parameter, dimension and lookup is read.

    Raises:
        DataError: An assumption the data fails, naming it, the parameters it
            reads, and the coordinates where it fails — a missing row reading
            as false, as in any mask.
    """
    written = {name: a for name, a in program.assumptions.items() if isinstance(a, Holds)}
    if not written:
        return
    scope = Scope(program, attach(program, sources), {})
    for name, assumption in written.items():
        read = assumption.predicate.dims | (assumption.where.dims if assumption.where is not None else frozenset())
        dims = tuple(d for d in program.dimensions if d in read)
        admitted = masked(scope, dims, assumption.where)
        carrier, holds = compile_predicate(scope, admitted, assumption.predicate, dims)
        keys = list(dims) or [UNIT]
        surviving = carrier.filter(falsy_if_null(holds)).select(*keys)
        failing = admitted.join(surviving, on=keys, how='anti').collect()
        if failing.height:
            raise DataError(_failed_message(name, assumption, failing, dims))


def _failed_message(name: str, assumption: Holds, failing: pl.DataFrame, dims: tuple[str, ...]) -> str:
    """The language's sentence, then where: the count and the first :data:`SHOWN` coordinates in the dimensions' own order."""
    sentence = assumption_message(name, assumption)
    if not dims:
        return sentence
    rows = failing.select(*dims).sort(*dims).head(SHOWN).iter_rows()
    spelled = ', '.join('(' + ', '.join(f'{d}={v}' for d, v in zip(dims, row, strict=True)) + ')' for row in rows)
    more = f', and {failing.height - SHOWN} more' if failing.height > SHOWN else ''
    return f'{sentence} at {failing.height} coordinates: {spelled}{more}'
