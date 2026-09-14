"""Evaluate a spec as arithmetic: its named expressions, read straight off the data.

A spec that declares no variables is a calculation rather than an optimisation —
every expression reads only parameters and lookups, so each has a value with no
solver and no chosen point. This attaches the data and hands back the same
deferred readers a solve does (:func:`~lpspec.relational.engines.polars.readback.deferred_readers`),
over a compiler carrying no solution: a variable never reached, every leaf a
number already.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpspec.relational.engines.polars import readback
from lpspec.relational.engines.polars.attaching import attach
from lpspec.relational.engines.polars.compiler import PolarsCompiler

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    import polars as pl
    from math_spec import program


def expression_readers(
    program: program.Program,
    sources: Mapping[str, pl.LazyFrame],
    lower: Callable[[str | Mapping[str, Any]], program.ExpressionNode] | None,
) -> tuple[dict[str, Callable[[], pl.DataFrame]], Callable[[str | Mapping[str, Any]], pl.DataFrame] | None]:
    """Attach *sources* and defer the reads an :class:`~lpspec.relational.result.Evaluation` is built from.

    Args:
        program: A lowered program with no variables — a calculation.
        sources: Tidied sources, as :func:`~lpspec.sources.tidy_sources` produces.
        lower: How an ad-hoc expression becomes a plan node in the model's
            namespace, or ``None`` where ad-hoc evaluation is not offered.

    Returns:
        One deferred reader per declared named expression, and the ad-hoc
        evaluator (or ``None``); calling either compiles and evaluates against
        the attached data.
    """
    attached = attach(program, sources)
    compiler = PolarsCompiler(program, attached, {}, None)
    return readback.evaluation_readers(compiler, program.named_expressions, lower)
