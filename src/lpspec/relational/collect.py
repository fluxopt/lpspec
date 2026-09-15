"""The one place that decides which polars engine materialises a frame.

Every collect in the package asks for the streaming engine, and a polars built
without it — the browser's, for one — refuses the request with a panic rather
than falling back. So the question is put to polars once, and every collect
reads the answer.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import polars as pl

__all__ = ['polars_engine']


@cache
def polars_engine() -> Literal['streaming', 'in-memory']:
    """The engine every ``collect`` names: streaming where this polars has it, in-memory otherwise."""
    try:
        pl.LazyFrame({'probe': [0]}).collect(engine='streaming')
    except BaseException:  # a refusal is a pyo3 panic, which is not an Exception
        return 'in-memory'
    return 'streaming'
