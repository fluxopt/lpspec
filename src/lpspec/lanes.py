"""What sits above both lanes: what a spec and its sources may arrive as, and what each lane can build.

None of these facts belongs to a lane. ``Buildable`` and ``Source`` are what
every verb in the package takes; ``LANES`` is read by ``check`` with no extra
installed and by the eager lane when it refuses, so it is data here rather
than a property of a lane that may not be importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from lpspec.relational.sinks.capabilities import Capabilities

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from datetime import datetime
    from pathlib import Path

    import pandas as pd
    import polars as pl
    from math_spec import Spec
    from math_spec.program import Program

#: Anything a verb takes as the spec: a YAML path, a mapping, or a spec the
#: language has already read — a ``Spec`` from :func:`math_spec.to_spec`, or a
#: ``Program`` from :func:`lpspec.check`. Each is handed straight to
#: :func:`math_spec.to_program`.
Buildable: TypeAlias = 'str | Path | dict[str, Any] | Spec | Program'


@runtime_checkable
class ArrowTable(Protocol):
    """Any table that exposes the Arrow PyCapsule stream — pyarrow, DuckDB, ibis — read without importing it."""

    def __arrow_c_stream__(self, requested_schema: object = None) -> object: ...


#: A label along a dimension, and so a slice's key: the Python type of each
#: dtype an index may declare (:data:`math_spec.program.DimensionDtype`).
Label: TypeAlias = 'int | float | str | datetime'

#: Anything a verb takes under one name of ``sources``. A parameter: a parquet
#: path, a table — polars, pandas, or any :class:`ArrowTable` — or one of the
#: plain-Python shapes a hand-written model reaches for, a ``{label: value}``
#: map, a sequence in the dimension's own label order, and one number for
#: every coordinate. A dimension's index: a table carrying a column named
#: after it, or a bare sequence of its labels. A lookup: the table of the
#: rows it maps.
Source: TypeAlias = 'str | Path | pl.DataFrame | pl.LazyFrame | pd.DataFrame | pd.Series | ArrowTable | Mapping[Label, float] | Collection[Label] | float'  # fmt: skip

#: What each **lane** can build, beside what each sink can ingest: both lanes
#: accept the same language, and one cannot build a quadratic constraint —
#: ``linopy.Model.add_constraints`` refuses a ``QuadraticExpression`` outright
#: and no reformulation of it is exact.
LANES: Mapping[str, Capabilities] = {
    'linopy': Capabilities(
        supports={
            'integrality': 'native',
            'sos': 'native',
            'quadratic_objective': 'native',
            'nonconvex_quadratic_objective': 'native',
        },
    ),
}
