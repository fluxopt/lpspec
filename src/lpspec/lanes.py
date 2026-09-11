"""What sits above both lanes: what a spec and its sources may arrive as, and what each lane can build.

None of these facts belongs to a lane. ``Buildable`` and ``Source`` are what
every verb in the package takes; ``LANES`` is read by ``check`` with no extra
installed and by the eager lane when it refuses, so it is data here rather
than a property of a lane that may not be importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from math_spec import to_spec
from math_spec.program import Program

from lpspec.errors import LpspecError
from lpspec.relational.sinks.capabilities import Capabilities

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from datetime import datetime
    from pathlib import Path

    import pandas as pd
    import polars as pl
    from math_spec import Spec

#: Anything a verb takes as the spec: a YAML path, a mapping, or a ``Spec``
#: the language has already read. **Not** a ``Program``: lowering has no
#: inverse, so an answer from one could not name the model it came from, and
#: nothing built from one can be archived. Keeping the ``Spec`` is both the
#: archivable form and the cheaper one — reading a file costs about ten times
#: what lowering it does (#1579).
type Buildable = str | Path | dict[str, Any] | Spec


def declared(spec: Buildable) -> Spec:
    """*spec* as the document it is, whatever shape it arrived in.

    The one door every verb reads a model through, so the refusal below is
    written once. A lowered ``Program`` reaching :func:`math_spec.to_spec`
    raises ``argument of type 'Program' is not iterable``, which names neither
    what is wrong nor what to pass.

    Args:
        spec: A YAML path, a mapping, or a ``Spec``.

    Raises:
        LpspecError: A lowered ``Program``.
        LanguageError: Anything the language does not accept.
    """
    if isinstance(spec, Program):
        raise LpspecError(
            'a lowered Program is not a model this takes. Lowering has no inverse, so an answer from one '
            'could not say which document it came back from, and nothing built from one could be archived. '
            'Pass what it was lowered from — a path, a mapping, or math_spec.to_spec() of either, which is '
            'the form worth keeping: reading a file costs about ten times what lowering it does. '
            'lps.check() still hands back the Program, for reading the plan.'
        )
    return to_spec(spec)


@runtime_checkable
class ArrowTable(Protocol):
    """Any table that exposes the Arrow PyCapsule stream — pyarrow, DuckDB, ibis — read without importing it."""

    def __arrow_c_stream__(self, requested_schema: object = None) -> object: ...


#: A label along a dimension, and so a slice's key: the Python type of each
#: dtype an index may declare (:data:`math_spec.program.DimensionDtype`).
type Label = int | float | str | datetime

#: Anything a verb takes under one name of ``sources``. A parameter: a parquet
#: path, a table — polars, pandas, or any :class:`ArrowTable` — or one of the
#: plain-Python shapes a hand-written model reaches for, a ``{label: value}``
#: map, a sequence in the dimension's own label order, and one number for
#: every coordinate. A dimension's index: a table carrying a column named
#: after it, or a bare sequence of its labels. A lookup: the table of the
#: rows it maps.
type Source = (
    str
    | Path
    | pl.DataFrame
    | pl.LazyFrame
    | pd.DataFrame
    | pd.Series
    | ArrowTable
    | Mapping[Label, float]
    | Collection[Label]
    | float
)

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
