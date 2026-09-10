"""What sits above both lanes: what a spec and its sources may arrive as, and what each lane can build.

None of these facts belongs to a lane. ``Buildable`` and ``Source`` are what
every verb in the package takes; ``LANES`` is read by ``check`` with no extra
installed and by the eager lane when it refuses, so it is data here rather
than a property of a lane that may not be importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from math_spec import to_program

from lpspec.errors import LpspecError
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
type Buildable = str | Path | dict[str, Any] | Spec | Program


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


def _case_collision(program: Program) -> str | None:
    """Two declarations of one namespace whose names differ only by case, as the sentence refusing them.

    The namespaces are the language's own — the flat one every expression
    refers into, and constraints beside it — read off the program rather than
    re-derived, so this and math-spec cannot come to disagree about what being
    declared twice means.
    """
    flat = (
        *(('dimension', name) for name in program.dimensions),
        *(('lookup', lookup.name) for _, lookup in program.lookups),
        *(('parameter', name) for name in program.parameters),
        *(('variable', name) for name in program.variables),
        *(('named expression', name) for name in program.named_expressions),
    )
    for namespace in (flat, tuple(('constraint', name) for name in program.constraints)):
        seen: dict[str, tuple[str, str]] = {}
        for kind, name in namespace:
            if (earlier := seen.get(name.casefold())) is not None:
                return (
                    f"{kind} '{name}' and {earlier[0]} '{earlier[1]}' differ only by case, and one answer "
                    f'on disk cannot hold both: a declaration is written as a file named after it, and a '
                    f'case-insensitive filesystem — a stock macOS volume, a stock Windows one — folds the '
                    f'two into one, so the second overwrites the first and one name comes back carrying '
                    f"the other's values. Tell them apart by a suffix rather than a capital: 'p_rated' "
                    f"beside 'p'."
                )
            seen[name.casefold()] = (kind, name)
    return None


def lowered(spec: Buildable) -> Program:
    """*spec* as a program, refusing what this package cannot keep apart.

    Every door lowers through here rather than through ``to_program``, so what
    :func:`check` refuses :func:`build` and an archive refuse too: a rule only
    the front door enforced is one ``solve`` walks past.

    Raises:
        LanguageError: A construct outside the streaming language.
        LpspecError: Two declarations of one namespace whose names differ only
            by case.
    """
    program = to_program(spec)
    if (collision := _case_collision(program)) is not None:
        raise LpspecError(collision)
    return program
