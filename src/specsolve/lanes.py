"""What sits above both lanes: what a spec and its sources may arrive as, and what each lane can build.

None of these facts belongs to a lane. ``Buildable`` and ``Source`` are what
every verb in the package takes; ``LANES`` is read by ``check`` with no extra
installed and by the eager lane when it refuses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from math_spec import to_program, to_spec
from math_spec.program import Program

from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.capabilities import Capabilities

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
#: nothing built from one can be archived.
type Buildable = str | Path | dict[str, Any] | Spec


def declared(spec: Buildable) -> Spec:
    """*spec* as the document it is, whatever shape it arrived in.

    The one door every verb reads a model through.

    Args:
        spec: A YAML path, a mapping, or a ``Spec``.

    Raises:
        SpecsolveError: A lowered ``Program``.
        LanguageError: Anything the language does not accept.
    """
    if isinstance(spec, Program):
        raise SpecsolveError(
            'a lowered Program is not a model this takes. Lowering has no inverse, so an answer from one '
            'could not say which document it came back from, and nothing built from one could be archived. '
            'Pass what it was lowered from — a path, a mapping, or math_spec.to_spec() of either, which is '
            'the form worth keeping: reading a file costs about ten times what lowering it does. '
            'sps.check() still hands back the Program, for reading the plan.'
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
#: after it, or a bare sequence of its labels. A relation: the table of the
#: rows it holds, one column per column it declares.
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
    refers into, and constraints beside it.
    """
    flat = (
        *(('dimension', name) for name in program.dimensions),
        *(('relation', name) for name in program.relations),
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

    Every door lowers through here, so what :func:`check` refuses
    :func:`build` and an archive refuse too.

    Raises:
        LanguageError: A construct outside the streaming language.
        SpecsolveError: Two declarations of one namespace whose names differ only
            by case.
    """
    program = to_program(spec)
    if (refused := _case_collision(program)) is not None:
        raise SpecsolveError(refused)
    return program
