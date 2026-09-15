"""The linopy lane: a YAML spec built as a ``linopy.Model``.

Requires the ``[linopy]`` extra (linopy, xarray).

One language, two lanes: the same file either attaches relationally and solves
through :mod:`lpspec.api`, or is constructed here as a ``linopy.Model`` the
caller then owns. Both accept *exactly* the same language, which is what makes
the differential tests an oracle rather than a comparison of dialects.

Two functions — a producer and a reader — and both are **pure**: YAML goes in,
a model or a value comes out, and nothing is retained. :func:`evaluate` takes
``sources`` again rather than remembering what :func:`build` saw::

    from lpspec import linopy as lpspec_linopy

    m = lpspec_linopy.build('spec.yaml', {...})
    m.solve(...)
    lpspec_linopy.evaluate(m, 'spec.yaml', 'co2', {...})  # a name the file declares
    lpspec_linopy.evaluate(m, 'spec.yaml', 'sum(p * rate)', {...})  # one it never did

The same spec on the other lane, which streams::

    import lpspec as lps

    with lps.solve('spec.yaml', {...}) as result:
        result.primal('p')

**Importing this module sets** ``linopy.options['semantics'] = 'v1'``. This
lane speaks v1 and the option is global: linopy's ``legacy`` default fills every
absent slot with 0, where the relational lane drops the row, so left alone the
two lanes answer the same YAML with different numbers. linopy's own context
manager cannot scope it — its ``__exit__`` resets *every* option to its default
rather than its prior value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:
    import linopy
    import xarray
except ModuleNotFoundError as exc:
    msg = 'The linopy lane requires the [linopy] extra: pip install "lpspec[linopy]"'
    raise ModuleNotFoundError(msg) from exc


from lpspec import expressions
from lpspec.lanes import declared, lowered
from lpspec.linopy._notes import note
from lpspec.linopy.builder import _eval, build_model
from lpspec.linopy.loader import dimension_coords, load_parameters
from lpspec.linopy.where import EvaluationContext
from lpspec.sources import tidy_sources

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from lpspec.lanes import Buildable, Source

linopy.options['semantics'] = 'v1'

__all__ = ['build', 'evaluate']


def build(spec: Buildable, sources: Mapping[str, Source]) -> linopy.Model:
    """Attach *sources* to *spec* and build it as a ``linopy.Model``.

    Args:
        spec: As :func:`lpspec.check` takes it.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — an index table, a parquet path,
            or a bare sequence.

    Returns:
        A model carrying every declaration the file makes.

    Raises:
        LanguageError: A construct the language does not accept — the same
            verdict :func:`lpspec.check` gives, reached through the same
            lowering pass, so neither lane accepts a file the other refuses.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    with note(f'while loading {_named(spec)}'):
        program = lowered(spec)

        tidy = tidy_sources(program, sources)
        master_coords, relations = dimension_coords(program, tidy)
        dataset = load_parameters(program, tidy, master_coords)

        built = linopy.Model()
        build_model(built, program, dataset, master_coords, relations)

    return built


def evaluate(
    built: linopy.Model,
    spec: Buildable,
    expression: str | Mapping[str, Any],
    sources: Mapping[str, Source],
) -> xarray.DataArray:
    """Evaluate *expression*, written in *spec*'s namespace, at *built*'s solution.

    The eager half of :meth:`lpspec.Result.evaluate`. Pure like :func:`build` —
    nothing is retained, so *sources* is taken again rather than remembered.

    Args:
        built: A solved model carrying this file's variables.
        spec: The model the expression is written against, as :func:`build`
            takes it — but not a lowered ``Program``.
        expression: What ``expressions:`` takes — a string, or the mapping
            carrying ``cases:`` with ``dims:`` and ``otherwise:``. A name
            *spec* declares works too.
        sources: As :func:`build` takes them.

    Returns:
        The expression's value over its own dims, as an ``xarray.DataArray``
        (0-dimensional for a variable-free scalar expression).

    Raises:
        LanguageError: A construct the language does not accept, in the file or
            in the expression, or a name *spec* does not declare.
        DataError: A source that does not fit the file.
        LpspecError: A lowered ``Program`` as *spec*, or an expression that
            reads a dual where the solve left none.
    """
    with note(f'while evaluating an expression against {_named(spec)}'):
        written = declared(spec)
        node = expressions.lower(written, expression)
        program = lowered(written)
        tidy = tidy_sources(program, sources)
        master_coords, relations = dimension_coords(program, tidy)
        dataset = load_parameters(program, tidy, master_coords)
        context = EvaluationContext(dataset, master_coords, built, relations, program, solved=True)
        value = _eval(node, context)
        if isinstance(value, xarray.DataArray):
            return value
        return xarray.DataArray(float(value))


def _named(spec: Buildable) -> str:
    """What to call *spec* in an error note.

    A path names itself; a mapping or an already-loaded schema has no name.
    """
    return f"YAML '{spec}'" if isinstance(spec, (str, Path)) else 'the spec passed in'
