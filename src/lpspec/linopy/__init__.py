"""The linopy lane: a YAML spec built as a ``linopy.Model``.

Requires the ``[linopy]`` extra (linopy, xarray).

One language, two lanes: the same file either attaches relationally and solves
through :mod:`lpspec.api`, or is constructed here as a ``linopy.Model`` the
caller then owns. Both accept *exactly* the same language, which is what makes
the differential tests an oracle rather than a comparison of dialects.

Two functions — a producer and a reader — and both are **pure**: YAML goes in,
a model or a value comes out, and nothing is retained, which is why
:func:`evaluate` takes ``sources`` again rather than remembering what
:func:`build` saw::

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


from math_spec import to_program, to_spec
from math_spec.program import Program

from lpspec import expressions
from lpspec.errors import LpspecError, no_written_model_message
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
    """Bind *sources* to *spec* and build it as a ``linopy.Model``.

    :func:`lpspec.build`'s signature: which lane builds a file is the caller's
    choice, so the call cannot differ.

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
        program = to_program(spec)

        tidy = tidy_sources(program, sources)
        master_coords, dim_coords = dimension_coords(program, tidy)
        dataset = load_parameters(program, tidy, master_coords)

        built = linopy.Model()
        build_model(built, program, dataset, master_coords, dim_coords)

    return built


def evaluate(
    built: linopy.Model,
    spec: Buildable,
    expression: str | Mapping[str, Any],
    sources: Mapping[str, Source],
) -> xarray.DataArray:
    """Evaluate *expression*, written in *spec*'s namespace, at *built*'s solution.

    The eager half of :meth:`lpspec.Result.evaluate`, so the differential suite
    holds the two lanes to one answer. Pure like :func:`build`: nothing is
    retained, which is why *sources* is taken again rather than remembered.

    Args:
        built: A solved model carrying this file's variables.
        spec: The model the expression is written against, as :func:`build`
            takes it — bar a lowered ``Program``, which is not a model as
            written and so has no namespace to read an expression in.
        expression: What ``expressions:`` takes — a string, or the mapping
            that carries ``cases:`` with its ``foreach:`` and ``otherwise:``.
            A name *spec* declares is an expression like any other, the
            language substituting it where it stands.
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
        if isinstance(spec, Program):
            raise LpspecError(no_written_model_message())
        written = to_spec(spec)
        node = expressions.lower(written, expression)
        program = to_program(written)
        tidy = tidy_sources(program, sources)
        master_coords, dim_coords = dimension_coords(program, tidy)
        dataset = load_parameters(program, tidy, master_coords)
        context = EvaluationContext(dataset, master_coords, built, dim_coords, program, solved=True)
        value = _eval(node, context)
        if isinstance(value, xarray.DataArray):
            return value
        return xarray.DataArray(float(value))


def _named(spec: Buildable) -> str:
    """What to call *spec* in an error note.

    A path names itself; a mapping or an already-loaded schema has no name, and
    saying so beats printing a dict into a traceback.
    """
    return f"YAML '{spec}'" if isinstance(spec, (str, Path)) else 'the spec passed in'
