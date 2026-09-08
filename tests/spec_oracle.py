"""The differential oracle: linopy's own build of the same spec.

**A test module reaches this through** :mod:`tests.oracle`, never directly:
that module holds the ``[linopy]`` guard, and a bare ``import linopy`` in a
collected module would beat it. ``differential/pypsa/parity.py`` imports it
straight, pytest collecting nothing there and the harness having no bare
install to run on.

**The oracle is another package now.** It used to be `lpspec/linopy/`, a second
lane in this repository reading the same YAML through the same
`sources.tidy_sources`; linopy reads math-spec natively (PyPSA/linopy#922) and
that lane was deleted in favour of it. What a differential compares is
therefore two implementations rather than two passes over one, and the reader
is part of what it compares: `linopy.spec.attach` enforces the language's
binding rules in its own code, so a defect in ours is no longer a defect in
both.

Two verbs, keeping the signatures the deleted lane had so that a differential
test reads as it did — a producer and a reader, both pure:

    m = spec_oracle.build('spec.yaml', {...})
    m.solve(...)
    spec_oracle.expression(m, 'spec.yaml', 'co2', {...})

**What sits between is a spelling change and nothing else.** The two packages
take the same data in different containers: this one reads tables, linopy reads
pandas and xarray, and a dimension's labels are a one-column table here and a
sequence there. :func:`linopy_sources` moves a source from the first spelling
to the second when it is in the first, and hands over anything else untouched
— a defect included. Nothing here checks a value, a label, a dtype or a
coverage: every one of those questions is `attach`'s to answer in its own
words, which is the whole point of measuring against it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import linopy
import pandas as pd
import polars as pl
import xarray as xr
from linopy.spec import SpecDataError
from math_spec import to_program

from lpspec.frames import to_pandas

if TYPE_CHECKING:
    from collections.abc import Mapping

    from math_spec.program import Program

linopy.options['semantics'] = 'v1'

__all__ = ['DATETIME_SPELLING', 'SPARSE_COEFFICIENT', 'SpecDataError', 'build', 'expression', 'linopy_sources']

#: Why a differential over a sparse coefficient table is expected to fail.
#:
#: The one place the two implementations read the language differently today,
#: and it is the oracle that is wrong: math-spec's rule 8 says a missing
#: parameter row reads as whatever contributes nothing — **zero as a
#: coefficient** — and refuses only where no such reading exists (a divisor, a
#: bound, a constant side, a piecewise breakpoint). linopy's port made absence
#: uniform and refuses a missing row wherever it is used, coefficients
#: included, so it turns the language's own sparsity idiom into a
#: ``SpecDataError``.
#:
#: Every xfail carrying this reason is strict, so each one XPASSes and this
#: constant comes out the day the oracle is fixed upstream.
SPARSE_COEFFICIENT = (
    'the oracle refuses a sparse coefficient table, which math-spec rule 8 reads as a zero '
    'coefficient — an over-refusal in linopy.spec.coverage, not a disagreement about this model '
    '(PyPSA/linopy#922)'
)

#: Why a differential over a ``dtype: datetime`` dimension is expected to fail.
#:
#: The second place the two read the language differently. A dimension's labels
#: are canonicalised against the declared dtype here (#1076), so a
#: ``datetime.date`` index and a ``datetime64`` one are the same instant and
#: the same label. The oracle compares the objects it was handed, and xarray
#: promotes a lookup's values to ``datetime64`` on the way through — so a
#: ``date`` index makes every value look like a stranger, and which spelling
#: the caller reached for decides whether the model loads at all.
DATETIME_SPELLING = (
    'the oracle does not canonicalise a datetime dimension against its declared dtype, so a '
    'date-spelled index and a datetime64-spelled value are two labels there and one here '
    '(PyPSA/linopy#922)'
)


def build(spec: str | Path | dict[str, Any], sources: Mapping[str, Any]) -> linopy.Model:
    """Bind *sources* to *spec* and build it as a ``linopy.Model``.

    ``retain='all'`` because a differential reads expressions the model's own
    report closure would not keep, and a test that had to say which is a test
    about ``retain`` rather than about the language.
    """
    return linopy.Model.from_spec(spec, linopy_sources(spec, sources), retain='all')


def expression(
    built: linopy.Model,
    spec: str | Path | dict[str, Any],
    name: str,
    sources: Mapping[str, Any],
) -> xr.DataArray:
    """Evaluate named expression *name* of *spec* at *built*'s solution.

    *spec* is what the sources are classified against; linopy reads the
    expression off the model, which carries the spec text the build saw.
    """
    value = built.spec.evaluate(name, linopy_sources(spec, sources)).solution
    return value if isinstance(value, xr.DataArray) else xr.DataArray(float(value))


def linopy_sources(spec: str | Path | dict[str, Any], sources: Mapping[str, Any]) -> dict[str, Any]:
    """*sources* with every table respelled as the container linopy reads.

    What each key is — a dimension, a lookup, a parameter — is read off the
    lowered *spec*, since the three take different containers and a table
    cannot say which it is. A key the spec does not declare, and a source in a
    container linopy already reads, pass through: `attach` has a sentence for
    both.
    """
    program = to_program(spec)
    lookups = {lookup.name: (over, lookup.target or lookup.name) for over, lookup in program.lookups}
    labels = {name: _labels(name, source) for name, source in sources.items() if name in program.dimensions}
    return {name: _respelled(name, source, program, lookups.get(name), labels) for name, source in sources.items()}


def _respelled(
    name: str,
    source: Any,
    program: Program,
    relation: tuple[str, str] | None,
    labels: Mapping[str, Any],
) -> Any:
    """One source as linopy takes it, or unchanged when it already is.

    Defensive on purpose, and narrow on purpose: a table whose columns are not
    exactly the ones this package documents, or a sequence of the wrong
    length, is a defect under test. Respelling it anyway would repair it on the
    way past, and the differential would then compare a defect against a fix.
    """
    declared = program.parameters.get(name)
    if declared is not None and len(declared.dims) == 1 and isinstance(source, (list, tuple, range)):
        index = labels.get(declared.dims[0])
        if index is not None and len(index) == len(source):
            return pd.Series(list(source), index=pd.Index(index, name=declared.dims[0]))
        return source
    frame = _tabular(source)
    if frame is None:
        return source
    columns = list(frame.columns)
    if relation is not None:
        over, value = relation
        return frame.set_index(over)[value] if columns == [over, value] else source
    if name in program.dimensions:
        return frame[name].to_list() if columns == [name] else source
    if declared is not None and not declared.dims and columns == ['value'] and len(frame) == 1:
        return frame['value'].iloc[0]
    return frame


def _labels(name: str, source: Any) -> Any:
    """Dimension *name*'s labels as a sequence, or ``None`` if the source is not one.

    Only a frame whose one column is the dimension's own name: a frame carrying
    anything else is a defect the differential wants linopy to refuse, and
    reading labels out of it here would repair it on the way past.
    """
    frame = _tabular(source)
    if frame is not None:
        return frame[name].to_list() if list(frame.columns) == [name] else None
    return source if hasattr(source, '__len__') else None


def _tabular(source: Any) -> Any | None:
    """*source* as a pandas frame if it is a table this package spells, else ``None``."""
    if isinstance(source, (str, Path)) and str(source).endswith('.parquet'):
        return to_pandas(pl.read_parquet(source))
    if isinstance(source, pl.LazyFrame):
        return to_pandas(source.collect())
    if isinstance(source, pl.DataFrame):
        return to_pandas(source)
    if isinstance(source, pl.Series):
        return to_pandas(source.to_frame())
    if type(source).__module__.startswith('pandas') and hasattr(source, 'columns'):
        return source
    return None
