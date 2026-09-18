"""Declarative optimisation: YAML math on a streaming engine.

Specs build relationally on polars and stream to the solver — see
docs/about/architecture.md. linopy is not imported here; with the ``[linopy]``
extra it is the second lane a file can be built on, and the differential-test
oracle (``from specsolve import linopy as specsolve_linopy``).

Example::

    import specsolve as sps

    result = sps.solve('spec.yaml', {'p_max': 'p_max.parquet', 'load': 'load.parquet'})
    result.objective
    result.primal('p')  # tidy polars.DataFrame
    result.to_dataarray('p')  # labelled, for array post-processing

``__version__`` reads the installed metadata: the git tag is the source of
truth, and hatch-vcs bakes it in at build time. A source tree with nothing
installed reads ``0.0.0``.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version

from specsolve.api import Model, build, check, evaluate, load_result, scan_result, solve, write
from specsolve.archive import SolveArchive, SweepArchive, load_archive, scan_archive
from specsolve.errors import (
    DataError,
    DimensionError,
    LaneError,
    LanguageError,
    LayoutError,
    NoSolutionError,
    PiecewiseExpansionError,
    SchemaError,
    SpecsolveError,
    SpecsolveWarning,
)
from specsolve.relational.result import Result
from specsolve.strategy import EachCoordinate, EachWindow, Runs, load_runs, scan_runs, solve_over

__all__ = [
    'DataError',
    'DimensionError',
    'EachCoordinate',
    'EachWindow',
    'LaneError',
    'LanguageError',
    'LayoutError',
    'Model',
    'NoSolutionError',
    'PiecewiseExpansionError',
    'Result',
    'Runs',
    'SchemaError',
    'SolveArchive',
    'SpecsolveError',
    'SpecsolveWarning',
    'SweepArchive',
    'build',
    'check',
    'evaluate',
    'load_archive',
    'load_result',
    'load_runs',
    'scan_archive',
    'scan_result',
    'scan_runs',
    'solve',
    'solve_over',
    'write',
]

try:
    __version__ = _installed_version('specsolve')
except _PackageNotFoundError:
    __version__ = '0.0.0'
