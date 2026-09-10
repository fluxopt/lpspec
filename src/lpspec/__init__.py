"""Declarative optimisation: YAML math on a streaming engine.

Specs build relationally on polars and stream to the solver — see
docs/about/architecture.md. linopy is not imported here; with the ``[linopy]``
extra it is the second lane a file can be built on, and the differential-test
oracle (``from lpspec import linopy as lpspec_linopy``).

Example::

    import lpspec as lps

    result = lps.solve('spec.yaml', {'p_max': 'p_max.parquet', 'load': 'load.parquet'})
    result.objective
    result.primal('p')  # tidy polars.DataFrame
    result.to_dataarray('p')  # labelled, for array post-processing

``__version__`` reads the installed metadata: the git tag is the source of
truth, and hatch-vcs bakes it in at build time. A source tree with nothing
installed reads ``0.0.0``.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version

from lpspec.api import Model, build, check, load_result, solve, write
from lpspec.artifact import SolveArtifact, SweepArtifact, load_artifact
from lpspec.errors import (
    DataError,
    DimensionError,
    LaneError,
    LanguageError,
    LayoutError,
    LpspecError,
    LpspecWarning,
    NoSolutionError,
    PiecewiseExpansionError,
    SchemaError,
)
from lpspec.relational.result import Result
from lpspec.strategy import EachCoordinate, EachWindow, Runs, load_runs, solve_over

__all__ = [
    'DataError',
    'DimensionError',
    'EachCoordinate',
    'EachWindow',
    'LaneError',
    'LanguageError',
    'LayoutError',
    'LpspecError',
    'LpspecWarning',
    'Model',
    'NoSolutionError',
    'PiecewiseExpansionError',
    'Result',
    'Runs',
    'SchemaError',
    'SolveArtifact',
    'SweepArtifact',
    'build',
    'check',
    'load_artifact',
    'load_result',
    'load_runs',
    'solve',
    'solve_over',
    'write',
]

try:
    __version__ = _installed_version('lpspec')
except _PackageNotFoundError:
    __version__ = '0.0.0'
