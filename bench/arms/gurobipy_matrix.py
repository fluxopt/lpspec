"""The `gurobipy-matrix` arm. Its verbs are `bench.arms.gurobipy`'s; what makes
it an arm of its own is which formulation module they reach for."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench.arms import gurobipy as runtime

if TYPE_CHECKING:
    from collections.abc import Mapping

DIALECT = 'gurobipy-matrix'
SINKS = runtime.SINKS

#: The runtime's, plus the CSR the shared formulation builds — `gurobipy-loop`
#: writes its model call by call and needs no `scipy`, so this cannot be the
#: runtime's list unchanged.
REQUIRES = (*runtime.REQUIRES, 'scipy')

build_and_emit = runtime.build_and_emit
build_only = runtime.build_only
objective = runtime.objective


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Any:
    return runtime.prepare(DIALECT, case_name, size, paths, options)
