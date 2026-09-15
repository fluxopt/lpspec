"""`storage`, in every dialect that has one."""

from bench.models.storage import gurobipy_loop, linopy, matrix, pyomo

FORMULATIONS = {
    'linopy': linopy,
    'pyomo': pyomo,
    'gurobipy-loop': gurobipy_loop,
    'gurobipy-matrix': matrix,
    'highspy-matrix': matrix,
}
