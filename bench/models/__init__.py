"""One directory per case: the model in every dialect that can express it.

`bench/cases.py` holds what a case *is* — its ladder, its cardinalities, the
parquet its generator writes. Here is what it *says*, once per language that
says it: `model.yaml` for lpspec, and one module per hand-written dialect.

A case with no alternative formulation is just its `model.yaml`; the ones that
have one carry a `FORMULATIONS` map naming what they hold. Written out rather
than discovered by scanning, so a misnamed module is an import error rather
than an arm that quietly measures nothing.

`matrix.py` is the exception to one module per dialect: the two matrix arms
map to it together, because what they differ in is the solver they push an
`Lp` into rather than the matrix they build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import numpy as np


class Lp(NamedTuple):
    """A case as a matrix, in the vocabulary every solver's bulk API shares.

    Columns carry `lower`, `upper` and `obj`; rows carry `matrix`, one `senses`
    character each and `rhs`. `senses` is `'='` or `'<'` — no case here needs a
    third, and an arm meeting one raises rather than guesses.

    Nothing on it is solver-specific, which is the whole reason it exists: a
    case's matrix had one home per matrix arm and would have had two once
    `highspy-matrix` arrived, so the second copy is the one that drifts and the
    published floor is what it would have taken down with it.

    Attributes:
        lower: Column lower bounds, one per column.
        upper: Column upper bounds, one per column.
        obj: Objective coefficient per column; the sense is always minimise.
        matrix: The constraint matrix, scipy CSR, rows x columns.
        senses: One of `'='` or `'<'` per row.
        rhs: The right-hand side, one per row.
    """

    lower: np.ndarray
    upper: np.ndarray
    obj: np.ndarray
    matrix: Any
    senses: np.ndarray
    rhs: np.ndarray


def formulation(case_name: str, dialect: str):
    """The case's model in *dialect*, or None where nobody has written one.

    A case package names what it holds in `FORMULATIONS`; a case that holds
    nothing has no package at all, which is the same answer. What `build`
    takes is the *arm's* contract rather than a shared one — `gurobipy-loop`
    hands its formulations an `Env`, because an environment is arm-level
    knowledge and a model file should not be creating one, where the matrix
    dialects take the tables alone and hand back an `Lp`.
    """
    import importlib

    try:
        case = importlib.import_module(f'bench.models.{case_name}')
    except ModuleNotFoundError:
        return None
    return getattr(case, 'FORMULATIONS', {}).get(dialect)
