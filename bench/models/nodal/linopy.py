"""`nodal` as a linopy user writes it.

The one case on the ladder whose mask is not vacuous. `dispatch` carries a
`where` its generator never triggers, so both lanes build the same dense
product there; here a quarter of the (node, tech) pairs exist and the other
three quarters are what the case measures.

`mask=` is linopy's spelling of the YAML's `where: installed > 0`, and the
shape of the argument is the point: the mask is node x tech where the variable
is snapshot x node x tech. Structural sparsity is time-invariant, so linopy
broadcasts one plane along the snapshot axis rather than storing the product —
the eager lane's best case for this shape, not a handicap arranged for it.

**`fillna(0)` is what makes the sum mean what the YAML means**, and it is the
line to read twice. An absent slot contributes zero to `sum(p, over=tech)`,
which is legacy linopy's default and v1's only under `fillna` — the same
mapping `lpspec.linopy` uses for a variable under `absence: zero`. Without it
the pinned linopy warns and the arm builds whichever model the option happens
to be set to, which is how a benchmark comes to measure a different model.

**The pivot to a dense node x tech frame is inside `build`, and is timed.**
`installed` arrives tidy, one row per pair that exists, because that is the
shape the sparsity lives in; squaring it up is work a linopy user does, and the
arm's contract is that each lane pays for its own ingestion. Labels come from
the declared dimension tables rather than from the pivot — a tech no node
installed would otherwise drop out of the model's coordinates, which is a
different model rather than a smaller one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import linopy
    import pandas as pd
    import xarray as xr

    snapshots = pd.Index(tables['snapshot']['snapshot'], name='snapshot')
    nodes = pd.Index(tables['node']['node'], name='node')
    techs = pd.Index(tables['tech']['tech'], name='tech')

    capacity = xr.DataArray(
        tables['installed'].pivot(index='node', columns='tech', values='value').reindex(index=nodes, columns=techs)
    ).fillna(0.0)
    demand = xr.DataArray(
        tables['demand'].pivot(index='snapshot', columns='node', values='value').reindex(index=snapshots, columns=nodes)
    )
    cost = tables['cost'].set_index('tech')['value'].reindex(techs)

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=capacity, coords=[snapshots, nodes, techs], mask=capacity > 0, name='p')
    m.add_constraints(p.fillna(0).sum('tech') == demand, name='balance')
    m.add_objective((p.fillna(0) * cost).sum())
    return m
