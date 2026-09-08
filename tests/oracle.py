"""Guarded access to the differential oracle — linopy's own build of a spec.

Importing this module skips the importing test module when the ``[linopy]``
extra is absent — ``pytest.importorskip`` raises ``Skipped``, and pytest turns
that into a skipped module at collection time.

Import the oracle *through here* rather than importing linopy or xarray
directly, so the guard cannot be bypassed by import ordering: isort sorts a
bare ``import xarray`` above a first-party import, and it would then blow up
as a collection error before any guard ran. ``tests.spec_oracle`` — the oracle's two
verbs — imports linopy at module level for that reason and is reached only
from here.

**pandas is re-exported for the same reason.** It is no longer a runtime
dependency — it ships with the ``[linopy]`` extra, for the oracle and for
``Result.to_pandas`` — so a bare ``import pandas as pd`` in a test module is
exactly the ordering bug described above, one dependency down. Test modules
take ``pd`` from here instead, and the guard covers it.

This replaces a hand-maintained list of filenames in ``conftest.py``, which
had to be edited every time a test module was added and silently mis-skipped
when it was not.

**The oracle is v1, and only v1.** A differential test is an oracle only if the
thing it compares against is the convention we implement. Legacy is the one
linopy is retiring: it fills every absent slot with 0, so it agrees with a lane
that keeps a constraint row whose variable is masked, and disagrees with one
that drops it. Measuring against legacy would pin this package to the behaviour
v1 classifies as a bug (PyPSA/linopy#712) — and ``add_spec`` refuses to build
under it at all. So the guards below raise rather than skipping: a skip would
be the worst outcome available, the suite going green having quietly stopped
checking this engine against the other implementation.

**One thing no guard here can check: that both sides lower with the same
math-spec.** Two versions of the language is two languages, and a differential
across them reports as a disagreement between implementations what is a
disagreement between releases. It is checked by resolution rather than at
runtime — one environment holds one math-spec, and two exact pins that disagree
fail the install — so what keeps it true is that the two pins move together:
``[project.dependencies]`` here, and the rev of the ``[linopy]`` extra, whose
own ``spec`` group names the version the oracle was written against.
"""

from __future__ import annotations

import pytest

_REASON = 'needs the [linopy] extra (linopy, xarray, pandas)'

linopy = pytest.importorskip('linopy', reason=_REASON)
xr = pytest.importorskip('xarray', reason=_REASON)
pd = pytest.importorskip('pandas', reason=_REASON)

if 'semantics' not in getattr(linopy.options, '_defaults', {}):
    raise RuntimeError(
        f'linopy {linopy.__version__} has no options["semantics"], so it cannot speak the v1 '
        f'arithmetic convention this package is written against. The oracle would silently '
        f'measure against the legacy convention instead. Install the pin in pyproject.toml '
        f'(the [linopy] extra: PyPSA/linopy at the rev it names) — `pixi install`.'
    )
if not hasattr(linopy.Model, 'from_spec'):
    raise RuntimeError(
        f'linopy {linopy.__version__} has no Model.from_spec, so there is no oracle to measure '
        f"against: what the differential compares this engine to is linopy's own build of the "
        f'same spec. Install the pin in pyproject.toml (the [linopy] extra) — `pixi install`.'
    )

from tests import spec_oracle  # noqa: E402  — must follow the guards above

__all__ = [
    'linopy',
    'pd',
    'spec_oracle',
    'transport_eager_objective',
    'xr',
]


def transport_eager_objective(gens, lines, load) -> float:
    gi = gens.set_index('generator')
    li = lines.set_index('line')
    snapshots = pd.Index(sorted(load['snapshot'].unique()), name='snapshot')
    buses = pd.Index(sorted(load['bus'].unique()), name='bus')

    load_da = xr.DataArray.from_series(load.set_index(['snapshot', 'bus'])['value'])
    p_max = xr.DataArray.from_series(gi['p_max'])
    cost = xr.DataArray.from_series(gi['cost'])
    cap = xr.DataArray.from_series(li['cap'])

    gen_at = xr.DataArray(
        (gi['bus'].to_numpy()[None, :] == buses.to_numpy()[:, None]).astype(float),
        coords={'bus': buses, 'generator': gi.index},
        dims=['bus', 'generator'],
    )
    line_in = xr.DataArray(
        (li['to_bus'].to_numpy()[None, :] == buses.to_numpy()[:, None]).astype(float),
        coords={'bus': buses, 'line': li.index},
        dims=['bus', 'line'],
    )
    line_out = xr.DataArray(
        (li['from_bus'].to_numpy()[None, :] == buses.to_numpy()[:, None]).astype(float),
        coords={'bus': buses, 'line': li.index},
        dims=['bus', 'line'],
    )

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, gi.index], name='p')
    f = m.add_variables(lower=-cap, upper=cap, coords=[snapshots, li.index], name='f')
    injection = (p * gen_at).sum('generator') + (f * line_in).sum('line') - (f * line_out).sum('line')
    m.add_constraints(injection == load_da, name='balance')
    m.add_objective((p * cost).sum())
    m.solve(solver_name='highs', output_flag=False)
    return float(m.objective.value)
