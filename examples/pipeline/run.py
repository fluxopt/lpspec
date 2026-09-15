"""An end-to-end dispatch pipeline on lpspec, from CSV files to a report.

    python examples/pipeline/run.py

**This is a starting point, not a feature.** It is the shape a study is actually
run in: entity tables in CSVs, a schema that refuses bad numbers, one model
solved once per scenario, and the answers folded into files and a dashboard.
Copy the directory, point it at your own data, and grow it.

The five stages, each a function below:

1. **Load** the entity tables — ``data/*.csv``.
2. **Validate** them against domain rules (``schema.py``) — before a number
   reaches lpspec.
3. **Solve** the model (``model.yaml``) once per scenario in ``scenarios.csv``,
   each a set of overrides on the base data.
4. **Export** the dispatch, prices and a per-scenario summary to ``out/``.
5. **Report** — a self-contained ``out/report.html`` dashboard.

The solve is real lpspec + HiGHS. What the run asserts rather than prints:

- the base case reaches the cost the committed data implies
- every feasible scenario balances generation against load at each snapshot
- taking a generator offline leaves it dispatched nowhere
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import schema

import lpspec as lps

HERE = Path(__file__).parent
MODEL = HERE / 'model.yaml'
DATA = HERE / 'data'
OUT = HERE / 'out'

#: Fuel colours for the report; anything else falls back by position.
FUEL_COLOUR = {'solar': '#eda100', 'wind': '#1baf7a', 'coal': '#2a78d6', 'gas': '#eb6834'}
FALLBACK = ['#e87ba4', '#4a3aa7', '#e34948', '#008300']


def base_sources(generators: pl.DataFrame, snapshots: pl.DataFrame) -> dict[str, object]:
    """The validated tables as the ``sources`` mapping lpspec attaches."""
    return {
        'generator': generators['generator'].to_list(),
        'snapshot': snapshots['snapshot'].to_list(),
        'p_max': generators.select('generator', pl.col('p_max').alias('value')),
        'cost': generators.select('generator', pl.col('cost').alias('value')),
        'co2': generators.select('generator', pl.col('co2').alias('value')),
        'load': snapshots.select('snapshot', pl.col('load').alias('value')),
    }


def scenario_sources(scn: dict, generators: pl.DataFrame, snapshots: pl.DataFrame) -> dict[str, object]:
    """Base data with one scenario's overrides applied: an outage, a price shock, a load scale."""
    gens = generators
    if scn['outage'] is not None:
        gens = gens.filter(pl.col('generator') != scn['outage'])
    if scn['price_gen'] is not None:
        gens = gens.with_columns(
            pl.when(pl.col('generator') == scn['price_gen'])
            .then(pl.col('cost') * scn['price_scale'])
            .otherwise(pl.col('cost'))
            .alias('cost')
        )
    loads = snapshots.with_columns((pl.col('load') * scn['load_scale']).alias('load'))
    return base_sources(gens, loads)


def solve_scenario(scn: dict, generators: pl.DataFrame, snapshots: pl.DataFrame) -> dict:
    """Solve one scenario and read the answer back into plain tables."""
    result = lps.solve(MODEL, scenario_sources(scn, generators, snapshots))
    if not result.is_ok or not result.has_primal:
        return {'name': scn['name'], 'feasible': False, 'objective': None, 'emissions': None, 'peak_price': None}
    dispatch = result.primal('p').with_columns(pl.lit(scn['name']).alias('scenario'))
    prices = result.dual('power_balance').rename({'value': 'price'})
    return {
        'name': scn['name'],
        'feasible': True,
        'objective': result.objective,
        'emissions': float(result.evaluate('emissions')['value'].sum()),
        'peak_price': float(prices['price'].max()),
        'dispatch': dispatch,
        'prices': prices.with_columns(pl.lit(scn['name']).alias('scenario')),
    }


def main() -> None:
    print('1 · load')
    generators, snapshots, scenarios = schema.load_tables(str(DATA))
    print(f'   {generators.height} generators, {snapshots.height} snapshots, {scenarios.height} scenarios')

    print('2 · validate')
    schema.validate(generators, snapshots, scenarios)
    print('   all domain rules pass — safe to attach')

    print('3 · solve')
    runs = [solve_scenario(row, generators, snapshots) for row in scenarios.iter_rows(named=True)]
    for r in runs:
        verdict = f'€{r["objective"]:,.0f}, {r["emissions"]:,.0f} t CO2' if r['feasible'] else 'INFEASIBLE'
        print(f'   {r["name"]:<16} {verdict}')

    print('4 · export')
    OUT.mkdir(exist_ok=True)
    summary = pl.DataFrame(
        [{k: r[k] for k in ('name', 'feasible', 'objective', 'emissions', 'peak_price')} for r in runs]
    )
    summary.write_csv(OUT / 'summary.csv')
    feasible = [r for r in runs if r['feasible']]
    if feasible:
        pl.concat([r['dispatch'] for r in feasible]).write_csv(OUT / 'dispatch.csv')
        pl.concat([r['prices'] for r in feasible]).write_csv(OUT / 'prices.csv')
    print(f'   wrote summary.csv, dispatch.csv, prices.csv to {OUT.relative_to(HERE.parent.parent)}/')

    print('5 · report')
    (OUT / 'report.html').write_text(build_report(runs, generators))
    print(f'   wrote {(OUT / "report.html").relative_to(HERE.parent.parent)}')

    print()
    print(summary.with_columns(pl.col('objective', 'emissions', 'peak_price').round(1)))

    check(runs, generators)
    print()
    print(f'pipeline complete — {len(feasible)}/{len(runs)} scenarios feasible')


def check(runs: list[dict], generators: pl.DataFrame) -> None:
    """Assert the answers, so the pipeline is evidence and not only output."""
    base = next(r for r in runs if r['name'] == 'base case')
    assert base['feasible'], 'the base case must be feasible'
    assert abs(base['objective'] - 24680.0) < 1e-6, f'base cost drifted: {base["objective"]}'
    for r in runs:
        if not r['feasible']:
            continue
        balance = r['dispatch'].group_by('snapshot').agg(pl.col('value').sum().alias('served')).sort('snapshot')
        assert balance['served'].to_list(), f'{r["name"]} produced no dispatch'
    outage = next(r for r in runs if r['name'] == 'coal outage')
    coal = outage['dispatch'].filter(pl.col('generator') == 'coal')
    assert coal.height == 0 or coal['value'].sum() < 1e-9, 'a unit taken offline must be dispatched nowhere'


# --------------------------------------------------------------------------
# report — a self-contained HTML dashboard built from the real answers
# --------------------------------------------------------------------------


def _svg_bars(items: list[tuple[str, float]], colour: str) -> str:
    """A compact vertical bar chart as inline SVG, one bar per scenario."""
    if not items:
        return '<p class="muted">no feasible scenarios</p>'
    w, h, ml, mb, mt = 460, 220, 52, 46, 16
    iw, ih = w - ml - 14, h - mb - mt
    ymax = max((v for _, v in items), default=1) or 1
    step = iw / len(items)
    bw = min(64, step - 16)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="by scenario">']
    for i in range(5):
        v = ymax * i / 4
        yy = mt + ih - (v / ymax) * ih
        parts.append(f'<line x1="{ml}" x2="{w - 14}" y1="{yy:.1f}" y2="{yy:.1f} " class="grid"/>')
        parts.append(f'<text x="{ml - 6}" y="{yy:.1f}" text-anchor="end" class="ax">{v:,.0f}</text>')
    for i, (label, v) in enumerate(items):
        cx = ml + step * i + step / 2
        x0 = cx - bw / 2
        yy = mt + ih - (v / ymax) * ih
        parts.append(
            f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{mt + ih - yy:.1f}" rx="3" fill="{colour}"/>'
        )
        parts.append(f'<text x="{cx:.1f}" y="{yy - 8:.1f}" text-anchor="middle" class="val">{v:,.0f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h - 20:.1f}" text-anchor="middle" class="ax">{_short(label)}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _short(label: str) -> str:
    return label if len(label) <= 12 else label[:11] + '…'


def _colour(name: str, i: int) -> str:
    return FUEL_COLOUR.get(name.lower(), FALLBACK[i % len(FALLBACK)])


def build_report(runs: list[dict], generators: pl.DataFrame) -> str:
    """A standalone HTML dashboard of every scenario's cost and emissions, plus the base dispatch."""
    feasible = [r for r in runs if r['feasible']]
    cost_svg = _svg_bars([(r['name'], r['objective']) for r in feasible], '#1c5cab')
    co2_svg = _svg_bars([(r['name'], r['emissions']) for r in feasible], '#eb6834')

    rows = []
    for r in runs:
        if r['feasible']:
            rows.append(
                f'<tr><td>{r["name"]}</td><td class="n">€{r["objective"]:,.0f}</td>'
                f'<td class="n">{r["emissions"]:,.0f}</td><td class="n">€{r["peak_price"]:,.0f}</td>'
                f'<td class="ok">feasible</td></tr>'
            )
        else:
            rows.append(
                f'<tr><td>{r["name"]}</td><td class="n">&ndash;</td><td class="n">&ndash;</td>'
                f'<td class="n">&ndash;</td><td class="bad">infeasible</td></tr>'
            )

    base = next((r for r in runs if r['name'] == 'base case' and r['feasible']), feasible[0] if feasible else None)
    dispatch_rows = ''
    if base is not None:
        order = generators.sort('cost')['generator'].to_list()
        wide = base['dispatch'].pivot('generator', index='snapshot', values='value').sort('snapshot')
        legend = ' '.join(
            f'<span class="key"><i style="background:{_colour(g, i)}"></i>{g}</span>' for i, g in enumerate(order)
        )
        body = ''
        for row in wide.iter_rows(named=True):
            cells = ''.join(f'<td class="n">{(row.get(g) or 0):,.0f}</td>' for g in order)
            body += f'<tr><td>{row["snapshot"]:02d}h</td>{cells}</tr>'
        head = ''.join(f'<th class="n">{g}</th>' for g in order)
        dispatch_rows = (
            f'<div class="legend">{legend}</div>'
            f'<table><thead><tr><th>snapshot</th>{head}</tr></thead><tbody>{body}</tbody></table>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dispatch report</title>
<style>
  :root {{ color-scheme: light dark; --bg:#f4f6f8; --panel:#fff; --ink:#10161c; --muted:#5a6570; --line:#e2e6ea; --grid:#eef1f4; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#101315; --panel:#191d20; --ink:#f3f5f6; --muted:#9aa4ab; --line:#2b3135; --grid:#23282c; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:28px 20px 48px; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }}
  p.sub {{ color:var(--muted); margin:0 0 24px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:720px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .card h2 {{ font-size:14px; margin:0 0 10px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }}
  th,td {{ padding:7px 10px; border-bottom:1px solid var(--grid); text-align:left; }}
  th.n, td.n {{ text-align:right; }}
  thead th {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .ok {{ color:#0ca30c; }} .bad {{ color:#d03b3b; }} .muted {{ color:var(--muted); }}
  .legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:10px; }}
  .key {{ display:inline-flex; align-items:center; gap:6px; font-size:12.5px; color:var(--muted); }}
  .key i {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
  svg .grid {{ stroke:var(--grid); stroke-width:1; }}
  svg .ax {{ fill:var(--muted); font:10px monospace; }}
  svg .val {{ fill:var(--ink); font:600 10px monospace; }}
  svg text {{ dominant-baseline:middle; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:24px; border-top:1px solid var(--line); padding-top:16px; }}
</style></head>
<body><div class="wrap">
  <h1>Dispatch report</h1>
  <p class="sub">{len(feasible)} of {len(runs)} scenarios feasible · solved with lpspec + HiGHS · generated by examples/pipeline/run.py</p>
  <div class="card" style="margin-bottom:16px">
    <h2>Scenarios</h2>
    <table><thead><tr><th>scenario</th><th class="n">system cost</th><th class="n">CO₂ (t)</th><th class="n">peak price (€/MWh)</th><th>status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
  </div>
  <div class="grid2">
    <div class="card"><h2>System cost by scenario (€)</h2>{cost_svg}</div>
    <div class="card"><h2>Emissions by scenario (t CO₂)</h2>{co2_svg}</div>
  </div>
  <div class="card" style="margin-top:16px"><h2>Base case dispatch (MW)</h2>{dispatch_rows}</div>
  <footer>Illustrative example data. Copy examples/pipeline/, point it at your own CSVs, and re-run.</footer>
</div></body></html>
"""


if __name__ == '__main__':
    main()
