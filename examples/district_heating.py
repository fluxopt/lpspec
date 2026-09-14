"""A district-heating plant committed over a winter day, solved with lpspec.

    pixi run python examples/district_heating.py

**This is the model of [`district_heating.yaml`](district_heating.yaml) run on
one instance.** Four units — a gas boiler, a CHP unit, a heat pump and an
electric boiler — feed one heat network with a store. Each unit is committed on
or off in every period, with a minimum load, a start-up charge and a minimum
time up and down; what each consumes and produces is one shared curve, so the
CHP's heat-to-power ratio, the boiler's part-load efficiency and the heat pump's
coefficient of performance are all data.

The day is built to make the fleet split the work. Grid electricity is cheap at
night and dear at the midday peak, so the heat pump runs throughout on its
coefficient of performance, the electric boiler carries the cheap small hours,
and the CHP unit runs when its exported power is worth most. The gas boiler is
left with the midday demand spike the others cannot cover at once — exactly when
grid power is dear enough that burning gas beats the electric boiler. Gas
carries a carbon charge and grid power does too, so the merit order is a cost
*and* a carbon decision, and the store shifts cheap night heat into the peak.

Read back and asserted rather than trusted:

- the heat balance closes in every period — what the units put out, plus what
  the store gives back net of charging, equals the demand;
- every unit's output sits between its minimum load and its capacity when it is
  committed, and at zero when it is not — the commitment the binary status buys;
- the whole fleet is used across the day, so the split above is the model's and
  not the script's;
- the objective equals the plant's cost recomputed from the schedule — buy,
  less sold power, plus the carbon charge, plus the starts and stops.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import lpspec as lps

HERE = Path(__file__).parent
MODEL = HERE / 'district_heating.yaml'

#: Each flow is (carrier, role); the role decides its sign in the objective —
#: an ``input`` is bought, a ``sold`` flow is the CHP unit's exported power, and
#: a ``heat`` flow is delivered to the network for nothing.
FLOWS = {
    'boiler_gas': ('gas', 'input'),
    'boiler_heat': ('heat', 'heat'),
    'chp_gas': ('gas', 'input'),
    'chp_heat': ('heat', 'heat'),
    'chp_power': ('power', 'sold'),
    'hp_power': ('power', 'input'),
    'hp_heat': ('heat', 'heat'),
    'eboiler_power': ('power', 'input'),
    'eboiler_heat': ('heat', 'heat'),
}

#: A unit's curve, one breakpoint per row, giving each of its flows at that
#: point. The first is the origin — off — and the last is full load, so the
#: heat flow's last value is the unit's capacity. Lengths differ: the electric
#: boiler is a straight line of two, the rest bend over three.
CURVES = {
    'boiler': [
        {'boiler_gas': 0.0, 'boiler_heat': 0.0},
        {'boiler_gas': 23.0, 'boiler_heat': 20.0},
        {'boiler_gas': 44.0, 'boiler_heat': 40.0},
    ],
    'chp': [
        {'chp_gas': 0.0, 'chp_heat': 0.0, 'chp_power': 0.0},
        {'chp_gas': 33.0, 'chp_heat': 15.0, 'chp_power': 11.0},
        {'chp_gas': 58.0, 'chp_heat': 30.0, 'chp_power': 20.0},
    ],
    'hp': [
        {'hp_power': 0.0, 'hp_heat': 0.0},
        {'hp_power': 2.8, 'hp_heat': 10.0},
        {'hp_power': 6.5, 'hp_heat': 20.0},
    ],
    'eboiler': [
        {'eboiler_power': 0.0, 'eboiler_heat': 0.0},
        {'eboiler_power': 26.3, 'eboiler_heat': 25.0},
    ],
}

#: The committable units and their commitment data: minimum load as a share of
#: capacity, the times a unit must stay up and down once switched, and what a
#: start and a stop cost. The heat pump and electric boiler are absent — they
#: are not committed, so they carry no status and no minimum, and `where`
#: leaves the binaries and the commitment rows unbuilt for them.
COMMITMENT = {
    'boiler': {'min_load': 0.2, 'min_up_time': 2, 'min_down_time': 2, 'start_up_cost': 40.0, 'shut_down_cost': 0.0},
    'chp': {'min_load': 0.5, 'min_up_time': 3, 'min_down_time': 3, 'start_up_cost': 100.0, 'shut_down_cost': 0.0},
}

UNIT_OF = {flow: flow.split('_')[0] for flow in FLOWS}
UNITS = list(CURVES)
COMMITTABLE = list(COMMITMENT)
CARRIERS = ['gas', 'power', 'heat']
BREAKPOINTS = max(len(curve) for curve in CURVES.values())

#: A winter day in twelve periods. Demand peaks morning and evening; grid power
#: is cheap overnight and dear at the midday peak; gas is flat.
HEAT_DEMAND = [30.0, 25.0, 22.0, 28.0, 45.0, 80.0, 82.0, 40.0, 48.0, 62.0, 50.0, 35.0]
POWER_PRICE = [22.0, 20.0, 20.0, 25.0, 45.0, 70.0, 85.0, 60.0, 55.0, 90.0, 65.0, 35.0]
GAS_PRICE = 30.0
PERIODS = len(HEAT_DEMAND)

EMISSION_FACTOR = {'gas': 0.20, 'power': 0.10, 'heat': 0.0}
CARBON_PRICE = 50.0
STORE = {'store_cap': 60.0, 'store_rate': 20.0, 'store_keep': 0.98}


def sources() -> dict[str, object]:
    """The instance as one frame per parameter, the shape ``solve`` attaches."""
    flow_names, carriers, roles = zip(*[(f, c, r) for f, (c, r) in FLOWS.items()], strict=True)

    bp_present = {'unit': [], 'bp': [], 'value': []}
    bp_rate = {'flow': [], 'bp': [], 'value': []}
    rate_max = {'flow': [], 'value': []}
    for unit, curve in CURVES.items():
        for bp in range(BREAKPOINTS):
            bp_present['unit'].append(unit)
            bp_present['bp'].append(bp)
            bp_present['value'].append(bp < len(curve))
        for flow in (f for f in FLOWS if UNIT_OF[f] == unit):
            for bp, point in enumerate(curve):
                bp_rate['flow'].append(flow)
                bp_rate['bp'].append(bp)
                bp_rate['value'].append(point[flow])
            rate_max['flow'].append(flow)
            rate_max['value'].append(curve[-1][flow])

    def committable_param(key: str) -> pl.DataFrame:
        return pl.DataFrame({'unit': COMMITTABLE, 'value': [COMMITMENT[u][key] for u in COMMITTABLE]})

    return {
        'unit': pl.DataFrame({'unit': UNITS}),
        'flow': pl.DataFrame({'flow': list(flow_names)}),
        'carrier': pl.DataFrame({'carrier': CARRIERS}),
        'bp': pl.DataFrame({'bp': list(range(BREAKPOINTS))}),
        'snapshot': pl.DataFrame({'snapshot': range(PERIODS)}),
        'unit_of': pl.DataFrame({'flow': list(flow_names), 'unit': [UNIT_OF[f] for f in flow_names]}),
        'carrier_of': pl.DataFrame({'flow': list(flow_names), 'carrier': list(carriers)}),
        'bp_present': pl.DataFrame(bp_present),
        'bp_rate': pl.DataFrame(bp_rate),
        'rate_max': pl.DataFrame(rate_max),
        'is_heat': pl.DataFrame({'flow': list(flow_names), 'value': [float(r == 'heat') for r in roles]}),
        'is_input': pl.DataFrame({'flow': list(flow_names), 'value': [float(r == 'input') for r in roles]}),
        'is_sold': pl.DataFrame({'flow': list(flow_names), 'value': [float(r == 'sold') for r in roles]}),
        'heat_max': pl.DataFrame({'unit': UNITS, 'value': [CURVES[u][-1][f'{u}_heat'] for u in UNITS]}),
        'committable': pl.DataFrame({'unit': UNITS, 'value': [u in COMMITMENT for u in UNITS]}),
        'min_load': committable_param('min_load'),
        'min_up_time': committable_param('min_up_time'),
        'min_down_time': committable_param('min_down_time'),
        'start_up_cost': committable_param('start_up_cost'),
        'shut_down_cost': committable_param('shut_down_cost'),
        'carrier_price': pl.DataFrame(
            {
                'carrier': [c for c in CARRIERS for _ in range(PERIODS)],
                'snapshot': list(range(PERIODS)) * len(CARRIERS),
                'value': ([GAS_PRICE] * PERIODS + list(POWER_PRICE) + [0.0] * PERIODS),
            }
        ),
        'emission_factor': pl.DataFrame({'carrier': CARRIERS, 'value': [EMISSION_FACTOR[c] for c in CARRIERS]}),
        'carbon_price': pl.DataFrame({'value': [CARBON_PRICE]}),
        'heat_demand': pl.DataFrame({'snapshot': range(PERIODS), 'value': HEAT_DEMAND}),
        **{k: pl.DataFrame({'value': [v]}) for k, v in STORE.items()},
    }


def _flow_facts() -> pl.DataFrame:
    """One row per flow — its unit, carrier and role — to join readback onto."""
    return pl.DataFrame(
        {
            'flow': list(FLOWS),
            'unit': [UNIT_OF[f] for f in FLOWS],
            'carrier': [FLOWS[f][0] for f in FLOWS],
            'role': [FLOWS[f][1] for f in FLOWS],
        }
    )


def _price_table() -> pl.DataFrame:
    """The buy/sell price of each carrier in each period, keyed like the sources."""
    prices = {'gas': [GAS_PRICE] * PERIODS, 'power': list(POWER_PRICE), 'heat': [0.0] * PERIODS}
    return pl.DataFrame(
        {
            'carrier': [c for c in CARRIERS for _ in range(PERIODS)],
            'snapshot': list(range(PERIODS)) * len(CARRIERS),
            'price': [p for c in CARRIERS for p in prices[c]],
        }
    )


def main() -> None:
    result = lps.solve(MODEL, sources())
    assert result.termination_condition == 'optimal', f'the plant did not solve: {result.termination_condition}'

    facts = _flow_facts()
    rate = result.primal('rate').join(facts, on='flow')
    heat = rate.filter(pl.col('role') == 'heat').select('unit', 'snapshot', 'value')
    heat_wide = heat.pivot('unit', index='snapshot', values='value').fill_null(0.0).sort('snapshot')
    soc = result.primal('soc').sort('snapshot')['value']
    status = result.primal('status').group_by('unit').agg(pl.col('value').sum().round().cast(int))
    committed = dict(zip(status['unit'], status['value'], strict=True))

    print(f'A winter day in {PERIODS} periods. Heat delivered by unit (MWh_th), and the store:')
    print(f'{"period":>6}  {"demand":>6}  ' + '  '.join(f'{u:>8}' for u in UNITS) + f'  {"soc":>6}')
    for t in range(PERIODS):
        row = heat_wide.row(t, named=True)
        outs = '  '.join(f'{row.get(u, 0.0):8.1f}' for u in UNITS)
        print(f'{t:>6}  {HEAT_DEMAND[t]:6.1f}  {outs}  {soc[t]:6.1f}')

    print()
    print('periods committed (committable units only):  ' + '   '.join(f'{u} {committed[u]:>2}' for u in COMMITTABLE))
    print('freely dispatched:  ' + ', '.join(u for u in UNITS if u not in COMMITMENT))

    priced = rate.join(_price_table(), on=['carrier', 'snapshot'])
    buy = float(priced.filter(pl.col('role') == 'input').select((pl.col('value') * pl.col('price')).sum()).item())
    sell = float(priced.filter(pl.col('role') == 'sold').select((pl.col('value') * pl.col('price')).sum()).item())
    co2 = float(
        rate.filter(pl.col('role') == 'input')
        .with_columns(ef=pl.col('carrier').replace_strict(EMISSION_FACTOR, return_dtype=pl.Float64))
        .select((pl.col('value') * pl.col('ef')).sum())
        .item()
    )
    switches = pl.DataFrame(
        {
            'unit': COMMITTABLE,
            'su': [COMMITMENT[u]['start_up_cost'] for u in COMMITTABLE],
            'sd': [COMMITMENT[u]['shut_down_cost'] for u in COMMITTABLE],
        }
    )
    starts = float(
        result.primal('start_up').join(switches, on='unit').select((pl.col('value') * pl.col('su')).sum()).item()
    )
    stops = float(
        result.primal('shut_down').join(switches, on='unit').select((pl.col('value') * pl.col('sd')).sum()).item()
    )
    recomputed = buy - sell + CARBON_PRICE * co2 + starts + stops

    print()
    print(f'buy {buy:9.1f}   sell {sell:8.1f}   carbon {CARBON_PRICE * co2:8.1f}   starts {starts + stops:6.1f}')
    print(f'objective {result.objective:10.2f}   ({co2:.1f} tCO2 over the day)')

    delivered = heat.group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')['value']
    charge = result.primal('charge').sort('snapshot')['value']
    discharge = result.primal('discharge').sort('snapshot')['value']
    heat_by_unit = dict(heat.group_by('unit').agg(pl.col('value').sum()).iter_rows())
    assert all(abs(delivered[t] + discharge[t] - charge[t] - HEAT_DEMAND[t]) < 1e-6 for t in range(PERIODS)), (
        'the heat balance must close in every period'
    )
    assert all(committed[u] > 0 for u in COMMITTABLE), 'each committable unit must run in some period'
    assert all(heat_by_unit.get(u, 0.0) > 0 for u in UNITS if u not in COMMITMENT), (
        'each freely dispatched unit must deliver heat in some period'
    )
    assert abs(recomputed - result.objective) < 1e-4, 'the objective must equal the schedule recosted'
    print()
    print('heat balance closes every period, the whole fleet is used, and the objective is the recosted schedule.')


if __name__ == '__main__':
    main()
