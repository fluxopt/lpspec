# district_heating

A heating plant that meets one network's heat from four committed units — a
boiler, a CHP unit, a heat pump and an electric boiler — each on its own
conversion curve.

## The problem

A heating plant meets a heat demand every period. Each unit converts a bought
carrier into heat along one [piecewise conversion](piecewise_conversion.md)
curve, and the CHP unit turns gas into both heat and the power it sells. The
curve is where the shapes live: the boiler's part-load efficiency, the CHP's
heat-to-power ratio, and the heat pump's coefficient of performance (COP, heat
delivered per unit of electricity drawn). None of them is a clause in the file.

On top of the curve, the two burners are committed. A binary status per period
says whether a unit runs, a minimum load says how low it may run when it does,
and a start-up charge and a minimum time up and down bind its schedule across
periods — the [unit commitment](pypsa_unit_commitment.md) and
[minimum up and down times](pypsa_min_up_down.md) shapes.

Commitment is opt-in. The heat pump and electric boiler are fast and carry no
minimum, so they are not committed: `where: committable` leaves them out of the
status variable and every commitment row, and their curve alone bounds them
between zero and capacity.

The two compose in one row. The status bounds the unit's heat output, and the
curve maps that output to the fuel and power it takes and makes:

```yaml
unit_heat - heat_max * status <= 0
```

An uncommitted unit is pinned to zero heat, which the curve reads back as its
origin — no fuel, no power. A committed one is free above its minimum load, and
the curve prices every point on the way up.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost operation of a district-heating plant: a gas boiler, a CHP unit, a heat pump and an electric boiler feed one heat network with a store. The two burners are committed on or off with a minimum load, a start-up charge and a minimum time up and down; the electric units carry none of that and dispatch freely, which `where: committable` decides. What a unit consumes and produces is one shared curve per unit — a boiler tying gas to heat, the CHP tying gas to heat and power, the heat pump tying electricity to heat at its coefficient of performance — so the number of flows a unit has, and the shape of its efficiency, are data. Heat sells at nothing, gas and grid electricity are bought, CHP power is sold, and every unit of gas burned is charged a carbon price.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` (`int` coordinates) — dispatch periods in order, cyclic at the horizon for the store |
| $`\mathcal{U}`$ | index $`u`$ — `unit` — the converting units, some committed on or off and some freely dispatched |
| $`\mathcal{F}`$ | index $`f`$ — `flow` with $`\mathrm{unit\_of}: \mathcal{F} \to \mathcal{U},\ \mathrm{carrier\_of}: \mathcal{F} \to \mathcal{C}`$ — a unit's inputs and outputs, one row each |
| $`\mathcal{C}`$ | index $`c`$ — `carrier` — what a flow carries — gas, grid electricity or heat |
| $`\mathcal{B}`$ | index $`b`$ — `bp` — breakpoints of a unit's curve, as many as the longest curve needs |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{bp\_rate}`$ | `bp_rate` over $`\mathcal{F} \times \mathcal{B}`$ — what each flow runs at, at each breakpoint of its unit's curve |
| $`\mathrm{bp\_present}`$ | `bp_present` over $`\mathcal{U} \times \mathcal{B}`$ — how far each unit's curve runs, so a shorter curve pads nothing |
| $`\mathrm{rate}^{\mathrm{max}}`$ | `rate_max` over $`\mathcal{F}`$ — what each flow runs at when its unit is at its last breakpoint |
| $`\mathrm{is\_heat}`$ | `is_heat` over $`\mathcal{F}`$ — which flows deliver heat to the network |
| $`\mathrm{is\_input}`$ | `is_input` over $`\mathcal{F}`$ — which flows are bought — gas into a burner, electricity into a heat pump |
| $`\mathrm{is\_sold}`$ | `is_sold` over $`\mathcal{F}`$ — which flows are sold back — the CHP unit's power export |
| $`\mathrm{heat\_max}`$ | `heat_max` over $`\mathcal{U}`$ — a unit's heat output at full load, the size its commitment switches |
| $`\mathrm{committable}`$ | `committable` over $`\mathcal{U}`$ — which units are committed on or off rather than freely dispatched — the thermal burners, whose minimum load and switching costs matter, and not the electric units, which follow the price from zero up |
| $`\mathrm{min\_load}`$ | `min_load` over $`\mathcal{U}`$ — the share of its heat output a committed unit must run at least at |
| $`\mathrm{min\_up\_time}`$ | `min_up_time` over $`\mathcal{U}`$ — how many periods a unit must stay on once it has started |
| $`\mathrm{min\_down\_time}`$ | `min_down_time` over $`\mathcal{U}`$ — how many periods a unit must stay off once it has stopped |
| $`\mathrm{start\_up\_cost}`$ | `start_up_cost` over $`\mathcal{U}`$ — what bringing a unit up costs, once per start |
| $`\mathrm{shut\_down\_cost}`$ | `shut_down_cost` over $`\mathcal{U}`$ — what taking a unit down costs, once per stop |
| $`\mathrm{carrier\_price}`$ | `carrier_price` over $`\mathcal{C} \times \mathcal{T}`$ — what a unit of each carrier costs to buy or earns to sell, per period |
| $`\mathrm{emission\_factor}`$ | `emission_factor` over $`\mathcal{C}`$ — carbon emitted per unit of each carrier bought |
| $`\mathrm{carbon\_price}`$ | `carbon_price` (scalar) — what a unit of emitted carbon is charged |
| $`\mathrm{heat\_demand}`$ | `heat_demand` over $`\mathcal{T}`$ — heat the network must be supplied with |
| $`\mathrm{store\_cap}`$ | `store_cap` (scalar) — most the heat store may hold |
| $`\mathrm{store\_rate}`$ | `store_rate` (scalar) — most the store may take in or give back in one period |
| $`\mathrm{store\_keep}`$ | `store_keep` (scalar) — the share of a period's stored heat still there the next period |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{rate}`$ | `rate` over $`\mathcal{F} \times \mathcal{T}`$ — what each flow runs at |
| $`\mathit{weight}`$ | `weight` over $`\mathcal{U} \times \mathcal{T} \times \mathcal{B}`$ — how much of each breakpoint a unit's operating point is made of — one convex combination per unit and period, over the breakpoints its own curve runs to |
| $`\mathit{status}`$ | `status` over $`\mathcal{U} \times \mathcal{T}`$ — is this unit committed in this period? Declared only for committable units |
| $`\mathit{start\_up}`$ | `start_up` over $`\mathcal{U} \times \mathcal{T}`$ — does this unit come up entering this period? |
| $`\mathit{shut\_down}`$ | `shut_down` over $`\mathcal{U} \times \mathcal{T}`$ — does this unit go down entering this period? |
| $`\mathit{charge}`$ | `charge` over $`\mathcal{T}`$ — heat taken into the store |
| $`\mathit{discharge}`$ | `discharge` over $`\mathcal{T}`$ — heat given back by the store |
| $`\mathit{soc}`$ | `soc` over $`\mathcal{T}`$ — heat held in the store at the end of a period |

#### Definitions

| Symbol | Meaning |
|---|---|
| $`\mathit{unit\_heat}`$ | `unit_heat` over $`\mathcal{T} \times \mathcal{U}`$ — the heat a unit puts out in a period, its committed quantity |

Upright is what the model is given — a parameter such as $`\mathrm{bp\_rate}`$, a coordinate map, a label — and italic is what the solver chooses, such as $`\mathit{rate}`$. An index is italic too, being what a quantifier chooses, and a set is script.

$`t \ominus k`$ denotes cyclic translation: index $`t-k`$ taken modulo the size of the dimension (`roll`). Plain $`t-k`$ (`shift`) has no wraparound — terms translated past the edge are simply absent.

$`t \boxminus_{v} k`$ denotes translation with $`v`$ standing where index $`t-k`$ leaves the dimension (`shift(edge=v)`), so the row at that boundary is built and carries $`v`$ rather than being dropped.

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ f \in \mathcal{F}} \mathit{rate}_{f,t} \cdot \mathrm{is\_input}_{f} \cdot \mathrm{carrier\_price}_{\mathrm{carrier\_of}(f),t} - \left( \sum_{t \in \mathcal{T},\ f \in \mathcal{F}} \mathit{rate}_{f,t} \cdot \mathrm{is\_sold}_{f} \cdot \mathrm{carrier\_price}_{\mathrm{carrier\_of}(f),t} \right) + \mathrm{carbon\_price} \cdot \left( \sum_{t \in \mathcal{T},\ f \in \mathcal{F}} \mathit{rate}_{f,t} \cdot \mathrm{is\_input}_{f} \cdot \mathrm{emission\_factor}_{\mathrm{carrier\_of}(f)} \right) + \sum_{t \in \mathcal{T},\ u \in \mathcal{U}} \mathit{start\_up}_{u,t} \cdot \mathrm{start\_up\_cost}_{u} + \sum_{t \in \mathcal{T},\ u \in \mathcal{U}} \mathit{shut\_down}_{u,t} \cdot \mathrm{shut\_down\_cost}_{u}
```

#### Subject to

**`one_operating_point`**

```math
\sum_{b \in \mathcal{B}} \mathit{weight}_{u,t,b} = 1 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T}
```

**`on_the_curve`**

```math
\mathit{rate}_{f,t} = \sum_{b \in \mathcal{B}} \mathit{weight}_{\mathrm{unit\_of}(f),t,b} \cdot \mathrm{bp\_rate}_{f,b} \qquad \forall\, f \in \mathcal{F},\ t \in \mathcal{T}
```

**`commitment_max`**

```math
\mathit{unit\_heat}_{t,u} - \mathrm{heat\_max}_{u} \cdot \mathit{status}_{u,t} \le 0 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u}
```

**`commitment_min`**

```math
\mathit{unit\_heat}_{t,u} - \mathrm{min\_load}_{u} \cdot \mathrm{heat\_max}_{u} \cdot \mathit{status}_{u,t} \ge 0 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u}
```

**`start_up`**

```math
\mathit{start\_up}_{u,t} - \mathit{status}_{u,t} + \mathit{status}_{u,t \boxminus_{0} 1} \ge 0 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u}
```

**`shut_down`**

```math
\mathit{shut\_down}_{u,t} + \mathit{status}_{u,t} - \mathit{status}_{u,t - 1} \ge 0 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u}
```

**`min_up_time`**

```math
\sum_{t' \in \mathcal{T} \,:\, 0 \le t - t' < \mathrm{min\_up\_time}} \mathit{start\_up}_{u,t'} \le \mathit{status}_{u,t} \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u} \wedge t > 0
```

**`min_down_time`**

```math
\mathit{status}_{u,t} + \sum_{t' \in \mathcal{T} \,:\, 0 \le t - t' < \mathrm{min\_down\_time}} \mathit{shut\_down}_{u,t'} \le 1 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u} \wedge t > 0
```

**`heat_balance`**

```math
\sum_{f \in \mathcal{F}} \mathit{rate}_{f,t} \cdot \mathrm{is\_heat}_{f} + \mathit{discharge}_{t} - \mathit{charge}_{t} = \mathrm{heat\_demand}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`store_balance`**

```math
\mathit{soc}_{t} = \mathit{soc}_{t \ominus 1} \cdot \mathrm{store\_keep} + \mathit{charge}_{t} - \mathit{discharge}_{t} \qquad \forall\, t \in \mathcal{T}
```

#### Definitions

**`unit_heat`**

```math
\mathit{unit\_heat}_{t,u} = \sum_{f \in \mathcal{F} \,:\, \mathrm{unit\_of}(f) = u} \mathit{rate}_{f,t} \cdot \mathrm{is\_heat}_{f} \qquad \forall\, t \in \mathcal{T},\ u \in \mathcal{U}
```

#### Variable domains

**`rate`**

```math
0 \le \mathit{rate}_{f,t} \le \mathrm{rate}^{\mathrm{max}}_{f} \qquad \forall\, f \in \mathcal{F},\ t \in \mathcal{T}
```

**`weight`**

```math
0 \le \mathit{weight}_{u,t,b} \le 1 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T},\ b \in \mathcal{B} \,:\, \mathrm{bp\_present}_{u,b}
```

**`weight sos`**

```math
\left( \mathit{weight}_{u,t,b} \right)_{b \in \mathcal{B}} \in \mathrm{SOS}2 \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T}
```

**`status`**

```math
\mathit{status}_{u,t} \in \{0, 1\} \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u}
```

**`start_up`**

```math
\mathit{start\_up}_{u,t} \in \{0, 1\} \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u}
```

**`shut_down`**

```math
\mathit{shut\_down}_{u,t} \in \{0, 1\} \qquad \forall\, u \in \mathcal{U},\ t \in \mathcal{T} \,:\, \mathrm{committable}_{u}
```

**`charge`**

```math
0 \le \mathit{charge}_{t} \le \mathrm{store\_rate} \qquad \forall\, t \in \mathcal{T}
```

**`discharge`**

```math
0 \le \mathit{discharge}_{t} \le \mathrm{store\_rate} \qquad \forall\, t \in \mathcal{T}
```

**`soc`**

```math
0 \le \mathit{soc}_{t} \le \mathrm{store\_cap} \qquad \forall\, t \in \mathcal{T}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per
parameter.

=== "lpspec"

    ```yaml
    description: >-
      Least-cost operation of a district-heating plant: a gas boiler, a CHP unit, a
      heat pump and an electric boiler feed one heat network with a store. The two
      burners are committed on or off with a minimum load, a start-up charge and a
      minimum time up and down; the electric units carry none of that and dispatch
      freely, which `where: committable` decides. What a unit consumes and produces
      is one shared curve per unit —
      a boiler tying gas to heat, the CHP tying gas to heat and power, the heat pump
      tying electricity to heat at its coefficient of performance — so the number of
      flows a unit has, and the shape of its efficiency, are data. Heat sells at
      nothing, gas and grid electricity are bought, CHP power is sold, and every
      unit of gas burned is charged a carbon price.

    dimensions:
      snapshot:
        description: dispatch periods in order, cyclic at the horizon for the store
        dtype: int
      unit:
        description: the converting units, some committed on or off and some freely dispatched
        dtype: str
      flow:
        description: a unit's inputs and outputs, one row each
        dtype: str
      carrier:
        description: what a flow carries — gas, grid electricity or heat
        dtype: str
      bp:
        description: breakpoints of a unit's curve, as many as the longest curve needs
        dtype: int

    lookups:
      unit_of:
        description: which unit a flow belongs to
        over: flow
        into: unit
      carrier_of:
        description: which carrier a flow moves
        over: flow
        into: carrier

    parameters:
      bp_rate:
        description: what each flow runs at, at each breakpoint of its unit's curve
        dims: [flow, bp]
      bp_present:
        description: how far each unit's curve runs, so a shorter curve pads nothing
        dims: [unit, bp]
        dtype: bool
      rate_max:
        description: what each flow runs at when its unit is at its last breakpoint
        dims: [flow]
      is_heat:
        description: which flows deliver heat to the network
        dims: [flow]
      is_input:
        description: which flows are bought — gas into a burner, electricity into a heat pump
        dims: [flow]
      is_sold:
        description: which flows are sold back — the CHP unit's power export
        dims: [flow]
      heat_max:
        description: a unit's heat output at full load, the size its commitment switches
        dims: [unit]
      committable:
        description: >-
          which units are committed on or off rather than freely dispatched — the
          thermal burners, whose minimum load and switching costs matter, and not the
          electric units, which follow the price from zero up
        dims: [unit]
        dtype: bool
      min_load:
        description: the share of its heat output a committed unit must run at least at
        dims: [unit]
      min_up_time:
        description: how many periods a unit must stay on once it has started
        dims: [unit]
        dtype: int
      min_down_time:
        description: how many periods a unit must stay off once it has stopped
        dims: [unit]
        dtype: int
      start_up_cost:
        description: what bringing a unit up costs, once per start
        dims: [unit]
      shut_down_cost:
        description: what taking a unit down costs, once per stop
        dims: [unit]
      carrier_price:
        description: what a unit of each carrier costs to buy or earns to sell, per period
        dims: [carrier, snapshot]
      emission_factor:
        description: carbon emitted per unit of each carrier bought
        dims: [carrier]
      carbon_price:
        description: what a unit of emitted carbon is charged
        dims: []
      heat_demand:
        description: heat the network must be supplied with
        dims: [snapshot]
      store_cap:
        description: most the heat store may hold
        dims: []
      store_rate:
        description: most the store may take in or give back in one period
        dims: []
      store_keep:
        description: the share of a period's stored heat still there the next period
        dims: []

    variables:
      rate:
        description: what each flow runs at
        foreach: [flow, snapshot]
        bounds:
          lower: 0
          upper: rate_max
      weight:
        description: >-
          how much of each breakpoint a unit's operating point is made of — one
          convex combination per unit and period, over the breakpoints its own curve
          runs to
        foreach: [unit, snapshot, bp]
        where: bp_present
        bounds:
          lower: 0
          upper: 1
      status:
        description: is this unit committed in this period? Declared only for committable units
        foreach: [unit, snapshot]
        where: committable
        domain: binary
      start_up:
        description: does this unit come up entering this period?
        foreach: [unit, snapshot]
        where: committable
        domain: binary
      shut_down:
        description: does this unit go down entering this period?
        foreach: [unit, snapshot]
        where: committable
        domain: binary
      charge:
        description: heat taken into the store
        foreach: [snapshot]
        bounds:
          lower: 0
          upper: store_rate
      discharge:
        description: heat given back by the store
        foreach: [snapshot]
        bounds:
          lower: 0
          upper: store_rate
      soc:
        description: heat held in the store at the end of a period
        foreach: [snapshot]
        bounds:
          lower: 0
          upper: store_cap

    sos:
      on_one_segment:
        description: >-
          at most two of a unit's weights, and those two neighbours — which puts the
          operating point on a segment of its curve rather than anywhere in the hull
        variable: weight
        over: bp
        type: 2
        big_m: 1

    expressions:
      unit_heat:
        description: the heat a unit puts out in a period, its committed quantity
        expression: sum(rate * is_heat, by=unit_of)

    constraints:
      one_operating_point:
        description: each unit sits somewhere on its curve, in every period
        foreach: [unit, snapshot]
        expression: sum(weight, over=bp) == 1
      on_the_curve:
        description: every flow reads its own value off its unit's weights
        foreach: [flow, snapshot]
        expression: rate == sum(at(weight, by=unit_of) * bp_rate, over=bp)
      commitment_max:
        description: >-
          a committed unit puts out no more than its full-load heat and an
          uncommitted period is pinned to zero — capacity times status is a parameter
          against a variable, so the product stays degree 1. A unit that is not
          committable builds no such row and is capped by its curve instead
        foreach: [unit, snapshot]
        where: committable
        expression: unit_heat - heat_max * status <= 0
      commitment_min:
        description: a committed unit runs at no less than its minimum load, an off period at zero
        foreach: [unit, snapshot]
        where: committable
        expression: unit_heat - min_load * heat_max * status >= 0
      start_up:
        description: >-
          a committable unit whose status rises entering this period pays for a start.
          It begins the horizon off, which is the 0 the first period reads where it
          has no predecessor
        foreach: [unit, snapshot]
        where: committable
        expression: start_up - status + shift(status, over=snapshot, offset=1, edge=0) >= 0
      shut_down:
        description: a committable unit whose status falls entering this period pays for a stop
        foreach: [unit, snapshot]
        where: committable
        expression: shut_down + status - shift(status, over=snapshot, offset=1) >= 0
      min_up_time:
        description: >-
          over the last `min_up_time` periods a unit may have started at most as often
          as it is running now, which stops it starting and stopping inside its window
        foreach: [unit, snapshot]
        where: "committable AND snapshot > 0"
        expression: sum_back(start_up, over=snapshot, within=min_up_time) <= status
      min_down_time:
        description: the mirror — having stopped inside the window and running now cannot both hold
        foreach: [unit, snapshot]
        where: "committable AND snapshot > 0"
        expression: status + sum_back(shut_down, over=snapshot, within=min_down_time) <= 1
      heat_balance:
        description: what the units put out, plus what the store gives back net of charging, meets the demand
        foreach: [snapshot]
        expression: sum(rate * is_heat, over=flow) + discharge - charge == heat_demand
      store_balance:
        description: >-
          the heat held at the end of a period is what was held the period before,
          kept net of standing loss, plus what was charged and less what was taken —
          and it wraps at the horizon, so the first period inherits from the last
        foreach: [snapshot]
        expression: soc == shift(soc, over=snapshot, offset=1, edge='wrap') * store_keep + charge - discharge

    objective:
      sense: minimize
      description: >-
        what the bought carriers cost, less what sold power earns, plus the carbon
        charge on what was burned, plus what the starts and stops cost
      expression: >-
        sum(rate * is_input * at(carrier_price, by=carrier_of))
        - sum(rate * is_sold * at(carrier_price, by=carrier_of))
        + carbon_price * sum(rate * is_input * at(emission_factor, by=carrier_of))
        + sum(start_up * start_up_cost)
        + sum(shut_down * shut_down_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/district_heating.yaml', sources) as solution:
        solution.objective  # 14228.04
        solution.primal('status')  # which units are committed, per period
        solution.primal('rate')  # what every flow runs at
    ```

## What it exercises

**Commitment rides on a conversion curve.** The binary status decides whether a
unit runs and its minimum load how hard; the shared curve decides what running
costs, in fuel bought and power sold. Neither construct knows about the other —
`unit_heat` is the one expression that ties them, and it is the unit's heat read
off the same flows the balance sums.

**Commitment is per unit, through `where`.** `where: committable` on the status
variable and on every commitment row builds them for the two burners and not the
electric units. An uncommitted unit has no status and no minimum: it is one
`where` clause away from a committed one, not a second kind of unit.

**The fleet is data.** Nothing in the file says there are four units, or that
the CHP has three flows and the boiler two. The `unit_of` and `carrier_of`
lookups say it, the curves are rows keyed by flow and breakpoint, and a fifth
unit type is more rows rather than a new declaration.

**Carbon is one term.** Gas and grid power each carry an emission factor, and
the objective charges the carbon on what is bought at a price. Raising that
price reorders the merit order without touching a constraint.

The model is mixed-integer, so it has no dual solution: there is no marginal
heat price to read back, only the committed schedule and its cost. A model that
needs the price relaxes the status to continuous.

---

[`examples/district_heating.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/district_heating.yaml)
· [`examples/district_heating.py`](https://github.com/fluxopt/lpspec/blob/main/examples/district_heating.py)
· back to [all models](index.md)
