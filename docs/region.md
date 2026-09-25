# See the feasible region

A solve answers one question — what *should* the plant do — and a wrong model
answers it confidently. The question that catches a wrong model is the other
one: what *can* the plant do? Its feasible set has one dimension per column,
which nothing draws, but its shadow on two quantities you can name is a
polygon, and for a plant that polygon is the operating chart every engineer
already knows — heat against power.

`sps.project` traces that polygon by solving the model along a sequence of
directions: each vertex is one solve, on the fast path, and an edge is kept
only once a solve along its outward normal finds nothing beyond it. So the
polygon is exact rather than sampled, and its edges are the constraints that
bound the plant. This page draws one for a small CHP plant, hour by hour,
state by state, and then breaks the model on purpose to show what the region
says about it.

Every block on this page runs when the site is built, and what you see under it
is what it printed on this commit. A block that raises fails the build.

```python exec="true" source="material-block" session="region"
import polars as pl
import yaml
from mathspec import to_markdown

import specsolve as sps

HOURS = range(4)
UNITS = ['chp', 'boiler', 'peaker']


def show(figure):
    print(figure.to_html(full_html=False, include_plotlyjs='cdn'))
```

`show` is for this page: the site runs each block without a browser, so a
figure is printed as the HTML the page shows, with plotly's script loaded from
its CDN. In a session of your own, `region.plot()` displays as it is.

## The plant

One gas well feeds three units: a CHP engine that makes heat *and* power, a
boiler that makes only heat, a peaker that makes only power. Each unit is on
or off, and on means at least its minimum load. The plant must cover a heat
load and a power load every hour, and may over-supply either — heat can be
dumped and power exported — which is what makes the region two-dimensional
rather than a point.

The two quantities the region is drawn on are declared as named expressions,
`heat` and `power`, over the hours. That is the whole of what `project` needs
from the file: two names, and nothing about the objective.

```python exec="true" source="material-block" session="region"
PLANT = yaml.safe_load("""
dimensions:
  t: {dtype: int, description: hours}
  unit: {dtype: str, description: the three conversions on the gas well}

parameters:
  well: {dims: [], description: what the gas well delivers in an hour}
  gas_price: {dims: []}
  p_nom: {dims: [unit], description: the most gas a unit burns in an hour}
  min_load: {dims: [unit], description: the share of p_nom a unit burns at least while on}
  to_heat: {dims: [unit], description: heat per unit of gas}
  to_power: {dims: [unit], description: power per unit of gas}
  heat_load: {dims: [t]}
  power_load: {dims: [t]}

variables:
  gas:
    description: gas burnt by a unit in an hour
    dims: [t, unit]
    bounds: {lower: 0}
  running:
    description: whether the unit runs in the hour
    dims: [t, unit]
    domain: binary

expressions:
  heat: sum(gas * to_heat, over=unit)
  power: sum(gas * to_power, over=unit)

constraints:
  capacity:
    dims: [t, unit]
    expression: gas <= p_nom * running
  minimum_load:
    dims: [t, unit]
    expression: gas >= min_load * p_nom * running
  the_well:
    dims: [t]
    expression: sum(gas, over=unit) <= well
  heat_demand:
    dims: [t]
    expression: heat >= heat_load
  power_demand:
    dims: [t]
    expression: power >= power_load

objective:
  sense: minimize
  expression: sum(gas) * gas_price
""")

sources = {
    't': pl.DataFrame({'t': HOURS}),
    'unit': pl.DataFrame({'unit': UNITS}),
    'well': 200.0,
    'gas_price': 10.0,
    'p_nom': pl.DataFrame({'unit': UNITS, 'value': [120.0, 100.0, 100.0]}),
    'min_load': pl.DataFrame({'unit': UNITS, 'value': [0.4, 0.2, 0.3]}),
    'to_heat': pl.DataFrame({'unit': UNITS, 'value': [0.4, 0.8, 0.0]}),
    'to_power': pl.DataFrame({'unit': UNITS, 'value': [0.4, 0.0, 0.5]}),
    'heat_load': pl.DataFrame({'t': HOURS, 'value': [36.0, 60.0, 80.0, 40.0]}),
    'power_load': pl.DataFrame({'t': HOURS, 'value': [40.0, 55.0, 64.0, 30.0]}),
}

print(to_markdown(PLANT, legend=False, numbered=False))
```

## One hour, binaries free

`at={'t': 0}` reads both quantities in the first hour. Every other dim a
quantity carries would be summed, and these carry only `t`, so the axes are
the plant's heat and power in that hour. The default leaves the binaries free,
and the polygon that comes back is what the solver can reach with any
combination of units on or off.

A region is four frames, and every one keeps its schema whatever was asked:
`vertices`, `pieces`, `edges` and `optimum`, all keyed by `piece`. With the
binaries free there is one piece, numbered `0`.

```python exec="true" source="material-block" result="text" session="region"
free = sps.project(PLANT, sources, x='heat', y='power', at={'t': 0})
print(free.vertices)
```

Six vertices, counter-clockwise from the lowest-leftmost, each one a solve.
The picture is one call, and the model's own optimum is already on it: the
region does not depend on the objective, but where the objective lands in the
region is the first thing to look at.

```python exec="true" source="material-block" html="true" session="region"
show(free.plot(name='binaries free'))
```

Hover a vertex for its coordinates, and the middle of an edge for what bounds
it. The frames behind both are next.

```python exec="true" source="material-block" result="text" session="region"
print(free.optimum)
```

The optimum sits on the floor, and not in the corner. The power load binds,
and the cheapest way to make `40` of power is the CHP alone — which makes
`40` of heat with it, four more than the hour asks for. The CHP's ratio sets
the heat, not the load, and the four are dumped. That is the kind of fact a
solve reports as a number and the region shows as a distance from a wall.

The rest of the polygon is what the plant *could* do at a price, and every
edge of it is a constraint the plant ran into. Which one is not a matter of
reading the data by hand: each vertex was a solve, so the bounds and rows the
solver sat on at both ends of an edge are known, and `edges` names them. Edge
`i` runs from vertex `i` to the next.

```python exec="true" source="material-block" result="text" session="region"
with pl.Config(tbl_rows=-1):
    print(free.edges)
```

Read against the picture, counter-clockwise from the lower-left corner:

- edge `0`, the floor, is `power_demand` at its lower side — the load;
- edge `1`, the steep one at the right, is `the_well` with the peaker off:
  gas moves from the boiler to the CHP and the well stays spent;
- edge `2`, the long one, is `the_well` with the CHP at its `capacity`: gas
  moves from the boiler to the peaker;
- edge `3` is `the_well` a third time, with the boiler off: CHP to peaker.
  Each pair of units trading gas gives its edge a slope of its own, which
  is why one well makes three edges;
- edge `4`, the short one at the top left, is the peaker's `capacity`;
- edge `5`, the left wall, is `heat_demand` at its lower side.

A unit that is off shows three times — its `gas` on its lower bound, its
`capacity` and its `minimum_load` both at zero — because all three hold at
zero and the frame reports every one the solver sat on.

A row named there that is not in your head is the whole point of drawing the
region — and a wall you expected that is missing, such as a unit's cap that
never appears, is a constraint the model does not actually hold.

## Every hour

The loads move hour to hour, and with them the floor and the left wall. A
loop over `at` is a loop over traces, and `vertices` is already the long
form — one row per vertex, in polygon order — so a stack of hours is one frame
with an `hour` column prepended, the way a sweep's answers stack.

```python exec="true" source="material-block" result="text" session="region"
by_hour = {hour: sps.project(PLANT, sources, x='heat', y='power', at={'t': hour}) for hour in HOURS}

stacked = pl.concat(region.vertices.select(pl.lit(hour).alias('hour'), pl.all()) for hour, region in by_hour.items())
print(stacked)
```

```python exec="true" source="material-block" html="true" session="region"
figure = None
for hour, region in by_hour.items():
    figure = region.plot(figure, optimum=False, name=f'hour {hour}')
show(figure)
```

The right wall, the ceiling and the well's edges never move — they are the
plant — while the two load walls walk through the polygon. Hour 2 is the
tight one: `80` of heat and `64` of power leave a sliver, and a load a little
higher in either would leave nothing. That sliver is the number a modeller
wants before the solve reports `infeasible` with no further comment.

The same frame answers it as a table: the extent each hour leaves on each axis.

```python exec="true" source="material-block" result="text" session="region"
print(
    stacked.group_by('hour', maintain_order=True).agg(
        pl.col('heat').min().alias('heat_min'),
        pl.col('heat').max().alias('heat_max'),
        pl.col('power').min().alias('power_min'),
        pl.col('power').max().alias('power_max'),
        pl.len().alias('vertices'),
    )
)
```

## Every state

A binary makes the region a union of polygons, one per combination of the
units' states, and a solve along a direction only ever finds a vertex of the
hull of that union. So the polygon above is exact as a hull and blind to what
is inside it: the strip a unit's minimum load forbids, the corner only
reachable with a unit off.

`binaries='each'` pins every combination of the binary columns `at` reaches
— the three units' states in hour 0 — and traces the region each leaves as a
piece of its own. A combination that cannot meet the loads is left out rather
than reported. The `pieces` frame says what each piece pinned, one row per
binary column, the coordinate as typed columns rather than a name to parse.

```python exec="true" source="material-block" result="text" session="region"
each = sps.project(PLANT, sources, x='heat', y='power', at={'t': 0}, binaries='each')

print(f'{each.pieces["piece"].n_unique()} of 8 combinations can meet the loads in hour 0')
print(each.pieces)
```

Five states can meet hour 0; the three without either a heat maker or a
power maker cannot, and are not drawn. `label` spells a piece for the legend
from that frame, dropping the hour every piece agrees on — and a click on a
legend entry hides that state, which is how to look at one at a time:

```python exec="true" source="material-block" html="true" session="region"
show(each.plot())
```

```python exec="true" source="material-block" result="text" session="region"
print(each.optimum)
```

Without the CHP the plant is a box: the boiler alone sets heat, the peaker
alone sets power, and neither trades against the other. The CHP alone is a
segment along its own ratio — `0.4` of heat for `0.4` of power per unit of
gas, from the power load up to its cap — and it is the piece the optimum
lands in. Each pair with the CHP is a polygon on one side of the box, and
all three together is the largest piece.

What the hull hid is now visible. The hull's long slanted edge, from
`(112, 48)` to `(48, 88)`, belongs to no state: it is the well with the CHP
at its cap and *only* one other unit drawing, and with all three on, the
third unit's minimum load takes well capacity of its own. The all-on piece
falls inside that edge — a point like `100` of heat with `55` of power is
inside the hull and reachable by no combination of the units.

`edges` says the same in a table, for every piece at once. Below, what
bounds the all-on piece: its own well edges, and the minimum loads that pull
it inside the hull.

```python exec="true" source="material-block" result="text" session="region"
all_on = each.pieces.filter(pl.col('value') == 1)['piece'].value_counts().filter(pl.col('count') == 3).item(0, 'piece')
with pl.Config(tbl_rows=-1):
    print(each.edges.filter(pl.col('piece') == all_on))
```

The frames share one key, so the questions a reader has are joins and
filters rather than code: the extent of each state, or one state on its own.

```python exec="true" source="material-block" result="text" session="region"
print(
    each.vertices.group_by('piece', maintain_order=True)
    .agg(
        pl.len().alias('vertices'),
        pl.col('heat').max().alias('heat_max'),
        pl.col('power').max().alias('power_max'),
    )
    .with_columns(pl.col('piece').map_elements(each.label, return_dtype=pl.String).alias('state'))
)
```

## Breaking the model on purpose

The reason to draw the region is to see a model that is wrong before it is
solved. Two mistakes a plant model makes often, and what each looks like.

**A cap that is not there.** Drop the well and each unit is still capped, so
the region is larger and still bounded; drop the units' capacity rows and the
well still caps what they burn together. Drop both, and nothing caps the
CHP. `project` does not draw an unbounded region: it stops at the first
direction nothing caps and names it, which is the variable missing its
bound — and a model with a hole like this solves fine as long as the
objective happens to point the other way.

```python exec="true" source="material-block" result="text" session="region"
uncapped = {
    **PLANT,
    'constraints': {k: v for k, v in PLANT['constraints'].items() if k not in ('capacity', 'the_well')},
}

try:
    sps.project(uncapped, sources, x='heat', y='power', at={'t': 0})
except sps.SpecsolveError as exc:
    print(exc)
```

**A minimum load that is too high.** Raise the boiler's minimum load from a
fifth of its capacity to half, and the boiler becomes a unit that cannot
idle: on means `40` of heat before anything else runs. The hull barely
notices — its extreme points are the same units flat out, and one corner
moves — while the states tell the story: every combination with the boiler
on loses its left part. With the boiler running the plant cannot make less
than `40` of heat, and with all three units on not less than `59`, where
before it could sit on the load. The `edges` frame names the row that did
it: `minimum_load` for the boiler, on the edges facing left of every piece
the boiler runs in.

```python exec="true" source="material-block" html="true" session="region"
stiff = sources | {'min_load': pl.DataFrame({'unit': UNITS, 'value': [0.4, 0.5, 0.3]})}
stiff_each = sps.project(PLANT, stiff, x='heat', y='power', at={'t': 0}, binaries='each')
show(stiff_each.plot())
```

```python exec="true" source="material-block" result="text" session="region"
boiler_on = stiff_each.pieces.filter((pl.col('unit') == 'boiler') & (pl.col('value') == 1))['piece']
print(
    stiff_each.edges.filter(
        pl.col('piece').is_in(boiler_on) & (pl.col('name') == 'minimum_load') & (pl.col('unit') == 'boiler')
    )
)
```

The hull of these pieces is the polygon the first section drew, so a solve
with the binaries free would report the same extremes and never say that the
plant can no longer run its boiler below half load. The pieces say it in one
look, and the edges say which row to go and read.

## What it costs, and where it stops

Every vertex is one solve, the optimum one more, and every combination is
one trace, all of them on the model the solver already holds: the probe
changes three costs between solves, and a combination is a pair of bounds
pushed as data, so the matrix is handed over once. Ten pinned binary columns
is where `project` refuses — `1024` traces — and `at` is how to ask about
fewer.

Three things it does not do:

- **`integer` variables are never pinned**, so a piece holding one is, again,
  a hull of what that variable allows.
- **the two axes share one `at`.** A quantity wanted at its own coordinate —
  heat at one bus, power at another — is declared as an expression that
  selects it, which is what `heat` and `power` above already are.
- **an unbounded region is an error, not a picture**, because in a plant model
  an uncapped direction is the finding.

The whole of the verb is [`project`](reference/api.md#specsolve.project).
