# Debugging a wrong answer

What to read when a model solves to the wrong number, or does not solve, in
the order that finds the fault soonest. Each step needs the file, or the file
and its data, and none needs a solver run you have not already paid for.

## 1. Check the file against the sink

```python
import lpspec as lps

lps.check('dispatch.yaml', sink='highs')
```

`check` raises on a construct outside the language, and with `sink=` on one
the solver cannot take. A warning names a rewrite the sink will make: an
`sos:` set on `highs` arrives as binaries, so the solve comes back with no
duals ([checking against a sink](../reference/api.md#checking-against-a-sink)).

## 2. Read the shape the build produced

```python
model = lps.build('dispatch.yaml', sources)
report = model.diagnostics()
report.columns, report.rows, report.nonzeros  # 8, 4, 8
report.omissions  # rows a constraint declared and did not build
report.sparse_parameters  # parameters whose table is short of their coordinates
```

A count smaller than you expected is a mask, or a table with a row missing.
`omissions` names the constraint, `sparse_parameters` the parameter; which one
you have is the difference between a `where:` you wrote and a row you lost
([diagnostics](../reference/api.md#diagnostics)).

## 3. Read the row that is wrong

```python
print(model.row('power_balance', snapshot=2))
# power_balance[snapshot=2]: +1 p[2, wind] +1 p[2, gas] == 180
```

The line is the row as the solver got it: every coefficient the data
produced, and no term for a variable a `where:` removed. Here `solar` is
absent because its `p_max` is `0.0`. A term you expected and do not see is a
mask; a coefficient you did not expect is the data
([reading one row](../reference/api.md#reading-one-row)).

## 4. When the solve is infeasible

```python
result = model.solve()
result.status, result.termination_condition  # 'warning', 'infeasible'
result.has_primal  # False: nothing to read, and every reader raises
```

There is no IIS (irreducible infeasible subsystem) read-back. Locate the
fault instead with a slack: add a variable to the balance row, minimise it,
and read where it is nonzero. The
[feasibility model](../about/decomposition.md#when-the-subproblem-is-infeasible)
is that file for a dispatch. Then read the row at a snapshot the slack lands
on, as in step 3.

## 5. When the number is wrong and the rows look right

Build the same file on the other lane and compare the objectives:

```python
from lpspec import linopy as lpspec_linopy

m = lpspec_linopy.build('dispatch.yaml', sources)
m.solve()
m.objective.value  # against result.objective
```

Two lanes agreeing on a number you still believe is wrong means the file says
something other than what you meant. Render it as math and read the
constraint as written:
[typeset](https://math-spec.readthedocs.io/en/latest/reference/typeset/).
