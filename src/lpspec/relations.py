"""The shape of relation both lanes build, and the one place that reads one.

The language's relation is a table over any number of dimensions, keyed by any
number of its columns, and walked in whichever direction a call names. What
either lane here builds is the **single-valued map**: one value column the key
determines, so a walk trades the dimension it consumes for the one it produces
and joins on whatever else the key names. A key of several columns is a map
conditioned on them — a generator's zone by period — and the join carries them
through. :func:`refusal` is where a table neither lane builds is turned away,
and it is asked once, at :func:`lpspec.lanes.lowered`, so both lanes and an
archive refuse the same file.

The door and the linopy lane read a declared relation through the accessors
here, which is why the guard has one home: each is undefined on a table the
guard would have refused. The relational engine reads the plan's own ``Walk``
and declaration instead, because its subpackage imports nothing from this
package (hard rule 2, ``tests/test_architecture.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_spec import program as _program

if TYPE_CHECKING:
    from collections.abc import Iterator

    from math_spec.program import (
        DimensionPositionNode,
        Mask,
        Program,
        RelationDeclaration,
        Translate,
        Window,
    )


def refusal(program: Program) -> str | None:
    """The sentence refusing a relation neither lane builds, or ``None``.

    Raises nothing: the caller decides which exception carries it.
    """
    for name, relation in program.relations.items():
        if not relation.key or len(relation.values) != 1:
            return _wider_than_a_map_message(name, relation)
    for node in _partitioning(program):
        assert node.partition is not None, '_partitioning yields only the nodes that carry one'
        if len(node.partition.key) > 1:
            return _partition_by_a_conditioned_map_message(node.partition.relation)
    return None


def _wider_than_a_map_message(name: str, relation: RelationDeclaration) -> str:
    """A relation wider than the single-valued map, named with the rewrite its own shape asks for.

    Two shapes reach here and they want different sentences: a table with no
    key is one ``key:`` away from being a map, and one the key does not
    determine a single column of has to be split or carried as a parameter.
    Saying "split it" to the first would name a rewrite that is not the one it
    needs.
    """
    if not relation.key:
        return (
            f"relation '{name}' declares no key, and this package builds the single-valued map: one "
            f'row per key, so a walk reaches one label rather than a set of them. Declare '
            f'key: on it — one of {list(relation.roles)}, whichever holds each label once, or '
            f'several of them together — or carry the membership as a bool parameter over '
            f'{list(relation.dims)} and select on it with a where string.'
        )
    return (
        f"relation '{name}' has {len(relation.values)} columns its key does not determine "
        f'({list(relation.values)}), and this package builds the single-valued map: one value column '
        f'per key, so a walk trades one dimension for one other. Split it into one relation per value '
        f"column — '{name}' keyed by {list(relation.key)} becomes a map per column — or carry the "
        f'wider table as a parameter over its dimensions and select on it with a where string.'
    )


def _partition_by_a_conditioned_map_message(relation: RelationDeclaration) -> str:
    """A ``shift``, ``sum_back`` or ``position`` grouped by a map keyed on more than the dimension it walks."""
    return (
        f"relation '{relation.name}' is keyed by {list(relation.key)}, and a partitioned shift, "
        f'sum_back or position groups by a map keyed by the one dimension it walks: which group a '
        f'coordinate is in would otherwise depend on the other key columns, and the walk has no row '
        f"to read them at. Declare a relation keyed by '{relation.dim(relation.key[0])}' alone for "
        f'the grouping, or drop the by= and walk the whole axis.'
    )


def key_roles(relation: RelationDeclaration) -> tuple[str, ...]:
    """The columns a row is identified by, in declared order."""
    return relation.key


def value_role(relation: RelationDeclaration) -> str:
    """The column the key determines — what a ``sum(by=)`` lands terms on."""
    return relation.values[0]


def key_dims(relation: RelationDeclaration) -> tuple[str, ...]:
    """The dimensions the map is keyed by, in declared order."""
    return tuple(relation.dim(role) for role in relation.key)


def value_dim(relation: RelationDeclaration) -> str:
    """The dimension the map's values are labels of."""
    return relation.dim(value_role(relation))


def maps_out_of(program: Program, dim: str) -> dict[str, str]:
    """Each map *dim* is part of the key of, to the dimension its values are labels of.

    The question every consumer of a ``by=`` asks. A relation sits under every
    dimension it has a column over, so a map onto *dim* is not a map out of it:
    the two answer different questions and only this one places terms.
    """
    return {r.name: value_dim(r) for r in program.dimension(dim).relations if dim in key_dims(r)}


def partition_of(node: Translate | Window | DimensionPositionNode) -> str | None:
    """The relation a shift, a window or a position counts inside, by name.

    ``None`` where the node counts along the whole axis. The walk itself says
    nothing more here than its name does: it consumes the key column over the
    dimension walked and produces the one value column, which is the map this
    package builds and every reader of a partition already assumes.
    """
    return None if node.partition is None else node.partition.name


def _partitioning(program: Program) -> Iterator[Translate | Window | DimensionPositionNode]:
    """Every node that groups by a relation — the ones a partition's own rule is about.

    Both halves of the language reach one: an operator inside an expression,
    and a ``position(by=)`` inside a mask, including the masks a cased
    expression carries in its regions.
    """
    bodies = (*program.expressions, *(e.expression for e in program.named_expressions.values()))
    for node in _program.walk(*bodies):
        if isinstance(node, (_program.Translate, _program.Window)) and node.partition is not None:
            yield node
        if isinstance(node, _program.Cases):
            for region in node.regions:
                yield from _positions(region.when)
    for declaration in (*program.variables.values(), *program.constraints.values()):
        yield from _positions(declaration.where)


def _positions(mask: Mask | None) -> Iterator[DimensionPositionNode]:
    """The grouped positions one mask tests, or nothing."""
    if mask is None:
        return
    for atom in mask.atoms:
        if isinstance(atom, _program.DimensionPositionNode) and atom.partition is not None:
            yield atom
