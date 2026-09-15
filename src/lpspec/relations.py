"""The shape of relation both lanes build, and the one place that reads one.

The language's relation is a table over any number of dimensions, keyed by any
number of its columns, and walked in whichever direction a call names. What
either lane here builds is the single-valued map: two columns, one of them the
key, so a walk consumes one dimension and produces one, joins on nothing, and
lands where a single equi-join lands it. :func:`refusal` is where a wider table
is turned away, and it is asked once, at :func:`lpspec.lanes.lowered`, so both
lanes and an archive refuse the same file.

The door and the linopy lane read a declared relation through the accessors
here, which is why the guard has one home: each is undefined on a table the
guard would have refused. The relational engine reads the plan's own ``Walk``
and declaration instead, because its subpackage imports nothing from this
package (hard rule 2, ``tests/test_architecture.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from math_spec.program import DimensionPositionNode, Program, RelationDeclaration, Translate, Window


def refusal(program: Program) -> str | None:
    """The sentence refusing a relation neither lane builds, or ``None``.

    Raises nothing: the caller decides which exception carries it.
    """
    for name, relation in program.relations.items():
        if len(relation.columns) != 2 or len(relation.key) != 1:
            return _wider_than_a_map_message(name, relation)
    return None


def _wider_than_a_map_message(name: str, relation: RelationDeclaration) -> str:
    """A relation wider than the single-valued map, named with the rewrite its own shape asks for.

    Two shapes reach here and they want different sentences: a table with no
    key is one ``key:`` away from being a map, and a wider one has to be split
    or carried as a parameter. Saying "split it" to the first would name a
    rewrite that is not the one it needs.
    """
    if not relation.key:
        return (
            f"relation '{name}' declares no key, and this package builds the single-valued map: one "
            f'row per key, so a walk reaches one label rather than a set of them. Declare '
            f'key: on it — one of {list(relation.roles)}, whichever holds each label once — or carry '
            f'the membership as a bool parameter over {list(relation.dims)} and select on it with a '
            f'where string.'
        )
    return (
        f"relation '{name}' has {len(relation.columns)} columns keyed by {len(relation.key)}, and "
        f'this package builds the single-valued map: two columns, one of them the key, so a walk '
        f'trades one dimension for one other and joins on nothing. Split it into one relation per '
        f"pair — '{name}' over {list(relation.dims)} becomes a map per value column, each keyed by "
        f'the same single column — or carry the wider table as a parameter over its dimensions and '
        f'select on it with a where string.'
    )


def key_role(relation: RelationDeclaration) -> str:
    """The column a row is identified by."""
    return relation.key[0]


def value_role(relation: RelationDeclaration) -> str:
    """The column the key determines — what a ``sum(by=)`` lands terms on."""
    return relation.values[0]


def key_dim(relation: RelationDeclaration) -> str:
    """The dimension the map runs out of."""
    return relation.dim(key_role(relation))


def value_dim(relation: RelationDeclaration) -> str:
    """The dimension the map's values are labels of."""
    return relation.dim(value_role(relation))


def maps_out_of(program: Program, dim: str) -> dict[str, str]:
    """Each map keyed over *dim*, to the dimension its values are labels of.

    The question every consumer of a ``by=`` asks. A relation sits under every
    dimension it has a column over, so a map onto *dim* is not a map out of it:
    the two answer different questions and only this one places terms.
    """
    return {r.name: value_dim(r) for r in program.dimension(dim).relations if key_dim(r) == dim}


def partition_of(node: Translate | Window | DimensionPositionNode) -> str | None:
    """The relation a shift, a window or a position counts inside, by name.

    ``None`` where the node counts along the whole axis. The walk itself says
    nothing more here than its name does: it consumes the key column over the
    dimension walked and produces the one value column, which is the map this
    package builds and every reader of a partition already assumes.
    """
    return None if node.partition is None else node.partition.name
