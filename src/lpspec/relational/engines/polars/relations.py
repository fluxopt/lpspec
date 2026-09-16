"""A relation's table as a walk reads it — the one place a role becomes a column.

The plan's :class:`~math_spec.program.Walk` names *roles*: which columns of a
relation an operator consumes, produces and joins on. The engine reads by
*dimension*, since an operand carries its coordinates under the dimensions'
names. Everything here is that translation, spelled once:

- a group or a pullback trades the :class:`Ends` its walks name through
  :func:`walk_join`, against the :func:`mapping` table;
- a partition ranks the walked dimension inside a :class:`Grouping`;
- a ``where`` reads a value column at the key through :func:`keyed`.

Nothing here reads data or holds state: every function takes the attached
frames and returns a lazy query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from math_spec import program

    from lpspec.relational.engines.polars.attaching import AttachedSources

#: What a :class:`Grouping` adds to a dimension table: a coordinate's rank
#: inside its group, and the group's size — the position and span a
#: partitioned walk reads.
GROUP_RANK = '__pos in group__'
GROUP_SIZE = '__group size__'


def landing(dim: str) -> str:
    """The column a walk's produced column waits under until the consumed one is dropped.

    A self-map produces the dimension it consumes, and a frame carries a
    dimension once; the spaces make the name unrepresentable as a declared
    one.
    """
    return f'__landing {dim}__'


def group_column(role: str) -> str:
    """The column a :class:`Grouping` carries one group-making value column of the relation under.

    Named for the role rather than its dimension, since a group may hold two
    columns over one dimension, and kept apart from the dimension's own name,
    which the operand may carry as an axis of its own.
    """
    return f'__group {role}__'


# ---------------------------------------------------------------------------
# a group or a pullback: the ends of a walk, and the join that trades them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ends:
    """The dimensions a node's walks trade: consumed for produced, at each coordinate of the joined.

    Each dimension once. Every walk of one node consumes the same dimensions,
    and several pullbacks land on the same fine ones, so the union is the
    node's answer rather than a walk's.
    """

    consumed: tuple[str, ...]
    joined: tuple[str, ...]
    produced: tuple[str, ...]

    @classmethod
    def of(cls, walks: Sequence[program.Walk]) -> Ends:
        return cls(
            tuple(dict.fromkeys(d for walk in walks for d in walk.consumed_dims)),
            tuple(dict.fromkeys(d for walk in walks for d in walk.joined_dims)),
            tuple(dict.fromkeys(d for walk in walks for d in walk.produced_dims)),
        )

    @property
    def fine(self) -> tuple[str, ...]:
        """The dimensions a pullback's result is keyed by: the joined ones and the produced ones."""
        return (*self.joined, *self.produced)


def mapping(relations: Mapping[str, pl.LazyFrame], walks: Sequence[program.Walk]) -> pl.LazyFrame:
    """The table a group or a pullback joins against — every walk's relation, met on the columns they share by **inner** joins.

    Each relation is read as its walk names it: consumed and joined columns
    under their dimensions, produced ones under :func:`landing`. A key some
    walk does not map has no row in that relation and so none here, which is
    what "reaches no slot" means for the whole tuple: it exists exactly where
    every walk does.
    """
    first, *rest = (_walked(relations[walk.name], walk) for walk in walks)
    for other in rest:
        names = other.collect_schema().names()
        shared = [c for c in first.collect_schema().names() if c in names]
        first = first.join(other, on=shared, how='inner')
    return first


def _walked(table: pl.LazyFrame, walk: program.Walk) -> pl.LazyFrame:
    return table.select(
        *(pl.col(role).alias(walk.dim(role)) for role in (*walk.consumed, *walk.joined)),
        *(pl.col(role).alias(landing(walk.dim(role))) for role in walk.produced),
    )


def landed(mapping: pl.LazyFrame, ends: Ends) -> pl.LazyFrame:
    """*mapping* at the coordinates it lands on: the joined dimensions and the produced ones, under their names."""
    return mapping.select(*ends.joined, *(pl.col(landing(d)).alias(d) for d in ends.produced))


def walk_join(
    frame: pl.LazyFrame, mapping: pl.LazyFrame, ends: Ends, have: Sequence[str], columns: Sequence[str] = ()
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """*frame* traded through *mapping*: one inner equi-join, and the dimensions the result is over.

    The join keys on the consumed and joined dimensions, and on a produced
    dimension the frame already carries — the masked sum the language reads
    that as. The result keeps every dimension of *have* but the consumed
    ones, gains every other produced dimension under its own name, and keeps
    *columns* beside them. The consumed and the gained may be one dimension,
    which a self-map does, so the two are traded in a single select.

    Args:
        frame: The operand, carrying *have* and *columns*.
        mapping: :func:`mapping` for the same walks.
        ends: :class:`Ends` of the same walks.
        have: The dimensions *frame* carries.
        columns: The other columns to keep — a fragment's carried ones.

    Returns:
        The traded frame, and the dimensions it is over, in order.
    """
    keep = [d for d in have if d not in ends.consumed]
    carried = [d for d in ends.produced if d in keep]
    gained = [d for d in ends.produced if d not in keep]
    traded = frame.join(
        mapping,
        left_on=[*ends.consumed, *ends.joined, *carried],
        right_on=[*ends.consumed, *ends.joined, *(landing(d) for d in carried)],
        how='inner',
    ).select(*keep, *(pl.col(landing(d)).alias(d) for d in gained), *columns)
    return traded, (*keep, *gained)


# ---------------------------------------------------------------------------
# a partition: the walked dimension ranked inside its groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grouping:
    """A dimension ranked inside the groups a partition makes — what ``shift``, ``sum_back`` and ``position`` count along.

    A group is the walk's produced columns at each coordinate of its joined
    dimensions — a season per generator, where the relation is keyed by both.
    The inner join behind :attr:`table` is where "this coordinate is in no
    group" comes from: it has no row in the relation, so it has none here,
    and every rank, span and neighbour a walk reads sees only coordinates
    that are in one.

    Attributes:
        dimension: The dimension walked.
        joined: The partition's other key dimensions, which the operand carries.
        groups: The group-making columns, one per produced column, under
            :func:`group_column`.
        grouped: The dimension each group column is over, in the same order.
        table: ``(val, ord, joined…, groups…, GROUP_RANK, GROUP_SIZE)``, one
            row per coordinate the partition places in a group.
    """

    dimension: str
    joined: tuple[str, ...]
    groups: tuple[str, ...]
    grouped: tuple[str, ...]
    table: pl.LazyFrame

    @classmethod
    def of(cls, data: AttachedSources, walk: program.Walk) -> Grouping:
        """Rank the dimension *walk* consumes inside the groups it produces."""
        (consumed,) = walk.consumed
        dimension = walk.dim(consumed)
        joined = walk.joined_dims
        groups = tuple(group_column(role) for role in walk.produced)
        rows = data.relations[walk.name].select(
            pl.col(consumed).alias('val'),
            *(pl.col(role).alias(dim) for role, dim in zip(walk.joined, joined, strict=True)),
            *(pl.col(role).alias(column) for role, column in zip(walk.produced, groups, strict=True)),
        )
        key = [*joined, *groups]
        table = (
            data.dimensions[dimension]
            .join(rows, on='val', how='inner')
            .with_columns(
                (pl.col('ord').rank('ordinal').over(key) - 1).cast(pl.Int64).alias(GROUP_RANK),
                pl.len().over(key).cast(pl.Int64).alias(GROUP_SIZE),
            )
        )
        return cls(dimension, joined, groups, walk.produced_dims, table)

    @property
    def key(self) -> tuple[str, ...]:
        """What one group is identified by: its joined coordinates and its group columns."""
        return (*self.joined, *self.groups)

    @property
    def keys(self) -> tuple[str, ...]:
        """What a coordinate of the walked dimension is identified by: the dimension, and the joined ones."""
        return (self.dimension, *self.joined)

    def placed(self) -> pl.LazyFrame:
        """The coordinates the partition places in some group, under :attr:`keys`.

        The rest belong to none, so a partitioned walk reaches nothing for
        them and their rows are not built — the reading ``sum(by=)`` gives a
        label the map has no row for, and the one an edge policy cannot speak
        about.
        """
        return self.table.select(pl.col('val').alias(self.dimension), *self.joined).unique()

    def column_of(self, dim: str) -> str | None:
        """The group column carrying *dim*'s labels, or ``None`` where the partition does not group into it.

        Where two produced columns are over one dimension, the first declared
        carries it.
        """
        return next((column for column, over in zip(self.groups, self.grouped, strict=True) if over == dim), None)


# ---------------------------------------------------------------------------
# a where: a relation read at its key
# ---------------------------------------------------------------------------


def keyed(
    table: pl.LazyFrame, declaration: program.RelationDeclaration, dims: tuple[str, ...], column: str | None, alias: str
) -> pl.LazyFrame:
    """*declaration*'s table read at *dims* — one value column under *alias*, or with ``None`` a marker that a row is there.

    The frame supplies the key's dimensions for a keyed relation and every
    column's for a bare one, which is what the plan stamps on the leaf; the
    columns read at are the relation's own, matched to *dims* in order.
    """
    roles = declaration.key or declaration.roles
    assert tuple(declaration.dim(role) for role in roles) == dims, f"relation '{declaration.name}' is read at its key"
    read = pl.col(column) if column is not None else pl.lit(value=True)
    return table.select(*(pl.col(role).alias(dim) for role, dim in zip(roles, dims, strict=True)), read.alias(alias))
