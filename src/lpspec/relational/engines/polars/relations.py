"""A relation's table as a walk reads it — the one place a role becomes a column.

The plan's :class:`~math_spec.program.Walk` names *roles*: which columns of a
relation an operator consumes, produces and joins on. The engine reads by
*dimension*, since an operand carries its coordinates under the dimensions'
names. Everything here is that translation, spelled once:

- a group or a pullback trades the dimensions its walk consumes for the
  ones it produces through :func:`walk_join`, against the :func:`mapping`
  table;
- a partition ranks the walked dimension inside a :class:`Grouping`.

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
    which the operand may carry as a dimension of its own.
    """
    return f'__group {role}__'


# ---------------------------------------------------------------------------
# a group or a pullback: the ends of a walk, and the join that trades them
# ---------------------------------------------------------------------------


def mapping(relations: Mapping[str, pl.LazyFrame], walk: program.Walk) -> pl.LazyFrame:
    """The table a group or a pullback joins against — the walk's relation, read as the walk names it.

    Consumed and joined columns arrive under their dimensions, produced ones
    under :func:`landing`. A key the walk does not map has no row in the
    relation and so none here, which is what "reaches no slot" means.
    """
    table = relations[walk.name]
    return table.select(
        *(pl.col(role).alias(walk.dim(role)) for role in (*walk.consumed, *walk.joined)),
        *(pl.col(role).alias(landing(walk.dim(role))) for role in walk.produced),
    )


def landed(mapping: pl.LazyFrame, node: program.GroupSum | program.At) -> pl.LazyFrame:
    """*mapping* at the coordinates it lands on: the joined dimensions and the produced ones, under their names."""
    return mapping.select(*node.joined, *(pl.col(landing(d)).alias(d) for d in node.walk.produced_dims))


def walk_join(
    frame: pl.LazyFrame,
    mapping: pl.LazyFrame,
    node: program.GroupSum | program.At,
    have: Sequence[str],
    columns: Sequence[str] = (),
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """*frame* traded through *mapping*: one inner equi-join, and the dimensions the result is over.

    The join keys on the dimensions the walk consumes and joins on. The result
    keeps every dimension of *have* but the consumed ones, gains every
    produced dimension under its own name, and keeps *columns* beside them. A
    walk brings the dimensions it lands on — the language refuses one the
    operand already carries — so the gained are exactly the produced. The
    consumed and a gained one may still be a single dimension, which a
    self-map does, so the two are traded in a single select.

    Args:
        frame: The operand, carrying *have* and *columns*.
        mapping: :func:`mapping` for the node's walk.
        node: The group or the pullback, whose walk says what is consumed
            and produced and whose ``joined`` says what is joined on.
        have: The dimensions *frame* carries.
        columns: The other columns to keep — a fragment's carried ones.

    Returns:
        The traded frame, and the dimensions it is over, in order.
    """
    gained = node.walk.produced_dims
    keep = [d for d in have if d not in node.walk.consumed_dims]
    traded = frame.join(mapping, on=[*node.walk.consumed_dims, *node.joined], how='inner').select(
        *keep, *(pl.col(landing(d)).alias(d) for d in gained), *columns
    )
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
    def whole(cls, data: AttachedSources, dimension: str) -> Grouping:
        """*dimension* as one group: the rank is the ordinal, the size the cardinality, and every label is placed.

        What an unpartitioned walk counts along, so ``shift`` and ``sum_back``
        read one table shape whether or not a ``by=`` was written.
        """
        table = data.dimensions[dimension].with_columns(
            pl.col('ord').alias(GROUP_RANK),
            pl.lit(data.cardinality[dimension], dtype=pl.Int64).alias(GROUP_SIZE),
        )
        return cls(dimension, (), (), (), table)

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

    @property
    def partial(self) -> bool:
        """Whether a coordinate can be in no group: a relation places only the labels it holds, the whole dimension every one."""
        return bool(self.groups)

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
