"""A relation's table as a call reads it — the one place a role becomes a column.

The plan's :class:`~math_spec.program.Join` names *roles*: which columns of
a relation an operator joins on and which it groups by. The engine reads by
*dimension*, since an operand carries its coordinates under the dimensions'
names. Everything here is that translation, spelled once:

- a group or a lookup joins the operand to the :func:`mapping` table through
  :func:`join_relation`, dropping the dimensions joined on and not grouped by
  and gaining the ones grouped by and not joined on;
- a :class:`~math_spec.program.Partition` ranks the dimension it steps along
  inside a :class:`Grouping`.

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
    """The column a join's added column waits under until the dropped one is gone.

    A self-map groups by the dimension it joins on, and a frame carries a
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
# a group or a lookup: the join, and what it drops and adds
# ---------------------------------------------------------------------------


def mapping(relations: Mapping[str, pl.LazyFrame], join: program.Join) -> pl.LazyFrame:
    """The table a group or a lookup joins against — the relation, as *join* names its columns.

    Columns joined on arrive under their dimensions, columns grouped by and
    not joined on under :func:`landing`. A key the relation does not map has
    no row in the relation and so none here, which is what "reaches no slot"
    means.
    """
    table = relations[join.name]
    return table.select(
        *(pl.col(role).alias(join.dim(role)) for role in join.joined),
        *(pl.col(role).alias(landing(join.dim(role))) for role in join.added),
    )


def grouped(mapping: pl.LazyFrame, node: program.GroupSum | program.Lookup) -> pl.LazyFrame:
    """*mapping* at the coordinates the join groups by, under their own names."""
    join = node.join
    return mapping.select(*join.kept_dims, *(pl.col(landing(d)).alias(d) for d in join.added_dims))


def join_relation(
    frame: pl.LazyFrame,
    mapping: pl.LazyFrame,
    node: program.GroupSum | program.Lookup,
    have: Sequence[str],
    columns: Sequence[str] = (),
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """*frame* joined to *mapping*: one inner equi-join, and the dimensions the result is over.

    The join keys on every dimension the node's join joins on. The result
    keeps every dimension of *have* but the ones joined on and not grouped
    by, gains every dimension grouped by and not joined on under its own
    name, and keeps *columns* beside them. A call brings the dimensions it
    groups by — the language refuses one the operand already carries — so the
    gained are exactly the added. A dropped and a gained one may still be a
    single dimension, which a self-map does, so the two are traded in a single
    select.

    Args:
        frame: The operand, carrying *have* and *columns*.
        mapping: :func:`mapping` for the node's join.
        node: The group or the lookup, whose join says what is joined on and
            what is grouped by.
        have: The dimensions *frame* carries.
        columns: The other columns to keep — a fragment's carried ones.

    Returns:
        The joined frame, and the dimensions it is over, in order.
    """
    join = node.join
    gained = join.added_dims
    keep = [d for d in have if d not in join.dropped_dims]
    traded = frame.join(mapping, on=list(join.joined_dims), how='inner').select(
        *keep, *(pl.col(landing(d)).alias(d) for d in gained), *columns
    )
    return traded, (*keep, *gained)


# ---------------------------------------------------------------------------
# a partition: the walked dimension ranked inside its groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grouping:
    """A dimension ranked inside the groups a partition makes — what ``shift``, ``sum_back`` and ``position`` count along.

    A group is the partition's group columns at each coordinate of its joined
    dimensions — a season per generator, where the relation is keyed by both.
    The inner join behind :attr:`table` is where "this coordinate is in no
    group" comes from: it has no row in the relation, so it has none here,
    and every rank, span and neighbour a partitioned call reads sees only
    coordinates that are in one.

    Attributes:
        dimension: The dimension stepped along.
        joined: The partition's other key dimensions, which the operand carries.
        groups: The group-making columns, one per group column, under
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

        What an unpartitioned call counts along, so ``shift`` and ``sum_back``
        read one table shape whether or not a ``by=`` was written.
        """
        table = data.dimensions[dimension].with_columns(
            pl.col('ord').alias(GROUP_RANK),
            pl.lit(data.cardinality[dimension], dtype=pl.Int64).alias(GROUP_SIZE),
        )
        return cls(dimension, (), (), (), table)

    @classmethod
    def of(cls, data: AttachedSources, partition: program.Partition) -> Grouping:
        """Rank the dimension *partition* steps along inside the groups its ``within=`` columns make."""
        dimension = partition.along_dim
        joined = partition.joined_dims
        groups = tuple(group_column(role) for role in partition.grouped)
        rows = data.relations[partition.name].select(
            pl.col(partition.along).alias('val'),
            *(pl.col(role).alias(dim) for role, dim in zip(partition.joined, joined, strict=True)),
            *(pl.col(role).alias(column) for role, column in zip(partition.grouped, groups, strict=True)),
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
        return cls(dimension, joined, groups, tuple(partition.dim(role) for role in partition.grouped), table)

    @property
    def key(self) -> tuple[str, ...]:
        """What one group is identified by: its joined coordinates and its group columns."""
        return (*self.joined, *self.groups)

    @property
    def keys(self) -> tuple[str, ...]:
        """What a coordinate of the dimension stepped along is identified by: the dimension, and the joined ones."""
        return (self.dimension, *self.joined)

    @property
    def partial(self) -> bool:
        """Whether a coordinate can be in no group: a relation places only the labels it holds, the whole dimension every one."""
        return bool(self.groups)

    def placed(self) -> pl.LazyFrame:
        """The coordinates the partition places in some group, under :attr:`keys`.

        The rest belong to none, so a partitioned call reaches nothing for
        them and their rows are not built — the reading ``sum(by=)`` gives a
        label the map has no row for, and the one an edge policy cannot speak
        about.
        """
        return self.table.select(pl.col('val').alias(self.dimension), *self.joined).unique()

    def column_of(self, dim: str) -> str | None:
        """The group column carrying *dim*'s labels, or ``None`` where the partition does not group into it.

        Where two group columns are over one dimension, the first declared
        carries it.
        """
        return next((column for column, over in zip(self.groups, self.grouped, strict=True) if over == dim), None)
