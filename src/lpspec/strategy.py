"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a *process* may loop over plans
(docs/about/ceiling.md). So a strategy is a driver above :mod:`lpspec.api`,
built from the public verbs — never a language or engine feature.

Every strategy is the same fold: **partition → attach → solve → carry → stitch**.
Only how slices are cut and whether they couple differs. A serial fold builds
once and updates each slice (:func:`_serially`); under a process pool it builds
per slice (:func:`_pooled`), a built model being the one thing that cannot
cross. Both yield an :class:`_Answer`, and the fold that absorbs them is
written once.

    scenario / sweep    ``EachCoordinate('scenario')``            independent
    myopic pathway      ``EachCoordinate('period')``              + ``carry``
    rolling horizon     ``EachWindow('snapshot', 48, 24, 't')``   + ``carry``

**A partition is a filter on the sources, not a narrower index** — the
containment check refuses parameter rows outside a narrowed index, so an axis
rewrites the rows and the index together.

The caller-facing rules are [docs/reference/sweeps.md](../../docs/reference/sweeps.md).
"""

from __future__ import annotations

import io
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import polars as pl
from math_spec import to_program, to_spec
from math_spec.program import Program

from lpspec.api import build, check
from lpspec.errors import DataError, LpspecError, LpspecWarning, did_you_mean
from lpspec.frames import as_frame
from lpspec.relational.result import tidy_to_dataarray, tidy_to_dataset, tidy_to_pandas
from lpspec.sources import least_value

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    import pandas as pd
    import xarray as xr
    from math_spec import Spec

    from lpspec.api import Model
    from lpspec.lanes import Buildable
    from lpspec.relational.result import Diagnostics, Keep

#: Parquet rather than pickle, and not a knob: zstd measured smaller *and*
#: faster than pickling the frame, on compressible and incompressible data
#: alike (#459).
_COMPRESSION = 'zstd'


class _Cut(NamedTuple):
    """One slice of a sweep: the key, and the sources that build it.

    A tuple on purpose: a hand-built axis is a plain list of ``(key, sources)``,
    and those unpack the same way.
    """

    key: Any
    sources: Mapping[str, Any]


class _SliceMeta(NamedTuple):
    """One row of :attr:`Runs.objective`: how a slice terminated, and its objective."""

    status: str
    termination_condition: str
    objective: float


#: The phases :attr:`Runs.diagnostics` clocks, in the order they run.
_PHASES = ('attach', 'build', 'handoff', 'solve')


def _slice_cost(after: Diagnostics, before: Diagnostics | None) -> dict[str, Any]:
    """One slice's row of :attr:`Runs.diagnostics`, off the model's cumulative counters.

    A serial fold reuses one model, whose clocks and ``loads`` keep summing
    across slices: *before* is the reading taken as the previous slice
    finished, and the difference is this slice's own. A model built for one
    slice alone has no *before*.
    """
    earlier = before.timings if before is not None else {}
    return {
        'columns': after.columns,
        'rows': after.rows,
        'nonzeros': after.nonzeros,
        'loaded': after.loads > (before.loads if before is not None else 0),
        **{phase: after.timings.get(phase, 0.0) - earlier.get(phase, 0.0) for phase in _PHASES},
    }


@dataclass(frozen=True)
class _CarryRule:
    """One resolved carry: which variable moves into a parameter, and how.

    ``dropped`` is the one dimension the carry collapses — ``None`` when the
    whole frame moves forward — and ``index`` names a coordinate of it.
    """

    variable: str
    dropped: str | None
    index: int | None

    @classmethod
    def resolved(cls, program: Program, parameter: str, variable: str, index: int | None) -> _CarryRule:
        """One carry checked against the plan — construction and validation, together.

        The variable's dims minus the parameter's is the one dimension the
        carry collapses; everything else passes through, so a myopic pathway
        hands a whole capacity vector forward rather than one number at a time.
        Nothing here reads data, which is why the plan resolves before the axis
        cuts any.
        """
        if parameter not in program.parameters:
            raise LpspecError(f'carry writes parameter {parameter!r}, which the spec does not declare')
        if variable not in program.variables:
            raise LpspecError(f'carry reads variable {variable!r}, which the spec does not declare')
        over = list(program.parameters[parameter].dims)
        source = list(program.variables[variable].dims)
        if missing := [d for d in over if d not in source]:
            raise LpspecError(
                f'carry {parameter!r} <- {variable!r} cannot line up: {parameter!r} is over {over}, and '
                f'{variable!r} is over {source}, which has no {missing}. A carry copies a variable into a '
                f'parameter, so the parameter cannot be over more than the variable is.'
            )
        dropped = [d for d in source if d not in over]
        if len(dropped) > 1:
            raise LpspecError(
                f'carry {parameter!r} <- {variable!r} would collapse {dropped} at once: {variable!r} is over '
                f'{source} and {parameter!r} over {over}. An index names a coordinate of one dimension, so '
                f'reduce the others in the YAML — a derived variable is where the oracle can see the math.'
            )
        if dropped and index is None:
            raise LpspecError(
                f'carry {parameter!r} <- {variable!r} drops {dropped[0]!r} and so needs an index: '
                f'({variable!r}, <{dropped[0]}>). With overlap the coordinate to carry is the last one you '
                f'*keep* — EachWindow(length=48, step=24) carries at 23, not 47 — which is why there is no default.'
            )
        if not dropped and index is not None:
            raise LpspecError(
                f'carry {parameter!r} <- ({variable!r}, {index}) has nothing to index: both are over {over}, '
                f'so the whole frame is what moves forward. Pass ({variable!r}, None).'
            )
        return cls(variable, dropped[0] if dropped else None, index)

    def value_from(self, frames: Mapping[str, pl.DataFrame], parameter: str, key: Any) -> pl.DataFrame:
        """What this rule hands the next slice, read out of one slice's primals.

        ``index`` is a **coordinate** of the dropped dimension, never a row
        number — the only one of the two that still means something once a
        second dimension is there.

        Raises:
            LpspecError: The slice built no row of the variable at ``index``
                — a ``where`` or an absence rule took it.
        """
        frame = frames[self.variable]
        if self.dropped is None:
            return frame
        picked = frame.filter(pl.col(self.dropped) == self.index).drop(self.dropped)
        if picked.is_empty():
            raise LpspecError(
                f'carry {parameter!r} <- ({self.variable!r}, {self.index}) has nothing to copy: slice {key!r} '
                f'built no {self.variable!r} at {self.dropped} == {self.index}, so there is no value there '
                f'to hand forward.'
            )
        return picked


@dataclass(frozen=True)
class _Answer:
    """One slice, solved and read out — what the fold absorbs.

    What :func:`_serially` and :func:`_pooled` both produce. Plain data
    throughout, because a worker returns it and so it has to pickle: frames,
    strings and numbers, never a result or a model.
    """

    meta: _SliceMeta
    #: This slice's row of :attr:`Runs.diagnostics`, from :func:`_slice_cost`.
    cost: dict[str, Any]
    primals: dict[str, pl.DataFrame]
    duals: dict[str, pl.DataFrame]
    #: Every declared named expression, evaluated at this slice's solution.
    expressions: dict[str, pl.DataFrame]
    #: Why this slice has none, when it has none. Carried rather than raised:
    #: one mixed-integer slice must not fail a whole sweep.
    no_duals: str | None
    #: Per expression, why this slice could not evaluate it — the same
    #: carried-not-raised rule, per name because each fails on its own data.
    no_expressions: dict[str, str]


@dataclass(frozen=True)
class _OriginalIndex:
    """The way back from a windowed sweep's slices to the dimension it sliced.

    ``owned`` is ``(key, local, dim)`` for the coordinates each window is
    *responsible* for — its first ``step``, the rest being lookahead the next
    window recomputes. One-way: the lookahead rows are not in it, so a sliced
    frame cannot be rebuilt from it — slicing stays
    :meth:`EachWindow._slices`' business.
    """

    local: str
    dim: str
    owned: pl.DataFrame

    def restore(self, frame: pl.DataFrame, key_name: str) -> pl.DataFrame:
        """*frame* over the dimension the axis sliced, rather than over its slices.

        The inner join against ``owned`` is the whole operation: it restores
        the original coordinate, and because a coordinate may appear only once
        under its own index, the lookahead rows have nowhere to go. Sorted on
        the restored dimension, so a sweep reads back in the caller's order.

        Raises:
            LpspecError: *frame* has no ``local`` column — the quantity was
                reduced over the sliced dimension, so there is no way back.
        """
        if self.local not in frame.columns:
            raise LpspecError(
                f'cannot read this over {self.dim!r}: the frame has no {self.local!r} column, because the '
                f'quantity was reduced over the sliced dimension — each row already covers a whole slice, '
                f'lookahead included under an overlapping window. Read it without original_index for the '
                f'per-slice values, or read a quantity that keeps {self.local!r} and aggregate the '
                f'stitched frame.'
            )
        keys = [key_name, self.local]
        restored = frame.join(self.owned, on=keys, how='inner').drop(keys)
        rest = [column for column in restored.columns if column not in (self.dim, 'value')]
        return restored.select(self.dim, *rest, 'value').sort(self.dim, *rest)


def _listed(entries: Mapping[str, str]) -> str:
    return '\n'.join(f'  {label}: {reason}' for label, reason in entries.items())


def _least(program: Program, sources: Mapping[str, Any], name: str) -> int:
    """The least value of parameter *name*, which decides how far its rows read ahead; an empty one reads nowhere.

    Read through :func:`~lpspec.sources.least_value`, so every shape a source
    may arrive in — a parquet path, a table, a scalar, a ``{label: value}``
    map, a sequence — is the one door's business rather than this driver's.

    Raises:
        DataError: *name* is a parameter nothing supplies.
    """
    if name not in sources:
        raise DataError(f"no data provided for parameter '{name}'")
    least = least_value(name, program.parameter(name), sources[name])
    return 0 if least is None else int(least)


# ---------------------------------------------------------------------------
# axes — how slices are cut
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EachCoordinate:
    """One slice per coordinate of *dim* — a column the sources carry.

    Scenarios, draws, investment periods. Sources carrying *dim* are filtered
    to one coordinate and the column dropped, so the model never mentions it —
    a *dim* the spec declares is refused; every other source passes through
    untouched. The slices run in the coordinates' sorted order, which is the
    order a ``carry`` chains them in.
    """

    dim: str

    def cuts(self, sources: Mapping[str, Any]) -> list[tuple[Any, Mapping[str, Any]]]:
        """The ``(key, sources)`` list this axis would run — what ``axis=`` takes hand-built.

        For building one slice alone: ``lps.build(spec, axis.cuts(sources)[3][1])``.
        """
        return list(self._slices(sources, self._key_name())[0])

    def _key_name(self) -> str:
        """The dimension itself: a slice key *is* a coordinate of it."""
        return self.dim

    def _check_the_program(self, program: Program, sources: Mapping[str, Any]) -> None:
        """Refuse a sweep over a dimension the spec declares, which nothing would then supply.

        The only thing a coordinate sweep needs of the program, and the reason
        it needs nothing else: the column is dropped from every source, so a
        spec that declared it could not be built — and a program that never
        sees the axis cannot tie it together either.

        Raises:
            LpspecError: The spec declares *dim*.
        """
        del sources
        if self.dim in program.dimensions:
            raise LpspecError(
                f'EachCoordinate({self.dim!r}) drops {self.dim!r} from every source, and the spec declares '
                f'it, so each slice would build a dimension nothing supplies. A coordinate sweep cuts an '
                f'axis the model does not have; to cut one it does, window it.'
            )

    def _slices(self, sources: Mapping[str, Any], key_name: str) -> tuple[list[_Cut], _OriginalIndex | None]:
        """One cut per coordinate, keyed by it. Sources without *dim* pass through.

        No :class:`_OriginalIndex`: nothing was re-indexed, so a slice's frames
        already carry the coordinates they were solved over.
        """
        del key_name
        carrying, coordinates = _coordinates(sources, self.dim, 'slice')
        out: list[_Cut] = []
        for key in coordinates:
            cut = {name: table.filter(pl.col(self.dim) == key).drop(self.dim) for name, table in carrying.items()}
            out.append(_Cut(key, {**sources, **cut}))
        return out, None


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive coordinates of *dim*.

    ``length`` is what the solver sees, ``step`` is what the window keeps, and
    ``length > step`` is overlap. Both count coordinates rather than coordinate
    values, so *dim* need only be orderable — datetimes, strings and gapped
    integers all work. The dimension is re-indexed rather than dropped, into a
    dense ``0..n-1`` column the model addresses by the name ``into`` gives it,
    which the spec has to declare.

    Whether the model *can* be cut this way is asked before it is — the
    coupling, the reach and the overlap they need are
    :meth:`_check_the_program`.
    """

    dim: str
    length: int
    step: int
    into: str

    def cuts(self, sources: Mapping[str, Any]) -> list[tuple[Any, Mapping[str, Any]]]:
        """The ``(key, sources)`` list this axis would run — what ``axis=`` takes hand-built.

        For building one window alone: ``lps.build(spec, axis.cuts(sources)[37][1])``.
        Solved as a list it keys by ``key_name=`` and stitches nothing —
        ``original_index`` is the axis's own.
        """
        return list(self._slices(sources, self._key_name())[0])

    def __post_init__(self) -> None:
        if self.length < 1 or self.step < 1:
            raise ValueError(f'length and step must be positive (got length={self.length}, step={self.step})')
        if self.step > self.length:
            raise ValueError(
                f'step={self.step} exceeds length={self.length}, which would skip coordinates between '
                f'windows. step == length is contiguous; step < length overlaps.'
            )
        if not self.into:
            raise ValueError('into must name the local index the spec declares — it has no default')
        if self.into == self.dim:
            raise ValueError(f'into={self.into!r} must differ from dim — the local index replaces the global one')

    def _key_name(self) -> str:
        """Where the window *started* — never ``dim`` itself.

        A column called ``snapshot`` holding window starts would join against
        real snapshot-indexed data and keep a fraction of it, silently.
        """
        return f'{self.dim}_start'

    def _check_the_program(self, program: Program, sources: Mapping[str, Any]) -> None:
        """Refuse a window the program's rows cannot be whole inside, before one is cut.

        The program answers through
        :attr:`~math_spec.program.Program.separability` and nothing here walks
        it: a window needs ``into`` *windowable*, and
        its lookahead to cover what the rows read ahead. Where a reach is an
        offset the data decides, the parameter's least value is read off the
        data and :meth:`~math_spec.program.Separability.resolved` folds it in,
        so the rule turning a value into a reach stays in the language.

        What the rows read *behind* is not refused: it is what a window's
        first rows meet the edge policy with, which is the rolling-horizon
        seed and the caller's to carry. A position the program counts is
        reported as a warning, every window restarting it.

        Raises:
            LpspecError: ``into`` names no dimension the spec declares, the
                program ties the axis together, a reach turns on a lookup, which
                this driver does not resolve, or the window looks ahead by
                less than its rows read.
            DataError: A parameter deciding a reach has no data.
        """
        if self.into not in program.dimensions:
            raise LpspecError(
                f'EachWindow(into={self.into!r}) names the local index the model addresses a window by, and '
                f'the spec declares no such dimension. ' + did_you_mean(self.into, program.dimensions)
            )
        verdict = program.separability[self.into]
        named = {reach.name for reach in verdict.undecided if reach.kind == 'offset'}
        verdict = verdict.resolved({name: _least(program, sources, name) for name in named})
        if verdict.coupled:
            raise LpspecError(
                f"EachWindow('{self.dim}', …, into='{self.into}') cuts '{self.into}', which the model ties "
                f'together, so no window holds every row whole:\n{_listed(verdict.coupled)}\n'
                f'Each names the change that would lift it.'
            )
        if verdict.undecided:
            raise LpspecError(
                f"EachWindow('{self.dim}', …, into='{self.into}') cuts '{self.into}', which the model reaches "
                f'along through a lookup whose groups a window may cut, and this driver does not resolve a '
                f'reach the lookup decides:\n'
                f'{_listed({r.label: f"through the lookup {r.name!r}" for r in verdict.undecided})}\n'
                f'Cut a dimension the lookup does not group.'
            )
        if self.length - self.step < verdict.ahead:
            raise LpspecError(
                f'EachWindow(length={self.length}, step={self.step}) looks ahead by '
                f'{self.length - self.step} coordinate(s), and the model reads {verdict.ahead} ahead along '
                f"'{self.into}' — a row near a window's end would read past it. Raise length to at least "
                f'step + {verdict.ahead}.'
            )
        if verdict.restarts:
            warnings.warn(
                f"the model counts a position along '{self.into}':\n{_listed(verdict.restarts)}\n"
                f'Every window restarts the count at its first row — what a rolling horizon seeding its '
                f'opening state means, and once per horizon otherwise.',
                LpspecWarning,
                stacklevel=3,
            )

    def _slices(self, sources: Mapping[str, Any], key_name: str) -> tuple[list[_Cut], _OriginalIndex]:
        """One cut per window, keyed by its **first coordinate**.

        Keyed by the coordinate rather than the window's position, which is
        what names a window in the caller's own terms. Sources without *dim*
        pass through untouched.

        The filter leads because it is what a scan can push down; the
        re-indexing that follows is over a frame already cut to one window.

        **A window owns its first ``step`` coordinates**, and the
        :class:`_OriginalIndex` records which — the rest is lookahead the next window
        recomputes. The final window can hold no more than ``step``, its start
        being the last multiple of ``step`` below the end, so the same rule
        keeps all of it and nothing is dropped off the tail.
        """
        carrying, coordinates = _coordinates(sources, self.dim, 'window')
        out: list[_Cut] = []
        owned: list[dict[str, Any]] = []
        for start in range(0, len(coordinates), self.step):
            window = coordinates[start : start + self.length]
            local = {coordinate: position for position, coordinate in enumerate(window)}
            cut = {
                name: (
                    table.filter(pl.col(self.dim).is_in(window))
                    .with_columns(pl.col(self.dim).replace_strict(local, return_dtype=pl.Int64).alias(self.into))
                    .drop(self.dim)
                )
                for name, table in carrying.items()
            }
            out.append(_Cut(window[0], {**sources, **cut, self.into: range(len(window))}))
            owned.extend(
                {key_name: window[0], self.into: position, self.dim: coordinate}
                for position, coordinate in enumerate(window[: self.step])
            )
        return out, _OriginalIndex(self.into, self.dim, pl.DataFrame(owned))


#: What ``axis=`` accepts. A plain list of ``(key, sources)`` is also
#: taken, so an irregular ladder or a hand-built draw needs no third class.
Axis = EachCoordinate | EachWindow


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Runs:
    """What a fold returned: frames keyed by slice, never a scalar.

    :class:`~lpspec.relational.result.Result`'s readers one dimension wider —
    same names, same shapes, the slice key prepended. Nothing is combined
    across slices: each row says which slice computed it. A windowed sweep
    reads over that key unless a reader asks ``original_index=True``, which
    gives the dimension the axis sliced and drops the lookahead rows every
    overlapping window recomputed.
    """

    key_name: str
    #: ``(key, status, termination_condition, objective)``, in slice order —
    #: how every slice terminated, whether or not it produced an answer.
    objective: pl.DataFrame
    #: ``(key, columns, rows, nonzeros, loaded, attach, build, handoff, solve)``,
    #: in slice order — :meth:`~lpspec.api.Model.diagnostics` one dimension
    #: wider, its counts and clocks only. ``loaded`` says the solver took the
    #: model from scratch: under a serial fold the first slice does and the
    #: rest are pushed values, so a later ``True`` is a slice whose data moved
    #: a mask; under an executor every slice builds alone and every one loads.
    #: The clocks are this slice's own seconds per phase, so a slow sweep says
    #: which slice, and which phase of it.
    diagnostics: pl.DataFrame
    #: Per slice, not concatenated. Joining them is the reader's work so a
    #: sweep pays it for the names actually read, and so the concatenated copy
    #: never exists beside the pieces it was built from.
    _primals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _duals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _expressions: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _no_duals: str | None = field(repr=False, default=None)
    _no_expressions: dict[str, str] = field(repr=False, default_factory=dict)
    _original: _OriginalIndex | None = field(repr=False, default=None)

    @classmethod
    def _folded(
        cls, key_name: str, original: _OriginalIndex | None, answered: Generator[tuple[Any, _Answer], None, None]
    ) -> Runs:
        """Every slice's answer absorbed, in the order they arrive.

        The stream is closed here, which is what releases the serial branch's
        model when a fold is abandoned part way; a reason a slice could not
        produce something is kept from the *first* slice that gave one, since
        a later slice's silence is not a second reason.

        Args:
            key_name: What to call the column holding each slice's key.
            original: The way back to the sliced dimension, or ``None`` where
                the axis re-indexed nothing.
            answered: ``(key, answer)`` per slice, in slice order.
        """
        rows: list[dict[str, Any]] = []
        costs: list[dict[str, Any]] = []
        primals: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)
        duals: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)
        expressions: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)
        no_duals: str | None = None
        no_expressions: dict[str, str] = {}
        with closing(answered) as stream:
            for key, answer in stream:
                no_duals = no_duals or answer.no_duals
                for name, reason in answer.no_expressions.items():
                    no_expressions.setdefault(name, reason)
                rows.append({key_name: key, **answer.meta._asdict()})
                costs.append({key_name: key, **answer.cost})
                for into, produced in (
                    (primals, answer.primals),
                    (duals, answer.duals),
                    (expressions, answer.expressions),
                ):
                    for name, frame in produced.items():
                        into[name].append(frame.select(pl.lit(key).alias(key_name), pl.all()))
        return cls(
            key_name=key_name,
            objective=pl.DataFrame(rows),
            diagnostics=pl.DataFrame(costs),
            _primals=dict(primals),
            _duals=dict(duals),
            _expressions=dict(expressions),
            _no_duals=no_duals,
            _no_expressions=no_expressions,
            _original=original,
        )

    @property
    def keys(self) -> list[Any]:
        return self.objective[self.key_name].to_list()

    def _read(
        self, held: Mapping[str, list[pl.DataFrame]], kind: str, name: str, absent: str | None = None
    ) -> pl.DataFrame:
        """*name*'s frames from *held*, concatenated — or why there are none.

        *absent* is a reason the fold already knows, which beats one derived
        from what the sweep happens to hold.
        """
        if name not in held:
            raise LpspecError(absent or _nothing_to_read(kind, name, held, self.objective))
        return pl.concat(held[name])

    def primal(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One variable's values across every slice, the slice key prepended.

        A slice that reached no solution contributes no rows, so this can be
        shorter than the sweep; :attr:`objective` is one row per slice always.

        Args:
            name: A variable the sweep's spec declares.
            original_index: Read over the dimension the axis sliced instead of
                over the slice key.

        Raises:
            LpspecError: No slice of the sweep produced *name*.
        """
        return self._reindexed(self._read(self._primals, 'variable', name), original_index=original_index)

    def dual(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One constraint's shadow prices across every slice, the key prepended.

        :meth:`primal`'s shape and arguments. A slice whose model had an
        integer variable contributes no duals; over the original index each
        coordinate carries the price of the window that owns it, never a blend
        of several.

        Raises:
            LpspecError: No slice produced duals for *name* — the message says
                which of the two it was.
        """
        return self._reindexed(
            self._read(self._duals, 'constraint', name, self._no_duals), original_index=original_index
        )

    def expression(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One named expression's values across every slice, the slice key prepended.

        :meth:`primal`'s shape and arguments, for the quantities the spec
        declares under ``expressions:`` — each slice's value was evaluated at
        that slice's solution when the fold read it.

        Over the original index each coordinate carries the value of the window
        that owns it — the recomputed lookahead rows are dropped, which is what
        makes summing the stitched frame safe where summing per-window values
        double-counts.

        Raises:
            LpspecError: No slice produced *name* — an evaluation that failed
                on every slice carries its own reason — or ``original_index``
                on a quantity reduced over the sliced dimension.
        """
        return self._reindexed(
            self._read(self._expressions, 'named expression', name, self._no_expressions.get(name)),
            original_index=original_index,
        )

    def _reindexed(self, frame: pl.DataFrame, *, original_index: bool) -> pl.DataFrame:
        """*frame* over the dimension the axis sliced, rather than over its slices.

        Every axis answers it: :class:`EachCoordinate` and a hand-built axis
        re-indexed nothing — their key column already *is* a coordinate of the
        answer — so there the frame comes back unchanged, a satisfied request
        rather than an ignored one.
        """
        if not original_index or self._original is None:
            return frame
        return self._original.restore(frame, self.key_name)

    def to_pandas(self, name: str, *, original_index: bool = False) -> pd.DataFrame:
        """:meth:`primal` as a tidy :class:`pandas.DataFrame`.

        The name is resolved before pandas is imported, so a sweep that never
        held *name* says so on any install.
        """
        return tidy_to_pandas(self.primal(name, original_index=original_index))

    def to_dataarray(self, name: str, *, original_index: bool = False) -> xr.DataArray:
        """:meth:`primal` as a :class:`xarray.DataArray`, the slice key a dimension.

        The extra dimension is named by the axis — a scenario sweep gives
        ``(scenario, …)`` and a window ``(<dim>_start, …)``. A slice that
        reached no solution has no rows and comes back NaN, the same answer a
        masked coordinate gets from ``Result``. ``original_index=True`` gives
        the array over the dimension the axis sliced instead, so a rolling
        horizon comes back indexed by time.
        """
        return tidy_to_dataarray(self.to_pandas(name, original_index=original_index), name)

    def to_dataset(self, *names: str) -> xr.Dataset:
        """Kept variables as one :class:`xarray.Dataset`; all of them by default.

        Costs more than ``Result``'s does — each variable arrives dense over
        its own dims *and* over every slice. Name the few you need, or use
        :meth:`to_parquet`.

        No ``original_index``: this and :meth:`to_parquet` export what the
        sweep *holds*, and the original index is lossy — a bulk export is the
        wrong place to drop the lookahead rows.

        Raises:
            LpspecError: The sweep holds no variable values at all.
        """
        return tidy_to_dataset(names or self._variables_held(), self.to_dataarray)

    def to_parquet(self, directory: str | Path) -> dict[str, Path]:
        """One parquet file per variable the sweep holds, ``(key, dims…, value)``.

        Written in :meth:`primal`'s order, so the same sweep writes the same
        bytes.

        Returns:
            Each variable's name, mapped to the file it was written to.

        Raises:
            LpspecError: The sweep holds no variable values at all.
        """
        held = self._variables_held()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name in held:
            path = directory / f'{name}.parquet'
            self.primal(name).write_parquet(path)
            written[name] = path
        return written

    def _variables_held(self) -> tuple[str, ...]:
        """Every variable some slice produced, sorted — what a bulk export writes.

        Raises:
            LpspecError: No slice produced any, which both exports refuse
                rather than writing an empty directory or an empty dataset.
        """
        if not self._primals:
            raise LpspecError(_nothing_to_read('variable', 'anything', self._primals, self.objective))
        return tuple(sorted(self._primals))

    def __len__(self) -> int:
        return self.objective.height


def _nothing_to_read(kind: str, name: str, held: Mapping[str, object], objective: pl.DataFrame) -> str:
    """Why *name* has no frame.

    A sweep keeps everything every slice produced, so a declared name arrives
    here only when no slice produced it; an undeclared name arrives here too,
    and the two are told apart by what the sweep did hold.
    """
    conditions = ', '.join(sorted(set(objective['termination_condition'].to_list())))
    if held:
        listed = ', '.join(repr(k) for k in sorted(held))
        return (
            f'no {kind} {name!r} in this sweep — it holds {listed}. '
            f'If the spec declares it, no slice produced one: all {objective.height} terminated {conditions}.'
        )
    return (
        f'this sweep holds no {kind} frames at all — every one of its {objective.height} slices '
        f'terminated {conditions}. The fold ran; the models did not solve. '
        f'runs.objective carries the status of each slice.'
    )


def solve_over(
    spec: Buildable,
    sources: Mapping[str, Any],
    axis: Axis | Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    carry: Mapping[str, tuple[str, int | None]] | None = None,
    key_name: str | None = None,
    executor: Any = None,
    workers_share_fs: bool | None = None,
    solver_options: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
    keep: Keep = 'solver',
) -> Runs:
    """Solve *spec* once per slice of *axis* and fold the answers together.

    The rules — what a carry copies, how the key column is named, which
    executor to choose — are [docs/reference/sweeps.md](../../docs/reference/sweeps.md).

    Args:
        spec: As :func:`~lpspec.api.check` takes it. Parsed once, whichever
            executor runs the slices.
        sources: As :func:`~lpspec.api.build` takes them, every shape
            included; the axis filters the tables that carry it and passes
            the rest through.
        axis: :class:`EachCoordinate`, :class:`EachWindow`, or a list of
            ``(key, sources)`` cut by hand.
        carry: ``{parameter: (variable, index)}`` — one slice's answer copied
            into the next slice's data. The first slice takes the parameter
            from *sources*, its seed.
        key_name: What to call the slice column; a class axis names its own,
            a hand-built list has to be told.
        executor: Any :class:`concurrent.futures.Executor`; ``None`` runs the
            slices in order on one model. A process pool must be ``spawn``
            or ``forkserver`` — a forked worker hangs.
        workers_share_fs: Whether the executor's workers can read this
            process's paths. Decided for the stdlib pools; anything else is
            assumed not to, and paths travel as bytes.
        solver_options: As :meth:`~lpspec.api.Model.solve` takes them.
        solver_name: As :meth:`~lpspec.api.Model.solve` takes it.
        keep: As :meth:`~lpspec.api.Model.solve` takes it, reaching every
            slice. Under an executor every slice is a first solve and keeps
            nothing, whatever was asked.

    Returns:
        Every slice's answers, keyed by slice.

    Raises:
        LpspecError: A carry that cannot line up, has no seed, reads a
            coordinate the next window recomputes, or is asked together with
            an executor; a key that collides with a column the frames carry;
            a ``Program`` handed to an executor that crosses a process; an
            axis the program does not allow. All refused before a slice is
            cut, and every one answerable from the declarations before a
            source is read.
        DataError: No source carries the axis, or the axis produced no
            slices.

    Warns:
        LpspecWarning: A source carrying the axis that is short of a
            coordinate another has — that slice builds it empty — or a
            position the model counts, which every window restarts.
    """
    if isinstance(spec, Program) and executor is not None and _crosses_a_process(executor):
        raise LpspecError(
            'a Program cannot cross a process: pass the spec as the path or mapping it was lowered '
            'from, and each worker lowers it itself.'
        )
    if carry and executor is not None:
        raise LpspecError(
            'carry and executor are mutually exclusive: a carried value makes slice i+1 depend on '
            "slice i's answer, so the slices cannot run concurrently. Drop the executor, or drop the carry."
        )
    parsed, program = _parsed(spec)
    plan = {p: _CarryRule.resolved(program, p, v, i) for p, (v, i) in (carry or {}).items()}
    key_name = _key_column(axis, key_name, program)

    if isinstance(axis, (EachCoordinate, EachWindow)):
        _check_the_carry(plan, axis, sources)
        axis._check_the_program(program, sources)
        cut, original = axis._slices(sources, key_name)
    else:
        cut, original = list(axis), None
        _check_the_carry(plan, axis, cut[0][1] if cut else {})
    cuts = [_Cut(*entry) for entry in cut]
    if not cuts:
        raise DataError('the axis produced no slices')
    solving = {'solver_name': solver_name, 'solver_options': dict(solver_options or {}) or None}
    answered = (
        _serially(program, cuts, solving, plan, keep)
        if executor is None
        else _pooled(executor, workers_share_fs, program, parsed, cuts, solving)
    )
    return Runs._folded(key_name, original, answered)


def _parsed(spec: Buildable) -> tuple[Spec | Program, Program]:
    """*spec* read once: what a worker across a process is handed, and the program this process runs.

    The copy is taken before the lowering, which caches its expansion on the
    instance it is given — a cache that does not pickle, where a fresh
    ``Spec`` does.
    """
    if isinstance(spec, Program):
        return spec, spec
    parsed = to_spec(spec)
    return parsed.model_copy(), check(parsed)


def _check_the_carry(plan: Mapping[str, _CarryRule], axis: Any, first: Mapping[str, Any]) -> None:
    """Refuse a carry with no seed, or one that reads what the next window recomputes.

    Both are answered before a source is read: the seed is a key of the first
    slice's sources, and the lookahead is arithmetic on the window. A carry
    at ``index >= step`` hands forward a row the next window solves again,
    which is never the state at the seam.
    """
    for parameter, rule in plan.items():
        if parameter not in first:
            raise LpspecError(
                f'carry writes {parameter!r} from the second slice on, and the first slice has nothing to '
                f'start from: supply {parameter!r} in sources as the seed.'
            )
        if not (isinstance(axis, EachWindow) and rule.dropped == axis.into and rule.index is not None):
            continue
        if rule.index >= axis.length:
            raise LpspecError(
                f'carry {parameter!r} <- ({rule.variable!r}, {rule.index}) is out of range: a window of '
                f'length {axis.length} holds {axis.into} 0..{axis.length - 1}.'
            )
        if rule.index >= axis.step:
            raise LpspecError(
                f'carry {parameter!r} <- ({rule.variable!r}, {rule.index}) reads {axis.into} == {rule.index}, '
                f'which is in the lookahead: EachWindow(length={axis.length}, step={axis.step}) keeps '
                f'{axis.into} 0..{axis.step - 1} and the next window recomputes the rest. The state at the '
                f'seam is the last coordinate kept, {axis.step - 1}.'
            )


def _serially(
    program: Program,
    cuts: Sequence[_Cut],
    solving: Mapping[str, Any],
    plan: Mapping[str, _CarryRule],
    keep: Keep,
) -> Generator[tuple[Any, _Answer], None, None]:
    """Each slice's answer, off one model updated in place.

    Every slice of a sweep is the same math over different numbers, which is
    what :meth:`~lpspec.api.Model.update` is for; a rebuild releases the
    previous model before it starts, so the fold holds one slice's model
    however many there are.

    **A slice that names something else is rebuilt, not updated.** A cut is
    *total* where ``update`` is partial by construction: the two agree only
    while every slice names the same sources, which the class axes guarantee
    and a hand-built list does not. Compared by *name* — values are what a
    update exists to replace.

    **A generator because of the carry**: slice ``i+1``'s sources are not
    known until slice ``i``'s frames have been read, and resuming after the
    yield is where that happens. The caller closes this — that is what
    releases the model when a fold is abandoned part way.
    """
    model: Model | None = None
    named: frozenset[str] | None = None
    state: dict[str, Any] = {}
    try:
        for position, cut in enumerate(cuts):
            sources = {**cut.sources, **state}
            names = frozenset(sources)
            with _named_slice(cut.key, position, len(cuts)):
                if model is not None and names == named:
                    before = model.diagnostics()
                    model.update(sources)
                else:
                    if model is not None:
                        model.close()
                    model, named, before = build(program, sources), names, None
                result = model.solve(**solving, keep=keep)
                answer = _answers(result, program, _slice_cost(model.diagnostics(), before))
            yield cut.key, answer
            if plan and position < len(cuts) - 1:
                if not answer.primals:
                    raise LpspecError(
                        f'slice {cut.key!r} ({position + 1} of {len(cuts)}) terminated '
                        f'{answer.meta.termination_condition}, so slice {cuts[position + 1].key!r} has no '
                        f'{sorted({rule.variable for rule in plan.values()})} to start from. A carried sweep '
                        f'stops at the first slice that leaves nothing to carry; the {position} before it solved.'
                    )
                state = {p: rule.value_from(answer.primals, p, cut.key) for p, rule in plan.items()}
    finally:
        if model is not None:
            model.close()


@contextmanager
def _named_slice(key: Any, position: int, count: int) -> Generator[None, None, None]:
    """Whatever a slice raises leaves naming the slice, as a note on the exception.

    A note rather than a new message: the error stays the engine's own, so a
    caller matching on it still matches, and the traceback of a fifty-window
    sweep says which window without anyone counting.
    """
    try:
        yield
    except Exception as exc:
        exc.add_note(f'in slice {key!r} ({position + 1} of {count})')
        raise


def _pooled(
    executor: Any,
    workers_share_fs: bool | None,
    program: Program,
    parsed: Spec | Program,
    cuts: Sequence[_Cut],
    solving: Mapping[str, Any],
) -> Generator[tuple[Any, _Answer], None, None]:
    """The same, from slices built independently and possibly elsewhere.

    Yielded in **slice order, never completion order**: the futures are walked
    in the order they were submitted, so a sweep cannot reorder itself under a
    pool. A built model cannot cross a process, so this branch builds per
    slice — the same fact that makes ``carry`` and ``executor`` mutually
    exclusive. A worker in this process is handed the lowered program; one
    across a boundary the validated model, which pickles where a program
    does not, and lowers it itself. Neither reads the YAML again.
    """
    crosses = _crosses_a_process(executor)
    shared = _shares_filesystem(executor, workers_share_fs)
    call = dict(solving)
    memo: dict[str, tuple[Any, Any]] = {}
    futures = [
        executor.submit(
            _run_slice,
            parsed if crosses else program,
            _encode(cut.sources, memo, workers_share_fs=shared) if crosses else dict(cut.sources),
            crosses,
            call,
        )
        for cut in cuts
    ]
    for position, (cut, future) in enumerate(zip(cuts, futures, strict=True)):
        with _named_slice(cut.key, position, len(cuts)):
            answer = future.result()
        yield (
            cut.key,
            replace(
                answer,
                primals=_decode(answer.primals),
                duals=_decode(answer.duals),
                expressions=_decode(answer.expressions),
            ),
        )


def _answers(result: Any, program: Program, cost: dict[str, Any]) -> _Answer:
    """One slice's answer, read out of *result*: its meta row, its cost, and its frames.

    Read here rather than held, so that what a sweep accumulates is frames and
    never results — holding a result per slice would hold that slice's label
    frames with it. Every declared expression is evaluated here,
    eagerly, for the same reason: the deferred reader holds the build's frames.

    **A slice that answered nothing is not a failure**, and neither is one
    whose duals are undefined: an integer variable makes them so, and one such
    slice must not fail a whole sweep. ``Result.dual`` already writes the
    sentence saying why, so it is caught and carried rather than rewritten.
    """
    meta = _SliceMeta(
        status=result.status,
        termination_condition=result.termination_condition,
        objective=result.objective if result.has_primal else float('nan'),
    )
    if not result.has_primal:
        return _Answer(meta, cost, {}, {}, {}, None, {})
    primals = {name: result.primal(name) for name in program.variables}
    expressions: dict[str, pl.DataFrame] = {}
    no_expressions: dict[str, str] = {}
    for name in program.named_expressions:
        try:
            expressions[name] = result.expression(name)
        except LpspecError as exc:
            no_expressions[name] = str(exc)
    try:
        duals = {name: result.dual(name) for name in program.constraints}
        return _Answer(meta, cost, primals, duals, expressions, None, no_expressions)
    except LpspecError as exc:
        return _Answer(meta, cost, primals, {}, expressions, str(exc), no_expressions)


def _run_slice(
    spec: Spec | Program,
    encoded: dict[str, Any],
    encode_out: bool,
    call: dict[str, Any],
) -> _Answer:
    """One slice, start to finish, over plain data — the *pooled* branch.

    Module-level and closure-free on purpose: a remote executor has to pickle
    what it is handed, and a bound method or a lambda over the axis object
    cannot cross.
    """
    program = to_program(spec)
    with build(program, _decode(encoded)) as model, model.solve(**call) as result:
        answer = _answers(result, program, _slice_cost(model.diagnostics(), None))
        if not encode_out:
            return answer
        return replace(
            answer,
            primals=_encode(answer.primals, {}),
            duals=_encode(answer.duals, {}),
            expressions=_encode(answer.expressions, {}),
        )


def _key_column(
    axis: Axis | Sequence[tuple[Any, Mapping[str, Any]]],
    key_name: str | None,
    program: Program,
) -> str:
    """What to call the column holding the slice key.

    Two rules, both the caller's rather than any axis's: an axis that cannot
    name its own key has to be told, and no key may be a column the frames
    already carry — a dimension the spec declares, or one of the fixed names
    every reader and :attr:`Runs.objective` use. What a class axis calls its
    key when it is not told is :meth:`EachCoordinate._key_name` and
    :meth:`EachWindow._key_name`.

    Raises:
        LpspecError: A hand-built axis with no ``key_name``, a name the spec
            declares as a dimension, or a fixed column's name.
    """
    if key_name is None:
        if not isinstance(axis, (EachCoordinate, EachWindow)):
            raise LpspecError(
                'a hand-built axis needs key_name=: a list of cuts does not say what its keys are '
                "coordinates of, and 'slice' would be this library naming your axis for you. Pass "
                "key_name='draw', key_name='period', or whatever the keys actually are."
            )
        key_name = axis._key_name()
    if key_name in program.dimensions:
        raise LpspecError(
            f'key_name={key_name!r} is a dimension the spec declares, so the slice key would collide '
            f'with a column the frames already carry. Name it something the spec does not use.'
        )
    fixed = ('value', *_SliceMeta._fields)
    if key_name in fixed:
        raise LpspecError(
            f'key_name={key_name!r} is a column every sweep frame carries ({", ".join(fixed)}), so the slice '
            f'key would replace it rather than sit beside it. Name it something else.'
        )
    return key_name


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------


def _shares_filesystem(executor: Any, declared: bool | None) -> bool:
    """Whether *executor*'s workers can read this process's paths.

    The two stdlib pools are the ones whose deployment is knowable: both run
    here, so both read the paths here. An executor this package did not ship is
    a transport it cannot ask, so it is assumed remote until *declared* says
    otherwise.
    """
    if declared is not None:
        return declared
    return isinstance(executor, ProcessPoolExecutor)


def _crosses_a_process(executor: Any) -> bool:
    """Whether a slice's sources have to be encoded to reach *executor*.

    A thread pool runs in this process, so encoding would be a parquet round
    trip for a boundary that is not there. Every other executor is assumed to
    cross, none of them being answerable.
    """
    return not isinstance(executor, ThreadPoolExecutor)


def _encode(
    sources: Mapping[str, Any], memo: dict[str, tuple[Any, Any]], *, workers_share_fs: bool = False
) -> dict[str, Any]:
    """Sources in the shape a worker can be handed.

    A path the workers can reach stays a path. A path they cannot travels as
    **its own bytes, untouched** — decoding and re-encoding a parquet file
    produces byte-identical output for 79x the CPU (#459).
    A table held in memory is written to parquet, which beats pickling the
    frame on size and time; a source that is not a table — a number, a map,
    a bare sequence — crosses as itself.

    *memo* keeps a source no slice rewrote — the static tables, which is most
    of them — from being encoded once per slice. ``bytes`` is what
    :func:`_decode` reads back, and cannot be confused with a path.
    """
    out: dict[str, Any] = {}
    for name, obj in sources.items():
        is_path = isinstance(obj, (str, Path))
        if is_path and workers_share_fs:
            out[name] = obj
            continue
        cached = memo.get(name)
        if cached is not None and cached[0] is obj:
            out[name] = cached[1]
            continue
        table = None if is_path else as_frame(obj)
        if is_path:
            out[name] = Path(obj).read_bytes()
        elif table is None:
            out[name] = obj
        else:
            buffer = io.BytesIO()
            table.collect().write_parquet(buffer, compression=_COMPRESSION)
            out[name] = buffer.getvalue()
        memo[name] = (obj, out[name])
    return out


def _decode(encoded: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of :func:`_encode`, and a pass-through for what never crossed.

    Called on every returned frame rather than only the encoded ones: a frame
    that stayed in this process is not ``bytes`` and comes back untouched, so
    the caller needs no branch and the two paths cannot answer differently.
    """
    return {name: pl.read_parquet(io.BytesIO(v)) if isinstance(v, bytes) else v for name, v in encoded.items()}


# ---------------------------------------------------------------------------
# reading a source without attaching it
# ---------------------------------------------------------------------------


def _table(obj: Any) -> pl.LazyFrame | None:
    """One source as a lazy frame — a scan for a path, so a filter pushes down.

    ``None`` for a source that is not a table — a number, a ``{label: value}``
    map, a bare sequence — which carries no column and so no axis, and passes
    through every cut as it is.
    """
    if isinstance(obj, (str, Path)):
        return pl.scan_parquet(obj)
    return as_frame(obj)


def _coordinates(sources: Mapping[str, Any], dim: str, verb: str) -> tuple[dict[str, pl.LazyFrame], list[Any]]:
    """The sources a slice has to filter, by name, and the ordered coordinates to cut.

    *carrying* is derived rather than declared: a source that carries the slice
    key and is *not* filtered produces a duplicate-coordinate error at attach
    time, so the derivation cannot silently miss one.

    The coordinates are sorted as **values of the column**, so a window is a
    span of those and never of the numbers in them.

    Raises:
        DataError: No source carries *dim*.

    Warns:
        LpspecWarning: A source carrying *dim* is short of a coordinate
            another has. The slice there builds it empty, and an absent row
            reads as zero — which is how a model masks, and so is reported
            rather than refused, the way the engine reports sparsity.
    """
    tables = {name: table for name, obj in sources.items() if (table := _table(obj)) is not None}
    carrying = {name: table for name, table in tables.items() if dim in table.collect_schema().names()}
    if not carrying:
        raise DataError(
            f"no source carries a '{dim}' column, so there is nothing to {verb} over. "
            f'EachCoordinate names a column the data has; a span of consecutive coordinates is EachWindow.'
        )
    held = {name: set(table.select(pl.col(dim).unique()).collect()[dim]) for name, table in carrying.items()}
    coordinates = sorted(set().union(*held.values()))
    for name, mine in held.items():
        if missing := sorted(set(coordinates) - mine):
            other = next(o for o, theirs in held.items() if missing[0] in theirs)
            warnings.warn(
                f"'{name}' has no rows for {dim} {missing[0]!r}, which '{other}' has, so the slice at "
                f"{missing[0]!r} builds '{name}' empty and every row of it reads as absent. Supply the rows "
                f'if a value was meant; where the absence is, this is the record of it.',
                LpspecWarning,
                stacklevel=4,
            )
    return carrying, coordinates
