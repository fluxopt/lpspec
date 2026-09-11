"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a *process* may loop over plans
(math-spec's docs/about/limits.md). So a strategy is a driver above :mod:`lpspec.api`,
built from the public verbs — never a language or engine feature.

Every strategy is the same fold: **partition → attach → solve → carry → stitch**.
Only how the sources are sliced and whether the slices couple differs. A serial fold builds
once and updates each slice (:func:`_serially`); under a process pool it builds
per slice (:func:`_pooled`), a built model being the one thing that cannot
cross. Both yield an :class:`_Answer`, and the fold that absorbs them is
written once.

    scenario / sweep    ``EachCoordinate('scenario')``            independent
    myopic pathway      ``EachCoordinate('period')``              + ``carry``
    rolling horizon     ``EachWindow('snapshot', steps=24, lookahead=24, into='t')``  + ``carry``

**A partition is a filter on the sources, not a narrower index** — the
containment check refuses parameter rows outside a narrowed index, so an axis
rewrites the rows and the index together.

The caller-facing rules are [docs/reference/sweeps.md](../../docs/reference/sweeps.md).
"""

from __future__ import annotations

import io
import json
import warnings
from collections import defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import polars as pl

from lpspec.api import build, check
from lpspec.errors import DataError, LayoutError, LpspecError, LpspecWarning, did_you_mean
from lpspec.frames import as_frame
from lpspec.relational.parquet import (
    KINDS,
    LABELS,
    RECORD_SCHEMA,
    Record,
    check_format,
    read_reasons,
    reader_kind,
    write_format,
    write_reasons,
    write_whole,
)
from lpspec.relational.result import tidy_to_dataarray, tidy_to_dataset, tidy_to_pandas
from lpspec.sources import least_value

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    import pandas as pd
    import xarray as xr
    from math_spec.program import Program

    from lpspec.api import Model
    from lpspec.lanes import Buildable, Label, Source
    from lpspec.relational.result import Diagnostics, Keep, Result

#: A frame lazy or not, going in and coming back out the same way.
_Frame = TypeVar('_Frame', pl.DataFrame, pl.LazyFrame)

#: Parquet rather than pickle, and not a knob: zstd measured smaller *and*
#: faster than pickling the frame, on compressible and incompressible data
#: alike (#459).
_COMPRESSION = 'zstd'


class _Slice(NamedTuple):
    """One slice of a sweep: the key, the sources that build it, and what it owns.

    A tuple on purpose: a hand-built axis is a plain list of ``(key, sources)``,
    and those unpack the same way — which is why ``owns`` has a default.

    ``owns`` is how many coordinates of the re-indexed dimension this slice is
    responsible for, the rest being lookahead the next slice recomputes. It is
    what a ``carry`` reads the seam off, and ``None`` on an axis that re-indexed
    nothing, where a carry cannot drop a dimension at all.
    """

    key: Label
    sources: Mapping[str, Source]
    owns: int | None = None


#: What a spilled sweep carries beside its frames: the manifest saying whose
#: sweep the directory is, and the coordinates each window owns. Named here
#: because :class:`_Spill` writes them and :func:`load_runs` reads them back.
_MANIFEST_FILE = 'sweep.json'
_OWNED_FILE = 'owned.parquet'

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

    ``dropped`` is the one dimension the carry collapses, and ``None`` where the
    whole frame moves forward. Which coordinate of it is handed on is not
    recorded, because it is not the caller's to choose: a carry exists to meet
    the next slice at the seam, so the coordinate is the last one this slice
    owns, and only the axis knows which that is.
    """

    variable: str
    dropped: str | None

    @classmethod
    def resolved(cls, program: Program, parameter: str, variable: str) -> _CarryRule:
        """One carry checked against the plan — construction and validation, together.

        The variable's dims minus the parameter's is the one dimension the
        carry collapses; everything else passes through, so a myopic pathway
        hands a whole capacity vector forward rather than one number at a time.
        Nothing here reads data, which is why the plan resolves before the axis
        slices any. Whether the dropped dimension is one the axis can answer for
        is :func:`_check_the_carry`'s, which is where the axis is.
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
                f'{source} and {parameter!r} over {over}. A carry collapses the one dimension the sweep '
                f'advances along, so reduce the others in the YAML — a derived variable is where the oracle '
                f'can see the math.'
            )
        return cls(variable, dropped[0] if dropped else None)

    def value_from(
        self, frames: Mapping[str, pl.DataFrame], parameter: str, key: Label, owns: int | None
    ) -> pl.DataFrame:
        """What this rule hands the next slice, read out of one slice's primals.

        *owns* is how many coordinates of the dropped dimension this slice is
        responsible for, so the coordinate handed on is ``owns - 1`` — the last
        one kept rather than the last one solved. Those differ under an
        overlapping window, and the lookahead row is never the state at the
        seam.

        Raises:
            LpspecError: The slice built no row of the variable there — a
                ``where`` or an absence rule took it.
        """
        frame = frames[self.variable]
        if self.dropped is None:
            return frame
        assert owns is not None, 'a carry that drops a dimension is refused unless the axis owns it'
        seam = owns - 1
        picked = frame.filter(pl.col(self.dropped) == seam).drop(self.dropped)
        if picked.is_empty():
            raise LpspecError(
                f'carry {parameter!r} <- {self.variable!r} has nothing to copy: slice {key!r} built no '
                f'{self.variable!r} at {self.dropped} == {seam}, the last coordinate it owns, so there is no '
                f'value there to hand forward.'
            )
        return picked


@dataclass(frozen=True)
class _Answer:
    """One slice, solved and read out — what the fold absorbs.

    What :func:`_serially` and :func:`_pooled` both produce. Plain data
    throughout, because a worker returns it and so it has to pickle: frames,
    strings and numbers, never a result or a model.
    """

    meta: Record
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
    *responsible* for — the coordinates its block names, the rest being lookahead the next
    window recomputes. One-way: the lookahead rows are not in it, so a sliced
    frame cannot be rebuilt from it — slicing stays
    :meth:`EachWindow._slice`'s business.
    """

    local: str
    dim: str
    owned: pl.DataFrame

    def restore(self, frame: _Frame, key_name: str) -> _Frame:
        """*frame* over the dimension the axis sliced, rather than over its slices.

        The inner join against ``owned`` is the whole operation: it restores
        the original coordinate, and because a coordinate may appear only once
        under its own index, the lookahead rows have nowhere to go. Sorted on
        the restored dimension, so a sweep reads back in the caller's order.
        Lazy in, lazy out: a spilled sweep stitches the same way.

        Raises:
            LpspecError: *frame* has no ``local`` column — the quantity was
                reduced over the sliced dimension, so there is no way back.
        """
        columns = frame.collect_schema().names()
        if self.local not in columns:
            raise LpspecError(
                f'cannot read this over {self.dim!r}: the frame has no {self.local!r} column, because the '
                f'quantity was reduced over the sliced dimension — each row already covers a whole slice, '
                f'lookahead included under an overlapping window. Read it without original_index for the '
                f'per-slice values, or read a quantity that keeps {self.local!r} and aggregate the '
                f'stitched frame.'
            )
        keys = [key_name, self.local]
        rest = [column for column in columns if column not in (*keys, 'value')]
        restored = frame.lazy().join(self.owned.lazy(), on=keys, how='inner').drop(keys)
        stitched = restored.select(self.dim, *rest, 'value').sort(self.dim, *rest)
        return stitched if isinstance(frame, pl.LazyFrame) else stitched.collect()  # pyrefly: ignore[bad-return]  — the branch matches the frame's own kind


def _one_key_type(keys: Sequence[Label], key_name: str) -> pl.DataType:
    """The type every file writes *key_name* as, settled over the whole sweep.

    Refused rather than coerced where the keys disagree: an ``int`` beside a
    ``float`` would widen every key to a float, so a sweep keyed 1 and 2 would
    read back keyed 1.0 and 2.0 — the caller's own labels, changed to make the
    files line up.

    Raises:
        LpspecError: The keys are of more than one type.
    """
    try:
        return pl.Series(keys).dtype
    except TypeError as mixed:
        kinds = sorted({type(key).__name__ for key in keys})
        raise LpspecError(
            f'the keys of this sweep are of more than one type ({", ".join(kinds)}), so its files could not '
            f'all write {key_name!r} as one. Every file carries the key, and a column that changes type '
            f'between them cannot be concatenated or loaded into one table. Key the slices consistently.'
        ) from mixed


def _keyed(frame: pl.DataFrame, key_name: str, key: Label, dtype: pl.DataType) -> pl.DataFrame:
    """*frame* with the slice key prepended — the shape every reader returns.

    The literal takes *dtype* rather than ``pl.lit``'s own, which reads a
    Python int as ``Int32`` where every dict-built frame here reads it as
    ``Int64``. A sweep whose record and whose frames disagree about the type
    of its own key still joins in polars and still casts in duckdb, but cannot
    be concatenated or loaded into one typed table — and the files outlive the
    process that could paper over it.

    *dtype* is the whole sweep's, never this key's: keys of ``[1, 2.5]`` infer
    per file to ``Int64`` and ``Float64``, which is the same split one file
    later, so the type has to be settled over the keys before any of them is
    written.
    """
    return frame.select(pl.lit(key, dtype=dtype).alias(key_name), pl.all())


@dataclass(frozen=True)
class _Spill:
    """A sweep's answers on disk instead of in memory, one file per slice and name.

    Under ``directory``: ``<kind>/<name>/<position>.parquet`` for the frames,
    the slice key a column of each; ``objective/`` and ``diagnostics/`` for
    the record, one row per position. Every file lands under its final name
    only whole, and the objective file is written last: it is what marks a
    slice done, so one interrupted part way is solved again rather than read
    back short. ``sweep.json`` names the key and the keys, so a directory
    answers for one sweep and another pointed at it is refused; it also
    carries what :func:`load_runs` cannot infer from the frames — whether the
    axis was hand-built, and the dimension a window sliced, whose owned
    coordinates go beside it in ``owned.parquet``.
    """

    directory: Path
    key_name: str
    #: What every file writes the key column as — settled over the sweep's
    #: keys by whoever opened this, never inferred per file.
    key_dtype: pl.DataType

    @classmethod
    def opened(
        cls,
        directory: str | Path,
        key_name: str,
        keys: Sequence[Label],
        key_dtype: pl.DataType,
        original: _OriginalIndex | None = None,
        hand_built: bool = False,
    ) -> _Spill:
        """The directory ready to take this sweep, or refused as another's.

        A directory already holding a sweep is **checked**, never re-stamped:
        resuming into one an earlier build wrote would otherwise overwrite the
        layout it is in and mix two under one manifest, which is the one
        failure the stamp exists to catch. Only a directory that holds no
        sweep yet is stamped, and it is stamped with the manifest.

        Raises:
            LayoutError: The directory holds a sweep in another layout.
            LpspecError: The directory holds a sweep keyed differently, or
                over other keys.
        """
        directory = Path(directory)
        manifest: dict[str, Any] = {
            'key_name': key_name,
            'keys': [str(key) for key in keys],
            'hand_built': hand_built,
            'original': None if original is None else {'local': original.local, 'dim': original.dim},
        }
        record = directory / _MANIFEST_FILE
        if record.exists():
            check_format(directory)
            found = json.loads(record.read_text())
            if found != manifest:
                raise LpspecError(
                    f'{str(directory)!r} holds a sweep keyed by {found["key_name"]!r} over {found["keys"]}, and '
                    f'this one is keyed by {key_name!r} over {manifest["keys"]}. A directory holds one sweep: '
                    f'point to= at an empty one, or delete this one to solve it again.'
                )
        else:
            write_format(directory)
            record.write_text(json.dumps(manifest))
            if original is not None:
                write_whole(original.owned, directory / _OWNED_FILE)
        return cls(directory, key_name, key_dtype)

    def _file(self, kind: str, position: int, name: str | None = None) -> Path:
        under = self.directory / kind if name is None else self.directory / kind / name
        return under / f'{position:06d}.parquet'

    def done(self, position: int) -> bool:
        return self._file('objective', position).exists()

    def write(self, position: int, key: Label, answer: _Answer) -> _Answer:
        """*answer*'s frames and record on disk, and the answer with the frames released."""
        for kind, produced in zip(KINDS, (answer.primals, answer.duals, answer.expressions), strict=True):
            for name, frame in produced.items():
                write_whole(_keyed(frame, self.key_name, key, self.key_dtype), self._file(kind, position, name))
        write_whole(pl.DataFrame([{self.key_name: key, **answer.cost}]), self._file('diagnostics', position))
        write_whole(
            pl.DataFrame([{self.key_name: key, **answer.meta._asdict()}], schema_overrides=RECORD_SCHEMA),
            self._file('objective', position),
        )
        return replace(answer, primals={}, duals={}, expressions={})

    def read_back(self, position: int) -> _Answer:
        """A done slice's record — meta and cost — with no frames, which stay on disk."""
        row = pl.read_parquet(self._file('objective', position)).drop(self.key_name).row(0, named=True)
        cost = pl.read_parquet(self._file('diagnostics', position)).drop(self.key_name).row(0, named=True)
        return _Answer(Record(**row), dict(cost), {}, {}, {}, None, {})

    def primals(self, position: int, names: Iterable[str]) -> dict[str, pl.DataFrame]:
        """The named primals a done slice wrote, for a carry to read; a name it did not write is absent."""
        found = {name: self._file('primal', position, name) for name in names}
        return {name: pl.read_parquet(path).drop(self.key_name) for name, path in found.items() if path.exists()}

    def held(self, kind: str) -> list[str]:
        under = self.directory / kind
        return sorted(path.name for path in under.iterdir()) if under.is_dir() else []

    def scan(self, kind: str, name: str) -> pl.LazyFrame | None:
        """Every slice's frame of *name*, lazily and in slice order, or ``None`` where no slice wrote one."""
        under = self.directory / kind / name
        return pl.scan_parquet(sorted(under.glob('*.parquet'))) if under.is_dir() else None


def _listed(entries: Mapping[str, str]) -> str:
    return '\n'.join(f'  {label}: {reason}' for label, reason in entries.items())


def _least(program: Program, sources: Mapping[str, Source], name: str) -> int:
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
# axes — how the sources are sliced
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

    def slices(self, sources: Mapping[str, Source]) -> list[tuple[Label, Mapping[str, Source]]]:
        """The ``(key, sources)`` list this axis would run — what ``axis=`` takes hand-built.

        For building one slice alone: ``lps.build(spec, axis.slices(sources)[3][1])``.
        """
        return [(current.key, current.sources) for current in self._slice(sources, self._key_name())[0]]

    def _key_name(self) -> str:
        """The dimension itself: a slice key *is* a coordinate of it."""
        return self.dim

    def _check_the_program(self, program: Program, sources: Mapping[str, Source]) -> None:
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
                f'it, so each slice would build a dimension nothing supplies. A coordinate sweep slices an '
                f'axis the model does not have; to slice one it does, window it.'
            )

    def _slice(self, sources: Mapping[str, Source], key_name: str) -> tuple[list[_Slice], _OriginalIndex | None]:
        """One slice per coordinate, keyed by it. Sources without *dim* pass through.

        No :class:`_OriginalIndex`: nothing was re-indexed, so a slice's frames
        already carry the coordinates they were solved over.
        """
        del key_name
        carrying, coordinates = _coordinates(sources, self.dim, 'slice')
        out: list[_Slice] = []
        for key in coordinates:
            filtered = {name: table.filter(pl.col(self.dim) == key).drop(self.dim) for name, table in carrying.items()}
            out.append(_Slice(key, {**sources, **filtered}))
        return out, None


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive coordinates of *dim*.

    ``steps`` is what each window keeps and ``lookahead`` is what it sees beyond
    that, so a window is ``steps + lookahead`` coordinates long and a
    ``lookahead`` above zero is overlap. An ``int`` keeps the same number every
    window; a sequence keeps those numbers in order, which is a telescoping
    horizon or a month at a time. Both count coordinates rather than coordinate
    values, so *dim* need only be orderable — datetimes, strings and gapped
    integers all work. The dimension is re-indexed rather than dropped, into a
    dense ``0..n-1`` column the model addresses by the name ``into`` gives it,
    which the spec has to declare.

    Whether the model *can* be sliced this way is asked before it is — the
    coupling, the reach and the lookahead they need are
    :meth:`_check_the_program`.
    """

    dim: str
    steps: int | Sequence[int] = field(kw_only=True)
    lookahead: int = field(kw_only=True)
    into: str = field(kw_only=True)

    def slices(self, sources: Mapping[str, Source]) -> list[tuple[Label, Mapping[str, Source]]]:
        """The ``(key, sources)`` list this axis would run — what ``axis=`` takes hand-built.

        For building one window alone: ``lps.build(spec, axis.slices(sources)[37][1])``.
        Pairs, so a window's ownership is not in them: solved as a list the
        slices key by ``key_name=``, ``original_index`` is refused and a
        ``carry`` cannot collapse a dimension, because none of the three has
        anything to read the seam off. Windowing stays this axis's own.
        """
        return [(current.key, current.sources) for current in self._slice(sources, self._key_name())[0]]

    def __post_init__(self) -> None:
        blocks = [self.steps] if isinstance(self.steps, int) else list(self.steps)
        if not blocks:
            raise ValueError('steps is empty, so no window would keep anything — pass an int, or one size per window')
        if short := [block for block in blocks if block < 1]:
            raise ValueError(f'every window must keep at least one coordinate (got steps with {short})')
        if self.lookahead < 0:
            raise ValueError(
                f'lookahead={self.lookahead} is negative; zero is contiguous windows, and above it overlaps'
            )
        if not isinstance(self.steps, int):
            object.__setattr__(self, 'steps', tuple(blocks))
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

    def _check_the_program(self, program: Program, sources: Mapping[str, Source]) -> None:
        """Refuse a window the program's rows cannot be whole inside, before one is taken.

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
                f"EachWindow('{self.dim}', …, into='{self.into}') slices '{self.into}', which the model ties "
                f'together, so no window holds every row whole:\n{_listed(verdict.coupled)}\n'
                f'Each names the change that would lift it.'
            )
        if verdict.undecided:
            raise LpspecError(
                f"EachWindow('{self.dim}', …, into='{self.into}') slices '{self.into}', which the model reaches "
                f'along through a lookup whose groups a window may split, and this driver does not resolve a '
                f'reach the lookup decides:\n'
                f'{_listed({r.label: f"through the lookup {r.name!r}" for r in verdict.undecided})}\n'
                f'Cut a dimension the lookup does not group.'
            )
        if self.lookahead < verdict.ahead:
            raise LpspecError(
                f'EachWindow(lookahead={self.lookahead}) looks ahead by {self.lookahead} coordinate(s), and '
                f"the model reads {verdict.ahead} ahead along '{self.into}' — a row near a window's end would "
                f'read past it. Raise lookahead to at least {verdict.ahead}.'
            )
        if verdict.restarts:
            warnings.warn(
                f"the model counts a position along '{self.into}':\n{_listed(verdict.restarts)}\n"
                f'Every window restarts the count at its first row — what a rolling horizon seeding its '
                f'opening state means, and once per horizon otherwise.',
                LpspecWarning,
                stacklevel=3,
            )

    def _slice(self, sources: Mapping[str, Source], key_name: str) -> tuple[list[_Slice], _OriginalIndex]:
        """One slice per window, keyed by its **first coordinate**.

        Keyed by the coordinate rather than the window's position, which is
        what names a window in the caller's own terms. Sources without *dim*
        pass through untouched.

        The filter leads because it is what a scan can push down; the
        re-indexing that follows is over a frame already filtered to one window.

        **A window owns the coordinates its block names**, and the
        :class:`_OriginalIndex` records which — the rest is lookahead the next
        window recomputes. :meth:`_blocks` trims the last block to what is left,
        so the tail window owns all of itself and nothing falls off the end.
        """
        carrying, coordinates = _coordinates(sources, self.dim, 'window')
        out: list[_Slice] = []
        owned: list[dict[str, Any]] = []
        start = 0
        for owns in self._blocks(len(coordinates)):
            window = coordinates[start : start + owns + self.lookahead]
            local = {coordinate: position for position, coordinate in enumerate(window)}
            filtered = {
                name: (
                    table.filter(pl.col(self.dim).is_in(window))
                    .with_columns(pl.col(self.dim).replace_strict(local, return_dtype=pl.Int64).alias(self.into))
                    .drop(self.dim)
                )
                for name, table in carrying.items()
            }
            out.append(_Slice(window[0], {**sources, **filtered, self.into: range(len(window))}, owns))
            owned.extend(
                {key_name: window[0], self.into: position, self.dim: coordinate}
                for position, coordinate in enumerate(window[:owns])
            )
            start += owns
        return out, _OriginalIndex(self.into, self.dim, pl.DataFrame(owned))

    def _blocks(self, total: int) -> list[int]:
        """How many coordinates each window owns, in order, summing to exactly *total*.

        An ``int`` repeats until the axis runs out, the last window owning
        whatever is left — which is why no window in the middle of a sweep can
        own fewer than ``steps``, and so why a carry always finds its seam. A
        sequence is taken as written, and one that stops short of the axis is
        refused rather than dropping the coordinates it never reached.

        Raises:
            DataError: A sequence of blocks that does not cover the axis.
        """
        if isinstance(self.steps, int):
            blocks = [self.steps] * -(-total // self.steps)
        else:
            blocks = list(self.steps)
            if sum(blocks) < total:
                raise DataError(
                    f'steps keeps {sum(blocks)} coordinate(s) across {len(blocks)} window(s), and '
                    f"'{self.dim}' has {total} — the last {total - sum(blocks)} would be solved by no window. "
                    f'List a block for them, or pass an int to repeat one size to the end.'
                )
        out: list[int] = []
        left = total
        for block in blocks:
            if left <= 0:
                break
            out.append(min(block, left))
            left -= block
        return out


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
    #: ``(key, status, termination_condition, objective, has_primal, spec_digest)``,
    #: in slice order — how every slice terminated, whether or not it produced
    #: an answer, ``has_primal`` saying which of the two it was and ``spec_digest``
    #: which document every slice answered. A slice that reached no objective
    #: holds null there rather than ``nan``, so the column aggregates over the
    #: slices that solved.
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
    #: Whether the axis was a hand-built list, which names no sliced dimension,
    #: so ``original_index`` is refused rather than answered with the keyed
    #: frame. Not the same fact as ``_original is None``, which
    #: :class:`EachCoordinate` is too and where the keyed frame *is* the answer.
    _hand_built: bool = field(repr=False, default=False)
    #: Where the frames are instead, for a sweep solved with ``to=``.
    _spill: _Spill | None = field(repr=False, default=None)

    @classmethod
    def _folded(
        cls,
        key_name: str,
        original: _OriginalIndex | None,
        hand_built: bool,
        answered: Generator[tuple[Any, _Answer], None, None],
        spill: _Spill | None,
        key_dtype: pl.DataType,
    ) -> Runs:
        """Every slice's answer absorbed, in the order they arrive.

        The stream is closed here, which is what releases the serial branch's
        model when a fold is abandoned part way; a reason a slice could not
        produce something is kept from the *first* slice that gave one, since
        a later slice's silence is not a second reason. A spilled sweep's
        answers arrive with their frames already written and released, so
        the fold absorbs the record alone.

        Args:
            key_name: What to call the column holding each slice's key.
            original: The way back to the sliced dimension, or ``None`` where
                the axis re-indexed nothing.
            hand_built: Whether the axis was a list rather than a class, which
                names no dimension to read the keys back over.
            answered: ``(key, answer)`` per slice, in slice order.
            spill: Where the frames went, or ``None`` where they are held.
            key_dtype: What to write the key column as, settled over the
                sweep's keys rather than inferred from each one.
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
                        into[name].append(_keyed(frame, key_name, key, key_dtype))
        return cls(
            key_name=key_name,
            objective=pl.DataFrame(rows, schema_overrides=RECORD_SCHEMA),
            diagnostics=pl.DataFrame(costs),
            _primals=dict(primals),
            _duals=dict(duals),
            _expressions=dict(expressions),
            _no_duals=no_duals,
            _no_expressions=no_expressions,
            _original=original,
            _hand_built=hand_built,
            _spill=spill,
        )

    @property
    def keys(self) -> list[Label]:
        return self.objective[self.key_name].to_list()

    def _read(
        self, held: Mapping[str, list[pl.DataFrame]], kind: str, name: str, absent: str | None = None
    ) -> pl.DataFrame:
        """*name*'s frames from *held*, concatenated — or why there are none.

        *absent* is a reason the fold already knows, which beats one derived
        from what the sweep happens to hold.

        Raises:
            LpspecError: The sweep was spilled, so nothing is held: the
                message names :meth:`scan`.
        """
        self._held_here()
        if name not in held:
            raise LpspecError(absent or _nothing_to_read(kind, name, held, self.objective))
        return pl.concat(held[name])

    def _held_here(self) -> None:
        if self._spill is not None:
            raise LpspecError(
                f'this sweep was spilled to {str(self._spill.directory)!r}, so its frames are on disk rather '
                f'than in memory: runs.scan(name) reads them back as a LazyFrame, and collecting it is the '
                f'choice this reader would otherwise make for you.'
            )

    def scan(self, name: str, kind: str = 'primal', *, original_index: bool = False) -> pl.LazyFrame:
        """One name's values across every slice as a :class:`polars.LazyFrame`, the slice key prepended.

        The reader for a sweep solved with ``to=``, whose frames are on disk;
        on one held in memory it is :meth:`primal`, :meth:`dual` or
        :meth:`expression` made lazy, so the same line reads either.

        Args:
            name: A variable, a constraint or a named expression the spec
                declares, as *kind* says.
            kind: ``primal``, ``dual`` or ``expression`` — the reader this
                stands in for.
            original_index: Read over the dimension the axis sliced instead
                of over the slice key.

        Raises:
            LpspecError: No slice produced *name*, or a *kind* that names no
                reader.
        """
        if self._spill is None:
            return self._frame(name, kind, original_index=original_index).lazy()
        frame = self._spill.scan(reader_kind(kind), name)
        if frame is None:
            absent = {'primal': None, 'dual': self._no_duals, 'expression': self._no_expressions.get(name)}[kind]
            held = dict.fromkeys(self._spill.held(kind))
            raise LpspecError(absent or _nothing_to_read(LABELS[kind], name, held, self.objective))
        return self._reindexed(frame, original_index=original_index)

    def primal(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One variable's values across every slice, the slice key prepended.

        A slice that reached no solution contributes no rows, so this can be
        shorter than the sweep; :attr:`objective` is one row per slice always.

        Args:
            name: A variable the sweep's spec declares.
            original_index: Read over the dimension the axis sliced instead of
                over the slice key.

        Raises:
            LpspecError: No slice of the sweep produced *name*, or
                ``original_index`` on a sweep whose axis was hand-built and so
                named no dimension to read the keys back over.
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

    def _reindexed(self, frame: _Frame, *, original_index: bool) -> _Frame:
        """*frame* over the dimension the axis sliced, rather than over its slices.

        Three answers, and the axis decides which. :class:`EachWindow` carries
        the way back. :class:`EachCoordinate` re-indexed nothing and its key
        column already *is* a coordinate of the answer, so the frame comes back
        unchanged — a satisfied request rather than an ignored one. A hand-built
        list says neither, and there the keyed frame answers a different
        question than the one asked, so it is refused.

        Raises:
            LpspecError: The sweep ran a hand-built axis, which named no
                dimension to read its keys back over.
        """
        if not original_index:
            return frame
        if self._hand_built:
            raise LpspecError(
                f'a hand-built axis does not say what its keys are coordinates of, so this sweep has no '
                f'dimension to read {self.key_name!r} back over. Read it keyed, which is what its slices '
                f'were solved over, or slice with EachWindow — it keys by where each window started, records '
                f'which coordinates each one owns, and stitches.'
            )
        if self._original is None:
            return frame
        return self._original.restore(frame, self.key_name)

    def _frame(self, name: str, kind: str, *, original_index: bool) -> pl.DataFrame:
        """*name* through the reader *kind* names — the dispatch every bridge and :meth:`scan` share."""
        reader = {'primal': self.primal, 'dual': self.dual, 'expression': self.expression}[reader_kind(kind)]
        return reader(name, original_index=original_index)

    def to_pandas(self, name: str, kind: str = 'primal', *, original_index: bool = False) -> pd.DataFrame:
        """One name's values across every slice as a tidy :class:`pandas.DataFrame`.

        The name is resolved before pandas is imported, so a sweep that never
        held *name* says so on any install.

        Args:
            name: A variable, a constraint or a named expression, as *kind*
                says.
            kind: ``primal``, ``dual`` or ``expression`` — the reader this
                stands in for.
            original_index: Read over the dimension the axis sliced instead
                of over the slice key.
        """
        return tidy_to_pandas(self._frame(name, kind, original_index=original_index))

    def to_dataarray(self, name: str, kind: str = 'primal', *, original_index: bool = False) -> xr.DataArray:
        """One name's values as a :class:`xarray.DataArray`, the slice key a dimension; :meth:`to_pandas`'s arguments.

        The extra dimension is named by the axis — a scenario sweep gives
        ``(scenario, …)`` and a window ``(<dim>_start, …)``. A slice that
        reached no solution has no rows and comes back NaN, the same answer a
        masked coordinate gets from ``Result``. ``original_index=True`` gives
        the array over the dimension the axis sliced instead, so a rolling
        horizon's dispatch, or its price, comes back indexed by time.
        """
        return tidy_to_dataarray(self.to_pandas(name, kind, original_index=original_index), name)

    def to_dataset(self, *names: str, kind: str = 'primal') -> xr.Dataset:
        """The named values of one *kind* as one :class:`xarray.Dataset`; all of that kind by default.

        One kind per call: a dual and a variable of the same name would
        collide, and mean something else per row. Costs more than
        ``Result``'s does — each name arrives dense over its own dims *and*
        over every slice. Name the few you need, or use :meth:`save`,
        which writes every kind.

        No ``original_index``: this and :meth:`save` export what the
        sweep *holds*, and the original index is lossy — a bulk export is the
        wrong place to drop the lookahead rows.

        Args:
            names: What to include; none means every name of *kind* some
                slice produced.
            kind: ``primal``, ``dual`` or ``expression``.

        Raises:
            LpspecError: The sweep holds no values of *kind* at all, or is
                spilled — its frames are on disk already.
        """
        return tidy_to_dataset(names or self._names_held(kind), lambda name: self.to_dataarray(name, kind))

    def save(self, directory: str | Path) -> Path:
        """Everything the sweep holds, written as ``to=`` would have written it.

        The same layout: ``<kind>/<name>/<position>.parquet`` for every
        primal, dual and expression, the slice key a column of each, with
        ``objective/``, ``diagnostics/`` and the manifest beside them. So the
        directory is a spilled sweep: :meth:`scan` reads it, and the call
        that made this sweep, pointed at it with ``to=``, reads it back
        without solving a slice.

        Returns:
            The directory.

        Raises:
            LpspecError: The sweep holds no variable values at all, or is
                spilled — its frames are in a directory already.
        """
        self._names_held('primal')
        spill = _Spill.opened(
            directory,
            self.key_name,
            self.keys,
            self.objective[self.key_name].dtype,
            self._original,
            self._hand_built,
        )
        write_reasons(spill.directory, self._no_duals, self._no_expressions)
        by_key = {
            kind: {name: _by_key(frames, self.key_name) for name, frames in held.items()}
            for kind, held in zip(KINDS, (self._primals, self._duals, self._expressions), strict=True)
        }
        for position, key in enumerate(self.keys):
            meta = Record(**self.objective.drop(self.key_name).row(position, named=True))
            cost = self.diagnostics.drop(self.key_name).row(position, named=True)
            frames = {
                kind: {name: keyed[key] for name, keyed in names.items() if key in keyed}
                for kind, names in by_key.items()
            }
            answer = _Answer(meta, dict(cost), frames['primal'], frames['dual'], frames['expression'], None, {})
            spill.write(position, key, answer)
        return spill.directory

    def _names_held(self, kind: str) -> tuple[str, ...]:
        """Every name of *kind* some slice produced, sorted — what a bulk export takes by default.

        Raises:
            LpspecError: No slice produced any, which the exports refuse
                rather than writing an empty directory or an empty dataset —
                for the duals, with the reason the first slice gave.
        """
        self._held_here()
        held = {'primal': self._primals, 'dual': self._duals, 'expression': self._expressions}[reader_kind(kind)]
        if not held:
            absent = self._no_duals if kind == 'dual' else None
            raise LpspecError(absent or _nothing_to_read(LABELS[kind], 'anything', held, self.objective))
        return tuple(sorted(held))

    def __len__(self) -> int:
        return self.objective.height


def _by_key(frames: Sequence[pl.DataFrame], key_name: str) -> dict[Label, pl.DataFrame]:
    """One name's held frames by the slice key each carries, the key column dropped.

    Held per slice that produced the name, not per slice, so the key is read
    off the frame rather than counted; an empty frame carries none and is
    left out, which is what the spill would have written for it.
    """
    return {frame[key_name][0]: frame.drop(key_name) for frame in frames if frame.height}


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


def load_runs(directory: str | Path) -> Runs:
    """Read back a sweep :meth:`Runs.save` wrote, or one ``solve_over(to=)`` spilled.

    The sweep comes back **spilled**: its frames stay in *directory* and
    :meth:`Runs.scan` reads them, which is what a sweep solved with ``to=``
    already is. :attr:`Runs.objective` and :attr:`Runs.diagnostics` are read
    whole — they are one row per slice — and ``original_index`` works, the
    manifest carrying the dimension a window sliced.

    Args:
        directory: Where the sweep was written.

    Returns:
        The sweep, keyed as it was solved.

    Raises:
        LayoutError: A directory holding no ``sweep.json``, which is what
            every sweep written there carries, or one whose layout has moved
            since it was written.
    """
    under = Path(directory)
    manifest = under / _MANIFEST_FILE
    if not manifest.is_file():
        raise LayoutError(
            f'{str(under)!r} holds no {_MANIFEST_FILE!r}, so it is not a sweep save() or to= wrote. A '
            f'single solve writes no manifest and is read by load_result.'
        )
    check_format(under)
    found = json.loads(manifest.read_text())
    original = found['original']
    no_duals, no_expressions = read_reasons(under)
    key_name = found['key_name']
    objective = pl.read_parquet(sorted((under / 'objective').glob('*.parquet')))
    return Runs(
        key_name=key_name,
        objective=objective,
        diagnostics=pl.read_parquet(sorted((under / 'diagnostics').glob('*.parquet'))),
        _no_duals=no_duals,
        _no_expressions=no_expressions,
        _original=None
        if original is None
        else _OriginalIndex(original['local'], original['dim'], pl.read_parquet(under / _OWNED_FILE)),
        _hand_built=found['hand_built'],
        _spill=_Spill(under, key_name, objective[key_name].dtype),
    )


def solve_over(
    spec: Buildable,
    sources: Mapping[str, Source],
    axis: Axis | Sequence[tuple[Label, Mapping[str, Source]]],
    *,
    carry: Mapping[str, str] | None = None,
    key_name: str | None = None,
    executor: Executor | None = None,
    workers_share_fs: bool | None = None,
    solver_options: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
    keep: Keep = 'solver',
    to: str | Path | None = None,
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
            ``(key, sources)`` written by hand.
        carry: ``{parameter: variable}`` — one slice's answer copied into the
            next slice's data. Where the two are over different dimensions the
            value handed on is the last coordinate the slice owns, which is the
            only one that meets the next slice at the seam. The first slice
            takes the parameter from *sources*, its seed.
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
        to: A directory to write each slice's frames to as the fold goes,
            so the sweep's memory stays at one slice however many there
            are. Read back through :meth:`Runs.scan`. A directory holds
            one sweep: run the same sweep at it again and the slices already
            there are not solved again, which is how an interrupted sweep
            resumes.

    Returns:
        Every slice's answers, keyed by slice.

    Raises:
        LpspecError: A carry that cannot line up, has no seed, collapses a
            dimension the axis does not advance along, or is asked together with
            an executor; a key that collides with a column the frames carry;
            an axis the program does not allow; a *to* directory holding
            another sweep. All refused before a slice is taken, and every
            one answerable from the declarations before a source is read.
        DataError: No source carries the axis, or the axis produced no
            slices.

    Warns:
        LpspecWarning: A source carrying the axis that is short of a
            coordinate another has — that slice builds it empty — or a
            position the model counts, which every window restarts.
    """
    if carry and executor is not None:
        raise LpspecError(
            'carry and executor are mutually exclusive: a carried value makes slice i+1 depend on '
            "slice i's answer, so the slices cannot run concurrently. Drop the executor, or drop the carry."
        )
    program = check(spec)
    plan = {p: _CarryRule.resolved(program, p, v) for p, v in (carry or {}).items()}
    key_name = _key_column(axis, key_name, program)

    if isinstance(axis, (EachCoordinate, EachWindow)):
        _check_the_carry(plan, axis, sources)
        axis._check_the_program(program, sources)
        slices, original = axis._slice(sources, key_name)
        hand_built = False
    else:
        slices = [_Slice(*entry) for entry in axis]
        original, hand_built = None, True
        _check_the_carry(plan, axis, slices[0].sources if slices else {})
    if not slices:
        raise DataError('the axis produced no slices')
    solving = {'solver_name': solver_name, 'solver_options': dict(solver_options or {}) or None}
    keys = [current.key for current in slices]
    key_dtype = _one_key_type(keys, key_name)
    spill = None if to is None else _Spill.opened(to, key_name, keys, key_dtype, original, hand_built)
    answered = (
        _serially(program, slices, solving, plan, keep, spill)
        if executor is None
        else _pooled(executor, workers_share_fs, program, slices, solving, spill)
    )
    folded = Runs._folded(key_name, original, hand_built, answered, spill, key_dtype)
    if spill is not None:
        write_reasons(spill.directory, folded._no_duals, folded._no_expressions)
    return folded


def _check_the_carry(
    plan: Mapping[str, _CarryRule],
    axis: Axis | Sequence[tuple[Label, Mapping[str, Source]]],
    first: Mapping[str, Source],
) -> None:
    """Refuse a carry with no seed, or one whose dropped dimension no axis can answer for.

    Both are answered before a source is read: the seed is a key of the first
    slice's sources, and which dimension an axis owns is the axis itself.

    A carry that collapses a dimension hands on the last coordinate the slice
    owns, so the dimension has to be the one the axis advances along —
    :attr:`EachWindow.into`. Any other is a coordinate nothing here can choose,
    and choosing one would be model arithmetic in a driver argument.
    """
    for parameter, rule in plan.items():
        if parameter not in first:
            raise LpspecError(
                f'carry writes {parameter!r} from the second slice on, and the first slice has nothing to '
                f'start from: supply {parameter!r} in sources as the seed.'
            )
        if rule.dropped is None:
            continue
        if isinstance(axis, EachWindow) and rule.dropped == axis.into:
            continue
        owned = f'this axis advances along {axis.into!r}' if isinstance(axis, EachWindow) else 'this axis owns none'
        raise LpspecError(
            f'carry {parameter!r} <- {rule.variable!r} collapses {rule.dropped!r}, and {owned}: a carry hands '
            f'on the last coordinate a slice owns, so it can only collapse the dimension the sweep advances '
            f'along. Reduce {rule.dropped!r} in the YAML — a derived variable is where the oracle can see the '
            f'math — so that {parameter!r} and {rule.variable!r} are over the same dimensions.'
        )


def _serially(
    program: Program,
    slices: Sequence[_Slice],
    solving: Mapping[str, Any],
    plan: Mapping[str, _CarryRule],
    keep: Keep,
    spill: _Spill | None,
) -> Generator[tuple[Any, _Answer], None, None]:
    """Each slice's answer, off one model updated in place.

    Every slice of a sweep is the same math over different numbers, which is
    what :meth:`~lpspec.api.Model.update` is for; a rebuild releases the
    previous model before it starts, so the fold holds one slice's model
    however many there are.

    **A slice that names something else is rebuilt, not updated.** A slice is
    *total* where ``update`` is partial by construction: the two agree only
    while every slice names the same sources, which the class axes guarantee
    and a hand-built list does not. Compared by *name* — values are what a
    update exists to replace.

    **A generator because of the carry**: slice ``i+1``'s sources are not
    known until slice ``i``'s frames have been read, and resuming after the
    yield is where that happens. The caller closes this — that is what
    releases the model when a fold is abandoned part way.

    **A slice the spill already holds is read back, not solved**, its frames
    staying on disk — a carry reads the one it needs from there — and one
    solved here is written before its answer is yielded, so an abandoned
    fold leaves every slice it finished.
    """
    model: Model | None = None
    named: frozenset[str] | None = None
    state: dict[str, Any] = {}
    try:
        for position, current in enumerate(slices):
            if spill is not None and spill.done(position):
                answer = spill.read_back(position)
                primals = spill.primals(position, {rule.variable for rule in plan.values()})
                yield current.key, answer
                state = _carried(plan, primals, current, position, slices, answer)
                continue
            sources = {**current.sources, **state}
            names = frozenset(sources)
            with _named_slice(current.key, position, len(slices)):
                if model is not None and names == named:
                    before = model.diagnostics()
                    model.update(sources)
                else:
                    if model is not None:
                        model.close()
                    model, named, before = build(program, sources), names, None
                result = model.solve(**solving, keep=keep)
                answer = _answers(result, program, _slice_cost(model.diagnostics(), before))
            primals = answer.primals
            if spill is not None:
                answer = spill.write(position, current.key, answer)
            yield current.key, answer
            state = _carried(plan, primals, current, position, slices, answer)
    finally:
        if model is not None:
            model.close()


def _carried(
    plan: Mapping[str, _CarryRule],
    primals: Mapping[str, pl.DataFrame],
    current: _Slice,
    position: int,
    slices: Sequence[_Slice],
    answer: _Answer,
) -> dict[str, Any]:
    """What the next slice starts from, read out of *primals* — nothing for the last slice, or with no plan.

    Raises:
        LpspecError: The slice left nothing to carry, having reached no
            solution, and a next slice is waiting on it.
    """
    if not plan or position == len(slices) - 1:
        return {}
    if not primals:
        raise LpspecError(
            f'slice {current.key!r} ({position + 1} of {len(slices)}) terminated '
            f'{answer.meta.termination_condition}, so slice {slices[position + 1].key!r} has no '
            f'{sorted({rule.variable for rule in plan.values()})} to start from. A carried sweep '
            f'stops at the first slice that leaves nothing to carry; the {position} before it solved.'
        )
    return {p: rule.value_from(primals, p, current.key, current.owns) for p, rule in plan.items()}


@contextmanager
def _named_slice(key: Label, position: int, count: int) -> Generator[None, None, None]:
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
    executor: Executor,
    workers_share_fs: bool | None,
    program: Program,
    slices: Sequence[_Slice],
    solving: Mapping[str, Any],
    spill: _Spill | None,
) -> Generator[tuple[Any, _Answer], None, None]:
    """The same, from slices built independently and possibly elsewhere.

    Yielded in **slice order, never completion order**: the futures are walked
    in the order they were submitted, so a sweep cannot reorder itself under a
    pool. A built model cannot cross a process, so this branch builds per
    slice — the same fact that makes ``carry`` and ``executor`` mutually
    exclusive. Every worker is handed the lowered program, so none reads the
    YAML or lowers it again.

    A slice the spill already holds is never submitted; one that comes back
    is written here, by the process that owns the directory.
    """
    crosses = _crosses_a_process(executor)
    shared = _shares_filesystem(executor, workers_share_fs)
    call = dict(solving)
    memo: dict[str, tuple[Any, Any]] = {}
    futures = [
        None
        if spill is not None and spill.done(position)
        else executor.submit(
            _run_slice,
            program,
            _encode(current.sources, memo, workers_share_fs=shared) if crosses else dict(current.sources),
            crosses,
            call,
        )
        for position, current in enumerate(slices)
    ]
    for position, (current, future) in enumerate(zip(slices, futures, strict=True)):
        if future is None:
            assert spill is not None, 'a slice is skipped only where a spill holds it'
            yield current.key, spill.read_back(position)
            continue
        with _named_slice(current.key, position, len(slices)):
            answer = future.result()
        answer = replace(
            answer,
            primals=_decode(answer.primals),
            duals=_decode(answer.duals),
            expressions=_decode(answer.expressions),
        )
        yield current.key, spill.write(position, current.key, answer) if spill is not None else answer


def _answers(result: Result, program: Program, cost: dict[str, Any]) -> _Answer:
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
    meta = Record.of(
        result.termination_condition, result.objective, has_primal=result.has_primal, spec_digest=result.spec_digest
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
    program: Program,
    encoded: dict[str, Any],
    encode_out: bool,
    call: dict[str, Any],
) -> _Answer:
    """One slice, start to finish, over plain data — the *pooled* branch.

    Module-level and closure-free on purpose: a remote executor has to pickle
    what it is handed, and a bound method or a lambda over the axis object
    cannot cross.
    """
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
    axis: Axis | Sequence[tuple[Label, Mapping[str, Source]]],
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
                'a hand-built axis needs key_name=: a list of slices does not say what its keys are '
                "coordinates of, and 'slice' would be this library naming your axis for you. Pass "
                "key_name='draw', key_name='period', or whatever the keys actually are."
            )
        key_name = axis._key_name()
    if key_name in program.dimensions:
        raise LpspecError(
            f'key_name={key_name!r} is a dimension the spec declares, so the slice key would collide '
            f'with a column the frames already carry. Name it something the spec does not use.'
        )
    fixed = ('value', *Record._fields)
    if key_name in fixed:
        raise LpspecError(
            f'key_name={key_name!r} is a column every sweep frame carries ({", ".join(fixed)}), so the slice '
            f'key would replace it rather than sit beside it. Name it something else.'
        )
    return key_name


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------


def _shares_filesystem(executor: Executor, declared: bool | None) -> bool:
    """Whether *executor*'s workers can read this process's paths.

    The two stdlib pools are the ones whose deployment is knowable: both run
    here, so both read the paths here. An executor this package did not ship is
    a transport it cannot ask, so it is assumed remote until *declared* says
    otherwise.
    """
    if declared is not None:
        return declared
    return isinstance(executor, ProcessPoolExecutor)


def _crosses_a_process(executor: Executor) -> bool:
    """Whether a slice's sources have to be encoded to reach *executor*.

    A thread pool runs in this process, so encoding would be a parquet round
    trip for a boundary that is not there. Every other executor is assumed to
    cross, none of them being answerable.
    """
    return not isinstance(executor, ThreadPoolExecutor)


def _encode(
    sources: Mapping[str, Source], memo: dict[str, tuple[Any, Any]], *, workers_share_fs: bool = False
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
        if isinstance(obj, (str, Path)) and workers_share_fs:
            out[name] = obj
            continue
        cached = memo.get(name)
        if cached is not None and cached[0] is obj:
            out[name] = cached[1]
            continue
        if isinstance(obj, (str, Path)):
            out[name] = Path(obj).read_bytes()
        elif (table := as_frame(obj)) is None:
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


def carries(sources: Mapping[str, Source], dim: str) -> dict[str, pl.LazyFrame]:
    """The sources that carry a column called *dim*, by name.

    Derived rather than declared, and the one home for that derivation: a
    source carrying the slice key that is *not* filtered produces a
    duplicate-coordinate error at attach time, so a sweep and an archive have
    to agree about which they are.
    """
    tables = {name: table for name, obj in sources.items() if (table := as_frame(obj)) is not None}
    return {name: table for name, table in tables.items() if dim in table.collect_schema().names()}


def _coordinates(sources: Mapping[str, Source], dim: str, verb: str) -> tuple[dict[str, pl.LazyFrame], list[Label]]:
    """The sources a slice has to filter, by name, and the ordered coordinates to slice.

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
    carrying = carries(sources, dim)
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
