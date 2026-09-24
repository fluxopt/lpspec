"""What every solver is: a loaded model, and the rule for keeping it.

A solver sink holds the model it was given and outlives the solve it was loaded
for, so that an updated model (:meth:`~specsolve.api.Model.update`) has its new
numbers *pushed* onto what the solver already has and re-solves from the basis
the last one ended on.

**This module imports no solver.** It is the one thing ``solvers/`` members may
read besides ``tables.py``.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from specsolve.relational.sinks.capabilities import Capabilities
    from specsolve.relational.sinks.tables import Tables
    from specsolve.relational.status import SolveStatus


@dataclass(frozen=True)
class WarmStart:
    """What one solve leaves for a later session: a basis, or an incumbent.

    Read with :meth:`Solver.warm_start`, applied with :meth:`Solver.warm`.
    Which fields are filled is the reading solver's decision: an LP leaves its
    simplex basis (both status vectors, paired), a mixed-integer solve leaves
    no valid basis anywhere and carries its incumbent instead.

    Opaque, and the statuses are the reading solver's own encoding, so only a
    session of the same solver takes them back. Nothing above ``solvers/``
    carries one.
    """

    #: Which member of ``SOLVERS`` read it; only that member takes it back.
    solver: str
    #: Basis status per column in label order, or ``None`` after a solve that
    #: left no valid basis.
    column_statuses: Any | None
    #: Basis status per row in label order; filled exactly when
    #: :attr:`column_statuses` is.
    row_statuses: Any | None
    #: Primal value per column in label order — a mixed-integer incumbent —
    #: or ``None`` where the basis carries the start instead.
    column_values: Any | None

    def basis(self) -> tuple[Any, Any] | None:
        """Both status vectors — filled exactly together — or ``None`` where the incumbent carries the start."""
        if self.column_statuses is not None and self.row_statuses is not None:
            return self.column_statuses, self.row_statuses
        return None


@dataclass(frozen=True)
class SolveAnswer:
    """What a solve concluded, and the vectors it left.

    Any vector may be ``None``, for different reasons: no ``primal`` means the
    solve left nothing worth reading — and ``activity``, each row's left-hand
    side at that point, travels with it — while no ``dual`` is narrower: a
    mixed-integer model has none at all, and neither does a run stopped short
    of a simplex basis.

    ``dual_ray`` is the one vector an *unreadable* answer can carry, and the
    only one: an infeasible solve has no solution to report and may still
    report why there is none.
    """

    status: SolveStatus
    objective: float
    primal: pl.Series | None
    dual: pl.Series | None
    activity: pl.Series | None
    #: A weight per row certifying that the constraints cannot all hold, in
    #: the sign convention of :meth:`Solver.dual_ray`, or ``None`` where the
    #: solve was not infeasible or the solver produced none.
    dual_ray: pl.Series | None = None

    @classmethod
    def unreadable(cls, status: SolveStatus, dual_ray: pl.Series | None = None) -> SolveAnswer:
        """The answer for a solve that left nothing worth reading.

        An unreadable status carries a NaN objective and no vector but the ray.
        """
        return cls(status, float('nan'), None, None, None, dual_ray)


class Solver(ABC):
    """One solver, holding one model. Subclassed once per member of ``SOLVERS``.

    A driver never constructs one directly: :func:`~specsolve.relational.sinks.solvers.loaded`
    is the whole of "reuse or load again", and what it hands back is run and,
    eventually, closed::

        solver = solvers.loaded(held, name, tables, options)
        solver.run(tables)  # …repeatedly
        solver.close()

    This class records the structure of what was loaded and the options it was
    loaded with; a subclass owns the hand-off — loading, pushing values,
    running, releasing.
    """

    def __init__(
        self,
        tables: Tables,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        #: The options the loaded model was told, set at the load.
        self._options = dict(solver_options or {})
        self._load(tables, batch_rows)
        #: The build's own frames, until :meth:`structure` reads their digest
        #: and lets them go.
        self._tables: Tables | None = tables
        #: The digest of everything a re-solve may not change, or ``None``
        #: before :meth:`structure` is first asked. Read through it, never here.
        self._structure: bytes | None = None
        #: The loaded model's spans, read by :meth:`_takes` alone.
        self._columns = tables.column_count
        self._rows = tables.row_count

    #: The packages this member imports lazily, and so the ones an environment
    #: has to have for it to run at all.
    requires: ClassVar[tuple[str, ...]]

    #: What this member can ingest, and what it refuses in combination. A
    #: member states it; the family acts on it
    #: (:func:`~specsolve.relational.sinks.refusal`).
    capabilities: ClassVar[Capabilities]

    #: What to tell a caller when :meth:`is_available` says no — which package
    #: is missing, and whether it ships or needs an extra.
    unavailable_message: ClassVar[str]

    def structure(self) -> bytes:
        """The digest of the loaded model, hashed off its frames the first time it is asked.

        Reading it lets the frames go. Idempotent.
        """
        if self._structure is None:
            assert self._tables is not None, 'a solver holds the tables it loaded until their digest replaces them'
            self._structure = self._tables.structure
            self._tables = None
        return self._structure

    def keeps(self, tables: Tables, solver_options: Mapping[str, Any] | None) -> bool:
        """Whether this held solver may keep its load and take *tables* by value."""
        return self._options == dict(solver_options or {}) and self.structure() == tables.structure

    @classmethod
    def imported(cls) -> Any:
        """Every package in :attr:`requires`, imported — or :attr:`unavailable_message`.

        Returns the first, the member's own library; the rest are imported only
        to fail here.
        """
        try:
            modules = [__import__(package) for package in cls.requires]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(cls.unavailable_message) from exc
        return modules[0]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this build can actually run this solver.

        A probe of the import system rather than an import, and it does not
        raise. Probed at the top-level name — ``find_spec`` on a dotted one
        imports the parent.
        """
        return all(importlib.util.find_spec(package.partition('.')[0]) is not None for package in cls.requires)

    @abstractmethod
    def _load(self, tables: Tables, batch_rows: int | None) -> None:
        """Hand *tables* to the solver and hold whatever reads it back.

        Called by ``__init__`` rather than by a caller.
        """

    @abstractmethod
    def push(self, tables: Tables) -> None:
        """*tables*'s bounds, costs and right-hand sides onto the loaded model.

        Everything an update may change without moving a label, and only ever
        after *tables*'s digest matched the loaded one. Whole vectors rather
        than a diff.
        """

    @abstractmethod
    def warm_start(self) -> WarmStart | None:
        """What the loaded model holds to warm a later session, if anything.

        Returns:
            The basis after an LP solve, the incumbent after a mixed-integer
            one — a solved MIP leaves no valid basis on any solver — and
            ``None`` where the model holds neither, which is every model not
            yet solved.
        """

    def warm(self, ws: WarmStart) -> None:
        """Start the next :meth:`run` from *ws* instead of from scratch.

        The caller vouches that *ws* was read from a model with this one's
        label set; what is checked here is what can be — the sink it came
        from, and that its vectors span the loaded model.

        Raises:
            SpecsolveError: A warm start read from another solver, or whose
                vectors do not span the loaded model.
        """
        self._takes(ws)
        self._warm(ws)

    def _takes(self, ws: WarmStart) -> None:
        """Refuse a warm start that describes a different model.

        Raises:
            SpecsolveError: A start from another solver — statuses are each
                solver's own encoding — or one whose vectors have the wrong
                span, which a basis being positional makes a start about a
                different model.
        """
        mine = type(self).__name__.lower()
        if ws.solver != mine:
            raise SpecsolveError(
                f'this warm start was read from {ws.solver!r} and cannot warm a {mine!r} session: '
                f"basis statuses and incumbents are the reading solver's own encoding, so applied "
                f'elsewhere they would start the solve from a state that means something else. '
                f'Read a warm start from the solver that will take it back.'
            )
        spans = (
            ('column statuses', ws.column_statuses, self._columns, 'columns'),
            ('row statuses', ws.row_statuses, self._rows, 'rows'),
            ('column values', ws.column_values, self._columns, 'columns'),
        )
        for quantity, values, expected, axis in spans:
            if values is not None and len(values) != expected:
                raise SpecsolveError(
                    f'this warm start carries {len(values)} {quantity} for a model with {expected} '
                    f'{axis}. A basis and an incumbent are positional, so one read from a '
                    f'differently shaped model would start the solve from a state about a '
                    f'different one — carry a warm start only across builds whose label set '
                    f'is unchanged.'
                )

    @abstractmethod
    def _warm(self, ws: WarmStart) -> None:
        """Apply *ws* onto the loaded model, its spans already checked.

        Reached only through :meth:`warm`, so a member may assume the vectors
        span the model it holds and that filled fields pair the way
        :class:`WarmStart` says they do.
        """

    def run(self, tables: Tables) -> SolveAnswer:
        """Solve what is loaded, read it back, and refuse a vector that lies.

        Reading a solution back is positional, so a vector that does not span
        the model is an answer about a *different* one, and is refused here.
        """
        answer = self._run(tables)
        self._check_span('primal', answer.primal, tables.column_count)
        self._check_span('dual', answer.dual, tables.row_count)
        self._check_span('activity', answer.activity, tables.row_count)
        self._check_span('dual ray', answer.dual_ray, tables.row_count)
        return answer

    def _check_span(self, quantity: str, values: pl.Series | None, expected: int) -> None:
        """Check that a solver vector spans the model.

        ``None`` is not a wrong length — a mixed-integer model has no duals,
        and neither does a run stopped short of a simplex basis.

        Raises:
            SpecsolveError: A vector of any other length, which describes a
                different model.
        """
        if values is not None and len(values) != expected:
            raise SpecsolveError(
                f'{type(self).__name__} returned {len(values)} {quantity} values for a model with '
                f'{expected}. Reading a solution back is positional, so a vector that does not span '
                f'the model describes a different one. This is an engine bug rather than a problem '
                f'with the model — please report it.'
            )

    @abstractmethod
    def _run(self, tables: Tables) -> SolveAnswer:
        """Solve what is loaded and read it back.

        *tables* is asked only for what has no column and so was never loaded —
        the objective's constant. When either vector may be ``None`` is
        :class:`SolveAnswer`'s docstring. An infeasible solve calls
        :meth:`dual_ray` and returns what it gives.
        """

    def dual_ray(self) -> pl.Series | None:
        """A weight per row certifying that this infeasible model has no solution.

        Called only after an infeasible solve. The weights combine the rows
        into one that demands more than the columns can deliver inside their
        bounds — Farkas' lemma, and the cut a Benders master needs when a
        subproblem cannot be dispatched at all.

        **One convention across the sinks**, since a caller reading a ray
        cannot be asked which solver signed it: a row's weight carries the
        sign the row is written with, which is HiGHS's and Xpress's. A sink
        whose solver signs the other way negates what it reads.

        Returns:
            The weights in row order, or ``None`` where this solver produced
            none — which is a solver setting rather than a property of the
            model, so the engine's message names the setting.
        """
        return None

    @abstractmethod
    def forget(self) -> None:
        """Discard the work the last solve did, keeping the model loaded.

        The middle rung of :data:`~specsolve.relational.result.KEEPS`: the matrix
        stays handed over, and the next run begins as if it had never been
        solved. A member with nothing to discard implements this as a no-op.
        """

    @property
    @abstractmethod
    def handle(self) -> Any:
        """The native object the load handed back, or ``None`` once closed.

        The library's own model — what ``build_<solver>`` gives a caller who
        stops at the hand-off, and what a test reads the load back through.
        Owned by this holder: the caller does not release it, :meth:`close`
        does.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the loaded model, and anything outside this process with it.

        Idempotent. Afterwards :attr:`handle` is ``None``.

        **The same release happens to a holder dropped without closing.** A
        member whose library releases its object on collection has that for
        free; one that does not — or that holds two objects, a model on an
        environment, where the order is innermost first — registers a
        finalizer over the objects rather than over itself.
        ``tests/test_solver_release.py`` asks every member.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
