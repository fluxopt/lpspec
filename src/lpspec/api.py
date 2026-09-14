"""The runner: attach data to a YAML spec and execute it. Not a modeling API.

Math is defined in YAML only — there is no Python API for constructing specs,
and the logical plan is internal. Four verbs run a model: ``check``, ``build``
(YAML + sources → a :class:`Model`), ``solve`` and ``write``. ``load_result`` reads back an
answer :meth:`Result.save` wrote and ``scan_result`` leaves it on disk; the
question and the answer as one archive is :class:`lpspec.archive.SolveArchive`.

This is the relational lane (docs/about/architecture.md): validated at load
time, lowered to the plan, executed relationally. The same file builds as a
``linopy.Model`` through ``lpspec.linopy``, on the same call and the same
sources — which lane a caller wants is theirs to pick, and this one needs no
optional extra.

Example::

    import lpspec as lps

    result = lps.solve(
        'spec.yaml',
        {'p_max': 'p_max.parquet', 'load': 'load.parquet', 'snapshot': range(8760)},
    )
    result.objective
    result.primal('p')  # tidy polars.DataFrame (coords..., value)
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from math_spec import advice

from lpspec.errors import DataError, LayoutError, LpspecError, LpspecWarning
from lpspec.lanes import LANES, Buildable, Label, Source, declared, lowered
from lpspec.layout import beside, check_the_target, write_archive
from lpspec.relational import sinks
from lpspec.relational.engines.polars.engine import PolarsEngine
from lpspec.relational.parquet import (
    METRICS_FILE,
    METRICS_SCHEMA,
    RECORD_FILE,
    Record,
    check_format,
    digest_of,
    read_reasons,
    write_whole,
)
from lpspec.relational.result import Result
from lpspec.relational.sinks import solver, writer
from lpspec.relational.sinks.capabilities import lane_cannot_build_message, required
from lpspec.sources import attachable, tidy_sources, unknown_source_keys_message

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from math_spec.program import Program

    from lpspec.relational.result import ConstraintRow, Diagnostics, Keep

__all__ = ['build', 'check', 'load_result', 'scan_result', 'solve', 'write']


def _portability(program: Program, sink: str) -> tuple[str | None, list[str]]:
    """*sink*'s reason for refusing *program*, and what it would rewrite to take it.

    A lane rewrites nothing — everything it supports it builds natively — so
    its second answer is empty.
    """
    if (lane := LANES.get(sink)) is not None:
        missing = lane.missing(required(program))
        return (lane_cannot_build_message(sink, missing) if missing else None), []
    refused = sinks.refusal(program, sink)
    return refused, [] if refused else sinks.relaxations(program, sink)


def check(spec: Buildable, sink: str | None = None) -> Program:
    """Parse, expand, validate and lower a spec; attach no data.

    With *sink*, also: **will that sink take it?** Bare ``check`` says nothing
    about portability. The answer is read off a declared table with no data
    attached, so it needs no solver installed. The solver-independent advice
    is issued either way.

    Args:
        spec: A YAML path, a mapping, or a ``Spec``.
        sink: A solver name (``highs``, ``gurobi``, ``xpress``), an output
            suffix (``.lp``, ``.mps``), or a lane (``linopy``). ``None`` asks
            only whether the spec is sayable.

    Returns:
        The lowered program: what a build reads rows off, and what every verb
        here takes back without parsing the file again. It is the language's
        own type — typeset it, or read its declarations, through
        :mod:`math_spec`.

    Raises:
        LanguageError: A construct outside the streaming language.
        LpspecError: A *sink* that cannot take this spec, or a name belonging
            to no sink.
        ValueError: A schema or expression that does not parse.

    Warns:
        LpspecWarning: Advice short of an error — a declared dimension nothing
            uses as an axis, a variable the objective drives to infinity with
            nothing to stop it, a construct the named sink takes only
            reformulated. Issued here and nowhere else.
    """
    program = lowered(spec)
    notes = [str(note) for note in advice(program)]
    refused: str | None = None
    relaxed: list[str] = []
    if sink is not None:
        refused, relaxed = _portability(program, sink)
    for note in (*notes, *relaxed):
        warnings.warn(note, LpspecWarning, stacklevel=2)
    if refused is not None:
        raise LpspecError(refused)
    return program


class Model:
    """A spec with your data attached to it — what :func:`build` returns.

    Three nouns, each arrow adding one thing: a ``Program`` is the math,
    a ``Model`` is the math with your data, a ``Result`` is one answer.

        ``check`` → ``Program`` → ``build`` → ``Model`` → ``solve`` → ``Result``

    One build feeds any number of sinks — :meth:`solve` and :meth:`write` on
    the same object — :meth:`update` puts new numbers on it without re-reading
    the YAML or re-lowering the plan, and :meth:`diagnostics` says what it did.
    Nothing has to be released; :meth:`close` hands a large model back early.
    """

    def __init__(self, spec: Buildable, sources: Mapping[str, Source]) -> None:
        self._spec = declared(spec)
        self._program = lowered(self._spec)
        #: What every answer of this model carries, so two of them can be told
        #: to have answered the same document.
        self._digest = digest_of(self._spec.to_yaml())
        self._sources = dict(sources)
        self._engine = PolarsEngine()
        self._fill()

    def _fill(self) -> None:
        """Build the frames from whatever is attached now.

        A failure releases the half-built model and re-raises, so an update
        that raises leaves a closed handle rather than a stale one.
        """
        try:
            self._engine.build(self._program, tidy_sources(self._program, self._sources))
        except BaseException:
            self._engine.close()
            raise

    def update(self, sources: Mapping[str, Source]) -> Model:
        """Put new numbers on the same model, in place.

        ::

            model.update({'cap_hat': capacity}).solve()

        Any new data is accepted: ``model.update(x)`` answers what
        ``build(spec, sources | x)`` answers, whatever changed. Data that moves
        a mask renumbers labels, so the model is rebuilt and solved cold
        instead of pushed onto a loaded solver, and
        :attr:`~lpspec.relational.result.Diagnostics.loads` says which ran.

        Results taken before the update keep reading: each owns the frames it
        reads, and an update builds new ones rather than touching those. A
        retained result keeps its build's label frames alive until it is
        dropped or :meth:`~lpspec.relational.result.Result.close` is called.

        Args:
            sources: Only what changed; the rest keeps what :func:`build`
                attached. A dimension's labels as well as a parameter, which is
                how a coordinate set grows.

        Returns:
            This object, so a driver can chain.

        Raises:
            DataError: A name the spec does not declare.
        """
        _refuse_unknown(sources, attachable(self._program))
        self._sources.update(sources)
        self._fill()
        return self

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, Any] | None = None,
        keep: Keep = 'solver',
        archive: str | Path | None = None,
    ) -> Result:
        """Hand the built model to a solver and solve it.

        A solver that can stay loaded is kept between calls, so an updated
        model skips the hand-off and only its numbers are pushed. Whether the
        *work* that solver did is kept too is *keep*, off by default. How much
        this solve actually kept is its
        :attr:`~lpspec.relational.result.Result.kept`.

        Args:
            solver_name: ``highs``, which ships with the package, or
                ``gurobi``, which needs the ``[gurobi]`` extra.
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary (``{'time_limit': 60}``).
            keep: How much of the session this solve may keep — one of
                :data:`~lpspec.relational.result.KEEPS`. ``solver``, the
                default, reuses the solver holding the model and discards the
                work it did; ``progress`` keeps that work too, which is what
                an iterating driver moving one step at a time wants;
                ``nothing`` keeps neither, which is what timing a build or
                comparing against a cold baseline needs and what no solver
                option can promise. A preference: a model whose structure
                moved is loaded again whatever was asked.
            archive: Where to write the whole thing — the spec, the data
                attached to it **now**, and this answer — so that
                :func:`~lpspec.archive.load_archive` gives all three back and
                the model solves again from the file alone. A ``.zip`` suffix
                packs it into one file and anything else is a directory. What
                the build and its solves have spent goes in beside the answer,
                as :class:`~lpspec.relational.parquet.Metrics`.

        Returns:
            The solution, holding this model.

        Raises:
            LpspecError: A solver name nothing serves, one this environment
                cannot run, or a *keep* outside
                :data:`~lpspec.relational.result.KEEPS`.
            LayoutError: An *archive* directory that already holds something,
                refused before the solve rather than after it.
        """
        out = None if archive is None else _the_archive_target(Path(archive))
        answered = replace(
            self._engine.solve(solver_name, solver_options=solver_options, keep=keep),
            _spec_digest=self._digest,
            _solved_at=datetime.now(UTC),
        )
        if out is not None:
            self._archive(out, answered)
        return answered

    def _archive(self, out: Path, answered: Result) -> None:
        """Pack this model, what is attached to it now, and *answered* into one zip.

        The answer is laid out in a scratch directory beside *out* first, in
        the layout an archive's ``answer/`` holds. The metrics row is written
        after the answer rather than through :meth:`Result.save`, which cannot
        write it: a result is one solve, and the diagnostics the metrics come
        from span the model's whole life.
        """
        with beside(out) as scratch:
            answer = answered.save(scratch)
            taken = self._engine.diagnostics().metrics()
            write_whole(pl.DataFrame([taken._asdict()], schema_overrides=METRICS_SCHEMA), answer / METRICS_FILE)
            write_archive(
                out,
                self._spec,
                self._sources,
                checked=self._sources,
                whole={},
                axis=None,
                answer=answer,
            )

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
            LpspecError: A construct the format has no section for, the same as
                :func:`check`'s ``sink=`` answer.
        """
        self._engine.write(path)

    def row(self, name: str, /, **coordinate: Label) -> ConstraintRow:
        """One built constraint row at one coordinate — its terms, sense and right-hand side.

        The verb for *this row is wrong and I do not know why*. ``to_latex``
        and its siblings render the spec as math before any data, and
        :meth:`~lpspec.relational.result.Result.dual` gives a row's number
        without its terms; this gives the row the build actually produced, at
        the coordinate you name.

        Reads the **built** model and needs no solve, so it answers on a model
        that never reached a solver — and it is the built row, so a term whose
        variable was absent is missing from it and a row a ``where`` masked out
        is not there at all. It shows what the model says rather than what the
        file appears to say.

        Args:
            name: A declared constraint. Positional, so that a dimension may
                be called ``name`` and still be named in *coordinate*.
            coordinate: One label per dim of that declaration, all of them —
                a partial coordinate names a set of rows rather than one.

        Returns:
            The terms as ``(variable, coordinate, coefficient)``, beside the
            comparison and the right-hand side.

        Raises:
            KeyError: No constraint is called *name*.
            LpspecError: The coordinate names the wrong dims, matches no row
                the build produced, or the model has been closed.

        Example:
            >>> print(model.row('balance', snapshot=1))  # doctest: +SKIP
            balance[snapshot=1]: +1 p[1, wind] +50 p[1, gas] >= 60
        """
        return self._engine.row(name, coordinate)

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`, and after a build that raised: every
        field is a count, a clock or a small frame the engine keeps, not a read
        of the model it releases. A raise leaves the sizes at zero — they are
        taken once a model is whole — and everything measured before it stands.
        """
        return self._engine.diagnostics()

    def close(self) -> None:
        """Release the built model, and any solver still holding it."""
        self._engine.close()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _refuse_unknown(given: Mapping[str, Any], declared: Mapping[str, Any]) -> None:
    """Refuse an update naming anything *declared* does not hold."""
    if unknown := set(given) - set(declared):
        raise DataError(unknown_source_keys_message(unknown, declared))


def _the_archive_target(out: Path) -> Path:
    """*out*, once it is somewhere an archive can be written.

    Checked before the solve, not after.
    """
    check_the_target(out)
    return out


def build(spec: Buildable, sources: Mapping[str, Source]) -> Model:
    """Attach *sources* to *spec* and build it — the model with your data on it.

    Args:
        spec: As :func:`check` takes it.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — an index table, a parquet path,
            or a bare sequence — wherever the YAML declares none.

    Returns:
        The built model. It feeds any number of sinks — ``model.solve()`` and
        ``model.write(path)`` on the same object — and ``model.update(...)``
        puts new numbers on it.

    Raises:
        LanguageError: A construct outside the streaming language.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    return Model(spec, sources)


def solve(
    spec: Buildable,
    sources: Mapping[str, Source],
    solver_name: str = 'highs',
    *,
    solver_options: Mapping[str, Any] | None = None,
    archive: str | Path | None = None,
) -> Result:
    """Build *spec* and solve it in one call.

    The one-shot spelling: a caller who will solve the same spec again with
    new numbers wants :func:`build` and :meth:`Model.update`.

    There is no ``keep`` here — this builds the model it solves, so the solve
    is the first of that model's life and
    :attr:`~lpspec.relational.result.Result.kept` is always ``nothing``.
    Choosing what to keep is :meth:`Model.solve`.

    Args:
        spec: As :func:`check` takes it.
        sources: As :func:`build` takes them.
        solver_name: ``highs``, which ships with the package, or ``gurobi``,
            which needs the ``[gurobi]`` extra.
        solver_options: Forwarded to the solver verbatim, in its own
            vocabulary (``{'time_limit': 60}``).
        archive: Where to write the spec, its data and this answer, as
            :meth:`Model.solve` takes it — a ``.zip``, or a directory.

    Returns:
        The solution, self-contained: it owns the frames it reads, so the built
        model and the solver are released before this returns and there is
        nothing to manage. ``result.close()`` drops its own hold early.

    Raises:
        LpspecError: A solver name nothing serves — checked before the build.
    """
    solver(solver_name)
    model = build(spec, sources)
    try:
        return model.solve(solver_name, solver_options=solver_options, archive=archive)
    finally:
        model.close()


def write(
    spec: Buildable,
    sources: Mapping[str, Source],
    out: str | Path,
) -> Path:
    """Build *spec* and stream it to a file, in the format *out*'s suffix names.

    Args:
        spec: As :func:`check` takes it.
        sources: As :func:`build` takes them.
        out: Where to write; ``.lp`` and ``.mps`` are what ship.

    Returns:
        The path written.

    Raises:
        ValueError: A suffix nothing writes — checked before the build.
        LpspecError: A construct the format has no section for, which is
            ``check(spec, sink=out.suffix)``'s answer with no data attached.
    """
    out = Path(out)
    writer(out.suffix.lower())
    with build(spec, sources) as model:
        model.write(out)
    return out


def _whole(file: Path) -> pl.LazyFrame:
    """*file* read into memory, behind the :class:`polars.LazyFrame` a saved frame is held as.

    The :data:`Reading` a ``load_`` uses: the bytes are here when it returns.
    """
    return pl.read_parquet(file).lazy()


#: How a saved frame is read — the one difference between ``load_`` and
#: ``scan_``. :func:`_whole` reads it now, so what comes back owes the
#: directory nothing; :func:`polars.scan_parquet` reads it at the first
#: collect, so the directory has to outlive what was read off it.
type Reading = Callable[[Path], pl.LazyFrame]


def _saved_frames(under: Path, read: Reading) -> dict[str, pl.LazyFrame]:
    """Every ``<name>.parquet`` under *under*, keyed by name; empty where it does not exist.

    The kinds a solve did not answer with are simply missing directories,
    which is how the writer says a kind is absent.
    """
    if not under.is_dir():
        return {}
    return {file.stem: read(file) for file in sorted(under.glob('*.parquet'))}


def _absent(reason: str) -> Callable[[], pl.DataFrame]:
    """A named expression's reader, for one the solve could not evaluate.

    Deferred like every expression reader: it raises at the read, with the
    reason the solve gave.
    """

    def read() -> pl.DataFrame:
        raise LpspecError(reason)

    return read


def load_result(directory: str | Path) -> Result:
    """Read back an answer :meth:`Result.save` wrote — a solve, off disk.

    Every reader answers what it answered in the session that solved: the
    values, the duals and activities, each named expression, and the reason
    behind anything the solve could not produce. A `Result` is frames and a
    few scalars, so none of it needs the build that made it or the solver
    that filled it — which is what makes an archived answer comparable with
    one solved today.

    Two things do not come back, both being facts about a session rather than
    about an answer: :attr:`~lpspec.relational.result.Result.kept` reads
    ``nothing``, this result holding no solver, and the solver's verbatim
    wording behind a refusal is not recorded — the termination condition is. A
    solve that reached no objective wrote null and reads back as ``nan``,
    which is what :attr:`~lpspec.relational.result.Result.objective` has to
    return, being a float.

    Args:
        directory: Where :meth:`~lpspec.relational.result.Result.save` wrote
            it. One that came out of an archive is
            :func:`~lpspec.archive.load_archive`'s to find.

    Returns:
        The result, read whole: the frames are in memory when this returns, so
        it owes *directory* nothing. :func:`scan_result` is the same answer left
        on disk.

    Raises:
        LayoutError: A directory holding no ``objective.parquet``, which is
            what every answer written there carries, or one whose layout has
            moved since it was written.
    """
    return _answer_under(Path(directory), _whole)


def scan_result(directory: str | Path) -> Result:
    """The answer under *directory*, read as its readers are called rather than now.

    :func:`load_result`'s other half, and the same value: every reader answers
    what that one's does. What differs is when the bytes move — each frame is
    a :func:`polars.scan_parquet` of the file it lies in, so an answer far
    larger than memory is readable a name at a time, and one whose names go
    unread costs nothing to open.

    The files stay where they are, so **they have to outlive the result**: a
    name read after the directory is gone raises where the scan is collected.

    Args:
        directory: As :func:`load_result` takes it.

    Raises:
        LayoutError: As :func:`load_result` raises it.
    """
    return _answer_under(Path(directory), pl.scan_parquet)


def _answer_under(out: Path, read: Reading) -> Result:
    """The saved answer under *out*, its frames read *read*'s way.

    Shared body of :func:`load_result` and :func:`scan_result`; only the
    reading differs.
    """
    record_file = out / RECORD_FILE
    if not record_file.is_file():
        raise LayoutError(
            f'{str(out)!r} holds no {RECORD_FILE!r}, so it is not an answer save() wrote. Every one '
            f'carries that record whether or not the solve produced values.'
        )
    check_format(out)
    record = Record(**pl.read_parquet(record_file).row(0, named=True))
    status = record.solve_status
    objective = float('nan') if record.objective is None else record.objective
    carried = {'_spec_digest': record.spec_digest, '_solved_at': record.solved_at}
    if not status.is_readable:
        return Result(status, objective, {}, {}, {}, 'nothing', **carried)

    no_duals, no_expressions = read_reasons(out)
    expressions: dict[str, Callable[[], pl.DataFrame]] = {
        name: (lambda frame=frame: frame.collect()) for name, frame in _saved_frames(out / 'expression', read).items()
    }
    expressions.update({name: _absent(why) for name, why in no_expressions.items()})
    return Result(
        status,
        objective,
        _saved_frames(out / 'primal', read),
        _saved_frames(out / 'dual', read),
        _saved_frames(out / 'activity', read),
        'nothing',
        expressions,
        no_duals,
        **carried,
    )
