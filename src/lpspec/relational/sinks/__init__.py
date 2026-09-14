"""Sinks: how a built model leaves the engine. See README.md.

**Two families.** A *solver* takes the tables and runs them (``solvers/``,
chosen by name); a *writer* renders them to a file (``writers/``, chosen by
suffix). They are directories, so ``tests/test_architecture.py`` reads
membership off the path.

``tables.py`` is what both read, and neither family imports the other.
``capabilities.py`` is what both *declare*, and the functions below are where
a caller's model meets those declarations — the only place the two families are
asked one question together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpspec.errors import LpspecError, unknown_name_message
from lpspec.relational.sinks import capabilities as caps
from lpspec.relational.sinks import sos
from lpspec.relational.sinks.capabilities import spelled
from lpspec.relational.sinks.solvers import SOLVERS, Solver, loaded, solver
from lpspec.relational.sinks.tables import Tables
from lpspec.relational.sinks.writers import WRITERS, writer

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from math_spec import program

__all__ = [
    'SOLVERS',
    'WRITERS',
    'Solver',
    'Tables',
    'ingestible',
    'loaded',
    'refusal',
    'relaxations',
    'sink_capabilities',
    'solver',
    'writer',
]


def sink_capabilities(name: str) -> caps.Capabilities:
    """What the sink called *name* can ingest — a solver name, or a suffix.

    Answered without importing the solver: a capability is a declared table
    rather than a probe, so a repository of models can be checked against every
    sink they will eventually be solved on.

    Raises:
        LpspecError: A name belonging to neither family.
    """
    if name in SOLVERS:
        return SOLVERS[name].capabilities
    if (suffix := name.lower()) in WRITERS:
        return WRITERS[suffix].capabilities
    raise LpspecError(unknown_name_message('sink', name, (*SOLVERS, *WRITERS)))


def _blocker(name: str, needed: Collection[caps.Capability]) -> Callable[[Sequence[str]], str] | None:
    """Why the sink called *name* refuses capabilities *needed*, or ``None``.

    The one home for what "takes" means, so the refusal and the takers it names
    cannot disagree. What comes back is the message short of its third clause, a
    function of the takers.
    """
    table = sink_capabilities(name)
    if missing := table.missing(needed):
        return lambda takers: _sink_refuses_message(name, missing, takers)
    if combination := table.excluded(needed):
        return lambda takers: _sink_refuses_combination_message(name, sorted(combination), takers)
    return None


def refusal(program: program.Program, name: str) -> str | None:
    """Why the sink called *name* cannot take *program*, or ``None``.

    The refusal names **the construct, the sink, and the sinks that do take
    it**. Two shapes: a capability the sink lacks outright, and a pair it has
    both halves of and refuses together.
    """
    needed = caps.required(program)
    if (refuses := _blocker(name, needed)) is None:
        return None
    return refuses([other for other in (*SOLVERS, *WRITERS) if other != name and _blocker(other, needed) is None])


def _instead(takers: Sequence[str]) -> str:
    """The third clause of the refusal contract: who *does* take it."""
    if not takers:
        return 'No sink this build has takes it.'
    return f'Sinks that do take it: {", ".join(sorted(takers))}.'


def _sink_refuses_combination_message(sink: str, combination: Sequence[str], takers: Sequence[str]) -> str:
    """A sink that has both halves of a pair and refuses them together."""
    return (
        f'the {sink!r} sink takes {spelled(combination)} separately and refuses them together, '
        f'which is a limit of that solver rather than of the model. {_instead(takers)}'
    )


def _sink_refuses_message(sink: str, missing: Sequence[str], takers: Sequence[str]) -> str:
    """A sink asked for a capability it does not have at all."""
    return (
        f'the {sink!r} sink cannot take {spelled(missing)}: it has no such concept, so there is '
        f'nothing to hand the model to. {_instead(takers)}'
    )


def relaxations(program: program.Program, name: str) -> list[str]:
    """What the sink called *name* would rewrite to take *program*.

    Not refusals — the model solves — but it answers a question slightly
    different from the one asked.

    In :data:`~lpspec.relational.sinks.capabilities.CAPABILITIES` order.
    """
    table = sink_capabilities(name)
    needed = caps.required(program)
    return [
        _sink_reformulates_message(
            name,
            c,
            integrality_added=c in caps.REWRITTEN_AS_INTEGRALITY and 'integrality' not in needed,
        )
        for c in caps.CAPABILITIES
        if c in needed and table.support(c) == 'reformulated'
    ]


def _sink_reformulates_message(sink: str, capability: str, *, integrality_added: bool) -> str:
    """A sink meeting a capability by rewriting the model into one it takes.

    *integrality_added* marks a model that declared no integrality of its own
    and reaches the solver mixed-integer, so it comes back without duals.
    """
    cost = (
        ' The model declared no integrality of its own and reaches the solver mixed-integer, '
        'so it will come back without duals.'
        if integrality_added
        else ''
    )
    return (
        f'the {sink!r} sink has no native support for {spelled([capability])} and '
        f'will take it reformulated, so what reaches the solver is not what the file declares.{cost}'
    )


def ingestible(name: str, tables: Tables, program: program.Program | None = None) -> Tables:
    """*tables* in the form the named solver can take it — sets included.

    A solver that cannot ingest a special-ordered set is handed
    :func:`~lpspec.relational.sinks.sos.reformulated` tables, so everything that
    reads a solve back — the span check, the label slices — sees the one model
    the solver actually holds.

    *program* is what the refusal is decided on, and is optional. Given one, a
    model this sink cannot take is refused **here**, before the load; without it
    the refusal falls to the solver, which reports it as an error code from
    inside a library.

    Only ``reformulated`` capabilities are rewritten: a sink is never handed a
    rewrite of a construct it declared it has no concept of.

    Returns:
        *tables* itself where nothing has to change, which is every model
        declaring no sets.

    Raises:
        LpspecError: A *program* carrying a construct this sink has no concept
            of, or a combination it refuses.
    """
    if program is not None and (refused := refusal(program, name)) is not None:
        raise LpspecError(refused)
    if tables.sos.height and sink_capabilities(name).support('sos') == 'reformulated':
        return sos.reformulated(tables)
    return tables
