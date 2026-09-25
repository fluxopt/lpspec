"""What a sink can ingest — the axis that is not the ceiling.

The ceiling is about streamability and is solver-independent (mathspec's
docs/about/limits.md); what a *sink* can take is separate. One descriptor per
sink, so a construct the language says and a sink cannot take is a refusal
naming both rather than a ``kError`` from inside a library.

Four shapes:

- **Two-valued.** A sink takes a construct or it does not; nothing is
  rewritten at the hand-off. A set on a sink with no SOS concept is refused,
  and the refusal names the expansion that writes it out as binaries and rows.
- **Exclusions.** HiGHS takes a Hessian, takes integrality, and refuses the
  pair.
- **Some entries are data-time.** Convexity is a property of coefficients, so
  ``check`` cannot answer it (rule 2). ``nonconvex_quadratic_objective`` is
  declared anyway: the sink that discovers it at solve time reads the sinks
  that would have taken it off this table.
- **A descriptor describes the sink as shipped, not the library it wraps.** A
  solver that takes a Hessian through an entry point this package does not call
  cannot ingest a quadratic objective *here*. An entry moves when the hand-off
  lands, so the benchmarks table may be ahead of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from mathspec.program import Program

#: What a model may need a sink to have. ``indicator`` and ``semi-continuous``
#: are absent: they have rows in the benchmarks table but no spelling in the
#: language.
Capability = Literal[
    'integrality',
    'sos',
    'quadratic_objective',
    'nonconvex_quadratic_objective',
    'quadratic_constraint',
]

#: Whether a sink takes one capability.
Support = Literal['native', 'absent']

CAPABILITIES: tuple[Capability, ...] = get_args(Capability)


@dataclass(frozen=True)
class Capabilities:
    """One sink's answer for every capability, and the pairs it refuses.

    Attributes:
        supports: What the sink does with each capability it has; a name left
            out is ``absent``, so a descriptor lists only what it *can* do.
        excludes: Sets of capabilities it has individually and refuses together.
    """

    supports: Mapping[Capability, Support]
    excludes: tuple[frozenset[Capability], ...] = ()

    def __post_init__(self) -> None:
        """Take a read-only copy of *supports*, which a sink holds as a ``ClassVar``."""
        object.__setattr__(self, 'supports', MappingProxyType(dict(self.supports)))

    def support(self, capability: Capability) -> Support:
        """What this sink does with *capability* — ``absent`` where it says nothing."""
        return self.supports.get(capability, 'absent')

    def missing(self, required: Collection[Capability]) -> list[Capability]:
        """Those of *required* this sink cannot take at all.

        In :data:`CAPABILITIES` order rather than the caller's.
        """
        return [c for c in CAPABILITIES if c in required and self.support(c) == 'absent']

    def excluded(self, required: Collection[Capability]) -> frozenset[Capability] | None:
        """The first conjunction *required* contains that this sink refuses.

        Returns:
            The excluded set, or ``None``. Each member is one the sink supports
            on its own; one it simply lacks is :meth:`missing`'s answer.
        """
        for combination in self.excludes:
            if combination <= set(required):
                return combination
        return None


def required(program: Program, /) -> frozenset[Capability]:
    """What *program* needs a sink to have, decided with no data attached.

    Exactly what the model declares.

    Only what rule 2 can decide appears here, so convexity never does.
    """
    footprint = program.footprint
    needed: set[Capability] = set()
    if footprint.domains - {'continuous'}:
        needed.add('integrality')
    if 'objective' in footprint.quadratic:
        needed.add('quadratic_objective')
    if 'constraint' in footprint.quadratic:
        needed.add('quadratic_constraint')
    if footprint.sos_types:
        needed.add('sos')
    return frozenset(needed)


#: How a capability reads in a sentence.
_SPELLED: Mapping[str, str] = {
    'integrality': 'binary or integer variables',
    'sos': 'special-ordered sets (`sos:`)',
    'quadratic_objective': 'a quadratic objective',
    'nonconvex_quadratic_objective': 'a nonconvex quadratic objective',
    'quadratic_constraint': 'a quadratic constraint',
}


def spelled(capabilities: Collection[str]) -> str:
    """Capabilities as a refusal names them."""
    return ', '.join(_SPELLED[c] for c in capabilities)
