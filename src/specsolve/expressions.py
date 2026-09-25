"""Expressions the file never named, valued against a spec the language has already read.

The expressions are spliced into the spec as named ones and the whole spec is
lowered again, so an ad-hoc read passes every rule a declared read passes; then
everything but the nodes is thrown away. It sits above both lanes because
lowering reads the spec **as written**, which nothing under ``relational/`` may
see (docs/about/architecture.md, hard rule 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from specsolve.lanes import lowered

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mathspec import Spec
    from mathspec.program import Expression

#: The name a single unnamed expression is spliced under. Stepped over rather
#: than overwritten where a spec declares it — [`_free_name`][].
_EVALUATED = '_evaluated'

#: The one section a caller may hand in. Every other declaration needs data or
#: builds rows, and neither is a read — see [`_splice`][].
_SECTION = 'expressions'


def lower(spec: Spec, expression: str | Mapping[str, object]) -> Expression:
    """One unnamed expression as a plan node, read in *spec*'s namespace.

    Args:
        spec: The spec the expression is written against. It supplies every
            name the expression may use; one it does not declare is refused.
        expression: What one ``expressions:`` entry takes — a string, or the
            mapping carrying ``cases:`` with ``dims:`` and ``otherwise:``.

    Returns:
        The node a declared named expression of *spec* lowers to.

    Raises:
        LanguageError: A construct outside the language, or a name *spec* does
            not declare.
        SchemaError: Something that is not an expression.
    """
    written = spec.to_dict()
    name = _free_name(written)
    return _splice(written, {name: expression})[name]


def _splice(written: dict[str, object], entries: Mapping[str, object]) -> dict[str, Expression]:
    """*entries* added to the spec *written* and lowered with it, as nodes.

    The lowered program is read for these nodes and dropped — its variables,
    constraints and objective are the ones already solved, lowered again only to
    check what is spliced against them. Through the same door the lanes use, so
    an added name is held to the rules a declared one is.
    """
    section = written.get(_SECTION)
    written[_SECTION] = {**section, **entries} if isinstance(section, dict) else dict(entries)
    named = lowered(written).expressions
    return {name: named[name].expression for name in entries}


def _free_name(written: Mapping[str, object]) -> str:
    """A name no declaration in *written* holds — where an unnamed expression is spliced."""
    taken = _declared(written)
    name = _EVALUATED
    while name in taken:
        name += '_'
    return name


def _declared(written: Mapping[str, object]) -> set[str]:
    """Every name *written* declares.

    Read off the spec's own mappings, the way the language checks the same
    thing, so a section added later is covered.
    """
    return {name for section in written.values() if isinstance(section, dict) for name in section}
