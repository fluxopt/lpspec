"""Expressions the file never named, valued against a model the language has already read.

The expressions are spliced into the model as named ones and the whole model is
lowered again, so an ad-hoc read passes every rule a declared read passes; then
everything but the nodes is thrown away. It sits above both lanes because
lowering reads the model **as written**, which nothing under ``relational/`` may
see (docs/about/architecture.md, hard rule 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from specsolve.lanes import lowered

if TYPE_CHECKING:
    from collections.abc import Mapping

    from math_spec import Spec
    from math_spec.program import ExpressionNode

#: The name a single unnamed expression is spliced under. Stepped over rather
#: than overwritten where a model declares it — :func:`_free_name`.
_EVALUATED = '_evaluated'

#: The one section a caller may hand in. Every other declaration needs data or
#: builds rows, and neither is a read — see :func:`_entries`.
_SECTION = 'expressions'


def lower(spec: Spec, expression: str | Mapping[str, Any]) -> ExpressionNode:
    """One unnamed expression as a plan node, read in *spec*'s namespace.

    Args:
        spec: The model the expression is written against. It supplies every
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


def _splice(written: dict[str, Any], entries: Mapping[str, Any]) -> dict[str, ExpressionNode]:
    """*entries* added to the model *written* and lowered with it, as nodes.

    The lowered program is read for these nodes and dropped — its variables,
    constraints and objective are the ones already solved, lowered again only to
    check what is spliced against them. Through the same door the lanes use, so
    an added name is held to the rules a declared one is.
    """
    written.setdefault(_SECTION, {}).update(entries)
    named = lowered(written).named_expressions
    return {name: named[name].expression for name in entries}


def _free_name(written: Mapping[str, Any]) -> str:
    """A name no declaration in *written* holds — where an unnamed expression is spliced."""
    taken = _declared(written)
    name = _EVALUATED
    while name in taken:
        name += '_'
    return name


def _declared(written: Mapping[str, Any]) -> set[str]:
    """Every name *written* declares.

    Read off the model's own mappings, the way the language checks the same
    thing, so a section added later is covered.
    """
    return {name for section in written.values() if isinstance(section, dict) for name in section}
