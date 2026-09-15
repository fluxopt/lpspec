"""Expressions the file never named, valued against a model the language has already read.

The expressions are spliced into the model as named ones and the whole model is
lowered again, so an ad-hoc read passes every rule a declared read passes; then
everything but the nodes is thrown away. It sits above both lanes because
lowering reads the model **as written**, which nothing under ``relational/`` may
see (docs/about/architecture.md, hard rule 2).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lpspec.errors import LanguageError, SchemaError
from lpspec.lanes import lowered

if TYPE_CHECKING:
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
            mapping carrying ``cases:`` with ``foreach:`` and ``otherwise:``.

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


def lower_all(
    spec: Spec,
    carried: Mapping[str, Any],
    added: Mapping[str, Any],
) -> tuple[dict[str, ExpressionNode], dict[str, Any]]:
    """Every entry of an ``expressions:`` block as a plan node, read in *spec*'s namespace.

    One lowering, however many entries the block has.

    Args:
        spec: As :func:`lower` takes it.
        carried: Entries an earlier call added, spliced again so this one may
            read them like a declared name; their nodes are dropped.
        added: A model fragment carrying ``expressions:`` and nothing else,
            each entry as :func:`lower` takes one.

    Returns:
        The nodes keyed as *added* keys them, and everything to carry into the
        next call.

    Raises:
        LanguageError: A construct outside the language, a name *spec* does not
            declare, or an entry named after something *spec* already declares.
        SchemaError: A fragment carrying any section but ``expressions:``.
    """
    entries = _entries(added)
    written = spec.to_dict()
    _refuse_a_name_the_model_declares(written, entries)
    merged = dict(carried) | dict(entries)
    nodes = _splice(written, merged)
    return {name: nodes[name] for name in entries}, merged


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


def _entries(added: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``expressions:`` of a fragment that declares nothing else.

    Reading is not building: a ``parameters:`` or ``lookups:`` entry would want
    a source, and ``variables:``/``constraints:``/``sos:``/``objective:``/
    ``dimensions:`` would want rows the solve never had. The mapping is checked
    before its keys are read — a ``str`` is iterable, and an expression handed
    here would otherwise be reported one character at a time.
    """
    if not isinstance(added, Mapping):
        raise SchemaError(
            f'a fragment to read expressions from is a mapping of sections, and this is a '
            f"{type(added).__name__}. One expression is evaluate()'s argument, not a block of them; "
            f"YAML text is yaml.safe_load()'s, and what that returns is what belongs here."
        )
    if strays := sorted(set(added) - {_SECTION}):
        raise SchemaError(
            f'a fragment to read expressions from carries {strays}, and reading takes '
            f"'{_SECTION}:' and nothing else: a parameter would want data attached and a variable "
            f'or a constraint would want rows, neither of which a solved model can grow. Declare '
            f'them in the model and build it.'
        )
    entries = added.get(_SECTION, {})
    if not entries:
        raise SchemaError(f"a fragment to read expressions from declares no '{_SECTION}:' entry, so it reads nothing.")
    return entries


def _refuse_a_name_the_model_declares(written: Mapping[str, Any], entries: Mapping[str, Any]) -> None:
    """Refuse an entry named after a declaration of *written*.

    Names share one flat namespace, so such an entry would shadow the very
    declaration the expression is there to read — silently, a dict update being
    the merge.
    """
    if clash := sorted(set(entries) & _declared(written)):
        raise LanguageError(
            f'{clash} name declarations this model already makes, and names share one flat namespace, '
            f'so reading under those names would shadow what the expressions are written to read. '
            f'Rename them.'
        )


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
