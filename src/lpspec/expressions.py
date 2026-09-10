"""Expressions the file never named, valued against a model the language has already read.

The expressions are spliced into the model as named ones and the whole model is
lowered again, so an ad-hoc read passes every rule a declared read passes —
names resolved against the same flat namespace, dims checked, an absent name
refused with the language's own sentence. Everything but the nodes is then
thrown away: nothing is built, and no row exists that did not exist before.

It sits above both lanes because lowering reads the model **as written**, and
nothing under ``relational/`` may see that (docs/about/architecture.md, hard
rule 2). What crosses into a lane is the node.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from math_spec import to_program

from lpspec.errors import LanguageError, SchemaError

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
            mapping that carries ``cases:`` with its ``foreach:`` and
            ``otherwise:``.

    Returns:
        The node a declared named expression of *spec* lowers to. Held to no
        degree and free to read a ``dual``, the math reading nothing spliced.

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

    One lowering however many entries there are, which is what makes handing in
    a block cheaper than handing in each of its entries.

    Args:
        spec: As :func:`lower` takes it.
        carried: Entries an earlier call added, so that this one may read them
            the way it reads a declared name. They are spliced again and their
            nodes dropped: what comes back is what *added* asked for.
        added: A model fragment carrying ``expressions:`` and nothing else,
            each entry as :func:`lower` takes one.

    Returns:
        The nodes, keyed as *added* keys them, and everything to carry into a
        call after this one.

    Raises:
        LanguageError: A construct outside the language, a name *spec* does not
            declare, or an entry named after something *spec* already declares.
        SchemaError: A fragment carrying any section but ``expressions:``.
    """
    entries = _entries(added)
    written = spec.to_dict()
    _refuse_a_name_the_model_declares(written, entries)
    merged = dict(carried) | dict(entries)
    lowered = _splice(written, merged)
    return {name: lowered[name] for name in entries}, merged


def _splice(written: dict[str, Any], entries: Mapping[str, Any]) -> dict[str, ExpressionNode]:
    """*entries* added to the model *written* and lowered with it, as nodes.

    The lowered program is read for these nodes and dropped: its variables,
    constraints and objective are the ones the caller already solved, lowered a
    second time only so that what is spliced is checked against them.
    """
    written.setdefault(_SECTION, {}).update(entries)
    lowered = to_program(written).named_expressions
    return {name: lowered[name].expression for name in entries}


def _entries(added: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``expressions:`` of a fragment that declares nothing else.

    Reading is not building. A ``parameters:`` or ``lookups:`` entry would want
    a source attached, and a ``variables:``, ``constraints:``, ``sos:``,
    ``objective:`` or ``dimensions:`` entry would want rows that the solve
    being read never had — so a fragment carrying one is refused here rather
    than lowered into a model the answer does not belong to.

    The mapping is checked for before its keys are read, a ``str`` being
    iterable: an expression handed here would otherwise be reported one
    character at a time.
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

    Names share one flat namespace, so such an entry would both replace what
    the splice merged over and shadow the declaration the expression is there
    to read — silently, a dict update being the merge.
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

    Read off the model's own mappings rather than a list of sections, the way
    the language checks the same thing, so a section added later is covered.
    """
    return {name for section in written.values() if isinstance(section, dict) for name in section}
