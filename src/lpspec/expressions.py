"""One expression, valued against a model the language has already read.

Both lanes read back a quantity the file never named, and neither owns this:
the expression is spliced into the model as a named expression and the whole
model is lowered again, so an ad-hoc read passes every rule a declared read
passes — names resolved against the same flat namespace, dims checked, an
absent name refused with the language's own sentence.

It sits above both lanes because lowering reads the model **as written**, and
nothing under ``relational/`` may see that (docs/about/architecture.md, hard
rule 2). What crosses into a lane is the node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from math_spec import to_program

if TYPE_CHECKING:
    from collections.abc import Mapping

    from math_spec import Spec
    from math_spec.program import ExpressionNode

#: The name the expression under evaluation is spliced under. Stepped over
#: rather than overwritten where a model declares it — :func:`_free_name`.
_EVALUATED = '_evaluated'


def lower(spec: Spec, expression: str | Mapping[str, Any]) -> ExpressionNode:
    """*expression* as a plan node, read in *spec*'s namespace.

    The whole model is lowered again, so this costs what :func:`lpspec.check`
    costs and nothing a lane can amortise. A declared name is cheaper read
    through the reader that already holds it.

    Args:
        spec: The model the expression is written against. It supplies every
            name the expression may use; one it does not declare is refused.
        expression: What ``expressions:`` takes — a string, or the mapping
            that carries ``cases:`` with its ``foreach:`` and ``otherwise:``.

    Returns:
        The node a declared named expression of *spec* lowers to. Held to no
        degree and free to read a ``dual``, the math reading nothing spliced.

    Raises:
        LanguageError: A construct outside the language, or a name *spec* does
            not declare.
        SchemaError: A mapping that is not a named expression.
    """
    written = spec.to_dict()
    name = _free_name(written)
    written.setdefault('expressions', {})[name] = expression
    return to_program(written).named_expressions[name].expression


def _free_name(written: Mapping[str, Any]) -> str:
    """A name no declaration in *written* holds.

    Names share one flat namespace, so a spliced entry landing on a declared
    one would shadow the very declaration the expression is there to read.
    Read off the model's own mappings rather than a list of sections, the way
    the language checks the same thing, so a section added later is covered.
    """
    taken = {name for section in written.values() if isinstance(section, dict) for name in section}
    name = _EVALUATED
    while name in taken:
        name += '_'
    return name
