"""Hooks the documentation build runs."""

from __future__ import annotations

import re

#: A fenced block, indented or not — its contents are nobody's to rewrite, and
#: a ```math one is the superfences entry's in `mkdocs.yml`.
FENCED_BLOCK = re.compile(r'^[ \t]*```.*?^[ \t]*```$', re.DOTALL | re.MULTILINE)

#: GitHub's verbatim inline math, `$`…`$` — the pair math-spec's typesetter
#: prints so that GitHub's escape pass cannot reach into the span. A backtick on
#: either outer edge means a code span quoting the syntax rather than math using
#: it.
GITHUB_INLINE_MATH = re.compile(r'(?<!`)\$`(?P<math>[^`\n]+)`\$(?!`)')


def on_page_markdown(markdown: str, **kwargs: object) -> str:
    """Rewrite GitHub's verbatim inline math into the ``$…$`` arithmatex reads.

    ``math_spec.to_markdown`` prints GitHub-flavoured Markdown, where inline
    math is delimited ``$`…`$`` so that GitHub hands the span to MathJax
    untouched. Arithmatex has no syntax for it: python-markdown's own inline
    code processor claims the backtick span first. Nothing is escaped away on
    the way in, so ``$…$`` carries the same math the generated page was written
    with.

    A fenced block is left exactly as it is — the YAML a gallery page shows
    beside each equation is not math, and a ```math one is handled by the
    superfences entry in ``mkdocs.yml``, which reads it where it is indented
    inside a tab as well.

    Args:
        markdown: Page source, as the file holds it.
        **kwargs: Automatic mkdocs hook inputs.

    Returns:
        The same page with inline math arithmatex can find.
    """
    spans: list[str] = []
    kept = FENCED_BLOCK.sub(lambda m: spans.append(m[0]) or f'\x00{len(spans) - 1}\x00', markdown)
    rewritten = GITHUB_INLINE_MATH.sub(lambda m: f'${m["math"]}$', kept)
    return re.sub(r'\x00(\d+)\x00', lambda m: spans[int(m[1])], rewritten)
