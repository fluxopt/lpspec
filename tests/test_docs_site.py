"""The docs are read in two places; these are the checks that keep them honest in both.

``docs/`` is browsed on GitHub and served as a site, from one set of files. A
link *inside* ``docs/`` is relative and the build validates it — ``--strict``
in CI fails on a dead one. A link *outside* ``docs/`` cannot be relative,
because the site has no `../CONTRIBUTING.md` to resolve to, so it is written as
a full GitHub URL.

That convention is the whole mechanism, and it is unenforceable by the builder
in both directions: a relative link escaping ``docs/`` builds a silent 404, and
a blob URL is opaque to every checker there is — the file it names can be
deleted and nothing anywhere fails. Hence this module.

``docs/README.md`` used to be exempted here, because ``exclude_docs`` kept it
out of the site and its relative links out of the tree were correct on GitHub.
zensical builds no page from a ``README.md`` and validates its links anyway, so
the page follows the convention like every other one and the exemption is gone.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / 'docs'
REPO_URL = 'https://github.com/fluxopt/lpspec'
BLOB = f'{REPO_URL}/blob/main'

#: `](target)` and `[label]: target`, the two ways markdown names a destination.
_TARGETS = re.compile(r'\]\(\s*([^)\s]+)|^\[[^\]]+\]:\s+(\S+)', re.MULTILINE)

#: Already absolute, a bare fragment, or a protocol that names no path.
_ABSOLUTE = re.compile(r'^([a-z][a-z0-9+.-]*:|//|#|/)', re.IGNORECASE)


@functools.cache
def _pages() -> tuple[Path, ...]:
    """Every Markdown file under `docs/`, `README.md` included.

    `README.md` is the folder view GitHub renders rather than a page of the
    site, and it is held to the same two rules: it is read in the tree, where a
    link above `docs/` is as wrong as it is on the site if it is spelled
    relatively and the file moves.
    """
    return tuple(sorted(DOCS.rglob('*.md')))


def _targets(page: Path) -> list[str]:
    return [inline or reference for inline, reference in _TARGETS.findall(page.read_text())]


def test_no_relative_link_escapes_the_docs_tree():
    """The failure mkdocs cannot see.

    `[x](../CONTRIBUTING.md)` is correct in the repo and a 404 on the site.
    mkdocs resolves it against `docs/`, finds nothing above the root, and —
    because the target is outside the tree it knows about — does not treat it
    as a broken internal link. It just ships. Write the full GitHub URL.
    """
    escaping = []
    for page in _pages():
        for target in _targets(page):
            if _ABSOLUTE.match(target):
                continue
            path = target.partition('#')[0]
            if not path:
                continue
            resolved = (page.parent / path).resolve()
            if resolved != DOCS and DOCS not in resolved.parents:
                escaping.append(f'{page.relative_to(REPO)} -> {target}')
    assert not escaping, (
        f'relative links pointing outside docs/, which 404 on the site: {escaping}\nwrite them as {BLOB}/<path> instead'
    )


def test_every_blob_url_names_a_file_that_exists():
    """The other half: a blob URL is checked by nothing at all.

    mkdocs treats it as external and never follows it; the repo has no reason
    to notice it. So a page can go on pointing at `bench/results/latest.json`
    long after the file moves, and the first report is a reader hitting
    GitHub's 404.
    """
    broken = []
    for page in _pages():
        for target in _targets(page):
            if not target.startswith(BLOB):
                continue
            relative = target.removeprefix(f'{BLOB}/').partition('#')[0]
            if not (REPO / relative).exists():
                broken.append(f'{page.relative_to(REPO)} -> {relative}')
    assert not broken, f'links to repo files that no longer exist: {broken}'


def test_links_to_our_own_files_are_all_spelled_the_same_way():
    """One spelling, so the check above cannot be dodged.

    A link at a file in this repo written any other way — `tree/`, `raw/`, a
    permalinked sha, a branch that will vanish — reaches the right page today
    and is skipped by the existence check, which only recognises `blob/main`.
    Issue and PR links are not file links and are left alone.
    """
    file_shaped = re.compile(rf'^{re.escape(REPO_URL)}/(blob|tree|raw|blame)/')
    stray = [
        f'{page.relative_to(REPO)} -> {target}'
        for page in _pages()
        for target in _targets(page)
        if file_shaped.match(target) and not target.startswith(f'{BLOB}/')
    ]
    assert not stray, f'links at repo files not written as {BLOB}/<path>: {stray}'


def test_the_convention_is_actually_in_use():
    """A guard on the guards.

    Every assertion above passes vacuously on a docs tree with no outbound
    links at all — including one where a bad refactor stripped them. Pin that
    the arrangement they describe exists.
    """
    urls = [t for page in _pages() for t in _targets(page) if t.startswith(BLOB)]
    assert len(urls) >= 15, f'expected the docs to link out to the repo; found {len(urls)}'


def _nav_pages(entries: list[Any]) -> list[str]:
    """Every page the nav points at, depth first, as `mkdocs.yml` spells it.

    An entry is a bare path or a one-key mapping whose value is a path or a
    deeper list. A value that is not a page in this tree — an address on
    another site, the hand-written chart page — is not one of these.
    """
    found: list[str] = []
    for entry in entries:
        target = entry if isinstance(entry, str) else next(iter(entry.values()))
        if isinstance(target, list):
            found.extend(_nav_pages(target))
        elif isinstance(target, str) and target.endswith('.md'):
            found.append(target)
    return found


def test_every_page_under_docs_has_a_nav_entry():
    """The strict build stopped asking this when the site moved to zensical.

    mkdocs failed the build on a page with no nav entry, under
    `validation.nav.omitted_files`. zensical validates links and leaves
    navigation alone, so an orphan page builds, ships and is reachable only by
    search. Both directions are asked here, because a nav entry naming a file
    that is not there is dropped just as quietly.

    `README.md` is the folder view GitHub renders and the site builds no page
    from it, so it is the one file under `docs/` that belongs in no nav.
    """
    config = yaml.safe_load((REPO / 'mkdocs.yml').read_text())
    nav = set(_nav_pages(config['nav']))
    pages = {page.relative_to(DOCS).as_posix() for page in _pages()} - {'README.md'}
    assert pages == nav, (
        f'pages with no nav entry in mkdocs.yml: {sorted(pages - nav)}; nav entries with no page: {sorted(nav - pages)}'
    )


def test_the_home_page_still_carries_its_math_block():
    """`tools.gallery_math --check` also fills the tabs on `docs/index.md`, and
    it fills what it finds — a page whose markers were dropped in an edit stops
    being checked without anything failing. Pin that they are there.

    The content itself is not asserted here; that is the generator's job, and
    `test_the_gallery_math_is_current` runs it.
    """
    from tools import gallery_math

    page = (DOCS / 'index.md').read_text()
    assert gallery_math.HOME_BEGIN in page and gallery_math.HOME_END in page, (
        f'docs/index.md lost its {gallery_math.HOME_BEGIN}/{gallery_math.HOME_END} markers — '
        f'the LaTeX tabs are generated, and an unmarked page silently opts out'
    )


# --------------------------------------------------------------------------
# the ten rules, and the pages that elaborate them
# --------------------------------------------------------------------------

LANGUAGE = DOCS / 'reference' / 'language'
RULES = LANGUAGE / 'index.md'

#: A rule row: `| 7 | text | [Absence](absence.md#how-absence-travels) |`
_RULE_ROW = re.compile(r'^\|\s*(\d+)\s*\|(.+?)\|([^|]*)\|\s*$', re.MULTILINE)


def _rules() -> list[tuple[str, str, str]]:
    text = RULES.read_text()
    start = text.index('## Ten rules the language reduces to')
    return _RULE_ROW.findall(text[start : text.index('\n## The pages', start)])


def _headings(page: Path) -> set[str]:
    """Every heading in *page* as GitHub would slug it."""
    slugs = set()
    for line in page.read_text().splitlines():
        if line.startswith('#'):
            title = line.lstrip('#').strip()
            slugs.add(re.sub(r'[^a-z0-9 -]', '', title.lower()).replace(' ', '-'))
    return slugs


# --------------------------------------------------------------------------
# the operators as math


# --------------------------------------------------------------------------
# every construct as math


# --------------------------------------------------------------------------
# the error tree


# the lane, as a translation


def test_the_translation_table_names_every_built_in_operator():
    """`What a construct becomes` is a copy of the builder, so something checks it.

    The page's own rule, one section up: a copy nobody checks is a copy that
    rots. What it would rot into is a reader believing the lane translates a
    construct it no longer has, or — worse for the oracle — missing one it
    gained, since an operator with no row is an operator nobody wrote down the
    linopy call for.
    """
    from math_spec import BUILTIN_NAMES

    page = (DOCS / 'about' / 'linopy.md').read_text()
    section = page.split('### What a construct becomes')[1].split('### The same language')[0]
    expressions = section.split('| In an expression |')[1].split('| A `where:` |')[0]
    shown = set(re.findall(r'^\| `(\w+)\(', expressions, re.MULTILINE))

    assert shown == set(BUILTIN_NAMES), (
        f"the translation table shows {sorted(shown)} against the language's "
        f'{sorted(BUILTIN_NAMES)} — every built-in needs the linopy call it becomes'
    )


def _gen_bus_walk(program: Any) -> Any:
    """One map walked one way — what a `by=` node stands on, whichever operator takes it."""
    gen_bus = program.RelationDeclaration('gen_bus', (('g', 'g'), ('bus', 'bus')), ('g',))
    return program.Walk(gen_bus, ('g',), ('bus',), ())


def test_the_plan_table_names_every_expression_node():
    """`The plan, node for node` is a copy of two dispatches, so something checks it.

    The same rule the table above answers to, one layer down: a node with no
    row is a node whose two readings nobody wrote down, and the row is where a
    reader learns that the lanes agree at all. The fan-in cell is read back
    off the language, since that column is one the compiler *acts* on rather
    than merely documents.
    """
    from math_spec import program

    page = (DOCS / 'about' / 'architecture.md').read_text()
    section = page.split('## The plan, node for node')[1].split('## The relational lane')[0]
    rows = dict(re.findall(r'^\| `(\w+)` \|[^|]*\| ([^|]*) \|', section, re.MULTILINE))

    x = program.Variable('x')
    nodes = {
        type(node).__name__: node
        for node in (
            program.Constant(1.0),
            program.Parameter('p'),
            x,
            program.Dual('c'),
            program.Negate(x),
            program.Add(x, x),
            program.Multiply(x, x),
            program.Divide(x, program.Parameter('p')),
            program.Power(program.Parameter('p'), program.Constant(2.0)),
            program.Sum(x, ('t',)),
            program.GroupSum(x, _gen_bus_walk(program)),
            program.At(x, _gen_bus_walk(program)),
            program.Translate(x, 't', 1, wrap=False),
            program.Window(x, 't', 3, wrap=False),
            program.Cases((program.Region(program.Mask(program.BooleanLiteralNode(True)), x),)),
        )
    }
    assert set(nodes) == {c.__name__ for c in program.Expression.__subclasses__()}, (
        'the instances below stand for every expression node, so a node added to the language is one here too'
    )
    assert set(rows) == set(nodes), (
        f"the table shows {sorted(rows)} against the plan's {sorted(nodes)} — "
        f'every expression node needs the two readings it becomes'
    )
    shown = {name: cell.strip() for name, cell in rows.items()}
    declared = {name: program.fan_in(node) for name, node in nodes.items()}
    assert shown == declared, f'the table calls these {shown}, the language answers {declared}'
