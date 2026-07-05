"""Tests for the HTML STRUCTURED-tier adapter.

`HtmlAdapter` (src/archex/parse/adapters/html.py) builds on the M11
`StructuredAdapter` base to produce an element/script/style outline for
`.html`/`.htm` files without ever claiming programming symbols. M12's
tier-flip PR lands `html` at `LanguageTier.STRUCTURED` for real in
`archex.languages`, so every test below builds `HtmlAdapter()` straight off
the production registry entry -- no monkeypatched stand-in is needed
anymore. This module also covers the local `script`/`link`/`img`/`a`
reference extraction and resolution that ships alongside the tier flip:
`extract_references` and `resolve_import` only ever surface *local*
file-path references, never a claimed programming symbol.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from archex.api import file_outline
from archex.languages import get_language_tier
from archex.models import (
    ChunkRange,
    Config,
    ImportStatement,
    LanguageTier,
    RepoSource,
    SymbolKind,
    SymbolOutline,
)
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.html import HtmlAdapter
from archex.parse.engine import TreeSitterEngine
from tests.conftest import _init_fixture_repo  # pyright: ignore[reportPrivateUsage]

FIXTURES_DIR = "tests/fixtures/html_structured"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> HtmlAdapter:
    return HtmlAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "html")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: HtmlAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


def test_html_registered_at_structured_tier() -> None:
    """M12 flips `html` to STRUCTURED for real in `archex.languages` -- this
    pins that registry fact directly, independent of any single adapter
    call, so a tier regression fails here even if some other test's mocks
    would otherwise mask it."""
    assert get_language_tier("html") == LanguageTier.STRUCTURED


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: HtmlAdapter) -> None:
    assert adapter.language_id == "html"


def test_file_extensions(adapter: HtmlAdapter) -> None:
    assert adapter.file_extensions == [".html", ".htm"]


def test_tree_sitter_name(adapter: HtmlAdapter) -> None:
    assert adapter.tree_sitter_name == "html"


# ---------------------------------------------------------------------------
# extract_chunk_ranges: element / script_element / style_element outline
# ---------------------------------------------------------------------------


def test_extract_chunk_ranges_covers_element_script_and_style_siblings(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """Each declared chunk node type produces its own outline entry when the
    nodes are siblings instead of nested inside one another -- proves
    `script_element`/`style_element` are genuinely honored, not just
    `element`."""
    source = (
        b"<script>const ready = true;</script>\n"
        b"<style>\n"
        b"body { margin: 0; }\n"
        b"</style>\n"
        b"<div>\n"
        b"  <p>hello</p>\n"
        b"</div>\n"
    )

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "fragment.html")

    assert ranges == [
        ChunkRange(start_line=1, end_line=1),
        ChunkRange(start_line=2, end_line=4),
        ChunkRange(start_line=5, end_line=7),
    ]


def test_extract_chunk_ranges_collapses_nested_element_into_enclosing_outline_entry(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """A nested `<span>` is not a second outline entry -- it folds into its
    enclosing `<div>`'s range, the same non-overlapping-outermost-wins rule
    `ChunkOnlyAdapter` uses."""
    source = b"<div>\n  <span>inner</span>\n</div>\n"

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "nested.html")

    assert ranges == [ChunkRange(start_line=1, end_line=3)]


def test_extract_chunk_ranges_on_realistic_fixture_outlines_the_document_element(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """Against a full nested html/head/body/main document, the root `<html>`
    element is the sole outline entry: head, body, main, link, script, and
    style are all nested inside it and fold away."""
    with open(f"{FIXTURES_DIR}/index.html", "rb") as f:
        source = f.read()

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "index.html")

    assert ranges == [ChunkRange(start_line=2, end_line=36)]


# ---------------------------------------------------------------------------
# extract_symbols: never claims programming symbols (M12 invariant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "script_body"),
    [
        ("function_declaration", b"function greet(name) {\n  return name;\n}\n"),
        ("class_declaration", b"class Widget {\n  render() {}\n}\n"),
        ("arrow_function_const", b"const add = (a, b) => a + b;\n"),
        ("object_literal_method", b"const api = {\n  get() { return 1; }\n};\n"),
    ],
)
def test_extract_symbols_ignores_function_and_class_shaped_javascript(
    engine: TreeSitterEngine, adapter: HtmlAdapter, shape: str, script_body: bytes
) -> None:
    source = b"<script>\n" + script_body + b"</script>\n"

    symbols = adapter.extract_symbols(parse(engine, source), source, "widget.html")

    assert symbols == [], f"{shape} leaked a symbol from embedded JavaScript"


def test_extract_symbols_on_realistic_fixture_with_function_and_class_in_script(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """The fixture's embedded `<script>` defines a real `function` and a real
    `class` -- HTML STRUCTURED must still report zero symbols for the file."""
    with open(f"{FIXTURES_DIR}/index.html", "rb") as f:
        source = f.read()
    text = source.decode("utf-8")
    assert "function computeTotal" in text
    assert "class Cart" in text

    symbols = adapter.extract_symbols(parse(engine, source), source, "index.html")

    assert symbols == []


# ---------------------------------------------------------------------------
# extract_references: local script/link/img/a reference extraction (M12)
# ---------------------------------------------------------------------------


def test_extract_references_extracts_local_script_link_img_and_anchor_targets(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """Against the realistic fixture, the adapter extracts exactly the four
    local reference-bearing attributes -- `<link href>`, `<script src>`,
    `<a href>`, `<img src>` -- in document order, each pinned to its source
    line. The fixture's *inline* `<script>`/`<style>` blocks carry no
    `src`/`href` attribute and must not contribute an entry."""
    with open(f"{FIXTURES_DIR}/index.html", "rb") as f:
        source = f.read()

    references = adapter.extract_references(parse(engine, source), source, "index.html")

    assert [(imp.module, imp.line, imp.is_relative) for imp in references] == [
        ("./styles/main.css", 6, True),
        ("./scripts/app.js", 7, True),
        ("./about.html", 12, True),
        ("./images/logo.png", 13, True),
    ]
    assert all(imp.file_path == "index.html" for imp in references)


# ---------------------------------------------------------------------------
# extract_references: external / non-file targets never become references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "snippet"),
    [
        ("https_scheme", b'<a href="https://example.com/about.html">About</a>\n'),
        ("http_scheme", b'<script src="http://cdn.example.com/lib.js"></script>\n'),
        ("protocol_relative", b'<script src="//cdn.example.com/lib.js"></script>\n'),
        ("data_uri", b'<img src="data:image/png;base64,AAAA">\n'),
        ("mailto_scheme", b'<a href="mailto:hello@example.com">Mail</a>\n'),
        ("javascript_scheme", b'<a href="javascript:void(0)">Click</a>\n'),
        ("fragment_only", b'<a href="#section-2">Jump</a>\n'),
    ],
)
def test_extract_references_ignores_external_and_non_file_targets(
    engine: TreeSitterEngine, adapter: HtmlAdapter, case: str, snippet: bytes
) -> None:
    references = adapter.extract_references(parse(engine, snippet), snippet, "page.html")

    assert references == [], f"{case} produced a reference for a non-local target"


# ---------------------------------------------------------------------------
# extract_references: query strings and fragments are stripped
# ---------------------------------------------------------------------------


def test_extract_references_strips_query_string_and_fragment_before_resolution(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """A same-page-relative link decorated with a query string and a
    fragment resolves as if neither were present -- both are stripped from
    the extracted module before `resolve_import` ever sees it."""
    source = b'<a href="./about.html?tab=2#intro">About</a>\n'

    references = adapter.extract_references(parse(engine, source), source, "index.html")

    assert len(references) == 1
    assert references[0].module == "./about.html"
    resolved = adapter.resolve_import(references[0], {"about.html": "about.html"})
    assert resolved == "about.html"


# ---------------------------------------------------------------------------
# resolve_import: relative and root-relative resolution
# ---------------------------------------------------------------------------


def test_resolve_import_resolves_relative_reference_against_containing_file_directory(
    adapter: HtmlAdapter,
) -> None:
    """A `./`-relative reference resolves against the directory of the HTML
    file that contains it, not against the repo root."""
    imp = ImportStatement(
        module="./about.html", file_path="pages/index.html", line=12, is_relative=True
    )

    resolved = adapter.resolve_import(imp, {"pages/about.html": "pages/about.html"})

    assert resolved == "pages/about.html"


def test_resolve_import_root_relative_reference_ignores_containing_file_nesting(
    adapter: HtmlAdapter,
) -> None:
    """A `/`-rooted reference resolves against the repo root file map -- the
    containing file's own directory nesting must not be joined in, unlike a
    relative reference."""
    imp = ImportStatement(
        module="/assets/shared/logo.png",
        file_path="deeply/nested/pages/about.html",
        line=3,
        is_relative=False,
    )

    resolved = adapter.resolve_import(
        imp, {"assets/shared/logo.png": "/repo/assets/shared/logo.png"}
    )

    assert resolved == "/repo/assets/shared/logo.png"


def test_resolve_import_returns_none_for_unresolvable_local_reference(adapter: HtmlAdapter) -> None:
    imp = ImportStatement(
        module="./missing.html", file_path="pages/index.html", line=1, is_relative=True
    )

    assert adapter.resolve_import(imp, {"pages/about.html": "pages/about.html"}) is None


def test_extract_and_resolve_all_reference_kinds_against_realistic_fixture(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """script src, link href, img src, and a href in the fixture all resolve
    against a file map keyed by the fixture's own relative layout -- the
    containing file (`index.html`) sits beside every referenced target, so
    each reference resolves without a directory prefix."""
    with open(f"{FIXTURES_DIR}/index.html", "rb") as f:
        source = f.read()

    references = adapter.extract_references(parse(engine, source), source, "index.html")
    file_map = {
        "styles/main.css": "styles/main.css",
        "scripts/app.js": "scripts/app.js",
        "about.html": "about.html",
        "images/logo.png": "images/logo.png",
    }

    resolved = {imp.module: adapter.resolve_import(imp, file_map) for imp in references}

    assert resolved == {
        "./styles/main.css": "styles/main.css",
        "./scripts/app.js": "scripts/app.js",
        "./about.html": "about.html",
        "./images/logo.png": "images/logo.png",
    }


# ---------------------------------------------------------------------------
# Reference extraction never becomes a programming-symbol claim (M12 invariant)
# ---------------------------------------------------------------------------


def test_extract_symbols_stays_empty_while_extract_references_finds_local_targets(
    engine: TreeSitterEngine, adapter: HtmlAdapter
) -> None:
    """M12 adds reference extraction on top of the M11 STRUCTURED base -- it
    must not also start claiming programming symbols. On the realistic
    fixture, references are non-empty but symbols remain empty."""
    with open(f"{FIXTURES_DIR}/index.html", "rb") as f:
        source = f.read()
    tree = parse(engine, source)

    references = adapter.extract_references(tree, source, "index.html")
    symbols = adapter.extract_symbols(tree, source, "index.html")

    assert references != []
    assert symbols == []


# ---------------------------------------------------------------------------
# archex.api.file_outline: end-to-end M12 outline acceptance
# ---------------------------------------------------------------------------


def test_file_outline_returns_html_outline_and_local_references_end_to_end(
    tmp_path: Path,
) -> None:
    """M12 acceptance: `archex.api.file_outline` against the realistic fixture
    surfaces the HTML element outline plus the local references its
    `<head>`/`<body>` declare -- without ever claiming a function/class/
    method/interface symbol for markup. File-outline reference resolution uses
    the repository's actual file tree, so local CSS, JavaScript, HTML, and image
    targets all resolve to fixture-relative paths without adding graph edges."""
    repo = _init_fixture_repo(tmp_path, "html_structured")
    source = RepoSource(local_path=str(repo))

    result = file_outline(
        source, file_path="index.html", config=Config(languages=["html"], cache=False)
    )

    assert result.language == "html"
    assert result.symbols == []
    assert [(item.start_line, item.end_line) for item in result.outline_ranges] == [(1, 1), (2, 36)]

    def _iter_kinds(symbols: Sequence[SymbolOutline]) -> list[SymbolKind]:
        kinds: list[SymbolKind] = []
        for sym in symbols:
            kinds.append(sym.kind)
            kinds.extend(_iter_kinds(sym.children))
        return kinds

    programming_kinds = {
        SymbolKind.FUNCTION,
        SymbolKind.CLASS,
        SymbolKind.METHOD,
        SymbolKind.INTERFACE,
    }
    assert not programming_kinds & set(_iter_kinds(result.symbols))

    resolved_by_module = {ref.module: ref.resolved_path for ref in result.references}
    assert resolved_by_module == {
        "./styles/main.css": "styles/main.css",
        "./scripts/app.js": "scripts/app.js",
        "./about.html": "about.html",
        "./images/logo.png": "images/logo.png",
    }
