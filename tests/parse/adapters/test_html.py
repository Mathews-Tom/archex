"""Tests for the HTML STRUCTURED-tier adapter.

`HtmlAdapter` (src/archex/parse/adapters/html.py) builds on the M11
`StructuredAdapter` base to produce an element/script/style outline for
`.html`/`.htm` files without ever claiming programming symbols. `html`
remains registered at `LanguageTier.CHUNK_ONLY` in `archex.languages` until
M12's tier-flip PR lands on top of this one, so every test that instantiates
`HtmlAdapter` first monkeypatches a STRUCTURED-tier copy of the existing
registry entry. Local script/link/img/anchor reference extraction is out of
scope here -- that lands in the follow-up PR that overrides
`extract_references`.
"""

from __future__ import annotations

import pytest

from archex import languages
from archex.languages import LanguageSupport
from archex.models import ChunkRange, LanguageTier
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.html import HtmlAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = "tests/fixtures/html_structured"


@pytest.fixture
def _html_structured_registered(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register `html` at STRUCTURED tier for the duration of one test.

    Copies extensions/pack_name/chunk_node_types straight from the real
    (still CHUNK_ONLY) registry entry so only the tier under test changes --
    the actual `languages.py` flip is a separate, later PR.
    """
    existing = languages.LANGUAGE_SUPPORT["html"]
    stub = LanguageSupport(
        language_id="html",
        display_name=existing.display_name,
        extensions=existing.extensions,
        tier=LanguageTier.STRUCTURED,
        pack_name=existing.pack_name,
        chunk_node_types=existing.chunk_node_types,
    )
    monkeypatch.setitem(languages.LANGUAGE_SUPPORT, "html", stub)


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter(_html_structured_registered: None) -> HtmlAdapter:
    return HtmlAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "html")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: HtmlAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


def test_rejects_instantiation_while_html_registry_entry_is_still_chunk_only() -> None:
    """PR-1 ships the adapter class before the registry tier flip: building it
    against the real, unpatched registry (still CHUNK_ONLY) must fail loudly
    instead of silently accepting the mismatched tier."""
    with pytest.raises(ValueError, match="registered as"):
        HtmlAdapter()


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
