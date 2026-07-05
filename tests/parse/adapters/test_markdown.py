"""Tests for the Markdown STRUCTURED-tier adapter.

`MarkdownAdapter` (src/archex/parse/adapters/markdown.py) builds on the
shared `StructuredAdapter` base to produce a section outline for
`.md`/`.markdown` files plus Markdown's native link and section-anchor
cross-reference mechanisms, without claiming programming symbols.
`markdown` is registered at `LanguageTier.STRUCTURED` for real in
`archex.languages`, so every test below builds `MarkdownAdapter()`
straight off the production registry entry -- no monkeypatched stand-in
is needed. Inline link/image targets are recovered by reparsing each
opaque `inline` block-grammar node with the companion `markdown_inline`
grammar (both bundled in `tree-sitter-language-pack`); reference-style
links (`[text][label]`) are resolved against `[label]: target`
definitions found anywhere in the same document; a fragment-only target
(`#heading`) is intra-document and resolves back to its own containing
file.
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
from archex.parse.adapters.markdown import MarkdownAdapter
from archex.parse.engine import TreeSitterEngine
from tests.conftest import _init_fixture_repo  # pyright: ignore[reportPrivateUsage]

FIXTURES_DIR = "tests/fixtures/markdown_structured"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> MarkdownAdapter:
    return MarkdownAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "markdown")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: MarkdownAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


def test_markdown_registered_at_structured_tier() -> None:
    """Pins that `markdown` is registered at STRUCTURED tier directly in
    the registry, independent of any single adapter call, so a tier
    regression fails here even if some other test's mocks would
    otherwise mask it."""
    assert get_language_tier("markdown") == LanguageTier.STRUCTURED


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: MarkdownAdapter) -> None:
    assert adapter.language_id == "markdown"


def test_file_extensions(adapter: MarkdownAdapter) -> None:
    assert adapter.file_extensions == [".md", ".markdown"]


def test_tree_sitter_name(adapter: MarkdownAdapter) -> None:
    assert adapter.tree_sitter_name == "markdown"


# ---------------------------------------------------------------------------
# extract_chunk_ranges: section outline
# ---------------------------------------------------------------------------


def test_extract_chunk_ranges_on_realistic_fixture_covers_the_whole_document(
    engine: TreeSitterEngine, adapter: MarkdownAdapter
) -> None:
    """The top-level `# Guide` section spans the whole document -- the
    nested `## Background` subsection folds into it, the same
    non-overlapping-outermost-wins rule `ChunkOnlyAdapter` uses."""
    with open(f"{FIXTURES_DIR}/index.md", "rb") as f:
        source = f.read()

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "index.md")

    assert ranges == [ChunkRange(start_line=1, end_line=13)]


# ---------------------------------------------------------------------------
# extract_symbols: never claims programming symbols
# ---------------------------------------------------------------------------


def test_extract_symbols_is_always_empty(
    engine: TreeSitterEngine, adapter: MarkdownAdapter
) -> None:
    with open(f"{FIXTURES_DIR}/index.md", "rb") as f:
        source = f.read()

    assert adapter.extract_symbols(parse(engine, source), source, "index.md") == []


def test_extract_symbols_ignores_fenced_code_blocks_with_real_function_and_class_definitions(
    engine: TreeSitterEngine, adapter: MarkdownAdapter
) -> None:
    """A fenced code block containing a genuine `def`/`class` definition
    is still just prose content to Markdown -- `extract_symbols` is
    `@final` on the `StructuredAdapter` base and can never be overridden
    to reach into a fenced block's language and claim its symbols."""
    source = (
        b"# Cart\n\n"
        b"```python\n"
        b"def compute_total(items):\n"
        b"    return sum(items)\n\n"
        b"class Cart:\n"
        b"    pass\n"
        b"```\n"
    )

    assert adapter.extract_symbols(parse(engine, source), source, "adversarial.md") == []


def test_extract_references_ignores_fenced_code_block_content(
    engine: TreeSitterEngine, adapter: MarkdownAdapter
) -> None:
    """The same fenced code block must not leak an import-shaped reference
    either -- only Markdown's own link/image/reference-definition syntax
    counts as a native cross-reference, never a fenced block's language
    content."""
    source = b"# Cart\n\n```python\nfrom ./sibling import helper\n```\n"

    assert adapter.extract_references(parse(engine, source), source, "adversarial.md") == []


# ---------------------------------------------------------------------------
# extract_references: link / image / reference-style / section-anchor extraction
# ---------------------------------------------------------------------------


def test_extract_references_extracts_every_native_link_form_in_document_order(
    engine: TreeSitterEngine, adapter: MarkdownAdapter
) -> None:
    """Against the realistic fixture: an inline link to another doc, an
    intra-doc section-anchor link, an inline image, a reference-style link
    usage, and the reference definition itself all surface -- the external
    `https://example.com/x` link does not."""
    with open(f"{FIXTURES_DIR}/index.md", "rb") as f:
        source = f.read()

    references = adapter.extract_references(parse(engine, source), source, "index.md")

    assert [(imp.module, imp.line, imp.is_relative) for imp in references] == [
        ("./about.md", 3, True),
        ("#background", 3, False),
        ("./assets/logo.png", 5, True),
        ("./about.md", 7, True),
        ("./about.md", 13, True),
    ]
    assert all(imp.file_path == "index.md" for imp in references)


@pytest.mark.parametrize(
    ("case", "snippet"),
    [
        ("https_scheme", b"[About](https://example.com/about.md)\n"),
        ("http_scheme", b"[About](http://example.com/about.md)\n"),
        ("protocol_relative", b"[About](//example.com/about.md)\n"),
        ("data_uri", b"![img](data:image/png;base64,AAAA)\n"),
        ("mailto_scheme", b"[Mail](mailto:hello@example.com)\n"),
    ],
)
def test_extract_references_ignores_external_and_non_file_targets(
    engine: TreeSitterEngine, adapter: MarkdownAdapter, case: str, snippet: bytes
) -> None:
    references = adapter.extract_references(parse(engine, snippet), snippet, "page.md")

    assert references == [], f"{case} produced a reference for a non-local target"


def test_parse_imports_delegates_to_extract_references(
    engine: TreeSitterEngine, adapter: MarkdownAdapter
) -> None:
    with open(f"{FIXTURES_DIR}/about.md", "rb") as f:
        source = f.read()
    tree = parse(engine, source)

    assert adapter.parse_imports(tree, source, "about.md") == adapter.extract_references(
        tree, source, "about.md"
    )


# ---------------------------------------------------------------------------
# resolve_import: relative file links, section anchors, unresolvable targets
# ---------------------------------------------------------------------------


def test_resolve_import_resolves_relative_link_against_containing_file_directory(
    adapter: MarkdownAdapter,
) -> None:
    imp = ImportStatement(module="./index.md", file_path="about.md", line=3, is_relative=True)

    resolved = adapter.resolve_import(imp, {"index.md": "index.md"})

    assert resolved == "index.md"


def test_resolve_import_resolves_section_anchor_to_its_own_containing_file(
    adapter: MarkdownAdapter,
) -> None:
    imp = ImportStatement(module="#background", file_path="index.md", line=3, is_relative=False)

    resolved = adapter.resolve_import(imp, {"index.md": "index.md"})

    assert resolved == "index.md"


def test_resolve_import_returns_none_for_unresolvable_local_reference(
    adapter: MarkdownAdapter,
) -> None:
    imp = ImportStatement(module="./missing.md", file_path="index.md", line=3, is_relative=True)

    assert adapter.resolve_import(imp, {"about.md": "about.md"}) is None


# ---------------------------------------------------------------------------
# archex.api.file_outline: end-to-end outline acceptance
# ---------------------------------------------------------------------------


def test_file_outline_returns_markdown_outline_and_resolved_references_end_to_end(
    tmp_path: Path,
) -> None:
    """Acceptance for Markdown: `archex.api.file_outline` surfaces the
    section outline plus every native link/image/section-anchor reference,
    resolved against the fixture's own file tree, without ever claiming a
    function/class/method/interface symbol for prose."""
    repo = _init_fixture_repo(tmp_path, "markdown_structured")
    source = RepoSource(local_path=str(repo))

    result = file_outline(
        source, file_path="index.md", config=Config(languages=["markdown"], cache=False)
    )

    assert result.language == "markdown"
    assert result.symbols == []
    assert [(item.start_line, item.end_line) for item in result.outline_ranges] == [(1, 13)]

    resolved_by_line = {ref.line: ref.resolved_path for ref in result.references}
    assert resolved_by_line == {
        3: "index.md",  # two references share line 3: about.md, then #background
        5: "assets/logo.png",
        7: "about.md",
        13: "about.md",
    }

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


# ---------------------------------------------------------------------------
# Cross-stack verification: JSON and TOML are unaffected
# ---------------------------------------------------------------------------


def test_json_and_toml_remain_chunk_only() -> None:
    """JSON and TOML have no generic cross-file reference syntax and stay
    `CHUNK_ONLY` permanently. This pins that neither was swept up by the
    XML/YAML/Markdown/CSS STRUCTURED promotion in this stack."""
    from archex.languages import CHUNK_ONLY_LANGUAGE_IDS

    assert get_language_tier("json") == LanguageTier.CHUNK_ONLY
    assert get_language_tier("toml") == LanguageTier.CHUNK_ONLY
    assert {"json", "toml"} <= CHUNK_ONLY_LANGUAGE_IDS
