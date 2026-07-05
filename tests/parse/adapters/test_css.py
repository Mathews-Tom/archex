"""Tests for the CSS STRUCTURED-tier adapter.

`CssAdapter` (src/archex/parse/adapters/css.py) builds on the shared
`StructuredAdapter` base to produce a rule outline for `.css` files plus
CSS's native `@import`/`url()` cross-file reference mechanisms, without
claiming programming symbols. `css` is registered at
`LanguageTier.STRUCTURED` for real in `archex.languages`, so every test
below builds `CssAdapter()` straight off the production registry entry --
no monkeypatched stand-in is needed. `@import` (both the bare-string and
`url(...)`-wrapped forms) and every other `url()` call (in property
values such as `background`/`background-image`) are covered; non-`url`
function calls (`rgba()`, `calc()`, ...), external targets, and
fragment-only targets (`url(#gradient)`, an SVG-internal reference, not a
file reference) never surface as extracted references.
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
from archex.parse.adapters.css import CssAdapter
from archex.parse.engine import TreeSitterEngine
from tests.conftest import _init_fixture_repo  # pyright: ignore[reportPrivateUsage]

FIXTURES_DIR = "tests/fixtures/css_structured"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> CssAdapter:
    return CssAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "css")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: CssAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


def test_css_registered_at_structured_tier() -> None:
    """Pins that `css` is registered at STRUCTURED tier directly in the
    registry, independent of any single adapter call, so a tier
    regression fails here even if some other test's mocks would
    otherwise mask it."""
    assert get_language_tier("css") == LanguageTier.STRUCTURED


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: CssAdapter) -> None:
    assert adapter.language_id == "css"


def test_file_extensions(adapter: CssAdapter) -> None:
    assert adapter.file_extensions == [".css"]


def test_tree_sitter_name(adapter: CssAdapter) -> None:
    assert adapter.tree_sitter_name == "css"


# ---------------------------------------------------------------------------
# extract_chunk_ranges: rule/media/import outline (unchanged by the tier flip)
# ---------------------------------------------------------------------------


def test_extract_chunk_ranges_on_realistic_fixture(
    engine: TreeSitterEngine, adapter: CssAdapter
) -> None:
    with open(f"{FIXTURES_DIR}/main.css", "rb") as f:
        source = f.read()

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "main.css")

    assert ranges == [
        ChunkRange(start_line=1, end_line=1),
        ChunkRange(start_line=2, end_line=2),
        ChunkRange(start_line=3, end_line=3),
        ChunkRange(start_line=5, end_line=7),
        ChunkRange(start_line=9, end_line=11),
        ChunkRange(start_line=13, end_line=17),
    ]


# ---------------------------------------------------------------------------
# extract_symbols: never claims programming symbols
# ---------------------------------------------------------------------------


def test_extract_symbols_is_always_empty(engine: TreeSitterEngine, adapter: CssAdapter) -> None:
    with open(f"{FIXTURES_DIR}/main.css", "rb") as f:
        source = f.read()

    assert adapter.extract_symbols(parse(engine, source), source, "main.css") == []


def test_extract_symbols_ignores_selectors_and_custom_properties_named_like_programming_constructs(
    engine: TreeSitterEngine, adapter: CssAdapter
) -> None:
    """A class selector, custom property, and id selector named
    `function-name`/`class-name`/`interface-id` are still just CSS
    selectors and properties -- `extract_symbols` is `@final` on the
    `StructuredAdapter` base and can never be overridden to claim them
    as programming symbols, regardless of how symbol-shaped the names
    look."""
    source = (
        b".function-name {\n"
        b"  --class-name: Cart;\n"
        b"  color: red;\n"
        b"}\n"
        b"#interface-id {\n"
        b"  width: 10px;\n"
        b"}\n"
    )

    assert adapter.extract_symbols(parse(engine, source), source, "adversarial.css") == []


def test_extract_references_ignores_plain_string_values_outside_url_or_import(
    engine: TreeSitterEngine, adapter: CssAdapter
) -> None:
    """A quoted string value that merely *looks* like a file path (e.g. a
    `content: "styles/other.css";` declaration) is not CSS's native
    cross-reference mechanism unless it is wrapped in `url(...)` or is an
    `@import` target -- treating any path-shaped string as a reference
    would invent semantics CSS's grammar does not assign to it."""
    source = b'.a::before { content: "styles/other.css"; }\n'

    assert adapter.extract_references(parse(engine, source), source, "adversarial.css") == []


# ---------------------------------------------------------------------------
# extract_references: @import and url() native reference extraction
# ---------------------------------------------------------------------------


def test_extract_references_covers_import_and_declaration_value_urls(
    engine: TreeSitterEngine, adapter: CssAdapter
) -> None:
    """The fixture's bare-string `@import`, `url()`-wrapped `@import`, and
    two declaration-value `url()` calls (quoted and unquoted) all surface;
    the external `@import url(https://...)` does not."""
    with open(f"{FIXTURES_DIR}/main.css", "rb") as f:
        source = f.read()

    references = adapter.extract_references(parse(engine, source), source, "main.css")

    assert [(imp.module, imp.line, imp.is_relative) for imp in references] == [
        ("./base.css", 1, True),
        ("./variables.css", 2, True),
        ("./assets/logo.png", 6, True),
        ("hero-bg.png", 10, True),
    ]
    assert all(imp.file_path == "main.css" for imp in references)


@pytest.mark.parametrize(
    ("case", "snippet"),
    [
        ("https_url_import", b"@import url(https://fonts.example.com/font.css);\n"),
        ("https_string_import", b'@import "https://fonts.example.com/font.css";\n'),
        ("data_uri", b".a { background: url(data:image/png;base64,AAAA); }\n"),
        ("fragment_only_url", b".a { fill: url(#gradient1); }\n"),
        ("non_url_function_call", b".a { color: rgba(0,0,0,0.5); width: calc(100% - 8px); }\n"),
        ("protocol_relative", b".a { background: url(//cdn.example.com/x.png); }\n"),
    ],
)
def test_extract_references_ignores_external_and_non_file_targets(
    engine: TreeSitterEngine, adapter: CssAdapter, case: str, snippet: bytes
) -> None:
    references = adapter.extract_references(parse(engine, snippet), snippet, "page.css")

    assert references == [], f"{case} produced a reference for a non-local target"


def test_parse_imports_delegates_to_extract_references(
    engine: TreeSitterEngine, adapter: CssAdapter
) -> None:
    with open(f"{FIXTURES_DIR}/main.css", "rb") as f:
        source = f.read()
    tree = parse(engine, source)

    assert adapter.parse_imports(tree, source, "main.css") == adapter.extract_references(
        tree, source, "main.css"
    )


# ---------------------------------------------------------------------------
# resolve_import: relative resolution and unresolvable targets
# ---------------------------------------------------------------------------


def test_resolve_import_resolves_relative_reference_against_containing_file_directory(
    adapter: CssAdapter,
) -> None:
    imp = ImportStatement(
        module="./base.css", file_path="styles/main.css", line=1, is_relative=True
    )

    resolved = adapter.resolve_import(imp, {"styles/base.css": "styles/base.css"})

    assert resolved == "styles/base.css"


def test_resolve_import_returns_none_for_unresolvable_local_reference(adapter: CssAdapter) -> None:
    imp = ImportStatement(module="missing.css", file_path="main.css", line=1, is_relative=True)

    assert adapter.resolve_import(imp, {"base.css": "base.css"}) is None


# ---------------------------------------------------------------------------
# archex.api.file_outline: end-to-end outline acceptance
# ---------------------------------------------------------------------------


def test_file_outline_returns_css_outline_and_resolved_references_end_to_end(
    tmp_path: Path,
) -> None:
    """Acceptance for CSS: `archex.api.file_outline` surfaces the rule
    outline plus every native `@import`/`url()` reference, resolved
    against the fixture's own file tree, without ever claiming a
    function/class/method/interface symbol for stylesheet rules."""
    repo = _init_fixture_repo(tmp_path, "css_structured")
    source = RepoSource(local_path=str(repo))

    result = file_outline(
        source, file_path="main.css", config=Config(languages=["css"], cache=False)
    )

    assert result.language == "css"
    assert result.symbols == []

    resolved_by_module = {ref.module: ref.resolved_path for ref in result.references}
    assert resolved_by_module == {
        "./base.css": "base.css",
        "./variables.css": "variables.css",
        "./assets/logo.png": "assets/logo.png",
        "hero-bg.png": "hero-bg.png",
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
