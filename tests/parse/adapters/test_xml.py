"""Tests for the generic XML STRUCTURED-tier adapter.

`XmlAdapter` (src/archex/parse/adapters/xml.py) builds on the shared
`StructuredAdapter` base to produce an element outline for `.xml` files
without claiming programming symbols or inventing a cross-reference
mechanism generic XML does not have. `xml` is registered at
`LanguageTier.STRUCTURED` for real in `archex.languages`, so every test
below builds `XmlAdapter()` straight off the production registry entry --
no monkeypatched stand-in is needed. Unlike HTML, generic XML has no native
cross-file reference syntax: an attribute that merely *looks* like a
reference (e.g. `ref="other.xml"`) is dialect-specific convention, not XML
grammar, so `extract_references` must stay empty even when such attributes
are present.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from archex.api import file_outline
from archex.languages import get_language_tier
from archex.models import ChunkRange, Config, LanguageTier, RepoSource, SymbolKind, SymbolOutline
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.xml import XmlAdapter
from archex.parse.engine import TreeSitterEngine
from tests.conftest import _init_fixture_repo  # pyright: ignore[reportPrivateUsage]

FIXTURES_DIR = "tests/fixtures/xml_structured"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> XmlAdapter:
    return XmlAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "xml")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: XmlAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


def test_xml_registered_at_structured_tier() -> None:
    """Pins that `xml` is registered at STRUCTURED tier directly in the
    registry, independent of any single adapter call, so a tier
    regression fails here even if some other test's mocks would
    otherwise mask it."""
    assert get_language_tier("xml") == LanguageTier.STRUCTURED


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: XmlAdapter) -> None:
    assert adapter.language_id == "xml"


def test_file_extensions(adapter: XmlAdapter) -> None:
    assert adapter.file_extensions == [".xml"]


def test_tree_sitter_name(adapter: XmlAdapter) -> None:
    assert adapter.tree_sitter_name == "xml"


# ---------------------------------------------------------------------------
# extract_chunk_ranges: element outline
# ---------------------------------------------------------------------------


def test_extract_chunk_ranges_collapses_nested_elements_into_the_root(
    engine: TreeSitterEngine, adapter: XmlAdapter
) -> None:
    """Nested `<product>`/`<name>`/`<price>` elements fold into the single
    enclosing `<catalog>` root -- the same non-overlapping-outermost-wins
    rule `ChunkOnlyAdapter` uses, unaffected by the STRUCTURED tier flip."""
    with open(f"{FIXTURES_DIR}/catalog.xml", "rb") as f:
        source = f.read()

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "catalog.xml")

    assert ranges == [ChunkRange(start_line=2, end_line=11)]


def test_extract_chunk_ranges_on_sibling_elements_under_one_root(
    engine: TreeSitterEngine, adapter: XmlAdapter
) -> None:
    with open(f"{FIXTURES_DIR}/library.xml", "rb") as f:
        source = f.read()

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "library.xml")

    assert ranges == [ChunkRange(start_line=2, end_line=9)]


# ---------------------------------------------------------------------------
# extract_symbols: never claims programming symbols
# ---------------------------------------------------------------------------


def test_extract_symbols_is_always_empty(engine: TreeSitterEngine, adapter: XmlAdapter) -> None:
    with open(f"{FIXTURES_DIR}/catalog.xml", "rb") as f:
        source = f.read()

    assert adapter.extract_symbols(parse(engine, source), source, "catalog.xml") == []


def test_extract_symbols_ignores_elements_named_like_programming_constructs(
    engine: TreeSitterEngine, adapter: XmlAdapter
) -> None:
    """Elements literally named `<function>`/`<class>`/`<interface>` are
    still just XML elements -- `extract_symbols` is `@final` on the
    `StructuredAdapter` base and can never be overridden to claim them as
    programming symbols, regardless of how symbol-shaped the tag names
    look."""
    source = (
        b'<function name="computeTotal">\n'
        b'  <class name="Cart"/>\n'
        b'  <interface implements="Payable"/>\n'
        b"</function>\n"
    )

    assert adapter.extract_symbols(parse(engine, source), source, "adversarial.xml") == []


def test_extract_references_ignores_elements_named_like_programming_constructs(
    engine: TreeSitterEngine, adapter: XmlAdapter
) -> None:
    """The same symbol-shaped tag names must not leak into references
    either -- generic XML's `extract_references` stays on the empty-list
    default no matter what the elements are named."""
    source = (
        b'<function name="computeTotal">\n'
        b'  <class name="Cart"/>\n'
        b'  <interface implements="Payable"/>\n'
        b"</function>\n"
    )

    assert adapter.extract_references(parse(engine, source), source, "adversarial.xml") == []


# ---------------------------------------------------------------------------
# extract_references: generic XML has no native cross-reference mechanism
# ---------------------------------------------------------------------------


def test_extract_references_never_invents_a_reference_from_a_ref_like_attribute(
    engine: TreeSitterEngine, adapter: XmlAdapter
) -> None:
    """`library.xml` contains a `ref="catalog.xml"` attribute that *looks*
    like a cross-file reference. Generic XML has no native syntax that
    makes an attribute a reference -- that is dialect-specific convention
    (out of scope until the XML dialect-plugin milestone) -- so this must
    not surface as an extracted reference."""
    with open(f"{FIXTURES_DIR}/library.xml", "rb") as f:
        source = f.read()
    text = source.decode("utf-8")
    assert 'ref="catalog.xml"' in text

    references = adapter.extract_references(parse(engine, source), source, "library.xml")

    assert references == []


def test_parse_imports_is_always_empty(engine: TreeSitterEngine, adapter: XmlAdapter) -> None:
    with open(f"{FIXTURES_DIR}/catalog.xml", "rb") as f:
        source = f.read()

    assert adapter.parse_imports(parse(engine, source), source, "catalog.xml") == []


def test_resolve_import_is_always_none(adapter: XmlAdapter) -> None:
    from archex.models import ImportStatement

    imp = ImportStatement(module="other.xml", file_path="library.xml", line=7, is_relative=False)

    assert adapter.resolve_import(imp, {"catalog.xml": "catalog.xml"}) is None


# ---------------------------------------------------------------------------
# archex.api.file_outline: end-to-end outline acceptance
# ---------------------------------------------------------------------------


def test_file_outline_returns_xml_element_outline_with_no_references_end_to_end(
    tmp_path: Path,
) -> None:
    """Acceptance for generic XML: `archex.api.file_outline` surfaces the
    element outline and zero programming symbols, and -- because generic
    XML's only native cross-reference mechanism is its outline itself --
    zero references, even though the fixture contains a `ref`-like
    attribute. Line 1 (the XML prolog, before the `<library>` root) is not
    claimed by any `element` node, so the chunker's generic gap-fill
    behavior surfaces it as its own leading range -- the same pattern
    HTML's DOCTYPE line exercises in `test_html.py`."""
    repo = _init_fixture_repo(tmp_path, "xml_structured")
    source = RepoSource(local_path=str(repo))

    result = file_outline(
        source, file_path="library.xml", config=Config(languages=["xml"], cache=False)
    )

    assert result.language == "xml"
    assert result.symbols == []
    assert [(item.start_line, item.end_line) for item in result.outline_ranges] == [(1, 1), (2, 9)]
    assert result.references == []

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


def test_json_remains_chunk_only_and_gained_no_dedicated_adapter() -> None:
    """JSON has no generic cross-file reference syntax and stays
    `CHUNK_ONLY` permanently. This pins that JSON was not swept up by the
    XML/YAML/Markdown/CSS STRUCTURED promotion in this stack, and that no
    dedicated `archex/parse/adapters/json.py` module was added -- `json`
    is still served by the generic chunk-only factory, the same as every
    other untouched `CHUNK_ONLY` language."""
    import importlib

    from archex.languages import CHUNK_ONLY_LANGUAGE_IDS

    assert get_language_tier("json") == LanguageTier.CHUNK_ONLY
    assert "json" in CHUNK_ONLY_LANGUAGE_IDS

    from archex.parse.adapters import default_adapter_registry

    json_adapter_cls = default_adapter_registry.get("json")
    assert json_adapter_cls is not None
    assert json_adapter_cls.__name__ == "JsonChunkOnlyAdapter"
    assert json_adapter_cls.__module__ == "archex.parse.adapters.chunk_only"

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("archex.parse.adapters.json")


def test_toml_remains_chunk_only_and_gained_no_dedicated_adapter() -> None:
    """Same guarantee as JSON above, for TOML -- both are explicitly out
    of scope for this stack per the report's exclusion."""
    import importlib

    from archex.languages import CHUNK_ONLY_LANGUAGE_IDS

    assert get_language_tier("toml") == LanguageTier.CHUNK_ONLY
    assert "toml" in CHUNK_ONLY_LANGUAGE_IDS

    from archex.parse.adapters import default_adapter_registry

    toml_adapter_cls = default_adapter_registry.get("toml")
    assert toml_adapter_cls is not None
    assert toml_adapter_cls.__name__ == "TomlChunkOnlyAdapter"
    assert toml_adapter_cls.__module__ == "archex.parse.adapters.chunk_only"

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("archex.parse.adapters.toml")
