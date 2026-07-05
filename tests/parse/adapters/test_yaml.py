"""Tests for the YAML STRUCTURED-tier adapter.

`YamlAdapter` (src/archex/parse/adapters/yaml.py) builds on the shared
`StructuredAdapter` base to produce a document outline for `.yaml`/`.yml`
files plus YAML's native anchor (`&name`) / alias (`*name`) cross-reference
mechanism, without claiming programming symbols. `yaml` is registered
at `LanguageTier.STRUCTURED` for real in `archex.languages`, so every
test below builds `YamlAdapter()` straight off the production registry
entry -- no monkeypatched stand-in is needed. Anchors/aliases are always
intra-document (YAML has no native cross-file import syntax), so a
resolved reference always points back at the alias's own containing file.
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
from archex.parse.adapters.yaml import YamlAdapter
from archex.parse.engine import TreeSitterEngine
from tests.conftest import _init_fixture_repo  # pyright: ignore[reportPrivateUsage]

FIXTURES_DIR = "tests/fixtures/yaml_structured"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> YamlAdapter:
    return YamlAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "yaml")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: YamlAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


def test_yaml_registered_at_structured_tier() -> None:
    """Pins that `yaml` is registered at STRUCTURED tier directly in the
    registry, independent of any single adapter call, so a tier
    regression fails here even if some other test's mocks would
    otherwise mask it."""
    assert get_language_tier("yaml") == LanguageTier.STRUCTURED


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: YamlAdapter) -> None:
    assert adapter.language_id == "yaml"


def test_file_extensions(adapter: YamlAdapter) -> None:
    assert adapter.file_extensions == [".yaml", ".yml"]


def test_tree_sitter_name(adapter: YamlAdapter) -> None:
    assert adapter.tree_sitter_name == "yaml"


# ---------------------------------------------------------------------------
# extract_chunk_ranges: document outline
# ---------------------------------------------------------------------------


def test_extract_chunk_ranges_covers_whole_document(
    engine: TreeSitterEngine, adapter: YamlAdapter
) -> None:
    with open(f"{FIXTURES_DIR}/config.yaml", "rb") as f:
        source = f.read()

    ranges = adapter.extract_chunk_ranges(parse(engine, source), source, "config.yaml")

    assert ranges == [ChunkRange(start_line=1, end_line=13)]


# ---------------------------------------------------------------------------
# extract_symbols: never claims programming symbols
# ---------------------------------------------------------------------------


def test_extract_symbols_is_always_empty(engine: TreeSitterEngine, adapter: YamlAdapter) -> None:
    with open(f"{FIXTURES_DIR}/config.yaml", "rb") as f:
        source = f.read()

    assert adapter.extract_symbols(parse(engine, source), source, "config.yaml") == []


def test_extract_symbols_ignores_keys_named_like_programming_constructs(
    engine: TreeSitterEngine, adapter: YamlAdapter
) -> None:
    """Mapping keys literally named `function`/`class`/`interface` are
    still just YAML scalar keys -- `extract_symbols` is `@final` on the
    `StructuredAdapter` base and can never be overridden to claim them
    as programming symbols, regardless of how symbol-shaped the key
    names look."""
    source = b"function: computeTotal\nclass: Cart\ninterface: Payable\n"

    assert adapter.extract_symbols(parse(engine, source), source, "adversarial.yaml") == []


def test_extract_references_ignores_plain_scalars_that_look_like_file_paths(
    engine: TreeSitterEngine, adapter: YamlAdapter
) -> None:
    """A plain scalar value that merely *looks* like a relative file path
    (`path: ./other.yaml`) is not YAML's native cross-reference mechanism
    -- only a genuine `&anchor`/`*alias` pair is. Treating arbitrary
    path-shaped strings as references would invent a cross-file import
    syntax YAML does not have."""
    source = b"path: ./other.yaml\nfunction: computeTotal\n"

    assert adapter.extract_references(parse(engine, source), source, "adversarial.yaml") == []


# ---------------------------------------------------------------------------
# extract_references: anchor/alias native cross-reference extraction
# ---------------------------------------------------------------------------


def test_extract_references_finds_every_alias_with_a_matching_anchor(
    engine: TreeSitterEngine, adapter: YamlAdapter
) -> None:
    """The fixture merge-keys `<<: *defaults` into both `production` and
    `staging`, each aliasing the `&defaults` anchor -- both usages must be
    extracted, each pinned to its own alias line, in document order."""
    with open(f"{FIXTURES_DIR}/config.yaml", "rb") as f:
        source = f.read()

    references = adapter.extract_references(parse(engine, source), source, "config.yaml")

    assert [(imp.module, imp.line, imp.is_relative) for imp in references] == [
        ("defaults", 6, False),
        ("defaults", 10, False),
    ]
    assert all(imp.file_path == "config.yaml" for imp in references)


def test_extract_references_drops_an_alias_with_no_matching_anchor(
    engine: TreeSitterEngine, adapter: YamlAdapter
) -> None:
    """`fallback_host: *undefined_anchor` is a syntactically valid alias
    node, but no `&undefined_anchor` anchor exists anywhere in the
    document. Reporting it as a 'correctly extracted' reference would be
    unverifiable, so it must not appear in the extracted list."""
    with open(f"{FIXTURES_DIR}/config.yaml", "rb") as f:
        source = f.read()
    text = source.decode("utf-8")
    assert "*undefined_anchor" in text

    references = adapter.extract_references(parse(engine, source), source, "config.yaml")

    assert "undefined_anchor" not in [imp.module for imp in references]


def test_parse_imports_delegates_to_extract_references(
    engine: TreeSitterEngine, adapter: YamlAdapter
) -> None:
    with open(f"{FIXTURES_DIR}/config.yaml", "rb") as f:
        source = f.read()
    tree = parse(engine, source)

    assert adapter.parse_imports(tree, source, "config.yaml") == adapter.extract_references(
        tree, source, "config.yaml"
    )


# ---------------------------------------------------------------------------
# resolve_import: anchors/aliases always resolve to their own file
# ---------------------------------------------------------------------------


def test_resolve_import_resolves_to_the_containing_file(adapter: YamlAdapter) -> None:
    imp = ImportStatement(module="defaults", file_path="config.yaml", line=6, is_relative=False)

    resolved = adapter.resolve_import(imp, {"config.yaml": "config.yaml"})

    assert resolved == "config.yaml"


def test_resolve_import_returns_none_when_containing_file_is_absent_from_the_file_map(
    adapter: YamlAdapter,
) -> None:
    imp = ImportStatement(module="defaults", file_path="config.yaml", line=6, is_relative=False)

    assert adapter.resolve_import(imp, {"other.yaml": "other.yaml"}) is None


# ---------------------------------------------------------------------------
# archex.api.file_outline: end-to-end outline acceptance
# ---------------------------------------------------------------------------


def test_file_outline_returns_yaml_outline_and_anchor_alias_references_end_to_end(
    tmp_path: Path,
) -> None:
    """Acceptance for YAML: `archex.api.file_outline` surfaces the
    document outline plus the two valid `&defaults`/`*defaults`
    references, resolved to the fixture's own file, without ever claiming
    a function/class/method/interface symbol for configuration data."""
    repo = _init_fixture_repo(tmp_path, "yaml_structured")
    source = RepoSource(local_path=str(repo))

    result = file_outline(
        source, file_path="config.yaml", config=Config(languages=["yaml"], cache=False)
    )

    assert result.language == "yaml"
    assert result.symbols == []
    assert [(item.start_line, item.end_line) for item in result.outline_ranges] == [(1, 13)]

    resolved_by_line = {ref.line: ref.resolved_path for ref in result.references}
    assert resolved_by_line == {6: "config.yaml", 10: "config.yaml"}
    assert all(ref.module == "defaults" for ref in result.references)

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
# Cross-stack verification: TOML is unaffected
# ---------------------------------------------------------------------------


def test_toml_remains_chunk_only_and_gained_no_dedicated_adapter() -> None:
    """TOML has no generic cross-file reference syntax and stays
    `CHUNK_ONLY` permanently. This pins that TOML was not swept up by the
    XML/YAML/Markdown/CSS STRUCTURED promotion in this stack, and that no
    dedicated `archex/parse/adapters/toml.py` module was added -- `toml`
    is still served by the generic chunk-only factory, the same as every
    other untouched `CHUNK_ONLY` language."""
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
