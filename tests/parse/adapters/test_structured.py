"""Tests for the shared STRUCTURED-tier base adapter.

`StructuredAdapter` (src/archex/parse/adapters/structured.py) is the M11 base
for outline-plus-native-cross-reference languages: it must never claim
programming symbols, must produce chunk-node-driven outline ranges the same
way `ChunkOnlyAdapter` does, and must route `parse_imports` through an
overridable `extract_references` hook so concrete languages (HTML, XML, ...)
can extract native cross-file references without re-implementing the
outline/symbol invariants.
"""

from __future__ import annotations

import re

import pytest

from archex import languages
from archex.languages import LanguageSupport
from archex.models import ChunkRange, ImportStatement, LanguageTier
from archex.parse.adapters.structured import StructuredAdapter, make_structured_adapter

_REF_PATTERN = re.compile(r'ref="([^"]+)"')

# A tiny two-"section" fixture with a filler node of a different named type
# (must be excluded from the outline) and a native reference embedded in the
# second section (must be extracted by the hook).
FIXTURE_SOURCE = (
    b"<section>Intro</section>\n"
    b"<note>see also</note>\n"
    b'<section ref="./shared.struct">Details</section>\n'
)


class _FakeNode:
    """Minimal duck-typed stand-in for a tree-sitter node.

    Only the attributes `extract_chunk_ranges`'s named-node walk relies on:
    `.children`, `.is_named`, `.type`, `.start_point`, `.end_point`.
    """

    def __init__(
        self,
        node_type: str,
        start_line: int,
        end_line: int,
        *,
        is_named: bool = True,
        children: list[_FakeNode] | None = None,
    ) -> None:
        self.type = node_type
        self.is_named = is_named
        self.start_point = (start_line - 1, 0)
        self.end_point = (end_line - 1, 10)
        self.children: list[_FakeNode] = children or []


class _FakeTree:
    def __init__(self, root: _FakeNode) -> None:
        self.root_node = root


def _fixture_tree() -> _FakeTree:
    root = _FakeNode(
        "document",
        1,
        3,
        children=[
            _FakeNode("section", 1, 1),
            _FakeNode("note", 2, 2),
            _FakeNode("section", 3, 3),
        ],
    )
    return _FakeTree(root)


class _StubStructuredAdapter(StructuredAdapter):
    """Stands in for a future concrete adapter (e.g. M12's html.py)."""

    _language_id = "structured_stub"

    def extract_references(
        self, tree: object, source: bytes, file_path: str
    ) -> list[ImportStatement]:
        references: list[ImportStatement] = []
        for line_no, line in enumerate(source.decode("utf-8").splitlines(), start=1):
            match = _REF_PATTERN.search(line)
            if match is None:
                continue
            target = match.group(1)
            references.append(
                ImportStatement(
                    module=target,
                    file_path=file_path,
                    line=line_no,
                    is_relative=target.startswith("./") or target.startswith("../"),
                )
            )
        return references


@pytest.fixture
def _structured_stub_registered(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = LanguageSupport(
        language_id="structured_stub",
        display_name="Structured Stub",
        extensions=(".structstub",),
        tier=LanguageTier.STRUCTURED,
        pack_name="structured_stub",
        chunk_node_types=frozenset({"section"}),
    )
    monkeypatch.setitem(languages.LANGUAGE_SUPPORT, "structured_stub", stub)


def test_extract_chunk_ranges_produces_outline_only_for_declared_node_types(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()

    ranges = adapter.extract_chunk_ranges(_fixture_tree(), FIXTURE_SOURCE, "pkg/main.struct")

    assert ranges == [
        ChunkRange(start_line=1, end_line=1),
        ChunkRange(start_line=3, end_line=3),
    ]


def test_extract_symbols_never_claims_programming_symbols(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()

    assert adapter.extract_symbols(_fixture_tree(), FIXTURE_SOURCE, "pkg/main.struct") == []


def test_parse_imports_delegates_to_extract_references_hook(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()
    tree = _fixture_tree()

    imports = adapter.parse_imports(tree, FIXTURE_SOURCE, "pkg/main.struct")

    assert imports == adapter.extract_references(tree, FIXTURE_SOURCE, "pkg/main.struct")
    assert len(imports) == 1
    assert imports[0].module == "./shared.struct"
    assert imports[0].line == 3
    assert imports[0].is_relative is True


def test_resolve_import_resolves_relative_local_reference(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()
    imp = ImportStatement(
        module="./shared.struct",
        file_path="pkg/main.struct",
        line=3,
        is_relative=True,
    )

    resolved = adapter.resolve_import(imp, {"pkg/shared.struct": "pkg/shared.struct"})

    assert resolved == "pkg/shared.struct"


def test_resolve_import_matches_production_file_map_values(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()
    imp = ImportStatement(
        module="./shared.struct",
        file_path="pkg/main.struct",
        line=3,
        is_relative=True,
    )

    resolved = adapter.resolve_import(imp, {"pkg.shared": "pkg/shared.struct"})

    assert resolved == "pkg/shared.struct"


def test_resolve_import_infers_relative_reference_syntax(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()
    imp = ImportStatement(
        module="./shared.struct",
        file_path="pkg/main.struct",
        line=3,
        is_relative=False,
    )

    resolved = adapter.resolve_import(imp, {"pkg.shared": "pkg/shared.struct"})

    assert resolved == "pkg/shared.struct"


def test_resolve_import_accepts_direct_file_map_match(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()
    imp = ImportStatement(
        module="pkg/shared.struct",
        file_path="pkg/main.struct",
        line=4,
        is_relative=False,
    )

    resolved = adapter.resolve_import(imp, {"pkg/shared.struct": "pkg/shared.struct"})

    assert resolved == "pkg/shared.struct"


def test_make_structured_adapter_sets_language_id_and_name(
    _structured_stub_registered: None,
) -> None:
    adapter_cls = make_structured_adapter("structured_stub")

    assert adapter_cls.__name__ == "StructuredStubStructuredAdapter"
    assert adapter_cls().language_id == "structured_stub"
    assert adapter_cls().tree_sitter_name == "structured_stub"
    assert adapter_cls().file_extensions == [".structstub"]


def test_structured_adapter_rejects_wrong_tier_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_support = LanguageSupport(
        language_id="chunk_stub",
        display_name="Chunk Stub",
        extensions=(".chunkstub",),
        tier=LanguageTier.CHUNK_ONLY,
        pack_name="chunk_stub",
        chunk_node_types=frozenset({"section"}),
    )
    monkeypatch.setitem(languages.LANGUAGE_SUPPORT, "chunk_stub", chunk_support)

    adapter_cls = make_structured_adapter("chunk_stub")
    with pytest.raises(ValueError, match="registered as"):
        adapter_cls()


def test_structured_adapter_rejects_unregistered_language() -> None:
    adapter_cls = make_structured_adapter("missing_structured_stub")

    with pytest.raises(ValueError, match="not registered"):
        adapter_cls()


def test_resolve_import_returns_none_for_unresolvable_reference(
    _structured_stub_registered: None,
) -> None:
    adapter = _StubStructuredAdapter()
    imp = ImportStatement(
        module="https://example.com/shared.struct",
        file_path="pkg/main.struct",
        line=5,
        is_relative=False,
    )

    assert adapter.resolve_import(imp, {"pkg/shared.struct": "pkg/shared.struct"}) is None
