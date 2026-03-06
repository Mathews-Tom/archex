"""Tests for the LSAP integration wrapper."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from archex.exceptions import LSAPError
from archex.integrations.lsap_models import (
    DefinitionLocation,
    HoverInfo,
    ReferenceLocation,
)
from archex.models import (
    DetectedPattern,
    ParsedFile,
    PatternCategory,
    PatternEvidence,
    Symbol,
    SymbolKind,
    SymbolSource,
    Visibility,
)


@pytest.fixture(autouse=True)
def _enable_lsap(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Patch _lsap_available=True so tests run without lsp-client installed."""
    import archex.integrations.lsap as lsap_module

    monkeypatch.setattr(lsap_module, "_lsap_available", True)


def _make_symbol(
    name: str = "get_user", sid: str = "src/repo.py::get_user#function"
) -> SymbolSource:
    return SymbolSource(
        symbol_id=sid,
        name=name,
        kind=SymbolKind.METHOD,
        file_path="src/repo.py",
        start_line=10,
        end_line=20,
        signature=f"def {name}(self, user_id: int) -> User",
        visibility=Visibility.PUBLIC,
        source=f"def {name}(self, user_id): ...",
        imports_context="from models import User",
        token_count=30,
    )


def _mock_client() -> AsyncMock:
    client = AsyncMock()
    client.request_hover = AsyncMock(
        return_value={
            "contents": "def get_user(self, user_id: int) -> User\nFetch a user by ID.",
        }
    )
    client.request_references = AsyncMock(
        return_value=[
            {
                "uri": "src/service.py",
                "range": {"start": {"line": 5, "character": 8}},
                "context": "repo.get_user(42)",
            },
            {
                "uri": "src/handler.py",
                "range": {"start": {"line": 12, "character": 4}},
                "context": "self.repo.get_user(uid)",
            },
        ]
    )
    client.request_definition = AsyncMock(
        return_value=[
            {
                "uri": "src/repo.py",
                "range": {"start": {"line": 10, "character": 4}},
                "context": "def get_user(self, user_id: int) -> User:",
            },
        ]
    )
    return client


class TestImportGuard:
    def test_raises_when_lsp_client_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import archex.integrations.lsap as lsap_module

        monkeypatch.setattr(lsap_module, "_lsap_available", False)
        with pytest.raises(LSAPError, match="lsp-client"):
            lsap_module.LSAPEnrichedLookup(lsp_client=object())  # type: ignore[arg-type]


class TestHoverEnrichment:
    def test_hover_populates_info(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        lookup = LSAPEnrichedLookup(lsp_client=client)
        hover = asyncio.get_event_loop().run_until_complete(lookup.get_hover("src/repo.py", 10))
        assert isinstance(hover, HoverInfo)
        assert "get_user" in hover.type_signature
        assert hover.documentation == "Fetch a user by ID."
        assert hover.raw_content != ""

    def test_hover_returns_empty_on_none(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        client.request_hover = AsyncMock(return_value=None)
        lookup = LSAPEnrichedLookup(lsp_client=client)
        hover = asyncio.get_event_loop().run_until_complete(lookup.get_hover("src/repo.py", 10))
        assert hover.type_signature == ""
        assert hover.raw_content == ""


class TestReferences:
    def test_references_returns_locations(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        lookup = LSAPEnrichedLookup(lsp_client=client)
        refs = asyncio.get_event_loop().run_until_complete(lookup.get_references("src/repo.py", 10))
        assert len(refs) == 2
        assert all(isinstance(r, ReferenceLocation) for r in refs)
        assert refs[0].file_path == "src/service.py"
        assert refs[0].line == 5
        assert refs[1].file_path == "src/handler.py"

    def test_references_empty_on_none(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        client.request_references = AsyncMock(return_value=None)
        lookup = LSAPEnrichedLookup(lsp_client=client)
        refs = asyncio.get_event_loop().run_until_complete(lookup.get_references("src/repo.py", 10))
        assert refs == []


class TestDefinition:
    def test_definition_returns_location(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        lookup = LSAPEnrichedLookup(lsp_client=client)
        defn = asyncio.get_event_loop().run_until_complete(lookup.get_definition("src/repo.py", 10))
        assert isinstance(defn, DefinitionLocation)
        assert defn.file_path == "src/repo.py"
        assert defn.line == 10

    def test_definition_returns_none_on_empty(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        client.request_definition = AsyncMock(return_value=None)
        lookup = LSAPEnrichedLookup(lsp_client=client)
        defn = asyncio.get_event_loop().run_until_complete(lookup.get_definition("src/repo.py", 10))
        assert defn is None


class TestEnrichSymbol:
    def test_enrich_populates_all_fields(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        lookup = LSAPEnrichedLookup(lsp_client=client)
        symbol = _make_symbol()
        enriched = asyncio.get_event_loop().run_until_complete(lookup.enrich_symbol(symbol))
        assert enriched.lsap_enrichment is not None
        enr = enriched.lsap_enrichment
        assert enr.hover is not None
        assert enr.hover.type_signature != ""
        assert len(enr.references) == 2
        assert enr.reference_count == 2
        assert enr.definition is not None
        # Original is unmodified.
        assert symbol.lsap_enrichment is None

    def test_partial_enrichment_on_hover_failure(self) -> None:
        """Hover fails but references and definition succeed."""
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        client.request_hover = AsyncMock(side_effect=RuntimeError("server error"))
        lookup = LSAPEnrichedLookup(lsp_client=client)
        symbol = _make_symbol()
        enriched = asyncio.get_event_loop().run_until_complete(lookup.enrich_symbol(symbol))
        assert enriched.lsap_enrichment is not None
        enr = enriched.lsap_enrichment
        assert enr.hover is None  # failed
        assert len(enr.references) == 2  # succeeded
        assert enr.definition is not None  # succeeded


class TestBatchConcurrency:
    def test_batch_enriches_all_symbols(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup

        client = _mock_client()
        lookup = LSAPEnrichedLookup(lsp_client=client)
        symbols = [
            _make_symbol(name=f"method_{i}", sid=f"src/repo.py::method_{i}#method")
            for i in range(10)
        ]
        enriched = asyncio.get_event_loop().run_until_complete(
            lookup.enrich_symbols_batch(symbols, concurrency=3)
        )
        assert len(enriched) == 10
        assert all(s.lsap_enrichment is not None for s in enriched)
        # Verify hover was called for each symbol.
        assert client.request_hover.call_count == 10


class TestPatternVerifier:
    def _make_repository_pattern(self) -> DetectedPattern:
        return DetectedPattern(
            name="Repository",
            display_name="Repository / DAO",
            description="Data access layer with CRUD operations",
            category=PatternCategory.STRUCTURAL,
            confidence=0.60,
            evidence=[
                PatternEvidence(
                    file_path="src/repo.py",
                    start_line=5,
                    end_line=30,
                    symbol="UserRepository",
                    explanation="CRUD methods: get_user, save_user",
                ),
            ],
        )

    def _make_parsed_file(self) -> ParsedFile:
        return ParsedFile(
            path="src/repo.py",
            language="python",
            symbols=[
                Symbol(
                    name="get_user",
                    qualified_name="UserRepository.get_user",
                    kind=SymbolKind.METHOD,
                    file_path="src/repo.py",
                    start_line=10,
                    end_line=20,
                    visibility=Visibility.PUBLIC,
                ),
            ],
            imports=[],
            lines=50,
            tokens=300,
        )

    def test_boosts_confidence_with_datastore_indicators(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup, verify_repository_pattern

        client = _mock_client()
        # Hover returns content containing "session" — a datastore indicator.
        client.request_hover = AsyncMock(
            return_value={
                "contents": "def get_user(self, session: AsyncSession) -> User",
            }
        )
        lookup = LSAPEnrichedLookup(lsp_client=client)
        pattern = self._make_repository_pattern()
        parsed_files = [self._make_parsed_file()]

        adjusted = asyncio.get_event_loop().run_until_complete(
            verify_repository_pattern(lookup, pattern, parsed_files)
        )
        assert adjusted is not None
        assert adjusted > pattern.confidence  # boosted

    def test_reduces_confidence_without_indicators(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup, verify_repository_pattern

        client = _mock_client()
        # Hover returns content with no datastore indicators.
        client.request_hover = AsyncMock(
            return_value={
                "contents": "def get_user(self, user_id: int) -> UserDTO",
            }
        )
        lookup = LSAPEnrichedLookup(lsp_client=client)
        pattern = self._make_repository_pattern()
        parsed_files = [self._make_parsed_file()]

        adjusted = asyncio.get_event_loop().run_until_complete(
            verify_repository_pattern(lookup, pattern, parsed_files)
        )
        assert adjusted is not None
        assert adjusted < pattern.confidence  # reduced

    def test_returns_none_for_non_repository_pattern(self) -> None:
        from archex.integrations.lsap import LSAPEnrichedLookup, verify_repository_pattern

        client = _mock_client()
        lookup = LSAPEnrichedLookup(lsp_client=client)
        pattern = DetectedPattern(
            name="Middleware",
            display_name="Middleware / Chain-of-Responsibility",
            description="Request processing pipeline",
            category=PatternCategory.BEHAVIORAL,
            confidence=0.85,
            evidence=[],
        )
        result = asyncio.get_event_loop().run_until_complete(
            verify_repository_pattern(lookup, pattern, [])
        )
        assert result is None
