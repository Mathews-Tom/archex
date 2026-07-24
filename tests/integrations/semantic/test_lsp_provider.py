"""Tests for the LSP evidence provider."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from archex.integrations.semantic.lsp_provider import LspEvidenceProvider
from archex.integrations.semantic.models import (
    ProviderAvailability,
    SemanticEdgeKind,
    SemanticProviderName,
)
from archex.models import ParsedFile, Symbol, SymbolKind


@pytest.fixture(autouse=True)
def _enable_lsap(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Patch lsap_available()=True so tests run without lsp-client installed."""
    import archex.integrations.lsap as lsap_module
    import archex.integrations.semantic.lsp_provider as lsp_provider_module

    monkeypatch.setattr(lsap_module, "_lsap_available", True)
    monkeypatch.setattr(lsp_provider_module, "lsap_available", lambda: True)


def _parsed_files() -> list[ParsedFile]:
    return [
        ParsedFile(
            path="a.py",
            language="python",
            symbols=[
                Symbol(
                    name="foo",
                    qualified_name="a.foo",
                    kind=SymbolKind.FUNCTION,
                    file_path="a.py",
                    start_line=3,
                    end_line=5,
                )
            ],
        )
    ]


def _mock_client(
    *,
    definition: dict[str, Any] | list[Any] | None = None,
    references: list[Any] | None = None,
    implementation: dict[str, Any] | list[Any] | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.request_definition = AsyncMock(return_value=definition)
    client.request_references = AsyncMock(return_value=references)
    client.request_implementation = AsyncMock(return_value=implementation)
    return client


class TestProbe:
    def test_unavailable_when_lsap_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import archex.integrations.semantic.lsp_provider as lsp_provider_module

        monkeypatch.setattr(lsp_provider_module, "lsap_available", lambda: False)
        provider = LspEvidenceProvider(client=_mock_client())
        receipt = provider.probe(tmp_path)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "lsp-client not installed" in receipt.reason

    def test_unavailable_when_no_client_configured(self, tmp_path: Path) -> None:
        provider = LspEvidenceProvider(client=None)
        receipt = provider.probe(tmp_path)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "no LSP client configured" in receipt.reason

    def test_available_when_client_configured(self, tmp_path: Path) -> None:
        provider = LspEvidenceProvider(client=_mock_client())
        receipt = provider.probe(tmp_path)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.provider == SemanticProviderName.LSP


class TestCollect:
    def test_no_client_returns_no_edges_and_reason(self, tmp_path: Path) -> None:
        provider = LspEvidenceProvider(client=None)
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert receipt.reason

    def test_cross_file_definition_reference_implementation_edges(self, tmp_path: Path) -> None:
        client = _mock_client(
            definition={"uri": "b.py", "range": {"start": {"line": 10, "character": 2}}},
            references=[{"uri": "c.py", "range": {"start": {"line": 1, "character": 0}}}],
            implementation={"uri": "d.py", "range": {"start": {"line": 20, "character": 4}}},
        )
        provider = LspEvidenceProvider(client=client)
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)

        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.provider == SemanticProviderName.LSP
        assert receipt.evidence_count == 3

        by_kind = {e.kind: e for e in evidence}
        assert by_kind[SemanticEdgeKind.DEFINITION].target.file_path == "b.py"
        assert by_kind[SemanticEdgeKind.DEFINITION].source.file_path == "a.py"
        assert by_kind[SemanticEdgeKind.REFERENCE].target.file_path == "c.py"
        assert by_kind[SemanticEdgeKind.IMPLEMENTATION].target.file_path == "d.py"
        for e in evidence:
            assert e.provider == SemanticProviderName.LSP
            assert 0.0 < e.confidence <= 1.0

    def test_same_file_results_are_dropped(self, tmp_path: Path) -> None:
        client = _mock_client(
            definition={"uri": "a.py", "range": {"start": {"line": 0, "character": 0}}},
        )
        provider = LspEvidenceProvider(client=client)
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.evidence_count == 0

    def test_lookup_failure_is_isolated_not_raised(self, tmp_path: Path) -> None:
        client = _mock_client()
        client.request_definition = AsyncMock(side_effect=RuntimeError("boom"))
        provider = LspEvidenceProvider(client=client)
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.AVAILABLE

    def test_symbol_cap_marks_partial(self, tmp_path: Path) -> None:
        many_symbols = [
            Symbol(
                name=f"sym{i}",
                qualified_name=f"a.sym{i}",
                kind=SymbolKind.FUNCTION,
                file_path="a.py",
                start_line=i,
                end_line=i,
            )
            for i in range(5)
        ]
        parsed = [ParsedFile(path="a.py", language="python", symbols=many_symbols)]
        client = _mock_client()
        provider = LspEvidenceProvider(client=client, max_symbols=2)
        _evidence, receipt = provider.collect(parsed, tmp_path)
        assert receipt.availability == ProviderAvailability.PARTIAL
        assert "capped" in receipt.reason
        assert client.request_definition.call_count == 2
