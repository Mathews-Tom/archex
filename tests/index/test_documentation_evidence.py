"""Tests for the conditional documentation-graph evidence dispatcher (M9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.index.documentation_evidence import collect_documentation_evidence
from archex.integrations.docs.models import (
    AdrRecord,
    DocEvidenceProviderName,
    DocProviderReceipt,
    DocumentationLink,
    OwnershipRecord,
    ProviderAvailability,
)
from archex.models import EdgeKind, IndexConfig

_REVISION = "a" * 40


class _StubDocLinkProvider:
    def __init__(self, *, raise_error: bool = False) -> None:
        self._raise_error = raise_error

    @property
    def name(self) -> str:
        return "doc_link"

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        del repo_root, expected_revision
        return DocProviderReceipt(
            provider=DocEvidenceProviderName.DOC_LINK,
            availability=ProviderAvailability.AVAILABLE,
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[DocumentationLink], DocProviderReceipt]:
        del repo_root
        if self._raise_error:
            raise RuntimeError("boom")
        return (
            [
                DocumentationLink(
                    doc_path="README.md",
                    target_path="src/a.py",
                    link_text="a",
                    revision=expected_revision,
                )
            ],
            DocProviderReceipt(
                provider=DocEvidenceProviderName.DOC_LINK,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=1,
            ),
        )


class _StubAdrProvider:
    @property
    def name(self) -> str:
        return "adr"

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        del repo_root, expected_revision
        return DocProviderReceipt(
            provider=DocEvidenceProviderName.ADR, availability=ProviderAvailability.AVAILABLE
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[AdrRecord], DocProviderReceipt]:
        del repo_root
        return (
            [
                AdrRecord(
                    adr_id="0001",
                    title="Use X",
                    status="Accepted",
                    doc_path="docs/adr/0001.md",
                    revision=expected_revision,
                )
            ],
            DocProviderReceipt(
                provider=DocEvidenceProviderName.ADR,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=1,
            ),
        )


class _StubOwnershipProvider:
    @property
    def name(self) -> str:
        return "ownership"

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        del repo_root, expected_revision
        return DocProviderReceipt(
            provider=DocEvidenceProviderName.OWNERSHIP,
            availability=ProviderAvailability.AVAILABLE,
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[OwnershipRecord], DocProviderReceipt]:
        del repo_root
        return (
            [
                OwnershipRecord(
                    path_pattern="/src/",
                    owners=["@team"],
                    source_path="CODEOWNERS",
                    revision=expected_revision,
                )
            ],
            DocProviderReceipt(
                provider=DocEvidenceProviderName.OWNERSHIP,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=1,
            ),
        )


class TestCollectDocumentationEvidence:
    def test_returns_empty_when_no_providers_requested(self, tmp_path: Path) -> None:
        links, adr, ownership, receipts = collect_documentation_evidence(
            tmp_path, [], expected_revision=_REVISION
        )
        assert links == []
        assert adr == []
        assert ownership == []
        assert receipts == []

    def test_rejects_unknown_provider_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown documentation evidence providers"):
            collect_documentation_evidence(tmp_path, ["bogus"], expected_revision=_REVISION)

    def test_uses_injected_doc_link_provider(self, tmp_path: Path) -> None:
        links, adr, ownership, receipts = collect_documentation_evidence(
            tmp_path,
            ["doc_link"],
            expected_revision=_REVISION,
            doc_link_provider=_StubDocLinkProvider(),
        )
        assert len(links) == 1
        assert adr == []
        assert ownership == []
        assert len(receipts) == 1
        assert receipts[0].provider == DocEvidenceProviderName.DOC_LINK

    def test_runs_all_three_providers_when_all_requested(self, tmp_path: Path) -> None:
        links, adr, ownership, receipts = collect_documentation_evidence(
            tmp_path,
            ["doc_link", "adr", "ownership"],
            expected_revision=_REVISION,
            doc_link_provider=_StubDocLinkProvider(),
            adr_provider=_StubAdrProvider(),
            ownership_provider=_StubOwnershipProvider(),
        )
        assert len(links) == 1
        assert len(adr) == 1
        assert len(ownership) == 1
        assert len(receipts) == 3

    def test_provider_exception_degrades_to_unavailable_receipt(self, tmp_path: Path) -> None:
        links, _adr, _ownership, receipts = collect_documentation_evidence(
            tmp_path,
            ["doc_link"],
            expected_revision=_REVISION,
            doc_link_provider=_StubDocLinkProvider(raise_error=True),
        )
        assert links == []
        assert len(receipts) == 1
        assert receipts[0].availability == ProviderAvailability.UNAVAILABLE
        assert "boom" in receipts[0].reason

    def test_default_providers_report_unavailable_on_bare_directory(self, tmp_path: Path) -> None:
        links, adr, ownership, receipts = collect_documentation_evidence(
            tmp_path, ["doc_link", "adr", "ownership"], expected_revision=_REVISION
        )
        assert links == []
        assert adr == []
        assert ownership == []
        assert {r.provider for r in receipts} == {
            DocEvidenceProviderName.DOC_LINK,
            DocEvidenceProviderName.ADR,
            DocEvidenceProviderName.OWNERSHIP,
        }
        assert all(r.availability == ProviderAvailability.UNAVAILABLE for r in receipts)


class TestDocumentationEvidenceGraphDistinctness:
    """M9's typed doc/ADR/ownership relations must never fold into DependencyGraph.

    Structurally distinct from M6's semantic-evidence channel, which does add
    typed edges to the graph: documentation evidence is association, never a
    code dependency, so it is stored and surfaced separately (never as an
    ``Edge``/``EdgeKind`` member) and reading it back never produces graph
    edges of any kind.
    """

    def test_full_index_never_adds_documentation_graph_edges(
        self, python_simple_repo: Path
    ) -> None:
        from archex.api import index_repository
        from archex.models import Config, RepoSource

        source = RepoSource(local_path=str(python_simple_repo))
        (python_simple_repo / "README.md").write_text("See [main](main.py) for the entry point.\n")

        baseline_store = index_repository(
            source, config=Config(cache=False), index_config=IndexConfig()
        )
        try:
            baseline_edges = baseline_store.get_edges()
            assert baseline_store.get_documentation_provider_receipts() == []
        finally:
            baseline_store.close()

        store = index_repository(
            source,
            config=Config(cache=False),
            index_config=IndexConfig(documentation_evidence_providers=["doc_link"]),
        )
        try:
            edges = store.get_edges()
            assert len(edges) == len(baseline_edges)
            assert all(edge.kind != EdgeKind.SEMANTIC_DEFINITION for edge in edges)

            [receipt] = store.get_documentation_provider_receipts()
            assert receipt.provider == DocEvidenceProviderName.DOC_LINK
            assert receipt.availability == ProviderAvailability.AVAILABLE

            links = store.get_documentation_links()
            assert any(link.target_path == "main.py" for link in links)
        finally:
            store.close()
