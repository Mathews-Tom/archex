"""Tests for M6 semantic-evidence pipeline wiring: collect_semantic_evidence() and
the end-to-end index_repository() integration.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from archex.index.semantic_evidence import (
    _default_provider,  # pyright: ignore[reportPrivateUsage]
    collect_semantic_evidence,
)
from archex.integrations.semantic.lsp_provider import LspEvidenceProvider
from archex.integrations.semantic.models import (
    ProviderAvailability,
    SemanticEdgeEvidence,
    SemanticEdgeKind,
    SemanticEvidenceLocation,
    SemanticProviderName,
    SemanticProviderReceipt,
)
from archex.integrations.semantic.scip_provider import ScipEvidenceProvider
from archex.models import EdgeKind, IndexConfig, ParsedFile

if TYPE_CHECKING:
    from pathlib import Path


class _FakeProvider:
    def __init__(self, name: SemanticProviderName, evidence: list[SemanticEdgeEvidence]) -> None:
        self._name = name
        self._evidence = evidence

    @property
    def name(self) -> SemanticProviderName:
        return self._name

    def probe(self, repo_root: Path) -> SemanticProviderReceipt:
        del repo_root
        return SemanticProviderReceipt(
            provider=self._name, availability=ProviderAvailability.AVAILABLE
        )

    def collect(
        self, parsed_files: list[ParsedFile], repo_root: Path
    ) -> tuple[list[SemanticEdgeEvidence], SemanticProviderReceipt]:
        del parsed_files, repo_root
        return self._evidence, SemanticProviderReceipt(
            provider=self._name,
            availability=ProviderAvailability.AVAILABLE,
            evidence_count=len(self._evidence),
        )


class _RaisingProvider:
    """A misbehaving provider that violates the never-raise contract."""

    @property
    def name(self) -> SemanticProviderName:
        return SemanticProviderName.SCIP

    def probe(self, repo_root: Path) -> SemanticProviderReceipt:
        del repo_root
        return SemanticProviderReceipt(
            provider=SemanticProviderName.SCIP, availability=ProviderAvailability.AVAILABLE
        )

    def collect(
        self, parsed_files: list[ParsedFile], repo_root: Path
    ) -> tuple[list[SemanticEdgeEvidence], SemanticProviderReceipt]:
        del parsed_files, repo_root
        raise RuntimeError("boom: unexpected provider bug")


def _evidence_item(provider: SemanticProviderName) -> SemanticEdgeEvidence:
    return SemanticEdgeEvidence(
        provider=provider,
        provider_version="1.0",
        kind=SemanticEdgeKind.DEFINITION,
        source=SemanticEvidenceLocation(file_path="a.py", line=1, character=0),
        target=SemanticEvidenceLocation(file_path="b.py", line=2, character=0),
        confidence=0.9,
    )


class TestCollectSemanticEvidence:
    def test_empty_config_returns_nothing(self, tmp_path: Path) -> None:
        evidence, receipts = collect_semantic_evidence(
            [], tmp_path, IndexConfig(semantic_evidence_providers=[])
        )
        assert evidence == []
        assert receipts == []

    def test_injected_provider_is_used(self, tmp_path: Path) -> None:
        fake = _FakeProvider(SemanticProviderName.SCIP, [_evidence_item(SemanticProviderName.SCIP)])
        evidence, receipts = collect_semantic_evidence(
            [],
            tmp_path,
            IndexConfig(semantic_evidence_providers=["scip"]),
            providers={"scip": fake},
        )
        assert len(evidence) == 1
        assert len(receipts) == 1
        assert receipts[0].availability == ProviderAvailability.AVAILABLE

    def test_unconfigured_provider_falls_back_to_default(self, tmp_path: Path) -> None:
        # No SCIP index present at tmp_path -> default ScipEvidenceProvider reports
        # UNAVAILABLE rather than raising or inventing edges.
        evidence, receipts = collect_semantic_evidence(
            [], tmp_path, IndexConfig(semantic_evidence_providers=["scip"])
        )
        assert evidence == []
        [receipt] = receipts
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert receipt.provider == SemanticProviderName.SCIP

    def test_provider_raising_degrades_to_unavailable_not_a_crash(self, tmp_path: Path) -> None:
        # A provider that violates 'never raise' must not abort the whole
        # index build -- it degrades to an explicit UNAVAILABLE receipt.
        evidence, receipts = collect_semantic_evidence(
            [],
            tmp_path,
            IndexConfig(semantic_evidence_providers=["scip"]),
            providers={"scip": _RaisingProvider()},
        )
        assert evidence == []
        [receipt] = receipts
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert receipt.provider == SemanticProviderName.SCIP
        assert "RuntimeError" in receipt.reason
        assert "boom" in receipt.reason

    def test_one_provider_raising_does_not_block_the_others(self, tmp_path: Path) -> None:
        lsp_fake = _FakeProvider(
            SemanticProviderName.LSP, [_evidence_item(SemanticProviderName.LSP)]
        )
        evidence, receipts = collect_semantic_evidence(
            [],
            tmp_path,
            IndexConfig(semantic_evidence_providers=["scip", "lsp"]),
            providers={"scip": _RaisingProvider(), "lsp": lsp_fake},
        )
        assert len(evidence) == 1
        assert evidence[0].provider == SemanticProviderName.LSP
        assert [r.availability for r in receipts] == [
            ProviderAvailability.UNAVAILABLE,
            ProviderAvailability.AVAILABLE,
        ]

    def test_multiple_providers_aggregate_in_order(self, tmp_path: Path) -> None:
        scip_fake = _FakeProvider(
            SemanticProviderName.SCIP, [_evidence_item(SemanticProviderName.SCIP)]
        )
        lsp_fake = _FakeProvider(
            SemanticProviderName.LSP, [_evidence_item(SemanticProviderName.LSP)]
        )
        evidence, receipts = collect_semantic_evidence(
            [],
            tmp_path,
            IndexConfig(semantic_evidence_providers=["scip", "lsp"]),
            providers={"scip": scip_fake, "lsp": lsp_fake},
        )
        expected_order = [SemanticProviderName.SCIP, SemanticProviderName.LSP]
        assert [e.provider for e in evidence] == expected_order
        assert [r.provider for r in receipts] == expected_order

    def test_default_provider_resolves_scip_and_lsp(self) -> None:
        assert isinstance(_default_provider("scip"), ScipEvidenceProvider)
        assert isinstance(_default_provider("lsp"), LspEvidenceProvider)

    def test_default_provider_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown semantic evidence provider"):
            _default_provider("bogus")


class TestIndexRepositoryWiring:
    def test_default_config_produces_no_semantic_edges(self, python_simple_repo: Path) -> None:
        from archex.api import index_repository
        from archex.models import Config, RepoSource

        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(source, config=Config(cache=False), index_config=IndexConfig())
        try:
            edges = store.get_edges()
            assert edges
            assert all(edge.provider is None for edge in edges)
            assert store.get_semantic_provider_receipts() == []
        finally:
            store.close()

    def test_scip_index_produces_semantic_edges(self, python_simple_repo: Path) -> None:
        from archex.api import index_repository
        from archex.integrations.semantic import scip_pb2
        from archex.models import Config, RepoSource

        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-python"
        index.metadata.tool_info.version = "0.5.0"

        main_doc = index.documents.add()
        main_doc.relative_path = "main.py"
        main_doc.language = "python"
        definition = main_doc.occurrences.add()
        definition.symbol = "scip-python python . . main/entry()."
        definition.symbol_roles = scip_pb2.SymbolRole.Definition
        definition.single_line_range.line = 0
        definition.single_line_range.start_character = 4

        models_doc = index.documents.add()
        models_doc.relative_path = "models.py"
        models_doc.language = "python"
        usage = models_doc.occurrences.add()
        usage.symbol = "scip-python python . . main/entry()."
        usage.range.extend([3, 0, 4])

        (python_simple_repo / "index.scip").write_bytes(index.SerializeToString())

        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(
            source,
            config=Config(cache=False),
            index_config=IndexConfig(semantic_evidence_providers=["scip"]),
        )
        try:
            edges = store.get_edges()
            semantic_edges = [e for e in edges if e.kind == EdgeKind.SEMANTIC_DEFINITION]
            assert len(semantic_edges) == 1
            assert semantic_edges[0].provider == "scip"
            assert semantic_edges[0].provider_version == "0.5.0"

            [receipt] = store.get_semantic_provider_receipts()
            assert receipt.provider == SemanticProviderName.SCIP
            assert receipt.availability == ProviderAvailability.AVAILABLE
            assert receipt.evidence_count > 0
        finally:
            store.close()


def _write_scip_index(repo_path: Path) -> None:
    from archex.integrations.semantic import scip_pb2

    index = scip_pb2.Index()
    index.metadata.tool_info.name = "scip-python"
    index.metadata.tool_info.version = "0.5.0"

    main_doc = index.documents.add()
    main_doc.relative_path = "main.py"
    main_doc.language = "python"
    definition = main_doc.occurrences.add()
    definition.symbol = "scip-python python . . main/entry()."
    definition.symbol_roles = scip_pb2.SymbolRole.Definition
    definition.single_line_range.line = 0
    definition.single_line_range.start_character = 4

    models_doc = index.documents.add()
    models_doc.relative_path = "models.py"
    models_doc.language = "python"
    usage = models_doc.occurrences.add()
    usage.symbol = "scip-python python . . main/entry()."
    usage.range.extend([3, 0, 4])

    (repo_path / "index.scip").write_bytes(index.SerializeToString())


class TestQueryCacheHitReceipt:
    def test_cached_query_still_reports_semantic_provider_receipts(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        """A warm cache-hit query must not silently drop the receipt.

        Regression test: the cache-hit branch in query() previously called
        _finalize_context_bundle without semantic_providers, so a cached
        query returned a receipt with semantic_providers=[] while its
        included_edges carried provider="scip" -- internally contradictory.
        """
        from archex.api import query
        from archex.models import Config, PipelineTiming, RepoSource

        _write_scip_index(python_simple_repo)
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=True, cache_dir=str(tmp_path / "cache"))
        index_config = IndexConfig(semantic_evidence_providers=["scip"])

        first_timing = PipelineTiming()
        _first = query(
            source,
            "how does entry work",
            config=config,
            index_config=index_config,
            timing=first_timing,
            token_budget=50,
            explicit_token_budget=True,
        )
        assert first_timing.strategy == "full"

        second_timing = PipelineTiming()
        second = query(
            source,
            "how does entry work",
            config=config,
            index_config=index_config,
            timing=second_timing,
            token_budget=50,
            explicit_token_budget=True,
        )
        assert second_timing.strategy == "cached"
        assert second.receipt is not None
        assert second.receipt.semantic_providers != []
        assert any(
            r.provider == SemanticProviderName.SCIP
            and r.availability == ProviderAvailability.AVAILABLE
            for r in second.receipt.semantic_providers
        )

    def test_warm_runtime_snapshot_query_reports_receipts_without_crashing(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        """A QueryRuntime-backed warm-snapshot query must not use a closed store.

        Regression test: when a runtime snapshot is reused, the original
        `store` handle is closed early (api.py) and retrieval switches to
        `search_store` (the warm snapshot's own store). Reading semantic
        provider receipts off the wrong (closed) handle raised
        sqlite3.ProgrammingError instead of returning the receipt.
        """
        from archex.api import query
        from archex.models import Config, RepoSource
        from archex.serve.runtime import QueryRuntime

        _write_scip_index(python_simple_repo)
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=True, cache_dir=str(tmp_path / "cache"))
        index_config = IndexConfig(semantic_evidence_providers=["scip"])

        runtime = QueryRuntime()
        try:
            query(
                source,
                "how does entry work",
                config=config,
                index_config=index_config,
                token_budget=50,
                explicit_token_budget=True,
                runtime=runtime,
            )
            second = query(
                source,
                "how does entry work",
                config=config,
                index_config=index_config,
                token_budget=50,
                explicit_token_budget=True,
                runtime=runtime,
            )
            assert second.receipt is not None
            assert second.receipt.semantic_providers != []
        finally:
            runtime.close()
