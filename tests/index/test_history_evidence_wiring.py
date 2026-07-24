"""Tests for M8's index_repository() wiring: cache metadata, generation
fingerprint invalidation, and end-to-end repository-memory evidence
persistence through a real index build.
"""

from __future__ import annotations

from pathlib import Path

from archex.api import index_repository
from archex.index.store import IndexStore
from archex.integrations.history.models import ProviderAvailability
from archex.models import Config, IndexConfig, RepoSource


class TestIndexRepositoryHistoryEvidenceWiring:
    def test_default_config_produces_no_history_evidence(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(source, config=Config(cache=False), index_config=IndexConfig())
        try:
            assert store.get_history_provider_receipts() == []
            assert store.get_history_change_cards() == []
        finally:
            store.close()

    def test_enabled_git_log_provider_collects_real_commit_history(
        self, python_simple_repo: Path
    ) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(
            source,
            config=Config(cache=False),
            index_config=IndexConfig(history_evidence_providers=["git_log"]),
        )
        try:
            receipts = store.get_history_provider_receipts()
            assert len(receipts) == 1
            assert receipts[0].availability == ProviderAvailability.AVAILABLE
            assert receipts[0].window_commit_count >= 1
            cards = store.get_history_change_cards()
            assert len(cards) >= 1
        finally:
            store.close()

    def test_enabled_operator_rationale_without_evidence_reports_unavailable(
        self, python_simple_repo: Path
    ) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(
            source,
            config=Config(cache=False),
            index_config=IndexConfig(history_evidence_providers=["operator_rationale"]),
        )
        try:
            receipts = store.get_history_provider_receipts()
            assert len(receipts) == 1
            assert receipts[0].availability == ProviderAvailability.UNAVAILABLE
            assert store.get_history_operator_rationale() == []
        finally:
            store.close()


class TestIndexConfigMetadataCacheValidity:
    def test_history_evidence_providers_change_invalidates_cache(self, tmp_path: Path) -> None:
        from archex.api import (
            _index_config_metadata_matches,  # pyright: ignore[reportPrivateUsage]
            _set_index_config_metadata,  # pyright: ignore[reportPrivateUsage]
        )

        db_path = tmp_path / "index.db"
        store = IndexStore(db_path)
        try:
            _set_index_config_metadata(store, IndexConfig(history_evidence_providers=[]))
            assert _index_config_metadata_matches(store, IndexConfig(history_evidence_providers=[]))
            assert not _index_config_metadata_matches(
                store, IndexConfig(history_evidence_providers=["git_log"])
            )
        finally:
            store.close()
