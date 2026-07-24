"""Tests for M7's index_repository() wiring: cache metadata, generation
fingerprint invalidation, and end-to-end runtime/coverage evidence
persistence through a real index build.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from archex.api import index_repository
from archex.index.store import IndexStore
from archex.integrations.runtime.models import ProviderAvailability
from archex.models import Config, IndexConfig, RepoSource


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


class TestIndexRepositoryRuntimeEvidenceWiring:
    def test_default_config_produces_no_runtime_evidence(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(source, config=Config(cache=False), index_config=IndexConfig())
        try:
            assert store.get_runtime_provider_receipts() == []
            assert store.get_runtime_coverage_evidence() == []
        finally:
            store.close()

    def test_enabled_provider_without_evidence_reports_unavailable(
        self, python_simple_repo: Path
    ) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(
            source,
            config=Config(cache=False),
            index_config=IndexConfig(runtime_evidence_providers=["coverage"]),
        )
        try:
            receipts = store.get_runtime_provider_receipts()
            assert len(receipts) == 1
            assert receipts[0].availability == ProviderAvailability.UNAVAILABLE
            assert store.get_runtime_coverage_evidence() == []
        finally:
            store.close()

    def test_enabled_provider_with_real_evidence_persists_coverage(
        self, python_simple_repo: Path
    ) -> None:
        head = _git_head(python_simple_repo)
        evidence_dir = python_simple_repo / ".archex" / "runtime-evidence" / "coverage"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "manifest.json").write_text(
            json.dumps({"revision": head, "tool": "coverage.py", "tool_version": "7.6.0"})
        )
        (evidence_dir / "coverage.xml").write_text(
            '<?xml version="1.0" ?>\n'
            '<coverage line-rate="1.0">\n'
            "<packages><package><classes>\n"
            '<class filename="main.py" line-rate="1.0">\n'
            '<lines><line number="1" hits="2"/></lines>\n'
            "</class>\n"
            "</classes></package></packages>\n"
            "</coverage>\n"
        )

        source = RepoSource(local_path=str(python_simple_repo))
        store = index_repository(
            source,
            config=Config(cache=False),
            index_config=IndexConfig(runtime_evidence_providers=["coverage"]),
        )
        try:
            receipts = store.get_runtime_provider_receipts()
            assert len(receipts) == 1
            assert receipts[0].availability == ProviderAvailability.AVAILABLE
            coverage_evidence = store.get_runtime_coverage_evidence()
            assert len(coverage_evidence) == 1
            assert coverage_evidence[0].file_path == "main.py"
            assert coverage_evidence[0].revision == head
        finally:
            store.close()


class TestIndexConfigMetadataCacheValidity:
    def test_runtime_evidence_providers_change_invalidates_cache(self, tmp_path: Path) -> None:
        from archex.api import (
            _index_config_metadata_matches,  # pyright: ignore[reportPrivateUsage]
        )

        db_path = tmp_path / "index.db"
        store = IndexStore(db_path)
        try:
            from archex.api import (
                _set_index_config_metadata,  # pyright: ignore[reportPrivateUsage]
            )

            _set_index_config_metadata(store, IndexConfig(runtime_evidence_providers=[]))
            assert _index_config_metadata_matches(store, IndexConfig(runtime_evidence_providers=[]))
            assert not _index_config_metadata_matches(
                store, IndexConfig(runtime_evidence_providers=["coverage"])
            )
        finally:
            store.close()
