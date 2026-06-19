"""Tests for the benchmark-only archex_query_summary_sidecar strategy."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkRetrievalOptions, BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _filter_bundle_to_files,  # pyright: ignore[reportPrivateUsage]
    _select_summary_files,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    reset_benchmark_retrieval_options,
    run_archex_query_summary_sidecar,
    set_benchmark_retrieval_options,
)
from archex.benchmark.summary_sidecar import (
    SummaryEntry,
    SummaryGranularity,
    SummarySidecar,
    build_summary_sidecar,
    is_entry_stale,
    summarize_chunk,
)
from archex.models import (
    CodeChunk,
    ContextBundle,
    ContextReceipt,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    RankedChunk,
    RetrievalMetadata,
)
from archex.scout import CHUNK_HANDLE_PREFIX, chunk_handle

if TYPE_CHECKING:
    from pathlib import Path


def _entry(
    *,
    file_path: str,
    source_hash: str,
    summary: str,
    handle: str,
    symbol: str | None = None,
) -> SummaryEntry:
    return SummaryEntry(
        source_file_path=file_path,
        source_content_hash=source_hash,
        index_revision="rev1",
        generated_at="2026-01-01T00:00:00+00:00",
        granularity=SummaryGranularity.SYMBOL,
        summary=summary,
        symbol_name=symbol,
        start_line=1,
        end_line=3,
        fetch_original_handle=handle,
    )


def _index_chunks(repo_path: Path) -> list[CodeChunk]:
    from archex.api import index_repository
    from archex.models import Config, IndexConfig, RepoSource

    source = RepoSource(local_path=str(repo_path), stable_identity="summary-sidecar-test@1")
    store = index_repository(
        source, config=Config(cache=False), index_config=IndexConfig(vector=False)
    )
    return store.get_chunks()


def _build_fixture_sidecar(repo_path: Path, dest: Path) -> SummarySidecar:
    sidecar = build_summary_sidecar(
        _index_chunks(repo_path),
        repo_path,
        repo="test",
        commit="1",
        index_revision="rev1",
    )
    sidecar.save(dest)
    return sidecar


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR)
            is run_archex_query_summary_sidecar
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR.value == "archex_query_summary_sidecar"
        assert (
            Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR.value in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        assert Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR not in DEFAULT_STRATEGIES


class TestSummarizeChunk:
    def test_digest_prefers_signature_and_docstring(self) -> None:
        chunk = CodeChunk(
            id="c1",
            content="def parse(): ...",
            file_path="p.py",
            start_line=1,
            end_line=2,
            language="python",
            signature="def parse(path: str) -> AST",
            docstring="Parse a file into an AST.\nMore detail.",
            breadcrumbs="mod > parse",
        )
        digest = summarize_chunk(chunk)
        assert "def parse(path: str) -> AST" in digest
        assert "Parse a file into an AST." in digest
        # Only the first docstring line is kept.
        assert "More detail." not in digest

    def test_digest_falls_back_to_first_code_line(self) -> None:
        chunk = CodeChunk(
            id="c2",
            content="\n\nx = compute_total(items)\n",
            file_path="p.py",
            start_line=1,
            end_line=3,
            language="python",
        )
        assert summarize_chunk(chunk) == "x = compute_total(items)"


class TestSidecarRoundTripAndStaleness:
    def test_build_save_load_roundtrip(self, python_simple_repo: Path, tmp_path: Path) -> None:
        dest = tmp_path / "sidecar.json"
        built = _build_fixture_sidecar(python_simple_repo, dest)
        assert built.entries, "expected at least one summary entry"

        loaded = SummarySidecar.load(dest)
        assert loaded.entries == built.entries
        assert loaded.granularity is SummaryGranularity.SYMBOL
        # Every entry carries the metadata needed to trust and fetch it.
        for entry in loaded.entries:
            assert entry.source_content_hash
            assert entry.index_revision == "rev1"
            assert entry.generated_at
            assert entry.fetch_original_handle.startswith(CHUNK_HANDLE_PREFIX)

    def test_stale_detection_after_edit(self, python_simple_repo: Path, tmp_path: Path) -> None:
        built = _build_fixture_sidecar(python_simple_repo, tmp_path / "sidecar.json")
        assert not any(is_entry_stale(e, python_simple_repo) for e in built.entries)

        target = python_simple_repo / "models.py"
        target.write_text(target.read_text() + "\n# drift\n", encoding="utf-8")

        stale = [e for e in built.entries if is_entry_stale(e, python_simple_repo)]
        assert stale, "edited file's summaries must be detected stale"
        assert all(e.source_file_path == "models.py" for e in stale)

    def test_missing_file_is_stale(self, tmp_path: Path) -> None:
        entry = _entry(file_path="gone.py", source_hash="deadbeef", summary="x", handle="chunk:1")
        assert is_entry_stale(entry, tmp_path) is True


class TestSelectSummaryFiles:
    def test_ranks_by_overlap_and_excludes_stale(self, tmp_path: Path) -> None:
        real = tmp_path / "real.py"
        real.write_text("def query(): pass", encoding="utf-8")
        real_hash = hashlib.sha256(real.read_bytes()).hexdigest()

        fresh = _entry(
            file_path="real.py",
            source_hash=real_hash,
            summary="query function entrypoint",
            handle="chunk:1",
            symbol="query",
        )
        stale = _entry(
            file_path="gone.py",
            source_hash="stale",
            summary="query helper",
            handle="chunk:2",
        )
        sidecar = SummarySidecar(
            repo="r",
            commit="c",
            index_revision="rev1",
            generated_at="t",
            granularity=SummaryGranularity.SYMBOL,
            entries=[fresh, stale],
        )

        selection = _select_summary_files(sidecar, tmp_path, question="query", file_cap=10)

        # The stale entry's file is excluded; only the fresh match is selected.
        assert selection.selected_files == ["real.py"]
        assert selection.stats["entries_fresh"] == 1
        assert selection.stats["entries_stale"] == 1

    def test_no_overlap_selects_nothing(self, tmp_path: Path) -> None:
        real = tmp_path / "real.py"
        real.write_text("def query(): pass", encoding="utf-8")
        real_hash = hashlib.sha256(real.read_bytes()).hexdigest()
        entry = _entry(
            file_path="real.py", source_hash=real_hash, summary="auth login", handle="chunk:1"
        )
        sidecar = SummarySidecar(
            repo="r",
            commit="c",
            index_revision="rev1",
            generated_at="t",
            granularity=SummaryGranularity.SYMBOL,
            entries=[entry],
        )
        selection = _select_summary_files(
            sidecar, tmp_path, question="database migration", file_cap=10
        )
        assert selection.selected_files == []


def _ranked(chunk_id: str, file_path: str, content: str) -> RankedChunk:
    chunk = CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=3,
        language="python",
        token_count=10,
    )
    return RankedChunk(chunk=chunk, final_score=1.0)


def _bundle_with(ranked: list[RankedChunk]) -> ContextBundle:
    items = [
        ContextReceiptItem(
            handle=chunk_handle(rc.chunk.id),
            file_path=rc.chunk.file_path,
            start_line=1,
            end_line=3,
            content_hash=f"h-{rc.chunk.id}",
        )
        for rc in ranked
    ]
    receipt = ContextReceipt(
        query="q",
        token_budget=ContextReceiptTokenBudget(requested=4096, consumed=0),
        index_revision="rev",
        returned_context=items,
        returned_total=len(items),
    )
    metadata = RetrievalMetadata(seed_file_paths=[rc.chunk.file_path for rc in ranked])
    return ContextBundle(
        query="q",
        chunks=ranked,
        token_count=sum(rc.chunk.token_count for rc in ranked),
        retrieval_metadata=metadata,
        receipt=receipt,
    )


class TestFilterBundleToFiles:
    def test_keeps_selected_files_preserves_handles_and_resets_diagnostics(self) -> None:
        keep = _ranked("a", "keep.py", "def keep():\n    return 1")
        drop = _ranked("b", "drop.py", "def drop():\n    return 2")
        bundle = _bundle_with([keep, drop])

        filtered = _filter_bundle_to_files(bundle, {"keep.py"})

        assert [rc.chunk.file_path for rc in filtered.chunks] == ["keep.py"]
        # Returned content is the original code, never a summary digest.
        assert filtered.chunks[0].chunk.content == "def keep():\n    return 1"
        # Fetch-original handles are preserved and realigned to the kept set.
        assert filtered.receipt is not None
        assert [item.handle for item in filtered.receipt.returned_context] == [chunk_handle("a")]
        assert filtered.receipt.returned_total == 1
        # Seed diagnostics are reset to the kept set; the dropped file does not leak.
        assert filtered.retrieval_metadata.seed_file_paths == []
        assert filtered.retrieval_metadata.seed_files_found == 1
        assert filtered.token_count == 10


class TestRunSummarySidecarFixture:
    def _task(self, question: str, token_budget: int = 4096) -> BenchmarkTask:
        return BenchmarkTask(
            task_id="summary_sidecar_test",
            repo="test/repo",
            commit="abc",
            question=question,
            expected_files=["main.py"],
            token_budget=token_budget,
        )

    def test_absent_sidecar_falls_back_to_plain_retrieval(self, python_simple_repo: Path) -> None:
        # No opt-in configured: summary-first is disabled, plain retrieval runs.
        result = run_archex_query_summary_sidecar(
            self._task("main.py main function module"), python_simple_repo
        )
        assert result.strategy == Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR
        assert result.provenance["sidecar"] == "absent"
        assert result.provenance["summary_first"] == "false"

    def test_summary_first_selection_and_handles(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "sidecar.json"
        sidecar = _build_fixture_sidecar(python_simple_repo, dest)

        token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(summary_sidecar_path=str(dest))
        )
        try:
            result = run_archex_query_summary_sidecar(
                self._task("main entry point function module"), python_simple_repo
            )
        finally:
            reset_benchmark_retrieval_options(token)

        prov = result.provenance
        assert prov["sidecar"] == "loaded"
        assert prov["summary_first"] == "true"
        assert prov["original_code_retrieved"] == "true"
        assert prov["fetch_original_preserved"] == "true"
        assert int(prov["entries_total"]) == len(sidecar.entries)
        # Returned files are within the summary-selected set.
        selected = set(prov["summary_selected_files"].split(", "))
        assert set(result.result_files).issubset(selected)
        assert result.result_files, "summary-first selection returned no files"

    def test_stale_sidecar_entries_reported(self, python_simple_repo: Path, tmp_path: Path) -> None:
        dest = tmp_path / "sidecar.json"
        _build_fixture_sidecar(python_simple_repo, dest)
        # Drift the source after the sidecar was built.
        target = python_simple_repo / "models.py"
        target.write_text(target.read_text() + "\n# drift\n", encoding="utf-8")

        token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(summary_sidecar_path=str(dest))
        )
        try:
            result = run_archex_query_summary_sidecar(
                self._task("models data class fields"), python_simple_repo
            )
        finally:
            reset_benchmark_retrieval_options(token)

        assert int(result.provenance["entries_stale"]) > 0
