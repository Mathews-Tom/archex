"""Tests for the benchmark-only archex_query_conditional_rerank strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import archex.benchmark.strategies as strategies
import archex.index.rerank as rerank_module
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _conditional_rerank_bundle,  # pyright: ignore[reportPrivateUsage]
    _rerank_model_storage_bytes,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    run_archex_query,
    run_archex_query_conditional_rerank,
)
from archex.models import (
    CodeChunk,
    ContextBundle,
    ContextReceipt,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    RankedChunk,
)
from archex.scout import chunk_handle

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _chunk(chunk_id: str) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=f"body of {chunk_id}",
        file_path=f"{chunk_id}.py",
        start_line=1,
        end_line=5,
        language="python",
        token_count=10,
        symbol_name=chunk_id,
    )


def _bundle(scores: list[float]) -> ContextBundle:
    ranked = [
        RankedChunk(chunk=_chunk(f"c{i}"), final_score=score) for i, score in enumerate(scores)
    ]
    items = [
        ContextReceiptItem(
            handle=chunk_handle(rc.chunk.id),
            file_path=rc.chunk.file_path,
            start_line=rc.chunk.start_line,
            end_line=rc.chunk.end_line,
            content_hash=f"hash-{rc.chunk.id}",
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
    return ContextBundle(query="q", chunks=ranked, token_count=0, receipt=receipt)


class _SpyReranker:
    """Fake local reranker recording invocations and forcing a reorder."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self, query: str, candidates: list[tuple[CodeChunk, float]], top_k: int
    ) -> list[tuple[CodeChunk, float]]:
        self.calls += 1
        return [
            (chunk, float(len(candidates) - i)) for i, (chunk, _) in enumerate(reversed(candidates))
        ]


def _install_spy(monkeypatch: pytest.MonkeyPatch) -> _SpyReranker:
    spy = _SpyReranker()
    monkeypatch.setattr(strategies, "_load_local_reranker", lambda: spy)
    monkeypatch.setattr(rerank_module, "loaded_reranker_model_names", lambda: ["test/model"])
    return spy


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_CONDITIONAL_RERANK)
            is run_archex_query_conditional_rerank
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_CONDITIONAL_RERANK.value == "archex_query_conditional_rerank"
        assert (
            Strategy.ARCHEX_QUERY_CONDITIONAL_RERANK.value
            in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        assert Strategy.ARCHEX_QUERY_CONDITIONAL_RERANK in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_CONDITIONAL_RERANK not in DEFAULT_STRATEGIES


class TestModelStorageBytes:
    def test_local_model_directory_is_measured(self, tmp_path: Path) -> None:
        (tmp_path / "model.bin").write_bytes(b"x" * 128)
        (tmp_path / "config.json").write_bytes(b"y" * 64)
        assert _rerank_model_storage_bytes(str(tmp_path)) == 192

    def test_non_local_model_is_unmeasured(self) -> None:
        # An HF id (not a local path) is never walked / downloaded; reports None.
        assert _rerank_model_storage_bytes("jinaai/jina-reranker-v3") is None


class TestConditionalRerankBundle:
    def test_no_local_model_returns_bundle_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        bundle = _bundle([1.0, 0.99, 0.98])
        rerank = _conditional_rerank_bundle(bundle, question="how does retrieval work")
        assert rerank.provenance["cross_encoder_status"] == "skipped:unavailable"
        assert rerank.provenance["rerank_invoked"] == "false"
        assert rerank.storage_bytes is None
        assert [rc.chunk.id for rc in rerank.bundle.chunks] == ["c0", "c1", "c2"]

    def test_ambiguous_bm25_invokes_model_and_reorders(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _install_spy(monkeypatch)
        bundle = _bundle([1.0, 0.99, 0.98, 0.97])  # flat -> ambiguous
        rerank = _conditional_rerank_bundle(bundle, question="how does retrieval work overall")
        assert spy.calls == 1
        assert rerank.provenance["cross_encoder_status"] == "applied"
        assert rerank.provenance["bm25_ambiguous"] == "true"
        # Spy reverses the head; the full candidate set is preserved.
        assert {rc.chunk.id for rc in rerank.bundle.chunks} == {"c0", "c1", "c2", "c3"}
        assert [rc.chunk.id for rc in rerank.bundle.chunks][0] == "c3"

    def test_confident_bm25_skips_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = _install_spy(monkeypatch)
        bundle = _bundle([10.0, 1.0, 1.0, 1.0])  # clear separation -> confident
        rerank = _conditional_rerank_bundle(bundle, question="parse config loader")
        assert spy.calls == 0
        assert rerank.provenance["cross_encoder_status"] == "skipped:confident_bm25"
        assert rerank.provenance["bm25_ambiguous"] == "false"
        assert [rc.chunk.id for rc in rerank.bundle.chunks] == ["c0", "c1", "c2", "c3"]

    def test_receipt_realigned_on_applied_rerank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_spy(monkeypatch)
        bundle = _bundle([1.0, 0.99, 0.98])
        rerank = _conditional_rerank_bundle(bundle, question="how does retrieval work overall")
        assert rerank.bundle.receipt is not None
        receipt_handles = [item.handle for item in rerank.bundle.receipt.returned_context]
        chunk_handles = [chunk_handle(rc.chunk.id) for rc in rerank.bundle.chunks]
        assert receipt_handles == chunk_handles


class TestRunFixture:
    def test_runs_without_model_and_matches_archex_query_recall(
        self, python_simple_repo: Path
    ) -> None:
        # No reranker is loaded in the benchmark env, so the lane is deterministic
        # and equivalent to archex_query retrieval (it only ever reorders).
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        task = BenchmarkTask(
            task_id="conditional_rerank_test",
            repo="test/repo",
            commit="abc",
            question="main entry point function",
            expected_files=["main.py"],
            token_budget=4096,
        )
        result = run_archex_query_conditional_rerank(task, python_simple_repo)
        plain = run_archex_query(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_CONDITIONAL_RERANK
        assert result.provenance["cross_encoder_status"] == "skipped:unavailable"
        for key in ("bm25_cv", "bm25_ambiguous", "rerank_model_storage_bytes"):
            assert key in result.provenance
        # Conditional rerank only reorders; the retrieved file set is unchanged.
        assert result.recall == plain.recall
        assert set(result.result_files) == set(plain.result_files)
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
