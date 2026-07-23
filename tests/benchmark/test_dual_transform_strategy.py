"""Tests for the benchmark-only archex_query_dual_transform strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import archex.benchmark.strategies as strategies
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.query_transform import transform_query
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _fuse_dual_bundles,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    run_archex_query_dual_transform,
)
from archex.models import (
    CodeChunk,
    ContextBundle,
    ContextReceipt,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    IndexConfig,
    PipelineTiming,
    RankedChunk,
)
from archex.scout import chunk_handle

if TYPE_CHECKING:
    from pathlib import Path


def _ranked(chunk_id: str, score: float, *, tokens: int = 10) -> RankedChunk:
    chunk = CodeChunk(
        id=chunk_id,
        content=f"content for {chunk_id}",
        file_path=f"{chunk_id}.py",
        start_line=1,
        end_line=5,
        language="python",
        token_count=tokens,
    )
    return RankedChunk(chunk=chunk, final_score=score)


def _bundle_with_receipt(chunks: list[RankedChunk], *, query: str = "q") -> ContextBundle:
    items = [
        ContextReceiptItem(
            handle=chunk_handle(rc.chunk.id),
            file_path=rc.chunk.file_path,
            start_line=rc.chunk.start_line,
            end_line=rc.chunk.end_line,
            content_hash=f"hash-{rc.chunk.id}",
        )
        for rc in chunks
    ]
    receipt = ContextReceipt(
        query=query,
        token_budget=ContextReceiptTokenBudget(requested=4096, consumed=0),
        index_revision="rev",
        returned_context=items,
        returned_total=len(items),
    )
    total = sum(rc.chunk.token_count for rc in chunks)
    return ContextBundle(query=query, chunks=chunks, token_count=total, receipt=receipt)


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_DUAL_TRANSFORM)
            is run_archex_query_dual_transform
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_DUAL_TRANSFORM.value == "archex_query_dual_transform"
        assert (
            Strategy.ARCHEX_QUERY_DUAL_TRANSFORM.value in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        # Discoverable as a benchmark lane, but never part of the product default.
        assert Strategy.ARCHEX_QUERY_DUAL_TRANSFORM in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_DUAL_TRANSFORM not in DEFAULT_STRATEGIES

    def test_default_strategies_unchanged(self) -> None:
        assert DEFAULT_STRATEGIES == [
            Strategy.RAW_FILES,
            Strategy.RAW_RIPGREP,
            Strategy.ARCHEX_QUERY,
        ]


class TestTransformQuery:
    """Deterministic transformation across the five required query shapes."""

    def test_bug_report(self) -> None:
        dual = transform_query(
            "The login button throws a NullPointerException when the session expires"
        )
        # Structural surfaces the error/exception type.
        assert "NullPointerException" in dual.structural
        # Behavioural preserves the natural-language symptom and domain nouns.
        assert "login" in dual.behavioral
        assert "session" in dual.behavioral
        assert "expires" in dual.behavioral
        assert not dual.structural_from_fallback
        assert not dual.behavioral_from_fallback

    def test_stack_trace(self) -> None:
        dual = transform_query(
            "Traceback (most recent call last):\n"
            '  File "auth/session.py", line 42, in refresh\n'
            "    raise TokenError(msg)\n"
            "TokenError: token expired"
        )
        assert dual.has_stack_trace
        # Structural surfaces the failing path and exception type.
        assert "auth/session.py" in dual.structural
        assert "TokenError" in dual.structural
        # The trace frame is stripped from behavioural prose-noise counting.
        assert "expired" in dual.behavioral

    def test_api_question(self) -> None:
        dual = transform_query(
            "How do I use the query() function to retrieve a ContextBundle within a budget?"
        )
        # Structural captures the call site and the symbol.
        assert "query" in dual.structural.split()
        assert "ContextBundle" in dual.structural
        # Behavioural keeps the how-to phrasing and the domain noun.
        assert "How" in dual.behavioral
        assert "retrieve" in dual.behavioral
        assert "context" in dual.behavioral

    def test_architecture_question(self) -> None:
        dual = transform_query(
            "How is the retrieval pipeline structured across the indexing and serving layers?"
        )
        # Pure natural language: no code tokens, so structural falls back to the
        # original query while behavioural carries the architecture vocabulary.
        assert dual.structural_from_fallback
        assert dual.structural == dual.original
        assert "retrieval" in dual.behavioral
        assert "pipeline" in dual.behavioral
        assert "structured" in dual.behavioral

    def test_pure_code_query(self) -> None:
        dual = transform_query("RankedChunk.final_score CodeChunk.token_count parse_imports")
        # Structural keeps the dotted symbols and snake_case identifier verbatim.
        assert "RankedChunk.final_score" in dual.structural
        assert "CodeChunk.token_count" in dual.structural
        assert "parse_imports" in dual.structural
        # Behavioural recovers the domain nouns embedded in the identifiers.
        assert "ranked" in dual.behavioral
        assert "imports" in dual.behavioral
        assert not dual.behavioral_from_fallback

    def test_import_targets_in_structural(self) -> None:
        dual = transform_query("import os\nfrom pkg.mod import helper, other")
        structural_tokens = dual.structural.split()
        assert "os" in structural_tokens
        assert "pkg.mod" in structural_tokens
        assert "helper" in structural_tokens
        assert "other" in structural_tokens

    def test_provenance_records_both_subqueries(self) -> None:
        dual = transform_query("AuthManager.validate_token fails on logout")
        prov = dual.to_provenance()
        assert prov["subquery_structural"] == dual.structural
        assert prov["subquery_behavioral"] == dual.behavioral
        for key in (
            "structural_token_count",
            "behavioral_token_count",
            "structural_from_fallback",
            "behavioral_from_fallback",
            "has_stack_trace",
        ):
            assert key in prov, key


class TestFuseDualBundles:
    def test_reciprocal_rank_fusion_dedups_and_reorders(self) -> None:
        structural = _bundle_with_receipt([_ranked("a", 5.0), _ranked("b", 3.0)])
        behavioral = _bundle_with_receipt([_ranked("b", 0.9), _ranked("c", 0.8)])

        fused, stats = _fuse_dual_bundles(structural, behavioral, query="orig", token_budget=4096)

        ids = [rc.chunk.id for rc in fused.chunks]
        # b appears in both lists -> highest RRF score -> ranked first.
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}
        assert stats["structural_candidates"] == 2
        assert stats["behavioral_candidates"] == 2
        assert stats["fused_candidates"] == 3
        assert stats["fused_included"] == 3
        assert fused.query == "orig"

    def test_budget_cap_keeps_top_even_when_it_alone_exceeds_budget(self) -> None:
        # Top hit alone exceeds the budget; the guard must keep it, and the
        # smaller second chunk must not slip in ahead of the dropped overflow.
        structural = _bundle_with_receipt([_ranked("a", 5.0, tokens=50)])
        behavioral = _bundle_with_receipt([_ranked("b", 0.9, tokens=10)])

        fused, stats = _fuse_dual_bundles(structural, behavioral, query="orig", token_budget=30)

        assert stats["fused_included"] == 1
        assert [rc.chunk.id for rc in fused.chunks] == ["a"]
        assert fused.token_count == 50

    def test_receipt_realigned_to_fused_chunks(self) -> None:
        structural = _bundle_with_receipt([_ranked("a", 5.0, tokens=30)])
        behavioral = _bundle_with_receipt([_ranked("b", 0.9, tokens=30)])

        fused, _ = _fuse_dual_bundles(structural, behavioral, query="orig", token_budget=30)

        assert fused.receipt is not None
        handles = {item.handle for item in fused.receipt.returned_context}
        kept = {chunk_handle(rc.chunk.id) for rc in fused.chunks}
        # Receipt rows describe exactly the chunks the fused bundle returned.
        assert handles == kept
        assert fused.receipt.returned_total == len(fused.chunks)


class TestRunDualTransformFixture:
    @pytest.fixture(autouse=True)
    def _no_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Keep the lane hermetic: BM25-only, never reach for embedding backends.
        monkeypatch.setattr(strategies, "_vector_dependencies_available", lambda: False)

    def _task(self, question: str, token_budget: int = 4096) -> BenchmarkTask:
        return BenchmarkTask(
            task_id="dual_transform_test",
            repo="test/repo",
            commit="abc",
            question=question,
            expected_files=["main.py"],
            token_budget=token_budget,
        )

    def test_runs_end_to_end(self, python_simple_repo: Path) -> None:
        task = self._task("How does main.py start the application and call main()?")
        result = run_archex_query_dual_transform(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_DUAL_TRANSFORM
        assert result.tool_calls == 1
        assert 0.0 <= result.required_file_recall <= 1.0
        # Added latency is reported like every other lane.
        assert result.wall_time_ms is not None
        assert result.wall_time_ms >= 0.0
        prov = result.provenance
        assert prov["subquery_structural"]
        assert prov["subquery_behavioral"]
        for key in (
            "structural_candidates",
            "behavioral_candidates",
            "fused_candidates",
            "fused_included",
        ):
            assert key in prov, key

    def test_index_config_never_enables_rerank(
        self, monkeypatch: pytest.MonkeyPatch, python_simple_repo: Path
    ) -> None:
        seen: list[IndexConfig] = []
        real_query_bundle = strategies._query_bundle  # pyright: ignore[reportPrivateUsage]

        def spy(
            task: BenchmarkTask,
            repo_path: Path,
            *,
            strategy: Strategy,
            index_config: IndexConfig,
            cache: bool,
        ) -> tuple[ContextBundle, IndexConfig, PipelineTiming]:
            seen.append(index_config)
            return real_query_bundle(
                task, repo_path, strategy=strategy, index_config=index_config, cache=cache
            )

        monkeypatch.setattr(strategies, "_query_bundle", spy)
        run_archex_query_dual_transform(
            self._task("main.py main function module"), python_simple_repo
        )

        # Two retrieval passes (structural + behavioural), neither reranked.
        assert len(seen) == 2
        assert all(cfg.rerank is False for cfg in seen)
