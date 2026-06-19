"""Tests for the benchmark-only archex_query_bounded_rerank strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import archex.benchmark.strategies as strategies
from archex.benchmark.bounded_rerank import (
    RerankCaps,
    evidence_score,
    query_signals,
    symbolic_scores,
)
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _bounded_rerank_bundle,  # pyright: ignore[reportPrivateUsage]
    _load_local_reranker,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    run_archex_query_bounded_rerank,
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


def _chunk(
    chunk_id: str,
    *,
    file_path: str | None = None,
    content: str = "",
    symbol: str | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content or f"body of {chunk_id}",
        file_path=file_path or f"{chunk_id}.py",
        start_line=1,
        end_line=5,
        language="python",
        token_count=10,
        symbol_name=symbol,
    )


def _ranked(chunk: CodeChunk, score: float) -> RankedChunk:
    return RankedChunk(chunk=chunk, final_score=score)


def _bundle(ranked: list[RankedChunk], *, query: str = "q") -> ContextBundle:
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
        query=query,
        token_budget=ContextReceiptTokenBudget(requested=4096, consumed=0),
        index_revision="rev",
        returned_context=items,
        returned_total=len(items),
    )
    return ContextBundle(query=query, chunks=ranked, token_count=0, receipt=receipt)


class _SpyReranker:
    """Fake local reranker recording candidate-set sizes and forcing a reorder."""

    def __init__(self) -> None:
        self.candidate_counts: list[int] = []

    def rerank(
        self, query: str, candidates: list[tuple[CodeChunk, float]], top_k: int
    ) -> list[tuple[CodeChunk, float]]:
        self.candidate_counts.append(len(candidates))
        # Reverse order with descending scores so the last candidate scores best.
        return [
            (chunk, float(len(candidates) - i)) for i, (chunk, _) in enumerate(reversed(candidates))
        ]


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_BOUNDED_RERANK)
            is run_archex_query_bounded_rerank
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_BOUNDED_RERANK.value == "archex_query_bounded_rerank"
        assert (
            Strategy.ARCHEX_QUERY_BOUNDED_RERANK.value in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        assert Strategy.ARCHEX_QUERY_BOUNDED_RERANK in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_BOUNDED_RERANK not in DEFAULT_STRATEGIES


class TestEvidenceScoring:
    def test_path_match_outranks_no_evidence(self) -> None:
        signals = query_signals("bug in auth/session.py around validate_token")
        unmatched = _chunk("a", file_path="util/io.py", content="unrelated body")
        matched = _chunk("b", file_path="auth/session.py", symbol="validate_token")
        # Evidence-rich chunk is placed LAST so its path/symbol weight must
        # overcome the (higher) rank prior of the first position.
        scored = symbolic_scores([unmatched, matched], signals)
        assert scored[1] > scored[0]

    def test_rank_prior_breaks_ties(self) -> None:
        signals = query_signals("nothing matches here xyzzy")
        first = _chunk("a")
        second = _chunk("b")
        # No path/symbol/term evidence: only the rank prior differs by position.
        s_first = evidence_score(first, signals, rank=0, total=2)
        s_second = evidence_score(second, signals, rank=1, total=2)
        assert s_first > s_second


class TestBoundedRerankBundle:
    def test_symbolic_reorder_promotes_strong_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        weak = _ranked(_chunk("weak", file_path="util/io.py"), 9.0)
        strong = _ranked(
            _chunk("strong", file_path="auth/session.py", symbol="validate_token"), 1.0
        )
        bundle = _bundle([weak, strong])

        result = _bounded_rerank_bundle(
            bundle, question="auth/session.py validate_token", caps=RerankCaps()
        )

        # Despite a lower retrieval score, the evidence-rich chunk reranks first.
        assert [rc.chunk.id for rc in result.bundle.chunks] == ["strong", "weak"]
        assert result.provenance["cross_encoder_status"] == "skipped:unavailable"
        assert result.provenance["rerank_method"] == "symbolic"

    def test_candidate_cap_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        ranked = [_ranked(_chunk(f"c{i}"), float(20 - i)) for i in range(12)]
        bundle = _bundle(ranked)
        caps = RerankCaps(candidate_cap=8, latency_cap_ms=750.0)

        result = _bounded_rerank_bundle(bundle, question="c0 c1 module", caps=caps)

        assert result.provenance["candidate_cap"] == "8"
        assert result.provenance["candidates_reranked"] == "8"
        assert result.provenance["candidates_total"] == "12"
        # Chunks past the cap keep their original relative order (the tail).
        tail_ids = [rc.chunk.id for rc in result.bundle.chunks[8:]]
        assert tail_ids == ["c8", "c9", "c10", "c11"]
        # The full chunk set is preserved, only reordered within the cap.
        assert {rc.chunk.id for rc in result.bundle.chunks} == {f"c{i}" for i in range(12)}

    def test_cross_encoder_never_sees_more_than_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = _SpyReranker()
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: spy)
        ranked = [_ranked(_chunk(f"c{i}"), float(20 - i)) for i in range(12)]
        bundle = _bundle(ranked)
        caps = RerankCaps(candidate_cap=8, latency_cap_ms=10_000.0)

        result = _bounded_rerank_bundle(bundle, question="c0 c1", caps=caps)

        # The expensive model is invoked once, over exactly the compact set.
        assert spy.candidate_counts == [8]
        assert result.provenance["cross_encoder_status"] == "applied"
        assert result.provenance["rerank_method"] == "symbolic+cross_encoder"

    def test_unavailable_reranker_skips_with_provenance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        bundle = _bundle([_ranked(_chunk("a"), 1.0), _ranked(_chunk("b"), 0.5)])

        result = _bounded_rerank_bundle(bundle, question="a b", caps=RerankCaps())

        assert result.provenance["cross_encoder_status"] == "skipped:unavailable"
        assert result.provenance["rerank_method"] == "symbolic"
        assert result.provenance["rerank_ms"] == "0.00"

    def test_latency_cap_aborts_cross_encoder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ranked = [
            _ranked(_chunk("a", file_path="auth/session.py", symbol="validate_token"), 1.0),
            _ranked(_chunk("b", file_path="util/io.py"), 0.5),
        ]
        question = "auth/session.py validate_token"

        # True symbolic-only order, with no cross-encoder in play.
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        symbolic_order = [
            rc.chunk.id
            for rc in _bounded_rerank_bundle(
                bundle=_bundle(ranked), question=question, caps=RerankCaps(candidate_cap=8)
            ).bundle.chunks
        ]

        # A spy cross-encoder would reverse the order, but a zero latency budget
        # forces the measured pass to be discarded.
        spy = _SpyReranker()
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: spy)
        result = _bounded_rerank_bundle(
            bundle=_bundle(ranked),
            question=question,
            caps=RerankCaps(candidate_cap=8, latency_cap_ms=0.0),
        )

        assert result.provenance["cross_encoder_status"] == "aborted:latency"
        # The cross-encoder was invoked (and measured) but its reorder is discarded.
        assert spy.candidate_counts == [2]
        assert [rc.chunk.id for rc in result.bundle.chunks] == symbolic_order

    def test_receipt_realigned_to_reordered_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        weak = _ranked(_chunk("weak", file_path="util/io.py"), 9.0)
        strong = _ranked(
            _chunk("strong", file_path="auth/session.py", symbol="validate_token"), 1.0
        )
        bundle = _bundle([weak, strong])

        result = _bounded_rerank_bundle(
            bundle, question="auth/session.py validate_token", caps=RerankCaps()
        )

        assert result.bundle.receipt is not None
        receipt_handles = [item.handle for item in result.bundle.receipt.returned_context]
        chunk_handles = [chunk_handle(rc.chunk.id) for rc in result.bundle.chunks]
        # Receipt order tracks the reordered chunk order.
        assert receipt_handles == chunk_handles

    def test_provenance_fields_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        result = _bounded_rerank_bundle(
            _bundle([_ranked(_chunk("a"), 1.0)]), question="a", caps=RerankCaps()
        )
        for key in (
            "candidate_cap",
            "candidates_reranked",
            "candidates_total",
            "latency_cap_ms",
            "rerank_ms",
            "rerank_method",
            "cross_encoder_status",
        ):
            assert key in result.provenance, key


class TestLoadLocalReranker:
    def test_returns_none_without_loaded_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _no_models() -> list[str]:
            return []

        monkeypatch.setattr("archex.index.rerank.loaded_reranker_model_names", _no_models)
        assert _load_local_reranker() is None

    def test_reuses_cached_model_without_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from archex.benchmark.models import BenchmarkRetrievalOptions
        from archex.benchmark.strategies import (
            reset_benchmark_retrieval_options,
            set_benchmark_retrieval_options,
        )
        from archex.index import rerank as rerank_mod

        fake_model = object()
        cache = rerank_mod._MODEL_CACHE  # pyright: ignore[reportPrivateUsage]
        # Seed an already-loaded model under the transformers-rerank-API name so
        # _load_model's cache short-circuit returns it without padding-token work.
        monkeypatch.setitem(cache, rerank_mod.DEFAULT_MODEL, fake_model)

        def _boom(*args: object, **kwargs: object) -> object:
            raise AssertionError("must not resolve or download a model")

        monkeypatch.setattr(rerank_mod, "resolve_hf_model_path", _boom)

        # The default reranker requires remote code; mirror the opt-in so warming
        # the already-cached model is allowed (it still never downloads).
        token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions(allow_remote_code=True))
        try:
            reranker = _load_local_reranker()
        finally:
            reset_benchmark_retrieval_options(token)

        assert reranker is not None
        # The warmed reranker reuses the cached object; no download path was taken.
        assert reranker._model is fake_model  # pyright: ignore[reportPrivateUsage]

    def test_cached_model_skipped_without_remote_code_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from archex.index import rerank as rerank_mod

        cache = rerank_mod._MODEL_CACHE  # pyright: ignore[reportPrivateUsage]
        # A remote-code model is cached, but the default benchmark options do not
        # opt in, so warming is rejected and the lane treats it as unavailable.
        monkeypatch.setitem(cache, rerank_mod.DEFAULT_MODEL, object())
        assert _load_local_reranker() is None


class TestRunBoundedRerankFixture:
    def _task(self, question: str, token_budget: int = 4096) -> BenchmarkTask:
        return BenchmarkTask(
            task_id="bounded_rerank_test",
            repo="test/repo",
            commit="abc",
            question=question,
            expected_files=["main.py"],
            token_budget=token_budget,
        )

    def test_runs_end_to_end(self, python_simple_repo: Path) -> None:
        task = self._task("main.py main function module")
        result = run_archex_query_bounded_rerank(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_BOUNDED_RERANK
        assert result.tool_calls == 1
        assert result.wall_time_ms >= 0.0
        prov = result.provenance
        # No model is loaded in a fresh benchmark process: the lane skips the CE.
        assert prov["cross_encoder_status"] == "skipped:unavailable"
        assert int(prov["candidates_reranked"]) <= int(prov["candidate_cap"])
