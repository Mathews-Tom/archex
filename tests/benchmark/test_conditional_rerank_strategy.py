"""Tests for the benchmark-only archex_query_conditional_rerank strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import archex.benchmark.strategies as strategies
import archex.index.rerank as rerank_module
from archex.benchmark.models import (
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
    SymbolicRerankMode,
)
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _conditional_rerank_bundle,  # pyright: ignore[reportPrivateUsage]
    _guarded_order,  # pyright: ignore[reportPrivateUsage]
    _minmax_normalize,  # pyright: ignore[reportPrivateUsage]
    _rerank_model_storage_bytes,  # pyright: ignore[reportPrivateUsage]
    _symbolic_rerank_bundle,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    reset_benchmark_retrieval_options,
    run_archex_query,
    run_archex_query_conditional_rerank,
    run_archex_query_symbolic_rerank,
    set_benchmark_retrieval_options,
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


# The strong-symbolic-evidence candidate is c0: its symbol_name "c0" matches the
# query term "c0", and the flat BM25 scores make the bundle ambiguous so the gate
# fires. The spy reranker reverses the head, which would demote c0 to last.
_EVIDENCE_QUESTION = "find c0 implementation now"


class TestSymbolicRerankStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK)
            is run_archex_query_symbolic_rerank
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK.value == "archex_query_symbolic_rerank"
        assert (
            Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK.value in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        assert Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK not in DEFAULT_STRATEGIES


class TestGuardedOrder:
    def test_protected_candidate_never_demoted(self) -> None:
        # The base order would put index 0 last; the guard floors it to rank 0.
        order, guard_fired = _guarded_order([0.0, 0.3, 0.7, 1.0], [True, False, False, False])
        assert order[0] == 0
        assert guard_fired == 1

    def test_no_protection_is_plain_sort(self) -> None:
        order, guard_fired = _guarded_order([0.0, 0.3, 0.7, 1.0], [False, False, False, False])
        assert order == [3, 2, 1, 0]
        assert guard_fired == 0

    def test_protected_already_on_top_does_not_fire(self) -> None:
        order, guard_fired = _guarded_order([1.0, 0.3, 0.2], [True, False, False])
        assert order == [0, 1, 2]
        assert guard_fired == 0

    def test_multiple_protected_keep_their_floor(self) -> None:
        # Two protected at indices 0 and 1; the base order reverses, but neither
        # may fall below its original rank.
        order, _ = _guarded_order([0.0, 0.1, 0.2, 0.3], [True, True, False, False])
        position = {index: rank for rank, index in enumerate(order)}
        assert position[0] <= 0
        assert position[1] <= 1


class TestMinMaxNormalize:
    def test_scales_to_unit_interval(self) -> None:
        assert _minmax_normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]

    def test_constant_maps_to_one(self) -> None:
        assert _minmax_normalize([2.0, 2.0]) == [1.0, 1.0]

    def test_empty(self) -> None:
        assert _minmax_normalize([]) == []


class TestSymbolicRerankBundle:
    def test_no_local_model_returns_bundle_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strategies, "_load_local_reranker", lambda: None)
        bundle = _bundle([1.0, 0.99, 0.98])
        rerank = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        assert rerank.provenance["cross_encoder_status"] == "skipped:unavailable"
        assert rerank.provenance["guard_fired"] == "0"
        assert [rc.chunk.id for rc in rerank.bundle.chunks] == ["c0", "c1", "c2"]

    def test_guard_never_demotes_strong_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_spy(monkeypatch)
        # GUARD mode orders by the cross-encoder alone, so the (reversed) spy would
        # demote the strong-evidence c0 to last; the floor must rescue it. This
        # isolates the never-demote floor from the blend's own lift on c0.
        token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(symbolic_rerank_mode=SymbolicRerankMode.GUARD)
        )
        try:
            bundle = _bundle([1.0, 0.99, 0.98, 0.97])  # flat -> ambiguous
            rerank = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        finally:
            reset_benchmark_retrieval_options(token)
        assert rerank.provenance["cross_encoder_status"] == "applied"
        assert rerank.provenance["symbolic_rerank_mode"] == "guard"
        order = [rc.chunk.id for rc in rerank.bundle.chunks]
        # The floor rescues c0 back to its pre-rerank rank 0.
        assert order[0] == "c0"
        assert int(rerank.provenance["guard_fired"]) >= 1
        assert set(order) == {"c0", "c1", "c2", "c3"}

    def test_guard_holds_in_blend_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_spy(monkeypatch)
        # alpha=0.8 lets the (reversed) cross-encoder dominate the blend and try to
        # demote the strong-evidence c0; the guard must still floor it to rank 0.
        token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(
                symbolic_rerank_mode=SymbolicRerankMode.BLEND,
                symbolic_rerank_alpha=0.8,
            )
        )
        try:
            bundle = _bundle([1.0, 0.99, 0.98, 0.97])
            rerank = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        finally:
            reset_benchmark_retrieval_options(token)
        assert rerank.provenance["symbolic_rerank_mode"] == "blend"
        order = [rc.chunk.id for rc in rerank.bundle.chunks]
        # The never-demote guard is a hard invariant in both modes, and here it
        # genuinely fires: the blend alone would have demoted c0.
        assert order[0] == "c0"
        assert int(rerank.provenance["guard_fired"]) >= 1

    def test_blend_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_spy(monkeypatch)
        token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(symbolic_rerank_mode=SymbolicRerankMode.BLEND)
        )
        try:
            bundle = _bundle([1.0, 0.99, 0.98, 0.97])
            first = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
            second = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        finally:
            reset_benchmark_retrieval_options(token)
        assert [rc.chunk.id for rc in first.bundle.chunks] == [
            rc.chunk.id for rc in second.bundle.chunks
        ]

    def test_confident_bm25_skips_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = _install_spy(monkeypatch)
        bundle = _bundle([10.0, 1.0, 1.0, 1.0])  # clear separation -> confident
        rerank = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        assert spy.calls == 0
        assert rerank.provenance["cross_encoder_status"] == "skipped:confident_bm25"
        assert rerank.provenance["bm25_ambiguous"] == "false"
        assert [rc.chunk.id for rc in rerank.bundle.chunks] == ["c0", "c1", "c2", "c3"]

    def test_latency_cap_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_spy(monkeypatch)
        bundle = _bundle([1.0, 0.99, 0.98])
        rerank = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        assert rerank.provenance["latency_cap_ms"] == "1500.0"

    def test_no_strong_evidence_matches_pure_cross_encoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # In GUARD mode (orders by the cross-encoder alone), with no query
        # symbol/path overlap no candidate is protected, the floor is inert, and
        # the lane's order equals the pure cross-encoder (conditional) order.
        _install_spy(monkeypatch)
        token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(symbolic_rerank_mode=SymbolicRerankMode.GUARD)
        )
        try:
            bundle = _bundle([1.0, 0.99, 0.98, 0.97])
            symbolic = _symbolic_rerank_bundle(bundle, question="unrelated query xyzzy")
        finally:
            reset_benchmark_retrieval_options(token)
        conditional = _conditional_rerank_bundle(bundle, question="unrelated query xyzzy")
        assert symbolic.provenance["guard_fired"] == "0"
        assert [rc.chunk.id for rc in symbolic.bundle.chunks] == [
            rc.chunk.id for rc in conditional.bundle.chunks
        ]

    def test_records_contributions_and_alpha(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_spy(monkeypatch)
        bundle = _bundle([1.0, 0.99, 0.98, 0.97])
        rerank = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        assert rerank.provenance["blend_alpha"] == "0.50"
        contributions = rerank.provenance["candidate_contributions"]
        assert contributions
        # c0 carries the strong-evidence flag in the per-candidate contributions.
        assert "0:ce=" in contributions
        assert "strong=1" in contributions

    def test_receipt_realigned_on_applied_rerank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_spy(monkeypatch)
        bundle = _bundle([1.0, 0.99, 0.98])
        rerank = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        assert rerank.bundle.receipt is not None
        receipt_handles = [item.handle for item in rerank.bundle.receipt.returned_context]
        chunk_handles = [chunk_handle(rc.chunk.id) for rc in rerank.bundle.chunks]
        assert receipt_handles == chunk_handles

    def test_conditional_rerank_output_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The symbolic lane must not alter the pure cross-encoder conditional lane.
        # On a strong-evidence candidate, conditional fully reverses (demoting it)
        # while symbolic guards it, so the two lanes diverge and conditional stays
        # the untouched pure-CE baseline.
        _install_spy(monkeypatch)
        bundle = _bundle([1.0, 0.99, 0.98, 0.97])
        conditional = _conditional_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        symbolic = _symbolic_rerank_bundle(bundle, question=_EVIDENCE_QUESTION)
        cond_order = [rc.chunk.id for rc in conditional.bundle.chunks]
        sym_order = [rc.chunk.id for rc in symbolic.bundle.chunks]
        # Conditional keeps the pure-CE order: strong-evidence c0 demoted to last.
        assert cond_order == ["c3", "c2", "c1", "c0"]
        # Symbolic guards c0 at its pre-rerank rank.
        assert sym_order[0] == "c0"
        assert cond_order != sym_order


class TestSymbolicRunFixture:
    def test_runs_without_model_and_matches_archex_query(self, python_simple_repo: Path) -> None:
        # No reranker is loaded in the benchmark env, so the lane is deterministic
        # and equivalent to archex_query retrieval (it only ever reorders).
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        task = BenchmarkTask(
            task_id="symbolic_rerank_test",
            repo="test/repo",
            commit="abc",
            question="main entry point function",
            expected_files=["main.py"],
            token_budget=4096,
        )
        result = run_archex_query_symbolic_rerank(task, python_simple_repo)
        plain = run_archex_query(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK
        assert result.provenance["cross_encoder_status"] == "skipped:unavailable"
        assert result.provenance["guard_fired"] == "0"
        # Symbolic rerank only reorders; with no model it equals archex_query.
        assert result.recall == plain.recall
        assert set(result.result_files) == set(plain.result_files)
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
