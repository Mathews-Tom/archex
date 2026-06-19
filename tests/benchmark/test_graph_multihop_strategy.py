"""Tests for the benchmark-only archex_query_graph_multihop strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import archex.benchmark.strategies as strategies
from archex.benchmark.graph_multihop import (
    ExpansionAction,
    ExpansionDecision,
    GraphEdge,
    MultihopCaps,
    MultihopResult,
    graph_multihop_expand,
)
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _apply_multihop_expansion,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    run_archex_query_graph_multihop,
)
from archex.models import (
    CodeChunk,
    ContextBundle,
    ContextOmittedEdgeReason,
    ContextReceipt,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    ContextSkippedReason,
    PipelineTiming,
    RankedChunk,
)
from archex.scout import chunk_handle

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from archex.index.store import IndexStore


# ---------------------------------------------------------------------------
# Pure expansion algorithm
# ---------------------------------------------------------------------------

_LINEAR_TOKENS = {name: 10 for name in "bcdefghij"}


class TestGraphMultihopExpand:
    def test_high_confidence_edge_expands(self) -> None:
        edges = [GraphEdge("a", "b", 0.9)]
        result = graph_multihop_expand(edges, ["a"], _LINEAR_TOKENS, MultihopCaps())
        assert result.added_files == ["b"]
        assert result.expanded_decisions()[0].confidence == 0.9

    def test_low_confidence_edge_suppressed(self) -> None:
        edges = [GraphEdge("a", "b", 0.2)]
        result = graph_multihop_expand(
            edges, ["a"], _LINEAR_TOKENS, MultihopCaps(confidence_threshold=0.5)
        )
        assert result.added_files == []
        assert result.action_count(ExpansionAction.SUPPRESSED_LOW_CONFIDENCE) == 1

    def test_hop_cap_limits_depth(self) -> None:
        # a -> b -> c -> d, all high confidence; hop_cap=2 stops before d.
        edges = [GraphEdge("a", "b", 0.9), GraphEdge("b", "c", 0.9), GraphEdge("c", "d", 0.9)]
        result = graph_multihop_expand(
            edges, ["a"], _LINEAR_TOKENS, MultihopCaps(hop_cap=2, frontier_cap=8)
        )
        assert result.added_files == ["b", "c"]
        assert result.hops_run == 2
        assert "d" not in result.added_files

    def test_frontier_cap_limits_breadth(self) -> None:
        # Four high-confidence neighbours of the seed; frontier_cap=2 keeps the best two.
        edges = [
            GraphEdge("a", "b", 0.9),
            GraphEdge("a", "c", 0.8),
            GraphEdge("a", "d", 0.7),
            GraphEdge("a", "e", 0.6),
        ]
        result = graph_multihop_expand(
            edges, ["a"], _LINEAR_TOKENS, MultihopCaps(hop_cap=1, frontier_cap=2)
        )
        # Ranked by confidence: b, c expand; d, e are frontier cuts.
        assert result.added_files == ["b", "c"]
        assert result.action_count(ExpansionAction.CUT_FRONTIER) == 2

    def test_token_budget_cuts_expansion(self) -> None:
        edges = [GraphEdge("a", "b", 0.9), GraphEdge("a", "c", 0.8)]
        # Budget fits one 10-token file only.
        result = graph_multihop_expand(
            edges, ["a"], {"b": 10, "c": 10}, MultihopCaps(hop_cap=1, token_budget=10)
        )
        assert result.added_files == ["b"]
        assert result.action_count(ExpansionAction.CUT_BUDGET) == 1

    def test_deterministic_ranking_alphabetical_tie_break(self) -> None:
        # Genuinely equal confidence so only the name tie-break can order b before c;
        # listing a->c first proves the sort (not insertion order) decides.
        edges = [GraphEdge("a", "c", 0.7), GraphEdge("a", "b", 0.7)]
        result = graph_multihop_expand(
            edges, ["a"], _LINEAR_TOKENS, MultihopCaps(hop_cap=1, frontier_cap=8)
        )
        assert result.added_files == ["b", "c"]

    def test_seed_never_readded_across_hops(self) -> None:
        # b expands at hop 1 then points back to the seed; a must stay excluded at hop 2.
        edges = [GraphEdge("a", "b", 0.9), GraphEdge("b", "a", 0.9)]
        result = graph_multihop_expand(
            edges, ["a"], _LINEAR_TOKENS, MultihopCaps(hop_cap=2, frontier_cap=8)
        )
        assert result.added_files == ["b"]

    def test_empty_inputs(self) -> None:
        result = graph_multihop_expand([], ["a"], {}, MultihopCaps())
        assert result.added_files == []
        assert result.hops_run == 0


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP)
            is run_archex_query_graph_multihop
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP.value == "archex_query_graph_multihop"
        assert (
            Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP.value in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        assert Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP not in DEFAULT_STRATEGIES


# ---------------------------------------------------------------------------
# Receipt / provenance integration
# ---------------------------------------------------------------------------


def _index_store(repo_path: Path) -> IndexStore:
    from archex.api import index_repository
    from archex.models import Config, IndexConfig, RepoSource

    source = RepoSource(local_path=str(repo_path), stable_identity="graph-multihop-test@1")
    return index_repository(
        source, config=Config(cache=False), index_config=IndexConfig(vector=False)
    )


def _bundle_from_chunks(query: str, chunks: list[CodeChunk]) -> ContextBundle:
    ranked = [RankedChunk(chunk=chunk, final_score=1.0) for chunk in chunks]
    items = [
        ContextReceiptItem(
            handle=chunk_handle(chunk.id),
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content_hash=f"h-{chunk.id}",
        )
        for chunk in chunks
    ]
    receipt = ContextReceipt(
        query=query,
        token_budget=ContextReceiptTokenBudget(requested=4096, consumed=0),
        index_revision="rev",
        returned_context=items,
        returned_total=len(items),
    )
    return ContextBundle(
        query=query,
        chunks=ranked,
        token_count=sum(chunk.token_count for chunk in chunks),
        receipt=receipt,
    )


class TestApplyMultihopExpansion:
    def test_appends_chunks_and_records_decisions(self, python_simple_repo: Path) -> None:
        store = _index_store(python_simple_repo)
        try:
            files = sorted({chunk.file_path for chunk in store.get_chunks()})
            seed = files[0]
            expand_file = next(path for path in files if path != seed)
            seed_chunks = store.get_chunks_for_files([seed])
            bundle = _bundle_from_chunks(seed, seed_chunks)

            decisions = [
                ExpansionDecision(
                    file=expand_file,
                    hop=1,
                    via=seed,
                    confidence=0.9,
                    tokens=10,
                    action=ExpansionAction.EXPANDED,
                ),
                ExpansionDecision(
                    file="low.py",
                    hop=1,
                    via=seed,
                    confidence=0.2,
                    tokens=5,
                    action=ExpansionAction.SUPPRESSED_LOW_CONFIDENCE,
                ),
                ExpansionDecision(
                    file="wide.py",
                    hop=1,
                    via=seed,
                    confidence=0.8,
                    tokens=5,
                    action=ExpansionAction.CUT_FRONTIER,
                ),
                ExpansionDecision(
                    file="big.py",
                    hop=1,
                    via=seed,
                    confidence=0.7,
                    tokens=999,
                    action=ExpansionAction.CUT_BUDGET,
                ),
            ]
            expansion = MultihopResult(added_files=[expand_file], decisions=decisions, hops_run=1)

            expanded = _apply_multihop_expansion(
                bundle, store, seed_files=[seed], expansion=expansion
            )

            # Expanded file's original chunks are appended to the seed bundle.
            files_in_bundle = {rc.chunk.file_path for rc in expanded.chunks}
            assert expand_file in files_in_bundle
            assert seed in files_in_bundle

            assert expanded.receipt is not None
            # The expanded edge is included; cuts are recorded as omitted edges.
            assert len(expanded.receipt.included_edges) == 1
            omit_reasons = {edge.reason for edge in expanded.receipt.omitted_edges}
            assert ContextOmittedEdgeReason.BELOW_THRESHOLD in omit_reasons
            assert ContextOmittedEdgeReason.OVER_BUDGET in omit_reasons
            # All three cut kinds appear as skipped candidates.
            skip_reasons = {cand.reason for cand in expanded.receipt.skipped_candidates}
            assert ContextSkippedReason.DEPENDENCY_FRONTIER_CUT in skip_reasons
            assert ContextSkippedReason.BELOW_THRESHOLD in skip_reasons
            assert ContextSkippedReason.OVER_BUDGET in skip_reasons
            # Seed/expansion diagnostics describe the returned bundle.
            assert expanded.retrieval_metadata.seed_file_paths == [seed]
            assert expanded.retrieval_metadata.expanded_file_paths == [expand_file]
            assert expanded.retrieval_metadata.expansion_files_added == 1
        finally:
            store.close()


class TestRunGraphMultihopFixture:
    def _task(self, question: str, token_budget: int = 4096) -> BenchmarkTask:
        return BenchmarkTask(
            task_id="graph_multihop_test",
            repo="test/repo",
            commit="abc",
            question=question,
            expected_files=["main.py"],
            token_budget=token_budget,
        )

    def test_runs_end_to_end_with_cap_provenance(self, python_simple_repo: Path) -> None:
        result = run_archex_query_graph_multihop(
            self._task("How does main call services and utils?"), python_simple_repo
        )
        assert result.strategy == Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP
        assert result.tool_calls == 1
        prov = result.provenance
        for key in (
            "hop_cap",
            "frontier_cap",
            "confidence_threshold",
            "expansion_token_budget",
            "seed_file_count",
            "hops_run",
            "files_expanded",
            "suppressed_low_confidence",
            "frontier_cuts",
            "budget_cuts",
        ):
            assert key in prov, key
        assert prov["hop_cap"] == "2"
        assert prov["frontier_cap"] == "8"

    def test_expands_graph_neighbours_from_single_seed(
        self, monkeypatch: pytest.MonkeyPatch, python_simple_repo: Path
    ) -> None:
        store = _index_store(python_simple_repo)
        try:
            seed_chunks = [c for c in store.get_chunks() if c.file_path == "main.py"][:1]
        finally:
            store.close()
        assert seed_chunks, "fixture must contain main.py chunks"
        seed_bundle = _bundle_from_chunks("main", seed_chunks)

        def fake_query_bundle(
            task: BenchmarkTask,
            repo_path: Path,
            *,
            strategy: Strategy,
            index_config: object,
            cache: bool,
        ) -> tuple[ContextBundle, object, PipelineTiming]:
            return seed_bundle, index_config, PipelineTiming()

        monkeypatch.setattr(strategies, "_query_bundle", fake_query_bundle)

        result = run_archex_query_graph_multihop(
            self._task("How does main import services and utils?"), python_simple_repo
        )

        # main.py depends on other fixture modules; with only main.py as a seed,
        # the bounded multi-hop expansion must add at least one graph neighbour.
        assert int(result.provenance["files_expanded"]) >= 1
        assert int(result.provenance["seed_file_count"]) == 1
