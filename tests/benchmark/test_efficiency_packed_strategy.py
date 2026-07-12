"""Tests for the benchmark-only archex_query_efficiency_packed strategy."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _pack_bundle,  # pyright: ignore[reportPrivateUsage]
    _pack_context_candidate_bundle,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    run_archex_query,
    run_archex_query_context_candidate,
    run_archex_query_efficiency_packed,
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
from archex.receipt import chunk_content_hash
from archex.reporting import count_tokens
from archex.scout import chunk_handle

if TYPE_CHECKING:
    from pathlib import Path


def _fixture_task(question: str, token_budget: int = 4096) -> BenchmarkTask:
    return BenchmarkTask(
        task_id="efficiency_packed_test",
        repo="test/repo",
        commit="abc",
        question=question,
        expected_files=["main.py"],
        token_budget=token_budget,
    )


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED)
            is run_archex_query_efficiency_packed
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED.value == "archex_query_efficiency_packed"
        assert (
            Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED.value
            in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        # Discoverable as a benchmark lane, but never part of the product default.
        assert Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED not in DEFAULT_STRATEGIES

    def test_context_candidate_is_available_but_not_default(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE)
            is run_archex_query_context_candidate
        )
        assert Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE not in DEFAULT_STRATEGIES


class TestProductPathIndependence:
    def test_pack_bundle_takes_no_benchmark_task(self) -> None:
        # The packer consumes a bundle + query only; it never receives the task or
        # its expected files/regions, so packing cannot read benchmark ground truth.
        params = set(inspect.signature(_pack_bundle).parameters)
        assert params == {"bundle", "question"}


class TestRunFixture:
    def test_runs_and_reports_packing_fields(self, python_simple_repo: Path) -> None:
        task = _fixture_task("main entry point function")
        result = run_archex_query_efficiency_packed(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED
        # Packing fields are populated (not the None default of other lanes).
        assert result.bundle_tokens_uncompressed is not None
        assert result.bundle_tokens_compressed is not None
        assert result.bundle_compression_ratio is not None
        assert result.packed_relevance_per_1k_tokens is not None
        assert result.packing_included_regions is not None
        # Required/direct targets are preserved, so no required region is hidden.
        assert result.compression_hidden_required_region_count == 0
        for key in ("include_count", "skip_count", "relevance_per_1k_tokens", "budget_tier"):
            assert key in result.provenance

    def test_default_query_unchanged(self, python_simple_repo: Path) -> None:
        task = _fixture_task("main entry point function")
        plain = run_archex_query(task, python_simple_repo)
        packed = run_archex_query_efficiency_packed(task, python_simple_repo)

        # The default lane never carries packing/compression fields.
        assert plain.bundle_compression_ratio is None
        assert plain.packed_relevance_per_1k_tokens is None
        # Direct/high-confidence targets are never compressed or hidden, so no
        # required region is hidden; the fixture's required file stays present.
        assert packed.compression_hidden_required_region_count == 0
        assert packed.all_required_files_present

    def test_packed_bundle_never_grows(self, python_simple_repo: Path) -> None:
        task = _fixture_task("main entry point function")
        packed = run_archex_query_efficiency_packed(task, python_simple_repo)
        assert packed.bundle_tokens_compressed is not None
        assert packed.bundle_tokens_uncompressed is not None
        # Packing only drops, anchors, or compresses; it never adds tokens.
        assert packed.bundle_tokens_compressed <= packed.bundle_tokens_uncompressed


def _ranked(
    chunk_id: str, *, file_path: str, score: float, body_lines: int, whole_file: bool = False
) -> RankedChunk:
    body = "\n".join(f"    acc = acc + value_{i}" for i in range(body_lines))
    content = f"def fn_{chunk_id}(value):\n    acc = 0\n{body}\n    return acc"
    chunk = CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=body_lines + 3,
        language="python",
        symbol_name=None if whole_file else f"fn_{chunk_id}",
    )
    return RankedChunk(chunk=chunk, final_score=score)


def _packing_bundle(
    chunks: list[RankedChunk],
    *,
    token_budget: int,
    seed_files: list[str],
    expanded_files: list[str],
) -> ContextBundle:
    items = [
        ContextReceiptItem(
            handle=chunk_handle(rc.chunk.id),
            file_path=rc.chunk.file_path,
            start_line=rc.chunk.start_line,
            end_line=rc.chunk.end_line,
            content_hash=chunk_content_hash(rc.chunk),
        )
        for rc in chunks
    ]
    receipt = ContextReceipt(
        query="q",
        token_budget=ContextReceiptTokenBudget(requested=token_budget, consumed=0),
        index_revision="rev",
        returned_context=items,
    )
    meta = RetrievalMetadata(
        seed_file_paths=list(seed_files), expanded_file_paths=list(expanded_files)
    )
    total = sum(count_tokens(rc.chunk.content) for rc in chunks)
    return ContextBundle(
        query="q",
        chunks=chunks,
        token_count=total,
        token_budget=token_budget,
        retrieval_metadata=meta,
        receipt=receipt,
    )


class TestPackBundle:
    def test_low_value_graph_distant_region_skipped_and_receipt_pruned(self) -> None:
        seed = _ranked("seed", file_path="main.py", score=1.0, body_lines=6)
        far = _ranked("far", file_path="util.py", score=0.1, body_lines=120)
        budget = count_tokens(seed.chunk.content) + 5  # only the direct seed fits
        bundle = _packing_bundle(
            [seed, far], token_budget=budget, seed_files=["main.py"], expanded_files=["util.py"]
        )
        packing = _pack_bundle(bundle, question="how does main start up")

        assert {rc.chunk.id for rc in packing.bundle.chunks} == {"seed"}
        assert packing.provenance["skip_count"] == "1"
        # Direct/required region is preserved verbatim.
        assert packing.result_fields["compression_hidden_required_region_count"] == 0
        # Receipt is pruned to the kept set and stays internally consistent.
        receipt = packing.bundle.receipt
        assert receipt is not None
        assert {item.handle for item in receipt.returned_context} == {chunk_handle("seed")}
        assert receipt.returned_total == 1
        assert receipt.token_budget.consumed == packing.bundle.token_count

    def test_compressed_region_keeps_original_hash_and_fetch_handle(self) -> None:
        seed = _ranked("seed", file_path="main.py", score=1.0, body_lines=4)
        # Whole-file, mid-score, compressible region from a graph-expanded file:
        # provisional COMPRESS under a standard budget with ample room.
        wf = _ranked("wf", file_path="svc.py", score=0.5, body_lines=60, whole_file=True)
        original_wf = wf.chunk.content
        bundle = _packing_bundle(
            [seed, wf], token_budget=4096, seed_files=["main.py"], expanded_files=["svc.py"]
        )
        packing = _pack_bundle(bundle, question="how does the service layer work")

        assert packing.provenance["compress_count"] == "1"
        receipt = packing.bundle.receipt
        assert receipt is not None
        wf_item = next(i for i in receipt.returned_context if i.handle == chunk_handle("wf"))
        assert wf_item.compression is not None
        assert wf_item.compression.fetch_original_handle == chunk_handle("wf")
        # The exact original region stays retrievable via its preserved hash.
        assert (
            wf_item.compression.original_content_hash != wf_item.compression.compressed_content_hash
        )
        # _pack_bundle works on a deep copy; the input bundle is untouched.
        assert bundle.chunks[1].chunk.content == original_wf

    def test_existing_packer_keeps_high_score_expansion_compressible(self) -> None:
        seed = _ranked("seed", file_path="main.py", score=0.1, body_lines=4)
        expanded = _ranked(
            "expanded",
            file_path="svc.py",
            score=1.0,
            body_lines=60,
            whole_file=True,
        )
        bundle = _packing_bundle(
            [seed, expanded],
            token_budget=4096,
            seed_files=["main.py"],
            expanded_files=["svc.py"],
        )

        packing = _pack_bundle(bundle, question="how does the service layer work")

        assert packing.provenance["compress_count"] == "1"
        assert packing.provenance["direct_match_count"] == "1"

    def test_packed_tokens_stay_within_budget(self) -> None:
        seed = _ranked("seed", file_path="main.py", score=1.0, body_lines=6)
        far = _ranked("far", file_path="util.py", score=0.1, body_lines=120)
        budget = count_tokens(seed.chunk.content) + 5
        bundle = _packing_bundle(
            [seed, far], token_budget=budget, seed_files=["main.py"], expanded_files=["util.py"]
        )
        packing = _pack_bundle(bundle, question="how does main start up")
        # Optional context is dropped/anchored so the packed bundle fits the budget.
        assert packing.bundle.token_count <= budget

    def test_context_candidate_preserves_receipt_backed_direct_and_graph_evidence(self) -> None:
        direct = _ranked("direct", file_path="target.py", score=0.2, body_lines=40)
        graph = _ranked("graph", file_path="dependency.py", score=0.1, body_lines=40)
        optional = _ranked("optional", file_path="tail.py", score=1.0, body_lines=120)
        bundle = _packing_bundle(
            [direct, graph, optional],
            token_budget=4096,
            seed_files=[],
            expanded_files=[],
        )
        assert bundle.receipt is not None
        items = {item.handle: item for item in bundle.receipt.returned_context}
        items[chunk_handle("direct")].reason_codes = ["coverage_seed:identifier:Target"]
        items[chunk_handle("graph")].reason_codes = ["coverage_neighbor:graph_import:target.py"]

        packing = _pack_context_candidate_bundle(
            bundle,
            question="Where is Target implemented?",
        )

        packed = {ranked.chunk.id: ranked.chunk for ranked in packing.bundle.chunks}
        assert packed["direct"].content == direct.chunk.content
        assert packed["graph"].content == graph.chunk.content
        receipt = packing.bundle.receipt
        assert receipt is not None
        packed_items = {item.handle: item for item in receipt.returned_context}
        assert "packing:include" in packed_items[chunk_handle("direct")].reason_codes
        assert (
            "packing_protection:direct_evidence"
            in packed_items[chunk_handle("direct")].reason_codes
        )
        assert (
            "packing_protection:required_graph_context"
            in packed_items[chunk_handle("graph")].reason_codes
        )
        assert packing.provenance["protected_direct_evidence_count"] == "1"
        assert packing.provenance["protected_graph_context_count"] == "1"
