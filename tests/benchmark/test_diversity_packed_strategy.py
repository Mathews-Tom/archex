"""Tests for the benchmark-only archex_query_diversity_packed strategy."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _diversity_pack_bundle,  # pyright: ignore[reportPrivateUsage]
    _pack_bundle,  # pyright: ignore[reportPrivateUsage]
    _query_aspect_count,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    run_archex_query,
    run_archex_query_diversity_packed,
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
        task_id="diversity_packed_test",
        repo="test/repo",
        commit="abc",
        question=question,
        expected_files=["main.py"],
        token_budget=token_budget,
    )


def _ranked(chunk_id: str, *, file_path: str, score: float, content: str) -> RankedChunk:
    chunk = CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=content.count("\n") + 1,
        language="python",
        symbol_name=f"sym_{chunk_id}",
    )
    return RankedChunk(chunk=chunk, final_score=score)


def _bundle(
    chunks: list[RankedChunk],
    *,
    seed_files: list[str],
    expanded_files: list[str],
    token_budget: int,
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
    return ContextBundle(
        query="q",
        chunks=chunks,
        token_count=sum(count_tokens(rc.chunk.content) for rc in chunks),
        token_budget=token_budget,
        retrieval_metadata=meta,
        receipt=receipt,
    )


_DUP_BODY = (
    "def handler(request):\n"
    "    payload = parse(request)\n"
    "    result = process(payload)\n"
    "    return respond(result)\n"
)


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_DIVERSITY_PACKED)
            is run_archex_query_diversity_packed
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_DIVERSITY_PACKED.value == "archex_query_diversity_packed"
        assert (
            Strategy.ARCHEX_QUERY_DIVERSITY_PACKED.value in default_strategy_registry.strategy_names
        )

    def test_available_but_not_default(self) -> None:
        assert Strategy.ARCHEX_QUERY_DIVERSITY_PACKED in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_DIVERSITY_PACKED not in DEFAULT_STRATEGIES


class TestProductPathIndependence:
    def test_diversity_pack_bundle_takes_no_benchmark_task(self) -> None:
        params = set(inspect.signature(_diversity_pack_bundle).parameters)
        assert params == {"bundle", "question"}


class TestQueryAspectCount:
    def test_single_keyword_is_one_aspect(self) -> None:
        assert _query_aspect_count("configure") == 1

    def test_multi_keyword_is_multi_aspect(self) -> None:
        assert _query_aspect_count("service layer initialization order") > 1


class TestDiversityPackBundle:
    def _dup_bundle(self) -> ContextBundle:
        seed = _ranked("seed", file_path="main.py", score=1.0, content="def main():\n    run()\n")
        dup_a = _ranked("dup_a", file_path="util.py", score=0.3, content=_DUP_BODY)
        dup_b = _ranked("dup_b", file_path="util.py", score=0.3, content=_DUP_BODY)
        return _bundle(
            [seed, dup_a, dup_b],
            seed_files=["main.py"],
            expanded_files=["util.py"],
            token_budget=4096,
        )

    def test_narrow_query_bypasses_diversity(self) -> None:
        packing = _diversity_pack_bundle(self._dup_bundle(), question="configure")
        assert packing.provenance["diversity_applied"] == "false"
        assert packing.provenance["deselected_for_diversity"] == "0"
        # Identical to efficiency packing on the same bundle (diversity off).
        baseline = _pack_bundle(self._dup_bundle(), question="configure")
        assert {rc.chunk.id for rc in packing.bundle.chunks} == {
            rc.chunk.id for rc in baseline.bundle.chunks
        }

    def test_multi_aspect_query_deselects_redundant_tail(self) -> None:
        packing = _diversity_pack_bundle(
            self._dup_bundle(), question="how does the request handler process payloads"
        )
        assert packing.provenance["diversity_applied"] == "true"
        # One of the two identical util.py regions is de-selected for redundancy.
        assert int(packing.provenance["deselected_for_diversity"]) == 1
        assert packing.result_fields["diversity_deselected_regions"] == 1
        kept_ids = {rc.chunk.id for rc in packing.bundle.chunks}
        # The direct seed and one util representative survive; recall is preserved.
        assert "seed" in kept_ids
        kept_files = {rc.chunk.file_path for rc in packing.bundle.chunks}
        assert {"main.py", "util.py"} <= kept_files

    def test_recall_superset_vs_efficiency_packing(self) -> None:
        question = "how does the request handler process payloads"
        diversity = _diversity_pack_bundle(self._dup_bundle(), question=question)
        baseline = _pack_bundle(self._dup_bundle(), question=question)
        baseline_files = {rc.chunk.file_path for rc in baseline.bundle.chunks}
        diversity_files = {rc.chunk.file_path for rc in diversity.bundle.chunks}
        assert baseline_files <= diversity_files

    def test_required_region_never_hidden(self) -> None:
        packing = _diversity_pack_bundle(
            self._dup_bundle(), question="how does the request handler process payloads"
        )
        assert packing.result_fields["compression_hidden_required_region_count"] == 0


class TestRunFixture:
    def test_runs_and_reports_diversity_fields(self, python_simple_repo: Path) -> None:
        task = _fixture_task("how does the main entry point start the program")
        result = run_archex_query_diversity_packed(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_DIVERSITY_PACKED
        assert result.diversity_deselected_regions is not None
        assert result.packing_included_regions is not None
        assert result.compression_hidden_required_region_count == 0
        for key in ("diversity_applied", "query_aspects", "deselected_for_diversity"):
            assert key in result.provenance

    def test_required_files_preserved_vs_archex_query(self, python_simple_repo: Path) -> None:
        task = _fixture_task("how does the main entry point start the program")
        plain = run_archex_query(task, python_simple_repo)
        packed = run_archex_query_diversity_packed(task, python_simple_repo)
        # Diversity never drops a required file: recall does not regress.
        assert packed.required_file_recall >= plain.required_file_recall
        assert packed.diversity_deselected_regions is not None
        # The default lane never carries diversity fields.
        assert plain.diversity_deselected_regions is None
