"""Tests for the benchmark-only archex_query_compressed strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import (
    _compress_bundle,
    _passthrough_required,
    default_strategy_registry,
    run_archex_query,
    run_archex_query_compressed,
)
from archex.models import (
    CodeChunk,
    CompressionMode,
    ContextBundle,
    ContextReceipt,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    RankedChunk,
)
from archex.receipt import chunk_content_hash
from archex.scout import chunk_handle

if TYPE_CHECKING:
    from pathlib import Path


def _chunk(chunk_id: str, *, score: float, body_lines: int = 40) -> RankedChunk:
    body = "\n".join(f"    acc = acc + value_{i}" for i in range(body_lines))
    content = f"def fn_{chunk_id}(value):\n    acc = 0\n{body}\n    return acc"
    chunk = CodeChunk(
        id=chunk_id,
        content=content,
        file_path=f"{chunk_id}.py",
        start_line=1,
        end_line=body_lines + 3,
        language="python",
    )
    return RankedChunk(chunk=chunk, final_score=score)


def _bundle(chunks: list[RankedChunk], *, query: str = "q") -> ContextBundle:
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
        query=query,
        token_budget=ContextReceiptTokenBudget(requested=4096, consumed=100),
        index_revision="rev",
        returned_context=items,
    )
    return ContextBundle(query=query, chunks=chunks, token_count=300, receipt=receipt)


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        assert (
            default_strategy_registry.get(Strategy.ARCHEX_QUERY_COMPRESSED)
            is run_archex_query_compressed
        )

    def test_strategy_value(self) -> None:
        assert Strategy.ARCHEX_QUERY_COMPRESSED.value == "archex_query_compressed"
        assert Strategy.ARCHEX_QUERY_COMPRESSED.value in default_strategy_registry.strategy_names


class TestPassthroughRequired:
    def test_top_hit_always_passes_through(self) -> None:
        ranked = _chunk("a", score=0.01)
        assert (
            _passthrough_required(
                0, ranked, seed_paths=set(), expanded_paths=set(), top_score=1.0
            )
            is True
        )

    def test_graph_frontier_expansion_is_compressible(self) -> None:
        ranked = _chunk("b", score=0.9)
        assert (
            _passthrough_required(
                3, ranked, seed_paths=set(), expanded_paths={"b.py"}, top_score=1.0
            )
            is False
        )

    def test_seed_region_passes_through(self) -> None:
        ranked = _chunk("c", score=0.01)
        assert (
            _passthrough_required(
                2, ranked, seed_paths={"c.py"}, expanded_paths=set(), top_score=1.0
            )
            is True
        )

    def test_high_score_passes_through_low_score_compresses(self) -> None:
        high = _chunk("d", score=0.7)
        low = _chunk("e", score=0.2)
        assert (
            _passthrough_required(
                1, high, seed_paths=set(), expanded_paths=set(), top_score=1.0
            )
            is True
        )
        assert (
            _passthrough_required(
                1, low, seed_paths=set(), expanded_paths=set(), top_score=1.0
            )
            is False
        )


class TestCompressBundle:
    def test_compression_metrics_and_passthrough(self) -> None:
        chunks = [_chunk("a", score=1.0), _chunk("b", score=0.1), _chunk("c", score=0.05)]
        bundle = _bundle(chunks)
        result = _compress_bundle(bundle, question="where is the widget rendered")

        fields = result.result_fields
        assert fields["bundle_tokens_uncompressed"] > fields["bundle_tokens_compressed"]
        assert 0.0 < fields["bundle_compression_ratio"] < 1.0
        # The top hit is required and passes through; its tokens are protected.
        assert fields["required_context_passthrough_tokens"] > 0
        # Required context is never compressed and nothing required is hidden.
        assert fields["required_context_compressed_tokens"] == 0
        assert fields["compression_hidden_required_region_count"] == 0
        assert result.provenance["compressed_region_count"] == "2"
        assert result.provenance["passthrough_region_count"] == "1"

    def test_compressed_rows_expose_original_handle_and_hash(self) -> None:
        chunks = [_chunk("a", score=1.0), _chunk("b", score=0.05)]
        bundle = _bundle(chunks)
        result = _compress_bundle(bundle, question="where is the widget rendered")

        items = {item.file_path: item for item in result.bundle.receipt.returned_context}
        top = items["a.py"].compression
        assert top is not None
        assert top.compression_mode == CompressionMode.PASSTHROUGH_REQUIRED
        # Original hash matches the receipt's original content hash exactly.
        assert top.original_content_hash == items["a.py"].content_hash

        compressed = items["b.py"].compression
        assert compressed is not None
        assert compressed.compression_mode == CompressionMode.STRUCTURAL_CODE_ELISION
        assert compressed.original_content_hash == items["b.py"].content_hash
        assert compressed.fetch_original_handle == chunk_handle("b")
        assert compressed.is_compressed is True
        assert f"fetch original: {chunk_handle('b')}" in result.bundle.chunks[1].chunk.content

    def test_does_not_mutate_original_bundle(self) -> None:
        chunks = [_chunk("a", score=1.0), _chunk("b", score=0.05)]
        bundle = _bundle(chunks)
        original_content = bundle.chunks[1].chunk.content
        _compress_bundle(bundle, question="where is the widget rendered")
        assert bundle.chunks[1].chunk.content == original_content
        assert bundle.receipt.returned_context[0].compression is None

    def test_protect_code_skips_elision_for_debugging_intent(self) -> None:
        chunks = [_chunk("a", score=1.0), _chunk("b", score=0.05)]
        bundle = _bundle(chunks)
        result = _compress_bundle(bundle, question="investigate the rendering bug")
        assert result.provenance["protect_code"] == "true"
        # Code bodies are protected during fix/debug/review; nothing compresses.
        assert result.provenance["compressed_region_count"] == "0"
        assert result.result_fields["bundle_compression_ratio"] == 1.0


def _fixture_task(question: str, token_budget: int = 4096) -> BenchmarkTask:
    return BenchmarkTask(
        task_id="compression_test",
        repo="test/repo",
        commit="abc",
        question=question,
        expected_files=["main.py"],
        token_budget=token_budget,
    )


class TestRunFixture:
    def test_runs_and_reports_compression_fields(self, python_simple_repo: Path) -> None:
        task = _fixture_task("main entry point function")
        result = run_archex_query_compressed(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_COMPRESSED
        # Compression fields are populated (not the None default of other lanes).
        assert result.bundle_tokens_uncompressed is not None
        assert result.bundle_tokens_compressed is not None
        assert result.bundle_compression_ratio is not None
        assert result.compression_hidden_required_region_count == 0
        assert result.token_efficiency_with_compression_and_completion is not None
        for key in ("query_intent", "compressed_region_count", "passthrough_region_count"):
            assert key in result.provenance

    def test_no_effect_on_archex_query_retrieval(self, python_simple_repo: Path) -> None:
        task = _fixture_task("main entry point function")
        plain = run_archex_query(task, python_simple_repo)
        compressed = run_archex_query_compressed(task, python_simple_repo)

        # Retrieval metrics are attributable to the same uncompressed retrieval set.
        assert compressed.recall == plain.recall
        assert compressed.precision == plain.precision
        assert compressed.f1_score == plain.f1_score
        assert compressed.result_files == plain.result_files
        assert compressed.required_file_recall == plain.required_file_recall
        assert compressed.token_efficiency_with_completion == plain.token_efficiency_with_completion
        # archex_query never carries compression fields.
        assert plain.bundle_compression_ratio is None
        assert plain.bundle_tokens_compressed is None
