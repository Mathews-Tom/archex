"""Tests for benchmark report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    Strategy,
    TaskCategory,
    TaskCompletionResult,
)
from archex.benchmark.reporter import (
    format_bucketed_summary,
    format_chunker_frontier_table,
    format_json,
    format_markdown,
    format_strategy_comparison,
    format_summary,
)

if TYPE_CHECKING:
    from archex.models import ChunkerName


def _make_result(
    strategy: Strategy,
    tokens: int = 1000,
    savings: float = 0.0,
    recall: float = 1.0,
    precision: float = 1.0,
    tokens_input: int = 2000,
    tokens_output: int = 1000,
    token_efficiency: float = 0.5,
    required_file_recall: float = 1.0,
    missed_required_file_rate: float = 0.0,
    missed_required_task_rate: float = 0.0,
    chunker: ChunkerName = "default",
    index_chunk_count: int = 10,
    mean_chunk_tokens: float = 50.0,
    category: TaskCategory | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="test",
        strategy=strategy,
        tokens_total=tokens,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        token_efficiency=token_efficiency,
        required_file_recall=required_file_recall,
        missed_required_file_rate=missed_required_file_rate,
        missed_required_task_rate=missed_required_task_rate,
        tool_calls=1,
        files_accessed=3,
        recall=recall,
        precision=precision,
        savings_vs_raw=savings,
        wall_time_ms=50.0,
        cached=False,
        timestamp="2025-01-01T00:00:00Z",
        chunker=chunker,
        index_chunk_count=index_chunk_count,
        mean_chunk_tokens=mean_chunk_tokens,
        category=category,
    )


def _make_report(results: list[BenchmarkResult] | None = None) -> BenchmarkReport:
    if results is None:
        results = [
            _make_result(Strategy.RAW_FILES, tokens=2000),
            _make_result(Strategy.ARCHEX_QUERY, tokens=500, savings=75.0, recall=0.8),
        ]
    return BenchmarkReport(
        task_id="test",
        repo="owner/repo",
        question="How does X work?",
        results=results,
        baseline_tokens=2000,
    )


class TestFormatMarkdown:
    def test_contains_header(self) -> None:
        md = format_markdown(_make_report())
        assert "## Benchmark: test" in md

    def test_contains_table_header(self) -> None:
        md = format_markdown(_make_report())
        assert "| Strategy |" in md
        assert "Required Recall" in md
        assert "Missed Task Rate" in md
        assert "Completion" in md
        assert "Receipt Accuracy" in md

    def test_contains_strategy_rows(self) -> None:
        md = format_markdown(_make_report())
        assert "raw_files" in md
        assert "archex_query" in md

    def test_contains_repo_and_question(self) -> None:
        md = format_markdown(_make_report())
        assert "owner/repo" in md
        assert "How does X work?" in md

    def test_contains_baseline(self) -> None:
        md = format_markdown(_make_report())
        assert "2,000" in md

    def test_contains_bundle_only_eval_section_when_present(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY)])
        result = report.results[0]
        result.bundle_only_success = TaskCompletionResult.FAIL
        result.needed_files_outside_returned = ["src/frontier.py", "src/absent.py"]
        result.needed_files_in_frontier_cut = ["src/frontier.py"]
        result.needed_files_in_top_candidates = ["src/frontier.py"]
        result.safe_to_act_false_positive = True
        result.post_bundle_read_turns = 2

        md = format_markdown(report)

        assert "Bundle-only evaluation" in md
        assert "src/frontier.py, src/absent.py" in md
        assert "Safe-to-act false positive" in md
        assert "| archex_query | fail |" in md
        assert "All Required | Post Reads | Completion" not in md
        assert "Safe-to-act false positive | Post reads" in md


class TestFormatJson:
    def test_valid_json(self) -> None:
        import json

        output = format_json(_make_report())
        data = json.loads(output)
        assert data["task_id"] == "test"
        assert len(data["results"]) == 2


class TestFormatSummary:
    def test_empty_reports(self) -> None:
        summary = format_summary([])
        assert "No benchmark results" in summary

    def test_summary_header(self) -> None:
        summary = format_summary([_make_report()])
        assert "# Benchmark Summary" in summary
        assert "**Tasks:** 1" in summary

    def test_summary_table(self) -> None:
        summary = format_summary([_make_report()])
        assert "| Strategy |" in summary
        assert "raw_files" in summary
        assert "archex_query" in summary
        assert "Avg Efficiency" in summary
        assert "Avg Required Recall" in summary
        assert "Missed File Rate" in summary
        assert "Missed Task Rate" in summary

    def test_multi_report_aggregation(self) -> None:
        r1 = _make_report(
            [
                _make_result(Strategy.RAW_FILES, tokens=2000),
                _make_result(Strategy.ARCHEX_QUERY, tokens=500, savings=75.0, recall=0.8),
            ]
        )
        r2 = _make_report(
            [
                _make_result(Strategy.RAW_FILES, tokens=3000),
                _make_result(Strategy.ARCHEX_QUERY, tokens=600, savings=80.0, recall=0.9),
            ]
        )
        summary = format_summary([r1, r2])
        assert "**Tasks:** 2" in summary


class TestFormatStrategyComparison:
    def test_empty_reports(self) -> None:
        result = format_strategy_comparison([])
        assert "No benchmark results" in result

    def test_contains_per_task_table(self) -> None:
        report = _make_report()
        result = format_strategy_comparison([report])
        assert "## test" in result
        assert "raw_files" in result
        assert "archex_query" in result
        assert "Tokens Total" in result
        assert "Required Recall" in result
        assert "Receipt Accuracy" in result

    def test_includes_missing_required_file_appendix(self) -> None:
        report = _make_report(
            [
                _make_result(Strategy.ARCHEX_QUERY),
                _make_result(Strategy.ARCHEX_SCOUT_FETCH),
            ]
        )
        report.results[0].required_files_missing = ["missing.py"]
        report.results[0].missed_required_file_rate = 1.0
        result = format_strategy_comparison([report])
        assert "Missing required files appendix" in result
        assert "missing.py" in result

    def test_strategy_comparison_includes_bundle_only_eval_section(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY)])
        report.results[0].bundle_only_success = TaskCompletionResult.PASS
        report.results[0].needed_files_outside_returned = ["src/absent.py"]

        result = format_strategy_comparison([report])

        assert "Bundle-only evaluation" in result
        assert "src/absent.py" in result

    def test_contains_head_to_head(self) -> None:
        report = _make_report()
        result = format_strategy_comparison([report])
        assert "Head-to-Head Wins" in result

    def test_contains_best_strategy(self) -> None:
        report = _make_report()
        result = format_strategy_comparison([report])
        assert "Best Strategy per Metric" in result


class TestFormatBucketedSummary:
    def test_empty_reports(self) -> None:
        assert "No benchmark results" in format_bucketed_summary([])

    def test_uncategorized_fallback(self) -> None:
        result = format_bucketed_summary([_make_report()])
        assert "uncategorized" in result

    def test_categorized_reports(self) -> None:
        r_self = _make_result(Strategy.ARCHEX_QUERY, recall=0.9, category=TaskCategory.SELF)
        r_ext = _make_result(
            Strategy.ARCHEX_QUERY, recall=0.5, category=TaskCategory.EXTERNAL_FRAMEWORK
        )
        report_self = BenchmarkReport(
            task_id="self_task",
            repo=".",
            question="q",
            results=[r_self],
            baseline_tokens=1000,
        )
        report_ext = BenchmarkReport(
            task_id="ext_task",
            repo="owner/repo",
            question="q",
            results=[r_ext],
            baseline_tokens=1000,
        )
        output = format_bucketed_summary([report_self, report_ext])
        assert "self (1 tasks)" in output
        assert "external-framework (1 tasks)" in output
        assert "All Tasks (2 tasks)" in output
        assert "Seed Recall" in output
        assert "Required Recall" in output
        assert "Missed File Rate" in output
        assert "Missed Task Rate" in output


class TestFormatChunkerFrontierTable:
    def test_contains_quality_latency_and_granularity_axes(self) -> None:
        baseline = _make_report(
            [
                _make_result(
                    Strategy.ARCHEX_QUERY_FUSION_RERANK,
                    recall=0.80,
                    precision=0.40,
                    token_efficiency=0.60,
                    chunker="default",
                    index_chunk_count=100,
                    mean_chunk_tokens=80.0,
                )
            ]
        )
        candidate = _make_report(
            [
                _make_result(
                    Strategy.ARCHEX_QUERY_FUSION_RERANK,
                    recall=0.82,
                    precision=0.42,
                    token_efficiency=0.61,
                    chunker="cast",
                    index_chunk_count=120,
                    mean_chunk_tokens=70.0,
                )
            ]
        )

        table = format_chunker_frontier_table([candidate], [baseline])

        assert "Chunker Frontier Comparison" in table
        assert "archex_query_fusion_rerank | default" in table
        assert "archex_query_fusion_rerank | cast" in table
        assert "Chunk Count" in table
        assert "Mean Chunk Tokens" in table


def _region_result(strategy: Strategy = Strategy.ARCHEX_QUERY) -> BenchmarkResult:
    return _make_result(strategy).model_copy(
        update={
            "region_recall": 0.5,
            "region_precision": 0.4,
            "region_f1": 0.44,
            "line_recall": 0.3,
            "line_precision": 0.6,
            "ranked_region_mrr": 0.5,
            "ranked_region_ndcg": 0.7,
            "context_noise_ratio": 0.25,
            "useful_tokens": 75,
            "wasted_tokens": 25,
            "relevance_per_1k_tokens": 5.0,
        }
    )


class TestRegionQualityReporting:
    def test_markdown_shows_region_table_when_labeled(self) -> None:
        md = format_markdown(_make_report([_region_result()]))
        assert "Region & Context Efficiency" in md
        assert "Noise Ratio" in md
        assert "Rel/1k" in md

    def test_markdown_omits_region_table_for_file_only_results(self) -> None:
        md = format_markdown(_make_report([_make_result(Strategy.ARCHEX_QUERY)]))
        assert "Region & Context Efficiency" not in md

    def test_summary_shows_region_table_when_labeled(self) -> None:
        summary = format_summary([_make_report([_region_result()])])
        assert "Region & Context Efficiency" in summary

    def test_summary_omits_region_table_for_file_only_results(self) -> None:
        summary = format_summary([_make_report([_make_result(Strategy.ARCHEX_QUERY)])])
        assert "Region & Context Efficiency" not in summary

    def test_json_includes_region_fields(self) -> None:
        import json

        data = json.loads(format_json(_make_report([_region_result()])))
        assert data["results"][0]["region_recall"] == 0.5
        assert data["results"][0]["context_noise_ratio"] == 0.25
        assert data["results"][0]["useful_tokens"] == 75

    def test_json_renders_labeled_metrics_and_unlabeled_nulls(self) -> None:
        import json

        report = _make_report(
            [
                _region_result(Strategy.ARCHEX_QUERY),
                _make_result(Strategy.RAW_FILES),
            ]
        )
        data = json.loads(format_json(report))

        labeled, unlabeled = data["results"]
        assert labeled["region_recall"] == 0.5
        assert labeled["line_recall"] == 0.3
        assert labeled["context_noise_ratio"] == 0.25
        assert labeled["relevance_per_1k_tokens"] == 5.0
        assert unlabeled["region_recall"] is None
        assert unlabeled["line_recall"] is None
        assert unlabeled["context_noise_ratio"] is None
        assert unlabeled["relevance_per_1k_tokens"] is None


def _task_aware_result() -> BenchmarkResult:
    result = _make_result(Strategy.ARCHEX_QUERY_TASK_AWARE, tokens=600)
    result.provenance = {
        "modality": "pl_to_pl",
        "budget_tier": "standard",
        "routing_decision": "bm25_only",
        "dense_expansion": "skipped:confident_sparse",
        "fusion_used": "false",
        "policy_candidate_cap": "40",
        "policy_dense_candidate_cap": "20",
        "policy_skipped_steps": "cross_encoder_rerank",
    }
    return result


class TestTaskAwarePolicyAppendix:
    def test_appendix_rendered_for_task_aware(self) -> None:
        md = format_markdown(_make_report([_task_aware_result()]))
        assert "### Task-aware policy" in md
        # Every populated column is rendered in the appendix section.
        appendix = md.split("### Task-aware policy", 1)[1]
        cells = ("pl_to_pl", "standard", "bm25_only", "40", "20", "false", "cross_encoder_rerank")
        for cell in cells:
            assert cell in appendix

    def test_appendix_uses_unknown_for_missing_keys(self) -> None:
        result = _make_result(Strategy.ARCHEX_QUERY_TASK_AWARE)
        result.provenance = {"modality": "nl_to_pl"}  # routing/caps/skipped absent
        md = format_markdown(_make_report([result]))
        appendix = md.split("### Task-aware policy", 1)[1]
        assert "nl_to_pl" in appendix
        # routing_decision / caps / skipped_steps fall back to "unknown".
        assert "unknown" in appendix

    def test_appendix_omitted_for_empty_provenance(self) -> None:
        result = _make_result(Strategy.ARCHEX_QUERY_TASK_AWARE)
        # Default provenance is empty; the appendix has nothing to show.
        md = format_markdown(_make_report([result]))
        assert "### Task-aware policy" not in md

    def test_appendix_absent_without_task_aware(self) -> None:
        md = format_markdown(_make_report([_make_result(Strategy.ARCHEX_QUERY)]))
        assert "### Task-aware policy" not in md

    def test_strategy_comparison_compares_lane_against_archex_query(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY), _task_aware_result()])
        md = format_strategy_comparison([report])
        # The lane appears alongside archex_query in the comparison table and
        # carries its compact policy appendix.
        assert Strategy.ARCHEX_QUERY.value in md
        assert Strategy.ARCHEX_QUERY_TASK_AWARE.value in md
        assert "### Task-aware policy" in md


def _compression_result() -> BenchmarkResult:
    return _make_result(Strategy.ARCHEX_QUERY_COMPRESSED, tokens=600).model_copy(
        update={
            "bundle_tokens_uncompressed": 1000,
            "bundle_tokens_compressed": 400,
            "bundle_compression_ratio": 0.4,
            "required_context_compressed_tokens": 0,
            "required_context_passthrough_tokens": 250,
            "compression_hidden_required_region_count": 0,
            "token_efficiency_with_compression_and_completion": 0.72,
        }
    )


class TestCompressionReporting:
    def test_appendix_rendered_for_compressed_strategy(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY), _compression_result()])
        md = format_markdown(report)
        assert "### Post-Retrieval Compression" in md
        assert "archex_query_compressed" in md
        assert "0.400" in md  # compression ratio
        assert "Efficiency (compressed+completion)" in md

    def test_appendix_absent_without_compression(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY)])
        assert "Post-Retrieval Compression" not in format_markdown(report)

    def test_summary_rendered_for_compressed_strategy(self) -> None:
        summary = format_summary([_make_report([_compression_result()])])
        assert "## Post-Retrieval Compression" in summary
        assert "Hidden Required" in summary
        assert "Compression cannot make incomplete context complete." in summary

    def test_summary_separates_retrieval_miss_from_compression_hiding(self) -> None:
        # Hidden Required is distinct from the missing-required-file appendix.
        summary = format_summary([_make_report([_compression_result()])])
        assert "retrieval misses are reported separately" in summary

    def test_summary_absent_without_compression(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY)])
        assert "Post-Retrieval Compression" not in format_summary([report])


def _packed_result() -> BenchmarkResult:
    result = _make_result(Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED, tokens=600).model_copy(
        update={
            "bundle_tokens_uncompressed": 1000,
            "bundle_tokens_compressed": 600,
            "bundle_compression_ratio": 0.6,
            "token_efficiency_with_completion": 0.74,
            "token_efficiency_with_compression_and_completion": 0.74,
            "relevance_per_1k_tokens": 1.8,
            "packed_relevance_per_1k_tokens": 2.5,
            "packing_included_regions": 4,
            "packing_compressed_regions": 1,
            "packing_elided_regions": 1,
            "packing_skipped_regions": 2,
        }
    )
    result.provenance = {
        "budget_tier": "standard",
        "include_count": "4",
        "compress_count": "1",
        "elide_count": "1",
        "skip_count": "2",
        "relevance_per_1k_tokens": "2.5000",
    }
    return result


class TestPackingReporting:
    def test_appendix_rendered_for_packed_strategy(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY), _packed_result()])
        md = format_markdown(report)
        assert "### Efficiency-Aware Packing" in md
        assert "archex_query_efficiency_packed" in md
        assert "2.50" in md  # packed relevance per 1k tokens
        assert "Include | Compress | Elide | Skip" in md

    def test_appendix_absent_without_packing(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY)])
        assert "Efficiency-Aware Packing" not in format_markdown(report)

    def test_lane_comparison_compares_normal_compressed_and_packed(self) -> None:
        report = _make_report(
            [_make_result(Strategy.ARCHEX_QUERY), _compression_result(), _packed_result()]
        )
        summary = format_summary([report])
        assert "## Packing Lane Comparison" in summary
        assert "normal packing" in summary
        assert "compressed packing" in summary
        assert "efficiency-aware packing" in summary
        assert "archex_query_efficiency_packed" in summary

    def test_lane_comparison_uses_post_reduction_efficiency(self) -> None:
        # The compressed lane sets only token_efficiency_with_compression_and_completion;
        # the comparison must report that (0.720), not its pre-compression
        # token_efficiency_with_completion (0.0), so the three lanes stay comparable.
        report = _make_report(
            [_make_result(Strategy.ARCHEX_QUERY), _compression_result(), _packed_result()]
        )
        section = format_summary([report]).split("## Packing Lane Comparison", 1)[1]
        assert "0.720" in section  # compressed lane, post-compression efficiency
        assert "0.740" in section  # efficiency-packed lane, post-packing efficiency

    def test_lane_comparison_absent_without_packed_lane(self) -> None:
        # The efficiency-packed lane is the section's subject; without it the
        # comparison is omitted so unrelated reports are unaffected.
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY), _compression_result()])
        assert "Packing Lane Comparison" not in format_summary([report])

    def test_appendix_rendered_in_strategy_comparison(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY), _packed_result()])
        md = format_strategy_comparison([report])
        assert "### Efficiency-Aware Packing" in md


def _dual_transform_result() -> BenchmarkResult:
    result = _make_result(Strategy.ARCHEX_QUERY_DUAL_TRANSFORM, tokens=600)
    result.provenance = {
        "subquery_structural": "AuthManager validate_token",
        "subquery_behavioral": "why does login fail",
        "fused_included": "5",
        "fused_candidates": "9",
    }
    return result


def _graph_multihop_result() -> BenchmarkResult:
    result = _make_result(Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP, tokens=700)
    result.provenance = {
        "hop_cap": "2",
        "frontier_cap": "8",
        "files_expanded": "3",
        "frontier_cuts": "1",
        "budget_cuts": "2",
        "suppressed_low_confidence": "4",
    }
    return result


def _bounded_rerank_result() -> BenchmarkResult:
    result = _make_result(Strategy.ARCHEX_QUERY_BOUNDED_RERANK, tokens=650)
    result.provenance = {
        "candidate_cap": "8",
        "candidates_reranked": "8",
        "candidates_total": "20",
        "cross_encoder_status": "skipped:unavailable",
        "rerank_method": "symbolic",
    }
    return result


def _summary_sidecar_result() -> BenchmarkResult:
    result = _make_result(Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR, tokens=550)
    result.provenance = {
        "sidecar": "loaded",
        "summary_first": "true",
        "entries_stale": "2",
    }
    return result


class TestAdvancedLanesReporting:
    def test_appendix_rendered_for_advanced_lane(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY), _dual_transform_result()])
        md = format_markdown(report)
        assert "### Advanced Quality Lanes" in md
        section = md.split("### Advanced Quality Lanes", 1)[1]
        assert "archex_query_dual_transform" in section
        # The lane note surfaces the fusion provenance.
        assert "fused 5/9 candidates" in section
        # The section states the benchmark-only / no-default-change framing.
        assert "never changes the product default" in section

    def test_appendix_notes_each_lane(self) -> None:
        report = _make_report(
            [
                _dual_transform_result(),
                _bounded_rerank_result(),
                _summary_sidecar_result(),
                _graph_multihop_result(),
            ]
        )
        section = format_markdown(report).split("### Advanced Quality Lanes", 1)[1]
        # Each lane's note surfaces its own provenance.
        assert "fused 5/9 candidates" in section
        assert "reranked 8/20 (ce=skipped:unavailable)" in section
        assert "summary_first=true, stale=2" in section
        assert "expanded 3" in section
        assert "frontier/budget/low-conf=1/2/4" in section

    def test_appendix_absent_without_advanced_lane(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY)])
        assert "Advanced Quality Lanes" not in format_markdown(report)

    def test_appendix_rendered_in_strategy_comparison(self) -> None:
        report = _make_report([_make_result(Strategy.ARCHEX_QUERY), _dual_transform_result()])
        md = format_strategy_comparison([report])
        assert "### Advanced Quality Lanes" in md
        assert "archex_query_dual_transform" in md
