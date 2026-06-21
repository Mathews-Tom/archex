"""Individual strategy implementations for benchmarking."""

from __future__ import annotations

import importlib.metadata
import logging
import math
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from archex.benchmark.bounded_rerank import (
    RerankCaps,
    query_signals,
    symbolic_scores,
)
from archex.benchmark.graph_multihop import (
    ExpansionAction,
    ExpansionDecision,
    GraphEdge,
    MultihopCaps,
    MultihopResult,
    graph_multihop_expand,
)
from archex.benchmark.models import (
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
    TaskCompletionResult,
)
from archex.benchmark.query_transform import transform_query
from archex.benchmark.region_metrics import (
    ReturnedRegion,
    compute_region_metrics,
)
from archex.benchmark.summary_sidecar import SummarySidecar, file_content_hash
from archex.benchmark.task_aware import DenseTrigger, TaskAwarePolicy, policy_for
from archex.cache import CacheManager
from archex.exceptions import ArchexError, ConfigError
from archex.index.churn import ChurnPriors, load_churn_priors
from archex.models import (
    ChunkerName,
    CodeChunk,
    CompressionLossRisk,
    CompressionMetadata,
    CompressionMode,
    ContextBundle,
    ContextCompletenessStatus,
    ContextOmittedEdgeReason,
    ContextReceiptEdge,
    ContextReceiptItem,
    ContextSkippedCandidate,
    ContextSkippedReason,
    EdgeConfidence,
    EdgeKind,
    IndexConfig,
    PipelineTiming,
    RankedChunk,
    RepoSource,
)
from archex.reporting import count_tokens
from archex.serve.context import CENTRALITY_MODE_PERSONALIZED_WEIGHTED
from archex.serve.modality import budget_tier, classify_query
from archex.serve.packing import (
    PackDecision,
    PackingCandidate,
    PackingSignals,
    pack_efficiently,
)

if TYPE_CHECKING:
    from archex.index.rerank import CrossEncoderReranker
    from archex.index.store import IndexStore

logger = logging.getLogger(__name__)

StrategyRunner = Callable[[BenchmarkTask, Path], BenchmarkResult]

_BENCHMARK_RETRIEVAL_OPTIONS: ContextVar[BenchmarkRetrievalOptions | None] = ContextVar(
    "benchmark_retrieval_options",
    default=None,
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "ought",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "where",
        "when",
        "why",
        "if",
        "then",
        "than",
        "but",
        "and",
        "or",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "each",
        "all",
        "any",
        "few",
        "more",
        "most",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "also",
        "as",
        "at",
        "before",
        "below",
        "between",
        "by",
        "down",
        "during",
        "for",
        "from",
        "in",
        "into",
        "of",
        "off",
        "on",
        "out",
        "over",
        "to",
        "up",
        "with",
    }
)


def _deduplicate_ranked(ranked_files: list[str]) -> list[str]:
    """Deduplicate file paths preserving first-occurrence order."""
    return list(dict.fromkeys(ranked_files))


def compute_f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision."""
    if recall + precision == 0.0:
        return 0.0
    return 2 * (recall * precision) / (recall + precision)


def compute_mrr(ranked_files: list[str], expected_files: list[str]) -> float:
    """Mean reciprocal rank: reciprocal of the rank of the first expected file found."""
    deduped = _deduplicate_ranked(ranked_files)
    expected_set = set(expected_files)
    for i, f in enumerate(deduped, 1):
        if f in expected_set:
            return 1.0 / i
    return 0.0


def compute_recall(result_files: set[str], expected_files: list[str]) -> float:
    """Fraction of expected files found in results."""
    if not expected_files:
        return 0.0
    found = sum(1 for f in expected_files if f in result_files)
    return found / len(expected_files)


def compute_precision(result_files: set[str], expected_files: list[str]) -> float:
    """Fraction of result files that are in the expected set."""
    if not result_files:
        return 0.0
    expected_set = set(expected_files)
    relevant = sum(1 for f in result_files if f in expected_set)
    return relevant / len(result_files)


def compute_ndcg(ranked_files: list[str], expected_files: list[str], k: int = 10) -> float:
    """Normalized discounted cumulative gain at k.

    Deduplicates ranked_files to prevent the same file from contributing
    relevance at multiple positions.
    """
    if not expected_files:
        return 0.0
    deduped = _deduplicate_ranked(ranked_files)
    expected_set = set(expected_files)
    # DCG
    dcg = 0.0
    for i, f in enumerate(deduped[:k]):
        rel = 1.0 if f in expected_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    # Ideal DCG
    ideal_count = min(len(expected_files), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_map(ranked_files: list[str], expected_files: list[str]) -> float:
    """Mean average precision.

    Deduplicates ranked_files to prevent the same file from inflating
    precision-at-k calculations.
    """
    if not expected_files:
        return 0.0
    deduped = _deduplicate_ranked(ranked_files)
    expected_set = set(expected_files)
    hits = 0
    sum_precision = 0.0
    for i, f in enumerate(deduped, 1):
        if f in expected_set:
            hits += 1
            sum_precision += hits / i
    if hits == 0:
        return 0.0
    return sum_precision / len(expected_files)


def count_file_tokens(repo_path: Path, files: list[str]) -> int:
    """Count tokens across a list of files relative to repo_path."""
    total = 0
    for f in files:
        full_path = repo_path / f
        if full_path.is_file():
            content = full_path.read_text(encoding="utf-8", errors="replace")
            total += count_tokens(content)
    return total


def extract_keywords(question: str, extra_keywords: list[str]) -> list[str]:
    """Extract search keywords from a question string, filtering stopwords."""
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question)
    filtered = [w.lower() for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
    for kw in extra_keywords:
        kw_lower = kw.lower()
        if kw_lower not in filtered:
            filtered.append(kw_lower)
    return filtered


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def compute_symbol_recall(result_symbols: set[str], expected_symbols: list[str]) -> float:
    """Fraction of expected symbols found in results."""
    if not expected_symbols:
        return 0.0
    found = sum(1 for s in expected_symbols if s in result_symbols)
    return found / len(expected_symbols)


def compute_token_efficiency(tokens_output: int, tokens_input: int) -> float:
    """Return higher-is-better token savings for the accessed context."""
    if tokens_input <= 0:
        return 0.0
    ratio = 1.0 - (tokens_output / tokens_input)
    return max(0.0, min(1.0, ratio))


def compute_bundle_completion_penalty(
    repo_path: Path,
    result_files: set[str],
    expected_files: list[str],
) -> tuple[int, list[str]]:
    """Return extra oracle-file tokens needed after an incomplete result bundle."""
    missing_files = [path for path in expected_files if path not in result_files]
    return count_file_tokens(repo_path, missing_files), missing_files


def compute_required_file_metrics(
    result_files: set[str],
    expected_files: list[str],
) -> tuple[float, float, float, bool, list[str], list[str]]:
    present = [path for path in expected_files if path in result_files]
    missing = [path for path in expected_files if path not in result_files]
    recall = len(present) / len(expected_files) if expected_files else 0.0
    all_present = not missing
    missed_file_rate = (len(missing) / len(expected_files)) if expected_files else 0.0
    missed_task_rate = 0.0 if all_present else 1.0
    return recall, missed_file_rate, missed_task_rate, all_present, present, missing


def compute_receipt_accuracy(
    bundle: ContextBundle | None,
    *,
    all_required_files_present: bool,
) -> bool | None:
    if bundle is None or bundle.receipt is None:
        return None
    status = bundle.receipt.context_complete
    if status == "unknown":
        return None
    predicted_complete = status == ContextCompletenessStatus.COMPLETE
    return predicted_complete == all_required_files_present


def completion_result_from_missing(missing_files: list[str]) -> TaskCompletionResult:
    return TaskCompletionResult.PASS if not missing_files else TaskCompletionResult.FAIL


def run_raw_files(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Baseline strategy: read all expected files, count tokens."""
    t0 = time.perf_counter()
    tokens = count_file_tokens(repo_path, task.expected_files)
    wall_ms = (time.perf_counter() - t0) * 1000
    required_metrics = compute_required_file_metrics(
        set(task.expected_files),
        task.expected_files,
    )
    (
        required_file_recall,
        missed_required_file_rate,
        missed_required_task_rate,
        all_required_files_present,
        present,
        missing,
    ) = required_metrics

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_FILES,
        tokens_total=tokens,
        tokens_input=tokens,
        tokens_output=tokens,
        token_efficiency=compute_token_efficiency(tokens, tokens),
        result_files=list(task.expected_files),
        required_file_recall=required_file_recall,
        missed_required_file_rate=missed_required_file_rate,
        missed_required_task_rate=missed_required_task_rate,
        all_required_files_present=all_required_files_present,
        required_files_present=present,
        required_files_missing=missing,
        post_bundle_search_turns=0,
        post_bundle_read_turns=0,
        task_completion_result=TaskCompletionResult.PASS,
        completion_preserved=True,
        token_efficiency_with_completion=compute_token_efficiency(tokens, tokens),
        tokens_raw_baseline=tokens,
        tool_calls=len(task.expected_files),
        files_accessed=len(task.expected_files),
        recall=1.0,
        precision=1.0,
        f1_score=1.0,
        mrr=1.0,
        ndcg=1.0,
        map_score=1.0,
        savings_vs_raw=0.0,
        wall_time_ms=wall_ms,
        cached=False,
        timestamp=now_iso(),
    )


def run_raw_grepped(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Grep-based strategy: search repo for keywords, read matched files."""
    t0 = time.perf_counter()
    keywords = extract_keywords(task.question, task.keywords)

    matched_files_seen: set[str] = set()
    matched_files_ordered: list[str] = []
    for kw in keywords:
        result = subprocess.run(
            [
                "grep",
                "-rl",
                "--include=*.py",
                "--include=*.ts",
                "--include=*.js",
                "--include=*.go",
                "--include=*.rs",
                "--include=*.java",
                "--include=*.kt",
                "--include=*.cs",
                "--include=*.swift",
                "-i",
                kw,
                ".",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                path = line.lstrip("./")
                if path and path not in matched_files_seen:
                    matched_files_seen.add(path)
                    matched_files_ordered.append(path)

    tokens = count_file_tokens(repo_path, matched_files_ordered)
    tokens_raw_baseline = count_file_tokens(repo_path, task.expected_files)
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(matched_files_seen, task.expected_files)
    precision = compute_precision(matched_files_seen, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(matched_files_ordered, task.expected_files)
    ndcg_val = compute_ndcg(matched_files_ordered, task.expected_files)
    map_val = compute_map(matched_files_ordered, task.expected_files)
    completion_tokens, completion_files = compute_bundle_completion_penalty(
        repo_path, matched_files_seen, task.expected_files
    )
    required_metrics = compute_required_file_metrics(
        matched_files_seen,
        task.expected_files,
    )
    (
        required_file_recall,
        missed_required_file_rate,
        missed_required_task_rate,
        all_required_files_present,
        present,
        missing,
    ) = required_metrics
    tokens_with_completion = tokens + completion_tokens

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_GREPPED,
        tokens_total=tokens,
        tokens_input=tokens,
        tokens_output=tokens,
        token_efficiency=compute_token_efficiency(tokens, tokens),
        result_files=_deduplicate_ranked(matched_files_ordered),
        required_file_recall=required_file_recall,
        missed_required_file_rate=missed_required_file_rate,
        missed_required_task_rate=missed_required_task_rate,
        all_required_files_present=all_required_files_present,
        required_files_present=present,
        required_files_missing=missing,
        post_bundle_read_turns=len(missing),
        task_completion_result=completion_result_from_missing(missing),
        bundle_completion_tokens=completion_tokens,
        bundle_completion_files=completion_files,
        token_efficiency_with_completion=compute_token_efficiency(
            tokens_with_completion, tokens + completion_tokens
        ),
        tokens_raw_baseline=tokens_raw_baseline,
        tool_calls=len(keywords),
        files_accessed=len(matched_files_seen),
        recall=recall,
        precision=precision,
        f1_score=f1,
        mrr=mrr_val,
        ndcg=ndcg_val,
        map_score=map_val,
        savings_vs_raw=0.0,
        wall_time_ms=wall_ms,
        cached=False,
        timestamp=now_iso(),
    )


_RIPGREP_INCLUDE_GLOBS = (
    "*.py",
    "*.ts",
    "*.js",
    "*.go",
    "*.rs",
    "*.java",
    "*.kt",
    "*.cs",
    "*.swift",
)
_RIPGREP_TIMEOUT_SECONDS = 30


def run_raw_ripgrep(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Ripgrep/read baseline: search keywords with rg, then read matched files."""
    t0 = time.perf_counter()
    rg_path = shutil.which("rg")
    if rg_path is None:
        raise RuntimeError("raw_ripgrep strategy requires ripgrep executable 'rg' on PATH")
    try:
        version = subprocess.run(
            [rg_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.splitlines()[0]
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("raw_ripgrep timed out while reading rg --version") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"raw_ripgrep failed to read rg --version: {message}") from exc
    keywords = extract_keywords(task.question, task.keywords)
    matched_files_seen: set[str] = set()
    matched_files_ordered: list[str] = []
    for keyword in keywords:
        command = [
            rg_path,
            "--files-with-matches",
            "--ignore-case",
            *[flag for glob in _RIPGREP_INCLUDE_GLOBS for flag in ("--glob", glob)],
            keyword,
            ".",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=_RIPGREP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"raw_ripgrep timed out after {_RIPGREP_TIMEOUT_SECONDS}s for keyword {keyword!r}"
            ) from exc
        if result.returncode == 1:
            continue
        if result.returncode != 0:
            raise RuntimeError(
                f"raw_ripgrep failed for keyword {keyword!r}: {result.stderr.strip()}"
            )
        for line in result.stdout.splitlines():
            path = line.lstrip("./")
            if path and path not in matched_files_seen:
                matched_files_seen.add(path)
                matched_files_ordered.append(path)
    tokens = count_file_tokens(repo_path, matched_files_ordered)
    tokens_raw_baseline = count_file_tokens(repo_path, task.expected_files)
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(matched_files_seen, task.expected_files)
    precision = compute_precision(matched_files_seen, task.expected_files)
    completion_tokens, completion_files = compute_bundle_completion_penalty(
        repo_path, matched_files_seen, task.expected_files
    )
    (
        required_file_recall,
        missed_required_file_rate,
        missed_required_task_rate,
        all_required_files_present,
        present,
        missing,
    ) = compute_required_file_metrics(matched_files_seen, task.expected_files)
    tokens_with_completion = tokens + completion_tokens
    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_RIPGREP,
        strategy_label="raw-ripgrep/read",
        tokens_total=tokens,
        tokens_input=tokens,
        tokens_output=tokens,
        token_efficiency=compute_token_efficiency(tokens, tokens),
        result_files=_deduplicate_ranked(matched_files_ordered),
        required_file_recall=required_file_recall,
        missed_required_file_rate=missed_required_file_rate,
        missed_required_task_rate=missed_required_task_rate,
        all_required_files_present=all_required_files_present,
        required_files_present=present,
        required_files_missing=missing,
        post_bundle_read_turns=len(missing),
        task_completion_result=completion_result_from_missing(missing),
        bundle_completion_tokens=completion_tokens,
        bundle_completion_files=completion_files,
        token_efficiency_with_completion=compute_token_efficiency(
            tokens_with_completion, tokens + completion_tokens
        ),
        tokens_raw_baseline=tokens_raw_baseline,
        tool_calls=len(keywords),
        files_accessed=len(matched_files_seen),
        recall=recall,
        precision=precision,
        f1_score=compute_f1(recall, precision),
        mrr=compute_mrr(matched_files_ordered, task.expected_files),
        ndcg=compute_ndcg(matched_files_ordered, task.expected_files),
        map_score=compute_map(matched_files_ordered, task.expected_files),
        savings_vs_raw=0.0,
        wall_time_ms=wall_ms,
        cached=False,
        timestamp=now_iso(),
        provenance={
            "rg_version": version,
            "keyword_count": str(len(keywords)),
            "include_globs": ",".join(_RIPGREP_INCLUDE_GLOBS),
            "timeout_seconds": str(_RIPGREP_TIMEOUT_SECONDS),
            "matched_file_count": str(len(matched_files_seen)),
        },
    )


class _ArchexFields:
    """Aggregated fields extracted from a ContextBundle for benchmark results."""

    __slots__ = (
        "tokens_input",
        "tokens_output",
        "token_efficiency",
        "tokens_raw_baseline",
        "symbol_recall",
        "unique_ranked_files",
        "seed_files",
        "expanded_files",
        "expansion_ratio",
        "seed_recall",
        "seed_precision",
        "expansion_eligible_seeds",
        "expansion_candidates_found",
        "expansion_import_neighbor_edges",
        "expansion_same_module_candidates",
        "expansion_hub_candidates",
        "expansion_test_candidates_skipped",
        "expansion_zero_candidate_reason",
        "expansion_reason_counts",
        "expanded_file_reasons",
        "chunker",
        "index_chunk_count",
        "mean_chunk_tokens",
    )

    chunker: ChunkerName
    index_chunk_count: int
    mean_chunk_tokens: float

    def __init__(
        self,
        *,
        tokens_input: int,
        tokens_output: int,
        token_efficiency: float,
        tokens_raw_baseline: int,
        symbol_recall: float,
        unique_ranked_files: int,
        seed_files: list[str],
        expanded_files: list[str],
        expansion_ratio: float,
        seed_recall: float,
        seed_precision: float,
        expansion_eligible_seeds: int,
        expansion_candidates_found: int,
        expansion_import_neighbor_edges: int,
        expansion_same_module_candidates: int,
        expansion_hub_candidates: int,
        expansion_test_candidates_skipped: int,
        expansion_zero_candidate_reason: str,
        expansion_reason_counts: dict[str, int],
        chunker: ChunkerName,
        index_chunk_count: int,
        mean_chunk_tokens: float,
        expanded_file_reasons: dict[str, list[str]],
    ) -> None:
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.token_efficiency = token_efficiency
        self.tokens_raw_baseline = tokens_raw_baseline
        self.symbol_recall = symbol_recall
        self.unique_ranked_files = unique_ranked_files
        self.seed_files = seed_files
        self.expanded_files = expanded_files
        self.expansion_ratio = expansion_ratio
        self.seed_recall = seed_recall
        self.seed_precision = seed_precision
        self.expansion_eligible_seeds = expansion_eligible_seeds
        self.expansion_candidates_found = expansion_candidates_found
        self.expansion_import_neighbor_edges = expansion_import_neighbor_edges
        self.expansion_same_module_candidates = expansion_same_module_candidates
        self.expansion_hub_candidates = expansion_hub_candidates
        self.expansion_test_candidates_skipped = expansion_test_candidates_skipped
        self.expansion_zero_candidate_reason = expansion_zero_candidate_reason
        self.expansion_reason_counts = expansion_reason_counts
        self.chunker = chunker
        self.index_chunk_count = index_chunk_count
        self.mean_chunk_tokens = mean_chunk_tokens
        self.expanded_file_reasons = expanded_file_reasons


def _archex_fields(
    bundle: object,
    task: BenchmarkTask,
    repo_path: Path,
) -> _ArchexFields:
    """Compute token efficiency and seed/expansion diagnostic fields."""
    from archex.models import ContextBundle

    assert isinstance(bundle, ContextBundle)
    unique_files = _deduplicate_ranked([c.chunk.file_path for c in bundle.chunks])
    tokens_input = count_file_tokens(repo_path, unique_files)
    tokens_output = bundle.token_count
    token_efficiency = compute_token_efficiency(tokens_output, tokens_input)
    tokens_raw_baseline = count_file_tokens(repo_path, task.expected_files)
    result_symbols = {c.chunk.symbol_name for c in bundle.chunks if c.chunk.symbol_name}
    symbol_recall = compute_symbol_recall(result_symbols, task.expected_symbols)

    # Seed vs expansion: candidates_found is a chunk count, while
    # seed_files_found is the file boundary for graph expansion diagnostics.
    meta = bundle.retrieval_metadata
    chunker = meta.chunker
    index_chunk_count = meta.index_chunk_count
    mean_chunk_tokens = meta.mean_chunk_tokens
    if meta.seed_file_paths or meta.expanded_file_paths:
        seed_files = list(meta.seed_file_paths)
        expanded_files = list(meta.expanded_file_paths)
        expansion_ratio = (
            len(expanded_files) / len(seed_files) if seed_files else float(bool(expanded_files))
        )
        seed_recall_val = compute_recall(set(seed_files), task.expected_files)
        seed_precision_val = compute_precision(set(seed_files), task.expected_files)
        return _ArchexFields(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            token_efficiency=token_efficiency,
            tokens_raw_baseline=tokens_raw_baseline,
            symbol_recall=symbol_recall,
            unique_ranked_files=len(unique_files),
            seed_files=seed_files,
            expanded_files=expanded_files,
            expansion_ratio=expansion_ratio,
            seed_recall=seed_recall_val,
            seed_precision=seed_precision_val,
            expansion_eligible_seeds=meta.expansion_eligible_seeds,
            expansion_candidates_found=meta.expansion_candidates_found,
            expansion_import_neighbor_edges=meta.expansion_import_neighbor_edges,
            expansion_same_module_candidates=meta.expansion_same_module_candidates,
            expansion_hub_candidates=meta.expansion_hub_candidates,
            expansion_test_candidates_skipped=meta.expansion_test_candidates_skipped,
            expansion_zero_candidate_reason=meta.expansion_zero_candidate_reason,
            expansion_reason_counts=dict(meta.expansion_reason_counts),
            expanded_file_reasons={
                path: list(reasons) for path, reasons in meta.expanded_file_reasons.items()
            },
            chunker=chunker,
            index_chunk_count=index_chunk_count,
            mean_chunk_tokens=mean_chunk_tokens,
        )

    seed_file_count = meta.seed_files_found

    # Build seed file list from first `seed_file_count` unique chunk file paths.
    all_chunk_files = [c.chunk.file_path for c in bundle.chunks]
    seen: set[str] = set()
    seed_files: list[str] = []
    expanded_files: list[str] = []
    # Chunks are ordered by score; unique file order is the observable boundary
    # preserved in benchmark results.
    seed_file_set: set[str] = set()
    for fp in all_chunk_files:
        if fp not in seen:
            seen.add(fp)
            if len(seed_file_set) < seed_file_count:
                seed_files.append(fp)
                seed_file_set.add(fp)
            else:
                expanded_files.append(fp)

    expansion_ratio = (
        meta.expansion_files_added / len(seed_files)
        if seed_files
        else float(meta.expansion_files_added > 0)
    )
    seed_recall_val = compute_recall(set(seed_files), task.expected_files)
    seed_precision_val = compute_precision(set(seed_files), task.expected_files)

    return _ArchexFields(
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        token_efficiency=token_efficiency,
        tokens_raw_baseline=tokens_raw_baseline,
        symbol_recall=symbol_recall,
        unique_ranked_files=len(unique_files),
        seed_files=seed_files,
        expanded_files=expanded_files,
        expansion_ratio=expansion_ratio,
        seed_recall=seed_recall_val,
        seed_precision=seed_precision_val,
        expansion_eligible_seeds=meta.expansion_eligible_seeds,
        expansion_candidates_found=meta.expansion_candidates_found,
        expansion_import_neighbor_edges=meta.expansion_import_neighbor_edges,
        expansion_same_module_candidates=meta.expansion_same_module_candidates,
        expansion_hub_candidates=meta.expansion_hub_candidates,
        expansion_test_candidates_skipped=meta.expansion_test_candidates_skipped,
        expansion_zero_candidate_reason=meta.expansion_zero_candidate_reason,
        expansion_reason_counts=dict(meta.expansion_reason_counts),
        expanded_file_reasons={
            path: list(reasons) for path, reasons in meta.expanded_file_reasons.items()
        },
        chunker=chunker,
        index_chunk_count=index_chunk_count,
        mean_chunk_tokens=mean_chunk_tokens,
    )


def _cache_state(timing: PipelineTiming) -> str:
    return "warm" if timing.cached else "cold"


def current_benchmark_retrieval_options() -> BenchmarkRetrievalOptions:
    return _BENCHMARK_RETRIEVAL_OPTIONS.get() or BenchmarkRetrievalOptions()


_VECTOR_CHUNKER_STRATEGIES = frozenset(
    {
        Strategy.ARCHEX_QUERY_VECTOR,
        Strategy.SURROGATE_VECTOR,
        Strategy.ARCHEX_QUERY_FUSION,
        Strategy.ARCHEX_QUERY_FUSION_RERANK,
        Strategy.ARCHEX_QUERY_HYBRID,
        Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT,
        Strategy.CROSS_LAYER_FUSION,
    }
)


def _chunker_for_strategy(
    strategy: Strategy,
    options: BenchmarkRetrievalOptions,
) -> ChunkerName:
    if strategy in _VECTOR_CHUNKER_STRATEGIES:
        return options.vector_chunker or options.chunker
    return options.bm25_chunker or options.chunker


def set_benchmark_retrieval_options(
    options: BenchmarkRetrievalOptions,
) -> Token[BenchmarkRetrievalOptions | None]:
    return _BENCHMARK_RETRIEVAL_OPTIONS.set(options)


def reset_benchmark_retrieval_options(token: Token[BenchmarkRetrievalOptions | None]) -> None:
    _BENCHMARK_RETRIEVAL_OPTIONS.reset(token)


def _embedder_cache_identity(embedder: str) -> str:
    if embedder != "jina-v2":
        return embedder
    from archex.index.embeddings import (
        JINA_BERT_CODE_REVISION,
        JINA_V2_MAX_SEQ_LENGTH,
        JINA_V2_MODEL_REVISION,
    )

    return (
        f"{embedder}@{JINA_V2_MODEL_REVISION}"
        f"+code={JINA_BERT_CODE_REVISION}"
        f"+max_seq={JINA_V2_MAX_SEQ_LENGTH}"
    )


def _retrieval_cache_suffix(
    options: BenchmarkRetrievalOptions,
    *,
    strategy: Strategy | None = None,
) -> str:
    enabled: list[str] = [f"embedder={_embedder_cache_identity(options.embedder)}"]
    if strategy is None:
        bm25_chunker = _chunker_for_strategy(Strategy.ARCHEX_QUERY, options)
        vector_chunker = _chunker_for_strategy(Strategy.ARCHEX_QUERY_FUSION, options)
        if bm25_chunker == vector_chunker:
            enabled.append(f"chunker={bm25_chunker}")
        else:
            enabled.append(f"bm25-chunker={bm25_chunker}")
            enabled.append(f"vector-chunker={vector_chunker}")
    else:
        enabled.append(f"chunker={_chunker_for_strategy(strategy, options)}")
    if strategy is Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT:
        enabled.append("quantize=4bit")
    if options.splade:
        enabled.append("splade")
    if options.module_prefilter:
        enabled.append("module-prefilter")
    return "+".join(enabled)


def _corpus_cache_suffix(task: BenchmarkTask) -> str:
    if not task.include_paths:
        return ""
    return "scope=" + "|".join(sorted(task.include_paths))


def benchmark_index_config(
    index_config: IndexConfig,
    *,
    strategy: Strategy | None = None,
) -> IndexConfig:
    options = current_benchmark_retrieval_options()
    strategy_chunker = (
        _chunker_for_strategy(strategy, options)
        if strategy is not None
        else (options.vector_chunker or options.chunker)
        if index_config.vector
        else (options.bm25_chunker or options.chunker)
    )
    updates: dict[str, bool | str | int] = {"chunker": strategy_chunker}
    if index_config.vector:
        updates["embedder"] = options.embedder
    if options.splade:
        updates["splade"] = True
    if options.module_prefilter and index_config.bm25:
        updates["module_prefilter"] = True
    if index_config.rerank and options.rerank_model is not None:
        updates["rerank_model"] = options.rerank_model
    if options.allow_remote_code:
        updates["allow_remote_code"] = True
    if index_config.rerank and options.rerank_candidate_limit is not None:
        updates["rerank_candidate_limit"] = options.rerank_candidate_limit
    if not updates:
        return index_config
    return index_config.model_copy(update=updates)


def benchmark_cache_enabled(default: bool) -> bool:
    options = current_benchmark_retrieval_options()
    return default or options.splade or options.module_prefilter


def benchmark_repo_source(
    task: BenchmarkTask,
    repo_path: Path,
    strategy: Strategy | None = None,
) -> RepoSource:
    commit = task.commit or CacheManager.git_head(str(repo_path))
    if not commit:
        raise ConfigError(
            f"Benchmark task {task.task_id!r} has no commit and {repo_path} has no git HEAD"
        )
    stable_identity = f"{task.repo}@{commit}"
    suffixes = [
        suffix
        for suffix in (
            _corpus_cache_suffix(task),
            _retrieval_cache_suffix(
                current_benchmark_retrieval_options(),
                strategy=strategy,
            ),
        )
        if suffix
    ]
    if suffixes:
        stable_identity = f"{stable_identity}#{'+'.join(suffixes)}"
    return RepoSource(
        local_path=str(repo_path),
        stable_identity=stable_identity,
    )


_FRESHNESS_MARKER = "archex_freshness_probe"


def _freshness_edit_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return f"\n\ndef {_FRESHNESS_MARKER}():\n    return 'freshness-probe'\n"
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return f"\n\nexport const {_FRESHNESS_MARKER} = 'freshness-probe';\n"
    if suffix == ".go":
        return f'\n\nfunc {_FRESHNESS_MARKER}() string {{ return "freshness-probe" }}\n'
    if suffix == ".rs":
        return f'\n\nfn {_FRESHNESS_MARKER}() -> &\'static str {{ "freshness-probe" }}\n'
    return f"\n\n{_FRESHNESS_MARKER} freshness-probe\n"


def _freshness_target(task: BenchmarkTask, repo_path: Path) -> Path | None:
    for expected in task.expected_files:
        candidate = repo_path / expected
        if candidate.is_file():
            return candidate
    return None


def measure_archex_freshness(task: BenchmarkTask, repo_path: Path) -> tuple[float, bool]:
    """Measure edit-to-correct-result latency on an isolated repo copy."""
    from archex.api import query
    from archex.models import Config

    workdir = Path(tempfile.mkdtemp(prefix="archex-freshness-"))
    try:
        target = workdir / repo_path.name
        shutil.copytree(repo_path, target)
        edit_target = _freshness_target(task, target)
        if edit_target is None:
            return (0.0, False)

        source = benchmark_repo_source(task, target, strategy=Strategy.ARCHEX_QUERY)
        index_config = benchmark_index_config(
            IndexConfig(vector=False),
            strategy=Strategy.ARCHEX_QUERY,
        )
        config = Config(cache=True, languages=task.languages, cache_dir=str(workdir / "cache"))
        query(source, _FRESHNESS_MARKER, config=config, index_config=index_config)
        with edit_target.open("a", encoding="utf-8") as handle:
            handle.write(_freshness_edit_text(edit_target))
        started = time.perf_counter()
        bundle = query(
            source,
            _FRESHNESS_MARKER,
            config=config,
            index_config=index_config,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        found = any(_FRESHNESS_MARKER in chunk.chunk.content for chunk in bundle.chunks)
        return (latency_ms, found)
    except (ArchexError, OSError, ValueError):
        return (0.0, False)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _bundle_returned_regions(bundle: ContextBundle) -> list[ReturnedRegion]:
    """Build ranked returned regions from a context bundle, preserving order."""
    return [
        ReturnedRegion(
            path=ranked.chunk.file_path,
            start_line=ranked.chunk.start_line,
            end_line=ranked.chunk.end_line,
            symbol=ranked.chunk.symbol_name,
            tokens=count_tokens(ranked.chunk.content),
        )
        for ranked in bundle.chunks
    ]


def _region_result_fields(
    bundle: ContextBundle,
    task: BenchmarkTask,
) -> dict[str, float | int | None]:
    """Region/line/ranking/context-efficiency result fields, empty without labels."""
    metrics = compute_region_metrics(_bundle_returned_regions(bundle), task.expected_regions)
    return metrics.as_result_fields() if metrics is not None else {}


def _query_bundle(
    task: BenchmarkTask,
    repo_path: Path,
    *,
    strategy: Strategy,
    index_config: IndexConfig,
    cache: bool,
    centrality_mode: str = "global",
    churn_priors: ChurnPriors | None = None,
) -> tuple[ContextBundle, IndexConfig, PipelineTiming]:
    """Run the query pipeline once; return the bundle and effective index config."""
    from archex.api import query
    from archex.models import Config

    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path, strategy=strategy)
    config = Config(cache=cache, languages=task.languages)
    effective_config = benchmark_index_config(index_config, strategy=strategy)
    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
        config=config,
        index_config=effective_config,
        timing=timing,
        centrality_mode=centrality_mode,
        churn_priors=churn_priors,
    )
    return bundle, effective_config, timing


def _assemble_query_result(
    task: BenchmarkTask,
    repo_path: Path,
    *,
    strategy: Strategy,
    index_config: IndexConfig,
    bundle: ContextBundle,
    timing: PipelineTiming,
    wall_ms: float,
    include_completion: bool = True,
    measure_freshness: bool = False,
) -> BenchmarkResult:
    """Build a BenchmarkResult from an already-retrieved bundle."""
    ranked_files = [chunk.chunk.file_path for chunk in bundle.chunks]
    unique_ranked = _deduplicate_ranked(ranked_files)
    result_files = set(unique_ranked)
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    required_metrics = compute_required_file_metrics(
        result_files,
        task.expected_files,
    )
    (
        required_file_recall,
        missed_required_file_rate,
        missed_required_task_rate,
        all_required_files_present,
        present,
        missing,
    ) = required_metrics
    af = _archex_fields(bundle, task, repo_path)
    result_fields: dict[str, Any] = {
        "task_id": task.task_id,
        "strategy": strategy,
        "tokens_total": bundle.token_count,
        "tokens_input": af.tokens_input,
        "tokens_output": af.tokens_output,
        "token_efficiency": af.token_efficiency,
        "tokens_raw_baseline": af.tokens_raw_baseline,
        "symbol_recall": af.symbol_recall,
        "tool_calls": 1,
        "files_accessed": len(result_files),
        "recall": recall,
        "precision": precision,
        "f1_score": compute_f1(recall, precision),
        "mrr": compute_mrr(ranked_files, task.expected_files),
        "ndcg": compute_ndcg(ranked_files, task.expected_files),
        "map_score": compute_map(ranked_files, task.expected_files),
        "savings_vs_raw": 0.0,
        "wall_time_ms": wall_ms,
        "cached": timing.cached,
        "timing": timing,
        "timestamp": now_iso(),
        "unique_ranked_files": af.unique_ranked_files,
        "seed_files": af.seed_files,
        "expanded_files": af.expanded_files,
        "expansion_ratio": af.expansion_ratio,
        "seed_recall": af.seed_recall,
        "seed_precision": af.seed_precision,
        "expansion_eligible_seeds": af.expansion_eligible_seeds,
        "expansion_candidates_found": af.expansion_candidates_found,
        "expansion_import_neighbor_edges": af.expansion_import_neighbor_edges,
        "expansion_same_module_candidates": af.expansion_same_module_candidates,
        "expansion_hub_candidates": af.expansion_hub_candidates,
        "expansion_test_candidates_skipped": af.expansion_test_candidates_skipped,
        "expansion_zero_candidate_reason": af.expansion_zero_candidate_reason,
        "expansion_reason_counts": af.expansion_reason_counts,
        "expanded_file_reasons": af.expanded_file_reasons,
        "result_files": unique_ranked,
        "required_file_recall": required_file_recall,
        "missed_required_file_rate": missed_required_file_rate,
        "all_required_files_present": all_required_files_present,
        "missed_required_task_rate": missed_required_task_rate,
        "required_files_present": present,
        "required_files_missing": missing,
        "receipt_accuracy": compute_receipt_accuracy(
            bundle,
            all_required_files_present=all_required_files_present,
        ),
        "chunker": af.chunker,
        "index_chunk_count": af.index_chunk_count,
        "mean_chunk_tokens": af.mean_chunk_tokens,
        "category": task.category,
        "family": task.family,
        "vector_mode": index_config.vector_mode,
        "surrogate_version": index_config.surrogate_version,
        "cache_state": _cache_state(timing),
    }
    if strategy == Strategy.ARCHEX_QUERY_PPR:
        meta = bundle.retrieval_metadata
        result_fields["provenance"] = {
            "centrality_variant": meta.centrality_variant,
            "centrality_personalized": str(meta.centrality_personalized).lower(),
            "centrality_fallback_reason": meta.centrality_fallback_reason,
            "centrality_latency_ms": f"{meta.centrality_latency_ms:.3f}",
            "centrality_subgraph_nodes": str(meta.centrality_subgraph_nodes),
            "centrality_subgraph_edges": str(meta.centrality_subgraph_edges),
        }
    if strategy == Strategy.ARCHEX_QUERY_CHURN:
        meta = bundle.retrieval_metadata
        applied = meta.churn_priors_applied
        result_fields["provenance"] = {
            "churn_source": meta.churn_source,
            "churn_intent_gated": str(meta.churn_intent_gated).lower(),
            "churn_files_boosted": str(len(applied)),
            "churn_max_multiplier": f"{max(applied.values(), default=1.0):.4f}",
        }
    if include_completion:
        completion_tokens, completion_files = compute_bundle_completion_penalty(
            repo_path, result_files, task.expected_files
        )
        result_fields.update(
            {
                "post_bundle_read_turns": len(completion_files),
                "task_completion_result": completion_result_from_missing(completion_files),
                "bundle_completion_tokens": completion_tokens,
                "bundle_completion_files": completion_files,
                "token_efficiency_with_completion": compute_token_efficiency(
                    af.tokens_output + completion_tokens,
                    af.tokens_input + completion_tokens,
                ),
            }
        )
    if measure_freshness:
        freshness_measured = current_benchmark_retrieval_options().freshness
        if freshness_measured:
            freshness_latency_ms, freshness_correct = measure_archex_freshness(task, repo_path)
        else:
            freshness_latency_ms, freshness_correct = (0.0, False)
        result_fields.update(
            {
                "freshness_latency_ms": freshness_latency_ms,
                "freshness_measured": freshness_measured,
                "freshness_correct": freshness_correct,
            }
        )
    result_fields.update(_region_result_fields(bundle, task))
    return BenchmarkResult(**result_fields)


def _run_query_strategy(
    task: BenchmarkTask,
    repo_path: Path,
    *,
    strategy: Strategy,
    index_config: IndexConfig,
    cache: bool,
    include_completion: bool = True,
    measure_freshness: bool = False,
    centrality_mode: str = "global",
    churn_priors: ChurnPriors | None = None,
) -> BenchmarkResult:
    t0 = time.perf_counter()
    bundle, effective_config, timing = _query_bundle(
        task,
        repo_path,
        strategy=strategy,
        index_config=index_config,
        cache=cache,
        centrality_mode=centrality_mode,
        churn_priors=churn_priors,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Strategy %s for %s: cached=%s, wall_time=%.1fms",
        strategy.value,
        task.task_id,
        timing.cached,
        wall_ms,
    )
    return _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=bundle,
        timing=timing,
        wall_ms=wall_ms,
        include_completion=include_completion,
        measure_freshness=measure_freshness,
    )


def run_archex_query(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """archex query strategy: use BM25-based retrieval."""
    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY,
        index_config=IndexConfig(vector=False),
        cache=benchmark_cache_enabled(default=False),
        include_completion=True,
        measure_freshness=True,
    )


def run_archex_query_ppr(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: BM25 retrieval with personalized structural centrality."""
    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_PPR,
        index_config=IndexConfig(vector=False),
        cache=benchmark_cache_enabled(default=False),
        include_completion=True,
        measure_freshness=True,
        centrality_mode=CENTRALITY_MODE_PERSONALIZED_WEIGHTED,
    )


# Operators may check a precomputed per-file churn fixture into the cloned repo
# at this path to drive the lane deterministically on shallow benchmark clones.
_CHURN_FIXTURE_RELPATH = ".archex/churn.json"


def run_archex_query_churn(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: BM25 retrieval with an intent-gated recency/churn prior.

    The prior is read from a checked-in churn fixture (``.archex/churn.json`` in
    the cloned repo) when present, else from the repo's real git history, else a
    neutral fallback. On a shallow benchmark clone with no fixture the prior is
    neutral, so this lane's ranking is identical to ``archex_query``.
    """
    churn_priors = load_churn_priors(repo_path, fixture_path=repo_path / _CHURN_FIXTURE_RELPATH)
    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_CHURN,
        index_config=IndexConfig(vector=False),
        cache=benchmark_cache_enabled(default=False),
        include_completion=True,
        measure_freshness=True,
        churn_priors=churn_priors,
    )


_PASSTHROUGH_SCORE_FRACTION = 0.6


def _passthrough_required(
    rank_index: int,
    ranked: RankedChunk,
    *,
    seed_paths: set[str],
    expanded_paths: set[str],
    top_score: float,
) -> bool:
    """Whether a region is required/direct/high-confidence and must not compress.

    Graph-frontier expansions are the only regions eligible for compression. The
    top hit, seed/direct retrievals, and any region scoring within
    ``_PASSTHROUGH_SCORE_FRACTION`` of the best score are protected. Uses only
    intrinsic retrieval signals — never benchmark ground truth.
    """
    path = ranked.chunk.file_path
    if path in expanded_paths and path not in seed_paths:
        return False
    if rank_index == 0:
        return True
    if path in seed_paths:
        return True
    return top_score > 0.0 and ranked.final_score >= top_score * _PASSTHROUGH_SCORE_FRACTION


@dataclass
class _BundleCompression:
    """Compressed bundle plus the benchmark result fields it produces."""

    bundle: ContextBundle
    result_fields: dict[str, float | int]
    provenance: dict[str, str]


def _compress_bundle(bundle: ContextBundle, *, question: str) -> _BundleCompression:
    """Apply deterministic compression after assembly, preserving receipts.

    Retrieval is untouched: this runs on an already-assembled bundle. Required,
    direct, and high-confidence regions pass through; only lower-confidence
    frontier regions are compressed. The original handle and hash for every
    compressed region stay retrievable through the receipt and metadata.
    """
    from archex.receipt import region_content_hash
    from archex.scout import chunk_handle
    from archex.serve.compression import compress_region
    from archex.serve.intent import QueryIntent, classify_intent

    intent = classify_intent(question)
    protect_code = intent == QueryIntent.DEBUGGING
    meta = bundle.retrieval_metadata
    seed_paths = set(meta.seed_file_paths)
    expanded_paths = set(meta.expanded_file_paths)
    top_score = max((rc.final_score for rc in bundle.chunks), default=0.0)

    compressed = bundle.model_copy(deep=True)
    receipt_items = (
        {item.handle: item for item in compressed.receipt.returned_context}
        if compressed.receipt is not None
        else {}
    )

    uncompressed_tokens = 0
    compressed_tokens = 0
    passthrough_required_tokens = 0
    compressed_required_tokens = 0
    hidden_required = 0
    compressed_regions = 0
    passthrough_regions = 0

    for index, ranked in enumerate(compressed.chunks):
        chunk = ranked.chunk
        handle = chunk_handle(chunk.id)
        original = chunk.content
        original_tokens = count_tokens(original)
        uncompressed_tokens += original_tokens
        required = _passthrough_required(
            index,
            ranked,
            seed_paths=seed_paths,
            expanded_paths=expanded_paths,
            top_score=top_score,
        )
        outcome = compress_region(
            original,
            language=chunk.language,
            fetch_original_handle=handle,
            required=required,
            protect_code=protect_code,
        )
        if outcome is None:
            compressed_tokens += original_tokens
            continue
        new_tokens = count_tokens(outcome.content)
        compressed_tokens += new_tokens
        ratio = (new_tokens / original_tokens) if original_tokens else 1.0
        chunk.content = outcome.content
        chunk.token_count = new_tokens
        item = receipt_items.get(handle)
        if item is not None:
            item.compression = CompressionMetadata(
                compression_mode=outcome.mode,
                original_tokens=original_tokens,
                compressed_tokens=new_tokens,
                compression_ratio=ratio,
                original_content_hash=region_content_hash(
                    chunk.file_path, chunk.start_line, chunk.end_line, original
                ),
                compressed_content_hash=region_content_hash(
                    chunk.file_path, chunk.start_line, chunk.end_line, outcome.content
                ),
                fetch_original_handle=handle,
                compression_loss_risk=outcome.loss_risk,
            )
        if outcome.mode is CompressionMode.PASSTHROUGH_REQUIRED:
            passthrough_required_tokens += new_tokens
            passthrough_regions += 1
        else:
            compressed_regions += 1
            if required:
                # Invariant guard: required regions never reach this branch.
                compressed_required_tokens += new_tokens
                hidden_required += 1

    compressed.token_count = compressed_tokens
    bundle_ratio = (compressed_tokens / uncompressed_tokens) if uncompressed_tokens else 1.0
    result_fields: dict[str, float | int] = {
        "bundle_tokens_uncompressed": uncompressed_tokens,
        "bundle_tokens_compressed": compressed_tokens,
        "bundle_compression_ratio": bundle_ratio,
        "required_context_compressed_tokens": compressed_required_tokens,
        "required_context_passthrough_tokens": passthrough_required_tokens,
        "compression_hidden_required_region_count": hidden_required,
    }
    provenance = {
        "query_intent": intent.value,
        "protect_code": str(protect_code).lower(),
        "compressed_region_count": str(compressed_regions),
        "passthrough_region_count": str(passthrough_regions),
        "bundle_compression_ratio": f"{bundle_ratio:.4f}",
    }
    return _BundleCompression(bundle=compressed, result_fields=result_fields, provenance=provenance)


def run_archex_query_compressed(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: archex query retrieval, then deterministic compression.

    Runs the exact ``archex_query`` retrieval and packing path so retrieval
    metrics stay attributable to the uncompressed set, then compresses the
    assembled bundle. Compression metrics describe the compressed output.
    Required/direct/high-confidence context passes through; the product default
    is unchanged.
    """
    strategy = Strategy.ARCHEX_QUERY_COMPRESSED
    t0 = time.perf_counter()
    bundle, effective_config, timing = _query_bundle(
        task,
        repo_path,
        strategy=strategy,
        index_config=IndexConfig(vector=False),
        cache=benchmark_cache_enabled(default=False),
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    result = _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=bundle,
        timing=timing,
        wall_ms=wall_ms,
        include_completion=True,
        measure_freshness=False,
    )
    compression = _compress_bundle(bundle, question=task.question)
    fields = dict(compression.result_fields)
    # Completion penalty is identical to the uncompressed lane: compression
    # cannot make incomplete retrieval complete. Scale measured output tokens by
    # the region compression ratio so non-region overhead stays comparable.
    ratio = float(compression.result_fields["bundle_compression_ratio"])
    completion = result.bundle_completion_tokens
    compressed_output = round(result.tokens_output * ratio)
    fields["token_efficiency_with_compression_and_completion"] = compute_token_efficiency(
        compressed_output + completion, result.tokens_input + completion
    )
    result = result.model_copy(update=fields)
    result.provenance = compression.provenance
    logger.info(
        "Strategy %s for %s: ratio=%.4f compressed=%s passthrough=%s wall=%.1fms",
        strategy.value,
        task.task_id,
        ratio,
        compression.provenance["compressed_region_count"],
        compression.provenance["passthrough_region_count"],
        wall_ms,
    )
    return result


# Loss risk recorded for a region whose body is dropped to a fetch anchor. The
# original stays retrievable via the handle and original hash, but the displayed
# text no longer carries the body, so it is not a low-risk representation.
_ELIDE_ANCHOR_RISK = CompressionLossRisk.MEDIUM


def _elide_anchor(original_tokens: int, handle: str) -> str:
    return (
        f"... [archex packed: {original_tokens}-token region elided; fetch original: {handle}] ..."
    )


def _shape_packed_region(
    chunk: CodeChunk,
    item: ContextReceiptItem | None,
    *,
    original: str,
    new_content: str,
    mode: CompressionMode,
    loss_risk: CompressionLossRisk,
) -> int:
    """Rewrite a region's content and record compression provenance; return new tokens."""
    from archex.receipt import region_content_hash

    original_tokens = count_tokens(original)
    new_tokens = count_tokens(new_content)
    chunk.content = new_content
    chunk.token_count = new_tokens
    if item is not None:
        item.compression = CompressionMetadata(
            compression_mode=mode,
            original_tokens=original_tokens,
            compressed_tokens=new_tokens,
            compression_ratio=(new_tokens / original_tokens) if original_tokens else 1.0,
            original_content_hash=region_content_hash(
                chunk.file_path, chunk.start_line, chunk.end_line, original
            ),
            compressed_content_hash=region_content_hash(
                chunk.file_path, chunk.start_line, chunk.end_line, new_content
            ),
            fetch_original_handle=item.handle,
            compression_loss_risk=loss_risk,
        )
    return new_tokens


@dataclass
class _BundlePacking:
    """Packed bundle plus the benchmark result fields and provenance it produces."""

    bundle: ContextBundle
    result_fields: dict[str, float | int | None]
    provenance: dict[str, str]


def _pack_bundle(bundle: ContextBundle, *, question: str) -> _BundlePacking:
    """Re-pack an assembled bundle for relevance-per-token within the same budget.

    Retrieval is untouched: the bundle's chunks are the candidate pool. The
    efficiency-aware packer preserves direct/high-confidence targets, prefers
    smaller enclosing evidence over whole-file context under non-large budgets,
    compresses only low-risk regions, and skips or anchors low-value graph-distant
    context. Receipts keep the original handle/hash for every shaped region so the
    exact source stays retrievable; no benchmark ground truth is consulted.
    """
    from archex.scout import chunk_handle
    from archex.serve.compression import RegionCompression, compress_region
    from archex.serve.intent import QueryIntent, classify_intent

    intent = classify_intent(question)
    protect_code = intent == QueryIntent.DEBUGGING
    meta = bundle.retrieval_metadata
    seed_paths = set(meta.seed_file_paths)
    expanded_paths = set(meta.expanded_file_paths)
    top_score = max((rc.final_score for rc in bundle.chunks), default=0.0)

    direct_by_id: dict[str, bool] = {}
    file_direct_counts: dict[str, int] = {}
    for index, ranked in enumerate(bundle.chunks):
        direct = _passthrough_required(
            index,
            ranked,
            seed_paths=seed_paths,
            expanded_paths=expanded_paths,
            top_score=top_score,
        )
        direct_by_id[ranked.chunk.id] = direct
        if direct:
            file_direct_counts[ranked.chunk.file_path] = (
                file_direct_counts.get(ranked.chunk.file_path, 0) + 1
            )

    candidates: list[PackingCandidate] = []
    outcomes: dict[str, RegionCompression] = {}
    original_tokens_by_id: dict[str, int] = {}
    score_by_id: dict[str, float] = {}
    for ranked in bundle.chunks:
        chunk = ranked.chunk
        direct = direct_by_id[chunk.id]
        handle = chunk_handle(chunk.id)
        original_tokens = count_tokens(chunk.content)
        original_tokens_by_id[chunk.id] = original_tokens
        normalized = (ranked.final_score / top_score) if top_score > 0 else 0.0
        score_by_id[chunk.id] = normalized
        elided_tokens = count_tokens(_elide_anchor(original_tokens, handle))
        outcome = compress_region(
            chunk.content,
            language=chunk.language,
            fetch_original_handle=handle,
            required=direct,
            protect_code=protect_code,
        )
        eligible = outcome is not None and outcome.mode is not CompressionMode.PASSTHROUGH_REQUIRED
        if eligible and outcome is not None:
            outcomes[chunk.id] = outcome
            compressed_tokens = count_tokens(outcome.content)
            loss_risk = outcome.loss_risk
        else:
            compressed_tokens = original_tokens
            loss_risk = CompressionLossRisk.NONE
        direct_or_seed = direct or chunk.file_path in seed_paths
        candidates.append(
            PackingCandidate(
                signals=PackingSignals(
                    candidate_id=chunk.id,
                    file_path=chunk.file_path,
                    retrieval_score=normalized,
                    direct_match=direct,
                    graph_distance=0 if direct_or_seed else 1,
                    graph_edge_confidence=1.0 if direct_or_seed else 0.5,
                    token_count=original_tokens,
                    compression_eligible=eligible,
                    compression_loss_risk=loss_risk,
                    handle_priority=0.0,
                    whole_file=chunk.symbol_name is None,
                    file_evidence_regions=file_direct_counts.get(chunk.file_path, 0),
                    protect_code=protect_code,
                ),
                compressed_token_count=compressed_tokens,
                elided_token_count=elided_tokens,
            )
        )

    tier = budget_tier(bundle.token_budget)
    plan = pack_efficiently(candidates, token_budget=bundle.token_budget, budget_tier=tier)
    decisions = {region.candidate_id: region.decision for region in plan.regions}

    packed = bundle.model_copy(deep=True)
    receipt_items = (
        {item.handle: item for item in packed.receipt.returned_context}
        if packed.receipt is not None
        else {}
    )

    kept: list[RankedChunk] = []
    kept_handles: set[str] = set()
    actual_counts = dict.fromkeys(PackDecision, 0)
    uncompressed_tokens = 0
    delivered_mass = 0.0
    passthrough_required_tokens = 0
    compressed_required_tokens = 0
    hidden_required = 0
    for ranked in packed.chunks:
        chunk = ranked.chunk
        original_tokens = original_tokens_by_id[chunk.id]
        uncompressed_tokens += original_tokens
        decision = decisions.get(chunk.id, PackDecision.SKIP)
        if decision is PackDecision.SKIP:
            actual_counts[PackDecision.SKIP] += 1
            continue
        direct = direct_by_id[chunk.id]
        handle = chunk_handle(chunk.id)
        item = receipt_items.get(handle)
        if decision is PackDecision.ELIDE:
            # Drop the body to a fetch anchor, unless the anchor would not be
            # smaller than the region — then keep it verbatim rather than grow it.
            anchor = _elide_anchor(original_tokens, handle)
            if count_tokens(anchor) < original_tokens:
                _shape_packed_region(
                    chunk,
                    item,
                    original=chunk.content,
                    new_content=anchor,
                    mode=CompressionMode.STRUCTURAL_CODE_ELISION,
                    loss_risk=_ELIDE_ANCHOR_RISK,
                )
                actual_counts[PackDecision.ELIDE] += 1
                if direct:
                    compressed_required_tokens += count_tokens(anchor)
                    hidden_required += 1
            else:
                decision = PackDecision.INCLUDE
        if decision is PackDecision.INCLUDE:
            actual_counts[PackDecision.INCLUDE] += 1
            delivered_mass += score_by_id[chunk.id]
            if direct:
                passthrough_required_tokens += original_tokens
        elif decision is PackDecision.COMPRESS:
            outcome = outcomes[chunk.id]
            new_tokens = _shape_packed_region(
                chunk,
                item,
                original=chunk.content,
                new_content=outcome.content,
                mode=outcome.mode,
                loss_risk=outcome.loss_risk,
            )
            actual_counts[PackDecision.COMPRESS] += 1
            delivered_mass += score_by_id[chunk.id]
            if direct:
                compressed_required_tokens += new_tokens
                hidden_required += 1
        kept.append(ranked)
        kept_handles.add(handle)

    packed.chunks = kept
    packed_tokens = sum(count_tokens(rc.chunk.content) for rc in kept)
    packed.token_count = packed_tokens
    if packed.receipt is not None:
        packed.receipt.returned_context = [
            item for item in packed.receipt.returned_context if item.handle in kept_handles
        ]
        packed.receipt.returned_total = len(packed.receipt.returned_context)
        packed.receipt.token_budget.consumed = packed_tokens

    relevance_per_1k = (delivered_mass / packed_tokens * 1000.0) if packed_tokens else 0.0
    bundle_ratio = (packed_tokens / uncompressed_tokens) if uncompressed_tokens else 1.0
    result_fields: dict[str, float | int | None] = {
        "bundle_tokens_uncompressed": uncompressed_tokens,
        "bundle_tokens_compressed": packed_tokens,
        "bundle_compression_ratio": bundle_ratio,
        "required_context_passthrough_tokens": passthrough_required_tokens,
        "required_context_compressed_tokens": compressed_required_tokens,
        "compression_hidden_required_region_count": hidden_required,
        "packed_relevance_per_1k_tokens": relevance_per_1k,
        "packing_included_regions": actual_counts[PackDecision.INCLUDE],
        "packing_compressed_regions": actual_counts[PackDecision.COMPRESS],
        "packing_elided_regions": actual_counts[PackDecision.ELIDE],
        "packing_skipped_regions": actual_counts[PackDecision.SKIP],
    }
    provenance = {
        "budget_tier": tier.value,
        "query_intent": intent.value,
        "protect_code": str(protect_code).lower(),
        "include_count": str(actual_counts[PackDecision.INCLUDE]),
        "compress_count": str(actual_counts[PackDecision.COMPRESS]),
        "elide_count": str(actual_counts[PackDecision.ELIDE]),
        "skip_count": str(actual_counts[PackDecision.SKIP]),
        "direct_match_count": str(sum(1 for v in direct_by_id.values() if v)),
        "hidden_required_count": str(hidden_required),
        "bundle_compression_ratio": f"{bundle_ratio:.4f}",
        "relevance_per_1k_tokens": f"{relevance_per_1k:.4f}",
    }
    return _BundlePacking(bundle=packed, result_fields=result_fields, provenance=provenance)


def run_archex_query_efficiency_packed(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: archex query retrieval, then efficiency-aware packing.

    Runs the exact ``archex_query`` retrieval path, then re-packs the assembled
    bundle with the efficiency-aware packer to maximize relevance per token while
    preserving direct/high-confidence targets. The packed bundle is the lane's
    returned context, so retrieval metrics reflect the packed selection; receipts
    and compression metadata are preserved. The product default is unchanged.
    """
    strategy = Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED
    t0 = time.perf_counter()
    bundle, effective_config, timing = _query_bundle(
        task,
        repo_path,
        strategy=strategy,
        index_config=IndexConfig(vector=False),
        cache=benchmark_cache_enabled(default=False),
    )
    packing = _pack_bundle(bundle, question=task.question)
    wall_ms = (time.perf_counter() - t0) * 1000
    result = _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=packing.bundle,
        timing=timing,
        wall_ms=wall_ms,
        include_completion=True,
        measure_freshness=False,
    )
    fields = dict(packing.result_fields)
    # The packed bundle is the returned context, so its with-completion efficiency
    # already reflects packing; mirror it into the shared compression-efficiency
    # field so reports can compare normal/compressed/efficiency-packed lanes.
    fields["token_efficiency_with_compression_and_completion"] = (
        result.token_efficiency_with_completion
    )
    result = result.model_copy(update=fields)
    result.provenance = packing.provenance
    logger.info(
        "Strategy %s for %s: ratio=%s included=%s skipped=%s wall=%.1fms",
        strategy.value,
        task.task_id,
        packing.provenance["bundle_compression_ratio"],
        packing.provenance["include_count"],
        packing.provenance["skip_count"],
        wall_ms,
    )
    return result


def _fuse_dual_bundles(
    structural: ContextBundle,
    behavioral: ContextBundle,
    *,
    query: str,
    token_budget: int,
) -> tuple[ContextBundle, dict[str, int]]:
    """Reciprocal-rank fuse two retrieved bundles into one budget-bounded bundle.

    Both inputs already ran the full retrieval + packing path for their subquery.
    Their ranked chunks are merged by reciprocal rank; the fused order is filled
    greedily up to ``token_budget`` (the top chunk is always kept). The structural
    bundle supplies receipt/metadata scaffolding; the receipt's returned_context
    and the seed/expansion diagnostics are rebuilt to match the fused chunk set so
    they stay aligned. Returns the fused bundle plus candidate-count stats.
    """
    from archex.index.fusion import reciprocal_rank_fusion
    from archex.scout import chunk_handle

    structural_ranked = [(rc.chunk, rc.final_score) for rc in structural.chunks]
    behavioral_ranked = [(rc.chunk, rc.final_score) for rc in behavioral.chunks]
    fused = reciprocal_rank_fusion(structural_ranked, behavioral_ranked)

    ranked_by_id: dict[str, RankedChunk] = {}
    for ranked in (*structural.chunks, *behavioral.chunks):
        ranked_by_id.setdefault(ranked.chunk.id, ranked)

    fused_chunks: list[RankedChunk] = []
    total_tokens = 0
    for chunk, rrf_score in fused:
        tokens = chunk.token_count or count_tokens(chunk.content)
        if fused_chunks and total_tokens + tokens > token_budget:
            continue
        base = ranked_by_id[chunk.id]
        fused_chunks.append(base.model_copy(update={"final_score": rrf_score}))
        total_tokens += tokens

    receipt = None
    if structural.receipt is not None:
        items_by_handle: dict[str, ContextReceiptItem] = {}
        for source in (structural, behavioral):
            if source.receipt is not None:
                for item in source.receipt.returned_context:
                    items_by_handle.setdefault(item.handle, item)
        returned = [
            items_by_handle[handle]
            for ranked in fused_chunks
            if (handle := chunk_handle(ranked.chunk.id)) in items_by_handle
        ]
        receipt = structural.receipt.model_copy(
            update={
                "query": query,
                "returned_context": returned,
                "returned_total": len(returned),
            }
        )

    # The fused selection has no single-pass "seed vs graph-expansion" split, so
    # the structural pass's expansion diagnostics would misattribute to this lane.
    # Reset them so every fused file reports as a direct seed with no expansion.
    fused_files = _deduplicate_ranked([ranked.chunk.file_path for ranked in fused_chunks])
    fused_metadata = structural.retrieval_metadata.model_copy(
        update={
            "seed_file_paths": [],
            "expanded_file_paths": [],
            "seed_files_found": len(fused_files),
            "expansion_files_added": 0,
            "expansion_eligible_seeds": 0,
            "expansion_candidates_found": 0,
            "expansion_import_neighbor_edges": 0,
            "expansion_same_module_candidates": 0,
            "expansion_hub_candidates": 0,
            "expansion_test_candidates_skipped": 0,
            "expansion_zero_candidate_reason": "",
            "expansion_reason_counts": {},
            "expanded_file_reasons": {},
        }
    )

    fused_bundle = structural.model_copy(
        update={
            "query": query,
            "chunks": fused_chunks,
            "token_count": total_tokens,
            "token_budget": token_budget,
            "retrieval_metadata": fused_metadata,
            "receipt": receipt,
        }
    )
    stats = {
        "structural_candidates": len(structural.chunks),
        "behavioral_candidates": len(behavioral.chunks),
        "fused_candidates": len(fused),
        "fused_included": len(fused_chunks),
    }
    return fused_bundle, stats


def run_archex_query_dual_transform(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: deterministic dual-perspective query transformation.

    Rewrites the query into a structural subquery (paths, identifiers, call
    sites, errors, stack-trace and import signals) and a behavioural subquery
    (natural-language symptoms and domain nouns) using rule-based transforms
    only, runs the BM25 retrieval path once per subquery sharing one warm index,
    then reciprocal-rank fuses the two results into one budget-bounded bundle.
    Both subqueries and the fusion stats are recorded in strategy provenance. No
    hosted LLM calls; the product default is unchanged.
    """
    strategy = Strategy.ARCHEX_QUERY_DUAL_TRANSFORM
    dual = transform_query(task.question)
    t0 = time.perf_counter()
    structural_bundle, _, structural_timing = _query_bundle(
        task.model_copy(update={"question": dual.structural}),
        repo_path,
        strategy=strategy,
        index_config=IndexConfig(vector=False),
        cache=True,
    )
    behavioral_bundle, effective_config, behavioral_timing = _query_bundle(
        task.model_copy(update={"question": dual.behavioral}),
        repo_path,
        strategy=strategy,
        index_config=IndexConfig(vector=False),
        cache=True,
    )
    fused_bundle, fusion_stats = _fuse_dual_bundles(
        structural_bundle,
        behavioral_bundle,
        query=task.question,
        token_budget=task.token_budget,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    # Report cache state across both passes: warm only if neither built work.
    behavioral_timing.cached = structural_timing.cached and behavioral_timing.cached
    result = _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=fused_bundle,
        timing=behavioral_timing,
        wall_ms=wall_ms,
        include_completion=True,
        measure_freshness=False,
    )
    result.provenance = dual.to_provenance() | {
        "structural_candidates": str(fusion_stats["structural_candidates"]),
        "behavioral_candidates": str(fusion_stats["behavioral_candidates"]),
        "fused_candidates": str(fusion_stats["fused_candidates"]),
        "fused_included": str(fusion_stats["fused_included"]),
    }
    logger.info(
        "Strategy %s for %s: structural=%d behavioral=%d fused=%d wall=%.1fms",
        strategy.value,
        task.task_id,
        fusion_stats["structural_candidates"],
        fusion_stats["behavioral_candidates"],
        fusion_stats["fused_included"],
        wall_ms,
    )
    return result


_BOUNDED_RERANK_CAPS = RerankCaps()


def _load_local_reranker() -> CrossEncoderReranker | None:
    """Reuse an in-process cross-encoder reranker; never load or download a model.

    Returns a warmed reranker only when one is already loaded in this process
    (e.g. a prior fusion-rerank pass warmed it). When no local model is loaded, or
    when warming the cached model is rejected (e.g. the model needs remote code and
    the benchmark run did not opt in), returns None so the lane skips the model
    rerank with explicit provenance. It never triggers a download, so it adds no
    network behaviour; the remote-code opt-in mirrors the benchmark options.
    """
    from archex.index.rerank import CrossEncoderReranker, loaded_reranker_model_names

    names = loaded_reranker_model_names()
    if not names:
        return None
    options = current_benchmark_retrieval_options()
    reranker = CrossEncoderReranker(names[0], allow_remote_code=options.allow_remote_code)
    try:
        reranker.warm()  # the model is already cached in-process; this never downloads
    except ArchexError:
        return None
    return reranker


def _normalize_ce_scores(results: list[tuple[CodeChunk, float]]) -> dict[str, float]:
    """Min-max normalise cross-encoder scores into [0, 1] keyed by chunk id."""
    if not results:
        return {}
    scores = [score for _, score in results]
    low, high = min(scores), max(scores)
    span = high - low
    if span <= 0:
        return {chunk.id: 1.0 for chunk, _ in results}
    return {chunk.id: (score - low) / span for chunk, score in results}


@dataclass
class _BoundedRerank:
    """Reordered bundle plus the provenance describing the bounded rerank."""

    bundle: ContextBundle
    provenance: dict[str, str]


def _bounded_rerank_bundle(
    bundle: ContextBundle, *, question: str, caps: RerankCaps
) -> _BoundedRerank:
    """Reorder a bounded compact candidate set by symbolic evidence (+ optional CE).

    Only the top ``caps.candidate_cap`` retrieved chunks are reranked; chunks past
    the cap keep their order. The deterministic symbolic evidence pass always runs.
    A local cross-encoder is reused only when already loaded and only if its
    measured pass stays under ``caps.latency_cap_ms``; otherwise the model rerank
    is skipped/aborted and the symbolic order stands. Receipt rows are realigned to
    the reordered chunk set so receipts stay aligned.
    """
    from archex.scout import chunk_handle

    signals = query_signals(question)
    compact = bundle.chunks[: caps.candidate_cap]
    tail = bundle.chunks[caps.candidate_cap :]
    candidate_count = len(compact)
    scores = symbolic_scores([ranked.chunk for ranked in compact], signals)

    reranker = _load_local_reranker()
    rerank_ms = 0.0
    if reranker is None:
        cross_encoder_status = "skipped:unavailable"
    elif not compact:
        cross_encoder_status = "skipped:no_candidates"
    else:
        candidates = [(ranked.chunk, score) for ranked, score in zip(compact, scores, strict=True)]
        t0 = time.perf_counter()
        ce_results = reranker.rerank(question, candidates, top_k=candidate_count)
        rerank_ms = (time.perf_counter() - t0) * 1000
        if rerank_ms > caps.latency_cap_ms:
            cross_encoder_status = "aborted:latency"
        else:
            ce_norm = _normalize_ce_scores(ce_results)
            scores = [
                score + ce_norm.get(ranked.chunk.id, 0.0)
                for ranked, score in zip(compact, scores, strict=True)
            ]
            cross_encoder_status = "applied"

    # Stable sort: higher score first, ties keep original retrieval order.
    order = sorted(range(candidate_count), key=lambda index: -scores[index])
    new_chunks = [compact[index] for index in order] + list(tail)

    reordered = bundle.model_copy(update={"chunks": new_chunks})
    if reordered.receipt is not None:
        items_by_handle = {item.handle: item for item in reordered.receipt.returned_context}
        ordered: list[ContextReceiptItem] = []
        seen: set[str] = set()
        for ranked in new_chunks:
            handle = chunk_handle(ranked.chunk.id)
            item = items_by_handle.get(handle)
            if item is not None and handle not in seen:
                ordered.append(item)
                seen.add(handle)
        # Preserve any receipt rows without a matching chunk (defensive).
        ordered.extend(
            item for item in reordered.receipt.returned_context if item.handle not in seen
        )
        reordered.receipt = reordered.receipt.model_copy(update={"returned_context": ordered})

    rerank_method = "symbolic+cross_encoder" if cross_encoder_status == "applied" else "symbolic"
    provenance = {
        "candidate_cap": str(caps.candidate_cap),
        "candidates_reranked": str(candidate_count),
        "candidates_total": str(len(bundle.chunks)),
        "latency_cap_ms": f"{caps.latency_cap_ms:.1f}",
        "rerank_ms": f"{rerank_ms:.2f}",
        "rerank_method": rerank_method,
        "cross_encoder_status": cross_encoder_status,
    }
    return _BoundedRerank(bundle=reordered, provenance=provenance)


def run_archex_query_bounded_rerank(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: bounded symbolic/evidence rerank over a compact set.

    Runs the ``archex_query`` BM25 retrieval path, then reranks only the top
    ``candidate_cap`` chunks by deterministic symbolic evidence (path / symbol /
    term overlap blended with the retrieval rank). A local cross-encoder reranker
    is reused only when already loaded in process and only if its measured pass
    stays within the latency cap; otherwise the model rerank is skipped or aborted
    with explicit provenance. Hard candidate and latency caps prevent unbounded
    rerank. The product default is unchanged.
    """
    strategy = Strategy.ARCHEX_QUERY_BOUNDED_RERANK
    t0 = time.perf_counter()
    bundle, effective_config, timing = _query_bundle(
        task,
        repo_path,
        strategy=strategy,
        index_config=IndexConfig(vector=False),
        cache=benchmark_cache_enabled(default=False),
    )
    rerank = _bounded_rerank_bundle(bundle, question=task.question, caps=_BOUNDED_RERANK_CAPS)
    wall_ms = (time.perf_counter() - t0) * 1000
    result = _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=rerank.bundle,
        timing=timing,
        wall_ms=wall_ms,
        include_completion=True,
        measure_freshness=False,
    )
    result.provenance = rerank.provenance
    logger.info(
        "Strategy %s for %s: reranked=%s/%s method=%s ce=%s wall=%.1fms",
        strategy.value,
        task.task_id,
        rerank.provenance["candidates_reranked"],
        rerank.provenance["candidates_total"],
        rerank.provenance["rerank_method"],
        rerank.provenance["cross_encoder_status"],
        wall_ms,
    )
    return result


_SUMMARY_FILE_CAP = 10


def _load_summary_sidecar() -> SummarySidecar | None:
    """Load the opted-in summary sidecar; None when not opted in or unreadable.

    The path comes from the benchmark retrieval options (the explicit offline
    opt-in). The lane never generates summaries — absence means summary-first
    selection is simply disabled.
    """
    path = current_benchmark_retrieval_options().summary_sidecar_path
    if not path:
        return None
    sidecar_path = Path(path)
    if not sidecar_path.is_file():
        return None
    try:
        return SummarySidecar.load(sidecar_path)
    except (OSError, ValueError):
        return None


@dataclass
class _SummarySelection:
    """Summary-first file selection plus stats for provenance."""

    selected_files: list[str]
    stats: dict[str, int]


def _select_summary_files(
    sidecar: SummarySidecar, repo_path: Path, *, question: str, file_cap: int
) -> _SummarySelection:
    """Rank non-stale sidecar entries by query overlap into selected source files.

    Stale entries (source file changed or missing) are excluded so summaries that
    no longer describe the current code never drive selection.
    """
    signals = query_signals(question)
    fresh = 0
    stale = 0
    file_scores: dict[str, int] = {}
    file_order: list[str] = []
    # Hash each source file at most once; every entry for a path is checked
    # against that single read rather than re-reading per symbol entry.
    current_hash: dict[str, str | None] = {}
    for entry in sidecar.entries:
        path = entry.source_file_path
        if path not in current_hash:
            current_hash[path] = file_content_hash(repo_path, path)
        live = current_hash[path]
        if live is None or live != entry.source_content_hash:
            stale += 1
            continue
        fresh += 1
        haystack = f"{entry.summary} {entry.symbol_name or ''} {path}".lower()
        overlap = sum(1 for term in signals.terms if term in haystack)
        if overlap <= 0:
            continue
        if path not in file_scores:
            file_order.append(path)
        file_scores[path] = file_scores.get(path, 0) + overlap
    ranked = sorted(file_order, key=lambda path: -file_scores[path])
    selected = ranked[:file_cap]
    stats = {
        "entries_total": len(sidecar.entries),
        "entries_fresh": fresh,
        "entries_stale": stale,
        "matched_files": len(file_scores),
        "selected_files": len(selected),
    }
    return _SummarySelection(selected_files=selected, stats=stats)


def _filter_bundle_to_files(bundle: ContextBundle, files: set[str]) -> ContextBundle:
    """Keep only chunks whose file was summary-selected; realign receipt + metadata.

    The kept chunks are the original retrieved code (the edit evidence); summaries
    only gate which files are returned. Seed/expansion diagnostics are reset to the
    gated set so they are not misattributed to the pre-filter retrieval.
    """
    from archex.scout import chunk_handle

    kept = [ranked for ranked in bundle.chunks if ranked.chunk.file_path in files]
    token_count = sum(
        ranked.chunk.token_count or count_tokens(ranked.chunk.content) for ranked in kept
    )
    receipt = bundle.receipt
    if receipt is not None:
        kept_handles = {chunk_handle(ranked.chunk.id) for ranked in kept}
        returned = [item for item in receipt.returned_context if item.handle in kept_handles]
        receipt = receipt.model_copy(
            update={"returned_context": returned, "returned_total": len(returned)}
        )
    kept_files = _deduplicate_ranked([ranked.chunk.file_path for ranked in kept])
    metadata = bundle.retrieval_metadata.model_copy(
        update={
            "seed_file_paths": [],
            "expanded_file_paths": [],
            "seed_files_found": len(kept_files),
            "expansion_files_added": 0,
            "expansion_eligible_seeds": 0,
            "expansion_candidates_found": 0,
            "expansion_import_neighbor_edges": 0,
            "expansion_same_module_candidates": 0,
            "expansion_hub_candidates": 0,
            "expansion_test_candidates_skipped": 0,
            "expansion_zero_candidate_reason": "",
            "expansion_reason_counts": {},
            "expanded_file_reasons": {},
        }
    )
    return bundle.model_copy(
        update={
            "chunks": kept,
            "token_count": token_count,
            "receipt": receipt,
            "retrieval_metadata": metadata,
        }
    )


def run_archex_query_summary_sidecar(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: summary-first selection, then original-code retrieval.

    Requires an explicit offline opt-in (a prebuilt sidecar path in the benchmark
    retrieval options); summaries are never auto-generated. Loads the sidecar,
    drops stale entries, ranks the rest by query overlap to select source files,
    then returns the original retrieved code restricted to those files — summaries
    gate selection but are never returned as edit evidence. Without an opt-in (or
    when the selection excludes everything retrieved) the lane returns the plain
    original-code bundle and records why. The product default is unchanged.
    """
    strategy = Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR
    t0 = time.perf_counter()
    bundle, effective_config, timing = _query_bundle(
        task,
        repo_path,
        strategy=strategy,
        index_config=IndexConfig(vector=False),
        cache=benchmark_cache_enabled(default=False),
    )
    current_revision = bundle.receipt.index_revision if bundle.receipt is not None else ""
    sidecar = _load_summary_sidecar()
    if sidecar is None:
        final_bundle = bundle
        provenance = {"sidecar": "absent", "summary_first": "false"}
    else:
        selection = _select_summary_files(
            sidecar, repo_path, question=task.question, file_cap=_SUMMARY_FILE_CAP
        )
        selected = set(selection.selected_files)
        gated = _filter_bundle_to_files(bundle, selected) if selected else bundle
        if gated.chunks and selected:
            final_bundle = gated
            summary_first = "true"
            fallback = "none"
        else:
            # Selection matched nothing retrievable; return the original-code
            # bundle so the lane still yields editable evidence.
            final_bundle = bundle
            summary_first = "false"
            fallback = "empty_selection"
        provenance = {
            "sidecar": "loaded",
            "summary_first": summary_first,
            "summary_fallback": fallback,
            "sidecar_granularity": sidecar.granularity.value,
            "sidecar_index_revision": sidecar.index_revision,
            "index_revision_match": str(sidecar.index_revision == current_revision).lower(),
            "entries_total": str(selection.stats["entries_total"]),
            "entries_fresh": str(selection.stats["entries_fresh"]),
            "entries_stale": str(selection.stats["entries_stale"]),
            "summary_selected_file_count": str(selection.stats["selected_files"]),
            "summary_selected_files": ", ".join(selection.selected_files) or "none",
            "original_code_retrieved": "true",
            "fetch_original_preserved": "true",
        }
    wall_ms = (time.perf_counter() - t0) * 1000
    result = _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=final_bundle,
        timing=timing,
        wall_ms=wall_ms,
        include_completion=True,
        measure_freshness=False,
    )
    result.provenance = provenance
    logger.info(
        "Strategy %s for %s: sidecar=%s summary_first=%s wall=%.1fms",
        strategy.value,
        task.task_id,
        provenance["sidecar"],
        provenance["summary_first"],
        wall_ms,
    )
    return result


_MULTIHOP_HOP_CAP = 2
_MULTIHOP_FRONTIER_CAP = 8
_MULTIHOP_CONFIDENCE_THRESHOLD = 0.5
# Fraction of the token budget retrieved as seeds; the remainder funds expansion.
_MULTIHOP_SEED_BUDGET_FRACTION = 0.5


def _file_token_map(store: IndexStore) -> dict[str, int]:
    """Total token count per file across the indexed chunk set."""
    totals: dict[str, int] = {}
    for chunk in store.get_chunks():
        totals[chunk.file_path] = totals.get(chunk.file_path, 0) + (
            chunk.token_count or count_tokens(chunk.content)
        )
    return totals


def _expansion_edge(
    decision: ExpansionDecision, *, reason: ContextOmittedEdgeReason | None = None
) -> ContextReceiptEdge:
    return ContextReceiptEdge(
        source=decision.via,
        target=decision.file,
        kind=EdgeKind.IMPORTS,
        confidence=EdgeConfidence.EXTRACTED,
        confidence_score=decision.confidence,
        evidence=[
            f"graph multihop hop {decision.hop} via {decision.via} "
            f"(confidence {decision.confidence:.2f})"
        ],
        reason=reason,
    )


def _expansion_skip(
    decision: ExpansionDecision, reason: ContextSkippedReason
) -> ContextSkippedCandidate:
    return ContextSkippedCandidate(
        file_path=decision.file,
        reason=reason,
        score=decision.confidence,
        detail=f"graph multihop hop {decision.hop} via {decision.via}",
    )


def _apply_multihop_expansion(
    bundle: ContextBundle,
    store: IndexStore,
    *,
    seed_files: list[str],
    expansion: MultihopResult,
) -> ContextBundle:
    """Append expanded files' original chunks and record the expansion in receipts.

    Expanded files contribute their original code (the edit evidence); every
    expansion decision becomes a receipt edge (included) or a cut (omitted edge /
    skipped candidate), and the seed/expansion diagnostics are set to the expanded
    bundle so they describe what the lane actually returned.
    """
    from archex.receipt import chunk_content_hash
    from archex.scout import chunk_handle

    confidence_by_file = {d.file: d.confidence for d in expansion.expanded_decisions()}
    chunks_by_file: dict[str, list[CodeChunk]] = {}
    for chunk in store.get_chunks_for_files(expansion.added_files):
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    expanded_ranked: list[RankedChunk] = []
    new_items: list[ContextReceiptItem] = []
    added_tokens = 0
    for file in expansion.added_files:
        confidence = confidence_by_file.get(file, 0.0)
        for chunk in sorted(chunks_by_file.get(file, []), key=lambda c: c.start_line):
            expanded_ranked.append(RankedChunk(chunk=chunk, final_score=confidence))
            added_tokens += chunk.token_count or count_tokens(chunk.content)
            new_items.append(
                ContextReceiptItem(
                    handle=chunk_handle(chunk.id),
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content_hash=chunk_content_hash(chunk),
                    reason_codes=["graph_multihop_expansion"],
                )
            )

    new_chunks = list(bundle.chunks) + expanded_ranked
    receipt = bundle.receipt
    if receipt is not None:
        included = list(receipt.included_edges)
        omitted = list(receipt.omitted_edges)
        skipped = list(receipt.skipped_candidates)
        for decision in expansion.decisions:
            if decision.action is ExpansionAction.EXPANDED:
                included.append(_expansion_edge(decision))
            elif decision.action is ExpansionAction.SUPPRESSED_LOW_CONFIDENCE:
                omitted.append(
                    _expansion_edge(decision, reason=ContextOmittedEdgeReason.BELOW_THRESHOLD)
                )
                skipped.append(_expansion_skip(decision, ContextSkippedReason.BELOW_THRESHOLD))
            elif decision.action is ExpansionAction.CUT_FRONTIER:
                skipped.append(
                    _expansion_skip(decision, ContextSkippedReason.DEPENDENCY_FRONTIER_CUT)
                )
            elif decision.action is ExpansionAction.CUT_BUDGET:
                omitted.append(
                    _expansion_edge(decision, reason=ContextOmittedEdgeReason.OVER_BUDGET)
                )
                skipped.append(_expansion_skip(decision, ContextSkippedReason.OVER_BUDGET))
        returned = list(receipt.returned_context) + new_items
        receipt = receipt.model_copy(
            update={
                "returned_context": returned,
                "returned_total": len(returned),
                "included_edges": included,
                "included_edges_total": len(included),
                "omitted_edges": omitted,
                "omitted_edges_total": len(omitted),
                "skipped_candidates": skipped,
                "skipped_total": len(skipped),
            }
        )

    metadata = bundle.retrieval_metadata.model_copy(
        update={
            "seed_file_paths": list(seed_files),
            "expanded_file_paths": list(expansion.added_files),
            "seed_files_found": len(seed_files),
            "expansion_files_added": len(expansion.added_files),
            "expansion_candidates_found": len(expansion.decisions),
            "expansion_import_neighbor_edges": expansion.action_count(ExpansionAction.EXPANDED),
        }
    )
    return bundle.model_copy(
        update={
            "chunks": new_chunks,
            "token_count": bundle.token_count + added_tokens,
            "receipt": receipt,
            "retrieval_metadata": metadata,
        }
    )


def run_archex_query_graph_multihop(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only lane: bounded multi-hop dependency-graph expansion.

    Retrieves seeds with a reduced budget, then expands the file dependency graph
    outward from those seeds under hard caps — an edge-confidence threshold, a
    per-hop frontier cap, a hop cap, and a token budget — so dependency expansion
    cannot flood the bundle. Expanded files contribute their original code; every
    expansion decision and cut is recorded in the receipt (included/omitted edges,
    skipped candidates) and in provenance. The product default is unchanged.
    """
    from archex.api import index_repository
    from archex.index.graph import DependencyGraph
    from archex.models import Config

    strategy = Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP
    t0 = time.perf_counter()
    seed_budget = max(1, int(task.token_budget * _MULTIHOP_SEED_BUDGET_FRACTION))
    seed_task = task.model_copy(update={"token_budget": seed_budget})
    bundle, effective_config, timing = _query_bundle(
        seed_task,
        repo_path,
        strategy=strategy,
        index_config=IndexConfig(vector=False),
        cache=True,
    )
    seed_files = _deduplicate_ranked([ranked.chunk.file_path for ranked in bundle.chunks])
    expansion_budget = max(0, task.token_budget - bundle.token_count)

    source = benchmark_repo_source(seed_task, repo_path, strategy=strategy)
    store = index_repository(
        source, config=Config(cache=True, languages=task.languages), index_config=effective_config
    )
    try:
        graph = DependencyGraph.from_edges(store.get_edges())
        edges = [
            GraphEdge(edge.source, edge.target, edge.confidence_score)
            for edge in graph.file_edges()
        ]
        file_tokens = _file_token_map(store)
        caps = MultihopCaps(
            hop_cap=_MULTIHOP_HOP_CAP,
            frontier_cap=_MULTIHOP_FRONTIER_CAP,
            confidence_threshold=_MULTIHOP_CONFIDENCE_THRESHOLD,
            token_budget=expansion_budget,
        )
        expansion = graph_multihop_expand(edges, seed_files, file_tokens, caps)
        expanded_bundle = _apply_multihop_expansion(
            bundle, store, seed_files=seed_files, expansion=expansion
        )
    finally:
        store.close()

    wall_ms = (time.perf_counter() - t0) * 1000
    result = _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=expanded_bundle,
        timing=timing,
        wall_ms=wall_ms,
        include_completion=True,
        measure_freshness=False,
    )
    result.provenance = {
        "hop_cap": str(_MULTIHOP_HOP_CAP),
        "frontier_cap": str(_MULTIHOP_FRONTIER_CAP),
        "confidence_threshold": f"{_MULTIHOP_CONFIDENCE_THRESHOLD:.2f}",
        "expansion_token_budget": str(expansion_budget),
        "seed_file_count": str(len(seed_files)),
        "hops_run": str(expansion.hops_run),
        "files_expanded": str(len(expansion.added_files)),
        "suppressed_low_confidence": str(
            expansion.action_count(ExpansionAction.SUPPRESSED_LOW_CONFIDENCE)
        ),
        "frontier_cuts": str(expansion.action_count(ExpansionAction.CUT_FRONTIER)),
        "budget_cuts": str(expansion.action_count(ExpansionAction.CUT_BUDGET)),
        "expanded_files": ", ".join(expansion.added_files) or "none",
    }
    logger.info(
        "Strategy %s for %s: seeds=%d expanded=%d hops=%d wall=%.1fms",
        strategy.value,
        task.task_id,
        len(seed_files),
        len(expansion.added_files),
        expansion.hops_run,
        wall_ms,
    )
    return result


def run_archex_scout_fetch(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Two-call scout map plus chunk-first exact fetch with direct-query guardrails."""
    from archex.api import query, scout_with_bundle
    from archex.models import Config
    from archex.scout import DEFAULT_SCOUT_TOKEN_BUDGET, render_scout

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path, strategy=Strategy.ARCHEX_SCOUT_FETCH)
    config = Config(
        cache=benchmark_cache_enabled(default=False),
        languages=task.languages,
    )
    index_config = benchmark_index_config(
        IndexConfig(vector=False),
        strategy=Strategy.ARCHEX_SCOUT_FETCH,
    )

    scout_result, direct_bundle = scout_with_bundle(
        source,
        task.question,
        token_budget=DEFAULT_SCOUT_TOKEN_BUDGET,
        output_format="markdown",
        config=config,
        index_config=index_config,
        timing=timing,
        refresh=True,
    )
    scout_tokens = count_tokens(render_scout(scout_result, output_format="markdown"))
    scout_ranked_files = [item.path for item in scout_result.ranked_files]
    missing_from_scout_map = sorted(set(task.expected_files) - set(scout_ranked_files))

    if scout_result.fetch_plan.recommended_strategy == "direct_query":
        bundle = direct_bundle
        fetch_mode = "direct_query"
        tool_calls = 1
        tokens_total = direct_bundle.token_count
        tokens_input = direct_bundle.token_count
    else:
        bundle = query(
            source,
            task.question,
            token_budget=task.token_budget,
            explicit_token_budget=True,
            config=config,
            index_config=index_config,
            handles=scout_result.fetch_plan.handles,
        )
        fetch_mode = scout_result.fetch_plan.recommended_strategy
        tool_calls = 2
        tokens_total = scout_tokens + bundle.token_count
        tokens_input = scout_tokens + bundle.token_count

    ranked_files = [chunk.chunk.file_path for chunk in bundle.chunks]
    unique_ranked = _deduplicate_ranked(ranked_files)
    result_files = set(unique_ranked)
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    required_metrics = compute_required_file_metrics(
        result_files,
        task.expected_files,
    )
    (
        required_file_recall,
        missed_required_file_rate,
        missed_required_task_rate,
        all_required_files_present,
        present,
        missing,
    ) = required_metrics
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    fetch_fields = _archex_fields(bundle, task, repo_path)
    completion_tokens, completion_files = compute_bundle_completion_penalty(
        repo_path, result_files, task.expected_files
    )
    tokens_with_completion = tokens_total + completion_tokens
    missing_from_fetch = sorted(set(task.expected_files) - result_files)
    extra_fetch_files = sorted(result_files - set(task.expected_files))
    scout_ranked_set = set(scout_ranked_files)
    if fetch_mode in {"chunk_first", "hybrid_fetch"}:
        extra_fetch_file_reasons = {
            path: scout_result.fetch_plan.file_reasons.get(path, "selected_handle reason=unknown")
            for path in extra_fetch_files
        }
        missing_from_fetch_reasons = {}
        for path in missing_from_fetch:
            if path in scout_result.fetch_plan.file_reasons:
                missing_from_fetch_reasons[path] = scout_result.fetch_plan.file_reasons[path]
            elif path not in scout_ranked_set:
                missing_from_fetch_reasons[path] = "not_in_scout_map"
            else:
                missing_from_fetch_reasons[path] = "ranked_without_reason"
    else:
        extra_fetch_file_reasons = {path: "direct_query_fallback" for path in extra_fetch_files}
        missing_from_fetch_reasons = {
            path: "direct_query_bundle_omission" for path in missing_from_fetch
        }
    result = BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_SCOUT_FETCH,
        tokens_total=tokens_total,
        tokens_input=tokens_input,
        tokens_output=tokens_total,
        token_efficiency=compute_token_efficiency(tokens_total, tokens_input),
        result_files=unique_ranked,
        required_file_recall=required_file_recall,
        missed_required_file_rate=missed_required_file_rate,
        missed_required_task_rate=missed_required_task_rate,
        all_required_files_present=all_required_files_present,
        required_files_present=present,
        required_files_missing=missing,
        post_bundle_read_turns=len(completion_files),
        task_completion_result=completion_result_from_missing(completion_files),
        receipt_accuracy=compute_receipt_accuracy(
            bundle,
            all_required_files_present=all_required_files_present,
        ),
        bundle_completion_tokens=completion_tokens,
        bundle_completion_files=completion_files,
        token_efficiency_with_completion=compute_token_efficiency(
            tokens_with_completion, tokens_input + completion_tokens
        ),
        tokens_raw_baseline=fetch_fields.tokens_raw_baseline,
        symbol_recall=fetch_fields.symbol_recall,
        tool_calls=tool_calls,
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
        f1_score=f1,
        mrr=mrr_val,
        ndcg=ndcg_val,
        map_score=map_val,
        savings_vs_raw=0.0,
        wall_time_ms=wall_ms,
        cached=timing.cached,
        timing=timing,
        timestamp=now_iso(),
        unique_ranked_files=len(unique_ranked),
        seed_files=scout_ranked_files,
        expanded_files=[],
        expansion_ratio=0.0,
        seed_recall=compute_recall(set(scout_ranked_files), task.expected_files),
        seed_precision=compute_precision(set(scout_ranked_files), task.expected_files),
        chunker=fetch_fields.chunker,
        index_chunk_count=fetch_fields.index_chunk_count,
        mean_chunk_tokens=fetch_fields.mean_chunk_tokens,
        category=task.category,
        family=task.family,
        cache_state=_cache_state(timing),
        provenance={
            "scout_token_budget": str(DEFAULT_SCOUT_TOKEN_BUDGET),
            "scout_tokens": str(scout_tokens),
            "fetch_handles": str(len(scout_result.fetch_plan.handles)),
            "fetch_mode": fetch_mode,
            "guardrail_reason": scout_result.fetch_plan.guardrail_reason or "none",
            "missing_from_scout_map": ",".join(missing_from_scout_map) or "none",
            "missing_from_fetch": ",".join(missing_from_fetch) or "none",
            "missing_from_fetch_reasons": "; ".join(
                f"{path}=>{reason}" for path, reason in missing_from_fetch_reasons.items()
            )
            or "none",
            "estimated_fetch_tokens": str(scout_result.fetch_plan.estimated_fetch_tokens),
            "estimated_fetch_files": str(scout_result.fetch_plan.estimated_fetch_files),
            "projected_coverage": f"{scout_result.fetch_plan.coverage_score_mass:.3f}",
            "target_coverage": f"{scout_result.fetch_plan.target_score_mass:.3f}",
            "projected_chunk_precision": f"{scout_result.fetch_plan.projected_precision:.3f}",
            "projected_direct_precision": f"{scout_result.fetch_plan.direct_query_precision:.3f}",
            "direct_query_tokens": str(scout_result.fetch_plan.direct_query_tokens),
            "direct_query_files": str(scout_result.fetch_plan.direct_query_files),
            "extra_fetch_file_reasons": "; ".join(
                f"{path}=>{reason}" for path, reason in extra_fetch_file_reasons.items()
            )
            or "none",
            "intent_class": task.category.value if task.category is not None else "uncategorized",
        },
    )
    return result.model_copy(update=_region_result_fields(bundle, task))


def run_archex_query_vector(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Pure vector retrieval strategy: vector search without BM25."""
    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_VECTOR,
        index_config=IndexConfig(bm25=False, vector=True, quantize_vectors=False),
        cache=True,
    )


def run_surrogate_vector(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Pure surrogate-vector retrieval strategy."""
    from archex.models import RetrievalPolicy, VectorMode

    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.SURROGATE_VECTOR,
        index_config=IndexConfig(
            bm25=False,
            vector=True,
            embedder=current_benchmark_retrieval_options().embedder,
            vector_mode=VectorMode.SURROGATE,
            retrieval_policy=RetrievalPolicy.VECTOR_ONLY,
        ),
        cache=True,
    )


def run_archex_query_fusion(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Full fusion strategy: BM25 + independent vector + confidence-aware RRF."""
    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_FUSION,
        index_config=IndexConfig(vector=True, quantize_vectors=False),
        cache=True,
    )


def run_archex_query_hybrid(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Alias for the BM25 + independent vector hybrid retrieval strategy."""
    result = _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_HYBRID,
        index_config=IndexConfig(vector=True, quantize_vectors=False),
        cache=True,
    )
    return result


def _vector_artifact_matches_config(path: Path, index_config: IndexConfig) -> bool:
    if not path.exists():
        return False

    try:
        import numpy as np

        with np.load(str(path), allow_pickle=False) as data:
            quantized = "quantize_meta" in data
            if not index_config.quantize_vectors:
                return not quantized
            if not quantized:
                return False
            meta = list(data["quantize_meta"])
            return int(meta[0]) == index_config.quantize_bits
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _ensure_vector_artifact(
    task: BenchmarkTask,
    repo_path: Path,
    *,
    strategy: Strategy,
    index_config: IndexConfig,
) -> Path:
    source = benchmark_repo_source(task, repo_path, strategy=strategy)
    effective_config = benchmark_index_config(index_config, strategy=strategy)
    cache = CacheManager()
    key = cache.cache_key(source)
    path = cache.vector_path(
        key,
        vector_mode=effective_config.vector_mode,
        surrogate_version=effective_config.surrogate_version,
    )
    if _vector_artifact_matches_config(path, effective_config):
        return path
    if path.exists():
        path.unlink()

    from archex.api import query
    from archex.models import Config

    cache.invalidate(key)
    timing = PipelineTiming()
    query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
        config=Config(cache=True, languages=task.languages),
        index_config=effective_config,
        timing=timing,
    )
    return path


def _quantized_storage_provenance(task: BenchmarkTask, repo_path: Path) -> dict[str, str]:
    unquantized_path = _ensure_vector_artifact(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_HYBRID,
        index_config=IndexConfig(vector=True, quantize_vectors=False),
    )
    quantized_path = _ensure_vector_artifact(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT,
        index_config=IndexConfig(vector=True, quantize_vectors=True, quantize_bits=4),
    )
    unquantized_bytes = unquantized_path.stat().st_size if unquantized_path.exists() else 0
    quantized_bytes = quantized_path.stat().st_size if quantized_path.exists() else 0
    compression_ratio = (
        unquantized_bytes / quantized_bytes if unquantized_bytes and quantized_bytes else 0.0
    )
    size_ratio = (
        quantized_bytes / unquantized_bytes if unquantized_bytes and quantized_bytes else 0.0
    )
    return {
        "unquantized_vector_npz_bytes": str(unquantized_bytes),
        "quantized_vector_npz_bytes": str(quantized_bytes),
        "vector_compression_ratio": f"{compression_ratio:.6f}",
        "quantized_vector_size_ratio": f"{size_ratio:.6f}",
    }


def run_archex_query_hybrid_quantized_4bit(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Hybrid retrieval with 4-bit TurboQuant vector storage."""
    provenance = _quantized_storage_provenance(task, repo_path)
    result = _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT,
        index_config=IndexConfig(vector=True, quantize_vectors=True, quantize_bits=4),
        cache=True,
    )
    result.provenance = result.provenance | provenance
    return result


def run_archex_query_fusion_rerank(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Fusion strategy with cross-encoder reranking: BM25 + vector + rerank."""
    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_FUSION_RERANK,
        index_config=IndexConfig(vector=True, rerank=True, quantize_vectors=False),
        cache=True,
    )


def run_cross_layer_fusion(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """BM25 over raw chunks plus vector retrieval over surrogates."""
    from archex.models import RetrievalPolicy, VectorMode

    return _run_query_strategy(
        task,
        repo_path,
        strategy=Strategy.CROSS_LAYER_FUSION,
        index_config=IndexConfig(
            vector=True,
            embedder=current_benchmark_retrieval_options().embedder,
            vector_mode=VectorMode.SURROGATE,
            retrieval_policy=RetrievalPolicy.CROSS_LAYER,
        ),
        cache=True,
    )


def _vector_dependencies_available() -> bool:
    """Return True when an embedding backend (fastembed/sentence-transformers) imports."""
    try:
        import fastembed as _fe  # noqa: F401  # pyright: ignore[reportUnusedImport]

        return True
    except ImportError:
        pass
    try:
        import sentence_transformers as _st  # noqa: F401  # pyright: ignore[reportUnusedImport]

        return True
    except ImportError:
        return False


def _sparse_confidence(bundle: ContextBundle, *, window: int) -> tuple[float, float, int]:
    """Return (top_score, relative_rank1_gap, candidate_count) over the top window.

    The relative gap between the first and second scores is the cheap diffuseness
    signal: a large gap means the top result clearly stands out (confident
    sparse), a small gap means the top scores are diffuse.
    """
    scores = [chunk.final_score for chunk in bundle.chunks[:window]]
    count = len(scores)
    top = scores[0] if scores else 0.0
    if count >= 2 and top > 0:
        relative_gap = (scores[0] - scores[1]) / top
    elif count == 1:
        relative_gap = 1.0
    else:
        relative_gap = 0.0
    return top, relative_gap, count


def _should_expand_dense(
    policy: TaskAwarePolicy, *, relative_gap: float, candidate_count: int
) -> bool:
    """Decide whether the conditional dense pass should run for a sparse-first policy.

    Both conditional triggers — ``LOW_SPARSE_CONFIDENCE`` (pl_to_pl) and
    ``DIFFUSE_TOP_SCORES`` (mixed) — share the same cheap signal: a small rank-1
    relative gap means the top result does not clearly dominate, which is the
    proxy for both "sparse confidence is low" and "top scores are diffuse". An
    empty candidate set always expands. The absolute top score is recorded as a
    diagnostic provenance signal but does not drive this decision.
    """
    if candidate_count == 0:
        return True
    return relative_gap < policy.diffuse_gap_threshold


def _task_aware_index_config(policy: TaskAwarePolicy, *, vector: bool) -> IndexConfig:
    """Bounded IndexConfig for a task-aware pass.

    The cross-encoder reranker is disabled; confidence-aware fusion is governed by
    the default retrieval policy when ``vector`` is True.
    """
    return IndexConfig(
        vector=vector,
        quantize_vectors=False,
        rerank=False,
        rerank_candidate_limit=policy.rerank_candidate_limit,
    )


def run_archex_query_task_aware(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Benchmark-only task-aware lane: route sparse/dense retrieval by modality + budget.

    Classifies the query modality and budget tier, derives a deterministic
    TaskAwarePolicy, and runs a sparse-first BM25 pass (or a bounded hybrid pass
    for nl_to_pl). For pl_to_pl/mixed it may run one targeted dense expansion pass
    when the top sparse scores are diffuse. The cross-encoder reranker is never
    run; confidence-aware fusion runs only on the hybrid/dense pass. Work is
    bounded to at most two retrieval passes. Strategy provenance records modality,
    budget tier, the routing decision, candidate caps, skipped expensive steps,
    and whether fusion ran. Product defaults are unchanged.
    """
    strategy = Strategy.ARCHEX_QUERY_TASK_AWARE
    classification = classify_query(task.question, task.token_budget)
    policy = policy_for(classification)
    vector_available = _vector_dependencies_available()

    t0 = time.perf_counter()
    initial_vector = policy.use_vector_initial and vector_available
    if policy.use_vector_initial and not vector_available:
        initial_pass = "bm25_only(vector_unavailable_fallback)"
    else:
        initial_pass = "hybrid" if initial_vector else "bm25_only"
    bundle, effective_config, timing = _query_bundle(
        task,
        repo_path,
        strategy=strategy,
        index_config=_task_aware_index_config(policy, vector=initial_vector),
        cache=True,
    )
    initial_cached = timing.cached
    initial_cache_state = _cache_state(timing)

    top_score, relative_gap, candidate_count = _sparse_confidence(
        bundle, window=policy.candidate_cap
    )

    if policy.dense_trigger is DenseTrigger.NEVER:
        dense_expansion = "disabled"
    elif policy.dense_trigger is DenseTrigger.ALWAYS:
        dense_expansion = "initial_hybrid" if initial_vector else "skipped:vector_unavailable"
    elif not _should_expand_dense(
        policy, relative_gap=relative_gap, candidate_count=candidate_count
    ):
        dense_expansion = "skipped:confident_sparse"
    elif not vector_available:
        dense_expansion = "skipped:vector_unavailable"
    else:
        bundle, effective_config, timing = _query_bundle(
            task,
            repo_path,
            strategy=strategy,
            index_config=_task_aware_index_config(policy, vector=True),
            cache=True,
        )
        # Report cache state across both passes: warm only if neither built work.
        timing.cached = initial_cached and timing.cached
        dense_expansion = "ran"

    wall_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Strategy %s for %s: modality=%s tier=%s dense=%s wall=%.1fms",
        strategy.value,
        task.task_id,
        classification.modality.value,
        classification.budget_tier.value,
        dense_expansion,
        wall_ms,
    )
    result = _assemble_query_result(
        task,
        repo_path,
        strategy=strategy,
        index_config=effective_config,
        bundle=bundle,
        timing=timing,
        wall_ms=wall_ms,
        include_completion=True,
        measure_freshness=False,
    )
    routing_decision = (
        f"{initial_pass}+dense_expansion" if dense_expansion == "ran" else initial_pass
    )
    fusion_used = initial_vector or dense_expansion == "ran"
    runtime_provenance = {
        "routing_decision": routing_decision,
        "initial_pass": initial_pass,
        "dense_expansion": dense_expansion,
        "fusion_used": str(fusion_used).lower(),
        "sparse_top_score": f"{top_score:.4f}",
        "sparse_relative_gap": f"{relative_gap:.4f}",
        "sparse_candidate_count": str(candidate_count),
        "initial_cache_state": initial_cache_state,
        "vector_dependencies_available": str(vector_available).lower(),
    }
    result.provenance = classification.to_provenance() | policy.to_provenance() | runtime_provenance
    return result


class StrategyRegistry:
    """Registry for benchmark strategy runners with entry-point support."""

    def __init__(self) -> None:
        self._runners: dict[str, StrategyRunner] = {}
        self._entry_points_loaded: bool = False
        self._entry_points_strict: bool = False

    def register(self, name: str, runner: StrategyRunner) -> None:
        """Register a strategy runner by name."""
        self._runners[name] = runner

    def get(self, strategy: Strategy | str) -> StrategyRunner | None:
        """Return the runner for a strategy, or None."""
        key = strategy.value if isinstance(strategy, Strategy) else strategy
        return self._runners.get(key)

    @property
    def strategy_names(self) -> list[str]:
        """Return sorted list of registered strategy names."""
        return sorted(self._runners.keys())

    def load_entry_points(
        self,
        group: str = "archex.benchmark_strategies",
        strict: bool = False,
    ) -> None:
        """Load strategy runners from installed entry points."""
        if self._entry_points_loaded and (not strict or self._entry_points_strict):
            return
        eps = sorted(importlib.metadata.entry_points(group=group), key=lambda ep: ep.name)
        for ep in eps:
            try:
                runner = ep.load()
                self._runners[ep.name] = runner
                logger.info("Loaded strategy %s from entry point", ep.name)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                if strict:
                    raise ConfigError(
                        f"Failed to load strategy entry point {ep.name!r}: {exc}"
                    ) from exc
                logger.warning("Failed to load strategy entry point %s: %s", ep.name, exc)
        self._entry_points_loaded = True
        self._entry_points_strict = strict


def _run_external_mcp_strategy(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    from archex.benchmark.external_mcp import run_external_mcp

    return run_external_mcp(task, repo_path)


default_strategy_registry = StrategyRegistry()
default_strategy_registry.register(Strategy.RAW_FILES.value, run_raw_files)
default_strategy_registry.register(Strategy.RAW_GREPPED.value, run_raw_grepped)
default_strategy_registry.register(Strategy.RAW_RIPGREP.value, run_raw_ripgrep)
default_strategy_registry.register(Strategy.ARCHEX_QUERY.value, run_archex_query)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_PPR.value, run_archex_query_ppr)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_CHURN.value, run_archex_query_churn)
default_strategy_registry.register(Strategy.ARCHEX_SCOUT_FETCH.value, run_archex_scout_fetch)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_VECTOR.value, run_archex_query_vector)
default_strategy_registry.register(Strategy.SURROGATE_VECTOR.value, run_surrogate_vector)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_FUSION.value, run_archex_query_fusion)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_HYBRID.value, run_archex_query_hybrid)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT.value,
    run_archex_query_hybrid_quantized_4bit,
)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_FUSION_RERANK.value, run_archex_query_fusion_rerank
)
default_strategy_registry.register(Strategy.CROSS_LAYER_FUSION.value, run_cross_layer_fusion)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_TASK_AWARE.value, run_archex_query_task_aware
)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_COMPRESSED.value, run_archex_query_compressed
)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED.value, run_archex_query_efficiency_packed
)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_DUAL_TRANSFORM.value, run_archex_query_dual_transform
)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_BOUNDED_RERANK.value, run_archex_query_bounded_rerank
)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR.value, run_archex_query_summary_sidecar
)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP.value, run_archex_query_graph_multihop
)
default_strategy_registry.register(Strategy.EXTERNAL_MCP.value, _run_external_mcp_strategy)
