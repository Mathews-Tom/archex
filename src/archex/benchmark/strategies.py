"""Individual strategy implementations for benchmarking."""

from __future__ import annotations

import importlib.metadata
import logging
import math
import re
import subprocess
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path

from archex.benchmark.models import (
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
)
from archex.cache import CacheManager
from archex.exceptions import ConfigError
from archex.models import IndexConfig, PipelineTiming, RepoSource
from archex.reporting import count_tokens

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


def run_raw_files(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Baseline strategy: read all expected files, count tokens."""
    t0 = time.perf_counter()
    tokens = count_file_tokens(repo_path, task.expected_files)
    wall_ms = (time.perf_counter() - t0) * 1000

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_FILES,
        tokens_total=tokens,
        tokens_input=tokens,
        tokens_output=tokens,
        token_efficiency=compute_token_efficiency(tokens, tokens),
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
                # Strip leading ./ from grep output
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

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_GREPPED,
        tokens_total=tokens,
        tokens_input=tokens,
        tokens_output=tokens,
        token_efficiency=compute_token_efficiency(tokens, tokens),
        tokens_raw_baseline=tokens_raw_baseline,
        tool_calls=len(keywords),
        files_accessed=len(matched_files_seen),
        recall=recall,
        precision=precision,
        f1_score=f1,
        mrr=mrr_val,
        ndcg=ndcg_val,
        map_score=map_val,
        savings_vs_raw=0.0,  # backfilled by runner
        wall_time_ms=wall_ms,
        cached=False,
        timestamp=now_iso(),
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
    )

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
    )


def _cache_state(timing: PipelineTiming) -> str:
    return "warm" if timing.cached else "cold"


def current_benchmark_retrieval_options() -> BenchmarkRetrievalOptions:
    return _BENCHMARK_RETRIEVAL_OPTIONS.get() or BenchmarkRetrievalOptions()


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


def _retrieval_cache_suffix(options: BenchmarkRetrievalOptions) -> str:
    enabled: list[str] = [f"embedder={_embedder_cache_identity(options.embedder)}"]
    if options.splade:
        enabled.append("splade")
    if options.module_prefilter:
        enabled.append("module-prefilter")
    return "+".join(enabled)


def _corpus_cache_suffix(task: BenchmarkTask) -> str:
    if not task.include_paths:
        return ""
    return "scope=" + "|".join(sorted(task.include_paths))


def benchmark_index_config(index_config: IndexConfig) -> IndexConfig:
    options = current_benchmark_retrieval_options()
    updates: dict[str, bool | str] = {}
    if index_config.vector:
        updates["embedder"] = options.embedder
    if options.splade:
        updates["splade"] = True
    if options.module_prefilter and index_config.bm25:
        updates["module_prefilter"] = True
    if not updates:
        return index_config
    return index_config.model_copy(update=updates)


def benchmark_cache_enabled(default: bool) -> bool:
    options = current_benchmark_retrieval_options()
    return default or options.splade or options.module_prefilter


def benchmark_repo_source(task: BenchmarkTask, repo_path: Path) -> RepoSource:
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
            _retrieval_cache_suffix(current_benchmark_retrieval_options()),
        )
        if suffix
    ]
    if suffixes:
        stable_identity = f"{stable_identity}#{'+'.join(suffixes)}"
    return RepoSource(
        local_path=str(repo_path),
        stable_identity=stable_identity,
    )


def run_archex_query(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """archex query strategy: use BM25-based retrieval."""
    from archex.api import query
    from archex.models import Config

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path)
    config = Config(
        cache=benchmark_cache_enabled(default=False),
        languages=task.languages,
    )
    index_config = benchmark_index_config(IndexConfig(vector=False))

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    result_files = set(_deduplicate_ranked(ranked_files))
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    af = _archex_fields(bundle, task, repo_path)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY,
        tokens_total=bundle.token_count,
        tokens_input=af.tokens_input,
        tokens_output=af.tokens_output,
        token_efficiency=af.token_efficiency,
        tokens_raw_baseline=af.tokens_raw_baseline,
        symbol_recall=af.symbol_recall,
        tool_calls=1,
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
        f1_score=f1,
        mrr=mrr_val,
        ndcg=ndcg_val,
        map_score=map_val,
        savings_vs_raw=0.0,  # backfilled by runner
        wall_time_ms=wall_ms,
        cached=timing.cached,
        timing=timing,
        timestamp=now_iso(),
        unique_ranked_files=af.unique_ranked_files,
        seed_files=af.seed_files,
        expanded_files=af.expanded_files,
        expansion_ratio=af.expansion_ratio,
        seed_recall=af.seed_recall,
        seed_precision=af.seed_precision,
        expansion_eligible_seeds=af.expansion_eligible_seeds,
        expansion_candidates_found=af.expansion_candidates_found,
        expansion_import_neighbor_edges=af.expansion_import_neighbor_edges,
        expansion_same_module_candidates=af.expansion_same_module_candidates,
        expansion_hub_candidates=af.expansion_hub_candidates,
        expansion_test_candidates_skipped=af.expansion_test_candidates_skipped,
        expansion_zero_candidate_reason=af.expansion_zero_candidate_reason,
        category=task.category,
        vector_mode=index_config.vector_mode,
        cache_state=_cache_state(timing),
    )


def run_archex_query_vector(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Pure vector retrieval strategy: vector search without BM25."""
    from archex.api import query
    from archex.models import Config

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(IndexConfig(bm25=False, vector=True))

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    result_files = set(_deduplicate_ranked(ranked_files))
    wall_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Strategy %s for %s: cached=%s, wall_time=%.1fms",
        "archex_query_vector",
        task.task_id,
        timing.cached,
        wall_ms,
    )
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    af = _archex_fields(bundle, task, repo_path)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY_VECTOR,
        tokens_total=bundle.token_count,
        tokens_input=af.tokens_input,
        tokens_output=af.tokens_output,
        token_efficiency=af.token_efficiency,
        tokens_raw_baseline=af.tokens_raw_baseline,
        symbol_recall=af.symbol_recall,
        tool_calls=1,
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
        f1_score=f1,
        mrr=mrr_val,
        ndcg=ndcg_val,
        map_score=map_val,
        savings_vs_raw=0.0,  # backfilled by runner
        wall_time_ms=wall_ms,
        cached=timing.cached,
        timing=timing,
        timestamp=now_iso(),
        unique_ranked_files=af.unique_ranked_files,
        seed_files=af.seed_files,
        expanded_files=af.expanded_files,
        expansion_ratio=af.expansion_ratio,
        seed_recall=af.seed_recall,
        seed_precision=af.seed_precision,
        expansion_eligible_seeds=af.expansion_eligible_seeds,
        expansion_candidates_found=af.expansion_candidates_found,
        expansion_import_neighbor_edges=af.expansion_import_neighbor_edges,
        expansion_same_module_candidates=af.expansion_same_module_candidates,
        expansion_hub_candidates=af.expansion_hub_candidates,
        expansion_test_candidates_skipped=af.expansion_test_candidates_skipped,
        expansion_zero_candidate_reason=af.expansion_zero_candidate_reason,
        category=task.category,
        vector_mode=index_config.vector_mode,
        cache_state=_cache_state(timing),
    )


def run_surrogate_vector(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Pure surrogate-vector retrieval strategy."""
    from archex.api import query
    from archex.models import Config, RetrievalPolicy, VectorMode

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(
        IndexConfig(
            bm25=False,
            vector=True,
            embedder=current_benchmark_retrieval_options().embedder,
            vector_mode=VectorMode.SURROGATE,
            retrieval_policy=RetrievalPolicy.VECTOR_ONLY,
        )
    )

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    result_files = set(_deduplicate_ranked(ranked_files))
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    af = _archex_fields(bundle, task, repo_path)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.SURROGATE_VECTOR,
        tokens_total=bundle.token_count,
        tokens_input=af.tokens_input,
        tokens_output=af.tokens_output,
        token_efficiency=af.token_efficiency,
        tokens_raw_baseline=af.tokens_raw_baseline,
        symbol_recall=af.symbol_recall,
        tool_calls=1,
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
        unique_ranked_files=af.unique_ranked_files,
        seed_files=af.seed_files,
        expanded_files=af.expanded_files,
        expansion_ratio=af.expansion_ratio,
        seed_recall=af.seed_recall,
        seed_precision=af.seed_precision,
        expansion_eligible_seeds=af.expansion_eligible_seeds,
        expansion_candidates_found=af.expansion_candidates_found,
        expansion_import_neighbor_edges=af.expansion_import_neighbor_edges,
        expansion_same_module_candidates=af.expansion_same_module_candidates,
        expansion_hub_candidates=af.expansion_hub_candidates,
        expansion_test_candidates_skipped=af.expansion_test_candidates_skipped,
        expansion_zero_candidate_reason=af.expansion_zero_candidate_reason,
        category=task.category,
        vector_mode=index_config.vector_mode,
        surrogate_version=index_config.surrogate_version,
        cache_state=_cache_state(timing),
    )


def run_archex_query_fusion(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Full fusion strategy: BM25 + independent vector + confidence-aware RRF."""
    from archex.api import query
    from archex.models import Config

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(IndexConfig(vector=True))

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    result_files = set(_deduplicate_ranked(ranked_files))
    wall_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Strategy %s for %s: cached=%s, wall_time=%.1fms",
        "archex_query_fusion",
        task.task_id,
        timing.cached,
        wall_ms,
    )
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    af = _archex_fields(bundle, task, repo_path)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY_FUSION,
        tokens_total=bundle.token_count,
        tokens_input=af.tokens_input,
        tokens_output=af.tokens_output,
        token_efficiency=af.token_efficiency,
        tokens_raw_baseline=af.tokens_raw_baseline,
        symbol_recall=af.symbol_recall,
        tool_calls=1,
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
        f1_score=f1,
        mrr=mrr_val,
        ndcg=ndcg_val,
        map_score=map_val,
        savings_vs_raw=0.0,  # backfilled by runner
        wall_time_ms=wall_ms,
        cached=timing.cached,
        timing=timing,
        timestamp=now_iso(),
        unique_ranked_files=af.unique_ranked_files,
        seed_files=af.seed_files,
        expanded_files=af.expanded_files,
        expansion_ratio=af.expansion_ratio,
        seed_recall=af.seed_recall,
        seed_precision=af.seed_precision,
        expansion_eligible_seeds=af.expansion_eligible_seeds,
        expansion_candidates_found=af.expansion_candidates_found,
        expansion_import_neighbor_edges=af.expansion_import_neighbor_edges,
        expansion_same_module_candidates=af.expansion_same_module_candidates,
        expansion_hub_candidates=af.expansion_hub_candidates,
        expansion_test_candidates_skipped=af.expansion_test_candidates_skipped,
        expansion_zero_candidate_reason=af.expansion_zero_candidate_reason,
        category=task.category,
        vector_mode=index_config.vector_mode,
        cache_state=_cache_state(timing),
    )


def run_archex_query_fusion_rerank(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Fusion strategy with cross-encoder reranking: BM25 + vector + rerank."""
    from archex.api import query
    from archex.models import Config

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(IndexConfig(vector=True, rerank=True))

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    result_files = set(_deduplicate_ranked(ranked_files))
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    af = _archex_fields(bundle, task, repo_path)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY_FUSION_RERANK,
        tokens_total=bundle.token_count,
        tokens_input=af.tokens_input,
        tokens_output=af.tokens_output,
        token_efficiency=af.token_efficiency,
        tokens_raw_baseline=af.tokens_raw_baseline,
        symbol_recall=af.symbol_recall,
        tool_calls=1,
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
        unique_ranked_files=af.unique_ranked_files,
        seed_files=af.seed_files,
        expanded_files=af.expanded_files,
        expansion_ratio=af.expansion_ratio,
        seed_recall=af.seed_recall,
        seed_precision=af.seed_precision,
        expansion_eligible_seeds=af.expansion_eligible_seeds,
        expansion_candidates_found=af.expansion_candidates_found,
        expansion_import_neighbor_edges=af.expansion_import_neighbor_edges,
        expansion_same_module_candidates=af.expansion_same_module_candidates,
        expansion_hub_candidates=af.expansion_hub_candidates,
        expansion_test_candidates_skipped=af.expansion_test_candidates_skipped,
        expansion_zero_candidate_reason=af.expansion_zero_candidate_reason,
        category=task.category,
        vector_mode=index_config.vector_mode,
        cache_state=_cache_state(timing),
    )


def run_cross_layer_fusion(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """BM25 over raw chunks plus vector retrieval over surrogates."""
    from archex.api import query
    from archex.models import Config, RetrievalPolicy, VectorMode

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(
        IndexConfig(
            vector=True,
            embedder=current_benchmark_retrieval_options().embedder,
            vector_mode=VectorMode.SURROGATE,
            retrieval_policy=RetrievalPolicy.CROSS_LAYER,
        )
    )

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    result_files = set(_deduplicate_ranked(ranked_files))
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    af = _archex_fields(bundle, task, repo_path)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.CROSS_LAYER_FUSION,
        tokens_total=bundle.token_count,
        tokens_input=af.tokens_input,
        tokens_output=af.tokens_output,
        token_efficiency=af.token_efficiency,
        tokens_raw_baseline=af.tokens_raw_baseline,
        symbol_recall=af.symbol_recall,
        tool_calls=1,
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
        unique_ranked_files=af.unique_ranked_files,
        seed_files=af.seed_files,
        expanded_files=af.expanded_files,
        expansion_ratio=af.expansion_ratio,
        seed_recall=af.seed_recall,
        seed_precision=af.seed_precision,
        expansion_eligible_seeds=af.expansion_eligible_seeds,
        expansion_candidates_found=af.expansion_candidates_found,
        expansion_import_neighbor_edges=af.expansion_import_neighbor_edges,
        expansion_same_module_candidates=af.expansion_same_module_candidates,
        expansion_hub_candidates=af.expansion_hub_candidates,
        expansion_test_candidates_skipped=af.expansion_test_candidates_skipped,
        expansion_zero_candidate_reason=af.expansion_zero_candidate_reason,
        category=task.category,
        vector_mode=index_config.vector_mode,
        surrogate_version=index_config.surrogate_version,
        cache_state=_cache_state(timing),
    )


def run_archex_symbol_lookup(
    task: BenchmarkTask,
    repo_path: Path,
) -> BenchmarkResult:
    """Symbol lookup strategy — requires Enhancement 1+2."""
    raise NotImplementedError("Requires Enhancement 1+2: Stable IDs + Precision Tools")


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


default_strategy_registry = StrategyRegistry()
default_strategy_registry.register(Strategy.RAW_FILES.value, run_raw_files)
default_strategy_registry.register(Strategy.RAW_GREPPED.value, run_raw_grepped)
default_strategy_registry.register(Strategy.ARCHEX_QUERY.value, run_archex_query)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_VECTOR.value, run_archex_query_vector)
default_strategy_registry.register(Strategy.SURROGATE_VECTOR.value, run_surrogate_vector)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_FUSION.value, run_archex_query_fusion)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_FUSION_RERANK.value, run_archex_query_fusion_rerank
)
default_strategy_registry.register(Strategy.CROSS_LAYER_FUSION.value, run_cross_layer_fusion)
default_strategy_registry.register(Strategy.ARCHEX_SYMBOL_LOOKUP.value, run_archex_symbol_lookup)
