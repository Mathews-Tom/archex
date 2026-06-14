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
from datetime import UTC, datetime
from pathlib import Path

from archex.benchmark.models import (
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
)
from archex.cache import CacheManager
from archex.exceptions import ArchexError, ConfigError
from archex.models import ChunkerName, IndexConfig, PipelineTiming, RepoSource
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


def compute_bundle_completion_penalty(
    repo_path: Path,
    result_files: set[str],
    expected_files: list[str],
) -> tuple[int, list[str]]:
    """Return extra oracle-file tokens needed after an incomplete result bundle."""
    missing_files = [path for path in expected_files if path not in result_files]
    return count_file_tokens(repo_path, missing_files), missing_files


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
        result_files=list(task.expected_files),
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
    completion_tokens, completion_files = compute_bundle_completion_penalty(
        repo_path, matched_files_seen, task.expected_files
    )
    tokens_with_completion = tokens + completion_tokens

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_GREPPED,
        tokens_total=tokens,
        tokens_input=tokens,
        tokens_output=tokens,
        token_efficiency=compute_token_efficiency(tokens, tokens),
        result_files=_deduplicate_ranked(matched_files_ordered),
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
        "expansion_reason_counts",
        "expanded_file_reasons",
        "chunker",
        "bm25_chunker",
        "vector_chunker",
        "index_chunk_count",
        "mean_chunk_tokens",
    )

    chunker: ChunkerName
    bm25_chunker: ChunkerName | None
    vector_chunker: ChunkerName | None
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
        expanded_file_reasons: dict[str, list[str]],
        chunker: ChunkerName,
        bm25_chunker: ChunkerName | None,
        vector_chunker: ChunkerName | None,
        index_chunk_count: int,
        mean_chunk_tokens: float,
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
        self.expanded_file_reasons = expanded_file_reasons
        self.chunker = chunker
        self.bm25_chunker = bm25_chunker
        self.vector_chunker = vector_chunker
        self.index_chunk_count = index_chunk_count
        self.mean_chunk_tokens = mean_chunk_tokens


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
    bm25_chunker = meta.bm25_chunker
    vector_chunker = meta.vector_chunker
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
            bm25_chunker=bm25_chunker,
            vector_chunker=vector_chunker,
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
        bm25_chunker=bm25_chunker,
        vector_chunker=vector_chunker,
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
        Strategy.CROSS_LAYER_FUSION,
    }
)

_DUAL_LEG_STRATEGIES = frozenset(
    {
        Strategy.ARCHEX_QUERY_FUSION,
        Strategy.ARCHEX_QUERY_FUSION_RERANK,
    }
)


def benchmark_dual_leg_enabled(
    strategy: Strategy,
    options: BenchmarkRetrievalOptions,
) -> bool:
    if strategy not in _DUAL_LEG_STRATEGIES or not options.dual_leg_orchestration or options.splade:
        return False
    return _chunker_for_strategy(Strategy.ARCHEX_QUERY, options) != _chunker_for_strategy(
        strategy,
        options,
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
    for candidate in repo_path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in {".py", ".js", ".ts", ".go", ".rs"}:
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


def run_archex_query(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """archex query strategy: use BM25-based retrieval."""
    from archex.api import query
    from archex.models import Config

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path, strategy=Strategy.ARCHEX_QUERY)
    config = Config(
        cache=benchmark_cache_enabled(default=False),
        languages=task.languages,
    )
    index_config = benchmark_index_config(
        IndexConfig(vector=False),
        strategy=Strategy.ARCHEX_QUERY,
    )

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    unique_ranked = _deduplicate_ranked(ranked_files)
    result_files = set(unique_ranked)
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    af = _archex_fields(bundle, task, repo_path)
    completion_tokens, completion_files = compute_bundle_completion_penalty(
        repo_path, result_files, task.expected_files
    )
    if current_benchmark_retrieval_options().freshness:
        freshness_latency_ms, freshness_correct = measure_archex_freshness(task, repo_path)
    else:
        freshness_latency_ms, freshness_correct = (0.0, False)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY,
        tokens_total=bundle.token_count,
        tokens_input=af.tokens_input,
        tokens_output=af.tokens_output,
        token_efficiency=af.token_efficiency,
        result_files=unique_ranked,
        bundle_completion_tokens=completion_tokens,
        bundle_completion_files=completion_files,
        token_efficiency_with_completion=compute_token_efficiency(
            af.tokens_output + completion_tokens, af.tokens_input + completion_tokens
        ),
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
        expansion_reason_counts=af.expansion_reason_counts,
        expanded_file_reasons=af.expanded_file_reasons,
        chunker=af.chunker,
        index_chunk_count=af.index_chunk_count,
        mean_chunk_tokens=af.mean_chunk_tokens,
        category=task.category,
        vector_mode=index_config.vector_mode,
        cache_state=_cache_state(timing),
        freshness_latency_ms=freshness_latency_ms,
        freshness_measured=current_benchmark_retrieval_options().freshness,
        freshness_correct=freshness_correct,
    )


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
    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_SCOUT_FETCH,
        tokens_total=tokens_total,
        tokens_input=tokens_input,
        tokens_output=tokens_total,
        token_efficiency=compute_token_efficiency(tokens_total, tokens_input),
        result_files=unique_ranked,
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


def run_archex_query_vector(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Pure vector retrieval strategy: vector search without BM25."""
    from archex.api import query
    from archex.models import Config

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path, strategy=Strategy.ARCHEX_QUERY_VECTOR)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(
        IndexConfig(bm25=False, vector=True),
        strategy=Strategy.ARCHEX_QUERY_VECTOR,
    )

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
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
        expansion_reason_counts=af.expansion_reason_counts,
        expanded_file_reasons=af.expanded_file_reasons,
        chunker=af.chunker,
        index_chunk_count=af.index_chunk_count,
        mean_chunk_tokens=af.mean_chunk_tokens,
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
    source = benchmark_repo_source(task, repo_path, strategy=Strategy.SURROGATE_VECTOR)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(
        IndexConfig(
            bm25=False,
            vector=True,
            embedder=current_benchmark_retrieval_options().embedder,
            vector_mode=VectorMode.SURROGATE,
            retrieval_policy=RetrievalPolicy.VECTOR_ONLY,
        ),
        strategy=Strategy.SURROGATE_VECTOR,
    )

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
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
        expansion_reason_counts=af.expansion_reason_counts,
        expanded_file_reasons=af.expanded_file_reasons,
        chunker=af.chunker,
        index_chunk_count=af.index_chunk_count,
        mean_chunk_tokens=af.mean_chunk_tokens,
        category=task.category,
        vector_mode=index_config.vector_mode,
        surrogate_version=index_config.surrogate_version,
        cache_state=_cache_state(timing),
    )


def _run_benchmark_fusion_query(
    task: BenchmarkTask,
    repo_path: Path,
    *,
    strategy: Strategy,
    rerank: bool,
    timing: PipelineTiming,
):
    from archex.api import query, query_dual_leg_benchmark
    from archex.models import Config

    options = current_benchmark_retrieval_options()
    config = Config(cache=True, languages=task.languages)
    if benchmark_dual_leg_enabled(strategy, options):
        bm25_source = benchmark_repo_source(task, repo_path, strategy=Strategy.ARCHEX_QUERY)
        vector_source = benchmark_repo_source(task, repo_path, strategy=strategy)
        bm25_index_config = benchmark_index_config(
            IndexConfig(vector=False),
            strategy=Strategy.ARCHEX_QUERY,
        )
        vector_index_config = benchmark_index_config(
            IndexConfig(vector=True, rerank=rerank),
            strategy=strategy,
        )
        return query_dual_leg_benchmark(
            bm25_source=bm25_source,
            vector_source=vector_source,
            question=task.question,
            token_budget=task.token_budget,
            config=config,
            bm25_index_config=bm25_index_config,
            vector_index_config=vector_index_config,
            timing=timing,
            file_stage_orchestration=options.file_stage_orchestration,
            direct_file_preservation=options.direct_file_preservation,
        )
    source = benchmark_repo_source(task, repo_path, strategy=strategy)
    index_config = benchmark_index_config(
        IndexConfig(vector=True, rerank=rerank),
        strategy=strategy,
    )
    return query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
        config=config,
        index_config=index_config,
        timing=timing,
    )


def _fusion_result_index_config(*, rerank: bool) -> IndexConfig:
    strategy = Strategy.ARCHEX_QUERY_FUSION_RERANK if rerank else Strategy.ARCHEX_QUERY_FUSION
    return benchmark_index_config(
        IndexConfig(vector=True, rerank=rerank),
        strategy=strategy,
    )


def run_archex_query_fusion(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Full fusion strategy: BM25 + independent vector + confidence-aware RRF."""
    t0 = time.perf_counter()
    timing = PipelineTiming()
    bundle = _run_benchmark_fusion_query(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_FUSION,
        rerank=False,
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
        expansion_reason_counts=af.expansion_reason_counts,
        expanded_file_reasons=af.expanded_file_reasons,
        category=task.category,
        vector_mode=_fusion_result_index_config(rerank=False).vector_mode,
        cache_state=_cache_state(timing),
        chunker=af.chunker,
        bm25_chunker=af.bm25_chunker,
        vector_chunker=af.vector_chunker,
        index_chunk_count=af.index_chunk_count,
        mean_chunk_tokens=af.mean_chunk_tokens,
    )


def run_archex_query_fusion_rerank(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Fusion strategy with cross-encoder reranking: BM25 + vector + rerank."""
    t0 = time.perf_counter()
    timing = PipelineTiming()
    bundle = _run_benchmark_fusion_query(
        task,
        repo_path,
        strategy=Strategy.ARCHEX_QUERY_FUSION_RERANK,
        rerank=True,
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
        expansion_reason_counts=af.expansion_reason_counts,
        expanded_file_reasons=af.expanded_file_reasons,
        category=task.category,
        vector_mode=_fusion_result_index_config(rerank=True).vector_mode,
        cache_state=_cache_state(timing),
        chunker=af.chunker,
        bm25_chunker=af.bm25_chunker,
        vector_chunker=af.vector_chunker,
        index_chunk_count=af.index_chunk_count,
        mean_chunk_tokens=af.mean_chunk_tokens,
    )


def run_cross_layer_fusion(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """BM25 over raw chunks plus vector retrieval over surrogates."""
    from archex.api import query
    from archex.models import Config, RetrievalPolicy, VectorMode

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = benchmark_repo_source(task, repo_path, strategy=Strategy.CROSS_LAYER_FUSION)
    config = Config(cache=True, languages=task.languages)
    index_config = benchmark_index_config(
        IndexConfig(
            vector=True,
            embedder=current_benchmark_retrieval_options().embedder,
            vector_mode=VectorMode.SURROGATE,
            retrieval_policy=RetrievalPolicy.CROSS_LAYER,
        ),
        strategy=Strategy.CROSS_LAYER_FUSION,
    )

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
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
        expansion_reason_counts=af.expansion_reason_counts,
        expanded_file_reasons=af.expanded_file_reasons,
        chunker=af.chunker,
        index_chunk_count=af.index_chunk_count,
        mean_chunk_tokens=af.mean_chunk_tokens,
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


def _run_external_mcp_strategy(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    from archex.benchmark.external_mcp import run_external_mcp

    return run_external_mcp(task, repo_path)


default_strategy_registry = StrategyRegistry()
default_strategy_registry.register(Strategy.RAW_FILES.value, run_raw_files)
default_strategy_registry.register(Strategy.RAW_GREPPED.value, run_raw_grepped)
default_strategy_registry.register(Strategy.ARCHEX_QUERY.value, run_archex_query)
default_strategy_registry.register(Strategy.ARCHEX_SCOUT_FETCH.value, run_archex_scout_fetch)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_VECTOR.value, run_archex_query_vector)
default_strategy_registry.register(Strategy.SURROGATE_VECTOR.value, run_surrogate_vector)
default_strategy_registry.register(Strategy.ARCHEX_QUERY_FUSION.value, run_archex_query_fusion)
default_strategy_registry.register(
    Strategy.ARCHEX_QUERY_FUSION_RERANK.value, run_archex_query_fusion_rerank
)
default_strategy_registry.register(Strategy.CROSS_LAYER_FUSION.value, run_cross_layer_fusion)
default_strategy_registry.register(Strategy.ARCHEX_SYMBOL_LOOKUP.value, run_archex_symbol_lookup)
default_strategy_registry.register(Strategy.EXTERNAL_MCP.value, _run_external_mcp_strategy)
