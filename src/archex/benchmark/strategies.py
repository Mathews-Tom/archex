"""Individual strategy implementations for benchmarking."""

from __future__ import annotations

import importlib.metadata
import logging
import math
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from archex.benchmark.models import BenchmarkResult, BenchmarkTask, Strategy
from archex.exceptions import ConfigError
from archex.models import PipelineTiming, RepoSource
from archex.reporting import count_tokens

logger = logging.getLogger(__name__)

StrategyRunner = Callable[[BenchmarkTask, Path], BenchmarkResult]

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


def compute_f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision."""
    if recall + precision == 0.0:
        return 0.0
    return 2 * (recall * precision) / (recall + precision)


def compute_mrr(ranked_files: list[str], expected_files: list[str]) -> float:
    """Mean reciprocal rank: reciprocal of the rank of the first expected file found."""
    expected_set = set(expected_files)
    for i, f in enumerate(ranked_files, 1):
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
    """Normalized discounted cumulative gain at k."""
    if not expected_files:
        return 0.0
    expected_set = set(expected_files)
    # DCG
    dcg = 0.0
    for i, f in enumerate(ranked_files[:k]):
        rel = 1.0 if f in expected_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    # Ideal DCG
    ideal_count = min(len(expected_files), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_map(ranked_files: list[str], expected_files: list[str]) -> float:
    """Mean average precision."""
    if not expected_files:
        return 0.0
    expected_set = set(expected_files)
    hits = 0
    sum_precision = 0.0
    for i, f in enumerate(ranked_files, 1):
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
        token_efficiency=1.0,
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
        token_efficiency=1.0 if tokens > 0 else 0.0,
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


def _archex_token_fields(
    bundle: object,
    task: BenchmarkTask,
    repo_path: Path,
) -> tuple[int, int, float, int, float]:
    """Compute token efficiency fields from a ContextBundle and task."""
    from archex.models import ContextBundle

    assert isinstance(bundle, ContextBundle)
    unique_files = list({c.chunk.file_path for c in bundle.chunks})
    tokens_input = count_file_tokens(repo_path, unique_files)
    tokens_output = bundle.token_count
    token_efficiency = tokens_output / tokens_input if tokens_input > 0 else 0.0
    tokens_raw_baseline = count_file_tokens(repo_path, task.expected_files)
    result_symbols = {c.chunk.symbol_name for c in bundle.chunks if c.chunk.symbol_name}
    symbol_recall = compute_symbol_recall(result_symbols, task.expected_symbols)
    return tokens_input, tokens_output, token_efficiency, tokens_raw_baseline, symbol_recall


def run_archex_query(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """archex query strategy: use BM25-based retrieval."""
    from archex.api import query
    from archex.models import Config, IndexConfig

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = RepoSource(local_path=str(repo_path))
    config = Config(cache=False, languages=task.languages)
    index_config = IndexConfig(vector=False)

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    result_files = {c.chunk.file_path for c in bundle.chunks}
    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    tokens_input, tokens_output, token_efficiency, tokens_raw_baseline, sym_recall = (
        _archex_token_fields(bundle, task, repo_path)
    )

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY,
        tokens_total=bundle.token_count,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        token_efficiency=token_efficiency,
        tokens_raw_baseline=tokens_raw_baseline,
        symbol_recall=sym_recall,
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
    )


def run_archex_query_hybrid(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """archex hybrid query strategy: BM25 + vector retrieval."""
    from archex.api import query
    from archex.models import Config, IndexConfig

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = RepoSource(local_path=str(repo_path))
    config = Config(cache=False)
    index_config = IndexConfig(vector=True, embedder="nomic")

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    result_files = {c.chunk.file_path for c in bundle.chunks}
    ranked_files = [c.chunk.file_path for c in bundle.chunks]
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    f1 = compute_f1(recall, precision)
    mrr_val = compute_mrr(ranked_files, task.expected_files)
    ndcg_val = compute_ndcg(ranked_files, task.expected_files)
    map_val = compute_map(ranked_files, task.expected_files)
    tokens_input, tokens_output, token_efficiency, tokens_raw_baseline, sym_recall = (
        _archex_token_fields(bundle, task, repo_path)
    )

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY_HYBRID,
        tokens_total=bundle.token_count,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        token_efficiency=token_efficiency,
        tokens_raw_baseline=tokens_raw_baseline,
        symbol_recall=sym_recall,
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
default_strategy_registry.register(Strategy.ARCHEX_QUERY_HYBRID.value, run_archex_query_hybrid)
default_strategy_registry.register(Strategy.ARCHEX_SYMBOL_LOOKUP.value, run_archex_symbol_lookup)

# Backward-compat reference
STRATEGY_RUNNERS = default_strategy_registry._runners
