"""Individual strategy implementations for benchmarking."""

from __future__ import annotations

import re
import subprocess
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkResult, BenchmarkTask, Strategy
from archex.models import PipelineTiming, RepoSource
from archex.reporting import count_tokens

if TYPE_CHECKING:
    from pathlib import Path

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


def run_raw_files(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Baseline strategy: read all expected files, count tokens."""
    t0 = time.perf_counter()
    tokens = count_file_tokens(repo_path, task.expected_files)
    wall_ms = (time.perf_counter() - t0) * 1000

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_FILES,
        tokens_total=tokens,
        tool_calls=len(task.expected_files),
        files_accessed=len(task.expected_files),
        recall=1.0,
        precision=1.0,
        savings_vs_raw=0.0,
        wall_time_ms=wall_ms,
        cached=False,
        timestamp=now_iso(),
    )


def run_raw_grepped(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """Grep-based strategy: search repo for keywords, read matched files."""
    t0 = time.perf_counter()
    keywords = extract_keywords(task.question, task.keywords)

    matched_files: set[str] = set()
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
                if path:
                    matched_files.add(path)

    tokens = count_file_tokens(repo_path, list(matched_files))
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(matched_files, task.expected_files)
    precision = compute_precision(matched_files, task.expected_files)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.RAW_GREPPED,
        tokens_total=tokens,
        tool_calls=len(keywords),
        files_accessed=len(matched_files),
        recall=recall,
        precision=precision,
        savings_vs_raw=0.0,  # backfilled by runner
        wall_time_ms=wall_ms,
        cached=False,
        timestamp=now_iso(),
    )


def run_archex_query(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    """archex query strategy: use BM25-based retrieval."""
    from archex.api import query
    from archex.models import Config, IndexConfig

    t0 = time.perf_counter()
    timing = PipelineTiming()
    source = RepoSource(local_path=str(repo_path))
    config = Config(cache=False)
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
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY,
        tokens_total=bundle.token_count,
        tool_calls=1,
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
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
    index_config = IndexConfig(vector=True)

    bundle = query(
        source,
        task.question,
        token_budget=task.token_budget,
        config=config,
        index_config=index_config,
        timing=timing,
    )

    result_files = {c.chunk.file_path for c in bundle.chunks}
    wall_ms = (time.perf_counter() - t0) * 1000
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY_HYBRID,
        tokens_total=bundle.token_count,
        tool_calls=1,
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
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


STRATEGY_RUNNERS = {
    Strategy.RAW_FILES: run_raw_files,
    Strategy.RAW_GREPPED: run_raw_grepped,
    Strategy.ARCHEX_QUERY: run_archex_query,
    Strategy.ARCHEX_QUERY_HYBRID: run_archex_query_hybrid,
    Strategy.ARCHEX_SYMBOL_LOOKUP: run_archex_symbol_lookup,
}
