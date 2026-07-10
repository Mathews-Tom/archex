"""Offline cross-tool token-efficiency comparison: archex retrieval vs a naive baseline.

This module models the token cost of *localizing* a task's required files two ways
and compares them at a fixed required-file recall:

- **archex**: the targeted regions the ``archex_query`` retrieval returns, in rank
  order. The cost to localize is the sum of region tokens consumed until the
  required files are covered.
- **naive baseline**: how a non-archex agent reaches the same files — grep the
  corpus for the task keywords, then either read the full grep-hit files
  (``full_file``) or read ``+/-K`` context windows around the grep hits
  (``grep_window``), in grep-relevance order. The cost to localize is the sum of
  file/window tokens consumed until the required files are covered.

The comparison is **offline and benchmark-only**: it never touches the query hot
path, the in-process metrics ledger, retrieval ranking, or any product default.
The naive token model is a pure, deterministic function of the corpus snapshot,
the task keywords, and the context window ``K`` (tokenized with the same
``cl100k_base`` encoder the rest of archex uses).

Recall is held equal across the compared paths: a token delta is reported only
for tasks where *both* paths reach the target required-file recall, so the
aggregate efficiency number never compares unequal recall.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from archex import __version__
from archex.acquire.discovery import discover_files
from archex.benchmark.models import BenchmarkTask, TaskCategory, TaskFamily
from archex.reporting import count_tokens

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from archex.benchmark.region_metrics import ReturnedRegion

logger = logging.getLogger(__name__)

# Source extensions the naive grep model scans, matching the existing raw grep/rg
# baselines so archex and naive see the same corpus.
_SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".ts", ".js", ".go", ".rs", ".java", ".kt", ".cs", ".swift"}
)

# Corpus buckets the cross-tool number is aggregated per. Localization is graded
# as its own corpus and never merged with comprehension.
CORPUS_SELF = "self"
CORPUS_EXTERNAL_COMPREHENSION = "external-comprehension"
CORPUS_EXTERNAL_LOCALIZATION = "external-localization"
_CORPUS_ORDER: tuple[str, ...] = (
    CORPUS_SELF,
    CORPUS_EXTERNAL_COMPREHENSION,
    CORPUS_EXTERNAL_LOCALIZATION,
)


class NaiveBaselineModel(StrEnum):
    """How the naive (non-archex) baseline spends tokens to read a grep-hit file."""

    FULL_FILE = "full_file"
    GREP_WINDOW = "grep_window"


@dataclass(frozen=True)
class RetrievalUnit:
    """One context unit a path surfaces, in the order that path ranks it.

    ``path`` is a repo-relative POSIX path; ``tokens`` is the cl100k_base token
    cost of opening this unit (a region, a full file, or a context window).
    """

    path: str
    tokens: int


@dataclass(frozen=True)
class _FileScan:
    """A scanned source file with at least one keyword hit."""

    path: str
    hit_lines: tuple[int, ...]
    content: str


class PathTokensAtRecall(BaseModel):
    """Tokens one path spends to reach a target required-file recall.

    ``recall_reached`` is the achieved required-file recall at the measurement
    point; ``target_reached`` is whether the target recall was reached at all.
    When the target is reached, ``tokens`` counts every unit consumed up to and
    including the one that crossed the target; otherwise it counts every unit.
    """

    tokens: int
    recall_reached: float
    target_reached: bool
    units_consumed: int


class NaiveBaselineResult(BaseModel):
    """A naive baseline model and its tokens-at-recall for one task."""

    model: NaiveBaselineModel
    at_recall: PathTokensAtRecall


class CrossToolTaskComparison(BaseModel):
    """Per-task archex-vs-naive tokens-at-fixed-recall comparison."""

    task_id: str
    repo: str
    corpus: str
    family: TaskFamily
    category: TaskCategory | None = None
    required_file_count: int
    target_recall: float
    context_window: int
    archex: PathTokensAtRecall
    naive: list[NaiveBaselineResult]


class CrossToolAggregate(BaseModel):
    """Per-corpus, per-naive-model aggregate over tasks where recall is held equal.

    ``comparable_count`` is the number of tasks where both archex and this naive
    model reach the target recall; every token figure is summed/averaged over
    exactly that comparable set, so the reduction never reflects unequal recall.
    """

    corpus: str
    model: NaiveBaselineModel
    task_count: int
    comparable_count: int
    archex_tokens: int
    naive_tokens: int
    mean_token_ratio: float
    median_token_ratio: float
    token_reduction_pct: float


class CrossToolReport(BaseModel):
    """Full cross-tool efficiency report: per-task comparisons plus aggregates."""

    generated_at: str
    strategy: str
    target_recall: float
    context_window: int
    archex_version: str
    comparisons: list[CrossToolTaskComparison]
    aggregates: list[CrossToolAggregate]


def corpus_of(task: BenchmarkTask) -> str:
    """Bucket a task into self / external-comprehension / external-localization.

    The cross-tool number publishes exactly these three corpora. Self-repo tasks
    bucket to ``self`` first (the maintained self corpus is comprehension-shaped;
    the localization family is external by construction). External tasks split
    so external localization is graded as its own corpus, never merged with
    external comprehension.
    """
    if task.repo == ".":
        return CORPUS_SELF
    if task.family is TaskFamily.LOCALIZATION:
        return CORPUS_EXTERNAL_LOCALIZATION
    return CORPUS_EXTERNAL_COMPREHENSION


def tokens_at_recall(
    units: Sequence[RetrievalUnit],
    required_files: Sequence[str],
    target_recall: float,
) -> PathTokensAtRecall:
    """Tokens consumed walking ``units`` until required-file recall reaches the target.

    Required files are covered the first time a unit in that file is consumed.
    Tokens accrue for every unit consumed (including grep false positives ranked
    ahead of the required file), modelling the true cost of localizing the
    required set along that path's own ranking.
    """
    required = set(required_files)
    if not required:
        return PathTokensAtRecall(
            tokens=0, recall_reached=1.0, target_reached=True, units_consumed=0
        )

    covered: set[str] = set()
    cumulative = 0
    consumed = 0
    for unit in units:
        cumulative += unit.tokens
        consumed += 1
        if unit.path in required:
            covered.add(unit.path)
        recall = len(covered) / len(required)
        if recall + 1e-9 >= target_recall:
            return PathTokensAtRecall(
                tokens=cumulative,
                recall_reached=recall,
                target_reached=True,
                units_consumed=consumed,
            )
    return PathTokensAtRecall(
        tokens=cumulative,
        recall_reached=len(covered) / len(required),
        target_reached=False,
        units_consumed=consumed,
    )


def archex_units(regions: Sequence[ReturnedRegion]) -> list[RetrievalUnit]:
    """One retrieval unit per returned region, preserving the bundle's rank order."""
    return [RetrievalUnit(path=region.path, tokens=region.tokens) for region in regions]


def _iter_source_files(repo_path: Path) -> list[tuple[str, Path]]:
    """Repo source files the naive model may open, as ``(rel_posix, absolute)``.

    Reuses archex's gitignore-aware discovery so the naive grep agent scans the
    same corpus archex indexes (no ``.venv``/build-artifact pollution), then
    narrows to the source extensions the grep/ripgrep baselines search. Sorted
    for determinism.
    """
    files = [
        (Path(discovered.path).as_posix(), Path(discovered.absolute_path))
        for discovered in discover_files(repo_path).files
        if Path(discovered.path).suffix in _SOURCE_EXTENSIONS
    ]
    return sorted(files)


def _scan_for_hits(repo_path: Path, keywords: Sequence[str]) -> list[_FileScan]:
    """Deterministic in-process grep: source files with >=1 case-insensitive hit.

    Pure Python (no subprocess for matching), so the naive token model is
    reproducible across machines regardless of the installed grep/ripgrep.
    """
    if not keywords:
        return []
    pattern = re.compile("|".join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)
    scans: list[_FileScan] = []
    for rel, absolute in _iter_source_files(repo_path):
        content = absolute.read_text(encoding="utf-8", errors="replace")
        hit_lines = tuple(
            number
            for number, line in enumerate(content.splitlines(), start=1)
            if pattern.search(line)
        )
        if hit_lines:
            scans.append(_FileScan(path=rel, hit_lines=hit_lines, content=content))
    return scans


def _window_tokens(content: str, hit_lines: Sequence[int], context_window: int) -> int:
    """Tokens of the merged ``+/-context_window`` line windows around grep hits."""
    lines = content.splitlines()
    line_count = len(lines)
    selected: set[int] = set()
    for hit in hit_lines:
        low = max(1, hit - context_window)
        high = min(line_count, hit + context_window)
        selected.update(range(low, high + 1))
    if not selected:
        return 0
    window_text = "\n".join(lines[number - 1] for number in sorted(selected))
    return count_tokens(window_text)


def naive_units(
    repo_path: Path,
    task: BenchmarkTask,
    *,
    model: NaiveBaselineModel,
    context_window: int,
) -> list[RetrievalUnit]:
    """Ordered naive-baseline units for a task: grep-hit files in relevance order.

    Files are ranked by hit count (descending) with a path tiebreak, the order a
    naive agent would triage grep output. ``full_file`` costs the whole file;
    ``grep_window`` costs only the merged context windows around the hits.
    """
    from archex.benchmark.strategies import extract_keywords

    keywords = extract_keywords(task.question, task.keywords)
    scans = _scan_for_hits(repo_path, keywords)
    ordered = sorted(scans, key=lambda scan: (-len(scan.hit_lines), scan.path))
    units: list[RetrievalUnit] = []
    for scan in ordered:
        if model is NaiveBaselineModel.FULL_FILE:
            tokens = count_tokens(scan.content)
        else:
            tokens = _window_tokens(scan.content, scan.hit_lines, context_window)
        units.append(RetrievalUnit(path=scan.path, tokens=tokens))
    return units


def compare_task(
    task: BenchmarkTask,
    repo_path: Path,
    regions: Sequence[ReturnedRegion],
    *,
    target_recall: float,
    context_window: int,
    models: Sequence[NaiveBaselineModel],
) -> CrossToolTaskComparison:
    """Compare archex's targeted token cost against each naive model for one task."""
    archex = tokens_at_recall(archex_units(regions), task.expected_files, target_recall)
    naive: list[NaiveBaselineResult] = []
    for model in models:
        units = naive_units(repo_path, task, model=model, context_window=context_window)
        naive.append(
            NaiveBaselineResult(
                model=model,
                at_recall=tokens_at_recall(units, task.expected_files, target_recall),
            )
        )
    return CrossToolTaskComparison(
        task_id=task.task_id,
        repo=task.repo,
        corpus=corpus_of(task),
        family=task.family,
        category=task.category,
        required_file_count=len(task.expected_files),
        target_recall=target_recall,
        context_window=context_window,
        archex=archex,
        naive=naive,
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _naive_for_model(
    comparison: CrossToolTaskComparison, model: NaiveBaselineModel
) -> NaiveBaselineResult | None:
    for result in comparison.naive:
        if result.model is model:
            return result
    return None


def aggregate_cross_tool(
    comparisons: Sequence[CrossToolTaskComparison],
    models: Sequence[NaiveBaselineModel],
) -> list[CrossToolAggregate]:
    """Aggregate per corpus and naive model over the recall-held-equal task set."""
    by_corpus: dict[str, list[CrossToolTaskComparison]] = {}
    for comparison in comparisons:
        by_corpus.setdefault(comparison.corpus, []).append(comparison)

    ordered_corpora = [corpus for corpus in _CORPUS_ORDER if corpus in by_corpus]
    ordered_corpora += [corpus for corpus in sorted(by_corpus) if corpus not in _CORPUS_ORDER]

    aggregates: list[CrossToolAggregate] = []
    for corpus in ordered_corpora:
        corpus_comparisons = by_corpus[corpus]
        for model in models:
            comparable = [
                comparison
                for comparison in corpus_comparisons
                if comparison.archex.target_reached
                and (naive := _naive_for_model(comparison, model)) is not None
                and naive.at_recall.target_reached
            ]
            archex_tokens = sum(comparison.archex.tokens for comparison in comparable)
            naive_tokens = 0
            ratios: list[float] = []
            for comparison in comparable:
                naive = _naive_for_model(comparison, model)
                assert naive is not None  # guaranteed by the comparable filter
                naive_tokens += naive.at_recall.tokens
                if comparison.archex.tokens > 0:
                    ratios.append(naive.at_recall.tokens / comparison.archex.tokens)
            reduction = (
                (naive_tokens - archex_tokens) / naive_tokens * 100.0 if naive_tokens > 0 else 0.0
            )
            aggregates.append(
                CrossToolAggregate(
                    corpus=corpus,
                    model=model,
                    task_count=len(corpus_comparisons),
                    comparable_count=len(comparable),
                    archex_tokens=archex_tokens,
                    naive_tokens=naive_tokens,
                    mean_token_ratio=(sum(ratios) / len(ratios)) if ratios else 0.0,
                    median_token_ratio=_median(ratios),
                    token_reduction_pct=reduction,
                )
            )
    return aggregates


def run_cross_tool(
    tasks: Sequence[BenchmarkTask],
    *,
    target_recall: float = 1.0,
    context_window: int = 5,
    models: Sequence[NaiveBaselineModel] = (
        NaiveBaselineModel.FULL_FILE,
        NaiveBaselineModel.GREP_WINDOW,
    ),
    strategy_label: str = "archex_query",
    regions_provider: Callable[[BenchmarkTask, Path], list[ReturnedRegion]] | None = None,
    on_task: Callable[[int, BenchmarkTask], None] | None = None,
) -> CrossToolReport:
    """Run the cross-tool comparison over ``tasks`` and aggregate per corpus.

    ``regions_provider`` returns archex's ranked returned regions for a task;
    it defaults to the real ``archex_query`` retrieval. Repos are cloned and
    scoped exactly as the standard benchmark runner does. A per-task clone or
    retrieval failure is isolated so one bad repo does not abort the batch.
    """
    from archex.benchmark.runner import repo_path_for_task
    from archex.benchmark.strategies import archex_returned_regions
    from archex.exceptions import ArchexIndexError, BenchmarkCloneError

    provider = regions_provider or archex_returned_regions
    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path] = {}
    cleanup_paths: list[Path] = []
    comparisons: list[CrossToolTaskComparison] = []
    try:
        for index, task in enumerate(tasks, start=1):
            if on_task is not None:
                on_task(index, task)
            try:
                repo_path = repo_path_for_task(task, repo_cache, cleanup_paths)
            except BenchmarkCloneError as exc:
                logger.warning("Skipping cross-tool task %s: %s", task.task_id, exc)
                continue
            try:
                regions = provider(task, repo_path)
            except (ArchexIndexError, NotImplementedError) as exc:
                logger.warning("Skipping cross-tool task %s: %s", task.task_id, exc)
                continue
            comparisons.append(
                compare_task(
                    task,
                    repo_path,
                    regions,
                    target_recall=target_recall,
                    context_window=context_window,
                    models=models,
                )
            )
    finally:
        for path in cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)

    return CrossToolReport(
        generated_at=datetime.now(tz=UTC).isoformat(),
        strategy=strategy_label,
        target_recall=target_recall,
        context_window=context_window,
        archex_version=__version__,
        comparisons=comparisons,
        aggregates=aggregate_cross_tool(comparisons, models),
    )
