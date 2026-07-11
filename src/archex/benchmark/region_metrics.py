"""Region, line, ranking, and context-efficiency metrics for benchmark results.

These metrics extend the file-level retrieval signals toward block/line-level
context quality, as motivated by ranked-region exploration benchmarks. They are
computed only when a task declares ``expected_regions``; file-only tasks report
``None`` for every region field so existing benchmark conventions are preserved.

The computation is deterministic and local: it depends only on the returned
context (in ranked order) and the task's ground-truth regions. No network, LLM,
or hosted calls are involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from archex.benchmark.models import ExpectedRegion, RegionGranularity


@dataclass(frozen=True)
class ReturnedRegion:
    """One returned context unit, in the order the bundle ranks it.

    ``start_line``/``end_line`` are 1-indexed inclusive line bounds. An
    elision anchor retains source coordinates and consumes context tokens but
    exposes no source evidence, so it is counted as noise without earning
    region, line, or ranking credit.
    """

    path: str
    start_line: int
    end_line: int
    symbol: str | None
    tokens: int
    source_evidence: bool = True


@dataclass(frozen=True)
class RegionMetrics:
    """Derived region/line/ranking/context-efficiency metrics for one result."""

    region_recall: float
    region_precision: float
    region_f1: float
    line_recall: float | None
    line_precision: float | None
    ranked_region_mrr: float
    ranked_region_ndcg: float
    context_noise_ratio: float
    useful_tokens: int
    wasted_tokens: int
    relevance_per_1k_tokens: float

    def as_result_fields(self) -> dict[str, float | int | None]:
        """Map to ``BenchmarkResult`` keyword arguments."""
        return {
            "region_recall": self.region_recall,
            "region_precision": self.region_precision,
            "region_f1": self.region_f1,
            "line_recall": self.line_recall,
            "line_precision": self.line_precision,
            "ranked_region_mrr": self.ranked_region_mrr,
            "ranked_region_ndcg": self.ranked_region_ndcg,
            "context_noise_ratio": self.context_noise_ratio,
            "useful_tokens": self.useful_tokens,
            "wasted_tokens": self.wasted_tokens,
            "relevance_per_1k_tokens": self.relevance_per_1k_tokens,
        }


def _ranges_overlap(
    r_start: int,
    r_end: int,
    e_start: int,
    e_end: int,
) -> bool:
    return r_start <= e_end and e_start <= r_end


def _symbol_matches(returned: str | None, expected: str | None) -> bool:
    if not returned or not expected:
        return False
    if returned == expected:
        return True
    # Returned chunk symbols are often unqualified (``run``) while labels may be
    # qualified (``Cli.run``). Match when the shorter dotted path is a trailing
    # segment-sequence of the longer one, so ``run`` matches ``Cli.run`` but
    # ``Server.run`` does not match ``Cli.run``.
    r_parts = returned.split(".")
    e_parts = expected.split(".")
    shorter, longer = (r_parts, e_parts) if len(r_parts) <= len(e_parts) else (e_parts, r_parts)
    return longer[-len(shorter) :] == shorter


def _overlaps(returned: ReturnedRegion, expected: ExpectedRegion) -> bool:
    """Whether a returned region covers an expected region at all."""
    if returned.path != expected.path:
        return False
    if expected.granularity is RegionGranularity.FILE:
        return True
    if expected.granularity in (RegionGranularity.SYMBOL, RegionGranularity.BLOCK):
        if _symbol_matches(returned.symbol, expected.symbol):
            return True
        # A symbol/block label may also pin explicit lines; fall through to a
        # line-overlap match when it does, otherwise it is a miss.
        if expected.start_line is None:
            return False
    if expected.start_line is None or expected.end_line is None:
        return False
    return _ranges_overlap(
        returned.start_line, returned.end_line, expected.start_line, expected.end_line
    )


def _useful_fraction(returned: ReturnedRegion, expected_same_path: list[ExpectedRegion]) -> float:
    """Fraction of a returned region's lines that fall inside expected regions.

    The fraction is computed over the union of all overlapping expected regions
    in the same file, so a chunk spanning several labeled regions is credited
    for every covered line rather than only the single best-overlapping region.
    """
    r_lines = returned.end_line - returned.start_line + 1
    if r_lines <= 0:
        return 0.0
    covered: set[int] = set()
    for expected in expected_same_path:
        if expected.granularity is RegionGranularity.FILE:
            return 1.0
        if expected.granularity in (RegionGranularity.SYMBOL, RegionGranularity.BLOCK):
            if _symbol_matches(returned.symbol, expected.symbol):
                return 1.0
            if expected.start_line is None:
                continue
        if expected.start_line is None or expected.end_line is None:
            continue
        lo = max(returned.start_line, expected.start_line)
        hi = min(returned.end_line, expected.end_line)
        if hi >= lo:
            covered.update(range(lo, hi + 1))
    return len(covered) / r_lines


def _line_set(region: ExpectedRegion) -> set[tuple[str, int]]:
    if region.start_line is None or region.end_line is None:
        return set()
    return {(region.path, line) for line in range(region.start_line, region.end_line + 1)}


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _line_metrics(
    returned_regions: list[ReturnedRegion],
    expected_regions: list[ExpectedRegion],
) -> tuple[float | None, float | None]:
    """Line recall/precision over expected regions that pin explicit line ranges."""
    expected_lines: set[tuple[str, int]] = set()
    labeled_files: set[str] = set()
    for expected in expected_regions:
        lines = _line_set(expected)
        if lines:
            expected_lines |= lines
            labeled_files.add(expected.path)
    if not expected_lines:
        return None, None
    returned_lines: set[tuple[str, int]] = set()
    for returned in returned_regions:
        if returned.path not in labeled_files:
            continue
        for line in range(returned.start_line, returned.end_line + 1):
            returned_lines.add((returned.path, line))
    covered = expected_lines & returned_lines
    line_recall = len(covered) / len(expected_lines)
    line_precision = (len(covered) / len(returned_lines)) if returned_lines else 0.0
    return line_recall, line_precision


def _ranking_metrics(
    returned_regions: list[ReturnedRegion],
    expected_regions: list[ExpectedRegion],
) -> tuple[float, float]:
    """MRR and weighted nDCG over expected regions using returned context order.

    nDCG credits each expected region at the rank of the first returned region
    that covers it, and normalizes by the ideal ordering of *all* expected-region
    weights (descending). Missing a heavily weighted region therefore lowers
    nDCG even when the regions that were returned are well ordered. Regions that
    are first surfaced by the same returned chunk are assigned distinct, adjacent
    effective ranks so a single chunk covering several regions cannot push DCG
    above the ideal (nDCG stays bounded in [0, 1]).
    """
    first_hit_rank = 0
    for index, returned in enumerate(returned_regions):
        if any(_overlaps(returned, expected) for expected in expected_regions):
            first_hit_rank = index + 1
            break
    mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0

    covered: list[tuple[int, float]] = []
    for expected in expected_regions:
        for rank_index, returned in enumerate(returned_regions):
            if _overlaps(returned, expected):
                covered.append((rank_index, expected.weight))
                break
    covered.sort(key=lambda item: (item[0], -item[1]))
    dcg = 0.0
    next_rank = 0
    for first_index, weight in covered:
        effective_rank = max(first_index, next_rank)
        dcg += weight / math.log2(effective_rank + 2)
        next_rank = effective_rank + 1
    ideal_weights = sorted((expected.weight for expected in expected_regions), reverse=True)
    idcg = sum(weight / math.log2(i + 2) for i, weight in enumerate(ideal_weights))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return mrr, ndcg


def compute_region_metrics(
    returned_regions: list[ReturnedRegion],
    expected_regions: list[ExpectedRegion],
) -> RegionMetrics | None:
    """Compute region/line/ranking/context metrics, or ``None`` without labels.

    ``returned_regions`` must be in the order the downstream model would see them
    (ranked order), not path-sorted, so ranking metrics reflect early context.
    """
    if not expected_regions:
        return None

    evidence_regions = [region for region in returned_regions if region.source_evidence]
    covered_weight = 0.0
    total_weight = 0.0
    for expected in expected_regions:
        total_weight += expected.weight
        if any(_overlaps(returned, expected) for returned in evidence_regions):
            covered_weight += expected.weight
    region_recall = (covered_weight / total_weight) if total_weight > 0 else 0.0

    overlapping_returned = sum(
        1
        for returned in evidence_regions
        if any(_overlaps(returned, expected) for expected in expected_regions)
    )
    region_precision = (overlapping_returned / len(evidence_regions)) if evidence_regions else 0.0
    region_f1 = _f1(region_precision, region_recall)

    line_recall, line_precision = _line_metrics(evidence_regions, expected_regions)
    mrr, ndcg = _ranking_metrics(evidence_regions, expected_regions)

    by_path: dict[str, list[ExpectedRegion]] = {}
    for expected in expected_regions:
        by_path.setdefault(expected.path, []).append(expected)

    # Token accounting is over returned chunk content only (the code context the
    # model reads), not the result's headline tokens_total, which may also include
    # scout maps or rendering. This keeps useful + wasted == returned-chunk tokens.
    total_tokens = 0
    useful_tokens = 0
    for returned in returned_regions:
        total_tokens += returned.tokens
        if not returned.source_evidence:
            continue
        same_path = by_path.get(returned.path)
        if not same_path:
            continue
        useful_tokens += round(_useful_fraction(returned, same_path) * returned.tokens)
    useful_tokens = min(useful_tokens, total_tokens)
    wasted_tokens = total_tokens - useful_tokens
    context_noise_ratio = (wasted_tokens / total_tokens) if total_tokens > 0 else 0.0
    relevance_per_1k_tokens = (region_recall * 1000.0 / total_tokens) if total_tokens > 0 else 0.0

    return RegionMetrics(
        region_recall=region_recall,
        region_precision=region_precision,
        region_f1=region_f1,
        line_recall=line_recall,
        line_precision=line_precision,
        ranked_region_mrr=mrr,
        ranked_region_ndcg=ndcg,
        context_noise_ratio=context_noise_ratio,
        useful_tokens=useful_tokens,
        wasted_tokens=wasted_tokens,
        relevance_per_1k_tokens=relevance_per_1k_tokens,
    )
