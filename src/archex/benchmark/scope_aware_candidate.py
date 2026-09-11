"""Deterministic benchmark-only ranking for R26 Candidate A."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from archex.index.bm25 import BM25Index
from archex.models import CodeChunk

if TYPE_CHECKING:
    from archex.index.store import IndexStore

SCOPE_MARKERS = ("package.json", "Cargo.toml")
CANDIDATE_LIMIT_PER_SCOPE = 150
PARTICIPATION_THRESHOLD = 0.25

ScopeSearch = Callable[[str, str, int], list[tuple[CodeChunk, float]]]


class ScopeAwareRankingError(ValueError):
    """Raised when the candidate cannot establish its frozen scope contract."""


@dataclass(frozen=True)
class ScopeMap:
    """Deterministic marker-root ownership for every indexed chunk path."""

    marker_roots: tuple[str, ...]
    scopes: tuple[str, ...]
    owners_by_path: dict[str, str]


@dataclass(frozen=True)
class ScopeSearchResult:
    """One scope's bounded raw BM25 candidates and participation decision."""

    scope: str
    candidates: tuple[tuple[CodeChunk, float], ...]
    raw_top_score: float
    normalized_top_score: float
    included: bool
    reason: str
    post_dedup_contribution_count: int


@dataclass(frozen=True)
class RankedScopeCandidate:
    """One deduplicated candidate with its repository-normalized score."""

    chunk: CodeChunk
    normalized_score: float
    scope: str


@dataclass(frozen=True)
class ScopeRanking:
    """Candidate ranking plus the counts needed by the R27 receipt contract."""

    searches: tuple[ScopeSearchResult, ...]
    candidates: tuple[RankedScopeCandidate, ...]
    repository_global_max_raw_score: float
    pre_dedup_candidate_count: int
    duplicate_candidate_count: int
    post_dedup_candidate_count: int


def scope_roots(repo_root: Path) -> tuple[str, ...]:
    """Return bytewise-sorted package/Cargo marker roots, matching the R27 freeze."""
    roots: set[str] = set()
    for marker in SCOPE_MARKERS:
        for path in repo_root.rglob(marker):
            if ".git" in path.parts:
                continue
            relative = path.parent.relative_to(repo_root).as_posix()
            roots.add("" if relative == "." else relative)
    return tuple(sorted(roots))


def owner_scope(path: str, roots: Sequence[str]) -> str | None:
    """Resolve one repository-relative path to its deepest marker-root prefix."""
    owners = [root for root in roots if not root or path == root or path.startswith(f"{root}/")]
    if not owners:
        return None
    return max(owners, key=lambda value: (len(value.split("/")), value))


def resolve_scope_map(repo_root: Path, chunks: Sequence[CodeChunk]) -> ScopeMap:
    """Resolve indexed chunks to non-empty scopes and discard empty marker roots."""
    roots = scope_roots(repo_root)
    owners_by_path: dict[str, str] = {}
    for path in sorted({chunk.file_path for chunk in chunks}):
        owner = owner_scope(path, roots)
        if owner is not None:
            owners_by_path[path] = owner
    scopes = tuple(sorted(set(owners_by_path.values())))
    if not scopes:
        raise ScopeAwareRankingError("no non-empty marker-root scope owns an indexed chunk")
    return ScopeMap(marker_roots=roots, scopes=scopes, owners_by_path=owners_by_path)


class _ScopedBM25Index(BM25Index):
    """BM25Index using one shared FTS corpus with a scope-owner predicate."""

    def __init__(self, store: IndexStore, scope_map: ScopeMap) -> None:
        super().__init__(store)
        self._scope = ""
        self._owners_by_path = scope_map.owners_by_path
        self._owner_function = f"archex_scope_owner_{id(self):x}"
        store.conn.create_function(
            self._owner_function,
            1,
            self._owner_for_sql,
            deterministic=True,
        )

    def _owner_for_sql(self, value: object) -> str | None:
        return self._owners_by_path.get(str(value))

    @staticmethod
    def _apply_path_bonus(
        results: list[tuple[CodeChunk, float]],
        tokens: list[str],
    ) -> list[tuple[CodeChunk, float]]:
        boosted = BM25Index._apply_path_bonus(  # pyright: ignore[reportPrivateUsage]
            results,
            tokens,
        )
        return sorted(
            boosted,
            key=lambda result: (
                -result[1],
                result[0].file_path,
                result[0].start_line,
                result[0].id,
            ),
        )

    def _execute_fts(
        self,
        escaped: str,
        top_k: int,
        weights: tuple[float, float, float, float, float, float] = (
            1.0,
            10.0,
            1.5,
            6.0,
            5.0,
            8.0,
        ),
    ) -> list[tuple[str, float]]:
        """Run the existing BM25 formula over rows owned by the active scope."""
        w_content, w_symbol, w_path, w_docstring, w_bc, w_summary = weights
        try:
            cursor = self._store.conn.execute(
                "SELECT chunk_id, "
                f"bm25(chunks_fts, {w_content}, {w_symbol}, {w_path}, "
                f"{w_docstring}, {w_bc}, {w_summary}) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? "
                f"AND {self._owner_function}(file_path) = ? "
                "ORDER BY score, file_path, chunk_id LIMIT ?",
                (escaped, self._scope, top_k),
            )
        except sqlite3.OperationalError as exc:
            raise ScopeAwareRankingError(
                f"scope-aware FTS5 query failed for scope {self._scope!r}: {escaped}"
            ) from exc
        return [(str(row[0]), float(row[1])) for row in cursor.fetchall()]

    def search_scope(
        self,
        scope: str,
        query: str,
        top_k: int = CANDIDATE_LIMIT_PER_SCOPE,
    ) -> list[tuple[CodeChunk, float]]:
        """Search one owner scope without changing the repository-wide BM25 corpus."""
        self._scope = scope
        return self.search(query, top_k=top_k)

    def close(self) -> None:
        """Remove the connection-local owner function installed for this search."""
        self._store.conn.create_function(self._owner_function, 1, None)


def rank_scope_candidates(
    scope_map: ScopeMap,
    query: str,
    search_scope: ScopeSearch,
) -> ScopeRanking | None:
    """Apply the frozen normalization, participation, deduplication, and order.

    ``None`` is the exact single-scope bypass signal. The caller must delegate
    that case to unchanged ``archex_query`` without invoking ``search_scope``.
    """
    if len(scope_map.scopes) == 1:
        return None
    if not scope_map.scopes:
        raise ScopeAwareRankingError("scope map contains no non-empty scopes")

    raw_results: list[tuple[str, tuple[tuple[CodeChunk, float], ...]]] = []
    for scope in scope_map.scopes:
        candidates = tuple(search_scope(scope, query, CANDIDATE_LIMIT_PER_SCOPE))
        if len(candidates) > CANDIDATE_LIMIT_PER_SCOPE:
            raise ScopeAwareRankingError(
                f"scope {scope!r} returned {len(candidates)} candidates; "
                f"limit is {CANDIDATE_LIMIT_PER_SCOPE}"
            )
        if any(not math.isfinite(score) for _, score in candidates):
            raise ScopeAwareRankingError(f"scope {scope!r} returned a non-finite score")
        raw_results.append((scope, candidates))
    raw_by_scope = tuple(raw_results)
    global_max = max(
        (score for _, candidates in raw_by_scope for _, score in candidates if score > 0.0),
        default=0.0,
    )
    pre_dedup_count = sum(len(candidates) for _, candidates in raw_by_scope)

    all_winners: dict[str, RankedScopeCandidate] = {}
    decisions: list[ScopeSearchResult] = []
    provisional: list[tuple[str, tuple[tuple[CodeChunk, float], ...], float, float, bool, str]] = []
    for scope, candidates in raw_by_scope:
        raw_top = max((score for _, score in candidates), default=0.0)
        if global_max <= 0.0:
            normalized_top = 0.0
            included = False
            reason = "non_positive_global_max"
        else:
            normalized_top = raw_top / global_max
            included = normalized_top >= PARTICIPATION_THRESHOLD
            reason = (
                "normalized_top_at_or_above_threshold"
                if included
                else "normalized_top_below_threshold"
            )
        provisional.append((scope, candidates, raw_top, normalized_top, included, reason))
        for chunk, raw_score in candidates:
            normalized = raw_score / global_max if global_max > 0.0 else 0.0
            candidate = RankedScopeCandidate(
                chunk=chunk,
                normalized_score=normalized,
                scope=scope,
            )
            previous = all_winners.get(chunk.id)
            if previous is None or candidate.normalized_score > previous.normalized_score:
                all_winners[chunk.id] = candidate

    contribution_counts: dict[str, int] = dict.fromkeys(scope_map.scopes, 0)
    for candidate in all_winners.values():
        contribution_counts[candidate.scope] += 1

    included_scopes = {scope for scope, _, _, _, included, _ in provisional if included}
    ranked = tuple(
        sorted(
            (candidate for candidate in all_winners.values() if candidate.scope in included_scopes),
            key=lambda candidate: (
                -candidate.normalized_score,
                candidate.chunk.file_path,
                candidate.chunk.start_line,
                candidate.chunk.id,
            ),
        )
    )
    for scope, candidates, raw_top, normalized_top, included, reason in provisional:
        decisions.append(
            ScopeSearchResult(
                scope=scope,
                candidates=candidates,
                raw_top_score=raw_top,
                normalized_top_score=normalized_top,
                included=included,
                reason=reason,
                post_dedup_contribution_count=contribution_counts[scope],
            )
        )

    post_dedup_count = len(all_winners)
    return ScopeRanking(
        searches=tuple(decisions),
        candidates=ranked,
        repository_global_max_raw_score=global_max,
        pre_dedup_candidate_count=pre_dedup_count,
        duplicate_candidate_count=pre_dedup_count - post_dedup_count,
        post_dedup_candidate_count=post_dedup_count,
    )


def rank_scopes(
    store: IndexStore,
    repo_root: Path,
    query: str,
    chunks: Sequence[CodeChunk],
) -> tuple[ScopeMap, ScopeRanking | None]:
    """Resolve scopes and rank them through one repository-wide BM25 store."""
    scope_map = resolve_scope_map(repo_root, chunks)
    if len(scope_map.scopes) == 1:
        return scope_map, None
    index = _ScopedBM25Index(store, scope_map)
    try:
        return scope_map, rank_scope_candidates(scope_map, query, index.search_scope)
    finally:
        index.close()
