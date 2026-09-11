"""Behavioral guards for the benchmark-only R28 scope-aware ranker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from archex.benchmark.scope_aware_candidate import (
    CANDIDATE_LIMIT_PER_SCOPE,
    ScopeAwareRankingError,
    ScopeMap,
    owner_scope,
    rank_scope_candidates,
    rank_scopes,
    resolve_scope_map,
)
from archex.index.bm25 import BM25Index
from archex.index.store import IndexStore
from archex.models import CodeChunk

if TYPE_CHECKING:
    from pathlib import Path


def _chunk(chunk_id: str, path: str, *, line: int = 1, content: str = "needle") -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=path,
        start_line=line,
        end_line=line,
        language="python",
        token_count=4,
    )


def _scope_map(*scopes: str) -> ScopeMap:
    return ScopeMap(
        marker_roots=tuple(scopes),
        scopes=tuple(scopes),
        owners_by_path={f"{scope}/file.py": scope for scope in scopes},
    )


def test_scope_resolution_uses_deepest_prefix_and_discards_empty_roots(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "packages" / "api" / "empty").mkdir(parents=True)
    (tmp_path / "packages" / "api" / "package.json").write_text("{}")
    (tmp_path / "packages" / "api" / "empty" / "package.json").write_text("{}")
    chunks = [
        _chunk("root", "src/root.py"),
        _chunk("api", "packages/api/src/api.py"),
    ]

    resolved = resolve_scope_map(tmp_path, chunks)

    assert resolved.marker_roots == ("", "packages/api", "packages/api/empty")
    assert resolved.scopes == ("", "packages/api")
    assert resolved.owners_by_path == {
        "packages/api/src/api.py": "packages/api",
        "src/root.py": "",
    }
    assert owner_scope("packages/api/src/nested.py", resolved.marker_roots) == "packages/api"


def test_single_scope_returns_bypass_without_searching() -> None:
    searched = False

    def _unexpected_search(_scope: str, _query: str, _limit: int) -> list[tuple[CodeChunk, float]]:
        nonlocal searched
        searched = True
        return []

    ranking = rank_scope_candidates(_scope_map("packages/api"), "needle", _unexpected_search)

    assert ranking is None
    assert searched is False


def test_one_store_filters_and_bounds_each_scope(tmp_path: Path) -> None:
    for scope in ("packages/api", "packages/web"):
        marker = tmp_path / scope / "package.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}")
    chunks = [
        _chunk(f"api-{index:03d}", f"packages/api/src/file_{index:03d}.py")
        for index in range(CANDIDATE_LIMIT_PER_SCOPE + 1)
    ] + [
        _chunk(f"web-{index:03d}", f"packages/web/src/file_{index:03d}.py")
        for index in range(CANDIDATE_LIMIT_PER_SCOPE + 1)
    ]
    store = IndexStore(tmp_path / "index.db")
    try:
        store.insert_chunks(chunks)
        BM25Index(store).build(chunks)

        scope_map, ranking = rank_scopes(store, tmp_path, "needle", chunks)

        assert scope_map.scopes == ("packages/api", "packages/web")
        assert ranking is not None
        assert ranking.repository_global_max_raw_score > 0.0
        assert [len(search.candidates) for search in ranking.searches] == [
            CANDIDATE_LIMIT_PER_SCOPE,
            CANDIDATE_LIMIT_PER_SCOPE,
        ]
        assert all(
            chunk.file_path.startswith(f"{search.scope}/")
            for search in ranking.searches
            for chunk, _score in search.candidates
        )
    finally:
        store.close()


def test_scoped_search_uses_repository_wide_idf(tmp_path: Path) -> None:
    for scope in ("packages/api", "packages/web"):
        marker = tmp_path / scope / "package.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}")
    matching = _chunk("match", "packages/api/src/match.py")
    unrelated = [
        _chunk(
            f"web-{index:03d}",
            f"packages/web/src/file_{index:03d}.py",
            content="unrelated",
        )
        for index in range(100)
    ]
    shared_chunks = [matching, *unrelated]
    shared_store = IndexStore(tmp_path / "shared.db")
    isolated_store = IndexStore(tmp_path / "isolated.db")
    try:
        shared_store.insert_chunks(shared_chunks)
        BM25Index(shared_store).build(shared_chunks)
        _scope_map_result, ranking = rank_scopes(
            shared_store,
            tmp_path,
            "needle",
            shared_chunks,
        )
        assert ranking is not None
        shared_score = ranking.searches[0].raw_top_score

        isolated_store.insert_chunks([matching])
        isolated_index = BM25Index(isolated_store)
        isolated_index.build([matching])
        isolated_score = isolated_index.search("needle", top_k=1)[0][1]

        assert shared_score > isolated_score
    finally:
        shared_store.close()
        isolated_store.close()


def test_scope_candidate_limit_fails_closed() -> None:
    candidates = [
        (_chunk(f"chunk-{index}", f"a/file_{index}.py"), 1.0)
        for index in range(CANDIDATE_LIMIT_PER_SCOPE + 1)
    ]

    with pytest.raises(ScopeAwareRankingError, match="limit is 150"):
        rank_scope_candidates(
            _scope_map("a", "b"),
            "needle",
            lambda _scope, _query, _limit: candidates,
        )


def test_global_normalization_inclusive_gate_dedup_and_order() -> None:
    top = _chunk("top", "a/top.py")
    duplicate_a = _chunk("duplicate", "a/duplicate.py")
    duplicate_b = duplicate_a.model_copy(update={"file_path": "b/duplicate.py"})
    boundary_late = _chunk("boundary-late", "z.py", line=1)
    boundary_early = _chunk("boundary-early", "a.py", line=2)
    boundary_first = _chunk("boundary-first", "a.py", line=1)
    rejected = _chunk("rejected", "c/rejected.py")
    raw = {
        "a": [(top, 4.0), (duplicate_a, 2.0)],
        "b": [
            (duplicate_b, 3.0),
            (boundary_late, 1.0),
            (boundary_early, 1.0),
            (boundary_first, 1.0),
        ],
        "c": [(rejected, 0.99)],
    }

    ranking = rank_scope_candidates(
        _scope_map("a", "b", "c"),
        "needle",
        lambda scope, _query, _limit: raw[scope],
    )

    assert ranking is not None
    assert ranking.repository_global_max_raw_score == 4.0
    assert [search.included for search in ranking.searches] == [True, True, False]
    assert [search.reason for search in ranking.searches] == [
        "normalized_top_at_or_above_threshold",
        "normalized_top_at_or_above_threshold",
        "normalized_top_below_threshold",
    ]
    assert ranking.searches[1].normalized_top_score == 0.75
    assert ranking.pre_dedup_candidate_count == 7
    assert ranking.duplicate_candidate_count == 1
    assert ranking.post_dedup_candidate_count == 6
    assert sum(search.post_dedup_contribution_count for search in ranking.searches) == 6
    assert [candidate.chunk.id for candidate in ranking.candidates] == [
        "top",
        "duplicate",
        "boundary-first",
        "boundary-early",
        "boundary-late",
    ]
    assert ranking.candidates[-3].normalized_score == 0.25


def test_non_positive_global_max_rejects_every_scope() -> None:
    raw = {
        "a": [(_chunk("a", "a/file.py"), 0.0)],
        "b": [(_chunk("b", "b/file.py"), -1.0)],
    }

    ranking = rank_scope_candidates(
        _scope_map("a", "b"),
        "needle",
        lambda scope, _query, _limit: raw[scope],
    )

    assert ranking is not None
    assert ranking.repository_global_max_raw_score == 0.0
    assert ranking.candidates == ()
    assert all(search.reason == "non_positive_global_max" for search in ranking.searches)
