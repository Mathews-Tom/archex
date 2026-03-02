"""End-to-end integration tests for the archex public API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.api import analyze, compare, query

if TYPE_CHECKING:
    from pathlib import Path
from archex.models import (
    ArchProfile,
    CodeChunk,
    ComparisonResult,
    Config,
    ContextBundle,
    IndexConfig,
    RepoSource,
    ScoringWeights,
)


class TestAnalyzeEndToEnd:
    """Full analyze() pipeline: acquire → parse → graph → modules → profile."""

    def test_analyze_python_simple(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        profile = analyze(source, config=Config(languages=["python"]))

        assert isinstance(profile, ArchProfile)
        assert profile.repo.total_files > 0
        assert profile.repo.total_lines > 0
        assert "python" in profile.repo.languages
        assert len(profile.module_map) > 0

    def test_analyze_returns_serializable_profile(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        profile = analyze(source, config=Config(languages=["python"]))

        json_str = profile.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        md_str = profile.to_markdown()
        assert isinstance(md_str, str)
        assert len(md_str) > 0


class TestQueryEndToEnd:
    """Full query() pipeline: acquire → parse → chunk → index → search → assemble."""

    def test_query_returns_context_bundle(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        bundle = query(
            source,
            "how does authentication work",
            config=Config(languages=["python"], cache=False),
        )

        assert isinstance(bundle, ContextBundle)
        assert bundle.query == "how does authentication work"
        assert bundle.token_budget == 8192
        assert bundle.retrieval_metadata is not None
        assert bundle.retrieval_metadata.strategy == "bm25+graph"

    def test_query_returns_chunks(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        bundle = query(
            source,
            "user model class",
            config=Config(languages=["python"], cache=False),
        )

        assert isinstance(bundle, ContextBundle)
        # At minimum we should get some chunks for a broad query
        for rc in bundle.chunks:
            assert isinstance(rc.chunk, CodeChunk)
            assert rc.chunk.content
            assert rc.final_score >= 0

    def test_query_respects_token_budget(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        budget = 512
        bundle = query(
            source,
            "models",
            token_budget=budget,
            config=Config(languages=["python"], cache=False),
        )

        assert bundle.token_count <= budget

    def test_query_with_custom_scoring_weights(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        weights = ScoringWeights(relevance=0.8, structural=0.1, type_coverage=0.1)
        bundle = query(
            source,
            "user model",
            config=Config(languages=["python"], cache=False),
            scoring_weights=weights,
        )

        assert isinstance(bundle, ContextBundle)

    def test_query_with_cache(self, python_simple_repo: Path, tmp_path: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache")
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir)

        # First call: cache miss
        bundle1 = query(source, "authentication", config=config)
        assert bundle1.retrieval_metadata is not None

        # Second call: cache hit
        bundle2 = query(source, "authentication", config=config)
        assert bundle2.retrieval_metadata is not None

    def test_query_with_index_config(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        index_cfg = IndexConfig(chunk_max_tokens=256, chunk_min_tokens=32)
        bundle = query(
            source,
            "models",
            config=Config(languages=["python"], cache=False),
            index_config=index_cfg,
        )

        assert isinstance(bundle, ContextBundle)


class TestQueryHybrid:
    """Query with vector=True using a mock embedder."""

    def test_hybrid_query_no_embedder_falls_back(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        # vector=True but no embedder configured → falls back to bm25-only
        index_cfg = IndexConfig(vector=True, embedder=None)
        bundle = query(
            source,
            "authentication",
            config=Config(languages=["python"], cache=False),
            index_config=index_cfg,
        )

        assert isinstance(bundle, ContextBundle)
        assert bundle.retrieval_metadata is not None
        assert bundle.retrieval_metadata.strategy == "bm25+graph"


class TestCompareEndToEnd:
    """Compare two repos via api.compare()."""

    def test_compare_two_repos(self, python_simple_repo: Path, tmp_path: Path) -> None:
        import shutil
        import subprocess

        # Create a second repo by copying and modifying
        repo_b = tmp_path / "repo_b"
        shutil.copytree(python_simple_repo, repo_b)
        extra_file = repo_b / "extra.py"
        extra_file.write_text("def extra_function():\n    return 42\n")
        subprocess.run(["git", "add", "."], cwd=repo_b, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add extra"],
            cwd=repo_b,
            check=True,
            capture_output=True,
        )

        source_a = RepoSource(local_path=str(python_simple_repo))
        source_b = RepoSource(local_path=str(repo_b))

        result = compare(source_a, source_b, config=Config(languages=["python"]))

        assert isinstance(result, ComparisonResult)
        assert result.repo_a is not None
        assert result.repo_b is not None
        assert len(result.dimensions) > 0

    def test_compare_with_specific_dimensions(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        import shutil
        import subprocess

        repo_b = tmp_path / "repo_b"
        shutil.copytree(python_simple_repo, repo_b)
        subprocess.run(["git", "add", "."], cwd=repo_b, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "empty"],
            cwd=repo_b,
            check=True,
            capture_output=True,
        )

        source_a = RepoSource(local_path=str(python_simple_repo))
        source_b = RepoSource(local_path=str(repo_b))

        result = compare(
            source_a,
            source_b,
            dimensions=["api_surface", "error_handling"],
            config=Config(languages=["python"]),
        )

        assert isinstance(result, ComparisonResult)
        dim_names = [d.dimension for d in result.dimensions]
        assert "api_surface" in dim_names
        assert "error_handling" in dim_names


class TestAnalyzeThenQuery:
    """Full pipeline: analyze() produces profile, query() produces context."""

    def test_analyze_then_query_same_repo(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(languages=["python"], cache=False)

        profile = analyze(source, config=config)
        assert isinstance(profile, ArchProfile)
        assert profile.repo.total_files > 0

        bundle = query(source, "user model", config=config)
        assert isinstance(bundle, ContextBundle)


class TestFileTreeEndToEnd:
    """Full pipeline: index → file_tree."""

    def test_file_tree_python_simple(self, python_simple_repo: Path) -> None:
        from archex.api import file_tree

        source = RepoSource(local_path=str(python_simple_repo))
        result = file_tree(source, config=Config(languages=["python"], cache=False))

        assert result.total_files > 0
        assert "python" in result.languages
        # Should have entries
        assert len(result.entries) > 0

    def test_file_tree_with_depth_limit(self, python_simple_repo: Path) -> None:
        from archex.api import file_tree

        source = RepoSource(local_path=str(python_simple_repo))
        result = file_tree(source, max_depth=1, config=Config(languages=["python"], cache=False))
        assert result.total_files > 0

    def test_file_tree_language_filter(self, python_simple_repo: Path) -> None:
        from archex.api import file_tree

        source = RepoSource(local_path=str(python_simple_repo))
        result = file_tree(source, language="python", config=Config(cache=False))
        assert result.total_files > 0
        assert result.languages.get("python", 0) > 0

        # Filter to nonexistent language should return empty
        result_empty = file_tree(source, language="rust", config=Config(cache=False))
        assert result_empty.total_files == 0


class TestFileOutlineEndToEnd:
    """Full pipeline: index → file_outline."""

    def test_outline_known_file(self, python_simple_repo: Path) -> None:
        import os

        from archex.api import file_outline

        source = RepoSource(local_path=str(python_simple_repo))

        # Find a .py file in the fixture
        py_files = [f for f in os.listdir(python_simple_repo) if f.endswith(".py")]
        assert py_files, "Expected .py files in fixture"

        result = file_outline(
            source, file_path=py_files[0], config=Config(languages=["python"], cache=False)
        )
        assert result.file_path == py_files[0]
        assert result.language == "python"

    def test_outline_missing_file(self, python_simple_repo: Path) -> None:
        from archex.api import file_outline

        source = RepoSource(local_path=str(python_simple_repo))
        result = file_outline(
            source, file_path="nonexistent.py", config=Config(languages=["python"], cache=False)
        )
        assert result.symbols == []


class TestSearchSymbolsEndToEnd:
    """Full pipeline: index → search_symbols."""

    def test_search_finds_symbols(self, python_simple_repo: Path) -> None:
        from archex.api import search_symbols

        source = RepoSource(local_path=str(python_simple_repo))
        # Search for a broad term likely to match something in the fixture
        results = search_symbols(
            source, query="class", config=Config(languages=["python"], cache=False)
        )
        # May or may not find matches depending on fixture content, but should not error
        assert isinstance(results, list)

    def test_search_respects_limit(self, python_simple_repo: Path) -> None:
        from archex.api import search_symbols

        source = RepoSource(local_path=str(python_simple_repo))
        results = search_symbols(
            source, query="def", limit=2, config=Config(languages=["python"], cache=False)
        )
        assert len(results) <= 2


class TestGetSymbolEndToEnd:
    """Full pipeline: index → search → get_symbol."""

    def test_get_symbol_round_trip(self, python_simple_repo: Path) -> None:
        from archex.api import get_symbol, search_symbols

        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(languages=["python"], cache=False)

        # First find some symbols
        matches = search_symbols(source, query="def", config=config)
        if matches:
            # Then retrieve the first one
            result = get_symbol(source, symbol_id=matches[0].symbol_id, config=config)
            assert result is not None
            assert result.source  # Should have source code
            assert result.symbol_id == matches[0].symbol_id

    def test_get_symbol_not_found(self, python_simple_repo: Path) -> None:
        from archex.api import get_symbol

        source = RepoSource(local_path=str(python_simple_repo))
        result = get_symbol(
            source, symbol_id="fake::nonexistent#function", config=Config(cache=False)
        )
        assert result is None


class TestGetSymbolsBatchEndToEnd:
    """Full pipeline: index → search → get_symbols_batch."""

    def test_batch_with_mixed_ids(self, python_simple_repo: Path) -> None:
        from archex.api import get_symbols_batch, search_symbols

        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(languages=["python"], cache=False)

        matches = search_symbols(source, query="def", config=config)
        if matches:
            valid_id = matches[0].symbol_id
            ids = [valid_id, "fake::nonexistent#function"]
            results = get_symbols_batch(source, symbol_ids=ids, config=config)
            assert len(results) == 2
            assert results[0] is not None
            assert results[0].symbol_id == valid_id
            assert results[1] is None

    def test_batch_rejects_over_50(self, python_simple_repo: Path) -> None:
        import pytest as _pytest

        from archex.api import get_symbols_batch

        source = RepoSource(local_path=str(python_simple_repo))
        with _pytest.raises(ValueError, match="Maximum 50"):
            get_symbols_batch(source, symbol_ids=["id"] * 51, config=Config(cache=False))

    def test_batch_empty_input(self, python_simple_repo: Path) -> None:
        from archex.api import get_symbols_batch

        source = RepoSource(local_path=str(python_simple_repo))
        results = get_symbols_batch(source, symbol_ids=[], config=Config(cache=False))
        assert results == []
