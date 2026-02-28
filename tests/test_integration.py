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
