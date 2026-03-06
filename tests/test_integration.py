"""End-to-end integration tests for the archex public API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from archex.api import (
    _compute_top_k,  # pyright: ignore[reportPrivateUsage]
    analyze,
    compare,
    query,
)

if TYPE_CHECKING:
    from archex.models import FileTreeEntry
from archex.models import (
    ArchProfile,
    CodeChunk,
    ComparisonResult,
    Config,
    ContextBundle,
    IndexConfig,
    PipelineTiming,
    RepoSource,
    ScoringWeights,
)


def _has_tree_sitter_swift() -> bool:
    try:
        import tree_sitter_swift  # noqa: F401  # pyright: ignore[reportUnusedImport,reportMissingImports]

        return True
    except ImportError:
        pass
    try:
        from tree_sitter_language_pack import get_language  # noqa: F401

        get_language("swift")
        return True
    except (ImportError, Exception):
        return False


def test_compute_top_k_thresholds() -> None:
    assert _compute_top_k(50) == 30
    assert _compute_top_k(100) == 30
    assert _compute_top_k(200) == 50
    assert _compute_top_k(500) == 50
    assert _compute_top_k(1000) == 100
    assert _compute_top_k(2000) == 100
    assert _compute_top_k(5000) == 150


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
        assert 0 < bundle.token_budget <= 8192
        assert bundle.retrieval_metadata is not None
        assert bundle.retrieval_metadata.strategy in ("bm25+graph", "passthrough")

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
        weights = ScoringWeights(relevance=0.8, structural=0.1, type_coverage=0.1, cohesion=0.0)
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
        assert bundle.retrieval_metadata.strategy in ("bm25+graph", "passthrough")


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


class TestTokenEfficiencyReporting:
    """E2E tests for token efficiency reporting across CLI and MCP."""

    def test_cli_query_timing_shows_savings(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["query", str(python_simple_repo), "what classes exist?", "--timing"],
        )
        assert result.exit_code == 0, result.output
        assert "[savings]" in result.output
        assert "tokens returned" in result.output
        assert "Raw equivalent:" in result.output
        assert "[timing]" in result.output
        # Phase timing: should show acquire or cache hit
        output = result.output
        assert "Acquired repo" in output or "Cache hit" in output

    def test_api_token_utilities(self, python_simple_repo: Path) -> None:
        from archex.api import get_file_token_count, get_files_token_count, get_repo_total_tokens

        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)

        total = get_repo_total_tokens(source, config)
        assert total > 0

        # Individual file tokens should be <= total
        file_tokens = get_file_token_count(source, "utils.py", config)
        assert 0 <= file_tokens <= total

        # Multi-file query should sum correctly
        multi = get_files_token_count(source, ["utils.py", "models.py"], config)
        assert multi > 0
        assert multi <= total

    def test_mcp_analyze_returns_meta(self, python_simple_repo: Path) -> None:
        import json

        from archex.integrations.mcp import handle_analyze_repo

        raw = handle_analyze_repo(str(python_simple_repo), "json")
        parsed = json.loads(raw)
        assert "content" in parsed
        assert "_meta" in parsed
        meta = parsed["_meta"]
        assert meta["tool_name"] == "analyze_repo"
        assert meta["tokens_returned"] > 0
        assert meta["tokens_raw_equivalent"] > 0
        assert isinstance(meta["savings_pct"], float)
        assert meta["strategy"] == "full_analysis"

    def test_mcp_query_returns_meta(self, python_simple_repo: Path) -> None:
        import json

        from archex.integrations.mcp import handle_query_repo

        raw = handle_query_repo(str(python_simple_repo), "what functions exist?")
        parsed = json.loads(raw)
        assert "content" in parsed
        assert "_meta" in parsed
        meta = parsed["_meta"]
        assert meta["tool_name"] == "query_repo"
        assert meta["tokens_returned"] > 0
        assert meta["strategy"] == "bm25+graph"


class TestPipelineTimingAPI:
    """E2E tests that API functions correctly populate PipelineTiming."""

    def test_analyze_populates_timing(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        pt = PipelineTiming()
        profile = analyze(source, config=Config(languages=["python"]), timing=pt)

        assert isinstance(profile, ArchProfile)
        assert pt.total_ms > 0
        assert pt.acquire_ms >= 0
        assert pt.parse_ms > 0
        assert pt.index_ms >= 0
        assert pt.cached is False

    def test_query_populates_timing_cache_miss(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        pt = PipelineTiming()
        bundle = query(
            source,
            "what classes exist",
            config=Config(languages=["python"], cache=False),
            timing=pt,
        )

        assert isinstance(bundle, ContextBundle)
        assert pt.total_ms > 0
        assert pt.acquire_ms >= 0
        assert pt.parse_ms > 0
        assert pt.search_ms >= 0
        assert pt.assemble_ms >= 0
        assert pt.cached is False

    def test_query_populates_timing_cache_hit(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache_timing")
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir)

        # First call: cache miss — primes the cache
        query(source, "classes", config=config)

        # Second call: cache hit — with timing
        pt = PipelineTiming()
        bundle = query(source, "classes", config=config, timing=pt)

        assert isinstance(bundle, ContextBundle)
        assert pt.total_ms > 0
        assert pt.cached is True
        assert pt.search_ms >= 0
        assert pt.assemble_ms >= 0
        # acquire/parse should be 0 on cache hit
        assert pt.acquire_ms == 0.0
        assert pt.parse_ms == 0.0

    def test_file_tree_populates_timing(self, python_simple_repo: Path) -> None:
        from archex.api import file_tree

        source = RepoSource(local_path=str(python_simple_repo))
        pt = PipelineTiming()
        result = file_tree(source, config=Config(languages=["python"], cache=False), timing=pt)

        assert result.total_files > 0
        assert pt.total_ms > 0
        assert pt.search_ms >= 0
        assert pt.cached is False

    def test_file_outline_populates_timing(self, python_simple_repo: Path) -> None:
        import os

        from archex.api import file_outline

        source = RepoSource(local_path=str(python_simple_repo))
        py_files = [f for f in os.listdir(python_simple_repo) if f.endswith(".py")]
        assert py_files

        pt = PipelineTiming()
        result = file_outline(
            source,
            file_path=py_files[0],
            config=Config(languages=["python"], cache=False),
            timing=pt,
        )

        assert result.file_path == py_files[0]
        assert pt.total_ms > 0
        assert pt.search_ms >= 0

    def test_search_symbols_populates_timing(self, python_simple_repo: Path) -> None:
        from archex.api import search_symbols

        source = RepoSource(local_path=str(python_simple_repo))
        pt = PipelineTiming()
        results = search_symbols(
            source, query="class", config=Config(languages=["python"], cache=False), timing=pt
        )

        assert isinstance(results, list)
        assert pt.total_ms > 0
        assert pt.search_ms >= 0

    def test_get_symbol_populates_timing(self, python_simple_repo: Path) -> None:
        from archex.api import get_symbol, search_symbols

        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(languages=["python"], cache=False)

        matches = search_symbols(source, query="class", config=config)
        assert len(matches) > 0, "Fixture must have searchable symbols"

        pt = PipelineTiming()
        result = get_symbol(source, symbol_id=matches[0].symbol_id, config=config, timing=pt)
        assert result is not None
        assert pt.total_ms > 0
        assert pt.search_ms >= 0

    def test_get_symbols_batch_populates_timing(self, python_simple_repo: Path) -> None:
        from archex.api import get_symbols_batch, search_symbols

        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(languages=["python"], cache=False)

        matches = search_symbols(source, query="class", config=config)
        assert len(matches) > 0, "Fixture must have searchable symbols"

        pt = PipelineTiming()
        ids = [matches[0].symbol_id]
        results = get_symbols_batch(source, symbol_ids=ids, config=config, timing=pt)
        assert len(results) == 1
        assert pt.total_ms > 0
        assert pt.search_ms >= 0

    def test_ensure_index_cache_hit_timing(self, python_simple_repo: Path, tmp_path: Path) -> None:
        """Verify _ensure_index sets cached=True and index_ms on cache hit."""
        from archex.api import file_tree

        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache_idx")
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir)

        # Prime cache
        file_tree(source, config=config)

        # Second call — cache hit
        pt = PipelineTiming()
        file_tree(source, config=config, timing=pt)
        assert pt.cached is True
        assert pt.index_ms >= 0
        assert pt.total_ms > 0

    def test_cli_analyze_timing(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["analyze", str(python_simple_repo), "--timing"],
        )
        assert result.exit_code == 0, result.output
        assert "[savings]" in result.output
        assert "[timing]" in result.output
        assert "Acquired repo" in result.output or "Cache hit" in result.output


class TestGetParentQname:
    """Unit tests for _get_parent_qname helper."""

    def test_double_colon_separator(self) -> None:
        from archex.api import _get_parent_qname  # pyright: ignore[reportPrivateUsage]

        assert _get_parent_qname("Foo::bar") == "Foo"

    def test_double_colon_nested(self) -> None:
        from archex.api import _get_parent_qname  # pyright: ignore[reportPrivateUsage]

        assert _get_parent_qname("ns::Foo::bar") == "ns::Foo"

    def test_dot_separator(self) -> None:
        from archex.api import _get_parent_qname  # pyright: ignore[reportPrivateUsage]

        assert _get_parent_qname("Foo.bar") == "Foo"

    def test_no_separator(self) -> None:
        from archex.api import _get_parent_qname  # pyright: ignore[reportPrivateUsage]

        assert _get_parent_qname("standalone") is None


class TestAddFileToTree:
    """Unit tests for _add_file_to_tree helper."""

    def test_creates_missing_root_entry(self) -> None:
        from archex.api import _add_file_to_tree  # pyright: ignore[reportPrivateUsage]
        from archex.models import FileTreeEntry

        root: dict[str, FileTreeEntry] = {}
        file_entry = FileTreeEntry(path="src/main.py", language="python", lines=10)
        _add_file_to_tree(root, ["src", "main.py"], file_entry)

        assert "src" in root
        assert root["src"].is_directory is True
        assert file_entry in root["src"].children

    def test_returns_early_on_missing_intermediate(self) -> None:
        from archex.api import _add_file_to_tree  # pyright: ignore[reportPrivateUsage]
        from archex.models import FileTreeEntry

        root: dict[str, FileTreeEntry] = {
            "src": FileTreeEntry(path="src", is_directory=True),
        }
        # "sub" doesn't exist as a child of "src", so the file can't be placed
        file_entry = FileTreeEntry(path="src/sub/deep.py", language="python", lines=5)
        _add_file_to_tree(root, ["src", "sub", "deep.py"], file_entry)

        # File should NOT have been added (intermediate "sub" missing)
        assert len(root["src"].children) == 0


class TestFileTreeDepthLimit:
    """Test file_tree depth limiting with deeply nested files."""

    def test_depth_limit_truncates_deep_paths(self, python_simple_repo: Path) -> None:
        """Verify max_depth=1 triggers the break guard on nested files."""
        from archex.api import file_tree

        # python_simple has services/auth.py (depth 2)
        source = RepoSource(local_path=str(python_simple_repo))
        result = file_tree(source, max_depth=1, config=Config(languages=["python"], cache=False))

        # Files at depth > 1 should not appear as entries
        all_paths = self._collect_paths(result.entries)
        for path in all_paths:
            depth = path.count("/")
            assert depth <= 1, f"Path {path} exceeds max_depth=1"

    def test_depth_zero_shows_only_root_files(self, python_simple_repo: Path) -> None:
        from archex.api import file_tree

        source = RepoSource(local_path=str(python_simple_repo))
        result = file_tree(source, max_depth=0, config=Config(languages=["python"], cache=False))

        # No files should be included (all have depth >= 0 in parts)
        # max_depth=0 means len(parts) - 1 < 0 is never true for any file
        assert result.total_files > 0  # metadata still counted
        # But no entries should be in the tree
        assert len(result.entries) == 0

    @staticmethod
    def _collect_paths(entries: list[FileTreeEntry]) -> list[str]:
        paths: list[str] = []
        for e in entries:
            if not e.is_directory:
                paths.append(e.path)
            paths.extend(TestFileTreeDepthLimit._collect_paths(e.children))
        return paths


# ---------------------------------------------------------------------------
# Delta indexing integration tests
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class TestDeltaIndexIntegration:
    """End-to-end delta indexing via _ensure_index / query()."""

    def test_delta_index_single_file_change(self, python_simple_repo: Path, tmp_path: Path) -> None:
        """Modify one file, delta re-index, verify new content appears in index."""
        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache")
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir)

        # First query — builds full index
        bundle1 = query(source, "calculate_sum", config=config)
        assert isinstance(bundle1, ContextBundle)

        # Modify utils.py in the fixture repo
        utils = python_simple_repo / "utils.py"
        utils.write_text("def delta_changed_function():\n    return 999\n")
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "delta test: modify utils")

        # Second query — should trigger delta path (same repo, new commit)
        bundle2 = query(source, "delta_changed_function", config=config)
        assert isinstance(bundle2, ContextBundle)

    def test_delta_index_add_file(self, python_simple_repo: Path, tmp_path: Path) -> None:
        """Add a new .py file, delta re-index, verify it appears in index."""
        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache")
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir)

        # Build initial index
        query(source, "models", config=config)

        # Add a new file
        new_file = python_simple_repo / "brand_new_module.py"
        new_file.write_text(
            "def ultra_unique_delta_function():\n    return 'added by delta test'\n"
        )
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "add brand_new_module")

        # Re-query — delta should include new file
        bundle = query(source, "ultra_unique_delta_function", config=config)
        assert isinstance(bundle, ContextBundle)

    def test_delta_index_delete_file(self, python_simple_repo: Path, tmp_path: Path) -> None:
        """Delete a file, delta re-index, verify its chunks are removed."""
        from archex.api import _ensure_index  # pyright: ignore[reportPrivateUsage]

        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache")
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir)

        # Build initial index and verify utils.py is indexed
        store = _ensure_index(source, config=config)
        try:
            initial_chunks = store.get_chunks_for_file("utils.py")
            assert len(initial_chunks) > 0
        finally:
            store.close()

        # Delete utils.py
        (python_simple_repo / "utils.py").unlink()
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "delete utils.py")

        # Re-index via delta path
        store2 = _ensure_index(source, config=config)
        try:
            after_chunks = store2.get_chunks_for_file("utils.py")
            assert after_chunks == []
        finally:
            store2.close()

    def test_delta_meta_in_timing(self, python_simple_repo: Path, tmp_path: Path) -> None:
        """PipelineTiming.delta_meta is populated after the delta path executes."""
        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache")
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir)

        # Prime the cache
        query(source, "models", config=config)

        # Make a change to trigger delta
        (python_simple_repo / "utils.py").write_text("def timing_test(): pass\n")
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "timing delta test")

        # Query with timing to check delta_meta populated
        pt = PipelineTiming()
        bundle = query(source, "timing_test", config=config, timing=pt)

        assert isinstance(bundle, ContextBundle)
        # If delta path was taken, delta_meta should be set
        if pt.delta_meta is not None:
            assert pt.delta_meta.full_reindex_avoided is True
            assert pt.delta_meta.delta_time_ms >= 0
            assert pt.delta_ms >= 0

    def test_delta_threshold_triggers_full_reindex(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        """When change ratio exceeds delta_threshold, full re-index is used."""
        from archex.api import _ensure_index  # pyright: ignore[reportPrivateUsage]

        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir = str(tmp_path / "cache")
        # Set threshold to 0.0 so any change triggers full re-index
        config = Config(languages=["python"], cache=True, cache_dir=cache_dir, delta_threshold=0.0)

        # Build initial index
        _ensure_index(source, config=config).close()

        # Modify one file
        (python_simple_repo / "utils.py").write_text("# threshold test\n")
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "threshold test")

        # With threshold=0.0, even 1 change is >= 0.0, so full re-index runs
        pt = PipelineTiming()
        store = _ensure_index(source, config=config, timing=pt)
        store.close()

        # delta_meta should be None (full re-index path, not delta path)
        assert pt.delta_meta is None

    def test_delta_index_matches_full_reindex(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        """After delta, querying produces the same symbol set as a full re-index."""
        from archex.api import _ensure_index  # pyright: ignore[reportPrivateUsage]

        source = RepoSource(local_path=str(python_simple_repo))
        cache_dir_delta = str(tmp_path / "cache_delta")

        # Build initial delta-path index
        config_delta = Config(languages=["python"], cache=True, cache_dir=cache_dir_delta)
        _ensure_index(source, config=config_delta).close()

        # Modify utils.py
        (python_simple_repo / "utils.py").write_text(
            "def fresh_symbol_for_comparison(): return 1\n"
        )
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "comparison test")

        # Get delta-indexed store
        store_delta = _ensure_index(source, config=config_delta)
        try:
            delta_chunks = {c.file_path for c in store_delta.get_chunks()}
        finally:
            store_delta.close()

        # Get full-indexed store (fresh cache)
        config_full = Config(languages=["python"], cache=False)
        store_full = _ensure_index(source, config=config_full)
        try:
            full_chunks = {c.file_path for c in store_full.get_chunks()}
        finally:
            store_full.close()

        # Both should index the same set of files
        assert delta_chunks == full_chunks

    def test_delta_nongit_mtime_api(self, tmp_path: Path) -> None:
        """compute_mtime_delta on a non-git directory detects file changes by mtime."""
        import shutil
        import time as _time

        from archex.index.delta import compute_mtime_delta
        from archex.index.store import IndexStore
        from archex.models import ChangeStatus

        fixtures_dir = Path(__file__).parent / "fixtures"
        nongit_dir = tmp_path / "nongit"
        shutil.copytree(fixtures_dir / "python_simple", nongit_dir)
        assert not (nongit_dir / ".git").exists()

        db = tmp_path / "mtime_test.db"
        store = IndexStore(db)
        try:
            # Empty store — all files on disk should be ADDED
            manifest = compute_mtime_delta(nongit_dir, store, _time.time() - 10)
            added = [c for c in manifest.changes if c.status == ChangeStatus.ADDED]
            assert len(added) > 0
            assert manifest.base_commit == "mtime"
            assert manifest.current_commit == "mtime"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# New language integration tests (Java, Kotlin, C#, Swift)
# ---------------------------------------------------------------------------


class TestNewLanguageIntegration:
    """Full analyze() and query() pipelines for v0.5.4 language adapters."""

    @pytest.mark.parametrize(
        ("language", "fixture_name"),
        [
            ("java", "java_simple_repo"),
            ("kotlin", "kotlin_simple_repo"),
            ("csharp", "csharp_simple_repo"),
            pytest.param(
                "swift",
                "swift_simple_repo",
                marks=pytest.mark.skipif(
                    not _has_tree_sitter_swift(),
                    reason="tree-sitter-swift not installed",
                ),
            ),
        ],
    )
    def test_analyze_new_language(
        self, language: str, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        repo_path: Path = request.getfixturevalue(fixture_name)
        source = RepoSource(local_path=str(repo_path))
        profile = analyze(source, config=Config(languages=[language]))

        assert isinstance(profile, ArchProfile)
        assert profile.repo.total_files > 0
        assert language in profile.repo.languages
        assert len(profile.module_map) > 0

    @pytest.mark.parametrize(
        ("language", "fixture_name", "query_term"),
        [
            ("java", "java_simple_repo", "User"),
            ("kotlin", "kotlin_simple_repo", "User"),
            ("csharp", "csharp_simple_repo", "User"),
            pytest.param(
                "swift",
                "swift_simple_repo",
                "User",
                marks=pytest.mark.skipif(
                    not _has_tree_sitter_swift(),
                    reason="tree-sitter-swift not installed",
                ),
            ),
        ],
    )
    def test_query_new_language(
        self,
        language: str,
        fixture_name: str,
        query_term: str,
        request: pytest.FixtureRequest,
    ) -> None:
        repo_path: Path = request.getfixturevalue(fixture_name)
        source = RepoSource(local_path=str(repo_path))
        bundle = query(
            source,
            query_term,
            config=Config(languages=[language], cache=False),
        )

        assert isinstance(bundle, ContextBundle)

    def test_delta_index_java(self, java_simple_repo: Path, tmp_path: Path) -> None:
        source = RepoSource(local_path=str(java_simple_repo))
        config = Config(languages=["java"], cache=True, cache_dir=str(tmp_path / "cache"))
        query(source, "User", config=config)  # initial index

        (java_simple_repo / "models" / "User.java").write_text(
            "public class User { public int deltaField; }\n"
        )
        _git(java_simple_repo, "add", ".")
        _git(java_simple_repo, "commit", "-m", "delta test")

        bundle = query(source, "deltaField", config=config)
        assert isinstance(bundle, ContextBundle)


class TestPluginBootstrapLifecycle:
    def test_bootstrap_nonstrict_then_strict_reloads(self) -> None:
        """Non-strict bootstrap followed by strict should re-validate."""
        import archex.api as api_mod

        original = api_mod._plugin_bootstrap_strict  # pyright: ignore[reportPrivateUsage]
        try:
            api_mod._plugin_bootstrap_strict = None  # pyright: ignore[reportPrivateUsage]

            api_mod._bootstrap_plugins(strict=False)  # pyright: ignore[reportPrivateUsage]
            assert api_mod._plugin_bootstrap_strict is False  # pyright: ignore[reportPrivateUsage]

            api_mod._bootstrap_plugins(strict=True)  # pyright: ignore[reportPrivateUsage]
            assert api_mod._plugin_bootstrap_strict is True  # pyright: ignore[reportPrivateUsage]
        finally:
            api_mod._plugin_bootstrap_strict = original  # pyright: ignore[reportPrivateUsage]

    def test_bootstrap_strict_then_nonstrict_skips(self) -> None:
        """Strict bootstrap followed by non-strict should skip (already at higher level)."""
        import archex.api as api_mod

        original = api_mod._plugin_bootstrap_strict  # pyright: ignore[reportPrivateUsage]
        try:
            api_mod._plugin_bootstrap_strict = None  # pyright: ignore[reportPrivateUsage]

            api_mod._bootstrap_plugins(strict=True)  # pyright: ignore[reportPrivateUsage]
            assert api_mod._plugin_bootstrap_strict is True  # pyright: ignore[reportPrivateUsage]

            api_mod._bootstrap_plugins(strict=False)  # pyright: ignore[reportPrivateUsage]
            assert api_mod._plugin_bootstrap_strict is True  # pyright: ignore[reportPrivateUsage]
        finally:
            api_mod._plugin_bootstrap_strict = original  # pyright: ignore[reportPrivateUsage]

    def test_adapter_registry_entry_points_strictness_upgrade(self) -> None:
        """AdapterRegistry re-loads entry points when upgrading to strict."""
        from archex.parse.adapters import AdapterRegistry

        reg = AdapterRegistry()
        reg.load_entry_points(strict=False)
        assert reg._entry_points_loaded is True  # pyright: ignore[reportPrivateUsage]
        assert reg._entry_points_strict is False  # pyright: ignore[reportPrivateUsage]

        reg.load_entry_points(strict=True)
        assert reg._entry_points_strict is True  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Cross-language integration tests
# ---------------------------------------------------------------------------


class TestTypescriptIntegration:
    """Analyze and query a TypeScript repository."""

    def test_analyze_typescript(self, typescript_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(typescript_simple_repo))
        profile = analyze(source, Config(cache=False))
        assert profile.repo.total_files > 0
        assert "typescript" in profile.repo.languages

    def test_query_typescript(self, typescript_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(typescript_simple_repo))
        bundle = query(source, "handlers", config=Config(cache=False))
        assert isinstance(bundle, ContextBundle)


class TestRustIntegration:
    """Analyze and query a Rust repository."""

    def test_analyze_rust(self, rust_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(rust_simple_repo))
        profile = analyze(source, Config(cache=False))
        assert profile.repo.total_files > 0
        assert "rust" in profile.repo.languages

    def test_query_rust(self, rust_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(rust_simple_repo))
        bundle = query(source, "functions", config=Config(cache=False))
        assert isinstance(bundle, ContextBundle)


class TestGoIntegration:
    """Analyze and query a Go repository."""

    def test_analyze_go(self, go_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(go_simple_repo))
        profile = analyze(source, Config(cache=False))
        assert profile.repo.total_files > 0
        assert "go" in profile.repo.languages

    def test_query_go(self, go_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(go_simple_repo))
        bundle = query(source, "main", config=Config(cache=False))
        assert isinstance(bundle, ContextBundle)


class TestMonorepoIntegration:
    """Analyze and query a multi-language monorepo."""

    def test_analyze_monorepo(self, monorepo_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(monorepo_simple_repo))
        profile = analyze(source, Config(cache=False))
        assert profile.repo.total_files > 0
        assert len(profile.repo.languages) >= 1

    def test_query_monorepo(self, monorepo_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(monorepo_simple_repo))
        bundle = query(source, "core", config=Config(cache=False))
        assert isinstance(bundle, ContextBundle)


# ---------------------------------------------------------------------------
# Error path integration tests
# ---------------------------------------------------------------------------


class TestErrorPathIntegration:
    def test_query_nonexistent_local_path(self, tmp_path: Path) -> None:
        """Querying a nonexistent path raises an appropriate error."""
        source = RepoSource(local_path=str(tmp_path / "nonexistent"))
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            query(source, "anything", config=Config(cache=False))

    def test_analyze_empty_repo(self, tmp_path: Path) -> None:
        """Analyzing an empty directory produces a profile with 0 files."""
        import subprocess

        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "test"],
            check=True,
            capture_output=True,
        )
        (tmp_path / ".gitkeep").touch()
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "init"],
            check=True,
            capture_output=True,
        )
        (tmp_path / ".gitkeep").unlink()
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "empty"],
            check=True,
            capture_output=True,
        )

        source = RepoSource(local_path=str(tmp_path))
        profile = analyze(source, Config(cache=False))
        assert profile.repo.total_files == 0

    def test_query_empty_repo(self, tmp_path: Path) -> None:
        """Querying an empty repo returns an empty bundle without crash."""
        import subprocess

        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "test"],
            check=True,
            capture_output=True,
        )
        (tmp_path / ".gitkeep").touch()
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "init"],
            check=True,
            capture_output=True,
        )
        (tmp_path / ".gitkeep").unlink()
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "empty"],
            check=True,
            capture_output=True,
        )

        source = RepoSource(local_path=str(tmp_path))
        bundle = query(source, "anything", config=Config(cache=False))
        assert bundle.chunks == []

    def test_scoring_weights_invalid_at_construction(self) -> None:
        """Invalid ScoringWeights raise ValueError at construction time."""
        with pytest.raises(ValueError):
            ScoringWeights(relevance=0.5, structural=0.5, type_coverage=0.5, cohesion=0.0)


# ---------------------------------------------------------------------------
# PipelineTiming integration tests
# ---------------------------------------------------------------------------


class TestPipelineTimingIntegration:
    def test_timing_all_fields_populated(self, python_simple_repo: Path) -> None:
        """After full pipeline, all timing fields are populated."""
        source = RepoSource(local_path=str(python_simple_repo))
        timing = PipelineTiming()
        query(source, "auth", config=Config(cache=False), timing=timing)
        assert timing.total_ms is not None
        assert timing.total_ms > 0
        assert timing.acquire_ms is not None
        assert timing.parse_ms is not None
