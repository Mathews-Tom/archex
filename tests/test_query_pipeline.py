"""Integration test for the full query → assemble_context → render pipeline.

Exercises the real pipeline end-to-end with the python_simple fixture,
verifying that query results contain expected files and that rendering
produces valid output in all three formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.api import query
from archex.models import (
    Config,
    ContextBundle,
    PipelineTiming,
    RepoSource,
    ScoringWeights,
)
from archex.serve.renderers.json import render_json
from archex.serve.renderers.markdown import render_markdown
from archex.serve.renderers.xml import render_xml

if TYPE_CHECKING:
    from pathlib import Path


class TestQueryPipelineEndToEnd:
    """Full query() → assemble_context() → render path with real fixture."""

    def test_query_returns_chunks_with_content(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        bundle = query(source, "How does authentication work?", config=config)

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.chunks) > 0
        # All returned RankedChunks should wrap CodeChunks with content
        for ranked in bundle.chunks:
            assert ranked.chunk.content
            assert ranked.chunk.file_path

    def test_query_retrieval_metadata_populated(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        bundle = query(source, "What models are defined?", config=config)

        meta = bundle.retrieval_metadata
        assert meta.retrieval_time_ms > 0
        assert meta.candidates_found > 0
        assert meta.chunks_included > 0

    def test_query_finds_relevant_files(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        bundle = query(source, "User model and Role enum", config=config)

        file_paths = {rc.chunk.file_path for rc in bundle.chunks}
        # models.py defines User and Role — should appear in results
        assert "models.py" in file_paths

    def test_query_with_timing(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        timing = PipelineTiming()
        bundle = query(source, "authentication", config=config, timing=timing)

        assert timing.total_ms > 0
        assert timing.search_ms is not None or timing.strategy == "passthrough"
        assert len(bundle.chunks) > 0

    def test_query_with_custom_weights(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        weights = ScoringWeights(relevance=1.0, structural=0.0, type_coverage=0.0, cohesion=0.0)
        bundle = query(source, "validate email", config=config, scoring_weights=weights)

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.chunks) > 0

    def test_query_with_small_budget(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        # Very small budget should still return some results
        bundle = query(source, "authentication", token_budget=100, config=config)

        assert isinstance(bundle, ContextBundle)
        total_tokens = sum(rc.chunk.token_count for rc in bundle.chunks)
        # Should respect budget (with some tolerance for the minimum chunk)
        assert total_tokens <= 200


class TestRenderPipeline:
    """Verify rendering produces valid output from query results."""

    def test_render_markdown(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        bundle = query(source, "What classes exist?", config=config)

        md = render_markdown(bundle)
        assert isinstance(md, str)
        assert len(md) > 0
        # Markdown should contain file path headers
        assert "```" in md

    def test_render_json(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        bundle = query(source, "What classes exist?", config=config)

        json_str = render_json(bundle)
        assert isinstance(json_str, str)
        import json

        data = json.loads(json_str)
        assert "chunks" in data
        assert len(data["chunks"]) > 0

    def test_render_xml(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        bundle = query(source, "What classes exist?", config=config)

        xml_str = render_xml(bundle)
        assert isinstance(xml_str, str)
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_str)
        assert root.tag == "context"
        chunks = root.findall(".//chunk")
        assert len(chunks) > 0
