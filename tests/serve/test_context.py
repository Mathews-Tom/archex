"""Tests for context assembly and renderers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from archex.index.graph import DependencyGraph
from archex.models import CodeChunk, ContextBundle, SymbolKind
from archex.serve.context import assemble_context
from archex.serve.renderers.json import render_json
from archex.serve.renderers.markdown import render_markdown
from archex.serve.renderers.xml import render_xml

# ruff: noqa: I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    chunk_id: str,
    file_path: str,
    content: str = "def foo(): pass",
    symbol_kind: SymbolKind | None = None,
    symbol_name: str | None = None,
    token_count: int = 10,
) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=5,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
        language="python",
        token_count=token_count,
    )


def make_graph_with_edges() -> DependencyGraph:
    graph = DependencyGraph()
    graph.add_file_node("auth.py")
    graph.add_file_node("models.py")
    graph.add_file_node("utils.py")
    graph.add_file_edge("auth.py", "models.py", kind="imports")
    graph.add_file_edge("utils.py", "auth.py", kind="imports")
    return graph


# ---------------------------------------------------------------------------
# assemble_context tests
# ---------------------------------------------------------------------------


def test_assemble_returns_context_bundle() -> None:
    graph = make_graph_with_edges()
    chunk = make_chunk("c1", "auth.py", token_count=50)
    results = [(chunk, 1.0)]
    bundle = assemble_context(results, graph, [chunk], "how does auth work?", token_budget=500)
    assert isinstance(bundle, ContextBundle)
    assert bundle.query == "how does auth work?"


def test_token_count_within_budget() -> None:
    graph = make_graph_with_edges()
    chunks = [make_chunk(f"c{i}", "auth.py", token_count=100) for i in range(10)]
    results = [(c, float(i + 1)) for i, c in enumerate(chunks)]
    bundle = assemble_context(results, graph, chunks, "query", token_budget=250)
    assert bundle.token_count <= 250


def test_truncated_flag_when_budget_exceeded() -> None:
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunks = [make_chunk(f"c{i}", "a.py", token_count=100) for i in range(5)]
    results = [(c, float(i + 1)) for i, c in enumerate(chunks)]
    bundle = assemble_context(results, graph, chunks, "q", token_budget=150)
    assert bundle.truncated is True


def test_not_truncated_when_budget_sufficient() -> None:
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunks = [make_chunk(f"c{i}", "a.py", token_count=10) for i in range(3)]
    results = [(c, float(i + 1)) for i, c in enumerate(chunks)]
    bundle = assemble_context(results, graph, chunks, "q", token_budget=1000)
    assert bundle.truncated is False


def test_chunks_ranked_by_score_descending() -> None:
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    c1 = make_chunk("c1", "a.py", token_count=10)
    c2 = make_chunk("c2", "a.py", token_count=10)
    c3 = make_chunk("c3", "a.py", token_count=10)
    # c3 has highest BM25
    results = [(c1, 1.0), (c2, 2.0), (c3, 3.0)]
    bundle = assemble_context(results, graph, [c1, c2, c3], "q", token_budget=1000)
    scores = [rc.final_score for rc in bundle.chunks]
    assert scores == sorted(scores, reverse=True)


def test_structural_expansion_adds_neighbor_chunks() -> None:
    graph = make_graph_with_edges()
    auth_chunk = make_chunk("c_auth", "auth.py", token_count=10)
    models_chunk = make_chunk("c_models", "models.py", token_count=10)
    all_chunks = [auth_chunk, models_chunk]
    # Only search result is auth.py — models.py is a neighbor
    results = [(auth_chunk, 1.0)]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=1000)
    included_ids = {rc.chunk.id for rc in bundle.chunks}
    assert "c_models" in included_ids


def test_type_definitions_extracted_from_class_chunks() -> None:
    graph = DependencyGraph()
    graph.add_file_node("models.py")
    class_chunk = make_chunk(
        "cls1",
        "models.py",
        content="class User: pass",
        symbol_kind=SymbolKind.CLASS,
        symbol_name="User",
        token_count=10,
    )
    results = [(class_chunk, 1.0)]
    bundle = assemble_context(results, graph, [class_chunk], "q", token_budget=1000)
    assert len(bundle.type_definitions) == 1
    assert bundle.type_definitions[0].symbol == "User"


def test_file_tree_built_from_included_chunks() -> None:
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    graph.add_file_node("b.py")
    ca = make_chunk("ca", "a.py", token_count=10)
    cb = make_chunk("cb", "b.py", token_count=10)
    results = [(ca, 1.0), (cb, 0.5)]
    bundle = assemble_context(results, graph, [ca, cb], "q", token_budget=1000)
    assert "a.py" in bundle.structural_context.file_tree
    assert "b.py" in bundle.structural_context.file_tree


def test_empty_search_results_returns_empty_bundle() -> None:
    graph = DependencyGraph()
    bundle = assemble_context([], graph, [], "q", token_budget=1000)
    assert bundle.chunks == []
    assert bundle.token_count == 0
    assert bundle.truncated is False


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


def _make_bundle() -> ContextBundle:
    chunk = make_chunk(
        "c1",
        "src/auth.py",
        content="def authenticate(): pass",
        symbol_name="authenticate",
        token_count=5,
    )
    from archex.models import RankedChunk, StructuralContext

    rc = RankedChunk(chunk=chunk, relevance_score=0.9, final_score=0.85)
    return ContextBundle(
        query="How does auth work?",
        chunks=[rc],
        structural_context=StructuralContext(file_tree="src/auth.py"),
        token_count=5,
        token_budget=1000,
    )


def test_xml_renderer_produces_valid_xml() -> None:
    bundle = _make_bundle()
    xml_str = render_xml(bundle)
    # Must parse without error
    root = ET.fromstring(xml_str)
    assert root.tag == "context"


def test_xml_renderer_includes_query_attribute() -> None:
    bundle = _make_bundle()
    xml_str = render_xml(bundle)
    assert 'query="How does auth work?"' in xml_str


def test_json_renderer_produces_valid_json() -> None:
    bundle = _make_bundle()
    json_str = render_json(bundle)
    parsed = json.loads(json_str)
    assert parsed["query"] == "How does auth work?"


def test_json_renderer_includes_chunks() -> None:
    bundle = _make_bundle()
    parsed = json.loads(render_json(bundle))
    assert len(parsed["chunks"]) == 1


def test_markdown_renderer_includes_query() -> None:
    bundle = _make_bundle()
    md = render_markdown(bundle)
    assert "How does auth work?" in md


def test_markdown_renderer_includes_file_tree() -> None:
    bundle = _make_bundle()
    md = render_markdown(bundle)
    assert "src/auth.py" in md


def test_markdown_renderer_includes_chunk_header() -> None:
    bundle = _make_bundle()
    md = render_markdown(bundle)
    assert "authenticate" in md
