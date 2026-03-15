"""Tests for context assembly and renderers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from archex.index.graph import DependencyGraph
from archex.models import CodeChunk, ContextBundle, Module, SymbolKind
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


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_token_budget_zero_truncates_all() -> None:
    """token_budget=0 includes no chunks, truncated=True."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunk = make_chunk("c1", "a.py", token_count=10)
    results = [(chunk, 1.0)]
    bundle = assemble_context(results, graph, [chunk], "q", token_budget=0)
    assert bundle.chunks == []
    assert bundle.truncated is True


def test_all_chunks_exceed_budget_none_included() -> None:
    """When every chunk exceeds the budget, none are included."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunks = [make_chunk(f"c{i}", "a.py", token_count=1000) for i in range(3)]
    results = [(c, float(i + 1)) for i, c in enumerate(chunks)]
    bundle = assemble_context(results, graph, chunks, "q", token_budget=100)
    assert bundle.chunks == []
    assert bundle.truncated is True


def test_estimate_tokens_fallback_on_zero_count() -> None:
    """_estimate_tokens uses word-based fallback when token_count is 0."""
    from archex.serve.context import estimate_tokens

    chunk = make_chunk("c1", "a.py", content="hello world foo bar", token_count=0)
    estimate = estimate_tokens(chunk)
    # 4 words * 1.3 ≈ 5
    assert estimate > 0
    assert estimate == int(4 * 1.3)


def test_graph_with_isolated_nodes_no_expansion() -> None:
    """Graph with isolated nodes (no edges) produces no structural expansion."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    graph.add_file_node("b.py")
    # No edges between nodes
    chunk_a = make_chunk("ca", "a.py", token_count=10)
    chunk_b = make_chunk("cb", "b.py", token_count=10)
    results = [(chunk_a, 1.0)]
    bundle = assemble_context(results, graph, [chunk_a, chunk_b], "q", token_budget=1000)
    # b.py should NOT be included since there's no edge from a.py to b.py
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "b.py" not in included_files


def test_empty_search_and_vector_results() -> None:
    """Both empty search_results and empty vector_results return empty bundle."""
    graph = DependencyGraph()
    bundle = assemble_context([], graph, [], "q", token_budget=1000, vector_results=[])
    assert bundle.chunks == []
    assert bundle.token_count == 0


# ---------------------------------------------------------------------------
# Cohesion signal tests
# ---------------------------------------------------------------------------


def test_cohesion_signal_boosts_co_module_files() -> None:
    """Two files in the same module → cohesion > 0."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    graph.add_file_node("b.py")
    graph.add_file_edge("a.py", "b.py", kind="imports")
    c_a = make_chunk("ca", "a.py", token_count=10)
    c_b = make_chunk("cb", "b.py", token_count=10)
    mod = Module(
        name="core",
        root_path=".",
        files=["a.py", "b.py"],
        cohesion_score=0.8,
    )
    results = [(c_a, 1.0), (c_b, 0.5)]
    bundle = assemble_context(
        results,
        graph,
        [c_a, c_b],
        "q",
        token_budget=1000,
        modules=[mod],
    )
    for rc in bundle.chunks:
        assert rc.cohesion_score > 0.0


def test_cohesion_signal_zero_without_modules() -> None:
    """No modules param → cohesion_score = 0.0."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunk = make_chunk("c1", "a.py", token_count=10)
    results = [(chunk, 1.0)]
    bundle = assemble_context(results, graph, [chunk], "q", token_budget=1000)
    for rc in bundle.chunks:
        assert rc.cohesion_score == 0.0


def test_cohesion_signal_zero_for_low_cohesion_module() -> None:
    """module.cohesion_score = 0.0 → signal = 0.0."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunk = make_chunk("c1", "a.py", token_count=10)
    mod = Module(name="low", root_path=".", files=["a.py"], cohesion_score=0.0)
    results = [(chunk, 1.0)]
    bundle = assemble_context(
        results,
        graph,
        [chunk],
        "q",
        token_budget=1000,
        modules=[mod],
    )
    for rc in bundle.chunks:
        assert rc.cohesion_score == 0.0


def test_signal_agreement_computed_with_vector() -> None:
    """Both BM25 and vector results → agreement ∈ [0, 1]."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    graph.add_file_node("b.py")
    c_a = make_chunk("ca", "a.py", token_count=10)
    c_b = make_chunk("cb", "b.py", token_count=10)
    bm25_results = [(c_a, 1.0)]
    vec_results = [(c_a, 0.9), (c_b, 0.5)]
    bundle = assemble_context(
        bm25_results,
        graph,
        [c_a, c_b],
        "q",
        token_budget=1000,
        vector_results=vec_results,
    )
    assert bundle.retrieval_metadata.signal_agreement is not None
    assert 0.0 <= bundle.retrieval_metadata.signal_agreement <= 1.0


def test_signal_agreement_none_without_vector() -> None:
    """BM25 only → agreement is None."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunk = make_chunk("c1", "a.py", token_count=10)
    results = [(chunk, 1.0)]
    bundle = assemble_context(results, graph, [chunk], "q", token_budget=1000)
    assert bundle.retrieval_metadata.signal_agreement is None


def test_ranked_chunk_carries_cohesion_score() -> None:
    """RankedChunk has cohesion_score field."""
    from archex.models import RankedChunk

    chunk = make_chunk("c1", "a.py", token_count=10)
    rc = RankedChunk(chunk=chunk, cohesion_score=0.5)
    assert rc.cohesion_score == 0.5


# ---------------------------------------------------------------------------
# BM25 propagation tests
# ---------------------------------------------------------------------------


def test_neighbor_of_high_bm25_seed_gets_relevance() -> None:
    """Neighbor of a high-BM25 seed gets propagated relevance > 0."""
    graph = make_graph_with_edges()
    # auth.py → models.py edge exists
    auth_chunk = make_chunk("c_auth", "auth.py", token_count=10)
    models_chunk = make_chunk("c_models", "models.py", token_count=10)
    # Only auth.py is a BM25 hit
    results = [(auth_chunk, 5.0)]
    bundle = assemble_context(results, graph, [auth_chunk, models_chunk], "q", token_budget=1000)
    # models.py is a neighbor — should get propagated relevance
    models_rc = next((rc for rc in bundle.chunks if rc.chunk.file_path == "models.py"), None)
    assert models_rc is not None
    assert models_rc.relevance_score > 0.0


def test_direct_hit_outranks_neighbor() -> None:
    """Direct BM25 hit has higher relevance than its neighbor."""
    graph = make_graph_with_edges()
    auth_chunk = make_chunk("c_auth", "auth.py", token_count=10)
    models_chunk = make_chunk("c_models", "models.py", token_count=10)
    results = [(auth_chunk, 5.0)]
    bundle = assemble_context(results, graph, [auth_chunk, models_chunk], "q", token_budget=1000)
    auth_rc = next(rc for rc in bundle.chunks if rc.chunk.file_path == "auth.py")
    models_rc = next(rc for rc in bundle.chunks if rc.chunk.file_path == "models.py")
    assert auth_rc.relevance_score > models_rc.relevance_score


def test_isolated_node_unaffected_by_propagation() -> None:
    """Node with no graph edges gets no propagated relevance."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    graph.add_file_node("b.py")
    # No edges
    chunk_a = make_chunk("ca", "a.py", token_count=10)
    chunk_b = make_chunk("cb", "b.py", token_count=10)
    results = [(chunk_a, 5.0)]
    bundle = assemble_context(results, graph, [chunk_a, chunk_b], "q", token_budget=1000)
    # b.py should NOT be included (no edge, no expansion)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "b.py" not in included_files


def test_weak_seed_does_not_expand() -> None:
    """A single weak seed does not trigger expansion into neighbors."""
    graph = DependencyGraph()
    graph.add_file_node("weak.py")
    graph.add_file_node("neighbor.py")
    graph.add_file_edge("weak.py", "neighbor.py", kind="imports")
    weak_chunk = make_chunk("c_weak", "weak.py", token_count=10)
    # Also add a strong seed on a different file so weak.py is proportionally low
    graph.add_file_node("strong.py")
    strong_chunk = make_chunk("c_strong", "strong.py", token_count=10)
    neighbor_chunk = make_chunk("c_neighbor", "neighbor.py", token_count=10)
    # strong.py has score 10.0, weak.py has score 0.5 (5% of max → below SEED_EXPANSION_MIN)
    results = [(strong_chunk, 10.0), (weak_chunk, 0.5)]
    all_chunks = [strong_chunk, weak_chunk, neighbor_chunk]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=1000)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    # neighbor.py should NOT appear — weak.py was below the expansion threshold
    assert "neighbor.py" not in included_files


def test_strong_seed_does_expand() -> None:
    """A strong seed triggers expansion into its imports."""
    graph = DependencyGraph()
    graph.add_file_node("strong.py")
    graph.add_file_node("dep.py")
    graph.add_file_edge("strong.py", "dep.py", kind="imports")
    strong_chunk = make_chunk("c_strong", "strong.py", token_count=10)
    dep_chunk = make_chunk("c_dep", "dep.py", token_count=10)
    results = [(strong_chunk, 5.0)]
    bundle = assemble_context(results, graph, [strong_chunk, dep_chunk], "q", token_budget=1000)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "dep.py" in included_files


def test_file_score_cutoff_excludes_low_files() -> None:
    """Files with aggregate score far below the top file are excluded."""
    graph = DependencyGraph()
    for name in ["top.py", "mid.py", "low.py"]:
        graph.add_file_node(name)
    top_chunk = make_chunk("c_top", "top.py", token_count=10)
    mid_chunk = make_chunk("c_mid", "mid.py", token_count=10)
    low_chunk = make_chunk("c_low", "low.py", token_count=10)
    # top has very high score, low has negligible score
    results = [(top_chunk, 100.0), (mid_chunk, 30.0), (low_chunk, 0.5)]
    all_chunks = [top_chunk, mid_chunk, low_chunk]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=1000)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "top.py" in included_files
    # low.py should be excluded by FILE_SCORE_CUTOFF
    assert "low.py" not in included_files


def test_high_relevance_low_centrality_beats_low_relevance_high_centrality() -> None:
    """With relevance-dominant weights, BM25 signal dominates over structural centrality."""
    graph = DependencyGraph()
    graph.add_file_node("hub.py")
    graph.add_file_node("leaf.py")
    # hub.py is highly central (many edges), leaf.py is not
    for i in range(10):
        node = f"dep{i}.py"
        graph.add_file_node(node)
        graph.add_file_edge(node, "hub.py", kind="imports")

    hub_chunk = make_chunk("c_hub", "hub.py", token_count=10)
    leaf_chunk = make_chunk("c_leaf", "leaf.py", token_count=10)
    # leaf has much higher BM25 score
    results = [(hub_chunk, 1.0), (leaf_chunk, 10.0)]
    all_chunks = [hub_chunk, leaf_chunk]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=1000)
    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    assert scores.get("leaf.py", 0) > scores.get("hub.py", 0)


# ---------------------------------------------------------------------------
# Fusion diagnostics tests
# ---------------------------------------------------------------------------


def test_fusion_weights_recorded_in_metadata_with_vector_results() -> None:
    """When vector_results is provided and fusion fires, weights are set."""
    graph = DependencyGraph()
    for name in ("a.py", "b.py", "c.py", "d.py"):
        graph.add_file_node(name)
    chunks = [make_chunk(f"c{i}", f"{chr(97 + i)}.py", token_count=10) for i in range(4)]
    # Provide enough results with flat BM25 scores (low CV) to trigger fusion
    bm25_results = [(chunks[0], 1.0), (chunks[1], 0.9), (chunks[2], 0.8)]
    vec_results = [(chunks[3], 0.9), (chunks[2], 0.8), (chunks[1], 0.7)]
    bundle = assemble_context(
        bm25_results,
        graph,
        chunks,
        "q",
        token_budget=1000,
        vector_results=vec_results,
    )
    meta = bundle.retrieval_metadata
    assert meta.fusion_bm25_weight is not None
    assert meta.fusion_vector_weight is not None
    assert abs(meta.fusion_bm25_weight + meta.fusion_vector_weight - 1.0) < 1e-9
    assert not meta.fusion_skipped


def test_fusion_skipped_when_bm25_confident() -> None:
    """When BM25 has clear score separation and signals agree, fusion is skipped."""
    graph = DependencyGraph()
    for name in ("a.py", "b.py", "c.py"):
        graph.add_file_node(name)
    chunks = [make_chunk(f"c{i}", f"{chr(97 + i)}.py", token_count=10) for i in range(3)]
    # High CV (very spread scores) + high agreement (same files)
    bm25_results = [(chunks[0], 10.0), (chunks[1], 1.0), (chunks[2], 0.1)]
    vec_results = [(chunks[0], 0.95), (chunks[1], 0.5), (chunks[2], 0.1)]
    bundle = assemble_context(
        bm25_results,
        graph,
        chunks,
        "q",
        token_budget=1000,
        vector_results=vec_results,
    )
    meta = bundle.retrieval_metadata
    assert meta.fusion_skipped
    assert meta.fusion_skip_reason.startswith("bm25_confident:")
    assert meta.fusion_bm25_weight is None


def test_fusion_weights_none_without_vector_results() -> None:
    """BM25-only retrieval leaves fusion weights as None."""
    graph = DependencyGraph()
    graph.add_file_node("a.py")
    chunk = make_chunk("c1", "a.py", token_count=10)
    results = [(chunk, 1.0)]
    bundle = assemble_context(results, graph, [chunk], "q", token_budget=1000)
    assert bundle.retrieval_metadata.fusion_bm25_weight is None
    assert bundle.retrieval_metadata.fusion_vector_weight is None


def test_fusion_weights_reflect_high_agreement() -> None:
    """When BM25 and vector agree completely, bm25_weight=0.85."""
    graph = DependencyGraph()
    for name in ("a.py", "b.py", "c.py"):
        graph.add_file_node(name)
    chunks = [make_chunk(f"c{i}", f"{chr(97 + i)}.py", token_count=10) for i in range(3)]
    # Perfect agreement: both signals return the same files with flat scores (low CV)
    bm25_results = [(chunks[0], 5.0), (chunks[1], 4.8), (chunks[2], 4.5)]
    vec_results = [(chunks[0], 0.9), (chunks[1], 0.85), (chunks[2], 0.8)]
    bundle = assemble_context(
        bm25_results,
        graph,
        chunks,
        "q",
        token_budget=1000,
        vector_results=vec_results,
    )
    meta = bundle.retrieval_metadata
    assert meta.fusion_bm25_weight == 0.85
    assert meta.fusion_vector_weight == 0.15


# ---------------------------------------------------------------------------
# Vector-seed retrieval tests
# ---------------------------------------------------------------------------


def test_vector_only_seed_surfaces_via_assemble_context() -> None:
    """BM25 returns file A, vector returns file B; both appear in the bundle.

    This verifies that vector_results are merged as independent seeds into
    assemble_context and that graph expansion from vector seeds works correctly.
    File A imports B — so B must appear at minimum via graph expansion when A
    is a BM25 seed.  With vector_results supplying B directly, B also acts as
    its own seed regardless of graph structure.
    Provides >= 3 results per signal to satisfy should_fuse min_results.
    """
    graph = DependencyGraph()
    for name in ("a.py", "b.py", "c.py", "d.py", "e.py"):
        graph.add_file_node(name)
    # a.py → b.py (import edge); b.py → c.py
    graph.add_file_edge("a.py", "b.py", kind="imports")
    graph.add_file_edge("b.py", "c.py", kind="imports")

    chunk_a = make_chunk("ca", "a.py", token_count=10)
    chunk_b = make_chunk("cb", "b.py", token_count=10)
    chunk_c = make_chunk("cc", "c.py", token_count=10)
    chunk_d = make_chunk("cd", "d.py", token_count=10)
    chunk_e = make_chunk("ce", "e.py", token_count=10)

    # BM25 knows about a, d, e; vector knows about b, d, e
    # Low agreement (b.py is novel) + flat BM25 scores → fusion fires
    bm25_results = [(chunk_a, 5.0), (chunk_d, 4.9), (chunk_e, 4.8)]
    vec_results = [(chunk_b, 0.8), (chunk_d, 0.7), (chunk_e, 0.6)]
    all_chunks = [chunk_a, chunk_b, chunk_c, chunk_d, chunk_e]

    bundle = assemble_context(
        search_results=bm25_results,
        graph=graph,
        all_chunks=all_chunks,
        question="how does a import b?",
        token_budget=1000,
        vector_results=vec_results,
    )

    included = {rc.chunk.file_path for rc in bundle.chunks}
    assert "a.py" in included, "BM25 seed a.py must be in bundle"
    assert "b.py" in included, "vector seed b.py must be in bundle"


def test_vector_only_recovery_when_bm25_empty() -> None:
    """When BM25 returns nothing, vector results alone seed the bundle.

    This is the critical edge case for external-large repos where BM25 produces
    zero results but the vector index can still locate the relevant file.
    """
    graph = DependencyGraph()
    graph.add_file_node("needle.py")
    graph.add_file_node("other.py")
    # No edge between them — other.py must not appear via expansion

    chunk_needle = make_chunk("c_needle", "needle.py", token_count=10)
    chunk_other = make_chunk("c_other", "other.py", token_count=10)

    vec_results = [(chunk_needle, 0.9)]

    bundle = assemble_context(
        search_results=[],
        graph=graph,
        all_chunks=[chunk_needle, chunk_other],
        question="find the needle",
        token_budget=1000,
        vector_results=vec_results,
    )

    assert len(bundle.chunks) > 0, "bundle must be non-empty when vector results exist"
    included = {rc.chunk.file_path for rc in bundle.chunks}
    assert "needle.py" in included, "vector-only seed needle.py must appear in bundle"
    assert "other.py" not in included, "isolated other.py must not appear"
