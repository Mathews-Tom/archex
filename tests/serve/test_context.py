"""Tests for context assembly and renderers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from archex.index.graph import DependencyGraph
from archex.index.rerank import DEFAULT_TOP_K, RERANK_CANDIDATE_LIMIT, CrossEncoderReranker
from archex.models import CodeChunk, ContextBundle, Module, SymbolKind
from archex.serve.context import assemble_context
from archex.serve.intent import INTENT_TOKEN_BUDGETS, QueryIntent
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
    start_line: int = 1,
    end_line: int = 5,
) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
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


class RecordingReranker(CrossEncoderReranker):
    def __init__(self) -> None:
        super().__init__()
        self.seen_candidate_count: int | None = None
        self.seen_top_k: int | None = None

    def rerank(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float]],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[tuple[CodeChunk, float]]:
        self.seen_candidate_count = len(candidates)
        self.seen_top_k = top_k
        return candidates[:top_k]


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


def test_definition_lookup_uses_smaller_intent_budget() -> None:
    graph = DependencyGraph()
    graph.add_file_node("models.py")
    chunks = [
        make_chunk("c1", "models.py", token_count=1500),
        make_chunk("c2", "models.py", token_count=1500),
    ]
    results = [(chunk, 10.0 - idx) for idx, chunk in enumerate(chunks)]

    bundle = assemble_context(
        results,
        graph,
        chunks,
        "Where is QuerySet defined?",
        token_budget=8192,
    )

    assert bundle.token_budget == INTENT_TOKEN_BUDGETS[QueryIntent.DEFINITION_LOOKUP]
    assert bundle.token_count <= INTENT_TOKEN_BUDGETS[QueryIntent.DEFINITION_LOOKUP]


def test_architecture_query_keeps_large_intent_budget() -> None:
    graph = DependencyGraph()
    graph.add_file_node("pipeline.py")
    chunk = make_chunk("c1", "pipeline.py", token_count=1500)

    bundle = assemble_context(
        [(chunk, 1.0)],
        graph,
        [chunk],
        "Explain the middleware architecture",
        token_budget=8192,
    )

    assert bundle.token_budget == INTENT_TOKEN_BUDGETS[QueryIntent.ARCHITECTURE_BROAD]


def test_underfilled_bundle_continues_below_score_floor() -> None:
    graph = DependencyGraph()
    graph.add_file_node("src/query_pipeline.py")
    high = make_chunk("high", "src/query_pipeline.py", token_count=100)
    low = make_chunk("low", "src/query_pipeline.py", token_count=100)

    bundle = assemble_context(
        [(high, 1.0), (low, 0.01)],
        graph,
        [high, low],
        "How does the query pipeline work?",
        token_budget=1000,
    )

    assert [rc.chunk.id for rc in bundle.chunks] == ["high", "low"]
    assert bundle.token_count == 200


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


def test_packing_covers_selected_files_before_extra_chunks() -> None:
    graph = DependencyGraph()
    graph.add_file_node("primary.py")
    graph.add_file_node("secondary.py")
    primary_a = make_chunk("primary_a", "primary.py", token_count=200)
    primary_b = make_chunk("primary_b", "primary.py", token_count=200)
    secondary = make_chunk("secondary", "secondary.py", token_count=50)

    bundle = assemble_context(
        [(primary_a, 10.0), (primary_b, 9.0), (secondary, 3.0)],
        graph,
        [primary_a, primary_b, secondary],
        "query",
        token_budget=460,
    )

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "secondary.py" in included_files
    assert [rc.chunk.id for rc in bundle.chunks] == ["primary_a", "secondary", "primary_b"]
    assert bundle.token_count <= 460


def test_cli_packing_limits_extra_chunks_per_file() -> None:
    graph = DependencyGraph()
    chunks: list[CodeChunk] = []
    for file_path in ("src/archex/cli/index_cmd.py", "src/archex/project.py"):
        graph.add_file_node(file_path)
        for index in range(4):
            chunks.append(
                make_chunk(
                    f"{file_path}:{index}",
                    file_path,
                    token_count=50,
                    start_line=index * 10 + 1,
                    end_line=index * 10 + 5,
                )
            )

    bundle = assemble_context(
        [(chunk, 10.0 - offset) for offset, chunk in enumerate(chunks)],
        graph,
        chunks,
        "How does archex explicitly build or refresh a repo-local index?",
        token_budget=1000,
    )

    counts: dict[str, int] = {}
    for ranked in bundle.chunks:
        counts[ranked.chunk.file_path] = counts.get(ranked.chunk.file_path, 0) + 1

    assert counts == {
        "src/archex/cli/index_cmd.py": 2,
        "src/archex/project.py": 2,
    }


def test_packing_delays_test_files_until_production_files_are_covered() -> None:
    graph = DependencyGraph()
    graph.add_file_node("tests/test_patterns.py")
    graph.add_file_node("src/archex/analyze/patterns.py")
    graph.add_file_node("src/archex/models.py")
    test_chunk = make_chunk("test_chunk", "tests/test_patterns.py", token_count=50)
    patterns = make_chunk("patterns", "src/archex/analyze/patterns.py", token_count=50)
    models = make_chunk("models", "src/archex/models.py", token_count=50)

    bundle = assemble_context(
        [(test_chunk, 40.0), (patterns, 10.0), (models, 8.0)],
        graph,
        [test_chunk, patterns, models],
        "How does archex detect architectural patterns?",
        token_budget=1000,
    )

    assert [rc.chunk.id for rc in bundle.chunks] == ["patterns", "models", "test_chunk"]


def test_cli_config_alignment_keeps_runtime_config_files_ahead_of_support_noise() -> None:
    graph = DependencyGraph()
    chunks = [
        make_chunk("project", "src/archex/project.py", symbol_name="project_dir", token_count=50),
        make_chunk("config", "src/archex/config.py", symbol_name="load_config", token_count=50),
        make_chunk("cache", "src/archex/cache.py", symbol_name="CacheManager", token_count=50),
        make_chunk(
            "context",
            "src/archex/serve/context.py",
            symbol_name="_query_terms",
            token_count=50,
        ),
        make_chunk(
            "compare_config",
            "src/archex/serve/compare/configuration.py",
            symbol_name="ConfigurationEvidence",
            token_count=50,
        ),
        make_chunk(
            "test_project",
            "tests/test_project.py",
            symbol_name="test_project_state",
            token_count=50,
        ),
    ]
    for chunk in chunks:
        graph.add_file_node(chunk.file_path)

    bundle = assemble_context(
        [
            (chunks[3], 12.0),
            (chunks[4], 11.0),
            (chunks[5], 10.0),
            (chunks[0], 9.0),
            (chunks[1], 8.0),
            (chunks[2], 7.0),
        ],
        graph,
        chunks,
        "How does archex resolve project settings into runtime configuration?",
        token_budget=220,
    )

    included_files = [rc.chunk.file_path for rc in bundle.chunks]
    assert included_files[:3] == [
        "src/archex/project.py",
        "src/archex/config.py",
        "src/archex/cache.py",
    ]


def test_mcp_query_prefers_mcp_command_over_generic_query_command() -> None:
    graph = DependencyGraph()
    chunks = [
        make_chunk("models", "src/archex/models.py", symbol_name="ContextBundle", token_count=60),
        make_chunk(
            "context",
            "src/archex/serve/context.py",
            symbol_name="assemble_context",
            token_count=60,
        ),
        make_chunk("api", "src/archex/api.py", symbol_name="query", token_count=60),
        make_chunk(
            "mcp",
            "src/archex/integrations/mcp.py",
            symbol_name="handle_query_repo",
            token_count=60,
        ),
        make_chunk("mcp_cmd", "src/archex/cli/mcp_cmd.py", symbol_name="mcp_cmd", token_count=60),
        make_chunk(
            "query_cmd",
            "src/archex/cli/query_cmd.py",
            symbol_name="query_cmd",
            token_count=60,
        ),
    ]
    for chunk in chunks:
        graph.add_file_node(chunk.file_path)

    bundle = assemble_context(
        [
            (chunks[0], 10.0),
            (chunks[1], 9.5),
            (chunks[2], 9.0),
            (chunks[3], 8.5),
            (chunks[5], 8.0),
            (chunks[4], 7.5),
        ],
        graph,
        chunks,
        "How does archex expose repository query workflows through MCP?",
        token_budget=300,
    )

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "src/archex/cli/mcp_cmd.py" in included_files
    assert "src/archex/cli/query_cmd.py" not in included_files


def test_cli_index_query_selects_api_as_fifth_product_file() -> None:
    graph = DependencyGraph()
    chunks = [
        make_chunk("index_cmd", "src/archex/cli/index_cmd.py", token_count=80),
        make_chunk("project", "src/archex/project.py", token_count=80),
        make_chunk("config", "src/archex/config.py", token_count=80),
        make_chunk("cache", "src/archex/cache.py", token_count=80),
        make_chunk("api", "src/archex/api.py", token_count=80),
        make_chunk("store", "src/archex/index/store.py", token_count=80),
    ]
    for chunk in chunks:
        graph.add_file_node(chunk.file_path)

    bundle = assemble_context(
        [
            (chunks[0], 10.0),
            (chunks[1], 9.0),
            (chunks[2], 8.0),
            (chunks[3], 7.0),
            (chunks[4], 6.0),
            (chunks[5], 5.0),
        ],
        graph,
        chunks,
        "How does archex explicitly build or refresh a repo-local index?",
        token_budget=500,
    )

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert {
        "src/archex/cli/index_cmd.py",
        "src/archex/project.py",
        "src/archex/config.py",
        "src/archex/cache.py",
        "src/archex/api.py",
    } <= included_files


def test_external_lifecycle_terms_preserve_expected_product_files() -> None:
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    celery_terms = _query_terms("How does Celery dispatch and execute distributed tasks?")
    assert {"amqp", "strategy", "task", "worker"} <= celery_terms
    assert _path_alignment_boost("celery/app/amqp.py", celery_terms) == 3.0
    assert _path_alignment_boost("celery/worker/strategy.py", celery_terms) == 3.0

    requests_terms = _query_terms("How does requests manage HTTP sessions and connection pooling?")
    assert {"adapter", "adapters", "model", "models", "session"} <= requests_terms
    assert _path_alignment_boost("src/requests/models.py", requests_terms) == 3.0

    orm_terms = _query_terms("How does Django's ORM build and execute SQL queries?")
    assert {"query", "queries", "model", "models"} <= orm_terms
    assert _path_alignment_boost("django/db/models/query.py", orm_terms) == 3.0


def test_configuration_query_terms_do_not_promote_generic_models_path() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    terms = _query_terms(
        "How does a query use project configuration and cache state across the indexing lifecycle?"
    )

    assert "models" not in terms


def test_packing_skips_nested_line_ranges() -> None:
    graph = DependencyGraph()
    graph.add_file_node("module.py")
    parent = make_chunk(
        "parent",
        "module.py",
        token_count=300,
        start_line=1,
        end_line=100,
    )
    child = make_chunk(
        "child",
        "module.py",
        token_count=100,
        start_line=20,
        end_line=30,
    )
    sibling = make_chunk(
        "sibling",
        "module.py",
        token_count=50,
        start_line=120,
        end_line=130,
    )

    bundle = assemble_context(
        [(parent, 10.0), (child, 9.0), (sibling, 8.0)],
        graph,
        [parent, child, sibling],
        "query",
        token_budget=1000,
    )

    included_ids = {rc.chunk.id for rc in bundle.chunks}
    assert "parent" in included_ids
    assert "child" not in included_ids
    assert "sibling" in included_ids


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


def test_preserved_files_survive_file_cutoff() -> None:
    graph = DependencyGraph()
    primary = make_chunk("primary", "src/archex/project.py", token_count=10)
    preserved = make_chunk("preserved", "src/archex/cli/main.py", token_count=10)
    other = make_chunk("other", "src/archex/cache.py", token_count=10)
    bundle = assemble_context(
        [(primary, 10.0), (preserved, 0.1), (other, 0.09)],
        graph,
        [primary, preserved, other],
        "How does the project init command work?",
        token_budget=1000,
        preserved_files={"src/archex/cli/main.py"},
        apply_intent_budget=False,
    )
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "src/archex/project.py" in included_files
    assert "src/archex/cli/main.py" in included_files


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


def test_expansion_metadata_records_seed_and_expanded_file_paths() -> None:
    graph = DependencyGraph()
    graph.add_file_node("seed.py")
    graph.add_file_node("expanded.py")
    graph.add_file_edge("seed.py", "expanded.py", kind="imports")
    seed_chunk = make_chunk("seed", "seed.py", token_count=10)
    expanded_chunk = make_chunk("expanded", "expanded.py", token_count=10)

    bundle = assemble_context(
        [(seed_chunk, 5.0)],
        graph,
        [seed_chunk, expanded_chunk],
        "graph expansion",
        token_budget=1000,
    )

    meta = bundle.retrieval_metadata
    assert meta.seed_files_found == 1
    assert meta.seed_file_paths == ["seed.py"]
    assert meta.expanded_file_paths == ["expanded.py"]
    assert meta.expansion_files_added == 1
    assert meta.expansion_eligible_seeds == 1
    assert meta.expansion_candidates_found == 1
    assert meta.expansion_import_neighbor_edges == 1
    assert meta.expansion_zero_candidate_reason == ""


def test_expansion_metadata_records_file_reasons_without_changing_output() -> None:
    graph = DependencyGraph()
    for file_path in (
        "src/app/main.py",
        "src/app/dep.py",
        "src/lib/helper.py",
        "tests/test_dep.py",
    ):
        graph.add_file_node(file_path)
    graph.add_file_edge("src/app/main.py", "src/app/dep.py", kind="imports")
    graph.add_file_edge("src/app/main.py", "src/lib/helper.py", kind="imports")
    graph.add_file_edge("src/app/main.py", "tests/test_dep.py", kind="imports")
    seed_chunk = make_chunk("seed", "src/app/main.py", token_count=10)
    same_module_chunk = make_chunk("dep", "src/app/dep.py", token_count=10)
    cross_module_chunk = make_chunk("helper", "src/lib/helper.py", token_count=10)
    test_chunk = make_chunk("test", "tests/test_dep.py", token_count=10)

    bundle = assemble_context(
        [(seed_chunk, 5.0)],
        graph,
        [seed_chunk, same_module_chunk, cross_module_chunk, test_chunk],
        "graph expansion",
        token_budget=1000,
        modules=[
            Module(name="app", root_path="src/app", files=["src/app/main.py", "src/app/dep.py"]),
            Module(name="lib", root_path="src/lib", files=["src/lib/helper.py"]),
        ],
    )

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert included_files == {"src/app/main.py", "src/app/dep.py", "src/lib/helper.py"}
    meta = bundle.retrieval_metadata
    assert meta.expanded_file_reasons["src/app/dep.py"] == ["import_target", "same_module"]
    assert meta.expanded_file_reasons["src/lib/helper.py"] == ["cross_module", "import_target"]
    assert meta.expansion_reason_counts["import_target"] == 3
    assert meta.expansion_reason_counts["same_module"] == 1
    assert meta.expansion_reason_counts["cross_module"] == 1
    assert meta.expansion_reason_counts["test_file"] == 1


def test_low_signal_hub_expansion_is_suppressed_without_dropping_aligned_file() -> None:
    graph = DependencyGraph()
    for file_path in (
        "lib/application.js",
        "lib/router/layer.js",
        "lib/utils.js",
        "lib/helper-a.js",
        "lib/helper-b.js",
        "lib/helper-c.js",
    ):
        graph.add_file_node(file_path)
    graph.add_file_edge("lib/application.js", "lib/router/layer.js", kind="imports")
    graph.add_file_edge("lib/application.js", "lib/utils.js", kind="imports")
    graph.add_file_edge("lib/utils.js", "lib/helper-a.js", kind="imports")
    graph.add_file_edge("lib/utils.js", "lib/helper-b.js", kind="imports")
    graph.add_file_edge("lib/helper-c.js", "lib/utils.js", kind="imports")

    seed_chunk = make_chunk("seed", "lib/application.js", token_count=10)
    layer_chunk = make_chunk("layer", "lib/router/layer.js", token_count=10)
    utils_chunk = make_chunk("utils", "lib/utils.js", token_count=10)

    bundle = assemble_context(
        [(seed_chunk, 5.0)],
        graph,
        [seed_chunk, layer_chunk, utils_chunk],
        "How does express implement the middleware chain and router layer?",
        token_budget=1000,
    )

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "lib/router/layer.js" in included_files
    assert "lib/utils.js" not in included_files
    assert bundle.retrieval_metadata.expansion_reason_counts["skipped"] == 1


def test_expansion_metadata_records_zero_candidate_reason() -> None:
    graph = DependencyGraph()
    graph.add_file_node("seed.py")
    seed_chunk = make_chunk("seed", "seed.py", token_count=10)

    bundle = assemble_context(
        [(seed_chunk, 5.0)],
        graph,
        [seed_chunk],
        "graph expansion",
        token_budget=1000,
    )

    meta = bundle.retrieval_metadata
    assert meta.seed_files_found == 1
    assert meta.expansion_eligible_seeds == 1
    assert meta.expansion_candidates_found == 0
    assert meta.expansion_import_neighbor_edges == 0
    assert meta.expansion_zero_candidate_reason == "no_import_neighbors"


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
    """When BM25 and vector agree completely, adaptive RSF uses bm25_weight=0.60."""
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
    # Adaptive RSF weights are more balanced than old RRF (was 0.85/0.15)
    assert meta.fusion_bm25_weight == 0.60
    assert meta.fusion_vector_weight == 0.40


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
    The query avoids dependency keywords so this test isolates fusion seed union
    from dependency-query file capping.
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
        question="how does a use b?",
        token_budget=1000,
        vector_results=vec_results,
    )

    included = {rc.chunk.file_path for rc in bundle.chunks}
    assert "a.py" in included, "BM25 seed a.py must be in bundle"
    assert "b.py" in included, "vector seed b.py must be in bundle"


def test_assemble_context_honors_reranker_default_top_k() -> None:
    graph = DependencyGraph()
    chunks = [make_chunk(f"c{i}", f"f{i}.py", token_count=10) for i in range(DEFAULT_TOP_K + 5)]
    for chunk in chunks:
        graph.add_file_node(chunk.file_path)
    search_results = [(chunk, float(DEFAULT_TOP_K + 5 - i)) for i, chunk in enumerate(chunks)]
    reranker = RecordingReranker()

    assemble_context(
        search_results=search_results,
        graph=graph,
        all_chunks=chunks,
        question="query",
        token_budget=1000,
        reranker=reranker,
    )
    assert reranker.seen_candidate_count == RERANK_CANDIDATE_LIMIT
    assert reranker.seen_top_k == DEFAULT_TOP_K


def test_assemble_context_honors_custom_rerank_candidate_limit() -> None:
    graph = DependencyGraph()
    chunks = [make_chunk(f"c{i}", f"f{i}.py", token_count=10) for i in range(DEFAULT_TOP_K + 5)]
    for chunk in chunks:
        graph.add_file_node(chunk.file_path)
    search_results = [(chunk, float(DEFAULT_TOP_K + 5 - i)) for i, chunk in enumerate(chunks)]
    reranker = RecordingReranker()

    assemble_context(
        search_results=search_results,
        graph=graph,
        all_chunks=chunks,
        question="query",
        token_budget=1000,
        reranker=reranker,
        rerank_candidate_limit=3,
    )

    assert reranker.seen_candidate_count == 3
    assert reranker.seen_top_k == DEFAULT_TOP_K


def test_assemble_context_reranker_preserves_candidates_outside_rerank_window() -> None:
    graph = DependencyGraph()
    distractors = [
        make_chunk(f"d{i}", f"distractor_{i}.py", token_count=10)
        for i in range(RERANK_CANDIDATE_LIMIT)
    ]
    needle = make_chunk("needle", "needle.py", symbol_kind=SymbolKind.CLASS, token_count=10)
    chunks = [*distractors, needle]
    for chunk in chunks:
        graph.add_file_node(chunk.file_path)

    search_results = [(chunk, 1.0) for chunk in distractors]
    search_results.append((needle, 100.0))
    reranker = RecordingReranker()

    bundle = assemble_context(
        search_results=search_results,
        graph=graph,
        all_chunks=chunks,
        question="needle",
        token_budget=1000,
        reranker=reranker,
    )

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert reranker.seen_candidate_count == RERANK_CANDIDATE_LIMIT
    assert bundle.retrieval_metadata.candidates_after_expansion == len(chunks)
    assert "needle.py" in included_files


def test_assemble_context_reranker_does_not_demote_existing_scores() -> None:
    class DemotingReranker(CrossEncoderReranker):
        def __init__(self) -> None:
            super().__init__()

        def rerank(
            self,
            query: str,
            candidates: list[tuple[CodeChunk, float]],
            top_k: int = DEFAULT_TOP_K,
        ) -> list[tuple[CodeChunk, float]]:
            del query, top_k
            return [(chunk, 0.0) for chunk, _ in candidates]

    graph = DependencyGraph()
    important = make_chunk("important", "important.py", token_count=10)
    other = make_chunk("other", "other.py", token_count=10)
    for chunk in (important, other):
        graph.add_file_node(chunk.file_path)

    bundle = assemble_context(
        search_results=[(important, 100.0), (other, 1.0)],
        graph=graph,
        all_chunks=[important, other],
        question="important",
        token_budget=1000,
        reranker=DemotingReranker(),
    )

    important_rank = next(rc for rc in bundle.chunks if rc.chunk.id == "important")
    assert important_rank.relevance_score == 1.0


def test_normalized_rerank_scores_shift_negative_logits() -> None:
    from archex.serve.context import _normalized_rerank_scores  # pyright: ignore[reportPrivateUsage]

    low = make_chunk("low", "low.py")
    high = make_chunk("high", "high.py")

    scores = _normalized_rerank_scores([(low, -3.0), (high, -1.0)])

    assert scores == {"low": 0.0, "high": 1.0}


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


# ---------------------------------------------------------------------------
# Entry-point boost tests
# ---------------------------------------------------------------------------


def test_entry_point_boost_ranks_mod_rs_higher() -> None:
    """mod.rs should score higher than a leaf module with equal BM25 score."""
    graph = DependencyGraph()
    graph.add_file_node("src/runtime/mod.rs")
    graph.add_file_node("src/runtime/scheduler/current_thread.rs")
    mod_chunk = make_chunk("c_mod", "src/runtime/mod.rs", token_count=10)
    leaf_chunk = make_chunk("c_leaf", "src/runtime/scheduler/current_thread.rs", token_count=10)
    results = [(mod_chunk, 1.0), (leaf_chunk, 1.0)]
    bundle = assemble_context(
        results,
        graph,
        [mod_chunk, leaf_chunk],
        "how does the execution model work?",
        token_budget=1000,
    )
    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    assert scores["src/runtime/mod.rs"] > scores["src/runtime/scheduler/current_thread.rs"]


def test_entry_point_boost_init_py() -> None:
    """__init__.py gets entry-point boost over regular .py files."""
    graph = DependencyGraph()
    graph.add_file_node("pkg/__init__.py")
    graph.add_file_node("pkg/utils.py")
    init_chunk = make_chunk("c_init", "pkg/__init__.py", token_count=10)
    utils_chunk = make_chunk("c_utils", "pkg/utils.py", token_count=10)
    results = [(init_chunk, 1.0), (utils_chunk, 1.0)]
    bundle = assemble_context(results, graph, [init_chunk, utils_chunk], "q", token_budget=1000)
    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    assert scores["pkg/__init__.py"] > scores["pkg/utils.py"]


def test_entry_point_boost_index_js() -> None:
    """index.js gets entry-point boost."""
    graph = DependencyGraph()
    graph.add_file_node("lib/router/index.js")
    graph.add_file_node("lib/router/layer.js")
    idx_chunk = make_chunk("c_idx", "lib/router/index.js", token_count=10)
    layer_chunk = make_chunk("c_layer", "lib/router/layer.js", token_count=10)
    results = [(idx_chunk, 1.0), (layer_chunk, 1.0)]
    bundle = assemble_context(results, graph, [idx_chunk, layer_chunk], "q", token_budget=1000)
    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    assert scores["lib/router/index.js"] > scores["lib/router/layer.js"]


# ---------------------------------------------------------------------------
# Directory-path alignment tests
# ---------------------------------------------------------------------------


def test_directory_alignment_boosts_matching_path() -> None:
    """File under lib/router/ scores higher when query mentions 'router'."""
    graph = DependencyGraph()
    graph.add_file_node("lib/router/route.js")
    graph.add_file_node("lib/application.js")
    route_chunk = make_chunk("c_route", "lib/router/route.js", token_count=10)
    app_chunk = make_chunk("c_app", "lib/application.js", token_count=10)
    results = [(route_chunk, 1.0), (app_chunk, 1.0)]
    bundle = assemble_context(
        results,
        graph,
        [route_chunk, app_chunk],
        "how does the router handle requests?",
        token_budget=1000,
    )
    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    assert scores["lib/router/route.js"] > scores["lib/application.js"]


def test_path_alignment_no_boost_without_match() -> None:
    """No path boost when query terms don't match the file path."""
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]

    assert _path_alignment_boost("lib/utils/helpers.py", {"router", "middleware"}) == 1.0


def test_path_alignment_matches_query_term_in_directory() -> None:
    """Directory matching returns a path-alignment boost."""
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]

    assert _path_alignment_boost("lib/router/index.js", {"router"}) == 1.35


def test_path_alignment_matches_query_term_in_filename() -> None:
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]

    assert _path_alignment_boost("src/archex/cli/index_cmd.py", {"index"}) > _path_alignment_boost(
        "src/archex/index/store.py", {"index"}
    )


def test_query_terms_expand_query_pipeline_to_bm25_context_signals() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    terms = _query_terms("How does archex implement the query pipeline?")
    assert {"api", "bm25", "context", "rank", "score"} <= terms


def test_query_pipeline_terms_boost_api_path() -> None:
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    terms = _query_terms("How does archex implement the query pipeline?")
    assert _path_alignment_boost("src/archex/api.py", terms) == 3.0


def test_query_terms_expand_index_to_cache_project_signals() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    terms = _query_terms("How does archex explicitly build or refresh a repo-local index?")
    assert {"cache", "config", "project", "store"} <= terms


def test_query_terms_expand_self_lifecycle_concepts() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    init_terms = _query_terms("How does archex initialize repo-local project state?")
    assert {"init", "cli", "main", "project", "config"} <= init_terms

    status_terms = _query_terms(
        "How does archex inspect whether a repo-local index is fresh, stale, dirty, or corrupt?"
    )
    assert {"status", "fresh", "stale", "dirty", "corrupt", "delta"} <= status_terms

    config_terms = _query_terms(
        "How does archex resolve project settings into runtime configuration?"
    )
    assert {"config", "settings", "runtime", "cache"} <= config_terms
    assert "models" not in config_terms


def test_self_lifecycle_terms_boost_command_paths() -> None:
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    init_terms = _query_terms("How does archex initialize repo-local project state?")
    assert _path_alignment_boost("src/archex/cli/init_cmd.py", init_terms) > _path_alignment_boost(
        "src/archex/status.py", init_terms
    )

    status_terms = _query_terms(
        "How does archex inspect whether a repo-local index is fresh, stale, dirty, or corrupt?"
    )
    assert _path_alignment_boost(
        "src/archex/cli/status_cmd.py", status_terms
    ) > _path_alignment_boost("src/archex/cli/cache_cmd.py", status_terms)


def test_query_terms_drop_repo_name_noise() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    assert "archex" not in _query_terms("How does archex implement the query pipeline?")


def test_query_terms_do_not_expand_generic_query_to_bm25() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    terms = _query_terms("How does archex expose repository query workflows through MCP?")
    assert "query" in terms
    assert "bm25" not in terms
    assert "rank" not in terms


def test_query_terms_expand_mcp_to_product_query_contract_files() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    terms = _query_terms("How does archex expose repository query workflows through MCP?")
    assert {"api", "context", "mcp_cmd", "models"} <= terms


def test_path_alignment_matches_private_module_stem_without_underscore() -> None:
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]

    assert _path_alignment_boost("pydantic/_internal/_validators.py", {"validators"}) == 3.0


def test_query_terms_expand_framework_semantics_without_path_hacks() -> None:
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    middleware_terms = _query_terms("How does express implement the middleware chain?")
    assert {"router", "route", "layer", "stack", "handler"} <= middleware_terms

    dependency_terms = _query_terms("How does FastAPI implement dependency injection?")
    assert {"depends", "dependant", "provider", "resolver"} <= dependency_terms

    validator_terms = _query_terms("How does Pydantic chain and apply field validators?")
    assert {"validate", "validation", "field_validator", "functional_validators"} <= validator_terms

    decorator_terms = _query_terms("How does click implement command decorators?")
    assert {"decorator", "parameter", "option", "argument", "wrapper"} <= decorator_terms

    orm_terms = _query_terms("How does Django's ORM build and execute SQL queries?")
    assert {"queryset", "compiler", "expression", "where"} <= orm_terms


def test_framework_semantic_terms_boost_router_and_layer_paths() -> None:
    from archex.serve.context import _path_alignment_boost  # pyright: ignore[reportPrivateUsage]
    from archex.serve.context import _query_terms  # pyright: ignore[reportPrivateUsage]

    terms = _query_terms("How does express implement the middleware chain and next function?")
    assert _path_alignment_boost("lib/router/layer.js", terms) > 1.0
    assert _path_alignment_boost("lib/router/route.js", terms) > 1.0
    assert _path_alignment_boost("lib/view.js", terms) == 1.0


def test_type_alignment_requires_query_overlap_for_general_queries() -> None:
    from archex.serve.context import _type_alignment_score  # pyright: ignore[reportPrivateUsage]

    chunk = make_chunk(
        "c_type",
        "src/archex/models.py",
        symbol_kind=SymbolKind.CLASS,
        symbol_name="Visibility",
    )
    assert _type_alignment_score(chunk, {"query", "pipeline"}, definition_lookup=False) == 0.0


def test_type_alignment_keeps_definition_lookup_bonus() -> None:
    from archex.serve.context import _type_alignment_score  # pyright: ignore[reportPrivateUsage]

    chunk = make_chunk(
        "c_type",
        "src/archex/models.py",
        symbol_kind=SymbolKind.CLASS,
        symbol_name="Visibility",
    )
    assert _type_alignment_score(chunk, {"visibility"}, definition_lookup=True) == 0.5


# ---------------------------------------------------------------------------
# Support-file penalty tests
# ---------------------------------------------------------------------------


def test_support_file_penalty_is_0_15() -> None:
    """Support files receive a 0.15 penalty unless the query asks for support artifacts."""
    graph = DependencyGraph()
    graph.add_file_node("src/auth.py")
    graph.add_file_node("tests/test_auth.py")
    src_chunk = make_chunk("c_src", "src/auth.py", token_count=10)
    test_chunk = make_chunk("c_test", "tests/test_auth.py", token_count=10)
    results = [(src_chunk, 1.0), (test_chunk, 1.0)]
    bundle = assemble_context(results, graph, [src_chunk, test_chunk], "q", token_budget=1000)
    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    ratio = scores["tests/test_auth.py"] / scores["src/auth.py"]
    assert 0.10 < ratio < 0.20, f"Expected ~0.15 ratio, got {ratio}"


# ---------------------------------------------------------------------------
# _is_entry_point unit tests
# ---------------------------------------------------------------------------


def test_is_entry_point_recognizes_known_names() -> None:
    from archex.serve.context import _is_entry_point  # pyright: ignore[reportPrivateUsage]

    assert _is_entry_point("src/runtime/mod.rs")
    assert _is_entry_point("pkg/__init__.py")
    assert _is_entry_point("lib/router/index.js")
    assert _is_entry_point("components/index.tsx")
    assert _is_entry_point("lib.rs")
    assert not _is_entry_point("src/utils.py")
    assert not _is_entry_point("tree.go")
    assert not _is_entry_point("mod_helper.rs")


# ---------------------------------------------------------------------------
# Convergent activation gate tests
# ---------------------------------------------------------------------------


def test_expansion_convergent_activation_bonus() -> None:
    """A file reached by two independent seeds ranks higher than one reached by one seed.

    file_a and file_b (both seeds) both import file_c — convergent evidence.
    file_d (seed) imports file_e — single-seed path.
    file_c must rank above file_e in expansion priority.
    """
    graph = DependencyGraph()
    for name in ("file_a.py", "file_b.py", "file_c.py", "file_d.py", "file_e.py"):
        graph.add_file_node(name)
    # Two seeds both import file_c — convergent evidence
    graph.add_file_edge("file_a.py", "file_c.py", kind="imports")
    graph.add_file_edge("file_b.py", "file_c.py", kind="imports")
    # Single seed imports file_e — single path
    graph.add_file_edge("file_d.py", "file_e.py", kind="imports")

    chunk_a = make_chunk("ca", "file_a.py", token_count=10)
    chunk_b = make_chunk("cb", "file_b.py", token_count=10)
    chunk_c = make_chunk("cc", "file_c.py", token_count=10)
    chunk_d = make_chunk("cd", "file_d.py", token_count=10)
    chunk_e = make_chunk("ce", "file_e.py", token_count=10)

    # All seeds have equal BM25 scores — only convergence distinguishes file_c vs file_e
    results = [(chunk_a, 5.0), (chunk_b, 5.0), (chunk_d, 5.0)]
    all_chunks = [chunk_a, chunk_b, chunk_c, chunk_d, chunk_e]

    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=1000)

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    # Both expansion targets must appear in the bundle
    assert "file_c.py" in included_files, "file_c.py (convergent) must be included"
    assert "file_e.py" in included_files, "file_e.py (single-seed) must be included"

    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    assert scores["file_c.py"] > scores["file_e.py"], (
        f"Convergent file_c.py ({scores['file_c.py']:.4f}) must outrank "
        f"single-seed file_e.py ({scores['file_e.py']:.4f})"
    )


def test_expansion_additive_accumulation() -> None:
    """Three seeds importing the same dependency accumulate higher priority than one seed.

    dep_shared is imported by all three seeds; dep_single is imported by only one.
    dep_shared must have higher expansion priority and thus rank higher after scoring.
    """
    graph = DependencyGraph()
    for name in ("seed_x.py", "seed_y.py", "seed_z.py", "dep_shared.py", "dep_single.py"):
        graph.add_file_node(name)
    # All three seeds import the shared dep
    graph.add_file_edge("seed_x.py", "dep_shared.py", kind="imports")
    graph.add_file_edge("seed_y.py", "dep_shared.py", kind="imports")
    graph.add_file_edge("seed_z.py", "dep_shared.py", kind="imports")
    # Only one seed imports the single dep
    graph.add_file_edge("seed_x.py", "dep_single.py", kind="imports")

    chunk_x = make_chunk("cx", "seed_x.py", token_count=10)
    chunk_y = make_chunk("cy", "seed_y.py", token_count=10)
    chunk_z = make_chunk("cz", "seed_z.py", token_count=10)
    chunk_shared = make_chunk("c_shared", "dep_shared.py", token_count=10)
    chunk_single = make_chunk("c_single", "dep_single.py", token_count=10)

    results = [(chunk_x, 5.0), (chunk_y, 5.0), (chunk_z, 5.0)]
    all_chunks = [chunk_x, chunk_y, chunk_z, chunk_shared, chunk_single]

    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=1000)

    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "dep_shared.py" in included_files, "dep_shared.py must be included"
    assert "dep_single.py" in included_files, "dep_single.py must be included"

    scores = {rc.chunk.file_path: rc.final_score for rc in bundle.chunks}
    assert scores["dep_shared.py"] > scores["dep_single.py"], (
        f"3-seed dep_shared.py ({scores['dep_shared.py']:.4f}) must outrank "
        f"1-seed dep_single.py ({scores['dep_single.py']:.4f})"
    )
