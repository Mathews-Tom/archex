"""Tests for retrieval recall: sum-based aggregation, adaptive limits, decay tuning."""

from __future__ import annotations

from archex.index.graph import DependencyGraph
from archex.models import CodeChunk, SymbolKind
from archex.serve.context import assemble_context


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


# ---------------------------------------------------------------------------
# Sum-based file aggregation
# ---------------------------------------------------------------------------


def test_sum_aggregation_rewards_multi_chunk_files() -> None:
    """A file matched by many chunks aggregates above a single-chunk file.

    Sum-based aggregation accumulates per-file chunk scores, so a central file
    with several relevant chunks ranks above a file with one equally scored
    chunk. Max-based aggregation collapsed each file to its best chunk and
    regressed recall on framework-heavy repos.
    """
    graph = DependencyGraph()
    graph.add_file_node("central.py")
    graph.add_file_node("peripheral.py")
    central = [make_chunk(f"cc{i}", "central.py", token_count=10) for i in range(6)]
    peripheral = [make_chunk("cp0", "peripheral.py", token_count=10)]
    all_chunks = central + peripheral
    results = [(c, 4.0) for c in central] + [(peripheral[0], 4.0)]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=5000)
    files = [rc.chunk.file_path for rc in bundle.chunks]
    assert "central.py" in files
    assert files.index("central.py") < files.index("peripheral.py"), (
        "multi-chunk file must rank above single-chunk file under sum aggregation"
    )


def test_single_chunk_file_survives_cutoff_when_above_threshold() -> None:
    """A strong single-chunk file still clears FILE_SCORE_CUTOFF under sum.

    Sum aggregation rewards chunk count but does not eliminate a single-chunk
    file whose score sits above the relative cutoff.
    """
    graph = DependencyGraph()
    graph.add_file_node("many.py")
    graph.add_file_node("solo.py")
    many = [make_chunk(f"cm{i}", "many.py", token_count=10) for i in range(3)]
    solo = [make_chunk("cs0", "solo.py", token_count=10)]
    all_chunks = many + solo
    results = [(c, 5.0) for c in many] + [(solo[0], 5.0)]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=5000)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "solo.py" in included_files


# ---------------------------------------------------------------------------
# Adaptive max files
# ---------------------------------------------------------------------------


def test_adaptive_max_files_minimum_is_five() -> None:
    """_adaptive_max_files never returns fewer than 5 for non-trivial inputs."""
    from archex.serve.context import _adaptive_max_files  # pyright: ignore[reportPrivateUsage]

    file_scores = [
        ("top.py", 10.0),
        ("a.py", 0.5),
        ("b.py", 0.3),
        ("c.py", 0.2),
        ("d.py", 0.1),
        ("e.py", 0.05),
    ]
    result = _adaptive_max_files(file_scores)
    assert result >= 5, f"Expected >= 5, got {result}"


def test_adaptive_max_files_returns_8_for_flat_scores() -> None:
    """Flat score distribution returns the full default of 8."""
    from archex.serve.context import _adaptive_max_files  # pyright: ignore[reportPrivateUsage]

    file_scores = [(f"f{i}.py", 5.0 - i * 0.1) for i in range(10)]
    result = _adaptive_max_files(file_scores)
    assert result == 8


def test_adaptive_max_files_returns_6_for_moderate_separation() -> None:
    """Moderate score separation (2-3x ratio) returns 6."""
    from archex.serve.context import _adaptive_max_files  # pyright: ignore[reportPrivateUsage]

    file_scores = [
        ("top.py", 5.0),
        ("a.py", 3.0),
        ("b.py", 2.0),  # median
        ("c.py", 1.5),
        ("d.py", 1.0),
    ]
    result = _adaptive_max_files(file_scores)
    # top/median = 5.0/2.0 = 2.5 → > 2.0 → 6
    assert result == 6


# ---------------------------------------------------------------------------
# Importer decay
# ---------------------------------------------------------------------------


def test_importer_file_included_with_higher_decay() -> None:
    """Importer (consumer) files with IMPORTER_DECAY=0.35 survive cutoff."""
    graph = DependencyGraph()
    graph.add_file_node("core.py")
    graph.add_file_node("consumer.py")
    graph.add_file_edge("consumer.py", "core.py", kind="imports")
    core_chunk = make_chunk("c_core", "core.py", token_count=10)
    consumer_chunk = make_chunk("c_consumer", "consumer.py", token_count=10)
    results = [(core_chunk, 5.0)]
    bundle = assemble_context(results, graph, [core_chunk, consumer_chunk], "q", token_budget=1000)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "consumer.py" in included_files, (
        "consumer.py (importer with 0.35 decay) must be included"
    )


def test_importer_relevance_score_reflects_decay() -> None:
    """Importer chunk gets relevance proportional to IMPORTER_DECAY."""
    from archex.serve.context import IMPORTER_DECAY

    graph = DependencyGraph()
    graph.add_file_node("seed.py")
    graph.add_file_node("importer.py")
    graph.add_file_edge("importer.py", "seed.py", kind="imports")
    seed_chunk = make_chunk("cs", "seed.py", token_count=10)
    imp_chunk = make_chunk("ci", "importer.py", token_count=10)
    results = [(seed_chunk, 5.0)]
    bundle = assemble_context(results, graph, [seed_chunk, imp_chunk], "q", token_budget=1000)
    imp_rc = next(rc for rc in bundle.chunks if rc.chunk.file_path == "importer.py")
    # Relevance should be seed_normalized * IMPORTER_DECAY = 1.0 * 0.35
    assert abs(imp_rc.relevance_score - IMPORTER_DECAY) < 0.01


# ---------------------------------------------------------------------------
# Graph expansion multi-file retrieval
# ---------------------------------------------------------------------------


def test_graph_expansion_retrieves_import_connected_files() -> None:
    """Graph expansion pulls import-connected files in alongside a BM25 seed.

    When the seed does not dominate by chunk count, sum aggregation keeps the
    import target above the cutoff so multi-file architecture queries still
    retrieve the connected files.
    """
    graph = DependencyGraph()
    graph.add_file_node("handlers/base.py")
    graph.add_file_node("handlers/wsgi.py")
    graph.add_file_edge("handlers/base.py", "handlers/wsgi.py", kind="imports")

    base_chunk = make_chunk("cb0", "handlers/base.py", token_count=10)
    wsgi_chunk = make_chunk("cw0", "handlers/wsgi.py", token_count=10)
    all_chunks = [base_chunk, wsgi_chunk]

    results = [(base_chunk, 6.0)]
    bundle = assemble_context(
        results,
        graph,
        all_chunks,
        "How does the handler chain handle requests?",
        token_budget=5000,
    )
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "handlers/base.py" in included_files
    assert "handlers/wsgi.py" in included_files, (
        "wsgi.py (import target of seed) must be retrieved via graph expansion"
    )


# ---------------------------------------------------------------------------
# Expansion file budget
# ---------------------------------------------------------------------------


def test_max_expansion_files_allows_8() -> None:
    """Up to 8 expansion files can enter the candidate pool."""
    from archex.serve.context import MAX_EXPANSION_FILES

    assert MAX_EXPANSION_FILES == 8


def test_file_score_cutoff_is_0_10() -> None:
    """FILE_SCORE_CUTOFF is 0.10 (lowered from 0.15 to reduce false elimination)."""
    from archex.serve.context import FILE_SCORE_CUTOFF

    assert FILE_SCORE_CUTOFF == 0.10
