from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from archex.index.store import IndexStore
from archex.models import (
    CodeChunk,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextOmittedEdgeReason,
    ContextReceipt,
    ContextReceiptEdge,
    ContextReceiptTokenBudget,
    ContextRecommendedAction,
    ContextSkippedCandidate,
    ContextSkippedReason,
    Edge,
    EdgeConfidence,
    EdgeKind,
    Module,
    RankedChunk,
    SymbolKind,
)
from archex.reporting import count_tokens
from archex.scout import (
    ScoutBudget,
    ScoutResult,
    assemble_scout_from_store,
    chunk_handle,
    file_handle,
    render_scout,
    symbol_handle,
)


def _populate_store(db_path: Path) -> IndexStore:
    store = IndexStore(db_path)
    store.insert_chunks(
        [
            CodeChunk(
                id="pkg/app.py::run#function",
                content="SECRET BODY def run():\n    return load_model()",
                file_path="pkg/app.py",
                start_line=1,
                end_line=2,
                symbol_name="run",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id="pkg/app.py::run#function",
                qualified_name="run",
                signature="def run()",
            ),
            CodeChunk(
                id="pkg/models.py::Model#class",
                content="SECRET BODY class Model: pass",
                file_path="pkg/models.py",
                start_line=1,
                end_line=1,
                symbol_name="Model",
                symbol_kind=SymbolKind.CLASS,
                language="python",
                token_count=5,
                symbol_id="pkg/models.py::Model#class",
                qualified_name="Model",
                signature="class Model",
            ),
        ]
    )
    store.insert_edges(
        [
            Edge(
                source="pkg/app.py",
                target="pkg/models.py",
                kind=EdgeKind.IMPORTS,
                location="pkg/app.py:1",
                confidence=EdgeConfidence.HEURISTIC,
                confidence_score=0.75,
                evidence=["import pkg.models"],
            )
        ]
    )
    store.insert_modules(
        [
            Module(
                name="pkg",
                root_path="pkg",
                files=["pkg/app.py", "pkg/models.py"],
                responsibility="application models",
                cohesion_score=0.5,
                file_count=2,
            )
        ]
    )
    return store


def test_scout_assembles_no_body_structural_map(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        app_chunk = store.get_chunk("pkg/app.py::run#function")
        assert app_chunk is not None
        result = assemble_scout_from_store(
            store,
            "how does app loading work",
            ranked_chunks=[RankedChunk(chunk=app_chunk, final_score=0.9)],
            token_budget=400,
        )

    rendered = render_scout(result)

    assert count_tokens(rendered) <= 400
    assert result.budget.token_count == count_tokens(rendered)
    assert result.ranked_files[0].path == "pkg/app.py"
    assert result.modules[0].name == "pkg"
    assert result.symbols[0].name == "run"
    assert result.symbols[0].chunk_handle == chunk_handle("pkg/app.py::run#function")
    assert result.symbols[0].symbol_handle == symbol_handle("pkg/app.py::run#function")
    assert result.ranked_files[0].handle == file_handle("pkg/app.py")
    assert result.ranked_files[0].primary_chunk_handle == chunk_handle("pkg/app.py::run#function")
    assert result.ranked_files[0].primary_symbol_handle == symbol_handle("pkg/app.py::run#function")
    assert result.fetch_plan.handles == [symbol_handle("pkg/app.py::run#function")]
    assert result.fetch_plan.recommended_strategy == "chunk_first"
    import_edge = next(edge for edge in result.graph if edge.kind == "imports")
    assert import_edge.confidence == "heuristic"
    assert "SECRET BODY" not in rendered


def test_scout_truncates_deterministically_under_cap(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        first = assemble_scout_from_store(store, "models", token_budget=140)
        second = assemble_scout_from_store(store, "models", token_budget=140)

    assert render_scout(first) == render_scout(second)
    assert first.budget.truncated is True
    assert count_tokens(render_scout(first)) <= 140
    assert first.budget.token_count == count_tokens(render_scout(first))


def test_scout_markdown_receipt_shows_actionable_details() -> None:
    receipt = ContextReceipt(
        query="models",
        token_budget=ContextReceiptTokenBudget(requested=120, consumed=80),
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
        skipped_candidates=[
            ContextSkippedCandidate(
                file_path="pkg/extra.py",
                reason=ContextSkippedReason.BELOW_THRESHOLD,
                handle=file_handle("pkg/extra.py"),
                score=0.45,
            )
        ],
        omitted_edges=[
            ContextReceiptEdge(
                source="pkg/app.py",
                target="pkg/extra.py",
                kind=EdgeKind.IMPORTS,
                reason=ContextOmittedEdgeReason.BELOW_THRESHOLD,
            )
        ],
        returned_total=2,
        skipped_total=7,
        omitted_edges_total=3,
        context_complete=ContextCompletenessStatus.INCOMPLETE,
        context_complete_reason=ContextCompletenessReason.DEPENDENCY_FRONTIER_CUT,
        recommended_next_action=ContextRecommendedAction.FETCH_SKIPPED_CANDIDATE,
    )
    scout_result = ScoutResult(
        query="models",
        budget=ScoutBudget(token_budget=120, token_count=80),
        receipt=receipt,
    )

    rendered = render_scout(scout_result)

    assert "## Receipt" in rendered
    assert "- Budget: 80 / 120 tokens" in rendered
    assert "- Returned: 0 shown / 2 total" in rendered
    assert "- Skipped: 1 shown / 7 total" in rendered
    assert "- Omitted dependency edges: 1 shown / 3 total" in rendered
    assert "pkg/extra.py `file:pkg/extra.py`: below_threshold, score=0.450" in rendered
    assert "pkg/app.py --imports--> pkg/extra.py: below_threshold" in rendered


def test_scout_guardrail_prefers_direct_query_when_fetch_is_not_cheaper(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        app_chunk = store.get_chunk("pkg/app.py::run#function")
        model_chunk = store.get_chunk("pkg/models.py::Model#class")
        assert app_chunk is not None
        assert model_chunk is not None
        result = assemble_scout_from_store(
            store,
            "how does app loading work",
            ranked_chunks=[
                RankedChunk(chunk=app_chunk, final_score=0.9),
                RankedChunk(chunk=model_chunk, final_score=0.8),
            ],
            token_budget=400,
            direct_query_tokens=10,
        )

    assert result.fetch_plan.recommended_strategy == "direct_query"
    assert result.fetch_plan.guardrail_reason == "estimated_total_not_better_than_query"
    assert result.fetch_plan.handles == [
        symbol_handle("pkg/app.py::run#function"),
        symbol_handle("pkg/models.py::Model#class"),
    ]


def test_scout_guardrail_prefers_direct_query_when_query_is_already_narrow(
    tmp_path: Path,
) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        app_chunk = store.get_chunk("pkg/app.py::run#function")
        assert app_chunk is not None
        result = assemble_scout_from_store(
            store,
            "how does app loading work",
            ranked_chunks=[RankedChunk(chunk=app_chunk, final_score=0.9)],
            token_budget=400,
            direct_query_tokens=400,
            direct_query_file_paths=["pkg/app.py"],
        )

    assert result.fetch_plan.recommended_strategy == "direct_query"
    assert result.fetch_plan.guardrail_reason == "direct_query_already_narrow"


def test_scout_adapts_handle_count_when_score_mass_is_spread(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        extras = [
            CodeChunk(
                id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                content=f"def extra_{idx}():\\n    return {idx}",
                file_path=f"pkg/extra_{idx}.py",
                start_line=1,
                end_line=2,
                symbol_name=f"extra_{idx}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                qualified_name=f"extra_{idx}",
                signature=f"def extra_{idx}()",
            )
            for idx in range(4)
        ]
        store.insert_chunks(extras)
        ranked_chunks = [
            RankedChunk(chunk=chunk, final_score=score)
            for chunk, score in zip(
                [store.get_chunk("pkg/app.py::run#function"), *extras],
                [1.0, 0.95, 0.9, 0.85, 0.8],
                strict=True,
            )
            if chunk is not None
        ]
        result = assemble_scout_from_store(
            store,
            "how do the modules coordinate across the codebase",
            ranked_chunks=ranked_chunks,
            token_budget=700,
            direct_query_tokens=5000,
        )

    assert len(result.fetch_plan.handles) >= 3
    assert result.fetch_plan.coverage_score_mass >= 0.7


def test_scout_guardrail_prefers_direct_query_when_coverage_is_weak(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        extras = [
            CodeChunk(
                id=f"pkg/wide_{idx}.py::wide_{idx}#function",
                content=f"def wide_{idx}():\\n    return {idx}",
                file_path=f"pkg/wide_{idx}.py",
                start_line=1,
                end_line=2,
                symbol_name=f"wide_{idx}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id=f"pkg/wide_{idx}.py::wide_{idx}#function",
                qualified_name=f"wide_{idx}",
                signature=f"def wide_{idx}()",
            )
            for idx in range(8)
        ]
        store.insert_chunks(extras)
        ranked_chunks = [RankedChunk(chunk=chunk, final_score=1.0) for chunk in extras]
        result = assemble_scout_from_store(
            store,
            "how do the modules coordinate across the codebase",
            ranked_chunks=ranked_chunks,
            token_budget=700,
            direct_query_tokens=900,
            direct_query_file_paths=[chunk.file_path for chunk in extras[:6]],
        )

    assert result.fetch_plan.recommended_strategy == "direct_query"
    assert result.fetch_plan.guardrail_reason in {
        "projected_coverage_weak",
        "direct_query_precision_proxy",
    }


def test_scout_prefers_hybrid_fetch_when_coverage_is_thin_and_query_is_not_narrow(
    tmp_path: Path,
) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        extras = [
            CodeChunk(
                id=f"pkg/hybrid_{idx}.py::hybrid_{idx}#function",
                content=f"def hybrid_{idx}():\\n    return {idx}",
                file_path=f"pkg/hybrid_{idx}.py",
                start_line=1,
                end_line=2,
                symbol_name=f"hybrid_{idx}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id=f"pkg/hybrid_{idx}.py::hybrid_{idx}#function",
                qualified_name=f"hybrid_{idx}",
                signature=f"def hybrid_{idx}()",
            )
            for idx in range(10)
        ]
        store.insert_chunks(extras)
        ranked_chunks = [RankedChunk(chunk=chunk, final_score=1.0) for chunk in extras]
        result = assemble_scout_from_store(
            store,
            "how do the modules coordinate across the codebase",
            ranked_chunks=ranked_chunks,
            token_budget=700,
            direct_query_tokens=4000,
            direct_query_file_paths=[chunk.file_path for chunk in extras],
        )

    assert result.fetch_plan.recommended_strategy == "hybrid_fetch"
    assert result.fetch_plan.guardrail_reason == "projected_coverage_thin"
    assert any(handle.startswith("file:") for handle in result.fetch_plan.handles)


def test_scout_handles_fetch_exact_symbols_and_query_chunks(tmp_path: Path) -> None:
    from archex.api import get_symbol, get_symbols_batch, query
    from archex.models import RepoSource

    store = _populate_store(tmp_path / "index.db")
    source = RepoSource(local_path="/fake")
    app_chunk = store.get_chunk("pkg/app.py::run#function")
    model_chunk = store.get_chunk("pkg/models.py::Model#class")
    assert app_chunk is not None
    assert model_chunk is not None
    with (
        patch("archex.api._ensure_index", return_value=store),
        patch.object(store, "close", return_value=None),
    ):
        result = assemble_scout_from_store(
            store,
            "fetch exact scout handles",
            ranked_chunks=[
                RankedChunk(chunk=app_chunk, final_score=1.0),
                RankedChunk(chunk=model_chunk, final_score=0.9),
            ],
            token_budget=400,
        )
        symbol = get_symbol(source, symbol_id=symbol_handle("pkg/app.py::run#function"))
        chunk_symbol = get_symbol(source, symbol_id=chunk_handle("pkg/models.py::Model#class"))
        batch = get_symbols_batch(
            source,
            symbol_ids=[
                symbol_handle("pkg/app.py::run#function"),
                chunk_handle("pkg/models.py::Model#class"),
            ],
        )
        bundle = query(
            source,
            "fetch exact scout handles",
            handles=result.fetch_plan.handles,
        )

    assert symbol is not None
    assert symbol.symbol_id == "pkg/app.py::run#function"
    assert chunk_symbol is not None
    assert chunk_symbol.symbol_id == "pkg/models.py::Model#class"
    assert [item.symbol_id if item is not None else None for item in batch] == [
        "pkg/app.py::run#function",
        "pkg/models.py::Model#class",
    ]
    assert [ranked.chunk.id for ranked in bundle.chunks] == [
        "pkg/app.py::run#function",
        "pkg/models.py::Model#class",
    ]
    assert bundle.retrieval_metadata.strategy == "scout_handle"


def test_scout_cli_emits_handles_and_budget() -> None:
    from click.testing import CliRunner

    from archex.cli.main import cli
    from archex.scout import ScoutBudget, ScoutFile, ScoutResult

    result_model = ScoutResult(
        query="delta indexing",
        ranked_files=[
            ScoutFile(
                path="src/archex/index/delta.py",
                language="python",
                lines=100,
                symbol_count=4,
                handle=file_handle("src/archex/index/delta.py"),
            )
        ],
        budget=ScoutBudget(token_budget=120),
    )
    with patch("archex.cli.scout_cmd.scout", return_value=result_model) as scout_mock:
        result = CliRunner().invoke(
            cli,
            ["scout", ".", "how does delta indexing work", "--budget", "120"],
        )

    assert result.exit_code == 0, result.output
    assert file_handle("src/archex/index/delta.py") in result.output
    assert scout_mock.call_args.kwargs["token_budget"] == 120


def test_scout_caps_files_at_definition_lookup_intent_limit(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        extras = [
            CodeChunk(
                id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                content=f"def extra_{idx}():\\n    return {idx}",
                file_path=f"pkg/extra_{idx}.py",
                start_line=1,
                end_line=2,
                symbol_name=f"extra_{idx}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                qualified_name=f"extra_{idx}",
                signature=f"def extra_{idx}()",
            )
            for idx in range(6)
        ]
        store.insert_chunks(extras)
        ranked_chunks = [
            RankedChunk(chunk=chunk, final_score=score)
            for chunk, score in zip(
                [store.get_chunk("pkg/app.py::run#function"), *extras],
                [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
                strict=True,
            )
            if chunk is not None
        ]
        result = assemble_scout_from_store(
            store,
            "implementation of run",
            ranked_chunks=ranked_chunks,
            token_budget=8000,
        )

    # 7 candidate files exist; DEFINITION_LOOKUP intent caps at 6 (old fixed
    # default of 12 would have kept all 7).
    assert len(result.ranked_files) == 6
    assert result.budget.truncated is False
    assert result.budget.omitted_files == 1


def test_scout_raises_file_cap_for_architecture_broad_intent(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        extras = [
            CodeChunk(
                id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                content=f"def extra_{idx}():\\n    return {idx}",
                file_path=f"pkg/extra_{idx}.py",
                start_line=1,
                end_line=2,
                symbol_name=f"extra_{idx}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                qualified_name=f"extra_{idx}",
                signature=f"def extra_{idx}()",
            )
            for idx in range(20)
        ]
        store.insert_chunks(extras)
        ranked_chunks = [
            RankedChunk(chunk=chunk, final_score=1.0 - idx * 0.03)
            for idx, chunk in enumerate([store.get_chunk("pkg/app.py::run#function"), *extras])
            if chunk is not None
        ]
        result = assemble_scout_from_store(
            store,
            "pipeline overview",
            ranked_chunks=ranked_chunks,
            token_budget=20000,
        )

    # 21 candidate files exist; ARCHITECTURE_BROAD intent caps at 16 (old fixed
    # default of 12 would have dropped 5 additional relevant files).
    assert len(result.ranked_files) == 16
    assert result.budget.truncated is False
    assert result.budget.omitted_files == 5


def test_scout_keeps_historical_default_cap_for_general_intent(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        extras = [
            CodeChunk(
                id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                content=f"def extra_{idx}():\\n    return {idx}",
                file_path=f"pkg/extra_{idx}.py",
                start_line=1,
                end_line=2,
                symbol_name=f"extra_{idx}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                qualified_name=f"extra_{idx}",
                signature=f"def extra_{idx}()",
            )
            for idx in range(15)
        ]
        store.insert_chunks(extras)
        ranked_chunks = [
            RankedChunk(chunk=chunk, final_score=1.0 - idx * 0.03)
            for idx, chunk in enumerate([store.get_chunk("pkg/app.py::run#function"), *extras])
            if chunk is not None
        ]
        result = assemble_scout_from_store(
            store,
            "models",
            ranked_chunks=ranked_chunks,
            token_budget=15000,
        )

    # 16 candidate files exist; GENERAL intent is unaffected by the adaptive
    # table and keeps the historical fixed default of 12.
    assert len(result.ranked_files) == 12
    assert result.budget.truncated is False
    assert result.budget.omitted_files == 4


def test_scout_explicit_file_limit_overrides_intent_classification(tmp_path: Path) -> None:
    with _populate_store(tmp_path / "index.db") as store:
        extras = [
            CodeChunk(
                id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                content=f"def extra_{idx}():\\n    return {idx}",
                file_path=f"pkg/extra_{idx}.py",
                start_line=1,
                end_line=2,
                symbol_name=f"extra_{idx}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=8,
                symbol_id=f"pkg/extra_{idx}.py::extra_{idx}#function",
                qualified_name=f"extra_{idx}",
                signature=f"def extra_{idx}()",
            )
            for idx in range(5)
        ]
        store.insert_chunks(extras)
        ranked_chunks = [
            RankedChunk(chunk=chunk, final_score=score)
            for chunk, score in zip(
                [store.get_chunk("pkg/app.py::run#function"), *extras],
                [1.0, 0.95, 0.9, 0.85, 0.8, 0.75],
                strict=True,
            )
            if chunk is not None
        ]
        # "pipeline overview" would classify as ARCHITECTURE_BROAD (cap 16),
        # but an explicit file_limit bypasses intent classification entirely.
        result = assemble_scout_from_store(
            store,
            "pipeline overview",
            ranked_chunks=ranked_chunks,
            token_budget=4000,
            file_limit=3,
        )

    assert len(result.ranked_files) == 3
    assert result.budget.truncated is False
    assert result.budget.omitted_files == 3
