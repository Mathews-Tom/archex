from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from archex.index.store import IndexStore
from archex.models import CodeChunk, Edge, EdgeConfidence, EdgeKind, Module, RankedChunk, SymbolKind
from archex.reporting import count_tokens
from archex.scout import (
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
