from __future__ import annotations

from pathlib import Path

from archex.index.store import IndexStore
from archex.models import CodeChunk, Edge, EdgeConfidence, EdgeKind, Module, RankedChunk, SymbolKind
from archex.reporting import count_tokens
from archex.scout import assemble_scout_from_store, render_scout


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
