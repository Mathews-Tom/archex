"""FastAPI HTTP API for archex."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from archex import api
from archex.exceptions import ArchexError
from archex.models import (
    ArchProfile,
    ComparisonResult,
    Config,
    ContextBundle,
    FileOutline,
    FileTree,
    IndexConfig,
    RepoSource,
    ScoringWeights,
    SymbolMatch,
    SymbolSource,
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    source: RepoSource
    config: Config | None = None


class QueryRequest(BaseModel):
    source: RepoSource
    question: str
    token_budget: int = 8192
    config: Config | None = None
    index_config: IndexConfig | None = None
    scoring_weights: ScoringWeights | None = None


class CompareRequest(BaseModel):
    source_a: RepoSource
    source_b: RepoSource
    dimensions: list[str] | None = None
    config: Config | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="archex", description="Architecture extraction & codebase intelligence API")

    # --- Health ---
    @app.get("/health")
    def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    # --- Core API ---
    @app.post("/analyze")
    def analyze_endpoint(req: AnalyzeRequest) -> ArchProfile:  # pyright: ignore[reportUnusedFunction]
        try:
            return api.analyze(req.source, req.config)
        except (FileNotFoundError, OSError, ArchexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/query")
    def query_endpoint(req: QueryRequest) -> ContextBundle:  # pyright: ignore[reportUnusedFunction]
        try:
            return api.query(
                req.source,
                req.question,
                token_budget=req.token_budget,
                config=req.config,
                index_config=req.index_config,
                scoring_weights=req.scoring_weights,
            )
        except (FileNotFoundError, OSError, ArchexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/compare")
    def compare_endpoint(req: CompareRequest) -> ComparisonResult:  # pyright: ignore[reportUnusedFunction]
        try:
            return api.compare(
                req.source_a, req.source_b, dimensions=req.dimensions, config=req.config
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, OSError, ArchexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- Precision tools ---
    @app.get("/tree")
    def tree_endpoint(local_path: str, depth: int = 5, language: str | None = None) -> FileTree:  # pyright: ignore[reportUnusedFunction]
        source = RepoSource(local_path=local_path)
        try:
            return api.file_tree(source, max_depth=depth, language=language)
        except (FileNotFoundError, OSError, ArchexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/outline")
    def outline_endpoint(local_path: str, file: str) -> FileOutline:  # pyright: ignore[reportUnusedFunction]
        source = RepoSource(local_path=local_path)
        try:
            return api.file_outline(source, file)
        except (FileNotFoundError, OSError, ArchexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/symbols")
    def symbols_endpoint(local_path: str, query: str, limit: int = 20) -> list[SymbolMatch]:  # pyright: ignore[reportUnusedFunction]
        source = RepoSource(local_path=local_path)
        try:
            return api.search_symbols(source, query, limit=limit)
        except (FileNotFoundError, OSError, ArchexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/symbol/{symbol_id:path}")
    def symbol_endpoint(symbol_id: str, local_path: str) -> SymbolSource:  # pyright: ignore[reportUnusedFunction]
        source = RepoSource(local_path=local_path)
        try:
            result = api.get_symbol(source, symbol_id)
        except (FileNotFoundError, OSError, ArchexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol_id} not found")
        return result

    # --- Benchmark endpoints ---
    @app.get("/benchmark/results")
    def benchmark_results() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return latest benchmark results if available."""
        baseline_path = Path.home() / ".archex" / "benchmark_baseline.json"
        if not baseline_path.exists():
            return {"results": [], "message": "No benchmark results found"}
        try:
            from archex.benchmark.baseline import load_baseline

            data = json.loads(baseline_path.read_text())
            baseline = load_baseline(data)
            return {"results": [e.model_dump() for e in baseline.entries]}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/benchmark/summary")
    def benchmark_summary() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return formatted benchmark summary."""
        baseline_path = Path.home() / ".archex" / "benchmark_baseline.json"
        if not baseline_path.exists():
            return {"summary": "No benchmark results found"}
        try:
            from archex.benchmark.baseline import load_baseline

            data = json.loads(baseline_path.read_text())
            baseline = load_baseline(data)
            if not baseline.entries:
                return {"summary": "No benchmark baseline found"}
            lines: list[str] = []
            lines.append("# Benchmark Baseline Summary")
            lines.append(f"**Created:** {baseline.created_at}")
            lines.append(f"**Version:** {baseline.archex_version or 'unknown'}")
            lines.append(f"**Entries:** {len(baseline.entries)}")
            return {"summary": "\n".join(lines)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/benchmark/gate")
    def benchmark_gate() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return quality gate check result based on baseline entries."""
        baseline_path = Path.home() / ".archex" / "benchmark_baseline.json"
        if not baseline_path.exists():
            return {"passed": False, "reason": "No benchmark baseline found"}
        try:
            from archex.benchmark.baseline import load_baseline

            data = json.loads(baseline_path.read_text())
            baseline = load_baseline(data)
            if not baseline.entries:
                return {"passed": False, "reason": "No benchmark baseline found"}
            # Check entries against minimum thresholds
            min_recall = 0.6
            min_f1 = 0.4
            violations: list[str] = []
            for entry in baseline.entries:
                if entry.recall < min_recall:
                    violations.append(
                        f"{entry.task_id}/{entry.strategy}: "
                        f"recall {entry.recall:.2f} < {min_recall}"
                    )
                if entry.f1_score < min_f1:
                    violations.append(
                        f"{entry.task_id}/{entry.strategy}: f1 {entry.f1_score:.2f} < {min_f1}"
                    )
            if violations:
                return {"passed": False, "violations": violations}
            return {"passed": True}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # --- Dashboard ---
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():

        @app.get("/dashboard", response_class=HTMLResponse)
        def dashboard() -> HTMLResponse:  # pyright: ignore[reportUnusedFunction]
            index_html = static_dir / "index.html"
            return HTMLResponse(content=index_html.read_text())

    return app
