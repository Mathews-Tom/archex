"""Tests for the FastAPI HTTP API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from fastapi.testclient import TestClient

from archex.serve.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyzeEndpoint:
    def test_analyze_with_valid_source(self, client: TestClient, python_simple_repo: Path) -> None:
        response = client.post(
            "/analyze",
            json={
                "source": {"local_path": str(python_simple_repo)},
                "config": {"cache": False},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "repo" in data
        assert "stats" in data

    def test_analyze_invalid_source_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/analyze",
            json={
                "source": {},
            },
        )
        assert response.status_code == 422


class TestQueryEndpoint:
    def test_query_with_valid_source(self, client: TestClient, python_simple_repo: Path) -> None:
        response = client.post(
            "/query",
            json={
                "source": {"local_path": str(python_simple_repo)},
                "question": "How does authentication work?",
                "config": {"cache": False},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "chunks" in data


class TestTreeEndpoint:
    def test_tree_returns_valid_data(self, client: TestClient, python_simple_repo: Path) -> None:
        response = client.get(
            "/tree",
            params={
                "local_path": str(python_simple_repo),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total_files" in data


class TestBenchmarkEndpoints:
    def test_benchmark_results_no_data(self, client: TestClient) -> None:
        response = client.get("/benchmark/results")
        assert response.status_code == 200

    def test_benchmark_summary_no_data(self, client: TestClient) -> None:
        response = client.get("/benchmark/summary")
        assert response.status_code == 200

    def test_benchmark_gate_no_data(self, client: TestClient) -> None:
        response = client.get("/benchmark/gate")
        assert response.status_code == 200


class TestDashboard:
    def test_dashboard_serves_html(self, client: TestClient) -> None:
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "archex Dashboard" in response.text


class TestErrorHandling:
    def test_nonexistent_repo_returns_error(self, client: TestClient) -> None:
        response = client.post(
            "/analyze",
            json={
                "source": {"local_path": "/nonexistent/path/to/repo"},
                "config": {"cache": False},
            },
        )
        assert response.status_code in (404, 500)
