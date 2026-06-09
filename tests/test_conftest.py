from __future__ import annotations

from tests.conftest import implementation_gate_paths


def test_implementation_gate_paths_match_benchmark_analyze_and_serve_slices() -> None:
    assert implementation_gate_paths(("tests/benchmark/test_gate.py", "tests/serve/")) is True
    assert implementation_gate_paths(("tests/analyze/", "tests/benchmark/test_gate.py")) is True


def test_implementation_gate_paths_reject_non_gate_slices() -> None:
    assert implementation_gate_paths(("tests/serve/", "tests/test_cli.py")) is False
    assert (
        implementation_gate_paths(("tests/benchmark/test_gate.py", "tests/integrations/")) is False
    )
    assert implementation_gate_paths(()) is False
