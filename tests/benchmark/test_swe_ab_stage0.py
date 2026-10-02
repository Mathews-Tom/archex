"""Tests for the Stage 0 checks that gate the Muna campaign; fakes only, no network."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

stage0: Any = importlib.import_module("swe_ab_stage0")

from archex.benchmark.swe_ab import CONFIGURATIONS  # noqa: E402

_KEY = "muna-secret-key-value"


def _capture(efforts: dict[str, Any]) -> Any:
    """A runner that returns, per configuration label, a body carrying that effort (or none)."""

    def capture(omp_command: list[str], config: Any, work: Path) -> dict[str, Any] | None:
        effort = efforts[config.label]
        return None if effort == "__no_request__" else {"model": "m", "reasoning_effort": effort}

    return capture


def _no_effort_capture(omp_command: list[str], config: Any, work: Path) -> dict[str, Any]:
    return {"model": "m", "enable_thinking": True}


def test_effort_request_shape_passes_when_every_request_carries_its_effort(tmp_path: Path) -> None:
    efforts = {config.label: config.thinking for config in CONFIGURATIONS}

    check = stage0.effort_request_shape_check(["omp"], tmp_path, capture=_capture(efforts))

    assert check["status"] == "pass"
    assert check["reasoning_effort"] == efforts


def test_effort_request_shape_fails_a_request_without_reasoning_effort(tmp_path: Path) -> None:
    check = stage0.effort_request_shape_check(["omp"], tmp_path, capture=_no_effort_capture)

    assert check["status"] == "fail"
    for config in CONFIGURATIONS:
        assert config.label in check["detail"]


def test_effort_request_shape_names_the_configuration_with_the_wrong_effort(
    tmp_path: Path,
) -> None:
    efforts = {config.label: config.thinking for config in CONFIGURATIONS}
    wrong = CONFIGURATIONS[1]
    efforts[wrong.label] = "low" if wrong.thinking == "high" else "high"

    check = stage0.effort_request_shape_check(["omp"], tmp_path, capture=_capture(efforts))

    assert check["status"] == "fail"
    assert wrong.label in check["detail"]
    assert all(c.label not in check["detail"] for c in CONFIGURATIONS if c is not wrong)


def test_effort_request_shape_fails_when_omp_sent_no_request(tmp_path: Path) -> None:
    efforts = {config.label: config.thinking for config in CONFIGURATIONS}
    efforts[CONFIGURATIONS[0].label] = "__no_request__"

    check = stage0.effort_request_shape_check(["omp"], tmp_path, capture=_capture(efforts))

    assert check["status"] == "fail"
    assert CONFIGURATIONS[0].label in check["detail"]


class _Http:
    """A fake POST: records what was sent and answers with a canned status and body."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.answer = (status, json.dumps(body))
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        self.calls.append((url, headers, json.loads(body)))
        return self.answer


def _keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUNA_ACCESS_KEY", _KEY)


def test_key_check_passes_on_model_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _keyed(monkeypatch)
    http = _Http(404, {"error": {"code": "model_not_found"}})

    check = stage0.muna_key_check(tmp_path / ".env", post=http)

    assert check["status"] == "pass"
    [(url, headers, body)] = http.calls
    assert url == "https://inference.muna.ai/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {_KEY}"
    assert body["max_tokens"] == 1
    assert body["model"] == "@nobody/does-not-exist"


def test_key_check_fails_a_rejected_key_without_leaking_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _keyed(monkeypatch)
    http = _Http(401, {"error": {"code": "invalid_api_key"}})

    check = stage0.muna_key_check(tmp_path / ".env", post=http)

    assert check["status"] == "fail"
    assert "rejected" in check["detail"]
    assert _KEY not in json.dumps(check)


def test_key_check_fails_any_other_answer_with_its_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _keyed(monkeypatch)

    check = stage0.muna_key_check(
        tmp_path / ".env", post=_Http(404, {"error": {"code": "something_else"}})
    )

    assert check["status"] == "fail"
    assert "404" in check["detail"]


def test_key_check_without_a_key_fails_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MUNA_ACCESS_KEY", raising=False)
    http = _Http(404, {"error": {"code": "model_not_found"}})

    check = stage0.muna_key_check(tmp_path / "absent.env", post=http)

    assert check["status"] == "fail"
    assert "MUNA_ACCESS_KEY" in check["detail"]
    assert http.calls == []


def test_key_check_reads_only_the_muna_line_of_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MUNA_ACCESS_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER_SECRET=nope\nMUNA_ACCESS_KEY={_KEY}\n")
    http = _Http(404, {"error": {"code": "model_not_found"}})

    check = stage0.muna_key_check(env_file, post=http)

    assert check["status"] == "pass"
    assert http.calls[0][1]["Authorization"] == f"Bearer {_KEY}"
    assert "nope" not in json.dumps(http.calls)


def _omp_listing(selectors: list[str], provider: str = "muna") -> Any:
    models = [{"provider": provider, "selector": s, "id": s.split("/", 1)[1]} for s in selectors]

    def run(argv: list[str], **kwargs: Any) -> Any:
        class Done:
            stdout = json.dumps({"models": models})

        return Done()

    return run


def test_route_check_requires_every_configuration_selector_under_muna() -> None:
    selectors = sorted({config.selector for config in CONFIGURATIONS})

    assert stage0.muna_route_check(["omp"], run=_omp_listing(selectors))["status"] == "pass"

    short = stage0.muna_route_check(["omp"], run=_omp_listing(selectors[:-1]))
    assert short["status"] == "fail"
    assert selectors[-1] in short["detail"]

    elsewhere = stage0.muna_route_check(["omp"], run=_omp_listing(selectors, provider="other"))
    assert elsewhere["status"] == "fail"
