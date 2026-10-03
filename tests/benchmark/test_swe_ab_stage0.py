"""Tests for the Stage 0 checks that gate a campaign; fakes only, no network."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from archex.benchmark.swe_ab import Campaign

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

stage0: Any = importlib.import_module("swe_ab_stage0")

from archex.benchmark.swe_ab import BASE_TOOLS, MCP_TOOLS, SweAbArm, load_campaign  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGNS = (
    "benchmarks/swe_ab/campaigns/muna.yml",
    "benchmarks/swe_ab/campaigns/openrouter-space-bunny.yml",
)
_KEY = "provider-secret-key-value"


def _campaign(path: str) -> Campaign:
    return load_campaign(path, root=_ROOT)


@pytest.fixture(params=_CAMPAIGNS)
def campaign(request: pytest.FixtureRequest) -> Campaign:
    return _campaign(str(request.param))


def _capture(efforts: dict[str, Any]) -> Any:
    """A runner that returns, per configuration label, a body carrying that effort (or none)."""

    def capture(
        omp_command: list[str], campaign: Any, config: Any, work: Path
    ) -> dict[str, Any] | None:
        effort = efforts[config.label]
        return None if effort == "__no_request__" else {"model": "m", "reasoning_effort": effort}

    return capture


def _no_effort_capture(
    omp_command: list[str], campaign: Any, config: Any, work: Path
) -> dict[str, Any]:
    return {"model": "m", "enable_thinking": True}


def test_effort_request_shape_passes_when_every_request_carries_its_effort(
    campaign: Campaign, tmp_path: Path
) -> None:
    efforts = {config.label: config.thinking for config in campaign.configurations}

    check = stage0.effort_request_shape_check(
        ["omp"], campaign, tmp_path, capture=_capture(efforts)
    )

    assert check["status"] == "pass"
    assert check["reasoning_effort"] == efforts


def test_effort_request_shape_fails_a_request_without_reasoning_effort(
    campaign: Campaign, tmp_path: Path
) -> None:
    check = stage0.effort_request_shape_check(
        ["omp"], campaign, tmp_path, capture=_no_effort_capture
    )

    assert check["status"] == "fail"
    for config in campaign.configurations:
        assert config.label in check["detail"]


def test_effort_request_shape_names_the_configuration_with_the_wrong_effort(
    tmp_path: Path,
) -> None:
    campaign = _campaign("benchmarks/swe_ab/campaigns/muna.yml")
    efforts = {config.label: config.thinking for config in campaign.configurations}
    wrong = campaign.configurations[1]
    efforts[wrong.label] = "low" if wrong.thinking == "high" else "high"

    check = stage0.effort_request_shape_check(
        ["omp"], campaign, tmp_path, capture=_capture(efforts)
    )

    assert check["status"] == "fail"
    assert wrong.label in check["detail"]
    assert all(c.label not in check["detail"] for c in campaign.configurations if c is not wrong)


def test_effort_request_shape_fails_when_omp_sent_no_request(
    campaign: Campaign, tmp_path: Path
) -> None:
    efforts = {config.label: config.thinking for config in campaign.configurations}
    efforts[campaign.configurations[0].label] = "__no_request__"

    check = stage0.effort_request_shape_check(
        ["omp"], campaign, tmp_path, capture=_capture(efforts)
    )

    assert check["status"] == "fail"
    assert campaign.configurations[0].label in check["detail"]


class _Http:
    """A fake POST: records what was sent and answers with a canned status and body."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.answer = (status, json.dumps(body))
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        self.calls.append((url, headers, json.loads(body)))
        return self.answer


def _keyed(monkeypatch: pytest.MonkeyPatch, campaign: Campaign) -> None:
    monkeypatch.setenv(campaign.credential_env, _KEY)


# The answers measured on 2026-10-03 for a key the provider authenticated, a bad key, and a
# nonexistent model.
_MUNA_GOOD = (404, {"error": {"code": "model_not_found"}})
_OPENROUTER_GOOD = (
    400,
    {"error": {"message": "nobody/does-not-exist is not a valid model ID", "code": 400}},
)


@pytest.mark.parametrize(
    "answer", [_MUNA_GOOD, _OPENROUTER_GOOD], ids=["muna-404", "openrouter-400"]
)
def test_key_check_passes_when_the_provider_rejects_only_the_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    campaign: Campaign,
    answer: tuple[int, dict[str, Any]],
) -> None:
    _keyed(monkeypatch, campaign)
    http = _Http(*answer)

    check = stage0.provider_key_check(campaign, tmp_path / ".env", post=http)

    assert check["id"] == "provider_key_accepted"
    assert check["status"] == "pass"
    [(url, headers, body)] = http.calls
    assert url == f"{campaign.base_url}/chat/completions"
    assert headers["Authorization"] == f"Bearer {_KEY}"
    assert body["max_tokens"] == 1
    assert body["model"] == "nobody/does-not-exist"
    assert _KEY not in json.dumps(check)


@pytest.mark.parametrize("status", [401, 403])
def test_key_check_fails_a_rejected_key_without_leaking_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign: Campaign, status: int
) -> None:
    _keyed(monkeypatch, campaign)
    http = _Http(status, {"error": {"message": "User not found.", "code": status}})

    check = stage0.provider_key_check(campaign, tmp_path / ".env", post=http)

    assert check["status"] == "fail"
    assert f"key rejected (HTTP {status})" in check["detail"]
    assert _KEY not in json.dumps(check)


@pytest.mark.parametrize("status", [200, 429, 500])
def test_key_check_fails_any_other_answer_with_its_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign: Campaign, status: int
) -> None:
    _keyed(monkeypatch, campaign)

    check = stage0.provider_key_check(
        campaign, tmp_path / ".env", post=_Http(status, {"error": {"code": "other"}})
    )

    assert check["status"] == "fail"
    assert "unexpected answer" in check["detail"]
    assert str(status) in check["detail"]


def test_key_check_without_a_key_fails_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign: Campaign
) -> None:
    monkeypatch.delenv(campaign.credential_env, raising=False)
    http = _Http(*_MUNA_GOOD)

    check = stage0.provider_key_check(campaign, tmp_path / "absent.env", post=http)

    assert check["status"] == "fail"
    assert campaign.credential_env in check["detail"]
    assert http.calls == []


def test_key_check_reads_only_the_campaigns_line_of_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign: Campaign
) -> None:
    monkeypatch.delenv(campaign.credential_env, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER_SECRET=nope\n{campaign.credential_env}={_KEY}\n")
    http = _Http(*_MUNA_GOOD)

    check = stage0.provider_key_check(campaign, env_file, post=http)

    assert check["status"] == "pass"
    assert http.calls[0][1]["Authorization"] == f"Bearer {_KEY}"
    assert "nope" not in json.dumps(http.calls)


def _omp_listing(selectors: list[str], provider: str) -> Any:
    models = [{"provider": provider, "selector": s, "id": s.split("/", 1)[1]} for s in selectors]

    def run(argv: list[str], **kwargs: Any) -> Any:
        class Done:
            stdout = json.dumps({"models": models})

        return Done()

    return run


def test_route_check_requires_every_configuration_selector_under_the_campaign_provider(
    campaign: Campaign,
) -> None:
    selectors = sorted({config.selector for config in campaign.configurations})
    listing = _omp_listing(selectors, campaign.provider)

    check = stage0.provider_route_check(["omp"], campaign, run=listing)
    assert check["id"] == "provider_route"
    assert check["status"] == "pass"

    elsewhere = stage0.provider_route_check(["omp"], campaign, run=_omp_listing(selectors, "other"))
    assert elsewhere["status"] == "fail"


def test_route_check_names_a_selector_omp_does_not_list() -> None:
    campaign = _campaign("benchmarks/swe_ab/campaigns/muna.yml")
    selectors = sorted({config.selector for config in campaign.configurations})

    short = stage0.provider_route_check(
        ["omp"], campaign, run=_omp_listing(selectors[:-1], campaign.provider)
    )

    assert short["status"] == "fail"
    assert selectors[-1] in short["detail"]


def test_container_checks_run_the_bundle_and_survive_an_instance_that_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign("benchmarks/swe_ab/campaigns/muna.yml")
    ran: list[list[str]] = []

    class FakeRuntime:
        def __init__(self, spec: Any, *, mounts: list[tuple[Path, str]]) -> None:
            if spec.task_id == "t0":
                raise RuntimeError("docker run failed (125): 401 Unauthorized")
            self.spec = spec

        def run(self, argv: list[str], **kwargs: Any) -> Any:
            ran.append(argv)
            out = "x86_64" if argv[0] == "uname" else ""
            if argv[-1] == "--version" and argv[0] == stage0.CONTAINER_OMP_COMMAND:
                out = "omp/18.4.4"
            return type("Done", (), {"stdout": out, "returncode": 0})()

        def close(self) -> None:
            return

    def score_patch(spec: Any, patch: Path) -> bool:
        return patch.stat().st_size > 0  # the gold patch resolves, the empty one does not

    def index_in_container(spec: Any, rt: Any) -> float:
        return 1.0

    def annotate_latency(rt: Any) -> dict[str, int]:
        return {"p50_ms": 100, "budget_ms": 500}

    def provider_container_check(campaign: Any, env_file: Path) -> dict[str, str]:
        return {"id": "provider_reachable_from_container", "status": "pass"}

    def bun_emulation_check() -> dict[str, str]:
        return {"id": "bun_runs_under_emulation", "status": "pass"}

    monkeypatch.setattr(stage0.cell_runner, "DockerRuntime", FakeRuntime)
    monkeypatch.setattr(stage0.cell_runner, "score_patch", score_patch)
    monkeypatch.setattr(stage0.cell_runner, "index_in_container", index_in_container)
    monkeypatch.setattr(stage0, "annotate_latency", annotate_latency)
    monkeypatch.setattr(stage0, "provider_container_check", provider_container_check)
    monkeypatch.setattr(stage0, "bun_emulation_check", bun_emulation_check)
    task = tmp_path / "tasks" / "t1" / "solution"
    task.mkdir(parents=True)
    (task / "gold_patch.diff").write_text("diff --git a/x b/x\n")
    (tmp_path / "instances.txt").write_text("t0\nt1\n")
    (tmp_path / "tasks" / "t0" / "solution").mkdir(parents=True)
    (tmp_path / "tasks" / "t0" / "solution" / "gold_patch.diff").write_text("diff --git a/x b/x\n")
    args = stage0.argparse.Namespace(
        instances=tmp_path / "instances.txt", tasks_root=tmp_path / "tasks",
        omp_command="/host/omp-18.4.4", container_omp_command=stage0.CONTAINER_OMP_COMMAND,
        profile_dir=tmp_path, archex_wheel=tmp_path / "w.whl", uv_binary=tmp_path / "uv",
        omp_dir=tmp_path, network="bridge", env_file=tmp_path / ".env",
    )  # fmt: skip

    checks = {c["id"]: c for c in stage0.host_checks(args, campaign)}

    # t0's container never starts: a named failure, and t1 is still checked with the bundle.
    assert checks["container_setup:t0"]["status"] == "fail"
    assert checks["omp_runs_in_container"]["per_instance"] == {"t0": False, "t1": True}
    assert not any("/host/omp-18.4.4" in part for argv in ran for part in argv)


# --- the MCP arm's checks ------------------------------------------------------------------------


def _advertised(
    *, m: list[str] | None = None, leak_into: str | None = None
) -> dict[str, list[str]]:
    base = sorted(BASE_TOOLS)
    tools = {arm.value: list(base) for arm in SweAbArm}
    tools["M"] = m if m is not None else sorted([*base, *MCP_TOOLS])
    if leak_into is not None:
        tools[leak_into] = sorted([*base, MCP_TOOLS[0]])
    return tools


def test_the_mcp_tools_check_passes_when_only_m_advertises_them() -> None:
    check = stage0.mcp_tools_check(_advertised())

    assert check["id"] == "mcp_tools_in_m_arm"
    assert check["status"] == "pass"


def test_the_mcp_tools_check_fails_when_m_lacks_a_tool() -> None:
    base = sorted(BASE_TOOLS)

    check = stage0.mcp_tools_check(_advertised(m=sorted([*base, MCP_TOOLS[0]])))

    assert check["status"] == "fail"
    assert MCP_TOOLS[1] in check["detail"]


def test_the_mcp_tools_check_fails_when_another_arm_advertises_one() -> None:
    check = stage0.mcp_tools_check(_advertised(leak_into="HC"))

    assert check["status"] == "fail"
    assert "HC" in check["detail"]


def _request(tool_text: str | None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
    if tool_text is not None:
        messages.append({"role": "tool", "content": tool_text})
    return {"messages": messages}


def test_the_mcp_call_check_passes_on_archex_context_and_records_only_its_size() -> None:
    text = '<context repo="x">secret source text</context>'

    check = stage0.mcp_call_check([_request(None), _request(text)], calls=1)

    assert check["status"] == "pass"
    assert check["tool_result_chars"] == len(text)
    assert "secret source text" not in json.dumps(check)


@pytest.mark.parametrize(
    ("requests", "calls"),
    [
        ([_request(None), _request("Error: repo not found")], 1),
        ([_request(None), _request("<context>x</context>")], 0),
        ([_request(None)], 0),
        ([], 0),
    ],
)
def test_the_mcp_call_check_fails_without_context_or_without_a_recorded_call(
    requests: list[dict[str, Any]], calls: int
) -> None:
    assert stage0.mcp_call_check(requests, calls=calls)["status"] == "fail"
