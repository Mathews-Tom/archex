"""Tests for the SWE A/B scripts on subscriptions: broker credentials, quota guard, Stage 0.

Everything here runs against fakes: no omp, no Docker, no broker, no model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

cell_runner: Any = importlib.import_module("run_swe_ab_cell")
suite: Any = importlib.import_module("run_swe_ab_suite")
stage0: Any = importlib.import_module("swe_ab_stage0")

from archex.benchmark.swe_ab import (  # noqa: E402
    BROKER_ENV_NAMES,
    MODELS,
    CellKey,
    FailureReason,
    QuotaEvidence,
    SweAbArm,
    Usage,
    load_cell,
    quota_blocked_relative_path,
)

MODEL = "anthropic/claude-sonnet-5-5"
TOKEN = "s3cret-broker-token"  # noqa: S105 - a fake, used to prove it never leaks


# --- the quota guard --------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _blocked_usage(reset_in_seconds: float, clock: _Clock) -> dict[str, Any]:
    limit = {
        "id": "5h",
        "amount": {"unit": "percent", "usedFraction": 1.0},
        "scope": {"provider": "anthropic"},
        "window": {"id": "5h", "resetsAt": int((clock.now + reset_in_seconds) * 1000)},
    }
    return {"reports": [{"provider": "anthropic", "limits": [limit]}]}


def _clear_usage() -> dict[str, Any]:
    limit = {"id": "5h", "amount": {"unit": "percent", "usedFraction": 0.2}, "scope": {}}
    return {"reports": [{"provider": "anthropic", "limits": [limit]}]}


class _Script:
    """A fetch that answers from a list (an Exception in the list is raised)."""

    def __init__(self, *answers: dict[str, Any] | Exception) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, url: str, token: str) -> dict[str, Any]:
        self.calls += 1
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _start() -> tuple[_Clock, list[dict[str, Any]]]:
    return _Clock(), []


def _no_invalidate(url: str, token: str) -> None:
    return None


def _guard(
    clock: _Clock,
    fetch: _Script,
    logs: list[dict[str, Any]],
    **overrides: Any,
) -> Any:
    settings: dict[str, Any] = {
        "broker_url": "http://broker",
        "token": TOKEN,
        "poll_seconds": 600.0,
        "max_wait_seconds": 3600.0,
        "fetch": fetch,
        "invalidate_cache": _no_invalidate,
        "sleep": clock.sleep,
        "clock": clock.time,
        "log": logs.append,
    }
    settings.update(overrides)
    return suite.QuotaGuard(**settings)


def test_the_guard_pauses_until_the_reported_reset_then_clears() -> None:
    clock, logs = _start()
    fetch = _Script(_blocked_usage(120, clock), _clear_usage())

    assert _guard(clock, fetch, logs).wait(MODEL) is True

    # The reset is 120 s away; the guard wakes 5 s after it rather than after a full poll.
    assert clock.slept == [pytest.approx(125.0)]  # pyright: ignore[reportUnknownMemberType]
    assert [entry["quota"] for entry in logs] == ["paused", "cleared"]


def test_the_guard_gives_up_when_the_quota_outlasts_its_wait_budget() -> None:
    clock, logs = _start()
    fetch = _Script(_blocked_usage(10_000, clock))

    guard = _guard(clock, fetch, logs, poll_seconds=100.0, max_wait_seconds=300.0)

    assert guard.wait(MODEL) is False
    assert sum(clock.slept) == pytest.approx(300.0)  # pyright: ignore[reportUnknownMemberType]
    assert logs[-1]["quota"] == "wait_budget_exhausted"


def test_an_unreachable_broker_is_waited_out_and_never_read_as_headroom() -> None:
    clock, logs = _start()
    failure = suite.BrokerError("broker GET /v1/usage failed: URLError")
    fetch = _Script(failure, failure, _clear_usage())

    assert _guard(clock, fetch, logs, poll_seconds=30.0).wait(MODEL) is True

    assert [entry["quota"] for entry in logs] == ["broker_unreachable"] * 2 + ["cleared"]
    assert TOKEN not in json.dumps(logs)


def test_a_broker_that_never_answers_ends_in_a_pause_not_a_free_pass() -> None:
    clock, logs = _start()
    fetch = _Script(suite.BrokerError("broker GET /v1/usage failed: URLError"))

    assert (
        _guard(clock, fetch, logs, poll_seconds=100.0, max_wait_seconds=250.0).wait(MODEL) is False
    )


def test_unknown_headroom_proceeds_and_is_reported_once_per_provider() -> None:
    clock, logs = _start()
    fetch = _Script({"reports": []})
    guard = _guard(clock, fetch, logs)

    assert guard.wait(MODEL) and guard.wait("anthropic/claude-opus-5-5")

    assert [(entry["quota"], entry["provider"]) for entry in logs] == [("unknown", "anthropic")]


def test_a_cooling_down_pause_comes_before_the_first_read() -> None:
    clock, logs = _start()
    fetch = _Script(_clear_usage())

    assert _guard(clock, fetch, logs).wait(MODEL, pause_first=300.0) is True

    assert clock.slept == [300.0]
    assert logs[0]["quota"] == "cooling_down"


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError("http://b/v1/usage", 401, "no", Message(), None),
        urllib.error.URLError("x"),
    ],
)
def test_broker_errors_name_the_endpoint_and_never_the_token(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def refuse(request: urllib.request.Request, timeout: float) -> Any:
        raise failure

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    with pytest.raises(suite.BrokerError) as caught:
        suite.fetch_usage("http://b", TOKEN)

    assert "/v1/usage" in str(caught.value)
    assert TOKEN not in str(caught.value)


# --- one cell with quota retries --------------------------------------------------------------


def _key() -> CellKey:
    return CellKey("t1", MODEL, SweAbArm.A0, 1)


def _cell_spec(tmp_path: Path, key: CellKey) -> dict[str, Any]:
    return {
        "task_id": key.task_id,
        "repo": "org/repo",
        "model": key.model,
        "arm": key.arm.value,
        "repetition": key.repetition,
        "runtime": "docker",
        "output": str(tmp_path / "out" / key.relative_path),
        "work_dir": str(tmp_path / "work"),
        "omp_command": ["omp"],
        "profile_dir": str(tmp_path / "profile"),
        "prior_blocked_attempts": 0,
    }


class _Cells:
    """A fake cell runner: each invocation records an artifact with the next scripted outcome."""

    def __init__(self, *outcomes: str) -> None:
        self.outcomes = list(outcomes)
        self.specs: list[dict[str, Any]] = []

    def __call__(self, spec: dict[str, Any], env: Any) -> None:
        self.specs.append(dict(spec))
        outcome = self.outcomes.pop(0)
        phase = {"blocked_first": "before_first_tool_call", "blocked_mid": "mid_run"}.get(outcome)
        cell = cell_runner.failed_cell(
            cell_runner.CellSpec(spec),
            FailureReason.QUOTA_BLOCK if phase else FailureReason.PROVIDER_ERROR,
            "usage limit" if phase else "500",
            quota=QuotaEvidence(
                block_phase=phase,  # pyright: ignore[reportArgumentType]
                prior_blocked_attempts=spec["prior_blocked_attempts"],
            ),
        ).model_copy(
            update={"usage": Usage(input=1, output=1, cache_read=0, cache_write=0, cost_usd=2.0)}
        )
        output = Path(spec["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(cell.model_dump_json(), encoding="utf-8")


def _run_one(
    tmp_path: Path,
    cells: _Cells,
    guard: Any,
    logs: list[dict[str, Any]],
    *,
    retries: int = 2,
) -> float:
    key = _key()
    return suite._run_one(  # pyright: ignore[reportPrivateUsage]
        _cell_spec(tmp_path, key),
        key,
        tmp_path / "out",
        guard=guard,
        env=None,
        quota_retries=retries,
        invoke=cells,
        log=logs.append,
    )


def test_a_quota_blocked_cell_is_filed_and_rerun_after_a_cooling_down_pause(
    tmp_path: Path,
) -> None:
    clock, logs = _start()
    cells = _Cells("blocked_first", "provider_error")
    guard = _guard(clock, _Script(_clear_usage()), logs)

    total = _run_one(tmp_path, cells, guard, logs)

    filed = tmp_path / "out" / quota_blocked_relative_path(_key(), 1)
    assert load_cell(filed).quota.block_phase == "before_first_tool_call"
    final = load_cell(tmp_path / "out" / _key().relative_path)
    assert final.failure_reason is FailureReason.PROVIDER_ERROR
    assert final.quota.prior_blocked_attempts == 1
    assert [spec["prior_blocked_attempts"] for spec in cells.specs] == [0, 1]
    assert clock.slept == [guard.min_pause_seconds]
    assert total == 4.0  # the blocked attempt's tokens still count toward the ceiling


def test_a_mid_run_block_is_logged_distinctly_from_a_task_failure(tmp_path: Path) -> None:
    logs: list[dict[str, Any]] = []
    cells = _Cells("blocked_mid", "provider_error")

    _run_one(tmp_path, cells, _guard(_Clock(), _Script(_clear_usage()), logs), logs)

    cell_lines = [line for line in logs if "reason" in line]
    assert [(line["reason"], line.get("quota_phase")) for line in cell_lines] == [
        ("quota_block", "mid_run"),
        ("provider_error", None),
    ]


def test_past_the_retry_budget_the_blocked_cell_stays_as_the_record(tmp_path: Path) -> None:
    logs: list[dict[str, Any]] = []
    cells = _Cells("blocked_mid", "blocked_mid")

    total = _run_one(
        tmp_path, cells, _guard(_Clock(), _Script(_clear_usage()), logs), logs, retries=1
    )

    assert load_cell(tmp_path / "out" / _key().relative_path).failure_reason is (
        FailureReason.QUOTA_BLOCK
    )
    assert (tmp_path / "out" / quota_blocked_relative_path(_key(), 1)).exists()
    assert total == 4.0
    assert any(line.get("quota") == "retry_budget_exhausted" for line in logs)


def test_without_a_guard_a_blocked_cell_is_recorded_and_not_rerun(tmp_path: Path) -> None:
    cells = _Cells("blocked_first")

    _run_one(tmp_path, cells, None, [])

    assert len(cells.specs) == 1


def test_a_quota_that_does_not_clear_stops_the_cell_before_it_starts(tmp_path: Path) -> None:
    clock, logs = _start()
    cells = _Cells("provider_error")
    guard = _guard(
        clock,
        _Script(_blocked_usage(99_999, clock)),
        logs,
        poll_seconds=100.0,
        max_wait_seconds=200.0,
    )

    with pytest.raises(suite.QuotaPauseError):
        _run_one(tmp_path, cells, guard, logs)

    assert cells.specs == []


# --- the suite: exit status, preflight, resume ------------------------------------------------


def _suite_args(
    tmp_path: Path, *extra: str, profile: Path | None = None, repetitions: int = 1
) -> list[str]:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "name": "p",
                "tasks": [{"task_id": "t1", "repo": "o/r"}, {"task_id": "t2", "repo": "o/r"}],
                "models": [MODEL],
                "repetitions": {"A0": repetitions},
                "cost_ceiling_usd": 100.0,
            }
        ),
        encoding="utf-8",
    )
    token = tmp_path / "auth-broker.token"
    token.write_text(TOKEN + "\n", encoding="utf-8")
    profile = profile or tmp_path / "profile"
    profile.mkdir(exist_ok=True)
    return [
        "run", "--plan", str(plan), "--runtime", "docker", "--output", str(tmp_path / "out"),
        "--work-root", str(tmp_path / "work"), "--profile-dir", str(profile),
        "--omp-dir", "/omp", "--archex-wheel", "/w.whl", "--uv-binary", "/uv",
        "--broker-url", "http://127.0.0.1:8765", "--broker-token-file", str(token), *extra,
    ]  # fmt: skip


def _no_broker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in BROKER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_a_paused_run_exits_4_and_names_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_broker_env(monkeypatch)
    seen: list[str] = []

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> float:
        seen.append(key.task_id)
        if key.task_id == "t2":
            raise suite.QuotaPauseError(
                "anthropic/claude-sonnet-5-5: subscription quota did not clear"
            )
        return 1.5

    monkeypatch.setattr(suite, "_run_one", fake)

    assert suite.main(_suite_args(tmp_path)) == suite.QUOTA_EXIT

    assert seen == ["t1", "t2"]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["aborted"] == "quota"
    assert TOKEN not in json.dumps(events)


def test_cells_get_the_container_broker_url_and_token_and_a_recorded_name_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_broker_env(monkeypatch)
    envs: list[dict[str, str]] = []
    specs: list[dict[str, Any]] = []

    def fake(
        spec: dict[str, Any], key: CellKey, *args: Any, env: dict[str, str], **kwargs: Any
    ) -> float:
        envs.append(env)
        specs.append(spec)
        return 0.0

    monkeypatch.setattr(suite, "_run_one", fake)

    assert suite.main(_suite_args(tmp_path)) == 0

    assert envs[0]["OMP_AUTH_BROKER_URL"] == "http://host.docker.internal:8765"
    assert envs[0]["OMP_AUTH_BROKER_TOKEN"] == TOKEN
    assert specs[0]["broker_url"] == "http://host.docker.internal:8765"
    assert TOKEN not in json.dumps(specs)
    banner = json.loads(capsys.readouterr().out.splitlines()[0])
    assert banner["agent_env_names"] == list(BROKER_ENV_NAMES)
    assert TOKEN not in json.dumps(banner)


def _record_prunes(monkeypatch: pytest.MonkeyPatch, pause_at: tuple[str, int] | None) -> list[str]:
    events: list[str] = []

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> float:
        events.append(f"{key.task_id}#{key.repetition}")
        if (key.task_id, key.repetition) == pause_at:
            raise suite.QuotaPauseError("quota did not clear")
        return 0.0

    def prune(image: str) -> None:
        events.append(f"prune {image}")

    monkeypatch.setattr(suite, "_run_one", fake)
    monkeypatch.setattr(suite, "_prune_image", prune)
    return events


def test_pruning_removes_a_task_image_only_after_the_last_cell_of_that_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_broker_env(monkeypatch)
    events = _record_prunes(monkeypatch, None)

    assert suite.main(_suite_args(tmp_path, "--prune-images", repetitions=2)) == 0

    image = "ghcr.io/scaleapi/swe-bench_pro-v2:"
    assert events == [
        "t1#1",
        "t1#2",
        f"prune {image}t1",
        "t2#1",
        "t2#2",
        f"prune {image}t2",
    ]


def test_pruning_keeps_the_image_of_a_task_whose_cell_did_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_broker_env(monkeypatch)
    events = _record_prunes(monkeypatch, ("t1", 2))

    assert suite.main(_suite_args(tmp_path, "--prune-images", repetitions=2)) == suite.QUOTA_EXIT

    assert not any(event.startswith("prune") for event in events)


def test_images_are_kept_unless_pruning_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_broker_env(monkeypatch)
    events = _record_prunes(monkeypatch, None)

    assert suite.main(_suite_args(tmp_path)) == 0

    assert not any(event.startswith("prune") for event in events)


def test_the_docker_runtime_needs_a_broker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_broker_env(monkeypatch)
    argv = _suite_args(tmp_path)
    at = argv.index("--broker-url")

    with pytest.raises(SystemExit, match="--broker-url is required"):
        suite.main([*argv[:at], *argv[at + 2 :]])


def test_a_profile_carrying_a_login_vault_is_refused_before_any_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_broker_env(monkeypatch)
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "agent.db").write_text("")

    with pytest.raises(SystemExit, match="credential stores"):
        suite.main(_suite_args(tmp_path, profile=profile))


def test_resuming_files_an_unfinished_quota_block_and_reruns_that_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_broker_env(monkeypatch)
    key = CellKey("t2", MODEL, SweAbArm.A0, 1)
    done = _Cells("provider_error", "blocked_mid")
    done(_cell_spec(tmp_path, CellKey("t1", MODEL, SweAbArm.A0, 1)), None)
    done(_cell_spec(tmp_path, key), None)
    ran: list[tuple[str, int]] = []

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> float:
        ran.append((key.task_id, spec["prior_blocked_attempts"]))
        return 0.0

    monkeypatch.setattr(suite, "_run_one", fake)

    assert suite.main(_suite_args(tmp_path)) == 0

    assert ran == [("t2", 1)]
    assert (tmp_path / "out" / quota_blocked_relative_path(key, 1)).exists()
    assert not (tmp_path / "out" / key.relative_path).exists()


# --- credentials reach the container only as the two broker variables -------------------------


def test_the_broker_token_travels_in_the_docker_client_environment_not_its_argv() -> None:
    args, client_env = cell_runner.docker_exec_env(
        {
            "OMP_AUTH_BROKER_URL": "http://host.docker.internal:8765",
            "OMP_AUTH_BROKER_TOKEN": TOKEN,
            "HOME": "/root",
        }
    )

    assert args == ["-e", "OMP_AUTH_BROKER_URL", "-e", "OMP_AUTH_BROKER_TOKEN", "-e", "HOME=/root"]
    assert TOKEN not in " ".join(args)
    assert client_env["OMP_AUTH_BROKER_TOKEN"] == TOKEN


def _docker_spec(tmp_path: Path, **overrides: Any) -> Any:
    spec = _cell_spec(tmp_path, _key())
    spec.update(overrides)
    return cell_runner.CellSpec(spec)


def test_an_agent_container_gets_the_allow_list_and_none_of_the_host_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "OMP_AUTH_BROKER_URL": "http://host.docker.internal:8765",
        "OMP_AUTH_BROKER_TOKEN": TOKEN,
        "ANTHROPIC_API_KEY": "decoy",
        "OPENAI_API_KEY": "decoy",
        "AWS_SECRET_ACCESS_KEY": "decoy",
        "GITHUB_TOKEN": "decoy",
    }.items():
        monkeypatch.setenv(name, value)
    runtime = SimpleNamespace(home="/root", out="/out", repo="/app", image_path="/usr/bin:/bin")

    env = cell_runner._agent_env(  # pyright: ignore[reportPrivateUsage]
        _docker_spec(tmp_path),
        runtime,
        cell_runner._Setup(),  # pyright: ignore[reportArgumentType, reportPrivateUsage]
    )

    assert set(env) == {
        "HOME",
        "PATH",
        "ARCHEX_ANNOTATION_LEDGER",
        "ARCHEX_HOOK_DIAGNOSTICS_LOG",
        *BROKER_ENV_NAMES,
    }


def test_the_local_rehearsal_runtime_carries_no_broker_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_AUTH_BROKER_TOKEN", TOKEN)

    assert cell_runner.broker_env(_docker_spec(tmp_path, runtime="local")) == {}


def test_a_cell_records_variable_names_and_emulation_never_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_AUTH_BROKER_URL", "http://host.docker.internal:8765")
    monkeypatch.setenv("OMP_AUTH_BROKER_TOKEN", TOKEN)
    monkeypatch.setattr(cell_runner.platform, "machine", lambda: "arm64")

    docker = cell_runner.failed_cell(_docker_spec(tmp_path), FailureReason.HARNESS_ERROR, "x")
    local = cell_runner.failed_cell(
        _docker_spec(tmp_path, runtime="local"), FailureReason.HARNESS_ERROR, "x"
    )

    assert docker.credential_env_names == sorted(BROKER_ENV_NAMES)
    assert docker.emulated is True
    assert TOKEN not in docker.model_dump_json()
    assert (local.credential_env_names, local.emulated) == ([], False)


class _FakeRuntime:
    repo = "/app"
    home = "/root"
    out = "/out"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[str]] = []

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        stdout = self.replies.pop(0) if self.replies else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    def put(self, source: Path, target: str) -> None:
        self.calls.append(["put", target])


def test_a_profile_with_a_login_vault_never_reaches_a_container(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "agent.db").write_text("")
    runtime = _FakeRuntime("abc123")

    with pytest.raises(ValueError, match="credential stores"):
        cell_runner._setup(  # pyright: ignore[reportPrivateUsage]
            _docker_spec(tmp_path, profile_dir=str(profile), prompt_file=str(profile / "p.md")),
            runtime,  # pyright: ignore[reportArgumentType]
        )

    assert ["put", "/root/.omp/profiles/swebench/agent"] not in runtime.calls


# --- the annotate pre-warm --------------------------------------------------------------------


def test_the_prewarm_runs_the_annotate_entry_on_a_real_search_hit_without_logging() -> None:
    reply = json.dumps({"eligible": True, "annotated": False, "reason": "all_units_visible"})
    runtime = _FakeRuntime("a.py:1:import os", reply)

    seconds = cell_runner._prewarm_annotate(runtime, "/opt/archex/venv/bin/python")  # pyright: ignore[reportArgumentType, reportPrivateUsage]

    command = runtime.calls[1][-1]
    assert "-m archex.integrations.annotate_hook" in command
    assert "ARCHEX_HOOK_DIAGNOSTICS_LOG=/dev/null" in command
    assert "a.py:1:import os" in command
    assert seconds >= 0.0


def test_a_prewarm_that_misses_the_search_path_fails_setup() -> None:
    runtime = _FakeRuntime(
        "a.py:1:x", json.dumps({"eligible": False, "reason": "unsupported_tool"})
    )

    with pytest.raises(RuntimeError, match="did not reach the search path"):
        cell_runner._prewarm_annotate(runtime, "python")  # pyright: ignore[reportArgumentType, reportPrivateUsage]


# --- Stage 0 -----------------------------------------------------------------------------------


def _catalog(*extra: dict[str, str]) -> list[dict[str, str]]:
    entries = [
        {"provider": model.split("/")[0], "id": model.split("/")[1], "selector": model}
        for model in MODELS
    ]
    return [*entries, *extra]


def _logins(tmp_path: Path, *providers: str) -> Path:
    path = tmp_path / "agent.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY, provider TEXT, "
            "credential_type TEXT, data TEXT, disabled_cause TEXT DEFAULT NULL)"
        )
        db.executemany(
            "INSERT INTO auth_credentials (provider, credential_type, data) "
            "VALUES (?, 'oauth', 'x')",
            [(provider,) for provider in providers],
        )
    return path


def _stub_omp_models(monkeypatch: pytest.MonkeyPatch, catalog: list[dict[str, str]]) -> None:
    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps({"models": catalog}), stderr=""
        )

    monkeypatch.setattr(stage0.subprocess, "run", fake)


def test_stage0_passes_when_every_model_routes_through_a_logged_in_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_omp_models(
        monkeypatch,
        _catalog(
            {
                "provider": "openrouter",
                "id": "openai/gpt-6-sol",
                "selector": "openrouter/openai/gpt-6-sol",
            }
        ),
    )

    check = stage0.model_ids_check(["omp"], _logins(tmp_path, "anthropic", "openai-codex"))

    assert check["status"] == "pass"
    assert check["routes"] == {model: model.split("/")[0] for model in MODELS}
    assert check["logins"]["openai-codex"]["enabled"] == 1


def test_stage0_fails_a_provider_without_an_enabled_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_omp_models(monkeypatch, _catalog())

    check = stage0.model_ids_check(["omp"], _logins(tmp_path, "anthropic"))

    assert check["status"] == "fail"
    assert "no enabled login for provider 'openai-codex'" in check["detail"]


def test_stage0_fails_a_model_that_is_absent_or_routed_off_its_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bedrock_only = [
        {
            "provider": "amazon-bedrock",
            "id": "global.anthropic.claude-sonnet-5-5",
            "selector": "amazon-bedrock/global.anthropic.claude-sonnet-5-5",
        },
        {"provider": "amazon-bedrock", "id": "gpt-6-sol", "selector": "openai-codex/gpt-6-sol"},
    ]
    _stub_omp_models(monkeypatch, bedrock_only)

    check = stage0.model_ids_check(["omp"], _logins(tmp_path, "anthropic", "openai-codex"))

    assert check["status"] == "fail"
    assert "anthropic/claude-sonnet-5-5 is not in this omp's model catalog" in check["detail"]
    assert "openai-codex/gpt-6-sol routes through 'amazon-bedrock'" in check["detail"]


def test_stage0_reports_an_unreadable_login_vault_as_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_omp_models(monkeypatch, _catalog())

    check = stage0.model_ids_check(["omp"], tmp_path / "absent.db")

    assert check["status"] == "fail"


class _Broker(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        if self.path == "/v1/healthz":
            body = b'{"ok":true,"version":"18.4.4"}'
        elif self.headers.get("authorization") == f"Bearer {TOKEN}":
            body = json.dumps(_clear_usage()).encode()
        else:
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def broker_url() -> Any:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Broker)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _broker_args(url: str | None, token_file: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        broker_url=url, broker_bind="127.0.0.1:0", broker_token_file=token_file
    )


def test_stage0_reads_the_brokers_health_and_the_usage_the_quota_guard_uses(
    tmp_path: Path, broker_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_broker_env(monkeypatch)
    token_file = tmp_path / "t"
    token_file.write_text(TOKEN)

    checks = {c["id"]: c for c in stage0.broker_checks(_broker_args(broker_url, token_file))}

    assert (checks["broker_healthy"]["status"], checks["broker_usage_readable"]["status"]) == (
        "pass",
        "pass",
    )
    assert checks["broker_usage_readable"]["quota"][MODELS[0]]["status"] == "clear"
    assert checks["broker_usage_readable"]["quota"][MODELS[2]]["status"] == "unknown"


def test_stage0_fails_the_usage_check_on_a_wrong_token_without_echoing_it(
    tmp_path: Path, broker_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_broker_env(monkeypatch)
    token_file = tmp_path / "t"
    token_file.write_text("wrong-token")

    checks = {c["id"]: c for c in stage0.broker_checks(_broker_args(broker_url, token_file))}

    assert checks["broker_usage_readable"]["status"] == "fail"
    assert "HTTP 401" in checks["broker_usage_readable"]["detail"]
    assert "wrong-token" not in json.dumps(checks)


def test_stage0_leaves_the_broker_checks_pending_until_one_is_running() -> None:
    checks = stage0.broker_checks(_broker_args(None))

    assert {c["status"] for c in checks} == {"requires_host"}


class _Docker:
    """A fake ``subprocess.run`` for the container probes."""

    def __init__(self, *, reachable: bool = True, env_names: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.reachable = reachable
        self.env_names = env_names or "HOME\nOMP_AUTH_BROKER_TOKEN\nOMP_AUTH_BROKER_URL\nPATH\n"

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        out, code = "", 0
        if "wget" in argv:
            out, code = ('{"ok":true,"version":"18.4.4"}', 0) if self.reachable else ("", 1)
        elif argv[-1].startswith("env |"):
            out = self.env_names
        return subprocess.CompletedProcess(argv, code, stdout=out, stderr="")


def test_a_container_probe_reaches_the_broker_and_sees_only_the_allow_list() -> None:
    docker = _Docker()

    check = stage0.broker_container_check(
        "http://host.docker.internal:8765", "127.0.0.1:8765", run=docker
    )

    assert check["status"] == "pass"
    assert check["bind"] == "127.0.0.1:8765"
    run_call = docker.calls[0]
    assert run_call[run_call.index("--add-host") + 1] == "host.docker.internal:host-gateway"
    assert docker.calls[-1][:3] == ["docker", "rm", "-f"]
    assert "stage0-probe" not in " ".join(" ".join(call) for call in docker.calls)


def test_a_container_probe_fails_when_the_broker_is_unreachable() -> None:
    check = stage0.broker_container_check(
        "http://host.docker.internal:1", None, run=_Docker(reachable=False)
    )

    assert (check["status"], check["reachable"]) == ("fail", False)


def test_a_container_probe_fails_when_a_host_variable_leaks_in() -> None:
    leaky = _Docker(
        env_names="HOME\nOMP_AUTH_BROKER_TOKEN\nOMP_AUTH_BROKER_URL\nSWE_AB_STAGE0_HOST_SENTINEL\n"
    )

    check = stage0.broker_container_check("http://host.docker.internal:8765", None, run=leaky)

    assert (check["status"], check["allow_list_only"]) == ("fail", False)


def _bun(default: int, baseline: int = 0) -> Any:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        code = default if "oven/bun:1-debian" in argv else baseline
        error = "Illegal instruction" if code else ""
        return subprocess.CompletedProcess(
            argv, code, stdout="1.3.0" if not code else "", stderr=error
        )

    return run


def test_bun_check_reports_which_build_runs_under_emulation() -> None:
    assert stage0.bun_emulation_check(run=_bun(0))["variant"] == "default"
    fallback = stage0.bun_emulation_check(run=_bun(132, 0))
    assert (fallback["status"], fallback["variant"]) == ("pass", "baseline")
    assert "Illegal instruction" in fallback["default_error"]
    assert stage0.bun_emulation_check(run=_bun(132, 132))["status"] == "fail"


def test_a_stopped_docker_daemon_leaves_every_container_check_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ok(check_id: str) -> dict[str, str]:
        return {"id": check_id, "status": "pass", "detail": ""}

    def omp_version(command: list[str]) -> dict[str, str]:
        return ok("omp_version")

    def model_routes(command: list[str], agent_db: Path) -> dict[str, str]:
        return ok("model_routes_and_logins")

    def no_stub_checks(command: list[str], work: Path) -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(stage0, "docker_running", lambda: False)
    monkeypatch.setattr(stage0, "omp_version_check", omp_version)
    monkeypatch.setattr(stage0, "model_ids_check", model_routes)
    monkeypatch.setattr(stage0, "stub_arm_checks", no_stub_checks)
    output = tmp_path / "stage0.json"

    code = stage0.main(["--output", str(output), "--host", "--agent-db", str(tmp_path / "x.db")])

    report = json.loads(output.read_text())
    pending = {c["id"]: c for c in report["checks"] if c["status"] == "requires_host"}
    assert code == 0
    assert report["gate"] == "pending_host"
    assert {
        "gold_empty_validity",
        "broker_reachable_from_container",
        "bun_runs_under_emulation",
        "emulated_wall_times",
        "one_real_cell_per_model",
    } <= set(pending)
    assert "not running" in pending["gold_empty_validity"]["detail"]


def test_docker_running_is_false_without_a_docker_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError

    monkeypatch.setattr(stage0.subprocess, "run", missing)

    assert stage0.docker_running() is False
