"""Tests for the SWE A/B cell runner and suite on Muna: campaign key, blocks, credits, scheduling.

Everything here runs against fakes (no Docker, no model, no network), except the rehearsal
smoke test, which drives the pinned omp against the local stub and skips without it.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

cell_runner: Any = importlib.import_module("run_swe_ab_cell")
suite: Any = importlib.import_module("run_swe_ab_suite")

from archex.benchmark.swe_ab import (  # noqa: E402
    CREDENTIAL_ENV_NAMES,
    OMP_CONFIG_PATH,
    CellKey,
    FailureReason,
    OmpRunEvents,
    QuotaEvidence,
    SweAbArm,
    SweAbError,
    Usage,
    load_cell,
    load_plan,
    quota_blocked_relative_path,
    validate_swe_ab_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen-3.8-27b@high"
LABELS = ("qwen-3.8-27b@low", "qwen-3.8-27b@high", "gemma-4-26b-a4b-it@high")
KEY_NAME = "MUNA_ACCESS_KEY"
SECRET = "muna-secret-key-123"  # noqa: S105 - a fake, used to prove it never leaks
PINNED_OMP = Path("/tmp/omp-18.4.4/bin/omp")
PROVIDER_CONFIG = REPO_ROOT / "benchmarks/swe_ab/muna-models.yml"


# --- one cell with block cooldown -------------------------------------------------------------


def _key(task_id: str = "t1", model: str = MODEL) -> CellKey:
    return CellKey(task_id, model, SweAbArm.A0, 1)


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
        "provider_config": str(PROVIDER_CONFIG),
        "omp_config": str(tmp_path / "omp-campaign.yml"),
        "prior_blocked_attempts": 0,
    }


_REASONS = {
    "blocked_first": (FailureReason.QUOTA_BLOCK, "before_first_tool_call"),
    "blocked_mid": (FailureReason.QUOTA_BLOCK, "mid_run"),
    "credit": (FailureReason.CREDIT_EXHAUSTED, "mid_run"),
}


class _Cells:
    """A fake cell runner: each invocation records an artifact with the next scripted outcome."""

    def __init__(self, *outcomes: str) -> None:
        self.outcomes = list(outcomes)
        self.specs: list[dict[str, Any]] = []

    def __call__(self, spec: dict[str, Any], env: Any) -> None:
        self.specs.append(dict(spec))
        outcome = self.outcomes.pop(0)
        reason, phase = _REASONS.get(outcome, (FailureReason.PROVIDER_ERROR, None))
        cell = cell_runner.failed_cell(
            cell_runner.CellSpec(spec),
            reason,
            outcome,
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
    logs: list[dict[str, Any]],
    sleeps: list[float],
    *,
    retries: int = 2,
    cooldown: float = 300.0,
) -> float:
    key = _key()
    return suite._run_one(  # pyright: ignore[reportPrivateUsage]
        _cell_spec(tmp_path, key),
        key,
        tmp_path / "out",
        env=None,
        quota_retries=retries,
        cooldown_seconds=cooldown,
        invoke=cells,
        sleep=sleeps.append,
        log=logs.append,
    )


def test_a_quota_blocked_cell_is_filed_and_rerun_after_the_cooldown(tmp_path: Path) -> None:
    cells = _Cells("blocked_first", "provider_error")
    sleeps: list[float] = []

    total = _run_one(tmp_path, cells, [], sleeps, cooldown=42.0)

    filed = tmp_path / "out" / quota_blocked_relative_path(_key(), 1)
    assert load_cell(filed).quota.block_phase == "before_first_tool_call"
    final = load_cell(tmp_path / "out" / _key().relative_path)
    assert final.failure_reason is FailureReason.PROVIDER_ERROR
    assert final.quota.prior_blocked_attempts == 1
    assert [spec["prior_blocked_attempts"] for spec in cells.specs] == [0, 1]
    assert sleeps == [42.0]
    assert total == 4.0  # the blocked attempt's tokens still count toward the ceiling


def test_each_rerun_waits_the_cooldown_up_to_the_retry_budget_and_the_block_then_stays(
    tmp_path: Path,
) -> None:
    cells = _Cells("blocked_mid", "blocked_mid", "blocked_mid")
    logs: list[dict[str, Any]] = []
    sleeps: list[float] = []

    total = _run_one(tmp_path, cells, logs, sleeps, retries=2, cooldown=7.0)

    assert sleeps == [7.0, 7.0]
    assert len(cells.specs) == 3
    assert load_cell(tmp_path / "out" / _key().relative_path).failure_reason is (
        FailureReason.QUOTA_BLOCK
    )
    for attempt in (1, 2):
        assert (tmp_path / "out" / quota_blocked_relative_path(_key(), attempt)).exists()
    assert total == 6.0
    assert any(line.get("quota") == "retry_budget_exhausted" for line in logs)


def test_a_mid_run_block_is_logged_distinctly_from_a_task_failure(tmp_path: Path) -> None:
    logs: list[dict[str, Any]] = []

    _run_one(tmp_path, _Cells("blocked_mid", "provider_error"), logs, [])

    cell_lines = [line for line in logs if "reason" in line]
    assert [(line["reason"], line.get("quota_phase")) for line in cell_lines] == [
        ("quota_block", "mid_run"),
        ("provider_error", None),
    ]


def test_credit_exhaustion_is_filed_never_rerun_and_stops_the_run(tmp_path: Path) -> None:
    cells = _Cells("credit", "provider_error")
    sleeps: list[float] = []

    with pytest.raises(suite.CreditExhaustedError) as stop:
        _run_one(tmp_path, cells, [], sleeps)

    filed = tmp_path / "out" / quota_blocked_relative_path(_key(), 1)
    assert load_cell(filed).failure_reason is FailureReason.CREDIT_EXHAUSTED
    assert not (tmp_path / "out" / _key().relative_path).exists()
    assert len(cells.specs) == 1
    assert sleeps == []
    assert stop.value.cost == 2.0


# --- the suite: exit status, preflight, resume ------------------------------------------------


def _suite_args(
    tmp_path: Path,
    *extra: str,
    profile: Path | None = None,
    repetitions: int = 1,
    models: tuple[str, ...] = (MODEL,),
    runtime: str = "docker",
    env_file: Path | None = None,
) -> list[str]:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "name": "p",
                "tasks": [{"task_id": "t1", "repo": "o/r"}, {"task_id": "t2", "repo": "o/r"}],
                "models": list(models),
                "repetitions": {"A0": repetitions},
                "cost_ceiling_usd": 100.0,
            }
        ),
        encoding="utf-8",
    )
    profile = profile or tmp_path / "profile"
    profile.mkdir(exist_ok=True)
    return [
        "run", "--plan", str(plan), "--runtime", runtime, "--output", str(tmp_path / "out"),
        "--work-root", str(tmp_path / "work"), "--profile-dir", str(profile),
        "--omp-dir", "/omp", "--archex-wheel", "/w.whl", "--uv-binary", "/uv",
        "--env-file", str(env_file or tmp_path / "absent.env"), *extra,
    ]  # fmt: skip


def _all_priced(_path: Path) -> list[str]:
    return []


def _recording(ran: list[CellKey]) -> Any:
    def fake(_spec: dict[str, Any], key: CellKey, *_args: Any, **_kwargs: Any) -> float:
        ran.append(key)
        return 0.0

    return fake


@pytest.fixture
def campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake campaign key in the environment and every model priced."""
    monkeypatch.setenv(KEY_NAME, SECRET)
    monkeypatch.setattr(suite, "unpriced_models", _all_priced)


def test_a_credit_stop_exits_5_after_the_running_cell_finishes_and_starts_nothing_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    order: list[str] = []
    credit_raised = threading.Event()

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> float:
        order.append(f"start {key.repetition}")
        if key.repetition == 1:
            credit_raised.set()
            raise suite.CreditExhaustedError("credits exhausted", 1.5)
        credit_raised.wait(5)
        time.sleep(0.1)
        order.append(f"finish {key.repetition}")
        return 0.5

    monkeypatch.setattr(suite, "_run_one", fake)

    assert suite.main(_suite_args(tmp_path, "--jobs", "2", repetitions=4)) == suite.CREDIT_EXIT

    assert order[:2] == ["start 1", "start 2"] or order[:2] == ["start 2", "start 1"]
    assert "finish 2" in order
    assert not any(event in order for event in ("start 3", "start 4"))


def test_cells_carry_the_key_in_their_environment_and_never_in_the_spec_or_banner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    campaign: None,
) -> None:
    del campaign
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

    assert envs[0][KEY_NAME] == SECRET
    assert SECRET not in json.dumps(specs)
    out = capsys.readouterr().out
    assert json.loads(out.splitlines()[0])["agent_env_names"] == [KEY_NAME]
    assert SECRET not in out


def test_a_key_from_the_env_file_reaches_the_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(KEY_NAME, raising=False)
    monkeypatch.setattr(suite, "unpriced_models", _all_priced)
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER=1\nexport {KEY_NAME}='from-file'\n", encoding="utf-8")
    envs: list[dict[str, str]] = []

    def fake(
        spec: dict[str, Any], key: CellKey, *args: Any, env: dict[str, str], **kwargs: Any
    ) -> float:
        envs.append(env)
        return 0.0

    monkeypatch.setattr(suite, "_run_one", fake)

    assert suite.main(_suite_args(tmp_path, env_file=env_file)) == 0

    assert envs[0][KEY_NAME] == "from-file"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"{KEY_NAME}=abc\n", "abc"),
        (f'{KEY_NAME}="quoted key"\n', "quoted key"),
        (f"{KEY_NAME}='single'\n", "single"),
        (f"export {KEY_NAME}=exported\n", "exported"),
        (f"{KEY_NAME}=plain # trailing comment\n", "plain"),
        (f"OTHER_KEY=nope\nXMUNA_ACCESS_KEY=nope\n{KEY_NAME}=right\n", "right"),
        ("OTHER_KEY=nope\nOPENAI_API_KEY=nope\n", None),
        (f"{KEY_NAME}=\n", None),
        (f"# {KEY_NAME}=commented\n", None),
    ],
)
def test_the_env_file_reader_takes_only_the_campaign_key_line(
    tmp_path: Path, text: str, expected: str | None
) -> None:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")

    assert suite.read_env_file_key(path) == expected


def test_a_missing_env_file_has_no_key(tmp_path: Path) -> None:
    assert suite.read_env_file_key(tmp_path / "absent.env") is None


@pytest.mark.parametrize("env_file_text", [None, "OTHER=1\n"])
def test_a_docker_run_without_a_key_anywhere_is_refused_before_any_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_file_text: str | None
) -> None:
    monkeypatch.delenv(KEY_NAME, raising=False)
    env_file = tmp_path / ".env"
    if env_file_text is not None:
        env_file.write_text(env_file_text, encoding="utf-8")
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _recording(ran))

    with pytest.raises(SystemExit, match=KEY_NAME):
        suite.main(_suite_args(tmp_path, env_file=env_file))

    assert ran == []
    assert not (tmp_path / "out").exists()


def test_a_docker_run_is_refused_while_prices_are_zero_and_never_prints_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The suite reads the frozen provider config from REPO_ROOT; point it at a zero-price copy.
    root = tmp_path / "repo"
    (root / "benchmarks/swe_ab").mkdir(parents=True)
    (root / "benchmarks/swe_ab/muna-models.yml").write_text(
        "providers:\n  muna:\n    models:\n"
        '      - id: "@qwen/qwen-3.8-27b"\n'
        "        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(suite, "REPO_ROOT", root)
    monkeypatch.setenv(KEY_NAME, SECRET)
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _recording(ran))

    with pytest.raises(SystemExit) as refused:
        suite.main(_suite_args(tmp_path))

    assert "@qwen/qwen-3.8-27b: input" in str(refused.value)
    assert SECRET not in str(refused.value)
    assert ran == []


def test_a_profile_carrying_a_login_vault_is_refused_before_any_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "agent.db").write_text("")
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _recording(ran))

    with pytest.raises(SystemExit, match="credential stores"):
        suite.main(_suite_args(tmp_path, profile=profile))

    assert ran == []


def test_resuming_files_an_unfinished_block_and_reruns_that_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    key = _key("t2")
    done = _Cells("provider_error", "credit")
    done(_cell_spec(tmp_path, _key("t1")), None)
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


# --- scheduling and image pruning -------------------------------------------------------------


def _record_runs(
    monkeypatch: pytest.MonkeyPatch, credit_at: tuple[str, str] | None = None
) -> list[str]:
    events: list[str] = []

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> float:
        events.append(f"{key.model} {key.task_id}")
        if (key.model, key.task_id) == credit_at:
            raise suite.CreditExhaustedError("credits exhausted", 0.0)
        return 0.0

    def prune(image: str, *args: Any) -> None:
        events.append(f"prune {image.rsplit(':', 1)[1]}")

    monkeypatch.setattr(suite, "_run_one", fake)
    monkeypatch.setattr(suite, "_prune_image", prune)
    return events


def test_cells_start_grouped_by_model_family_then_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    events = _record_runs(monkeypatch)

    assert suite.main(_suite_args(tmp_path, models=LABELS)) == 0

    assert events == [
        "gemma-4-26b-a4b-it@high t1",
        "gemma-4-26b-a4b-it@high t2",
        "qwen-3.8-27b@high t1",
        "qwen-3.8-27b@low t1",
        "qwen-3.8-27b@high t2",
        "qwen-3.8-27b@low t2",
    ]


def test_pruning_removes_a_task_image_once_its_cells_of_the_current_family_are_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    events = _record_runs(monkeypatch)

    assert suite.main(_suite_args(tmp_path, "--prune-images", models=LABELS)) == 0

    assert events == [
        "gemma-4-26b-a4b-it@high t1",
        "prune t1",
        "gemma-4-26b-a4b-it@high t2",
        "prune t2",
        "qwen-3.8-27b@high t1",
        "qwen-3.8-27b@low t1",
        "prune t1",
        "qwen-3.8-27b@high t2",
        "qwen-3.8-27b@low t2",
        "prune t2",
    ]


def test_pruning_keeps_the_image_of_a_task_whose_cell_did_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    events = _record_runs(monkeypatch, credit_at=("qwen-3.8-27b@low", "t1"))

    code = suite.main(_suite_args(tmp_path, "--prune-images", models=LABELS))

    assert code == suite.CREDIT_EXIT
    assert events.count("prune t1") == 1  # the gemma family's; qwen's never completed
    assert "qwen-3.8-27b@high t2" not in events


def test_images_are_kept_unless_pruning_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    events = _record_runs(monkeypatch)

    assert suite.main(_suite_args(tmp_path, models=LABELS)) == 0

    assert not any(event.startswith("prune") for event in events)


# --- credentials reach the container only as the campaign key ---------------------------------


def test_the_campaign_key_travels_in_the_docker_client_environment_not_its_argv() -> None:
    args, client_env = cell_runner.docker_exec_env({KEY_NAME: SECRET, "HOME": "/root"})

    assert args == ["-e", KEY_NAME, "-e", "HOME=/root"]
    assert SECRET not in " ".join(args)
    assert client_env[KEY_NAME] == SECRET


def _docker_spec(tmp_path: Path, **overrides: Any) -> Any:
    spec = _cell_spec(tmp_path, _key())
    spec.update(overrides)
    return cell_runner.CellSpec(spec)


def test_an_agent_container_gets_the_allow_list_and_none_of_the_host_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        KEY_NAME: SECRET,
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
        *CREDENTIAL_ENV_NAMES,
    }
    assert env[KEY_NAME] == SECRET


def test_the_local_rehearsal_runtime_carries_no_credentials_and_strips_a_stray_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_NAME, SECRET)
    spec = _docker_spec(tmp_path, runtime="local")
    runtime = SimpleNamespace(home="/h", out="/o", repo="/r")

    env = cell_runner._agent_env(  # pyright: ignore[reportPrivateUsage]
        spec,
        runtime,
        cell_runner._Setup(),  # pyright: ignore[reportArgumentType, reportPrivateUsage]
    )

    assert cell_runner.credential_env(spec) == {}
    assert KEY_NAME not in env
    assert SECRET not in json.dumps(env)


def test_a_cell_records_variable_names_and_emulation_never_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_NAME, SECRET)
    monkeypatch.setattr(cell_runner.platform, "machine", lambda: "arm64")

    docker = cell_runner.failed_cell(_docker_spec(tmp_path), FailureReason.HARNESS_ERROR, "x")
    local = cell_runner.failed_cell(
        _docker_spec(tmp_path, runtime="local"), FailureReason.HARNESS_ERROR, "x"
    )

    assert docker.credential_env_names == [KEY_NAME]
    assert docker.emulated is True
    assert SECRET not in docker.model_dump_json()
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


# --- block classification at the cell level ---------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("HTTP 402 insufficient credits", FailureReason.CREDIT_EXHAUSTED),
        ('{"error":{"code":"credits_required"}}', FailureReason.CREDIT_EXHAUSTED),
        (
            '429 {"error":{"code":"model_capacity_exhausted"}}',
            FailureReason.QUOTA_BLOCK,
        ),
        ("429 rate limit exceeded", FailureReason.QUOTA_BLOCK),
        ("500 internal error", None),
        (None, None),
    ],
)
def test_a_provider_error_is_classified_as_credit_exhaustion_a_quota_block_or_neither(
    message: str | None, expected: FailureReason | None
) -> None:
    assert cell_runner._block_reason(message) is expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("gave_up", "tool_calls", "reason", "phase"),
    [
        (
            "HTTP 402 insufficient credits",
            0,
            FailureReason.CREDIT_EXHAUSTED,
            "before_first_tool_call",
        ),
        ("HTTP 402 insufficient credits", 3, FailureReason.CREDIT_EXHAUSTED, "mid_run"),
        ("429 model_capacity_exhausted", 0, FailureReason.QUOTA_BLOCK, "before_first_tool_call"),
        ("429 model_capacity_exhausted", 2, FailureReason.QUOTA_BLOCK, "mid_run"),
    ],
)
def test_a_run_omp_gave_up_on_is_a_blocked_unscored_cell_with_its_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gave_up: str,
    tool_calls: int,
    reason: FailureReason,
    phase: str,
) -> None:
    spec = _docker_spec(tmp_path)

    def setup(_spec: Any, _rt: Any) -> Any:
        return cell_runner._Setup()  # pyright: ignore[reportPrivateUsage]

    def run_agent(*_args: Any, **_kwargs: Any) -> tuple[int, Path, str]:
        return 1, tmp_path, ""

    def events(_stdout: str) -> OmpRunEvents:
        return OmpRunEvents(gave_up=gave_up, tool_calls_started=tool_calls)

    def no_session(_dir: Any) -> None:
        return None

    monkeypatch.setattr(cell_runner, "_setup", setup)
    monkeypatch.setattr(cell_runner, "_run_agent", run_agent)
    monkeypatch.setattr(cell_runner, "_load_session", no_session)
    monkeypatch.setattr(cell_runner, "parse_omp_events", events)

    cell = cell_runner._run_cell(spec, _FakeRuntime(), 0.0)  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert cell.failure_reason is reason
    assert cell.quota.block_phase == phase
    assert cell.resolved is False
    assert cell.score_source == "failed_before_patch"


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


# --- the no-spend rehearsal -------------------------------------------------------------------


@pytest.mark.skipif(
    not PINNED_OMP.is_file() or not (REPO_ROOT / OMP_CONFIG_PATH).is_file(),
    reason="needs the pinned omp 18.4.4 and the frozen omp-campaign.yml",
)
def test_a_stub_rehearsal_ends_ok_and_the_validator_refuses_its_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv(KEY_NAME, raising=False)
    plan_path = tmp_path / "plan.json"
    plan = json.loads((REPO_ROOT / "benchmarks/swe_ab/dry-run-plan.json").read_text())
    plan["repetitions"] = {"A0": 1}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    out = tmp_path / "out"

    code = suite.main(
        [
            "run",
            "--plan",
            str(plan_path),
            "--runtime",
            "local",
            "--output",
            str(out),
            "--work-root",
            str(tmp_path / "work"),
            "--omp-command",
            str(PINNED_OMP),
            "--stub-script",
            str(REPO_ROOT / "benchmarks/swe_ab/stub-script.json"),
        ]  # fmt: skip
    )

    assert code == 0
    cells = [load_cell(path) for path in out.rglob("*.json")]
    assert len(cells) == 1
    cell = cells[0]
    assert cell.status.value == "ok"
    assert cell.provider_base_url is not None
    assert cell.provider_base_url.startswith("http://127.0.0.1:")
    assert cell.credential_env_names == []
    with pytest.raises(SweAbError, match="local stub"):
        validate_swe_ab_directory(out, load_plan(plan_path), require_complete=False)
