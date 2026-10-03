"""Tests for the SWE A/B cell runner and suite: campaign key, blocks, credits, ceilings, attempts.

Everything here runs against fakes (no Docker, no model, no network), except the rehearsal
smoke test, which drives the pinned omp against the local stub and skips without it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
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
    plan_campaign,
    quota_blocked_relative_path,
    summarize_ledger,
    validate_swe_ab_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen-3.8-27b@high"
LABELS = ("qwen-3.8-27b@low", "qwen-3.8-27b@high", "gemma-4-26b-a4b-it@high")
KEY_NAME = "MUNA_ACCESS_KEY"
SECRET = "muna-secret-key-123"  # noqa: S105 - a fake, used to prove it never leaks
PINNED_OMP = next(
    (
        path
        for path in (Path("/tmp/omp-18.4.4/bin/omp"), Path.home() / "swe-ab/omp-host/bin/omp")
        if path.is_file()
    ),
    Path("/tmp/omp-18.4.4/bin/omp"),
)
MUNA_CAMPAIGN = "benchmarks/swe_ab/campaigns/muna.yml"


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
        "campaign": MUNA_CAMPAIGN,
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
) -> Any:
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
    assert (total.cost_usd, total.tokens) == (4.0, 4)  # the blocked attempt counts toward ceilings


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
    assert (total.cost_usd, total.tokens) == (6.0, 6)
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
    assert (stop.value.spend.cost_usd, stop.value.spend.tokens) == (2.0, 2)


# --- the suite: exit status, preflight, resume ------------------------------------------------


def _suite_args(
    tmp_path: Path,
    *extra: str,
    profile: Path | None = None,
    repetitions: int = 1,
    models: tuple[str, ...] = (MODEL,),
    runtime: str = "docker",
    env_file: Path | None = None,
    campaign: str = MUNA_CAMPAIGN,
    token_ceiling: int | None = None,
) -> list[str]:
    plan = tmp_path / "plan.json"
    body: dict[str, Any] = {
        "name": "p",
        "campaign": campaign,
        "tasks": [{"task_id": "t1", "repo": "o/r"}, {"task_id": "t2", "repo": "o/r"}],
        "models": list(models),
        "repetitions": {"A0": repetitions},
        "cost_ceiling_usd": 100.0,
    }
    if token_ceiling is not None:
        body["token_ceiling"] = token_ceiling
    plan.write_text(json.dumps(body), encoding="utf-8")
    profile = profile or tmp_path / "profile"
    profile.mkdir(exist_ok=True)
    return [
        "run", "--plan", str(plan), "--runtime", runtime, "--output", str(tmp_path / "out"),
        "--work-root", str(tmp_path / "work"), "--profile-dir", str(profile),
        "--omp-dir", "/omp", "--archex-wheel", "/w.whl", "--uv-binary", "/uv",
        "--env-file", str(env_file or tmp_path / "absent.env"), *extra,
    ]  # fmt: skip


def _recording(ran: list[CellKey]) -> Any:
    def fake(_spec: dict[str, Any], key: CellKey, *_args: Any, **_kwargs: Any) -> Any:
        ran.append(key)
        return suite.Spend()

    return fake


@pytest.fixture
def campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake key for the Muna campaign in the environment (its prices are all set)."""
    monkeypatch.setenv(KEY_NAME, SECRET)


def test_a_credit_stop_exits_5_after_the_running_cell_finishes_and_starts_nothing_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    order: list[str] = []
    credit_raised = threading.Event()

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> Any:
        order.append(f"start {key.repetition}")
        if key.repetition == 1:
            credit_raised.set()
            raise suite.CreditExhaustedError("credits exhausted", suite.Spend(1.5, 15))
        credit_raised.wait(5)
        time.sleep(0.1)
        order.append(f"finish {key.repetition}")
        return suite.Spend(0.5, 5)

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
    ) -> Any:
        envs.append(env)
        specs.append(spec)
        return suite.Spend()

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
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER=1\nexport {KEY_NAME}='from-file'\n", encoding="utf-8")
    envs: list[dict[str, str]] = []

    def fake(
        spec: dict[str, Any], key: CellKey, *args: Any, env: dict[str, str], **kwargs: Any
    ) -> Any:
        envs.append(env)
        return suite.Spend()

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

    assert suite.read_env_file_key(path, KEY_NAME) == expected


def test_a_missing_env_file_has_no_key(tmp_path: Path) -> None:
    assert suite.read_env_file_key(tmp_path / "absent.env", KEY_NAME) is None


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


FREE_KEY_NAME = "FREE_API_KEY"


@pytest.fixture
def free_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A repository root (the suite's) holding a campaign whose only model is priced at 0."""
    root = tmp_path / "repo"
    (root / "benchmarks/swe_ab/campaigns").mkdir(parents=True)
    (root / "benchmarks/swe_ab/free-models.yml").write_text(
        "providers:\n"
        "  freeprov:\n"
        "    baseUrl: https://free.example/v1\n"
        f"    apiKey: {FREE_KEY_NAME}\n"
        "    models:\n"
        "      - id: stealth/free\n"
        "        thinking: {mode: effort, efforts: [low, high]}\n"
        "        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}\n",
        encoding="utf-8",
    )
    path = "benchmarks/swe_ab/campaigns/free.yml"
    (root / path).write_text(
        "name: free\n"
        "provider_config: benchmarks/swe_ab/free-models.yml\n"
        "configurations:\n"
        "  - {label: free@high, model: stealth/free, thinking: high}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(suite, "REPO_ROOT", root)
    return path


@pytest.mark.parametrize("key_present", [False, True])
def test_a_docker_run_of_an_unpriced_campaign_is_refused_without_a_token_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_campaign: str, key_present: bool
) -> None:
    if key_present:
        monkeypatch.setenv(FREE_KEY_NAME, SECRET)
    else:
        monkeypatch.delenv(FREE_KEY_NAME, raising=False)  # the refusal does not need the key
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _recording(ran))

    with pytest.raises(SystemExit) as refused:
        suite.main(_suite_args(tmp_path, campaign=free_campaign, models=("free@high",)))

    assert "stealth/free: input" in str(refused.value)
    assert "token_ceiling" in str(refused.value)
    assert SECRET not in str(refused.value)
    assert ran == []


def test_an_unpriced_campaign_runs_with_a_token_ceiling_and_its_own_key_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    free_campaign: str,
) -> None:
    monkeypatch.setenv(FREE_KEY_NAME, SECRET)
    envs: list[dict[str, str]] = []
    specs: list[dict[str, Any]] = []

    def fake(
        spec: dict[str, Any], key: CellKey, *args: Any, env: dict[str, str], **kwargs: Any
    ) -> Any:
        envs.append(env)
        specs.append(spec)
        return suite.Spend()

    monkeypatch.setattr(suite, "_run_one", fake)

    code = suite.main(
        _suite_args(
            tmp_path, campaign=free_campaign, models=("free@high",), token_ceiling=1_000_000
        )
    )

    assert code == 0
    assert envs[0][FREE_KEY_NAME] == SECRET
    assert {spec["campaign"] for spec in specs} == {free_campaign}
    banner = json.loads(capsys.readouterr().out.splitlines()[0])
    assert (banner["campaign"], banner["provider"], banner["base_url"]) == (
        "free",
        "freeprov",
        "https://free.example/v1",
    )
    assert banner["agent_env_names"] == [FREE_KEY_NAME]
    assert SECRET not in json.dumps(banner)


def _spending(tokens: int, ran: list[CellKey]) -> Any:
    def fake(_spec: dict[str, Any], key: CellKey, *_args: Any, **_kwargs: Any) -> Any:
        ran.append(key)
        return suite.Spend(0.0, tokens)

    return fake


def test_the_token_ceiling_stops_the_suite_with_exit_3_before_the_next_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    campaign: None,
) -> None:
    del campaign
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _spending(400, ran))

    code = suite.main(_suite_args(tmp_path, repetitions=4, token_ceiling=1000))

    assert code == suite.COST_EXIT == 3
    assert len(ran) == 3  # 400, 800, then 1200 >= 1000 stops the fourth
    stop = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (stop["aborted"], stop["spent_tokens"], stop["ceiling_tokens"]) == (
        "token ceiling",
        1200,
        1000,
    )


def test_without_a_token_ceiling_a_priced_campaign_is_bounded_by_cost_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _spending(400, ran))

    assert suite.main(_suite_args(tmp_path, repetitions=4)) == 0
    assert len(ran) == 8


def test_tokens_of_cells_recorded_by_an_earlier_run_count_toward_the_token_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    _Cells("provider_error")(_cell_spec(tmp_path, _key("t1")), None)  # 2 billed tokens
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _spending(0, ran))

    assert suite.main(_suite_args(tmp_path, token_ceiling=2)) == suite.COST_EXIT
    assert ran == []


def test_a_blocked_attempt_filed_on_resume_counts_toward_the_token_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: None
) -> None:
    del campaign
    _Cells("credit")(_cell_spec(tmp_path, _key("t1")), None)  # an unfiled block: 2 billed tokens
    ran: list[CellKey] = []
    monkeypatch.setattr(suite, "_run_one", _spending(0, ran))

    assert suite.main(_suite_args(tmp_path, token_ceiling=2)) == suite.COST_EXIT
    assert ran == []
    assert (tmp_path / "out" / quota_blocked_relative_path(_key("t1"), 1)).exists()


def test_in_flight_cells_are_reserved_at_the_largest_recorded_cell() -> None:
    hit = suite._ceiling_hit  # pyright: ignore[reportPrivateUsage]
    spent, largest = suite.Spend(10.0, 500), suite.Spend(1.0, 200)

    assert hit(spent, largest, 2, 100.0, 1000) is None  # 500 + 2 * 200 < 1000
    assert hit(spent, largest, 3, 100.0, 1000) == "token ceiling"
    assert hit(spent, largest, 3, 12.0, 1000) == "cost ceiling"  # 10 + 3 * 1 >= 12
    assert hit(spent, largest, 3, 100.0, None) is None


def test_the_token_ceiling_flag_only_lowers_the_plans_value() -> None:
    lower = suite._token_ceiling  # pyright: ignore[reportPrivateUsage]
    plan = SimpleNamespace(token_ceiling=1000)

    assert lower(plan, None) == 1000
    assert lower(plan, 400) == 400
    assert lower(plan, 5000) == 1000
    with pytest.raises(SystemExit, match="plan sets none"):
        lower(SimpleNamespace(token_ceiling=None), 400)


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

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> Any:
        ran.append((key.task_id, spec["prior_blocked_attempts"]))
        return suite.Spend()

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

    def fake(spec: dict[str, Any], key: CellKey, *args: Any, **kwargs: Any) -> Any:
        events.append(f"{key.model} {key.task_id}")
        if (key.model, key.task_id) == credit_at:
            raise suite.CreditExhaustedError("credits exhausted", suite.Spend())
        return suite.Spend()

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
    args, client_env = cell_runner.docker_exec_env(
        {KEY_NAME: SECRET, "HOME": "/root"}, secret_names=[KEY_NAME]
    )

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
        "ARCHEX_HOOK_TIMEOUT_SECONDS",
        KEY_NAME,
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


OPENROUTER_CAMPAIGN = "benchmarks/swe_ab/campaigns/openrouter-space-bunny.yml"
OPENROUTER_MODEL = "space-bunny-alpha@high"
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"


def test_another_campaign_authenticates_with_its_own_key_variable_and_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OPENROUTER_KEY_NAME, SECRET)
    monkeypatch.setenv(KEY_NAME, "muna-decoy")
    spec = _docker_spec(tmp_path, campaign=OPENROUTER_CAMPAIGN, model=OPENROUTER_MODEL)
    runtime = SimpleNamespace(home="/root", out="/out", repo="/app", image_path="/usr/bin:/bin")

    env = cell_runner._agent_env(  # pyright: ignore[reportPrivateUsage]
        spec,
        runtime,
        cell_runner._Setup(),  # pyright: ignore[reportArgumentType, reportPrivateUsage]
    )
    cell = cell_runner.failed_cell(spec, FailureReason.HARNESS_ERROR, "x")

    assert cell_runner.credential_env(spec) == {OPENROUTER_KEY_NAME: SECRET}
    assert env[OPENROUTER_KEY_NAME] == SECRET
    assert KEY_NAME not in env
    assert cell.credential_env_names == [OPENROUTER_KEY_NAME]
    assert (cell.provider_base_url, cell.thinking) == ("https://openrouter.ai/api/v1", "high")


# --- an attempt reads only its own files ------------------------------------------------------


def _session_text(tokens: int, cost: float) -> str:
    message = {
        "role": "assistant",
        "provider": "muna",
        "model": "m",
        "usage": {
            "input": tokens,
            "output": 1,
            "cacheRead": 0,
            "cacheWrite": 0,
            "cost": {"total": cost},
        },
        "stopReason": "stop",
        "content": [{"type": "text", "text": "done"}],
    }
    return json.dumps({"type": "message", "message": message}) + "\n"


class _FakeContainer:
    """A docker container whose filesystem is a host directory and whose `get` copies like
    `docker cp`: a directory goes INTO an existing target as `target/<basename>`."""

    repo = "/app"
    home = "/root"
    out = "/out"
    image_path = "/usr/bin:/bin"

    def __init__(self, root: Path, *, tokens: int, cost: float, patch: str, ledger: bool) -> None:
        self.root = root
        self.session = _session_text(tokens, cost)
        self.patch = patch
        self.ledger = ledger

    def _host(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        stdout = ""
        if argv[0] == "git":
            verb = argv[3] if argv[1] == "-c" else argv[1]
            stdout = {"rev-parse": "base\n", "diff": self.patch}.get(verb, "")
        elif argv[0] == "test":
            return subprocess.CompletedProcess(
                argv, 0 if self._host(argv[2]).exists() else 1, stdout="", stderr=""
            )
        elif "--session-dir" in argv:
            session = self._host(argv[argv.index("--session-dir") + 1])
            session.mkdir(parents=True)
            (session / "omp.jsonl").write_text(self.session, encoding="utf-8")
            if self.ledger:
                self._host(f"{self.out}/annotation-ledger.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
            stdout = f"omp ran in {self.root.name}\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    def put(self, source: Path, target: str) -> None:
        host = self._host(target)
        host.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, host, dirs_exist_ok=True)
        else:
            shutil.copy2(source, host)

    def get(self, source: str, target: Path) -> None:
        origin = self._host(source)
        if origin.is_dir():
            nested = target / origin.name if target.exists() else target
            shutil.copytree(origin, nested, dirs_exist_ok=True)
        else:
            shutil.copy2(origin, target)

    def close(self) -> None:
        return


def test_a_rerun_of_a_cell_records_its_own_session_and_keeps_the_earlier_attempts_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "profile").mkdir()
    (tmp_path / "omp-campaign.yml").write_text("retry: {}\n", encoding="utf-8")
    (tmp_path / "prompt.md").write_text("fix it\n", encoding="utf-8")
    spec = _docker_spec(tmp_path, prompt_file=str(tmp_path / "prompt.md"))
    work = spec.work_dir
    attempts = [
        _FakeContainer(tmp_path / "c1", tokens=100, cost=1.0, patch="patch one\n", ledger=True),
        _FakeContainer(tmp_path / "c2", tokens=700, cost=7.0, patch="patch two\n", ledger=False),
        _FakeContainer(tmp_path / "c3", tokens=900, cost=9.0, patch="patch three\n", ledger=False),
    ]
    containers = iter(attempts)

    def docker_runtime(_spec: object, *, mounts: object) -> _FakeContainer:
        return next(containers)

    def scored(_spec: object, _patch: object) -> bool:
        return True

    monkeypatch.setattr(cell_runner, "DockerRuntime", docker_runtime)
    monkeypatch.setattr(cell_runner, "score_patch", scored)

    first = cell_runner.run_cell(spec)
    second = cell_runner.run_cell(spec)
    third = cell_runner.run_cell(spec)

    # Each cell records the usage of its own session, never an earlier attempt's.
    assert [c.usage.input for c in (first, second, third)] == [100, 700, 900]
    assert [c.usage.cost_usd for c in (first, second, third)] == [1.0, 7.0, 9.0]
    # The third attempt's files are the live ones: one session, one patch, no stale ledger.
    assert [p.name for p in (work / "session-1").iterdir()] == ["omp.jsonl"]
    assert '"input": 900' in (work / "session-1/omp.jsonl").read_text(encoding="utf-8")
    assert (work / "model.patch").read_text(encoding="utf-8") == "patch three\n"
    assert not (work / "annotation-ledger.jsonl").exists()
    assert (work / "omp-stdout-1.jsonl").read_text(encoding="utf-8") == "omp ran in c3\n"
    # The earlier attempts' raw evidence moved aside, not deleted.
    prior = work / "prior-attempts"
    assert sorted(p.name for p in prior.iterdir()) == ["1", "2"]
    assert (prior / "1/model.patch").read_text(encoding="utf-8") == "patch one\n"
    assert (prior / "1/annotation-ledger.jsonl").exists()
    assert '"input": 100' in (prior / "1/session-1/omp.jsonl").read_text(encoding="utf-8")
    assert (prior / "1/omp-stdout-1.jsonl").read_text(encoding="utf-8") == "omp ran in c1\n"
    assert (prior / "2/model.patch").read_text(encoding="utf-8") == "patch two\n"
    assert not (prior / "2/annotation-ledger.jsonl").exists()
    assert '"input": 700' in (prior / "2/session-1/omp.jsonl").read_text(encoding="utf-8")


def test_a_local_rerun_starts_in_an_empty_work_dir_and_keeps_the_earlier_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    spec = _docker_spec(tmp_path, runtime="local", local_repo=str(repo))
    old_session = spec.work_dir / "out/session-1/old.jsonl"
    old_session.parent.mkdir(parents=True)
    old_session.write_text("earlier attempt\n", encoding="utf-8")
    seen: dict[str, list[str]] = {}

    def run_cell_body(_spec: Any, rt: Any, _started: float) -> str:
        seen["out"] = sorted(p.name for p in Path(rt.out).iterdir())
        return "ran"

    monkeypatch.setattr(cell_runner, "_run_cell", run_cell_body)

    assert cell_runner.run_cell(spec) == "ran"

    assert seen["out"] == []
    kept = spec.work_dir / "prior-attempts/1/out/session-1/old.jsonl"
    assert kept.read_text(encoding="utf-8") == "earlier attempt\n"


def test_a_docker_get_never_copies_into_an_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object.__new__(cell_runner.DockerRuntime)
    runtime.name = "c"
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(cell_runner.subprocess, "run", fake_run)
    existing = tmp_path / "session-1"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        runtime.get("/out/session-1", existing)
    assert calls == []

    runtime.get("/out/session-1", tmp_path / "fresh")
    assert calls == [["docker", "cp", "c:/out/session-1", str(tmp_path / "fresh")]]


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
        validate_swe_ab_directory(
            out,
            load_plan(plan_path),
            plan_campaign(load_plan(plan_path), root=REPO_ROOT),
            require_complete=False,
        )


_M_SCRIPT = [
    {
        "tool": "mcp__archex_query_repo",
        "args": {"repo_url": ".", "question": "where is hash_password defined?"},
    },
    {"text": "Looked it up."},
]


@pytest.mark.skipif(
    not PINNED_OMP.is_file() or not (REPO_ROOT / OMP_CONFIG_PATH).is_file(),
    reason="needs the pinned omp 18.4.4 and the frozen omp-campaign.yml",
)
def test_an_m_cell_records_its_mcp_config_calls_and_channel_while_others_carry_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv(KEY_NAME, raising=False)
    plan_path = tmp_path / "plan.json"
    plan = json.loads((REPO_ROOT / "benchmarks/swe_ab/dry-run-plan.json").read_text())
    plan["repetitions"] = {"A0": 1, "M": 1}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    script = tmp_path / "script.json"
    script.write_text(json.dumps(_M_SCRIPT), encoding="utf-8")
    out = tmp_path / "out"
    # The A0 cell would fail on the unadvertised MCP call, so run the two arms separately.
    for arm in ("A0", "M"):
        plan["repetitions"] = {arm: 1}
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        argv = [
            "run", "--plan", str(plan_path), "--runtime", "local", "--output", str(out),
            "--work-root", str(tmp_path / f"work-{arm}"), "--omp-command", str(PINNED_OMP),
            "--stub-script",
            str(script if arm == "M" else REPO_ROOT / "benchmarks/swe_ab/stub-script.json"),
        ]  # fmt: skip
        assert suite.main(argv) == 0

    cells = {c.arm.value: c for c in (load_cell(p) for p in out.rglob("*.json"))}
    m, a0 = cells["M"], cells["A0"]
    assert m.status.value == "ok"
    config_path = next((tmp_path / "work-M").rglob("agent/mcp.json"))
    config = json.loads(config_path.read_text())
    assert config["mcpServers"]["archex"]["args"] == ["mcp"]
    assert Path(config["mcpServers"]["archex"]["command"]).is_absolute()
    assert list(config) == ["$schema", "mcpServers"]
    assert m.mcp_config_sha256 == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert m.archex_mcp_calls == 1
    assert m.archex_mcp_tools == {"mcp__archex_query_repo": 1}
    assert m.channel_tokens_once["archex-MCP"] > 0
    assert m.channel_tokens_compounded["archex-MCP"] > 0
    assert len(m.isolation.tools_advertised) == 9
    assert m.hook_ledger is None
    assert a0.status.value == "ok"
    assert a0.mcp_config_sha256 is None
    assert a0.archex_mcp_calls == 0
    assert not list((tmp_path / "work-A0").rglob("mcp.json"))


@pytest.mark.parametrize("arm", ["A0", "M"])
def test_a_profile_carrying_an_mcp_config_is_refused_for_every_arm(
    tmp_path: Path, arm: str
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "mcp.json").write_text('{"mcpServers": {"x": {"command": "x"}}}')
    runtime = _FakeRuntime("abc123")

    with pytest.raises(ValueError, match=r"mcp\.json"):
        cell_runner._setup(  # pyright: ignore[reportPrivateUsage]
            _docker_spec(
                tmp_path, arm=arm, profile_dir=str(profile), prompt_file=str(profile / "p.md")
            ),
            runtime,  # pyright: ignore[reportArgumentType]
        )

    assert ["put", "/root/.omp/profiles/swebench/agent"] not in runtime.calls


def _exchange(call_id: str, tool: str, request_index: int) -> Any:
    return SimpleNamespace(call_id=call_id, tool=tool, request_index=request_index)


def test_a_call_issued_after_the_first_edit_in_the_same_request_counts_as_after_it() -> None:
    exchanges = [
        _exchange("read", "read", 0),
        _exchange("edit", "edit", 1),
        _exchange("grep", "grep", 1),  # same request as the edit, issued after it
        _exchange("late", "grep", 2),
    ]
    rows = [
        {"toolCallId": "read", "eligible": True, "annotated": False, "reason": "index_not_fresh"},
        {"toolCallId": "grep", "eligible": True, "annotated": False, "reason": "index_not_fresh"},
    ]

    after = cell_runner.calls_after_first_edit(exchanges)
    summary = summarize_ledger(rows, after)

    assert after == {"grep", "late"}
    assert summary.not_fresh_after_first_edit == 1


def test_no_edit_means_no_call_is_after_the_first_edit() -> None:
    assert cell_runner.calls_after_first_edit([_exchange("a", "grep", 0)]) == set()


def test_the_write_tool_is_a_first_edit_too() -> None:
    exchanges = [_exchange("w", "write", 0), _exchange("g", "grep", 0)]

    assert cell_runner.calls_after_first_edit(exchanges) == {"g"}


def _pull_runner(pull_codes: list[int], calls: list[list[str]]) -> Any:
    """A fake `subprocess.run`: the image is absent, and each pull answers the next code."""
    codes = iter(pull_codes)

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        code = 1 if argv[1] == "image" else next(codes)
        return subprocess.CompletedProcess(argv, code, stdout="", stderr="401 Unauthorized")

    return run


def test_a_transient_pull_failure_is_retried_until_the_image_arrives() -> None:
    calls: list[list[str]] = []
    pauses: list[float] = []

    cell_runner.ensure_image("img", run=_pull_runner([1, 1, 0], calls), sleep=pauses.append)

    assert [argv[1] for argv in calls] == ["image", "pull", "pull", "pull"]
    assert pauses == [30.0, 60.0]


def test_a_pull_that_keeps_failing_raises_after_the_last_attempt() -> None:
    calls: list[list[str]] = []
    pauses: list[float] = []

    with pytest.raises(RuntimeError, match="4 attempts"):
        cell_runner.ensure_image("img", run=_pull_runner([1, 1, 1, 1], calls), sleep=pauses.append)

    assert sum(argv[1] == "pull" for argv in calls) == 4
