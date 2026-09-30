"""The pre-registered R3x analysis on synthetic, validator-accepted result directories."""

from __future__ import annotations

import importlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from archex.benchmark.swe_ab import (
    BASE_TOOLS,
    BROKER_ENV_NAMES,
    CHANNELS,
    OMP_VERSION,
    CellKey,
    SweAbArm,
    SweAbCell,
    SweAbPlan,
    quota_blocked_relative_path,
    tool_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

analysis: Any = importlib.import_module("swe_ab_analysis")

SONNET = "anthropic/claude-sonnet-5-5"
SOL = "openai-codex/gpt-6-sol"
TASKS = [(f"t{r}{i}", repo) for r, repo in enumerate(["org/a", "org/b", "org/c"]) for i in range(2)]
STAGE1 = {"A0": 2, "H": 1, "HC": 1, "C": 1}


def _plan(repetitions: dict[str, int]) -> SweAbPlan:
    return SweAbPlan.model_validate(
        {
            "name": "synthetic",
            "tasks": [{"task_id": t, "repo": r} for t, r in TASKS],
            "models": [SONNET, SOL],
            "repetitions": repetitions,
            "cost_ceiling_usd": 100.0,
        }
    )


def _cell(key: CellKey, repo: str, **spec: Any) -> dict[str, Any]:
    """A campaign-valid cell; ``spec`` sets tokens, outcome, ledger, and channel totals."""
    arm = key.arm
    compounded = dict.fromkeys(CHANNELS, 0)
    compounded["read"] = spec.get("oop", 0)
    compounded["search"] = spec.get("search", 0)
    failure = spec.get("failure")
    ledger = spec.get("ledger", (0, 0, 0))
    return {
        "task_id": key.task_id,
        "repo": repo,
        "model": key.model,
        "arm": arm.value,
        "repetition": key.repetition,
        "status": "failed" if failure else "ok",
        "failure_reason": failure,
        "omp_version": OMP_VERSION,
        "archex_version": "0.34.0" if arm.archex_installed else None,
        "archex_wheel_sha256": "w" if arm.archex_installed else None,
        "hook_module_sha256": "h" if arm.hook else None,
        "cli_guide_sha256": "g" if arm.cli else None,
        "image": f"img:{key.task_id}",
        "tool_fingerprint": tool_fingerprint(BASE_TOOLS),
        "provider": key.model.split("/", 1)[0],
        "provider_endpoint_overridden": False,
        "emulated": spec.get("emulated", True),
        "network": "bridge",
        "credential_env_names": sorted(BROKER_ENV_NAMES),
        "usage": {
            "input": spec.get("billed", 1000),
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "cost_usd": 0.0,
        },
        "requests": 1,
        "tool_calls": 0,
        "channel_tokens_once": dict.fromkeys(CHANNELS, 0),
        "channel_tokens_compounded": compounded,
        "out_of_patch_read_tokens_compounded": spec.get("oop", 0),
        "hook_ledger": {
            "results": ledger[0],
            "eligible": ledger[0],
            "annotated": ledger[1],
            "units": ledger[1],
            "tokens": 10 * ledger[1],
            "not_fresh_after_first_edit": ledger[2],
        }
        if arm.hook
        else None,
        "archex_cli_calls": spec.get("cli", 0),
        "isolation": {
            "tools_source": "declared",
            "tools_advertised": sorted(BASE_TOOLS),
            "system_prompt_checked": False,
            "compressor_marker_seen": False,
            "annotation_seen": False,
        },
        "localization": {"gold_files": []},
        "patch_sha256": "p",
        "patch_bytes": 0,
        "resolved": spec.get("resolved", not failure),
        "score_source": "pro_verifier",
        "wall_seconds": 1.0,
        "setup_seconds": 1.0,
        "quota": {
            "prior_blocked_attempts": spec.get("prior_blocked", 0),
            "block_phase": spec.get("block_phase"),
        },
    }


def _write(path: Path, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SweAbCell.model_validate(cell).model_dump_json(), encoding="utf-8")


def _stage(
    root: Path, repetitions: dict[str, int], spec: Callable[[CellKey, int], dict[str, Any]]
) -> tuple[Path, Path]:
    """Write every declared cell (``spec(key, task_index)`` shapes each) and the plan."""
    plan = _plan(repetitions)
    cells = root / "cells"
    repos = dict(TASKS)
    index = {task: i for i, (task, _) in enumerate(TASKS)}
    for key in plan.cells():
        _write(
            cells / key.relative_path,
            _cell(key, repos[key.task_id], **spec(key, index[key.task_id])),
        )
    plan_path = root / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    return plan_path, cells


def _a0(i: int) -> int:
    return 1000 + 100 * i


def _run(plan: Path, cells: Path, out: Path, *extra: str) -> dict[str, Any]:
    assert (
        analysis.main(["--plan", str(plan), "--input", str(cells), "--output", str(out), *extra])
        == 0
    )
    return json.loads(out.read_text(encoding="utf-8"))


def _expected_ratio(ratios: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(r) for r in ratios))


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)


def test_efficiency_is_the_geometric_mean_ratio_with_holm_verdicts(tmp_path: Path) -> None:
    jitter = [0.99, 1.0, 1.01, 0.98, 1.02, 1.0]

    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H:
            return {"billed": round(_a0(i) * 0.7 * jitter[i])}
        if key.arm is SweAbArm.HC:
            return {"billed": round(_a0(i) * (1 + (0.004 if i % 2 else -0.004)))}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    report = _run(plan, cells, tmp_path / "out.json")

    h = report["efficiency"][SONNET]["H"]
    expected = _expected_ratio([round(_a0(i) * 0.7 * jitter[i]) / _a0(i) for i in range(6)])
    assert _close(h["geometric_mean_ratio"], expected)
    assert h["pairs"] == 6 and h["repositories"] == 3
    assert h["verdict"] == "superior"
    assert report["efficiency"][SONNET]["HC"]["verdict"] == "equivalent"
    assert report["guardrail"]["H"]["verdict"] == "non_inferior"


def test_a_failed_cell_enters_at_its_tokens_and_scores_unresolved(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H and key.model == SONNET and i == 0:
            return {"billed": 5 * _a0(i), "failure": "timeout"}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    report = _run(plan, cells, tmp_path / "out.json")

    assert _close(
        report["efficiency"][SONNET]["H"]["geometric_mean_ratio"],
        _expected_ratio([5.0, 1, 1, 1, 1, 1]),
    )
    # One of six Sonnet pairs lost a solve; Sol lost none; the model is the stratum.
    assert _close(report["guardrail"]["H"]["difference"], -1 / 12)
    assert report["outcomes"][SONNET]["H"]["failures"] == {"timeout": 1}


def test_a_zero_token_pair_is_listed_not_logged_and_stays_in_the_guardrail(
    tmp_path: Path,
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H and key.model == SOL and i == 3:
            return {"billed": 0, "failure": "harness_error"}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    report = _run(plan, cells, tmp_path / "out.json")

    sol = report["efficiency"][SOL]["H"]
    assert sol["pairs"] == 5
    assert [z["task_id"] for z in sol["zero_token_pairs"]] == ["t11"]
    assert report["guardrail"]["H"]["pairs"] == 12
    assert report["guardrail"]["H"]["mcnemar"]["control_only_resolved"] == 1


def test_quota_blocked_attempts_are_disclosed_and_excluded_in_the_sensitivity_check(
    tmp_path: Path,
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        blocked = key.arm is SweAbArm.H and key.model == SONNET and i == 0
        return {"billed": _a0(i), "prior_blocked": 1 if blocked else 0}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    key = CellKey("t00", SONNET, SweAbArm.H, 1)
    attempt = _cell(key, "org/a", billed=400, failure="quota_block", block_phase="mid_run")
    _write(cells / quota_blocked_relative_path(key, 1), attempt)
    report = _run(plan, cells, tmp_path / "out.json")

    assert report["blocked_attempts"]["total"] == 1
    slot = report["blocked_attempts"]["by_model_and_arm"][SONNET]["H"]
    assert (slot["attempts"], slot["billed_tokens"], slot["phase"]) == (1, 400, {"mid_run": 1})
    sensitivity = report["sensitivity_excluding_blocked"]
    assert sensitivity["efficiency"][SONNET]["H"]["pairs"] == 5
    assert sensitivity["efficiency"][SONNET]["H"]["excluded_blocked_pairs"] == 1
    assert sensitivity["guardrail"]["H"]["pairs"] == 11
    assert report["efficiency"][SONNET]["H"]["pairs"] == 6


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"emulated": False}, "emulated and native cells are mixed"),
        ({"resolved": None}, "never scored"),
    ],
)
def test_the_analysis_refuses_what_the_protocol_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], override: dict[str, Any], message: str
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        odd = key.arm is SweAbArm.C and key.model == SOL and i == 5
        return {"billed": _a0(i), **(override if odd else {})}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    code = analysis.main(
        ["--plan", str(plan), "--input", str(cells), "--output", str(tmp_path / "o.json")]
    )

    assert code == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_stage1_gates_follow_the_kill_criteria(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        cell: dict[str, Any] = {"billed": 10**6}
        if key.arm is SweAbArm.A0:
            if key.repetition == 2:
                cell["billed"] = round(10**6 * math.exp(0.1 if i % 2 else -0.1))
                cell["resolved"] = i != 0
            lever = 0.25 if key.model == SONNET else 0.10
            cell.update(oop=round(cell["billed"] * lever), search=round(cell["billed"] * 0.1))
        if key.arm is SweAbArm.H:
            cell["ledger"] = (3, 0, 0) if (key.model, i) == (SOL, 5) else (10, 4, 2)
        if key.arm is SweAbArm.HC:
            cell["cli"] = 2 if (key.model, i) == (SONNET, 0) else 0
        return cell

    plan, cells = _stage(tmp_path, STAGE1, spec)
    gates = _run(plan, cells, tmp_path / "out.json")["gates"]

    assert _close(gates["headroom"][SONNET]["H"]["share"], 0.25)
    assert gates["headroom"][SONNET]["H"]["passed"] is True
    assert _close(gates["headroom"][SONNET]["HC"]["share"], 0.35)
    assert gates["headroom"][SOL]["H"]["passed"] is False
    assert _close(gates["hc_adoption"]["share"], 1 / 12)
    assert gates["hc_adoption"]["passed"] is False
    # 11 cells annotate 4 of 10 eligible (2 declined as stale after an edit); one annotates none.
    hook = gates["hook_activity"]
    assert _close(hook["share"], 44 / (113 - 22))
    assert hook["passed"] is False
    assert hook["zero_annotation_cells"] == [f"{SOL}:t21"]
    noise = gates["a0_noise"][SONNET]
    diffs = [math.log(round(10**6 * math.exp(0.1 if i % 2 else -0.1)) / 10**6) for i in range(6)]
    sd = statistics.stdev(diffs)
    normal = statistics.NormalDist()
    z = normal.inv_cdf(0.975) + normal.inv_cdf(0.8)
    assert _close(noise["sd"], sd)
    assert noise["tasks_for_power_at_sesoi"] == math.ceil((z * sd / -math.log(0.9)) ** 2)
    assert _close(noise["flip_rate"], 1 / 6)


def test_stage2_has_no_gates_and_carries_the_pilot_headroom_verdict(tmp_path: Path) -> None:
    plan, cells = _stage(tmp_path, {"A0": 1, "H": 1, "HC": 1}, lambda key, i: {"billed": _a0(i)})
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {"gates": {"headroom": {SONNET: {"H": {"passed": False}, "HC": {"passed": True}}}}}
        ),
        encoding="utf-8",
    )
    report = _run(plan, cells, tmp_path / "out.json", "--pilot-analysis", str(pilot))

    assert report["gates"] is None
    assert report["efficiency"][SONNET]["H"]["hypothesis"].startswith("mis-specified")
    assert report["efficiency"][SONNET]["HC"]["hypothesis"] == "tested"


def test_the_report_is_byte_identical_across_runs(tmp_path: Path) -> None:
    plan, cells = _stage(tmp_path, STAGE1, lambda key, i: {"billed": _a0(i) + len(key.arm)})
    _run(plan, cells, tmp_path / "one.json")
    _run(plan, cells, tmp_path / "two.json")

    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()


@pytest.mark.parametrize(
    ("p_values", "adjusted"),
    [
        ({"H": 0.01, "HC": 0.04}, {"H": 0.02, "HC": 0.04}),
        ({"H": 0.03, "HC": 0.02}, {"H": 0.04, "HC": 0.04}),
        ({"H": 0.6}, {"H": 0.6}),
    ],
)
def test_holm_adjustment(p_values: dict[str, float], adjusted: dict[str, float]) -> None:
    adjusted_now = analysis.holm(p_values)
    assert adjusted_now.keys() == adjusted.keys()
    assert all(_close(adjusted_now[k], v) for k, v in adjusted.items())


def test_mcnemar_pools_discordant_pairs_across_strata() -> None:
    result = analysis.mcnemar({SONNET: (5, 1), SOL: (3, 1)})

    assert (result["treatment_only_resolved"], result["control_only_resolved"]) == (8, 2)
    assert _close(result["chi_square"], 3.6)
    assert _close(result["p_exact_two_sided"], 2 * (1 + 10 + 45) / 1024)


@pytest.mark.parametrize(
    ("q", "df", "quantile"),
    [(0.05, 10, 3.940299), (0.2, 23, 17.186506), (0.95, 23, 35.172462), (0.01, 99, 69.22989)],
)
def test_chi_square_quantiles_match_reference_values(q: float, df: int, quantile: float) -> None:
    assert _close(analysis.chi2_ppf(q, df), quantile)
