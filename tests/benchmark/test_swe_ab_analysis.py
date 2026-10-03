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
    CHANNELS,
    OMP_VERSION,
    PRIMARY_ARMS,
    CellKey,
    SweAbArm,
    SweAbCell,
    SweAbPlan,
    arm_tools,
    load_campaign,
    quota_blocked_relative_path,
    tool_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

analysis: Any = importlib.import_module("swe_ab_analysis")

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = load_campaign("benchmarks/swe_ab/campaigns/muna.yml", root=REPO_ROOT)
LOW = "qwen-3.8-27b@low"
GEMMA = "gemma-4-26b-a4b-it@high"
TASKS = [(f"t{r}{i}", repo) for r, repo in enumerate(["org/a", "org/b", "org/c"]) for i in range(2)]
STAGE1 = {"A0": 2, "H": 1, "HC": 1, "M": 1, "C": 1}
STAGE2 = {"A0": 1, "H": 1, "HC": 1, "M": 1}
NORMAL = statistics.NormalDist()


def _plan(repetitions: dict[str, int]) -> SweAbPlan:
    return SweAbPlan.model_validate(
        {
            "name": "synthetic",
            "tasks": [{"task_id": t, "repo": r} for t, r in TASKS],
            "models": [LOW, GEMMA],
            "campaign": CAMPAIGN.path,
            "repetitions": repetitions,
            "cost_ceiling_usd": 100.0,
        }
    )


def _cell(key: CellKey, repo: str, **spec: Any) -> dict[str, Any]:
    """A campaign-valid cell; ``spec`` sets tokens, outcome, ledger, calls, and channel totals."""
    arm = key.arm
    compounded = dict.fromkeys(CHANNELS, 0)
    compounded["read"] = spec.get("oop", 0)
    compounded["search"] = spec.get("search", 0)
    failure = spec.get("failure")
    ledger = spec.get("ledger", (0, 0, 0))
    mcp_calls = spec.get("mcp", 0)
    return {
        "artifact_version": 3,
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
        "mcp_config_sha256": "m" if arm.mcp else None,
        "image": f"img:{key.task_id}",
        "tool_fingerprint": tool_fingerprint(arm_tools(arm)),
        "thinking": CAMPAIGN.configuration(key.model).thinking,
        "provider": CAMPAIGN.provider,
        "provider_base_url": spec.get("base_url", CAMPAIGN.base_url),
        "provider_config_sha256": CAMPAIGN.provider_config_sha256,
        "omp_config_sha256": "o" * 64,
        "emulated": spec.get("emulated", True),
        "network": "bridge",
        "credential_env_names": [CAMPAIGN.credential_env],
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
            "fail_open": {"no_hits": spec["no_hits"]} if spec.get("no_hits") else {},
        }
        if arm.hook
        else None,
        "archex_cli_calls": spec.get("cli", 0),
        "archex_mcp_calls": mcp_calls,
        "archex_mcp_tools": {"mcp__archex_query_repo": mcp_calls} if mcp_calls else {},
        "isolation": {
            "tools_source": "declared",
            "tools_advertised": sorted(arm_tools(arm)),
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
        "hook_timeout_seconds": 5.0,
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


def _pairs_needed(arms: int) -> float:
    """n(m) from the pre-registration, re-derived here."""
    z_alpha = NORMAL.inv_cdf(1 - 0.05 / (2 * arms))
    z_beta = NORMAL.inv_cdf(0.80)
    return ((z_alpha * math.sqrt(0.15) + z_beta * math.sqrt(0.15 - 0.05**2)) / 0.05) ** 2


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

    h = report["efficiency"][LOW]["H"]
    expected = _expected_ratio([round(_a0(i) * 0.7 * jitter[i]) / _a0(i) for i in range(6)])
    assert _close(h["geometric_mean_ratio"], expected)
    assert h["pairs"] == 6 and h["repositories"] == 3
    assert h["verdict"] == "superior"
    assert report["efficiency"][LOW]["HC"]["verdict"] == "equivalent"
    assert report["efficiency"][LOW]["M"]["verdict"] == "equivalent"
    assert report["guardrail"]["H"]["verdict"] == "non_inferior"


def test_the_pilot_arm_c_is_reported_but_never_tested(tmp_path: Path) -> None:
    plan, cells = _stage(tmp_path, STAGE1, lambda key, i: {"billed": _a0(i)})
    report = _run(plan, cells, tmp_path / "out.json")

    assert report["outcomes"][LOW]["C"]["cells"] == 6
    for family in (report["efficiency"][LOW], report["guardrail"], report["completion"]):
        assert set(family) == {arm.value for arm in PRIMARY_ARMS}


def test_efficiency_reads_superior_worse_equivalent_or_inconclusive(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H:
            return {"billed": round(_a0(i) * 0.7)}
        if key.arm is SweAbArm.HC:
            return {"billed": round(_a0(i) * 1.3)}
        if key.arm is SweAbArm.M:
            return {"billed": round(_a0(i) * (1.003 if i % 2 else 0.997))}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE2, spec)
    efficiency = _run(plan, cells, tmp_path / "out.json")["efficiency"]

    for model in (LOW, GEMMA):
        assert efficiency[model]["H"]["verdict"] == "superior"
        assert efficiency[model]["HC"]["verdict"] == "worse"
        assert efficiency[model]["M"]["verdict"] == "equivalent"
    # Each direction is its own Holm family over the three primary arms.
    p_worse = {arm: efficiency[LOW][arm]["p_vs_worse"] for arm in ("H", "HC", "M")}
    p_superior = {arm: efficiency[LOW][arm]["p_vs_sesoi"] for arm in ("H", "HC", "M")}
    for arm in ("H", "HC", "M"):
        assert _close(efficiency[LOW][arm]["p_holm_worse"], analysis.holm(p_worse)[arm])
        assert _close(efficiency[LOW][arm]["p_holm"], analysis.holm(p_superior)[arm])
    assert _close(efficiency[LOW]["HC"]["p_vs_worse"], 1 / 10_001)


@pytest.mark.parametrize(
    ("factors", "reading"),
    [
        ([1.12] * 6, "worse"),
        # Inside the margin, however clearly above 1.
        ([1.08] * 6, "inconclusive"),
        # A geometric mean of 1.0996 with repositories on both sides of 1.10: no evidence.
        ([1.04, 1.04, 1.10, 1.10, 1.16, 1.16], "inconclusive"),
    ],
)
def test_worse_needs_the_ratio_clearly_above_the_margin(
    tmp_path: Path, factors: list[float], reading: str
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H:
            return {"billed": round(_a0(i) * factors[i])}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE2, spec)
    h = _run(plan, cells, tmp_path / "out.json")["efficiency"][LOW]["H"]

    assert h["verdict"] == reading
    if reading == "worse":
        assert h["p_holm_worse"] < 0.05
    else:
        assert h["p_holm_worse"] >= 0.05
        assert h["p_vs_worse"] > 0.05


def test_a_failed_cell_enters_at_its_tokens_and_scores_unresolved(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H and key.model == LOW and i == 0:
            return {"billed": 5 * _a0(i), "failure": "timeout"}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    report = _run(plan, cells, tmp_path / "out.json")

    assert _close(
        report["efficiency"][LOW]["H"]["geometric_mean_ratio"],
        _expected_ratio([5.0, 1, 1, 1, 1, 1]),
    )
    # One of six Sonnet pairs lost a solve; Sol lost none; the model is the stratum.
    assert _close(report["guardrail"]["H"]["difference"], -1 / 12)
    assert report["outcomes"][LOW]["H"]["failures"] == {"timeout": 1}


def test_a_zero_token_pair_is_listed_not_logged_and_stays_in_the_guardrail(
    tmp_path: Path,
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H and key.model == GEMMA and i == 3:
            return {"billed": 0, "failure": "harness_error"}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    report = _run(plan, cells, tmp_path / "out.json")

    sol = report["efficiency"][GEMMA]["H"]
    assert sol["pairs"] == 5
    assert [z["task_id"] for z in sol["zero_token_pairs"]] == ["t11"]
    assert report["guardrail"]["H"]["pairs"] == 12
    assert report["guardrail"]["H"]["mcnemar"]["control_only_resolved"] == 1
    assert report["completion"]["H"]["pairs"] == 12


def test_quota_blocked_attempts_are_disclosed_and_excluded_in_the_sensitivity_check(
    tmp_path: Path,
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        blocked = key.arm is SweAbArm.H and key.model == LOW and i == 0
        return {"billed": _a0(i), "prior_blocked": 1 if blocked else 0}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    key = CellKey("t00", LOW, SweAbArm.H, 1)
    attempt = _cell(key, "org/a", billed=400, failure="quota_block", block_phase="mid_run")
    _write(cells / quota_blocked_relative_path(key, 1), attempt)
    report = _run(plan, cells, tmp_path / "out.json")

    assert report["blocked_attempts"]["total"] == 1
    slot = report["blocked_attempts"]["by_model_and_arm"][LOW]["H"]
    assert (slot["attempts"], slot["billed_tokens"], slot["phase"]) == (1, 400, {"mid_run": 1})
    sensitivity = report["sensitivity_excluding_blocked"]
    assert sensitivity["efficiency"][LOW]["H"]["pairs"] == 5
    assert sensitivity["efficiency"][LOW]["H"]["excluded_blocked_pairs"] == 1
    assert sensitivity["guardrail"]["H"]["pairs"] == 11
    assert sensitivity["completion"]["H"]["pairs"] == 11
    assert report["efficiency"][LOW]["H"]["pairs"] == 6
    assert report["completion"]["H"]["pairs"] == 12


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"emulated": False}, "emulated and native cells are mixed"),
        ({"base_url": "http://127.0.0.1:9/v1"}, "a local stub or another endpoint"),
        ({"resolved": None}, "never scored"),
    ],
)
def test_the_analysis_refuses_what_the_protocol_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], override: dict[str, Any], message: str
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        odd = key.arm is SweAbArm.C and key.model == GEMMA and i == 5
        return {"billed": _a0(i), **(override if odd else {})}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    code = analysis.main(
        ["--plan", str(plan), "--input", str(cells), "--output", str(tmp_path / "o.json")]
    )

    assert code == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def _gate_spec(*, hc_adopts: bool, m_adopts: bool) -> Callable[[CellKey, int], dict[str, Any]]:
    """Stage 1 cells: A0 rep 2 jittered, lever shares 25% (LOW) / 10% (GEMMA), H hook ledger."""

    def spec(key: CellKey, i: int) -> dict[str, Any]:
        cell: dict[str, Any] = {"billed": 10**6}
        if key.arm is SweAbArm.A0:
            if key.repetition == 2:
                cell["billed"] = round(10**6 * math.exp(0.1 if i % 2 else -0.1))
                cell["resolved"] = i != 0
            lever = 0.25 if key.model == LOW else 0.10
            cell.update(oop=round(cell["billed"] * lever), search=round(cell["billed"] * 0.1))
        if key.arm is SweAbArm.H:
            lone = (key.model, i) == (GEMMA, 5)
            cell["ledger"] = (3, 0, 0) if lone else (10, 4, 2)
            cell["no_hits"] = 0 if lone else 1
        if key.arm is SweAbArm.HC:
            if hc_adopts:
                cell["cli"] = 1
            else:
                cell["cli"] = 2 if (key.model, i) == (LOW, 0) else 0
        if key.arm is SweAbArm.M:
            if m_adopts:
                cell["mcp"] = 1
            elif (key.model, i) in {(LOW, 0), (LOW, 1), (GEMMA, 0)}:
                cell["mcp"] = 1
                if key.model == GEMMA:
                    cell.update(failure="timeout")  # a failed cell with a call still counts
        return cell

    return spec


def test_stage1_gates_follow_the_kill_criteria(tmp_path: Path) -> None:
    plan, cells = _stage(tmp_path, STAGE1, _gate_spec(hc_adopts=False, m_adopts=False))
    gates = _run(plan, cells, tmp_path / "out.json")["gates"]

    assert _close(gates["headroom"][LOW]["H"]["share"], 0.25)
    assert gates["headroom"][LOW]["H"]["passed"] is True
    assert _close(gates["headroom"][LOW]["HC"]["share"], 0.35)
    assert _close(gates["headroom"][LOW]["M"]["share"], 0.35)
    assert gates["headroom"][LOW]["M"]["passed"] is True
    assert gates["headroom"][GEMMA]["H"]["passed"] is False
    assert abs(gates["headroom"][GEMMA]["M"]["share"] - 0.2) < 1e-5
    assert _close(gates["hc_adoption"]["share"], 1 / 12)
    assert gates["hc_adoption"]["passed"] is False
    # Three of twelve M cells call an archex MCP tool, one of them a failed cell: exactly 25%.
    mcp = gates["mcp_adoption"]
    assert (mcp["cells"], mcp["calls"]) == (12, 3)
    assert _close(mcp["share"], 0.25)
    assert mcp["passed"] is True
    assert _close(mcp["by_model"][LOW], 2 / 6) and _close(mcp["by_model"][GEMMA], 1 / 6)
    assert mcp["subcommands"] == {"mcp__archex_query_repo": 3}
    assert gates["c_adoption"]["calls"] == 0
    # Kill criterion 7: 11 cells annotate 4 of 10 eligible, each with 1 no-hit search and 2
    # declines as stale after an edit; one annotates none of its 3.
    hook = gates["hook_activity"]
    assert (hook["eligible"], hook["annotated"], hook["no_hits"]) == (113, 44, 11)
    assert hook["stale_after_first_edit"] == 22
    assert _close(hook["share"], 44 / (113 - 11 - 22))
    assert hook["passed"] is True
    assert hook["zero_annotation_cells"] == [f"{GEMMA}:t21"]


def test_stage1_sizes_stage2_from_the_primary_arms_that_clear_their_adoption_gate(
    tmp_path: Path,
) -> None:
    plan, cells = _stage(tmp_path, STAGE1, _gate_spec(hc_adopts=False, m_adopts=False))
    gates = _run(plan, cells, tmp_path / "out.json")["gates"]

    # HC fails its gate, H has none, M passes: m = 2; both configurations clear the floor: k = 2.
    size = gates["stage2_size"]
    assert size["primary_arms_kept"] == ["H", "M"]
    assert (size["passing_configurations"], size["estimable"]) == (2, True)
    assert _close(size["pairs_needed"], _pairs_needed(2))
    assert _close(size["target_pairs"], _pairs_needed(2))
    assert size["tasks"] == math.ceil(_pairs_needed(2) / 2)

    noise = gates["a0_noise"][LOW]
    diffs = [math.log(round(10**6 * math.exp(0.1 if i % 2 else -0.1)) / 10**6) for i in range(6)]
    sd = statistics.stdev(diffs)
    delta = -math.log(0.9)
    z = NORMAL.inv_cdf(1 - 0.05 / 2) + NORMAL.inv_cdf(0.8)
    assert _close(noise["sd"], sd)
    assert noise["tasks_for_power_at_sesoi"] == math.ceil((z * sd / delta) ** 2)
    assert _close(
        noise["min_detectable_ratio_at_stage2"], math.exp(-z * sd / math.sqrt(size["tasks"]))
    )
    assert _close(noise["flip_rate"], 1 / 6)


def test_efficiency_power_uses_the_holm_first_step_for_the_arms_kept(tmp_path: Path) -> None:
    plan, cells = _stage(tmp_path, STAGE1, _gate_spec(hc_adopts=True, m_adopts=True))
    gates = _run(plan, cells, tmp_path / "out.json")["gates"]

    size = gates["stage2_size"]
    assert size["primary_arms_kept"] == ["H", "HC", "M"]
    assert size["tasks"] == math.ceil(_pairs_needed(3) / 2)
    noise = gates["a0_noise"][LOW]
    diffs = [math.log(round(10**6 * math.exp(0.1 if i % 2 else -0.1)) / 10**6) for i in range(6)]
    sd = statistics.stdev(diffs)
    z = NORMAL.inv_cdf(1 - 0.05 / 3) + NORMAL.inv_cdf(0.8)
    assert noise["tasks_for_power_at_sesoi"] == math.ceil((z * sd / -math.log(0.9)) ** 2)
    assert _close(
        noise["power_at_stage2_tasks"],
        NORMAL.cdf(-math.log(0.9) * math.sqrt(size["tasks"]) / sd - NORMAL.inv_cdf(1 - 0.05 / 3)),
    )


def test_hook_activity_excludes_no_hit_searches_from_the_denominator(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is SweAbArm.H:
            # 10 eligible, 3 annotated, 4 searches with no hits, 1 stale decline after an edit.
            return {"billed": _a0(i), "ledger": (10, 3, 1), "no_hits": 4}
        return {"billed": _a0(i)}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    hook = _run(plan, cells, tmp_path / "out.json")["gates"]["hook_activity"]

    assert _close(hook["share"], 36 / (120 - 48 - 12))
    assert hook["passed"] is True
    assert hook["fail_open_by_reason"] == {"no_hits": 48}


@pytest.mark.parametrize(
    ("arms_kept", "passing", "configurations", "tasks"),
    [
        (PRIMARY_ARMS, 3, 3, 209),
        (PRIMARY_ARMS[:2], 3, 3, 190),
        (PRIMARY_ARMS[:1], 3, 3, 157),
        (PRIMARY_ARMS, 1, 3, 626),
        (PRIMARY_ARMS[:1], 7, 7, 100),  # 469 pairs / 7 = 67 tasks: the 100-task floor binds
    ],
)
def test_stage2_size_follows_the_pair_rule(
    arms_kept: tuple[SweAbArm, ...], passing: int, configurations: int, tasks: int
) -> None:
    size = analysis.stage2_size(passing, configurations, arms_kept)

    assert size["tasks"] == tasks
    assert size["estimable"] is True
    assert size["primary_arms_kept"] == [arm.value for arm in arms_kept]
    assert _close(size["pairs_needed"], _pairs_needed(len(arms_kept)))
    assert size["passing_configurations"] == passing


@pytest.mark.parametrize(("passing", "arms_kept"), [(0, PRIMARY_ARMS), (2, ())])
def test_stage2_has_no_size_without_a_passing_configuration_or_a_primary_arm(
    passing: int, arms_kept: tuple[SweAbArm, ...]
) -> None:
    size = analysis.stage2_size(passing, 3, arms_kept)

    assert size["tasks"] is None and size["estimable"] is False


def test_primary_arms_kept_drops_only_the_arms_that_failed_an_adoption_gate() -> None:
    stage = analysis.Stage(plan=_plan(STAGE1), cells={}, blocked=[], repo_of={}, coverage={})
    passed, failed = {"passed": True}, {"passed": False}

    def kept(**gates: dict[str, bool] | None) -> list[str]:
        return [arm.value for arm in analysis.primary_arms_kept(stage, gates)]

    assert kept(HC=passed, M=passed, C=failed) == ["H", "HC", "M"]
    assert kept(HC=failed, M=passed) == ["H", "M"]
    assert kept(HC=passed, M=failed) == ["H", "HC"]
    assert kept(HC=failed, M=failed) == ["H"]
    without_m = analysis.Stage(
        plan=_plan({"A0": 1, "H": 1, "HC": 1}), cells={}, blocked=[], repo_of={}, coverage={}
    )
    assert [a.value for a in analysis.primary_arms_kept(without_m, {"HC": passed})] == ["H", "HC"]


def test_stage2_has_no_gates_and_carries_the_pilot_headroom_verdict(tmp_path: Path) -> None:
    plan, cells = _stage(tmp_path, STAGE2, lambda key, i: {"billed": _a0(i)})
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "gates": {
                    "headroom": {
                        LOW: {
                            "H": {"passed": False},
                            "HC": {"passed": True},
                            "M": {"passed": False},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report = _run(plan, cells, tmp_path / "out.json", "--pilot-analysis", str(pilot))

    assert report["gates"] is None
    assert report["efficiency"][LOW]["H"]["hypothesis"].startswith("mis-specified")
    assert report["efficiency"][LOW]["HC"]["hypothesis"] == "tested"
    assert report["efficiency"][LOW]["M"]["hypothesis"].startswith("mis-specified")
    assert report["efficiency"][GEMMA]["M"]["hypothesis"] == "tested"


def test_solve_floor_counts_every_a0_cell_of_both_repetitions(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        if key.arm is not SweAbArm.A0:
            return {"billed": _a0(i)}
        # LOW solves 2 of 12 A0 cells, both only in repetition 2 (rep 1 alone: 0% would fail);
        # GEMMA solves 1 of 12, only in repetition 1 (rep 1 alone: 17% would pass).
        if key.model == LOW:
            solved = key.repetition == 2 and i < 2
        else:
            solved = key.repetition == 1 and i == 0
        return {"billed": _a0(i), "resolved": solved}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    gates = _run(plan, cells, tmp_path / "out.json")["gates"]

    assert gates["solve_floor"][LOW]["n"] == 12
    assert _close(gates["solve_floor"][LOW]["rate"], 2 / 12)
    assert gates["solve_floor"][LOW]["passed"] is True
    assert gates["solve_floor"][GEMMA]["n"] == 12
    assert _close(gates["solve_floor"][GEMMA]["rate"], 1 / 12)
    assert gates["solve_floor"][GEMMA]["passed"] is False
    # No HC or M cell called an archex tool: only H is kept (m = 1); one configuration passes.
    size = gates["stage2_size"]
    assert size["primary_arms_kept"] == ["H"]
    assert (size["passing_configurations"], size["estimable"]) == (1, True)
    assert size["tasks"] == math.ceil(_pairs_needed(1))


def test_stage1_with_no_configuration_above_the_floor_has_no_stage2_size(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        return {"billed": _a0(i) + 7 * key.repetition * i, "resolved": False}

    plan, cells = _stage(tmp_path, STAGE1, spec)
    gates = _run(plan, cells, tmp_path / "out.json")["gates"]

    assert gates["stage2_size"]["tasks"] is None
    assert gates["stage2_size"]["estimable"] is False
    assert gates["stage2_size"]["passing_configurations"] == 0
    assert "power_at_stage2_tasks" not in gates["a0_noise"][LOW]


def _floor_pilot(path: Path, passed: dict[str, bool]) -> Path:
    path.write_text(
        json.dumps({"gates": {"solve_floor": {m: {"passed": p} for m, p in passed.items()}}}),
        encoding="utf-8",
    )
    return path


def test_stage2_guardrail_pools_only_configurations_above_the_pilot_floor(tmp_path: Path) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        # Every GEMMA treatment cell loses its solve; LOW's treatment cells hold.
        lost = key.arm is not SweAbArm.A0 and key.model == GEMMA
        return {"billed": _a0(i), "resolved": not lost}

    plan, cells = _stage(tmp_path, STAGE2, spec)
    everything = _run(plan, cells, tmp_path / "all.json")
    pilot = _floor_pilot(tmp_path / "pilot.json", {LOW: True, GEMMA: False})
    report = _run(plan, cells, tmp_path / "out.json", "--pilot-analysis", str(pilot))

    assert _close(everything["guardrail"]["H"]["difference"], -0.5)
    h = report["guardrail"]["H"]
    assert h["pairs"] == 6
    assert _close(h["difference"], 0.0)
    assert list(h["solve_rate_by_model"]) == [LOW]
    assert h["uninformative"][GEMMA]["label"] == "uninformative: below the 15% solve-rate floor"
    assert _close(h["uninformative"][GEMMA]["treatment"], 0.0)
    assert report["sensitivity_excluding_blocked"]["guardrail"]["H"]["pairs"] == 6
    # The completion test pools the same configurations.
    assert report["completion"]["H"]["pairs"] == 6
    assert list(report["completion"]["H"]["per_stratum"]) == [LOW]
    assert report["completion"]["H"]["reading"] == "no_difference_detected"
    assert everything["completion"]["H"]["pairs"] == 12
    # Efficiency stays per configuration, floor or not.
    assert report["efficiency"][GEMMA]["H"]["pairs"] == 6


def test_stage2_guardrail_is_not_estimable_when_no_configuration_passed_the_floor(
    tmp_path: Path,
) -> None:
    plan, cells = _stage(tmp_path, STAGE2, lambda key, i: {"billed": _a0(i)})
    pilot = _floor_pilot(tmp_path / "pilot.json", {LOW: False, GEMMA: False})
    report = _run(plan, cells, tmp_path / "out.json", "--pilot-analysis", str(pilot))

    for arm in ("H", "HC", "M"):
        assert report["guardrail"][arm]["verdict"] == "not_estimable"
        assert set(report["guardrail"][arm]["uninformative"]) == {LOW, GEMMA}
        assert report["completion"][arm]["reading"] == "not_estimable"
        assert report["completion"][arm]["p_holm"] is None
    assert report["efficiency"][LOW]["H"]["pairs"] == 6


# --- the completion test (H5) -------------------------------------------------------------


@pytest.mark.parametrize(
    ("a0_solves", "arm", "arm_solves", "reading"),
    [(False, "H", True, "better"), (True, "M", False, "worse")],
)
def test_completion_reads_better_or_worse_on_a_clear_two_sided_difference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    a0_solves: bool,
    arm: str,
    arm_solves: bool,
    reading: str,
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        return {"billed": _a0(i), "resolved": arm_solves if key.arm.value == arm else a0_solves}

    plan, cells = _stage(tmp_path, STAGE2, spec)
    report = _run(plan, cells, tmp_path / "out.json")

    result = report["completion"][arm]
    assert result["reading"] == reading
    # All 12 pairs are discordant in one direction: p = 2 / 2^12, Holm over three arms.
    assert _close(result["p_two_sided"], 2 / 4096)
    assert _close(result["p_holm"], 3 * 2 / 4096)
    assert _close(result["difference"], 1.0 if arm_solves else -1.0)
    assert result["treatment_only_resolved"] + result["control_only_resolved"] == 12
    assert result["ci95"][0] <= result["difference"] <= result["ci95"][1]
    others = [a for a in ("H", "HC", "M") if a != arm]
    assert [report["completion"][a]["reading"] for a in others] == ["no_difference_detected"] * 2
    assert all(report["completion"][a]["p_holm"] == 1.0 for a in others)
    assert report["sensitivity_excluding_blocked"]["completion"][arm]["reading"] == reading
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["completion"][arm] == reading
    # The non-inferiority guardrail rides alongside, unchanged in kind.
    assert report["guardrail"][arm]["verdict"] == ("non_inferior" if arm_solves else "inferior")


def test_completion_p_values_are_holm_adjusted_over_the_primary_arms_present(
    tmp_path: Path,
) -> None:
    def spec(key: CellKey, i: int) -> dict[str, Any]:
        # A0 solves tasks 0-2 in both configurations. H solves everything (6 treatment-only
        # pairs); M solves exactly the complement (6 each way); HC mirrors A0.
        solved = {SweAbArm.H: True, SweAbArm.M: i >= 3}.get(key.arm, i < 3)
        return {"billed": _a0(i), "resolved": solved}

    plan, cells = _stage(tmp_path, STAGE2, spec)
    three = _run(plan, cells, tmp_path / "three.json")["completion"]
    only_h_plan, only_h_cells = _stage(tmp_path / "h", {"A0": 1, "H": 1}, spec)
    alone = _run(only_h_plan, only_h_cells, tmp_path / "alone.json")["completion"]

    # Raw p = 2 / 2^6 = 0.03125 < 0.05, but with three arms Holm multiplies it by 3.
    assert _close(three["H"]["p_two_sided"], 2 / 64)
    assert _close(three["H"]["p_holm"], 3 * 2 / 64)
    assert three["H"]["reading"] == "no_difference_detected"
    assert alone["H"]["reading"] == "better"
    assert _close(alone["H"]["p_holm"], 2 / 64)
    assert set(alone) == {"H"}
    # Discordant pairs split 6 / 6 across the strata: no evidence; none at all: p = 1.
    assert (three["M"]["treatment_only_resolved"], three["M"]["control_only_resolved"]) == (6, 6)
    assert three["M"]["p_holm"] == 1.0 and three["M"]["reading"] == "no_difference_detected"
    assert three["HC"]["p_holm"] == 1.0 and three["HC"]["reading"] == "no_difference_detected"


def test_completion_pools_discordant_pairs_across_configurations_as_strata() -> None:
    guardrails = {
        "H": {
            "verdict": "non_inferior",
            "pairs": 20,
            "excluded_blocked_pairs": 0,
            "difference": 0.3,
            "ci95": [0.1, 0.5],
            "mcnemar": analysis.mcnemar({LOW: (5, 0), GEMMA: (5, 0)}),
        },
        "M": {
            "verdict": "inconclusive",
            "pairs": 20,
            "excluded_blocked_pairs": 0,
            "difference": -0.2,
            "ci95": [-0.4, 0.0],
            "mcnemar": analysis.mcnemar({LOW: (2, 4), GEMMA: (1, 3)}),
        },
    }

    result = analysis.completion_family(guardrails)

    assert result["H"]["treatment_only_resolved"] == 10
    assert _close(result["H"]["p_two_sided"], 2 / 1024)
    assert result["H"]["reading"] == "better"
    assert _close(result["M"]["p_two_sided"], 2 * (1 + 10 + 45 + 120) / 1024)
    assert result["M"]["reading"] == "no_difference_detected"


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
        ({"H": 0.01, "HC": 0.02, "M": 0.04}, {"H": 0.03, "HC": 0.04, "M": 0.04}),
    ],
)
def test_holm_adjustment(p_values: dict[str, float], adjusted: dict[str, float]) -> None:
    adjusted_now = analysis.holm(p_values)
    assert adjusted_now.keys() == adjusted.keys()
    assert all(_close(adjusted_now[k], v) for k, v in adjusted.items())


def test_mcnemar_pools_discordant_pairs_across_strata() -> None:
    result = analysis.mcnemar({LOW: (5, 1), GEMMA: (3, 1)})

    assert (result["treatment_only_resolved"], result["control_only_resolved"]) == (8, 2)
    assert _close(result["chi_square"], 3.6)
    assert _close(result["p_exact_two_sided"], 2 * (1 + 10 + 45) / 1024)


@pytest.mark.parametrize(
    ("q", "df", "quantile"),
    [(0.05, 10, 3.940299), (0.2, 23, 17.186506), (0.95, 23, 35.172462), (0.01, 99, 69.22989)],
)
def test_chi_square_quantiles_match_reference_values(q: float, df: int, quantile: float) -> None:
    assert _close(analysis.chi2_ppf(q, df), quantile)
