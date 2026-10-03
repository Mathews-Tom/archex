"""Pre-registered analysis of one R3x stage (benchmarks/preregistrations/R3x-swe-archex-ab.md).

```bash
uv run python scripts/run_swe_ab_suite.py validate --plan stage1-plan.json \
    --input benchmarks/swe_ab/results/stage1
uv run python scripts/swe_ab_analysis.py --plan stage1-plan.json \
    --input benchmarks/swe_ab/results/stage1 --output benchmarks/evidence/r3x-swe-ab-pilot.json
# Stage 2, carrying the Stage 1 headroom gate into the hypothesis labels:
uv run python scripts/swe_ab_analysis.py --plan stage2-plan.json \
    --input benchmarks/swe_ab/results/stage2 --output benchmarks/evidence/r3x-swe-ab.json \
    --pilot-analysis benchmarks/evidence/r3x-swe-ab-pilot.json
```

The directory must pass `validate_swe_ab_directory` first (complete, one host kind, no cell left
as a quota or credit block, authenticated with the campaign's key, identities single per arm);
anything it refuses, and any unscored cell, ends the analysis with ``REFUSED`` and exit status 1.
Nothing is dropped: a failed
cell enters every metric at the tokens it consumed and scores unresolved.

What is computed, and nothing else:

* **Efficiency, per configuration** (H1, H2, H4). Per task, ``ln(treatment billed tokens / A0
  billed tokens)``, billed = input + output + cache read + cache write, summed over the cell's
  requests; the control is A0 repetition 1 (a second A0 repetition, Stage 1 only, feeds the noise
  estimate and nothing else). Geometric mean ratio = ``exp(mean)``. Repository-clustered
  percentile bootstrap, 10,000 resamples, seed 20260909. The primary arms are H, HC and M (C is a
  pilot arm: reported, never tested). Two one-sided tests per arm, each Holm-adjusted over the
  primary arms present within the model, as two separate families: H0 ratio ≥ 1 − SESOI, one-sided
  bootstrap p = (1 + #{resampled ratio ≥ 0.90}) / 10,001; and H0 ratio ≤ 1 + SESOI, p = (1 +
  #{resampled ratio ≤ 1.10}) / 10,001. Reading: ``superior`` (first adjusted p < 0.05), else
  ``worse`` (second adjusted p < 0.05), else ``equivalent`` (the 90% interval inside 1 ± EQM),
  else ``inconclusive``. A pair with a zero-token cell has no log ratio: it is counted and listed
  under ``zero_token_pairs``, and stays in the completion analyses as unresolved.
* **Completion test, pooled** (H5). Per primary arm, the two-sided paired solve-rate comparison
  with A0 over the floor-passing configurations, stratified by configuration: the exact
  conditional test of the stratified McNemar table (the sum of the per-stratum discordant counts
  is Binomial(N, 1/2) under H0, which is the Mantel–Haenszel-type statistic's exact form), with
  the repository-clustered bootstrap 95% interval of the pooled difference. Holm over the primary
  arms present at α = 0.05. Reading: ``better`` / ``worse`` (adjusted p < 0.05, by the sign of the
  pooled difference), else ``no_difference_detected``.
* **Quality guardrail, pooled** (H3). Per (model, task), ``resolved(treatment) − resolved(A0)``;
  the estimate is the mean of per-model means (the model is the stratum). Same bootstrap,
  resampling repositories with all their models' pairs. ``non_inferior`` when the one-sided 95%
  lower bound (5th percentile) is above the −5 pp margin, ``inferior`` when the 95th percentile is
  below it, else ``inconclusive``. Cross-check: McNemar over discordant pairs pooled across the
  model strata (chi-square statistic and exact two-sided binomial p).
* **Sensitivity.** The efficiency, guardrail and completion analyses again without the (model,
  task) pairs in which either cell had a quota-blocked attempt.
* **Stage 1 gates** (when A0 has two repetitions): the A0 lever share per configuration against
  2 × SESOI (kill criterion 5; H's lever is out-of-patch reads, HC's and M's search plus
  out-of-patch reads), HC and M adoption against 25% (criterion 6; the share of cells with at
  least one archex CLI call, respectively one archex MCP call, failed cells included), H hook
  activity against 50% (criterion 7; annotated ÷ eligible less ``no_hits`` declines and stale-index
  declines after the first edit), the A0 solve-rate floor (15% over all A0 cells, failures
  unresolved), the Stage 2 size for the m primary arms kept after the adoption gates and the k
  configurations passing the floor (pairs needed n(m), target pairs ``max(370, n(m))``, tasks
  ``max(100, ceil(target / k))``), and the A0-vs-A0 noise that sets efficiency power (Holm first
  step α/m), the minimum detectable ratio, and whether the equivalence margin is reachable at that
  Stage 2 size. With ``--pilot-analysis`` the Stage 2 guardrail and completion analyses pool only
  the configurations that passed the floor (none: ``not_estimable``).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from archex.benchmark.swe_ab import (
    PREREGISTRATION_PATH,
    PRIMARY_ARMS,
    QUOTA_BLOCKED_DIR,
    Campaign,
    CellKey,
    CellStatus,
    SweAbArm,
    SweAbCell,
    SweAbError,
    SweAbPlan,
    load_cell,
    load_plan,
    plan_campaign,
    validate_swe_ab_directory,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parent.parent
import swe_ab_sample as sampler  # noqa: E402 - sibling script, importable only via sys.path

SEED = 20260909
RESAMPLES = 10_000
ALPHA = 0.05
SESOI_RATIO = 0.90
"""H1/H2 hold when the geometric-mean billed-token ratio is at most this (a 10% reduction)."""
EQM = 0.05
"""Equivalence margin on the ratio, ±5% (set at freeze)."""
NIM = -0.05
"""Non-inferiority margin on the pooled solve-rate difference (−5 percentage points)."""
WORSE_RATIO = 1.10
"""H4 holds (a treatment costs more tokens) when the ratio is above this (a 10% increase)."""
LEVER_GATE = round(2 * (1 - SESOI_RATIO), 6)
ADOPTION_GATE = 0.25
HOOK_ACTIVITY_GATE = 0.50
SOLVE_FLOOR = 0.15
"""Stage 1 gate: a configuration's A0 solve rate (all A0 cells) must reach this."""
POWER = 0.80
_NORMAL = statistics.NormalDist()


# --- distributions (stdlib only; each is re-derivable by hand) ------------------------------


def binomial_half_cdf(k: int, n: int) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, 1/2)``, exactly."""
    return sum(math.comb(n, i) for i in range(k + 1)) / 2**n


def regularized_gamma_p(a: float, x: float) -> float:
    """The regularized lower incomplete gamma function ``P(a, x)``.

    Series below ``a + 1``, Lentz's continued fraction for ``Q = 1 - P`` above it.
    """
    if x <= 0:
        return 0.0
    prefix = math.exp(a * math.log(x) - x - math.lgamma(a))
    if x < a + 1:
        term = total = 1.0 / a
        denominator = a
        for _ in range(10_000):
            denominator += 1
            term *= x / denominator
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * prefix
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-16:
            break
    return 1.0 - prefix * h


def chi2_ppf(q: float, df: int) -> float:
    """The ``q`` quantile of the chi-square distribution with ``df`` degrees of freedom."""
    low, high = 0.0, float(max(1, df))
    while regularized_gamma_p(df / 2, high / 2) < q:
        high *= 2
    for _ in range(200):
        mid = (low + high) / 2
        low, high = (mid, high) if regularized_gamma_p(df / 2, mid / 2) < q else (low, mid)
    return (low + high) / 2


# --- loading -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    plan: SweAbPlan
    cells: dict[CellKey, SweAbCell]
    blocked: list[SweAbCell]
    repo_of: dict[str, str]
    coverage: dict[str, Any]


def load_stage(directory: Path, plan: SweAbPlan, campaign: Campaign) -> Stage:
    """The validated cells of one stage; raise `SweAbError` on anything the protocol refuses.

    ``campaign`` is the plan's (`plan_campaign`).
    """
    coverage = validate_swe_ab_directory(directory, plan, campaign)
    cells = {key: load_cell(directory / key.relative_path) for key in plan.cells()}
    if unscored := sorted(str(key.relative_path) for key, c in cells.items() if c.resolved is None):
        raise SweAbError(f"{len(unscored)} cells were never scored, first: {unscored[0]}")
    root = directory / QUOTA_BLOCKED_DIR
    blocked = [load_cell(path) for path in sorted(root.rglob("*.json"))] if root.is_dir() else []
    return Stage(
        plan=plan,
        cells=cells,
        blocked=blocked,
        repo_of={task.task_id: task.repo for task in plan.tasks},
        coverage=coverage.model_dump(),
    )


@dataclass(frozen=True)
class Pair:
    task_id: str
    repo: str
    model: str
    control: SweAbCell
    treatment: SweAbCell
    blocked: bool


def pairs(stage: Stage, arm: SweAbArm) -> list[Pair]:
    """(model, task) pairs of A0 repetition 1 and the treatment's repetition 1."""
    blocked_keys = {cell.key for cell in stage.blocked}

    def was_blocked(cell: SweAbCell) -> bool:
        return cell.quota.prior_blocked_attempts > 0 or cell.key in blocked_keys

    out: list[Pair] = []
    for model in stage.plan.models:
        for task in stage.plan.tasks:
            control = stage.cells[CellKey(task.task_id, model, SweAbArm.A0, 1)]
            treatment = stage.cells[CellKey(task.task_id, model, arm, 1)]
            out.append(
                Pair(
                    task_id=task.task_id,
                    repo=task.repo,
                    model=model,
                    control=control,
                    treatment=treatment,
                    blocked=was_blocked(control) or was_blocked(treatment),
                )
            )
    return out


# --- repository-clustered bootstrap ----------------------------------------------------------


def _draws(clusters: int) -> NDArray[np.int64]:
    """Cluster indices for every resample; the same seed for every estimate."""
    return np.random.default_rng(SEED).integers(0, clusters, size=(RESAMPLES, clusters))


def cluster_mean(values: Mapping[str, Sequence[float]]) -> tuple[float, NDArray[np.float64]]:
    """Mean over all items, and its resampled distribution (clusters drawn with replacement)."""
    names = sorted(name for name, items in values.items() if items)
    sums = np.array([math.fsum(values[name]) for name in names], dtype=np.float64)
    counts = np.array([len(values[name]) for name in names], dtype=np.float64)
    idx = _draws(len(names))
    return float(sums.sum() / counts.sum()), sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def stratified_cluster_mean(
    values: Mapping[tuple[str, str], Sequence[float]], strata: Sequence[str]
) -> tuple[float, NDArray[np.float64]]:
    """Mean of per-stratum means; clusters (the first key) are resampled with all their strata."""
    names = sorted({cluster for cluster, _ in values})
    sums = np.zeros((len(names), len(strata)))
    counts = np.zeros((len(names), len(strata)))
    for (cluster, stratum), items in values.items():
        i, j = names.index(cluster), strata.index(stratum)
        sums[i, j] += math.fsum(items)
        counts[i, j] += len(items)
    present = counts.sum(axis=0) > 0
    point = float((sums.sum(axis=0)[present] / counts.sum(axis=0)[present]).mean())
    idx = _draws(len(names))
    boot_sums = sums[idx].sum(axis=1)
    boot_counts = counts[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = boot_sums / boot_counts
    return point, np.nanmean(means[:, present], axis=1)


def _pct(boot: NDArray[np.float64], q: float) -> float:
    return float(np.percentile(boot, q))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error across the given comparisons)."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


# --- primary metrics -------------------------------------------------------------------------


def efficiency(stage_pairs: Iterable[Pair], *, exclude_blocked: bool = False) -> dict[str, Any]:
    """Geometric-mean billed-token ratio of one treatment over A0 for one model's pairs."""
    by_repo: dict[str, list[float]] = defaultdict(list)
    zero: list[dict[str, Any]] = []
    excluded = 0
    for pair in stage_pairs:
        if exclude_blocked and pair.blocked:
            excluded += 1
            continue
        control = pair.control.usage.total_billed
        treatment = pair.treatment.usage.total_billed
        if control == 0 or treatment == 0:
            zero.append(
                {
                    "task_id": pair.task_id,
                    "control_failure": pair.control.failure_reason,
                    "treatment_failure": pair.treatment.failure_reason,
                }
            )
            continue
        by_repo[pair.repo].append(math.log(treatment / control))
    result: dict[str, Any] = {
        "pairs": sum(len(v) for v in by_repo.values()),
        "repositories": len(by_repo),
        "zero_token_pairs": zero,
        "excluded_blocked_pairs": excluded,
    }
    if not by_repo:
        return {
            **result,
            "geometric_mean_ratio": None,
            "p_vs_sesoi": None,
            "p_vs_worse": None,
        }
    point, boot = cluster_mean(by_repo)
    ratios = np.exp(boot)
    ci90 = (_pct(ratios, 5), _pct(ratios, 95))
    return {
        **result,
        "geometric_mean_ratio": math.exp(point),
        "ci95": [_pct(ratios, 2.5), _pct(ratios, 97.5)],
        "ci90": list(ci90),
        "p_vs_sesoi": (1 + int((ratios >= SESOI_RATIO).sum())) / (RESAMPLES + 1),
        "p_vs_worse": (1 + int((ratios <= WORSE_RATIO).sum())) / (RESAMPLES + 1),
        "within_eqm": ci90[0] >= 1 - EQM and ci90[1] <= 1 + EQM,
        "increase": _pct(ratios, 2.5) > 1.0,
    }


def efficiency_family(
    stage: Stage,
    *,
    exclude_blocked: bool = False,
    no_headroom: Mapping[str, set[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """H1/H2/H4 per model; each direction Holm-adjusted over the primary arms present."""
    arms = [arm for arm in PRIMARY_ARMS if arm in stage.plan.repetitions]
    by_arm = {arm: pairs(stage, arm) for arm in arms}
    out: dict[str, dict[str, Any]] = {}
    for model in stage.plan.models:
        results = {
            arm.value: efficiency(
                (p for p in by_arm[arm] if p.model == model), exclude_blocked=exclude_blocked
            )
            for arm in arms
        }
        superior = holm(
            {arm: r["p_vs_sesoi"] for arm, r in results.items() if r["p_vs_sesoi"] is not None}
        )
        worse = holm(
            {arm: r["p_vs_worse"] for arm, r in results.items() if r["p_vs_worse"] is not None}
        )
        for arm, result in results.items():
            result["p_holm"] = superior.get(arm)
            result["p_holm_worse"] = worse.get(arm)
            if arm not in superior:
                result["verdict"] = "not_estimable"
            elif superior[arm] < ALPHA:
                result["verdict"] = "superior"
            elif worse[arm] < ALPHA:
                result["verdict"] = "worse"
            elif result["within_eqm"]:
                result["verdict"] = "equivalent"
            else:
                result["verdict"] = "inconclusive"
            if no_headroom is not None:
                result["hypothesis"] = (
                    "mis-specified: no attainable headroom (Stage 1 gate)"
                    if arm in no_headroom.get(model, set())
                    else "tested"
                )
        out[model] = results
    return out


def mcnemar(discordant: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    """McNemar over discordant pairs pooled across strata; ``b`` favours the treatment."""
    b = sum(pair[0] for pair in discordant.values())
    c = sum(pair[1] for pair in discordant.values())
    n = b + c
    return {
        "treatment_only_resolved": b,
        "control_only_resolved": c,
        "per_stratum": {
            k: {"treatment_only": v[0], "control_only": v[1]} for k, v in discordant.items()
        },
        "chi_square": (b - c) ** 2 / n if n else None,
        "p_exact_two_sided": min(1.0, 2 * binomial_half_cdf(min(b, c), n)) if n else None,
    }


def guardrail(
    stage_pairs: Sequence[Pair], models: Sequence[str], *, exclude_blocked: bool = False
) -> dict[str, Any]:
    """Pooled paired solve-rate difference, treatment − A0, with the model as stratum."""
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    discordant: dict[str, list[int]] = {model: [0, 0] for model in models}
    rates: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    excluded = 0
    for pair in stage_pairs:
        if exclude_blocked and pair.blocked:
            excluded += 1
            continue
        control = bool(pair.control.resolved)
        treatment = bool(pair.treatment.resolved)
        values[(pair.repo, pair.model)].append(float(treatment) - float(control))
        rates[pair.model].append((control, treatment))
        if treatment and not control:
            discordant[pair.model][0] += 1
        elif control and not treatment:
            discordant[pair.model][1] += 1
    result: dict[str, Any] = {
        "pairs": sum(len(v) for v in values.values()),
        "excluded_blocked_pairs": excluded,
        "solve_rate_by_model": {
            model: {
                "pairs": len(rows),
                "control": statistics.fmean(float(a) for a, _ in rows),
                "treatment": statistics.fmean(float(t) for _, t in rows),
            }
            for model, rows in sorted(rates.items())
        },
        "mcnemar": mcnemar({m: (d[0], d[1]) for m, d in discordant.items()}),
    }
    if not values:
        return {**result, "difference": None, "verdict": "not_estimable"}
    point, boot = stratified_cluster_mean(values, list(models))
    lower, upper = _pct(boot, 5), _pct(boot, 95)
    verdict = "non_inferior" if lower > NIM else ("inferior" if upper < NIM else "inconclusive")
    return {
        **result,
        "difference": point,
        "ci95": [_pct(boot, 2.5), _pct(boot, 97.5)],
        "one_sided_95_lower": lower,
        "verdict": verdict,
    }


UNINFORMATIVE = "uninformative: below the 15% solve-rate floor"


def guardrail_family(
    stage: Stage, *, exclude_blocked: bool = False, uninformative: Collection[str] = frozenset()
) -> dict[str, Any]:
    """H3 per primary arm over the configurations above the pilot's floor.

    Configurations in ``uninformative`` stay out of the pooled estimate and are reported alone;
    with none left the verdict is ``not_estimable``.
    """
    informative = [m for m in stage.plan.models if m not in uninformative]
    out: dict[str, Any] = {}
    for arm in PRIMARY_ARMS:
        if arm not in stage.plan.repetitions:
            continue
        stage_pairs = pairs(stage, arm)
        result = guardrail(
            [p for p in stage_pairs if p.model in informative],
            informative,
            exclude_blocked=exclude_blocked,
        )
        if not informative:
            result["reason"] = "no configuration passed the Stage 1 solve-rate floor"
        if uninformative:
            result["uninformative"] = {
                model: {
                    "label": UNINFORMATIVE,
                    **guardrail([p for p in stage_pairs if p.model == model], [model])[
                        "solve_rate_by_model"
                    ].get(model, {}),
                }
                for model in stage.plan.models
                if model in uninformative
            }
        out[arm.value] = result
    return out


def completion_family(guardrails: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """H5: the two-sided pooled paired solve-rate test per primary arm, Holm over the arms.

    ``guardrails`` is `guardrail_family`'s output, whose pooled difference, repository-clustered
    interval and discordant counts (strata: the configurations) this reuses. The p-value is the
    exact conditional test of the stratified McNemar table: given each stratum's discordant count
    the treatment-only count is Binomial(n, 1/2) under H0, so their sum is Binomial(N, 1/2) and
    ``mcnemar`` already returns its two-sided p. With no discordant pair the p-value is 1.
    """
    tested: dict[str, float] = {}
    for arm, result in guardrails.items():
        if result["verdict"] != "not_estimable":
            p = result["mcnemar"]["p_exact_two_sided"]
            tested[arm] = 1.0 if p is None else p
    adjusted = holm(tested)
    out: dict[str, dict[str, Any]] = {}
    for arm, result in guardrails.items():
        mc = result["mcnemar"]
        difference = result["difference"]
        p_holm = adjusted.get(arm)
        if p_holm is None:
            reading = "not_estimable"
        elif p_holm < ALPHA and difference > 0:
            reading = "better"
        elif p_holm < ALPHA and difference < 0:
            reading = "worse"
        else:
            reading = "no_difference_detected"
        out[arm] = {
            "pairs": result["pairs"],
            "excluded_blocked_pairs": result["excluded_blocked_pairs"],
            "difference": difference,
            "ci95": result.get("ci95"),
            "treatment_only_resolved": mc["treatment_only_resolved"],
            "control_only_resolved": mc["control_only_resolved"],
            "per_stratum": mc["per_stratum"],
            "p_two_sided": tested.get(arm),
            "p_holm": p_holm,
            "reading": reading,
        }
    return out


# --- disclosure: outcomes and blocked attempts -------------------------------------------


def outcomes(stage: Stage) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for model in stage.plan.models:
        out[model] = {}
        for arm in SweAbArm:
            if arm not in stage.plan.repetitions:
                continue
            cells = [c for k, c in stage.cells.items() if k.model == model and k.arm is arm]
            out[model][arm.value] = {
                "cells": len(cells),
                "ok": sum(1 for c in cells if c.status is CellStatus.OK),
                "failures": dict(
                    sorted(
                        Counter(str(c.failure_reason) for c in cells if c.failure_reason).items()
                    )
                ),
                "resolved": sum(1 for c in cells if c.resolved),
                "billed_tokens": sum(c.usage.total_billed for c in cells),
                "modelled_cost_usd": math.fsum(c.usage.cost_usd for c in cells),
                "omp_quota_retries": sum(c.quota.omp_retries for c in cells),
            }
    return out


def blocked_attempts(stage: Stage) -> dict[str, Any]:
    """Quota-blocked attempts per model and arm (never scored; tokens reported separately)."""
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in stage.blocked:
        slot = rows.setdefault(cell.model, {}).setdefault(
            cell.arm.value,
            {"attempts": 0, "billed_tokens": 0, "modelled_cost_usd": 0.0, "phase": {}},
        )
        slot["attempts"] += 1
        slot["billed_tokens"] += cell.usage.total_billed
        slot["modelled_cost_usd"] += cell.usage.cost_usd
        phase = str(cell.quota.block_phase)
        slot["phase"][phase] = slot["phase"].get(phase, 0) + 1
    return {"total": len(stage.blocked), "by_model_and_arm": rows}


# --- Stage 1 gates -------------------------------------------------------------------------


def _billed_input(cell: SweAbCell) -> int:
    return cell.usage.input + cell.usage.cache_read + cell.usage.cache_write


def lever_shares(stage: Stage) -> dict[str, Any]:
    """Kill criterion 5: per model, the A0 share of billed input each treatment could displace.

    H's lever is out-of-patch reads; HC's and M's is search plus out-of-patch reads (compounded
    tokens over the cell's billed input). Mean over A0 repetitions within a task, then over tasks.
    """
    out: dict[str, Any] = {}
    reps = stage.plan.repetitions[SweAbArm.A0]
    for model in stage.plan.models:
        by_repo: dict[str, dict[str, list[float]]] = {
            "H": defaultdict(list),
            "HC": defaultdict(list),
            "M": defaultdict(list),
        }
        no_input: list[str] = []
        for task in stage.plan.tasks:
            h: list[float] = []
            hc: list[float] = []
            for rep in range(1, reps + 1):
                cell = stage.cells[CellKey(task.task_id, model, SweAbArm.A0, rep)]
                billed = _billed_input(cell)
                if billed == 0:
                    continue
                read = cell.out_of_patch_read_tokens_compounded
                h.append(read / billed)
                hc.append((read + cell.channel_tokens_compounded["search"]) / billed)
            if not h:
                no_input.append(task.task_id)
                continue
            by_repo["H"][task.repo].append(statistics.fmean(h))
            by_repo["HC"][task.repo].append(statistics.fmean(hc))
            by_repo["M"][task.repo].append(statistics.fmean(hc))
        model_out: dict[str, Any] = {"tasks_without_billed_input": no_input}
        for arm, values in by_repo.items():
            if not values:
                model_out[arm] = {"share": None, "passed": False}
                continue
            point, boot = cluster_mean(values)
            model_out[arm] = {
                "share": point,
                "ci95": [_pct(boot, 2.5), _pct(boot, 97.5)],
                "threshold": LEVER_GATE,
                "passed": point >= LEVER_GATE,
            }
        out[model] = model_out
    return out


def adoption(stage: Stage, arm: SweAbArm) -> dict[str, Any]:
    """Share of the arm's cells that made at least one archex call (criterion 6 for HC and M).

    HC and C count archex CLI calls (``subcommands``: per subcommand); M counts archex MCP tool
    calls (``subcommands``: per MCP tool). Failed cells stay in the denominator.
    """
    cells = [c for k, c in stage.cells.items() if k.arm is arm]

    def calls(cell: SweAbCell) -> int:
        return cell.archex_mcp_calls if arm.mcp else cell.archex_cli_calls

    subcommands: Counter[str] = Counter()
    for cell in cells:
        subcommands.update(cell.archex_mcp_tools if arm.mcp else cell.archex_cli_subcommands)
    share = sum(1 for c in cells if calls(c) > 0) / len(cells)
    return {
        "cells": len(cells),
        "share": share,
        "by_model": {
            model: statistics.fmean(1.0 if calls(c) > 0 else 0.0 for c in cells if c.model == model)
            for model in stage.plan.models
        },
        "calls": sum(calls(c) for c in cells),
        "subcommands": dict(sorted(subcommands.items())),
        "threshold": ADOPTION_GATE,
        "passed": share >= ADOPTION_GATE,
    }


def hook_activity(stage: Stage, arm: SweAbArm) -> dict[str, Any]:
    """Criterion 7: annotated ÷ eligible calls, less no-hit searches and stale-index declines.

    The stale declines excluded are those after the first edit; a search with no hits has nothing
    to annotate (``fail_open["no_hits"]``). Searches the hook could not parse stay in.
    """

    def share(cells: Sequence[SweAbCell]) -> tuple[float | None, dict[str, int]]:
        ledgers = [c.hook_ledger for c in cells if c.hook_ledger is not None]
        totals = {
            "eligible": sum(ledger.eligible for ledger in ledgers),
            "annotated": sum(ledger.annotated for ledger in ledgers),
            "no_hits": sum(ledger.fail_open.get("no_hits", 0) for ledger in ledgers),
            "stale_after_first_edit": sum(ledger.not_fresh_after_first_edit for ledger in ledgers),
        }
        denominator = totals["eligible"] - totals["no_hits"] - totals["stale_after_first_edit"]
        return (totals["annotated"] / denominator if denominator > 0 else None), totals

    cells = [c for k, c in stage.cells.items() if k.arm is arm]
    pooled, totals = share(cells)
    fail_open: Counter[str] = Counter()
    for cell in cells:
        if cell.hook_ledger is not None:
            fail_open.update(cell.hook_ledger.fail_open)
    return {
        **totals,
        "share": pooled,
        "by_model": {m: share([c for c in cells if c.model == m])[0] for m in stage.plan.models},
        "fail_open_by_reason": dict(sorted(fail_open.items())),
        "zero_annotation_cells": sorted(
            f"{c.model}:{c.task_id}"
            for c in cells
            if c.hook_ledger is not None
            and c.hook_ledger.eligible > 0
            and not c.hook_ledger.annotated
        ),
        "threshold": HOOK_ACTIVITY_GATE,
        "passed": pooled is not None and pooled >= HOOK_ACTIVITY_GATE,
    }


def solve_floor(stage: Stage) -> dict[str, Any]:
    """Per configuration, the A0 solve rate over every A0 cell (all repetitions) against the floor.

    A failed cell is unresolved and counts in the denominator.
    """
    out: dict[str, Any] = {}
    for model in stage.plan.models:
        cells = [c for k, c in stage.cells.items() if k.model == model and k.arm is SweAbArm.A0]
        rate = sum(1 for c in cells if c.resolved) / len(cells)
        out[model] = {"rate": rate, "n": len(cells), "passed": rate >= SOLVE_FLOOR}
    return out


def primary_arms_kept(
    stage: Stage, gates: Mapping[str, Mapping[str, Any] | None]
) -> list[SweAbArm]:
    """The primary arms in the plan that survive the adoption gates (HC and M; H has none).

    ``gates`` maps an arm's value to its adoption gate; an arm whose gate failed is dropped.
    """
    kept: list[SweAbArm] = []
    for arm in PRIMARY_ARMS:
        if arm not in stage.plan.repetitions:
            continue
        gate = gates.get(arm.value)
        if gate is None or gate["passed"]:
            kept.append(arm)
    return kept


def stage2_size(passing: int, configurations: int, arms_kept: Sequence[SweAbArm]) -> dict[str, Any]:
    """The Stage 2 size the rule gives for ``passing`` of ``configurations`` and ``arms_kept``.

    Pairs needed ``n(m)`` for the two-sided completion test over the ``m`` primary arms kept
    (Holm first step ``alpha / m``), target pairs ``max(370, n(m))``, tasks
    ``max(100, ceil(target / k))``. With no arm kept or no passing configuration there is no size.
    """
    m = len(arms_kept)
    size: dict[str, Any] = {
        "passing_configurations": passing,
        "primary_arms_kept": [arm.value for arm in arms_kept],
        "pairs_needed": sampler.stage2_pairs_needed(m) if m else None,
        "target_pairs": sampler.stage2_target_pairs(m) if m else None,
    }
    if passing == 0 or m == 0:
        return {**size, "tasks": None, "estimable": False}
    return {
        **size,
        "tasks": sampler.stage2_task_count(m, passing, configurations),
        "estimable": True,
    }


def a0_noise(stage: Stage, stage2_tasks: int | None, arms_kept: int) -> dict[str, Any]:
    """A0 rep 2 vs rep 1 per configuration: token CV, flip rate, and what they imply for Stage 2.

    ``sd`` is the standard deviation of per-task ``ln(rep2 / rep1)``, the paired log-ratio spread
    a treatment with no effect would show. Power treats tasks as independent (Stage 1 has about
    two tasks per repository, too few to estimate a between-repository component) and uses the
    Holm first-step level ``alpha / m`` (``m`` = ``arms_kept``, the primary arms kept after the
    adoption gates; at least 1), one-sided.
    """
    z_alpha = _NORMAL.inv_cdf(1 - ALPHA / max(1, arms_kept))
    z_beta = _NORMAL.inv_cdf(POWER)
    delta = -math.log(SESOI_RATIO)
    eqm_delta = math.log(1 + EQM)
    out: dict[str, Any] = {}
    for model in stage.plan.models:
        diffs: list[float] = []
        flips = 0
        for task in stage.plan.tasks:
            first = stage.cells[CellKey(task.task_id, model, SweAbArm.A0, 1)]
            second = stage.cells[CellKey(task.task_id, model, SweAbArm.A0, 2)]
            flips += bool(first.resolved) != bool(second.resolved)
            if first.usage.total_billed and second.usage.total_billed:
                diffs.append(math.log(second.usage.total_billed / first.usage.total_billed))
        result: dict[str, Any] = {
            "tasks": len(stage.plan.tasks),
            "token_pairs": len(diffs),
            "flip_rate": flips / len(stage.plan.tasks),
        }
        if len(diffs) < 2:
            out[model] = {**result, "sd": None}
            continue
        sd = statistics.stdev(diffs)
        if sd == 0:
            # Identical repetitions: a degenerate noise estimate, not a precise one.
            out[model] = {**result, "sd": 0.0, "degenerate": True}
            continue
        df = len(diffs) - 1
        within = sd / math.sqrt(2)

        def tasks_needed(spread: float) -> int:
            return math.ceil(((z_alpha + z_beta) * spread / delta) ** 2)

        out[model] = {**result, "sd": sd, "within_task_cv": math.sqrt(math.exp(within**2) - 1)}
        out[model]["tasks_for_power_at_sesoi"] = tasks_needed(sd)
        if stage2_tasks is not None:
            half_width = _NORMAL.inv_cdf(1 - ALPHA) * sd / math.sqrt(stage2_tasks)
            out[model].update(
                {
                    "power_at_stage2_tasks": _NORMAL.cdf(
                        delta * math.sqrt(stage2_tasks) / sd - z_alpha
                    ),
                    "min_detectable_ratio_at_stage2": math.exp(
                        -(z_alpha + z_beta) * sd / math.sqrt(stage2_tasks)
                    ),
                    "eqm_ci90_half_width_at_stage2": half_width,
                    "eqm_reachable_at_stage2": half_width < eqm_delta,
                }
            )
        out[model].update(
            {
                "eqm_log_margin": eqm_delta,
                "sd_upper_limits": {
                    f"one_sided_{round(p * 100)}": {
                        "sd": upper,
                        "tasks_for_power_at_sesoi": tasks_needed(upper),
                    }
                    for p in (0.80, 0.90, 0.95)
                    for upper in [sd * math.sqrt(df / chi2_ppf(1 - p, df))]
                },
            }
        )
    return out


def stage1_gates(stage: Stage) -> dict[str, Any] | None:
    """Every Stage 1 gate, or ``None`` for a plan without the A0 replicate (Stage 2)."""
    if stage.plan.repetitions.get(SweAbArm.A0, 0) < 2:
        return None
    arms = stage.plan.repetitions
    floor = solve_floor(stage)
    passing = sum(1 for gate in floor.values() if gate["passed"])
    adoption_gates: dict[str, dict[str, Any] | None] = {
        arm.value: adoption(stage, arm) if arm in arms else None
        for arm in (SweAbArm.HC, SweAbArm.M, SweAbArm.C)
    }
    kept = primary_arms_kept(stage, adoption_gates)
    size = stage2_size(passing, len(stage.plan.models), kept)
    return {
        "headroom": lever_shares(stage),
        "solve_floor": floor,
        "stage2_size": size,
        "hc_adoption": adoption_gates["HC"],
        "mcp_adoption": adoption_gates["M"],
        "c_adoption": adoption_gates["C"],
        "hook_activity": hook_activity(stage, SweAbArm.H) if SweAbArm.H in arms else None,
        "hook_activity_hc": hook_activity(stage, SweAbArm.HC) if SweAbArm.HC in arms else None,
        "a0_noise": a0_noise(stage, size["tasks"], len(kept)),
    }


# --- report --------------------------------------------------------------------------------


def _no_headroom(pilot: Mapping[str, Any]) -> dict[str, set[str]]:
    pilot_gates = cast("dict[str, Any]", pilot.get("gates") or {})
    headroom = cast("dict[str, dict[str, Any]]", pilot_gates.get("headroom") or {})
    return {
        model: {
            arm
            for arm in ("H", "HC", "M")
            if arm in gates and not cast("dict[str, Any]", gates[arm])["passed"]
        }
        for model, gates in headroom.items()
    }


def _below_floor(pilot: Mapping[str, Any]) -> frozenset[str]:
    """Configurations whose Stage 1 solve-rate floor gate failed."""
    pilot_gates = cast("dict[str, Any]", pilot.get("gates") or {})
    floor = cast("dict[str, dict[str, Any]]", pilot_gates.get("solve_floor") or {})
    return frozenset(label for label, gate in floor.items() if not gate["passed"])


def _identities(stage: Stage) -> dict[str, Any]:
    cells = list(stage.cells.values())
    return {
        "emulated": cells[0].emulated,
        "network": cells[0].network,
        "omp_version": cells[0].omp_version,
        "by_arm": {
            arm.value: {
                "archex_wheel_sha256": next(
                    (c.archex_wheel_sha256 for c in cells if c.arm is arm), None
                ),
                "hook_module_sha256": next(
                    (c.hook_module_sha256 for c in cells if c.arm is arm), None
                ),
                "cli_guide_sha256": next((c.cli_guide_sha256 for c in cells if c.arm is arm), None),
                "mcp_config_sha256": next(
                    (c.mcp_config_sha256 for c in cells if c.arm is arm), None
                ),
            }
            for arm in SweAbArm
            if arm in stage.plan.repetitions
        },
    }


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in cast("dict[object, object]", value).items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in cast("list[object]", value)]
    return value


def analyse(stage: Stage, pilot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    no_headroom = _no_headroom(pilot) if pilot is not None else None
    uninformative = _below_floor(pilot) if pilot is not None else frozenset[str]()
    guardrails = guardrail_family(stage, uninformative=uninformative)
    blocked_free = guardrail_family(stage, exclude_blocked=True, uninformative=uninformative)
    report = {
        "analysis": "r3x-swe-ab",
        "preregistration": PREREGISTRATION_PATH,
        "plan": stage.plan.name,
        "constants": {
            "seed": SEED,
            "resamples": RESAMPLES,
            "alpha": ALPHA,
            "sesoi_ratio": SESOI_RATIO,
            "worse_ratio": WORSE_RATIO,
            "eqm": EQM,
            "nim": NIM,
            "lever_gate": LEVER_GATE,
            "adoption_gate": ADOPTION_GATE,
            "hook_activity_gate": HOOK_ACTIVITY_GATE,
            "solve_floor": SOLVE_FLOOR,
            "guardrail_pairs": sampler.GUARDRAIL_PAIRS,
            "primary_arms": [arm.value for arm in PRIMARY_ARMS],
            "completion_alpha": ALPHA,
        },
        "provenance": {**_identities(stage), "coverage": stage.coverage},
        "outcomes": outcomes(stage),
        "blocked_attempts": blocked_attempts(stage),
        "efficiency": efficiency_family(stage, no_headroom=no_headroom),
        "guardrail": guardrails,
        "completion": completion_family(guardrails),
        "sensitivity_excluding_blocked": {
            "efficiency": efficiency_family(stage, exclude_blocked=True, no_headroom=no_headroom),
            "guardrail": blocked_free,
            "completion": completion_family(blocked_free),
        },
        "gates": stage1_gates(stage),
    }
    return cast("dict[str, Any]", _rounded(report))


def render(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="the stage's result directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot-analysis",
        type=Path,
        help="Stage 1 analysis JSON; labels Stage 2 hypotheses its headroom gate declared "
        "mis-specified and drops configurations that failed its solve-rate floor from the "
        "pooled guardrail and completion tests",
    )
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        stage = load_stage(args.input, plan, plan_campaign(plan, root=REPO_ROOT))
    except SweAbError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    pilot = (
        cast("dict[str, Any]", json.loads(args.pilot_analysis.read_text(encoding="utf-8")))
        if args.pilot_analysis
        else None
    )
    report = analyse(stage, pilot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")
    summary = {
        model: {arm: r.get("verdict") for arm, r in arms.items()}
        for model, arms in report["efficiency"].items()
    }
    print(
        json.dumps(
            {
                "efficiency": summary,
                "guardrail": {arm: r["verdict"] for arm, r in report["guardrail"].items()},
                "completion": {arm: r["reading"] for arm, r in report["completion"].items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
