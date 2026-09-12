"""Validation, inference, and deterministic ledgers for R30 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from archex.benchmark.scope_aware_campaign import (
    CAMPAIGN_ID,
    MANIFEST_FILENAME,
    ScopeAwareCampaignError,
    validate_scope_aware_campaign,
)
from archex.benchmark.scope_aware_receipt import MultiScopeReceipt, SingleScopeReceipt
from archex.benchmark.scope_aware_run import (
    ARM_IDS,
    CELLS_DIRNAME,
    CONTROL_ARM,
    RUN_RECEIPT_FILENAME,
    TREATMENT_ARM,
    ScopeAwareCellArtifact,
    canonical_json_bytes,
    cell_artifact_path,
    load_cell_artifact,
)

EVIDENCE_SCHEMA_VERSION = 1
LEDGER_FILENAME = "ledger.json"
ANALYSIS_FILENAME = "analysis.json"
REPORT_FILENAME = "REPORT.md"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260913
MWG = 0.05
NIM = -0.02
EQM = 0.02
WARM_P95_LIMIT_MS = 3000.0


class ScopeAwareEvidenceError(ValueError):
    """Raised when R30 evidence contradicts the immutable campaign."""


@dataclass(frozen=True)
class ScopeAwareEvidenceCoverage:
    """Validated raw-cell coverage for the complete paired matrix."""

    cells: int
    planned_cells: int
    successes: int
    failures: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScopeAwareEvidenceError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeAwareEvidenceError(f"Invalid {path}: {exc}") from exc
    _require(isinstance(raw, dict), f"{path}: expected JSON object")
    return cast("dict[str, Any]", raw)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ScopeAwareEvidenceError(f"Cannot hash {path}: {exc}") from exc


def _cell_entries(cells_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = cells_payload.get("cells")
    if not isinstance(raw, list):
        raise ScopeAwareEvidenceError("cells.json: cells must be an array")
    entries: list[dict[str, Any]] = []
    for item in cast("list[Any]", raw):
        if not isinstance(item, dict):
            raise ScopeAwareEvidenceError("cells.json: cell must be an object")
        entries.append(cast("dict[str, Any]", item))
    return entries


def _load_campaign(campaign_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        validate_scope_aware_campaign(campaign_dir)
    except ScopeAwareCampaignError as exc:
        raise ScopeAwareEvidenceError(f"frozen campaign is invalid: {exc}") from exc
    return (
        _load_json(campaign_dir / "population.json"),
        _load_json(campaign_dir / "cells.json"),
        _load_json(campaign_dir / MANIFEST_FILENAME),
    )


def _expected_artifacts(evidence_dir: Path, cells: list[dict[str, Any]]) -> dict[str, Path]:
    expected: dict[str, Path] = {}
    for cell in cells:
        cell_id = cell.get("cell_id")
        task_id = cell.get("task_id")
        repo_id = cell.get("repository_id")
        arm = cell.get("arm")
        if not (
            isinstance(cell_id, str)
            and isinstance(task_id, str)
            and isinstance(repo_id, str)
            and isinstance(arm, str)
        ):
            raise ScopeAwareEvidenceError("cells.json: invalid cell identity")
        if arm not in ARM_IDS:
            raise ScopeAwareEvidenceError(f"{cell_id}: unknown arm")
        if cell_id in expected:
            raise ScopeAwareEvidenceError(f"cells.json: duplicate cell_id {cell_id}")
        expected[cell_id] = cell_artifact_path(evidence_dir / arm, repo_id, task_id)
    return expected


def _artifact_files(evidence_dir: Path) -> set[Path]:
    files: set[Path] = set()
    for arm in ARM_IDS:
        cells_dir = evidence_dir / arm / CELLS_DIRNAME
        if cells_dir.exists():
            files.update(path for path in cells_dir.rglob("*.json") if path.is_file())
    return files


def _validate_artifact(
    artifact: ScopeAwareCellArtifact,
    declared: dict[str, Any],
) -> None:
    _require(artifact.cell_id == declared["cell_id"], "artifact cell_id drift")
    _require(artifact.task_id == declared["task_id"], f"{artifact.cell_id}: task_id drift")
    _require(
        artifact.repository_id == declared["repository_id"], f"{artifact.cell_id}: repository drift"
    )
    _require(artifact.arm == declared["arm"], f"{artifact.cell_id}: arm drift")
    _require(artifact.kind == declared["kind"], f"{artifact.cell_id}: kind drift")
    _require(artifact.family == declared["family"], f"{artifact.cell_id}: family drift")
    expected = set(artifact.expected_files)
    returned = set(artifact.returned_files)
    _require(bool(expected), f"{artifact.cell_id}: no expected files")
    computed_recall = len(expected & returned) / len(expected)
    _require(
        math.isclose(artifact.required_file_recall, computed_recall, abs_tol=1e-12),
        f"{artifact.cell_id}: required-file recall does not reconcile",
    )
    if artifact.status == "failure":
        _require(artifact.required_file_recall == 0.0, f"{artifact.cell_id}: failure has recall")
        _require(not artifact.returned_files, f"{artifact.cell_id}: failure has returned files")
        _require(artifact.payload_sha256 is None, f"{artifact.cell_id}: failure has payload")
        _require(artifact.receipt is None, f"{artifact.cell_id}: failure has receipt")
        _require(artifact.error is not None, f"{artifact.cell_id}: failure lacks error")
    else:
        _require(artifact.payload_sha256 is not None, f"{artifact.cell_id}: success lacks payload")
        _require(
            artifact.payload_bytes is not None, f"{artifact.cell_id}: success lacks payload size"
        )
        _require(artifact.token_count is not None, f"{artifact.cell_id}: success lacks token count")
        _require(artifact.latency_ms is not None, f"{artifact.cell_id}: success lacks latency")
        _require(artifact.error is None, f"{artifact.cell_id}: success has error")
        if artifact.arm == CONTROL_ARM:
            _require(artifact.receipt is None, f"{artifact.cell_id}: control has receipt")
        else:
            _require(artifact.receipt is not None, f"{artifact.cell_id}: treatment lacks receipt")


def _validate_treatment_receipt(artifact: ScopeAwareCellArtifact) -> None:
    _require(artifact.arm == TREATMENT_ARM, "receipt validation requires treatment")
    if artifact.status == "failure":
        return
    receipt = artifact.receipt
    _require(receipt is not None, f"{artifact.cell_id}: treatment lacks receipt")
    try:
        if artifact.kind == "single_scope_control":
            parsed = SingleScopeReceipt.model_validate(receipt)
            _require(
                parsed.payload_sha256 == artifact.payload_sha256,
                f"{artifact.cell_id}: single-scope receipt payload drift",
            )
        else:
            parsed = MultiScopeReceipt.model_validate(receipt)
            _require(
                parsed.mode == "scope_aware_ranking", f"{artifact.cell_id}: wrong receipt mode"
            )
    except ValidationError as exc:
        raise ScopeAwareEvidenceError(f"{artifact.cell_id}: invalid scope receipt: {exc}") from exc
    _require(parsed.task_id == artifact.task_id, f"{artifact.cell_id}: receipt task drift")
    _require(
        parsed.repository_id == artifact.repository_id, f"{artifact.cell_id}: receipt repo drift"
    )
    _require(parsed.arm == TREATMENT_ARM, f"{artifact.cell_id}: receipt arm drift")


def _load_raw_artifacts(
    evidence_dir: Path, campaign_dir: Path
) -> tuple[
    dict[str, ScopeAwareCellArtifact],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    population, cells_payload, manifest = _load_campaign(campaign_dir)
    cells = _cell_entries(cells_payload)
    declared_by_id = {cast("str", cell["cell_id"]): cell for cell in cells}
    expected = _expected_artifacts(evidence_dir, cells)
    actual = _artifact_files(evidence_dir)
    expected_paths = set(expected.values())
    _require(
        actual == expected_paths,
        f"raw evidence paths mismatch: expected {len(expected_paths)}, observed {len(actual)}",
    )
    artifacts: dict[str, ScopeAwareCellArtifact] = {}
    for cell_id, path in expected.items():
        artifact = load_cell_artifact(path)
        _require(cell_id not in artifacts, f"duplicate result cell {cell_id}")
        _validate_artifact(artifact, declared_by_id[cell_id])
        if artifact.arm == TREATMENT_ARM:
            _validate_treatment_receipt(artifact)
        artifacts[cell_id] = artifact
    _require(len(artifacts) == len(cells), "raw evidence is missing a declared cell")
    for arm in ARM_IDS:
        receipt_path = evidence_dir / arm / RUN_RECEIPT_FILENAME
        receipt = _load_json(receipt_path)
        _require(receipt.get("campaign_id") == CAMPAIGN_ID, f"{arm}: receipt campaign drift")
        _require(receipt.get("arm") == arm, f"{arm}: receipt arm drift")
        arm_cells = sum(artifact.arm == arm for artifact in artifacts.values())
        _require(receipt.get("planned_cells") == arm_cells, f"{arm}: receipt planned count drift")
        _require(
            receipt.get("successes", 0) + receipt.get("failures", 0) == arm_cells,
            f"{arm}: receipt outcome count drift",
        )
    return artifacts, declared_by_id, population, cells_payload, manifest


def validate_scope_aware_evidence(
    evidence_dir: Path, *, campaign_dir: Path
) -> ScopeAwareEvidenceCoverage:
    """Fail closed on missing, duplicate, undeclared, or malformed R30 raw cells."""
    artifacts, _declared, _population, cells_payload, _manifest = _load_raw_artifacts(
        evidence_dir, campaign_dir
    )
    successes = sum(artifact.status == "success" for artifact in artifacts.values())
    failures = len(artifacts) - successes
    planned = cells_payload.get("planned_cells")
    if not isinstance(planned, int):
        raise ScopeAwareEvidenceError("cells.json: planned_cells missing")
    _require(len(artifacts) == planned, f"coverage {len(artifacts)}/{planned}")
    return ScopeAwareEvidenceCoverage(
        cells=len(artifacts),
        planned_cells=planned,
        successes=successes,
        failures=failures,
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    _require(bool(sorted_values), "cannot percentile an empty bootstrap distribution")
    _require(0.0 <= probability <= 1.0, "invalid percentile probability")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _bootstrap_intervals(repository_means: dict[str, float]) -> tuple[float, float, float, float]:
    ordered = [repository_means[key] for key in sorted(repository_means)]
    _require(len(ordered) == 16, f"primary requires 16 repository means, got {len(ordered)}")
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        sum(ordered[rng.randrange(len(ordered))] for _ in ordered) / len(ordered)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return (
        _percentile(samples, 0.025),
        _percentile(samples, 0.975),
        _percentile(samples, 0.05),
        _percentile(samples, 0.95),
    )


def _p95(values: list[float]) -> float | None:
    return _percentile(sorted(values), 0.95) if values else None


def _artifact_digest_ledger(evidence_dir: Path) -> dict[str, str]:
    paths = sorted(
        path for arm in ARM_IDS for path in (evidence_dir / arm).rglob("*.json") if path.is_file()
    )
    return {path.relative_to(evidence_dir).as_posix(): _sha256(path) for path in paths}


def build_scope_aware_ledger(evidence_dir: Path, *, campaign_dir: Path) -> dict[str, Any]:
    """Re-derive the raw evidence coverage and every source-artifact digest."""
    coverage = validate_scope_aware_evidence(evidence_dir, campaign_dir=campaign_dir)
    manifest = _load_json(campaign_dir / MANIFEST_FILENAME)
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "coverage": {
            "unique_cells": coverage.cells,
            "planned_cells": coverage.planned_cells,
            "successes": coverage.successes,
            "failures": coverage.failures,
        },
        "campaign_digests": {
            path.name: _sha256(path)
            for path in sorted(campaign_dir.glob("*.json"))
            if path.name != "candidate_identity.json"
        },
        "candidate_identity_sha256": _sha256(campaign_dir / "candidate_identity.json"),
        "canonical_manifest_sha256": manifest["manifest_sha256"],
        "raw_artifact_sha256": _artifact_digest_ledger(evidence_dir),
    }


def _language_subgroups(
    repository_means: dict[str, float], population: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for repository in cast("list[dict[str, Any]]", population["repositories"]):
        repo_id = cast("str", repository["repo_id"])
        for language in cast("list[str]", repository["languages"]):
            groups[language].append(repo_id)
    result: dict[str, dict[str, Any]] = {}
    for language, repositories in sorted(groups.items()):
        means = [repository_means[repo] for repo in repositories]
        mean = sum(means) / len(means)
        result[language] = {
            "repositories": sorted(repositories),
            "repository_weighted_mean_difference": mean,
            "regresses": mean < 0.0,
            "region_metric": "not_applicable_no_region_labels",
            "line_metric": "not_applicable_no_line_labels",
        }
    return result


def analyze_scope_aware_evidence(evidence_dir: Path, *, campaign_dir: Path) -> dict[str, Any]:
    """Compute the immutable R30 primary, intervals, gates, and disposition."""
    artifacts, declared, population, _cells_payload, manifest = _load_raw_artifacts(
        evidence_dir, campaign_dir
    )
    coverage = validate_scope_aware_evidence(evidence_dir, campaign_dir=campaign_dir)
    pair_by_task: dict[str, dict[str, ScopeAwareCellArtifact]] = defaultdict(dict)
    for artifact in artifacts.values():
        pair_by_task[artifact.task_id][artifact.arm] = artifact

    treatment_entries = [cell for cell in declared.values() if cell["kind"] == "treatment"]
    single_scope_entries = [
        cell for cell in declared.values() if cell["kind"] == "single_scope_control"
    ]
    _require(len(treatment_entries) == 2048, "primary selector is not 2,048 treatment pairs")
    _require(len(single_scope_entries) == 16, "single-scope invariant selector is not 16 pairs")

    repository_deltas: dict[str, list[float]] = defaultdict(list)
    family_deltas: dict[str, list[float]] = defaultdict(list)
    zero_recall_tasks: list[str] = []
    for entry in treatment_entries:
        task_id = cast("str", entry["task_id"])
        pair = pair_by_task[task_id]
        _require(set(pair) == set(ARM_IDS), f"{task_id}: incomplete arm pair")
        control = pair[CONTROL_ARM]
        treatment = pair[TREATMENT_ARM]
        difference = treatment.required_file_recall - control.required_file_recall
        repository_deltas[treatment.repository_id].append(difference)
        family_deltas[treatment.family].append(difference)
        if treatment.required_file_recall == 0.0 and control.required_file_recall > 0.0:
            zero_recall_tasks.append(task_id)
    repository_means = {
        repository_id: sum(values) / len(values)
        for repository_id, values in sorted(repository_deltas.items())
    }
    _require(
        all(len(values) == 128 for values in repository_deltas.values()), "primary repo size drift"
    )
    point_estimate = sum(repository_means.values()) / len(repository_means)
    lower_95, upper_95, lower_90, upper_90 = _bootstrap_intervals(repository_means)

    single_scope_payload_failures: list[str] = []
    single_scope_deltas: list[float] = []
    for entry in single_scope_entries:
        task_id = cast("str", entry["task_id"])
        pair = pair_by_task[task_id]
        _require(set(pair) == set(ARM_IDS), f"{task_id}: incomplete arm pair")
        control = pair[CONTROL_ARM]
        treatment = pair[TREATMENT_ARM]
        expected_payload = manifest["control_payload_sha256"][control.repository_id]
        if (
            control.status != "success"
            or treatment.status != "success"
            or control.payload_sha256 != expected_payload
            or treatment.payload_sha256 != expected_payload
        ):
            single_scope_payload_failures.append(task_id)
        single_scope_deltas.append(treatment.required_file_recall - control.required_file_recall)

    multi_scope_receipt_failures: list[str] = []
    for entry in treatment_entries:
        artifact = artifacts[cast("str", entry["cell_id"])]
        if artifact.status != "success":
            multi_scope_receipt_failures.append(artifact.task_id)
            continue
        try:
            _validate_treatment_receipt(artifact)
        except ScopeAwareEvidenceError:
            multi_scope_receipt_failures.append(artifact.task_id)

    subgroup = _language_subgroups(repository_means, population)
    subgroup_regressions = [
        name for name, values in subgroup.items() if cast("bool", values["regresses"])
    ]
    single_scope_mean = sum(single_scope_deltas) / len(single_scope_deltas)
    if single_scope_mean < 0.0:
        subgroup_regressions.append("single_scope_control")
    treatment_latencies = [
        artifact.latency_ms
        for artifact in artifacts.values()
        if artifact.arm == TREATMENT_ARM
        and artifact.status == "success"
        and artifact.latency_ms is not None
    ]
    latency_p95 = _p95(treatment_latencies)

    gates = {
        "complete_unique_cells": coverage.cells == coverage.planned_cells,
        "single_scope_payload": not single_scope_payload_failures,
        "multi_scope_receipts": not multi_scope_receipt_failures,
        "no_subgroup_regression": not subgroup_regressions,
        "no_new_zero_recall": not zero_recall_tasks,
        "treatment_warm_p95": latency_p95 is not None and latency_p95 <= WARM_P95_LIMIT_MS,
    }
    beneficial = lower_95 > 0.0
    non_inferior = lower_95 >= NIM
    equivalent = lower_90 >= -EQM and upper_90 <= EQM
    minimum_worthwhile = point_estimate >= MWG
    binding_failures = [name for name, passed in gates.items() if not passed]
    if not binding_failures and minimum_worthwhile and beneficial:
        disposition = (
            "BENCHMARK CONTINUATION ELIGIBLE — RELEASE: none — PRODUCT PROMOTION: forbidden"
        )
        terminal_reason = (
            "all binding gates pass; point estimate clears MWG; 95% lower bound is above zero"
        )
    else:
        if binding_failures:
            terminal_reason = f"binding gate: {', '.join(binding_failures)}"
        elif not minimum_worthwhile:
            terminal_reason = "point estimate below +0.05 MWG"
        elif not beneficial:
            terminal_reason = "95% beneficial interval is compatible with zero"
        elif non_inferior or equivalent:
            terminal_reason = "result is compatible only with NIM −0.02 or EQM ±0.02"
        else:
            terminal_reason = "frozen continuation rule not satisfied"
        disposition = f"EVIDENCE NO-GO — RELEASE: none — REASON: {terminal_reason}"
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "primary": {
            "selector": {"kind": "treatment", "pairs": 2048, "single_scope_excluded": 16},
            "aggregation": "equal-weight mean of 16 repository means",
            "repository_means": repository_means,
            "point_estimate": point_estimate,
            "bootstrap": {
                "method": "whole-repository percentile bootstrap",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "beneficial_95_interval": [lower_95, upper_95],
                "tost_equivalence_90_interval": [lower_90, upper_90],
            },
            "margins": {"MWG": MWG, "NIM": NIM, "EQM": EQM},
            "classifications": {
                "minimum_worthwhile_gain": minimum_worthwhile,
                "beneficial": beneficial,
                "non_inferior": non_inferior,
                "equivalent": equivalent,
            },
        },
        "invariants": {
            "single_scope_pairs": 16,
            "single_scope_payload_failures": sorted(single_scope_payload_failures),
            "multi_scope_receipt_failures": sorted(multi_scope_receipt_failures),
            "subgroups": subgroup,
            "single_scope_mean_difference": single_scope_mean,
            "subgroup_regressions": sorted(subgroup_regressions),
            "new_zero_recall_tasks": sorted(zero_recall_tasks),
            "treatment_warm_p95_ms": latency_p95,
            "treatment_warm_p95_limit_ms": WARM_P95_LIMIT_MS,
        },
        "family_means": {
            family: sum(values) / len(values) for family, values in sorted(family_deltas.items())
        },
        "coverage": {
            "unique_cells": coverage.cells,
            "planned_cells": coverage.planned_cells,
            "successes": coverage.successes,
            "failures": coverage.failures,
        },
        "binding_gates": gates,
        "disposition": disposition,
        "terminal_reason": terminal_reason,
    }


def generated_evidence_files(evidence_dir: Path, *, campaign_dir: Path) -> dict[Path, bytes]:
    """Return all derived R30 artifacts as deterministic bytes without writing."""
    from archex.benchmark.scope_aware_report import render_scope_aware_report

    ledger = build_scope_aware_ledger(evidence_dir, campaign_dir=campaign_dir)
    analysis = analyze_scope_aware_evidence(evidence_dir, campaign_dir=campaign_dir)
    return {
        evidence_dir / LEDGER_FILENAME: canonical_json_bytes(ledger),
        evidence_dir / ANALYSIS_FILENAME: canonical_json_bytes(analysis),
        evidence_dir / REPORT_FILENAME: render_scope_aware_report(ledger, analysis).encode(),
    }


def write_generated_evidence(
    evidence_dir: Path, *, campaign_dir: Path, check: bool = False
) -> None:
    """Write or byte-compare the deterministic R30 ledger, analysis, and report."""
    generated = generated_evidence_files(evidence_dir, campaign_dir=campaign_dir)
    for path, expected in generated.items():
        if check:
            try:
                observed = path.read_bytes()
            except OSError as exc:
                raise ScopeAwareEvidenceError(f"missing generated artifact {path}: {exc}") from exc
            _require(observed == expected, f"generated artifact drift: {path}")
        else:
            path.write_bytes(expected)
