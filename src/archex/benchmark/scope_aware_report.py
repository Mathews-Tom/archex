"""Deterministic Markdown renderer for the R30 terminal evidence report."""

from __future__ import annotations

from typing import Any, cast


def _number(value: object, *, digits: int = 6) -> str:
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return "not available"


def _bool(value: object) -> str:
    return "PASS" if value is True else "FAIL"


def render_scope_aware_report(ledger: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Render R30's fixed report from raw-derived ledger and analysis objects."""
    coverage = cast("dict[str, Any]", analysis["coverage"])
    primary = cast("dict[str, Any]", analysis["primary"])
    bootstrap = cast("dict[str, Any]", primary["bootstrap"])
    margins = cast("dict[str, Any]", primary["margins"])
    classifications = cast("dict[str, Any]", primary["classifications"])
    invariants = cast("dict[str, Any]", analysis["invariants"])
    gates = cast("dict[str, Any]", analysis["binding_gates"])
    beneficial = cast("list[float]", bootstrap["beneficial_95_interval"])
    equivalence = cast("list[float]", bootstrap["tost_equivalence_90_interval"])
    payload_failures = cast("list[str]", invariants["single_scope_payload_failures"])
    receipt_failures = cast("list[str]", invariants["multi_scope_receipt_failures"])
    zero_recall = cast("list[str]", invariants["new_zero_recall_tasks"])
    primary_line = (
        "- **Primary selector:** exactly `kind=treatment`, 2,048 paired tasks. "
        "The 16 `kind=single_scope_control` pairs are invariant-only and never "
        "enter the primary estimate."
    )
    aggregation_line = (
        "- **Aggregation:** task-level treatment-minus-control required-file recall "
        "is averaged within each repository; the 16 repository means receive equal weight."
    )
    classification_line = (
        f"- MWG `+{margins['MWG']:.2f}`: {_bool(classifications['minimum_worthwhile_gain'])}; "
        f"beneficial (95% lower bound > 0): {_bool(classifications['beneficial'])}; "
        f"NIM `{margins['NIM']:.2f}`: {_bool(classifications['non_inferior'])}; "
        f"EQM `±{margins['EQM']:.2f}`: {_bool(classifications['equivalent'])}."
    )
    lines = [
        "# R30 — Frozen scope-aware benchmark execution",
        "",
        "## Scope and terminal boundary",
        "",
        "This report is the terminal benchmark evidence for R26 Candidate A, "
        "scope-aware monorepo ranking. It makes no product registration, "
        "default-change, MCP, language, release, or promotion claim. "
        "A continuation-eligible result remains benchmark-only.",
        "",
        f"**Terminal disposition:** {analysis['disposition']}",
        "",
        "## Coverage and immutable identity",
        "",
        f"- Raw evidence: `{coverage['unique_cells']:,}/{coverage['planned_cells']:,}` "
        f"unique declared cells; `{coverage['successes']:,}` successes and "
        f"`{coverage['failures']:,}` recorded failures.",
        f"- Campaign: `{ledger['campaign_id']}`; canonical manifest SHA-256: "
        f"`{ledger['canonical_manifest_sha256']}`.",
        f"- Candidate identity SHA-256: `{ledger['candidate_identity_sha256']}`.",
        "- Every declared cell remains in the denominator. Recorded failures have "
        "required-file recall `0.0`; no cell is excluded, repaired, or rerun.",
        "",
        "## Primary inference",
        "",
        primary_line,
        aggregation_line,
        f"- Point estimate: **{_number(primary['point_estimate'])}**.",
        f"- Bootstrap: {bootstrap['resamples']:,} whole-repository percentile "
        f"resamples at seed `{bootstrap['seed']}`.",
        "- **95% beneficial/non-inferiority interval:** "
        f"[{_number(beneficial[0])}, {_number(beneficial[1])}].",
        "- **90% TOST equivalence interval:** "
        f"[{_number(equivalence[0])}, {_number(equivalence[1])}].",
        classification_line,
        "",
        "## Binding invariant gates",
        "",
    ]
    for name, passed in gates.items():
        lines.append(f"- `{name}`: {_bool(passed)}.")
    lines.extend(
        [
            "",
            "### Single-scope payloads and receipts",
            "",
            f"- Single-scope payload mismatches: `{len(payload_failures)}`.",
            f"- Multi-scope receipt failures: `{len(receipt_failures)}`.",
            f"- New zero-recall treatment tasks: `{len(zero_recall)}`.",
            f"- Treatment warm p95: "
            f"`{_number(invariants['treatment_warm_p95_ms'], digits=3)}` ms; "
            f"frozen limit: `{invariants['treatment_warm_p95_limit_ms']:.0f}` ms.",
            "",
            "### Declared subgroups",
            "",
        ]
    )
    subgroups = cast("dict[str, dict[str, Any]]", invariants["subgroups"])
    for language, values in subgroups.items():
        repository_count = len(cast("list[str]", values["repositories"]))
        lines.append(
            f"- `{language}`: {_number(values['repository_weighted_mean_difference'])} "
            "repository-weighted required-file-recall difference across "
            f"`{repository_count}` repositories; regression: "
            f"{_bool(not values['regresses'])}. Region and line metrics are not "
            "applicable because R27 supplies no region or line labels."
        )
    no_single_scope_regression = "single_scope_control" not in cast(
        "list[str]", invariants["subgroup_regressions"]
    )
    lines.extend(
        [
            f"- `single_scope_control`: "
            f"{_number(invariants['single_scope_mean_difference'])} paired "
            "required-file-recall difference; regression: "
            f"{_bool(no_single_scope_regression)}.",
            "",
            "## Interpretation",
            "",
            f"{analysis['terminal_reason']}.",
            "",
            "A result compatible only with NIM `−0.02` or EQM `±0.02` is "
            "`EVIDENCE NO-GO`; it does not authorize promotion. The frozen "
            "continuation rule requires a point estimate at least `+0.05`, a 95% "
            "beneficial lower bound above zero, and every binding gate to pass.",
            "",
        ]
    )
    return "\n".join(lines)
