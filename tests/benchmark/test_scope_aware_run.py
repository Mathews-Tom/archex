"""Regression tests for R30's frozen run and terminal report primitives."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from archex.benchmark.scope_aware_evidence import (
    _bootstrap_intervals,  # pyright: ignore[reportPrivateUsage]
    _percentile,  # pyright: ignore[reportPrivateUsage]
)
from archex.benchmark.scope_aware_report import render_scope_aware_report
from archex.benchmark.scope_aware_run import (
    _acquire_pinned_checkout,  # pyright: ignore[reportPrivateUsage]
)


def _commit_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    (path / "package.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "package.json"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "--message=fixture",
        ],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_pinned_checkout_reuses_single_scratch_clone_for_both_corpora(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    commit = _commit_repo(origin)
    repository = {
        "repo_id": "fixture-repo",
        "repository": "fixture/repo",
        "commit": commit,
        "url": str(origin),
    }
    scratch = tmp_path / "scratch"

    first = _acquire_pinned_checkout(repository, repo_cache=None, scratch=scratch)
    second = _acquire_pinned_checkout(repository, repo_cache=None, scratch=scratch)

    assert first == second
    assert (first / ".git").is_dir()


def test_cluster_bootstrap_is_seeded_and_uses_linear_percentiles() -> None:
    means = {f"repo-{index:02d}": index / 100 for index in range(16)}

    first = _bootstrap_intervals(means)
    second = _bootstrap_intervals(means)

    assert first == second
    assert first[0] <= first[1]
    assert first[2] <= first[3]
    assert _percentile([0.0, 1.0], 0.5) == 0.5


def test_terminal_report_states_frozen_selector_intervals_and_promotion_boundary() -> None:
    ledger: dict[str, Any] = {
        "campaign_id": "r27-scope-aware-monorepo-ranking",
        "canonical_manifest_sha256": "a" * 64,
        "candidate_identity_sha256": "b" * 64,
    }
    analysis: dict[str, Any] = {
        "disposition": "EVIDENCE NO-GO — RELEASE: none — REASON: point estimate below +0.05 MWG",
        "terminal_reason": "point estimate below +0.05 MWG",
        "coverage": {"unique_cells": 4128, "planned_cells": 4128, "successes": 4128, "failures": 0},
        "primary": {
            "point_estimate": 0.01,
            "bootstrap": {
                "resamples": 10000,
                "seed": 20260913,
                "beneficial_95_interval": [-0.01, 0.02],
                "tost_equivalence_90_interval": [-0.01, 0.01],
            },
            "margins": {"MWG": 0.05, "NIM": -0.02, "EQM": 0.02},
            "classifications": {
                "minimum_worthwhile_gain": False,
                "beneficial": False,
                "non_inferior": True,
                "equivalent": True,
            },
        },
        "invariants": {
            "single_scope_payload_failures": [],
            "multi_scope_receipt_failures": [],
            "new_zero_recall_tasks": [],
            "treatment_warm_p95_ms": 1.0,
            "treatment_warm_p95_limit_ms": 3000.0,
            "subgroups": {},
            "single_scope_mean_difference": 0.0,
            "subgroup_regressions": [],
        },
        "binding_gates": {
            "complete_unique_cells": True,
            "single_scope_payload": True,
            "multi_scope_receipts": True,
            "no_subgroup_regression": True,
            "no_new_zero_recall": True,
            "treatment_warm_p95": True,
        },
    }

    report = render_scope_aware_report(ledger, analysis)

    assert "Primary selector:** exactly `kind=treatment`, 2,048 paired tasks" in report
    assert "equal weight" in report
    assert "95% beneficial/non-inferiority interval" in report
    assert "90% TOST equivalence interval" in report
    assert "does not authorize promotion" in report
