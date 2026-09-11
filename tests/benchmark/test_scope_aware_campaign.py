"""Guards for the checked-in R27 scope-aware population freeze."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from archex.benchmark.scope_aware_campaign import (
    ScopeAwareCampaignError,
    validate_scope_aware_population,
)
from archex.cli.benchmark_cmd import benchmark_cmd

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = REPO_ROOT / "benchmarks" / "campaigns" / "r27_scope_aware"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _campaign_copy(tmp_path: Path) -> Path:
    target = tmp_path / "campaign"
    shutil.copytree(CAMPAIGN_DIR, target)
    return target


def _mutate_json(
    directory: Path,
    filename: str,
    mutate: Callable[[dict[str, Any]], None],
    *,
    repair_population_digest: bool = False,
) -> None:
    path = directory / filename
    payload: dict[str, Any] = json.loads(path.read_text())
    mutate(payload)
    if repair_population_digest:
        digest_payload = {
            key: value for key, value in payload.items() if key != "population_sha256"
        }
        payload["population_sha256"] = _canonical_sha256(digest_payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_checked_in_population_and_cli_validate() -> None:
    coverage = validate_scope_aware_population(CAMPAIGN_DIR)
    assert coverage.repositories == 16
    assert coverage.tasks == 2064

    result = CliRunner().invoke(
        benchmark_cmd,
        ["validate", "--kind", "scope-aware-population", "--input", str(CAMPAIGN_DIR)],
    )
    assert result.exit_code == 0
    assert result.output == "Valid R27 scope-aware population: 16 repositories / 2,064 tasks.\n"


def test_unknown_population_field_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    _mutate_json(
        directory,
        "population.json",
        lambda payload: payload.update({"candidate_revision": "unfrozen"}),
        repair_population_digest=True,
    )

    with pytest.raises(ScopeAwareCampaignError, match="candidate_revision"):
        validate_scope_aware_population(directory)


def test_population_digest_drift_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    _mutate_json(
        directory,
        "population.json",
        lambda payload: payload["tasks"][0].update({"question": "post-freeze edit"}),
    )

    with pytest.raises(ScopeAwareCampaignError, match="population digest drift"):
        validate_scope_aware_population(directory)


def test_missing_task_is_rejected_even_with_repaired_digest(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    _mutate_json(
        directory,
        "population.json",
        lambda payload: payload["tasks"].pop(),
        repair_population_digest=True,
    )

    with pytest.raises(ScopeAwareCampaignError, match="2,064 tasks"):
        validate_scope_aware_population(directory)


def test_control_payload_mismatch_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    _mutate_json(
        directory,
        "control_receipts.json",
        lambda payload: payload["repositories"][0]["warm_runs"][1].update(
            {"payload_sha256": "0" * 64}
        ),
    )

    with pytest.raises(ScopeAwareCampaignError, match="warm payload digest drift"):
        validate_scope_aware_population(directory)


def test_underpowered_result_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    _mutate_json(
        directory,
        "power.json",
        lambda payload: payload["result"].update({"power": 0.79}),
    )

    with pytest.raises(ScopeAwareCampaignError, match="power result drift"):
        validate_scope_aware_population(directory)
