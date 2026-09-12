"""Guards for the immutable R29 Candidate A source binding."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from archex.benchmark.scope_aware_identity import (
    EXPECTED_CANDIDATE_REVISION,
    ScopeAwareIdentityError,
    validate_scope_aware_identity,
)
from archex.cli.benchmark_cmd import benchmark_cmd

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = REPO_ROOT / "benchmarks" / "campaigns" / "r27_scope_aware"
IDENTITY_FILENAME = "candidate_identity.json"


def _campaign_copy(tmp_path: Path) -> Path:
    target = tmp_path / "campaign"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CAMPAIGN_DIR, target)
    return target


def _mutate_identity(directory: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = directory / IDENTITY_FILENAME
    payload = cast("dict[str, Any]", json.loads(path.read_text()))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_checked_in_identity_and_cli_validate() -> None:
    coverage = validate_scope_aware_identity(CAMPAIGN_DIR, repo_root=REPO_ROOT)
    assert coverage.revision == EXPECTED_CANDIDATE_REVISION
    assert coverage.implementation_files == 6

    result = CliRunner().invoke(
        benchmark_cmd,
        ["validate", "--kind", "scope-aware-identity", "--input", str(CAMPAIGN_DIR)],
    )
    assert result.exit_code == 0
    assert result.output == (
        "Valid R29 scope-aware candidate identity: "
        f"{EXPECTED_CANDIDATE_REVISION} / 6 implementation files.\n"
    )


@pytest.mark.parametrize("revision", ["main", EXPECTED_CANDIDATE_REVISION[:12]])
def test_branch_names_and_short_revisions_are_rejected(tmp_path: Path, revision: str) -> None:
    directory = _campaign_copy(tmp_path)
    _mutate_identity(directory, lambda payload: payload.__setitem__("revision", revision))

    with pytest.raises(ScopeAwareIdentityError, match="revision"):
        validate_scope_aware_identity(directory, repo_root=REPO_ROOT)


def test_other_full_revision_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    _mutate_identity(directory, lambda payload: payload.__setitem__("revision", "0" * 40))

    with pytest.raises(ScopeAwareIdentityError, match="externally merged R28 revision"):
        validate_scope_aware_identity(directory, repo_root=REPO_ROOT)


def test_missing_implementation_file_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)

    def remove_file(payload: dict[str, Any]) -> None:
        files = cast("dict[str, str]", payload["implementation_file_sha256"])
        files.pop("src/archex/benchmark/scope_aware_candidate.py")

    _mutate_identity(directory, remove_file)

    with pytest.raises(ScopeAwareIdentityError, match="implementation file set drift"):
        validate_scope_aware_identity(directory, repo_root=REPO_ROOT)


def test_implementation_digest_drift_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)

    def replace_digest(payload: dict[str, Any]) -> None:
        files = cast("dict[str, str]", payload["implementation_file_sha256"])
        files["src/archex/benchmark/scope_aware_candidate.py"] = "0" * 64

    _mutate_identity(directory, replace_digest)

    with pytest.raises(ScopeAwareIdentityError, match="candidate source digest drift"):
        validate_scope_aware_identity(directory, repo_root=REPO_ROOT)


def test_dependency_and_lock_digest_drift_are_rejected(tmp_path: Path) -> None:
    project_directory = _campaign_copy(tmp_path / "project")
    _mutate_identity(
        project_directory,
        lambda payload: cast("dict[str, Any]", payload["dependency_identity"]).__setitem__(
            "project_file_sha256", "0" * 64
        ),
    )
    with pytest.raises(ScopeAwareIdentityError, match="project dependency digest drift"):
        validate_scope_aware_identity(project_directory, repo_root=REPO_ROOT)

    lock_directory = _campaign_copy(tmp_path / "lock")
    _mutate_identity(
        lock_directory,
        lambda payload: payload.__setitem__("lock_file_sha256", "0" * 64),
    )
    with pytest.raises(ScopeAwareIdentityError, match="lock digest drift"):
        validate_scope_aware_identity(lock_directory, repo_root=REPO_ROOT)


def test_dirty_binding_checkout_is_rejected() -> None:
    probe = REPO_ROOT / ".r29-dirty-probe"
    probe.write_text("dirty\n")
    try:
        with pytest.raises(ScopeAwareIdentityError, match="checkout is dirty"):
            validate_scope_aware_identity(CAMPAIGN_DIR, repo_root=REPO_ROOT)
    finally:
        probe.unlink(missing_ok=True)


def test_protocol_and_result_fields_are_rejected(tmp_path: Path) -> None:
    for field in ("commands", "results"):
        directory = _campaign_copy(tmp_path / field)
        _mutate_identity(directory, lambda payload, field=field: payload.__setitem__(field, {}))

        with pytest.raises(ScopeAwareIdentityError, match=field):
            validate_scope_aware_identity(directory, repo_root=REPO_ROOT)


def test_result_directory_is_rejected(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    (directory / "results").mkdir()

    with pytest.raises(ScopeAwareIdentityError, match="result artifact is forbidden"):
        validate_scope_aware_identity(directory, repo_root=REPO_ROOT)


def test_identity_file_encoding_is_immutable(tmp_path: Path) -> None:
    directory = _campaign_copy(tmp_path)
    path = directory / IDENTITY_FILENAME
    payload = json.loads(path.read_text())
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(ScopeAwareIdentityError, match="identity artifact digest drift"):
        validate_scope_aware_identity(directory, repo_root=REPO_ROOT)
