"""Fail-closed validation for the R29 Candidate A source binding."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from archex.benchmark.scope_aware_campaign import (
    CAMPAIGN_ID,
    MANIFEST_FILENAME,
    ScopeAwareCampaignError,
    validate_scope_aware_campaign,
)

IDENTITY_FILENAME = "candidate_identity.json"
EXPECTED_CANDIDATE_REVISION = "f0ebd9bade164abfc3913419f06ea90709a20108"
EXPECTED_IDENTITY_FILE_SHA256 = "80729d35addc6af8679279cd0bcf78fa31c97a308f384b214665e5a1f2e6e938"
EXPECTED_R27_MANIFEST_SHA256 = "20caed725c4091a4ad675ead0d09f8bbd8ee6fcd1844b631480658a862175ca2"
EXPECTED_R27_MANIFEST_FILE_SHA256 = (
    "e4a35c04c468671b971dc920c250f0003ced0fbc036330aa80cd354998717e4b"
)
EXPECTED_IMPLEMENTATION_FILES = frozenset(
    {
        "src/archex/benchmark/models.py",
        "src/archex/benchmark/runner.py",
        "src/archex/benchmark/scope_aware_candidate.py",
        "src/archex/benchmark/scope_aware_receipt.py",
        "src/archex/benchmark/scope_aware_strategy.py",
        "src/archex/benchmark/strategies.py",
    }
)
_HEX_40 = r"^[0-9a-f]{40}$"
_HEX_64 = r"^[0-9a-f]{64}$"


class ScopeAwareIdentityError(ValueError):
    """Raised when the R29 candidate identity cannot be reproduced."""


class _IdentityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DependencyIdentity(_IdentityModel):
    """Dependency inputs resolved by the committed uv lock."""

    package_manager: Literal["uv"]
    project_file: Literal["pyproject.toml"]
    project_file_sha256: str = Field(pattern=_HEX_64)
    requires_python: Literal[">=3.11"]
    lock_file: Literal["uv.lock"]
    lock_format_version: Literal[1]
    lock_revision: Literal[1]


class CleanTreeProof(_IdentityModel):
    """Review-time proof that the immutable candidate tree was bound unchanged."""

    binding_checkout: Literal["clean"]
    candidate_source_diff: Literal["empty"]
    candidate_tree: str = Field(pattern=_HEX_40)


class ScopeAwareCandidateIdentity(_IdentityModel):
    """Only the source and dependency identity allowed by the R27 manifest."""

    schema_version: Literal[1]
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"]
    r27_manifest_sha256: str = Field(pattern=_HEX_64)
    r27_manifest_file_sha256: str = Field(pattern=_HEX_64)
    revision: str = Field(pattern=_HEX_40)
    implementation_file_sha256: dict[str, str]
    dependency_identity: DependencyIdentity
    lock_file_sha256: str = Field(pattern=_HEX_64)
    clean_tree_proof: CleanTreeProof


class ScopeAwareIdentityCoverage(_IdentityModel):
    """Validated immutable candidate identity summary."""

    revision: str
    implementation_files: int


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScopeAwareIdentityError(message)


def _git(repo_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ScopeAwareIdentityError(f"git {' '.join(args)} failed{suffix}")
    return completed.stdout


def _load_identity(path: Path) -> tuple[ScopeAwareCandidateIdentity, bytes]:
    try:
        encoded = path.read_bytes()
        identity = ScopeAwareCandidateIdentity.model_validate_json(encoded)
    except (OSError, ValidationError) as exc:
        raise ScopeAwareIdentityError(f"Invalid {path.name}: {exc}") from exc
    return identity, encoded


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = path.read_bytes()
        payload = cast("dict[str, Any]", json.loads(encoded))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeAwareIdentityError(f"Invalid {path.name}: {exc}") from exc
    return payload, encoded


def _validate_dependency_identity(
    identity: ScopeAwareCandidateIdentity,
    *,
    repo_root: Path,
) -> None:
    dependency = identity.dependency_identity
    project_bytes = _git(repo_root, "show", f"{identity.revision}:{dependency.project_file}")
    _require(
        _sha256(project_bytes) == dependency.project_file_sha256,
        "candidate project dependency digest drift",
    )
    try:
        project = tomllib.loads(project_bytes.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ScopeAwareIdentityError(f"Invalid candidate pyproject.toml: {exc}") from exc
    _require(
        project.get("project", {}).get("requires-python") == dependency.requires_python,
        "candidate Python requirement drift",
    )

    lock_bytes = _git(repo_root, "show", f"{identity.revision}:{dependency.lock_file}")
    _require(_sha256(lock_bytes) == identity.lock_file_sha256, "candidate lock digest drift")
    try:
        lock = tomllib.loads(lock_bytes.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ScopeAwareIdentityError(f"Invalid candidate uv.lock: {exc}") from exc
    _require(lock.get("version") == dependency.lock_format_version, "uv lock format drift")
    _require(lock.get("revision") == dependency.lock_revision, "uv lock revision drift")
    _require(
        lock.get("requires-python") == dependency.requires_python,
        "uv lock Python requirement drift",
    )


def _validate_git_identity(
    identity: ScopeAwareCandidateIdentity,
    *,
    repo_root: Path,
) -> None:
    _require(
        identity.revision == EXPECTED_CANDIDATE_REVISION,
        "candidate revision differs from the externally merged R28 revision",
    )
    _require(
        _git(repo_root, "cat-file", "-t", identity.revision).decode().strip() == "commit",
        "candidate revision does not resolve to a commit",
    )
    tree = _git(repo_root, "rev-parse", f"{identity.revision}^{{tree}}").decode().strip()
    _require(tree == identity.clean_tree_proof.candidate_tree, "candidate tree identity drift")

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", identity.revision, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    _require(ancestor.returncode == 0, "candidate revision is not an ancestor of HEAD")

    paths = set(identity.implementation_file_sha256)
    _require(paths == set(EXPECTED_IMPLEMENTATION_FILES), "candidate implementation file set drift")
    for path in sorted(paths):
        expected_sha256 = identity.implementation_file_sha256[path]
        _require(
            bool(re.fullmatch(_HEX_64, expected_sha256)),
            f"invalid implementation digest for {path}",
        )
        revision_bytes = _git(repo_root, "show", f"{identity.revision}:{path}")
        _require(
            _sha256(revision_bytes) == expected_sha256,
            f"candidate source digest drift: {path}",
        )

    _validate_dependency_identity(identity, repo_root=repo_root)

    diff = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            identity.revision,
            "--",
            *sorted(paths),
            identity.dependency_identity.project_file,
            identity.dependency_identity.lock_file,
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    _require(diff.returncode == 0, "candidate source diff is not empty")
    _require(
        not _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all").strip(),
        "candidate binding checkout is dirty",
    )


def validate_scope_aware_identity(
    directory: Path,
    *,
    repo_root: Path | None = None,
) -> ScopeAwareIdentityCoverage:
    """Validate the R29 identity artifact against immutable Git and R27 state."""
    try:
        validate_scope_aware_campaign(directory)
    except ScopeAwareCampaignError as exc:
        raise ScopeAwareIdentityError(str(exc)) from exc

    for child in directory.iterdir():
        _require("result" not in child.name.lower(), f"result artifact is forbidden: {child.name}")

    identity_path = directory / IDENTITY_FILENAME
    identity, identity_bytes = _load_identity(identity_path)
    manifest, manifest_bytes = _load_manifest(directory / MANIFEST_FILENAME)

    _require(identity.campaign_id == CAMPAIGN_ID, "candidate campaign identity drift")
    _require(
        identity.r27_manifest_sha256 == EXPECTED_R27_MANIFEST_SHA256,
        "R27 canonical manifest digest drift",
    )
    _require(
        identity.r27_manifest_file_sha256 == EXPECTED_R27_MANIFEST_FILE_SHA256,
        "R27 manifest file digest drift",
    )
    _require(
        _sha256(manifest_bytes) == identity.r27_manifest_file_sha256,
        "bound R27 manifest file digest drift",
    )
    _require(
        manifest.get("manifest_sha256") == identity.r27_manifest_sha256,
        "bound R27 canonical manifest digest drift",
    )

    root = (repo_root or Path.cwd()).resolve()
    _validate_git_identity(identity, repo_root=root)
    _require(
        _sha256(identity_bytes) == EXPECTED_IDENTITY_FILE_SHA256,
        "candidate identity artifact digest drift",
    )
    return ScopeAwareIdentityCoverage(
        revision=identity.revision,
        implementation_files=len(identity.implementation_file_sha256),
    )
