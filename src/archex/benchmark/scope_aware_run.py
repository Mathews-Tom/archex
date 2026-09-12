"""One-shot frozen executor for the R30 scope-aware campaign run.

This module executes exactly one arm of the immutable R27 cell matrix. It
never edits protocol artifacts, never excludes or repairs a cell, and fails
closed on any pre-existing output so a measured cell can never be rerun or
overwritten under the same campaign identity. Every declared cell produces
exactly one artifact: a valid measurement or a recorded failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from archex.api import query
from archex.benchmark.scope_aware_campaign import (
    CAMPAIGN_ID,
    CELLS_FILENAME,
    MANIFEST_FILENAME,
    PROTOCOL_ID,
    ScopeAwareCampaignError,
    validate_scope_aware_campaign,
)
from archex.benchmark.scope_aware_identity import (
    IDENTITY_FILENAME,
    ScopeAwareIdentityError,
    validate_scope_aware_identity,
)
from archex.benchmark.scope_aware_receipt import (
    canonical_context_payload,
)
from archex.benchmark.scope_aware_strategy import (
    execute_scope_aware_query,
    frozen_scope_aware_config,
    frozen_scope_aware_index_config,
)
from archex.models import Config, ContextBundle, IndexConfig, RepoSource

CONTROL_ARM = "archex_query_control"
TREATMENT_ARM = "scope_aware_candidate"
ARM_IDS = (CONTROL_ARM, TREATMENT_ARM)
CELLS_DIRNAME = "cells"
RUN_RECEIPT_FILENAME = "run_receipt.json"
REPO_CACHE_ENV = "ARCHEX_SCOPE_AWARE_REPO_CACHE"
_SNAPSHOT_DATE = "2026-09-11T20:00:00Z"
_HEX_64 = r"^[0-9a-f]{64}$"

ArmId = Literal["archex_query_control", "scope_aware_candidate"]
CellKind = Literal["treatment", "single_scope_control"]
CellFamily = Literal[
    "lexical_collision",
    "cross_scope_dependency",
    "weak_participation_control",
    "single_scope_control",
]
CorpusKind = Literal["workspace", "control_scope"]


class ScopeAwareRunError(ValueError):
    """Raised when the frozen run cannot start or continue safely."""


class ScopeAwareCellArtifact(BaseModel):
    """One immutable measured cell: a valid result or a recorded failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"] = CAMPAIGN_ID
    protocol_id: Literal["R26-CANDIDATE-A"] = PROTOCOL_ID
    cell_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    arm: ArmId
    kind: CellKind
    family: CellFamily
    corpus: CorpusKind
    sequence: int = Field(ge=1)
    status: Literal["success", "failure"]
    question: str = Field(min_length=1)
    expected_files: tuple[str, ...] = Field(min_length=1)
    returned_files: tuple[str, ...] = ()
    required_file_recall: float = Field(ge=0.0, le=1.0)
    payload_sha256: str | None = Field(default=None, pattern=_HEX_64)
    payload_bytes: int | None = Field(default=None, gt=0)
    token_count: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    receipt: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _PlannedCell:
    cell_id: str
    task_id: str
    repository_id: str
    arm: str
    kind: CellKind
    family: CellFamily
    sequence: int


@dataclass
class _CorpusState:
    """Lazily built snapshot corpus with its isolated per-arm cache."""

    kind: CorpusKind
    root: Path | None = None
    cache_dir: Path | None = None
    warmed: bool = False
    setup_error: str | None = None
    warmup_ms: float | None = None


@dataclass(frozen=True)
class ScopeAwareRunSummary:
    """Coverage counters for one completed arm execution."""

    arm: str
    cells: int
    successes: int
    failures: int
    output_dir: Path


@dataclass
class _RepositoryPlan:
    record: dict[str, Any]
    cells: list[_PlannedCell] = field(default_factory=lambda: list[_PlannedCell]())


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize evidence deterministically: sorted keys, compact, UTF-8."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def cell_artifact_path(output_dir: Path, repository_id: str, task_id: str) -> Path:
    """Return the single declared artifact path for one cell."""
    return output_dir / CELLS_DIRNAME / repository_id / f"{task_id}.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScopeAwareRunError(message)


def _run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown git error"
        raise ScopeAwareRunError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _snapshot_commit(target: Path, message: str) -> None:
    _run_git("init", "--quiet", cwd=target)
    _run_git("add", "--all", cwd=target)
    completed = subprocess.run(
        (
            "git",
            "-c",
            "user.name=R30 Runner",
            "-c",
            "user.email=r30@example.invalid",
            "commit",
            "--quiet",
            f"--message={message}",
        ),
        cwd=target,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": _SNAPSHOT_DATE,
            "GIT_COMMITTER_DATE": _SNAPSHOT_DATE,
        },
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown git error"
        raise ScopeAwareRunError(f"snapshot commit failed for {target}: {detail}")


def _copy_pinned_path(source_root: Path, destination_root: Path, relative_path: str) -> None:
    source = source_root / relative_path
    destination = destination_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        shutil.copy2(source, destination)
    else:
        raise ScopeAwareRunError(f"missing pinned corpus path: {source}")


def _acquire_pinned_checkout(
    repository: dict[str, Any],
    *,
    repo_cache: Path | None,
    scratch: Path,
) -> Path:
    """Return a checkout at the frozen commit, from a verified cache or a fetch."""
    commit = cast("str", repository["commit"])
    if repo_cache is not None:
        owner_repo = cast("str", repository["repository"]).replace("/", "__")
        for name in (owner_repo, cast("str", repository["repo_id"]), f"actual__{owner_repo}"):
            candidate = repo_cache / name
            if candidate.is_dir():
                revision = _run_git("rev-parse", "HEAD", cwd=candidate)
                _require(
                    revision == commit,
                    f"{repository['repo_id']}: cached checkout is {revision}, expected {commit}",
                )
                return candidate

    target = scratch / cast("str", repository["repository"]).replace("/", "__")
    if (target / ".git").is_dir():
        revision = _run_git("rev-parse", "HEAD", cwd=target)
        _require(
            revision == commit,
            f"{repository['repo_id']}: scratch checkout is {revision}, expected {commit}",
        )
        return target

    target.mkdir(parents=True)
    _run_git("init", "--quiet", cwd=target)
    _run_git("remote", "add", "origin", cast("str", repository["url"]), cwd=target)
    _run_git("fetch", "--quiet", "--depth=1", "origin", commit, cwd=target)
    _run_git("checkout", "--quiet", "FETCH_HEAD", cwd=target)
    revision = _run_git("rev-parse", "HEAD", cwd=target)
    _require(
        revision == commit,
        f"{repository['repo_id']}: fetched checkout is {revision}, expected {commit}",
    )
    return target


def _verify_license(repository: dict[str, Any], source_root: Path) -> None:
    license_record = cast("dict[str, Any]", repository["license"])
    license_path = source_root / cast("str", license_record["path"])
    try:
        observed = hashlib.sha256(license_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ScopeAwareRunError(
            f"{repository['repo_id']}: pinned license unreadable: {exc}"
        ) from exc
    _require(
        observed == license_record["sha256"],
        f"{repository['repo_id']}: pinned license digest drift",
    )


def _build_workspace_snapshot(repository: dict[str, Any], source_root: Path, scratch: Path) -> Path:
    workspace = scratch / f"workspace__{repository['repo_id']}"
    workspace.mkdir(parents=True)
    for corpus_path in cast("list[str]", repository["corpus_paths"]):
        _copy_pinned_path(source_root, workspace, corpus_path)
    _snapshot_commit(workspace, "freeze")
    return workspace


def _build_control_snapshot(repository: dict[str, Any], source_root: Path, scratch: Path) -> Path:
    control = cast("dict[str, Any]", repository["control"])
    control_root = scratch / f"control__{repository['repo_id']}"
    control_root.mkdir(parents=True)
    _copy_pinned_path(source_root, control_root, cast("str", control["scope"]))
    _snapshot_commit(control_root, "control")
    return control_root


def _frozen_control_config(languages: list[str], cache_dir: Path) -> Config:
    """Return the exact R27 control-arm runtime configuration."""
    return Config(
        cache=True,
        cache_dir=str(cache_dir),
        languages=languages,
        parallel=False,
        worktree_seed=False,
    )


def _measure_cell(
    *,
    arm: ArmId,
    corpus: _CorpusState,
    task: dict[str, Any],
    cell: _PlannedCell,
    languages: list[str],
    index_config: IndexConfig,
) -> ScopeAwareCellArtifact:
    root = corpus.root
    cache_dir = corpus.cache_dir
    _require(root is not None and cache_dir is not None, "corpus is not initialized")
    assert root is not None
    assert cache_dir is not None
    question = cast("str", task["question"])
    expected_files = tuple(cast("list[str]", task["expected_files"]))
    budget = int(cast("int", task["budget_tokens"]))
    source = RepoSource(local_path=str(root))
    receipt: dict[str, Any] | None = None
    started = time.perf_counter()
    if arm == CONTROL_ARM:
        config = _frozen_control_config(languages, cache_dir)
        bundle: ContextBundle = query(
            source,
            question,
            token_budget=budget,
            config=config,
            index_config=index_config,
            explicit_token_budget=True,
            refresh=False,
        )
    else:
        config = frozen_scope_aware_config(languages=languages, cache_dir=str(cache_dir))
        outcome = execute_scope_aware_query(
            source,
            question,
            repo_root=root,
            task_id=cell.task_id,
            repository_id=cell.repository_id,
            token_budget=budget,
            config=config,
            index_config=index_config,
        )
        bundle = outcome.bundle
        receipt = outcome.receipt.model_dump(mode="json")
    latency_ms = (time.perf_counter() - started) * 1000.0
    payload = canonical_context_payload(bundle)
    returned_files = tuple(sorted({ranked.chunk.file_path for ranked in bundle.chunks}))
    matched = sum(1 for path in expected_files if path in set(returned_files))
    return ScopeAwareCellArtifact(
        cell_id=cell.cell_id,
        task_id=cell.task_id,
        repository_id=cell.repository_id,
        arm=arm,
        kind=cell.kind,
        family=cell.family,
        corpus=corpus.kind,
        sequence=cell.sequence,
        status="success",
        question=question,
        expected_files=expected_files,
        returned_files=returned_files,
        required_file_recall=matched / len(expected_files),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=len(payload),
        token_count=bundle.token_count,
        latency_ms=latency_ms,
        receipt=receipt,
    )


def _failure_artifact(
    *,
    arm: ArmId,
    corpus_kind: CorpusKind,
    task: dict[str, Any],
    cell: _PlannedCell,
    error: str,
) -> ScopeAwareCellArtifact:
    return ScopeAwareCellArtifact(
        cell_id=cell.cell_id,
        task_id=cell.task_id,
        repository_id=cell.repository_id,
        arm=arm,
        kind=cell.kind,
        family=cell.family,
        corpus=corpus_kind,
        sequence=cell.sequence,
        status="failure",
        question=cast("str", task["question"]),
        expected_files=tuple(cast("list[str]", task["expected_files"])),
        returned_files=(),
        required_file_recall=0.0,
        error=error,
    )


def _write_cell_artifact(output_dir: Path, artifact: ScopeAwareCellArtifact) -> None:
    path = cell_artifact_path(output_dir, artifact.repository_id, artifact.task_id)
    _require(not path.exists(), f"measured cell already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(artifact.model_dump(mode="json")))


def _warm_up(
    *,
    arm: ArmId,
    corpus: _CorpusState,
    question: str,
    languages: list[str],
    index_config: IndexConfig,
    budget: int,
) -> None:
    """Execute the single unmeasured index/query warm-up for one corpus."""
    root = corpus.root
    cache_dir = corpus.cache_dir
    _require(root is not None and cache_dir is not None, "corpus is not initialized")
    assert root is not None
    assert cache_dir is not None
    source = RepoSource(local_path=str(root))
    started = time.perf_counter()
    if arm == CONTROL_ARM:
        query(
            source,
            question,
            token_budget=budget,
            config=_frozen_control_config(languages, cache_dir),
            index_config=index_config,
            explicit_token_budget=True,
            refresh=False,
        )
    else:
        execute_scope_aware_query(
            source,
            question,
            repo_root=root,
            task_id="warm-up",
            repository_id="warm-up",
            token_budget=budget,
            config=frozen_scope_aware_config(languages=languages, cache_dir=str(cache_dir)),
            index_config=index_config,
        )
    corpus.warmup_ms = (time.perf_counter() - started) * 1000.0
    corpus.warmed = True


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeAwareRunError(f"Invalid {path}: {exc}") from exc


def _planned_cells_for_arm(cells_payload: dict[str, Any], arm: str) -> list[_PlannedCell]:
    declared = cast("list[dict[str, Any]]", cells_payload["cells"])
    planned: list[_PlannedCell] = []
    for entry in declared:
        if entry["arm"] != arm:
            continue
        planned.append(
            _PlannedCell(
                cell_id=cast("str", entry["cell_id"]),
                task_id=cast("str", entry["task_id"]),
                repository_id=cast("str", entry["repository_id"]),
                arm=cast("str", entry["arm"]),
                kind=cast("CellKind", entry["kind"]),
                family=cast("CellFamily", entry["family"]),
                sequence=len(planned) + 1,
            )
        )
    _require(bool(planned), f"no planned cells declare arm {arm!r}")
    return planned


def _group_by_repository(planned: list[_PlannedCell]) -> list[list[_PlannedCell]]:
    groups: list[list[_PlannedCell]] = []
    seen: set[str] = set()
    for cell in planned:
        if groups and groups[-1][0].repository_id == cell.repository_id:
            groups[-1].append(cell)
            continue
        _require(
            cell.repository_id not in seen,
            f"cells for repository {cell.repository_id!r} are not contiguous",
        )
        seen.add(cell.repository_id)
        groups.append([cell])
    return groups


def _ensure_corpus(
    corpus: _CorpusState,
    *,
    repository: dict[str, Any],
    repo_cache: Path | None,
    scratch: Path,
    arm: ArmId,
) -> None:
    if corpus.root is not None or corpus.setup_error is not None:
        return
    try:
        source_root = _acquire_pinned_checkout(repository, repo_cache=repo_cache, scratch=scratch)
        _verify_license(repository, source_root)
        if corpus.kind == "workspace":
            corpus.root = _build_workspace_snapshot(repository, source_root, scratch)
        else:
            corpus.root = _build_control_snapshot(repository, source_root, scratch)
        cache_dir = scratch / "cache" / arm / cast("str", repository["repo_id"]) / corpus.kind
        cache_dir.mkdir(parents=True)
        corpus.cache_dir = cache_dir
    except (ScopeAwareRunError, OSError) as exc:
        corpus.setup_error = f"{type(exc).__name__}: {exc}"


def _run_repository(
    *,
    arm: ArmId,
    plan: _RepositoryPlan,
    tasks_by_id: dict[str, dict[str, Any]],
    index_config: IndexConfig,
    output_dir: Path,
    repo_cache: Path | None,
    scratch_root: Path,
) -> tuple[int, int, dict[str, Any]]:
    repository = plan.record
    repo_id = cast("str", repository["repo_id"])
    languages = cast("list[str]", repository["languages"])
    scratch = scratch_root / repo_id
    scratch.mkdir(parents=True)
    corpora: dict[CorpusKind, _CorpusState] = {
        "workspace": _CorpusState(kind="workspace"),
        "control_scope": _CorpusState(kind="control_scope"),
    }
    successes = 0
    failures = 0
    try:
        for cell in plan.cells:
            task = tasks_by_id.get(cell.task_id)
            if task is None:
                raise ScopeAwareRunError(f"declared cell has no frozen task: {cell.cell_id}")
            corpus_kind: CorpusKind = (
                "control_scope" if cell.kind == "single_scope_control" else "workspace"
            )
            corpus = corpora[corpus_kind]
            _ensure_corpus(
                corpus,
                repository=repository,
                repo_cache=repo_cache,
                scratch=scratch,
                arm=arm,
            )
            if corpus.setup_error is not None:
                artifact = _failure_artifact(
                    arm=arm,
                    corpus_kind=corpus_kind,
                    task=task,
                    cell=cell,
                    error=f"corpus setup failed: {corpus.setup_error}",
                )
                _write_cell_artifact(output_dir, artifact)
                failures += 1
                continue
            try:
                if not corpus.warmed:
                    _warm_up(
                        arm=arm,
                        corpus=corpus,
                        question=cast("str", task["question"]),
                        languages=languages,
                        index_config=index_config,
                        budget=int(cast("int", task["budget_tokens"])),
                    )
                artifact = _measure_cell(
                    arm=arm,
                    corpus=corpus,
                    task=task,
                    cell=cell,
                    languages=languages,
                    index_config=index_config,
                )
                successes += 1
            except ScopeAwareRunError:
                raise
            except Exception as exc:  # noqa: BLE001 - retain every failure as evidence
                artifact = _failure_artifact(
                    arm=arm,
                    corpus_kind=corpus_kind,
                    task=task,
                    cell=cell,
                    error=f"{type(exc).__name__}: {exc}",
                )
                failures += 1
            _write_cell_artifact(output_dir, artifact)
        receipt = {
            "repository_id": repo_id,
            "commit": repository["commit"],
            "cells": len(plan.cells),
            "warmups": {
                kind: {
                    "executed": state.warmed,
                    "duration_ms": state.warmup_ms,
                    "setup_error": state.setup_error,
                }
                for kind, state in sorted(corpora.items())
            },
        }
        return successes, failures, receipt
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_scope_aware_arm(
    *,
    manifest_path: Path,
    cells_path: Path,
    arm: str,
    output_dir: Path,
    candidate_identity_path: Path | None = None,
    repo_root: Path | None = None,
) -> ScopeAwareRunSummary:
    """Execute one frozen arm exactly once, retaining every success or failure."""
    _require(arm in ARM_IDS, f"unknown arm {arm!r}; expected one of {ARM_IDS}")
    arm_id = cast("ArmId", arm)
    _require(
        os.environ.get("PYTHONHASHSEED") == "0",
        "Set PYTHONHASHSEED=0 before executing a frozen cell",
    )
    _require(
        os.environ.get("ARCHEX_TELEMETRY") == "0",
        "Set ARCHEX_TELEMETRY=0 before executing a frozen cell",
    )
    campaign_dir = manifest_path.parent
    _require(manifest_path.name == MANIFEST_FILENAME, f"manifest must be {MANIFEST_FILENAME}")
    _require(cells_path.name == CELLS_FILENAME, f"cells must be {CELLS_FILENAME}")
    _require(cells_path.parent == campaign_dir, "manifest and cells must share one campaign")

    root = (repo_root or Path.cwd()).resolve()
    try:
        validate_scope_aware_campaign(campaign_dir)
    except ScopeAwareCampaignError as exc:
        raise ScopeAwareRunError(f"frozen campaign is invalid: {exc}") from exc
    if arm_id == TREATMENT_ARM:
        _require(
            candidate_identity_path is not None,
            "treatment arm requires --candidate-identity",
        )
        assert candidate_identity_path is not None
        _require(
            candidate_identity_path.resolve() == (campaign_dir / IDENTITY_FILENAME).resolve(),
            f"candidate identity must be the frozen {IDENTITY_FILENAME}",
        )
        try:
            validate_scope_aware_identity(campaign_dir, repo_root=root)
        except ScopeAwareIdentityError as exc:
            raise ScopeAwareRunError(f"candidate identity is invalid: {exc}") from exc
    else:
        _require(
            candidate_identity_path is None,
            "control arm must not bind a candidate identity",
        )

    population = _load_json(campaign_dir / "population.json")
    cells_payload = _load_json(cells_path)
    index_config = IndexConfig.model_validate(population["index_config"])
    if arm_id == TREATMENT_ARM:
        _require(
            index_config == frozen_scope_aware_index_config(),
            "frozen population index configuration drifted from the candidate contract",
        )

    planned = _planned_cells_for_arm(cells_payload, arm)
    groups = _group_by_repository(planned)
    repositories = {
        cast("str", record["repo_id"]): record
        for record in cast("list[dict[str, Any]]", population["repositories"])
    }
    tasks_by_id = {
        cast("str", task["task_id"]): task
        for task in cast("list[dict[str, Any]]", population["tasks"])
    }

    output_dir = output_dir.resolve()
    _require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        f"output already contains artifacts; a measured cell can never be rerun: {output_dir}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_cache_env = os.environ.get(REPO_CACHE_ENV)
    repo_cache = Path(repo_cache_env).resolve() if repo_cache_env else None

    scratch_root = output_dir / ".scratch"
    scratch_root.mkdir()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    successes = 0
    failures = 0
    repository_receipts: list[dict[str, Any]] = []
    try:
        for group in groups:
            record = repositories.get(group[0].repository_id)
            if record is None:
                raise ScopeAwareRunError(
                    f"declared cells name unknown repository {group[0].repository_id!r}"
                )
            group_successes, group_failures, receipt = _run_repository(
                arm=arm_id,
                plan=_RepositoryPlan(record=record, cells=group),
                tasks_by_id=tasks_by_id,
                index_config=index_config,
                output_dir=output_dir,
                repo_cache=repo_cache,
                scratch_root=scratch_root,
            )
            successes += group_successes
            failures += group_failures
            repository_receipts.append(receipt)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    receipt_payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "planned_cells": len(planned),
        "successes": successes,
        "failures": failures,
        "started_at": started_at,
        "finished_at": finished_at,
        "environment": {
            "python": platform.python_version(),
            "implementation": sys.implementation.name,
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "archex": _distribution_version("archex"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "archex_telemetry": os.environ.get("ARCHEX_TELEMETRY"),
        },
        "archex_revision": _run_git("rev-parse", "HEAD", cwd=root),
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "cells_file_sha256": hashlib.sha256(cells_path.read_bytes()).hexdigest(),
        "repositories": repository_receipts,
    }
    (output_dir / RUN_RECEIPT_FILENAME).write_bytes(canonical_json_bytes(receipt_payload))
    _require(
        successes + failures == len(planned),
        "runner accounting drift: written cells do not cover the planned matrix",
    )
    return ScopeAwareRunSummary(
        arm=arm,
        cells=successes + failures,
        successes=successes,
        failures=failures,
        output_dir=output_dir,
    )


def load_cell_artifact(path: Path) -> ScopeAwareCellArtifact:
    """Load and revalidate one immutable cell artifact."""
    try:
        return ScopeAwareCellArtifact.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ScopeAwareRunError(f"Invalid cell artifact {path}: {exc}") from exc
