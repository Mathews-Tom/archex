"""Create and validate immutable local benchmark evidence directories."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import ValidationError

from archex.benchmark.models import (
    BenchmarkEvidenceManifest,
    BenchmarkReport,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
)

EVIDENCE_MANIFEST_FILENAME = "manifest.json"


class BenchmarkEvidenceError(ValueError):
    """Raised when local benchmark evidence lacks a verifiable identity."""


def task_manifest_digest(tasks_dir: Path) -> str:
    """Return a stable digest of every top-level benchmark task file."""
    task_files = sorted(tasks_dir.glob("*.yaml"))
    if not task_files:
        msg = f"No benchmark task files found in {tasks_dir}"
        raise BenchmarkEvidenceError(msg)

    digest = hashlib.sha256()
    for task_file in task_files:
        digest.update(task_file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(task_file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_revision(repo_root: Path) -> str:
    """Resolve the clean Git revision that produced local benchmark evidence."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if revision.returncode != 0:
        detail = revision.stderr.strip() or "unknown git error"
        msg = f"Cannot resolve benchmark source revision: {detail}"
        raise BenchmarkEvidenceError(msg)

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if status.returncode != 0:
        detail = status.stderr.strip() or "unknown git error"
        msg = f"Cannot inspect benchmark source cleanliness: {detail}"
        raise BenchmarkEvidenceError(msg)
    if status.stdout:
        msg = "Cannot record immutable benchmark evidence from a dirty source tree"
        raise BenchmarkEvidenceError(msg)

    return revision.stdout.strip()


def archex_version() -> str:
    """Return the installed Archex distribution version used by the benchmark run."""
    try:
        installed_version = version("archex")
    except PackageNotFoundError as exc:
        msg = "Cannot resolve the installed archex distribution version"
        raise BenchmarkEvidenceError(msg) from exc
    if not installed_version:
        msg = "Cannot record an empty archex distribution version"
        raise BenchmarkEvidenceError(msg)
    return installed_version


def prepare_evidence_directory(output_dir: Path) -> None:
    """Create an empty output directory without mixing a new run with stale reports."""
    if output_dir.exists():
        if not output_dir.is_dir():
            msg = f"Benchmark output path is not a directory: {output_dir}"
            raise BenchmarkEvidenceError(msg)
        if any(output_dir.iterdir()):
            msg = f"Benchmark output directory is not empty: {output_dir}"
            raise BenchmarkEvidenceError(msg)
        return
    output_dir.mkdir(parents=True)


def build_evidence_manifest(
    reports: list[BenchmarkReport],
    tasks: list[BenchmarkTask],
    strategies: list[Strategy],
    retrieval_options: BenchmarkRetrievalOptions,
    *,
    source_sha: str,
    tasks_dir: Path,
    hardware_advisory: str | None = None,
) -> BenchmarkEvidenceManifest:
    """Build a manifest after asserting exact task and strategy coverage."""
    task_ids = [task.task_id for task in tasks]
    report_by_task = {report.task_id: report for report in reports}
    if len(report_by_task) != len(reports):
        msg = "Benchmark evidence contains duplicate report task IDs"
        raise BenchmarkEvidenceError(msg)
    if set(report_by_task) != set(task_ids):
        missing = sorted(set(task_ids) - set(report_by_task))
        unexpected = sorted(set(report_by_task) - set(task_ids))
        msg = f"Benchmark report coverage mismatch: missing={missing}, unexpected={unexpected}"
        raise BenchmarkEvidenceError(msg)

    expected_strategies = set(strategies)
    for report in reports:
        actual_strategies = {result.strategy for result in report.results}
        if actual_strategies != expected_strategies or len(report.results) != len(
            actual_strategies
        ):
            missing = sorted(strategy.value for strategy in expected_strategies - actual_strategies)
            unexpected = sorted(
                strategy.value for strategy in actual_strategies - expected_strategies
            )
            msg = (
                f"Benchmark strategy coverage mismatch for {report.task_id}: "
                f"missing={missing}, unexpected={unexpected}"
            )
            raise BenchmarkEvidenceError(msg)

    report_hashes = {
        report.task_id: hashlib.sha256(report.model_dump_json(indent=2).encode("utf-8")).hexdigest()
        for report in reports
    }
    return BenchmarkEvidenceManifest(
        source_revision=source_sha,
        archex_version=archex_version(),
        task_manifest_digest=task_manifest_digest(tasks_dir),
        task_ids=task_ids,
        strategies=strategies,
        retrieval_options=retrieval_options,
        generated_at=datetime.now(tz=UTC).isoformat(),
        hardware_advisory=hardware_advisory
        or f"{platform.system()} {platform.release()} {platform.machine()}",
        report_hashes=report_hashes,
    )


def write_evidence_manifest(output_dir: Path, manifest: BenchmarkEvidenceManifest) -> Path:
    """Write the manifest only after the reports already exist in *output_dir*."""
    manifest_path = output_dir / EVIDENCE_MANIFEST_FILENAME
    if manifest_path.exists():
        msg = f"Benchmark evidence manifest already exists: {manifest_path}"
        raise BenchmarkEvidenceError(msg)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path


def validate_evidence_directory(
    output_dir: Path,
    tasks_dir: Path,
    *,
    expected_source_sha: str | None = None,
) -> BenchmarkEvidenceManifest:
    """Validate manifest schema, task identity, report hashes, and exact coverage."""
    manifest_path = output_dir / EVIDENCE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        msg = f"Benchmark evidence manifest not found: {manifest_path}"
        raise BenchmarkEvidenceError(msg)
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BenchmarkEvidenceManifest.model_validate(raw_manifest)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        msg = f"Invalid benchmark evidence manifest {manifest_path}: {exc}"
        raise BenchmarkEvidenceError(msg) from exc

    actual_task_digest = task_manifest_digest(tasks_dir)
    if manifest.task_manifest_digest != actual_task_digest:
        msg = (
            "Benchmark evidence task-manifest digest mismatch: "
            f"expected={manifest.task_manifest_digest}, actual={actual_task_digest}"
        )
        raise BenchmarkEvidenceError(msg)
    if expected_source_sha is not None and manifest.source_revision != expected_source_sha:
        msg = (
            "Benchmark evidence source revision mismatch: "
            f"expected={expected_source_sha}, actual={manifest.source_revision}"
        )
        raise BenchmarkEvidenceError(msg)

    report_paths = {
        path.stem: path
        for path in output_dir.glob("*.json")
        if path.name != EVIDENCE_MANIFEST_FILENAME
    }
    expected_task_ids = set(manifest.task_ids)
    actual_task_ids = set(report_paths)
    if actual_task_ids != expected_task_ids:
        missing = sorted(expected_task_ids - actual_task_ids)
        unexpected = sorted(actual_task_ids - expected_task_ids)
        msg = (
            f"Benchmark evidence file coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
        raise BenchmarkEvidenceError(msg)

    expected_strategies = set(manifest.strategies)
    for task_id, report_path in report_paths.items():
        raw_report = report_path.read_bytes()
        actual_digest = hashlib.sha256(raw_report).hexdigest()
        expected_digest = manifest.report_hashes[task_id]
        if actual_digest != expected_digest:
            msg = (
                f"Benchmark evidence report hash mismatch for {report_path.name}: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
            raise BenchmarkEvidenceError(msg)
        try:
            report = BenchmarkReport.model_validate_json(raw_report)
        except ValidationError as exc:
            msg = f"Invalid benchmark report {report_path}: {exc}"
            raise BenchmarkEvidenceError(msg) from exc
        if report.task_id != task_id:
            msg = (
                "Benchmark report filename/task ID mismatch: "
                f"{report_path.name} != {report.task_id}"
            )
            raise BenchmarkEvidenceError(msg)
        report_strategies = {result.strategy for result in report.results}
        if report_strategies != expected_strategies or len(report.results) != len(
            report_strategies
        ):
            missing = sorted(strategy.value for strategy in expected_strategies - report_strategies)
            unexpected = sorted(
                strategy.value for strategy in report_strategies - expected_strategies
            )
            msg = (
                f"Benchmark evidence strategy coverage mismatch for {task_id}: "
                f"missing={missing}, unexpected={unexpected}"
            )
            raise BenchmarkEvidenceError(msg)

    return manifest
