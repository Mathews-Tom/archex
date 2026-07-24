"""Structural checks on archex's pinned, read-only CI example workflows.

Parses each workflow file directly rather than executing it (GitHub Actions
cannot run inside a unit test); these assertions are exactly the properties
each example claims: no write permissions, every action pinned by full
commit SHA rather than a floating tag, no step that could mutate the
repository/comment/push anything, and every uploaded artifact is one of the
workflow's own declared read-only report outputs.

Covers `report-diff.yml` (M4) and `status-card.yml` (M9's immutable-pinned,
read-only status/compatibility-artifact example).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOWS_DIR = Path(__file__).parent.parent / ".github" / "workflows"

_SHA_PIN_RE = re.compile(r"^[^@]+@[0-9a-f]{40}\b")
_MUTATING_PATTERNS = (
    "git push",
    "git commit",
    "gh pr comment",
    "gh pr edit",
    "gh pr merge",
    "git add",
)

#: Every path this suite verifies is uploaded as a build artifact, keyed by
#: the workflow file it belongs to. Uploading anything outside this
#: whitelist would be a scope change the test must catch.
_ALLOWED_UPLOAD_PATHS: dict[str, frozenset[str]] = {
    "report-diff.yml": frozenset({"report-delta.json", "report-delta.md", "report-diff.json"}),
    "status-card.yml": frozenset(
        {"status-card.json", "status-card.md", "compatibility-artifact.json"}
    ),
}

_WORKFLOW_NAMES = tuple(_ALLOWED_UPLOAD_PATHS)


def _load_workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((_WORKFLOWS_DIR / name).read_text())


@pytest.mark.parametrize("workflow_name", _WORKFLOW_NAMES)
def test_workflow_file_exists(workflow_name: str) -> None:
    assert (_WORKFLOWS_DIR / workflow_name).is_file()


@pytest.mark.parametrize("workflow_name", _WORKFLOW_NAMES)
def test_workflow_grants_no_write_permissions(workflow_name: str) -> None:
    workflow = _load_workflow(workflow_name)

    permissions = workflow["permissions"]
    assert permissions == {"contents": "read"}


@pytest.mark.parametrize("workflow_name", _WORKFLOW_NAMES)
def test_workflow_actions_are_pinned_to_full_commit_shas(workflow_name: str) -> None:
    workflow = _load_workflow(workflow_name)

    jobs = workflow["jobs"]
    uses_refs = [step["uses"] for job in jobs.values() for step in job["steps"] if "uses" in step]

    assert uses_refs, "expected at least one pinned action reference"
    for ref in uses_refs:
        assert _SHA_PIN_RE.match(ref), f"{ref!r} is not pinned to a full commit SHA"


@pytest.mark.parametrize("workflow_name", _WORKFLOW_NAMES)
def test_workflow_has_no_mutating_steps(workflow_name: str) -> None:
    workflow = _load_workflow(workflow_name)

    run_bodies = [
        step["run"] for job in workflow["jobs"].values() for step in job["steps"] if "run" in step
    ]
    combined = "\n".join(run_bodies)

    for pattern in _MUTATING_PATTERNS:
        assert pattern not in combined, f"found mutating command {pattern!r} in workflow"


@pytest.mark.parametrize("workflow_name", _WORKFLOW_NAMES)
def test_workflow_only_uploads_declared_read_only_outputs(workflow_name: str) -> None:
    workflow = _load_workflow(workflow_name)
    allowed = _ALLOWED_UPLOAD_PATHS[workflow_name]

    upload_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "upload-artifact" in step.get("uses", "")
    ]

    assert upload_steps, "expected an upload-artifact step"
    for step in upload_steps:
        paths = step["with"]["path"].strip().splitlines()
        assert paths
        for path in paths:
            assert path.strip() in allowed, f"{path!r} is not a declared read-only output"


def test_status_card_workflow_runs_the_m9_status_and_compatibility_commands() -> None:
    workflow = _load_workflow("status-card.yml")

    run_bodies = "\n".join(
        step["run"] for job in workflow["jobs"].values() for step in job["steps"] if "run" in step
    )

    assert "archex report status-card" in run_bodies
    assert "archex report release-artifact" in run_bodies
