"""Structural checks on the pinned, read-only report-diff CI workflow example.

Parses `.github/workflows/report-diff.yml` directly rather than executing it
(GitHub Actions cannot run inside a unit test); these assertions are exactly
the properties the CI example claims: no write permissions, every action
pinned by full commit SHA rather than a floating tag, and no step that could
mutate the repository, comment on the PR, or push anything.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_WORKFLOW_PATH = Path(__file__).parent.parent / ".github" / "workflows" / "report-diff.yml"
_SHA_PIN_RE = re.compile(r"^[^@]+@[0-9a-f]{40}\b")
_MUTATING_PATTERNS = (
    "git push",
    "git commit",
    "gh pr comment",
    "gh pr edit",
    "gh pr merge",
    "git add",
)


def _load_workflow() -> dict[str, Any]:
    return yaml.safe_load(_WORKFLOW_PATH.read_text())


def test_workflow_file_exists() -> None:
    assert _WORKFLOW_PATH.is_file()


def test_workflow_grants_no_write_permissions() -> None:
    workflow = _load_workflow()

    permissions = workflow["permissions"]
    assert permissions == {"contents": "read"}


def test_workflow_actions_are_pinned_to_full_commit_shas() -> None:
    workflow = _load_workflow()

    jobs = workflow["jobs"]
    uses_refs = [step["uses"] for job in jobs.values() for step in job["steps"] if "uses" in step]

    assert uses_refs, "expected at least one pinned action reference"
    for ref in uses_refs:
        assert _SHA_PIN_RE.match(ref), f"{ref!r} is not pinned to a full commit SHA"


def test_workflow_has_no_mutating_steps() -> None:
    workflow = _load_workflow()

    run_bodies = [
        step["run"] for job in workflow["jobs"].values() for step in job["steps"] if "run" in step
    ]
    combined = "\n".join(run_bodies)

    for pattern in _MUTATING_PATTERNS:
        assert pattern not in combined, f"found mutating command {pattern!r} in workflow"


def test_workflow_only_uploads_report_outputs() -> None:
    workflow = _load_workflow()

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
            assert path.strip().startswith("report-")
