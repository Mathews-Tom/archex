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
from typing import Any, cast

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
    "report-diff.yml": frozenset(
        {
            "report-delta.json",
            "report-delta.md",
            "report-diff.json",
            "arch-graph.json",
            "explorer-site",
        }
    ),
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


#: `packages: write` on the image-publish workflow is the one write scope this
#: repository grants. It reaches a container registry, never the repository
#: contents, an issue, or a pull request. Any other write grant, and any
#: workflow that omits a `permissions:` block and therefore inherits the
#: repository's default token scope, must fail this audit.
_ALLOWED_WRITE_SCOPES: dict[str, frozenset[str]] = {"docker.yml": frozenset({"packages"})}

#: Scopes that can mutate the repository or its pull requests. Deliberately
#: not the full GitHub scope list: the allowlist check below rejects every
#: undeclared write scope anyway, so this set exists to produce a specific
#: message for the scopes that matter most, not to be the only gate.
_FORBIDDEN_WRITE_SCOPES = frozenset(
    {"contents", "pull-requests", "issues", "actions", "checks", "deployments", "statuses"}
)

#: GitHub executes both extensions, so an audit that globs one of them can be
#: bypassed by naming a workflow with the other.
_WORKFLOW_GLOBS = ("*.yml", "*.yaml")


def _every_workflow() -> list[str]:
    return sorted(path.name for pattern in _WORKFLOW_GLOBS for path in _WORKFLOWS_DIR.glob(pattern))


def _permission_blocks(workflow: dict[str, Any]) -> list[tuple[str, object]]:
    """Every `permissions:` mapping in the file, workflow-level and job-level.

    A job-level block REPLACES the workflow-level one for that job, so auditing
    only the top level would pass a workflow that declares `contents: read`
    globally and grants write on one job.
    """
    blocks: list[tuple[str, object]] = [("workflow", workflow.get("permissions"))]
    jobs = cast("dict[str, dict[str, Any]]", workflow.get("jobs") or {})
    for job_id, job in jobs.items():
        if "permissions" in job:
            blocks.append((f"job {job_id}", job["permissions"]))
    return blocks


def test_the_audit_sees_every_workflow_file() -> None:
    """An empty parametrize list would make every audit below vacuously pass."""
    names = _every_workflow()

    assert names
    assert set(names) >= set(_WORKFLOW_NAMES)


@pytest.mark.parametrize("workflow_name", _every_workflow())
def test_every_workflow_declares_its_permissions_explicitly(workflow_name: str) -> None:
    """An omitted block inherits the repository default, which may include write."""
    workflow = _load_workflow(workflow_name)

    assert "permissions" in workflow, (
        f"{workflow_name} omits `permissions:` and inherits the repository default token scope"
    )


@pytest.mark.parametrize("workflow_name", _every_workflow())
def test_no_workflow_grants_repository_or_pull_request_write(workflow_name: str) -> None:
    workflow = _load_workflow(workflow_name)
    allowed_writes = _ALLOWED_WRITE_SCOPES.get(workflow_name, frozenset())

    for origin, raw_permissions in _permission_blocks(workflow):
        assert isinstance(raw_permissions, dict), (
            f"{workflow_name} ({origin}) must enumerate scopes explicitly, "
            f"not omit them or use a `write-all`/`read-all` preset"
        )
        permissions = cast("dict[str, str]", raw_permissions)
        for scope, level in permissions.items():
            if level != "write":
                continue
            assert scope not in _FORBIDDEN_WRITE_SCOPES, (
                f"{workflow_name} ({origin}) grants {scope}: write, which can mutate the "
                f"repository or its pull requests"
            )
            assert scope in allowed_writes, (
                f"{workflow_name} ({origin}) grants an undeclared {scope}: write"
            )


def test_report_diff_workflow_delivers_the_canonical_artifacts_and_explorer_bundle() -> None:
    workflow = _load_workflow("report-diff.yml")
    run_bodies = "\n".join(
        step["run"] for job in workflow["jobs"].values() for step in job["steps"] if "run" in step
    )

    # The canonical artifacts, then a projection of them -- not a second analysis.
    assert "archex report diff . --base" in run_bodies
    assert "archex graph export ." in run_bodies
    assert "archex explore report-diff.json --graph arch-graph.json" in run_bodies
    assert "--export explorer-site" in run_bodies


def test_report_diff_workflow_stays_read_only_and_comments_nothing() -> None:
    workflow = _load_workflow("report-diff.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert _permission_blocks(workflow) == [("workflow", {"contents": "read"})]

    # Scan every step surface, not just `run`: a credential can reach a step
    # through `env:` or an action's `with:` block without appearing in a shell
    # body, and `${{ github.token }}` does not contain the string GITHUB_TOKEN.
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    run_bodies = "\n".join(step["run"] for step in steps if "run" in step)
    every_surface = yaml.safe_dump(steps)

    assert "GITHUB_STEP_SUMMARY" in run_bodies
    for forbidden in (
        "gh api",
        "gh pr",
        "gh issue",
        "GITHUB_TOKEN",
        "github.token",
        "secrets.",
        "curl -X POST",
        "curl -X PATCH",
    ):
        assert forbidden not in every_surface, (
            f"{forbidden!r} implies a write, a credential, or an authenticated call"
        )
