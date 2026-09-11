"""Independently reproduce R27 scope counts and single-scope control payloads.

Run this file with the Archex checkout at the population's full control revision.
It intentionally parses raw JSON and never imports the R27 campaign validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from archex.api import index_repository, query
from archex.models import Config, IndexConfig, RepoSource

_MARKERS = ("package.json", "Cargo.toml")
_PAYLOAD_FIELDS = (
    "query",
    "chunks",
    "structural_context",
    "type_definitions",
    "dependency_summary",
    "token_count",
    "token_budget",
    "truncated",
)


def _run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _checkout_revision() -> str:
    return _run_git("rev-parse", "HEAD", cwd=Path.cwd())


def _repository_root(repository: dict[str, Any], repo_cache: Path, scratch: Path) -> Path:
    owner_repo = repository["repository"].replace("/", "__")
    for name in (owner_repo, repository["repo_id"], f"actual__{owner_repo}"):
        candidate = repo_cache / name
        if candidate.is_dir():
            revision = _run_git("rev-parse", "HEAD", cwd=candidate)
            if revision != repository["commit"]:
                raise ValueError(
                    f"{repository['repo_id']}: cached checkout is {revision}, "
                    f"expected {repository['commit']}"
                )
            return candidate

    target = scratch / owner_repo
    target.mkdir()
    _run_git("init", "--quiet", cwd=target)
    _run_git("remote", "add", "origin", repository["url"], cwd=target)
    _run_git("fetch", "--quiet", "--depth=1", "origin", repository["commit"], cwd=target)
    _run_git("checkout", "--quiet", "FETCH_HEAD", cwd=target)
    return target


def _copy_path(source_root: Path, destination_root: Path, relative_path: str) -> None:
    source = source_root / relative_path
    destination = destination_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"missing pinned corpus path: {source}")


def _initialize_snapshot(repository: dict[str, Any], source_root: Path, target: Path) -> None:
    for corpus_path in repository["corpus_paths"]:
        _copy_path(source_root, target, corpus_path)
    _run_git("init", "--quiet", cwd=target)
    _run_git("add", "--all", cwd=target)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=R27 Reproducer",
            "-c",
            "user.email=r27@example.invalid",
            "commit",
            "--quiet",
            "--message=freeze",
        ),
        cwd=target,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-09-11T20:00:00Z",
            "GIT_COMMITTER_DATE": "2026-09-11T20:00:00Z",
        },
    )


def _scope_roots(root: Path) -> tuple[str, ...]:
    roots: set[str] = set()
    for marker in _MARKERS:
        for path in root.rglob(marker):
            if ".git" in path.parts:
                continue
            relative = path.parent.relative_to(root).as_posix()
            roots.add("" if relative == "." else relative)
    return tuple(sorted(roots))


def _owner(path: str, roots: tuple[str, ...]) -> str | None:
    owners = [root for root in roots if not root or path == root or path.startswith(f"{root}/")]
    if not owners:
        return None
    return max(owners, key=lambda value: (len(value.split("/")), value))


def _canonical_payload(bundle: Any) -> bytes:
    raw = bundle.model_dump(mode="json")
    projection = {field: raw[field] for field in _PAYLOAD_FIELDS}
    return json.dumps(
        projection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _run_query(
    root: Path,
    repository: dict[str, Any],
    question: str,
    cache_dir: Path,
) -> Any:
    config = Config(
        cache=True,
        cache_dir=str(cache_dir),
        languages=repository["languages"],
        parallel=False,
        worktree_seed=False,
    )
    source = RepoSource(local_path=str(root))
    index_config = IndexConfig.model_validate(repository["index_config"])
    return query(
        source,
        question,
        token_budget=4000,
        config=config,
        index_config=index_config,
        explicit_token_budget=True,
        refresh=False,
    )


def _reproduce_repository(
    repository: dict[str, Any],
    source_root: Path,
    scratch: Path,
) -> dict[str, Any]:
    license_path = source_root / repository["license"]["path"]
    observed_license_sha256 = hashlib.sha256(license_path.read_bytes()).hexdigest()
    if observed_license_sha256 != repository["license"]["sha256"]:
        raise ValueError(f"{repository['repo_id']}: pinned license digest drift")
    workspace = scratch / f"workspace__{repository['repo_id']}"
    workspace.mkdir()
    _initialize_snapshot(repository, source_root, workspace)
    roots = _scope_roots(workspace)

    config = Config(
        cache=True,
        cache_dir=str(scratch / "full-cache"),
        languages=repository["languages"],
        parallel=False,
        worktree_seed=False,
    )
    store = index_repository(
        RepoSource(local_path=str(workspace)),
        config=config,
        index_config=IndexConfig.model_validate(repository["index_config"]),
    )
    try:
        counts: Counter[str] = Counter()
        for chunk in store.iter_chunks():
            owner = _owner(chunk.file_path, roots)
            if owner is not None:
                counts[owner] += 1
    finally:
        store.close()

    expected_counts = repository["scope_chunk_counts"]
    observed_counts = dict(sorted(counts.items()))
    if observed_counts != expected_counts:
        mismatches = {
            scope: (expected_counts.get(scope), observed_counts.get(scope))
            for scope in sorted(set(expected_counts) | set(observed_counts))
            if expected_counts.get(scope) != observed_counts.get(scope)
        }
        raise ValueError(f"{repository['repo_id']}: scope chunk counts drift: {mismatches}")

    control = repository["control"]
    control_root = scratch / f"control__{repository['repo_id']}"
    control_root.mkdir()
    _copy_path(source_root, control_root, control["scope"])
    _run_git("init", "--quiet", cwd=control_root)
    _run_git("add", "--all", cwd=control_root)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=R27 Reproducer",
            "-c",
            "user.email=r27@example.invalid",
            "commit",
            "--quiet",
            "--message=control",
        ),
        cwd=control_root,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-09-11T20:00:00Z",
            "GIT_COMMITTER_DATE": "2026-09-11T20:00:00Z",
        },
    )
    control_roots = _scope_roots(control_root)
    if len(control_roots) != 1:
        raise ValueError(
            f"{repository['repo_id']}: control has {len(control_roots)} marker-root scopes"
        )

    cache_dir = scratch / "control-cache"
    _run_query(control_root, repository, control["query"], cache_dir)
    measured = [_run_query(control_root, repository, control["query"], cache_dir) for _ in range(2)]
    runs: list[dict[str, Any]] = []
    for number, bundle in enumerate(measured, start=1):
        payload = _canonical_payload(bundle)
        paths = {ranked.chunk.file_path for ranked in bundle.chunks}
        runs.append(
            {
                "run": number,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_bytes": len(payload),
                "expected_file_returned": control["expected_file"] in paths,
            }
        )
    for run in runs:
        if run["payload_sha256"] != control["payload_sha256"]:
            raise ValueError(
                f"{repository['repo_id']}: control payload digest "
                f"{run['payload_sha256']} != {control['payload_sha256']}"
            )
        if run["payload_bytes"] != control["payload_bytes"]:
            raise ValueError(f"{repository['repo_id']}: control payload size drift")
        if not run["expected_file_returned"]:
            raise ValueError(f"{repository['repo_id']}: control expected file missing")

    return {
        "repository_id": repository["repo_id"],
        "scope_marker_count": len(roots),
        "indexed_chunk_count": sum(counts.values()),
        "minimum_dominant_ratio": min(
            repository["dominant_scope_chunks"] / counts[scope]
            for scope in repository["eligible_required_scopes"]
        ),
        "warm_runs": runs,
    }


def _verify_receipts(
    observed: list[dict[str, Any]], receipts_path: Path, repositories: dict[str, dict[str, Any]]
) -> None:
    expected = json.loads(receipts_path.read_text())
    expected_by_repo = {receipt["repository_id"]: receipt for receipt in expected["repositories"]}
    for repository_result in observed:
        repo_id = repository_result["repository_id"]
        expected_receipt = expected_by_repo[repo_id]
        expected_runs = expected_receipt["warm_runs"]
        for observed_run, expected_run in zip(
            repository_result["warm_runs"], expected_runs, strict=True
        ):
            if observed_run["payload_sha256"] != expected_run["payload_sha256"]:
                raise ValueError(f"{repo_id}: receipt payload digest drift")
            if observed_run["payload_bytes"] != expected_run["payload_bytes"]:
                raise ValueError(f"{repo_id}: receipt payload size drift")
        repository = repositories[repo_id]
        if repository_result["indexed_chunk_count"] != repository["indexed_chunk_count"]:
            raise ValueError(f"{repo_id}: indexed chunk count drift")


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 before invoking the reproducer")
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--repo-cache", type=Path, required=True)
    parser.add_argument("--repository", action="append", default=[])
    args = parser.parse_args()

    population = json.loads(args.population.read_text())
    if _checkout_revision() != population["control_archex_revision"]:
        raise SystemExit(
            "Run from an Archex checkout at control revision "
            f"{population['control_archex_revision']}"
        )
    repositories = {
        repo["repo_id"]: {**repo, "index_config": population["index_config"]}
        for repo in population["repositories"]
    }
    selected = args.repository or list(repositories)
    unknown = set(selected) - set(repositories)
    if unknown:
        raise SystemExit(f"Unknown repository IDs: {', '.join(sorted(unknown))}")

    with tempfile.TemporaryDirectory(prefix="archex-r27-reproduce-") as temp_dir:
        scratch = Path(temp_dir)
        observed = [
            _reproduce_repository(
                repositories[repo_id],
                _repository_root(repositories[repo_id], args.repo_cache, scratch),
                scratch,
            )
            for repo_id in selected
        ]
    _verify_receipts(observed, args.receipts, repositories)
    print(
        f"Reproduced R27 controls: {len(observed)} repositories; "
        "scope counts and two warm payloads verified."
    )


if __name__ == "__main__":
    main()
