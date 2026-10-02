#!/usr/bin/env python3
"""Deterministic stratified task sampler for the pre-registered R3x SWE A/B.

No model call, no network, no Docker. The draw is a pure function of the tasks-root
listing, the exclusions file, the stage, and (stage 2) the stage 1 plan. Within each
repository, instances are ranked by ``sha256(f"{SEED}:{instance_id}")``; selection
walks that order, skips excluded instances (and every stage 1 instance in stage 2),
and takes the first ``k_r``. A repository that runs out has its shortfall reassigned
one task at a time to the repository with the most remaining available instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from archex.benchmark.swe_ab import MODELS, SweAbArm, SweAbError, SweAbPlan, load_plan

SEED = 20260909
STAGE1_TASKS = 24
STAGE2_MIN_TASKS = 100
GUARDRAIL_PAIRS = 370
"""Task-pair count that powers the pooled non-inferiority guardrail."""


def stage2_task_count(passing_configurations: int) -> int:
    """Stage 2 tasks: ``max(100, ceil(370 / k))`` for ``k`` Stage 1 solve-floor configurations."""
    if not 1 <= passing_configurations <= len(MODELS):
        raise SweAbError(
            f"stage 2 size rule needs 1..{len(MODELS)} floor-passing configurations, "
            f"got {passing_configurations}"
        )
    return max(STAGE2_MIN_TASKS, -(-GUARDRAIL_PAIRS // passing_configurations))


STAGE1_PER_REPO = 2
STAGE1_EXTRA_REPOS = 2
EXCLUSION_REASONS = ("gold_not_resolved", "empty_not_failing")
_ID_RE = re.compile(r"^instance_(?P<org>.+?)__(?P<repo>.+)-(?P<sha>[0-9a-f]{40})(?:-v.+)?$")


@dataclass(frozen=True)
class Draw:
    stage: int
    pool_by_repo: dict[str, int]
    allocation: dict[str, int]
    walked: dict[str, list[dict[str, Any]]]
    selected: list[tuple[str, str]]  # (repo, instance_id), plan order


def parse_repo(instance_id: str) -> str:
    match = _ID_RE.match(instance_id)
    if match is None:
        raise SweAbError(f"unparseable instance id: {instance_id!r}")
    return f"{match['org']}/{match['repo']}"


def rank_key(instance_id: str) -> tuple[str, str]:
    return (hashlib.sha256(f"{SEED}:{instance_id}".encode()).hexdigest(), instance_id)


def list_pool(tasks_root: Path) -> dict[str, list[str]]:
    """Return repo -> instance ids in rank order."""
    if not tasks_root.is_dir():
        raise SweAbError(f"tasks root is not a directory: {tasks_root}")
    pool: dict[str, list[str]] = {}
    for entry in sorted(tasks_root.iterdir()):
        if entry.is_dir():
            pool.setdefault(parse_repo(entry.name), []).append(entry.name)
    if not pool:
        raise SweAbError(f"no task directories under {tasks_root}")
    return {repo: sorted(ids, key=rank_key) for repo, ids in sorted(pool.items())}


def load_exclusions(path: Path | None, pool: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweAbError(f"cannot read exclusions {path}: {exc}") from exc
    items: Any = cast("dict[str, Any]", raw).get("exclusions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise SweAbError('exclusions file must be {"exclusions": [...]}')
    known = {i for ids in pool.values() for i in ids}
    out: dict[str, dict[str, str]] = {}
    for raw_item in cast("list[Any]", items):
        if not isinstance(raw_item, dict):
            raise SweAbError(f"exclusion must be an object: {raw_item!r}")
        item = cast("dict[str, Any]", raw_item)
        if set(item) != {"instance_id", "reason", "source"}:
            raise SweAbError(f"exclusion needs exactly instance_id, reason, source: {item!r}")
        iid, reason, source = item["instance_id"], item["reason"], item["source"]
        if not all(isinstance(v, str) for v in (iid, reason, source)):
            raise SweAbError(f"exclusion fields must be strings: {item!r}")
        if reason not in EXCLUSION_REASONS:
            raise SweAbError(f"invalid exclusion reason {reason!r} for {iid}")
        if iid not in known:
            raise SweAbError(f"exclusion names an unknown instance id: {iid}")
        if iid in out:
            raise SweAbError(f"duplicate exclusion: {iid}")
        out[iid] = {"instance_id": iid, "reason": reason, "source": source}
    return out


def stage1_allocation(sizes: dict[str, int]) -> dict[str, int]:
    alloc = dict.fromkeys(sizes, STAGE1_PER_REPO)
    largest = sorted(sizes, key=lambda r: (-sizes[r], r))[:STAGE1_EXTRA_REPOS]
    for repo in largest:
        alloc[repo] += 1
    return alloc


def stage2_allocation(sizes: dict[str, int], total: int) -> dict[str, int]:
    pool = sum(sizes.values())
    base = {r: total * n // pool for r, n in sizes.items()}
    remainder = {r: total * n % pool for r, n in sizes.items()}
    order = sorted(sizes, key=lambda r: (-remainder[r], -sizes[r], r))
    for repo in order[: total - sum(base.values())]:
        base[repo] += 1
    return base


def cap_and_reassign(alloc: dict[str, int], available: dict[str, int]) -> dict[str, int]:
    out = {r: min(k, available[r]) for r, k in alloc.items()}
    short = sum(alloc.values()) - sum(out.values())
    for _ in range(short):
        repo = min(out, key=lambda r: (-(available[r] - out[r]), r))
        if available[repo] - out[repo] <= 0:
            raise SweAbError("not enough valid instances to fill the stage")
        out[repo] += 1
    return out


def draw(
    stage: int,
    pool: dict[str, list[str]],
    exclusions: dict[str, dict[str, str]],
    stage1_ids: frozenset[str] = frozenset(),
    stage2_tasks: int = STAGE2_MIN_TASKS,
) -> Draw:
    sizes = {r: len(ids) for r, ids in pool.items()}
    wanted = stage1_allocation(sizes) if stage == 1 else stage2_allocation(sizes, stage2_tasks)

    def disposition(iid: str) -> str | None:
        if iid in exclusions:
            return f"excluded:{exclusions[iid]['reason']}"
        if iid in stage1_ids:
            return "stage1"
        return None

    valid = {r: [i for i in ids if disposition(i) is None] for r, ids in pool.items()}
    alloc = cap_and_reassign(wanted, {r: len(v) for r, v in valid.items()})
    walked: dict[str, list[dict[str, Any]]] = {}
    selected: list[tuple[str, str]] = []
    for repo, ids in pool.items():
        chosen = 0
        rows: list[dict[str, Any]] = []
        for rank, iid in enumerate(ids):
            if chosen == alloc[repo]:
                break
            reason = disposition(iid)
            rows.append({"instance_id": iid, "rank": rank, "disposition": reason or "selected"})
            if reason is None:
                chosen += 1
                selected.append((repo, iid))
        walked[repo] = rows
    return Draw(stage, sizes, alloc, walked, selected)


def build_plan(result: Draw, cost_ceiling: float, without_hc: bool) -> SweAbPlan:
    if result.stage == 1:
        reps = {SweAbArm.A0: 2, SweAbArm.H: 1, SweAbArm.HC: 1, SweAbArm.C: 1}
    else:
        reps = {SweAbArm.A0: 1, SweAbArm.H: 1}
        if not without_hc:
            reps[SweAbArm.HC] = 1
    try:
        return SweAbPlan.model_validate(
            {
                "name": f"r3x-stage{result.stage}",
                "tasks": [{"task_id": i, "repo": r} for r, i in result.selected],
                "models": list(MODELS),
                "repetitions": reps,
                "cost_ceiling_usd": cost_ceiling,
            }
        )
    except ValueError as exc:
        raise SweAbError(f"invalid plan: {exc}") from exc


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def build_manifest(
    result: Draw,
    exclusions: dict[str, dict[str, str]],
    sums_sha: str | None,
    stage1_ids: list[str] | None,
    stage2_tasks: int | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "seed": SEED,
        "stage": result.stage,
        "tasks_root_sha256sums": sums_sha,
        "pool_size": sum(result.pool_by_repo.values()),
        "pool_by_repo": result.pool_by_repo,
        "allocation_by_repo": result.allocation,
        "exclusions": [exclusions[i] for i in sorted(exclusions)],
        "walk_by_repo": result.walked,
        "selected": [i for _, i in result.selected],
    }
    if stage1_ids is not None:
        manifest["stage1_plan_task_ids"] = stage1_ids
    if stage2_tasks is not None:
        manifest["stage2_tasks"] = stage2_tasks
    return manifest


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage: int = args.stage
    if args.cost_ceiling <= 0:
        raise SweAbError("--cost-ceiling must be > 0")
    if stage == 1 and (args.stage1_plan is not None or args.without_hc):
        raise SweAbError("--stage1-plan and --without-hc apply to stage 2 only")
    if stage == 1 and args.stage2_tasks is not None:
        raise SweAbError("--stage2-tasks applies to stage 2 only")
    stage2_tasks: int | None = args.stage2_tasks
    if stage == 2:
        if stage2_tasks is None:
            raise SweAbError("stage 2 requires --stage2-tasks")
        if stage2_tasks < STAGE2_MIN_TASKS:
            raise SweAbError(f"--stage2-tasks must be >= {STAGE2_MIN_TASKS}")
    if stage == 2 and args.stage1_plan is None:
        raise SweAbError("stage 2 requires --stage1-plan")
    outs = [args.plan_out, args.manifest_out, args.instances_out]
    if not args.force:
        for out in outs:
            if out.exists():
                raise SweAbError(f"refusing to overwrite {out} (use --force)")

    pool = list_pool(args.tasks_root)
    exclusions = load_exclusions(args.exclusions, pool)
    stage1_ids: list[str] | None = None
    if stage == 1:
        result = draw(1, pool, exclusions)
    else:
        given = load_plan(args.stage1_plan)
        recomputed = draw(1, pool, exclusions)
        stage1_ids = [t.task_id for t in given.tasks]
        if stage1_ids != [i for _, i in recomputed.selected]:
            raise SweAbError(
                "stage 1 plan does not match the recomputed stage 1 draw "
                "(different tasks root, exclusions, or plan)"
            )
        result = draw(2, pool, exclusions, frozenset(stage1_ids), stage2_tasks or 0)
        if set(stage1_ids) & {i for _, i in result.selected}:
            raise SweAbError("stage 2 overlaps stage 1")

    plan = build_plan(result, args.cost_ceiling, args.without_hc)
    sums = args.tasks_root.parent / "SHA256SUMS"
    sums_sha = hashlib.sha256(sums.read_bytes()).hexdigest() if sums.is_file() else None
    manifest = build_manifest(result, exclusions, sums_sha, stage1_ids, stage2_tasks)
    instances = "".join(f"{i}\n" for _, i in result.selected)
    _atomic_write(args.plan_out, _dump(plan.model_dump(mode="json", exclude_none=True)))
    _atomic_write(args.manifest_out, _dump(manifest))
    _atomic_write(args.instances_out, instances)
    return {
        "stage": stage,
        "selected": len(result.selected),
        "allocation": result.allocation,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-root", type=Path, required=True)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--cost-ceiling", type=float, required=True)
    p.add_argument("--exclusions", type=Path)
    p.add_argument("--stage1-plan", type=Path)
    p.add_argument("--without-hc", action="store_true")
    p.add_argument("--plan-out", type=Path, required=True)
    p.add_argument("--manifest-out", type=Path, required=True)
    p.add_argument("--instances-out", type=Path, required=True)
    p.add_argument("--stage2-tasks", type=int, help="stage 2 task count (stage 2 only; >= 100)")
    p.add_argument("--force", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except SweAbError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
