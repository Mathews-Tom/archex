"""Stage 0 feasibility checks for the SWE A/B campaign (spec §2.5, §8), as JSON.

```bash
# Local checks (no spend, no containers): runs every arm once against the
# local stub provider and inspects what omp actually sent.
uv run python scripts/swe_ab_stage0.py --output /tmp/stage0.json

# On the x86_64 Linux campaign host, add the container checks:
uv run python scripts/swe_ab_stage0.py --output stage0.json --host \
    --tasks-root SWE-bench_Pro-os/v2/tasks --instances instances.txt \
    --omp-dir /opt/omp-linux-x64 --omp-command /opt/omp/bin/omp \
    --archex-wheel dist/archex-0.33.0-py3-none-any.whl --uv-binary /opt/uv/uv
```

Each check reports ``pass``, ``fail``, or ``requires_host`` (needs an x86_64
Linux host with Docker, or a hosted model, and was not run). The gate passes
only when no check fails and none is left ``requires_host``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from archex.benchmark.swe_ab import (
    ANNOTATION_MARKER,
    BASE_TOOLS,
    CLI_GUIDE_PATH,
    COMPRESSOR_MARKERS,
    MODELS,
    OMP_VERSION,
    SweAbArm,
    SweAbError,
    load_cell,
    load_plan,
    search_routing_rules,
    sha256_file,
    system_prompt_violations,
    validate_swe_ab_directory,
)
from archex.client_setup import render_annotation_hook_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swe_ab_cell as cell_runner  # noqa: E402 - sibling script, importable only via sys.path

_ROOT = Path(__file__).resolve().parent.parent
_SUITE = _ROOT / "scripts" / "run_swe_ab_suite.py"
_DRY_RUN_PLAN = _ROOT / "benchmarks" / "swe_ab" / "dry-run-plan.json"
_STUB_SCRIPT = _ROOT / "benchmarks" / "swe_ab" / "stub-script.json"
_CAMPAIGN_PYTHON = "/opt/archex/venv/bin/python"


def _check(check_id: str, status: str, detail: str, **data: Any) -> dict[str, Any]:
    return {"id": check_id, "status": status, "detail": detail, **data}


def _requires_host(check_id: str, detail: str) -> dict[str, Any]:
    return _check(check_id, "requires_host", detail)


def omp_version_check(omp_command: list[str]) -> dict[str, Any]:
    done = subprocess.run([*omp_command, "--version"], capture_output=True, text=True, check=False)
    found = done.stdout.strip().removeprefix("omp/")
    status = "pass" if found == OMP_VERSION else "fail"
    return _check("omp_version", status, f"omp reports {found!r}; pinned {OMP_VERSION}")


def model_ids_check(omp_command: list[str]) -> dict[str, Any]:
    done = subprocess.run(
        [*omp_command, "models", "--json"], capture_output=True, text=True, check=False
    )
    try:
        listed = cast("dict[str, Any]", json.loads(done.stdout))["models"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return _check(
            "model_ids_resolve", "fail", "`omp models --json` did not return a model list"
        )
    routes: dict[str, list[str]] = {}
    for entry in cast("list[dict[str, Any]]", listed):
        routes.setdefault(str(entry.get("id")), []).append(str(entry.get("provider")))
    missing = [model for model in MODELS if model not in routes]
    return _check(
        "model_ids_resolve",
        "fail" if missing else "pass",
        "every campaign model id resolves in this host's omp catalog"
        if not missing
        else f"unresolved: {missing}",
        routes={model: routes.get(model, []) for model in MODELS},
        note="credential reachability through the `swebench` profile is part of the one-cell check",
    )


def stub_arm_checks(omp_command: list[str], work: Path) -> list[dict[str, Any]]:
    """Run every arm once against the stub and check what omp sent and received."""
    cells_dir = work / "cells"
    done = subprocess.run(
        [
            sys.executable, str(_SUITE), "run", "--plan", str(_DRY_RUN_PLAN), "--runtime", "local",
            "--output", str(cells_dir), "--work-root", str(work / "suite"),
            "--omp-command", shlex.join(omp_command), "--stub-script", str(_STUB_SCRIPT),
        ],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        return [_check("stub_rehearsal", "fail", done.stderr[-800:] or done.stdout[-800:])]
    checks: list[dict[str, Any]] = []
    tools: dict[str, list[str]] = {}
    violations: dict[str, list[str]] = {}
    routing: dict[str, list[str]] = {}
    annotation: dict[str, bool] = {}
    compressor: dict[str, bool] = {}
    statuses: dict[str, str] = {}
    for arm in SweAbArm:
        [cell_path] = list(cells_dir.rglob(f"{arm.value}/*.json"))
        cell = load_cell(cell_path)
        statuses[arm.value] = cell.status.value
        [request_path] = list(
            (work / "suite" / "cells").glob(f"*__{arm.value}__rep1/first-request.json")
        )
        request = cast("dict[str, Any]", json.loads(request_path.read_text()))
        messages = cast("list[dict[str, Any]]", request.get("messages") or [])
        system = "".join(str(m.get("content")) for m in messages if m.get("role") == "system")
        tools[arm.value] = sorted(
            str(cast("dict[str, Any]", t.get("function") or {}).get("name"))
            for t in cast("list[dict[str, Any]]", request.get("tools") or [])
        )
        violations[arm.value] = system_prompt_violations(system)
        routing[arm.value] = search_routing_rules(system)
        results = [str(m.get("content")) for m in messages if m.get("role") == "tool"]
        annotation[arm.value] = cell.isolation.annotation_seen
        compressor[arm.value] = any(marker in r for r in results for marker in COMPRESSOR_MARKERS)
    checks.append(_check(
        "stub_rehearsal", "pass" if set(statuses.values()) == {"ok"} else "fail",
        "one no-spend cell per arm through the full cell flow", cell_status=statuses,
    ))  # fmt: skip
    wrong = {arm: t for arm, t in tools.items() if t != sorted(BASE_TOOLS)}
    checks.append(_check(
        "tool_list_fingerprint", "fail" if wrong else "pass",
        "first request advertises exactly the base tools in every arm",
        advertised=tools, expected=sorted(BASE_TOOLS),
    ))  # fmt: skip
    leaked = {arm: v for arm, v in violations.items() if v}
    checks.append(_check(
        "system_prompt_isolation", "fail" if leaked else "pass",
        "no AGENTS.md, CLAUDE.md, memories, or compressor marker in the rendered system prompt",
        violations=violations,
    ))  # fmt: skip
    checks.append(_check(
        "search_routing_rules", "pass",
        "disclosure: system-prompt lines that route work to a search tool, per arm",
        rules=routing,
    ))  # fmt: skip
    misplaced = {arm: seen for arm, seen in annotation.items() if seen != SweAbArm(arm).hook}
    checks.append(_check(
        "annotation_only_in_hook_arms", "fail" if misplaced else "pass",
        f"`{ANNOTATION_MARKER}` appears in H and HC tool results and nowhere else",
        annotation_seen=annotation,
    ))  # fmt: skip
    checks.append(_check(
        "no_compressor_output", "fail" if any(compressor.values()) else "pass",
        "no tool-output compressor marker in any tool result", marker_seen=compressor,
    ))  # fmt: skip
    try:
        validate_swe_ab_directory(cells_dir, load_plan(_DRY_RUN_PLAN))
        refused = "fail"
        detail = "the validator accepted stub-endpoint cells for publication"
    except SweAbError as exc:
        refused = "pass" if "overridden provider endpoint" in str(exc) else "fail"
        detail = str(exc)
    checks.append(_check("validator_refuses_stub_cells", refused, detail))
    return checks


def identity_checks() -> list[dict[str, Any]]:
    module = render_annotation_hook_module(_CAMPAIGN_PYTHON)
    return [
        _check(
            "frozen_identities", "pass", "SHA-256 of the artifacts the pre-registration pins",
            cli_guide_sha256=sha256_file(_ROOT / CLI_GUIDE_PATH),
            hook_module_sha256=hashlib.sha256(module.encode()).hexdigest(),
            hook_module_python=_CAMPAIGN_PYTHON,
        )
    ]  # fmt: skip


def host_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Container checks; run only with --host on an x86_64 Linux Docker host."""
    instances = [line.strip() for line in args.instances.read_text().splitlines() if line.strip()]
    base = {
        "runtime": "docker", "model": MODELS[0], "arm": "HC", "repetition": 1, "repo": "stage0",
        "omp_command": shlex.split(args.omp_command), "profile_dir": str(args.profile_dir or "."),
        "archex_wheel": str(args.archex_wheel), "uv_binary": str(args.uv_binary),
        "omp_dir": str(args.omp_dir), "network": args.network,
    }  # fmt: skip
    checks: list[dict[str, Any]] = []
    validity: dict[str, dict[str, bool]] = {}
    index_seconds: dict[str, float | None] = {}
    omp_ok: dict[str, bool] = {}
    for instance in instances:
        task_dir = args.tasks_root / instance
        spec = cell_runner.CellSpec({
            **base, "task_id": instance, "image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{instance}",
            "task_dir": str(task_dir), "output": "/dev/null",
            "work_dir": tempfile.mkdtemp(prefix=f"stage0-{instance}-"),
        })  # fmt: skip
        empty = Path(spec.work_dir) / "empty.patch"
        empty.write_text("")
        validity[instance] = {
            "gold_resolves": cell_runner.score_patch(
                spec, task_dir / "solution" / "gold_patch.diff"
            ),
            "empty_fails": not cell_runner.score_patch(spec, empty),
        }
        rt = cell_runner.DockerRuntime(spec, mounts=[(args.omp_dir, "/opt/omp")])
        try:
            version = rt.run([*spec.omp_command, "--version"], cwd="/").stdout.strip()
            omp_ok[instance] = version == f"omp/{OMP_VERSION}"
            index_seconds[instance] = cell_runner.index_in_container(spec, rt)
        except Exception as exc:  # noqa: BLE001 - reported as a failed check, not raised
            index_seconds[instance] = None
            omp_ok.setdefault(instance, False)
            checks.append(_check(f"container_setup:{instance}", "fail", repr(exc)))
        finally:
            rt.close()
    invalid = [i for i, v in validity.items() if not (v["gold_resolves"] and v["empty_fails"])]
    checks.append(_check(
        "gold_empty_validity", "fail" if invalid else "pass",
        "gold patch resolves and empty patch fails, per instance (invalid ones leave the pool)",
        per_instance=validity, excluded=invalid,
    ))  # fmt: skip
    checks.append(_check(
        "omp_runs_in_container", "pass" if all(omp_ok.values()) else "fail",
        "the pinned omp build starts inside each Pro image", per_instance=omp_ok,
    ))  # fmt: skip
    checks.append(_check(
        "archex_indexes_in_container", "pass" if all(index_seconds.values()) else "fail",
        "archex installs into /opt/archex and indexes the checkout to `fresh`",
        index_seconds=index_seconds,
    ))  # fmt: skip
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omp-command", default=shutil.which("omp") or "omp")
    parser.add_argument("--host", action="store_true", help="also run the Docker checks")
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--instances", type=Path, help="one Pro instance id per line")
    parser.add_argument("--omp-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--archex-wheel", type=Path)
    parser.add_argument("--uv-binary", type=Path)
    parser.add_argument("--network", default="bridge")
    args = parser.parse_args(argv)
    omp_command = shlex.split(args.omp_command)
    started = time.monotonic()
    checks = [omp_version_check(omp_command), model_ids_check(omp_command), *identity_checks()]
    with tempfile.TemporaryDirectory(prefix="swe-ab-stage0-") as work:
        checks += stub_arm_checks(omp_command, Path(work))
    if args.host:
        checks += host_checks(args)
    else:
        checks += [
            _requires_host(
                "gold_empty_validity", "x86_64 Pro images: gold patch resolves, empty fails"
            ),
            _requires_host(
                "omp_runs_in_container", "the omp linux-x64 build inside each Pro image"
            ),
            _requires_host(
                "archex_indexes_in_container", "archex install and index inside a Pro image"
            ),
        ]
    checks.append(_requires_host(
        "one_real_cell_per_model",
        "hosted model call: route, auth, usage reporting, and the annotation ledger per model; "
        "run with scripts/run_swe_ab_suite.py and a one-task Stage 0 plan (RUNBOOK §4)",
    ))  # fmt: skip
    statuses = [check["status"] for check in checks]
    report = {
        "stage": "0",
        "omp_version_pinned": OMP_VERSION,
        "checks": checks,
        "summary": {s: statuses.count(s) for s in ("pass", "fail", "requires_host")},
        "gate": "fail"
        if "fail" in statuses
        else ("pending_host" if "requires_host" in statuses else "pass"),
        "seconds": round(time.monotonic() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "gate": report["gate"]}))
    return 1 if report["gate"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
