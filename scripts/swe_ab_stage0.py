"""Stage 0 feasibility checks for the SWE A/B campaign (spec §2.5, §8), as JSON.

```bash
# Local checks (no spend, no containers): model routes and logins from omp's own catalog and
# agent.db (login metadata only), the broker if one is running, and every arm once against the
# local stub provider, inspecting what omp actually sent.
uv run python scripts/swe_ab_stage0.py --output /tmp/stage0.json

# With a running auth broker (omp auth-broker serve --bind 127.0.0.1:8765):
uv run python scripts/swe_ab_stage0.py --output /tmp/stage0.json \
    --broker-url http://127.0.0.1:8765 --broker-bind 127.0.0.1:8765

# On a host with a running Docker daemon (x86_64 Linux, or Docker Desktop on Apple silicon,
# where cells are emulated and recorded as such), add the container checks:
uv run python scripts/swe_ab_stage0.py --output stage0.json --host \
    --tasks-root SWE-bench_Pro-os/v2/tasks --instances instances.txt \
    --omp-dir /opt/omp-linux-x64 --omp-command /opt/omp/bin/omp \
    --archex-wheel dist/archex-0.33.0-py3-none-any.whl --uv-binary /opt/uv/uv \
    --broker-url http://127.0.0.1:8765 --broker-bind 127.0.0.1:8765
```

Each check reports ``pass``, ``fail``, or ``requires_host`` (needs a running Docker daemon,
the omp bundle and task images, or a hosted model, and was not run). The gate passes only
when no check fails and none is left ``requires_host``. No check here calls a model: the
broker checks read ``/v1/healthz`` and ``/v1/usage``, and the container checks use only the
task images, ``alpine``, and Bun's own images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, cast

from archex.benchmark.swe_ab import (
    ANNOTATION_MARKER,
    BASE_TOOLS,
    BROKER_ENV_NAMES,
    CLI_GUIDE_PATH,
    COMPRESSOR_MARKERS,
    MODELS,
    OMP_VERSION,
    QUOTA_MIN_HEADROOM,
    SUBSCRIPTION_PROVIDERS,
    ProviderLogins,
    SweAbArm,
    SweAbError,
    is_emulated,
    load_cell,
    load_plan,
    provider_logins,
    quota_reading,
    search_routing_rules,
    sha256_file,
    system_prompt_violations,
    validate_swe_ab_directory,
)
from archex.client_setup import render_annotation_hook_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swe_ab_cell as cell_runner  # noqa: E402 - sibling script, importable only via sys.path
import run_swe_ab_suite as cell_suite  # noqa: E402 - sibling script, importable only via sys.path

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


def default_agent_db() -> Path:
    """omp's login vault: ``$PI_CODING_AGENT_DIR/agent.db``, else ``~/.omp/agent/agent.db``."""
    if agent_dir := os.environ.get("PI_CODING_AGENT_DIR"):
        return Path(agent_dir) / "agent.db"
    return Path.home() / (os.environ.get("PI_CONFIG_DIR") or ".omp") / "agent" / "agent.db"


def route_and_login_check(
    catalog: list[dict[str, Any]], logins: dict[str, ProviderLogins]
) -> dict[str, Any]:
    """Each campaign model resolves to its own subscription provider, which holds a login."""
    routes: dict[str, str | None] = {}
    problems: list[str] = []
    for model in MODELS:
        provider, _, model_id = model.partition("/")
        found = [
            str(entry.get("provider"))
            for entry in catalog
            if entry.get("selector") == model
            or (entry.get("provider") == provider and entry.get("id") == model_id)
        ]
        route = found[0] if found else None
        routes[model] = route
        if route is None:
            problems.append(f"{model} is not in this omp's model catalog")
        elif route != provider or route not in SUBSCRIPTION_PROVIDERS:
            problems.append(
                f"{model} routes through {route!r}, not the subscription provider {provider!r}"
            )
        if not logins.get(provider, ProviderLogins(0, 0, ())).enabled:
            problems.append(f"omp holds no enabled login for provider {provider!r}")
    return _check(
        "model_routes_and_logins",
        "fail" if problems else "pass",
        "every campaign model id routes through its subscription provider and that provider "
        "has an enabled login (catalog and login metadata only; no model call)"
        if not problems
        else "; ".join(problems),
        routes=routes,
        logins={
            provider: {
                "enabled": logins[provider].enabled if provider in logins else 0,
                "disabled": logins[provider].disabled if provider in logins else 0,
                "types": list(logins[provider].types) if provider in logins else [],
            }
            for provider in SUBSCRIPTION_PROVIDERS
        },
        note="the login is exercised, and its quota read, by the one-real-cell-per-model check",
    )


def model_ids_check(omp_command: list[str], agent_db: Path) -> dict[str, Any]:
    """Model routes from omp's own catalog and logins from ``agent.db``; makes no model call."""
    done = subprocess.run(
        [*omp_command, "models", "--json"], capture_output=True, text=True, check=False
    )
    try:
        catalog = cast("list[dict[str, Any]]", json.loads(done.stdout)["models"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return _check(
            "model_routes_and_logins", "fail", "`omp models --json` did not return a model list"
        )
    try:
        logins = provider_logins(agent_db)
    except SweAbError as exc:
        return _check("model_routes_and_logins", "fail", str(exc))
    return route_and_login_check(catalog, logins)


def broker_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The host-side broker: alive, and its usage report readable (the quota guard's input)."""
    if not args.broker_url:
        return [
            _requires_host("broker_healthy", "start `omp auth-broker serve` and pass --broker-url"),
            _requires_host("broker_usage_readable", "needs the running broker"),
        ]
    checks: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(
            f"{args.broker_url.rstrip('/')}/v1/healthz", timeout=10
        ) as reply:  # noqa: S310
            health = json.loads(reply.read())
        ok = health.get("ok") is True
        checks.append(_check(
            "broker_healthy", "pass" if ok else "fail",
            f"{args.broker_url}/v1/healthz answered", bind=args.broker_bind,
            broker_version=health.get("version"),
        ))  # fmt: skip
    except (OSError, ValueError) as exc:
        return [
            _check("broker_healthy", "fail", f"{args.broker_url}: {type(exc).__name__}"),
            _requires_host("broker_usage_readable", "needs a healthy broker"),
        ]
    try:
        token = cell_suite.read_broker_token(args.broker_token_file)
        usage = cell_suite.fetch_usage(args.broker_url, token)
    except (cell_suite.BrokerError, SystemExit) as exc:
        checks.append(_check("broker_usage_readable", "fail", str(exc)))
        return checks
    readings: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        reading = quota_reading(
            usage,
            cell_suite.provider_of(model),
            model_id=model.split("/", 1)[1],
            min_headroom=QUOTA_MIN_HEADROOM,
        )
        readings[model] = {
            "status": reading.status.value,
            "headroom": reading.headroom,
            "detail": reading.detail,
        }
    checks.append(_check(
        "broker_usage_readable", "pass",
        "GET /v1/usage answered; per-model subscription headroom as the quota guard reads it",
        reports=len(cast("list[object]", usage.get("reports") or [])), quota=readings,
    ))  # fmt: skip
    return checks


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


def docker_running() -> bool:
    """Whether a Docker daemon answers (``docker info``); Docker Desktop must be started first."""
    try:
        done = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def broker_container_check(
    container_url: str, bind: str | None, *, run: Any = subprocess.run
) -> dict[str, Any]:
    """From a throwaway container: reach the host broker, and see only the allow-listed env.

    The probe container is started like a cell's (``--add-host`` for Linux hosts) with no
    environment; the allow-listed variables are then passed the way the cell runner passes them,
    with dummy values, and the names visible inside are compared with what the host exports.
    Nothing here authenticates: ``/v1/healthz`` needs no token.
    """
    name = f"swe-ab-probe-{uuid.uuid4().hex[:8]}"
    started = time.monotonic()
    data: dict[str, Any] = {"bind": bind, "container_url": container_url}
    try:
        up = run(
            ["docker", "run", "-d", "--rm", "--name", name, "--add-host",
             "host.docker.internal:host-gateway", "alpine:3", "sleep", "120"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if up.returncode != 0:
            return _check("broker_reachable_from_container", "fail", up.stderr[-300:], **data)
        health = run(
            ["docker", "exec", name, "wget", "-q", "-O", "-", "-T", "10"]
            + [f"{container_url}/v1/healthz"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        reachable = health.returncode == 0 and '"ok":true' in health.stdout.replace(" ", "")
        env_args, client_env = cell_runner.docker_exec_env(
            {"OMP_AUTH_BROKER_URL": container_url, "OMP_AUTH_BROKER_TOKEN": "stage0-probe"}
        )
        client_env["SWE_AB_STAGE0_HOST_SENTINEL"] = "1"
        names = run(
            ["docker", "exec", *env_args, name, "sh", "-c", "env | cut -d= -f1 | sort"],
            capture_output=True, text=True, check=False, env=client_env,
        )  # fmt: skip
        seen = set(names.stdout.split())
        allowlisted = set(BROKER_ENV_NAMES) <= seen and "SWE_AB_STAGE0_HOST_SENTINEL" not in seen
    finally:
        run(["docker", "rm", "-f", name], capture_output=True, check=False)
    data.update(
        reachable=reachable,
        allow_list_only=allowlisted,
        seconds=round(time.monotonic() - started, 1),
    )
    ok = reachable and allowlisted
    return _check(
        "broker_reachable_from_container",
        "pass" if ok else "fail",
        "a container reaches the host broker's /v1/healthz and receives the two broker variables "
        "but nothing else from the host environment"
        if ok
        else f"reachable={reachable}, allow_list_only={allowlisted}: {health.stderr[-200:]}",
        **data,
    )


_BASELINE_BUN = (
    "apt-get update -qq && apt-get install -y -qq curl unzip ca-certificates >/dev/null && "
    "curl -fsSL https://github.com/oven-sh/bun/releases/latest/download/bun-linux-x64-baseline.zip "
    "-o /tmp/bun.zip && unzip -q /tmp/bun.zip -d /tmp && /tmp/bun-linux-x64-baseline/bun --version"
)


def bun_emulation_check(*, run: Any = subprocess.run) -> dict[str, Any]:
    """Whether Bun's default x86-64 build starts under emulation; else whether the baseline does."""
    default = run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "oven/bun:1-debian"]
        + ["bun", "--version"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if default.returncode == 0:
        return _check(
            "bun_runs_under_emulation", "pass", "Bun's default x86-64 build starts under emulation",
            variant="default", bun_version=default.stdout.strip(),
        )  # fmt: skip
    baseline = run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "debian:12-slim"]
        + ["sh", "-c", _BASELINE_BUN],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if baseline.returncode == 0:
        return _check(
            "bun_runs_under_emulation", "pass",
            "the default build failed under emulation; the baseline build starts: build the omp "
            "bundle with it (RUNBOOK §2)",
            variant="baseline", default_error=(default.stderr or default.stdout)[-200:],
            bun_version=baseline.stdout.strip(),
        )  # fmt: skip
    return _check(
        "bun_runs_under_emulation",
        "fail",
        "neither Bun's default nor its baseline x86-64 build starts",
        default_error=(default.stderr or default.stdout)[-200:],
        baseline_error=(baseline.stderr or baseline.stdout)[-200:],
    )


def host_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Container checks; run only with --host against a running Docker daemon."""
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
    timings: dict[str, dict[str, float | str | None]] = {}
    for instance in instances:
        task_dir = args.tasks_root / instance
        spec = cell_runner.CellSpec({
            **base, "task_id": instance, "image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{instance}",
            "task_dir": str(task_dir), "output": "/dev/null",
            "work_dir": tempfile.mkdtemp(prefix=f"stage0-{instance}-"),
        })  # fmt: skip
        empty = Path(spec.work_dir) / "empty.patch"
        empty.write_text("")
        timing: dict[str, float | str | None] = {}
        timings[instance] = timing
        lap = time.monotonic()
        gold_resolves = cell_runner.score_patch(spec, task_dir / "solution" / "gold_patch.diff")
        timing["gold_score_seconds"] = round(time.monotonic() - lap, 1)
        lap = time.monotonic()
        validity[instance] = {
            "gold_resolves": gold_resolves,
            "empty_fails": not cell_runner.score_patch(spec, empty),
        }
        timing["empty_score_seconds"] = round(time.monotonic() - lap, 1)
        lap = time.monotonic()
        rt = cell_runner.DockerRuntime(spec, mounts=[(args.omp_dir, "/opt/omp")])
        timing["container_start_seconds"] = round(time.monotonic() - lap, 1)
        try:
            timing["container_arch"] = rt.run(["uname", "-m"], cwd="/").stdout.strip()
            lap = time.monotonic()
            version = rt.run([*spec.omp_command, "--version"], cwd="/").stdout.strip()
            timing["omp_start_seconds"] = round(time.monotonic() - lap, 1)
            omp_ok[instance] = version == f"omp/{OMP_VERSION}"
            lap = time.monotonic()
            index_seconds[instance] = cell_runner.index_in_container(spec, rt)
            timing["install_and_index_seconds"] = round(time.monotonic() - lap, 1)
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
        "archex installs into /opt/archex, indexes the checkout to `fresh`, and the annotate "
        "entry pre-warms on a real search hit",
        index_seconds=index_seconds,
    ))  # fmt: skip
    emulated = is_emulated("docker", platform.machine())
    wrong_arch = [i for i, t in timings.items() if t.get("container_arch") != "x86_64"]
    checks.append(_check(
        "emulated_wall_times", "fail" if wrong_arch else "pass",
        f"emulated={emulated} on {platform.machine()}: per-instance wall times "
        f"({'not comparable with native runs' if emulated else 'native x86-64'}); "
        "every image is linux/amd64",
        emulated=emulated, host_machine=platform.machine(), per_instance=timings,
        non_amd64_instances=wrong_arch,
    ))  # fmt: skip
    container_url = cell_suite.container_broker_url(
        args.broker_url or "", args.container_broker_url
    )
    if args.broker_url:
        checks.append(broker_container_check(container_url, args.broker_bind))
    else:
        checks.append(
            _requires_host(
                "broker_reachable_from_container", "start the auth broker and pass --broker-url"
            )
        )
    checks.append(bun_emulation_check())
    return checks


def container_checks_pending(why: str) -> list[dict[str, Any]]:
    """The checks that need a running Docker daemon, left ``requires_host`` with the reason."""
    return [
        _requires_host(check_id, f"{detail}; {why}")
        for check_id, detail in (
            ("gold_empty_validity", "Pro images: gold patch resolves, empty fails"),
            ("omp_runs_in_container", "the omp linux-x64 build inside each Pro image"),
            (
                "archex_indexes_in_container",
                "archex install, index, and annotate pre-warm inside a Pro image",
            ),
            (
                "emulated_wall_times",
                "container start, gold/empty scoring, omp start, and index times per instance",
            ),
            (
                "broker_reachable_from_container",
                "a container reaches the host broker and gets only the allow-listed variables",
            ),
            (
                "bun_runs_under_emulation",
                "Bun's default x86-64 build starts under emulation, or the baseline build does",
            ),
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omp-command", default=shutil.which("omp") or "omp")
    parser.add_argument(
        "--agent-db",
        type=Path,
        default=default_agent_db(),
        help="omp's agent.db; read for login metadata only, never for secrets",
    )
    parser.add_argument("--host", action="store_true", help="also run the Docker checks")
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--instances", type=Path, help="one Pro instance id per line")
    parser.add_argument("--omp-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--archex-wheel", type=Path)
    parser.add_argument("--uv-binary", type=Path)
    parser.add_argument("--network", default="bridge")
    parser.add_argument("--broker-url", help="the running auth broker, e.g. http://127.0.0.1:8765")
    parser.add_argument(
        "--broker-bind",
        help="address the broker was started with (recorded in the report), e.g. 127.0.0.1:8765",
    )
    parser.add_argument(
        "--container-broker-url", help="default: http://host.docker.internal:<broker port>"
    )
    parser.add_argument("--broker-token-file", type=Path)
    args = parser.parse_args(argv)
    omp_command = shlex.split(args.omp_command)
    started = time.monotonic()
    checks = [
        omp_version_check(omp_command),
        model_ids_check(omp_command, args.agent_db),
        *identity_checks(),
        *broker_checks(args),
    ]
    with tempfile.TemporaryDirectory(prefix="swe-ab-stage0-") as work:
        checks += stub_arm_checks(omp_command, Path(work))
    if args.host and docker_running():
        checks += host_checks(args)
    else:
        checks += container_checks_pending(
            "the Docker daemon is not running (`docker info` failed); start Docker Desktop and "
            "rerun with --host"
            if args.host
            else "run with --host on a machine with a running Docker daemon"
        )
    checks.append(_requires_host(
        "one_real_cell_per_model",
        "hosted model call on the operator's subscriptions: route, auth through the broker, usage "
        "reporting, quota behaviour, and the annotation ledger per model; run "
        "scripts/run_swe_ab_suite.py with a one-task Stage 0 plan (RUNBOOK §4)",
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
