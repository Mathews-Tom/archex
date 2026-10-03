"""Stage 0 feasibility checks for the SWE A/B campaign (spec §2.5, §8), as JSON.

```bash
# Local checks (no spend, no containers): omp's route to the campaign's models, the
# provider key (a zero-token request for a model that does not exist), the
# `reasoning_effort` each configuration's request carries, and every arm once against the
# local stub provider, inspecting what omp actually sent. The campaign's key variable comes
# from the environment, else from its `NAME=` line in --env-file (default: repo-root .env).
uv run python scripts/swe_ab_stage0.py --campaign benchmarks/swe_ab/campaigns/muna.yml \
    --output /tmp/stage0.json

# On a host with a running Docker daemon (x86_64 Linux, or Docker Desktop on Apple silicon,
# where cells are emulated and recorded as such), add the container checks:
uv run python scripts/swe_ab_stage0.py --campaign benchmarks/swe_ab/campaigns/muna.yml \
    --output stage0.json --host \
    --tasks-root SWE-bench_Pro-os/v2/tasks --instances instances.txt \
    --omp-dir /opt/omp-linux-x64 --omp-command /opt/omp/bin/omp \
    --archex-wheel dist/archex-0.34.0-py3-none-any.whl --uv-binary /opt/uv/uv
```

Each check reports ``pass``, ``fail``, or ``requires_host`` (needs a running Docker daemon,
the omp bundle and task images, or a hosted model, and was not run). The gate passes only
when no check fails and none is left ``requires_host``. No check here generates a token: the
key check asks the provider for a model that does not exist (a 4xx rejection of the model for a
good key, 401 for a bad one),
and the container checks use only the task images, ``alpine``, and Bun's own images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, cast

from archex.benchmark.swe_ab import (
    ANNOTATION_MARKER,
    BASE_TOOLS,
    CLI_GUIDE_PATH,
    COMPRESSOR_MARKERS,
    HOOK_TIMEOUT_SECONDS,
    OMP_CONFIG_PATH,
    OMP_VERSION,
    PROFILE,
    Campaign,
    Configuration,
    SweAbArm,
    SweAbError,
    is_emulated,
    load_campaign,
    load_cell,
    load_plan,
    omp_argv,
    plan_campaign,
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
_STUB_PROVIDER = _ROOT / "scripts" / "swe_ab_stub_provider.py"
_DRY_RUN_PLAN = _ROOT / "benchmarks" / "swe_ab" / "dry-run-plan.json"
_STUB_SCRIPT = _ROOT / "benchmarks" / "swe_ab" / "stub-script.json"
_CAMPAIGN_PYTHON = "/opt/archex/venv/bin/python"
CONTAINER_OMP_COMMAND = "/opt/omp/bin/omp-entry"
"""omp's entry point inside a task container: `benchmarks/swe_ab/omp-entry.sh` in the bundle."""
_LATENCY_SAMPLES = 10
_LATENCY_DRIVER = """
import json, os, subprocess, sys, time
python, samples = sys.argv[1], int(sys.argv[2])
command = ["git", "grep", "-n", "-I", "-w", "-e", "return", "--",
           "*.py", "*.go", "*.js", "*.jsx", "*.ts", "*.tsx"]
hits = subprocess.run(command, capture_output=True, text=True).stdout.splitlines()[:20]
request = json.dumps({"host": "omp", "tool": "bash", "input": {"command": " ".join(command)},
                      "text": "\\n".join(hits), "cwd": os.getcwd()})
env = {**os.environ, "ARCHEX_HOOK_DIAGNOSTICS_LOG": "/dev/null"}
env.pop("ARCHEX_ANNOTATION_LEDGER", None)
rows = []
for _ in range(samples):
    started = time.monotonic()
    done = subprocess.run([python, "-m", "archex.integrations.annotate_hook"], input=request,
                          capture_output=True, text=True, env=env)
    ms = round((time.monotonic() - started) * 1000)
    try:
        record = json.loads(done.stdout)
    except ValueError:
        record = {}
    rows.append({"ms": ms, "annotated": record.get("annotated") is True,
                 "reason": record.get("reason")})
print(json.dumps({"hits": len(hits), "rows": rows}))
"""


def annotate_latency(rt: Any, python: str = _CAMPAIGN_PYTHON) -> dict[str, Any]:
    """Time the annotate entry end to end, as the hook spawns it, on a real multi-hit search.

    Runs inside the container after setup (index fresh, entry pre-warmed), so the numbers are
    the steady-state cost every agent search pays under this host's emulation, against the
    hook's wall-clock budget. A call over budget is dropped by the hook and adds nothing.
    """
    done = rt.run([python, "-c", _LATENCY_DRIVER, python, str(_LATENCY_SAMPLES)])
    report = cast("dict[str, Any]", json.loads(done.stdout))
    rows = cast("list[dict[str, Any]]", report["rows"])
    latencies = sorted(int(row["ms"]) for row in rows)
    budget_ms = round(HOOK_TIMEOUT_SECONDS * 1000)
    return {
        "hits": report["hits"],
        "samples": len(rows),
        "p50_ms": statistics.median(latencies),
        "p90_ms": latencies[max(0, math.ceil(0.9 * len(latencies)) - 1)],
        "max_ms": latencies[-1],
        "budget_ms": budget_ms,
        "over_budget": sum(1 for ms in latencies if ms > budget_ms),
        "annotated": sum(1 for row in rows if row["annotated"]),
        "reasons": sorted({str(row["reason"]) for row in rows if not row["annotated"]}),
    }


def _check(check_id: str, status: str, detail: str, **data: Any) -> dict[str, Any]:
    return {"id": check_id, "status": status, "detail": detail, **data}


def _requires_host(check_id: str, detail: str) -> dict[str, Any]:
    return _check(check_id, "requires_host", detail)


def omp_version_check(omp_command: list[str]) -> dict[str, Any]:
    done = subprocess.run([*omp_command, "--version"], capture_output=True, text=True, check=False)
    found = done.stdout.strip().removeprefix("omp/")
    status = "pass" if found == OMP_VERSION else "fail"
    return _check("omp_version", status, f"omp reports {found!r}; pinned {OMP_VERSION}")


_KEY_ACCEPTED_STATUSES = (400, 404, 422)
_KEY_PROBE_MODEL = "nobody/does-not-exist"


def _no_key(campaign: Campaign) -> str:
    return f"{campaign.credential_env} is neither exported nor in the env file; no request was made"


_STUB_REPLY = [{"text": "ok"}]


def _omp_env(home: Path, agent_dir: Path | None = None) -> dict[str, str]:
    """The host environment minus omp's own variables, under an isolated ``HOME``."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "PI_"))}
    env["HOME"] = str(home)
    if agent_dir is not None:
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    return env


def provider_route_check(
    omp_command: list[str], campaign: Campaign, *, run: Any = subprocess.run
) -> dict[str, Any]:
    """omp lists every configuration's selector under the campaign's provider; no model call.

    ``omp models --json`` runs in a throwaway profile holding only the campaign's provider config.
    """
    expected = sorted({config.selector for config in campaign.configurations})
    with tempfile.TemporaryDirectory(prefix="swe-ab-route-") as tmp:
        agent_dir, home = Path(tmp) / "agent", Path(tmp) / "home"
        agent_dir.mkdir()
        home.mkdir()
        shutil.copy2(_ROOT / campaign.provider_config, agent_dir / "models.yml")
        done = run(
            [*omp_command, "models", "--json"],
            capture_output=True, text=True, check=False, env=_omp_env(home, agent_dir),
            timeout=120,
        )  # fmt: skip
    try:
        catalog = cast("list[dict[str, Any]]", json.loads(done.stdout)["models"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return _check("provider_route", "fail", "`omp models --json` did not return a model list")
    found = sorted(
        str(entry.get("selector"))
        for entry in catalog
        if entry.get("provider") == campaign.provider
    )
    missing = [selector for selector in expected if selector not in found]
    return _check(
        "provider_route",
        "fail" if missing else "pass",
        f"omp lists every configuration's selector under provider {campaign.provider!r} "
        "(catalog only; no model call)"
        if not missing
        else f"not listed under provider {campaign.provider!r}: {', '.join(missing)}",
        expected=expected,
        listed=found,
    )


def campaign_api_key(campaign: Campaign, env_file: Path) -> str | None:
    """The key from the environment, else from the env file's ``<credential_env>=`` line only."""
    try:
        return cell_suite.campaign_key(campaign, env_file)
    except SystemExit:
        return None


def _post_json(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:  # noqa: S310
            return int(reply.status), reply.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def provider_key_check(
    campaign: Campaign, env_file: Path, *, post: Any = _post_json
) -> dict[str, Any]:
    """Zero-token key check: a chat request for a model that does not exist generates nothing.

    A good key is authenticated and the request is rejected for its model (Muna: 404
    ``model_not_found``; OpenRouter: 400 "not a valid model ID"); a bad or absent key is 401.
    Pass on 400, 404, or 422. The key and the Authorization header appear in no detail, log,
    or report.
    """
    check_id = "provider_key_accepted"
    key = campaign_api_key(campaign, env_file)
    if key is None:
        return _check(check_id, "fail", _no_key(campaign))
    body = json.dumps({
        "model": _KEY_PROBE_MODEL,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "."}],
    }).encode()  # fmt: skip
    try:
        status, _text = post(
            f"{campaign.base_url}/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            body,
        )
    except (OSError, ValueError) as exc:
        return _check(check_id, "fail", f"{campaign.provider} unreachable: {type(exc).__name__}")
    if status in _KEY_ACCEPTED_STATUSES:
        return _check(
            check_id, "pass",
            f"{campaign.provider} authenticated the key (HTTP {status} for a nonexistent model; "
            "no tokens spent)",
            http_status=status,
        )  # fmt: skip
    if status in (401, 403):
        return _check(check_id, "fail", f"key rejected (HTTP {status})", http_status=status)
    return _check(check_id, "fail", f"unexpected answer: HTTP {status}", http_status=status)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def capture_first_request(
    omp_command: list[str], campaign: Campaign, config: Configuration, work: Path
) -> dict[str, Any] | None:
    """One stub-backed omp run of ``config``; the first request body omp sent, if any.

    The command line is the one cells run (`omp_argv`, arm A0); the profile is the campaign's
    provider config with only its base URL pointed at the stub.
    """
    work.mkdir(parents=True, exist_ok=True)
    reply = work / "reply.json"
    reply.write_text(json.dumps(_STUB_REPLY), encoding="utf-8")
    capture = work / "capture"
    port = _free_port()
    home = work / "home"
    profile = home / ".omp" / "profiles" / PROFILE / "agent"
    profile.mkdir(parents=True)
    (profile / "models.yml").write_text(
        cell_suite.stub_provider_config(campaign, port), encoding="utf-8"
    )
    prompt = work / "prompt.md"
    prompt.write_text("Say ok.\n", encoding="utf-8")
    repo = work / "repo"
    repo.mkdir()
    argv = omp_argv(
        omp_command,
        arm=SweAbArm.A0,
        config=config,
        prompt_path=str(prompt),
        session_dir=str(work / "session"),
        hook_module_path=None,
        cli_guide_path=None,
        omp_config_path=str(_ROOT / OMP_CONFIG_PATH),
    )
    stub = subprocess.Popen(
        [sys.executable, str(_STUB_PROVIDER), "--port", str(port), "--script", str(reply)]
        + ["--capture-dir", str(capture)],
        stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert stub.stdout is not None
        stub.stdout.readline()  # "stub listening on ..."
        subprocess.run(
            argv, cwd=repo, env=_omp_env(home),
            capture_output=True, text=True, check=False, timeout=240,
        )  # fmt: skip
    finally:
        stub.terminate()
        stub.wait()
    first = sorted(capture.glob("request-*.json"))[:1]
    return cast("dict[str, Any]", json.loads(first[0].read_text())) if first else None


def effort_request_shape_check(
    omp_command: list[str], campaign: Campaign, work: Path, *, capture: Any = capture_first_request
) -> dict[str, Any]:
    """Each configuration's first request carries ``reasoning_effort`` equal to its effort.

    omp sends no effort for a Qwen model unless the model has ``thinkingFormat: openai``, which
    would make the low and high configurations byte-identical requests.
    """
    sent: dict[str, Any] = {}
    problems: list[str] = []
    for config in campaign.configurations:
        try:
            body = capture(omp_command, campaign, config, work / config.label)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            sent[config.label] = None
            problems.append(f"{config.label}: omp run failed ({type(exc).__name__})")
            continue
        if body is None:
            sent[config.label] = None
            problems.append(f"{config.label}: omp sent no request")
            continue
        effort = body.get("reasoning_effort")
        sent[config.label] = effort
        if effort != config.thinking:
            problems.append(
                f"{config.label}: expected reasoning_effort {config.thinking!r}, "
                f"the request carried {effort!r}"
            )
    return _check(
        "effort_request_shape",
        "fail" if problems else "pass",
        "every configuration's first request carries its thinking effort as `reasoning_effort`"
        if not problems
        else "; ".join(problems),
        reasoning_effort=sent,
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
        plan = load_plan(_DRY_RUN_PLAN)
        validate_swe_ab_directory(cells_dir, plan, plan_campaign(plan, root=_ROOT))
        refused = "fail"
        detail = "the validator accepted stub-endpoint cells for publication"
    except SweAbError as exc:
        refused = "pass" if "a local stub or another endpoint" in str(exc) else "fail"
        detail = str(exc)
    checks.append(_check("validator_refuses_stub_cells", refused, detail))
    return checks


def identity_checks(campaign: Campaign) -> list[dict[str, Any]]:
    module = render_annotation_hook_module(_CAMPAIGN_PYTHON)
    return [
        _check(
            "frozen_identities", "pass", "SHA-256 of the artifacts the pre-registration pins",
            cli_guide_sha256=sha256_file(_ROOT / CLI_GUIDE_PATH),
            hook_module_sha256=hashlib.sha256(module.encode()).hexdigest(),
            hook_module_python=_CAMPAIGN_PYTHON,
            campaign_sha256=campaign.sha256,
            provider_config_sha256=campaign.provider_config_sha256,
            omp_config_sha256=sha256_file(_ROOT / OMP_CONFIG_PATH),
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


def provider_container_check(
    campaign: Campaign, env_file: Path, *, run: Any = subprocess.run
) -> dict[str, Any]:
    """From a throwaway container: reach the provider, and see only the allow-listed environment.

    The probe container is started like a cell's; the allow-listed variable is then passed the
    way the cell runner passes it (a bare ``-e NAME``, the value in the docker client's
    environment), and the names visible inside are compared with what the host exports. The
    fetch is the unauthenticated ``GET <base_url>/models``; the key never goes on a command line.
    """
    check_id = "provider_reachable_from_container"
    key = campaign_api_key(campaign, env_file)
    if key is None:
        return _check(check_id, "fail", _no_key(campaign))
    name = f"swe-ab-probe-{uuid.uuid4().hex[:8]}"
    started = time.monotonic()
    try:
        up = run(
            ["docker", "run", "-d", "--rm", "--name", name, "alpine:3", "sleep", "120"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if up.returncode != 0:
            return _check(check_id, "fail", up.stderr[-300:])
        fetch = run(
            ["docker", "exec", name, "wget", "-q", "-O", "-", "-T", "15"]
            + [f"{campaign.base_url}/models"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        reachable = fetch.returncode == 0 and '"data"' in fetch.stdout
        env_args, client_env = cell_runner.docker_exec_env(
            {campaign.credential_env: key}, secret_names=[campaign.credential_env]
        )
        client_env["SWE_AB_STAGE0_HOST_SENTINEL"] = "1"
        names = run(
            ["docker", "exec", *env_args, name, "sh", "-c", "env | cut -d= -f1 | sort"],
            capture_output=True, text=True, check=False, env=client_env,
        )  # fmt: skip
        seen = set(names.stdout.split())
        allowlisted = campaign.credential_env in seen and "SWE_AB_STAGE0_HOST_SENTINEL" not in seen
    finally:
        run(["docker", "rm", "-f", name], capture_output=True, check=False)
    ok = reachable and allowlisted
    return _check(
        check_id,
        "pass" if ok else "fail",
        f"a container fetches {campaign.base_url}/models and receives the allow-listed variable "
        "but nothing else from the host environment"
        if ok
        else f"reachable={reachable}, allow_list_only={allowlisted}: {fetch.stderr[-200:]}",
        reachable=reachable,
        allow_list_only=allowlisted,
        seconds=round(time.monotonic() - started, 1),
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


def host_checks(args: argparse.Namespace, campaign: Campaign) -> list[dict[str, Any]]:
    """Container checks; run only with --host against a running Docker daemon."""
    instances = [line.strip() for line in args.instances.read_text().splitlines() if line.strip()]
    base = {
        "runtime": "docker", "model": campaign.labels[0], "campaign": campaign.path, "arm": "HC",
        "repetition": 1, "repo": "stage0",
        "omp_command": shlex.split(args.container_omp_command),
        "profile_dir": str(args.profile_dir or "."),
        "archex_wheel": str(args.archex_wheel), "uv_binary": str(args.uv_binary),
        "omp_dir": str(args.omp_dir), "network": args.network,
    }  # fmt: skip
    checks: list[dict[str, Any]] = []
    validity: dict[str, dict[str, bool]] = {}
    scoring_errors: dict[str, str] = {}
    index_seconds: dict[str, float | None] = {}
    latency: dict[str, dict[str, Any] | None] = {}
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
        try:
            gold_resolves = cell_runner.score_patch(spec, task_dir / "solution" / "gold_patch.diff")
            timing["gold_score_seconds"] = round(time.monotonic() - lap, 1)
            lap = time.monotonic()
            validity[instance] = {
                "gold_resolves": gold_resolves,
                "empty_fails": not cell_runner.score_patch(spec, empty),
            }
            timing["empty_score_seconds"] = round(time.monotonic() - lap, 1)
        except Exception as exc:  # noqa: BLE001 - an unscorable instance is a named failure
            # Not a validity verdict: the pool excludes only instances whose gold patch does not
            # resolve or whose empty patch does not fail, so this stops the gate instead.
            scoring_errors[instance] = repr(exc)[-300:]
        lap = time.monotonic()
        rt: Any = None
        try:
            rt = cell_runner.DockerRuntime(spec, mounts=[(args.omp_dir, "/opt/omp")])
            timing["container_start_seconds"] = round(time.monotonic() - lap, 1)
            timing["container_arch"] = rt.run(["uname", "-m"], cwd="/").stdout.strip()
            lap = time.monotonic()
            version = rt.run([*spec.omp_command, "--version"], cwd="/").stdout.strip()
            timing["omp_start_seconds"] = round(time.monotonic() - lap, 1)
            omp_ok[instance] = version == f"omp/{OMP_VERSION}"
            lap = time.monotonic()
            index_seconds[instance] = cell_runner.index_in_container(spec, rt)
            timing["install_and_index_seconds"] = round(time.monotonic() - lap, 1)
            latency[instance] = annotate_latency(rt)
        except Exception as exc:  # noqa: BLE001 - reported as a failed check, not raised
            index_seconds.setdefault(instance, None)
            latency.setdefault(instance, None)
            omp_ok.setdefault(instance, False)
            checks.append(_check(f"container_setup:{instance}", "fail", repr(exc)))
        finally:
            if rt is not None:
                rt.close()
    invalid = [i for i, v in validity.items() if not (v["gold_resolves"] and v["empty_fails"])]
    source = f"stage0 gold_empty_validity, emulated={is_emulated('docker', platform.machine())}"
    exclusions = [
        {
            "instance_id": instance,
            "reason": "gold_not_resolved"
            if not validity[instance]["gold_resolves"]
            else "empty_not_failing",
            "source": source,
        }
        for instance in invalid
    ]
    checks.append(_check(
        "gold_empty_validity", "fail" if invalid or scoring_errors else "pass",
        "gold patch resolves and empty patch fails, per instance (invalid ones leave the pool; "
        "an instance that could not be scored is an error, not an exclusion)",
        per_instance=validity, excluded=invalid, exclusions=exclusions,
        scoring_errors=scoring_errors,
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
    slow = [
        instance
        for instance, measured in latency.items()
        if measured is None or measured["p50_ms"] > measured["budget_ms"]
    ]
    checks.append(_check(
        "annotate_latency_in_container", "fail" if slow else "pass",
        f"steady-state annotate calls on a real search, {_LATENCY_SAMPLES} per instance, end to "
        "end as the hook spawns them; the median must fit the hook's wall-clock budget, or the "
        "hook arms drop most annotations (kill criterion 5)",
        per_instance=latency, over_budget_instances=slow,
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
    checks.append(provider_container_check(campaign, args.env_file))
    checks.append(bun_emulation_check())
    return checks


def container_checks_pending(campaign: Campaign, why: str) -> list[dict[str, Any]]:
    """The checks that need a running Docker daemon, left ``requires_host`` with the reason."""
    return [
        _requires_host(check_id, f"{detail}; {why}")
        for check_id, detail in (
            ("gold_empty_validity", "Pro images: gold patch resolves, empty fails"),
            (
                "annotate_latency_in_container",
                "steady-state annotate latency against the hook budget inside a Pro image",
            ),
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
                "provider_reachable_from_container",
                f"a container reaches {campaign.base_url}/models and gets only the allow-listed "
                "variable",
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
    parser.add_argument(
        "--omp-command",
        default=shutil.which("omp") or "omp",
        help="the host's omp (pinned build) for the local checks",
    )
    parser.add_argument(
        "--container-omp-command",
        default=CONTAINER_OMP_COMMAND,
        help="omp's entry point inside a task container, from the bundle mounted at /opt/omp",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="campaign file (repo-relative or absolute): the provider and configurations to check",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=_ROOT / ".env",
        help="read only the campaign's credential variable line, and only when it is not exported",
    )
    parser.add_argument("--host", action="store_true", help="also run the Docker checks")
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--instances", type=Path, help="one Pro instance id per line")
    parser.add_argument("--omp-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--archex-wheel", type=Path)
    parser.add_argument("--uv-binary", type=Path)
    parser.add_argument("--network", default="bridge")
    args = parser.parse_args(argv)
    campaign = load_campaign(args.campaign, root=_ROOT)
    omp_command = shlex.split(args.omp_command)
    started = time.monotonic()
    checks = [
        omp_version_check(omp_command),
        provider_route_check(omp_command, campaign),
        provider_key_check(campaign, args.env_file),
        *identity_checks(campaign),
    ]
    with tempfile.TemporaryDirectory(prefix="swe-ab-stage0-") as work:
        checks.append(effort_request_shape_check(omp_command, campaign, Path(work) / "effort"))
        checks += stub_arm_checks(omp_command, Path(work))
    if args.host and docker_running():
        checks += host_checks(args, campaign)
    else:
        checks += container_checks_pending(
            campaign,
            "the Docker daemon is not running (`docker info` failed); start Docker Desktop and "
            "rerun with --host"
            if args.host
            else "run with --host on a machine with a running Docker daemon",
        )
    checks.append(_requires_host(
        "one_real_cell_per_configuration",
        f"hosted model call on {campaign.provider}: route, auth, usage reporting, the effort the "
        f"request carried, and the annotation ledger per configuration (arm H, "
        f"{len(campaign.configurations)} cells); run scripts/run_swe_ab_suite.py with a one-task "
        "Stage 0 plan (RUNBOOK §4)",
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
