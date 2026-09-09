"""Run the paired product-loop baseline over the frozen population (R20).

Drives `scripts/run_product_loop_cell.py` over every planned cell — 2 arms x the
manifest's 19 tasks x 3 repetitions — and writes exactly one artifact per cell.
Each cell gets its own workspace, its own checkout, and its own wired copy of
the arm's shipped product, so no state crosses cells and neither arm inherits
the operator's ambient client, MCP, or index configuration.

Usage:

```bash
uv run python scripts/run_product_loop_suite.py \
    --manifest benchmarks/headtohead/manifest.yaml \
    --output benchmarks/product_loop/results \
    --graft-binary /path/to/graft
```

Frozen behaviour this script owns, from
`benchmarks/preregistrations/R20-product-loop-agent-baseline.md`:

* both products' shipped setup commands, with each arm's wiring committed under
  `-c core.excludesFile=/dev/null` so Archex's hook does not silently no-op on a
  dirty tree and the operator's global excludes cannot change what is captured;
* Graft's generated `npx -y @nanonets/graft mcp` entry repinned to the released
  binary R19 pinned, with both command shapes recorded;
* every hook command either product installed rewritten to the passthrough
  recorder, so hook invocations are counted at an instrumented boundary rather
  than from a transcript that does not carry them;
* an abort before the next cell once cumulative modelled cost reaches the
  ceiling, retaining everything already completed;
* resume: an existing artifact is never re-run and never overwritten.

`--dry-run` performs setup and every check except the agent invocation, so the
whole harness can be exercised without a hosted call.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from archex.benchmark.headtohead import load_headtohead_manifest, select_headtohead_tasks
from archex.benchmark.product_loop import (
    AGENT_NAME,
    AGENT_VERSION,
    COST_CEILING_USD,
    PREREGISTRATION_PATH,
    REPETITIONS,
    JsonValue,
    ProductLoopArm,
    ProductLoopError,
    as_json_array,
    as_json_object,
    assert_frozen_prompt,
    cell_filename,
    decode_json,
    sanitize_document,
)
from archex.benchmark.runner import repo_path_for_task

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkTask

_CELL_RUNNER = Path(__file__).resolve().parent / "run_product_loop_cell.py"
_GIT_NO_GLOBAL_EXCLUDES = ("-c", "core.excludesFile=/dev/null")
_RECORDER_MODULE = "archex.benchmark.product_loop_hook_recorder"


def _load_json(path: Path) -> JsonValue:
    """Read a JSON document as an explicitly typed value, not `Any`."""
    return decode_json(path.read_text(encoding="utf-8"))


def _number(value: JsonValue) -> float:
    """Read a recorded numeric field, defaulting to zero rather than guessing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


class SuiteAbortError(Exception):
    """The run stopped on a frozen abort condition; completed cells are retained."""


def _run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 1800) -> str:
    completed = subprocess.run(  # noqa: S603 - argv is built from frozen commands
        command, cwd=str(cwd), env=env, capture_output=True, text=True, check=False, timeout=timeout
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        message = f"{command[0]} failed ({completed.returncode}): {detail}"
        raise SuiteAbortError(message)
    return completed.stdout


def _commit_wiring(repo_path: Path, *, message: str, env: dict[str, str]) -> None:
    """Commit whatever the product wrote, ignoring the operator's global excludes."""
    base = ["git", "-C", str(repo_path), *_GIT_NO_GLOBAL_EXCLUDES]
    _run([*base, "add", "-A"], cwd=repo_path, env=env)
    _run(
        [
            *base,
            "-c",
            "user.email=benchmark@archex.invalid",
            "-c",
            "user.name=archex benchmark",
            "commit",
            "--allow-empty",
            "-m",
            message,
        ],
        cwd=repo_path,
        env=env,
    )


def _wire_archex(repo_path: Path, *, env: dict[str, str]) -> dict[str, JsonValue]:
    """Archex's documented install flow, up to but not including the commit."""
    _run(["archex", "init", str(repo_path), "--no-index"], cwd=repo_path, env=env)
    _run(
        ["archex", "install-client", "claude-code", str(repo_path), "--scope", "project", "-y"],
        cwd=repo_path,
        env=env,
    )
    _run(
        [
            "archex",
            "install-client",
            "claude-code",
            str(repo_path),
            "--scope",
            "project",
            "--hooks",
            "-y",
        ],
        cwd=repo_path,
        env=env,
    )
    return {}


def _wire_graft(repo_path: Path, *, env: dict[str, str], binary: str) -> dict[str, JsonValue]:
    """Graft's documented install flow, up to but not including the commit.

    `graft init` builds the graph itself, so nothing is left to construct inside
    the measured invocation.
    """
    _run([binary, "init", "--no-agents", str(repo_path)], cwd=repo_path, env=env)
    original, used = _repin_graft_mcp(repo_path, binary=binary)
    ignore_file = repo_path / ".ignore"
    greppable = ignore_file.is_file() and "!graft/" in ignore_file.read_text(encoding="utf-8")
    return {
        "mcp_command_original": original,
        "mcp_command_used": used,
        "graft_cards_greppable": greppable,
    }


def _repin_graft_mcp(repo_path: Path, *, binary: str) -> tuple[str, str]:
    """Replace `npx -y @nanonets/graft mcp` with the pinned released binary.

    `graft init` writes an `npx` entry that resolves whatever is latest on npm at
    call time, which would silently break R19's pin. Nothing else in the
    generated config is touched, and both command shapes are recorded.
    """
    config_path = repo_path / ".mcp.json"
    payload = _load_json(config_path)
    root = as_json_object(payload)
    servers = as_json_object(root["mcpServers"]) if root and "mcpServers" in root else None
    entry = as_json_object(servers["graft"]) if servers and "graft" in servers else None
    if entry is None:
        message = f"{config_path} has no graft MCP server entry to repin"
        raise SuiteAbortError(message)

    original = json.dumps(
        {"command": entry.get("command"), "args": entry.get("args")}, sort_keys=True
    )
    entry["command"] = binary
    entry["args"] = ["mcp"]
    existing_env = as_json_object(entry.get("env")) or {}
    entry["env"] = {**existing_env, "CI": "1", "DO_NOT_TRACK": "1"}
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The recorded command carries the pinned identity, not the install path:
    # an absolute binary path would violate the privacy contract.
    used = json.dumps(
        {"command": "<pinned-graft-binary>", "args": ["mcp"], "env": ["CI", "DO_NOT_TRACK"]},
        sort_keys=True,
    )
    return original, used


def _instrument_hooks(repo_path: Path, cell_dir: Path) -> int:
    """Wrap every installed hook command in the passthrough recorder.

    Applied identically to both arms. The recorder appends one row per
    invocation and reproduces the original command's stdout and exit code, so
    what the product does is unchanged and only its invocation becomes visible.
    """
    settings_path = repo_path / ".claude" / "settings.json"
    if not settings_path.is_file():
        return 0
    payload = _load_json(settings_path)
    root = as_json_object(payload)
    hooks = as_json_object(root["hooks"]) if root and "hooks" in root else None
    if hooks is None:
        return 0

    log = cell_dir / "hooks.jsonl"
    wrapped = 0
    for event, raw_groups in hooks.items():
        groups = as_json_array(raw_groups) or []
        for raw_group in groups:
            group = as_json_object(raw_group)
            if group is None:
                continue
            matcher = group.get("matcher")
            entries = as_json_array(group.get("hooks")) or []
            for raw_entry in entries:
                entry = as_json_object(raw_entry)
                if entry is None or entry.get("type") != "command":
                    continue
                original_command = entry.get("command")
                if not isinstance(original_command, str):
                    continue
                original_args = as_json_array(entry.get("args"))
                # Claude Code accepts two hook shapes and runs them differently:
                # a bare `command` string goes through a shell, a `command` plus
                # `args` list is exec'd. Archex ships the second, Graft the
                # first. The recorder must reproduce whichever it wrapped.
                use_shell = original_args is None
                tail = [original_command, *(str(part) for part in original_args or [])]
                entry["command"] = sys.executable
                entry["args"] = [
                    "-m",
                    _RECORDER_MODULE,
                    "--log",
                    str(log),
                    "--event",
                    str(event),
                    *(["--matcher", str(matcher)] if isinstance(matcher, str) else []),
                    *(["--shell"] if use_shell else []),
                    "--",
                    *tail,
                ]
                wrapped += 1
    settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return wrapped


def _freshness_command(arm: ProductLoopArm, repo_path: Path, *, graft_binary: str) -> list[str]:
    if arm is ProductLoopArm.ARCHEX:
        return ["archex", "status", str(repo_path), "--format", "json"]
    return [graft_binary, "check", "--json", str(repo_path)]


def _cell_env(cell_dir: Path, *, path: str, arm: ProductLoopArm) -> dict[str, str]:
    """Setup runs under the same home the agent will use.

    It has to: the Archex hook executes as a child of the agent, so an index
    written under a different `HOME` would be invisible to it. Subscription auth
    pins that home to the operator's real one, which means `~/.archex` is shared
    across cells. Cells stay distinct because each has its own checkout path,
    and the sharing is disclosed rather than hidden.
    """
    del cell_dir
    env = {
        "PATH": path,
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "")),
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }
    if arm is ProductLoopArm.GRAFT:
        env |= {"CI": "1", "DO_NOT_TRACK": "1"}
    return env


def _assert_agent(binary: str) -> None:
    """Refuse to run against anything but the frozen agent version."""
    completed = subprocess.run(  # noqa: S603 - binary comes from the operator's own flag
        [binary, "--version"], capture_output=True, text=True, check=False, timeout=60
    )
    reported = completed.stdout.strip()
    if completed.returncode != 0 or AGENT_VERSION not in reported:
        message = f"agent is {reported or 'unavailable'}, not the frozen {AGENT_VERSION}"
        raise SuiteAbortError(message)


def _run_cell(
    *,
    task: BenchmarkTask,
    arm: ProductLoopArm,
    repetition: int,
    cell_dir: Path,
    repo_path: Path,
    setup_seconds: float,
    extra: dict[str, JsonValue],
    agent_binary: str,
    graft_binary: str,
    path: str,
    output_dir: Path,
) -> dict[str, JsonValue]:
    payload = {
        "task": task.model_dump(mode="json"),
        "arm": arm.value,
        "repetition": repetition,
        "cell_dir": str(cell_dir),
        "repo_path": str(repo_path),
        "setup_seconds": setup_seconds,
        "agent_binary": agent_binary,
        "path": path,
        "freshness_command": _freshness_command(arm, repo_path, graft_binary=graft_binary),
        **extra,
    }
    completed = subprocess.run(  # noqa: S603 - argv is this repository's own cell runner
        [sys.executable, str(_CELL_RUNNER)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        message = f"{arm.value} cell for {task.task_id} rep {repetition} failed: {detail}"
        raise SuiteAbortError(message)

    document = sanitize_document(
        completed.stdout, replacements=[(repo_path, "<repo>"), (cell_dir, "<workspace>")]
    )
    arm_dir = output_dir / arm.value
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / cell_filename(task.task_id, repetition)).write_text(document, encoding="utf-8")
    return as_json_object(decode_json(document)) or {}


def _prepare_cell(
    *,
    arm: ProductLoopArm,
    source_repo: Path,
    cell_dir: Path,
    path: str,
    graft_binary: str,
) -> tuple[Path, float, dict[str, JsonValue]]:
    """Give the cell its own checkout, run the arm's shipped setup, instrument hooks.

    The step order is load-bearing and was fixed by a no-spend end-to-end probe
    rather than by inspection:

    1. wire the product, because both products write into the checkout;
    2. instrument the installed hooks, because that rewrites the tracked
       `.claude/settings.json` and must happen before the commit;
    3. commit, so the working tree is clean;
    4. only then index, because `archex init` indexes by default and an index
       taken before the commit sits behind `HEAD`, which makes `archex status`
       report `stale` and the shipped `PreToolUse` hook no-op on every call.

    Getting any of those wrong runs the Archex arm with its hook silently
    disabled while still producing a plausible-looking scored cell.
    """
    repo_path = cell_dir / "repo"
    (cell_dir / "home").mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_repo, repo_path)
    env = _cell_env(cell_dir, path=path, arm=arm)

    started = time.perf_counter()
    extra = (
        _wire_archex(repo_path, env=env)
        if arm is ProductLoopArm.ARCHEX
        else _wire_graft(repo_path, env=env, binary=graft_binary)
    )

    # Both products install hooks. Wrapping none of them means the shipped
    # wiring did not land, which would publish a silent zero for this arm's
    # hook-invocation metric instead of a visible failure.
    if _instrument_hooks(repo_path, cell_dir) == 0:
        message = f"{arm.value} installed no hook command under {repo_path.name}/.claude"
        raise SuiteAbortError(message)

    _commit_wiring(repo_path, message=f"wire {arm.mcp_server}", env=env)
    if arm is ProductLoopArm.ARCHEX:
        _run(["archex", "index", str(repo_path)], cwd=repo_path, env=env)
    setup_seconds = time.perf_counter() - started
    return repo_path, setup_seconds, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/headtohead/manifest.yaml")
    )
    parser.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/tasks"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/product_loop/results"))
    parser.add_argument("--preregistration", type=Path, default=Path(PREREGISTRATION_PATH))
    parser.add_argument("--agent-binary", default=AGENT_NAME)
    parser.add_argument("--graft-binary", default="graft")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--cost-ceiling", type=float, default=COST_CEILING_USD)
    parser.add_argument("--task", action="append", default=None, help="run only these task ids")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare and check every cell without invoking the agent",
    )
    args = parser.parse_args()

    try:
        assert_frozen_prompt(args.preregistration)
    except ProductLoopError as exc:
        print(f"frozen-prompt check failed: {exc}", file=sys.stderr)
        return 1
    if not args.dry_run:
        _assert_agent(args.agent_binary)

    manifest = load_headtohead_manifest(args.manifest)
    tasks = select_headtohead_tasks(manifest, args.tasks_dir)
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task.task_id in wanted]

    path = os.environ.get("PATH", "")
    workspace = Path(tempfile.mkdtemp(prefix="archex-product-loop-"))
    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path] = {}
    cleanup_paths: list[Path] = []
    spent = 0.0
    completed_cells = 0
    planned = len(tasks) * len(ProductLoopArm) * args.repetitions

    try:
        for task in tasks:
            source_repo = repo_path_for_task(task, repo_cache, cleanup_paths)
            for arm in ProductLoopArm:
                for repetition in range(1, args.repetitions + 1):
                    filename = cell_filename(task.task_id, repetition)
                    artifact_path = args.output / arm.value / filename
                    if artifact_path.is_file():
                        existing = as_json_object(_load_json(artifact_path)) or {}
                        spent += _number(existing.get("modelled_cost_usd"))
                        completed_cells += 1
                        print(f"resume: {arm.value}/{task.task_id}#{repetition}", flush=True)
                        continue
                    if spent >= args.cost_ceiling:
                        message = (
                            f"cost ceiling {args.cost_ceiling:.2f} reached at "
                            f"{spent:.4f}; aborting before "
                            f"{arm.value}/{task.task_id}#{repetition}"
                        )
                        raise SuiteAbortError(message)

                    cell_dir = workspace / arm.value / task.task_id / f"rep{repetition}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    repo_path, setup_seconds, extra = _prepare_cell(
                        arm=arm,
                        source_repo=source_repo,
                        cell_dir=cell_dir,
                        path=path,
                        graft_binary=args.graft_binary,
                    )
                    if args.dry_run:
                        print(
                            f"dry-run: {arm.value}/{task.task_id}#{repetition} "
                            f"setup={setup_seconds:.1f}s mcp="
                            f"{(repo_path / '.mcp.json').is_file()}",
                            flush=True,
                        )
                        shutil.rmtree(cell_dir, ignore_errors=True)
                        continue

                    cell = _run_cell(
                        task=task,
                        arm=arm,
                        repetition=repetition,
                        cell_dir=cell_dir,
                        repo_path=repo_path,
                        setup_seconds=setup_seconds,
                        extra=extra,
                        agent_binary=args.agent_binary,
                        graft_binary=args.graft_binary,
                        path=path,
                        output_dir=args.output,
                    )
                    spent += _number(cell.get("modelled_cost_usd"))
                    completed_cells += 1
                    print(
                        f"[{completed_cells}/{planned}] {arm.value}/{task.task_id}"
                        f"#{repetition} status={cell['status']} "
                        f"completeness={_number(cell.get('completeness')):.3f} "
                        f"tools={cell['tool_calls']} hooks={cell['hook_invocations']} "
                        f"cost={_number(cell.get('modelled_cost_usd')):.4f} spent={spent:.4f}",
                        flush=True,
                    )
                    shutil.rmtree(cell_dir / "repo", ignore_errors=True)
    except SuiteAbortError as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        print(f"retained {completed_cells}/{planned} cells, spent {spent:.4f}", file=sys.stderr)
        return 2
    finally:
        for path_to_clean in cleanup_paths:
            shutil.rmtree(path_to_clean, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)

    print(f"wrote {completed_cells}/{planned} cells, spent {spent:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
