"""Passthrough recorder for a product's shipped Claude Code hook (R20).

Claude Code surfaces `system`/`hook_started` and `hook_response` events for
`SessionStart` only. Archex's `PreToolUse` hook and Graft's `UserPromptSubmit`
and `PostToolUse` hooks execute but never appear in the transcript, so counting
hook invocations from the transcript would under-report both arms.

The product-loop runner therefore rewrites every hook `command` the product
installed to::

    python -m archex.benchmark.product_loop_hook_recorder \\
        --log <cell>/hooks.jsonl --event PreToolUse --matcher 'Glob|Grep' \\
        -- <original command and arguments>

This module appends one JSONL row describing the invocation, then runs the
original command with the same stdin and reproduces its stdout, stderr, and exit
code byte for byte. It must never change what the product does: the row is a
side effect, the passthrough is the contract, and a recorder failure is swallowed
rather than allowed to break the client's hook.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def _tool_name(stdin_bytes: bytes) -> str | None:
    """Best-effort tool name from a hook payload, without retaining the payload."""
    try:
        decoded: Any = json.loads(stdin_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    payload = cast("object", decoded)
    if not isinstance(payload, dict):
        return None
    name: object = cast("dict[str, object]", payload).get("tool_name")
    return name if isinstance(name, str) and name else None


def _append_row(log: Path, row: dict[str, object]) -> None:
    """Append one record, never failing the hook if the log is unwritable."""
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--matcher", default=None)
    parser.add_argument(
        "--shell",
        action="store_true",
        help=(
            "run the wrapped command through a shell, as Claude Code does for a bare command string"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = [part for part in args.command if part != "--"]
    stdin_bytes = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
    row: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": args.event,
        "matcher": args.matcher,
        "tool_name": _tool_name(stdin_bytes),
        "stdin_bytes": len(stdin_bytes),
        "stdout_bytes": 0,
        "exit_code": 0,
        "latency_ms": 0.0,
        "augmented": False,
    }
    if not command:
        row["recorder_error"] = "no wrapped command"
        _append_row(args.log, row)
        return 0

    started = time.perf_counter()
    try:
        if args.shell:
            # Claude Code runs a bare `command` string through a shell, so a
            # recorder that exec'd it directly would fail on every quoted or
            # multi-word hook command — which is exactly how Graft ships its
            # hooks, and would have silently recorded zero invocations.
            completed = subprocess.run(  # noqa: S602 - mirrors the client's own shell execution
                " ".join(command),
                input=stdin_bytes,
                capture_output=True,
                check=False,
                shell=True,
            )
        else:
            completed = subprocess.run(  # noqa: S603 - command comes from the cell's own settings
                command,
                input=stdin_bytes,
                capture_output=True,
                check=False,
            )
    except OSError as exc:
        # Fail open at the client boundary: a recorder problem must never become
        # a product failure, but it must be visible rather than look like a hook
        # that simply never fired.
        row["latency_ms"] = (time.perf_counter() - started) * 1000.0
        row["recorder_error"] = f"{type(exc).__name__}: {exc}"
        _append_row(args.log, row)
        return 0

    row["latency_ms"] = (time.perf_counter() - started) * 1000.0
    row["stdout_bytes"] = len(completed.stdout)
    row["exit_code"] = completed.returncode
    row["augmented"] = b"additionalContext" in completed.stdout
    _append_row(args.log, row)

    sys.stdout.buffer.write(completed.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(completed.stderr)
    sys.stderr.buffer.flush()
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
