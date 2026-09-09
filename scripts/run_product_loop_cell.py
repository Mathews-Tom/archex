"""Run one paired product-loop cell and emit its artifact on stdout (R20).

Reads a JSON payload on stdin and writes exactly one
:class:`~archex.benchmark.product_loop.ProductLoopCellArtifact` document to
stdout. Every planned cell produces a document: a scored measurement, or a
recorded failure with zeroed scoring and a sanitized reason. A cell is never
missing and never repaired in place.

Payload keys:

- ``task``: a serialized :class:`~archex.benchmark.models.BenchmarkTask`
- ``arm``: ``archex_product_loop`` or ``graft_product_loop``
- ``repetition``: 1-based repetition index
- ``cell_dir``: per-cell workspace holding ``home/`` and ``hooks.jsonl``
- ``repo_path``: the cell's own wired checkout
- ``setup_seconds``: time the arm's shipped setup took, measured by the suite
- ``agent_binary``, ``path``: resolved `claude` binary and the fixed ``PATH``
- ``freshness_command``: argv for the arm's freshness probe, run after the agent
  exits so it never enters the measured wall time
- ``mcp_command_original`` / ``mcp_command_used``: Graft's floating ``npx`` entry
  and the pinned rewrite, recorded for audit
- ``graft_cards_greppable``: whether the Graft arm's ``.ignore`` re-admits cards
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from archex.benchmark.models import BenchmarkTask
from archex.benchmark.product_loop import (
    AGENT_MODEL,
    AGENT_NAME,
    AGENT_VERSION,
    BASE_TOOLS,
    BILLING_MODE,
    CELL_DEADLINE_SECONDS,
    DENIED_TOOLS,
    INSTRUCTION_BLOCK_SHA256,
    HookRecord,
    JsonValue,
    ProductLoopAnswerFlag,
    ProductLoopArm,
    ProductLoopCellArtifact,
    ProductLoopCellStatus,
    ProductLoopError,
    ProductLoopFailureReason,
    TranscriptSummary,
    as_json_array,
    as_json_object,
    as_json_text,
    build_prompt,
    decode_json,
    extract_answer_paths,
    sanitize_document,
    score_completeness,
    summarize_transcript,
)


def _agent_command(*, arm: ProductLoopArm, prompt: str, repo_path: Path, binary: str) -> list[str]:
    """The frozen measured invocation; identical but for the arm's MCP config."""
    return [
        binary,
        "-p",
        prompt,
        "--model",
        AGENT_MODEL,
        "--output-format",
        "stream-json",
        "--verbose",
        "--mcp-config",
        str(repo_path / ".mcp.json"),
        "--strict-mcp-config",
        "--allowedTools",
        *sorted(BASE_TOOLS),
        f"mcp__{arm.mcp_server}",
        "--disallowedTools",
        *DENIED_TOOLS,
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
    ]


def _cell_env(*, cell_dir: Path, path: str, arm: ProductLoopArm) -> tuple[dict[str, str], bool]:
    """A scrubbed environment, plus whether the provider endpoint was overridden.

    Nothing ambient reaches the agent. The two `ANTHROPIC_*` overrides are the
    single exception: they exist so the whole harness can be exercised against a
    local stub endpoint without a hosted call. A cell that used them records the
    fact, and the validator refuses to publish it as evidence.
    """
    home = cell_dir / "home"
    env = {
        "PATH": path,
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }
    if arm is ProductLoopArm.GRAFT:
        env["CI"] = "1"
        env["DO_NOT_TRACK"] = "1"
    overridden = False
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(name)
        if value:
            env[name] = value
            overridden = overridden or name == "ANTHROPIC_BASE_URL"
    return env, overridden


def _run_agent(
    command: list[str], *, env: dict[str, str], cwd: Path, deadline: int
) -> tuple[str, int, bool]:
    """Run the agent under an external deadline, returning stdout and how it ended.

    The pinned agent exposes no turn or time cap, so the deadline is enforced by
    terminating the whole process group; a timeout and an operator cancel share
    this single path because the agent does not distinguish them either.
    """
    process = subprocess.Popen(  # noqa: S603 - argv is built from frozen constants
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=deadline)
        return stdout, process.returncode, False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, _ = process.communicate()
        return stdout or "", process.returncode, True


def _hook_records(cell_dir: Path) -> list[HookRecord]:
    """Read the instrumented hook boundary's rows, ignoring recorder anomalies."""
    log = cell_dir / "hooks.jsonl"
    if not log.is_file():
        return []
    records: list[HookRecord] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = as_json_object(decode_json(line))
        if row is None or "recorder_error" in row:
            continue
        records.append(
            HookRecord(
                event=as_json_text(row.get("event")) or "",
                matcher=as_json_text(row.get("matcher")),
                tool_name=as_json_text(row.get("tool_name")),
                stdin_bytes=_as_int(row.get("stdin_bytes")),
                stdout_bytes=_as_int(row.get("stdout_bytes")),
                exit_code=_as_int(row.get("exit_code")),
                latency_ms=_as_float(row.get("latency_ms")),
                augmented=row.get("augmented") is True,
            )
        )
    return records


def _as_int(value: JsonValue) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _as_float(value: JsonValue) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _probe_freshness(
    command: list[str], *, arm: ProductLoopArm, env: dict[str, str], cwd: Path
) -> tuple[str | None, bool]:
    """Read the arm's freshness state after the agent has exited.

    Never inside the measured wall time. With no write tools on either arm the
    frozen task family performs no edits, so this is expected to be `fresh` on
    every cell; it is recorded anyway because the later paired re-measurement
    needs the field and an edit-bearing family will make it informative.
    """
    if not command:
        return None, False
    completed = subprocess.run(  # noqa: S603 - argv is built by the suite from frozen commands
        command, cwd=str(cwd), env=env, capture_output=True, text=True, check=False, timeout=120
    )
    if completed.returncode not in (0, 1) or not completed.stdout.strip():
        return None, False
    try:
        payload = as_json_object(decode_json(completed.stdout))
    except json.JSONDecodeError:
        return None, False
    if payload is None:
        return None, False
    if arm is ProductLoopArm.ARCHEX:
        state = as_json_text(payload.get("state"))
        return state, state is not None and state != "fresh"
    graph = as_json_object(payload.get("graph"))
    if graph is None:
        return None, False
    ok = graph.get("ok") is True
    return ("fresh" if ok else "stale"), not ok


def _failure_reason(
    *, timed_out: bool, summary: TranscriptSummary | None, arm: ProductLoopArm
) -> ProductLoopFailureReason | None:
    """Classify a cell that cannot be scored, in the pre-registered order."""
    if timed_out:
        return ProductLoopFailureReason.DEADLINE_EXCEEDED
    if summary is None:
        return ProductLoopFailureReason.TRANSCRIPT_UNPARSABLE
    if summary.rate_limited:
        return ProductLoopFailureReason.RATE_LIMITED
    if not summary.saw_result or summary.is_error:
        return ProductLoopFailureReason.AGENT_ERROR
    if summary.mcp_status != "connected" or summary.mcp_dropped:
        return ProductLoopFailureReason.MCP_DISCONNECTED
    if sorted(summary.tools_advertised) != sorted(arm.expected_tools):
        return ProductLoopFailureReason.TOOL_PARITY_VIOLATION
    return None


def _build_artifact(
    *,
    task: BenchmarkTask,
    arm: ProductLoopArm,
    repetition: int,
    summary: TranscriptSummary | None,
    reason: ProductLoopFailureReason | None,
    detail: str | None,
    repo_path: Path,
    setup_seconds: float,
    wall_seconds: float,
    hook_records: list[HookRecord],
    freshness_state: str | None,
    stale_index_event: bool,
    graft_cards_greppable: bool | None,
    mcp_command_original: str | None,
    mcp_command_used: str | None,
    provider_endpoint_overridden: bool,
) -> ProductLoopCellArtifact:
    expected_count = len(task.expected_files)
    common: dict[str, Any] = {
        "instruction_block_sha256": INSTRUCTION_BLOCK_SHA256,
        "task_id": task.task_id,
        "repo": task.repo,
        "commit": task.commit,
        "arm": arm,
        "repetition": repetition,
        "agent_name": AGENT_NAME,
        "agent_version": AGENT_VERSION,
        "model_requested": AGENT_MODEL,
        "billing_mode": BILLING_MODE,
        "mcp_server": arm.mcp_server,
        "expected_file_count": expected_count,
        "setup_seconds": setup_seconds,
        "wall_seconds": wall_seconds,
        "freshness_state": freshness_state,
        "stale_index_event": stale_index_event,
        "hook_records": hook_records,
        "hook_invocations": len(hook_records),
        "graft_cards_greppable": graft_cards_greppable,
        "mcp_command_original": mcp_command_original,
        "mcp_command_used": mcp_command_used,
        "provider_endpoint_overridden": provider_endpoint_overridden,
    }
    if summary is not None:
        common |= {
            "models_observed": summary.models_observed,
            "mcp_status": summary.mcp_status,
            "mcp_dropped": summary.mcp_dropped,
            "tools_advertised": sorted(summary.tools_advertised),
            "tool_calls": summary.tool_calls,
            "tool_call_mix": summary.tool_call_mix,
            "product_tool_calls": summary.product_tool_calls,
            "no_product_use": summary.product_tool_calls == 0,
            "input_tokens": summary.input_tokens,
            "output_tokens": summary.output_tokens,
            "cache_read_tokens": summary.cache_read_tokens,
            "cache_creation_tokens": summary.cache_creation_tokens,
            "modelled_cost_usd": summary.modelled_cost_usd,
            "num_turns": summary.num_turns,
        }

    if reason is not None:
        return ProductLoopCellArtifact.model_validate(
            common
            | {
                "status": ProductLoopCellStatus.FAILED,
                "failure_reason": reason,
                "failure_detail": detail,
                "matched_file_count": 0,
                "completeness": 0.0,
                "answer_flag": ProductLoopAnswerFlag.ANSWER_UNPARSED,
                "answer_paths": [],
                "answer_path_count": 0,
                "answer_precision": 0.0,
                "no_product_use": common.get("no_product_use", True),
            }
        )

    assert summary is not None  # noqa: S101 - _failure_reason returns None only with a summary
    paths, flag = extract_answer_paths(summary.final_text, repo_root=repo_path)
    matched = score_completeness(paths, task.expected_files) if paths else 0
    completeness = 0.0 if flag is not ProductLoopAnswerFlag.SCORED else matched / expected_count
    precision = (matched / len(paths)) if paths else 0.0
    return ProductLoopCellArtifact.model_validate(
        common
        | {
            "status": ProductLoopCellStatus.OK,
            "matched_file_count": 0 if flag is not ProductLoopAnswerFlag.SCORED else matched,
            "completeness": completeness,
            "answer_flag": flag,
            "answer_paths": paths,
            "answer_path_count": len(paths),
            "answer_precision": precision,
        }
    )


def main() -> int:
    payload = as_json_object(decode_json(sys.stdin.read()))
    if payload is None:
        message = "cell payload must be a JSON object"
        raise ProductLoopError(message)
    task = BenchmarkTask.model_validate(payload["task"])
    arm = ProductLoopArm(as_json_text(payload["arm"]))
    repetition = _as_int(payload["repetition"])
    cell_dir = Path(as_json_text(payload["cell_dir"]) or "")
    repo_path = Path(as_json_text(payload["repo_path"]) or "")
    binary = as_json_text(payload.get("agent_binary")) or AGENT_NAME
    path = as_json_text(payload.get("path")) or os.environ.get("PATH", "")

    prompt = build_prompt(task.question)
    command = _agent_command(arm=arm, prompt=prompt, repo_path=repo_path, binary=binary)
    env, endpoint_overridden = _cell_env(cell_dir=cell_dir, path=path, arm=arm)

    started = time.perf_counter()
    stdout, _returncode, timed_out = _run_agent(
        command, env=env, cwd=repo_path, deadline=CELL_DEADLINE_SECONDS
    )
    wall_seconds = time.perf_counter() - started
    (cell_dir / "transcript.jsonl").write_text(stdout, encoding="utf-8")

    summary: TranscriptSummary | None = None
    detail: str | None = None
    try:
        summary = summarize_transcript(stdout.splitlines(), arm=arm)
    except ProductLoopError as exc:
        detail = str(exc)

    reason = _failure_reason(timed_out=timed_out, summary=summary, arm=arm)
    if reason is not None and detail is None:
        if summary is not None and summary.error_text:
            detail = summary.error_text
        elif timed_out:
            detail = f"agent exceeded the {CELL_DEADLINE_SECONDS}s cell deadline"
        elif summary is not None:
            detail = (
                f"advertised={sorted(summary.tools_advertised)} "
                f"mcp_status={summary.mcp_status} dropped={summary.mcp_dropped}"
            )

    freshness_state, stale_index_event = _probe_freshness(
        [str(part) for part in as_json_array(payload.get("freshness_command")) or []],
        arm=arm,
        env=env,
        cwd=repo_path,
    )

    artifact = _build_artifact(
        task=task,
        arm=arm,
        repetition=repetition,
        summary=summary,
        reason=reason,
        detail=detail,
        repo_path=repo_path,
        setup_seconds=_as_float(payload.get("setup_seconds")),
        wall_seconds=wall_seconds,
        hook_records=_hook_records(cell_dir),
        freshness_state=freshness_state,
        stale_index_event=stale_index_event,
        graft_cards_greppable=(
            payload.get("graft_cards_greppable") is True
            if "graft_cards_greppable" in payload
            else None
        ),
        mcp_command_original=as_json_text(payload.get("mcp_command_original")),
        mcp_command_used=as_json_text(payload.get("mcp_command_used")),
        provider_endpoint_overridden=endpoint_overridden,
    )
    document = json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(
        sanitize_document(document, replacements=[(repo_path, "<repo>"), (cell_dir, "<workspace>")])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
