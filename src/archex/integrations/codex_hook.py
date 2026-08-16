"""Codex CLI PreToolUse hook: diagnostics-only fallback (M21).

Confirmation-spike findings (read against the `openai/codex` Rust source,
`codex-rs/hooks/` and `codex-rs/core/src/tools/`, not secondary docs):

- Codex's `PreToolUse` hook output schema DOES support content/context
  augmentation. `PreToolUseHookSpecificOutputWire` (`codex-rs/hooks/src/
  schema.rs`) has an `additional_context: Option<String>` field that
  serializes to the wire as `additionalContext` — the same field name
  Claude Code's own `PreToolUse` hook uses. The DEVELOPMENT_PLAN.md §2 GAP
  about augmentation-field support is resolved: augmentation IS supported.
- However, Codex has no Grep/Glob-equivalent tool-call event to scope that
  augmentation to. Codex's hookable `PreToolUse` tool names are exactly:
  `Bash` (every shell invocation — `HookToolName::bash()` in
  `codex-rs/core/src/tools/hook_names.rs`), `apply_patch` (file edits,
  matcher-aliased to `Write`/`Edit`), `spawn_agent`, and MCP tools under
  their own canonical names. There is no tool name that means "this is a
  search," the way Claude Code's `Grep`/`Glob` or oh-my-pi/Pi's
  `grep`/`glob`/`find` do — file search and reads both happen through the
  generic `Bash` tool by shelling out to `grep`/`rg`/`find`/`cat`/etc.
- Hooking `Bash` unconditionally to inject `additionalContext` would
  intercept every shell command Codex runs, including destructive ones
  (`rm`, `git push --force`, arbitrary scripts) — a materially broader and
  riskier surface than M19/M20's read-only Grep/Glob-only pattern, and it
  would not satisfy "the installed hook config matches the Grep/Glob-
  equivalent tool only" (there is no such tool to match). Per the plan's
  own contingency for this GAP and the "never claim augmentation that
  wasn't verified" constraint, this module ships the documented
  diagnostics-only fallback instead: it recognizes `Bash` invocations
  shaped like a search command and logs what archex could have surfaced
  (or why it couldn't), purely for operator observability. It never
  returns `additionalContext`, `permissionDecision`, `updatedInput`, or any
  other `hookSpecificOutput` field, and never mutates or blocks the tool
  call.

Every code path exits 0, mirroring `archex.integrations.hook`'s contract: a
missing/stale index, a timeout, a malformed payload, or an internal error
all degrade to a silent no-op from Codex's point of view, logged instead to
the same local diagnostics log `archex.integrations.hook` already uses
(`ARCHEX_HOOK_DIAGNOSTICS_LOG`, default `~/.archex/hook-diagnostics.log`).

This module is a thin per-client shim: it translates Codex's `PreToolUse`
payload into a query and calls straight into `archex.integrations.hook`'s
lookup/timeout/freshness engine (`lookup_with_timeout`, `log_diagnostic`,
`IDENTIFIER_TOKEN_RE`) in-process — no lookup, ranking, or timeout logic is
reimplemented here, and no second subprocess is spawned.

Invoked as a subprocess: `python -m archex.integrations.codex_hook`, reading
Codex's `PreToolUse` JSON payload from stdin and always writing `{}` to
stdout (Codex's hook output schema defaults every field, so an empty object
means "no decision" — `continue: true`, no `hookSpecificOutput`).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, cast

from archex.integrations.hook import IDENTIFIER_TOKEN_RE, log_diagnostic, lookup_with_timeout

#: Codex's canonical `PreToolUse` tool name for every shell invocation
#: (`HookToolName::bash()` in `codex-rs/core/src/tools/hook_names.rs`).
#: Codex has no distinct Grep/Glob-equivalent tool name — this is the only
#: `PreToolUse` event that can ever carry a search-shaped command.
DIAGNOSED_TOOL_NAME = "Bash"

#: `matcher` regex the installer (`archex.client_setup`) writes into
#: `config.toml`'s `[[hooks.PreToolUse]]` table, kept alongside this
#: module's own runtime filter so the installed config and the runtime
#: dispatch can never drift apart. Codex matches `matcher` as a regex
#: against the tool name (confirmed via `codex-rs/config/src/
#: config_requirements.rs`'s own `matcher = "^Bash$"` fixture).
HOOK_MATCHER = f"^{DIAGNOSED_TOOL_NAME}$"

#: Leading-token detector for shell commands shaped like a search
#: invocation (optionally after `sudo`, and optionally following a
#: `;`/`&`/`|` separator so `cd x && grep ...` and `cat f | grep ...` both
#: match). Detection only — this never builds a query that gets surfaced to
#: the agent, only a diagnostics log line.
_SEARCH_COMMAND_RE = re.compile(r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:git\s+grep|rg|ag|grep|find|fd)\b")

#: Tokens from the recognized command name/verbs/flags themselves that would
#: otherwise pollute the extracted query (every matched command contains one
#: of these, so leaving them in would dilute or dominate the BM25 lookup).
_COMMAND_NOISE_TOKENS = frozenset({"sudo", "git", "grep", "rg", "ag", "find", "fd", "name"})


def main() -> None:
    """Entry point for `python -m archex.integrations.codex_hook`.

    Guarantees exit 0 and valid JSON stdout on every path, including an
    exception this module did not anticipate — see the module docstring's
    non-blocking, diagnostics-only contract.
    """
    try:
        _main_impl()
    except BaseException as exc:  # noqa: BLE001 - the non-blocking contract requires this
        log_diagnostic("codex_unhandled_exception", detail=repr(exc))
    sys.stdout.write("{}")
    sys.stdout.flush()
    os._exit(0)


def _main_impl() -> None:
    raw = sys.stdin.read()
    payload = _parse_codex_payload(raw)
    if payload is None:
        return
    handle_codex_pre_tool_use(payload)


def _parse_codex_payload(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_diagnostic("codex_malformed_payload", detail=f"invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        log_diagnostic("codex_malformed_payload", detail="payload is not a JSON object")
        return None
    return cast("dict[str, Any]", payload)


def handle_codex_pre_tool_use(payload: dict[str, Any]) -> None:
    """Diagnostics-only core handler: never returns or applies augmentation.

    Detects `Bash` invocations shaped like a search command and logs what
    archex would have surfaced (or why it couldn't) purely for operator
    observability. Never mutates or blocks the tool call, and never raises —
    any failure degrades to a diagnostics log line.
    """
    try:
        if payload.get("tool_name") != DIAGNOSED_TOOL_NAME:
            return
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return
        command = cast("dict[str, Any]", tool_input).get("command")
        if not isinstance(command, str) or not _SEARCH_COMMAND_RE.search(command):
            return
        query = " ".join(
            token
            for token in IDENTIFIER_TOKEN_RE.findall(command)
            if token.lower() not in _COMMAND_NOISE_TOKENS
        )
        if not query:
            return
        cwd_raw = payload.get("cwd")
        cwd = cwd_raw if isinstance(cwd_raw, str) and cwd_raw else os.getcwd()
        context = lookup_with_timeout(cwd, query)
        if context is None:
            # A missing/stale index, timeout, or lookup error already logged
            # its own diagnostic inside `lookup_with_timeout`/`_lookup`.
            return
        log_diagnostic(
            "codex_augmentation_withheld",
            detail=(
                f"Bash command looked like a search ({command!r}); archex has "
                "matches but Codex has no Grep/Glob-equivalent tool-call event "
                "to scope augmentation to safely — see "
                "docs/CLIENT_COMPATIBILITY_MATRIX.md"
            ),
            cwd=cwd,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to no-op, never raise
        log_diagnostic("codex_internal_error", detail=repr(exc))


if __name__ == "__main__":
    main()
