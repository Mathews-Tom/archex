"""Cursor `beforeSubmitPrompt` hook: diagnostics-only prompt-level lookup (M23).

Confirmation-spike findings (read directly against Cursor's own official
docs — `https://cursor.com/docs/hooks` and `https://cursor.com/docs/
reference/third-party-hooks`, fetched 2026-07-06 — not secondary sources):

- Cursor has no Grep/Glob-equivalent tool-call hook at all: `preToolUse`/
  `postToolUse` fire generically for every tool, with no per-tool
  augmentation scoping analogous to Claude Code's `Grep`/`Glob` matcher,
  oh-my-pi/Pi's `grep`/`glob`/`find` `tool_result` dispatch, or OpenCode's
  native-tool `tool.execute.after`. `beforeSubmitPrompt` is the closest
  content-adjacent hook that fires on ordinary agent use, matching
  `DEVELOPMENT_PLAN.md` §2's own framing of it as the necessarily weaker,
  prompt-level mechanism for this client.
- `beforeSubmitPrompt`'s own output schema, however, is `{"continue": bool,
  "user_message": str | None}` ONLY — there is no context-injection output
  field, nested or flat. This differs from `sessionStart` (which *does*
  support an `additional_context` output field, but fires once per
  conversation, not per prompt) and from `postToolUse` (which also supports
  `additional_context`). Cursor's own "Response Format Compatibility"
  section (`third-party-hooks.md`) documents a Claude-Code-style nested
  `hookSpecificOutput` translation only for `PreToolUse` and
  `Stop`/`SubagentStop` — none is documented for `UserPromptSubmit` (which
  Cursor maps to `beforeSubmitPrompt`), and no `additionalContext`
  passthrough exists for it despite Claude Code's own `UserPromptSubmit`
  hook supporting that field. `user_message` is documented as shown only
  when a submission is blocked (`continue: false`), which is out of scope
  for this milestone (deny/blocking behavior is not part of M23) — so it
  cannot carry injected context on the normal, non-blocking path either.
- The milestone's own objective describes "prompt-level context injection"
  as the mechanism; the finding above means that, as specified, Cursor's
  `beforeSubmitPrompt` cannot deliver it today. Per the same discipline M21
  applied when Codex's hook schema turned out to have no Grep/Glob-
  equivalent tool-call event to scope augmentation to, this module ships
  the plan's own accepted fallback instead: a diagnostics-only hook that
  performs the same lookup and logs what it would have injected, but never
  returns it to Cursor and never blocks prompt submission. Every invocation
  returns exactly `{"continue": true}`.

Every code path exits 0, mirroring `archex.integrations.hook`'s and
`archex.integrations.codex_hook`'s contract: a missing/stale index, a
timeout, a malformed payload, or an internal error all degrade to a
no-injection, non-blocking no-op from Cursor's point of view, logged instead
to the same local diagnostics log (`ARCHEX_HOOK_DIAGNOSTICS_LOG`, default
`~/.archex/hook-diagnostics.log`).

This module is a thin per-client shim, per the Section G architecture note
in `.docs/DEVELOPMENT_PLAN.md` §2: unlike the Claude Code/omp/Pi/OpenCode
hooks, which translate a *tool-call* payload into a Grep/Glob query, this
translates a *prompt-submission* payload (free natural-language text, not a
search pattern) into a query by extracting the single longest
identifier-like token from the prompt (see `_extract_query`), then calls
straight into `archex.integrations.hook`'s lookup/timeout/freshness engine
(`lookup_with_timeout`, `log_diagnostic`, `IDENTIFIER_TOKEN_RE`)
in-process — no lookup, ranking, or timeout logic is reimplemented here.

Invoked as a subprocess: `python -m archex.integrations.cursor_hook`,
reading Cursor's `beforeSubmitPrompt` JSON payload (`{"prompt": str,
"attachments": [...]}`, no `cwd` field — Cursor runs project hooks from the
project root per its own docs, so `os.getcwd()` is used directly) from
stdin and always writing `{"continue": true}` to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, cast

from archex.integrations.hook import IDENTIFIER_TOKEN_RE, log_diagnostic, lookup_with_timeout

#: Always-continue output. `beforeSubmitPrompt` has no context-injection
#: field to populate (see module docstring) and deny/blocking behavior is
#: out of scope for this milestone, so every invocation returns exactly this.
_ALWAYS_CONTINUE: dict[str, object] = {"continue": True}


def _extract_query(prompt: str) -> str:
    """Pick the single strongest search candidate out of free prompt text.

    `IndexStore.search_symbols` wraps its entire query in one quoted FTS5
    phrase (see `src/archex/index/store.py`), so a multi-word query only
    matches when that exact word sequence appears verbatim in the indexed
    content — true for a single Grep/Glob pattern fragment, never true for a
    natural-language sentence. The longest identifier-like token in the
    prompt is used as a heuristic stand-in for "the symbol the user is
    probably asking about": real identifiers mentioned in prose are
    typically longer than the surrounding English words.
    """
    tokens = IDENTIFIER_TOKEN_RE.findall(prompt)
    return max(tokens, key=len) if tokens else ""


def main() -> None:
    """Entry point for `python -m archex.integrations.cursor_hook`.

    Guarantees exit 0 and `{"continue": true}` stdout on every path,
    including an exception this module did not anticipate — see the module
    docstring's non-blocking, diagnostics-only contract.
    """
    try:
        _main_impl()
    except BaseException as exc:  # noqa: BLE001 - the non-blocking contract requires this
        log_diagnostic("cursor_unhandled_exception", detail=repr(exc))
    sys.stdout.write(json.dumps(_ALWAYS_CONTINUE))
    sys.stdout.flush()
    os._exit(0)


def _main_impl() -> None:
    raw = sys.stdin.read()
    payload = _parse_cursor_payload(raw)
    if payload is None:
        return
    handle_before_submit_prompt(payload)


def _parse_cursor_payload(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_diagnostic("cursor_malformed_payload", detail=f"invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        log_diagnostic("cursor_malformed_payload", detail="payload is not a JSON object")
        return None
    return cast("dict[str, Any]", payload)


def handle_before_submit_prompt(payload: dict[str, Any]) -> None:
    """Diagnostics-only core handler: never returns or applies context injection.

    Extracts the single strongest identifier-like token from the submitted
    prompt text (see `_extract_query`), looks it up through the shared
    engine, and logs what it would have injected (see module docstring for
    why the output field to carry that doesn't exist on this hook). Never
    mutates or blocks the submission, and never raises — any failure
    degrades to a diagnostics log line.
    """
    try:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return
        query = _extract_query(prompt)
        if not query:
            return
        cwd = os.getcwd()
        context = lookup_with_timeout(cwd, query)
        if context is None:
            # A missing/stale index, timeout, or lookup error already logged
            # its own diagnostic inside `lookup_with_timeout`/`_lookup`.
            return
        log_diagnostic(
            "cursor_context_injection_unsupported",
            detail=(
                "prompt lookup found archex matches, but Cursor's "
                "beforeSubmitPrompt hook has no context-injection output "
                "field to carry them (see docs/CLIENT_COMPATIBILITY_MATRIX.md); "
                f"withheld: {context}"
            ),
            cwd=cwd,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to no-op, never raise
        log_diagnostic("cursor_internal_error", detail=repr(exc))


if __name__ == "__main__":
    main()
