"""Shared omp/Pi post-edit module template."""

from __future__ import annotations

TS_POST_EDIT_MODULE_TEMPLATE = r"""/**
 * archex shared post-edit `tool_result` module (R21 — oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --post-edit-hooks` /
 * `archex install-client pi --post-edit-hooks` (opt-in; never installed by
 * default). A separate file from the search hook's `archex-hook.ts`, so the
 * two surfaces install and remove independently.
 *
 * Upstream contract (verified against each host's own source, not docs):
 * - Both hosts expose `pi.on("tool_result", handler)`, fired after a tool
 *   executes, with `{ toolName, input, content, details, isError }` and a
 *   partial-patch return. `isError` is the explicit success signal — this
 *   module records an edit only when it is not `true`.
 * - `tool_result` fires after execution and cannot block or fail the edit;
 *   the blocking event is the separate pre-execution `tool_call`.
 * - Both hosts' `edit` and `write` schemas carry the target in `path`
 *   (oh-my-pi `dist/types/edit/schemas.d.ts` and `dist/types/tools/write.d.ts`;
 *   Pi `dist/core/tools/edit.d.ts` and `.../write.d.ts`).
 *
 * Every path resolves without throwing. A spawn failure, a timeout, a stale
 * index, or a malformed response all degrade to returning `undefined` (no
 * content patch), logged to the same diagnostics file the Python hooks use.
 */
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client <client> --post-edit-hooks`) ---

/** Python interpreter active when the installer ran, so this always runs in
 * the environment archex was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
const ARCHEX_PYTHON_ARGS = ["-m", "archex.integrations.post_edit_hook"];

/** Wall-clock kill timer for the subprocess. Deliberately larger than the
 * search hook's 500ms — a post-edit cycle reparses changed files — and
 * larger than the Python side's own DEFAULT_POST_EDIT_TIMEOUT_SECONDS so
 * the Python deadline, which logs a diagnostic, normally fires first. */
const ARCHEX_POST_EDIT_TIMEOUT_MS = 15000;

const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";

/** Native edit tool name -> the Claude Code tool name the Python subprocess
 * expects. Verified against each host's own tool definitions, not docs. */
const ARCHEX_EDIT_TOOLS: Readonly<Record<string, "Edit" | "Write">> = {
  edit: "Edit",
  write: "Write",
};

/** Input fields that can hold the edited path, most-specific first. Both
 * oh-my-pi's and Pi's `edit`/`write` schemas name it `path`; OpenCode's name
 * it `filePath`. The others are accepted defensively — the Python side
 * rejects anything that is not a real path inside the repository, so a wrong
 * guess degrades to no output rather than a wrong claim. */
const ARCHEX_PATH_FIELDS = ["filePath", "path", "file_path", "notebook_path"] as const;

/** Declared to the Python subprocess as `archex_client` so the emitted
 * receipt names the host that actually ran the edit. The subprocess
 * allowlists this value; an unknown one falls back to `claude-code`. */
const ARCHEX_CLIENT = "__ARCHEX_CLIENT__";

function firstPath(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null;
  for (const field of ARCHEX_PATH_FIELDS) {
    const value = args[field];
    if (typeof value === "string" && value.trim().length > 0) return value;
  }
  return null;
}

// --- Diagnostics (parity with hook.py's `log_diagnostic`) ---

function diagnosticsLogPath(): string {
  const override = process.env[ARCHEX_DIAGNOSTICS_LOG_ENV_VAR];
  if (override && override.trim().length > 0) return override;
  return join(homedir(), ".archex", "hook-diagnostics.log");
}

function logDiagnostic(kind: string, detail: string, cwd?: string): void {
  try {
    const path = diagnosticsLogPath();
    mkdirSync(dirname(path), { recursive: true });
    const entry: Record<string, string> = {
      timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
  } catch {
    // Diagnostics logging must never raise into the host's edit path.
  }
}

// --- Subprocess call: `python -m archex.integrations.post_edit_hook` ---

function runArchexPostEditSubprocess(
  payload: Record<string, unknown>,
  cwd: string,
): Promise<string | null> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: string | null): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(ARCHEX_PYTHON_COMMAND, ARCHEX_PYTHON_ARGS, {
        cwd,
        stdio: ["pipe", "pipe", "ignore"],
      });
    } catch (err) {
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
      return;
    }

    const timer = setTimeout(() => {
      logDiagnostic(
        "ts_post_edit_timeout",
        `post-edit cycle exceeded ${ARCHEX_POST_EDIT_TIMEOUT_MS}ms`,
        cwd,
      );
      try {
        child.kill("SIGKILL");
      } catch {
        // Already exited.
      }
      finish(null);
    }, ARCHEX_POST_EDIT_TIMEOUT_MS);

    let stdout = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
    });
    child.on("close", () => {
      clearTimeout(timer);
      finish(stdout.length > 0 ? stdout : null);
    });

    try {
      child.stdin?.write(JSON.stringify(payload));
      child.stdin?.end();
    } catch (err) {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_stdin_error", String(err), cwd);
      finish(null);
    }
  });
}

function extractAdditionalContext(rawStdout: string): string | null {
  try {
    const parsed: unknown = JSON.parse(rawStdout);
    if (typeof parsed !== "object" || parsed === null) return null;
    const hookSpecificOutput = (parsed as Record<string, unknown>).hookSpecificOutput;
    if (typeof hookSpecificOutput !== "object" || hookSpecificOutput === null) return null;
    const context = (hookSpecificOutput as Record<string, unknown>).additionalContext;
    return typeof context === "string" && context.length > 0 ? context : null;
  } catch {
    return null;
  }
}

// --- Minimal structural types for the `tool_result` contract ---
//
// Declared locally (never imported from either host package) so this module
// has zero import-resolution dependency on which host loaded it.

interface ToolResultEventLike {
  toolName: string;
  input?: Record<string, unknown>;
  content?: unknown[];
  details?: unknown;
  isError?: boolean;
}

interface ToolResultPatch {
  content?: unknown[];
  details?: unknown;
  isError?: boolean;
}

type ToolResultHandler = (
  event: ToolResultEventLike,
  ctx: unknown,
) => Promise<ToolResultPatch | undefined>;

interface HookHost {
  on(event: "tool_result", handler: ToolResultHandler): unknown;
}

// --- Extension entry point ---

export default function archexPostEditHook(pi: HookHost): void {
  pi.on("tool_result", async (event) => {
    try {
      const claudeToolName = ARCHEX_EDIT_TOOLS[event.toolName];
      if (!claudeToolName) return undefined; // never touches read/grep/glob/bash
      if (event.isError === true) return undefined; // only successful edits count

      const filePath = firstPath(event.input);
      if (filePath === null) return undefined;

      const cwd = process.cwd();
      const rawStdout = await runArchexPostEditSubprocess(
        {
          tool_name: claudeToolName,
          tool_input: { file_path: filePath },
          cwd,
          archex_client: ARCHEX_CLIENT,
        },
        cwd,
      );
      if (rawStdout === null) return undefined;

      const context = extractAdditionalContext(rawStdout);
      if (context === null) return undefined;

      const existingContent = Array.isArray(event.content) ? event.content : [];
      return {
        content: [...existingContent, { type: "text", text: `\n\n${context}` }],
      };
    } catch (err) {
      logDiagnostic("ts_post_edit_internal_error", String(err));
      return undefined;
    }
  });
}
"""
