"""OpenCode post-edit plugin template."""

from __future__ import annotations

OPENCODE_POST_EDIT_MODULE_TEMPLATE = r"""/**
 * archex OpenCode post-edit `tool.execute.after` plugin (R21).
 *
 * Installed by `archex install-client opencode --post-edit-hooks` (opt-in).
 * A separate plugin file from the search hook's `archex-hook.ts`, so the two
 * surfaces install and remove independently. OpenCode auto-loads plugin
 * files from `.opencode/plugins/` and `~/.config/opencode/plugins/`, so no
 * `opencode.json` entry is written or needed.
 *
 * Upstream contract (verified against `packages/plugin/src/index.ts`):
 * - `"tool.execute.after"` receives `input { tool, sessionID, callID, args }`
 *   and a mutable `output { title, output, metadata }`. It runs after the
 *   tool executed and has no documented way to block or fail it.
 * - The hook exposes no error flag, so a failed edit is not distinguishable
 *   here; the Python side revalidates every recorded path against the real
 *   working tree before anything is emitted, so a phantom path simply
 *   produces no output.
 * - Text reaches the agent only by appending to `output.output`; there is no
 *   additional-context field on this event.
 * - `packages/opencode/src/tool/edit.ts` and `.../write.ts` define the
 *   built-in `edit` and `write` tools, both parameterised by `filePath`.
 *
 * Every path resolves without throwing and leaves `output` untouched when
 * archex has nothing fresh to add.
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

// --- Minimal structural types for the plugin contract ---

interface ToolExecuteAfterInput {
  tool: string;
  sessionID?: string;
  callID?: string;
  args?: Record<string, unknown>;
}

interface ToolExecuteAfterOutput {
  title?: string;
  output?: string;
  metadata?: unknown;
}

type Plugin = (context: { directory?: string }) => Promise<{
  "tool.execute.after"?: (
    input: ToolExecuteAfterInput,
    output: ToolExecuteAfterOutput,
  ) => Promise<void>;
}>;

// --- Plugin entry point ---

export const ArchexPostEditPlugin: Plugin = async ({ directory }) => {
  return {
    "tool.execute.after": async (input, output) => {
      try {
        const claudeToolName = ARCHEX_EDIT_TOOLS[input.tool];
        if (!claudeToolName) return; // never touches read/grep/glob/bash

        const filePath = firstPath(input.args);
        if (filePath === null) return;

        const cwd = directory ?? process.cwd();
        const rawStdout = await runArchexPostEditSubprocess(
          {
            tool_name: claudeToolName,
            tool_input: { file_path: filePath },
            cwd,
            archex_client: ARCHEX_CLIENT,
          },
          cwd,
        );
        if (rawStdout === null) return;

        const context = extractAdditionalContext(rawStdout);
        if (context === null) return;

        output.output = `${output.output ?? ""}\n\n${context}`;
      } catch (err) {
        logDiagnostic("ts_post_edit_internal_error", String(err));
      }
    },
  };
};

export default ArchexPostEditPlugin;
"""
