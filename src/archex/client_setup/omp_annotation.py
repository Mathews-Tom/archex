"""Shared omp/Pi annotation module template."""

from __future__ import annotations

TS_HOOK_MODULE_TEMPLATE = r"""/**
 * archex grep-result annotation module (oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --hooks` / `archex install-client
 * pi --hooks` (opt-in; never installed by default). On every `tool_result`
 * for a search tool it sends the tool name, the tool's input, and the
 * tool's own result text to `archex annotate`, and appends the returned
 * lines: one fact line per indexed code unit the search hit. Parsing,
 * resolution, freshness, and caps all live in `archex annotate`; this module
 * only gathers inputs and appends output.
 *
 * Contract:
 * - Augment, never replace. The returned `content` is a deep copy of the
 *   host's own content array with exactly one text block appended; every
 *   original block, including fields this module does not know about, is
 *   carried over unchanged. `details` and `isError` are never returned.
 * - Search tools only: `grep`, `glob` (oh-my-pi), `find` (Pi), and `bash`.
 *   `archex annotate` decides whether a bash command is a search (`rg`,
 *   `grep`, `ugrep`, `git grep`, or a path lister such as `find`, `fd`, or
 *   `git ls-files`); a bash command that mentions none of them is not sent.
 *   `read` and every other tool are never touched.
 * - Fail open. A stale or dirty index, an unrecognised format, a spawn
 *   failure, a timeout, or any error returns `undefined`, leaving the
 *   original result untouched; faults go to the diagnostics log
 *   (`ARCHEX_HOOK_DIAGNOSTICS_LOG` or `~/.archex/hook-diagnostics.log`).
 * - Bounded. The subprocess runs under the hook budget
 *   (`ARCHEX_HOOK_TIMEOUT_SECONDS`, default 0.5s) and is killed past it.
 * - Ledger. One JSONL line per search-tool result
 *   (`ARCHEX_ANNOTATION_LEDGER` or `~/.archex/annotation-ledger.jsonl`):
 *   whether it was eligible and annotated, the units and tokens added, the
 *   index freshness, and the reason when nothing was added. The transcript
 *   alone cannot show whether an annotation was applied; this can.
 */

import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client <client> --hooks`) ---

/** Python interpreter active when `--hooks` ran, so the module always runs
 * archex from the environment it was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
/** The host-neutral hook entry: it decides whether a call is a search before
 * importing the index or the tokenizer, and skips the full CLI's imports. */
const ARCHEX_ANNOTATE_ARGS = ["-m", "archex.integrations.annotate_hook"];

/** Matches `DEFAULT_HOOK_TIMEOUT_SECONDS` in `archex.integrations.diagnostics`. */
const ARCHEX_DEFAULT_TIMEOUT_MS = 500;
const ARCHEX_TIMEOUT_ENV_VAR = "ARCHEX_HOOK_TIMEOUT_SECONDS";
const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";
const ARCHEX_LEDGER_ENV_VAR = "ARCHEX_ANNOTATION_LEDGER";

/** Host tool name -> `archex annotate --tool` value. oh-my-pi's glob tool is
 * `glob`; Pi's glob-equivalent is `find`. Only names that exist on the
 * running host ever fire. */
const ANNOTATED_TOOLS: Readonly<Record<string, string>> = {
  grep: "grep",
  glob: "glob",
  find: "find",
  bash: "bash",
};

/** Spawn guard only: every command `archex annotate` accepts as a search
 * names one of these programs, so a command naming none is never sent. */
const BASH_SEARCH_MENTION = /rg|grep|find|bfs|fd|ls-files/;

// --- Minimal structural types for the `tool_result` contract ---
//
// Declared locally (never imported from either host package) so this module
// has zero import-resolution dependency on which host loaded it.

interface ToolResultEventLike {
  toolName: string;
  toolCallId?: string;
  input?: Record<string, unknown>;
  content?: unknown;
  details?: unknown;
  isError?: boolean;
}

interface ToolResultPatch {
  content?: unknown[];
}

type ToolResultHandler = (
  event: ToolResultEventLike,
  ctx: unknown,
) => Promise<ToolResultPatch | undefined>;

interface HookHost {
  on(event: "tool_result", handler: ToolResultHandler): unknown;
}

interface AnnotateRecord {
  annotation: string;
  eligible: boolean;
  annotated: boolean;
  units_hit: number;
  tokens: number;
  freshness: string;
  reason: string | null;
}

// --- Diagnostics and ledger ---

function envPath(name: string, fallback: string): string {
  const override = process.env[name];
  return override && override.trim().length > 0 ? override : fallback;
}

function appendJsonLine(path: string, entry: Record<string, unknown>): void {
  mkdirSync(dirname(path), { recursive: true });
  appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
}

function logDiagnostic(kind: string, detail: string, cwd?: string): void {
  try {
    const entry: Record<string, string> = {
      timestamp: new Date().toISOString(),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendJsonLine(
      envPath(ARCHEX_DIAGNOSTICS_LOG_ENV_VAR, join(homedir(), ".archex", "hook-diagnostics.log")),
      entry,
    );
  } catch {
    // Diagnostics logging must never raise into the hook's return path.
  }
}

function recordLedger(
  event: ToolResultEventLike,
  outcome: {
    eligible: boolean;
    annotated: boolean;
    units: number;
    tokens: number;
    freshness: string;
    reason: string | null;
    latencyMs: number;
  },
): void {
  try {
    appendJsonLine(
      envPath(ARCHEX_LEDGER_ENV_VAR, join(homedir(), ".archex", "annotation-ledger.jsonl")),
      {
        timestamp: new Date().toISOString(),
        host: "omp",
        toolCallId: typeof event.toolCallId === "string" ? event.toolCallId : null,
        tool: event.toolName,
        eligible: outcome.eligible,
        annotated: outcome.annotated,
        units: outcome.units,
        tokens: outcome.tokens,
        freshness: outcome.freshness,
        reason: outcome.reason,
        latency_ms: Math.round(outcome.latencyMs),
      },
    );
  } catch (err) {
    logDiagnostic("ts_ledger_error", String(err));
  }
}

// --- Subprocess call: `python -m archex.integrations.annotate_hook` ---

function timeoutMs(): number {
  const seconds = Number(process.env[ARCHEX_TIMEOUT_ENV_VAR]);
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : ARCHEX_DEFAULT_TIMEOUT_MS;
}

type SubprocessOutcome = { stdout: string } | { failure: string };

function runAnnotate(request: Record<string, unknown>, cwd: string): Promise<SubprocessOutcome> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: SubprocessOutcome): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(ARCHEX_PYTHON_COMMAND, ARCHEX_ANNOTATE_ARGS, {
        cwd,
        stdio: ["pipe", "pipe", "ignore"],
      });
    } catch (err) {
      logDiagnostic("ts_spawn_error", String(err), cwd);
      finish({ failure: "spawn_error" });
      return;
    }

    const budget = timeoutMs();
    const timer = setTimeout(() => {
      logDiagnostic("ts_timeout", `annotate exceeded ${budget}ms`, cwd);
      try {
        child.kill("SIGKILL");
      } catch {
        // Already exited.
      }
      finish({ failure: "timeout" });
    }, budget);

    let stdout = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      logDiagnostic("ts_spawn_error", String(err), cwd);
      finish({ failure: "spawn_error" });
    });
    child.on("close", () => {
      clearTimeout(timer);
      finish(stdout.length > 0 ? { stdout } : { failure: "no_output" });
    });

    try {
      child.stdin?.write(JSON.stringify(request));
      child.stdin?.end();
    } catch (err) {
      clearTimeout(timer);
      logDiagnostic("ts_stdin_error", String(err), cwd);
      finish({ failure: "stdin_error" });
    }
  });
}

function parseRecord(stdout: string): AnnotateRecord | null {
  try {
    const parsed: unknown = JSON.parse(stdout);
    if (typeof parsed !== "object" || parsed === null) return null;
    const record = parsed as Record<string, unknown>;
    if (typeof record.annotation !== "string" || typeof record.annotated !== "boolean") {
      return null;
    }
    return {
      annotation: record.annotation,
      eligible: record.eligible === true,
      annotated: record.annotated,
      units_hit: typeof record.units_hit === "number" ? record.units_hit : 0,
      tokens: typeof record.tokens === "number" ? record.tokens : 0,
      freshness: typeof record.freshness === "string" ? record.freshness : "unknown",
      reason: typeof record.reason === "string" ? record.reason : null,
    };
  } catch {
    return null;
  }
}

function resultText(content: unknown[]): string | null {
  const texts: string[] = [];
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    const candidate = block as Record<string, unknown>;
    if (candidate.type === "text" && typeof candidate.text === "string") {
      texts.push(candidate.text);
    }
  }
  return texts.length > 0 ? texts.join("\n") : null;
}

// --- Extension entry point ---

export default function archexHook(pi: HookHost): void {
  pi.on("tool_result", async (event) => {
    const tool = ANNOTATED_TOOLS[event.toolName];
    if (tool === undefined) return undefined; // never touches `read` or any other tool
    const started = Date.now();
    const outcome = {
      eligible: true,
      annotated: false,
      units: 0,
      tokens: 0,
      freshness: "unchecked",
      reason: null as string | null,
      latencyMs: 0,
    };
    const decline = (reason: string): undefined => {
      outcome.reason = reason;
      outcome.latencyMs = Date.now() - started;
      recordLedger(event, outcome);
      return undefined;
    };
    try {
      const input = event.input;
      if (typeof input !== "object" || input === null || Array.isArray(input)) {
        return decline("malformed_input");
      }
      if (tool === "bash") {
        const command = input.command;
        if (typeof command !== "string" || !BASH_SEARCH_MENTION.test(command)) {
          outcome.eligible = false;
          return decline("not_search_command");
        }
      }
      if (event.isError === true) return decline("tool_error");
      if (!Array.isArray(event.content)) return decline("malformed_content");
      const text = resultText(event.content);
      if (text === null) return decline("no_text_content");

      const cwd = process.cwd();
      const response = await runAnnotate({ host: "omp", tool, input, text, cwd }, cwd);
      if ("failure" in response) return decline(response.failure);
      const record = parseRecord(response.stdout);
      if (record === null) {
        logDiagnostic("ts_malformed_response", response.stdout.slice(0, 200), cwd);
        return decline("malformed_response");
      }
      outcome.eligible = record.eligible;
      outcome.units = record.units_hit;
      outcome.freshness = record.freshness;
      if (!record.annotated || record.annotation.length === 0) {
        return decline(record.reason ?? "not_annotated");
      }

      const content = structuredClone(event.content);
      content.push({ type: "text", text: `\n\n${record.annotation}` });
      outcome.annotated = true;
      outcome.tokens = record.tokens;
      outcome.latencyMs = Date.now() - started;
      recordLedger(event, outcome);
      return { content };
    } catch (err) {
      logDiagnostic("ts_internal_error", String(err));
      return decline("internal_error");
    }
  });
}
"""
