"""OpenCode annotation plugin template."""

from __future__ import annotations

OPENCODE_HOOK_MODULE_TEMPLATE = r"""/**
 * archex grep-result annotation plugin (OpenCode).
 *
 * Installed by `archex install-client opencode --hooks` (opt-in; never
 * installed by default) as a standalone plugin file OpenCode auto-loads from
 * its native plugin directory -- `.opencode/plugins/archex-hook.ts`
 * (project-local) or `~/.config/opencode/plugins/archex-hook.ts` (global).
 * No `opencode.json` entry is required.
 *
 * On every `tool.execute.after` for a search tool it sends the tool name, the
 * tool's arguments, and the tool's own output text to
 * `python -m archex.integrations.annotate_hook` and appends the returned
 * lines: one fact line per indexed code unit the search hit. Parsing,
 * resolution, freshness, and caps all live in Python; this module only
 * gathers inputs and appends output.
 *
 * Contract:
 * - Augment, never replace. `output.output` is only ever reassigned to
 *   `original + "\n\n" + annotation`, so the host's own text stays a
 *   byte-for-byte prefix. `output.title`, `output.metadata`, and every field
 *   this module does not know about are never touched.
 * - Search tools only: OpenCode's native `grep`, `glob`, and `bash`. `archex
 *   annotate` decides whether a bash command is a search (`rg`, `grep`,
 *   `ugrep`, `git grep`, or a path lister such as `find`, `fd`, or `git
 *   ls-files`); a bash command that mentions none of them is not sent.
 *   `read` and every other tool are never touched. MCP-routed tool calls
 *   are excluded on purpose: OpenCode registers every MCP tool under a
 *   `{server}_{tool}` id, so none can collide with these names, and for MCP
 *   `tool.execute.after` receives the raw MCP result (a `{content}` shape)
 *   that OpenCode rebuilds into the model-visible text after the hook runs,
 *   discarding any `output.output` mutation (`session/prompt.ts`, v1.14.33).
 * - Fail open. A stale or dirty index, an unrecognised format, a spawn
 *   failure, a timeout, or any error leaves `output` untouched; faults go to
 *   the diagnostics log (`ARCHEX_HOOK_DIAGNOSTICS_LOG` or
 *   `~/.archex/hook-diagnostics.log`), never to the agent flow.
 * - Bounded. The subprocess runs under the hook budget
 *   (`ARCHEX_HOOK_TIMEOUT_SECONDS`, default 0.5s) and is killed past it.
 * - Ledger. One JSONL line per search-tool call
 *   (`ARCHEX_ANNOTATION_LEDGER` or `~/.archex/annotation-ledger.jsonl`), with
 *   OpenCode's `callID` as `toolCallId`: whether the call was eligible and
 *   annotated, the units and tokens added, the index freshness, and the
 *   reason when nothing was added.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client opencode --hooks`) ---

/** Python interpreter active when `--hooks` ran, so the plugin always runs
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

/** OpenCode tool id -> the tool name `archex.annotate` expects for host
 * `opencode`. This table is the plugin's only tool dispatch. */
const ANNOTATED_TOOLS: Readonly<Record<string, string>> = {
  grep: "grep",
  glob: "glob",
  bash: "bash",
};

/** Spawn guard only: every command `archex annotate` accepts as a search
 * names one of these programs, so a command naming none is never sent. */
const BASH_SEARCH_MENTION = /rg|grep|find|bfs|fd|ls-files/;

// --- Minimal structural types for `tool.execute.after` ---
//
// The hook's public types omit fields OpenCode adds at runtime, so only the
// members this module reads are declared, and defensively.

interface ToolInputLike {
  tool?: unknown;
  callID?: unknown;
  args?: unknown;
}

interface ToolOutputLike {
  output?: unknown;
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

interface Outcome {
  eligible: boolean;
  annotated: boolean;
  units: number;
  tokens: number;
  freshness: string;
  reason: string | null;
  latencyMs: number;
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

function recordLedger(input: ToolInputLike, outcome: Outcome): void {
  try {
    appendJsonLine(
      envPath(ARCHEX_LEDGER_ENV_VAR, join(homedir(), ".archex", "annotation-ledger.jsonl")),
      {
        timestamp: new Date().toISOString(),
        host: "opencode",
        toolCallId: typeof input.callID === "string" ? input.callID : null,
        tool: typeof input.tool === "string" ? input.tool : null,
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

// --- Plugin entry point ---

export const ArchexHookPlugin: Plugin = async ({ directory }) => {
  return {
    "tool.execute.after": async (input: ToolInputLike, output: ToolOutputLike) => {
      const tool = typeof input?.tool === "string" ? ANNOTATED_TOOLS[input.tool] : undefined;
      if (tool === undefined) return; // never touches `read`, an MCP tool, or any other tool
      const started = Date.now();
      const outcome: Outcome = {
        eligible: true,
        annotated: false,
        units: 0,
        tokens: 0,
        freshness: "unchecked",
        reason: null,
        latencyMs: 0,
      };
      const decline = (reason: string): void => {
        outcome.reason = reason;
        outcome.latencyMs = Date.now() - started;
        recordLedger(input, outcome);
      };
      try {
        const args = input.args;
        if (typeof args !== "object" || args === null || Array.isArray(args)) {
          return decline("malformed_input");
        }
        if (tool === "bash") {
          const command = (args as Record<string, unknown>).command;
          if (typeof command !== "string" || !BASH_SEARCH_MENTION.test(command)) {
            outcome.eligible = false;
            return decline("not_search_command");
          }
        }
        const text = output?.output;
        if (typeof text !== "string") return decline("malformed_output");

        const cwd =
          typeof directory === "string" && directory.length > 0 ? directory : process.cwd();
        const response = await runAnnotate(
          { host: "opencode", tool, input: args, text, cwd },
          cwd,
        );
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

        output.output = `${text}\n\n${record.annotation}`;
        outcome.annotated = true;
        outcome.tokens = record.tokens;
        outcome.latencyMs = Date.now() - started;
        recordLedger(input, outcome);
      } catch (err) {
        logDiagnostic("ts_internal_error", String(err));
        decline("internal_error");
      }
    },
  };
};
"""
