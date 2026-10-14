"""TypeScript status adapter paths and rendering."""

from __future__ import annotations

from pathlib import Path

from archex.client_setup.models import TS_STATUS_KEY, ClientName, ClientScope
from archex.project import PROJECT_DIR_NAME
from archex.status_snapshot import (
    DEFAULT_STALE_AFTER_SECONDS,
    SNAPSHOT_FILENAME,
    STATUS_SNAPSHOT_VERSION,
    WATCH_OBSERVATION_TTL_SECONDS,
)

#: Filename of the installed omp/Pi status extension module. Its own name is
#: the ownership marker, exactly as the hook modules' filenames are.
_TS_STATUS_MODULE_FILENAME = "archex-status.ts"


def ts_status_module_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    if client == "omp":
        return (
            repo_root / ".omp" / "extensions" / _TS_STATUS_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".omp" / "agent" / "extensions" / _TS_STATUS_MODULE_FILENAME
        )
    return (
        repo_root / ".pi" / "extensions" / _TS_STATUS_MODULE_FILENAME
        if scope == "project"
        else Path.home() / ".pi" / "agent" / "extensions" / _TS_STATUS_MODULE_FILENAME
    )


def render_ts_status_module() -> str:
    """Render the omp/Pi status module with the snapshot contract baked in."""
    return (
        _TS_STATUS_MODULE_TEMPLATE.replace(
            "__ARCHEX_SNAPSHOT_VERSION__", str(STATUS_SNAPSHOT_VERSION)
        )
        .replace("__ARCHEX_STALE_DEFAULT__", str(DEFAULT_STALE_AFTER_SECONDS))
        .replace("__ARCHEX_WATCH_TTL__", str(WATCH_OBSERVATION_TTL_SECONDS))
        .replace("__ARCHEX_PROJECT_DIR__", PROJECT_DIR_NAME)
        .replace("__ARCHEX_SNAPSHOT_FILENAME__", SNAPSHOT_FILENAME)
        .replace("__ARCHEX_STATUS_KEY__", TS_STATUS_KEY)
    )


_TS_STATUS_MODULE_TEMPLATE = r"""/**
 * archex status extension module (R23 - oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --statusline` /
 * `archex install-client pi --statusline` (opt-in; never installed by
 * default). A separate file from the search and post-edit hook modules, so
 * every surface installs and removes independently.
 *
 * Upstream contract (verified against each host's own type declarations, not
 * docs):
 * - `ctx.ui.setStatus(key, text)` sets keyed footer/status-bar text, rendered
 *   by the built-in footer and by the `status` status-line segment, sorted by
 *   key (oh-my-pi `src/modes/controllers/extension-ui-controller.ts` wires it
 *   to `setHookStatus`; Pi `dist/core/extensions/types.d.ts` declares it on
 *   `ExtensionUIContext`).
 * - Both hosts declare `turn_start`, `turn_end`, and `tool_result` events with
 *   the same `(event, ctx)` handler shape, and `ctx.hasUI` is false in
 *   print/RPC modes where `setStatus` is a documented no-op.
 *
 * Unlike the hook modules, this one spawns no subprocess and opens no index:
 * a repaint is a single `readFileSync` of the bounded snapshot archex
 * publishes plus string formatting, inside a process the host already runs.
 * Every path resolves without throwing; an unreadable, malformed, or
 * unknown-version snapshot renders as an explicit state instead.
 */
import { existsSync, readFileSync } from "node:fs";
import { join, parse } from "node:path";

// --- Baked in at install time from the Python contract ---

const ARCHEX_SNAPSHOT_VERSION = __ARCHEX_SNAPSHOT_VERSION__;
const ARCHEX_STALE_AFTER_SECONDS = __ARCHEX_STALE_DEFAULT__;
const ARCHEX_WATCH_TTL_SECONDS = __ARCHEX_WATCH_TTL__;
const ARCHEX_PROJECT_DIR = "__ARCHEX_PROJECT_DIR__";
const ARCHEX_SNAPSHOT_FILENAME = "__ARCHEX_SNAPSHOT_FILENAME__";
const ARCHEX_STATUS_KEY = "__ARCHEX_STATUS_KEY__";

/** Override for tests and for a host started outside the repository. */
const ARCHEX_SNAPSHOT_ENV_VAR = "ARCHEX_STATUS_SNAPSHOT";

/** Freshness-budget override, honoured by every archex status renderer. */
const ARCHEX_STALE_ENV_VAR = "ARCHEX_STATUS_STALE_AFTER_SECONDS";

/** Reader-state lines, worded exactly as the shell renderer words them. */
const ARCHEX_MISSING_LINE = "archex missing - no status snapshot - run: archex index";
const ARCHEX_CORRUPT_LINE = "archex corrupt - unreadable snapshot - run: archex status";

interface ArchexSnapshot {
  version?: unknown;
  state?: unknown;
  index_revision?: unknown;
  files_indexed?: unknown;
  chunks_indexed?: unknown;
  pending_delta_files?: unknown;
  pending_view_complete?: unknown;
  reindex_required?: unknown;
  written_at_epoch?: unknown;
  watch_observed_epoch?: unknown;
}

function snapshotPath(start: string): string | null {
  const override = process.env[ARCHEX_SNAPSHOT_ENV_VAR];
  if (override && override.trim().length > 0) return override;
  let current = start;
  for (;;) {
    const candidate = join(current, ARCHEX_PROJECT_DIR, ARCHEX_SNAPSHOT_FILENAME);
    // Existence, not readability: a present-but-unreadable snapshot must
    // classify as `corrupt` here, exactly as it does in the shell renderer
    // and the Python reader. Probing with a read would skip it and report
    // either `missing` or -- worse -- a parent repository's status.
    if (existsSync(candidate)) return candidate;
    const parent = parse(current).dir;
    if (!parent || parent === current) return null;
    current = parent;
  }
}

/** Freshness budget in seconds, overridable exactly as the CLI allows. */
function staleAfterSeconds(): number {
  const raw = process.env[ARCHEX_STALE_ENV_VAR];
  if (raw === undefined) return ARCHEX_STALE_AFTER_SECONDS;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : ARCHEX_STALE_AFTER_SECONDS;
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function readText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function ageSegment(writtenAtEpoch: number, now: number): string {
  if (writtenAtEpoch <= 0) return "";
  const age = Math.max(0, now - writtenAtEpoch);
  if (age < 60) return ` - ${age}s ago`;
  if (age < 3600) return ` - ${Math.floor(age / 60)}m ago`;
  return ` - ${Math.floor(age / 3600)}h ago`;
}

/** Render the status text for `cwd`, or a state naming why it cannot. */
export function renderArchexStatus(cwd: string, nowSeconds?: number): string {
  const path = snapshotPath(cwd);
  if (path === null) return ARCHEX_MISSING_LINE;

  // An absent file is `missing`; unreadable or malformed bytes are
  // `corrupt`. The two have different remedies -- publish one versus
  // re-publish this one -- so they must not collapse into each other.
  let raw: string;
  try {
    raw = readFileSync(path, "utf-8");
  } catch (err) {
    const code = (err as { code?: string }).code;
    return code === "ENOENT" ? ARCHEX_MISSING_LINE : ARCHEX_CORRUPT_LINE;
  }

  let parsed: ArchexSnapshot;
  try {
    const document: unknown = JSON.parse(raw);
    if (typeof document !== "object" || document === null || Array.isArray(document)) {
      return ARCHEX_CORRUPT_LINE;
    }
    parsed = document as ArchexSnapshot;
  } catch {
    return ARCHEX_CORRUPT_LINE;
  }

  if (typeof parsed.version !== "number") return ARCHEX_CORRUPT_LINE;
  if (parsed.version !== ARCHEX_SNAPSHOT_VERSION) {
    return `archex unsupported - snapshot v${parsed.version} - upgrade archex`;
  }

  const persisted = readText(parsed.state);
  if (persisted !== "fresh" && persisted !== "dirty" && persisted !== "pending") {
    return ARCHEX_CORRUPT_LINE;
  }

  const now = nowSeconds ?? Math.floor(Date.now() / 1000);
  const writtenAtEpoch = readNumber(parsed.written_at_epoch);
  const age = writtenAtEpoch > 0 ? Math.max(0, now - writtenAtEpoch) : null;
  const stale = age !== null && age > staleAfterSeconds();
  const state = stale ? "stale" : persisted;

  let detail = "";
  if (state === "fresh") {
    const files = readNumber(parsed.files_indexed);
    const chunks = readNumber(parsed.chunks_indexed);
    detail = ` - ${files} files, ${chunks} chunks`;
  } else if (state === "pending") {
    const pending = readNumber(parsed.pending_delta_files);
    const complete = parsed.pending_view_complete !== false;
    detail = ` - ${pending}${complete ? "" : "+"} awaiting sync`;
  } else if (state === "dirty") {
    detail = parsed.reindex_required === true ? " - reindex required" : " - index behind tree";
  } else {
    detail = " - unverified since measurement";
  }

  const revision = readText(parsed.index_revision);
  const revisionSegment = revision.length > 0 ? ` - rev ${revision.slice(0, 8)}` : "";

  const watchEpoch = readNumber(parsed.watch_observed_epoch);
  const watching = watchEpoch > 0 && now - watchEpoch <= ARCHEX_WATCH_TTL_SECONDS;

  return `archex ${state}${detail}${revisionSegment}${ageSegment(writtenAtEpoch, now)}${
    watching ? " - watch" : ""
  }`;
}

// --- Minimal structural types for the events used ---
//
// Declared locally (never imported from either host package) so this module
// has zero import-resolution dependency on which host loaded it.

interface StatusUiLike {
  setStatus(key: string, text: string | undefined): void;
}

interface StatusContextLike {
  ui?: StatusUiLike;
  hasUI?: boolean;
}

type StatusHandler = (event: unknown, ctx: StatusContextLike) => void;

interface StatusHost {
  on(event: "turn_start" | "turn_end" | "tool_result", handler: StatusHandler): unknown;
}

/** Refresh the footer status, swallowing every failure. */
export function publishArchexStatus(ctx: StatusContextLike): void {
  try {
    if (ctx.hasUI === false || !ctx.ui) return;
    ctx.ui.setStatus(ARCHEX_STATUS_KEY, renderArchexStatus(process.cwd()));
  } catch {
    // A status refresh must never disturb the host's turn.
  }
}

export default function archexStatusExtension(pi: StatusHost): void {
  const refresh: StatusHandler = (_event, ctx) => {
    publishArchexStatus(ctx);
  };
  // Three refresh points: entering a turn, after each tool result (so an edit
  // shows as pending mid-turn), and at rest. All three read one small file.
  pi.on("turn_start", refresh);
  pi.on("tool_result", refresh);
  pi.on("turn_end", refresh);
}
"""
