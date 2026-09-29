"""Replay real omp search results through the annotation renderer (Stage −1 headroom).

The grep-result annotation hook can only change an agent's choice where one
search result spans two or more code units: that is the call where "which hit
is which" is a real question. This script measures how often that happens in
the operator's own omp sessions, and how many tokens the annotation would add,
before any paid campaign is funded.

```bash
uv run python scripts/annotation_headroom.py collect \
    --sessions ~/.omp/agent/sessions --output /tmp/annotation-headroom-records.json
uv run python scripts/annotation_headroom.py analyze \
    --records /tmp/annotation-headroom-records.json \
    --output benchmarks/evidence/annotation-headroom.json
```

`collect` joins every `toolCall` to its `toolResult` by `toolCallId`, keeps
the omp `grep` and `glob` results and `bash` results whose command runs `rg`,
`grep`, or `git grep`, and replays each through `archex.annotate` against the
repository's **current** index (a read-only snapshot, so no repository index
is migrated or written). Line numbers may have drifted since the session:
results whose files are gone or whose hit lines fall past the current file
length are skipped and counted. Results a tool-output extension already
compressed in the transcript cannot be replayed and are counted too.

Privacy: the records file and the evidence carry counts only. Repositories
are pseudonymous (`repo-01`, ...); no transcript text, path, prompt, or
repository name is written. Keep the records file outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from archex import __version__
from archex.annotate import (
    MAX_CALL_TOKENS,
    MAX_LINE_TOKENS,
    ParsedSearch,
    annotate_parsed,
    bash_search_base,
    parse_search_result,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260909
GATE_MIN_AMBIGUOUS_SHARE = 0.15
REPLAYED_TOOLS = ("grep", "glob", "bash")
#: Wrappers tool-output compressors leave in the transcript in place of the
#: original text (laconic's codec, omp's context shaker); the original is not
#: recoverable from the session file.
_COMPRESSED_MARKERS = ("[laconic ", "[shaken ~")


@dataclass
class _Repo:
    pseudonym: str
    snapshot: Path | None
    lines: dict[str, int | None] = field(default_factory=dict[str, "int | None"])


class _Collector:
    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = snapshot_dir
        self.git_roots: dict[str, Path | None] = {}
        self.repos: dict[Path, _Repo] = {}
        self.rows: list[dict[str, Any]] = []
        self.excluded: Counter[str] = Counter()
        self.drift_skipped: Counter[str] = Counter()
        self.eligible = 0
        self.first_day = ""
        self.last_day = ""

    def git_root(self, directory: Path) -> Path | None:
        key = str(directory)
        if key not in self.git_roots:
            root: Path | None = None
            if directory.is_dir():
                done = subprocess.run(
                    ["git", "-C", key, "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if done.returncode == 0 and done.stdout.strip():
                    root = Path(done.stdout.strip()).resolve()
            self.git_roots[key] = root
        return self.git_roots[key]

    def repo(self, root: Path) -> _Repo:
        if root not in self.repos:
            pseudonym = f"repo-{len(self.repos) + 1:02d}"
            self.repos[root] = _Repo(pseudonym, self._snapshot(root, pseudonym))
        return self.repos[root]

    def _snapshot(self, root: Path, pseudonym: str) -> Path | None:
        source = root / ".archex" / "index.db"
        if not source.is_file():
            return None
        target = self.snapshot_dir / f"{pseudonym}.db"
        target.unlink(missing_ok=True)
        try:
            src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
            try:
                dst = sqlite3.connect(target)
                try:
                    src.backup(dst)
                finally:
                    dst.close()
            finally:
                src.close()
        except sqlite3.Error:
            return None
        return target

    def line_count(self, repo: _Repo, path: Path) -> int | None:
        key = str(path)
        if key not in repo.lines:
            try:
                data = path.read_bytes()
            except OSError:
                repo.lines[key] = None
            else:
                ends_open = bool(data) and not data.endswith(b"\n")
                repo.lines[key] = data.count(b"\n") + int(ends_open)
        return repo.lines[key]


def _object(value: object) -> dict[str, Any] | None:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _objects(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]


def _iter_session(
    path: Path,
) -> tuple[str | None, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    """Session cwd and the (tool, input, toolResult message) triples in one file."""
    cwd: str | None = None
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    results: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if cwd is None and '"type":"session"' in line[:200]:
                try:
                    header = _object(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if header is not None and isinstance(header.get("cwd"), str):
                    cwd = str(header["cwd"])
                continue
            if '"toolCall"' not in line and '"toolResult"' not in line:
                continue
            try:
                entry = _object(json.loads(line))
            except json.JSONDecodeError:
                continue
            if entry is None or entry.get("type") != "message":
                continue
            message = _object(entry.get("message")) or {}
            if message.get("role") == "assistant":
                for block in _objects(message.get("content")):
                    if block.get("type") == "toolCall":
                        arguments = _object(block.get("arguments")) or {}
                        calls[str(block.get("id"))] = (str(block.get("name")), arguments)
            elif message.get("role") == "toolResult":
                call_id = str(message.get("toolCallId"))
                name, arguments = calls.get(call_id, (str(message.get("toolName")), {}))
                results.append((name, arguments, message))
                timestamp = entry.get("timestamp")
                if isinstance(timestamp, str) and len(timestamp) >= 10:
                    message["_day"] = timestamp[:10]
    return cwd, results


def _result_text(message: dict[str, Any]) -> str:
    return "\n".join(
        str(block.get("text", ""))
        for block in _objects(message.get("content"))
        if block.get("type") == "text"
    )


def _resolve_repo(
    collector: _Collector, parsed: ParsedSearch, cwd: Path
) -> tuple[Path | None, Path]:
    base = (cwd / parsed.base_dir).resolve() if parsed.base_dir else cwd
    for hits in parsed.files:
        candidate = (base / hits.path).resolve()
        if candidate.exists():
            return collector.git_root(candidate.parent), base
    return collector.git_root(cwd), base


def _drift(collector: _Collector, repo: _Repo, parsed: ParsedSearch, base: Path) -> str | None:
    if parsed.kind != "lines":
        return None
    for hits in parsed.files:
        count = collector.line_count(repo, (base / hits.path).resolve())
        if count is None:
            return "file_missing"
        if any(line > count for line in hits.matches):
            return "line_past_end"
    return None


def _replay_one(
    collector: _Collector,
    tool: str,
    arguments: dict[str, Any],
    message: dict[str, Any],
    cwd_text: str | None,
) -> None:
    if tool == "bash":
        command = arguments.get("command")
        if not isinstance(command, str) or bash_search_base(command) is None:
            return
    collector.eligible += 1
    day = message.get("_day")
    if isinstance(day, str):
        collector.first_day = min(collector.first_day or day, day)
        collector.last_day = max(collector.last_day, day)
    if message.get("isError") is True:
        collector.excluded["tool_error"] += 1
        return
    text = _result_text(message)
    if any(marker in text for marker in _COMPRESSED_MARKERS):
        collector.excluded["compressed_by_extension"] += 1
        return
    if cwd_text is None:
        collector.excluded["no_session_cwd"] += 1
        return
    parsed = parse_search_result(tool, arguments, text)
    if isinstance(parsed, str):
        collector.excluded[parsed] += 1
        return
    cwd = Path(cwd_text)
    root, base = _resolve_repo(collector, parsed, cwd)
    if root is None:
        collector.excluded["no_repository"] += 1
        return
    repo = collector.repo(root)
    if repo.snapshot is None:
        collector.excluded["no_index"] += 1
        return
    drift = _drift(collector, repo, parsed, base)
    if drift is not None:
        collector.drift_skipped[drift] += 1
        return
    try:
        outcome = annotate_parsed(
            parsed, cwd=cwd, repo_root=root, index_path=repo.snapshot, freshness="replay"
        )
    except Exception as exc:  # noqa: BLE001 - one unreadable index must not stop the replay
        collector.excluded[f"index_error:{type(exc).__name__}"] += 1
        return
    collector.rows.append(
        {
            "repo": repo.pseudonym,
            "tool": tool,
            "format": parsed.format,
            "units_hit": outcome.units_hit,
            "annotated": outcome.annotated,
            "tokens": outcome.tokens,
            "capped": outcome.capped,
            "reason": outcome.reason,
        }
    )


def collect(sessions: Path, output: Path, snapshot_dir: Path) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    collector = _Collector(snapshot_dir)
    files = sorted(sessions.rglob("*.jsonl"))
    for index, path in enumerate(files, 1):
        cwd, results = _iter_session(path)
        for tool, arguments, message in results:
            if tool in REPLAYED_TOOLS:
                _replay_one(collector, tool, arguments, message, cwd)
        if index % 250 == 0:
            print(f"{index}/{len(files)} sessions, {len(collector.rows)} rows", file=sys.stderr)
    document = {
        "session_files": len(files),
        "first_day": collector.first_day,
        "last_day": collector.last_day,
        "eligible_calls": collector.eligible,
        "excluded": dict(sorted(collector.excluded.items())),
        "drift_skipped": dict(sorted(collector.drift_skipped.items())),
        "repositories": len(collector.repos),
        "repositories_without_index": sum(1 for r in collector.repos.values() if not r.snapshot),
        "rows": collector.rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


# --- analysis -------------------------------------------------------------------


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _cluster_bootstrap_share(by_repo: dict[str, tuple[int, int]]) -> tuple[float, float, float]:
    """Pooled share sum(k)/sum(n) with a repository-clustered percentile 95% CI."""
    names = sorted(by_repo)
    total_k = sum(k for k, _ in by_repo.values())
    total_n = sum(n for _, n in by_repo.values())
    rng = random.Random(BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [by_repo[rng.choice(names)] for _ in names]
        n = sum(n for _, n in sample)
        draws.append(sum(k for k, _ in sample) / n if n else 0.0)
    draws.sort()
    low = draws[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))]
    high = draws[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))]
    return total_k / total_n, low, high


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def _share_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_repo: dict[str, tuple[int, int]] = {}
    for row in rows:
        k, n = by_repo.get(row["repo"], (0, 0))
        by_repo[row["repo"]] = (k + int(row["units_hit"] >= 2), n + 1)
    if len(by_repo) < 2:
        return {"calls": len(rows), "repositories": len(by_repo)}
    share, low, high = _cluster_bootstrap_share(by_repo)
    return {
        "calls": len(rows),
        "repositories": len(by_repo),
        "ambiguous_calls": sum(k for k, _ in by_repo.values()),
        "share": round(share, 4),
        "ci95_repository_clustered": [round(low, 4), round(high, 4)],
    }


def analyze(records: Path, output: Path) -> None:
    document = cast("dict[str, Any]", json.loads(records.read_text(encoding="utf-8")))
    rows = cast("list[dict[str, Any]]", document["rows"])
    annotated = [row for row in rows if row["annotated"]]
    tokens = [int(row["tokens"]) for row in annotated]
    primary = _share_block(rows)
    per_repo: list[dict[str, Any]] = []
    for name in sorted({row["repo"] for row in rows}):
        repo_rows = [row for row in rows if row["repo"] == name]
        k = sum(1 for row in repo_rows if row["units_hit"] >= 2)
        per_repo.append(
            {
                "repo": name,
                "calls": len(repo_rows),
                "ambiguous_calls": k,
                "share": round(k / len(repo_rows), 4),
                "ci95_wilson_call_level": list(_wilson(k, len(repo_rows))),
            }
        )
    share = float(primary.get("share", 0.0))
    evidence = {
        "artifact": "annotation-headroom",
        "stage": "Stage -1 (spec §2.2, §8)",
        "generated_on": datetime.now(UTC).strftime("%Y-%m-%d"),
        "archex_version": __version__,
        "question": (
            "Share of eligible omp search calls whose hits span two or more indexed code "
            "units — the only calls where an annotation can change which file the agent "
            "opens — and the tokens the annotation would add."
        ),
        "corpus": {
            "source": "local omp session transcripts (toolCall joined to toolResult by toolCallId)",
            "session_files": document["session_files"],
            "session_days": [document["first_day"], document["last_day"]],
            "eligible_calls_seen": document["eligible_calls"],
            "measured_calls": len(rows),
            "excluded": document["excluded"],
            "drift_skipped": document["drift_skipped"],
            "drift_skipped_total": sum(document["drift_skipped"].values()),
            "repositories_seen": document["repositories"],
            "repositories_without_index": document["repositories_without_index"],
            "repositories_measured": primary.get("repositories", 0),
        },
        "method": {
            "renderer": "archex.annotate.annotate_parsed (the shipped hook's renderer)",
            "index": (
                "each repository's current local index, read from a snapshot; line numbers "
                "may have drifted since the session. Results whose files are gone or whose "
                "hit lines fall past the current file length are skipped (drift_skipped)."
            ),
            "denominator": (
                "every measured eligible call, including calls with no hit or with hits "
                "only in files holding no code unit (0 units)"
            ),
            "ambiguous": "units_hit >= 2 (distinct units containing a hit, before suppression)",
            "ci": (
                f"repository-clustered percentile bootstrap, {BOOTSTRAP_RESAMPLES} resamples, "
                f"seed {BOOTSTRAP_SEED}"
            ),
            "caps": {"line_tokens": MAX_LINE_TOKENS, "call_tokens": MAX_CALL_TOKENS},
            "tokenizer": "cl100k_base (archex.reporting.count_tokens)",
            "privacy": "counts only; repositories pseudonymous; no text, paths, or prompts",
        },
        "ambiguous_share": primary,
        "ambiguous_share_by_tool": {
            tool: _share_block([row for row in rows if row["tool"] == tool])
            for tool in REPLAYED_TOOLS
        },
        "ambiguous_share_among_calls_with_a_unit": _share_block(
            [row for row in rows if row["units_hit"] >= 1]
        ),
        "annotation_tokens": {
            "annotated_calls": len(annotated),
            "annotated_share_of_measured": round(len(annotated) / len(rows), 4) if rows else 0.0,
            "p50": _percentile(tokens, 0.50) if tokens else 0,
            "p95": _percentile(tokens, 0.95) if tokens else 0,
            "max": max(tokens) if tokens else 0,
            "mean": round(statistics.fmean(tokens), 1) if tokens else 0.0,
            "mean_per_measured_call": round(sum(tokens) / len(rows), 1) if rows else 0.0,
            "cap_hit_rate": round(sum(1 for row in annotated if row["capped"]) / len(annotated), 4)
            if annotated
            else 0.0,
        },
        "not_annotated_reasons": dict(
            sorted(Counter(str(row["reason"]) for row in rows if not row["annotated"]).items())
        ),
        "per_repository": per_repo,
        "gate": {
            "rule": f"proceed only if ambiguous share >= {GATE_MIN_AMBIGUOUS_SHARE:.0%}",
            "threshold": GATE_MIN_AMBIGUOUS_SHARE,
            "verdict": "pass" if share >= GATE_MIN_AMBIGUOUS_SHARE else "fail",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ambiguous_share": primary, "gate": evidence["gate"]}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--sessions", type=Path, default=Path.home() / ".omp" / "agent" / "sessions")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("/tmp/archex-annotation-headroom-indexes"),
        help="where read-only index snapshots are written (outside every repository)",
    )
    a = sub.add_parser("analyze")
    a.add_argument("--records", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.sessions.expanduser(), args.output, args.snapshot_dir)
    else:
        analyze(args.records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
