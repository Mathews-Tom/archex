"""Deterministic impact analysis from git changes and index dependencies."""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from archex.index.graph import DependencyGraph
from archex.models import DiscoveredFile
from archex.parse.adapters import default_adapter_registry

if TYPE_CHECKING:
    from archex.index.store import IndexStore
    from archex.models import CodeChunk, Edge
    from archex.parse.adapters import LanguageAdapter


class ImpactError(ValueError):
    """Raised when impact analysis cannot run."""


class ImpactRiskLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ImpactFileChange(BaseModel):
    path: str
    status: str = "M"
    old_path: str | None = None


class ImpactRisk(BaseModel):
    level: ImpactRiskLevel
    reasons: list[str] = []


class SymbolRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SymbolRiskSignal(BaseModel):
    name: str
    detail: str


class SymbolImpact(BaseModel):
    file_path: str
    symbol_name: str | None = None
    qualified_name: str | None = None
    symbol_kind: str | None = None
    start_line: int
    end_line: int
    level: SymbolRiskLevel
    signals: list[SymbolRiskSignal] = []


class ImpactReport(BaseModel):
    changed_files: list[ImpactFileChange] = []
    affected_files: list[str] = []
    affected_modules: list[str] = []
    affected_interfaces: list[str] = []
    affected_tests: list[str] = []
    unmapped_files: list[str] = []
    risk: ImpactRisk = Field(default_factory=lambda: ImpactRisk(level=ImpactRiskLevel.LOW))
    diff_ref: str | None = None
    affected_symbols: list[SymbolImpact] = []

    def to_json(self) -> str:
        data = self.model_dump(mode="json")
        if self.diff_ref is None:
            data.pop("diff_ref", None)
            data.pop("affected_symbols", None)
        return json.dumps(data, indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Impact Analysis",
            "",
            "## Summary",
            "",
            f"- **Changed files:** {len(self.changed_files)}",
            f"- **Affected files:** {len(self.affected_files)}",
            f"- **Risk:** `{self.risk.level.value}`",
            "",
            "## Changed Files",
            "",
        ]
        lines.extend(f"- `{change.status}` `{change.path}`" for change in self.changed_files)
        lines.extend(["", "## Affected Modules", ""])
        lines.extend(_path_lines(self.affected_modules))
        lines.extend(["", "## Affected Files", ""])
        lines.extend(_path_lines(self.affected_files))
        lines.extend(["", "## Public Interface Impact", ""])
        lines.extend(_path_lines(self.affected_interfaces))
        lines.extend(["", "## Test Surface", ""])
        lines.extend(_path_lines(self.affected_tests))
        lines.extend(["", "## Risk Assessment", ""])
        lines.append(f"- **Level:** `{self.risk.level.value}`")
        lines.extend(f"- `{reason}`" for reason in self.risk.reasons)
        lines.extend(["", "## Unmapped Files", ""])
        lines.extend(_path_lines(self.unmapped_files))
        if self.diff_ref is not None:
            lines.extend(["", f"## Symbol Risk (diff: `{self.diff_ref}`)", ""])
            if not self.affected_symbols:
                lines.append("- None")
            for symbol in self.affected_symbols:
                label = symbol.qualified_name or symbol.symbol_name or "<unnamed>"
                lines.append(
                    f"- `{symbol.level.value}` `{symbol.file_path}::{label}` "
                    f"(lines {symbol.start_line}-{symbol.end_line})"
                )
                lines.extend(f"  - {signal.name}: {signal.detail}" for signal in symbol.signals)
        return "\n".join(lines).rstrip() + "\n"


def git_changed_files(repo_root: Path, base_ref: str) -> list[ImpactFileChange]:
    command = ["git", "diff", "--name-status", base_ref]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ImpactError(f"git diff failed for base {base_ref}: {stderr}")
    return [_parse_name_status(line) for line in completed.stdout.splitlines() if line.strip()]


_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_NEW_FILE_PATH_RE = re.compile(r"^\+\+\+ b/(.+)$")


def git_diff_hunks(repo_root: Path, ref: str) -> dict[str, list[tuple[int, int]]]:
    """Return per-file new-side line ranges touched by a diff (ref vs. working tree).

    Parses ``git diff --unified=0`` hunk headers. A pure-deletion hunk (zero new
    lines) has no corresponding new-file line range; it is recorded as a
    single-line marker at the deletion point so it can still be intersected
    against symbol spans.
    """
    command = ["git", "diff", "--unified=0", "--no-color", ref]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ImpactError(f"git diff failed for ref {ref}: {stderr}")

    hunks: dict[str, list[tuple[int, int]]] = defaultdict(list)
    current_path: str | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("+++ "):
            match = _NEW_FILE_PATH_RE.match(line)
            current_path = match.group(1) if match else None
            continue
        if current_path is None:
            continue
        hunk_match = _HUNK_HEADER_RE.match(line)
        if hunk_match is None:
            continue
        new_start = int(hunk_match.group(1))
        new_length = int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
        if new_length == 0:
            point = max(new_start, 1)
            hunks[current_path].append((point, point))
        else:
            hunks[current_path].append((new_start, new_start + new_length - 1))
    return dict(hunks)


def _symbols_touched_by_hunks(
    chunks: list[CodeChunk], hunks: list[tuple[int, int]]
) -> list[CodeChunk]:
    """Return chunks whose line span overlaps at least one hunk range."""
    return [
        chunk
        for chunk in chunks
        if chunk.symbol_name is not None
        and any(chunk.start_line <= end and start <= chunk.end_line for start, end in hunks)
    ]


def resolve_diff_symbols(
    store: IndexStore,
    changes: list[ImpactFileChange],
    hunks: dict[str, list[tuple[int, int]]],
) -> list[CodeChunk]:
    """Map a diff's changed files to the indexed symbols (chunks) their hunks touch.

    Deleted files have no symbols to resolve against the current index (which
    reflects the post-diff state) and are skipped; the file itself is still
    reported through the enclosing file-level ``ImpactReport``.
    """
    touched: list[CodeChunk] = []
    for change in changes:
        if change.status == "D":
            continue
        file_hunks = hunks.get(change.path)
        if not file_hunks:
            continue
        chunks = store.get_chunks_for_files([change.path])
        touched.extend(_symbols_touched_by_hunks(chunks, file_hunks))
    touched.sort(key=lambda chunk: (chunk.file_path, chunk.start_line, chunk.symbol_name or ""))
    return touched


_HUB_CENTRALITY_RATIO = 2.0
_HIGH_FAN_IN_THRESHOLD = 3
_ENTRY_POINT_MAX_DEPTH = 2


def _nearest_entry_point(
    graph: DependencyGraph,
    repo_root: Path,
    file_path: str,
    path_to_language: dict[str, str],
    adapters: dict[str, LanguageAdapter],
    max_depth: int,
) -> tuple[int, str] | None:
    """Reverse-BFS from ``file_path`` over import predecessors, bounded to
    ``max_depth`` hops, looking for the nearest file an adapter identifies as
    an entry point. Returns ``(depth, entry_point_path)`` or ``None``.
    """
    entry_cache: dict[str, bool] = {}

    def is_entry(path: str) -> bool:
        if path in entry_cache:
            return entry_cache[path]
        language = path_to_language.get(path)
        result = False
        if language is not None:
            adapter = adapters.get(language)
            if adapter is not None:
                discovered = DiscoveredFile(
                    path=path, absolute_path=str(repo_root / path), language=language
                )
                try:
                    result = path in adapter.detect_entry_points([discovered])
                except OSError:
                    result = False
        entry_cache[path] = result
        return result

    visited = {file_path}
    frontier = [file_path]
    for depth in range(max_depth + 1):
        for path in sorted(frontier):
            if is_entry(path):
                return depth, path
        next_frontier: list[str] = []
        for path in frontier:
            for predecessor in graph.imported_by(path):
                if predecessor not in visited:
                    visited.add(predecessor)
                    next_frontier.append(predecessor)
        if not next_frontier:
            return None
        frontier = next_frontier
    return None


def _classify_file_risk(
    file_path: str,
    graph: DependencyGraph,
    centrality: dict[str, float],
    mean_centrality: float,
    repo_root: Path,
    path_to_language: dict[str, str],
    adapters: dict[str, LanguageAdapter],
) -> tuple[SymbolRiskLevel, list[SymbolRiskSignal]]:
    """Classify a file's risk tier from deterministic graph signals only.

    Three signals, each file-scoped (edges are file-to-file; no CALLS edges
    exist to derive symbol-level blast radius):

    - structural_centrality: PageRank share >= _HUB_CENTRALITY_RATIO x the
      graph's mean node centrality.
    - fan_in: count of distinct files that directly import this file >=
      _HIGH_FAN_IN_THRESHOLD.
    - entry_point_proximity: an adapter-identified entry point is reachable
      within _ENTRY_POINT_MAX_DEPTH import hops (upstream) of this file.

    HIGH if the file *is* an entry point, or if 2+ signals fire; MEDIUM if
    exactly 1 fires; LOW if none fire.
    """
    signals: list[SymbolRiskSignal] = []
    fired = 0

    file_centrality = centrality.get(file_path, 0.0)
    hub_threshold = mean_centrality * _HUB_CENTRALITY_RATIO
    is_hub = mean_centrality > 0 and file_centrality >= hub_threshold
    signals.append(
        SymbolRiskSignal(
            name="structural_centrality",
            detail=(
                f"centrality={file_centrality:.4f} "
                f"{'>=' if is_hub else '<'} threshold={hub_threshold:.4f} "
                f"({_HUB_CENTRALITY_RATIO}x graph mean {mean_centrality:.4f})"
            ),
        )
    )
    fired += int(is_hub)

    fan_in = len(graph.imported_by(file_path))
    is_high_fan_in = fan_in >= _HIGH_FAN_IN_THRESHOLD
    signals.append(
        SymbolRiskSignal(
            name="fan_in",
            detail=(
                f"fan_in={fan_in} {'>=' if is_high_fan_in else '<'} "
                f"threshold={_HIGH_FAN_IN_THRESHOLD}"
            ),
        )
    )
    fired += int(is_high_fan_in)

    nearest = _nearest_entry_point(
        graph, repo_root, file_path, path_to_language, adapters, _ENTRY_POINT_MAX_DEPTH
    )
    is_entry_itself = nearest is not None and nearest[0] == 0
    if nearest is not None:
        depth, entry_path = nearest
        entry_detail = (
            f"entry_point_distance={depth} <= max_depth={_ENTRY_POINT_MAX_DEPTH} "
            f"(nearest entry point: {entry_path})"
        )
    else:
        entry_detail = f"no entry point within max_depth={_ENTRY_POINT_MAX_DEPTH}"
    signals.append(SymbolRiskSignal(name="entry_point_proximity", detail=entry_detail))
    fired += int(nearest is not None)

    if is_entry_itself or fired >= 2:
        level = SymbolRiskLevel.HIGH
    elif fired == 1:
        level = SymbolRiskLevel.MEDIUM
    else:
        level = SymbolRiskLevel.LOW
    return level, signals


def analyze_diff_impact(
    store: IndexStore,
    repo_root: Path,
    changes: list[ImpactFileChange],
    hunks: dict[str, list[tuple[int, int]]],
    diff_ref: str,
) -> ImpactReport:
    """Extend the file-level impact report with per-symbol risk classification.

    Reuses ``analyze_impact()`` for the existing file-level fields, then adds
    ``diff_ref`` and ``affected_symbols`` -- only the symbols the diff's hunks
    directly touch, each tagged with a deterministic risk tier and the
    signal rationale that produced it.
    """
    base_report = analyze_impact(store, changes)
    touched = resolve_diff_symbols(store, changes, hunks)
    if not touched:
        return base_report.model_copy(update={"diff_ref": diff_ref})

    graph = DependencyGraph.from_edges(store.get_edges())
    centrality = graph.structural_centrality()
    mean_centrality = sum(centrality.values()) / len(centrality) if centrality else 0.0
    path_to_language = {
        str(item["file_path"]): str(item["language"]) for item in store.get_file_metadata()
    }
    adapters = default_adapter_registry.build_all()

    risk_cache: dict[str, tuple[SymbolRiskLevel, list[SymbolRiskSignal]]] = {}
    symbol_impacts: list[SymbolImpact] = []
    for chunk in touched:
        if chunk.file_path not in risk_cache:
            risk_cache[chunk.file_path] = _classify_file_risk(
                chunk.file_path,
                graph,
                centrality,
                mean_centrality,
                repo_root,
                path_to_language,
                adapters,
            )
        level, signals = risk_cache[chunk.file_path]
        symbol_impacts.append(
            SymbolImpact(
                file_path=chunk.file_path,
                symbol_name=chunk.symbol_name,
                qualified_name=chunk.qualified_name,
                symbol_kind=str(chunk.symbol_kind) if chunk.symbol_kind is not None else None,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                level=level,
                signals=signals,
            )
        )
    return base_report.model_copy(update={"diff_ref": diff_ref, "affected_symbols": symbol_impacts})


def analyze_impact(
    store: IndexStore,
    changes: list[ImpactFileChange],
) -> ImpactReport:
    indexed_files = {str(item["file_path"]) for item in store.get_file_metadata()}
    edges = store.get_edges()
    changed_paths = {change.path for change in changes}
    mapped_changed = changed_paths & indexed_files
    affected_files = _affected_files(edges, mapped_changed)
    all_relevant_files = sorted(mapped_changed | affected_files)
    unmapped = sorted(changed_paths - indexed_files)
    chunks = store.get_chunks_for_files(all_relevant_files)
    affected_interfaces = _public_interfaces(chunks)
    affected_tests = sorted(path for path in all_relevant_files if _is_test_path(path))
    modules = sorted(
        {_module_for_path(path) for path in all_relevant_files if _module_for_path(path)}
    )
    reasons = _risk_reasons(
        store,
        changes,
        all_relevant_files,
        affected_interfaces,
        unmapped,
        modules,
    )
    return ImpactReport(
        changed_files=sorted(changes, key=lambda change: (change.path, change.status)),
        affected_files=all_relevant_files,
        affected_modules=modules,
        affected_interfaces=affected_interfaces,
        affected_tests=affected_tests,
        unmapped_files=unmapped,
        risk=ImpactRisk(level=_risk_level(reasons), reasons=reasons),
    )


def render_impact_report(report: ImpactReport, output_format: str) -> str:
    if output_format == "json":
        return report.to_json()
    if output_format == "markdown":
        return report.to_markdown()
    raise ImpactError(f"Unsupported impact output format: {output_format}")


def _parse_name_status(line: str) -> ImpactFileChange:
    parts = line.split("\t")
    status = parts[0]
    if status.startswith("R"):
        if len(parts) != 3:
            raise ImpactError(f"Malformed rename diff line: {line}")
        return ImpactFileChange(path=parts[2], status="R", old_path=parts[1])
    if len(parts) != 2:
        raise ImpactError(f"Malformed diff line: {line}")
    return ImpactFileChange(path=parts[1], status=status[:1])


def _affected_files(edges: list[Edge], changed_files: set[str]) -> set[str]:
    affected = set(changed_files)
    for edge in edges:
        if edge.target in changed_files:
            affected.add(edge.source)
        if edge.source in changed_files:
            affected.add(edge.target)
    return affected


def _public_interfaces(chunks: list[CodeChunk]) -> list[str]:
    interfaces: list[str] = []
    for chunk in chunks:
        if chunk.symbol_name is None or chunk.symbol_kind is None:
            continue
        if chunk.visibility not in (None, "public"):
            continue
        if str(chunk.symbol_kind) not in {"function", "class", "interface"}:
            continue
        qualified_name = chunk.qualified_name or chunk.symbol_name
        interfaces.append(f"{chunk.file_path}::{qualified_name}#{chunk.symbol_kind}")
    return sorted(interfaces)


def _risk_reasons(
    store: IndexStore,
    changes: list[ImpactFileChange],
    affected_files: list[str],
    affected_interfaces: list[str],
    unmapped_files: list[str],
    modules: list[str],
) -> list[str]:
    reasons: set[str] = set()
    changed_paths = {change.path for change in changes}
    if affected_interfaces:
        reasons.add("public_interface_changed")
    if len(modules) > 1:
        reasons.add("cross_module_impact")
    if unmapped_files:
        reasons.add("unmapped_file")
    if any(_is_config_path(path) for path in changed_paths):
        reasons.add("config_changed")
    if any(_is_test_path(path) for path in changed_paths):
        reasons.add("test_surface_changed")
    for path in affected_files:
        if (store.get_file_tokens(path) or 0) >= 1000:
            reasons.add("large_file_changed")
    fan_in = _fan_in(store.get_edges(), changed_paths)
    if fan_in >= 3:
        reasons.add("high_fan_in")
    return sorted(reasons)


def _risk_level(reasons: list[str]) -> ImpactRiskLevel:
    if {"public_interface_changed", "high_fan_in", "cross_module_impact"} & set(reasons):
        return ImpactRiskLevel.HIGH
    if reasons:
        return ImpactRiskLevel.MODERATE
    return ImpactRiskLevel.LOW


def _fan_in(edges: list[Edge], changed_paths: set[str]) -> int:
    return len({edge.source for edge in edges if edge.target in changed_paths})


def _module_for_path(path: str) -> str:
    if "/" not in path:
        return ""
    return path.split("/", maxsplit=1)[0]


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", maxsplit=1)[-1]
    return path.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py")


def _is_config_path(path: str) -> bool:
    name = path.rsplit("/", maxsplit=1)[-1]
    return name in {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "tsconfig.json"}


def _path_lines(paths: list[str]) -> list[str]:
    if not paths:
        return ["- None"]
    return [f"- `{path}`" for path in paths]
