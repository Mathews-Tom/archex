"""Deterministic impact analysis from git changes and index dependencies."""

from __future__ import annotations

import json
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from archex.index.store import IndexStore
    from archex.models import CodeChunk, Edge


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
    reasons: list[str] = Field(default_factory=list)


class ImpactReport(BaseModel):
    changed_files: list[ImpactFileChange] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    affected_modules: list[str] = Field(default_factory=list)
    affected_interfaces: list[str] = Field(default_factory=list)
    affected_tests: list[str] = Field(default_factory=list)
    unmapped_files: list[str] = Field(default_factory=list)
    risk: ImpactRisk = Field(default_factory=lambda: ImpactRisk(level=ImpactRiskLevel.LOW))

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

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
        if store.get_file_tokens(path) >= 1000:
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
