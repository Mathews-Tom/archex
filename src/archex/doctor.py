"""Read-only distribution and trust diagnostics for archex projects."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from archex.config import load_index_config
from archex.languages import LANGUAGE_SUPPORT
from archex.models import LanguageTier
from archex.parse.engine import TreeSitterEngine
from archex.project import ProjectState
from archex.status import ProjectStatus, inspect_project_status

CheckStatus = Literal["ok", "warning", "error"]

_JINA_V2_MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
_DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v3"
_DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"

_HF_EMBEDDER_MODELS: dict[str, str] = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "jina-v2": _JINA_V2_MODEL_ID,
    "sentence_transformers": _JINA_V2_MODEL_ID,
    "nomic": "nomic-ai/nomic-embed-code",
    "coderank": "nomic-ai/CodeRankEmbed",
}


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class DoctorReport:
    repo_root: Path
    status: CheckStatus
    checks: list[DoctorCheck]

    def has_errors(self) -> bool:
        return self.status == "error"

    def to_payload(self) -> dict[str, object]:
        return {
            "repo_root": str(self.repo_root),
            "status": self.status,
            "checks": [check.to_payload() for check in self.checks],
        }


def inspect_doctor(source: str | Path) -> DoctorReport:
    """Run read-only health diagnostics for a repo-local archex project."""
    project = ProjectState.resolve(source)
    checks: list[DoctorCheck] = []
    status = inspect_project_status(project.repo_root)
    checks.append(_index_health_check(status))
    checks.append(_index_staleness_check(status))
    checks.append(_model_cache_check(project.repo_root))
    checks.append(_grammar_check())
    checks.append(_mcp_registration_check(project.repo_root))
    checks.append(_disk_usage_check(project.project_dir))
    return DoctorReport(
        repo_root=project.repo_root,
        status=_overall_status(checks),
        checks=checks,
    )


def render_doctor_text(report: DoctorReport) -> str:
    """Render a compact text report suitable for terminal use."""
    lines = [
        f"archex doctor: {report.status}",
        f"Repository: {report.repo_root}",
        "",
    ]
    for check in report.checks:
        lines.append(f"[{check.status}] {check.name}: {check.message}")
        if check.name == "disk_usage":
            total_bytes = check.details.get("total_bytes", 0)
            lines.append(f"  .archex size: {_format_bytes(_int_value(total_bytes))}")
        if check.name == "grammars":
            full_value = check.details.get("full", {})
            chunk_value = check.details.get("chunk_only", {})
            if isinstance(full_value, dict) and isinstance(chunk_value, dict):
                full = cast("dict[str, object]", full_value)
                chunk = cast("dict[str, object]", chunk_value)
                full_available = _int_value(full.get("available", 0))
                full_total = _int_value(full.get("total", 0))
                chunk_available = _int_value(chunk.get("available", 0))
                chunk_total = _int_value(chunk.get("total", 0))
                lines.append(f"  full grammars: {full_available}/{full_total} available")
                lines.append(f"  chunk-only grammars: {chunk_available}/{chunk_total} available")
    return "\n".join(lines).rstrip() + "\n"


def _index_health_check(status: ProjectStatus) -> DoctorCheck:
    details = _status_details(status)
    if status.state == "fresh" or status.state in {"dirty", "stale", "needs_reindex"}:
        if status.files_indexed > 0 and status.chunks_indexed > 0:
            return DoctorCheck(
                name="index_health",
                status="ok",
                message=(
                    f"index readable with {status.files_indexed} files "
                    f"and {status.chunks_indexed} chunks"
                ),
                details=details,
            )
        return DoctorCheck(
            name="index_health",
            status="error",
            message="index is readable but empty",
            details=details,
        )
    return DoctorCheck(
        name="index_health",
        status="error",
        message=f"index state is {status.state}",
        details=details,
    )


def _index_staleness_check(status: ProjectStatus) -> DoctorCheck:
    details: dict[str, object] = {
        "state": status.state,
        "working_tree": status.working_tree,
        "current_commit": status.current_commit,
        "indexed_commit": status.indexed_commit,
    }
    if status.state == "fresh":
        return DoctorCheck(
            name="index_staleness",
            status="ok",
            message="index matches the current commit and working tree signature",
            details=details,
        )
    if status.state in {"dirty", "stale", "needs_reindex"}:
        return DoctorCheck(
            name="index_staleness",
            status="warning",
            message=f"index is {status.state}; run archex index to refresh",
            details=details,
        )
    return DoctorCheck(
        name="index_staleness",
        status="error",
        message="staleness cannot be evaluated until the index is healthy",
        details=details,
    )


def _model_cache_check(repo_root: Path) -> DoctorCheck:
    index_config = load_index_config(repo_root)
    required_models: list[str] = []
    unknown_models: list[str] = []
    embedder = index_config.embedder or "jina-v2"
    if index_config.vector:
        model_name = _HF_EMBEDDER_MODELS.get(embedder)
        if model_name is None:
            unknown_models.append(embedder)
        else:
            required_models.append(model_name)
    if index_config.splade:
        required_models.append(_DEFAULT_SPLADE_MODEL)
    if index_config.rerank:
        required_models.append(index_config.rerank_model or _DEFAULT_RERANK_MODEL)

    cache_dirs = _model_cache_dirs()
    model_paths = {model: _cached_model_paths(model, cache_dirs) for model in required_models}
    missing = [
        model for model, paths in model_paths.items() if not any(path.exists() for path in paths)
    ]
    missing.extend(unknown_models)
    details: dict[str, object] = {
        "required": bool(required_models or unknown_models),
        "configured_embedder": embedder,
        "models": required_models,
        "unknown_models": unknown_models,
        "cache_dirs": [str(path) for path in cache_dirs],
        "missing_models": missing,
    }
    if not required_models:
        return DoctorCheck(
            name="model_cache",
            status="ok",
            message="BM25-only configuration does not require a local model cache",
            details=details,
        )
    if missing:
        return DoctorCheck(
            name="model_cache",
            status="warning",
            message="one or more enabled local models are not present in the cache",
            details=details,
        )
    return DoctorCheck(
        name="model_cache",
        status="ok",
        message="enabled local models are present in the cache",
        details=details,
    )


def _grammar_check() -> DoctorCheck:
    engine = TreeSitterEngine()
    missing: dict[str, str] = {}
    available_by_tier: dict[LanguageTier, int] = {
        LanguageTier.FULL: 0,
        LanguageTier.CHUNK_ONLY: 0,
        LanguageTier.UNKNOWN: 0,
    }
    total_by_tier: dict[LanguageTier, int] = {
        LanguageTier.FULL: 0,
        LanguageTier.CHUNK_ONLY: 0,
        LanguageTier.UNKNOWN: 0,
    }
    for language_id, support in sorted(LANGUAGE_SUPPORT.items()):
        total_by_tier[support.tier] += 1
        try:
            engine.get_language(language_id)
        except Exception as exc:
            missing[language_id] = str(exc)
        else:
            available_by_tier[support.tier] += 1

    details: dict[str, object] = {
        "full": {
            "available": available_by_tier[LanguageTier.FULL],
            "total": total_by_tier[LanguageTier.FULL],
        },
        "chunk_only": {
            "available": available_by_tier[LanguageTier.CHUNK_ONLY],
            "total": total_by_tier[LanguageTier.CHUNK_ONLY],
        },
        "missing": missing,
    }
    if not missing:
        return DoctorCheck(
            name="grammars",
            status="ok",
            message="all declared tree-sitter grammars load",
            details=details,
        )
    full_missing = [
        language_id
        for language_id in missing
        if LANGUAGE_SUPPORT[language_id].tier == LanguageTier.FULL
    ]
    return DoctorCheck(
        name="grammars",
        status="error" if full_missing else "warning",
        message=f"{len(missing)} declared grammars failed to load",
        details=details,
    )


def _mcp_registration_check(repo_root: Path) -> DoctorCheck:
    package_available = importlib.util.find_spec("mcp") is not None
    checked_paths = _mcp_config_candidates(repo_root)
    registrations: list[str] = []
    invalid_configs: dict[str, str] = {}
    for path in checked_paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid_configs[str(path)] = str(exc)
            continue
        if _contains_archex_mcp_registration(data):
            registrations.append(str(path))

    details: dict[str, object] = {
        "package_available": package_available,
        "registered": bool(registrations),
        "registrations": registrations,
        "checked_paths": [str(path) for path in checked_paths],
        "invalid_configs": invalid_configs,
        "expected": {"command": "archex", "args": ["mcp"]},
    }
    if invalid_configs:
        return DoctorCheck(
            name="mcp_registration",
            status="warning",
            message="one or more MCP config files could not be read",
            details=details,
        )
    if registrations:
        return DoctorCheck(
            name="mcp_registration",
            status="ok",
            message="archex MCP registration found",
            details=details,
        )
    if package_available:
        return DoctorCheck(
            name="mcp_registration",
            status="ok",
            message=(
                "MCP package is installed; no archex client registration found "
                "in known config files"
            ),
            details=details,
        )
    return DoctorCheck(
        name="mcp_registration",
        status="warning",
        message="MCP package is not installed; install archex[mcp] before registering the server",
        details=details,
    )


def _disk_usage_check(project_dir: Path) -> DoctorCheck:
    total = _disk_usage_bytes(project_dir) if project_dir.exists() else 0
    return DoctorCheck(
        name="disk_usage",
        status="ok",
        message=f".archex uses {_format_bytes(total)}",
        details={"path": str(project_dir), "total_bytes": total},
    )


def _status_details(status: ProjectStatus) -> dict[str, object]:
    return {
        "repo_root": str(status.repo_root),
        "initialized": status.initialized,
        "state": status.state,
        "index_path": str(status.index_path),
        "current_commit": status.current_commit,
        "indexed_commit": status.indexed_commit,
        "working_tree": status.working_tree,
        "files_indexed": status.files_indexed,
        "chunks_indexed": status.chunks_indexed,
        "languages": status.languages,
        "vector_index_available": status.vector_index_available,
        "error": status.error,
    }


def _overall_status(checks: list[DoctorCheck]) -> CheckStatus:
    if any(check.status == "error" for check in checks):
        return "error"
    if any(check.status == "warning" for check in checks):
        return "warning"
    return "ok"


def _model_cache_dirs() -> list[Path]:
    env_home = os.environ.get("HF_HOME")
    base = Path(env_home).expanduser() if env_home else Path.home() / ".cache" / "huggingface"
    return [base / "hub", Path.home() / ".cache" / "fastembed"]


def _cached_model_paths(model: str, cache_dirs: list[Path]) -> list[Path]:
    if Path(model).expanduser().exists():
        return [Path(model).expanduser()]
    hf_name = "models--" + model.replace("/", "--")
    return [cache_dir / hf_name for cache_dir in cache_dirs]


def _mcp_config_candidates(repo_root: Path) -> list[Path]:
    home = Path.home()
    return [
        repo_root / ".mcp.json",
        repo_root / ".claude" / "settings.json",
        home / ".claude.json",
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        home / ".config" / "claude" / "claude_desktop_config.json",
    ]


def _contains_archex_mcp_registration(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    mapping = cast("dict[str, object]", value)
    servers = mapping.get("mcpServers")
    if isinstance(servers, dict):
        server_mapping = cast("dict[str, object]", servers)
        entry = server_mapping.get("archex")
        if isinstance(entry, dict):
            entry_mapping = cast("dict[str, object]", entry)
            command = entry_mapping.get("command")
            args = entry_mapping.get("args")
            if command == "archex" and isinstance(args, list) and "mcp" in args:
                return True
    return any(_contains_archex_mcp_registration(child) for child in mapping.values())


def _disk_usage_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def _int_value(value: object) -> int:
    return value if isinstance(value, int) else 0
