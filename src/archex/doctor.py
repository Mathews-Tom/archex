"""Read-only distribution and trust diagnostics for archex projects."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from archex.config import load_index_config
from archex.index.model_policy import (
    JINA_RERANKER_MODEL,
    ModelSecurityProfile,
    embedder_security_profile,
    reranker_security_profile,
)
from archex.languages import LANGUAGE_SUPPORT
from archex.models import LanguageTier
from archex.parse.engine import TreeSitterEngine
from archex.project import ProjectState
from archex.status import ProjectStatus, inspect_project_status

CheckStatus = Literal["ok", "warning", "error"]

_DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"

_HF_EMBEDDER_MODELS: dict[str, str] = {
    "fastembed": embedder_security_profile("fastembed").model_name,
    "jina-v2": embedder_security_profile("jina-v2").model_name,
    "sentence_transformers": embedder_security_profile("sentence_transformers").model_name,
    "nomic": embedder_security_profile("nomic").model_name,
    "coderank": embedder_security_profile("coderank").model_name,
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


def inspect_doctor(source: str | Path, *, security_only: bool = False) -> DoctorReport:
    """Run read-only health diagnostics for a repo-local archex project."""
    project = ProjectState.resolve(source)
    checks: list[DoctorCheck] = []
    if not security_only:
        status = inspect_project_status(project.repo_root)
        checks.append(_index_health_check(status))
        checks.append(_index_staleness_check(status))
        checks.append(_model_cache_check(project.repo_root))
        checks.append(_grammar_check())
        checks.append(_mcp_registration_check(project.repo_root))
        checks.append(_disk_usage_check(project.project_dir))
    checks.append(_model_security_check(project.repo_root))
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
        if check.name == "model_security":
            lines.append(f"  allow_remote_code: {check.details.get('allow_remote_code', False)}")
            for label in ("embedding", "reranker", "splade"):
                value = check.details.get(label, {})
                if isinstance(value, dict):
                    entry = cast("dict[str, object]", value)
                    lines.append(
                        "  "
                        f"{label}: provider={entry.get('provider')} "
                        f"model={entry.get('model')} "
                        f"remote_code={entry.get('requires_remote_code')} "
                        f"cache_present={entry.get('cache_present')}"
                    )
            lines.append(f"  network: {check.details.get('network_implications', '')}")
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
        required_models.append(index_config.rerank_model or JINA_RERANKER_MODEL)

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


def _model_security_check(repo_root: Path) -> DoctorCheck:
    index_config = load_index_config(repo_root)
    cache_dirs = _model_cache_dirs()
    if index_config.embedder is None:
        embedding = _disabled_model_entry(
            provider="none",
            vector_requested=index_config.vector,
        )
    else:
        embedder_profile = embedder_security_profile(index_config.embedder)
        embedding = _model_security_entry(
            enabled=index_config.vector,
            provider=index_config.embedder,
            profile=embedder_profile,
            allow_remote_code=index_config.allow_remote_code,
            cache_dirs=cache_dirs,
        )
    rerank_model = index_config.rerank_model or JINA_RERANKER_MODEL
    reranker_profile = reranker_security_profile(rerank_model)

    reranker = _model_security_entry(
        enabled=index_config.rerank,
        provider=rerank_model,
        profile=reranker_profile,
        allow_remote_code=index_config.allow_remote_code,
        cache_dirs=cache_dirs,
    )
    splade = _download_entry(
        enabled=index_config.splade,
        provider="splade",
        model_name=_DEFAULT_SPLADE_MODEL,
        cache_dirs=cache_dirs,
    )

    blocked = [
        entry["provider"]
        for entry in (embedding, reranker)
        if entry["enabled"] and entry["requires_remote_code"] and not index_config.allow_remote_code
    ]
    downloads = [
        entry["provider"]
        for entry in (embedding, reranker, splade)
        if entry["enabled"] and not entry["cache_present"]
    ]
    details: dict[str, object] = {
        "allow_remote_code": index_config.allow_remote_code,
        "offline": _offline_status(),
        "embedding": embedding,
        "reranker": reranker,
        "splade": splade,
        "cache_dirs": [str(path) for path in cache_dirs],
        "network_downloads_required": downloads,
        "network_implications": _network_implications(downloads),
    }
    if blocked:
        return DoctorCheck(
            name="model_security",
            status="error",
            message="remote-code model selected without allow_remote_code opt-in",
            details=details,
        )
    if downloads:
        return DoctorCheck(
            name="model_security",
            status="warning",
            message="one or more selected local models may download before first use",
            details=details,
        )
    return DoctorCheck(
        name="model_security",
        status="ok",
        message="selected model-loading paths satisfy the remote-code policy",
        details=details,
    )


def _disabled_model_entry(*, provider: str, vector_requested: bool) -> dict[str, object]:
    return {
        "enabled": False,
        "provider": provider,
        "model": None,
        "requires_remote_code": False,
        "allow_remote_code": False,
        "remote_code_allowed": False,
        "model_revision": None,
        "code_revision": None,
        "cache_present": False,
        "cache_paths": [],
        "network_download_required": False,
        "vector_requested": vector_requested,
    }


def _model_security_entry(
    *,
    enabled: bool,
    provider: str,
    profile: ModelSecurityProfile,
    allow_remote_code: bool,
    cache_dirs: list[Path],
) -> dict[str, object]:
    cache_paths = _cached_model_paths(profile.model_name, cache_dirs)
    cache_present = any(path.exists() for path in cache_paths)
    return {
        "enabled": enabled,
        "provider": provider,
        "model": profile.model_name,
        "requires_remote_code": profile.requires_remote_code,
        "allow_remote_code": allow_remote_code,
        "remote_code_allowed": profile.requires_remote_code and allow_remote_code,
        "model_revision": profile.model_revision,
        "code_revision": profile.code_revision,
        "cache_present": cache_present,
        "cache_paths": [str(path) for path in cache_paths],
        "network_download_required": enabled and not cache_present,
    }


def _download_entry(
    *,
    enabled: bool,
    provider: str,
    model_name: str,
    cache_dirs: list[Path],
) -> dict[str, object]:
    cache_paths = _cached_model_paths(model_name, cache_dirs)
    cache_present = any(path.exists() for path in cache_paths)
    return {
        "enabled": enabled,
        "provider": provider,
        "model": model_name,
        "requires_remote_code": False,
        "cache_present": cache_present,
        "cache_paths": [str(path) for path in cache_paths],
        "network_download_required": enabled and not cache_present,
    }


def _offline_status() -> dict[str, bool]:
    return {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "").lower()
        in {"1", "true", "yes", "on"},
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "").lower()
        in {"1", "true", "yes", "on"},
    }


def _network_implications(downloads: list[object]) -> str:
    if downloads:
        return (
            "selected model-backed features may contact Hugging Face or provider caches "
            "before first use"
        )
    return "no selected model-backed feature requires a model download from the current cache state"


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
