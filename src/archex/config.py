"""Default configuration values and config loading utilities."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, cast

from archex.models import Config, IndexConfig, RepoSource
from archex.project import ProjectState

DEFAULT_CONFIG = Config()

DEFAULT_INDEX_CONFIG = IndexConfig()

_CONFIG_FILE = Path("~/.archex/config.toml").expanduser()

_ENV_PREFIX = "ARCHEX_"

_BOOL_TRUE = {"1", "true", "yes", "on"}

DEFAULT_CACHE_DIR = "~/.archex/cache"


def _parse_env_value(key: str, value: str) -> Any:
    """Convert env string to appropriate Python type based on key suffix."""
    lower = key.lower()
    if lower.endswith("_size") or lower.endswith("_budget"):
        return int(value)
    if lower.endswith("_threshold"):
        try:
            return float(value)
        except ValueError:
            return value
    if value.lower() in _BOOL_TRUE:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data


def _config_overrides_from_mapping(data: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in data.items():
        if key not in Config.model_fields:
            continue
        if key == "languages" and value == []:
            overrides[key] = None
        else:
            overrides[key] = value
    return overrides


def _index_overrides_from_mapping(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k in IndexConfig.model_fields}


def _env_overrides(model_fields: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        config_key = env_key[len(_ENV_PREFIX) :].lower()
        if config_key in model_fields:
            overrides[config_key] = _parse_env_value(config_key, env_val)
    return overrides


def _project_state(source: RepoSource | str | Path | None) -> ProjectState | None:
    if source is None:
        return None
    if isinstance(source, RepoSource):
        if source.url or not source.local_path:
            return None
        path: str | Path = source.local_path
    else:
        path = source
    try:
        state = ProjectState.resolve(path)
    except ValueError:
        return None
    return state if state.settings_path.exists() else None


def _project_index_settings(
    source: RepoSource | str | Path | None,
) -> tuple[ProjectState, dict[str, Any]] | None:
    state = _project_state(source)
    if state is None:
        return None
    settings = _load_toml(state.settings_path)
    raw_index_settings = settings.get("index", {})
    if not isinstance(raw_index_settings, dict):
        raise ValueError(f"Invalid project index settings in {state.settings_path}")
    index_settings = cast("dict[str, Any]", raw_index_settings)
    return state, index_settings


def _resolve_project_path(value: Any, repo_root: Path) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(repo_root / path)


def load_config(source: RepoSource | str | Path | None = None) -> Config:
    """Load Config from TOML files and ARCHEX_* environment variables.

    Priority: env vars > project TOML > user TOML > defaults.
    """
    overrides: dict[str, Any] = {}

    if _CONFIG_FILE.exists():
        overrides.update(_config_overrides_from_mapping(_load_toml(_CONFIG_FILE)))

    project_settings = _project_index_settings(source)
    if project_settings is not None:
        state, index_settings = project_settings
        project_overrides = _config_overrides_from_mapping(index_settings)
        if "cache_dir" in project_overrides:
            project_overrides["cache_dir"] = _resolve_project_path(
                project_overrides["cache_dir"],
                state.repo_root,
            )
        overrides.update(project_overrides)

    overrides.update(_env_overrides(Config.model_fields))

    return Config(**overrides) if overrides else Config()


def load_index_config(source: RepoSource | str | Path | None = None) -> IndexConfig:
    """Load IndexConfig from project settings and ARCHEX_* environment variables."""
    overrides: dict[str, Any] = {}

    project_settings = _project_index_settings(source)
    if project_settings is not None:
        _state, index_settings = project_settings
        overrides.update(_index_overrides_from_mapping(index_settings))

    overrides.update(_env_overrides(IndexConfig.model_fields))

    return IndexConfig(**overrides) if overrides else IndexConfig()


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        items = cast("list[Any]", value)
        return "[" + ", ".join(_format_toml_value(item) for item in items) + "]"
    raise TypeError(f"Unsupported TOML value type for project index settings: {type(value)!r}")


def persist_project_index_settings(
    source: RepoSource | str | Path,
    updates: dict[str, Any],
) -> Path | None:
    """Persist supported IndexConfig keys into repo-local `.archex/settings.toml`."""
    state = _project_state(source)
    if state is None:
        return None
    filtered = {key: value for key, value in updates.items() if key in IndexConfig.model_fields}
    if not filtered:
        return state.settings_path

    # Validate existing settings and the merged index config before writing.
    _state, index_settings = _project_index_settings(source) or (state, {})
    IndexConfig(**(index_settings | filtered))

    lines = state.settings_path.read_text(encoding="utf-8").splitlines()
    section_start: int | None = None
    section_end = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[index]":
            section_start = idx
            continue
        if section_start is not None and idx > section_start and stripped.startswith("["):
            section_end = idx
            break

    if section_start is None:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("[index]")
        section_start = len(lines) - 1
        section_end = len(lines)

    remaining = dict(filtered)
    for idx in range(section_start + 1, section_end):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            lines[idx] = f"{key} = {_format_toml_value(remaining.pop(key))}"

    insert_at = section_end
    for key, value in remaining.items():
        lines.insert(insert_at, f"{key} = {_format_toml_value(value)}")
        insert_at += 1

    state.settings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state.settings_path
