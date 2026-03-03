"""Default configuration values and config loading utilities."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from archex.models import Config, IndexConfig

DEFAULT_CONFIG = Config()

DEFAULT_INDEX_CONFIG = IndexConfig()

_CONFIG_FILE = Path("~/.archex/config.toml").expanduser()

_ENV_PREFIX = "ARCHEX_"

_BOOL_TRUE = {"1", "true", "yes", "on"}

DEFAULT_CACHE_DIR = "~/.archex/cache"

# Default model IDs for LLM providers — single source of truth
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4.1",
    "openrouter": "anthropic/claude-sonnet-4-20250514",
}


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


def load_config() -> Config:
    """Load Config from TOML file and/or ARCHEX_* environment variables.

    Priority: env vars > TOML file > defaults.
    """
    overrides: dict[str, Any] = {}

    # Load from TOML if present
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE, "rb") as f:
            toml_data = tomllib.load(f)
        # Flatten top-level keys into overrides
        for k, v in toml_data.items():
            if k in Config.model_fields:
                overrides[k] = v

    # Environment variables override TOML
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        config_key = env_key[len(_ENV_PREFIX) :].lower()
        if config_key in Config.model_fields:
            overrides[config_key] = _parse_env_value(config_key, env_val)

    return Config(**overrides) if overrides else Config()
