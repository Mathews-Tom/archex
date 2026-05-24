"""Tests for archex.config — load_config and _parse_env_value."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest

import archex.config as cfg_module
from archex.config import (  # pyright: ignore[reportPrivateUsage]
    _parse_env_value,
    load_config,
    load_index_config,
)
from archex.models import Config, IndexConfig, RepoSource
from archex.project import init_project

# ---------------------------------------------------------------------------
# _parse_env_value
# ---------------------------------------------------------------------------


def test_parse_env_value_size_suffix_returns_int() -> None:
    result = _parse_env_value("max_file_size", "5000000")
    assert result == 5_000_000
    assert isinstance(result, int)


def test_parse_env_value_budget_suffix_returns_int() -> None:
    result = _parse_env_value("token_budget", "1024")
    assert result == 1024
    assert isinstance(result, int)


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_parse_env_value_bool_true_variants(raw: str) -> None:
    assert _parse_env_value("cache", raw) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE", "No"])
def test_parse_env_value_bool_false_variants(raw: str) -> None:
    assert _parse_env_value("cache", raw) is False


def test_parse_env_value_string_passthrough() -> None:
    result = _parse_env_value("cache_dir", "/tmp/archex")
    assert result == "/tmp/archex"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# load_config — TOML
# ---------------------------------------------------------------------------


def test_load_config_toml_valid_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_file = tmp_path / "config.toml"
    toml_file.write_text('depth = "shallow"\ncache = false\n', encoding="utf-8")
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", toml_file)
    monkeypatch.delenv("ARCHEX_DEPTH", raising=False)
    monkeypatch.delenv("ARCHEX_CACHE", raising=False)

    config = load_config()

    assert config.depth == "shallow"
    assert config.cache is False


def test_load_config_toml_ignores_unknown_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toml_file = tmp_path / "config.toml"
    toml_file.write_text('depth = "shallow"\nunknown_key = "ignored"\n', encoding="utf-8")
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", toml_file)

    config = load_config()

    assert config.depth == "shallow"
    assert not hasattr(config, "unknown_key")


# ---------------------------------------------------------------------------
# load_config — env vars
# ---------------------------------------------------------------------------


def test_load_config_env_var_prefix_stripping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "nonexistent.toml")
    monkeypatch.setenv("ARCHEX_CACHE_DIR", "/tmp/archex-cache")

    config = load_config()

    assert config.cache_dir == "/tmp/archex-cache"


def test_load_config_env_var_size_coercion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "nonexistent.toml")
    monkeypatch.setenv("ARCHEX_MAX_FILE_SIZE", "999")

    config = load_config()

    assert config.max_file_size == 999
    assert isinstance(config.max_file_size, int)


def test_load_config_env_var_ignores_unknown_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "nonexistent.toml")
    monkeypatch.setenv("ARCHEX_TOTALLY_UNKNOWN", "value")

    config = load_config()

    assert config == Config()


def test_load_config_default_when_no_toml_no_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "nonexistent.toml")
    for key in list(k for k in __import__("os").environ if k.startswith("ARCHEX_")):
        monkeypatch.delenv(key, raising=False)

    config = load_config()

    assert config == Config()


def test_load_config_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_file = tmp_path / "config.toml"
    toml_file.write_text('depth = "shallow"\n', encoding="utf-8")
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", toml_file)
    monkeypatch.setenv("ARCHEX_DEPTH", "full")

    config = load_config()

    assert config.depth == "full"


def test_load_config_project_settings_override_user_defaults(
    python_simple_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_config = tmp_path / "config.toml"
    user_config.write_text(
        'cache_dir = "/tmp/global-cache"\nlanguages = ["go"]\n', encoding="utf-8"
    )
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", user_config)
    monkeypatch.delenv("ARCHEX_CACHE_DIR", raising=False)
    monkeypatch.delenv("ARCHEX_LANGUAGES", raising=False)
    init_project(python_simple_repo)

    config = load_config(RepoSource(local_path=str(python_simple_repo)))

    assert config.cache_dir == str(python_simple_repo / ".archex")
    assert config.languages == []
    assert config.delta_threshold == 0.5


def test_load_config_environment_overrides_project_settings(
    python_simple_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "missing.toml")
    monkeypatch.setenv("ARCHEX_CACHE_DIR", "/tmp/env-cache")
    init_project(python_simple_repo)

    config = load_config(python_simple_repo)

    assert config.cache_dir == "/tmp/env-cache"


def test_load_index_config_project_settings(
    python_simple_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "missing.toml")
    monkeypatch.delenv("ARCHEX_VECTOR", raising=False)
    init_project(python_simple_repo)

    index_config = load_index_config(python_simple_repo)

    assert index_config == IndexConfig(vector=False)


def test_load_index_config_environment_overrides_project_settings(
    python_simple_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "missing.toml")
    monkeypatch.setenv("ARCHEX_VECTOR", "true")
    init_project(python_simple_repo)

    index_config = load_index_config(python_simple_repo)

    assert index_config.vector is True


@pytest.mark.parametrize(
    ("env_key", "raw_value", "field", "expected"),
    [
        ("ARCHEX_CACHE", "false", "cache", False),
        ("ARCHEX_PARALLEL", "1", "parallel", True),
        ("ARCHEX_STRICT", "yes", "strict", True),
    ],
)
def test_load_config_bool_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_key: str,
    raw_value: str,
    field: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(cfg_module, "_CONFIG_FILE", tmp_path / "nonexistent.toml")
    monkeypatch.setenv(env_key, raw_value)

    config = load_config()

    assert getattr(config, field) == expected
