"""Tests for StrategyRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from archex.benchmark.models import BenchmarkResult, BenchmarkTask, Strategy
from archex.benchmark.strategies import StrategyRegistry, default_strategy_registry
from archex.exceptions import ConfigError


def _dummy_runner(task: BenchmarkTask, repo_path: Path) -> BenchmarkResult:
    raise NotImplementedError


class TestStrategyRegistry:
    def test_register_and_get(self) -> None:
        reg = StrategyRegistry()
        reg.register("test_strat", _dummy_runner)
        assert reg.get("test_strat") is _dummy_runner

    def test_get_unknown_returns_none(self) -> None:
        reg = StrategyRegistry()
        assert reg.get("nonexistent") is None

    def test_get_with_strategy_enum(self) -> None:
        reg = StrategyRegistry()
        reg.register(Strategy.RAW_FILES.value, _dummy_runner)
        assert reg.get(Strategy.RAW_FILES) is _dummy_runner

    def test_strategy_names_reflects_registered(self) -> None:
        reg = StrategyRegistry()
        reg.register("b_strat", _dummy_runner)
        reg.register("a_strat", _dummy_runner)
        assert reg.strategy_names == ["a_strat", "b_strat"]

    def test_default_registry_has_builtin_strategies(self) -> None:
        names = default_strategy_registry.strategy_names
        assert Strategy.RAW_FILES.value in names
        assert Strategy.RAW_GREPPED.value in names
        assert Strategy.ARCHEX_QUERY.value in names
        assert Strategy.ARCHEX_QUERY_VECTOR.value in names
        assert Strategy.ARCHEX_QUERY_FUSION.value in names
        assert Strategy.ARCHEX_SYMBOL_LOOKUP.value in names

    def test_load_entry_points_registers_runner(self) -> None:
        reg = StrategyRegistry()
        mock_ep = MagicMock()
        mock_ep.name = "custom_strat"
        mock_ep.load.return_value = _dummy_runner
        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            reg.load_entry_points(group="archex.benchmark_strategies")
        assert reg.get("custom_strat") is _dummy_runner

    def test_load_entry_points_failure_logged(self) -> None:
        reg = StrategyRegistry()
        mock_ep = MagicMock()
        mock_ep.name = "bad_strat"
        mock_ep.load.side_effect = ImportError("no module")
        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            reg.load_entry_points(group="archex.benchmark_strategies")
        assert reg.get("bad_strat") is None

    def test_load_entry_points_strict_raises_config_error(self) -> None:
        reg = StrategyRegistry()
        mock_ep = MagicMock()
        mock_ep.name = "bad_strat"
        mock_ep.load.side_effect = ImportError("no module")
        with (
            patch("importlib.metadata.entry_points", return_value=[mock_ep]),
            pytest.raises(ConfigError, match="bad_strat"),
        ):
            reg.load_entry_points(group="archex.benchmark_strategies", strict=True)

    def test_idempotent_load_entry_points(self) -> None:
        reg = StrategyRegistry()
        call_count = 0

        def _mock_eps(group: str = "") -> list[MagicMock]:
            nonlocal call_count
            call_count += 1
            return []

        with patch("importlib.metadata.entry_points", side_effect=_mock_eps):
            reg.load_entry_points()
            reg.load_entry_points()
        assert call_count == 1
