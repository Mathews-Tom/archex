from __future__ import annotations

from .base import SortStrategy, T  # noqa: TC001
from .implementations import BubbleSort, MergeSort, QuickSort

STRATEGIES: dict[str, type[SortStrategy]] = {
    "bubble": BubbleSort,
    "merge": MergeSort,
    "quick": QuickSort,
}


class SortContext:
    """Context that delegates sorting to a pluggable strategy."""

    def __init__(self, strategy: SortStrategy | str = "quick") -> None:
        if isinstance(strategy, str):
            cls = STRATEGIES.get(strategy)
            if cls is None:
                raise ValueError(f"Unknown sort strategy: {strategy}")
            self._strategy = cls()
        else:
            self._strategy = strategy

    @property
    def strategy(self) -> SortStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: SortStrategy) -> None:
        self._strategy = value

    def sort(self, data: list[T], *, key: callable | None = None, reverse: bool = False) -> list[T]:
        return self._strategy.sort(data, key=key, reverse=reverse)

    def algorithm_name(self) -> str:
        return self._strategy.name()

    @staticmethod
    def available_strategies() -> list[str]:
        return list(STRATEGIES.keys())
