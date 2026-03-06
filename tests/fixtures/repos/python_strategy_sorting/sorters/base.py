from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T")


class SortStrategy(ABC):
    """Base class for all sorting strategy implementations."""

    @abstractmethod
    def sort(self, data: list[T], *, key: callable | None = None, reverse: bool = False) -> list[T]:
        """Sort the data using this strategy's algorithm."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the name of this sorting algorithm."""
        ...

    def compare(self, a: T, b: T, *, key: callable | None = None, reverse: bool = False) -> int:
        """Compare two elements, returning negative, zero, or positive."""
        va = key(a) if key else a
        vb = key(b) if key else b
        result = (va > vb) - (va < vb)
        return -result if reverse else result
