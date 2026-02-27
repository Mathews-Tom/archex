from __future__ import annotations

from typing import Protocol


class SortStrategy(Protocol):
    def sort(self, data: list[int]) -> list[int]: ...


class BubbleSort:
    def sort(self, data: list[int]) -> list[int]:
        result = data[:]
        for i in range(len(result)):
            for j in range(len(result) - i - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
        return result


class QuickSort:
    def sort(self, data: list[int]) -> list[int]:
        if len(data) <= 1:
            return data[:]
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        mid = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + mid + self.sort(right)


class Sorter:
    def __init__(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def sort(self, data: list[int]) -> list[int]:
        return self._strategy.sort(data)
