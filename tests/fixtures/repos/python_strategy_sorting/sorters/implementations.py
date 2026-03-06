from __future__ import annotations

from .base import SortStrategy, T


class BubbleSort(SortStrategy):
    def sort(self, data: list[T], *, key: callable | None = None, reverse: bool = False) -> list[T]:
        result = list(data)
        n = len(result)
        for i in range(n):
            swapped = False
            for j in range(n - i - 1):
                if self.compare(result[j], result[j + 1], key=key, reverse=reverse) > 0:
                    result[j], result[j + 1] = result[j + 1], result[j]
                    swapped = True
            if not swapped:
                break
        return result

    def name(self) -> str:
        return "bubble_sort"


class MergeSort(SortStrategy):
    def sort(self, data: list[T], *, key: callable | None = None, reverse: bool = False) -> list[T]:
        if len(data) <= 1:
            return list(data)
        mid = len(data) // 2
        left = self.sort(data[:mid], key=key, reverse=reverse)
        right = self.sort(data[mid:], key=key, reverse=reverse)
        return self._merge(left, right, key=key, reverse=reverse)

    def _merge(
        self,
        left: list[T],
        right: list[T],
        *,
        key: callable | None,
        reverse: bool,
    ) -> list[T]:
        result: list[T] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if self.compare(left[i], right[j], key=key, reverse=reverse) <= 0:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def name(self) -> str:
        return "merge_sort"


class QuickSort(SortStrategy):
    def sort(self, data: list[T], *, key: callable | None = None, reverse: bool = False) -> list[T]:
        if len(data) <= 1:
            return list(data)
        pivot = data[len(data) // 2]
        left = [x for x in data if self.compare(x, pivot, key=key, reverse=reverse) < 0]
        middle = [x for x in data if self.compare(x, pivot, key=key, reverse=reverse) == 0]
        right = [x for x in data if self.compare(x, pivot, key=key, reverse=reverse) > 0]
        left_sorted = self.sort(left, key=key, reverse=reverse)
        right_sorted = self.sort(right, key=key, reverse=reverse)
        return left_sorted + middle + right_sorted

    def name(self) -> str:
        return "quick_sort"
