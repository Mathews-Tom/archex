from __future__ import annotations

from hub import shared_helper


def isolated(value: int) -> int:
    return shared_helper(value) - 1
