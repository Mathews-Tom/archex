from __future__ import annotations

from hub import shared_helper


def use_a(value: int) -> int:
    return shared_helper(value)
