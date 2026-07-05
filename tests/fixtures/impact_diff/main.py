from __future__ import annotations

from consumer_a import use_a
from hub import other_helper


def run() -> None:
    print(use_a(1))
    print(other_helper(2))


if __name__ == "__main__":
    run()
