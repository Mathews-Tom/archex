from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoreConfig:
    name: str
    debug: bool = False
    max_workers: int = 4
    tags: list[str] = field(default_factory=list)


def initialize(name: str, *, debug: bool = False) -> CoreConfig:
    config = CoreConfig(name=name, debug=debug)
    if debug:
        print(f"[core] initialized with config: {config}")
    return config
