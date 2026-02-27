from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

Handler = Callable[[dict], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event: str, handler: Handler) -> None:
        self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Handler) -> None:
        self._handlers[event].remove(handler)

    def emit(self, event: str, payload: dict) -> None:
        for handler in self._handlers[event]:
            handler(payload)
