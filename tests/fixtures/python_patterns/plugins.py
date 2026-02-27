from __future__ import annotations

from typing import Protocol


class Plugin(Protocol):
    name: str

    def execute(self, context: dict) -> dict: ...


class PluginRegistry:
    _plugins: dict[str, Plugin] = {}

    @classmethod
    def register(cls, plugin: Plugin) -> None:
        cls._plugins[plugin.name] = plugin

    @classmethod
    def get(cls, name: str) -> Plugin:
        if name not in cls._plugins:
            raise KeyError(f"Plugin '{name}' not registered")
        return cls._plugins[name]

    @classmethod
    def all(cls) -> list[Plugin]:
        return list(cls._plugins.values())
