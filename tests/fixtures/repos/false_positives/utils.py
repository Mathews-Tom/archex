class DataProcessor:
    """Utility class that has process/handle/get/all but is NOT a pattern."""

    def process(self, data: list) -> list:
        return [x * 2 for x in data]

    def handle(self, item: str) -> str:
        return item.upper()


class ConfigManager:
    """Has get and all methods but is NOT a repository."""

    def get(self, key: str) -> str:
        return ""

    def all(self) -> dict:
        return {}


class ButtonWidget:
    """Has an 'on' method but is NOT an event bus."""

    def on(self, event_name: str) -> None:
        pass
