from __future__ import annotations


class Middleware:
    def __init__(self, name: str) -> None:
        self.name = name
        self._next: Middleware | None = None

    def set_next(self, middleware: Middleware) -> Middleware:
        self._next = middleware
        return middleware

    def process(self, request: dict) -> dict:
        if self._next is not None:
            return self._next.process(request)
        return request


class LoggingMiddleware(Middleware):
    def process(self, request: dict) -> dict:
        print(f"[{self.name}] processing {request}")
        return super().process(request)


class AuthMiddleware(Middleware):
    def process(self, request: dict) -> dict:
        if "token" not in request:
            raise PermissionError("Missing token")
        return super().process(request)
