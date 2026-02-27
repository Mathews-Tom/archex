from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    email: str


class UserRepository:
    def __init__(self) -> None:
        self._store: dict[int, User] = {}

    def get(self, id: int) -> User | None:
        return self._store.get(id)

    def save(self, user: User) -> None:
        self._store[user.id] = user

    def delete(self, id: int) -> None:
        self._store.pop(id, None)

    def all(self) -> list[User]:
        return list(self._store.values())
