from __future__ import annotations

import enum


class Role(enum.Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class User:
    def __init__(self, id: int, name: str, email: str, role: Role = Role.USER) -> None:
        self.id = id
        self.name = name
        self.email = email
        self.role = role

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, role={self.role!r})"
