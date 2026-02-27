from __future__ import annotations

from typing import TYPE_CHECKING

from utils import hash_password

if TYPE_CHECKING:
    from models import User


class AuthService:
    def __init__(self) -> None:
        self._sessions: dict[str, User] = {}

    def login(self, user: User, password: str) -> str:
        token = hash_password(f"{user.id}:{password}")
        self._sessions[token] = user
        return token

    def logout(self, token: str) -> None:
        self._sessions.pop(token, None)

    def verify_token(self, token: str) -> User | None:
        return self._sessions.get(token)
