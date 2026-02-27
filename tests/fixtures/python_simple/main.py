from __future__ import annotations

from models import Role, User
from services.auth import AuthService
from utils import validate_email


def run() -> None:
    service = AuthService()
    user = User(id=1, name="Alice", email="alice@example.com", role=Role.ADMIN)
    if not validate_email(user.email):
        raise ValueError(f"Invalid email: {user.email}")
    token = service.login(user, "secret")
    verified = service.verify_token(token)
    print(f"Logged in: {verified}")


if __name__ == "__main__":
    run()
