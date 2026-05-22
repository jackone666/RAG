"""
Simple JSON file-based user store with pbkdf2 password hashing.
"""

import hashlib
import json
import os
from pathlib import Path

USER_STORE_PATH = Path("data/users.json")


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return dk.hex(), salt.hex()


def _load_users() -> dict:
    if not USER_STORE_PATH.exists():
        return {}
    return json.loads(USER_STORE_PATH.read_text())


def _save_users(users: dict) -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_STORE_PATH.write_text(json.dumps(users, indent=2))


def init_default_users() -> None:
    """Seed the user store with a default admin account if it doesn't exist."""
    if USER_STORE_PATH.exists():
        return

    pw_hash, salt = _hash_password("123")
    users = {
        "123": {
            "password_hash": pw_hash,
            "salt": salt,
            "tenant_id": "default",
            "role": "admin",
            "display_name": "Administrator",
        }
    }
    _save_users(users)


def authenticate_user(username: str, password: str) -> dict | None:
    """Validate credentials and return user info dict, or None."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return None

    pw_hash, _ = _hash_password(password, bytes.fromhex(user["salt"]))
    if pw_hash != user["password_hash"]:
        return None

    return {
        "username": username,
        "tenant_id": user.get("tenant_id", "default"),
        "role": user.get("role", "viewer"),
        "display_name": user.get("display_name", username),
    }
