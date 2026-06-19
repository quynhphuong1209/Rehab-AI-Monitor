"""Authentication helpers for the standalone backend API."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from auth.accounts import find_user_key, find_user_key_by_email, normalize_auth_text
from auth.passwords import password_record_update, verify_password_record
from models.schemas import PATIENT_ROLE
from storage.app_json import update_app_json

from backend.repository import JsonRepository


class RegistrationError(ValueError):
    """Raised when a self-service account registration is not acceptable."""

    status_code: int

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def public_user_record(username: str, record: dict[str, Any]) -> dict[str, Any]:
    """Return user fields that are safe to send to a client."""
    return {
        "username": username,
        "full_name": record.get("full_name") or username,
        "role": record.get("role") or "",
        "email": record.get("email") or "",
        "must_change_password": bool(record.get("must_change_password")),
        "active": bool(record.get("active", True)),
    }


@dataclass
class TokenStore:
    _tokens: dict[str, dict[str, Any]] = field(default_factory=dict)

    def issue(self, actor: dict[str, Any]) -> str:
        token = secrets.token_urlsafe(32)
        self._tokens[token] = dict(actor)
        return token

    def get(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        actor = self._tokens.get(token)
        return dict(actor) if actor else None

    def revoke(self, token: str | None) -> bool:
        if not token:
            return False
        return self._tokens.pop(token, None) is not None

    def revoke_actor(self, username: str) -> int:
        target = normalize_auth_text(username).casefold()
        if not target:
            return 0
        revoked = [
            token
            for token, actor in self._tokens.items()
            if normalize_auth_text(actor.get("username")).casefold() == target
        ]
        for token in revoked:
            self._tokens.pop(token, None)
        return len(revoked)

    def revoke_all(self) -> int:
        count = len(self._tokens)
        self._tokens.clear()
        return count

    def replace(self, token: str | None, actor: dict[str, Any]) -> bool:
        if not token or token not in self._tokens:
            return False
        self._tokens[token] = dict(actor)
        return True


def authenticate_user(
    repo: JsonRepository,
    username: str,
    password: str,
) -> dict[str, Any] | None:
    users = repo.users()
    user_key = find_user_key(users, username)
    if not user_key:
        return None
    record = users.get(user_key) or {}
    if record.get("active") is False:
        return None
    verification = verify_password_record(password, record)
    if not verification.ok:
        return None
    return public_user_record(user_key, record)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def register_patient_user(
    repo: JsonRepository,
    *,
    username: str,
    password: str,
    confirm_password: str,
    full_name: str = "",
    email: str = "",
) -> dict[str, Any]:
    """Register a self-service patient account and return a public user."""
    username_clean = normalize_auth_text(username)
    email_clean = normalize_auth_text(email)
    full_name_clean = normalize_auth_text(full_name) or username_clean
    password = str(password or "")
    confirm_password = str(confirm_password or "")

    if not username_clean or not email_clean or not password:
        raise RegistrationError("missing required registration fields")
    if len(username_clean) < 3:
        raise RegistrationError("username must be at least 3 characters")
    if len(password) < 6:
        raise RegistrationError("password must be at least 6 characters")
    if password != confirm_password:
        raise RegistrationError("password confirmation does not match")

    created_user: dict[str, Any] | None = None

    def _add_user(current: Any) -> dict[str, Any]:
        nonlocal created_user
        users = current if isinstance(current, dict) else {}
        if find_user_key(users, username_clean):
            raise RegistrationError("username already exists", 409)
        if find_user_key_by_email(users, email_clean):
            raise RegistrationError("email already exists", 409)

        now = _utc_now_iso()
        record = {
            **password_record_update(password, updated_at=now, must_change_password=False),
            "username": username_clean,
            "full_name": full_name_clean,
            "email": email_clean,
            "role": PATIENT_ROLE,
            "created_at": now,
            "active": True,
            "assigned_patient_usernames": [],
            "assigned_doctor_username": "",
            "team_usernames": [],
        }
        updated = dict(users)
        updated[username_clean] = record
        created_user = public_user_record(username_clean, record)
        return updated

    update_app_json(repo.config.users_file, _add_user, default={})
    if not created_user:
        raise RegistrationError("registration failed")
    return created_user


def bearer_token_from_header(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()
