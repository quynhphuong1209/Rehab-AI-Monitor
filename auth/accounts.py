"""Pure account lookup helpers for authentication flows."""

from __future__ import annotations

import unicodedata
from typing import Any


DEFAULT_PATIENT_ROLE = "Bệnh nhân"


def normalize_auth_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.strip().split())


def find_user_key(users: dict[str, Any] | None, username: Any) -> str | None:
    """Find a username key without matching full names or other display fields."""
    users = users or {}
    wanted = normalize_auth_text(username)
    if not wanted:
        return None
    if wanted in users:
        return wanted
    wanted_fold = wanted.casefold()
    for key in users:
        if normalize_auth_text(key).casefold() == wanted_fold:
            return key
    return None


def find_user_key_by_email(users: dict[str, Any] | None, email: Any) -> str | None:
    users = users or {}
    wanted = normalize_auth_text(email).casefold()
    if not wanted:
        return None
    for key, record in users.items():
        if normalize_auth_text((record or {}).get("email")).casefold() == wanted:
            return key
    return None


def roles_match(stored_role: Any, selected_role: Any, *, default_role: str = DEFAULT_PATIENT_ROLE) -> bool:
    return normalize_auth_text(stored_role or default_role) == normalize_auth_text(selected_role or default_role)


def find_user_uniqueness_issues(users: dict[str, Any] | None) -> list[str]:
    users = users or {}
    seen_usernames: set[str] = set()
    seen_emails: dict[str, str] = {}
    issues: list[str] = []
    for username, record in users.items():
        username_key = normalize_auth_text(username).casefold()
        if username_key in seen_usernames:
            issues.append("duplicate username")
        seen_usernames.add(username_key)

        email_key = normalize_auth_text((record or {}).get("email")).casefold()
        if email_key:
            if email_key in seen_emails and seen_emails[email_key] != username:
                issues.append("duplicate email")
            seen_emails[email_key] = username
    return sorted(set(issues))
