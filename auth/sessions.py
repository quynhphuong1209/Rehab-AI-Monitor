"""Global session version storage for coarse session revocation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from storage.json_store import read_json, write_json


DEFAULT_SESSION_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_session_state(path: str) -> dict[str, Any]:
    data = read_json(path, {})
    if not isinstance(data, dict):
        data = {}
    version = data.get("global_session_version")
    if not isinstance(version, int) or version < DEFAULT_SESSION_VERSION:
        version = DEFAULT_SESSION_VERSION
        data["global_session_version"] = version
        data.setdefault("updated_at", _now_iso())
        write_json(path, data)
    return data


def get_global_session_version(path: str) -> int:
    return int(read_session_state(path).get("global_session_version", DEFAULT_SESSION_VERSION))


def bump_global_session_version(path: str, *, actor: str = "system", reason: str = "") -> int:
    data = read_session_state(path)
    current = int(data.get("global_session_version", DEFAULT_SESSION_VERSION))
    data.update(
        {
            "global_session_version": current + 1,
            "updated_at": _now_iso(),
            "updated_by": actor,
            "reason": reason,
        }
    )
    write_json(path, data)
    return current + 1


def session_is_current(path: str, session_version: int | None) -> bool:
    if session_version is None:
        return False
    try:
        return int(session_version) == get_global_session_version(path)
    except (TypeError, ValueError):
        return False
