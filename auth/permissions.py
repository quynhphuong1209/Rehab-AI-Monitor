"""Role and patient-scope helpers.

This module intentionally avoids Streamlit and storage access. Callers provide
the current actor and user directory, then handle UI feedback/audit logging at
the app boundary.
"""

from __future__ import annotations

from typing import Any, Iterable

from models.schemas import (
    ADMIN_ROLE,
    RESEARCHER_ROLE,
    filter_records_for_actor,
    pseudonymize_records,
)


PERMISSION_DENIED_MESSAGE = "Bạn không có quyền thực hiện thao tác này."
MISSING_PATIENT_MESSAGE = "Thiếu bệnh nhân đích cho thao tác này."
PATIENT_SCOPE_DENIED_MESSAGE = "Bạn không có quyền thao tác trên bệnh nhân này."


def actor_has_role(actor: dict[str, Any] | None, allowed_roles: Iterable[str]) -> bool:
    actor = actor or {}
    return actor.get("role") in set(allowed_roles or [])


def require_actor_role(actor: dict[str, Any], allowed_roles: Iterable[str]) -> dict[str, Any]:
    if not actor_has_role(actor, allowed_roles):
        raise PermissionError(PERMISSION_DENIED_MESSAGE)
    return actor


def scope_records_for_actor(
    records: list[dict[str, Any]] | None,
    actor: dict[str, Any] | None,
    users: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return filter_records_for_actor(records or [], actor or {}, users or {})


def scope_patient_usernames_for_actor(
    usernames: Iterable[str] | None,
    actor: dict[str, Any] | None,
    users: dict[str, Any] | None = None,
) -> set[str]:
    records = [{"patient_username": username} for username in usernames or [] if username]
    scoped = scope_records_for_actor(records, actor, users)
    return {record.get("patient_username") for record in scoped if record.get("patient_username")}


def actor_can_access_patient(
    actor: dict[str, Any] | None,
    patient_username: str | None,
    users: dict[str, Any] | None = None,
) -> bool:
    if not patient_username:
        return False
    actor = actor or {}
    if actor.get("role") == ADMIN_ROLE:
        return True
    scoped = scope_records_for_actor([{"patient_username": patient_username}], actor, users)
    return bool(scoped)


def require_actor_patient_scope(
    actor: dict[str, Any],
    patient_username: str | None,
    users: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not patient_username:
        raise PermissionError(MISSING_PATIENT_MESSAGE)
    if not actor_can_access_patient(actor, patient_username, users):
        raise PermissionError(PATIENT_SCOPE_DENIED_MESSAGE)
    return actor


def researcher_view_records_for_actor(
    records: list[dict[str, Any]] | None,
    actor: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if (actor or {}).get("role") == RESEARCHER_ROLE:
        return pseudonymize_records(records or [])
    return records or []


def patient_display_label_for_actor(
    record: dict[str, Any] | None,
    actor: dict[str, Any] | None,
    *,
    include_username: bool = True,
) -> str:
    record = record or {}
    username = record.get("username") or record.get("patient_username") or record.get("subject_code") or ""
    if (actor or {}).get("role") == RESEARCHER_ROLE:
        pseudonymized = pseudonymize_records([{"patient_username": username}])[0] if username else {}
        return pseudonymized.get("subject_code", "SUBJ-UNKNOWN")
    full_name = record.get("full_name") or record.get("patient_name") or username or "Không rõ"
    if include_username and username and full_name != username:
        return f"{full_name} ({username})"
    return full_name
