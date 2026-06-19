"""Backend response scoping helpers.

These helpers keep authorization and response shaping out of route handlers.
They also avoid importing Streamlit or the legacy `app.py` module.
"""

from __future__ import annotations

from typing import Any

from auth.permissions import (
    actor_can_access_patient,
    researcher_view_records_for_actor,
    scope_records_for_actor,
)
from models.schemas import (
    ADMIN_ROLE,
    DOCTOR_ROLE,
    PATIENT_ROLE,
    RESEARCHER_ROLE,
    pseudonymize_record,
)


def scoped_records_for_response(
    records: list[dict[str, Any]] | None,
    actor: dict[str, Any],
    users: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return records visible to actor, with researcher PII removed."""
    visible = scope_records_for_actor(records or [], actor, users or {})
    return researcher_view_records_for_actor(visible, actor)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def public_patient_record(
    username: str,
    record: dict[str, Any],
    actor: dict[str, Any],
) -> dict[str, Any]:
    """Shape a patient user record for the current API actor."""
    base = {
        "username": username,
        "full_name": record.get("full_name") or username,
        "role": record.get("role") or PATIENT_ROLE,
        "email": record.get("email") or "",
        "active": bool(record.get("active", True)),
        "assigned_doctor_username": record.get("assigned_doctor_username") or "",
        "assigned_patient_usernames": _as_list(record.get("assigned_patient_usernames")),
        "team_usernames": _as_list(record.get("team_usernames")),
    }

    role = actor.get("role")
    if role == RESEARCHER_ROLE:
        pseudonymized = pseudonymize_record(base)
        return {
            "subject_code": pseudonymized.get("subject_code", "SUBJ-UNKNOWN"),
            "username": pseudonymized.get("username", "SUBJ-UNKNOWN"),
            "role": base["role"],
            "active": base["active"],
        }

    if role == ADMIN_ROLE:
        return base

    if role == DOCTOR_ROLE:
        return {
            "username": base["username"],
            "full_name": base["full_name"],
            "role": base["role"],
            "active": base["active"],
            "assigned_doctor_username": base["assigned_doctor_username"],
        }

    return {
        "username": base["username"],
        "full_name": base["full_name"],
        "role": base["role"],
        "email": base["email"],
        "active": base["active"],
        "assigned_doctor_username": base["assigned_doctor_username"],
    }


def patient_records_for_actor(
    users: dict[str, Any] | None,
    actor: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return patient user records visible to actor without password fields."""
    users = users or {}
    visible: list[dict[str, Any]] = []
    for username, record in users.items():
        if not isinstance(record, dict) or record.get("role") != PATIENT_ROLE:
            continue
        if actor_can_access_patient(actor, username, users):
            visible.append(public_patient_record(username, record, actor))
    return visible
