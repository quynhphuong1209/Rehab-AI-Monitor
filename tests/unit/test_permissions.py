import pytest

from auth.permissions import (
    PATIENT_SCOPE_DENIED_MESSAGE,
    actor_can_access_patient,
    patient_display_label_for_actor,
    require_actor_patient_scope,
    require_actor_role,
    researcher_view_records_for_actor,
    scope_patient_usernames_for_actor,
    scope_records_for_actor,
)
from models.schemas import ADMIN_ROLE, DOCTOR_ROLE, PATIENT_ROLE, RESEARCHER_ROLE


def test_require_actor_role_allows_and_denies_by_role():
    actor = {"username": "doctor01", "role": DOCTOR_ROLE}

    assert require_actor_role(actor, [DOCTOR_ROLE]) is actor

    with pytest.raises(PermissionError):
        require_actor_role(actor, [ADMIN_ROLE])


def test_scope_records_for_actor_uses_user_assignments():
    users = {
        "doctor01": {"role": DOCTOR_ROLE, "assigned_patient_usernames": ["bn01"]},
        "bn01": {"role": PATIENT_ROLE},
        "bn02": {"role": PATIENT_ROLE},
    }
    records = [{"patient_username": "bn01"}, {"patient_username": "bn02"}]

    assert scope_records_for_actor(records, {"username": "doctor01"}, users) == [{"patient_username": "bn01"}]
    assert scope_patient_usernames_for_actor(["bn01", "bn02"], {"username": "doctor01"}, users) == {"bn01"}


def test_actor_can_access_patient_admin_bypasses_patient_scope():
    actor = {"username": "admin", "role": ADMIN_ROLE}

    assert actor_can_access_patient(actor, "bn01", users={})


def test_require_actor_patient_scope_reports_denied_scope():
    users = {
        "doctor01": {"role": DOCTOR_ROLE, "assigned_patient_usernames": ["bn01"]},
        "bn02": {"role": PATIENT_ROLE},
    }

    with pytest.raises(PermissionError, match=PATIENT_SCOPE_DENIED_MESSAGE):
        require_actor_patient_scope({"username": "doctor01", "role": DOCTOR_ROLE}, "bn02", users)


def test_researcher_view_and_label_use_pseudonyms():
    actor = {"username": "ncv01", "role": RESEARCHER_ROLE}
    records = [{"patient_username": "bn01", "full_name": "Private Name", "general_result": "Đúng"}]

    pseudonymized = researcher_view_records_for_actor(records, actor)

    assert pseudonymized[0]["patient_username"].startswith("SUBJ-")
    assert "full_name" not in pseudonymized[0]
    assert patient_display_label_for_actor(records[0], actor).startswith("SUBJ-")
