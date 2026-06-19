from models.schemas import (
    DOCTOR_ROLE,
    PATIENT_ROLE,
    RESEARCHER_ROLE,
    filter_records_for_actor,
    normalize_evaluations,
    normalize_patient_symptoms,
    normalize_schedules,
    normalize_users,
    normalize_video_list,
    pseudonymize_records,
)


def test_video_schema_fills_missing_optional_fields_and_skips_broken_rows():
    result = normalize_video_list([
        {"username": "bn01", "video_name": "clip.mp4"},
        "broken",
    ])

    assert result.changed
    assert len(result.data) == 1
    assert result.data[0]["full_name"] == "bn01"
    assert result.data[0]["metrics"] == {}
    assert result.issues


def test_users_schema_adds_assignment_fields():
    result = normalize_users({"doctor01": {"role": DOCTOR_ROLE, "assigned_patient_usernames": "bn01,bn02"}})

    assert result.data["doctor01"]["assigned_patient_usernames"] == ["bn01", "bn02"]
    assert result.data["doctor01"]["active"] is True


def test_evaluation_schedule_symptom_roots_reject_wrong_type():
    assert normalize_evaluations({}).data == []
    assert normalize_schedules({}).data == []
    assert normalize_patient_symptoms({}).data == []


def test_filter_records_for_actor_uses_doctor_assignments():
    users = {
        "doctor01": {"role": DOCTOR_ROLE, "assigned_patient_usernames": ["bn01"]},
        "bn01": {"role": PATIENT_ROLE},
        "bn02": {"role": PATIENT_ROLE},
    }
    records = [{"patient_username": "bn01"}, {"patient_username": "bn02"}]

    scoped = filter_records_for_actor(records, {"username": "doctor01"}, users)

    assert scoped == [{"patient_username": "bn01"}]


def test_filter_records_for_actor_uses_patient_assigned_doctor():
    users = {
        "doctor01": {"role": DOCTOR_ROLE},
        "bn01": {"role": PATIENT_ROLE, "assigned_doctor_username": "doctor01"},
        "bn02": {"role": PATIENT_ROLE, "assigned_doctor_username": "doctor02"},
    }
    records = [{"patient_username": "bn01"}, {"patient_username": "bn02"}]

    scoped = filter_records_for_actor(records, {"username": "doctor01"}, users)

    assert scoped == [{"patient_username": "bn01"}]


def test_researcher_pseudonymization_removes_direct_identifiers():
    records = [
        {
            "patient_username": "bn01",
            "full_name": "Patient One",
            "comments": "private note",
            "subject_code": "REAL-ID-001",
            "general_result": "Đúng",
        }
    ]

    pseudonymized = pseudonymize_records(records)

    assert pseudonymized[0]["patient_username"].startswith("SUBJ-")
    assert pseudonymized[0]["subject_code"].startswith("SUBJ-")
    assert pseudonymized[0]["subject_code"] != "REAL-ID-001"
    assert "full_name" not in pseudonymized[0]
    assert "comments" not in pseudonymized[0]
    assert pseudonymized[0]["general_result"] == "Đúng"
