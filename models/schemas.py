"""Schema normalization for JSON-backed app data.

The app still accepts legacy JSON files, so these validators are intentionally
conservative: reject malformed roots, skip severely broken rows, and fill safe
defaults for optional fields without dropping unknown metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Callable


PATIENT_ROLE = "Bệnh nhân"
DOCTOR_ROLE = "Bác sĩ / KTV PHCN"
RESEARCHER_ROLE = "Nghiên cứu viên"
ADMIN_ROLE = "Quản trị viên"
AI_RESEARCHER = "AI_Researcher"


@dataclass
class ValidationIssue:
    path: str
    message: str
    severity: str = "warning"


@dataclass
class ValidationResult:
    data: Any
    issues: list[ValidationIssue] = field(default_factory=list)
    changed: bool = False


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "co", "có"}:
            return True
        if lowered in {"0", "false", "no", "n", "khong", "không"}:
            return False
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _copy_with_defaults(record: dict[str, Any], defaults: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    out = dict(record)
    changed = False
    for key, default in defaults.items():
        if key not in out or out[key] is None:
            out[key] = default() if callable(default) else default
            changed = True
    return out, changed


def normalize_user_record(username: str, record: Any) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, dict):
        issues.append(ValidationIssue(f"users.{username}", "record is not an object", "error"))
        return None, issues, False

    normalized, changed = _copy_with_defaults(
        record,
        {
            "username": username,
            "full_name": username,
            "email": "",
            "role": PATIENT_ROLE,
            "must_change_password": False,
            "created_at": "",
            "updated_at": "",
            "hash_version": "",
            "assigned_patient_usernames": list,
            "assigned_doctor_username": "",
            "team_usernames": list,
            "active": True,
        },
    )
    if not _as_str(normalized.get("username")):
        normalized["username"] = username
        changed = True
    if normalized.get("username") != username:
        normalized["username"] = username
        changed = True

    normalized["full_name"] = _as_str(normalized.get("full_name"), username)
    normalized["email"] = _as_str(normalized.get("email"), "")
    normalized["role"] = _as_str(normalized.get("role"), PATIENT_ROLE)
    normalized["must_change_password"] = _as_bool(normalized.get("must_change_password"), False)
    normalized["active"] = _as_bool(normalized.get("active"), True)

    assigned = normalized.get("assigned_patient_usernames")
    if isinstance(assigned, str):
        assigned = [part.strip() for part in assigned.split(",") if part.strip()]
        changed = True
    elif not isinstance(assigned, list):
        assigned = []
        changed = True
    normalized["assigned_patient_usernames"] = [str(item).strip() for item in assigned if str(item).strip()]

    team = normalized.get("team_usernames")
    if isinstance(team, str):
        team = [part.strip() for part in team.split(",") if part.strip()]
        changed = True
    elif not isinstance(team, list):
        team = []
        changed = True
    normalized["team_usernames"] = [str(item).strip() for item in team if str(item).strip()]
    normalized["assigned_doctor_username"] = _as_str(normalized.get("assigned_doctor_username"), "")
    return normalized, issues, changed


def normalize_users(root: Any) -> ValidationResult:
    if not isinstance(root, dict):
        return ValidationResult({}, [ValidationIssue("users", "root is not an object", "error")], True)
    out: dict[str, Any] = {}
    issues: list[ValidationIssue] = []
    changed = False
    for username, record in root.items():
        key = _as_str(username)
        if not key:
            issues.append(ValidationIssue("users", "empty username skipped", "error"))
            changed = True
            continue
        normalized, row_issues, row_changed = normalize_user_record(key, record)
        issues.extend(row_issues)
        if normalized is None:
            changed = True
            continue
        out[key] = normalized
        changed = changed or row_changed or normalized != record
    return ValidationResult(out, issues, changed)


def normalize_video_record(record: Any, index: int = 0) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, dict):
        issues.append(ValidationIssue(f"videos[{index}]", "record is not an object", "error"))
        return None, issues, False

    normalized, changed = _copy_with_defaults(
        record,
        {
            "username": "",
            "full_name": "",
            "video_name": "",
            "exercise": "",
            "accuracy": 0.0,
            "time": "",
            "video_path": "",
            "processed_path": None,
            "status": "Chờ xử lý",
            "metrics": dict,
            "df_path": None,
            "frames_zip_path": None,
            "all_frames_data_path": None,
            "original_filename": "",
            "stored_filename": "",
        },
    )
    normalized["username"] = _as_str(normalized.get("username"))
    normalized["full_name"] = _as_str(normalized.get("full_name"), normalized["username"] or "Không rõ")
    normalized["video_name"] = _as_str(normalized.get("video_name"))
    normalized["exercise"] = _as_str(normalized.get("exercise"), "Không rõ")
    normalized["status"] = _as_str(normalized.get("status"), "Chờ xử lý")
    normalized["time"] = _as_str(normalized.get("time"))
    normalized["accuracy"] = _as_float(normalized.get("accuracy"), 0.0)
    if not isinstance(normalized.get("metrics"), dict):
        normalized["metrics"] = {}
        changed = True
    if not normalized["username"] and not normalized["video_name"]:
        issues.append(ValidationIssue(f"videos[{index}]", "missing username and video_name", "error"))
        return None, issues, True
    return normalized, issues, changed or normalized != record


def normalize_video_list(root: Any) -> ValidationResult:
    if not isinstance(root, list):
        return ValidationResult([], [ValidationIssue("videos", "root is not a list", "error")], True)
    return _normalize_list(root, normalize_video_record)


def normalize_evaluation_record(record: Any, index: int = 0) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, dict):
        issues.append(ValidationIssue(f"evaluations[{index}]", "record is not an object", "error"))
        return None, issues, False
    normalized, changed = _copy_with_defaults(
        record,
        {
            "patient_username": "",
            "doctor_username": "",
            "doctor_name": "",
            "video_name": "",
            "exercise": "",
            "doctor_result": "",
            "errors": list,
            "comments": "",
            "comments_ncv": "",
            "plan": "",
            "time": "",
        },
    )
    for key in ("patient_username", "doctor_username", "doctor_name", "video_name", "exercise", "doctor_result", "comments", "comments_ncv", "plan", "time"):
        normalized[key] = _as_str(normalized.get(key))
    if not isinstance(normalized.get("errors"), list):
        normalized["errors"] = []
        changed = True
    normalized["errors"] = [str(item) for item in normalized["errors"] if str(item).strip()]
    if not normalized["patient_username"] and not normalized["video_name"]:
        issues.append(ValidationIssue(f"evaluations[{index}]", "missing patient_username and video_name", "error"))
        return None, issues, True
    return normalized, issues, changed or normalized != record


def normalize_evaluations(root: Any) -> ValidationResult:
    if not isinstance(root, list):
        return ValidationResult([], [ValidationIssue("evaluations", "root is not a list", "error")], True)
    return _normalize_list(root, normalize_evaluation_record)


def normalize_schedule_record(record: Any, index: int = 0) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, dict):
        issues.append(ValidationIssue(f"schedules[{index}]", "record is not an object", "error"))
        return None, issues, False
    normalized, changed = _copy_with_defaults(
        record,
        {
            "id": lambda: index,
            "type": "appointment",
            "title": "",
            "datetime": "",
            "notes": "",
            "patient_username": "",
            "patient_name": "",
            "doctor_username": "",
            "doctor_name": "",
            "exercise_name": "",
            "frequency": "",
            "medication_name": "",
            "dosage": "",
            "taken": False,
        },
    )
    normalized["type"] = _as_str(normalized.get("type"), "appointment")
    if normalized["type"] not in {"appointment", "exercise", "medication"}:
        issues.append(ValidationIssue(f"schedules[{index}].type", "unknown schedule type"))
        normalized["type"] = "appointment"
        changed = True
    for key in ("title", "datetime", "notes", "patient_username", "patient_name", "doctor_username", "doctor_name", "exercise_name", "frequency", "medication_name", "dosage"):
        normalized[key] = _as_str(normalized.get(key))
    normalized["taken"] = _as_bool(normalized.get("taken"), False)
    if normalized.get("id") in (None, ""):
        normalized["id"] = index
        changed = True
    return normalized, issues, changed or normalized != record


def normalize_schedules(root: Any) -> ValidationResult:
    if not isinstance(root, list):
        return ValidationResult([], [ValidationIssue("schedules", "root is not a list", "error")], True)
    return _normalize_list(root, normalize_schedule_record)


def normalize_patient_symptom_record(record: Any, index: int = 0) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, dict):
        issues.append(ValidationIssue(f"symptoms[{index}]", "record is not an object", "error"))
        return None, issues, False
    normalized, changed = _copy_with_defaults(
        record,
        {
            "username": "",
            "full_name": "",
            "age": 0,
            "gender": "",
            "patient_id": "",
            "symptoms": "",
            "vas": 0,
            "time": "",
            "exercises": list,
            "exercise": "",
        },
    )
    for key in ("username", "full_name", "gender", "patient_id", "symptoms", "time", "exercise"):
        normalized[key] = _as_str(normalized.get(key))
    normalized["age"] = _as_int(normalized.get("age"), 0)
    normalized["vas"] = max(0, min(10, _as_int(normalized.get("vas"), 0)))
    if not isinstance(normalized.get("exercises"), list):
        normalized["exercises"] = []
        changed = True
    if not normalized["patient_id"]:
        normalized["patient_id"] = normalized["username"]
        changed = True
    if not normalized["username"] and not normalized["patient_id"]:
        issues.append(ValidationIssue(f"symptoms[{index}]", "missing username/patient_id", "error"))
        return None, issues, True
    return normalized, issues, changed or normalized != record


def normalize_patient_symptoms(root: Any) -> ValidationResult:
    if not isinstance(root, list):
        return ValidationResult([], [ValidationIssue("symptoms", "root is not a list", "error")], True)
    return _normalize_list(root, normalize_patient_symptom_record)


def normalize_research_record(record: Any, index: int = 0) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, dict):
        issues.append(ValidationIssue(f"research[{index}]", "record is not an object", "error"))
        return None, issues, False
    normalized, changed = _copy_with_defaults(
        record,
        {
            "patient_username": "",
            "subject_code": "",
            "interviewer": "",
            "timestamp": "",
            "exercises": list,
            "general_result": "",
            "specialist_comment": "",
            "role": "",
            "submitted_by": "",
        },
    )
    for key in ("patient_username", "subject_code", "interviewer", "timestamp", "general_result", "specialist_comment", "role", "submitted_by"):
        normalized[key] = _as_str(normalized.get(key))
    if not isinstance(normalized.get("exercises"), list):
        normalized["exercises"] = []
        changed = True
    if not normalized["patient_username"] and not normalized["subject_code"]:
        issues.append(ValidationIssue(f"research[{index}]", "missing patient_username/subject_code", "error"))
        return None, issues, True
    return normalized, issues, changed or normalized != record


def normalize_research_data(root: Any) -> ValidationResult:
    if not isinstance(root, list):
        return ValidationResult([], [ValidationIssue("research", "root is not a list", "error")], True)
    return _normalize_list(root, normalize_research_record)


def normalize_history_record(record: Any, index: int = 0) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, dict):
        issues.append(ValidationIssue(f"history[{index}]", "record is not an object", "error"))
        return None, issues, False
    normalized, changed = _copy_with_defaults(
        record,
        {"username": "", "full_name": "", "bai_tap": "", "accuracy": 0.0, "ngay": ""},
    )
    for key in ("username", "full_name", "bai_tap", "ngay"):
        normalized[key] = _as_str(normalized.get(key))
    normalized["accuracy"] = _as_float(normalized.get("accuracy"), 0.0)
    return normalized, issues, changed or normalized != record


def normalize_history(root: Any) -> ValidationResult:
    if not isinstance(root, list):
        return ValidationResult([], [ValidationIssue("history", "root is not a list", "error")], True)
    return _normalize_list(root, normalize_history_record)


def _normalize_list(root: list[Any], normalizer: Callable[[Any, int], tuple[dict[str, Any] | None, list[ValidationIssue], bool]]) -> ValidationResult:
    out: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    changed = False
    for index, record in enumerate(root):
        normalized, row_issues, row_changed = normalizer(record, index)
        issues.extend(row_issues)
        if normalized is None:
            changed = True
            continue
        out.append(normalized)
        changed = changed or row_changed
    return ValidationResult(out, issues, changed)


SCHEMA_NORMALIZERS: dict[str, Callable[[Any], ValidationResult]] = {
    "users.json": normalize_users,
    "video_list.json": normalize_video_list,
    "doctor_evaluations.json": normalize_evaluations,
    "schedules.json": normalize_schedules,
    "patient_symptoms.json": normalize_patient_symptoms,
    "research_data.json": normalize_research_data,
    "lich_su_tap_luyen.json": normalize_history,
}


def normalize_json_data(file_path: str | Path, data: Any) -> ValidationResult:
    normalizer = SCHEMA_NORMALIZERS.get(Path(file_path).name)
    if not normalizer:
        return ValidationResult(data, [], False)
    return normalizer(data)


def migrate_json_file(path: str | Path, *, read_json: Callable[[str | Path, Any], Any], write_json: Callable[[str | Path, Any], bool]) -> ValidationResult:
    file_path = Path(path)
    default: Any = {} if file_path.name == "users.json" else []
    result = normalize_json_data(file_path, read_json(file_path, default))
    if result.changed:
        write_json(file_path, result.data)
    return result


def patient_scope_for_user(username: str, users: dict[str, Any]) -> set[str] | None:
    user = _as_dict(users.get(username))
    role = user.get("role")
    if role == ADMIN_ROLE or role == RESEARCHER_ROLE:
        return None
    if role == PATIENT_ROLE:
        return {username}
    assigned = user.get("assigned_patient_usernames")
    if isinstance(assigned, str):
        assigned = [part.strip() for part in assigned.split(",") if part.strip()]
    if isinstance(assigned, list) and assigned:
        return {str(item).strip() for item in assigned if str(item).strip()}

    derived = {
        patient_username
        for patient_username, patient in users.items()
        if isinstance(patient, dict)
        and patient.get("role") == PATIENT_ROLE
        and patient.get("assigned_doctor_username") == username
    }
    return derived


def record_patient_username(record: dict[str, Any]) -> str:
    return _as_str(record.get("patient_username") or record.get("username") or record.get("subject_code"))


def filter_records_for_actor(records: list[dict[str, Any]], actor: dict[str, Any], users: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    users = users or {}
    scope = patient_scope_for_user(_as_str(actor.get("username")), users)
    if scope is None:
        return list(records or [])
    return [record for record in records or [] if record_patient_username(record) in scope]


PII_KEYS = {
    "full_name",
    "patient_name",
    "patient_id",
    "doctor_name",
    "email",
    "phone",
    "address",
    "interviewer",
    "subject_code",
    "submitted_by",
    "specialist_comment",
    "comments",
    "comments_ncv",
    "symptoms",
}


def pseudonymize_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    patient = record_patient_username(record)
    if patient:
        digest = hashlib.sha256(patient.encode("utf-8")).hexdigest()[:10].upper()
        out["subject_code"] = f"SUBJ-{digest}"
    for key, value in record.items():
        if key in PII_KEYS:
            continue
        if key in {"patient_username", "username"}:
            out[key] = out.get("subject_code", "SUBJ-00000")
        elif key == "subject_code":
            continue
        else:
            out[key] = value
    return out


def pseudonymize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [pseudonymize_record(record) for record in records or []]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
