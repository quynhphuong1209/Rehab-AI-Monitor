"""Shared Streamlit-parity helpers for analysis result previews.

These functions keep the standalone backend aligned with the existing
Streamlit interpretation of frame phases and REF/ML labels without importing
the Streamlit app.
"""

from __future__ import annotations

import math
from typing import Any


PHASE_ORDER = ("G1", "G2", "G3")
PHASE_THRESHOLDS = {"G1": 45.0, "G2": 30.0, "G3": 15.0}
FRAME_FILTERS = {"ALL", "G1", "G2", "G3", "PASS", "NEAR", "FAIL"}
PHASE_METRIC_KEYS = {
    "accuracy": ("do_chinh_xac", "ty_le_tong_the", "ai_accuracy", "accuracy"),
    "mae": ("mae_tong", "mae"),
    "f1": ("f1_score", "f1"),
    "icc": ("icc",),
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def safe_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 3)


def boolish(value: Any) -> bool:
    if value is True:
        return True
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "co", "có", "dung", "đúng", "pass"}


def _first_number(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = safe_number(record.get(key))
        if number is not None:
            return number
    return None


def _average_numbers(record: dict[str, Any], *keys: str) -> float | None:
    values = [safe_number(record.get(key)) for key in keys]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 3)


def _first_present(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _nested_eval_info(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("eval_info")
    return value if isinstance(value, dict) else {}


def shoulder_angle(record: dict[str, Any]) -> float | None:
    return _first_present(
        _first_number(record, "goc_vai"),
        _average_numbers(record, "goc_vai_trai", "goc_vai_phai"),
        _average_numbers(record, "goc_vai_left", "goc_vai_right"),
        _first_number(record, "goc_vai_phai", "goc_vai_right", "goc_vai_trai", "goc_vai_left"),
    )


def elbow_angle(record: dict[str, Any]) -> float | None:
    return _first_present(
        _first_number(record, "goc_khuyu"),
        _average_numbers(record, "goc_khuyu_trai", "goc_khuyu_phai"),
        _average_numbers(record, "goc_khuyu_left", "goc_khuyu_right"),
        _first_number(record, "goc_khuyu_phai", "goc_khuyu_right", "goc_khuyu_trai", "goc_khuyu_left"),
    )


def shoulder_ref(record: dict[str, Any]) -> float:
    eval_info = _nested_eval_info(record)
    return _first_present(
        _first_number(record, "vai_chuan", "chuan_vai", "shoulder_ref"),
        safe_number(eval_info.get("shoulder_ref")),
        90.0,
    ) or 90.0


def elbow_ref(record: dict[str, Any]) -> float:
    eval_info = _nested_eval_info(record)
    return _first_present(
        _first_number(record, "khuyu_chuan", "chuan_khuyu", "elbow_ref"),
        safe_number(eval_info.get("elbow_ref")),
        170.0,
    ) or 170.0


def legacy_frame_label(record: dict[str, Any]) -> str:
    for key in ("label", "status", "phase_status", "ref_status"):
        value = clean_text(record.get(key)).upper()
        if value in {"PASS", "NEAR", "NEARLY", "FAIL"}:
            return "NEAR" if value == "NEARLY" else value
    if boolish(record.get("dung")):
        return "PASS"
    if boolish(record.get("gan_dung")):
        return "NEAR"
    if "dung" in record or "gan_dung" in record:
        return "FAIL"
    return "FAIL"


def is_stick_exercise(exercise: Any) -> bool:
    text = clean_text(exercise).lower()
    return any(keyword in text for keyword in ("gậy", "gay", "pulley", "stick"))


def frame_phase_status(record: dict[str, Any], threshold: float, exercise: Any = "") -> str:
    cv = shoulder_ref(record)
    ck = elbow_ref(record)

    if is_stick_exercise(exercise):
        vt = safe_number(record.get("goc_vai_trai"))
        vp = safe_number(record.get("goc_vai_phai"))
        kt = safe_number(record.get("goc_khuyu_trai"))
        kp = safe_number(record.get("goc_khuyu_phai"))
        if None not in (vt, vp, kt, kp):
            pass_ok = all(abs(value - ref) <= threshold for value, ref in ((vt, cv), (vp, cv), (kt, ck), (kp, ck)))
            near_ok = all(abs(value - ref) <= threshold * 1.5 for value, ref in ((vt, cv), (vp, cv), (kt, ck), (kp, ck)))
            if pass_ok:
                return "PASS"
            return "NEAR" if near_ok else "FAIL"

    gv = shoulder_angle(record)
    gk = elbow_angle(record)
    if gv is None or gk is None:
        return legacy_frame_label(record)
    shoulder_delta = abs(gv - cv)
    elbow_delta = abs(gk - ck)
    if shoulder_delta <= threshold and elbow_delta <= threshold:
        return "PASS"
    if shoulder_delta <= threshold * 1.5 and elbow_delta <= threshold * 1.5:
        return "NEAR"
    return "FAIL"


def segment_frame_bounds(records: list[dict[str, Any]]) -> list[int]:
    total = len(records)
    if total <= 0:
        return [0, 0, 0, 0]
    try:
        from video.metrics import segment_frames

        bounds = segment_frames(records)
    except Exception:
        bounds = [0, total // 3, (2 * total) // 3, total]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        bounds = [0, total // 3, (2 * total) // 3, total]
    cleaned = [max(0, min(total, int(value))) for value in bounds]
    cleaned[0] = 0
    cleaned[-1] = total
    cleaned[1] = max(cleaned[0], min(cleaned[1], cleaned[2]))
    cleaned[2] = max(cleaned[1], min(cleaned[2], cleaned[3]))
    return cleaned


def phase_for_index(raw_index: int, bounds: list[int]) -> str:
    _, n1, n2, n3 = bounds
    if raw_index < n1:
        return "G1"
    if raw_index < n2:
        return "G2"
    if raw_index < n3:
        return "G3"
    return "G3"


def threshold_for_phase(phase: str) -> float:
    return PHASE_THRESHOLDS.get(phase, PHASE_THRESHOLDS["G2"])


def normalized_frame_filter(value: Any) -> str:
    label = clean_text(value).upper()
    if label == "NEARLY":
        return "NEAR"
    return label if label in FRAME_FILTERS else "ALL"


def phase_label_for_record(record: dict[str, Any], raw_index: int, bounds: list[int], exercise: Any = "") -> tuple[str, str, float]:
    phase = phase_for_index(raw_index, bounds)
    threshold = threshold_for_phase(phase)
    return phase, frame_phase_status(record, threshold, exercise), threshold


def frame_passes_filter(record: dict[str, Any], raw_index: int, bounds: list[int], exercise: Any, frame_filter: str) -> bool:
    phase, label, _threshold = phase_label_for_record(record, raw_index, bounds, exercise)
    if frame_filter == "ALL":
        return True
    if frame_filter in PHASE_ORDER:
        return phase == frame_filter
    return label == frame_filter


def frame_delta_payload(record: dict[str, Any]) -> dict[str, float | None]:
    gv = shoulder_angle(record)
    gk = elbow_angle(record)
    cv = shoulder_ref(record)
    ck = elbow_ref(record)
    return {
        "shoulder_ref": cv,
        "elbow_ref": ck,
        "shoulder_delta": round(abs(gv - cv), 3) if gv is not None else None,
        "elbow_delta": round(abs(gk - ck), 3) if gk is not None else None,
    }


def ml_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    raw_label = clean_text(record.get("ml_label") or record.get("ml_label_text"))
    label_text = clean_text(record.get("ml_label_text")) or raw_label
    confidence = safe_number(record.get("ml_confidence") or record.get("ml_score"))
    probabilities_value = record.get("ml_probabilities")
    probabilities: dict[str, float] = {}
    if isinstance(probabilities_value, dict):
        probabilities = {
            clean_text(key): number
            for key, value in probabilities_value.items()
            if clean_text(key) and (number := safe_number(value)) is not None
        }
    elif isinstance(probabilities_value, list):
        for index, value in enumerate(probabilities_value):
            number = safe_number(value)
            if number is not None:
                probabilities[str(index)] = number
    if not raw_label and not label_text and confidence is None and not probabilities:
        return None
    return {
        "label": raw_label or label_text,
        "label_text": label_text or raw_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def frame_summary(records: list[dict[str, Any]], bounds: list[int], exercise: Any = "") -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(records),
        "PASS": 0,
        "NEAR": 0,
        "FAIL": 0,
        "phases": {
            phase: {"total": 0, "PASS": 0, "NEAR": 0, "FAIL": 0, "threshold": PHASE_THRESHOLDS[phase]}
            for phase in PHASE_ORDER
        },
    }
    for raw_index, record in enumerate(records):
        phase, label, _threshold = phase_label_for_record(record, raw_index, bounds, exercise)
        summary[label] += 1
        summary["phases"][phase]["total"] += 1
        summary["phases"][phase][label] += 1
    return summary


def frame_public_payload(record: dict[str, Any], raw_index: int, public_index: int, image_id: str, bounds: list[int], exercise: Any = "") -> dict[str, Any]:
    phase, label, threshold = phase_label_for_record(record, raw_index, bounds, exercise)
    payload = {
        "index": record.get("index") if record.get("index") not in (None, "") else public_index + 1,
        "timestamp": record.get("timestamp") or record.get("time") or "",
        "label": label,
        "phase": phase,
        "phase_label": f"{phase} {label}",
        "phase_threshold": threshold,
        "image_id": image_id,
        "has_image": bool(image_id),
        "goc_vai": record.get("goc_vai"),
        "goc_khuyu": record.get("goc_khuyu"),
        "goc_vai_trai": record.get("goc_vai_trai"),
        "goc_vai_phai": record.get("goc_vai_phai"),
        "goc_khuyu_trai": record.get("goc_khuyu_trai"),
        "goc_khuyu_phai": record.get("goc_khuyu_phai"),
        "detected": record.get("detected"),
        "filtered_stranger": record.get("filtered_stranger"),
        **frame_delta_payload(record),
    }
    ml = ml_payload(record)
    if ml:
        payload["ml"] = ml
    return payload


def chart_point_payload(record: dict[str, Any], raw_index: int, bounds: list[int], exercise: Any = "") -> dict[str, Any]:
    phase, label, threshold = phase_label_for_record(record, raw_index, bounds, exercise)
    return {
        "index": raw_index + 1,
        "frame": record.get("frame") or record.get("frame_idx") or record.get("index") or raw_index + 1,
        "timestamp": record.get("timestamp") or record.get("time") or "",
        "label": label,
        "phase": phase,
        "phase_threshold": threshold,
        "goc_vai": shoulder_angle(record),
        "goc_khuyu": elbow_angle(record),
        "vai_chuan": shoulder_ref(record),
        "khuyu_chuan": elbow_ref(record),
    }


def filter_points(points: list[dict[str, Any]], frame_filter: str) -> list[dict[str, Any]]:
    if frame_filter == "ALL":
        return points
    if frame_filter in PHASE_ORDER:
        return [point for point in points if point.get("phase") == frame_filter]
    return [point for point in points if point.get("label") == frame_filter]


def _metric_first(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        number = safe_number(source.get(key))
        if number is not None:
            return number
    return None


def phase_metrics_from_record(record: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    payload: dict[str, dict[str, float | None]] = {}
    for phase in PHASE_ORDER:
        key = phase.lower()
        source = metrics.get(f"metrics_{key}") if isinstance(metrics.get(f"metrics_{key}"), dict) else {}
        phase_payload: dict[str, float | None] = {"threshold": PHASE_THRESHOLDS[phase]}
        for public_key, metric_keys in PHASE_METRIC_KEYS.items():
            value = _metric_first(source, metric_keys)
            if value is None:
                value = safe_number(metrics.get(f"ai_accuracy_{key}") if public_key == "accuracy" else metrics.get(f"{public_key}_{key}"))
            phase_payload[public_key] = value
        payload[phase] = phase_payload
    return payload
