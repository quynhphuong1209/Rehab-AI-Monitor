# -*- coding: utf-8 -*-
"""YouTube reference pose matching — khớp tư thế theo động tác 1/2/3, không theo từng giây."""

from __future__ import annotations

import json
import os
from typing import Any

# Sai số cho phép theo giai đoạn PHCN (Δ vai & Δ khuỷu)
PHASE_ERROR = {"g1": 45, "g2": 30, "g3": 15}
PHASE_ERROR_DEFAULT = 30
NEAR_ERROR_MULTIPLIER = 1.5

PHASE_UI_LABELS = {
    "g1": "Giai đoạn 1: Khởi đầu (Sai số ±45°)",
    "g2": "Giai đoạn 2: Hồi phục (Sai số ±30°)",
    "g3": "Giai đoạn 3: Chuẩn xác (Sai số ±15°)",
}

PHASE_UI_SHORT = {
    "g1": "G1 · ss ±45°",
    "g2": "G2 · ss ±30°",
    "g3": "G3 · ss ±15°",
}

# Chữ ký góc để nhận dạng động tác con (học máy rule-based từ mẫu YouTube)
_MOTION_SIGNATURES: dict[str, dict[str, dict[str, float]]] = {
    "codman": {
        "1": {"vai_min": 15, "vai_max": 90, "khuyu_min": 160, "khuyu_max": 180},
        "2": {"vai_min": 20, "vai_max": 85, "khuyu_min": 160, "khuyu_max": 180},
        "3": {"vai_min": 25, "vai_max": 75, "khuyu_min": 165, "khuyu_max": 178},
    },
    "gay": {
        "1": {"vai_min": 70, "vai_max": 180, "khuyu_min": 155, "khuyu_max": 180},
        "2": {"vai_min": 25, "vai_max": 90, "khuyu_min": 70, "khuyu_max": 120},
        "3": {"vai_min": 10, "vai_max": 70, "khuyu_min": 80, "khuyu_max": 150},
    },
}


def is_codman_exercise(exercise_name: str | None) -> bool:
    text = str(exercise_name or "").lower()
    return "codman" in text or "con lắc" in text or "con lac" in text


def is_gay_exercise(exercise_name: str | None) -> bool:
    text = str(exercise_name or "").lower()
    return any(kw in text for kw in ["gậy", "gay", "pulley", "stick"])


def is_day_exercise(exercise_name: str | None) -> bool:
    text = str(exercise_name or "").lower()
    return any(kw in text for kw in ["dây", "day", "kháng lực", "khang", "theraband", "band"])


def get_phase_error(phase_key: str) -> int:
    return int(PHASE_ERROR.get(str(phase_key).lower(), PHASE_ERROR_DEFAULT))


def get_phase_error_for_segment(frame_index: int, n0: int, n1: int, n2: int, n3: int) -> int:
    if frame_index < n1:
        return PHASE_ERROR["g1"]
    if frame_index < n2:
        return PHASE_ERROR["g2"]
    return PHASE_ERROR["g3"]


def phase_frame_label(phase_key: str, status: str) -> str:
    """Nhãn vẽ trên frame: G1(ss±45°): PASS"""
    key = str(phase_key).lower()
    ss = get_phase_error(key)
    return f"G{key[-1]}(ss±{ss}°): {status}"


def _exercise_family(exercise_name: str | None) -> str | None:
    if is_codman_exercise(exercise_name):
        return "codman"
    if is_gay_exercise(exercise_name):
        return "gay"
    if is_day_exercise(exercise_name):
        return "day"
    return None


def _pose_in_signature(
    vai: float,
    khuyu: float,
    sig: dict[str, float],
) -> bool:
    return (
        sig["vai_min"] <= vai <= sig["vai_max"]
        and sig["khuyu_min"] <= khuyu <= sig["khuyu_max"]
    )


def detect_motion_subtype(
    exercise_name: str | None,
    vai: float | None,
    khuyu: float | None,
    *,
    vai_trai: float | None = None,
    vai_phai: float | None = None,
    khuyu_trai: float | None = None,
    khuyu_phai: float | None = None,
) -> str | None:
    """
    Nhận dạng động tác con 1/2/3 từ góc hiện tại (vd. dơ gậy cao → động tác 1).
    Trả về "1", "2", "3" hoặc None nếu không xác định được.
    """
    family = _exercise_family(exercise_name)
    if not family or family not in _MOTION_SIGNATURES:
        return None

    if family == "gay":
        vai_use = max(
            float(vai_trai if vai_trai is not None else (vai or 0)),
            float(vai_phai if vai_phai is not None else (vai or 0)),
        )
        khuyu_use = float(khuyu if khuyu is not None else 170)
        if khuyu_trai is not None and khuyu_phai is not None:
            khuyu_use = (float(khuyu_trai) + float(khuyu_phai)) / 2
    elif family == "codman":
        vai_use = float(vai_phai if vai_phai is not None else (vai or 0))
        khuyu_use = float(khuyu_phai if khuyu_phai is not None else (khuyu or 170))
    else:
        vai_use = float(vai or 0)
        khuyu_use = float(khuyu or 170)

    signatures = _MOTION_SIGNATURES[family]
    scores: list[tuple[str, int]] = []
    for ex_id, sig in signatures.items():
        if _pose_in_signature(vai_use, khuyu_use, sig):
            scores.append((ex_id, 2))
        elif _pose_in_signature(vai_use, khuyu_use, {**sig, "vai_min": sig["vai_min"] - 15, "vai_max": sig["vai_max"] + 15}):
            scores.append((ex_id, 1))

    if not scores:
        return None
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def _normalize_raw_reference(raw: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Hỗ trợ format v1 (list) và v2 (object có exercises + poses)."""
    meta: dict[str, Any] | None = None
    if isinstance(raw, dict):
        meta = {k: raw[k] for k in ("version", "youtube", "side", "exercises") if k in raw}
        poses = raw.get("poses") or []
        if not poses and raw.get("exercises"):
            poses = []
            for ex_id, ex_data in raw["exercises"].items():
                for p in ex_data.get("poses", []):
                    row = dict(p)
                    row.setdefault("exercise_id", int(ex_id))
                    row.setdefault("motion_type", ex_data.get("motion_type", ""))
                    poses.append(row)
        return list(poses), meta
    if isinstance(raw, list):
        return raw, None
    return [], None


def filter_reference_poses(
    raw_refs: list[dict[str, Any]],
    exercise_name: str | None,
    motion_subtype: str | None = None,
) -> list[dict[str, Any]]:
    """
    Giữ tư thế thuộc đoạn động tác liên quan.
    Nếu có motion_subtype (1/2/3) thì chỉ lấy mẫu cùng động tác.
    """
    if not raw_refs:
        return raw_refs

    refs = raw_refs
    if motion_subtype:
        tagged = [
            r for r in refs
            if str(r.get("exercise_id", "")) == str(motion_subtype)
            or str(r.get("motion_type", "")).lower() in _subtype_motion_types(exercise_name, motion_subtype)
        ]
        if len(tagged) >= 3:
            refs = tagged

    tagged = [
        r for r in refs
        if str(r.get("motion_type", "")).lower() in {"circular", "xoay", "rotation", "pendulum"}
    ]
    if len(tagged) >= 5:
        return tagged

    if is_codman_exercise(exercise_name):
        filtered = [
            r for r in refs
            if 20 <= float(r.get("vai", r.get("vai_phai", 0))) <= 80
            and 165 <= float(r.get("khuyu", r.get("khuyu_phai", 180))) <= 180
        ]
        if len(filtered) >= 5:
            return filtered

    if is_gay_exercise(exercise_name):
        if motion_subtype == "1":
            filtered = [r for r in refs if float(r.get("vai", 0)) >= 50]
        elif motion_subtype == "2":
            filtered = [
                r for r in refs
                if 25 <= float(r.get("vai", 0)) <= 100
                and float(r.get("khuyu", 170)) <= 130
            ]
        elif motion_subtype == "3":
            filtered = [r for r in refs if float(r.get("vai", 0)) <= 75]
        else:
            filtered = [r for r in refs if float(r.get("vai", 0)) >= 8]
        if len(filtered) >= 3:
            return filtered

    return refs


def _subtype_motion_types(exercise_name: str | None, subtype: str) -> set[str]:
    family = _exercise_family(exercise_name)
    if family == "codman":
        return {
            "1": {"sagittal", "pendulum"},
            "2": {"frontal", "lateral"},
            "3": {"circular", "xoay", "rotation", "pendulum"},
        }.get(subtype, set())
    if family == "gay":
        return {
            "1": {"flexion", "raise", "nang"},
            "2": {"external_rotation", "external", "xoay_ngoai"},
            "3": {"internal_rotation", "internal", "xoay_trong"},
        }.get(subtype, set())
    return set()


def _pose_distance(
    ref: dict[str, Any],
    vai: float,
    khuyu: float,
    *,
    vai_trai: float | None = None,
    vai_phai: float | None = None,
    khuyu_trai: float | None = None,
    khuyu_phai: float | None = None,
    exercise_name: str | None = None,
    weight_vai: float = 1.0,
    weight_khuyu: float = 1.0,
) -> float:
    if is_gay_exercise(exercise_name) and vai_trai is not None and vai_phai is not None:
        rv_t = float(ref.get("vai_trai", ref.get("vai", 90)))
        rk_t = float(ref.get("khuyu_trai", ref.get("khuyu", 170)))
        rv_p = float(ref.get("vai_phai", ref.get("vai", 90)))
        rk_p = float(ref.get("khuyu_phai", ref.get("khuyu", 170)))
        kt = float(khuyu_trai if khuyu_trai is not None else khuyu)
        kp = float(khuyu_phai if khuyu_phai is not None else khuyu)
        d_t = (weight_vai * (rv_t - vai_trai) ** 2) + (weight_khuyu * (rk_t - kt) ** 2)
        d_p = (weight_vai * (rv_p - vai_phai) ** 2) + (weight_khuyu * (rk_p - kp) ** 2)
        return max(d_t, d_p)

    if is_codman_exercise(exercise_name):
        rv = float(ref.get("vai_phai", ref.get("vai", 90)))
        rk = float(ref.get("khuyu_phai", ref.get("khuyu", 170)))
        v = float(vai_phai if vai_phai is not None else vai)
        k = float(khuyu_phai if khuyu_phai is not None else khuyu)
    else:
        rv = float(ref.get("vai", 90))
        rk = float(ref.get("khuyu", 170))
        v, k = float(vai), float(khuyu)

    return (weight_vai * (rv - v) ** 2) + (weight_khuyu * (rk - k) ** 2)


def find_closest_reference_pose(
    refs: list[dict[str, Any]] | None,
    vai: float | None,
    khuyu: float | None,
    exercise_name: str | None,
    *,
    vai_trai: float | None = None,
    vai_phai: float | None = None,
    khuyu_trai: float | None = None,
    khuyu_phai: float | None = None,
    motion_subtype: str | None = None,
) -> dict[str, Any] | None:
    """
    Tìm tư thế mẫu YouTube gần nhất theo góc vai/khuỷu — KHÔNG dùng trường `time`.
    Ưu tiên khớp trong cùng động tác 1/2/3 nếu nhận dạng được.
    """
    if not refs or vai is None:
        return None

    khuyu_val = float(khuyu if khuyu is not None else 170)

    subtype = motion_subtype or detect_motion_subtype(
        exercise_name,
        vai,
        khuyu_val,
        vai_trai=vai_trai,
        vai_phai=vai_phai,
        khuyu_trai=khuyu_trai,
        khuyu_phai=khuyu_phai,
    )
    scoped = filter_reference_poses(refs, exercise_name, subtype)

    if is_day_exercise(exercise_name):
        return min(scoped, key=lambda x: abs(float(x.get("khuyu", 170)) - khuyu_val), default=None)

    return min(
        scoped,
        key=lambda x: _pose_distance(
            x,
            float(vai),
            khuyu_val,
            vai_trai=vai_trai,
            vai_phai=vai_phai,
            khuyu_trai=khuyu_trai,
            khuyu_phai=khuyu_phai,
            exercise_name=exercise_name,
        ),
        default=None,
    )


def load_reference_poses(ref_path: str, exercise_name: str | None) -> list[dict[str, Any]]:
    with open(ref_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    poses, _meta = _normalize_raw_reference(raw)
    return filter_reference_poses(poses, exercise_name)


def load_reference_bundle(ref_path: str) -> dict[str, Any]:
    """Nạp đầy đủ metadata + poses (dùng cho UI / debug)."""
    with open(ref_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    poses, meta = _normalize_raw_reference(raw)
    bundle = meta or {}
    bundle["poses"] = poses
    bundle["path"] = ref_path
    return bundle


def normalize_phase_selection(value: str | None) -> str:
    text = str(value or "")
    if "Giai đoạn 1" in text:
        return PHASE_UI_LABELS["g1"]
    if "Giai đoạn 3" in text:
        return PHASE_UI_LABELS["g3"]
    if value in PHASE_UI_LABELS.values():
        return str(value)
    return PHASE_UI_LABELS["g2"]


def resolve_reference_file(ref_name: str, db_dir: str, app_dir: str) -> str | None:
    search_paths = [
        os.path.join(db_dir, f"reference_{ref_name}.json"),
        os.path.join("database", f"reference_{ref_name}.json"),
        f"reference_{ref_name}.json",
        os.path.join(app_dir, f"reference_{ref_name}.json"),
        os.path.join(os.getcwd(), f"reference_{ref_name}.json"),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None
