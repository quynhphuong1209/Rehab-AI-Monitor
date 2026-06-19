"""Small, scoped previews for analysis CSV/JSON artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

from backend.analysis_parity import (
    chart_point_payload,
    filter_points,
    frame_summary,
    normalized_frame_filter,
    phase_metrics_from_record,
    segment_frame_bounds,
)


MAX_CHART_SOURCE_BYTES = 10 * 1024 * 1024
MAX_CHART_SOURCE_ROWS = 20_000
MAX_CHART_POINTS = 180
ANGLE_SERIES_KEYS = ("goc_vai", "goc_khuyu", "vai_chuan", "khuyu_chuan")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_number(value: Any) -> float | None:
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


def _first_number(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _safe_number(record.get(key))
        if number is not None:
            return number
    return None


def _first_present(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _average_numbers(record: dict[str, Any], *keys: str) -> float | None:
    values = [_safe_number(record.get(key)) for key in keys]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 3)


def _boolish(value: Any) -> bool:
    if value is True:
        return True
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "co", "có", "dung", "đúng", "pass"}


def _label_for_record(record: dict[str, Any]) -> str:
    for key in ("label", "status", "phase_status", "ref_status"):
        value = _clean_text(record.get(key)).upper()
        if value in {"PASS", "NEAR", "NEARLY", "FAIL"}:
            return "NEAR" if value == "NEARLY" else value
    if _boolish(record.get("dung")):
        return "PASS"
    if _boolish(record.get("gan_dung")):
        return "NEAR"
    if "dung" in record or "gan_dung" in record:
        return "FAIL"
    return ""


def _nested_eval_info(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("eval_info")
    return value if isinstance(value, dict) else {}


def _shoulder_angle(record: dict[str, Any]) -> float | None:
    return _first_present(
        _first_number(record, "goc_vai"),
        _average_numbers(record, "goc_vai_trai", "goc_vai_phai"),
        _average_numbers(record, "goc_vai_left", "goc_vai_right"),
        _first_number(record, "goc_vai_phai", "goc_vai_right", "goc_vai_trai", "goc_vai_left"),
    )


def _elbow_angle(record: dict[str, Any]) -> float | None:
    return _first_present(
        _first_number(record, "goc_khuyu"),
        _average_numbers(record, "goc_khuyu_trai", "goc_khuyu_phai"),
        _average_numbers(record, "goc_khuyu_left", "goc_khuyu_right"),
        _first_number(record, "goc_khuyu_phai", "goc_khuyu_right", "goc_khuyu_trai", "goc_khuyu_left"),
    )


def _normal_chart_point(record: dict[str, Any], raw_index: int) -> dict[str, Any]:
    eval_info = _nested_eval_info(record)
    point: dict[str, Any] = {
        "index": raw_index + 1,
        "frame": record.get("frame") or record.get("frame_idx") or record.get("index") or raw_index + 1,
        "timestamp": record.get("timestamp") or record.get("time") or "",
        "label": _label_for_record(record),
        "goc_vai": _shoulder_angle(record),
        "goc_khuyu": _elbow_angle(record),
        "vai_chuan": _first_present(
            _first_number(record, "vai_chuan", "chuan_vai", "shoulder_ref"),
            _safe_number(eval_info.get("shoulder_ref")),
        ),
        "khuyu_chuan": _first_present(
            _first_number(record, "khuyu_chuan", "chuan_khuyu", "elbow_ref"),
            _safe_number(eval_info.get("elbow_ref")),
        ),
    }
    return point


def _read_csv_rows(path: str) -> list[dict[str, Any]]:
    if os.path.getsize(path) > MAX_CHART_SOURCE_BYTES:
        raise ValueError("chart CSV is too large")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        for row in reader:
            if len(rows) >= MAX_CHART_SOURCE_ROWS:
                raise ValueError("chart CSV contains too many rows")
            rows.append(dict(row))
    return rows


def _read_json_rows(path: str) -> list[dict[str, Any]]:
    if os.path.getsize(path) > MAX_CHART_SOURCE_BYTES:
        raise ValueError("chart JSON is too large")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    rows = [item for item in data if isinstance(item, dict)]
    if len(rows) > MAX_CHART_SOURCE_ROWS:
        raise ValueError("chart JSON contains too many rows")
    return rows


def _downsample(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) <= MAX_CHART_POINTS:
        return points
    step = (len(points) - 1) / (MAX_CHART_POINTS - 1)
    indexes = sorted({round(index * step) for index in range(MAX_CHART_POINTS)})
    return [points[index] for index in indexes]


def _series_summary(points: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for key in ANGLE_SERIES_KEYS:
        values = [point.get(key) for point in points if isinstance(point.get(key), (int, float))]
        numbers = [float(value) for value in values if math.isfinite(float(value))]
        if not numbers:
            summary[key] = {"count": 0, "min": None, "max": None, "avg": None}
            continue
        summary[key] = {
            "count": len(numbers),
            "min": round(min(numbers), 2),
            "max": round(max(numbers), 2),
            "avg": round(sum(numbers) / len(numbers), 2),
        }
    return summary


def _label_summary(points: list[dict[str, Any]]) -> dict[str, int]:
    labels = {"total": len(points), "PASS": 0, "NEAR": 0, "FAIL": 0}
    for point in points:
        label = point.get("label")
        if label in {"PASS", "NEAR", "FAIL"}:
            labels[str(label)] += 1
    return labels


def analysis_chart_preview(
    record: dict[str, Any],
    *,
    csv_path: str | None,
    frames_json_path: str | None,
    label_filter: str = "ALL",
) -> dict[str, Any]:
    source = "none"
    source_label = "No chart artifact"
    rows: list[dict[str, Any]] = []
    if csv_path:
        rows = _read_csv_rows(csv_path)
        source = "csv"
        source_label = "CSV angles"
    elif frames_json_path:
        rows = _read_json_rows(frames_json_path)
        source = "frames-json"
        source_label = "Frames JSON"

    frame_filter = normalized_frame_filter(label_filter)
    bounds = segment_frame_bounds(rows)
    exercise = record.get("exercise") or ""
    points = [chart_point_payload(row, index, bounds, exercise) for index, row in enumerate(rows)]
    filtered_points = filter_points(points, frame_filter)
    columns = [
        key
        for key in ANGLE_SERIES_KEYS
        if any(isinstance(point.get(key), (int, float)) for point in filtered_points)
    ]
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return {
        "source": source,
        "source_label": source_label,
        "filter": frame_filter,
        "total_rows": len(points),
        "filtered_rows": len(filtered_points),
        "sampled_rows": min(len(filtered_points), MAX_CHART_POINTS),
        "columns": columns,
        "segment_bounds": bounds,
        "summary": {
            "series": _series_summary(filtered_points),
            "labels": _label_summary(filtered_points),
        },
        "phase_summary": frame_summary(rows, bounds, exercise),
        "phase_metrics": phase_metrics_from_record(record),
        "metrics": metrics,
        "series": _downsample(filtered_points),
    }
