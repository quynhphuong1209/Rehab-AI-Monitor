# -*- coding: utf-8 -*-
"""Second-stage pose classifier utilities.

MediaPipe is still the pose/keypoint extractor. This module trains and applies a
RandomForest classifier on the extracted landmark CSV files.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


KEY_POINTS = [11, 12, 13, 14, 15, 16, 23, 24]
COORDINATE_COLS: list[str] = []
for point_idx in KEY_POINTS:
    COORDINATE_COLS.extend(
        [
            f"pt{point_idx}_x",
            f"pt{point_idx}_y",
            f"pt{point_idx}_z",
            f"pt{point_idx}_vis",
        ]
    )

FEATURE_COLS = ["goc_vai", "goc_khuyu"] + COORDINATE_COLS
MODEL_FILENAME = "pose_classifier.pkl"
FEATURES_FILENAME = "pose_classifier_features.json"
LABEL_NAMES = {
    0: "Sai",
    1: "Gan dung",
    2: "Dung",
}


def get_model_paths(db_dir: str = "database") -> tuple[str, str]:
    return (
        os.path.join(db_dir, MODEL_FILENAME),
        os.path.join(db_dir, FEATURES_FILENAME),
    )


def get_pose_classifier_status(db_dir: str = "database") -> dict[str, Any]:
    model_path, features_path = get_model_paths(db_dir)
    ready = os.path.exists(model_path) and os.path.exists(features_path)
    return {
        "ready": ready,
        "model_path": model_path,
        "features_path": features_path,
        "model_mtime": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
        if os.path.exists(model_path)
        else None,
    }


def _labels_to_int(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").round().astype("Int64")

    mapping = {
        "true": 1,
        "1": 1,
        "yes": 1,
        "pass": 1,
        "dung": 1,
        "đúng": 1,
        "dat": 1,
        "đạt": 1,
        "false": 0,
        "0": 0,
        "no": 0,
        "fail": 0,
        "sai": 0,
        "khong dat": 0,
        "không đạt": 0,
    }
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(mapping).astype("Int64")


def _build_training_labels(df: pd.DataFrame) -> pd.Series:
    dung = _labels_to_int(df["dung"]).fillna(0).astype(int)
    if "gan_dung" in df.columns:
        gan_dung = _labels_to_int(df["gan_dung"]).fillna(0).astype(int)
    else:
        gan_dung = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    labels = np.where(dung == 1, 2, np.where(gan_dung == 1, 1, 0))
    return pd.Series(labels, index=df.index, dtype=int)


def load_training_data(
    processed_dir: str = "processed_results",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None, dict[str, Any]]:
    feature_cols = feature_cols or FEATURE_COLS
    csv_files = sorted(glob.glob(os.path.join(processed_dir, "*_data.csv")))
    summary: dict[str, Any] = {
        "csv_files": len(csv_files),
        "valid_files": 0,
        "skipped_files": [],
        "samples": 0,
        "label_distribution": {},
    }
    if not csv_files:
        summary["error"] = f"Khong tim thay CSV trong {processed_dir}"
        return None, None, summary

    frames: list[pd.DataFrame] = []
    required = feature_cols + ["dung"]
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            missing = [col for col in required if col not in df.columns]
            if missing:
                summary["skipped_files"].append(
                    {"file": os.path.basename(csv_path), "missing": missing[:5]}
                )
                continue

            optional_cols = ["gan_dung"] if "gan_dung" in df.columns else []
            part = df[required + optional_cols].copy()
            part["ml_label"] = _build_training_labels(part)
            for col in feature_cols:
                part[col] = pd.to_numeric(part[col], errors="coerce")
            part = part.dropna(subset=feature_cols)
            if part.empty:
                summary["skipped_files"].append(
                    {"file": os.path.basename(csv_path), "missing": ["valid_rows"]}
                )
                continue

            frames.append(part)
            summary["valid_files"] += 1
        except Exception as exc:
            summary["skipped_files"].append(
                {"file": os.path.basename(csv_path), "error": str(exc)}
            )

    if not frames:
        summary["error"] = "Khong co CSV hop le de train"
        return None, None, summary

    merged = pd.concat(frames, ignore_index=True)
    X = merged[feature_cols]
    y = merged["ml_label"].astype(int)
    summary["samples"] = int(len(X))
    summary["label_distribution"] = {
        f"{int(label)}_{LABEL_NAMES.get(int(label), 'Unknown')}": int(count)
        for label, count in y.value_counts().sort_index().items()
    }
    return X, y, summary


def train_pose_classifier(
    processed_dir: str = "processed_results",
    db_dir: str = "database",
    min_samples: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    X, y, summary = load_training_data(processed_dir)
    if X is None or y is None:
        return {"success": False, "message": summary.get("error", "Khong co du lieu"), **summary}
    if len(X) < min_samples:
        return {
            "success": False,
            "message": f"Khong du du lieu train, can toi thieu {min_samples} dong",
            **summary,
        }
    if y.nunique() < 2:
        return {
            "success": False,
            "message": "Can it nhat 2 nhan trong 3 lop Sai/Gan dung/Dung de train classifier",
            **summary,
        }

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    import joblib

    label_counts = y.value_counts()
    stratify = y if label_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Sai (0)", "Gan dung (1)", "Dung (2)"],
        output_dict=True,
        zero_division=0,
    )

    os.makedirs(db_dir, exist_ok=True)
    model_path, features_path = get_model_paths(db_dir)
    joblib.dump(clf, model_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": FEATURE_COLS,
                "label_names": LABEL_NAMES,
                "label_source": "dung + gan_dung frame labels",
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    return {
        "success": True,
        "message": "Da train va luu pose classifier",
        "accuracy": round(accuracy * 100, 2),
        "model_path": model_path,
        "features_path": features_path,
        "classification_report": report,
        **summary,
    }


def _load_classifier(db_dir: str = "database"):
    import joblib

    model_path, features_path = get_model_paths(db_dir)
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError("Chua co pose classifier. Hay train model truoc.")
    clf = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        raw_features = json.load(f)
    feature_cols = raw_features.get("feature_cols", FEATURE_COLS) if isinstance(raw_features, dict) else raw_features
    return clf, feature_cols


def create_pose_classifier_predictor(db_dir: str = "database") -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Load the trained classifier once and return a per-frame predictor."""
    clf, feature_cols = _load_classifier(db_dir)
    classes = list(getattr(clf, "classes_", []))
    pass_label = _pass_label_from_classes(classes)

    def predict_row(row: Mapping[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([{col: row.get(col) for col in feature_cols}])
        X = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        prediction = int(clf.predict(X)[0])
        result: dict[str, Any] = {
            "ml_label": prediction,
            "ml_label_text": LABEL_NAMES.get(prediction, str(prediction)),
            "dung_ml": bool(prediction == pass_label),
            "gan_dung_ml": bool(prediction == 1) if pass_label == 2 else False,
        }

        if hasattr(clf, "predict_proba"):
            probabilities = clf.predict_proba(X)[0]
            if pass_label in classes:
                result["ml_score"] = round(float(probabilities[classes.index(pass_label)] * 100), 2)
            if prediction in classes:
                result["ml_confidence"] = round(float(probabilities[classes.index(prediction)] * 100), 2)
        return result

    return predict_row


def _is_gay_exercise(exercise_name: str | None) -> bool:
    text = str(exercise_name or "").lower()
    return any(keyword in text for keyword in ["gậy", "gay", "pulley", "stick"])


def _is_codman_exercise(exercise_name: str | None) -> bool:
    text = str(exercise_name or "").lower()
    return "codman" in text or "con lắc" in text or "con lac" in text


def _default_phase_bounds(total_rows: int) -> list[int]:
    return [0, total_rows // 3, (2 * total_rows) // 3, total_rows]


def segment_codman_frames(df: pd.DataFrame) -> list[int]:
    """Phan 3 giai doan Codman — dong bo logic segment_frames() trong app.py."""
    total = len(df)
    if total < 30:
        return _default_phase_bounds(total)

    goc_v = df["goc_vai"].fillna(90).tolist()
    goc_k = df["goc_khuyu"].fillna(170).tolist()
    var_v = np.std(goc_v)
    var_k = np.std(goc_k)
    angles = np.array(goc_v) if var_v > var_k else np.array(goc_k)

    window_size = min(15, max(5, total // 30))
    smoothed = np.convolve(angles, np.ones(window_size) / window_size, mode="same")

    valleys: list[int] = []
    threshold_val = np.percentile(smoothed, 50)
    min_dist = max(15, total // 8)

    for i in range(window_size, total - window_size):
        is_min = all(smoothed[i] <= smoothed[j] for j in range(i - window_size, i + window_size + 1))
        if is_min and smoothed[i] < threshold_val:
            if not valleys or (i - valleys[-1] >= min_dist):
                valleys.append(i)

    filtered_valleys = [v for v in valleys if v > total // 10 and v < total - total // 10]

    if len(filtered_valleys) >= 2:
        if len(filtered_valleys) == 2:
            n1, n2 = filtered_valleys[0], filtered_valleys[1]
        else:
            best_diff = float("inf")
            n1, n2 = total // 3, (2 * total) // 3
            for i in range(len(filtered_valleys)):
                for j in range(i + 1, len(filtered_valleys)):
                    p1, p2 = filtered_valleys[i], filtered_valleys[j]
                    sizes = [p1, p2 - p1, total - p2]
                    diff = max(sizes) - min(sizes)
                    if diff < best_diff:
                        best_diff = diff
                        n1, n2 = p1, p2
    elif len(filtered_valleys) == 1:
        v = filtered_valleys[0]
        if v < total // 2:
            n1, n2 = v, v + (total - v) // 2
        else:
            n1, n2 = v // 2, v
    else:
        n1, n2 = total // 3, (2 * total) // 3

    return [0, n1, n2, total]


def _pass_label_from_classes(classes: list[int] | np.ndarray | None) -> int:
    if classes is None:
        classes = []
    else:
        classes = list(classes)
    return 2 if 2 in classes else 1


def _accuracy_from_predictions(predictions: np.ndarray, pass_label: int = 1) -> float:
    if len(predictions) == 0:
        return 0.0
    return round(float(np.mean(predictions == pass_label) * 100), 1)


def _phase_accuracy(
    predictions: np.ndarray,
    phase_bounds: list[int] | tuple[int, int, int, int] | None,
    exercise_name: str | None,
    pass_label: int = 1,
) -> dict[str, float]:
    overall = _accuracy_from_predictions(predictions, pass_label)
    if len(predictions) == 0 or _is_gay_exercise(exercise_name):
        return {"overall": overall, "g1": overall, "g2": overall, "g3": overall}

    bounds = list(phase_bounds or _default_phase_bounds(len(predictions)))
    if len(bounds) != 4:
        bounds = _default_phase_bounds(len(predictions))
    n0, n1, n2, n3 = [max(0, min(len(predictions), int(v))) for v in bounds]
    if not (n0 <= n1 <= n2 <= n3):
        n0, n1, n2, n3 = _default_phase_bounds(len(predictions))

    return {
        "overall": overall,
        "g1": _accuracy_from_predictions(predictions[n0:n1], pass_label),
        "g2": _accuracy_from_predictions(predictions[n1:n2], pass_label),
        "g3": _accuracy_from_predictions(predictions[n2:n3], pass_label),
    }


def apply_classifier_to_dataframe(
    df: pd.DataFrame,
    db_dir: str = "database",
    phase_bounds: list[int] | tuple[int, int, int, int] | None = None,
    phase_bounds_fn: Callable[[pd.DataFrame], list[int] | tuple[int, int, int, int]] | None = None,
    exercise_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clf, feature_cols = _load_classifier(db_dir)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV thieu cot dac trung: {missing[:5]}")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    predictions = clf.predict(X)
    classes = list(getattr(clf, "classes_", []))
    pass_label = _pass_label_from_classes(classes)

    out_df = df.copy()
    out_df["ml_label"] = predictions
    out_df["ml_label_text"] = [LABEL_NAMES.get(int(label), str(label)) for label in predictions]
    out_df["dung_ml"] = predictions == pass_label
    if pass_label == 2:
        out_df["gan_dung_ml"] = predictions == 1
    if hasattr(clf, "predict_proba"):
        if pass_label in classes:
            class_idx = classes.index(pass_label)
            out_df["ml_score"] = np.round(clf.predict_proba(X)[:, class_idx] * 100, 2)

    if phase_bounds_fn is not None and phase_bounds is None:
        try:
            phase_bounds = phase_bounds_fn(out_df)
        except Exception:
            phase_bounds = None

    ml_phases = _phase_accuracy(predictions, phase_bounds, exercise_name, pass_label)
    return out_df, {
        "ml_phases": ml_phases,
        "overall_correct": int(np.sum(predictions == pass_label)),
        "overall_nearly": int(np.sum(predictions == 1)) if pass_label == 2 else 0,
        "overall_fail": int(np.sum(predictions == 0)),
        "total_rows": int(len(predictions)),
        "label_distribution": {
            LABEL_NAMES.get(int(label), str(label)): int(count)
            for label, count in pd.Series(predictions).value_counts().sort_index().items()
        },
        "is_codman": _is_codman_exercise(exercise_name),
        "is_gay": _is_gay_exercise(exercise_name),
    }


def merge_ml_metrics(metrics: dict[str, Any] | None, ml_result: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics if isinstance(metrics, dict) else {}
    phases = ml_result.get("ml_phases", {})
    metrics["ml_do_chinh_xac"] = phases.get("overall", 0.0)
    metrics["ml_frame_dung"] = ml_result.get("overall_correct", 0)
    metrics["ml_frame_gan_dung"] = ml_result.get("overall_nearly", 0)
    metrics["ml_frame_sai"] = ml_result.get("overall_fail", 0)
    metrics["ml_tong_frame"] = ml_result.get("total_rows", 0)

    for key, phase_key in [("metrics_g1", "g1"), ("metrics_g2", "g2"), ("metrics_g3", "g3")]:
        block = metrics.get(key, {})
        if not isinstance(block, dict):
            block = {}
        block["ml_do_chinh_xac"] = phases.get(phase_key, phases.get("overall", 0.0))
        metrics[key] = block
    return metrics


def _read_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_local_path(
    path_str: str | None,
    data_dir: str = ".",
    processed_dir: str = "processed_results",
    db_dir: str = "database",
) -> str | None:
    if not path_str:
        return None

    clean = str(path_str).replace("\\", "/").replace("/data/", "")
    if clean.startswith("/"):
        clean = clean[1:]
    basename = os.path.basename(clean)
    candidates = [
        path_str,
        clean,
        os.path.join(data_dir, clean),
        os.path.join(processed_dir, basename),
        os.path.join(data_dir, "processed_results", basename),
        os.path.join(db_dir, basename),
        os.path.abspath(clean),
        os.path.abspath(os.path.join(processed_dir, basename)),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.path.getsize(candidate) > 100:
            return candidate
    return None


def _badge_color_for_rule(dung: bool, gan_dung: bool) -> tuple[str, tuple[int, int, int]]:
    if dung:
        return "PASS", (0, 220, 80)
    if gan_dung:
        return "NEARLY", (0, 165, 255)
    return "FAIL", (0, 0, 230)


def _badge_color_for_ml(ml_label_text: str | None) -> tuple[int, int, int]:
    label_key = str(ml_label_text or "").strip().lower()
    if "dung" in label_key and "gan" not in label_key:
        return (0, 220, 80)
    if "gan" in label_key:
        return (0, 165, 255)
    return (0, 0, 230)


def draw_rule_badge(
    frame_output,
    dung: bool,
    gan_dung: bool,
    scale_factor: float = 1.0,
):
    """Ve nhan doi chieu YouTube (PASS / NEARLY / FAIL) tren frame."""
    import cv2

    status, color = _badge_color_for_rule(bool(dung), bool(gan_dung))
    text = f"REF: {status}"
    h, w = frame_output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, 0.58 * scale_factor)
    thickness = max(1, int(2 * scale_factor))
    pad_x = max(8, int(10 * scale_factor))
    pad_y = max(6, int(8 * scale_factor))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + pad_x * 2
    box_h = text_h + baseline + pad_y * 2
    margin = max(10, int(15 * scale_factor))
    x1, y1 = margin, margin
    x2, y2 = min(w - margin, x1 + box_w), min(h - margin, y1 + box_h)
    overlay = frame_output.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 18, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame_output, 0.22, 0, frame_output)
    cv2.rectangle(frame_output, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame_output,
        text,
        (x1 + pad_x, y2 - pad_y - baseline),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame_output


def draw_ml_badge(frame_output, ml_info: Mapping[str, Any] | None, scale_factor: float = 1.0):
    """Ve nhan ket qua model ML tren frame."""
    if not ml_info:
        return frame_output
    import cv2

    label = str(ml_info.get("ml_label_text") or "N/A").upper()
    score = ml_info.get("ml_score", ml_info.get("ml_confidence"))
    if score is not None:
        try:
            label = f"{label} {float(score):.0f}%"
        except (TypeError, ValueError):
            pass
    text = f"ML: {label}"
    color = _badge_color_for_ml(ml_info.get("ml_label_text"))

    h, w = frame_output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, 0.58 * scale_factor)
    thickness = max(1, int(2 * scale_factor))
    pad_x = max(8, int(10 * scale_factor))
    pad_y = max(6, int(8 * scale_factor))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + pad_x * 2
    box_h = text_h + baseline + pad_y * 2
    margin = max(10, int(15 * scale_factor))
    x1 = max(margin, w - box_w - margin)
    y1 = max(margin, int(48 * scale_factor))
    x2 = min(w - margin, x1 + box_w)
    y2 = min(h - margin, y1 + box_h)
    overlay = frame_output.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 18, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame_output, 0.22, 0, frame_output)
    cv2.rectangle(frame_output, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame_output,
        text,
        (x1 + pad_x, y2 - pad_y - baseline),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame_output


def refresh_saved_frame_labels(
    frames_json_path: str,
    data_dir: str = ".",
    processed_dir: str = "processed_results",
    db_dir: str = "database",
) -> dict[str, Any]:
    """Cap nhat anh frame JPG da luu: them nhan REF + ML tu JSON frame data."""
    import cv2

    frame_data = _read_json(frames_json_path, [])
    if not isinstance(frame_data, list) or not frame_data:
        return {"success": False, "message": "Khong co du lieu frame JSON", "updated": 0}

    updated = 0
    skipped = 0
    for frame_item in frame_data:
        if not isinstance(frame_item, dict):
            skipped += 1
            continue
        img_path = resolve_local_path(
            frame_item.get("path"),
            data_dir=data_dir,
            processed_dir=processed_dir,
            db_dir=db_dir,
        )
        if not img_path or not os.path.exists(img_path):
            skipped += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        scale = img.shape[1] / 640.0
        draw_rule_badge(
            img,
            bool(frame_item.get("dung")),
            bool(frame_item.get("gan_dung")),
            scale_factor=scale,
        )
        ml_info = {
            "ml_label_text": frame_item.get("ml_label_text"),
            "ml_score": frame_item.get("ml_score"),
            "ml_confidence": frame_item.get("ml_confidence"),
        }
        if ml_info.get("ml_label_text"):
            draw_ml_badge(img, ml_info, scale_factor=scale)
        if cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            updated += 1
        else:
            skipped += 1

    return {
        "success": True,
        "updated": updated,
        "skipped": skipped,
        "frames_json_path": frames_json_path,
    }


def ensure_classifier_ready(
    processed_dir: str = "processed_results",
    db_dir: str = "database",
    min_samples: int = 10,
    auto_train: bool = True,
) -> dict[str, Any]:
    """Nap model neu co; tu dong train tu CSV neu chua co va du du lieu."""
    status = get_pose_classifier_status(db_dir)
    if status.get("ready"):
        return {"ready": True, "trained": False, **status}
    if not auto_train:
        return {"ready": False, "trained": False, "message": "Chua co model ML"}
    train_result = train_pose_classifier(processed_dir=processed_dir, db_dir=db_dir, min_samples=min_samples)
    status = get_pose_classifier_status(db_dir)
    return {
        "ready": bool(status.get("ready")),
        "trained": bool(train_result.get("success")),
        "train_result": train_result,
        **status,
    }


def reprocess_videos_with_classifier(
    videos_file: str,
    evaluations_file: str | None = None,
    processed_dir: str = "processed_results",
    db_dir: str = "database",
    data_dir: str = ".",
    phase_bounds_fn: Callable[[pd.DataFrame], list[int] | tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    get_pose_classifier_status(db_dir)
    if not get_pose_classifier_status(db_dir)["ready"]:
        return {"success": False, "message": "Chua co model pose_classifier.pkl"}

    if phase_bounds_fn is None:
        phase_bounds_fn = segment_codman_frames

    video_list = _read_json(videos_file, [])
    evaluations_list = _read_json(evaluations_file, []) if evaluations_file else []
    updated = 0
    results: list[dict[str, Any]] = []

    for v in video_list:
        csv_path = resolve_local_path(v.get("df_path"), data_dir, processed_dir, db_dir)
        if not csv_path:
            results.append({"video": v.get("video_name"), "error": "Khong tim thay CSV"})
            continue

        try:
            df = pd.read_csv(csv_path)
            predicted_df, ml_result = apply_classifier_to_dataframe(
                df,
                db_dir=db_dir,
                phase_bounds_fn=phase_bounds_fn,
                exercise_name=v.get("exercise"),
            )
            predicted_df.to_csv(csv_path, index=False)

            frames_json_path = resolve_local_path(
                v.get("all_frames_data_path"), data_dir, processed_dir, db_dir
            )
            if frames_json_path:
                frame_data = _read_json(frames_json_path, [])
                if isinstance(frame_data, list):
                    ml_cols = [
                        "ml_label",
                        "ml_label_text",
                        "ml_score",
                        "ml_confidence",
                        "dung_ml",
                        "gan_dung_ml",
                    ]
                    for idx, frame_item in enumerate(frame_data[: len(predicted_df)]):
                        if not isinstance(frame_item, dict):
                            continue
                        for col in ml_cols:
                            if col in predicted_df.columns:
                                frame_item[col] = _json_safe_scalar(predicted_df.iloc[idx][col])
                    _write_json(frames_json_path, frame_data)
                    refresh_saved_frame_labels(
                        frames_json_path,
                        data_dir=data_dir,
                        processed_dir=processed_dir,
                        db_dir=db_dir,
                    )

            ml_phases = ml_result["ml_phases"]
            v["ml_accuracy"] = ml_phases["overall"]
            v["metrics"] = merge_ml_metrics(v.get("metrics", {}), ml_result)

            is_codman = _is_codman_exercise(v.get("exercise"))
            for eval_entry in evaluations_list:
                same_patient = (
                    eval_entry.get("patient_username") == v.get("username")
                    or eval_entry.get("patient_username") == v.get("full_name")
                )
                same_video = os.path.basename(eval_entry.get("video_name", "")) == os.path.basename(
                    v.get("video_name", "")
                )
                if same_patient and same_video and eval_entry.get("doctor_username") == "AI_Researcher":
                    eval_entry["ml_accuracy"] = ml_phases["overall"]
                    if is_codman:
                        eval_entry["ml_accuracy_g1"] = ml_phases["g1"]
                        eval_entry["ml_accuracy_g2"] = ml_phases["g2"]
                        eval_entry["ml_accuracy_g3"] = ml_phases["g3"]

            updated += 1
            results.append({"video": v.get("video_name"), "ml_accuracy": ml_phases["overall"]})
        except Exception as exc:
            results.append({"video": v.get("video_name"), "error": str(exc)})

    _write_json(videos_file, video_list)
    if evaluations_file and evaluations_list:
        _write_json(evaluations_file, evaluations_list)
    return {"success": True, "updated": updated, "results": results}
