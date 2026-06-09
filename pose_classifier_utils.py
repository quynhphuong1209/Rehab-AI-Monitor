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
from typing import Any, Callable

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

            part = df[required].copy()
            part["dung"] = _labels_to_int(part["dung"])
            part = part.dropna(subset=["dung"])
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
    y = merged["dung"].astype(int)
    summary["samples"] = int(len(X))
    summary["label_distribution"] = {
        str(int(label)): int(count) for label, count in y.value_counts().sort_index().items()
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
            "message": "Can it nhat 2 nhan dung/sai de train classifier",
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
        labels=[0, 1],
        target_names=["Sai (0)", "Dung (1)"],
        output_dict=True,
        zero_division=0,
    )

    os.makedirs(db_dir, exist_ok=True)
    model_path, features_path = get_model_paths(db_dir)
    joblib.dump(clf, model_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=4)

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


def _is_gay_exercise(exercise_name: str | None) -> bool:
    text = str(exercise_name or "").lower()
    return any(keyword in text for keyword in ["gậy", "gay", "pulley", "stick"])


def _is_codman_exercise(exercise_name: str | None) -> bool:
    text = str(exercise_name or "").lower()
    return "codman" in text or "con lắc" in text or "con lac" in text


def _default_phase_bounds(total_rows: int) -> list[int]:
    return [0, total_rows // 3, (2 * total_rows) // 3, total_rows]


def _accuracy_from_predictions(predictions: np.ndarray) -> float:
    if len(predictions) == 0:
        return 0.0
    return round(float(np.mean(predictions == 1) * 100), 1)


def _phase_accuracy(
    predictions: np.ndarray,
    phase_bounds: list[int] | tuple[int, int, int, int] | None,
    exercise_name: str | None,
) -> dict[str, float]:
    overall = _accuracy_from_predictions(predictions)
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
        "g1": _accuracy_from_predictions(predictions[n0:n1]),
        "g2": _accuracy_from_predictions(predictions[n1:n2]),
        "g3": _accuracy_from_predictions(predictions[n2:n3]),
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

    out_df = df.copy()
    out_df["dung_ml"] = predictions == 1
    if hasattr(clf, "predict_proba"):
        classes = list(getattr(clf, "classes_", []))
        if 1 in classes:
            class_idx = classes.index(1)
            out_df["ml_score"] = np.round(clf.predict_proba(X)[:, class_idx] * 100, 2)

    if phase_bounds_fn is not None and phase_bounds is None:
        try:
            phase_bounds = phase_bounds_fn(out_df)
        except Exception:
            phase_bounds = None

    ml_phases = _phase_accuracy(predictions, phase_bounds, exercise_name)
    return out_df, {
        "ml_phases": ml_phases,
        "overall_correct": int(np.sum(predictions == 1)),
        "total_rows": int(len(predictions)),
        "is_codman": _is_codman_exercise(exercise_name),
        "is_gay": _is_gay_exercise(exercise_name),
    }


def merge_ml_metrics(metrics: dict[str, Any] | None, ml_result: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics if isinstance(metrics, dict) else {}
    phases = ml_result.get("ml_phases", {})
    metrics["ml_do_chinh_xac"] = phases.get("overall", 0.0)
    metrics["ml_frame_dung"] = ml_result.get("overall_correct", 0)
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
