"""Backend pose-classifier jobs.

The legacy Streamlit/tools path already knows how to train and apply the
second-stage pose classifier. This module wraps that logic in small JSON-backed
jobs so the API can trigger ML work without blocking a request.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from storage.app_json import update_app_json
from storage.json_store import read_json, update_json, write_json
from utils.checksum import checksum_sidecar_path, read_sha256_sidecar
from utils.pose_classifier_utils import (
    ML_LABEL_DISPLAY,
    ML_PROB_KEYS,
    apply_classifier_to_dataframe,
    get_model_paths,
    get_pose_classifier_status,
    load_training_data,
    merge_ml_metrics,
    refresh_saved_frame_labels,
    resolve_local_path,
    segment_codman_frames,
    train_pose_classifier,
)


ML_TERMINAL_STATUSES = frozenset({"success", "error"})


@dataclass(frozen=True)
class PoseClassifierJobRequest:
    actor_username: str
    action: str
    dry_run: bool = False
    stored_filename: str = ""
    min_samples: int = 10


class PoseClassifierJobs:
    def __init__(
        self,
        *,
        repo_root: Path,
        database_dir: Path,
        processed_dir: Path,
        videos_file: Path,
        evaluations_file: Path,
    ) -> None:
        self.repo_root = repo_root
        self.database_dir = database_dir
        self.processed_dir = processed_dir
        self.videos_file = videos_file
        self.evaluations_file = evaluations_file
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, PoseClassifierJobRequest, float]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running_job_id = ""
        self._queued_job_ids: set[str] = set()

    def configure(
        self,
        *,
        repo_root: Path,
        database_dir: Path,
        processed_dir: Path,
        videos_file: Path,
        evaluations_file: Path,
    ) -> None:
        self.repo_root = repo_root
        self.database_dir = database_dir
        self.processed_dir = processed_dir
        self.videos_file = videos_file
        self.evaluations_file = evaluations_file

    def latest_file(self) -> Path:
        return self.processed_dir / "pose_classifier_job_latest.json"

    def history_file(self) -> Path:
        return self.processed_dir / "pose_classifier_job_history.json"

    def read_latest(self) -> dict[str, Any] | None:
        data = read_json(self.latest_file(), None)
        return data if isinstance(data, dict) else None

    def read_history(self) -> list[dict[str, Any]]:
        data = read_json(self.history_file(), [])
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def public_model_status(self) -> dict[str, Any]:
        raw = get_pose_classifier_status(str(self.database_dir))
        model_path, features_path = get_model_paths(str(self.database_dir))
        feature_info = read_json(features_path, {}) if Path(features_path).exists() else {}
        feature_cols = feature_info.get("feature_cols") if isinstance(feature_info, dict) else []
        labels = feature_info.get("label_names") if isinstance(feature_info, dict) else {}
        checksum = read_sha256_sidecar(model_path) if Path(model_path).exists() else None
        return {
            "ready": bool(raw.get("ready")),
            "checksum_ok": bool(raw.get("checksum_ok")),
            "checksum_required": True,
            "model_path": self._public_path(model_path),
            "features_path": self._public_path(features_path),
            "checksum_path": self._public_path(checksum_sidecar_path(model_path)),
            "checksum": checksum,
            "model_mtime": raw.get("model_mtime"),
            "feature_count": len(feature_cols) if isinstance(feature_cols, list) else 0,
            "labels": labels if isinstance(labels, dict) else {},
        }

    def start(self, request: PoseClassifierJobRequest) -> dict[str, Any]:
        with self._lock:
            current = self.read_latest()
            if self._running_job_id or self._queued_job_ids:
                return {"started": False, "reason": "already_running", "job": current}
            job_id = self._new_job_id(request.action)
            started_at = time.time()
            self._queued_job_ids.add(job_id)
            job = self._write_job(
                job_id,
                request,
                status="queued",
                progress=0.01,
                status_msg="Da nhan yeu cau ML, dang cho worker.",
                started_at=started_at,
            )
            self._ensure_worker_locked()
            self._queue.put((job_id, request, started_at))
            return {"started": True, "reason": "", "job": job}

    def _ensure_worker_locked(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="rehab-pose-classifier-worker",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            job_id, request, started_at = self._queue.get()
            with self._lock:
                self._queued_job_ids.discard(job_id)
                self._running_job_id = job_id
            try:
                self._run_job(job_id, request, started_at)
            finally:
                with self._lock:
                    if self._running_job_id == job_id:
                        self._running_job_id = ""
                self._queue.task_done()

    def _run_job(self, job_id: str, request: PoseClassifierJobRequest, started_at: float) -> None:
        try:
            self._write_job(
                job_id,
                request,
                status="processing",
                progress=0.10,
                status_msg="Worker ML dang xu ly.",
                started_at=started_at,
            )
            if request.action == "train":
                result = self._run_train_job(job_id, request, started_at)
            elif request.action == "apply":
                result = self._run_apply_job(job_id, request, started_at)
            else:
                result = {"success": False, "message": "unsupported ML action"}

            ok = bool(result.get("success"))
            self._write_job(
                job_id,
                request,
                status="success" if ok else "error",
                progress=1.0,
                status_msg=str(result.get("message") or "ML job da hoan tat."),
                error_msg="" if ok else str(result.get("message") or "ML job failed"),
                result=result,
                started_at=started_at,
            )
        except Exception as exc:
            self._write_job(
                job_id,
                request,
                status="error",
                progress=1.0,
                status_msg="ML job gap loi.",
                error_msg=str(exc),
                started_at=started_at,
            )

    def _run_train_job(
        self,
        job_id: str,
        request: PoseClassifierJobRequest,
        started_at: float,
    ) -> dict[str, Any]:
        self._write_job(
            job_id,
            request,
            status="processing",
            progress=0.25,
            status_msg="Dang doc CSV training.",
            started_at=started_at,
        )
        if request.dry_run:
            X, y, summary = load_training_data(str(self.processed_dir))
            samples = len(X) if X is not None else 0
            label_count = int(y.nunique()) if y is not None else 0
            can_train = bool(X is not None and y is not None and samples >= request.min_samples and label_count >= 2)
            return {
                "success": True,
                "dry_run": True,
                "can_train": can_train,
                "message": "Dry-run train hoan tat.",
                "min_samples": request.min_samples,
                "samples": samples,
                "label_count": label_count,
                **self._public_result(summary),
            }

        result = train_pose_classifier(
            processed_dir=str(self.processed_dir),
            db_dir=str(self.database_dir),
            min_samples=request.min_samples,
        )
        self._write_job(
            job_id,
            request,
            status="processing",
            progress=0.90,
            status_msg="Da train xong, dang kiem tra checksum.",
            result=self._public_result(result),
            started_at=started_at,
        )
        return self._public_result(result)

    def _run_apply_job(
        self,
        job_id: str,
        request: PoseClassifierJobRequest,
        started_at: float,
    ) -> dict[str, Any]:
        if not request.stored_filename:
            return {"success": False, "message": "stored_filename is required"}
        status = get_pose_classifier_status(str(self.database_dir))
        if not status.get("ready"):
            return {
                "success": False,
                "message": "Chua co model pose classifier hop le.",
                "model_status": self.public_model_status(),
            }

        records = read_json(self.videos_file, [])
        records = [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []
        record_index = next(
            (index for index, record in enumerate(records) if self._record_media_filename(record) == request.stored_filename),
            -1,
        )
        if record_index < 0:
            return {"success": False, "message": "video not found"}
        record = records[record_index]
        csv_path = resolve_local_path(
            str(record.get("df_path") or ""),
            str(self.repo_root),
            str(self.processed_dir),
            str(self.database_dir),
        )
        if not csv_path:
            return {"success": False, "message": "Khong tim thay CSV da phan tich cho video."}

        import pandas as pd

        self._write_job(
            job_id,
            request,
            status="processing",
            progress=0.35,
            status_msg="Dang ap dung classifier len CSV.",
            started_at=started_at,
        )
        df = pd.read_csv(csv_path)
        predicted_df, ml_result = apply_classifier_to_dataframe(
            df,
            db_dir=str(self.database_dir),
            phase_bounds_fn=segment_codman_frames,
            exercise_name=str(record.get("exercise") or ""),
        )

        frames_json_path = resolve_local_path(
            str(record.get("all_frames_data_path") or ""),
            str(self.repo_root),
            str(self.processed_dir),
            str(self.database_dir),
        )
        model_path, features_path = get_model_paths(str(self.database_dir))
        would_read = [csv_path, model_path, features_path]
        would_write = [csv_path, str(self.videos_file), str(self.evaluations_file)]
        if frames_json_path:
            would_read.append(frames_json_path)
            would_write.append(frames_json_path)

        if request.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message": "Dry-run apply hoan tat.",
                "video": self._video_summary(record),
                "ml_result": self._public_result(ml_result),
                "would_read": self._public_paths(would_read),
                "would_write": self._public_paths(would_write),
            }

        self._write_job(
            job_id,
            request,
            status="processing",
            progress=0.72,
            status_msg="Dang ghi CSV/frame JSON va metadata ML.",
            started_at=started_at,
        )
        predicted_df.to_csv(csv_path, index=False)
        frames_updated = self._update_frames_json(frames_json_path, predicted_df) if frames_json_path else 0
        if frames_json_path:
            refresh_saved_frame_labels(
                frames_json_path,
                data_dir=str(self.repo_root),
                processed_dir=str(self.processed_dir),
                db_dir=str(self.database_dir),
            )
        videos_updated = self._update_video_metadata(request.stored_filename, ml_result)
        evaluations_updated = self._update_evaluation_metadata(record, ml_result)
        return {
            "success": True,
            "dry_run": False,
            "message": "Da ap dung pose classifier cho video.",
            "video": self._video_summary(record),
            "ml_result": self._public_result(ml_result),
            "updated": videos_updated,
            "evaluations_updated": evaluations_updated,
            "frames_updated": frames_updated,
            "would_read": self._public_paths(would_read),
            "would_write": self._public_paths(would_write),
        }

    def _update_frames_json(self, frames_json_path: str, predicted_df: Any) -> int:
        frame_data = read_json(frames_json_path, [])
        if not isinstance(frame_data, list):
            return 0
        ml_cols = [
            "ml_label",
            "ml_label_text",
            "ml_score",
            "ml_confidence",
            "dung_ml",
            "gan_dung_ml",
            *ML_PROB_KEYS.values(),
        ]
        updated = 0
        for idx, frame_item in enumerate(frame_data[: len(predicted_df)]):
            if not isinstance(frame_item, dict):
                continue
            row = predicted_df.iloc[idx]
            for col in ml_cols:
                if col in predicted_df.columns:
                    frame_item[col] = self._json_safe_scalar(row[col])
            probabilities: dict[str, Any] = {}
            for cls_id, col_name in ML_PROB_KEYS.items():
                if col_name in predicted_df.columns:
                    probabilities[ML_LABEL_DISPLAY.get(cls_id, str(cls_id))] = self._json_safe_scalar(row[col_name])
            if probabilities:
                frame_item["ml_probabilities"] = probabilities
            updated += 1
        write_json(frames_json_path, frame_data)
        return updated

    def _update_video_metadata(self, stored_filename: str, ml_result: dict[str, Any]) -> int:
        updated_count = 0

        def _update(current: Any) -> list[dict[str, Any]]:
            nonlocal updated_count
            records = [record for record in (current if isinstance(current, list) else []) if isinstance(record, dict)]
            for record in records:
                if self._record_media_filename(record) != stored_filename:
                    continue
                phases = ml_result.get("ml_phases") if isinstance(ml_result, dict) else {}
                if isinstance(phases, dict):
                    record["ml_accuracy"] = phases.get("overall", 0.0)
                record["metrics"] = merge_ml_metrics(record.get("metrics", {}), ml_result)
                updated_count += 1
                break
            return records

        update_app_json(self.videos_file, _update, default=[])
        return updated_count

    def _update_evaluation_metadata(self, video_record: dict[str, Any], ml_result: dict[str, Any]) -> int:
        updated_count = 0
        phases = ml_result.get("ml_phases") if isinstance(ml_result, dict) else {}
        if not isinstance(phases, dict):
            phases = {}

        def _update(current: Any) -> list[dict[str, Any]]:
            nonlocal updated_count
            records = [record for record in (current if isinstance(current, list) else []) if isinstance(record, dict)]
            for record in records:
                same_patient = (
                    record.get("patient_username") == video_record.get("username")
                    or record.get("patient_username") == video_record.get("patient_username")
                    or record.get("patient_username") == video_record.get("full_name")
                )
                same_video = self._path_basename(record.get("video_name")) == self._path_basename(
                    video_record.get("video_name")
                )
                if same_patient and same_video and record.get("doctor_username") == "AI_Researcher":
                    record["ml_accuracy"] = phases.get("overall", 0.0)
                    record["ml_accuracy_g1"] = phases.get("g1", phases.get("overall", 0.0))
                    record["ml_accuracy_g2"] = phases.get("g2", phases.get("overall", 0.0))
                    record["ml_accuracy_g3"] = phases.get("g3", phases.get("overall", 0.0))
                    updated_count += 1
            return records

        update_app_json(self.evaluations_file, _update, default=[])
        return updated_count

    def _write_job(
        self,
        job_id: str,
        request: PoseClassifierJobRequest,
        *,
        status: str,
        progress: float,
        status_msg: str = "",
        error_msg: str = "",
        result: dict[str, Any] | None = None,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        start_time = started_at or time.time()
        data = {
            "job_id": job_id,
            "action": request.action,
            "status": status,
            "progress": max(0.0, min(1.0, float(progress or 0.0))),
            "requested_by": request.actor_username,
            "dry_run": bool(request.dry_run),
            "stored_filename": request.stored_filename,
            "min_samples": request.min_samples,
            "start_time": start_time,
            "elapsed": max(0.0, time.time() - start_time),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status_msg": self._sanitize_message(status_msg),
            "error_msg": self._sanitize_message(error_msg),
            "result": self._public_result(result or {}),
            "model_status": self.public_model_status(),
        }
        write_json(self.latest_file(), data)
        self._write_history_entry(data)
        return data

    def _write_history_entry(self, data: dict[str, Any]) -> None:
        job_id = str(data.get("job_id") or "")
        if not job_id:
            return

        def _update(current: Any) -> list[dict[str, Any]]:
            items = [item for item in (current if isinstance(current, list) else []) if isinstance(item, dict)]
            next_items: list[dict[str, Any]] = []
            replaced = False
            for item in items:
                if item.get("job_id") == job_id:
                    next_items.append(data)
                    replaced = True
                else:
                    next_items.append(item)
            if not replaced:
                next_items.append(data)
            return next_items[-50:]

        update_json(self.history_file(), _update, default=[])

    def _new_job_id(self, action: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"pose_{action}_{stamp}_{uuid.uuid4().hex[:8]}"

    def _public_result(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._public_result(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._public_result(item) for item in value]
        if isinstance(value, tuple):
            return [self._public_result(item) for item in value]
        if isinstance(value, (str, os.PathLike)):
            return self._public_path(str(value))
        return self._json_safe_scalar(value)

    def _public_paths(self, paths: list[str]) -> list[str]:
        return sorted({self._public_path(path) for path in paths if path})

    def _public_path(self, value: str | os.PathLike[str] | None) -> str:
        if not value:
            return ""
        text = str(value)
        try:
            resolved = Path(text).resolve()
        except OSError:
            return text
        try:
            return str(resolved.relative_to(self.repo_root.resolve())).replace("\\", "/")
        except ValueError:
            try:
                return str(resolved.relative_to(self.database_dir.resolve())).replace("\\", "/")
            except ValueError:
                return resolved.name

    def _sanitize_message(self, message: str) -> str:
        sanitized = str(message or "")
        for root in {self.repo_root, self.database_dir, self.processed_dir}:
            try:
                resolved = str(Path(root).resolve())
            except OSError:
                resolved = str(root)
            if resolved:
                sanitized = sanitized.replace(resolved, "[path]")
                sanitized = sanitized.replace(resolved.replace("\\", "/"), "[path]")
        return sanitized

    def _video_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "username": record.get("username") or record.get("patient_username"),
            "video_name": record.get("video_name"),
            "stored_filename": self._record_media_filename(record),
            "exercise": record.get("exercise"),
        }

    def _record_media_filename(self, record: dict[str, Any]) -> str:
        for value in (
            record.get("stored_filename"),
            self._path_basename(record.get("video_path")),
            self._path_basename(record.get("processed_path")),
            record.get("video_name"),
        ):
            text = str(value or "").strip()
            if text:
                return text
        return ""

    def _path_basename(self, value: Any) -> str:
        return str(value or "").replace("\\", "/").split("/")[-1]

    def _json_safe_scalar(self, value: Any) -> Any:
        try:
            if value is None or value != value:
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value
