"""Backend analysis job progress helpers.

This module keeps the backend API contract for analysis jobs separate from the
legacy Streamlit orchestration. It uses the same progress_<md5>.json convention
as the Streamlit app so a future worker can pick up the same files.
"""

from __future__ import annotations

import hashlib
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from storage.json_store import read_json, update_json, write_json
from video.io import (
    build_background_upload_h264_command,
    ffprobe_video_codecs,
    ffprobe_video_has_readable_duration,
    final_h264_path,
    temp_h264_path,
)
from video.serving import allowed_media_file_path, video_media_allowed_roots
from video.validation import ALLOWED_UPLOAD_VIDEO_EXTENSIONS


TERMINAL_STATUSES = frozenset({"success", "error", "ready_for_ai_worker", "canceled"})
RUNNING_STATUSES = frozenset({"processing", "cancel_requested"})
INTERNAL_OPTION_KEYS = frozenset({"media_path", "analysis_input_path", "prep_result"})
ANALYSIS_STEPS = (
    ("validate_transcode", "Validate/transcode"),
    ("mediapipe_pass", "MediaPipe pass 1"),
    ("overlay_export", "Overlay/export"),
    ("artifact_persist", "Artifact/persist"),
)


@dataclass(frozen=True)
class AnalysisJobRequest:
    actor_username: str
    username: str
    video_name: str
    video_path: str
    exercise: str
    options: dict[str, Any]
    run_id: str = ""
    action: str = "start"


class BackendAnalysisJobs:
    def __init__(
        self,
        *,
        repo_root: Path,
        upload_dir: Path,
        processed_dir: Path | None = None,
        runner: Callable[[AnalysisJobRequest, Callable[..., None]], dict[str, Any]] | None = None,
        ai_runner: Callable[[AnalysisJobRequest, str, Callable[..., None]], dict[str, Any]] | None = None,
        result_handler: Callable[[AnalysisJobRequest, dict[str, Any]], None] | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
        ffmpeg_threads: int = 2,
        transcode_timeout_seconds: int = 1800,
    ) -> None:
        self.repo_root = repo_root
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir or repo_root / "processed_results"
        self.runner = runner or self._validate_transcode_runner
        self.ai_runner = ai_runner
        self.result_handler = result_handler
        self.command_runner = command_runner
        self.ffmpeg_threads = max(1, int(ffmpeg_threads or 1))
        self.transcode_timeout_seconds = max(30, int(transcode_timeout_seconds or 30))
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[AnalysisJobRequest, float]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running: dict[str, threading.Thread] = {}
        self._queued: dict[str, AnalysisJobRequest] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    def configure(self, *, repo_root: Path, upload_dir: Path, processed_dir: Path | None = None) -> None:
        self.repo_root = repo_root
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir or repo_root / "processed_results"

    def job_id_for_video_path(self, video_path: str | os.PathLike[str] | None) -> str:
        normalized = str(video_path or "").replace("\\", "/")
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def progress_file_for_video_path(self, video_path: str | os.PathLike[str] | None) -> Path:
        return self.processed_dir / f"progress_{self.job_id_for_video_path(video_path)}.json"

    def history_file_for_video_path(self, video_path: str | os.PathLike[str] | None) -> Path:
        return self.processed_dir / f"analysis_job_history_{self.job_id_for_video_path(video_path)}.json"

    def read_progress(self, video_path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
        progress_path = self.progress_file_for_video_path(video_path)
        data = read_json(progress_path, None)
        return data if isinstance(data, dict) else None

    def read_history(self, video_path: str | os.PathLike[str] | None) -> list[dict[str, Any]]:
        history_path = self.history_file_for_video_path(video_path)
        data = read_json(history_path, [])
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _new_run_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"run_{stamp}_{uuid.uuid4().hex[:8]}"

    def _public_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        public: dict[str, Any] = {}
        for key, value in (options or {}).items():
            if key in INTERNAL_OPTION_KEYS:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                public[key] = value
        return public

    def _sanitize_message(self, message: str) -> str:
        sanitized = str(message or "")
        for root in {self.repo_root, self.upload_dir, self.processed_dir}:
            try:
                resolved = str(Path(root).resolve())
            except OSError:
                resolved = str(root)
            if resolved:
                sanitized = sanitized.replace(resolved, "[path]")
                sanitized = sanitized.replace(resolved.replace("\\", "/"), "[path]")
        for token_marker in ("token=", "access_token=", "hf_"):
            if token_marker in sanitized:
                if token_marker == "hf_":
                    sanitized = sanitized.replace(token_marker, "hf_[redacted]")
                else:
                    head, _, _tail = sanitized.partition(token_marker)
                    sanitized = f"{head}{token_marker}[redacted]"
        return sanitized

    def _steps_for_status(self, status: str, progress: float) -> list[dict[str, str]]:
        if status == "success":
            return [{"key": key, "label": label, "status": "done"} for key, label in ANALYSIS_STEPS]
        if status in {"error", "canceled"}:
            active_index = 0
            if progress >= 0.92:
                active_index = 3
            elif progress >= 0.70:
                active_index = 2
            elif progress >= 0.42:
                active_index = 1
            steps = []
            for index, (key, label) in enumerate(ANALYSIS_STEPS):
                if index < active_index:
                    step_status = "done"
                elif index == active_index:
                    step_status = "error" if status == "error" else "canceled"
                else:
                    step_status = "pending"
                steps.append({"key": key, "label": label, "status": step_status})
            return steps
        if status == "ready_for_ai_worker":
            return [
                {"key": "validate_transcode", "label": "Validate/transcode", "status": "done"},
                {"key": "mediapipe_pass", "label": "MediaPipe pass 1", "status": "pending"},
                {"key": "overlay_export", "label": "Overlay/export", "status": "pending"},
                {"key": "artifact_persist", "label": "Artifact/persist", "status": "pending"},
            ]
        if progress >= 0.92:
            active_index = 3
        elif progress >= 0.70:
            active_index = 2
        elif progress >= 0.42:
            active_index = 1
        else:
            active_index = 0
        steps = []
        for index, (key, label) in enumerate(ANALYSIS_STEPS):
            if index < active_index:
                step_status = "done"
            elif index == active_index:
                step_status = "active"
            else:
                step_status = "pending"
            steps.append({"key": key, "label": label, "status": step_status})
        return steps

    def _write_history_entry(self, data: dict[str, Any]) -> None:
        run_id = str(data.get("run_id") or "")
        if not run_id:
            return
        history_path = self.history_file_for_video_path(data.get("video_path"))
        entry = {
            key: value
            for key, value in data.items()
            if key
            in {
                "job_id",
                "run_id",
                "video_path",
                "username",
                "video_name",
                "exercise",
                "status",
                "progress",
                "elapsed",
                "start_time",
                "heartbeat",
                "updated_at",
                "status_msg",
                "error_msg",
                "result",
                "job_meta",
                "steps",
            }
        }

        def _update(current: Any) -> list[dict[str, Any]]:
            items = [item for item in (current if isinstance(current, list) else []) if isinstance(item, dict)]
            updated: list[dict[str, Any]] = []
            replaced = False
            for item in items:
                if str(item.get("run_id") or "") == run_id:
                    old_status = str(item.get("status") or "")
                    new_status = str(entry.get("status") or "")
                    if old_status in TERMINAL_STATUSES and new_status not in TERMINAL_STATUSES:
                        updated.append(item)
                    else:
                        updated.append(entry)
                    replaced = True
                else:
                    updated.append(item)
            if not replaced:
                updated.append(entry)
            return updated[-50:]

        update_json(history_path, _update, default=[])

    def write_progress(
        self,
        request: AnalysisJobRequest,
        *,
        status: str,
        progress: float,
        status_msg: str = "",
        error_msg: str = "",
        result: dict[str, Any] | None = None,
        start_time: float | None = None,
        meta_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.read_progress(request.video_path) or {}
        run_id = request.run_id or str(existing.get("run_id") or "") or self._new_run_id()
        if (
            str(existing.get("run_id") or "") == run_id
            and str(existing.get("status") or "") in TERMINAL_STATUSES
            and status not in TERMINAL_STATUSES
        ):
            return existing
        started_at = start_time if start_time is not None else existing.get("start_time") or time.time()
        elapsed = max(0.0, time.time() - float(started_at or time.time()))
        progress_value = max(0.0, min(1.0, float(progress or 0.0)))
        previous_meta = existing.get("job_meta") if existing.get("run_id") == run_id and isinstance(existing.get("job_meta"), dict) else {}
        job_meta = {
            **previous_meta,
            "requested_by": previous_meta.get("requested_by") or request.actor_username,
            "options": previous_meta.get("options") or self._public_options(request.options),
            "action": previous_meta.get("action") or request.action,
        }
        if meta_updates:
            job_meta.update(meta_updates)
        data = {
            "job_id": self.job_id_for_video_path(request.video_path),
            "run_id": run_id,
            "video_path": request.video_path,
            "username": request.username,
            "video_name": request.video_name,
            "exercise": request.exercise,
            "status": status,
            "progress": progress_value,
            "elapsed": elapsed,
            "start_time": started_at,
            "heartbeat": time.time(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status_msg": status_msg,
            "error_msg": self._sanitize_message(error_msg),
            "result": result,
            "job_meta": job_meta,
            "steps": self._steps_for_status(status, progress_value),
        }
        write_json(self.progress_file_for_video_path(request.video_path), data)
        self._write_history_entry(data)
        return data

    def resolve_media_path(self, video_path: str | os.PathLike[str] | None) -> str | None:
        if not video_path:
            return None
        path = Path(str(video_path))
        if not path.is_absolute():
            path = self.repo_root / Path(*[part for part in str(video_path).replace("\\", "/").split("/") if part])
        roots = video_media_allowed_roots(
            data_dir=self.repo_root,
            upload_dir=self.upload_dir,
            processed_dir=self.processed_dir,
        )
        return allowed_media_file_path(path, roots, allowed_extensions=frozenset(ALLOWED_UPLOAD_VIDEO_EXTENSIONS))

    def is_running(self, video_path: str | os.PathLike[str] | None) -> bool:
        job_id = self.job_id_for_video_path(video_path)
        thread = self._running.get(job_id)
        return bool(job_id in self._queued or (thread and thread.is_alive()))

    def is_cancel_requested(self, video_path: str | os.PathLike[str] | None) -> bool:
        event = self._cancel_events.get(self.job_id_for_video_path(video_path))
        return bool(event and event.is_set())

    def _raise_if_canceled(self, request: AnalysisJobRequest) -> None:
        if self.is_cancel_requested(request.video_path):
            raise AnalysisJobCanceled("analysis job was canceled")

    def _ensure_worker_locked(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="rehab-analysis-job-worker",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            run_request, started_at = self._queue.get()
            job_id = self.job_id_for_video_path(run_request.video_path)
            with self._lock:
                queued_request = self._queued.get(job_id)
                if queued_request and queued_request.run_id == run_request.run_id:
                    self._queued.pop(job_id, None)
                self._running[job_id] = threading.current_thread()
            try:
                self._run_job(run_request, started_at)
            finally:
                with self._lock:
                    current = self._running.get(job_id)
                    if current is threading.current_thread():
                        self._running.pop(job_id, None)
                    self._cancel_events.pop(job_id, None)
                self._queue.task_done()

    def _run_job(self, run_request: AnalysisJobRequest, started_at: float) -> None:
        try:
            self._raise_if_canceled(run_request)
            self.write_progress(
                run_request,
                status="processing",
                progress=0.05,
                status_msg="Đang kiểm tra file video trên backend.",
                start_time=started_at,
            )
            media_path = self.resolve_media_path(run_request.video_path)
            if not media_path:
                self.write_progress(
                    run_request,
                    status="error",
                    progress=1.0,
                    status_msg="Không tìm thấy file video hợp lệ để phân tích.",
                    error_msg="video file is missing or outside allowed media roots",
                    start_time=started_at,
                )
                return

            def _progress(**kwargs: Any) -> None:
                self._raise_if_canceled(run_request)
                self.write_progress(run_request, start_time=started_at, **kwargs)

            runner_request = AnalysisJobRequest(
                actor_username=run_request.actor_username,
                username=run_request.username,
                video_name=run_request.video_name,
                video_path=run_request.video_path,
                exercise=run_request.exercise,
                options={**run_request.options, "media_path": media_path},
                run_id=run_request.run_id,
                action=run_request.action,
            )
            result = self.runner(runner_request, _progress) or {}
            if not isinstance(result, dict):
                result = {}
            self._raise_if_canceled(run_request)

            terminal_request = runner_request
            terminal_status = str(result.get("status") or "ready_for_ai_worker")
            terminal_result = result
            terminal_payload = result.get("result") if isinstance(result.get("result"), dict) else None
            if terminal_status == "ready_for_ai_worker" and self.ai_runner:
                analysis_input_path = str((terminal_payload or {}).get("analysis_input_path") or media_path)
                self.write_progress(
                    run_request,
                    status="processing",
                    progress=max(float(result.get("progress") or 0.0), 0.42),
                    status_msg="Video đã sẵn sàng, đang chạy worker AI.",
                    result=terminal_payload,
                    start_time=started_at,
                )
                terminal_request = AnalysisJobRequest(
                    actor_username=runner_request.actor_username,
                    username=runner_request.username,
                    video_name=runner_request.video_name,
                    video_path=runner_request.video_path,
                    exercise=runner_request.exercise,
                    options={
                        **runner_request.options,
                        "analysis_input_path": analysis_input_path,
                        "prep_result": result,
                    },
                    run_id=runner_request.run_id,
                    action=runner_request.action,
                )
                terminal_result = self.ai_runner(terminal_request, analysis_input_path, _progress) or {}
                if not isinstance(terminal_result, dict):
                    terminal_result = {}
                self._raise_if_canceled(run_request)
                terminal_status = str(terminal_result.get("status") or "success")
                terminal_payload = (
                    terminal_result.get("result") if isinstance(terminal_result.get("result"), dict) else None
                )

            terminal_progress = 1.0 if terminal_status in {"success", "error"} else 0.12
            if terminal_status == "success" and terminal_payload and self.result_handler:
                self.result_handler(terminal_request, terminal_payload)
            self.write_progress(
                run_request,
                status=terminal_status,
                progress=float(terminal_result.get("progress", terminal_progress)),
                status_msg=str(
                    terminal_result.get("status_msg") or "Video đã sẵn sàng, đang chờ worker AI/transcode."
                ),
                error_msg=str(terminal_result.get("error_msg") or ""),
                result=terminal_payload,
                start_time=started_at,
            )
        except AnalysisJobCanceled:
            current = self.read_progress(run_request.video_path) or {}
            if str(current.get("status") or "") != "canceled":
                self.write_progress(
                    run_request,
                    status="canceled",
                    progress=float(current.get("progress") or 0.0),
                    status_msg="Job phân tích đã được hủy.",
                    result=current.get("result") if isinstance(current.get("result"), dict) else None,
                    start_time=started_at,
                )
        except Exception as exc:
            self.write_progress(
                run_request,
                status="error",
                progress=1.0,
                status_msg="Lỗi khi chuẩn bị job phân tích.",
                error_msg=str(exc),
                start_time=started_at,
            )

    def cancel(self, request: AnalysisJobRequest, *, canceled_by: str) -> dict[str, Any]:
        job_id = self.job_id_for_video_path(request.video_path)
        current = self.read_progress(request.video_path)
        if not current:
            return {"ok": False, "reason": "not_found", "job": None}
        if str(current.get("status") or "") in TERMINAL_STATUSES and not self.is_running(request.video_path):
            return {"ok": False, "reason": "not_running", "job": current}

        event = self._cancel_events.get(job_id)
        if event is None:
            event = threading.Event()
            self._cancel_events[job_id] = event
        event.set()

        run_request = replace(
            request,
            run_id=str(current.get("run_id") or ""),
            action=str((current.get("job_meta") or {}).get("action") or request.action),
            options=(current.get("job_meta") or {}).get("options") if isinstance(current.get("job_meta"), dict) else request.options,
        )
        job = self.write_progress(
            run_request,
            status="canceled",
            progress=float(current.get("progress") or 0.0),
            status_msg="Job phân tích đã được hủy.",
            start_time=float(current.get("start_time") or time.time()),
            result=current.get("result") if isinstance(current.get("result"), dict) else None,
            meta_updates={"canceled_by": canceled_by, "canceled_at": datetime.now().isoformat(timespec="seconds")},
        )
        return {"ok": True, "reason": "", "job": job}

    def start(self, request: AnalysisJobRequest) -> dict[str, Any]:
        job_id = self.job_id_for_video_path(request.video_path)
        with self._lock:
            thread = self._running.get(job_id)
            if job_id in self._queued or (thread and thread.is_alive()):
                current = self.read_progress(request.video_path) or {}
                return {
                    "started": False,
                    "reason": "already_running",
                    "job": current,
                }

            started_at = time.time()
            run_request = replace(request, run_id=request.run_id or self._new_run_id())
            cancel_event = threading.Event()
            self._cancel_events[job_id] = cancel_event
            job = self.write_progress(
                run_request,
                status="processing",
                progress=0.01,
                status_msg="Đã nhận yêu cầu phân tích, đang chuẩn bị video.",
                start_time=started_at,
            )
            self._queued[job_id] = run_request
            self._ensure_worker_locked()
            self._queue.put((run_request, started_at))
            return {"started": True, "reason": "", "job": job}

    def _validate_transcode_runner(
        self,
        request: AnalysisJobRequest,
        progress: Callable[..., None],
    ) -> dict[str, Any]:
        media_path = str(request.options.get("media_path") or self.resolve_media_path(request.video_path) or "")
        if not media_path:
            return {
                "status": "error",
                "progress": 1.0,
                "status_msg": "Không tìm thấy file video để chuẩn bị phân tích.",
                "error_msg": "media path is missing",
            }

        self._raise_if_canceled(request)
        progress(
            status="processing",
            progress=0.08,
            status_msg="Đang đọc metadata video bằng ffprobe.",
        )
        if not ffprobe_video_has_readable_duration(media_path, runner=self.command_runner, timeout=10):
            return {
                "status": "error",
                "progress": 1.0,
                "status_msg": "Không đọc được metadata video.",
                "error_msg": "ffprobe could not read video duration",
            }

        self._raise_if_canceled(request)
        video_codec, audio_codec = ffprobe_video_codecs(media_path, runner=self.command_runner, timeout=10)
        ext = Path(media_path).suffix.lower()
        is_h264_mp4 = ext == ".mp4" and video_codec == "h264"
        if is_h264_mp4:
            progress(
                status="processing",
                progress=0.18,
                status_msg="Video đã là MP4/H.264, sẵn sàng cho worker AI.",
            )
            return {
                "status": "ready_for_ai_worker",
                "progress": 0.22,
                "status_msg": "Video đã sẵn sàng cho worker AI.",
                "result": {
                    "analysis_input_path": media_path,
                    "transcoded": False,
                    "video_codec": video_codec,
                    "audio_codec": audio_codec,
                },
            }

        self._raise_if_canceled(request)
        output_path = final_h264_path(media_path)
        if not output_path:
            return {
                "status": "error",
                "progress": 1.0,
                "status_msg": "Không tạo được đường dẫn H.264 đầu ra.",
                "error_msg": "invalid H.264 output path",
            }

        temp_output_path = temp_h264_path(output_path)
        try:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except OSError:
            pass

        progress(
            status="processing",
            progress=0.24,
            status_msg="Đang chuyển video sang MP4/H.264.",
        )
        self._raise_if_canceled(request)
        cmd = build_background_upload_h264_command(
            media_path,
            temp_output_path,
            ffmpeg_threads=self.ffmpeg_threads,
        )
        result = self.command_runner(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.transcode_timeout_seconds,
        )
        if result.returncode != 0:
            return {
                "status": "error",
                "progress": 1.0,
                "status_msg": "FFmpeg không chuyển mã được video.",
                "error_msg": str(getattr(result, "stderr", "") or "ffmpeg failed"),
            }
        self._raise_if_canceled(request)
        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) <= 0:
            return {
                "status": "error",
                "progress": 1.0,
                "status_msg": "FFmpeg không tạo file đầu ra hợp lệ.",
                "error_msg": "transcoded output is missing or empty",
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_output_path, output_path)
        progress(
            status="processing",
            progress=0.34,
            status_msg="Đã tạo MP4/H.264, đang kiểm tra file đầu ra.",
        )
        out_video_codec, out_audio_codec = ffprobe_video_codecs(output_path, runner=self.command_runner, timeout=10)
        if out_video_codec != "h264":
            return {
                "status": "error",
                "progress": 1.0,
                "status_msg": "File sau transcode chưa phải H.264.",
                "error_msg": f"unexpected output codec: {out_video_codec or 'unknown'}",
            }

        return {
            "status": "ready_for_ai_worker",
            "progress": 0.40,
            "status_msg": "Video H.264 đã sẵn sàng cho worker AI.",
            "result": {
                "analysis_input_path": output_path,
                "transcoded": True,
                "source_path": media_path,
                "video_codec": out_video_codec,
                "audio_codec": out_audio_codec,
            },
        }


class AnalysisJobCanceled(RuntimeError):
    pass
