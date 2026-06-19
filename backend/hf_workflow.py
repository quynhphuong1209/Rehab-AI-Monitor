"""Guarded Hugging Face Dataset sync/report workflow for the backend."""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from cloud.hf_sync import (
    HF_JSON_DOWNLOAD_FILES,
    HfPathPolicy,
    download_dataset_file_bytes,
    hf_token_fingerprint,
    list_dataset_files,
    upload_dataset_file,
    verify_dataset_via_http,
)
from scripts.sync_data_and_report import generate_report
from storage.json_store import read_json, update_json, write_json


HF_TERMINAL_STATUSES = frozenset({"success", "error"})
HF_SAFE_SYNC_FILES = tuple(sorted(HF_JSON_DOWNLOAD_FILES - {"users.json"}))


@dataclass(frozen=True)
class HfWorkflowRequest:
    actor_username: str
    action: str
    dry_run: bool = True
    files: tuple[str, ...] = ()
    stored_filename: str = ""
    artifact_kind: str = ""
    local_path: str = ""
    report_format: str = "markdown"
    request_get: Callable[..., Any] | None = field(default=None, compare=False)
    list_repo_files_fn: Callable[..., Any] | None = field(default=None, compare=False)
    api_factory: Callable[..., Any] | None = field(default=None, compare=False)


class HfWorkflowJobs:
    def __init__(
        self,
        *,
        repo_root: Path,
        database_dir: Path,
        upload_dir: Path,
        processed_dir: Path,
        token: str = "",
        dataset_id: str = "",
    ) -> None:
        self.repo_root = repo_root
        self.database_dir = database_dir
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        self.token = token
        self.dataset_id = dataset_id
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, HfWorkflowRequest, float]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running_job_id = ""
        self._queued_job_ids: set[str] = set()

    def configure(
        self,
        *,
        repo_root: Path,
        database_dir: Path,
        upload_dir: Path,
        processed_dir: Path,
        token: str = "",
        dataset_id: str = "",
    ) -> None:
        self.repo_root = repo_root
        self.database_dir = database_dir
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        self.token = token
        self.dataset_id = dataset_id

    def policy(self) -> HfPathPolicy:
        return HfPathPolicy(
            data_dir=str(self.repo_root),
            upload_dir=str(self.upload_dir),
            processed_dir=str(self.processed_dir),
            db_dir=str(self.database_dir),
        )

    def latest_file(self) -> Path:
        return self.processed_dir / "hf_sync_job_latest.json"

    def history_file(self) -> Path:
        return self.processed_dir / "hf_sync_job_history.json"

    def read_latest(self) -> dict[str, Any] | None:
        data = read_json(self.latest_file(), None)
        return data if isinstance(data, dict) else None

    def read_history(self) -> list[dict[str, Any]]:
        data = read_json(self.history_file(), [])
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def public_status(self, *, verify: bool = False, list_files_flag: bool = False) -> dict[str, Any]:
        ok = False
        message = ""
        files: list[str] = []
        if verify and self.token and self.dataset_id:
            ok, err = verify_dataset_via_http(self.token, self.dataset_id)
            message = err or ""
        elif verify:
            message = "Chua cau hinh HF_TOKEN hoac HF_DATASET_ID."
        if list_files_flag and self.token and self.dataset_id:
            listed, err = list_dataset_files(self.token, self.dataset_id)
            if listed is not None:
                files = [item for item in listed if _safe_file_name(item) in HF_SAFE_SYNC_FILES or item.startswith(("patient_uploads/", "processed_results/"))][:200]
            elif err:
                message = err
        return {
            "configured": bool(self.dataset_id),
            "token_configured": bool(self.token),
            "dataset_id": self.dataset_id,
            "token_fingerprint": hf_token_fingerprint(self.token, self.dataset_id) if self.token or self.dataset_id else "",
            "verify_ok": bool(ok),
            "message": _sanitize_message(message, self.repo_root, self.database_dir, self.processed_dir),
            "allowed_sync_files": list(HF_SAFE_SYNC_FILES),
            "files": files,
        }

    def start(self, request: HfWorkflowRequest) -> dict[str, Any]:
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
                status_msg="Da nhan yeu cau HF, dang cho worker.",
                started_at=started_at,
            )
            self._ensure_worker_locked()
            self._queue.put((job_id, request, started_at))
            return {"started": True, "reason": "", "job": job}

    def _ensure_worker_locked(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="rehab-hf-sync-worker")
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

    def _run_job(self, job_id: str, request: HfWorkflowRequest, started_at: float) -> None:
        try:
            self._write_job(
                job_id,
                request,
                status="processing",
                progress=0.10,
                status_msg="Worker HF dang xu ly.",
                started_at=started_at,
            )
            if not self.dataset_id:
                result = {"success": False, "message": "Chua cau hinh HF_DATASET_ID."}
            elif request.action == "sync":
                result = self._sync_metadata(job_id, request, started_at)
            elif request.action == "upload":
                result = self._upload_artifact(job_id, request, started_at)
            elif request.action == "report":
                result = self._generate_report(job_id, request, started_at)
            else:
                result = {"success": False, "message": "unsupported HF action"}
            ok = bool(result.get("success"))
            self._write_job(
                job_id,
                request,
                status="success" if ok else "error",
                progress=1.0,
                status_msg=str(result.get("message") or "HF job da hoan tat."),
                error_msg="" if ok else str(result.get("message") or "HF job failed"),
                result=result,
                started_at=started_at,
            )
        except Exception as exc:
            self._write_job(
                job_id,
                request,
                status="error",
                progress=1.0,
                status_msg="HF job gap loi.",
                error_msg=str(exc),
                started_at=started_at,
            )

    def _sync_metadata(self, job_id: str, request: HfWorkflowRequest, started_at: float) -> dict[str, Any]:
        requested_files = [item for item in request.files if item] or list(HF_SAFE_SYNC_FILES)
        files = [item for item in requested_files if item in HF_SAFE_SYNC_FILES]
        skipped = [item for item in requested_files if item not in HF_SAFE_SYNC_FILES]
        results = []
        for index, rel_path in enumerate(files):
            self._write_job(
                job_id,
                request,
                status="processing",
                progress=0.15 + (0.75 * (index / max(1, len(files)))),
                status_msg=f"Dang sync {rel_path}.",
                result={"completed": results, "skipped": skipped},
                started_at=started_at,
            )
            raw, err = download_dataset_file_bytes(
                rel_path,
                token=self.token,
                dataset_id=self.dataset_id,
                request_get=request.request_get,
            )
            if raw is None:
                results.append({"file": rel_path, "status": "error", "message": err or "download failed"})
                continue
            try:
                remote_data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                results.append({"file": rel_path, "status": "error", "message": f"invalid JSON: {exc}"})
                continue
            local_path = self.database_dir / rel_path
            local_before = read_json(local_path, [] if isinstance(remote_data, list) else {})
            summary = _merge_json_summary(local_before, remote_data)
            backup_path = "" if request.dry_run else self._backup_file(local_path, action="hf_sync", target=rel_path)
            if not request.dry_run:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(local_path, remote_data, indent=2)
            results.append(
                {
                    "file": rel_path,
                    "status": "dry_run" if request.dry_run else "updated",
                    "local_path": self._public_path(local_path),
                    "backup_path": self._public_path(backup_path),
                    **summary,
                }
            )
        return {
            "success": True,
            "dry_run": request.dry_run,
            "message": "HF metadata sync hoan tat.",
            "results": results,
            "skipped": skipped,
        }

    def _upload_artifact(self, job_id: str, request: HfWorkflowRequest, started_at: float) -> dict[str, Any]:
        if not request.local_path:
            return {"success": False, "message": "local artifact path is required"}
        self._write_job(
            job_id,
            request,
            status="processing",
            progress=0.45,
            status_msg="Dang kiem tra artifact upload.",
            started_at=started_at,
        )
        if request.dry_run:
            try:
                from cloud.hf_sync import hf_upload_rel_path_for_local

                rel_path = hf_upload_rel_path_for_local(request.local_path, self.policy())
                return {
                    "success": True,
                    "dry_run": True,
                    "message": "Dry-run upload hoan tat.",
                    "local_path": self._public_path(request.local_path),
                    "path_in_repo": rel_path,
                }
            except Exception as exc:
                return {"success": False, "dry_run": True, "message": str(exc)}
        rel_path, err = upload_dataset_file(
            request.local_path,
            token=self.token,
            dataset_id=self.dataset_id,
            policy=self.policy(),
            api_factory=request.api_factory,
        )
        return {
            "success": bool(rel_path and not err),
            "dry_run": False,
            "message": "Da upload artifact len HF Dataset." if rel_path and not err else (err or "upload failed"),
            "local_path": self._public_path(request.local_path),
            "path_in_repo": rel_path or "",
        }

    def _generate_report(self, job_id: str, request: HfWorkflowRequest, started_at: float) -> dict[str, Any]:
        del job_id, started_at
        content = generate_report(dataset_id=self.dataset_id, synced=False)
        output = self.repo_root / "docs" / "generated" / "sync_report.md"
        if not request.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "dry_run": request.dry_run,
            "message": "Da tao bao cao sanitize." if not request.dry_run else "Dry-run report hoan tat.",
            "output_path": self._public_path(output),
            "preview": content[:1200],
        }

    def _write_job(
        self,
        job_id: str,
        request: HfWorkflowRequest,
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
            "files": list(request.files),
            "stored_filename": request.stored_filename,
            "artifact_kind": request.artifact_kind,
            "start_time": start_time,
            "elapsed": max(0.0, time.time() - start_time),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status_msg": _sanitize_message(status_msg, self.repo_root, self.database_dir, self.processed_dir),
            "error_msg": _sanitize_message(error_msg, self.repo_root, self.database_dir, self.processed_dir),
            "result": self._public_result(result or {}),
            "hf_status": self.public_status(),
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
            updated = []
            replaced = False
            for item in items:
                if item.get("job_id") == job_id:
                    updated.append(data)
                    replaced = True
                else:
                    updated.append(item)
            if not replaced:
                updated.append(data)
            return updated[-50:]

        update_json(self.history_file(), _update, default=[])

    def _backup_file(self, path: Path, *, action: str, target: str) -> str:
        if not path.exists():
            return ""
        backup_dir = self.repo_root / "backups" / "hf_sync"
        backup_dir.mkdir(parents=True, exist_ok=True)
        safe_target = _safe_file_name(target).replace(".", "_") or "target"
        backup_path = backup_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{action}_{safe_target}_{path.name}"
        shutil.copy2(path, backup_path)
        return str(backup_path)

    def _public_result(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._public_result(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._public_result(item) for item in value]
        if isinstance(value, tuple):
            return [self._public_result(item) for item in value]
        if isinstance(value, (str, os.PathLike)):
            return self._public_path(value)
        return value

    def _public_path(self, value: str | os.PathLike[str] | None) -> str:
        if not value:
            return ""
        text = str(value)
        try:
            resolved = Path(text).resolve()
            return str(resolved.relative_to(self.repo_root.resolve())).replace("\\", "/")
        except (OSError, ValueError):
            return text.replace("\\", "/")

    def _new_job_id(self, action: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"hf_{action}_{stamp}_{uuid.uuid4().hex[:8]}"


def _merge_json_summary(local_data: Any, remote_data: Any) -> dict[str, Any]:
    local_count = len(local_data) if isinstance(local_data, (list, dict)) else 0
    remote_count = len(remote_data) if isinstance(remote_data, (list, dict)) else 0
    return {
        "local_count": local_count,
        "remote_count": remote_count,
        "changed": local_data != remote_data,
    }


def _safe_file_name(value: Any) -> str:
    return Path(str(value or "").replace("\\", "/")).name


def _sanitize_message(message: str, *roots: Path) -> str:
    sanitized = str(message or "")
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved:
            sanitized = sanitized.replace(resolved, "[path]")
            sanitized = sanitized.replace(resolved.replace("\\", "/"), "[path]")
    for marker in ("token=", "access_token=", "hf_"):
        if marker in sanitized:
            if marker == "hf_":
                sanitized = sanitized.replace(marker, "hf_[redacted]")
            else:
                head, _, _tail = sanitized.partition(marker)
                sanitized = f"{head}{marker}[redacted]"
    return sanitized
