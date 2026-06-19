"""Hugging Face Dataset sync helpers.

This module centralizes Hugging Face token handling, low-level network calls,
and security-sensitive Dataset path allowlist logic.
"""

from __future__ import annotations

import os
import hashlib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from utils.path_security import (
    PathSecurityError,
    normalize_relative_path,
    path_is_within,
    safe_data_path,
)


class HfSyncError(RuntimeError):
    """Raised for Hugging Face service operations that should be handled at UI/script boundaries."""


HF_JSON_DOWNLOAD_FILES = frozenset(
    {
        "patient_symptoms.json",
        "doctor_evaluations.json",
        "schedules.json",
        "video_list.json",
        "research_data.json",
        "lich_su_tap_luyen.json",
        "phan_hoi.json",
    }
)

HF_MODEL_ARTIFACT_FILES = frozenset(
    {
        "pose_classifier.pkl",
        "pose_classifier.pkl.sha256",
        "pose_classifier_features.json",
    }
)

HF_LIBRARY_ERROR_MARKERS = (
    "cannot import name",
    "importerror",
    "no module named 'huggingface_hub'",
    "no module named huggingface_hub",
)

HF_AUTH_ERROR_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "permission",
    "credentials",
)

HF_NOT_FOUND_MARKERS = (
    "404",
    "not found",
    "entry not found",
)


@dataclass(frozen=True)
class HfPathPolicy:
    data_dir: str | os.PathLike[str]
    upload_dir: str | os.PathLike[str]
    processed_dir: str | os.PathLike[str]
    db_dir: str | os.PathLike[str]
    json_files: frozenset[str] = HF_JSON_DOWNLOAD_FILES
    model_artifact_files: frozenset[str] = HF_MODEL_ARTIFACT_FILES

    @property
    def media_roots(self) -> list[str | os.PathLike[str]]:
        return [self.upload_dir, self.processed_dir]

    @property
    def db_roots(self) -> list[str | os.PathLike[str]]:
        return [self.db_dir]


def data_allowed_roots(policy: HfPathPolicy) -> list[str | os.PathLike[str]]:
    return [policy.upload_dir, policy.processed_dir, policy.db_dir]


def hf_min_size_for_path(path: str | os.PathLike[str] | None) -> int:
    if not path:
        return 5 * 1024
    low = str(path).lower()
    if low.endswith(".csv"):
        return 80
    if low.endswith(".json"):
        return 2
    return 5 * 1024


def hf_token_fingerprint(token: str | None, dataset_id: str | None) -> str:
    return hashlib.md5(f"{token or ''}:{dataset_id or ''}".encode()).hexdigest()[:12]


def is_hf_library_error(err_text: object) -> bool:
    err = str(err_text or "").lower()
    return any(marker in err for marker in HF_LIBRARY_ERROR_MARKERS)


def is_hf_auth_error(err_text: object) -> bool:
    err = str(err_text or "").lower()
    return any(marker in err for marker in HF_AUTH_ERROR_MARKERS)


def is_hf_not_found_error(err_text: object) -> bool:
    err = str(err_text or "").lower()
    return any(marker in err for marker in HF_NOT_FOUND_MARKERS)


def hf_auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_dataset_file_url(dataset_id: str, rel_path: str | os.PathLike[str]) -> str:
    rel_norm = normalize_relative_path(rel_path)
    rel_enc = urllib.parse.quote(rel_norm, safe="/")
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{rel_enc}"


def hf_verify_dataset_status_message(status_code: int, dataset_id: str) -> str | None:
    if status_code in (401, 403):
        return (
            f"Token không có quyền đọc Dataset `{dataset_id}`. "
            "Hãy dùng token có phạm vi tối thiểu và thêm tài khoản chạy app làm collaborator nếu cần."
        )
    if status_code == 404:
        return (
            f"Không tìm thấy Dataset `{dataset_id}`. "
            "Kiểm tra cấu hình HF_DATASET_ID trong env hoặc Streamlit secrets."
        )
    if status_code != 200:
        return f"HTTP {status_code} khi kiểm tra Dataset."
    return None


def hf_download_status_error(status_code: int, rel_path: str | os.PathLike[str]) -> str | None:
    if status_code in (401, 403):
        return "Token không có quyền tải file từ Dataset."
    if status_code == 404:
        return f"Chưa có trên Dataset: `{rel_path}`"
    return None


def hf_repo_info(
    token: str | None,
    dataset_id: str | None,
    *,
    api_factory: Callable[..., Any] | None = None,
) -> tuple[bool, str | None]:
    if not (token and dataset_id):
        return False, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
    try:
        if api_factory is None:
            from huggingface_hub import HfApi

            api_factory = HfApi
        api = api_factory(token=token)
        api.repo_info(repo_id=dataset_id, repo_type="dataset")
        return True, None
    except Exception as exc:
        return False, str(exc)


def ensure_dataset_repo(
    token: str | None,
    dataset_id: str | None,
    *,
    private: bool = True,
    api_factory: Callable[..., Any] | None = None,
) -> tuple[bool, str | None]:
    if not (token and dataset_id):
        return False, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
    try:
        if api_factory is None:
            from huggingface_hub import HfApi

            api_factory = HfApi
        api = api_factory(token=token)
        api.create_repo(repo_id=dataset_id, repo_type="dataset", private=private, exist_ok=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def list_dataset_files(
    token: str | None,
    dataset_id: str | None,
    *,
    list_repo_files_fn: Callable[..., Any] | None = None,
) -> tuple[list[str] | None, str | None]:
    if not (token and dataset_id):
        return None, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
    try:
        if list_repo_files_fn is None:
            from huggingface_hub import list_repo_files

            list_repo_files_fn = list_repo_files
        files = list_repo_files_fn(dataset_id, repo_type="dataset", token=token)
        return list(files or []), None
    except Exception as exc:
        return None, str(exc)


def upload_dataset_file(
    local_path: str | os.PathLike[str],
    *,
    token: str | None,
    dataset_id: str | None,
    policy: HfPathPolicy,
    api_factory: Callable[..., Any] | None = None,
) -> tuple[str | None, str | None]:
    if not (token and dataset_id and local_path):
        return None, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
    try:
        rel_path = hf_upload_rel_path_for_local(local_path, policy)
        if api_factory is None:
            from huggingface_hub import HfApi

            api_factory = HfApi
        api = api_factory(token=token)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path,
            repo_id=dataset_id,
            repo_type="dataset",
            token=token,
        )
        return rel_path, None
    except PathSecurityError:
        return None, "Đường dẫn file không nằm trong thư mục dữ liệu được phép."
    except Exception as exc:
        return None, str(exc)


def download_dataset_file(
    rel_path: str | os.PathLike[str],
    *,
    token: str | None,
    dataset_id: str | None,
    policy: HfPathPolicy,
    min_size: int | None = None,
    hf_hub_download_fn: Callable[..., Any] | None = None,
    request_get: Callable[..., Any] | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Download a Dataset file via huggingface_hub, then HTTP fallback.

    Returns ``(local_path, error_message, hub_error_message)``. ``hub_error`` is
    populated when the hub client failed before HTTP fallback was attempted.
    """
    if not (token and dataset_id and rel_path):
        return None, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID.", None
    try:
        target, local_dir, rel_norm = hf_download_target_for_rel_path(rel_path, policy)
    except PathSecurityError:
        return None, "Đường dẫn cloud không hợp lệ.", None
    if min_size is None:
        min_size = hf_min_size_for_path(rel_norm)

    hub_error = None
    try:
        if hf_hub_download_fn is None:
            from huggingface_hub import hf_hub_download

            hf_hub_download_fn = hf_hub_download
        local_fp = hf_hub_download_fn(
            repo_id=dataset_id,
            filename=rel_norm,
            repo_type="dataset",
            token=token,
            local_dir=local_dir,
        )
        if (
            local_fp
            and path_is_within(local_fp, local_dir)
            and os.path.exists(local_fp)
            and os.path.getsize(local_fp) >= min_size
        ):
            return local_fp, None, None
        if os.path.exists(target) and os.path.getsize(target) >= min_size:
            return target, None, None
        hub_error = f"File `{rel_norm}` tải về nhưng kích thước không hợp lệ."
    except Exception as exc:
        hub_error = str(exc)
        if is_hf_not_found_error(hub_error):
            return None, f"Chưa có trên Dataset: `{rel_norm}`", hub_error
        if is_hf_auth_error(hub_error):
            return None, "Token không có quyền tải file từ Dataset.", hub_error

    http_path, http_err = download_dataset_file_via_http(
        rel_norm,
        token=token,
        dataset_id=dataset_id,
        policy=policy,
        min_size=min_size,
        request_get=request_get,
    )
    if http_path:
        return http_path, None, hub_error
    return None, http_err or hub_error, hub_error


def download_dataset_file_bytes(
    rel_path: str | os.PathLike[str],
    *,
    token: str | None,
    dataset_id: str | None,
    request_get: Callable[..., Any] | None = None,
    timeout: int | float = 60,
) -> tuple[bytes | None, str | None]:
    if not (dataset_id and rel_path):
        return None, "Chưa cấu hình HF_DATASET_ID."
    try:
        rel_norm = normalize_relative_path(rel_path)
        if request_get is None:
            import requests

            request_get = requests.get
        resp = request_get(
            hf_dataset_file_url(dataset_id, rel_norm),
            headers=hf_auth_headers(token),
            timeout=timeout,
        )
        status_error = hf_download_status_error(resp.status_code, rel_norm)
        if status_error:
            return None, status_error
        resp.raise_for_status()
        return resp.content, None
    except PathSecurityError:
        return None, "Đường dẫn cloud không hợp lệ."
    except Exception as exc:
        return None, str(exc)


def download_dataset_file_with_progress(
    rel_path: str | os.PathLike[str],
    target_path: str | os.PathLike[str],
    *,
    token: str | None,
    dataset_id: str | None,
    progress_callback: Callable[[int, int], None] | None = None,
    request_get: Callable[..., Any] | None = None,
    chunk_size: int = 512 * 1024,
    min_size: int = 5 * 1024,
    timeout: int | float = 30,
) -> tuple[str | None, str | None]:
    if not (token and dataset_id and rel_path and target_path):
        return None, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
    try:
        rel_norm = normalize_relative_path(rel_path)
        if request_get is None:
            import requests

            request_get = requests.get
        response = request_get(
            hf_dataset_file_url(dataset_id, rel_norm),
            headers=hf_auth_headers(token),
            stream=True,
            timeout=timeout,
        )
        status_error = hf_download_status_error(response.status_code, rel_norm)
        if status_error:
            return None, status_error
        if response.status_code != 200:
            return None, f"HTTP {response.status_code} khi tải `{rel_norm}`"
        response.raise_for_status()
        os.makedirs(os.path.dirname(str(target_path)) or ".", exist_ok=True)
        total_size = int(response.headers.get("content-length", 0) or 0)
        downloaded = 0
        with open(target_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total_size)
        if os.path.exists(target_path) and os.path.getsize(target_path) >= min_size:
            return str(target_path), None
        return None, f"File `{rel_norm}` tải về nhưng kích thước không hợp lệ."
    except PathSecurityError:
        return None, "Đường dẫn cloud không hợp lệ."
    except Exception as exc:
        return None, str(exc)


def dataset_file_exists(
    rel_path: str | os.PathLike[str],
    *,
    token: str | None,
    dataset_id: str | None,
    request_head: Callable[..., Any] | None = None,
) -> bool:
    if not (dataset_id and rel_path):
        return False
    try:
        if request_head is None:
            import requests

            request_head = requests.head
        response = request_head(
            hf_dataset_file_url(dataset_id, rel_path),
            headers=hf_auth_headers(token),
            timeout=3.0,
        )
        return response.status_code == 200
    except Exception:
        return False


def verify_dataset_via_http(
    token: str | None,
    dataset_id: str | None,
    *,
    request_get: Callable[..., Any] | None = None,
) -> tuple[bool, str | None]:
    if not (token and dataset_id):
        return False, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
    try:
        if request_get is None:
            import requests

            request_get = requests.get
        resp = request_get(
            hf_dataset_file_url(dataset_id, "video_list.json"),
            headers=hf_auth_headers(token),
            timeout=30,
            stream=True,
        )
        status_msg = hf_verify_dataset_status_message(resp.status_code, dataset_id)
        if status_msg:
            return False, status_msg
        if int(resp.headers.get("content-length") or 0) < 2:
            chunk = next(resp.iter_content(64), b"")
            if len(chunk) < 2:
                return False, "Dataset phản hồi nhưng file video_list.json trống."
        return True, None
    except Exception as exc:
        return False, f"Không kết nối Dataset qua HTTP: {exc}"


def download_dataset_file_via_http(
    rel_path: str | os.PathLike[str],
    *,
    token: str | None,
    dataset_id: str | None,
    policy: HfPathPolicy,
    min_size: int = 80,
    request_get: Callable[..., Any] | None = None,
    chunk_size: int = 256 * 1024,
) -> tuple[str | None, str | None]:
    if not (token and dataset_id and rel_path):
        return None, "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
    try:
        target, _local_dir, rel_norm = hf_download_target_for_rel_path(rel_path, policy)
        if request_get is None:
            import requests

            request_get = requests.get
        resp = request_get(
            hf_dataset_file_url(dataset_id, rel_norm),
            headers=hf_auth_headers(token),
            timeout=180,
            stream=True,
        )
        status_error = hf_download_status_error(resp.status_code, rel_path)
        if status_error:
            return None, status_error
        resp.raise_for_status()
        os.makedirs(os.path.dirname(target) or str(policy.data_dir), exist_ok=True)
        with open(target, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
        if os.path.exists(target) and os.path.getsize(target) >= min_size:
            return target, None
        return None, f"File `{rel_path}` tải về nhưng kích thước không hợp lệ."
    except PathSecurityError as exc:
        return None, "Đường dẫn cloud không hợp lệ."
    except Exception as exc:
        return None, str(exc)


def dataset_rel_path_from_local(path: str | os.PathLike[str], policy: HfPathPolicy) -> str:
    raw = str(path or "")
    raw_slash = raw.replace("\\", "/")
    for folder in ("patient_uploads", "processed_results"):
        idx = raw_slash.find(folder)
        if idx != -1:
            rel_hint = normalize_relative_path(raw_slash[idx:])
            safe_data_path(rel_hint, policy.media_roots, base_dir=policy.data_dir)
            return rel_hint

    resolved = Path(raw).expanduser().resolve(strict=False)
    data_root = Path(policy.data_dir).expanduser().resolve(strict=False)
    db_root = Path(policy.db_dir).expanduser().resolve(strict=False)
    if path_is_within(resolved, policy.upload_dir) or path_is_within(resolved, policy.processed_dir):
        rel = resolved.relative_to(data_root).as_posix()
        return normalize_relative_path(rel)
    if path_is_within(resolved, policy.db_dir):
        rel = resolved.relative_to(db_root).as_posix()
        return normalize_relative_path(rel)
    return normalize_relative_path(os.path.basename(raw))


def hf_download_target_for_rel_path(rel_path: str | os.PathLike[str], policy: HfPathPolicy) -> tuple[str, str | os.PathLike[str], str]:
    rel_norm = normalize_relative_path(rel_path)
    if rel_norm in policy.json_files:
        return safe_data_path(rel_norm, policy.db_roots, base_dir=policy.db_dir), policy.db_dir, rel_norm
    if rel_norm in policy.model_artifact_files:
        return safe_data_path(rel_norm, policy.db_roots, base_dir=policy.db_dir), policy.db_dir, rel_norm
    if rel_norm.startswith(("patient_uploads/", "processed_results/")):
        return safe_data_path(rel_norm, policy.media_roots, base_dir=policy.data_dir), policy.data_dir, rel_norm
    raise PathSecurityError("HF path is outside allowed dataset roots")


def hf_upload_rel_path_for_local(local_path: str | os.PathLike[str], policy: HfPathPolicy) -> str:
    resolved = Path(str(local_path or "")).expanduser().resolve(strict=False)
    data_root = Path(policy.data_dir).expanduser().resolve(strict=False)
    db_root = Path(policy.db_dir).expanduser().resolve(strict=False)
    if path_is_within(resolved, policy.upload_dir) or path_is_within(resolved, policy.processed_dir):
        rel_norm = resolved.relative_to(data_root).as_posix()
        rel_norm = normalize_relative_path(rel_norm)
        safe_data_path(rel_norm, policy.media_roots, base_dir=policy.data_dir)
        return rel_norm
    if path_is_within(resolved, policy.db_dir):
        rel_norm = resolved.relative_to(db_root).as_posix()
        rel_norm = normalize_relative_path(rel_norm)
        if rel_norm in policy.json_files or rel_norm in policy.model_artifact_files:
            return rel_norm
    raise PathSecurityError("local path is not allowed for HF upload")
