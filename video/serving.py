"""Helpers for serving local video files through short-lived media URLs."""

from __future__ import annotations

import os
import secrets
import tempfile
import time
import urllib.parse
from collections.abc import Callable, MutableMapping


ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".webm"})
LOCAL_VIDEO_ORIGIN_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def safe_realpath(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return ""
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def path_is_within(child: str | os.PathLike[str], parent: str | os.PathLike[str]) -> bool:
    try:
        child_real = safe_realpath(child)
        parent_real = safe_realpath(parent)
        return os.path.commonpath([child_real, parent_real]) == parent_real
    except Exception:
        return False


def video_media_allowed_roots(
    *,
    data_dir: str | os.PathLike[str] = ".",
    upload_dir: str | os.PathLike[str] | None = None,
    processed_dir: str | os.PathLike[str] | None = None,
    temp_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Return existing media roots that may be exposed by the local video server."""
    data_dir_text = str(data_dir)
    candidates = {
        "uploads": upload_dir or os.path.join(data_dir_text, "patient_uploads"),
        "processed": processed_dir or os.path.join(data_dir_text, "processed_results"),
        "temp": temp_dir or os.path.join(tempfile.gettempdir(), "rehab_videos"),
    }
    roots: dict[str, str] = {}
    for name, raw_path in candidates.items():
        root = safe_realpath(raw_path)
        if root and os.path.isdir(root):
            roots[name] = root
    return roots


def allowed_media_file_path(
    path: str | os.PathLike[str] | None,
    roots: dict[str, str],
    *,
    allowed_extensions: frozenset[str] = ALLOWED_VIDEO_EXTENSIONS,
) -> str | None:
    real_path = safe_realpath(path)
    if not real_path or not os.path.isfile(real_path):
        return None
    ext = os.path.splitext(real_path)[1].lower()
    if ext not in allowed_extensions:
        return None
    for root in roots.values():
        if path_is_within(real_path, root):
            return real_path
    return None


def cleanup_media_tokens(tokens: MutableMapping[str, dict], *, now: float | None = None) -> None:
    current = time.time() if now is None else now
    expired = [token for token, meta in tokens.items() if meta.get("expires_at", 0) < current]
    for token in expired:
        tokens.pop(token, None)


def register_media_token(
    tokens: MutableMapping[str, dict],
    path: str | os.PathLike[str] | None,
    roots: dict[str, str],
    *,
    ttl_seconds: int,
    now: float | None = None,
    token_factory: Callable[[int], str] = secrets.token_urlsafe,
) -> str | None:
    real_path = allowed_media_file_path(path, roots)
    if not real_path:
        return None
    current = time.time() if now is None else now
    cleanup_media_tokens(tokens, now=current)
    token = token_factory(32)
    tokens[token] = {
        "path": real_path,
        "expires_at": current + ttl_seconds,
    }
    return token


def resolve_media_token(
    tokens: MutableMapping[str, dict],
    token: str | None,
    roots: dict[str, str],
    *,
    now: float | None = None,
) -> str | None:
    if not token:
        return None
    cleanup_media_tokens(tokens, now=now)
    meta = tokens.get(token)
    if not meta:
        return None
    real_path = allowed_media_file_path(meta.get("path"), roots)
    if not real_path:
        tokens.pop(token, None)
        return None
    return real_path


def is_allowed_video_origin(origin: str | None) -> bool:
    if not origin:
        return False
    try:
        parsed = urllib.parse.urlsplit(origin)
        return parsed.scheme in ("http", "https") and parsed.hostname in LOCAL_VIDEO_ORIGIN_HOSTS
    except Exception:
        return False


def media_token_from_request_path(request_path: str | None) -> str | None:
    try:
        parsed_path = urllib.parse.urlsplit(request_path or "").path
        parts = [urllib.parse.unquote(p) for p in parsed_path.split("/") if p]
        if len(parts) < 3 or parts[0] != "_media":
            return None
        filename = parts[-1]
        if "\\" in filename or "/" in filename or ".." in parts:
            return None
        return parts[1]
    except Exception:
        return None


def build_video_media_url(port: int | str | None, token: str | None, video_path: str | os.PathLike[str]) -> str | None:
    if port is None or not token:
        return None
    try:
        filename = urllib.parse.quote(os.path.basename(str(video_path)))
        return f"http://127.0.0.1:{port}/_media/{token}/{filename}"
    except Exception:
        return None
