"""Upload/video metadata validation utilities."""

from __future__ import annotations

import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any


MAX_UPLOAD_SIZE_MB = 300
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_UPLOAD_DURATION_SECONDS = 60 * 60
MAX_UPLOAD_WIDTH = 3840
MAX_UPLOAD_HEIGHT = 2160
ALLOWED_UPLOAD_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ALLOWED_UPLOAD_MIME_TYPES = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}


def sanitize_filename(original_name: str | None, fallback: str = "upload") -> str:
    raw_name = os.path.basename(str(original_name or ""))
    raw_name = unicodedata.normalize("NFKD", raw_name)
    raw_name = raw_name.encode("ascii", "ignore").decode("ascii")
    raw_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
    if not raw_name:
        raw_name = fallback

    base, ext = os.path.splitext(raw_name)
    ext = ext[:20].lower()
    base = base[: max(1, 200 - len(ext))]
    return f"{base}{ext}"[:200]


def upload_video_magic_matches(ext: str | None, prefix: bytes) -> bool:
    if not prefix:
        return False
    ext = (ext or "").lower()
    if ext in {".mp4", ".mov", ".m4v"}:
        return len(prefix) >= 12 and prefix[4:8] == b"ftyp"
    if ext == ".avi":
        return prefix.startswith(b"RIFF") and prefix[8:12] == b"AVI "
    if ext in {".mkv", ".webm"}:
        return prefix.startswith(b"\x1a\x45\xdf\xa3")
    return False


def validate_upload_metadata(uploaded_file: Any) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "Chua chon file video."

    size = getattr(uploaded_file, "size", None)
    if size is None:
        return False, "Khong xac dinh duoc dung luong file upload."
    if size <= 0:
        return False, "File upload dang rong."
    if size > MAX_UPLOAD_SIZE_BYTES:
        return False, f"File vuot qua gioi han {MAX_UPLOAD_SIZE_MB}MB."

    name = os.path.basename(str(getattr(uploaded_file, "name", "") or ""))
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_UPLOAD_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_VIDEO_EXTENSIONS))
        return False, f"Dinh dang file khong duoc ho tro. Chi nhan: {allowed}."

    mime_type = (getattr(uploaded_file, "type", "") or "").lower()
    if mime_type not in ALLOWED_UPLOAD_MIME_TYPES and not mime_type.startswith("video/"):
        return False, f"MIME type khong hop le cho video: {mime_type or 'khong ro'}."

    try:
        old_pos = uploaded_file.tell()
    except Exception:
        old_pos = None
    try:
        uploaded_file.seek(0)
        prefix = uploaded_file.read(64)
    except Exception:
        prefix = b""
    finally:
        try:
            uploaded_file.seek(old_pos or 0)
        except Exception:
            pass

    if not upload_video_magic_matches(ext, prefix):
        return False, "Header file khong khop dinh dang video da chon."
    return True, ""


def validate_video_file_for_processing(
    file_path: str | os.PathLike[str],
    *,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
    max_duration_seconds: int = MAX_UPLOAD_DURATION_SECONDS,
    max_width: int = MAX_UPLOAD_WIDTH,
    max_height: int = MAX_UPLOAD_HEIGHT,
) -> tuple[bool, str]:
    path = Path(file_path)
    if not file_path or not path.exists():
        return False, "Khong tim thay file video tam."
    if path.stat().st_size > max_size_bytes:
        return False, f"File vuot qua gioi han {max_size_bytes // (1024 * 1024)}MB."

    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,width,height,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if result.returncode != 0:
            return False, "Khong doc duoc metadata video bang ffprobe."
        info = json.loads(result.stdout or "{}")
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not streams:
            return False, "File khong co video stream hop le."
        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width <= 0 or height <= 0:
            return False, "Khong xac dinh duoc do phan giai video."
        if width > max_width or height > max_height:
            return False, f"Video vuot qua do phan giai toi da {max_width}x{max_height}."
        duration_raw = stream.get("duration") or (info.get("format") or {}).get("duration")
        if duration_raw:
            duration = float(duration_raw)
            if duration <= 0:
                return False, "Thoi luong video khong hop le."
            if duration > max_duration_seconds:
                return False, "Video vuot qua thoi luong toi da 60 phut."
        return True, ""
    except FileNotFoundError:
        return False, "Khong tim thay ffprobe de kiem tra video."
    except (json.JSONDecodeError, subprocess.TimeoutExpired, ValueError, OSError) as exc:
        return False, f"Khong kiem tra duoc metadata video: {exc}"

