"""Video file IO helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MAX_FRAME_ZIP_ENTRIES = 20000
MAX_FRAME_ZIP_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_FRAME_ZIP_ENTRY_BYTES = 25 * 1024 * 1024
ALLOWED_FRAME_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
NON_VIDEO_ARTIFACT_EXTENSIONS = frozenset({".json", ".csv", ".zip", ".jpg", ".jpeg", ".png", ".webp"})


@dataclass(frozen=True)
class ExtractedFramesSummary:
    count: int
    total_bytes: int


def ffprobe_video_has_readable_duration(
    path: str | os.PathLike[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int = 5,
) -> bool:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = runner(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if result.returncode != 0:
            return False
        duration_str = str(result.stdout or "").strip()
        if not duration_str:
            return False
        float(duration_str)
        return True
    except Exception:
        return False


def ffprobe_video_codecs(
    path: str | os.PathLike[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int = 10,
) -> tuple[str | None, str | None]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ]
        result = runner(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if result.returncode != 0:
            return None, None
        info = json.loads(result.stdout or "{}")
        video_codec = None
        audio_codec = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_codec = stream.get("codec_name")
            elif stream.get("codec_type") == "audio":
                audio_codec = stream.get("codec_name")
        return video_codec, audio_codec
    except Exception:
        return None, None


def temp_h264_path(dst_path: str | os.PathLike[str]) -> str:
    text = str(dst_path)
    tmp_path = text.replace("_f.mp4", "_ftmp.mp4")
    if tmp_path == text:
        tmp_path = text + ".ftmp.mp4"
    return tmp_path


def build_h264_transcode_command(
    src_path: str | os.PathLike[str],
    dst_path: str | os.PathLike[str],
    *,
    audio_path: str | os.PathLike[str] | None = None,
    audio_exists: bool = False,
    ffmpeg_threads: int | str = 2,
    crf: int = 28,
    preset: str = "ultrafast",
    scale_filter: str = "scale=trunc(iw/2)*2:trunc(ih/2)*2",
) -> list[str]:
    cmd = ["ffmpeg", "-y", "-i", str(src_path)]
    if audio_path and audio_exists:
        cmd.extend(["-i", str(audio_path), "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    else:
        cmd.extend(["-map", "0:v:0", "-an"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
            "-vf",
            scale_filter,
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            "-threads",
            str(ffmpeg_threads),
            "-f",
            "mp4",
            str(dst_path),
        ]
    )
    return cmd


def build_upload_h264_command(
    src_path: str | os.PathLike[str],
    dst_path: str | os.PathLike[str],
    *,
    has_audio: bool,
    ffmpeg_threads: int | str = 2,
    crf: int = 28,
    maxrate: str = "800k",
    bufsize: str = "1600k",
    scale_filter: str = "scale=trunc(iw/2)*2:trunc(ih/2)*2",
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-crf",
        str(crf),
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-vf",
        scale_filter,
        "-threads",
        str(ffmpeg_threads),
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac"])
    else:
        cmd.extend(["-an"])
    cmd.append(str(dst_path))
    return cmd


def build_async_h264_command(
    src_path: str | os.PathLike[str],
    dst_path: str | os.PathLike[str],
    *,
    has_audio: bool,
    ffmpeg_threads: int | str = 2,
    crf: int = 30,
    maxrate: str = "500k",
    bufsize: str = "1000k",
    scale_filter: str = r"scale=-2:min(480\,ih)",
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-vf",
        scale_filter,
        "-crf",
        str(crf),
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-movflags",
        "+faststart",
        "-threads",
        str(ffmpeg_threads),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac"])
    else:
        cmd.extend(["-an"])
    cmd.extend(["-f", "mp4", str(dst_path)])
    return cmd


def build_background_upload_h264_command(
    src_path: str | os.PathLike[str],
    dst_path: str | os.PathLike[str],
    *,
    ffmpeg_threads: int | str = 2,
    crf: int = 28,
    maxrate: str = "800k",
    bufsize: str = "1600k",
    scale_filter: str = "scale=-2:720",
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-crf",
        str(crf),
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-vf",
        scale_filter,
        "-threads",
        str(ffmpeg_threads),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:a",
        "aac",
        str(dst_path),
    ]


def mov_to_mp4_path(input_path: str | os.PathLike[str]) -> str:
    return str(input_path).replace(".mov", ".mp4").replace(".MOV", ".mp4")


def build_ffmpeg_version_command() -> list[str]:
    return ["ffmpeg", "-version"]


def build_mov_to_mp4_command(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    preset: str = "fast",
    crf: int = 23,
) -> list[str]:
    target = mov_to_mp4_path(input_path) if output_path is None else str(output_path)
    return [
        "ffmpeg",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-y",
        target,
    ]


def build_cut_segment_command(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    start: float,
    duration: float,
    crf: int = 26,
    preset: str = "ultrafast",
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def build_frame_extract_command(
    video_path: str | os.PathLike[str],
    frame_path: str | os.PathLike[str],
    *,
    timestamp: float,
    quality: int = 5,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.4f}",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        str(quality),
        str(frame_path),
    ]


def ffprobe_video_duration_text(
    path: str | os.PathLike[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int = 3,
) -> tuple[str, str | None]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = runner(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        duration = str(result.stdout or "").strip()
        if result.returncode != 0:
            return duration, str(result.stderr or "").strip()
        return duration, None
    except Exception as exc:
        return "", str(exc)


def is_non_playable_video_artifact(path: str | os.PathLike[str] | None) -> bool:
    if not path:
        return True
    normalized = str(path).replace("\\", "/")
    low = normalized.lower()
    base = os.path.basename(low)
    if low.endswith(tuple(NON_VIDEO_ARTIFACT_EXTENSIONS)):
        return True
    if not low.endswith(tuple(VIDEO_EXTENSIONS)):
        return True
    if (
        base.endswith("_frames.mp4")
        or base.endswith("_frames_f.mp4")
        or "_frames_" in base
        or ("/processed_results/processed_" in low and "_frames/" in low)
    ):
        return True
    return False


def final_h264_path(video_path: str | os.PathLike[str] | None) -> str:
    if not video_path or is_non_playable_video_artifact(video_path):
        return ""
    text = str(video_path)
    if text.endswith("_f.mp4"):
        return text
    base, _ = os.path.splitext(text)
    if base.endswith("_f"):
        return base + ".mp4"
    return base + "_f.mp4"


def video_fallback_paths_for(
    file_path: str | os.PathLike[str] | None,
    *,
    local_frame_path_resolver=None,
) -> list[str]:
    if not file_path:
        return []
    try:
        norm = local_frame_path_resolver(file_path) or file_path if local_frame_path_resolver else file_path
    except Exception:
        norm = file_path
    norm = str(norm)
    if is_non_playable_video_artifact(norm):
        return []
    if norm.endswith("_f.mp4"):
        candidates = [norm, norm.replace("_f.mp4", ".mp4")]
    elif norm.endswith("_ffmp.mp4"):
        candidates = [norm, norm.replace("_ffmp.mp4", ".mp4")]
    else:
        h264 = final_h264_path(norm)
        candidates = [h264, norm] if h264 != norm else [norm]

    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def safe_extract_frames_zip(
    zip_path: str | os.PathLike[str],
    frames_dir: str | os.PathLike[str],
    *,
    max_entries: int = MAX_FRAME_ZIP_ENTRIES,
    max_total_bytes: int = MAX_FRAME_ZIP_TOTAL_BYTES,
    max_entry_bytes: int = MAX_FRAME_ZIP_ENTRY_BYTES,
    allowed_extensions: frozenset[str] = ALLOWED_FRAME_IMAGE_EXTENSIONS,
) -> ExtractedFramesSummary:
    frames_root = Path(frames_dir).expanduser().resolve(strict=False)
    frames_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        infos = zip_ref.infolist()
        if len(infos) > max_entries:
            raise ValueError("ZIP chứa quá nhiều file frame.")

        total_size = 0
        planned: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in infos:
            if info.is_dir():
                continue

            raw_name = info.filename or ""
            normalized = raw_name.replace("\\", "/")
            base_name = os.path.basename(normalized)
            if not base_name or base_name != normalized or ".." in normalized.split("/"):
                raise ValueError("ZIP chứa đường dẫn frame không hợp lệ.")

            ext = os.path.splitext(base_name)[1].lower()
            if ext not in allowed_extensions:
                raise ValueError("ZIP chứa file không phải ảnh frame.")

            if info.file_size > max_entry_bytes:
                raise ValueError("Một frame trong ZIP vượt quá dung lượng cho phép.")

            total_size += int(info.file_size or 0)
            if total_size > max_total_bytes:
                raise ValueError("ZIP frame vượt quá tổng dung lượng cho phép.")

            target_path = (frames_root / base_name).resolve(strict=False)
            if target_path != frames_root and frames_root not in target_path.parents:
                raise ValueError("ZIP cố ghi file ra ngoài thư mục frame.")

            planned.append((info, target_path))

        for info, target_path in planned:
            with zip_ref.open(info, "r") as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

    return ExtractedFramesSummary(count=len(planned), total_bytes=total_size)
