"""Frame gallery helpers for the standalone backend.

The gallery endpoints read small pages of frame metadata and individual frame
images on demand. ZIP files are inspected by name and one entry is read only
when that image is requested.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from backend.analysis_parity import (
    frame_public_payload,
    frame_summary as parity_frame_summary,
    normalized_frame_filter,
    phase_for_index,
    segment_frame_bounds,
    threshold_for_phase,
    frame_passes_filter,
)
from video.io import ALLOWED_FRAME_IMAGE_EXTENSIONS
from video.serving import allowed_media_file_path, path_is_within


MAX_FRAME_JSON_BYTES = 25 * 1024 * 1024
MAX_ZIP_ENTRIES_FOR_GALLERY = 20_000
MAX_FRAME_IMAGE_BYTES = 8 * 1024 * 1024


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _path_basename(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _relative_path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value or "").strip())
    if path.is_absolute():
        return path
    return repo_root / Path(*[part for part in str(value or "").replace("\\", "/").split("/") if part])


def frame_roots(repo_root: Path, processed_dir: Path) -> dict[str, str]:
    candidates = {
        "processed": processed_dir,
        "repo_processed": repo_root / "processed_results",
        "temp_rehab": Path(tempfile.gettempdir()) / "rehab_videos",
        "temp": Path(tempfile.gettempdir()),
    }
    roots: dict[str, str] = {}
    for key, path in candidates.items():
        root = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
        if root and os.path.isdir(root):
            roots[key] = root
    return roots


def resolve_frame_json_path(record: dict[str, Any], repo_root: Path, roots: dict[str, str]) -> str | None:
    raw_path = record.get("all_frames_data_path")
    if not raw_path:
        return None
    path = _relative_path(repo_root, raw_path)
    return allowed_media_file_path(path, roots, allowed_extensions=frozenset({".json"}))


def resolve_frames_zip_path(record: dict[str, Any], repo_root: Path, roots: dict[str, str]) -> str | None:
    raw_path = record.get("frames_zip_path") or record.get("frames_zip")
    if not raw_path:
        return None
    path = _relative_path(repo_root, raw_path)
    return allowed_media_file_path(path, roots, allowed_extensions=frozenset({".zip"}))


def load_frame_records(json_path: str | None) -> list[dict[str, Any]]:
    if not json_path:
        return []
    if os.path.getsize(json_path) > MAX_FRAME_JSON_BYTES:
        raise ValueError("frame JSON is too large")
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def frame_label(record: dict[str, Any]) -> str:
    for key in ("label", "status", "phase_status", "ref_status"):
        value = _clean_text(record.get(key)).upper()
        if value in {"PASS", "NEAR", "NEARLY", "FAIL"}:
            return "NEAR" if value == "NEARLY" else value
    if record.get("dung") is True:
        return "PASS"
    if record.get("gan_dung") is True:
        return "NEAR"
    return "FAIL"


def safe_zip_image_names(zip_path: str | None) -> set[str]:
    if not zip_path:
        return set()
    names: set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        infos = zip_file.infolist()
        if len(infos) > MAX_ZIP_ENTRIES_FOR_GALLERY:
            raise ValueError("frames ZIP contains too many entries")
        for info in infos:
            if info.is_dir():
                continue
            normalized = (info.filename or "").replace("\\", "/")
            base = os.path.basename(normalized)
            ext = os.path.splitext(base)[1].lower()
            if not base or base != normalized or ".." in normalized.split("/") or ext not in ALLOWED_FRAME_IMAGE_EXTENSIONS:
                raise ValueError("frames ZIP contains an unsafe entry")
            if info.file_size > MAX_FRAME_IMAGE_BYTES:
                raise ValueError("a frame image in ZIP is too large")
            names.add(base)
    return names


def resolve_frame_image_path(record: dict[str, Any], repo_root: Path, roots: dict[str, str]) -> str | None:
    raw_path = record.get("path") or record.get("image_path") or record.get("frame_path")
    if not raw_path:
        return None
    return allowed_media_file_path(
        _relative_path(repo_root, raw_path),
        roots,
        allowed_extensions=ALLOWED_FRAME_IMAGE_EXTENSIONS,
    )


def frame_image_id(record: dict[str, Any], raw_index: int, zip_names: set[str], repo_root: Path, roots: dict[str, str]) -> str:
    image_path = resolve_frame_image_path(record, repo_root, roots)
    if image_path:
        return f"frame:{raw_index}"
    base = _path_basename(record.get("path") or record.get("image_path") or record.get("frame_path"))
    if base and base in zip_names:
        return f"frame:{raw_index}"
    return ""


def frame_gallery_page(
    record: dict[str, Any],
    *,
    repo_root: Path,
    processed_dir: Path,
    page: int,
    page_size: int,
    label_filter: str,
) -> dict[str, Any]:
    roots = frame_roots(repo_root, processed_dir)
    frames = load_frame_records(resolve_frame_json_path(record, repo_root, roots))
    zip_path = resolve_frames_zip_path(record, repo_root, roots)
    zip_names = safe_zip_image_names(zip_path)
    label = normalized_frame_filter(label_filter)
    bounds = segment_frame_bounds(frames)
    exercise = record.get("exercise") or ""
    indexed_frames = list(enumerate(frames))
    filtered = [
        (raw_index, item)
        for raw_index, item in indexed_frames
        if frame_passes_filter(item, raw_index, bounds, exercise, label)
    ]

    total = len(filtered)
    page = max(1, page)
    page_size = max(1, min(48, page_size))
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    page_records = filtered[start : start + page_size]
    items = [
        frame_public_payload(item, raw_index, raw_index, frame_image_id(item, raw_index, zip_names, repo_root, roots), bounds, exercise)
        for raw_index, item in page_records
    ]
    phase_ranges = {
        "G1": {"start": bounds[0], "end": bounds[1], "threshold": threshold_for_phase("G1")},
        "G2": {"start": bounds[1], "end": bounds[2], "threshold": threshold_for_phase("G2")},
        "G3": {"start": bounds[2], "end": bounds[3], "threshold": threshold_for_phase("G3")},
    }
    return {
        "items": items,
        "summary": parity_frame_summary(frames, bounds, exercise),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        },
        "filter": label,
        "segment_bounds": bounds,
        "phase_ranges": phase_ranges,
        "sources": {
            "frames_json": bool(frames),
            "frames_zip": bool(zip_path),
        },
    }


def frame_raw_index_from_id(image_id: str) -> int | None:
    prefix = "frame:"
    if not image_id.startswith(prefix):
        return None
    try:
        raw_index = int(image_id[len(prefix) :])
    except ValueError:
        return None
    if raw_index < 0:
        return None
    return raw_index


def read_zip_image(zip_path: str, name: str) -> tuple[bytes, str] | None:
    normalized = name.replace("\\", "/")
    base = os.path.basename(normalized)
    ext = os.path.splitext(base)[1].lower()
    if not base or base != normalized or ".." in normalized.split("/") or ext not in ALLOWED_FRAME_IMAGE_EXTENSIONS:
        return None
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        info = zip_file.getinfo(base)
        if info.file_size > MAX_FRAME_IMAGE_BYTES:
            return None
        with zip_file.open(info, "r") as handle:
            data = handle.read(MAX_FRAME_IMAGE_BYTES + 1)
    if len(data) > MAX_FRAME_IMAGE_BYTES:
        return None
    media_type = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
    return data, media_type


def resolve_gallery_image(
    record: dict[str, Any],
    *,
    image_id: str,
    repo_root: Path,
    processed_dir: Path,
) -> tuple[str, str | bytes, str] | None:
    roots = frame_roots(repo_root, processed_dir)
    raw_index = frame_raw_index_from_id(image_id)
    if raw_index is None:
        return None
    frames = load_frame_records(resolve_frame_json_path(record, repo_root, roots))
    if raw_index >= len(frames):
        return None
    frame = frames[raw_index]
    image_path = resolve_frame_image_path(frame, repo_root, roots)
    if image_path:
        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
        return "file", image_path, media_type
    zip_path = resolve_frames_zip_path(record, repo_root, roots)
    if not zip_path:
        return None
    if not path_is_within(zip_path, roots.get("processed", "")) and not any(path_is_within(zip_path, root) for root in roots.values()):
        return None
    base = _path_basename(frame.get("path") or frame.get("image_path") or frame.get("frame_path"))
    zip_image = read_zip_image(zip_path, base)
    if not zip_image:
        return None
    data, media_type = zip_image
    return "bytes", data, media_type
