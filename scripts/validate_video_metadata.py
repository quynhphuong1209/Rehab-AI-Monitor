#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate saved video metadata paths and timestamp consistency."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_LIST = ROOT / "database" / "video_list.json"

REQUIRED_KEYS = {"username", "video_name", "video_path", "status"}
PATH_KEYS = {
    "video_path",
    "processed_path",
    "df_path",
    "all_frames_data_path",
    "frames_zip",
    "frames_zip_path",
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def timestamp_from(value: Any) -> str | None:
    if not value:
        return None
    match = re.search(r"processed_(\d+)", str(value))
    return match.group(1) if match else None


def contained_path(value: str, allowed_roots: list[Path]) -> bool:
    raw = str(value or "").replace("\\", "/")
    if raw.startswith("http://") or raw.startswith("https://"):
        return True
    clean = raw.replace("/data/", "").lstrip("/")
    candidates = [Path(raw), ROOT / clean]
    basename = Path(clean).name
    for folder in ("patient_uploads", "processed_results", "database"):
        candidates.append(ROOT / folder / basename)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if any(resolved == root or root in resolved.parents for root in allowed_roots):
            return True
    return False


def validate_records(records: Any, *, fix: bool = False) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(records, list):
        return [{"index": None, "error": "video_list root must be a list"}], False

    allowed_roots = [
        ROOT.resolve(),
        (ROOT / "patient_uploads").resolve(),
        (ROOT / "processed_results").resolve(),
        (ROOT / "database").resolve(),
    ]
    findings: list[dict[str, Any]] = []
    changed = False

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            findings.append({"index": idx, "error": "record is not an object"})
            continue

        missing = sorted(key for key in REQUIRED_KEYS if key not in record)
        if missing:
            findings.append({"index": idx, "video": record.get("video_name"), "missing": missing})

        for key in PATH_KEYS:
            value = record.get(key)
            if value and not contained_path(str(value), allowed_roots):
                findings.append({"index": idx, "video": record.get("video_name"), "unsafe_path": {key: value}})

        processed_ts = timestamp_from(record.get("processed_path"))
        if processed_ts:
            expected_zip = str(ROOT / "processed_results" / f"processed_{processed_ts}_frames.zip")
            for key in ("frames_zip", "frames_zip_path"):
                current = record.get(key)
                current_ts = timestamp_from(current)
                if current and current_ts and current_ts != processed_ts:
                    findings.append(
                        {
                            "index": idx,
                            "video": record.get("video_name"),
                            "timestamp_mismatch": {
                                "processed_path": record.get("processed_path"),
                                key: current,
                                "expected": expected_zip,
                            },
                        }
                    )
                    if fix:
                        record[key] = expected_zip
                        changed = True
                elif key not in record and fix:
                    record[key] = expected_zip
                    changed = True

    return findings, changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate video_list metadata.")
    parser.add_argument("--file", default=str(DEFAULT_VIDEO_LIST), help="Path to video_list.json")
    parser.add_argument("--fix", action="store_true", help="Fix timestamp-mismatched frames ZIP metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Report only. This is the default unless --fix is set.")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = (ROOT / path).resolve()

    records = load_json(path, [])
    findings, changed = validate_records(records, fix=args.fix and not args.dry_run)
    print(json.dumps({"file": str(path), "findings": findings, "changed": changed}, ensure_ascii=False, indent=2))

    if changed:
        write_json(path, records)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

