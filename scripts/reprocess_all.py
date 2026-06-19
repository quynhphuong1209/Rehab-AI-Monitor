# -*- coding: utf-8 -*-
"""Apply the trained pose classifier to existing processed CSV files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in [ROOT, os.path.join(ROOT, "utils")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pose_classifier_utils import reprocess_videos_with_classifier


def reprocess(dry_run: bool = False, sync_report: bool = True) -> bool:
    result = reprocess_videos_with_classifier(
        videos_file=os.path.join(ROOT, "database", "video_list.json"),
        evaluations_file=os.path.join(ROOT, "database", "doctor_evaluations.json"),
        processed_dir=os.path.join(ROOT, "processed_results"),
        db_dir=os.path.join(ROOT, "database"),
        data_dir=ROOT,
        dry_run=dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("success") and not dry_run and sync_report:
        sync_script = os.path.join(ROOT, "scripts", "sync_data_and_report.py")
        if os.path.exists(sync_script):
            try:
                subprocess.run([sys.executable, sync_script], cwd=ROOT, check=False)
            except Exception as exc:
                print(f"Khong the chay sync_data_and_report.py: {exc}")

    return bool(result.get("success"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the trained pose classifier to saved videos.")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be read/written without changing them.")
    parser.add_argument("--no-sync-report", action="store_true", help="Do not run sync_data_and_report.py after applying.")
    args = parser.parse_args()
    raise SystemExit(0 if reprocess(dry_run=args.dry_run, sync_report=not args.no_sync_report) else 1)
