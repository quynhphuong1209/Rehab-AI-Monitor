# -*- coding: utf-8 -*-
"""Apply the trained pose classifier to existing processed CSV files."""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pose_classifier_utils import reprocess_videos_with_classifier


def reprocess() -> bool:
    result = reprocess_videos_with_classifier(
        videos_file=os.path.join(ROOT, "database", "video_list.json"),
        evaluations_file=os.path.join(ROOT, "database", "doctor_evaluations.json"),
        processed_dir=os.path.join(ROOT, "processed_results"),
        db_dir=os.path.join(ROOT, "database"),
        data_dir=ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("success"):
        sync_script = os.path.join(ROOT, "scripts", "sync_data_and_report.py")
        if os.path.exists(sync_script):
            try:
                subprocess.run([sys.executable, sync_script], cwd=ROOT, check=False)
            except Exception as exc:
                print(f"Khong the chay sync_data_and_report.py: {exc}")

    return bool(result.get("success"))


if __name__ == "__main__":
    raise SystemExit(0 if reprocess() else 1)
