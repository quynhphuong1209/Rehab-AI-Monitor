# -*- coding: utf-8 -*-
"""Train the second-stage pose classifier from extracted MediaPipe CSV files."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pose_classifier_utils import train_pose_classifier


def train() -> bool:
    result = train_pose_classifier(
        processed_dir=os.path.join(ROOT, "processed_results"),
        db_dir=os.path.join(ROOT, "database"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(result.get("success"))


if __name__ == "__main__":
    raise SystemExit(0 if train() else 1)
