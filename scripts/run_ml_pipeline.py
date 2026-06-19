#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline ML Pose Classifier - train and/or apply to processed patient videos.

Usage:
  python scripts/run_ml_pipeline.py train
  python scripts/run_ml_pipeline.py apply
  python scripts/run_ml_pipeline.py all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(name: str, *extra_args: str) -> int:
    path = os.path.join(ROOT, "scripts", name)
    return subprocess.call([sys.executable, path, *extra_args], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Pose Classifier MediaPipe")
    parser.add_argument(
        "action",
        choices=["train", "apply", "all"],
        help="train | apply | all",
    )
    parser.add_argument("--dry-run", action="store_true", help="For apply/all, list writes without changing files.")
    args = parser.parse_args()

    if args.action in ("train", "all"):
        code = run_script("train_classifier.py")
        if code != 0:
            return code

    if args.action in ("apply", "all"):
        extra = ["--dry-run"] if args.dry_run else []
        return run_script("reprocess_all.py", *extra)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
