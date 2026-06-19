# -*- coding: utf-8 -*-
"""
Trích xuất góc vai/khuỷu từ video YouTube mẫu (MediaPipe) và lưu reference JSON
theo từng động tác 1, 2, 3 của Codman, bài tập với gậy hoặc dây kháng lực.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mediapipe as mp

mp_pose = mp.solutions.pose


def tinh_goc(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def get_pose_model(model_type="MediaPipe Heavy", min_confidence=0.5):
    complexity = 2 if "Heavy" in model_type else (0 if "Lite" in model_type else 1)
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

# Khoảng thời gian (giây) ước lượng cho từng động tác trong video hướng dẫn PHCN.
# Có thể chỉnh sau khi xem lại video; script vẫn trích toàn bộ góc trong khoảng đó.
EXERCISE_SEGMENTS: dict[str, dict[str, Any]] = {
    "codman": {
        "youtube": "https://youtu.be/a4eCRWuqO40",
        "side": "right",
        "exercises": {
            "1": {
                "name": "Đung trước-sau (Sagittal)",
                "motion_type": "sagittal",
                "time_start": 5.0,
                "time_end": 22.0,
            },
            "2": {
                "name": "Đung sang ngang (Frontal)",
                "motion_type": "frontal",
                "time_start": 22.0,
                "time_end": 40.0,
            },
            "3": {
                "name": "Xoay vòng tròn / con lắc (Circular)",
                "motion_type": "circular",
                "time_start": 40.0,
                "time_end": 58.0,
            },
        },
    },
    "gay": {
        "youtube": "https://www.youtube.com/watch?v=s2O8WHT5o2k",
        "side": "both",
        "exercises": {
            "1": {
                "name": "Nâng gậy ra trước — dơ gậy cao (Flexion)",
                "motion_type": "flexion",
                "time_start": 20.0,
                "time_end": 55.0,
            },
            "2": {
                "name": "Xoay vai ngoài (External rotation)",
                "motion_type": "external_rotation",
                "time_start": 55.0,
                "time_end": 90.0,
            },
            "3": {
                "name": "Xoay vai trong (Internal rotation)",
                "motion_type": "internal_rotation",
                "time_start": 90.0,
                "time_end": 130.0,
            },
        },
    },
    "day": {
        "youtube": "https://www.youtube.com/watch?v=njDHDnZ6lis",
        "side": "both",
        "exercises": {
            "1": {
                "name": "Xoay vai ngoài (External rotation)",
                "motion_type": "external_rotation",
                "time_start": 88.0,
                "time_end": 118.0,
            },
            "2": {
                "name": "Xoay vai trong (Internal rotation)",
                "motion_type": "internal_rotation",
                "time_start": 118.0,
                "time_end": 148.0,
            },
            "3": {
                "name": "Dang vai (Abduction)",
                "motion_type": "abduction",
                "time_start": 148.0,
                "time_end": 198.0,
            },
        },
    },
}


def _ensure_yt_dlp() -> None:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Missing dependency yt-dlp. Install project dependencies with "
            "`requirements-prod.txt`."
        )


def download_youtube(url: str, out_dir: str) -> str:
    _ensure_yt_dlp()
    import yt_dlp

    out_template = os.path.join(out_dir, "ref_video.%(ext)s")
    opts = {
        "format": "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        ext = info.get("ext", "mp4")
    path = os.path.join(out_dir, f"ref_video.{ext}")
    if not os.path.exists(path):
        for f in os.listdir(out_dir):
            if f.startswith("ref_video"):
                return os.path.join(out_dir, f)
    return path


def _landmark_xy(lm, idx: int, w: int, h: int) -> tuple[int, int]:
    p = lm[idx]
    return int(p.x * w), int(p.y * h)


def extract_angles_from_video(
    video_path: str,
    side: str = "right",
    sample_every_n: int = 3,
    time_ranges: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = get_pose_model("MediaPipe Heavy", 0.5)
    frames: list[dict[str, Any]] = []
    frame_idx = 0

    def _in_ranges(t: float) -> bool:
        if not time_ranges:
            return True
        return any(t0 <= t <= t1 for t0, t1 in time_ranges)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            if not _in_ranges(t):
                frame_idx += 1
                continue
            if frame_idx % sample_every_n != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = model.process(rgb)
            t = round(t, 2)

            if not result.pose_landmarks:
                frame_idx += 1
                continue

            lm = result.pose_landmarks.landmark
            vai_t = _landmark_xy(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
            khuyu_t = _landmark_xy(lm, mp_pose.PoseLandmark.LEFT_ELBOW, w, h)
            co_tay_t = _landmark_xy(lm, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
            hong_t = _landmark_xy(lm, mp_pose.PoseLandmark.LEFT_HIP, w, h)

            vai_p = _landmark_xy(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
            khuyu_p = _landmark_xy(lm, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
            co_tay_p = _landmark_xy(lm, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
            hong_p = _landmark_xy(lm, mp_pose.PoseLandmark.RIGHT_HIP, w, h)

            g_vai_t = float(tinh_goc(hong_t, vai_t, khuyu_t))
            g_khuyu_t = float(tinh_goc(vai_t, khuyu_t, co_tay_t))
            g_vai_p = float(tinh_goc(hong_p, vai_p, khuyu_p))
            g_khuyu_p = float(tinh_goc(vai_p, khuyu_p, co_tay_p))

            row: dict[str, Any] = {
                "time": t,
                "vai_trai": round(g_vai_t, 1),
                "khuyu_trai": round(g_khuyu_t, 1),
                "vai_phai": round(g_vai_p, 1),
                "khuyu_phai": round(g_khuyu_p, 1),
            }
            if side == "right":
                row["vai"] = row["vai_phai"]
                row["khuyu"] = row["khuyu_phai"]
            elif side == "left":
                row["vai"] = row["vai_trai"]
                row["khuyu"] = row["khuyu_trai"]
            else:
                row["vai"] = round((g_vai_t + g_vai_p) / 2, 1)
                row["khuyu"] = round((g_khuyu_t + g_khuyu_p) / 2, 1)

            frames.append(row)
            frame_idx += 1
    finally:
        cap.release()
        model.close()

    return frames


def _downsample(poses: list[dict], max_points: int = 80) -> list[dict]:
    if len(poses) <= max_points:
        return poses
    idxs = np.linspace(0, len(poses) - 1, max_points, dtype=int)
    return [poses[i] for i in idxs]


def build_reference_json(exercise_key: str, all_frames: list[dict]) -> dict[str, Any]:
    cfg = EXERCISE_SEGMENTS[exercise_key]
    exercises_out: dict[str, Any] = {}

    for ex_id, ex_cfg in cfg["exercises"].items():
        t0 = float(ex_cfg["time_start"])
        t1 = float(ex_cfg["time_end"])
        segment = [f for f in all_frames if t0 <= f["time"] <= t1]
        segment = _downsample(segment)

        for p in segment:
            p["exercise_id"] = int(ex_id)
            p["motion_type"] = ex_cfg["motion_type"]

        exercises_out[ex_id] = {
            "name": ex_cfg["name"],
            "motion_type": ex_cfg["motion_type"],
            "time_start": t0,
            "time_end": t1,
            "poses": segment,
        }

    flat: list[dict] = []
    for ex_id in sorted(exercises_out.keys(), key=int):
        flat.extend(exercises_out[ex_id]["poses"])

    return {
        "version": 2,
        "source": "youtube_mediapipe",
        "youtube": cfg["youtube"],
        "side": cfg["side"],
        "exercises": exercises_out,
        "poses": flat,
    }


def extract_and_save(exercise_key: str, output_path: str | None = None) -> str:
    cfg = EXERCISE_SEGMENTS[exercise_key]
    if output_path is None:
        output_path = os.path.join(ROOT, "database", f"reference_{exercise_key}.json")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[{exercise_key}] Tải video YouTube...")
        video_path = download_youtube(cfg["youtube"], tmp)
        ranges = [(float(e["time_start"]), float(e["time_end"])) for e in cfg["exercises"].values()]
        print(f"[{exercise_key}] Trich xuat goc MediaPipe tu {video_path} (doan {ranges})...")
        frames = extract_angles_from_video(video_path, side=cfg["side"], time_ranges=ranges)
        data = build_reference_json(exercise_key, frames)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_ex = len(data["exercises"])
    n_poses = len(data["poses"])
    print(f"[{exercise_key}] Đã lưu {n_poses} tư thế / {n_ex} động tác → {output_path}")
    return output_path


def main() -> int:
    keys = sys.argv[1:] if len(sys.argv) > 1 else ["codman", "gay", "day"]
    for key in keys:
        if key not in EXERCISE_SEGMENTS:
            print(f"Bỏ qua key không hợp lệ: {key}")
            continue
        extract_and_save(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
