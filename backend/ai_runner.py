"""Opt-in MediaPipe AI runner for backend analysis jobs.

This module deliberately does not import ``app.py``. It builds the dependency
object expected by ``video.processing`` from small backend-safe helpers and the
already extracted ``utils`` / ``video`` modules.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from backend.analysis_jobs import AnalysisJobRequest
from video.io import build_h264_transcode_command, final_h264_path, temp_h264_path
from video.metrics import recalc_metrics, segment_frames, tinh_metrics_chi_tiet


ProcessingFn = Callable[..., tuple[Any, ...]]
ProgressFn = Callable[..., None]


@dataclass(frozen=True)
class BackendAIOptions:
    model_type: str = "MediaPipe Heavy"
    min_confidence: float = 0.5
    skip_step: int | None = 0
    resize_width: int | None = 720
    force_train_classifier: bool = False
    enable_pose_classifier: bool = False
    ffmpeg_threads: int = 2
    transcode_timeout_seconds: int = 1800


EXERCISES: dict[str, dict[str, Any]] = {
    "codman": {
        "ten": "Bai tap con lac Codman",
        "chuan": {"sai_so": 30, "kieu": "dynamic"},
    },
    "gay": {
        "ten": "Bai tap voi gay (Pulley Exercise)",
        "chuan": {"sai_so": 30, "kieu": "dynamic"},
    },
    "khang_luc": {
        "ten": "Bai tap voi day khang luc (Theraband Exercise)",
        "chuan": {"sai_so": 30, "kieu": "dynamic"},
    },
}


def normalize_exercise_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(keyword in text for keyword in ("gay", "gậy", "pulley", "stick")):
        return "gay"
    if any(keyword in text for keyword in ("day", "dây", "khang", "kháng", "theraband", "band")):
        return "khang_luc"
    return "codman"


def reference_exercise_name(ref_name: str) -> str:
    if ref_name == "gay":
        return "Bai tap voi gay (Pulley Exercise)"
    if ref_name == "day":
        return "Bai tap voi day khang luc (Theraband)"
    return "Bai tap con lac Codman"


def phase_error_from_label(value: Any, *, default: int = 30) -> int:
    text = str(value or "").lower()
    if "g1" in text or "giai doan 1" in text or "giai đoạn 1" in text:
        return 45
    if "g3" in text or "giai doan 3" in text or "giai đoạn 3" in text:
        return 15
    if "g2" in text or "giai doan 2" in text or "giai đoạn 2" in text:
        return 30
    return default


def tinh_goc(a: Any, b: Any, c: Any) -> float:
    pa, pb, pc = np.array(a), np.array(b), np.array(c)
    ba = pa - pb
    bc = pc - pb
    cos_value = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def ve_cung_tron_goc(image: Any, point1: Any, center: Any, point3: Any, angle: float, color: Any, radius: int = 40) -> Any:
    del angle
    try:
        import cv2

        v1 = np.array(point1) - np.array(center)
        v2 = np.array(point3) - np.array(center)
        angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
        angle2 = np.degrees(np.arctan2(v2[1], v2[0]))
        overlay = image.copy()
        cv2.ellipse(overlay, center, (radius, radius), 0, angle1, angle2, color, -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        cv2.ellipse(image, center, (radius, radius), 0, angle1, angle2, color, 2)
    except Exception:
        pass
    return image


def ve_khung_xuong_custom(
    frame_output: Any,
    current_landmarks: Any,
    active_side: str | None = None,
    mau_tong: tuple[int, int, int] = (0, 255, 0),
    scale_factor: float = 1.0,
) -> Any:
    try:
        import cv2

        height, width = frame_output.shape[:2]
        landmarks = current_landmarks.landmark
        points = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in range(33)]
        line_thickness = max(2, int(3 * scale_factor))
        circle_radius = max(3, int(4 * scale_factor))
        left_links = [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (23, 25), (25, 27), (27, 29), (27, 31)]
        right_links = [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (24, 26), (26, 28), (28, 30), (28, 32)]
        torso_links = [(11, 12), (11, 23), (12, 24), (23, 24)]
        face_links = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (6, 8), (9, 10)]

        for start, end in torso_links + face_links:
            cv2.line(frame_output, points[start], points[end], (180, 180, 180), max(1, line_thickness - 1))
        for start, end in left_links:
            color = mau_tong if active_side in {"LEFT", "BOTH"} else (180, 180, 180)
            cv2.line(frame_output, points[start], points[end], color, line_thickness)
        for start, end in right_links:
            color = mau_tong if active_side in {"RIGHT", "BOTH"} else (180, 180, 180)
            cv2.line(frame_output, points[start], points[end], color, line_thickness)
        for idx, point in enumerate(points):
            is_left = idx in {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
            is_active = active_side == "BOTH" or (active_side == "LEFT" and is_left) or (active_side == "RIGHT" and not is_left)
            color = (0, 235, 255) if is_active and idx >= 11 else (240, 240, 240)
            cv2.circle(frame_output, point, circle_radius if idx >= 11 else max(2, circle_radius - 1), color, -1)
    except Exception:
        pass
    return frame_output


def _draw_rule_badge_fallback(frame_output: Any, dung: bool, gan_dung: bool, scale_factor: float = 1.0) -> Any:
    try:
        import cv2

        label = "PASS" if dung else ("NEAR" if gan_dung else "FAIL")
        color = (0, 210, 0) if dung else ((0, 180, 255) if gan_dung else (0, 0, 220))
        pad = int(10 * scale_factor)
        cv2.rectangle(frame_output, (pad, pad), (pad + int(120 * scale_factor), pad + int(36 * scale_factor)), color, -1)
        cv2.putText(
            frame_output,
            label,
            (pad + int(8 * scale_factor), pad + int(25 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * scale_factor,
            (255, 255, 255),
            max(1, int(2 * scale_factor)),
        )
    except Exception:
        pass
    return frame_output


def _draw_ml_badge_fallback(frame_output: Any, ml_info: dict[str, Any] | None, scale_factor: float = 1.0) -> Any:
    if not ml_info:
        return frame_output
    try:
        import cv2

        label = str(ml_info.get("ml_label") or ml_info.get("label") or "ML")
        pad = int(10 * scale_factor)
        y = pad + int(44 * scale_factor)
        cv2.rectangle(frame_output, (pad, y), (pad + int(150 * scale_factor), y + int(32 * scale_factor)), (40, 40, 40), -1)
        cv2.putText(
            frame_output,
            label[:16],
            (pad + int(8 * scale_factor), y + int(22 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale_factor,
            (255, 255, 255),
            max(1, int(1.5 * scale_factor)),
        )
    except Exception:
        pass
    return frame_output


def get_pose_model(model_type: str = "MediaPipe Heavy", min_confidence: float = 0.5) -> Any:
    import mediapipe as mp

    complexity = 1
    if "Lite" in str(model_type):
        complexity = 0
    elif "Heavy" in str(model_type):
        complexity = 2
    last_error: Exception | None = None
    for model_complexity in (complexity, 1, 0):
        try:
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"could not initialize MediaPipe Pose: {last_error}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and value != value:
        return 0.0
    return value


def build_stats_data(
    df: Any,
    exercise_config: dict[str, Any],
    *,
    valid_frames: int,
    total_frames: int,
    warnings: list[str],
    sai_so: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    metrics = tinh_metrics_chi_tiet(df, exercise_config)
    is_gay_ex = any(keyword in str(exercise_config.get("ten") or "").lower() for keyword in ("gay", "gậy", "pulley", "stick"))
    if is_gay_ex:
        metrics_overall = recalc_metrics(df, sai_so, exercise_config.get("ten", ""))
        metrics_g1 = metrics_overall
        metrics_g2 = metrics_overall
        metrics_g3 = metrics_overall
        metrics["ty_le_tong_the"] = metrics_overall.get("do_chinh_xac", metrics_overall.get("ty_le_tong_the", 0.0))
    else:
        n0, n1, n2, n3 = segment_frames(df)
        metrics_g1 = recalc_metrics(df.iloc[n0:n1], 45, exercise_config.get("ten", ""))
        metrics_g2 = recalc_metrics(df.iloc[n1:n2], 30, exercise_config.get("ten", ""))
        metrics_g3 = recalc_metrics(df.iloc[n2:n3], 15, exercise_config.get("ten", ""))

    return _json_safe({
        "do_chinh_xac": _safe_float(metrics.get("ty_le_tong_the")),
        "ty_le_gan_dung": _safe_float(metrics.get("ty_le_gan_dung")),
        "ty_le_vai_dung": _safe_float(metrics.get("ty_le_vai_dung")),
        "ty_le_khuyu_dung": _safe_float(metrics.get("ty_le_khuyu_dung")),
        "frame_dung": int(_safe_float(metrics.get("frame_dung"))),
        "frame_gan_dung": int(_safe_float(metrics.get("frame_gan_dung"))),
        "tong_frame_hop_le": int(valid_frames),
        "tb_goc_vai": _safe_float(metrics.get("tb_goc_vai")),
        "tb_goc_khuyu": _safe_float(metrics.get("tb_goc_khuyu")),
        "min_goc_vai": _safe_float(metrics.get("min_goc_vai")),
        "max_goc_vai": _safe_float(metrics.get("max_goc_vai")),
        "min_goc_khuyu": _safe_float(metrics.get("min_goc_khuyu")),
        "max_goc_khuyu": _safe_float(metrics.get("max_goc_khuyu")),
        "std_goc_vai": _safe_float(metrics.get("std_goc_vai")),
        "std_goc_khuyu": _safe_float(metrics.get("std_goc_khuyu")),
        "mae_tong": _safe_float(metrics.get("mae_tong")),
        "precision": _safe_float(metrics.get("precision")),
        "recall": _safe_float(metrics.get("recall")),
        "f1_score": _safe_float(metrics.get("f1_score")),
        "icc": _safe_float(metrics.get("icc")),
        "tb_vai_chuan": _safe_float(metrics.get("tb_vai_chuan"), 90.0),
        "tb_khuyu_chuan": _safe_float(metrics.get("tb_khuyu_chuan"), 170.0),
        "thoi_gian": _safe_float(elapsed_seconds),
        "tong_frame": int(total_frames),
        "warnings": warnings,
        "metrics_g1": metrics_g1,
        "metrics_g2": metrics_g2,
        "metrics_g3": metrics_g3,
    })


class BackendMediaPipeAIRunner:
    is_backend_mediapipe_ai_runner = True

    def __init__(
        self,
        *,
        repo_root: Path,
        database_dir: Path,
        processed_dir: Path,
        options: BackendAIOptions | None = None,
        processing_fn: ProcessingFn | None = None,
        command_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.database_dir = database_dir
        self.processed_dir = processed_dir
        self.options = options or BackendAIOptions()
        self.processing_fn = processing_fn
        self.command_runner = command_runner or subprocess.run

    def __call__(self, request: AnalysisJobRequest, analysis_input_path: str, progress: ProgressFn) -> dict[str, Any]:
        started_at = time.time()
        try:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            deps = self._build_processing_deps()
            processing_fn = self.processing_fn or self._load_processing_fn()
            exercise_key = normalize_exercise_key(request.options.get("exercise_key") or request.exercise)
            exercise_config = {**EXERCISES.get(exercise_key, EXERCISES["codman"])}
            exercise_config["chuan"] = dict(exercise_config.get("chuan") or {})
            sai_so = phase_error_from_label(
                request.options.get("giai_doan") or request.options.get("phase"),
                default=int(exercise_config["chuan"].get("sai_so", 30)),
            )
            exercise_config["chuan"]["sai_so"] = sai_so

            progress(status="processing", progress=0.43, status_msg="AI runner dang khoi tao MediaPipe.")

            output = processing_fn(
                deps,
                duong_dan_video=analysis_input_path,
                chuan=exercise_config["chuan"],
                callback=self._progress_mapper(progress),
                model_type=str(request.options.get("model_type") or self.options.model_type),
                min_confidence=float(request.options.get("min_confidence") or self.options.min_confidence),
                exercise_name=exercise_config["ten"],
                skip_step=request.options.get("skip_step", self.options.skip_step),
                resize_width=request.options.get("resize_width", self.options.resize_width),
                force_train_classifier=bool(
                    request.options.get("force_train_classifier", self.options.force_train_classifier)
                ),
                checkpoint_video_path=request.video_path,
            )
            parsed = self._parse_processing_output(output)
            if parsed["valid_frames"] <= 0 or not parsed["angle_data"]:
                return {
                    "status": "error",
                    "progress": 1.0,
                    "status_msg": "AI khong nhan dien duoc khung hinh hop le.",
                    "error_msg": "no valid pose frames",
                    "result": parsed,
                }

            import pandas as pd

            progress(status="processing", progress=0.94, status_msg="Dang tinh metrics va ghi CSV.")
            df = pd.DataFrame(parsed["angle_data"])
            stats_data = build_stats_data(
                df,
                exercise_config,
                valid_frames=parsed["valid_frames"],
                total_frames=parsed["total_frames"],
                warnings=parsed["warnings"],
                sai_so=sai_so,
                elapsed_seconds=time.time() - started_at,
            )
            df_csv_path = str(parsed["processed_path"]).replace(".mp4", "_data.csv")
            df.to_csv(df_csv_path, index=False)
            result = {
                "processed_path": parsed["processed_path"],
                "processed_video_path": parsed["processed_path"],
                "metrics": stats_data,
                "stats": stats_data,
                "df_path": df_csv_path,
                "all_frames_data_path": parsed["all_frames_data_path"],
                "frames_zip": parsed["frames_zip_path"],
                "frames_zip_path": parsed["frames_zip_path"],
                "accuracy": round(_safe_float(stats_data.get("do_chinh_xac")), 1),
                "exercise": reference_exercise_name(parsed["ref_name"]),
                "sai_so": sai_so,
                "giai_doan": request.options.get("giai_doan") or request.options.get("phase") or "",
                "analysis_input_path": analysis_input_path,
                "total_frames": parsed["total_frames"],
                "valid_frames": parsed["valid_frames"],
                "warnings": parsed["warnings"],
            }
            return {
                "status": "success",
                "progress": 1.0,
                "status_msg": "AI da phan tich xong.",
                "result": result,
            }
        except Exception as exc:
            return {
                "status": "error",
                "progress": 1.0,
                "status_msg": "AI runner gap loi.",
                "error_msg": str(exc),
            }

    def _build_processing_deps(self) -> SimpleNamespace:
        import cv2
        import pandas as pd
        from utils import checkpoint_utils, reference_utils

        if self.options.enable_pose_classifier:
            from utils import pose_classifier_utils

            create_pose_classifier_predictor = pose_classifier_utils.create_pose_classifier_predictor
            ensure_classifier_ready = pose_classifier_utils.ensure_classifier_ready
            train_pose_classifier = pose_classifier_utils.train_pose_classifier
            draw_rule_badge = pose_classifier_utils.draw_rule_badge
            draw_ml_badge = pose_classifier_utils.draw_ml_badge
        else:
            create_pose_classifier_predictor = None
            ensure_classifier_ready = None
            train_pose_classifier = None
            draw_rule_badge = _draw_rule_badge_fallback
            draw_ml_badge = _draw_ml_badge_fallback

        return SimpleNamespace(
            st=None,
            cv2=cv2,
            pd=pd,
            np=np,
            DB_DIR=str(self.database_dir),
            PROCESSED_DIR=str(self.processed_dir),
            MAX_FRAMES=0,
            SKIP_FRAMES=self.options.skip_step or 0,
            RESIZE_WIDTH=self.options.resize_width or 720,
            PHASE_ERROR=reference_utils.PHASE_ERROR,
            PHASE_ERROR_DEFAULT=reference_utils.PHASE_ERROR_DEFAULT,
            NEAR_ERROR_MULTIPLIER=reference_utils.NEAR_ERROR_MULTIPLIER,
            get_phase_error_for_segment=reference_utils.get_phase_error_for_segment,
            phase_frame_label=reference_utils.phase_frame_label,
            resolve_reference_file=reference_utils.resolve_reference_file,
            load_reference_poses=reference_utils.load_reference_poses,
            detect_motion_subtype=reference_utils.detect_motion_subtype,
            find_closest_reference_pose=reference_utils.find_closest_reference_pose,
            get_checkpoint_path=checkpoint_utils.get_checkpoint_path,
            build_config_hash=checkpoint_utils.build_config_hash,
            load_checkpoint=checkpoint_utils.load_checkpoint,
            save_checkpoint=checkpoint_utils.save_checkpoint,
            clear_checkpoint=checkpoint_utils.clear_checkpoint,
            checkpoint_ui_progress=checkpoint_utils.checkpoint_ui_progress,
            CHECKPOINT_INTERVAL_PASS2=checkpoint_utils.CHECKPOINT_INTERVAL_PASS2,
            assemble_video_from_jpgs=checkpoint_utils.assemble_video_from_jpgs,
            serialize_pass1_item=checkpoint_utils.serialize_pass1_item,
            deserialize_pass1_item=checkpoint_utils.deserialize_pass1_item,
            get_final_h264_path=final_h264_path,
            sync_transcode_to_h264=self._sync_transcode_to_h264,
            ensure_voice_files=lambda force_voice=False: "",
            _xoa_cache_h264_video=self._remove_h264_cache,
            _day_progress_checkpoint_len_hf=lambda *args, **kwargs: None,
            get_pose_model=get_pose_model,
            tinh_goc=tinh_goc,
            ve_cung_tron_goc=ve_cung_tron_goc,
            ve_khung_xuong_custom=ve_khung_xuong_custom,
            ve_nhan_rule_classifier=lambda frame, dung, gan_dung, scale_factor=1.0: draw_rule_badge(
                frame, dung, gan_dung, scale_factor=scale_factor
            ),
            ve_nhan_ml_classifier=lambda frame, ml_info, scale_factor=1.0: draw_ml_badge(
                frame, ml_info, scale_factor=scale_factor
            ),
            create_pose_classifier_predictor=create_pose_classifier_predictor,
            ensure_classifier_ready=ensure_classifier_ready,
            train_pose_classifier=train_pose_classifier,
        )

    def _sync_transcode_to_h264(
        self,
        src_path: str,
        dst_path: str | None = None,
        audio_path: str | None = None,
        timeout: int | None = None,
        on_tick: Callable[[], None] | None = None,
    ) -> str | None:
        if not src_path or not os.path.exists(src_path):
            return None
        dst = dst_path or final_h264_path(src_path)
        if not dst:
            return None
        tmp_dst = temp_h264_path(dst)
        for cleanup_path in (tmp_dst,):
            try:
                if os.path.exists(cleanup_path):
                    os.remove(cleanup_path)
            except OSError:
                pass
        has_audio = bool(audio_path and os.path.exists(audio_path))
        cmd = build_h264_transcode_command(
            src_path,
            tmp_dst,
            audio_path=audio_path,
            audio_exists=has_audio,
            ffmpeg_threads=self.options.ffmpeg_threads,
        )
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.time() + float(timeout or self.options.transcode_timeout_seconds)
            last_tick = time.time()
            while proc.poll() is None:
                if time.time() > deadline:
                    proc.kill()
                    return None
                if on_tick and time.time() - last_tick >= 2.0:
                    on_tick()
                    last_tick = time.time()
                time.sleep(0.5)
            _, stderr_text = proc.communicate(timeout=10)
            if proc.returncode != 0:
                print(f"[Backend AI Transcode] ffmpeg failed ({proc.returncode}): {(stderr_text or '')[-800:]}")
                return None
            if not os.path.exists(tmp_dst) or os.path.getsize(tmp_dst) <= 0:
                return None
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_dst, dst)
            return dst
        except Exception as exc:
            print(f"[Backend AI Transcode] error: {exc}")
            return None
        finally:
            try:
                if os.path.exists(tmp_dst):
                    os.remove(tmp_dst)
            except OSError:
                pass

    def _remove_h264_cache(self, video_path: str | None) -> None:
        if not video_path:
            return
        for candidate in {video_path, final_h264_path(video_path)}:
            if candidate and str(candidate).endswith("_f.mp4"):
                try:
                    if os.path.exists(candidate):
                        os.remove(candidate)
                except OSError:
                    pass

    def _load_processing_fn(self) -> ProcessingFn:
        from video.processing import xu_ly_video_day_du

        return xu_ly_video_day_du

    def _progress_mapper(self, progress: ProgressFn) -> Callable[..., None]:
        def _mapped(internal_progress: float, frame_count: int | None = None, total_frames: int | None = None) -> None:
            try:
                inner = max(0.0, min(1.0, float(internal_progress or 0.0)))
            except (TypeError, ValueError):
                inner = 0.0
            mapped = min(0.99, 0.42 + inner * 0.55)
            if frame_count is not None and total_frames:
                status_msg = f"AI dang phan tich frame {frame_count}/{total_frames}."
            else:
                status_msg = "AI dang phan tich video."
            progress(status="processing", progress=mapped, status_msg=status_msg)

        return _mapped

    def _parse_processing_output(self, output: tuple[Any, ...]) -> dict[str, Any]:
        values = tuple(output or ())
        return {
            "processed_path": str(values[0] or "") if len(values) > 0 else "",
            "ref_name": str(values[1] or "codman") if len(values) > 1 else "codman",
            "angle_data": list(values[3] or []) if len(values) > 3 else [],
            "total_frames": int(values[4] or 0) if len(values) > 4 else 0,
            "valid_frames": int(values[5] or 0) if len(values) > 5 else 0,
            "frames_dir": str(values[6] or "") if len(values) > 6 else "",
            "frames_zip_path": str(values[7] or "") if len(values) > 7 and values[7] else "",
            "frame_paths": list(values[8] or []) if len(values) > 8 else [],
            "all_frames_data_path": str(values[10] or "") if len(values) > 10 and values[10] else "",
            "warnings": list(values[11] or []) if len(values) > 11 else [],
        }
