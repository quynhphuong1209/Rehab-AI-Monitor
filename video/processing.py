"""Core frame and video processing helpers.

The functions still receive app-level services through deps while Phase 5 keeps
behavior unchanged. This module intentionally does not import Streamlit.
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from video.metrics import segment_frames


def _bind_deps(deps: Any) -> None:
    if deps is None:
        return
    globals().update(
        {k: v for k, v in vars(deps).items() if not k.startswith("__")}
    )


def _session_get(key: str, default: Any = None) -> Any:
    session_state = getattr(globals().get("st", None), "session_state", None)
    if session_state is None:
        return default
    try:
        return session_state.get(key, default)
    except Exception:
        return default


def _session_set(key: str, value: Any) -> None:
    session_state = getattr(globals().get("st", None), "session_state", None)
    if session_state is None:
        return
    try:
        session_state[key] = value
    except Exception:
        pass


def _toast(message: str, icon: str | None = None) -> None:
    toast_fn = getattr(globals().get("st", None), "toast", None)
    if not toast_fn:
        return
    try:
        toast_fn(message, icon=icon)
    except Exception:
        pass


def xu_ly_frame(deps, frame, model, chuan, frame_idx, fps=30, dynamic_chuan=None, active_side=None, last_pose_landmarks=None, precomputed_landmarks=None, exercise_name="codman"):
    _bind_deps(deps)

    # 1. LẤY KÍCH THƯỚC VÀ CHUYỂN ĐỔI MÀU (Không dùng padding gây lệch)
    h, w = frame.shape[:2]

    frame_output = frame.copy()
    GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)
    YELLOW, CYAN, ORANGE = (0, 255, 255), (255, 255, 0), (0, 165, 255)

    thoi_gian_giay = frame_idx / fps
    phut = int(thoi_gian_giay // 60)
    giay = int(thoi_gian_giay % 60)
    timestamp_str = f"{phut:02d}:{giay:02d}"

    # Sử dụng landmarks hiện tại hoặc khôi phục từ frame trước nếu mất dấu
    current_landmarks = None
    if precomputed_landmarks is not None:
        current_landmarks = precomputed_landmarks
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. AI XỬ LÝ TRỰC TIẾP TRÊN FRAME GỐC
        ket_qua = model.process(rgb) if model else None
        if ket_qua and ket_qua.pose_landmarks:
            current_landmarks = ket_qua.pose_landmarks
        elif last_pose_landmarks:
            current_landmarks = last_pose_landmarks
        del rgb
        del ket_qua

    if not current_landmarks:
        # Box thông tin khi không có pose - CẢI THIỆN
        overlay = frame_output.copy()
        cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
        frame_output = cv2.addWeighted(overlay, 0.6, frame_output, 0.4, 0)
        cv2.rectangle(frame_output, (10, 10), (400, 130), (255, 255, 255), 2)
        cv2.putText(frame_output, f"FRAME #{frame_idx}", (20, 45),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_output, f"TIME: {timestamp_str}", (20, 80),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame_output, "NO POSE DETECTED", (20, 115),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
        gc.collect()
        return frame_output, None, None, None, None, [], None

    # Import cục bộ để tránh phụ thuộc vào biến global
    import mediapipe as _mp
    _mp_drawing = _mp.solutions.drawing_utils
    _mp_drawing_styles = _mp.solutions.drawing_styles
    _mp_pose = _mp.solutions.pose

    # LẤY TỌA ĐỘ CÁC KHỚP QUAN TRỌNG (ĐẢM BẢO KHỚP 100% VỚI FRAME)
    lm = current_landmarks.landmark
    def get_coords(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    # Bên trái
    vai_t = get_coords(_mp_pose.PoseLandmark.LEFT_SHOULDER)
    khuyu_t = get_coords(_mp_pose.PoseLandmark.LEFT_ELBOW)
    co_tay_t = get_coords(_mp_pose.PoseLandmark.LEFT_WRIST)
    hong_t = get_coords(_mp_pose.PoseLandmark.LEFT_HIP)

    # Bên phải
    vai_p = get_coords(_mp_pose.PoseLandmark.RIGHT_SHOULDER)
    khuyu_p = get_coords(_mp_pose.PoseLandmark.RIGHT_ELBOW)
    co_tay_p = get_coords(_mp_pose.PoseLandmark.RIGHT_WRIST)
    hong_p = get_coords(_mp_pose.PoseLandmark.RIGHT_HIP)

    # Các điểm khác để vẽ khung xương
    mui = get_coords(_mp_pose.PoseLandmark.NOSE)
    tai_t = get_coords(_mp_pose.PoseLandmark.LEFT_EAR)
    tai_p = get_coords(_mp_pose.PoseLandmark.RIGHT_EAR)
    dau_goi_t = get_coords(_mp_pose.PoseLandmark.LEFT_KNEE)
    dau_goi_p = get_coords(_mp_pose.PoseLandmark.RIGHT_KNEE)
    co_chan_t = get_coords(_mp_pose.PoseLandmark.LEFT_ANKLE)
    co_chan_p = get_coords(_mp_pose.PoseLandmark.RIGHT_ANKLE)

    # TÍNH TOÁN GÓC CẢ HAI BÊN
    goc_vai_t = tinh_goc(hong_t, vai_t, khuyu_t)
    goc_khuyu_t = tinh_goc(vai_t, khuyu_t, co_tay_t)

    goc_vai_p = tinh_goc(hong_p, vai_p, khuyu_p)
    goc_khuyu_p = tinh_goc(vai_p, khuyu_p, co_tay_p)

    # Chọn bên tay tập chủ đạo (ưu tiên khóa tay tập nếu có active_side)
    if active_side == "LEFT":
        goc_vai, goc_khuyu = goc_vai_t, goc_khuyu_t
        khop_chinh = vai_t
        khop_phu = khuyu_t
        pts_vai = (hong_t, vai_t, khuyu_t)
        pts_khuyu = (vai_t, khuyu_t, co_tay_t)
    elif active_side == "RIGHT":
        goc_vai, goc_khuyu = goc_vai_p, goc_khuyu_p
        khop_chinh = vai_p
        khop_phu = khuyu_p
        pts_vai = (hong_p, vai_p, khuyu_p)
        pts_khuyu = (vai_p, khuyu_p, co_tay_p)
    else:
        # Tự động chọn bên đang tập động (nếu không khóa cứng)
        if abs(goc_vai_t - 10) > abs(goc_vai_p - 10):
            goc_vai, goc_khuyu = goc_vai_t, goc_khuyu_t
            khop_chinh = vai_t
            khop_phu = khuyu_t
            pts_vai = (hong_t, vai_t, khuyu_t)
            pts_khuyu = (vai_t, khuyu_t, co_tay_t)
        else:
            goc_vai, goc_khuyu = goc_vai_p, goc_khuyu_p
            khop_chinh = vai_p
            khop_phu = khuyu_p
            pts_vai = (hong_p, vai_p, khuyu_p)
            pts_khuyu = (vai_p, khuyu_p, co_tay_p)

    # MẶC ĐỊNH (Nếu không có dynamic, hệ thống sẽ dùng số an toàn nhưng ưu tiên tuyệt đối dynamic)
    thoi_gian_giay = frame_idx / fps
    chuan_vai = chuan.get("vai", 90)
    chuan_khuyu = chuan.get("khuyu", 170)

    # NẾU CÓ DỮ LIỆU DYNAMIC (BẢN CHUẨN YOUTUBE) -> KHỚP TƯ THẾ GẦN NHẤT (KHÔNG THEO TỪNG GIÂY)
    chuan_vai_t, chuan_khuyu_t = chuan_vai, chuan_khuyu
    chuan_vai_p, chuan_khuyu_p = chuan_vai, chuan_khuyu
    motion_subtype_detected = None
    if dynamic_chuan:
        is_gay_ex = any(kw in str(exercise_name or '').lower() for kw in ["gậy", "gay", "pulley", "stick"])
        is_day_ex = any(kw in str(exercise_name or '').lower() for kw in ["dây", "day", "kháng lực", "khang", "theraband", "band"])
        is_both_sides = is_gay_ex or is_day_ex

        current_vai = (goc_vai_t + goc_vai_p) / 2 if is_both_sides else goc_vai
        current_khuyu = (goc_khuyu_t + goc_khuyu_p) / 2 if is_both_sides else goc_khuyu
        motion_subtype_detected = detect_motion_subtype(
            exercise_name,
            current_vai,
            current_khuyu,
            vai_trai=goc_vai_t,
            vai_phai=goc_vai_p,
            khuyu_trai=goc_khuyu_t,
            khuyu_phai=goc_khuyu_p,
        )

        closest_ref = find_closest_reference_pose(
            dynamic_chuan,
            current_vai,
            current_khuyu,
            exercise_name,
            vai_trai=goc_vai_t,
            vai_phai=goc_vai_p,
            khuyu_trai=goc_khuyu_t,
            khuyu_phai=goc_khuyu_p,
            motion_subtype=motion_subtype_detected,
        )

        if closest_ref:
            chuan_vai = closest_ref.get('vai', chuan_vai)
            chuan_khuyu = closest_ref.get('khuyu', chuan_khuyu)
            chuan_vai_t = closest_ref.get('vai_trai', chuan_vai)
            chuan_khuyu_t = closest_ref.get('khuyu_trai', chuan_khuyu)
            chuan_vai_p = closest_ref.get('vai_phai', chuan_vai)
            chuan_khuyu_p = closest_ref.get('khuyu_phai', chuan_khuyu)

    ss = chuan["sai_so"]

    ex_clean = str(exercise_name or '').lower()
    is_gay = any(kw in ex_clean for kw in ["gậy", "gay", "pulley", "stick"])
    is_codman = any(kw in ex_clean for kw in ["codman"])
    is_day = any(kw in ex_clean for kw in ["dây", "day", "kháng lực", "khang", "theraband", "band"])

    if is_gay or is_day:
        vai_diff_t = abs(goc_vai_t - chuan_vai_t)
        vai_diff_p = abs(goc_vai_p - chuan_vai_p)
        khuyu_diff_t = abs(goc_khuyu_t - chuan_khuyu_t)
        khuyu_diff_p = abs(goc_khuyu_p - chuan_khuyu_p)

        vai_diff = max(vai_diff_t, vai_diff_p)
        khuyu_diff = max(khuyu_diff_t, khuyu_diff_p)

        # Để thống kê / hiển thị, lấy góc trung bình hai bên
        goc_vai = (goc_vai_t + goc_vai_p) / 2
        goc_khuyu = (goc_khuyu_t + goc_khuyu_p) / 2
    elif is_codman:
        # Chỉ so sánh góc bên phải
        vai_diff = abs(goc_vai_p - chuan_vai)
        khuyu_diff = abs(goc_khuyu_p - chuan_khuyu)
        goc_vai = goc_vai_p
        goc_khuyu = goc_khuyu_p
    else:
        # Mặc định theo active_side
        vai_diff = abs(goc_vai - chuan_vai)
        khuyu_diff = abs(goc_khuyu - chuan_khuyu)

    vai_dung = vai_diff <= ss
    khuyu_dung = khuyu_diff <= ss

    # Gần đúng: Trong khoảng 1.5 lần sai số
    vai_gan_dung = vai_diff <= (ss * 1.5)
    khuyu_gan_dung = khuyu_diff <= (ss * 1.5)

    # Logic tổng thể:
    # - ĐÚNG: Cả hai đều đúng
    # - GẦN ĐÚNG: Không đạt "đúng" nhưng cả hai đều nằm trong vùng "gần đúng"
    # - SAI: Vượt quá ngưỡng gần đúng

    tong_the = vai_dung and khuyu_dung
    gan_dung_tong_the = (vai_gan_dung and khuyu_gan_dung) and not tong_the

    # TÍNH TOÁN 3 GIAI ĐOẠN (sai số cho phép G1±45°, G2±30°, G3±15°) CHO HIỂN THỊ TRÊN FRAME
    def _phase_status(v_diff, k_diff, threshold):
        v_ok = v_diff <= threshold
        k_ok = k_diff <= threshold
        v_near = v_diff <= (threshold * NEAR_ERROR_MULTIPLIER)
        k_near = k_diff <= (threshold * NEAR_ERROR_MULTIPLIER)
        if v_ok and k_ok:
            return "PASS", (0, 200, 80)
        elif v_near and k_near:
            return "NEAR", (0, 165, 255)
        else:
            return "FAIL", (0, 0, 220)

    g1_text, g1_color = _phase_status(vai_diff, khuyu_diff, PHASE_ERROR["g1"])
    g2_text, g2_color = _phase_status(vai_diff, khuyu_diff, PHASE_ERROR["g2"])
    g3_text, g3_color = _phase_status(vai_diff, khuyu_diff, PHASE_ERROR["g3"])

    # MÀU SẮC: Xanh (Đúng), Cam (Gần đúng), Đỏ (Sai)
    ORANGE_BGR = (0, 165, 255)
    mau_vai = (0, 255, 0) if vai_dung else (ORANGE_BGR if vai_gan_dung else (0, 0, 255))
    mau_khuyu = (0, 255, 0) if khuyu_dung else (ORANGE_BGR if khuyu_gan_dung else (0, 0, 255))
    mau_tong = (0, 255, 0) if tong_the else (ORANGE_BGR if gan_dung_tong_the else (0, 0, 255))

    # Tính riêng cho Trái/Phải phục vụ vẽ góc (chuẩn YouTube từng bên khi có)
    vai_diff_t = abs(goc_vai_t - chuan_vai_t)
    khuyu_diff_t = abs(goc_khuyu_t - chuan_khuyu_t)
    vai_dung_t = vai_diff_t <= ss
    khuyu_dung_t = khuyu_diff_t <= ss
    vai_gan_dung_t = vai_diff_t <= (ss * 1.5)
    khuyu_gan_dung_t = khuyu_diff_t <= (ss * 1.5)
    mau_vai_t = (0, 255, 0) if vai_dung_t else (ORANGE_BGR if vai_gan_dung_t else (0, 0, 255))
    mau_khuyu_t = (0, 255, 0) if khuyu_dung_t else (ORANGE_BGR if khuyu_gan_dung_t else (0, 0, 255))

    vai_diff_p = abs(goc_vai_p - chuan_vai_p)
    khuyu_diff_p = abs(goc_khuyu_p - chuan_khuyu_p)
    vai_dung_p = vai_diff_p <= ss
    khuyu_dung_p = khuyu_diff_p <= ss
    vai_gan_dung_p = vai_diff_p <= (ss * 1.5)
    khuyu_gan_dung_p = khuyu_diff_p <= (ss * 1.5)
    mau_vai_p = (0, 255, 0) if vai_dung_p else (ORANGE_BGR if vai_gan_dung_p else (0, 0, 255))
    mau_khuyu_p = (0, 255, 0) if khuyu_dung_p else (ORANGE_BGR if khuyu_gan_dung_p else (0, 0, 255))

    scale_factor = w / 640.0
    line_thickness = max(2, int(2 * scale_factor))
    circle_rad = max(2, int(2 * scale_factor))

    font_scale = 0.8 * scale_factor
    font_scale_small = 0.55 * scale_factor
    font_scale_tiny = 0.38 * scale_factor
    font_scale_mini = 0.42 * scale_factor
    font_scale_medium = 0.72 * scale_factor
    font_scale_large = 0.78 * scale_factor
    font_scale_g = 0.52 * scale_factor
    text_thick = max(1, int(2 * scale_factor))
    text_thick_thin = max(1, int(1 * scale_factor))

    # === 0. VẼ KHUNG XƯƠNG ĐỘNG 33 ĐIỂM TỰ VẼ ===
    custom_active_side = "BOTH" if exercise_name == "gay" else ("RIGHT" if exercise_name == "codman" else active_side)
    ve_khung_xuong_custom(frame_output, current_landmarks, active_side=custom_active_side, mau_tong=mau_tong, scale_factor=scale_factor)

    # === 1. VẼ HEADER TRÊN CÙNG (TOP BAR) ===
    header_h = int(35 * scale_factor)
    cv2.rectangle(frame_output, (0, 0), (w, header_h), (10, 10, 10), -1) # Nền đen
    cv2.rectangle(frame_output, (0, 0), (w, header_h), (80, 80, 80), text_thick)    # Viền xám trung tính
    cv2.putText(frame_output, f"Frame #{frame_idx}", (w // 2 - int(50 * scale_factor), int(22 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale_factor, (255, 255, 255), text_thick)

    # === 2. VẼ CUNG TRÒN VÀ SỐ ĐO TẠI KHỚP (JOINT LABELS) ===
    # Vẽ góc và cung tròn tùy thuộc vào bài tập
    if exercise_name == "gay":
        # Vẽ cả hai bên (Trái và Phải)
        # Bên Trái
        ve_cung_tron_goc(frame_output, hong_t, vai_t, khuyu_t, goc_vai_t, mau_vai_t, radius=int(35 * scale_factor))
        ve_cung_tron_goc(frame_output, vai_t, khuyu_t, co_tay_t, goc_khuyu_t, mau_khuyu_t, radius=int(30 * scale_factor))
        cv2.putText(frame_output, f"{int(goc_vai_t)}", (vai_t[0] - int(45 * scale_factor), vai_t[1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_vai_t, text_thick)
        cv2.putText(frame_output, f"{int(goc_khuyu_t)}", (khuyu_t[0] - int(45 * scale_factor), khuyu_t[1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_khuyu_t, text_thick)

        # Bên Phải
        ve_cung_tron_goc(frame_output, hong_p, vai_p, khuyu_p, goc_vai_p, mau_vai_p, radius=int(35 * scale_factor))
        ve_cung_tron_goc(frame_output, vai_p, khuyu_p, co_tay_p, goc_khuyu_p, mau_khuyu_p, radius=int(30 * scale_factor))
        cv2.putText(frame_output, f"{int(goc_vai_p)}", (vai_p[0] + int(15 * scale_factor), vai_p[1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_vai_p, text_thick)
        cv2.putText(frame_output, f"{int(goc_khuyu_p)}", (khuyu_p[0] + int(15 * scale_factor), khuyu_p[1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_khuyu_p, text_thick)
    elif exercise_name == "codman":
        # Chỉ vẽ bên Phải (Tay tập cố định)
        ve_cung_tron_goc(frame_output, hong_p, vai_p, khuyu_p, goc_vai_p, mau_vai_p, radius=int(35 * scale_factor))
        ve_cung_tron_goc(frame_output, vai_p, khuyu_p, co_tay_p, goc_khuyu_p, mau_khuyu_p, radius=int(30 * scale_factor))
        cv2.putText(frame_output, f"{int(goc_vai_p)}", (vai_p[0] + int(15 * scale_factor), vai_p[1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_vai_p, text_thick)
        cv2.putText(frame_output, f"{int(goc_khuyu_p)}", (khuyu_p[0] + int(15 * scale_factor), khuyu_p[1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_khuyu_p, text_thick)
    else:
        # Vẽ theo active_side mặc định (cho bài tập khác như Dây kháng lực)
        ve_cung_tron_goc(frame_output, pts_vai[0], pts_vai[1], pts_vai[2], goc_vai, mau_vai, radius=int(35 * scale_factor))
        ve_cung_tron_goc(frame_output, pts_khuyu[0], pts_khuyu[1], pts_khuyu[2], goc_khuyu, mau_khuyu, radius=int(30 * scale_factor))
        cv2.putText(frame_output, f"{int(goc_vai)}", (pts_vai[1][0] + int(15 * scale_factor), pts_vai[1][1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_vai, text_thick)
        cv2.putText(frame_output, f"{int(goc_khuyu)}", (pts_khuyu[1][0] + int(15 * scale_factor), pts_khuyu[1][1] - int(15 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, mau_khuyu, text_thick)

    # === 3. VẼ BOX THÔNG TIN CHI TIẾT (TOP-LEFT BOX) — MỞ RỘNG 3 GIAI ĐOẠN ===
    box_x, box_y = int(15 * scale_factor), int(50 * scale_factor)
    is_gay = any(kw in str(exercise_name or '').lower() for kw in ["gậy", "gay", "pulley", "stick"])
    box_w = int(345 * scale_factor)
    box_h = int(205 * scale_factor) if is_gay else int(235 * scale_factor)   # Tăng chiều cao để chứa thêm 3 giai đoạn (gậy ẩn 3 giai đoạn nên thu nhỏ)

    # TỐI ƯU: Chỉ tạo overlay cho vùng Box thay vì toàn bộ frame
    # Đảm bảo box không vượt quá biên frame
    safe_box_y2 = min(box_y + box_h, h)
    safe_box_x2 = min(box_x + box_w, w)
    box_roi = frame_output[box_y:safe_box_y2, box_x:safe_box_x2]
    overlay = box_roi.copy()
    cv2.rectangle(overlay, (0, 0), (safe_box_x2 - box_x, safe_box_y2 - box_y), (20, 20, 35), -1)
    cv2.addWeighted(overlay, 0.72, box_roi, 0.28, 0, box_roi)
    frame_output[box_y:safe_box_y2, box_x:safe_box_x2] = box_roi
    cv2.rectangle(frame_output, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), text_thick)

    # Text thông tin trong Box
    status_text = "PASS" if tong_the else ("NEARLY" if gan_dung_tong_the else "FAIL")
    CYAN = (255, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dòng 1: FRAME & STATUS
    cv2.putText(frame_output, f"FRAME #{frame_idx}", (box_x + int(15 * scale_factor), box_y + int(28 * scale_factor)), font, font_scale_medium, CYAN, text_thick)
    # Không vẽ status_text tổng thể tại đây để tránh mâu thuẫn giữa các giai đoạn (được quản lý động bằng HTML Card Badge)

    # Dòng 2: TIME
    time_sec = frame_idx / fps
    time_str = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
    has_real_pose = (precomputed_landmarks is not None) or (ket_qua and ket_qua.pose_landmarks if 'ket_qua' in locals() and ket_qua else False)
    if not has_real_pose and last_pose_landmarks:
        cv2.putText(frame_output, f"TIME: {time_str} (EST)", (box_x + int(15 * scale_factor), box_y + int(52 * scale_factor)), font, font_scale_small, (0, 165, 255), text_thick_thin)
    else:
        cv2.putText(frame_output, f"TIME: {time_str}", (box_x + int(15 * scale_factor), box_y + int(52 * scale_factor)), font, font_scale_small, (180, 180, 180), text_thick_thin)

    sep_y = box_y + int(162 * scale_factor) if is_gay else box_y + int(120 * scale_factor)

    if is_gay:
        # Dòng 3 (LEFT SIDE)
        cv2.putText(frame_output, "LEFT SIDE: SHOULDER / ELBOW", (box_x + int(15 * scale_factor), box_y + int(78 * scale_factor)), font, 0.44 * scale_factor, (200, 200, 200), text_thick_thin)
        cv2.putText(frame_output, f"{int(goc_vai_t)}", (box_x + int(15 * scale_factor), box_y + int(102 * scale_factor)), font, font_scale_medium, mau_vai_t, text_thick)
        cv2.putText(frame_output, f"/{int(chuan_vai)}", (box_x + int(60 * scale_factor), box_y + int(102 * scale_factor)), font, font_scale_tiny, (140, 140, 140), text_thick_thin)

        cv2.putText(frame_output, f"{int(goc_khuyu_t)}", (box_x + int(180 * scale_factor), box_y + int(102 * scale_factor)), font, font_scale_medium, mau_khuyu_t, text_thick)
        cv2.putText(frame_output, f"/{int(chuan_khuyu)}", (box_x + int(225 * scale_factor), box_y + int(102 * scale_factor)), font, font_scale_tiny, (140, 140, 140), text_thick_thin)

        # Dòng 4 (RIGHT SIDE)
        cv2.putText(frame_output, "RIGHT SIDE: SHOULDER / ELBOW", (box_x + int(15 * scale_factor), box_y + int(126 * scale_factor)), font, 0.44 * scale_factor, (200, 200, 200), text_thick_thin)
        cv2.putText(frame_output, f"{int(goc_vai_p)}", (box_x + int(15 * scale_factor), box_y + int(150 * scale_factor)), font, font_scale_medium, mau_vai_p, text_thick)
        cv2.putText(frame_output, f"/{int(chuan_vai)}", (box_x + int(60 * scale_factor), box_y + int(150 * scale_factor)), font, font_scale_tiny, (140, 140, 140), text_thick_thin)

        cv2.putText(frame_output, f"{int(goc_khuyu_p)}", (box_x + int(180 * scale_factor), box_y + int(150 * scale_factor)), font, font_scale_medium, mau_khuyu_p, text_thick)
        cv2.putText(frame_output, f"/{int(chuan_khuyu)}", (box_x + int(225 * scale_factor), box_y + int(150 * scale_factor)), font, font_scale_tiny, (140, 140, 140), text_thick_thin)
    else:
        # Chỉ vẽ bên tập chủ đạo hoặc bên Phải
        cv2.putText(frame_output, "SHOULDER", (box_x + int(15 * scale_factor), box_y + int(82 * scale_factor)), font, 0.48 * scale_factor, (200, 200, 200), text_thick_thin)
        cv2.putText(frame_output, f"{int(goc_vai)}", (box_x + int(15 * scale_factor), box_y + int(108 * scale_factor)), font, font_scale_large, mau_vai, text_thick)
        cv2.putText(frame_output, f"/{int(chuan_vai)}", (box_x + int(70 * scale_factor), box_y + int(108 * scale_factor)), font, font_scale_small, (140, 140, 140), text_thick_thin)
        cv2.putText(frame_output, "ELBOW", (box_x + int(180 * scale_factor), box_y + int(82 * scale_factor)), font, 0.48 * scale_factor, (200, 200, 200), text_thick_thin)
        cv2.putText(frame_output, f"{int(goc_khuyu)}", (box_x + int(180 * scale_factor), box_y + int(108 * scale_factor)), font, font_scale_large, mau_khuyu, text_thick)
        cv2.putText(frame_output, f"/{int(chuan_khuyu)}", (box_x + int(240 * scale_factor), box_y + int(108 * scale_factor)), font, font_scale_small, (140, 140, 140), text_thick_thin)

    # === ĐƯỜNG PHÂN CÁCH VÀ CẢNH BÁO / 3 GIAI ĐOẠN ===
    cv2.line(frame_output, (box_x + int(8 * scale_factor), sep_y), (box_x + box_w - int(8 * scale_factor), sep_y), (80, 80, 100), text_thick_thin)

    if not is_gay:
        cv2.putText(frame_output, "3 GIAI DOAN (ss+/-45/30/15):", (box_x + int(15 * scale_factor), sep_y + int(17 * scale_factor)), font, font_scale_mini, (150, 200, 255), text_thick_thin)

        g1_label = phase_frame_label("g1", g1_text)
        cv2.putText(frame_output, g1_label, (box_x + int(15 * scale_factor), sep_y + int(40 * scale_factor)), font, font_scale_g, g1_color, text_thick)

        g2_label = phase_frame_label("g2", g2_text)
        cv2.putText(frame_output, g2_label, (box_x + int(15 * scale_factor), sep_y + int(63 * scale_factor)), font, font_scale_g, g2_color, text_thick)

        g3_label = phase_frame_label("g3", g3_text)
        cv2.putText(frame_output, g3_label, (box_x + int(15 * scale_factor), sep_y + int(86 * scale_factor)), font, font_scale_g, g3_color, text_thick)

        warnings_list = get_warning_message(goc_vai, goc_khuyu, chuan_vai, chuan_khuyu, ss)
        if warnings_list:
            w_text = warnings_list[0][:38] + ".." if len(warnings_list[0]) > 38 else warnings_list[0]
            cv2.putText(frame_output, f"! {w_text}", (box_x + int(15 * scale_factor), sep_y + int(108 * scale_factor)), font, font_scale_tiny, (0, 255, 255), text_thick_thin)
    else:
        warnings_list = get_warning_message(goc_vai, goc_khuyu, chuan_vai, chuan_khuyu, ss)
        if warnings_list:
            w_text = warnings_list[0][:38] + ".." if len(warnings_list[0]) > 38 else warnings_list[0]
            cv2.putText(frame_output, f"! {w_text}", (box_x + int(15 * scale_factor), sep_y + int(25 * scale_factor)), font, font_scale_tiny, (0, 255, 255), text_thick_thin)

    # Đảm bảo trả về kiểu dữ liệu Python chuẩn
    goc_vai = float(goc_vai)
    goc_khuyu = float(goc_khuyu)
    tong_the = bool(tong_the)
    vai_dung = bool(vai_dung)
    khuyu_dung = bool(khuyu_dung)
    gan_dung_tong_the = bool(gan_dung_tong_the)

    return frame_output, goc_vai, goc_khuyu, tong_the, {"nearly_correct": gan_dung_tong_the, "shoulder_correct": vai_dung, "elbow_correct": khuyu_dung, "shoulder_ref": float(chuan_vai), "elbow_ref": float(chuan_khuyu)}, warnings_list, current_landmarks

def xu_ly_video_day_du(deps, duong_dan_video, chuan, callback=None, model_type="MediaPipe Heavy", min_confidence=0.5, exercise_name="codman", skip_step=None, resize_width=None, force_train_classifier=False, checkpoint_video_path=None):
    _bind_deps(deps)

    import gc
    import json
    import os

    temp_copy_path = duong_dan_video

    # 1. LOAD DYNAMIC REFERENCE (BẢN CHUẨN YOUTUBE)
    dynamic_chuan = None
    try:
        # Chuẩn hóa tên bài tập và tên file video để so khớp từ khóa
        exercise_name_clean = exercise_name.lower().strip()
        video_filename_clean = os.path.basename(duong_dan_video).lower()

        # Mặc định là codman
        ref_name = "codman"

        # Ưu tiên kiểm tra từ khóa trong tên file video trước (do người dùng đặt tên file cụ thể thường chính xác hơn)
        if "codman" in video_filename_clean:
            ref_name = "codman"
        elif any(kw in video_filename_clean for kw in ["gậy", "gay", "pulley", "stick"]):
            ref_name = "gay"
        elif any(kw in video_filename_clean for kw in ["dây", "day", "kháng lực", "khang", "theraband", "band"]):
            ref_name = "day"
        # Nếu tên file không chứa từ khóa đặc trưng, kiểm tra tên bài tập được chọn từ metadata
        else:
            if "codman" in exercise_name_clean:
                ref_name = "codman"
            elif any(kw in exercise_name_clean for kw in ["gậy", "gay", "pulley", "stick"]):
                ref_name = "gay"
            elif any(kw in exercise_name_clean for kw in ["dây", "day", "kháng lực", "khang", "theraband", "band"]):
                ref_name = "day"
            else:
                ref_name = "codman"

        # Sử dụng đường dẫn tuyệt đối để đảm bảo nạp được file trên mọi môi trường
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ref_file = os.path.join(current_dir, f"reference_{ref_name}.json")

        ref_file_found = resolve_reference_file(ref_name, DB_DIR, current_dir)
        if ref_file_found:
            dynamic_chuan = load_reference_poses(ref_file_found, ref_name)
            if callback: callback(0.01)
            print(f"[AI Process] Da nap {len(dynamic_chuan)} tu the chuan ({ref_name})")
        else:
            print(f"[AI Process] Khong tim thay file chuan: reference_{ref_name}.json")
    except Exception as e:
        print(f"[AI Process] Loi nap chuan: {e}")

    cap = cv2.VideoCapture(duong_dan_video)
    if not cap.isOpened():
        raise Exception(
            f"Video Error — không mở được file: {os.path.basename(duong_dan_video or '')}"
        )

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    fps_export = max(10, fps // 2) # BẬT LẠI LÀM CHẬM 0.5X (SLOW MOTION)
    tong_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if tong_frame <= 0:
        print(f"[AI Process] Cảnh báo: CAP_PROP_FRAME_COUNT trả về {tong_frame}. Thiết lập dự đoán là 1000 frames.")
        tong_frame = 1000
    if MAX_FRAMES and tong_frame > MAX_FRAMES: tong_frame = MAX_FRAMES

    w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Lấy giá trị skip và resolution từ tham số hoặc session_state
    if skip_step is None:
        skip_step = _session_get('ncv_skip_frames', SKIP_FRAMES)
    if resize_width is None:
        resize_width = _session_get('ncv_resize_width', RESIZE_WIDTH)

    # Pass 1 chỉ cần landmarks (tọa độ chuẩn hóa 0-1) → dùng resolution thấp hơn để nhanh hơn.
    # Pass 2 giữ nguyên resize_width gốc cho chất lượng hình ảnh overlay.
    pass1_resize_width = min(int(resize_width), 480)
    if tong_frame > 10000:
        pass1_resize_width = min(pass1_resize_width, 360)

    ckpt_path = get_checkpoint_path(checkpoint_video_path or duong_dan_video, PROCESSED_DIR)
    cfg_hash = build_config_hash(
        checkpoint_video_path or duong_dan_video, model_type, min_confidence,
        exercise_name, skip_step, resize_width
    )
    ckpt = load_checkpoint(ckpt_path)
    ckpt_valid = (
        ckpt
        and ckpt.get("config_hash") == cfg_hash
        and ckpt.get("phase") in ("pass1_done", "pass2")
        and ckpt.get("pass1_data")
    )
    # Không check analysis_input_path — đường dẫn temp thay đổi mỗi lần HF Space restart
    # (config_hash đã đảm bảo đúng video+tham số)
    if ckpt and not ckpt_valid:
        print(f"[Checkpoint] Bo qua checkpoint cu (cau hinh/video khac): {ckpt_path}")
        clear_checkpoint(ckpt_path)
        ckpt = None

    resume_pass1 = False
    pass2_resume_from = 0
    use_jpg_assembly = False

    if ckpt_valid:
        timestamp = ckpt.get("timestamp") or int(time.time())
        out_path = ckpt.get("out_path") or os.path.join(PROCESSED_DIR, f"processed_{timestamp}.mp4")
        thu_muc_frame = ckpt.get("thu_muc_frame") or os.path.join(PROCESSED_DIR, f"processed_{timestamp}_frames")
        import tempfile
        local_temp_dir = ckpt.get("local_temp_dir") or ""
        if not local_temp_dir or not os.path.isdir(local_temp_dir):
            local_temp_dir = tempfile.mkdtemp(prefix=f"frames_processed_{timestamp}_")
        os.makedirs(thu_muc_frame, exist_ok=True)
        raw_pass1_data = [deserialize_pass1_item(x) for x in ckpt.get("pass1_data", [])]
        active_side = ckpt.get("active_side", "RIGHT")
        segment_bounds = ckpt.get("segment_bounds")
        resume_pass1 = True
        if ckpt.get("phase") == "pass2":
            pass2_resume_from = int(ckpt.get("pass2_processed_count") or 0)
            use_jpg_assembly = True
        print(f"[Checkpoint] Resume phase={ckpt.get('phase')} pass2_from={pass2_resume_from} frames={len(raw_pass1_data)}")
        if callback:
            try:
                ui_p, _ = checkpoint_ui_progress(ckpt)
                callback(0.5 if ckpt.get("phase") == "pass1_done" else (0.5 + min(pass2_resume_from / max(len(raw_pass1_data), 1), 1.0) * 0.40))
            except Exception:
                pass
    else:
        timestamp = int(time.time())
        out_path = os.path.join(PROCESSED_DIR, f'processed_{timestamp}.mp4')
        thu_muc_frame = os.path.join(PROCESSED_DIR, f'processed_{timestamp}_frames')
        import tempfile
        local_temp_dir = tempfile.mkdtemp(prefix=f"frames_processed_{timestamp}_")
        raw_pass1_data = []
        active_side = "RIGHT"
        segment_bounds = None

    from concurrent.futures import ThreadPoolExecutor
    img_writer_executor = ThreadPoolExecutor(max_workers=6)
    frame_write_futures = []

    model = None if resume_pass1 else get_pose_model(model_type=model_type, min_confidence=min_confidence)
    du_lieu_goc = list(ckpt.get("du_lieu_goc", [])) if (ckpt_valid and ckpt.get("phase") == "pass2") else []
    danh_sach_frame_paths = list(ckpt.get("danh_sach_frame_paths", [])) if (ckpt_valid and ckpt.get("phase") == "pass2") else []
    danh_sach_frame_data = list(ckpt.get("danh_sach_frame_data", [])) if (ckpt_valid and ckpt.get("phase") == "pass2") else []
    all_warnings = list(ckpt.get("all_warnings", [])) if ckpt_valid else []

    frame_count = 0
    processed_count = 0
    last_progress = 0
    writer = None

    audio_events = list(ckpt.get("audio_events", [])) if (ckpt_valid and ckpt.get("phase") == "pass2") else []
    last_state = ckpt.get("last_state") if (ckpt_valid and ckpt.get("phase") == "pass2") else None
    last_audio_time = float(ckpt.get("last_audio_time", -10.0)) if (ckpt_valid and ckpt.get("phase") == "pass2") else -10.0
    last_pose_landmarks = None
    last_known_center = None
    has_multiple_people_warning = False
    pass1_data_serialized = (
        list(ckpt.get("pass1_data", []))
        if (ckpt_valid and ckpt.get("pass1_data"))
        else None
    )
    ckpt_save_executor = ThreadPoolExecutor(max_workers=1)
    ckpt_save_busy = [False]

    def _persist_checkpoint(phase, pass2_done=0):
        if not ckpt_path:
            return
        nonlocal pass1_data_serialized
        if pass1_data_serialized is None and raw_pass1_data:
            pass1_data_serialized = [serialize_pass1_item(x) for x in raw_pass1_data]
        payload = {
            "config_hash": cfg_hash,
            "video_path": checkpoint_video_path or duong_dan_video,
            "analysis_input_path": duong_dan_video,
            "phase": phase,
            "timestamp": timestamp,
            "out_path": out_path,
            "thu_muc_frame": thu_muc_frame,
            "local_temp_dir": local_temp_dir,
            "ref_name": ref_name,
            "active_side": active_side,
            "segment_bounds": segment_bounds,
            "fps": fps,
            "fps_export": fps_export,
            "tong_frame": tong_frame,
            "skip_step": skip_step,
            "resize_width": resize_width,
            "model_type": model_type,
            "exercise_name": exercise_name,
            "pass1_data": pass1_data_serialized or [],
            "pass2_processed_count": pass2_done,
            "du_lieu_goc": du_lieu_goc,
            "danh_sach_frame_paths": danh_sach_frame_paths,
            "danh_sach_frame_data": danh_sach_frame_data,
            "audio_events": audio_events,
            "last_state": last_state,
            "last_audio_time": last_audio_time,
            "all_warnings": all_warnings,
        }

        def _save_job():
            try:
                ok = save_checkpoint(ckpt_path, payload)
                if ok and phase == "pass1_done":
                    print(f"[Checkpoint] Da luu Pass 1 ({len(raw_pass1_data)} frame) -> {ckpt_path}")
                    _day_progress_checkpoint_len_hf(
                        checkpoint_video_path or duong_dan_video, force=True, progress=0.5, status="processing"
                    )
            finally:
                ckpt_save_busy[0] = False

        if ckpt_save_busy[0] and phase != "pass1_done":
            return
        ckpt_save_busy[0] = True
        if phase == "pass1_done":
            # Pass 1 xong — ghi đồng bộ để thread khác / HF không đọc file dở
            _save_job()
        else:
            ckpt_save_executor.submit(_save_job)

    # Tự động phát hiện bên tay tập chủ đạo (LEFT hoặc RIGHT) để tránh nhảy bên gây lỗi trích xuất
    left_deviations = []
    right_deviations = []
    detect_count_limit = 60

    # PASS 1: Trích xuất landmarks và tọa độ (bỏ qua nếu đã có checkpoint)
    if not resume_pass1:
        gc.collect()  # giải phóng RAM trước khi MediaPipe bắt đầu xử lý

        # Theo dõi tốc độ để tự động hạ model nếu CPU quá chậm
        _current_complexity = 2 if "Heavy" in model_type else (1 if "Full" in model_type else 0)
        _speed_check_done = False   # chỉ check 1 lần sau 30 frames
        _speed_check_done2 = False  # check lần 2 sau khi đã hạ xuống Full
        _frame_times = []
        # Heavy → Full nếu > 70ms/frame; Full → Lite nếu > 45ms/frame
        _THRESH_HEAVY_TO_FULL = 0.070
        _THRESH_FULL_TO_LITE  = 0.045

        def _downgrade_model(to_complexity):
            nonlocal model
            try:
                model.close()
            except Exception:
                pass
            import mediapipe as _mp
            _mp_pose = _mp.solutions.pose
            _names = {2: "Heavy", 1: "Full", 0: "Lite"}
            print(f"[AutoModel] Tốc độ chậm — tự động chuyển sang MediaPipe {_names.get(to_complexity, to_complexity)}")
            model = _mp_pose.Pose(
                static_image_mode=False,
                model_complexity=to_complexity,
                smooth_landmarks=True,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            return to_complexity

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (MAX_FRAMES and processed_count >= MAX_FRAMES): break

                # Nhả CPU / GIL để luồng chính Streamlit phản hồi kịp nút bấm
                time.sleep(0.001)

                frame_count += 1
                if skip_step > 0 and frame_count % (skip_step + 1) != 1:
                    continue

                processed_count += 1

                h_orig, w_orig = frame.shape[:2]
                if w_orig != pass1_resize_width:
                    scale = pass1_resize_width / w_orig
                    new_h = int(h_orig * scale)
                    if new_h % 2 != 0: new_h -= 1
                    frame = cv2.resize(frame, (pass1_resize_width, new_h), interpolation=cv2.INTER_LINEAR)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _t0 = time.time()
                ket_qua = model.process(rgb)
                _frame_times.append(time.time() - _t0)

                # Sau 30 frames đầu: đánh giá tốc độ, tự động hạ model nếu cần
                if not _speed_check_done and len(_frame_times) >= 30:
                    _speed_check_done = True
                    _avg = sum(_frame_times) / len(_frame_times)
                    if _current_complexity == 2 and _avg > _THRESH_HEAVY_TO_FULL:
                        _current_complexity = _downgrade_model(1)
                        _frame_times.clear()   # reset để check lần 2 cho Full
                    elif _current_complexity == 1 and _avg > _THRESH_FULL_TO_LITE:
                        _current_complexity = _downgrade_model(0)

                # Sau khi đã hạ xuống Full, check thêm 30 frames rồi quyết định xuống Lite không
                if _speed_check_done and not _speed_check_done2 and _current_complexity == 1 and len(_frame_times) >= 30:
                    _speed_check_done2 = True
                    _avg2 = sum(_frame_times) / len(_frame_times)
                    if _avg2 > _THRESH_FULL_TO_LITE:
                        _current_complexity = _downgrade_model(0)

                current_landmarks = None
                detected_this_frame = False
                filtered_stranger_this_frame = False
                if ket_qua and ket_qua.pose_landmarks:
                    # Trích xuất trọng tâm thân người (Torso center) để theo dõi và lọc người lạ
                    lm = ket_qua.pose_landmarks.landmark
                    # Dùng trung bình cộng các điểm vai (11, 12) và hông (23, 24) làm trọng tâm đại diện
                    torso_idx = [11, 12, 23, 24]
                    torso_x = [lm[i].x for i in torso_idx]
                    torso_y = [lm[i].y for i in torso_idx]
                    current_center = (sum(torso_x) / 4.0, sum(torso_y) / 4.0)

                    if last_known_center is None:
                        # Lần đầu tiên phát hiện -> Khóa vị trí bệnh nhân
                        last_known_center = current_center
                        current_landmarks = ket_qua.pose_landmarks
                        last_pose_landmarks = current_landmarks
                        detected_this_frame = True
                    else:
                        # Tính khoảng cách dịch chuyển trọng tâm
                        dist = math.sqrt((current_center[0] - last_known_center[0])**2 + (current_center[1] - last_known_center[1])**2)
                        # Nếu khoảng cách nhảy quá lớn (> 0.18 trong hệ tọa độ chuẩn hóa),
                        # chứng tỏ có người khác xuất hiện ở vị trí khác và bị nhận dạng thay thế bệnh nhân
                        if dist <= 0.18:
                            last_known_center = current_center
                            current_landmarks = ket_qua.pose_landmarks
                            last_pose_landmarks = current_landmarks
                            detected_this_frame = True
                        else:
                            filtered_stranger_this_frame = True
                elif last_pose_landmarks is None:
                    # Không có gì cả - thực sự không nhận dạng được người nào
                    detected_this_frame = False
                    filtered_stranger_this_frame = False
                elif last_pose_landmarks:
                    current_landmarks = last_pose_landmarks
                    detected_this_frame = False
                    filtered_stranger_this_frame = False

                goc_v_left, goc_k_left = None, None
                goc_v_right, goc_k_right = None, None

                if current_landmarks:
                    lm = current_landmarks.landmark
                    def get_coords_det(idx):
                        return (int(lm[idx].x * w), int(lm[idx].y * h))

                    goc_v_left = tinh_goc(get_coords_det(23), get_coords_det(11), get_coords_det(13))
                    goc_k_left = tinh_goc(get_coords_det(11), get_coords_det(13), get_coords_det(15))
                    goc_v_right = tinh_goc(get_coords_det(24), get_coords_det(12), get_coords_det(14))
                    goc_k_right = tinh_goc(get_coords_det(12), get_coords_det(14), get_coords_det(16))

                    # Tích lũy độ lệch để tự động phát hiện bên tay tập
                    if len(left_deviations) < detect_count_limit:
                        left_deviations.append(abs(goc_v_left - 10))
                        right_deviations.append(abs(goc_v_right - 10))

                raw_pass1_data.append({
                    'frame_idx': frame_count,
                    'processed_count': processed_count,
                    'landmarks': current_landmarks,
                    'detected': detected_this_frame,           # True = AI thực sự nhận dạng bệnh nhân frame này
                    'filtered_stranger': filtered_stranger_this_frame,  # True = có người lạ bị lọc bỏ
                    'goc_vai_left': goc_v_left,
                    'goc_khuyu_left': goc_k_left,
                    'goc_vai_right': goc_v_right,
                    'goc_khuyu_right': goc_k_right,
                    'goc_vai': None,
                    'goc_khuyu': None
                })

                if callback and tong_frame > 0:
                    if frame_count > tong_frame:
                        tong_frame = frame_count + 100
                    # Dùng processed_count để % không đứng im khi skip_frame > 0
                    frames_to_process = max(1, (tong_frame + skip_step) // (skip_step + 1))
                    prog = min(processed_count / frames_to_process, 1.0) * 0.5
                    # Truyền processed_count/frames_to_process thay vì frame_count/tong_frame
                    # để UI hiện "Frame X/Y đã xử lý" đúng với skip setting
                    callback(prog, frame_count=processed_count, total_frames=frames_to_process)
                    if processed_count % 500 == 1 or processed_count == frames_to_process:
                        print(f"[AI Process] Pass 1: Frame {processed_count}/{frames_to_process} processed (video frame {frame_count}/{tong_frame}, {prog*100:.1f}%)")

                if processed_count % 200 == 0:
                    gc.collect()

        except Exception as e:
            import traceback as _tb
            print(f"[AI Process] LOI PASS 1 tai frame {frame_count}: {e}\n{_tb.format_exc()}")
            if callback:
                try:
                    callback(min(processed_count / max(tong_frame, 1), 0.499) * 0.5,
                             frame_count=frame_count, total_frames=tong_frame)
                except Exception:
                    pass

        # Xác định bên tay tập chủ đạo dựa trên dữ liệu tích lũy
        if ref_name == "codman":
            active_side = "RIGHT"
            print("[AI Process] Bai tap Codman: co dinh TAY PHAI (RIGHT)")
        else:
            active_side = "RIGHT"
            if left_deviations and right_deviations:
                mean_left = float(np.mean(left_deviations))
                mean_right = float(np.mean(right_deviations))
                std_left = float(np.std(left_deviations))
                std_right = float(np.std(right_deviations))

                score_left = mean_left + std_left * 2
                score_right = mean_right + std_right * 2

                if score_left > score_right:
                    active_side = "LEFT"
                else:
                    active_side = "RIGHT"
            _toast(
                f"🤖 AI phát hiện bên tập chủ đạo: {'TAY TRÁI (LEFT)' if active_side == 'LEFT' else 'TAY PHẢI (RIGHT)'}",
                icon="🦾",
            )

        for item in raw_pass1_data:
            if active_side == "LEFT":
                item['goc_vai'] = item['goc_vai_left']
                item['goc_khuyu'] = item['goc_khuyu_left']
            else:
                item['goc_vai'] = item['goc_vai_right']
                item['goc_khuyu'] = item['goc_khuyu_right']

        segment_bounds = segment_frames(raw_pass1_data)
        # Báo 100% Pass 1 trước khi lưu checkpoint — UI thấy "Pass 1: 100%" rõ ràng
        if callback:
            try:
                p1_total = max(1, len(raw_pass1_data))
                callback(0.5, frame_count=p1_total, total_frames=p1_total)
            except Exception:
                pass
        _persist_checkpoint("pass1_done", 0)  # print "Da luu" duoc log trong _save_job sau khi ghi thanh cong
    else:
        if not segment_bounds:
            segment_bounds = segment_frames(raw_pass1_data)

    _session_set("segment_bounds", segment_bounds)
    _session_set("last_processed_video_for_bounds", out_path)
    n0, n1, n2, n3 = segment_bounds

    if resume_pass1:
        force_train_classifier = False

    ml_predict_row = None
    if create_pose_classifier_predictor and ensure_classifier_ready:
        try:
            if force_train_classifier and train_pose_classifier:
                if callback:
                    try: callback(0.501)
                    except: pass
                import threading as _thr_train
                _train_hb_stop = _thr_train.Event()
                def _train_heartbeat():
                    while not _train_hb_stop.wait(5.0):
                        if callback:
                            try:
                                callback(0.503)
                            except Exception:
                                pass
                _train_hb = _thr_train.Thread(target=_train_heartbeat, daemon=True)
                _train_hb.start()
                try:
                    train_state = train_pose_classifier(PROCESSED_DIR, DB_DIR)
                finally:
                    _train_hb_stop.set()
                if train_state.get("success"):
                    _toast("Da cap nhat/train lai ML classifier truoc khi gan nhan.", icon="ML")
                else:
                    print(f"[Pose Classifier] Khong train lai duoc, se thu nap model hien co: {train_state.get('message')}")
                if callback:
                    try: callback(0.505)
                    except: pass
            if callback and not (force_train_classifier and train_pose_classifier):
                try: callback(0.502)
                except: pass
            _clf_hb_stop = threading.Event()
            def _clf_heartbeat():
                while not _clf_hb_stop.wait(5.0):
                    if callback:
                        try: callback(0.503)
                        except: pass
            _clf_hb_thread = threading.Thread(target=_clf_heartbeat, daemon=True)
            _clf_hb_thread.start()
            try:
                clf_state = ensure_classifier_ready(PROCESSED_DIR, DB_DIR, auto_train=True)
            finally:
                _clf_hb_stop.set()
            if clf_state.get("ready"):
                ml_predict_row = create_pose_classifier_predictor(DB_DIR)
                if clf_state.get("trained"):
                    _toast("Da tu dong train pose classifier tu CSV da trich xuat.", icon="ML")
                _toast("Da nap model ML classifier cho video hien tai.", icon="ML")
        except Exception as ml_load_err:
            print(f"[Pose Classifier] Khong the nap/train model truoc Pass 2: {ml_load_err}")

    # PASS 2: Reset video capture và vẽ đè/ghi video với sai số động theo giai đoạn
    if cap: cap.release()

    # Pass 2 đọc lại trực tiếp từ video sau khi cap đã release.
    # Không copy nguyên video GB sang *_pass2.mp4 vì bước copy làm progress đứng rất lâu.
    temp_copy_path = duong_dan_video
    if callback:
        try: callback(0.505)
        except: pass

    cap = cv2.VideoCapture(temp_copy_path)

    frame_count = 0
    processed_count = 0
    last_state = None
    last_audio_time = -10.0

    try:
        while cap.isOpened() and processed_count < len(raw_pass1_data):
            ret, frame = cap.read()
            if not ret: break

            # Nhả CPU / GIL để luồng chính Streamlit phản hồi kịp nút bấm
            time.sleep(0.001)

            frame_count += 1
            if skip_step > 0 and frame_count % (skip_step + 1) != 1:
                continue

            if processed_count < pass2_resume_from:
                processed_count += 1
                continue

            p1_data = raw_pass1_data[processed_count]
            processed_count += 1

            h_orig, w_orig = frame.shape[:2]
            if w_orig != resize_width:
                scale = resize_width / w_orig
                new_h = int(h_orig * scale)
                if new_h % 2 != 0: new_h -= 1
                frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_LINEAR)

            # Sai số cho phép theo giai đoạn PHCN (G1±45°, G2±30°, G3±15°)
            is_gay_ex = any(kw in str(ref_name or '').lower() for kw in ["gậy", "gay", "pulley", "stick"])
            if is_gay_ex:
                ss_dynamic = chuan.get("sai_so", PHASE_ERROR_DEFAULT)
            else:
                ss_dynamic = get_phase_error_for_segment(processed_count - 1, n0, n1, n2, n3)

            chuan_dynamic = chuan.copy()
            chuan_dynamic['sai_so'] = ss_dynamic

            try:
                xu_ly, goc_v, goc_k, dung, eval_info, warnings_list, _ = xu_ly_frame(
                    deps, frame, None, chuan_dynamic, frame_count, fps,
                    dynamic_chuan=dynamic_chuan, active_side=active_side,
                    last_pose_landmarks=None,
                    precomputed_landmarks=p1_data['landmarks'],
                    exercise_name=ref_name
                )

                if goc_v is not None:
                    current_state = "sai"
                    if dung:
                        current_state = "dung"
                    elif eval_info and eval_info.get("nearly_correct"):
                        current_state = "gan_dung"
                else:
                    current_state = last_state

                ts_frame_export = frame_count / fps_export
                if current_state != last_state:
                    if ts_frame_export - last_audio_time >= 1.5:
                        audio_events.append({"time": ts_frame_export, "state": current_state})
                        last_audio_time = ts_frame_export
                        last_state = current_state

            except Exception as e:
                print(f"Lỗi vẽ đè frame {frame_count}: {e}")
                continue

            persistent_frame_path = os.path.join(thu_muc_frame, f"f_{processed_count:06d}.jpg")
            local_frame_path = os.path.join(local_temp_dir, f"f_{processed_count:06d}.jpg")
            danh_sach_frame_paths.append(persistent_frame_path)

            ts_frame_goc = frame_count / fps
            time_str = f"{int(ts_frame_goc // 60):02d}:{int(ts_frame_goc % 60):02d}"

            if warnings_list: all_warnings.extend(warnings_list)

            d_frame = {
                'index': frame_count, 'timestamp': time_str, 'path': persistent_frame_path,
                'goc_vai': goc_v, 'goc_khuyu': goc_k,
                'goc_vai_trai': p1_data.get('goc_vai_left'), 'goc_khuyu_trai': p1_data.get('goc_khuyu_left'),
                'goc_vai_phai': p1_data.get('goc_vai_right'), 'goc_khuyu_phai': p1_data.get('goc_khuyu_right'),
                'dung': dung,
                'gan_dung': eval_info['nearly_correct'] if eval_info else False,
                'eval_info': eval_info if eval_info else {},
                'detected': p1_data.get('detected', False),            # AI thực sự nhận dạng BN frame này
                'filtered_stranger': p1_data.get('filtered_stranger', False)  # Có người lạ bị lọc bỏ
            }
            danh_sach_frame_data.append(d_frame)

            row_data = {
                'frame': frame_count, 'timestamp': time_str, 'timestamp_seconds': ts_frame_goc,
                'goc_vai': goc_v, 'goc_khuyu': goc_k,
                'goc_vai_trai': p1_data.get('goc_vai_left'), 'goc_khuyu_trai': p1_data.get('goc_khuyu_left'),
                'goc_vai_phai': p1_data.get('goc_vai_right'), 'goc_khuyu_phai': p1_data.get('goc_khuyu_right'),
                'dung': dung if goc_v is not None else False,
                'gan_dung': eval_info['nearly_correct'] if (eval_info and goc_v is not None) else False,
                'vai_dung': eval_info['shoulder_correct'] if (eval_info and goc_v is not None) else False,
                'khuyu_dung': eval_info['elbow_correct'] if (eval_info and goc_v is not None) else False,
                'vai_chuan': eval_info['shoulder_ref'] if (eval_info and goc_v is not None) else 90.0,
                'khuyu_chuan': eval_info['elbow_ref'] if (eval_info and goc_v is not None) else 170.0
            }

            if p1_data['landmarks'] is not None:
                for idx, lm_pt in enumerate(p1_data['landmarks'].landmark):
                    row_data[f"pt{idx}_x"] = lm_pt.x
                    row_data[f"pt{idx}_y"] = lm_pt.y
                    row_data[f"pt{idx}_z"] = lm_pt.z
                    row_data[f"pt{idx}_vis"] = lm_pt.visibility
            else:
                for idx in range(33):
                    row_data[f"pt{idx}_x"] = None
                    row_data[f"pt{idx}_y"] = None
                    row_data[f"pt{idx}_z"] = None
                    row_data[f"pt{idx}_vis"] = None

            ml_info = None
            if ml_predict_row and goc_v is not None:
                try:
                    ml_info = ml_predict_row(row_data)
                    row_data.update(ml_info)
                    d_frame.update(ml_info)
                except Exception as ml_pred_err:
                    print(f"[Pose Classifier] Khong the du doan ML frame {frame_count}: {ml_pred_err}")

            if ml_info:
                try:
                    scale_lbl = xu_ly.shape[1] / 640.0
                    rule_dung = bool(dung) if goc_v is not None else False
                    rule_gan = bool(eval_info.get("nearly_correct")) if eval_info else False
                    ve_nhan_rule_classifier(xu_ly, rule_dung, rule_gan, scale_factor=scale_lbl)
                    ve_nhan_ml_classifier(xu_ly, ml_info, scale_factor=scale_lbl)
                except Exception as ml_draw_err:
                    print(f"[Pose Classifier] Khong the ve nhan frame {frame_count}: {ml_draw_err}")
            elif goc_v is not None:
                try:
                    scale_lbl = xu_ly.shape[1] / 640.0
                    rule_gan = bool(eval_info.get("nearly_correct")) if eval_info else False
                    ve_nhan_rule_classifier(xu_ly, bool(dung), rule_gan, scale_factor=scale_lbl)
                except Exception as rule_draw_err:
                    print(f"[Rule Classifier] Khong the ve nhan REF frame {frame_count}: {rule_draw_err}")

            if not use_jpg_assembly:
                if writer is None:
                    curr_h, curr_w = xu_ly.shape[:2]
                    curr_w -= curr_w % 2
                    curr_h -= curr_h % 2
                    if curr_w < 2 or curr_h < 2:
                        curr_w, curr_h = max(2, curr_w), max(2, curr_h)
                    if curr_w != xu_ly.shape[1] or curr_h != xu_ly.shape[0]:
                        xu_ly = xu_ly[:curr_h, :curr_w]
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_export, (curr_w, curr_h))
                writer.write(xu_ly)

            # Save extracted frames after all overlays have been drawn.
            try:
                frame_write_futures.append(
                    img_writer_executor.submit(cv2.imwrite, local_frame_path, xu_ly, [cv2.IMWRITE_JPEG_QUALITY, 80])
                )
            except Exception as write_err:
                print("Loi submit ghi anh:", write_err)

            du_lieu_goc.append(row_data)

            if callback and tong_frame > 0:
                p_len = len(raw_pass1_data)
                # Pass 2 chỉ đi tới 90%; phần sau dành cho chờ ghi ảnh/ZIP/đóng gói H.264.
                prog = 0.5 + (min(processed_count / p_len, 1.0) * 0.40 if p_len > 0 else 0.40)
                callback(prog, frame_count=processed_count, total_frames=p_len)
                if processed_count % 500 == 1 or processed_count == p_len:
                    print(f"[AI Process] Pass 2: Frame {processed_count}/{p_len} (Tiến độ: {prog*100:.1f}%)")

            if processed_count % CHECKPOINT_INTERVAL_PASS2 == 0 or processed_count == len(raw_pass1_data):
                _persist_checkpoint("pass2", processed_count)

            if processed_count % 500 == 0:
                gc.collect()
    except Exception as e:
        print("Lỗi trong Pass 2:", e)
    finally:
        if cap: cap.release()
        if writer: writer.release()
        if model:
            try: model.close()
            except: pass
        if os.path.exists(temp_copy_path) and temp_copy_path != duong_dan_video:
            try: os.unlink(temp_copy_path)
            except: pass
        if 'img_writer_executor' in locals():
            if frame_write_futures:
                total_futures = len(frame_write_futures)
                for fut_idx, fut in enumerate(frame_write_futures, start=1):
                    try:
                        fut.result()
                    except Exception as write_wait_err:
                        print("Loi cho ghi anh:", write_wait_err)
                    if callback and (fut_idx % 25 == 0 or fut_idx == total_futures):
                        try:
                            callback(0.90 + min(fut_idx / max(total_futures, 1), 1.0) * 0.02)
                        except:
                            pass
            img_writer_executor.shutdown(wait=False)
        if 'ckpt_save_executor' in locals():
            ckpt_save_executor.shutdown(wait=True)
        gc.collect()

    if use_jpg_assembly:
        if writer:
            try:
                writer.release()
            except Exception:
                pass
            writer = None
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 5 * 1024:
            print(f"[Checkpoint] Ghep video tu {len(du_lieu_goc)} anh JPG...")
            if not assemble_video_from_jpgs(local_temp_dir, out_path, fps_export):
                warn_asm = "Khong ghep duoc video tu checkpoint JPG; can chay lai Pass 2."
                print(f"[Checkpoint] {warn_asm}")
                all_warnings.append(warn_asm)

    # SAU KHI XỬ LÝ XONG, TIẾN HÀNH TRỘN ÂM THANH NẾU CÓ THAY ĐỔI
    audio_mixed = False
    mixed_audio_path = out_path.replace('.mp4', '_audio.wav')
    # Pass 2 hoàn tất — báo 100% trước khi chuyển sang bước đóng gói
    if callback:
        try:
            _p2_total = len(raw_pass1_data) if raw_pass1_data else 1
            callback(0.92, frame_count=_p2_total, total_frames=_p2_total)
        except Exception:
            pass
    if callback:
        try: callback(0.925)
        except: pass

    if not audio_events:
        warn_audio = "Khong co su kien am thanh — tư thế khong doi hoac khong nhan dien duoc goc khop."
        print(f"[Audio] {warn_audio}")
        all_warnings.append(warn_audio)

    try:
        from pydub import AudioSegment
        sounds_dir = ensure_voice_files(force_voice=True)
        if sounds_dir and audio_events:
            total_duration_ms = int((tong_frame / fps_export) * 1000) + 1000
            # GIẢI PHÁP CHỐNG SẬP WEB (OOM ERROR): Giới hạn 40 sự kiện âm thanh để tránh tràn RAM
            from pydub import AudioSegment
            if len(audio_events) > 40:
                step = len(audio_events) // 40
                audio_events = audio_events[::step][:40]

            sounds = {}
            for s in ["dung", "gan_dung", "sai"]:
                sp = os.path.join(sounds_dir, f"{s}.mp3")
                if os.path.exists(sp):
                    sounds[s] = AudioSegment.from_mp3(sp)
            if not sounds:
                raise RuntimeError("Thieu file am thanh dung/gan_dung/sai trong thu muc sounds/")

            final_audio = AudioSegment.empty()
            last_ms = 0

            for ev in audio_events:
                state = ev['state']
                time_ms = int(ev['time'] * 1000)
                if state in sounds:
                    snd = sounds[state]
                    silence_dur = time_ms - last_ms

                    if silence_dur > 0:
                        final_audio += AudioSegment.silent(duration=silence_dur)
                    elif silence_dur < 0:
                        continue # Tránh lỗi âm thanh bị đè nếu thao tác quá nhanh

                    final_audio += snd
                    last_ms = time_ms + len(snd)

                    gc.collect() # Dọn rác bộ nhớ ngay lập tức để không bị sập RAM

            # Thêm khoảng lặng cuối video nếu cần
            if total_duration_ms > last_ms:
                final_audio += AudioSegment.silent(duration=total_duration_ms - last_ms)
            else:
                final_audio = final_audio[:total_duration_ms]

            final_audio.export(mixed_audio_path, format="wav")
            del final_audio
            audio_mixed = True
            gc.collect()
    except Exception as e:
        warn_mix = f"Loi tron am thanh: {e}"
        print(f"[Audio] {warn_mix}")
        all_warnings.append(warn_mix)

    gc.collect()

    # Tự động tạo tệp ZIP của các khung hình, không tốn RAM do ghi trực tiếp (z.write)
    zip_path = out_path.replace('.mp4', '_frames.zip')
    try:
        import zipfile
        total_zip_frames = max(len(danh_sach_frame_data), 1)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for z_idx, f_data in enumerate(danh_sach_frame_data, start=1):
                f_name = os.path.basename(f_data.get('path'))
                local_f_path = os.path.join(local_temp_dir, f_name)
                if os.path.exists(local_f_path):
                    z.write(local_f_path, f_name)
                if callback and (z_idx % 100 == 0 or z_idx == total_zip_frames):
                    try:
                        callback(0.94 + min(z_idx / total_zip_frames, 1.0) * 0.015)
                    except:
                        pass
    except Exception as e:
        print(f"Lỗi tự động tạo file ZIP frames: {e}")
        zip_path = None

    json_path = os.path.join(PROCESSED_DIR, f'f_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(danh_sach_frame_data, f, ensure_ascii=False)

    if callback:
        try: callback(0.96)
        except: pass

    final_video_path = out_path
    final_h264 = get_final_h264_path(out_path)
    if callback:
        try:
            callback(0.96)
        except Exception:
            pass
    audio_aux = mixed_audio_path if (audio_mixed and os.path.exists(mixed_audio_path)) else None
    if audio_aux:
        _xoa_cache_h264_video(final_h264)
    start_transcode_time = time.time()

    def _transcode_tick():
        if callback:
            try:
                elapsed_t = time.time() - start_transcode_time
                mock_prog = 0.96 + min(elapsed_t / 60.0, 1.0) * 0.03
                callback(mock_prog)
            except Exception:
                pass

    h264_out = sync_transcode_to_h264(out_path, final_h264, audio_path=audio_aux, on_tick=_transcode_tick)
    if h264_out:
        final_video_path = h264_out
    else:
        warn_h264 = "Khong chuyen duoc video sang H.264; file tai ve co the khong mo duoc tren Windows."
        print(f"[Integrity] {warn_h264}")
        all_warnings.append(warn_h264)
    if callback:
        try:
            elapsed_t = time.time() - start_transcode_time
            mock_prog = 0.96 + min(elapsed_t / 30.0, 1.0) * 0.03
            callback(mock_prog)
        except Exception:
            pass

    gc.collect()
    valid_count = sum(1 for row in du_lieu_goc if row['goc_vai'] is not None)

    # ============================================================
    # KIỂM TRA TOÀN VẸN CUỐI PIPELINE (ĐẢM BẢO KHÔNG SÓT FRAME)
    # ============================================================
    expected_frames = len(raw_pass1_data)
    frame_data_count = len(danh_sach_frame_data)
    csv_row_count = len(du_lieu_goc)

    # Đếm số ảnh .jpg đã ghi thực tế trước khi dọn thư mục tạm
    jpg_written = 0
    if 'local_temp_dir' in locals() and local_temp_dir and os.path.exists(local_temp_dir):
        try:
            jpg_written = sum(1 for fn in os.listdir(local_temp_dir) if fn.lower().endswith('.jpg'))
        except Exception as count_err:
            print(f"[Integrity] Khong dem duoc anh jpg: {count_err}")

    # Đếm số frame trong file JSON đã lưu
    json_frame_count = 0
    try:
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as jf:
                json_frame_count = len(json.load(jf))
    except Exception as json_err:
        print(f"[Integrity] Khong doc duoc JSON frame: {json_err}")

    print(
        f"[Integrity] Pass1={expected_frames} | Pass2_frame_data={frame_data_count} | "
        f"CSV_rows={csv_row_count} | JPG={jpg_written} | JSON={json_frame_count}"
    )

    # Cảnh báo nếu Pass 2 xử lý thiếu so với Pass 1 (lệch > 1 frame để bỏ qua sai số làm tròn cuối video)
    if expected_frames > 0 and (expected_frames - frame_data_count) > 1:
        missing = expected_frames - frame_data_count
        warn_msg = (
            f"⚠️ Phát hiện thiếu {missing} khung hình khi vẽ nhãn "
            f"(Pass 1: {expected_frames}, Pass 2: {frame_data_count}). "
            "Khuyến nghị chạy lại phân tích để đảm bảo đầy đủ frame."
        )
        print(f"[Integrity] {warn_msg}")
        all_warnings.append(warn_msg)

    # Cảnh báo nếu số ảnh JPG ghi ra thiếu so với số frame dữ liệu
    if frame_data_count > 0 and jpg_written > 0 and (frame_data_count - jpg_written) > 1:
        warn_msg_img = (
            f"⚠️ Có {frame_data_count - jpg_written} khung hình chưa ghi được ảnh "
            f"(frame data: {frame_data_count}, ảnh JPG: {jpg_written})."
        )
        print(f"[Integrity] {warn_msg_img}")
        all_warnings.append(warn_msg_img)

    # Cảnh báo nếu JSON lưu thiếu frame so với dữ liệu trong RAM
    if frame_data_count > 0 and json_frame_count != frame_data_count:
        warn_msg_json = (
            f"⚠️ File dữ liệu frame (JSON) lưu {json_frame_count}/{frame_data_count} khung hình."
        )
        print(f"[Integrity] {warn_msg_json}")
        all_warnings.append(warn_msg_json)

    # Dọn dẹp thư mục tạm chứa các frame cục bộ để giải phóng dung lượng đĩa
    if 'local_temp_dir' in locals() and local_temp_dir and os.path.exists(local_temp_dir):
        try:
            import shutil
            shutil.rmtree(local_temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"Lỗi dọn dẹp thư mục tạm frames: {cleanup_err}")

    clear_checkpoint(ckpt_path)
    print(f"[Checkpoint] Hoan tat — da xoa checkpoint: {ckpt_path}")

    return final_video_path, ref_name, None, du_lieu_goc, frame_count, valid_count, thu_muc_frame, zip_path, danh_sach_frame_paths, {}, json_path, all_warnings
