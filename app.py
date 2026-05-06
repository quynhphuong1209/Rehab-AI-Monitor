# -*- coding: utf-8 -*-
import os
import sys

# FIX LỖI LIBGL CHO OPENCV TRÊN HEADLESS ENVIRONMENT
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# CẤU HÌNH GPU CHO MEDIAPIPE
os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import zipfile
from io import BytesIO
import subprocess
import hashlib
import threading
import queue
import gc
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

# IMPORT MEDIAPIPE
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    else:
        raise AttributeError("MediaPipe solutions not found")
except (ImportError, AttributeError) as e:
    st.error(f"Lỗi import MediaPipe: {e}")
    import mediapipe.python.solutions as mp_solutions
    mp = mp_solutions
    mp_pose = mp.pose
    mp_drawing = mp.drawing_utils
    mp_drawing_styles = mp.drawing_styles

warnings.filterwarnings("ignore")

# ============================================
# CẤU HÌNH TRANG
# ============================================
st.set_page_config(
    page_title="Hệ thống giám sát tập PHCN từ xa - Đề tài NCKH",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix lỗi hiển thị chữ "upload" và "arrow_down" do trình duyệt không tải được font Material Symbols
st.markdown("""
<style>
.material-symbols-rounded {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

MAX_FILE_SIZE_MB = 500

# ============================================
# CẤU HÌNH XỬ LÝ - CHẤT LƯỢNG CAO
# ============================================
SKIP_FRAMES = 1
RESIZE_WIDTH = 540 # Giảm thêm một chút để ổn định RAM
OUTPUT_QUALITY = 50 # Giảm chất lượng ảnh lưu đĩa để tiết kiệm RAM khi đọc
MAX_FRAMES = 3000 # Nâng lên 3000 frames (đã an toàn nhờ cơ chế lưu JSON trên đĩa)
THUMBNAIL_QUALITY = 90
THUMBNAIL_WIDTH = 400

# ============================================
# HÀM CHUYỂN ĐỔI MOV SANG MP4
# ============================================
def convert_mov_to_mp4(input_path):
    output_path = input_path.replace('.mov', '.mp4').replace('.MOV', '.mp4')
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            return input_path
        
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-preset', 'fast',
            '-crf', '23',
            '-y',
            output_path
        ], check=True, capture_output=True, text=True, timeout=300)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return input_path
    except Exception as e:
        print(f"Lỗi chuyển đổi MOV: {e}")
        return input_path

# ============================================
# KHỞI TẠO SESSION STATE
# ============================================
if 'has_data' not in st.session_state:
    st.session_state.has_data = False
if 'angle_df' not in st.session_state:
    st.session_state.angle_df = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'frames_zip' not in st.session_state:
    st.session_state.frames_zip = None
if 'exercise' not in st.session_state:
    st.session_state.exercise = None
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'output_video_bytes' not in st.session_state:
    st.session_state.output_video_bytes = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = False
if 'current_upload_key' not in st.session_state:
    st.session_state.current_upload_key = None
if 'all_frames_paths' not in st.session_state:
    st.session_state.all_frames_paths = []
if 'temp_video_file' not in st.session_state:
    st.session_state.temp_video_file = None
if 'video_ready' not in st.session_state:
    st.session_state.video_ready = False
if 'frames_ready' not in st.session_state:
    st.session_state.frames_ready = False

if 'frames_loaded' not in st.session_state:
    st.session_state.frames_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'processed_video_bytes' not in st.session_state:
    st.session_state.processed_video_bytes = None
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'all_frames_data' not in st.session_state:
    st.session_state.all_frames_data = []

if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = queue.Queue()
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None

# LỊCH NHẮC NHỞ
if 'appointments' not in st.session_state:
    st.session_state.appointments = []
if 'exercise_reminders' not in st.session_state:
    st.session_state.exercise_reminders = []
if 'medication_reminders' not in st.session_state:
    st.session_state.medication_reminders = []
if 'reminder_id_counter' not in st.session_state:
    st.session_state.reminder_id_counter = 0

# ============================================
# HÀM TÍNH GÓC
# ============================================
def tinh_goc(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

# ============================================
# MEDIAPIPE VỚI GPU
# ============================================
@st.cache_resource
def get_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # Dùng bản Lite để tiết kiệm RAM tối đa
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=False,
        smooth_landmarks=True
    )

# ============================================
# THÔNG BÁO LỖI ĐỘNG TÁC
# ============================================
def get_warning_message(goc_vai, goc_khuyu, chuan_vai, chuan_khuyu, sai_so):
    warnings_list = []
    
    if abs(goc_vai - chuan_vai) > sai_so:
        if goc_vai > chuan_vai:
            warnings_list.append("WARNING: SHOULDER TOO HIGH")
        else:
            warnings_list.append("WARNING: SHOULDER TOO LOW")
    
    if abs(goc_khuyu - chuan_khuyu) > sai_so:
        if goc_khuyu > chuan_khuyu:
            warnings_list.append("WARNING: ELBOW TOO STRAIGHT")
        else:
            warnings_list.append("WARNING: ELBOW TOO BENT")
    
    return warnings_list



# ============================================
# XỬ LÝ FRAME - CẢI THIỆN BOX THÔNG TIN
# ============================================
def xu_ly_frame(frame, model, chuan, frame_idx, fps=30):
    h, w = frame.shape[:2]
    
    if w > RESIZE_WIDTH:
        scale = RESIZE_WIDTH / w
        new_w = RESIZE_WIDTH
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ket_qua = model.process(rgb)
    
    frame_output = frame.copy()
    
    GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)
    YELLOW, CYAN, ORANGE = (0, 255, 255), (255, 255, 0), (0, 165, 255)
    
    thoi_gian_giay = frame_idx / fps
    phut = int(thoi_gian_giay // 60)
    giay = int(thoi_gian_giay % 60)
    timestamp_str = f"{phut:02d}:{giay:02d}"
    
    if not ket_qua.pose_landmarks:
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
        return frame_output, None, None, None, None, []
    
    # Vẽ landmarks
    mp_drawing.draw_landmarks(
        frame_output, ket_qua.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    )
    
    lm = ket_qua.pose_landmarks.landmark
    
    # Lấy tọa độ cả hai bên
    # Bên trái
    vai_t = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    khuyu_t = (int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w), int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
    co_tay_t = (int(lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w), int(lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
    hong_t = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h))
    
    # Bên phải
    vai_p = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    khuyu_p = (int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
    co_tay_p = (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))
    hong_p = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
    
    # Tính toán góc cả hai bên
    goc_vai_t = tinh_goc(hong_t, vai_t, khuyu_t)
    goc_khuyu_t = tinh_goc(vai_t, khuyu_t, co_tay_t)
    
    goc_vai_p = tinh_goc(hong_p, vai_p, khuyu_p)
    goc_khuyu_p = tinh_goc(vai_p, khuyu_p, co_tay_p)
    
    # Tự động chọn bên đang tập (bên có góc vai lớn hơn hoặc đang vận động)
    # Với Codman, tay tập thường đưa ra xa thân mình hơn tay vịn
    nhip_t = abs(goc_vai_t - 45) # Giả định chuẩn Codman là 45
    nhip_p = abs(goc_vai_p - 45)
    
    # Chọn bên có góc gần với mục tiêu tập luyện hơn hoặc có sự thay đổi
    if abs(goc_vai_t - 10) > abs(goc_vai_p - 10): # So sánh với tư thế đứng thẳng (10 độ)
        goc_vai, goc_khuyu = goc_vai_t, goc_khuyu_t
        khop_chinh = vai_t
        khop_phu = khuyu_t
    else:
        goc_vai, goc_khuyu = goc_vai_p, goc_khuyu_p
        khop_chinh = vai_p
        khop_phu = khuyu_p

    chuan_vai = chuan["vai"]
    chuan_khuyu = chuan["khuyu"]
    ss = chuan["sai_so"]
    
    vai_dung = abs(goc_vai - chuan_vai) <= ss
    khuyu_dung = abs(goc_khuyu - chuan_khuyu) <= ss
    tong_the = vai_dung and khuyu_dung
    
    mau_vai = GREEN if vai_dung else RED
    mau_khuyu = GREEN if khuyu_dung else RED
    mau_tong = GREEN if tong_the else RED
    
    warnings_list = get_warning_message(goc_vai, goc_khuyu, chuan_vai, chuan_khuyu, ss)
    
    # Vẽ góc tại khớp
    cv2.putText(frame_output, f"{goc_vai:.0f}", (khop_chinh[0]-50, khop_chinh[1]-25), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, mau_vai, 2)
    cv2.putText(frame_output, f"{goc_khuyu:.0f}", (khop_phu[0]+25, khop_phu[1]-25), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, mau_khuyu, 2)
    
    # BOX THÔNG TIN - CẢI THIỆN RÕ NÉT
    overlay = frame_output.copy()
    cv2.rectangle(overlay, (10, 10), (420, 180), (0, 0, 0), -1)
    frame_output = cv2.addWeighted(overlay, 0.65, frame_output, 0.35, 0)
    cv2.rectangle(frame_output, (10, 10), (420, 180), (255, 255, 255), 2)
    
    # Frame info
    cv2.putText(frame_output, f"FRAME #{frame_idx}", (20, 40), 
               cv2.FONT_HERSHEY_DUPLEX, 0.7, CYAN, 2)
    cv2.putText(frame_output, f"TIME: {timestamp_str}", (20, 65), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)
    
    # Status
    status_text = "PASS" if tong_the else "FAIL"
    status_color = mau_tong
    cv2.putText(frame_output, status_text, (150, 40), 
               cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2)
    
    # Shoulder info
    cv2.putText(frame_output, "SHOULDER", (20, 95), 
               cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame_output, f"{goc_vai:.0f}", (20, 120), 
               cv2.FONT_HERSHEY_DUPLEX, 0.7, mau_vai, 2)
    cv2.putText(frame_output, f"/ {chuan_vai}", (80, 120), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (150, 150, 150), 1)
    
    # Elbow info
    cv2.putText(frame_output, "ELBOW", (200, 95), 
               cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame_output, f"{goc_khuyu:.0f}", (200, 120), 
               cv2.FONT_HERSHEY_DUPLEX, 0.7, mau_khuyu, 2)
    cv2.putText(frame_output, f"/ {chuan_khuyu}", (260, 120), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (150, 150, 150), 1)
    
    # Warning
    if warnings_list:
        cv2.putText(frame_output, warnings_list[0][:40], (20, 160), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, YELLOW, 1)
    
    return frame_output, goc_vai, goc_khuyu, tong_the, {
        'shoulder_correct': vai_dung,
        'elbow_correct': khuyu_dung,
        'shoulder_ref': chuan_vai,
        'elbow_ref': chuan_khuyu,
        'warnings': warnings_list
    }, warnings_list

# ============================================
# XỬ LÝ VIDEO
# ============================================
def xu_ly_video_day_du(duong_dan_video, chuan, callback=None):
    cap = cv2.VideoCapture(duong_dan_video)
    if not cap.isOpened():
        raise Exception(f"Không thể mở file video: {duong_dan_video}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    tong_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if MAX_FRAMES and tong_frame > MAX_FRAMES:
        tong_frame = MAX_FRAMES
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait = width < height
    output_width, output_height = (width, height) if is_portrait else (height, width)
    rotate_needed = not is_portrait
    
    timestamp = int(time.time())
    out_path = os.path.join(tempfile.gettempdir(), f'processed_video_{timestamp}.webm')
    thu_muc_frame = tempfile.mkdtemp()
    
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (output_width, output_height))
    if not writer.isOpened():
        out_path = os.path.join(tempfile.gettempdir(), f'processed_video_{timestamp}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (output_width, output_height))
    
    model = get_pose_model()
    du_lieu_goc = []
    danh_sach_frame_paths = []
    danh_sach_frame_data = []
    all_warnings = []
    
    frame_count = 0
    processed_count = 0
    last_progress = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (MAX_FRAMES and processed_count >= MAX_FRAMES):
            break
            
        frame_count += 1
        # XỬ LÝ 100% FRAME (KHÔNG SKIP) THEO YÊU CẦU CỦA USER
        if SKIP_FRAMES > 1 and frame_count % SKIP_FRAMES != 0:
            continue
            
        processed_count += 1
        if processed_count % 30 == 0:
            gc.collect() # Giải phóng RAM thường xuyên hơn
        
        if rotate_needed:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        h, w = frame.shape[:2]
        if w > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        
        xu_ly, goc_v, goc_k, dung, eval_info, warnings_list = xu_ly_frame(
            frame, model, chuan, frame_count, fps
        )
        
        if xu_ly.shape[1] != output_width or xu_ly.shape[0] != output_height:
            xu_ly = cv2.resize(xu_ly, (output_width, output_height))
        
        writer.write(xu_ly)
        
        frame_path = os.path.join(thu_muc_frame, f"frame_{processed_count:06d}.jpg")
        cv2.imwrite(frame_path, xu_ly, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_QUALITY])
        danh_sach_frame_paths.append(frame_path)
        
        ts_frame = frame_count / fps
        time_str = f"{int(ts_frame // 60):02d}:{int(ts_frame % 60):02d}"
        
        if warnings_list: all_warnings.extend(warnings_list)
        
        # Chuyển đổi sang kiểu dữ liệu Python thuần túy để JSON có thể serialize được
        danh_sach_frame_data.append({
            'index': int(frame_count), 
            'timestamp': str(time_str), 
            'path': str(frame_path),
            'goc_vai': float(goc_v) if goc_v is not None else None, 
            'goc_khuyu': float(goc_k) if goc_k is not None else None, 
            'dung': bool(dung) if dung is not None else False, 
            'eval_info': {
                'shoulder_correct': bool(eval_info['shoulder_correct']) if 'shoulder_correct' in eval_info else False,
                'elbow_correct': bool(eval_info['elbow_correct']) if 'elbow_correct' in eval_info else False,
                'shoulder_ref': float(eval_info['shoulder_ref']) if 'shoulder_ref' in eval_info else 0,
                'elbow_ref': float(eval_info['elbow_ref']) if 'elbow_ref' in eval_info else 0,
                'warnings': [str(w) for w in eval_info.get('warnings', [])]
            } if eval_info else {}
        })
        
        if goc_v is not None:
            du_lieu_goc.append({
                'frame': frame_count, 'timestamp': time_str, 'timestamp_seconds': ts_frame,
                'goc_vai': float(goc_v), 'goc_khuyu': float(goc_k), 'dung': bool(dung),
                'vai_dung': eval_info['shoulder_correct'], 'khuyu_dung': eval_info['elbow_correct'],
                'vai_chuan': eval_info['shoulder_ref'], 'khuyu_chuan': eval_info['elbow_ref']
            })
        
        if callback and tong_frame > 0:
            progress = min(frame_count / tong_frame, 1.0)
            if progress - last_progress >= 0.05:
                callback(progress)
                last_progress = progress
        
        # GIẢI PHÓNG BỘ NHỚ TRIỆT ĐỂ
        del frame
        del xu_ly
        if processed_count % 100 == 0:
            gc.collect()
    
    cap.release()
    writer.release()
    
    # TẠO ZIP TRÊN ĐĨA ĐỂ TIẾT KIỆM RAM
    zip_path = os.path.join(tempfile.gettempdir(), f'frames_{timestamp}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for p in danh_sach_frame_paths:
            if os.path.exists(p):
                zipf.write(p, os.path.basename(p))

    # LƯU DỮ LIỆU KHUNG HÌNH RA FILE JSON ĐỂ TIẾT KIỆM RAM
    json_path = os.path.join(tempfile.gettempdir(), f'frames_data_{timestamp}.json')
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(danh_sach_frame_data, f, ensure_ascii=False)
    
    gc.collect()
    return out_path, None, None, du_lieu_goc, frame_count, len(du_lieu_goc), thu_muc_frame, zip_path, danh_sach_frame_paths, {}, json_path, all_warnings


# ============================================
# TÍNH TOÁN METRICS CHI TIẾT
# ============================================
def tinh_metrics_chi_tiet(df):
    if len(df) == 0:
        return {}
    
    ty_le_vai_dung = df['vai_dung'].sum() / len(df) * 100
    ty_le_khuyu_dung = df['khuyu_dung'].sum() / len(df) * 100
    ty_le_tong_the = df['dung'].sum() / len(df) * 100
    
    return {
        "ty_le_tong_the": ty_le_tong_the,
        "ty_le_vai_dung": ty_le_vai_dung,
        "ty_le_khuyu_dung": ty_le_khuyu_dung,
        "tb_goc_vai": df['goc_vai'].mean(),
        "tb_goc_khuyu": df['goc_khuyu'].mean(),
        "frame_dung": int(df['dung'].sum()),
        "min_goc_vai": df['goc_vai'].min(),
        "max_goc_vai": df['goc_vai'].max(),
        "min_goc_khuyu": df['goc_khuyu'].min(),
        "max_goc_khuyu": df['goc_khuyu'].max(),
        "std_goc_vai": df['goc_vai'].std(),
        "std_goc_khuyu": df['goc_khuyu'].std()
    }

# ============================================
# VẼ BIỂU ĐỒ SÁNG TẠO
# ============================================
def ve_bieu_do_goc_vai(df, bt):
    """Vẽ biểu đồ góc vai với thiết kế đẹp mắt"""
    chuan_vai = bt['chuan']['vai']
    sai_so = bt['chuan']['sai_so']
    
    fig = go.Figure()
    
    # Thêm vùng chuẩn
    fig.add_hrect(y0=chuan_vai-sai_so, y1=chuan_vai+sai_so,
                  fillcolor="rgba(0, 255, 0, 0.15)", line_width=0,
                  annotation_text="Vùng chuẩn", annotation_position="top left")
    
    # Thêm đường góc bệnh nhân
    fig.add_trace(go.Scatter(
        y=df['goc_vai'],
        mode='lines+markers',
        line=dict(color='#00CED1', width=3),
        marker=dict(size=4, color='#00CED1', symbol='circle'),
        name='Góc vai bệnh nhân',
        hovertemplate='Frame: %{x}<br>Góc vai: %{y:.1f}°<extra></extra>'
    ))
    
    # Thêm đường chuẩn
    fig.add_hline(y=chuan_vai, line_dash='dash', line_color='#00FF00',
                 line_width=2, annotation_text=f"Chuẩn: {chuan_vai}°",
                 annotation_position="top right")
    
    # Tô màu vùng ngoài chuẩn
    fig.add_hrect(y0=0, y1=chuan_vai-sai_so, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
    fig.add_hrect(y0=chuan_vai+sai_so, y1=180, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
    
    fig.update_layout(
        title=dict(
            text="<b>📈 BIỂU ĐỒ GÓC VAI THEO THỜI GIAN</b>",
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5
        ),
        xaxis=dict(title=dict(text="<b>Số Frame</b>", font=dict(size=14, color='white')), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text="<b>Góc (độ)</b>", font=dict(size=14, color='white')), gridcolor='rgba(255,255,255,0.1)',
                   range=[0, 180]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(26,26,46,0.9)',
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white', size=12)
        ),
        margin=dict(l=50, r=50, t=70, b=50)
    )
    
    return fig


def ve_bieu_do_goc_khuyu(df, bt):
    """Vẽ biểu đồ góc khuỷu với thiết kế đẹp mắt"""
    chuan_khuyu = bt['chuan']['khuyu']
    sai_so = bt['chuan']['sai_so']
    
    fig = go.Figure()
    
    fig.add_hrect(y0=chuan_khuyu-sai_so, y1=chuan_khuyu+sai_so,
                  fillcolor="rgba(0, 255, 0, 0.15)", line_width=0,
                  annotation_text="Vùng chuẩn", annotation_position="top left")
    
    fig.add_trace(go.Scatter(
        y=df['goc_khuyu'],
        mode='lines+markers',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=4, color='#FF6B6B', symbol='circle'),
        name='Góc khuỷu bệnh nhân',
        hovertemplate='Frame: %{x}<br>Góc khuỷu: %{y:.1f}°<extra></extra>'
    ))
    
    fig.add_hline(y=chuan_khuyu, line_dash='dash', line_color='#00FF00',
                 line_width=2, annotation_text=f"Chuẩn: {chuan_khuyu}°",
                 annotation_position="top right")
    
    fig.add_hrect(y0=0, y1=chuan_khuyu-sai_so, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
    fig.add_hrect(y0=chuan_khuyu+sai_so, y1=180, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
    
    fig.update_layout(
        title=dict(
            text="<b>📈 BIỂU ĐỒ GÓC KHUỶU THEO THỜI GIAN</b>",
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5
        ),
        xaxis=dict(title=dict(text="<b>Số Frame</b>", font=dict(size=14, color='white')), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text="<b>Góc (độ)</b>", font=dict(size=14, color='white')), gridcolor='rgba(255,255,255,0.1)',
                   range=[0, 180]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(26,26,46,0.9)',
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white', size=12)
        ),
        margin=dict(l=50, r=50, t=70, b=50)
    )
    
    return fig


def ve_bieu_do_histogram(df, bt):
    """Vẽ biểu đồ histogram phân phối góc"""
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("<b>Phân phối góc vai</b>", "<b>Phân phối góc khuỷu</b>"),
                        shared_yaxes=True)
    
    fig.add_trace(go.Histogram(
        x=df['goc_vai'],
        nbinsx=25,
        marker_color='#00CED1',
        opacity=0.8,
        name='Góc vai',
        hovertemplate='Góc: %{x:.1f}°<br>Tần suất: %{y}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=df['goc_khuyu'],
        nbinsx=25,
        marker_color='#FF6B6B',
        opacity=0.8,
        name='Góc khuỷu',
        hovertemplate='Góc: %{x:.1f}°<br>Tần suất: %{y}<extra></extra>'
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text="<b>📊 PHÂN PHỐI GÓC KHỚP (HISTOGRAM)</b>",
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(26,26,46,0.9)',
        showlegend=False,
        height=500,
        bargap=0.05
    )
    
    fig.update_xaxes(title=dict(text="<b>Góc (độ)</b>", font=dict(size=12, color='white')), 
                     gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
    fig.update_xaxes(title=dict(text="<b>Góc (độ)</b>", font=dict(size=12, color='white')),
                     gridcolor='rgba(255,255,255,0.1)', row=1, col=2)
    fig.update_yaxes(title=dict(text="<b>Tần suất</b>", font=dict(size=12, color='white')),
                     gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
    
    return fig


def ve_bieu_do_boxplot(df):
    """Vẽ biểu đồ boxplot so sánh"""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=df['goc_vai'],
        name='Góc vai',
        marker_color='#00CED1',
        boxmean='sd',
        hovertemplate='Góc vai: %{y:.1f}°<extra></extra>'
    ))
    
    fig.add_trace(go.Box(
        y=df['goc_khuyu'],
        name='Góc khuỷu',
        marker_color='#FF6B6B',
        boxmean='sd',
        hovertemplate='Góc khuỷu: %{y:.1f}°<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>📦 SO SÁNH PHÂN PHỐI GÓC (BOX PLOT)</b>",
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5
        ),
        yaxis=dict(title=dict(text="<b>Góc (độ)</b>", font=dict(size=14, color='white')),
                   gridcolor='rgba(255,255,255,0.1)', range=[0, 180]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(26,26,46,0.9)',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white', size=12)
        )
    )
    
    return fig

# ============================================
# LƯU BIỂU ĐỒ THÀNH ẢNH
# ============================================
def save_figure_as_image(fig, filename):
    """Lưu figure thành ảnh"""
    fig.write_image(filename, width=1200, height=600, scale=2)
    return filename


# ============================================
# DỮ LIỆU BÀI TẬP
# ============================================
BAI_TAP = {
    "codman": {
        "ten": "Bài tập con lắc Codman",
        "icon": "🔄",
        "mo_ta": "Bài tập dao động tay thụ động theo quán tính, giúp thả lỏng khớp vai, giảm đau và chống dính khớp.",
        "chuan": {"vai": 45, "khuyu": 160, "sai_so": 20},
        "youtube": "https://youtu.be/a4eCRWuqO40",
        "thoi_gian": 30, 
        "lan": 10,
        # THÊM DÒNG NÀY - BẢN NGẮN CHO HIỂN THỊ CHÍNH
        "huong_dan": "1. Đứng thẳng, cúi người về phía trước 30-45 độ\n2. Tay bệnh thả lỏng hoàn toàn\n3. Đung đưa tay nhẹ nhàng theo chiều trước-sau, sang ngang và vòng tròn\n4. Mỗi động tác 10 lần, nghỉ 5-10 giây\n5. Thở ra khi đưa tay lên, hít vào khi hạ tay xuống",
        "loi_ich": [
            "✓ Giảm đau khớp vai sau chấn thương hoặc viêm quanh khớp",
            "✓ Tăng tuần hoàn máu tại chỗ, thúc đẩy quá trình lành thương",
            "✓ Chống dính khớp và duy trì tầm vận động khớp vai",
            "✓ Giảm co cứng cơ quanh khớp",
            "✓ Chuẩn bị cho các bài tập chủ động ở giai đoạn sau"
        ],
        "chi_dinh": [
            "✓ Viêm quanh khớp vai thể đông cứng",
            "✓ Sau phẫu thuật khớp vai (giai đoạn sớm)",
            "✓ Hội chứng chóp xoay",
            "✓ Đau vai do thoái hóa khớp",
            "✓ Bệnh nhân sau đột quỵ có liệt mềm chi trên"
        ],
        "chu_y": [
            "⚠️ Không thực hiện khi đau vai cấp tính chưa rõ nguyên nhân",
            "⚠️ Tránh thực hiện động tác quá mạnh hoặc quá nhanh",
            "⚠️ Bệnh nhân loãng xương nặng cần thận trọng",
            "⚠️ Không tập khi có trật khớp vai chưa nắn chỉnh",
            "⚠️ Dừng tập nếu đau tăng hoặc xuất hiện đau mới"
        ],
        "huong_dan_chi_tiet": """
        📌 **HƯỚNG DẪN CHUYÊN SÂU THEO PHCN:**
        
        **1. TƯ THẾ CHUẨN BỊ:**
        - Đứng thẳng, hai chân rộng bằng vai
        - Gập thân về phía trước khoảng 30-45 độ
        - Tay bệnh thả lỏng hoàn toàn, duỗi tự nhiên
        - Tay lành có thể chống lên bàn hoặc ghế để giữ thăng bằng
        
        **2. KỸ THUẬT THỰC HIỆN:**
        - Đung đưa tay bệnh theo chiều trước - sau (động tác 1)
        - Đung đưa tay theo chiều sang ngang (động tác 2)
        - Đung đưa tay theo vòng tròn (ngược chiều kim đồng hồ rồi xuôi chiều)
        - Mỗi động tác thực hiện 10 lần, nghỉ 5-10 giây giữa các động tác
        
        **3. HƯỚNG DẪN THỞ:**
        - Thở ra khi đưa tay lên/xa khỏi người
        - Hít vào khi đưa tay về vị trí ban đầu
        
        **4. THỜI GIAN VÀ LIỀU LƯỢNG:**
        - Thời gian: 30 giây/động tác
        - Số lần: 5-10 lần/ngày, 5-7 ngày/tuần
        - Tổng thời gian điều trị: 4-6 tuần
        
        **5. THEO DÕI TIẾN TRIỂN:**
        - Ghi nhận mức độ đau theo thang VAS trước và sau tập
        - Đo tầm vận động khớp vai (gập, duỗi, dạng, xoay) mỗi tuần
        - Đánh giá khả năng sinh hoạt hàng ngày (ADL)
        """,
        "tieu_chi_danh_gia": """
        📊 **TIÊU CHÍ ĐÁNH GIÁ KẾT QUẢ:**
        - Góc vai đạt 45° ± 15° (30° - 60°)
        - Góc khuỷu duy trì 160° ± 15° (145° - 175°)
        - Bệnh nhân không có biểu hiện đau khi thực hiện (VAS < 3)
        - Thực hiện động tác mượt mà, không giật cục
        - Duy trì được nhịp thở đều đặn trong khi tập
        """
    },
    "gay": {
        "ten": "Bài tập với gậy (Pulley Exercise)",
        "icon": "🏒",
        "mo_ta": "Sử dụng gậy hoặc ròng rọc hỗ trợ nâng tay và xoay vai bị hạn chế vận động.",
        "chuan": {"vai": 90, "khuyu": 170, "sai_so": 20},
        "youtube": "https://www.youtube.com/watch?v=s2O8WHT5o2k",
        "thoi_gian": 45, 
        "lan": 12,
        # THÊM DÒNG NÀY - BẢN NGẮN CHO HIỂN THỊ CHÍNH
        "huong_dan": "1. Cầm gậy bằng hai tay, tay lành cầm một đầu, tay bệnh cầm đầu kia\n2. Tay lành dùng lực đẩy gậy lên cao, kéo tay bệnh theo\n3. Giữ 5-10 giây ở tư thế cao nhất, hạ từ từ\n4. Thực hiện 10 lần mỗi động tác: nâng trước, xoay ngoài, xoay trong\n5. Thở ra khi nâng gậy lên, hít vào khi hạ xuống",
        "loi_ich": [
            "✓ Cải thiện tầm vận động khớp vai chủ động và thụ động",
            "✓ Tăng cường sức mạnh cơ vai một cách an toàn",
            "✓ Giảm chênh lệch vận động giữa hai tay",
            "✓ Phục hồi khả năng với tay và nâng vật trên cao",
            "✓ Duy trì tính linh hoạt của bao khớp và dây chằng"
        ],
        "chi_dinh": [
            "✓ Hội chứng chóp xoay giai đoạn phục hồi",
            "✓ Viêm quanh khớp vai thể đông cứng giai đoạn tan dính",
            "✓ Sau phẫu thuật tái tạo chóp xoay (giai đoạn muộn)",
            "✓ Bệnh nhân sau đột quỵ giai đoạn phục hồi chức năng",
            "✓ Teo cơ delta, cơ trên gai do bất động kéo dài"
        ],
        "chu_y": [
            "⚠️ Không kéo gậy quá tầm chịu đựng của bệnh nhân",
            "⚠️ Tránh thực hiện khi đau vai cấp (VAS > 5)",
            "⚠️ Bệnh nhân trật khớp vai chưa ổn định tuyệt đối chống chỉ định",
            "⚠️ Theo dõi dấu hiệu chèn ép rễ thần kinh (tê bì tay)",
            "⚠️ Điều chỉnh chiều cao gậy phù hợp với từng bệnh nhân"
        ],
        "huong_dan_chi_tiet": """
        📌 **HƯỚNG DẪN CHUYÊN SÂU THEO PHCN:**
        
        **1. DỤNG CỤ HỖ TRỢ:**
        - Gậy dài khoảng 80-100cm (có thể dùng chổi, cán lau nhà)
        - Hoặc hệ thống ròng rọc treo tường (nếu có)
        - Khăn mềm để cuốn quanh tay nếu đau khi cầm nắm
        
        **2. ĐỘNG TÁC 1 - NÂNG TAY RA TRƯỚC (Flexion):**
        - Nằm hoặc ngồi, hai tay cầm hai đầu gậy
        - Tay lành dùng lực đẩy gậy lên cao, kéo tay bệnh theo
        - Nâng đến mức tối đa chịu đựng hoặc đến khi tay thẳng đứng
        - Giữ 5-10 giây ở tư thế cao nhất
        - Hạ từ từ về vị trí ban đầu
        - Lặp lại 10 lần
        
        **3. ĐỘNG TÁC 2 - XOAY VAI NGOÀI (External Rotation):**
        - Nằm ngửa, tay bệnh gập khuỷu 90°, cẳng tay hướng lên trần
        - Tay lành cầm gậy đẩy tay bệnh ra ngoài
        - Giữ 5 giây ở tư thế xoay tối đa
        - Thực hiện 10 lần
        
        **4. ĐỘNG TÁC 3 - XOAY VAI TRONG (Internal Rotation):**
        - Tay bệnh đưa ra sau lưng
        - Tay lành cầm gậy từ phía trên kéo tay bệnh lên
        - Giữ 5 giây, thực hiện 10 lần
        
        **5. HƯỚNG DẪN THỞ:**
        - Thở ra khi nâng/kéo gậy lên
        - Hít vào khi hạ gậy về
        - Không nín thở trong khi thực hiện
        
        **6. TIẾN TRIỂN BÀI TẬP (Theo tuần):**
        - Tuần 1-2: Thực hiện với biên độ 50% tầm vận động tối đa
        - Tuần 3-4: Tăng lên 75% tầm vận động
        - Tuần 5-6: Thực hiện toàn bộ tầm vận động (90°)
        - Tuần 7-8: Thêm tạ nhẹ (0.5-1kg) nếu không đau
        """,
        "tieu_chi_danh_gia": """
        📊 **TIÊU CHÍ ĐÁNH GIÁ KẾT QUẢ:**
        - Góc vai đạt 90° ± 15° (75° - 105°)
        - Góc khuỷu duy trì gần duỗi thẳng (170° ± 15°)
        - Không có hiện tượng bù trừ (nghiêng người, nhún vai)
        - Bệnh nhân có thể tự thực hiện với mức độ trợ giúp tối thiểu
        - Cải thiện khả năng với tay lên cao (lấy đồ trên kệ, móc áo)
        """
    },
    "khang_luc": {
        "ten": "Bài tập với dây kháng lực (Theraband Exercise)",
        "icon": "💪",
        "mo_ta": "Tăng cường sức mạnh cơ chóp xoay và cơ quanh khớp vai bằng dây thun kháng lực.",
        "chuan": {"vai": 60, "khuyu": 90, "sai_so": 20},
        "youtube": "https://www.youtube.com/watch?v=njDHDnZ6lis",
        "thoi_gian": 40, 
        "lan": 15,
        # THÊM DÒNG NÀY - BẢN NGẮN CHO HIỂN THỊ CHÍNH
        "huong_dan": "1. Bắt đầu với dây kháng lực thấp nhất (màu vàng hoặc đỏ)\n2. Xoay vai ngoài: Nằm nghiêng, khuỷu gập 90°, xoay cẳng tay ra ngoài\n3. Xoay vai trong: Đứng hoặc nằm nghiêng, kéo dây vào trong áp sát bụng\n4. Dang vai: Đứng, dẫm dây dưới chân, dang tay sang ngang 60°\n5. Gập vai: Đứng, nâng tay ra trước 60°\n6. Mỗi động tác 10-15 lần x 3 hiệp, nghỉ 30 giây giữa hiệp",
        "loi_ich": [
            "✓ Tăng sức mạnh và sức bền của cơ chóp xoay",
            "✓ Ổn định khớp vai trong các hoạt động hàng ngày",
            "✓ Phòng ngừa tái phát chấn thương vai",
            "✓ Cải thiện khả năng kiểm soát vận động",
            "✓ Duy trì thành quả sau điều trị dài hạn"
        ],
        "chi_dinh": [
            "✓ Hội chứng chóp xoay giai đoạn phục hồi chức năng",
            "✓ Sau phẫu thuật tái tạo dây chằng vai (giai đoạn muộn)",
            "✓ Bệnh nhân yếu cơ vai do bất động kéo dài",
            "✓ Phòng ngừa chấn thương vai ở vận động viên",
            "✓ Điều trị hội chứng va đập dưới mỏm cùng vai"
        ],
        "chu_y": [
            "⚠️ Bắt đầu với dây kháng lực thấp nhất (màu vàng hoặc đỏ)",
            "⚠️ Không tập khi đau cấp hoặc viêm khớp vai đang tiến triển",
            "⚠️ Theo dõi dấu hiệu quá tải: đau kéo dài > 24 giờ sau tập",
            "⚠️ Bệnh nhân cao huyết áp cần thận trọng với tư thế gập người",
            "⚠️ Không thực hiện động tác quá nhanh, kiểm soát ở cả 2 thì"
        ],
        "huong_dan_chi_tiet": """
        📌 **HƯỚNG DẪN CHUYÊN SÂU THEO PHCN:**
        
        **1. LỰA CHỌN DÂY KHÁNG LỰC (Theraband):**
        - Cấp độ 1 (Yếu nhất): Màu vàng hoặc đỏ - cho bệnh nhân yếu
        - Cấp độ 2 (Trung bình): Màu xanh lá - cho bệnh nhân trung bình
        - Cấp độ 3 (Mạnh): Màu xanh dương, đen - cho VĐV, người khỏe
        
        **2. ĐỘNG TÁC 1 - XOAY VAI NGOÀI:**
        - Tư thế: Nằm nghiêng về phía tay lành
        - Tay bệnh: Gập khuỷu 90°, đặt sát thân mình
        - Cố định dây: Buộc dây vào vật chắc chắn ngang thắt lưng
        - Động tác: Xoay cẳng tay ra ngoài, giữ khuỷu sát người
        - Giữ 2-3 giây ở cuối tầm, trở về chậm
        - Thực hiện 10-15 lần, 3 hiệp
        
        **3. ĐỘNG TÁC 2 - XOAY VAI TRONG:**
        - Tư thế: Đứng hoặc nằm nghiêng về phía tay bệnh
        - Tay bệnh: Gập khuỷu 90°
        - Cố định dây: Buộc dây ở phía cùng bên
        - Động tác: Kéo dây vào trong, áp sát tay vào bụng
        - Thực hiện 10-15 lần, 3 hiệp
        
        **4. ĐỘNG TÁC 3 - DANG VAI (Abduction):**
        - Tư thế: Đứng, tay bệnh duỗi thẳng, dây dẫm dưới chân
        - Động tác: Dang tay sang ngang đến 60°
        - Giữ 2 giây, hạ về chậm
        - Thực hiện 10-12 lần, 3 hiệp
        
        **5. ĐỘNG TÁC 4 - GẬP VAI (Flexion):**
        - Tư thế: Đứng, dây dẫm dưới chân
        - Động tác: Nâng tay ra trước đến 60°
        - Kiểm soát ở cả chiều lên và xuống
        - Thực hiện 10-12 lần, 3 hiệp
        
        **6. NGUYÊN TẮC TĂNG TIẾN:**
        - Tuần 1-2: 10 lần x 2 hiệp, nghỉ 30s, dây cấp độ 1
        - Tuần 3-4: 12 lần x 3 hiệp, tăng số lần trước
        - Tuần 5-6: 15 lần x 3 hiệp, tăng cấp độ dây nếu dễ dàng
        - Tuần 7-8: Thêm bài tập phức hợp và tăng kháng lực
        
        **7. THEO DÕI CƯỜNG ĐỘ TẬP (RPE - Rating of Perceived Exertion):**
        - RPE 3-4: Hơi mệt, có thể trò chuyện trong khi tập
        - RPE 5-6: Mệt vừa, khó nói chuyện
        - RPE 7-8: Mệt nhiều - GIẢM CƯỜNG ĐỘ NGAY
        """,
        "tieu_chi_danh_gia": """
        📊 **TIÊU CHÍ ĐÁNH GIÁ KẾT QUẢ:**
        - Góc vai đạt 60° ± 15° (45° - 75°)
        - Góc khuỷu gập 90° ± 15° (75° - 105°)
        - Thực hiện đúng kỹ thuật, không bù trừ bằng cơ vai khác
        - Bệnh nhân có thể thực hiện 3 hiệp 15 lần với dây cấp độ phù hợp
        - Không đau trong và sau khi tập (VAS < 2)
        - Cải thiện sức mạnh (test cơ manual muscle testing tăng 1-2 cấp độ)
        """,
        "tien_trinh_dieu_tri": """
        📅 **TIẾN TRÌNH ĐIỀU TRỊ THEO GIAI ĐOẠN:**
        
        **Giai đoạn 1 (Tuần 1-2): Làm quen**
        - Dây kháng lực thấp nhất (màu vàng)
        - 10 lần x 2 hiệp, mỗi động tác
        - Tần suất: 3-4 buổi/tuần
        
        **Giai đoạn 2 (Tuần 3-4): Tăng số lần**
        - Dây cấp độ 1 (màu đỏ hoặc xanh lá nhạt)
        - 12-15 lần x 3 hiệp
        - Tần suất: 4-5 buổi/tuần
        
        **Giai đoạn 3 (Tuần 5-6): Tăng kháng lực**
        - Dây cấp độ 2 (màu xanh lá đậm)
        - 12 lần x 3 hiệp
        - Thêm bài tập phức hợp (kết hợp nhiều động tác)
        
        **Giai đoạn 4 (Tuần 7-8): Duy trì**
        - Dây cấp độ phù hợp với sức bệnh nhân
        - 15 lần x 3 hiệp
        - Kết hợp với các bài tập chức năng (ném bóng, kéo dây)
        """
    }
}


# ============================================
# CSS KẾT HỢP ĐẦY ĐỦ
# ============================================
st.markdown("""
<style>
    * { font-family: 'Times New Roman', Times, serif !important; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #0f0f1a 50%, #1a1a2e 100%); }
    
    /* HEADER */
    .main-header {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #2a5298;
    }
    .main-header h1 { color: #ffffff !important; font-size: 1.8rem; margin: 0; }
    .main-header p { color: #aaa !important; margin: 0.5rem 0 0 0; }
    
    /* RESEARCH BADGE */
    .research-badge {
        background: linear-gradient(135deg, #2a5298, #1a73e8);
        padding: 0.3rem 1rem;
        border-radius: 50px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .research-badge span { color: white; font-size: 0.8rem; font-weight: bold; }
    
    /* INFO BOX */
    .info-box {
        background: rgba(26,26,46,0.8);
        padding: 1.2rem;
        border-radius: 16px;
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    
    /* BUTTON - CÓ HOVER EFFECT */
    .stButton > button {
        background: linear-gradient(135deg, #2a5298 0%, #1a73e8 100%) !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* MEMBER CARD */
    .member-card {
        background: linear-gradient(135deg, rgba(26,26,46,0.9) 0%, rgba(22,33,62,0.9) 100%);
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .member-name { color: #fff; font-size: 1.1rem; font-weight: bold; }
    .member-role { color: #ffd700; font-size: 0.85rem; }
    
    /* LECTURER CARD */
    .lecturer-card {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 2px solid #ffd700;
    }
    .lecturer-name { color: #ffd700; font-size: 1.3rem; font-weight: bold; }
    
    /* FRAME THUMBNAIL */
    .frame-thumbnail {
        transition: transform 0.3s;
        cursor: pointer;
        width: 100%;
        border-radius: 12px;
    }
    .frame-thumbnail:hover {
        transform: scale(1.02);
    }
    
    /* VIDEO */
    video {
        width: 100%;
        border-radius: 16px;
        background: black;
        max-height: 70vh;
    }
    
    /* WARNING BOX */
    .warning-box {
        background: rgba(255,100,0,0.2);
        border-left: 4px solid #FFA500;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* TABS - CÓ FONT SIZE LỚN HƠN */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
    }
    
    /* METRIC CARD - MỚI THÊM */
    .metric-card {
        background: linear-gradient(135deg, rgba(26,26,46,0.95) 0%, rgba(22,33,62,0.95) 100%);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a5298;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        border-color: #ffd700;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00CED1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.5rem;
    }
    
    /* CUSTOM SCROLLBAR (TÙY CHỌN THÊM) */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #2a5298;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #1a73e8;
    }
    
    /* DATA FRAME STYLE */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame div[data-testid="stDataFrame"] {
        border-radius: 12px;
    }
    
    /* EXPANDER STYLE */
    .streamlit-expanderHeader {
        background: rgba(26,26,46,0.8);
        border-radius: 12px;
        font-weight: bold;
    }
    
    /* SUCCESS/WARNING/INFO MESSAGES */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    .stAlert[data-baseweb="notification"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HÀM HIỂN THỊ TAB 2 - THIẾT KẾ LẠI
# ============================================
def hien_thi_tab_phan_tich():
    """Hiển thị tab phân tích với thiết kế sáng tạo"""
    
    if not st.session_state.has_data or not st.session_state.stats:
        st.info("ℹ️ Chưa có kết quả. Vui lòng upload video ở tab TRANG CHỦ.")
        return
    
    bt = st.session_state.exercise
    tk = st.session_state.stats
    df = st.session_state.angle_df
    
    # Header thông tin
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                border-radius: 20px; padding: 1.5rem; margin-bottom: 2rem; 
                border: 1px solid #2a5298;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="color: #ffd700; margin: 0;">📊 KẾT QUẢ PHÂN TÍCH</h2>
                <p style="color: #aaa; margin: 0.5rem 0 0 0;">
                    🏥 Bài tập: {bt['ten']} | ⏱️ Thời gian xử lý: {tk['thoi_gian']:.1f} giây
                </p>
            </div>
            <div style="text-align: right;">
                <p style="color: #00CED1; margin: 0;">✅ Độ chính xác: {tk['do_chinh_xac']:.1f}%</p>
                <p style="color: #FF6B6B; margin: 0;">📸 Tổng số frame: {tk['tong_frame']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Thống kê nhanh - Card design
    st.markdown("### 📈 THỐNG KÊ TỔNG QUAN")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{tk['do_chinh_xac']:.1f}%</div>
            <div class="metric-label">🎯 Độ chính xác tổng thể</div>
            <div style="font-size: 0.7rem; color: #aaa;">{tk['frame_dung']}/{tk['tong_frame_hop_le']} frame đúng</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{tk['ty_le_vai_dung']:.1f}%</div>
            <div class="metric-label">🦾 Tỉ lệ đúng góc vai</div>
            <div style="font-size: 0.7rem; color: #aaa;">Chuẩn: {bt['chuan']['vai']}° ±{bt['chuan']['sai_so']}°</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{tk['ty_le_khuyu_dung']:.1f}%</div>
            <div class="metric-label">💪 Tỉ lệ đúng góc khuỷu</div>
            <div style="font-size: 0.7rem; color: #aaa;">Chuẩn: {bt['chuan']['khuyu']}° ±{bt['chuan']['sai_so']}°</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{tk['tb_goc_vai']:.1f}°</div>
            <div class="metric-label">📐 Góc vai trung bình</div>
            <div style="font-size: 0.7rem; color: #aaa;">Min: {tk['min_goc_vai']:.0f}° | Max: {tk['max_goc_vai']:.0f}°</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TẠO CÁC TAB CON TRONG TAB PHÂN TÍCH
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📈 GÓC VAI", 
        "📊 GÓC KHUỶU", 
        "⚠️ CẢNH BÁO CHI TIẾT",
        "📁 XUẤT DỮ LIỆU"
    ])
    
    # === SUB TAB 1: BIỂU ĐỒ GÓC VAI ===
    with sub_tab1:
        st.markdown("### 📈 PHÂN TÍCH CHI TIẾT GÓC VAI")
        
        # Biểu đồ đường góc vai
        fig_vai = ve_bieu_do_goc_vai(df, bt)
        st.plotly_chart(fig_vai, width='stretch')
        
        # Thống kê góc vai
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Giá trị trung bình", f"{tk['tb_goc_vai']:.1f}°", 
                     delta=f"Chuẩn: {bt['chuan']['vai']}°")
        with col2:
            st.metric("📈 Giá trị lớn nhất", f"{tk['max_goc_vai']:.1f}°")
        with col3:
            st.metric("📉 Giá trị nhỏ nhất", f"{tk['min_goc_vai']:.1f}°")
        
        st.markdown("---")
        
        # Biểu đồ Histogram cho góc vai
        st.markdown("#### 📊 PHÂN PHỐI GÓC VAI (HISTOGRAM)")
        fig_hist_vai = go.Figure()
        fig_hist_vai.add_trace(go.Histogram(
            x=df['goc_vai'],
            nbinsx=25,
            marker_color='#00CED1',
            opacity=0.8,
            name='Góc vai',
            hovertemplate='Góc: %{x:.1f}°<br>Tần suất: %{y}<extra></extra>'
        ))
        fig_hist_vai.add_vline(x=bt['chuan']['vai'], line_dash='dash', line_color='#00FF00',
                               annotation_text=f"Chuẩn: {bt['chuan']['vai']}°", annotation_position="top right")
        fig_hist_vai.add_vrect(x0=bt['chuan']['vai']-bt['chuan']['sai_so'], 
                               x1=bt['chuan']['vai']+bt['chuan']['sai_so'],
                               fillcolor='rgba(0, 255, 0, 0.15)', line_width=0)
        fig_hist_vai.update_layout(
            title="Phân phối góc vai theo tần suất",
            xaxis_title="Góc (độ)",
            yaxis_title="Tần suất",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(26,26,46,0.9)',
            height=450,
            font=dict(color='white')
        )
        st.plotly_chart(fig_hist_vai, width='stretch')
        
        st.markdown("---")
        
        # Biểu đồ Boxplot so sánh góc vai và góc khuỷu
        st.markdown("#### 📦 SO SÁNH PHÂN PHỐI GÓC (BOX PLOT)")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df['goc_vai'],
            name='Góc vai',
            marker_color='#00CED1',
            boxmean='sd',
            hovertemplate='Góc vai: %{y:.1f}°<extra></extra>'
        ))
        fig_box.add_trace(go.Box(
            y=df['goc_khuyu'],
            name='Góc khuỷu',
            marker_color='#FF6B6B',
            boxmean='sd',
            hovertemplate='Góc khuỷu: %{y:.1f}°<extra></extra>'
        ))
        fig_box.update_layout(
            yaxis_title="Góc (độ)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(26,26,46,0.9)',
            height=500,
            font=dict(color='white'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='white',
                borderwidth=1
            )
        )
        st.plotly_chart(fig_box, width='stretch')
        
        # Nút tải biểu đồ
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Tải biểu đồ góc vai (PNG)", width='stretch', key="save_vai"):
                try:
                    img_path = "temp_vai_chart.png"
                    fig_vai.write_image(img_path, width=1200, height=600, scale=2)
                    with open(img_path, "rb") as f:
                        st.download_button("📥 Click để tải xuống", f, "bieu_do_goc_vai.png", "image/png")
                    os.remove(img_path)
                except:
                    st.warning("⚠️ Cần cài đặt kaleido để xuất ảnh: pip install -U kaleido")
        with col2:
            if st.button("💾 Tải Histogram + Boxplot (PNG)", width='stretch', key="save_vai_stats"):
                try:
                    # Tạo figure tổng hợp
                    fig_combined = make_subplots(rows=2, cols=1, 
                                                  subplot_titles=("Histogram góc vai", "Boxplot so sánh"),
                                                  vertical_spacing=0.15)
                    fig_combined.add_trace(go.Histogram(x=df['goc_vai'], nbinsx=25, marker_color='#00CED1', name='Histogram'), row=1, col=1)
                    fig_combined.add_trace(go.Box(y=df['goc_vai'], name='Góc vai', marker_color='#00CED1', boxmean='sd'), row=2, col=1)
                    fig_combined.add_trace(go.Box(y=df['goc_khuyu'], name='Góc khuỷu', marker_color='#FF6B6B', boxmean='sd'), row=2, col=1)
                    fig_combined.update_layout(height=800, title="Thống kê góc vai", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(26,26,46,0.9)', font=dict(color='white'))
                    img_path = "temp_vai_stats.png"
                    fig_combined.write_image(img_path, width=1000, height=800, scale=2)
                    with open(img_path, "rb") as f:
                        st.download_button("✅ Click để tải", f, "thong_ke_goc_vai.png", "image/png")
                    os.remove(img_path)
                except:
                    st.warning("⚠️ Cần cài đặt kaleido để xuất ảnh")
    
    # === SUB TAB 2: BIỂU ĐỒ GÓC KHUỶU ===
    with sub_tab2:
        st.markdown("### 📊 PHÂN TÍCH CHI TIẾT GÓC KHUỶU")
        
        # Biểu đồ đường góc khuỷu
        fig_khuyu = ve_bieu_do_goc_khuyu(df, bt)
        st.plotly_chart(fig_khuyu, width='stretch')
        
        # Thống kê góc khuỷu
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Giá trị trung bình", f"{tk['tb_goc_khuyu']:.1f}°",
                     delta=f"Chuẩn: {bt['chuan']['khuyu']}°")
        with col2:
            st.metric("📈 Giá trị lớn nhất", f"{tk['max_goc_khuyu']:.1f}°")
        with col3:
            st.metric("📉 Giá trị nhỏ nhất", f"{tk['min_goc_khuyu']:.1f}°")
        
        st.markdown("---")
        
        # Biểu đồ Histogram cho góc khuỷu
        st.markdown("#### 📊 PHÂN PHỐI GÓC KHUỶU (HISTOGRAM)")
        fig_hist_khuyu = go.Figure()
        fig_hist_khuyu.add_trace(go.Histogram(
            x=df['goc_khuyu'],
            nbinsx=25,
            marker_color='#FF6B6B',
            opacity=0.8,
            name='Góc khuỷu',
            hovertemplate='Góc: %{x:.1f}°<br>Tần suất: %{y}<extra></extra>'
        ))
        fig_hist_khuyu.add_vline(x=bt['chuan']['khuyu'], line_dash='dash', line_color='#00FF00',
                                 annotation_text=f"Chuẩn: {bt['chuan']['khuyu']}°", annotation_position="top right")
        fig_hist_khuyu.add_vrect(x0=bt['chuan']['khuyu']-bt['chuan']['sai_so'], 
                                 x1=bt['chuan']['khuyu']+bt['chuan']['sai_so'],
                                 fillcolor='rgba(0, 255, 0, 0.15)', line_width=0)
        fig_hist_khuyu.update_layout(
            title="Phân phối góc khuỷu theo tần suất",
            xaxis_title="Góc (độ)",
            yaxis_title="Tần suất",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(26,26,46,0.9)',
            height=450,
            font=dict(color='white')
        )
        st.plotly_chart(fig_hist_khuyu, width='stretch')
        
        st.markdown("---")
        
        # Biểu đồ Boxplot so sánh (giống như trên)
        st.markdown("#### 📦 SO SÁNH PHÂN PHỐI GÓC (BOX PLOT)")
        fig_box2 = go.Figure()
        fig_box2.add_trace(go.Box(
            y=df['goc_vai'],
            name='Góc vai',
            marker_color='#00CED1',
            boxmean='sd'
        ))
        fig_box2.add_trace(go.Box(
            y=df['goc_khuyu'],
            name='Góc khuỷu',
            marker_color='#FF6B6B',
            boxmean='sd'
        ))
        fig_box2.update_layout(
            yaxis_title="Góc (độ)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(26,26,46,0.9)',
            height=500,
            font=dict(color='white'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='white',
                borderwidth=1
            )
        )
        st.plotly_chart(fig_box2, width='stretch')
        
        # Nút tải biểu đồ
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Tải biểu đồ góc khuỷu (PNG)", width='stretch', key="save_khuyu"):
                try:
                    img_path = "temp_khuyu_chart.png"
                    fig_khuyu.write_image(img_path, width=1200, height=600, scale=2)
                    with open(img_path, "rb") as f:
                        st.download_button("📥 Click để tải xuống", f, "bieu_do_goc_khuyu.png", "image/png")
                    os.remove(img_path)
                except:
                    st.warning("⚠️ Cần cài đặt kaleido để xuất ảnh: pip install -U kaleido")
        with col2:
            if st.button("💾 Tải Histogram + Boxplot (PNG)", width='stretch', key="save_khuyu_stats"):
                try:
                    fig_combined = make_subplots(rows=2, cols=1, 
                                                  subplot_titles=("Histogram góc khuỷu", "Boxplot so sánh"),
                                                  vertical_spacing=0.15)
                    fig_combined.add_trace(go.Histogram(x=df['goc_khuyu'], nbinsx=25, marker_color='#FF6B6B', name='Histogram'), row=1, col=1)
                    fig_combined.add_trace(go.Box(y=df['goc_vai'], name='Góc vai', marker_color='#00CED1', boxmean='sd'), row=2, col=1)
                    fig_combined.add_trace(go.Box(y=df['goc_khuyu'], name='Góc khuỷu', marker_color='#FF6B6B', boxmean='sd'), row=2, col=1)
                    fig_combined.update_layout(height=800, title="Thống kê góc khuỷu", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(26,26,46,0.9)', font=dict(color='white'))
                    img_path = "temp_khuyu_stats.png"
                    fig_combined.write_image(img_path, width=1000, height=800, scale=2)
                    with open(img_path, "rb") as f:
                        st.download_button("✅ Click để tải", f, "thong_ke_goc_khuyu.png", "image/png")
                    os.remove(img_path)
                except:
                    st.warning("⚠️ Cần cài đặt kaleido để xuất ảnh")
    
    # === SUB TAB 3: CẢNH BÁO CHI TIẾT ===
    with sub_tab3:
        st.markdown("### ⚠️ PHÂN TÍCH LỖI ĐỘNG TÁC")
        
        if tk.get('warnings') and len(tk['warnings']) > 0:
            warning_counts = {}
            for w in tk['warnings']:
                warning_counts[w] = warning_counts.get(w, 0) + 1
            
            # Hiển thị tổng quan lỗi
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #FF6B6B;">{len(tk['warnings'])}</div>
                    <div class="metric-label">⚠️ Tổng số cảnh báo</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #FFD700;">{len(warning_counts)}</div>
                    <div class="metric-label">📋 Loại lỗi khác nhau</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### 📋 CHI TIẾT CÁC LỖI PHÁT HIỆN:")
            
            for warning, count in warning_counts.items():
                st.markdown(f"""
                <div class="warning-box" style="margin: 10px 0;">
                    <strong style="font-size: 1rem;">{warning}</strong><br>
                    <span style="color: #aaa;">📊 Xuất hiện {count} lần trong video ({count/tk['tong_frame_hop_le']*100:.1f}% số frame hợp lệ)</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Biểu đồ tròn thống kê lỗi
            if len(warning_counts) > 0:
                st.markdown("#### 📊 BIỂU ĐỒ THỐNG KÊ LỖI")
                fig_warning = go.Figure(data=[go.Pie(
                    labels=list(warning_counts.keys()),
                    values=list(warning_counts.values()),
                    hole=0.4,
                    marker_colors=['#FF6B6B', '#FFB347', '#4ECDC4', '#45B7D1', '#96CEB4']
                )])
                fig_warning.update_layout(
                    title="Phân bố các loại lỗi",
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=500
                )
                st.plotly_chart(fig_warning, width='stretch')
        else:
            st.success("🎉 Không phát hiện lỗi nào! Bệnh nhân thực hiện động tác rất tốt.")
    
    # === SUB TAB 4: XUẤT DỮ LIỆU ===
    with sub_tab4:
        st.markdown("### 📁 XUẤT DỮ LIỆU PHÂN TÍCH")
        
        st.markdown("""
        <div style="background: rgba(26,26,46,0.8); border-radius: 16px; padding: 1rem; margin-bottom: 1rem;">
            <p style="color: #aaa;">📌 Tại đây bạn có thể tải xuống toàn bộ dữ liệu phân tích dưới dạng file CSV và hình ảnh biểu đồ.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 DỮ LIỆU DẠNG BẢNG")
            st.dataframe(df, width='stretch', height=300)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Tải file CSV đầy đủ",
                csv_data,
                f"ket_qua_phan_tich_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                width='stretch'
            )
        
        with col2:
            st.markdown("#### 📊 DỮ LIỆU THỐNG KÊ")
            stats_df = pd.DataFrame({
                'Chỉ số': ['Độ chính xác tổng thể', 'Tỉ lệ đúng góc vai', 'Tỉ lệ đúng góc khuỷu',
                          'Góc vai TB', 'Góc khuỷu TB', 'Số frame đúng', 'Tổng frame hợp lệ'],
                'Giá trị': [f"{tk['do_chinh_xac']:.1f}%", f"{tk['ty_le_vai_dung']:.1f}%", f"{tk['ty_le_khuyu_dung']:.1f}%",
                           f"{tk['tb_goc_vai']:.1f}°", f"{tk['tb_goc_khuyu']:.1f}°", 
                           f"{tk['frame_dung']}", f"{tk['tong_frame_hop_le']}"]
            })
            st.dataframe(stats_df, width='stretch', hide_index=True)
            
            stats_csv = stats_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                "📥 Tải thống kê CSV",
                stats_csv,
                f"thong_ke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                width='stretch'
            )
        
        st.markdown("---")
        st.markdown("#### 🖼️ TẢI XUỐNG TẤT CẢ BIỂU ĐỒ")
        
        if st.button("📸 Tải xuống tất cả biểu đồ (ZIP)", width='stretch'):
            try:
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Biểu đồ góc vai
                    fig_vai = ve_bieu_do_goc_vai(df, bt)
                    fig_vai.write_image("temp_vai.png", width=1200, height=600, scale=2)
                    zip_file.write("temp_vai.png", "bieu_do_goc_vai.png")
                    os.remove("temp_vai.png")
                    
                    # Biểu đồ góc khuỷu
                    fig_khuyu = ve_bieu_do_goc_khuyu(df, bt)
                    fig_khuyu.write_image("temp_khuyu.png", width=1200, height=600, scale=2)
                    zip_file.write("temp_khuyu.png", "bieu_do_goc_khuyu.png")
                    os.remove("temp_khuyu.png")
                    
                    # Biểu đồ histogram
                    fig_hist = ve_bieu_do_histogram(df, bt)
                    fig_hist.write_image("temp_hist.png", width=1200, height=500, scale=2)
                    zip_file.write("temp_hist.png", "histogram.png")
                    os.remove("temp_hist.png")
                    
                    # Biểu đồ boxplot
                    fig_box = ve_bieu_do_boxplot(df)
                    fig_box.write_image("temp_box.png", width=1000, height=500, scale=2)
                    zip_file.write("temp_box.png", "boxplot.png")
                    os.remove("temp_box.png")
                
                zip_buffer.seek(0)
                st.download_button(
                    "✅ Click để tải xuống",
                    zip_buffer,
                    "tat_ca_bieu_do.zip",
                    "application/zip"
                )
            except Exception as e:
                st.error(f"❌ Lỗi xuất ảnh: {e}\nVui lòng cài đặt: pip install -U kaleido")
                
# ============================================
# HÀM HIỂN THỊ LỊCH NHẮC NHỞ
# ============================================
def hien_thi_lich_nhac_nho():
    """Hiển thị lịch nhắc nhở chi tiết"""
    st.markdown("## ⏰ LỊCH NHẮC NHỞ CHI TIẾT")
    
    current_time = datetime.now()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Hôm nay", current_time.strftime("%d/%m/%Y"))
    with col2:
        st.metric("⏰ Hiện tại", current_time.strftime("%H:%M:%S"))
    with col3:
        st.metric("📆 Thứ", current_time.strftime("%A"))
    with col4:
        tong = len(st.session_state.appointments) + len(st.session_state.exercise_reminders) + len(st.session_state.medication_reminders)
        st.metric("📊 Tổng lịch", tong)
    
    st.markdown("---")
    
    tab_lich1, tab_lich2, tab_lich3, tab_lich4 = st.tabs([
        "🩺 Lịch hẹn khám", "🏋️ Lịch tập luyện", "💊 Lịch uống thuốc", "➕ Thêm mới"
    ])
    
    with tab_lich1:
        st.subheader("🩺 Lịch hẹn với bác sĩ")
        if not st.session_state.appointments:
            st.info("📭 Không có lịch hẹn nào.")
        else:
            for app in st.session_state.appointments:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(26,26,46,0.8); border-radius: 16px; padding: 1rem; margin-bottom: 0.8rem;">
                        <strong style="color: #ffd700;">📌 {app['title']}</strong><br>
                        🕒 {app['datetime']}<br>
                        📝 {app.get('notes', '')}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_app_{app['id']}"):
                        st.session_state.appointments.remove(app)
                        st.rerun()
    
    with tab_lich2:
        st.subheader("🏋️ Lịch tập luyện")
        if not st.session_state.exercise_reminders:
            st.info("📭 Không có lịch tập nào.")
        else:
            for ex in st.session_state.exercise_reminders:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(26,26,46,0.8); border-radius: 16px; padding: 1rem; margin-bottom: 0.8rem;">
                        <strong style="color: #ffd700;">💪 {ex['exercise_name']}</strong><br>
                        🕒 {ex['datetime']}<br>
                        🔁 {ex.get('frequency', 'Một lần')}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_ex_{ex['id']}"):
                        st.session_state.exercise_reminders.remove(ex)
                        st.rerun()
    
    with tab_lich3:
        st.subheader("💊 Lịch uống thuốc")
        if not st.session_state.medication_reminders:
            st.info("📭 Không có lịch uống thuốc nào.")
        else:
            for med in st.session_state.medication_reminders:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(26,26,46,0.8); border-radius: 16px; padding: 1rem; margin-bottom: 0.8rem;">
                        <strong style="color: #ffd700;">💊 {med['medication_name']}</strong><br>
                        🕒 {med['datetime']}<br>
                        💊 Liều: {med.get('dosage', 'Theo chỉ định')}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_med_{med['id']}"):
                        st.session_state.medication_reminders.remove(med)
                        st.rerun()
    
    with tab_lich4:
        st.subheader("➕ Thêm lịch nhắc nhở mới")
        loai = st.radio("Chọn loại:", ["Lịch hẹn khám", "Lịch tập luyện", "Lịch uống thuốc"], horizontal=True)
        
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Ngày", min_value=datetime.now().date())
        with col2:
            time_input = st.time_input("Giờ")
        
        if loai == "Lịch hẹn khám":
            title = st.text_input("Tiêu đề")
            notes = st.text_area("Ghi chú")
            if st.button("➕ Thêm", key="add_appointment"):
                if title:
                    new_id = len(st.session_state.appointments) + len(st.session_state.exercise_reminders) + len(st.session_state.medication_reminders) + 1
                    st.session_state.appointments.append({
                        'id': new_id,
                        'title': title,
                        'datetime': f"{date} {time_input}",
                        'notes': notes
                    })
                    st.success("✅ Đã thêm!")
                    st.rerun()
        
        elif loai == "Lịch tập luyện":
            exercise = st.selectbox("Bài tập", list(BAI_TAP.keys()), format_func=lambda x: BAI_TAP[x]['ten'])
            frequency = st.selectbox("Tần suất", ["Một lần", "Hàng ngày", "Thứ 2-4-6", "Thứ 3-5-7"])
            notes = st.text_area("Ghi chú")
            if st.button("➕ Thêm", key="add_exercise"):
                new_id = len(st.session_state.appointments) + len(st.session_state.exercise_reminders) + len(st.session_state.medication_reminders) + 1
                st.session_state.exercise_reminders.append({
                    'id': new_id,
                    'exercise_name': BAI_TAP[exercise]['ten'],
                    'datetime': f"{date} {time_input}",
                    'frequency': frequency,
                    'notes': notes
                })
                st.success("✅ Đã thêm!")
                st.rerun()
        
        else:
            med_name = st.text_input("Tên thuốc")
            dosage = st.text_input("Liều lượng")
            notes = st.text_area("Ghi chú")
            if st.button("➕ Thêm", key="add_medication"):
                if med_name:
                    new_id = len(st.session_state.appointments) + len(st.session_state.exercise_reminders) + len(st.session_state.medication_reminders) + 1
                    st.session_state.medication_reminders.append({
                        'id': new_id,
                        'medication_name': med_name,
                        'dosage': dosage,
                        'datetime': f"{date} {time_input}",
                        'notes': notes,
                        'taken': False
                    })
                    st.success("✅ Đã thêm!")
                    st.rerun()

# ============================================
# HÀM HIỂN THỊ LỊCH FRAMES ĐẦY ĐỦ
# ============================================
def hien_thi_frames_day_du():
    """Hiển thị tất cả frames với phân trang - TỐI ƯU TỐC ĐỘ"""
    
    if not st.session_state.get('all_frames_data_path') or not os.path.exists(st.session_state.all_frames_data_path):
        st.info("📭 Không có dữ liệu khung hình để hiển thị.")
        return
    
    # Đọc dữ liệu từ file JSON thay vì RAM
    import json
    with open(st.session_state.all_frames_data_path, 'r', encoding='utf-8') as f:
        all_frames_data = json.load(f)
    
    total_frames = len(all_frames_data)
    
    # Chuẩn bị dữ liệu
    frame_paths = [f['path'] for f in all_frames_data]
    dung_flags = [f.get('dung', False) for f in all_frames_data]
    timestamps = [f.get('timestamp', '00:00') for f in all_frames_data]
    goc_vai_list = [f.get('goc_vai') for f in all_frames_data]
    goc_khuyu_list = [f.get('goc_khuyu') for f in all_frames_data]
    frame_indices = [f.get('index', i) for i, f in enumerate(all_frames_data)]
    
    st.markdown(f"### 📸 TẤT CẢ FRAMES ĐÃ XỬ LÝ (Tổng: {total_frames} frames)")
    
    # Bộ lọc
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        loc_frame = st.selectbox("🔍 Lọc theo kết quả", ["Tất cả", "PASS (Đúng)", "FAIL (Sai)"], key="filter_select")
    with col2:
        quality_mode = st.selectbox("⚡ Chất lượng ảnh", ["Nhanh", "Trung bình", "Chất lượng cao"], index=0, key="quality_select")
    with col3:
        frames_per_page = st.selectbox("📄 Số frame/trang", [12, 24, 36, 48], index=1, key="per_page_select")
    with col4:
        st.write("")
        st.write("")
        if st.button("🔄 Làm mới", width='stretch', key="refresh_thumbnails"):
            st.cache_data.clear()
            st.rerun()
    
    # Chất lượng theo chế độ
    if quality_mode == "Chất lượng cao":
        thumb_quality = 85
        thumb_width = 380
    elif quality_mode == "Trung bình":
        thumb_quality = 70
        thumb_width = 320
    else:
        thumb_quality = 55
        thumb_width = 260
    
    # Lọc indices
    if loc_frame == "Tất cả":
        filtered_indices = list(range(total_frames))
    elif loc_frame == "PASS (Đúng)":
        filtered_indices = [i for i, f in enumerate(all_frames_data) if f.get('dung')]
    else:
        filtered_indices = [i for i, f in enumerate(all_frames_data) if not f.get('dung')]
    
    total_filtered = len(filtered_indices)
    total_pages = max(1, (total_filtered + frames_per_page - 1) // frames_per_page)
    
    # Khởi tạo current_page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    
    st.markdown("---")
    col_prev, col_page, col_next, col_info = st.columns([1, 2, 1, 2])
    
    with col_prev:
        if st.button("◀ Trang trước", width='stretch', key="prev_page"):
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
    
    with col_page:
        page = st.number_input("Trang", min_value=1, max_value=total_pages, 
                              value=st.session_state.current_page, 
                              step=1, label_visibility="collapsed", key="page_input")
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()
    
    with col_next:
        if st.button("Trang sau ▶", width='stretch', key="next_page"):
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()
    
    with col_info:
        st.caption(f"📊 Hiển thị {min(frames_per_page, total_filtered)}/{total_filtered} frame | Trang {st.session_state.current_page}/{total_pages}")
    
    # Lấy indices của trang hiện tại
    start_idx = (st.session_state.current_page - 1) * frames_per_page
    end_idx = min(start_idx + frames_per_page, total_filtered)
    page_indices = filtered_indices[start_idx:end_idx]
    
    # === XỬ LÝ THUMBNAIL CHO TRANG HIỆN TẠI ===
    # Hiển thị grid 4 cột (mỗi hàng 4 ảnh)
    cols_per_row = 4
    for i in range(0, len(page_indices), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(page_indices):
                original_idx = page_indices[idx]
                path = frame_paths[original_idx]
                if not os.path.exists(path):
                    continue
                
                frame_data = all_frames_data[original_idx]
                border_color = "#00FF00" if frame_data.get('dung') else "#FF4444"
                
                with cols[j]:
                    st.markdown(f"""
                    <div style="text-align:center; background: rgba(0,0,0,0.4); border-radius: 12px 12px 0 0; padding: 4px; border-top: 3px solid {border_color}; border-left: 3px solid {border_color}; border-right: 3px solid {border_color};">
                        <span style="color:#aaa; font-size:0.8rem; font-weight:bold;">⏱️ Frame #{frame_data['index']} | {frame_data.get('timestamp', '00:00')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(path, use_container_width=True)
                    st.markdown(f"""
                    <div style="height: 15px; margin-bottom: 15px;"></div>
                    """, unsafe_allow_html=True)
    
    # Thống kê
    st.markdown("---")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("📊 Tổng số frames", total_frames)
    with col_stat2:
        pass_count = sum(1 for f in all_frames_data if f.get('dung'))
        st.metric("✅ Số frame PASS", pass_count)
    with col_stat3:
        fail_count = total_frames - pass_count
        st.metric("❌ Số frame FAIL", fail_count)
    with col_stat4:
        st.metric("📄 Tổng số trang", total_pages)


# ============================================
# MAIN - GIỮ NGUYÊN CẤU TRÚC TAB
# ============================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Hệ thống giám sát tập luyện Phục hồi chức năng từ xa</h1>
        <p>Dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision)</p>
        <div class="research-badge"><span>📚 ĐỀ TÀI NGHIÊN CỨU KHOA HỌC CẤP TRƯỜNG - NĂM HỌC 2025-2026</span></div>
        <p style="font-size: 0.8rem;">Bệnh viện Đa khoa Phạm Ngọc Thạch - Trường Đại học Y tế Công cộng</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 👤 THÔNG TIN BỆNH NHÂN")
        ten_benh_nhan = st.text_input("Họ và tên", placeholder="VD: Nguyễn Văn A")
        ma_benh_nhan = st.text_input("Mã số bệnh nhân", placeholder="VD: BN0001")
        col1, col2 = st.columns(2)
        with col1: tuoi = st.number_input("Tuổi", 0, 120, 22)
        with col2: gioi_tinh = st.selectbox("Giới tính", ["", "Nam", "Nữ"])
        
        st.markdown("### 🩺 THÔNG TIN LÂM SÀNG")
        chan_doan = st.selectbox("Chẩn đoán", ["","Viêm quanh khớp vai", "Hội chứng chóp xoay", "Đông cứng khớp vai"])
        muc_do_dau = st.slider("Mức độ đau (VAS 0-10)", 0, 10, 3)
        
        st.markdown("### 🎯 CHỌN BÀI TẬP")
        ma_bai_tap = st.selectbox("Bài tập", list(BAI_TAP.keys()), format_func=lambda x: f"{BAI_TAP[x]['icon']} {BAI_TAP[x]['ten']}")
        bai_tap = BAI_TAP[ma_bai_tap]
        
        st.markdown("### 📺 VIDEO HƯỚNG DẪN")
        st.video(bai_tap["youtube"])
        
        st.markdown("---")
        st.markdown("**👨‍🏫 Giảng viên hướng dẫn:** TS. Trần Hồng Việt")
        st.markdown("**👩‍⚕️ Chủ nhiệm đề tài:** Đinh Lê Quỳnh Phương")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 TRANG CHỦ", "📊 PHÂN TÍCH", "🎬 VIDEO & ẢNH",
        "⏰ LỊCH NHẮC NHỞ", "📚 ĐỀ TÀI NCKH", "👥 THÀNH VIÊN"
    ])
    
    # ==================== TAB 1: TRANG CHỦ ====================
    with tab1:
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h3>{bai_tap['icon']} {bai_tap['ten']}</h3>
                <p>{bai_tap['mo_ta']}</p>
                <p><strong>⏱️ Thời gian:</strong> {bai_tap['thoi_gian']} giây/lần</p>
                <p><strong>🔄 Số lần:</strong> {bai_tap['lan']} lần/ngày</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📖 HƯỚNG DẪN TẬP LUYỆN", expanded=True):
                st.markdown(bai_tap['huong_dan'])
            with st.expander("✨ LỢI ÍCH CỦA BÀI TẬP", expanded=False):
                for loi_ich in bai_tap['loi_ich']:
                    st.markdown(f"- {loi_ich}")
        
        with col2:
            chuan = bai_tap['chuan']
            st.markdown(f"""
            <div style="background:rgba(26,26,46,0.8); padding:1.2rem; border-radius:16px; text-align:center; border:1px solid #2a5298;">
                <h4 style="color:#fff;">🎯 THÔNG SỐ CHUẨN</h4>
                <p style="color:#00CED1;">🦾 Góc vai: <strong>{chuan['vai']}°</strong> ±{chuan['sai_so']}°</p>
                <p style="color:#FF6B6B;">💪 Góc khuỷu: <strong>{chuan['khuyu']}°</strong> ±{chuan['sai_so']}°</p>
                <hr style="margin:10px 0;">
                <p style="color:#aaa; font-size:0.8rem;">✅ Đạt: Cả 2 góc trong vùng cho phép</p>
                <p style="color:#aaa; font-size:0.8rem;">❌ Không đạt: Một hoặc cả 2 góc ngoài vùng cho phép</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info(f"📁 Hỗ trợ upload file tối đa {MAX_FILE_SIZE_MB}MB (MP4, MOV, AVI, MKV)")
        
        file_upload = st.file_uploader(
            "📤 Tải lên video tập luyện của bệnh nhân", 
            type=["mp4", "mov", "avi", "mkv", "MP4", "MOV"],
            help=f"Hỗ trợ file tối đa {MAX_FILE_SIZE_MB}MB"
        )
        
        if file_upload is not None and not st.session_state.processing:
            file_size_mb = file_upload.size / (1024 * 1024)
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File quá lớn! {file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
            else:
                st.success(f"✅ Đã chọn file: {file_upload.name} ({file_size_mb:.1f} MB)")
                
                if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", width='stretch'):
                    st.session_state.processing = True
                    st.session_state.has_data = False
                    st.session_state.all_frames_data = []
                    st.session_state.angle_df = None
                    st.session_state.stats = None
                    st.session_state.frames_zip = None
                    st.session_state.temp_video_file = None
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.info("📤 Đang đọc file video...")
                        try:
                            file_content = file_upload.getvalue()
                        except ValueError:
                            st.warning("⚠️ File tải lên đã hết hạn lưu trữ tạm thời do trang web vừa tải lại. Vui lòng CHỌN LẠI FILE VIDEO và bấm Bắt đầu phân tích!")
                            st.session_state.processing = False
                            st.stop()
                        
                        progress_bar.progress(0.1)
                        status_text.info("💾 Đang lưu file tạm...")
                        
                        is_mov = file_upload.name.lower().endswith('.mov')
                        suffix = '.mp4' if not is_mov else '.mov'
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(file_content)
                            video_path = tmp_file.name
                        
                        if is_mov:
                            status_text.info("🔄 Đang chuyển đổi MOV sang MP4...")
                            converted_path = convert_mov_to_mp4(video_path)
                            if converted_path != video_path:
                                video_path = converted_path
                        
                        progress_bar.progress(0.2)
                        status_text.info("🎬 Đang xử lý video với AI... (có thể mất vài phút)")
                        
                        start_time = time.time()
                        
                        def update_progress(p):
                            progress_bar.progress(0.2 + p * 0.7)
                            status_text.info(f"🔄 Đang xử lý frame... {p*100:.0f}%")
                        
                        output_path, _, _, angle_data, total_frames, valid_frames, temp_folder, zip_data, frame_paths, _, all_frames_data, all_warnings = xu_ly_video_day_du(
                            video_path, bai_tap['chuan'], update_progress
                        )
                        
                        progress_bar.progress(0.95)
                        status_text.info("📦 Đang hoàn tất...")
                        
                        process_time = time.time() - start_time
                        
                        if valid_frames > 0 and len(angle_data) > 0:
                            df = pd.DataFrame(angle_data)
                            metrics = tinh_metrics_chi_tiet(df)
                            
                            st.session_state.has_data = True
                            st.session_state.angle_df = df
                            st.session_state.stats = {
                                "do_chinh_xac": metrics["ty_le_tong_the"],
                                "ty_le_vai_dung": metrics["ty_le_vai_dung"],
                                "ty_le_khuyu_dung": metrics["ty_le_khuyu_dung"],
                                "frame_dung": metrics["frame_dung"],
                                "tong_frame_hop_le": valid_frames,
                                "tb_goc_vai": metrics["tb_goc_vai"],
                                "tb_goc_khuyu": metrics["tb_goc_khuyu"],
                                "min_goc_vai": metrics["min_goc_vai"],
                                "max_goc_vai": metrics["max_goc_vai"],
                                "min_goc_khuyu": metrics["min_goc_khuyu"],
                                "max_goc_khuyu": metrics["max_goc_khuyu"],
                                "std_goc_vai": metrics["std_goc_vai"],
                                "std_goc_khuyu": metrics["std_goc_khuyu"],
                                "thoi_gian": process_time,
                                "tong_frame": total_frames,
                                "warnings": all_warnings
                            }
                            st.session_state.frames_zip = zip_data
                            st.session_state.exercise = bai_tap
                            st.session_state.all_frames_paths = frame_paths
                            st.session_state.temp_video_file = output_path
                            st.session_state.all_frames_data_path = all_frames_data
                            
                            try:
                                os.unlink(video_path)
                            except:
                                pass
                            
                            status_text.empty()
                            progress_bar.empty()
                            st.balloons()
                            st.success(f"✅ Xử lý hoàn tất trong {process_time:.1f} giây!")
                            st.info(f"📊 Tổng số frame: {total_frames} | Hợp lệ: {valid_frames} frames | Độ chính xác: {metrics['ty_le_tong_the']:.1f}%")
                            
                            st.session_state.processing = False
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Không phát hiện khung xương! Vui lòng quay video rõ người tập hơn.")
                            st.session_state.processing = False
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi xử lý: {str(e)}")
                        st.session_state.processing = False
                        progress_bar.empty()
                        status_text.empty()
        
        elif st.session_state.processing:
            st.warning("⏳ Đang xử lý video, vui lòng chờ...")
            
            if st.button("❌ Hủy xử lý", width='stretch'):
                st.session_state.processing = False
                st.rerun()
        
        elif st.session_state.has_data:
            st.success("✅ Đã có kết quả phân tích! Hãy xem các tab PHÂN TÍCH và VIDEO & ẢNH.")
            
            if st.button("🔄 PHÂN TÍCH VIDEO MỚI", width='stretch'):
                keys_to_clear = ['has_data', 'angle_df', 'stats', 'frames_zip', 'exercise', 
                                'temp_video_file', 'processed_video_bytes', 
                                'all_frames_data', 'all_frames_paths']
                for key in keys_to_clear:
                    if key in st.session_state:
                        st.session_state[key] = None if key != 'has_data' else False
                st.session_state.processing = False
                st.rerun()
    
    # ==================== TAB 2: PHÂN TÍCH (THIẾT KẾ MỚI) ====================
    with tab2:
        hien_thi_tab_phan_tich()
    
    # ==================== TAB 3: VIDEO & ẢNH ====================
    with tab3:
        if st.session_state.has_data and st.session_state.temp_video_file and os.path.exists(st.session_state.temp_video_file):
            st.markdown("### 🎬 VIDEO ĐÃ PHÂN TÍCH")
            
            # Đọc video và hiển thị bằng Streamlit video (Tránh dùng Base64 gây lag trình duyệt)
            st.video(st.session_state.temp_video_file)
            
            # Đọc dữ liệu video để phục vụ cho nút tải xuống
            file_ext = os.path.splitext(st.session_state.temp_video_file)[1]
            mime_type = "video/webm" if file_ext == ".webm" else "video/mp4"
            col1, col2 = st.columns(2)
            with col1:
                with open(st.session_state.temp_video_file, 'rb') as f:
                    st.download_button("📥 Tải video xuống", f, f"video_processed{file_ext}", mime_type)
            with col2:
                if st.session_state.frames_zip and os.path.exists(st.session_state.frames_zip):
                    with open(st.session_state.frames_zip, "rb") as f:
                        st.download_button("📥 Tải tất cả frames (ZIP)", f, "tat_ca_frames.zip", "application/zip")
            
            st.markdown("---")
            
            hien_thi_frames_day_du()
            
        else:
            st.info("ℹ️ Chưa có video. Hãy upload và xử lý video ở tab TRANG CHỦ.")
    
    # ==================== TAB 4: LỊCH NHẮC NHỞ ====================
    with tab4:
        hien_thi_lich_nhac_nho()
    
    # ==================== TAB 5: ĐỀ TÀI NCKH ====================
    with tab5:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center; border: 1px solid #2a5298;">
            <h2 style="color: white; margin: 0;">📚 ĐỀ TÀI NGHIÊN CỨU KHOA HỌC</h2>
            <p style="color: #ffd700; font-size: 1.1rem; margin-top: 0.5rem;">Phát triển Mô hình thử nghiệm giám sát tập luyện Phục hồi chức năng từ xa</p>
            <p style="color: #ccc;">Dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision)</p>
            <p style="color: #aaa; font-size: 0.9rem;">Bệnh viện Đa khoa Phạm Ngọc Thạch - Trường Đại học Y tế Công cộng (2025-2026)</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📌 ĐẶT VẤN ĐỀ", expanded=True):
            st.markdown("""
            Trong những năm gần đây, cùng với sự gia tăng của các bệnh lý cơ xương khớp, chấn thương thể thao và đột quỵ, nhu cầu phục hồi chức năng (PHCN) trên toàn thế giới ngày càng tăng cao. 
            
            Theo Tổ chức Y tế Thế giới (WHO), hiện có khoảng 2,4 tỷ người cần ít nhất một hình thức phục hồi chức năng, chiếm gần một phần ba dân số toàn cầu. Tại Việt Nam, theo Hội Phục hồi chức năng Việt Nam (2023), có khoảng 7,06% dân số từ 2 tuổi trở lên là người khuyết tật, trong đó phần lớn cần được can thiệp PHCN.
            
            Mặc dù nhu cầu PHCN lớn, song năng lực cung cấp dịch vụ này tại Việt Nam vẫn còn hạn chế. Trung bình 10.000 người dân chỉ có 0,25 nhân viên phục hồi chức năng, thấp hơn đáng kể so với khuyến nghị của WHO là 0,5-1 người/10.000 dân. Thực tế này khiến nhiều bệnh nhân phải tự tập luyện tại nhà sau khi xuất viện mà thiếu sự giám sát chuyên môn.
            
            Xuất phát từ thực tiễn trên, nhóm nghiên cứu quyết định thực hiện đề tài: **"Phát triển Mô hình thử nghiệm giám sát tập luyện Phục hồi chức năng từ xa dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision)"**.
            """)
        
        with st.expander("🎯 MỤC TIÊU NGHIÊN CỨU", expanded=True):
            st.markdown("""
            **Mục tiêu 1:** Xây dựng mô hình nhận diện và đánh giá 3 bài tập phục hồi chức năng cho bệnh nhân viêm quanh khớp vai, bao gồm:
            - Bài tập con lắc Codman
            - Bài tập với gậy
            - Bài tập với dây kháng lực
            
            **Mục tiêu 2:** So sánh độ chính xác của mô hình với đánh giá thủ công trên một tập dữ liệu nhỏ.
            """)
        
        with st.expander("🔬 ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU", expanded=True):
            st.markdown("""
            **Đối tượng nghiên cứu:** 05 bệnh nhân viêm quanh khớp vai + nhóm chuyên gia PHCN tại Khoa Phục hồi chức năng, Bệnh viện Đa khoa Phạm Ngọc Thạch.
            
            **Thiết kế nghiên cứu:** Nghiên cứu định lượng, phát triển mô hình học máy.
            
            **Công nghệ sử dụng:** 
            - MediaPipe Pose Estimation cho ước lượng tư thế
            - Python và các thư viện xử lý ảnh (OpenCV, NumPy, Pandas)
            - Streamlit cho giao diện người dùng
            - Plotly cho trực quan hóa dữ liệu
            
            **Cỡ mẫu dự kiến:** 500-1000 chuỗi chuyển động.
            """)
        
        with st.expander("📊 KẾT QUẢ DỰ KIẾN", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Độ chính xác (Accuracy)", "≥ 90%")
                st.metric("F1-Score", "≥ 0.85")
            with col2:
                st.metric("Sai số MAE", "< 5°")
                st.metric("Hệ số ICC", "≥ 0.75")
            with col3:
                st.metric("Precision", "≥ 0.85")
                st.metric("Recall", "≥ 0.85")
        
        with st.expander("🎁 ĐÓNG GÓP CỦA ĐỀ TÀI", expanded=True):
            st.markdown("""
            **- Về khoa học và đào tạo:** Xây dựng mô hình nhận diện động tác PHCN, tạo bộ dữ liệu chuẩn hóa, là tài liệu thực hành cho sinh viên ngành Khoa học dữ liệu y sinh.
            
            **- Về phát triển kinh tế:** Giảm chi phí đi lại, giảm tải cho nhân viên y tế, tối ưu nguồn lực bệnh viện.
            
            **- Về xã hội:** Tăng khả năng tiếp cận dịch vụ PHCN, thúc đẩy chuyển đổi số y tế, xây dựng hệ thống chăm sóc sức khỏe thông minh.
            """)
        
        with st.expander("📚 TÀI LIỆU THAM KHẢO", expanded=False):
            st.markdown("""
            1. WHO. Rehabilitation 2030: A call for action.
            2. Cieza A, et al. Global estimates of the need for rehabilitation. Lancet. 2021.
            3. Lugaresi C, et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv. 2019.
            4. Cao Z, et al. OpenPose: Realtime Multi-Person 2D Pose Estimation. arXiv. 2019.
            5. Hellstén T, et al. Reliability and validity of computer vision-based markerless human pose estimation. Healthc Technol Lett. 2025.
            6. Ino T, et al. Validity and Reliability of OpenPose-Based Motion Analysis. J Sports Sci Med. 2024.
            7. Aguilar-Ortega R, et al. UCO Physical Rehabilitation: New Dataset and Study. Sensors. 2023.
            8. Nguyễn Thị Ngọc Lan, et al. Thực trạng nhu cầu phục hồi chức năng tại Việt Nam. Tạp chí Y học Việt Nam. 2024.
            """)
    
    # ==================== TAB 6: THÀNH VIÊN ====================
    with tab6:
        st.markdown("### 👨‍🏫 GIẢNG VIÊN HƯỚNG DẪN")
        st.markdown("""
        <div class="lecturer-card">
            <div class="lecturer-name">TS. Trần Hồng Việt</div>
            <p style="color: #ccc; margin-top: 0.5rem;">Giảng viên hướng dẫn</p>
            <p style="color: #aaa; font-size: 0.9rem;">Trường Đại học Y tế Công cộng</p>
            <p style="color: #aaa; font-size: 0.85rem;">Chuyên ngành: Khoa học dữ liệu Y sinh</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 👩‍⚕️ CHỦ NHIỆM ĐỀ TÀI")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="member-card" style="border-color: #ffd700; border: 2px solid #ffd700;">
                <div class="member-name">Đinh Lê Quỳnh Phương</div>
                <div class="member-role">⭐ Chủ nhiệm đề tài ⭐</div>
                <div class="member-class">Chuyên ngành Khoa học dữ liệu Y sinh</div>
                <div class="member-id">MSSV: 2211090031</div>
                <div class="member-id">📧 2211090031@studenthuph.edu.vn</div>
                <div class="member-id">📱 0382665916</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 👥 THÀNH VIÊN NGHIÊN CỨU")
        thanh_vien = [
            ("Kim Mạnh Hưng", "Thành viên", "CNCQ KHDL1-1A", "2211090016"),
            ("Nguyễn Hải An", "Thành viên", "CNCQ KHDL1-1A", "2211090001"),
            ("Phan Vân Anh", "Thành viên", "CNCQ KHDL1-1A", "2211090004"),
            ("Nguyễn Thị Thanh Nga", "Thành viên", "CNCQ KHDL1-1A", "2211090027"),
        ]
        
        cols = st.columns(4)
        for i, (ten, vai_tro, lop, mssv) in enumerate(thanh_vien):
            with cols[i]:
                st.markdown(f"""
                <div class="member-card">
                    <div class="member-name">{ten}</div>
                    <div class="member-role">{vai_tro}</div>
                    <div class="member-class">{lop}</div>
                    <div class="member-id">MSSV: {mssv}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🩺 CHUYÊN GIA LÂM SÀNG")
        chuyen_gia = [
            ("Nguyễn Thị Thơm", "Chuyên gia PHCN", "CNCQ KTPHCN3-1A", "2216030122"),
            ("Nguyễn Thị Thu Hương", "Chuyên gia PHCN", "CNCQYTCC22-1A", "2317010071"),
        ]
        
        cols = st.columns(2)
        for i, (ten, vai_tro, lop, mssv) in enumerate(chuyen_gia):
            with cols[i]:
                st.markdown(f"""
                <div class="member-card">
                    <div class="member-name">{ten}</div>
                    <div class="member-role">{vai_tro}</div>
                    <div class="member-class">{lop}</div>
                    <div class="member-id">MSSV: {mssv}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🏥 ĐƠN VỊ PHỐI HỢP")
        st.markdown("""
        <div style="background: rgba(26,26,46,0.8); border-radius: 16px; padding: 1.5rem; text-align: center;">
            <p style="color: #ffd700; font-weight: bold;">Bệnh viện Đa khoa Phạm Ngọc Thạch</p>
            <p style="color: #ccc;">Khoa Phục hồi chức năng</p>
            <p style="color: #aaa; font-size: 0.9rem;">Địa chỉ: 1A Đ. Đức Thắng, Đông Ngạc, Hà Nội</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"💥 Lỗi khởi động ứng dụng: {e}")
        import traceback
        st.code(traceback.format_exc())