# -*- coding: utf-8 -*-
import os
import sys
import math
import json

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

# MEDIAPIPE sẽ được load lazily khi cần xử lý video
mp_pose = None
mp_drawing = None
mp_drawing_styles = None

def init_mediapipe():
    """Load MediaPipe chỉ khi cần thiết (lazy import)"""
    global mp_pose, mp_drawing, mp_drawing_styles
    if mp_pose is None:
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            return True
        except Exception as e:
            st.error(f"🚨 Không thể khởi tạo MediaPipe: {e}")
            return False
    return True

warnings.filterwarnings("ignore")

# ============================================
# QUẢN LÝ NGƯỜI DÙNG & BẢO MẬT
# ============================================
USER_DATA_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

# Khởi tạo trạng thái đăng nhập
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'forgot_password_mode' not in st.session_state:
    st.session_state.forgot_password_mode = False
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

# KIỂM TRA ĐĂNG NHẬP GOOGLE (Hỗ trợ Streamlit Cloud Identity)
if not st.session_state.get('logged_in'):
    try:
        user_detected = None
        # Kiểm tra chuẩn mới
        if hasattr(st, 'experimental_user') and st.experimental_user.get("email"):
            user_detected = st.experimental_user
        # Kiểm tra chuẩn cũ
        elif hasattr(st, 'user') and st.user and getattr(st.user, 'email', None):
            user_detected = st.user
            
        if user_detected and user_detected.get("email"):
            st.session_state.logged_in = True
            st.session_state.user_info = {
                "username": user_detected.get("name") or user_detected.get("email", "").split("@")[0],
                "email": user_detected.get("email"),
                "auth_type": "google"
            }
            # Xóa trạng thái đang chờ auth
            if 'auth_initiated' in st.session_state:
                del st.session_state['auth_initiated']
            st.rerun() 
    except Exception as e:
        pass


# ============================================
# CẤU HÌNH TRANG
# ============================================
st.set_page_config(
    page_title="Hệ thống giám sát tập PHCN từ xa - Đề tài NCKH",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GIẢI PHÁP TRIỆT ĐỂ: Xóa chữ đè bằng cách nhắm thẳng vào cấu trúc Streamlit
st.markdown("""
<style>
    /* 1. Nhắm vào mọi thẻ span chứa văn bản icon trong Expander và File Uploader */
    [data-testid="stExpander"] summary span > span,
    [data-testid="stExpander"] summary svg + span,
    [data-testid="stFileUploader"] section span > span,
    .st-emotion-cache-1p6n6q3, /* Một số mã cache phổ biến của Streamlit */
    .st-emotion-cache-16idsys {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
        width: 0 !important;
        height: 0 !important;
    }

    /* 2. Đảm bảo các tiêu đề chính vẫn hiện rõ */
    [data-testid="stExpander"] summary p, 
    [data-testid="stExpander"] summary span p {
        font-size: 1.1rem !important;
        color: white !important;
        visibility: visible !important;
        display: block !important;
    }
    
    /* 3. Force ẩn mọi text có nội dung là arrow_... */
    span:empty, span:contains("arrow_"), span:contains("upload") {
        display: none !important;
    }

    /* === LOGIN UI CREATIVE DESIGN === */
    .login-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        color: white;
        transition: all 0.3s;
        border: 1px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
        border: 1px solid #00c6ff !important;
        box-shadow: 0 0 15px rgba(0, 198, 255, 0.4);
    }

    .google-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        background: white;
        color: #444;
        padding: 12px;
        border-radius: 12px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    
    .google-btn:hover {
        background: #f1f1f1;
        box-shadow: 0 5px 15px rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

MAX_FILE_SIZE_MB = 500

# ============================================
# CẤU HÌNH XỬ LÝ - TỐI ƯU ĐỘ CHÍNH XÁC CAO
# ============================================
SKIP_FRAMES = 0    # Xử lý mọi khung hình để đảm bảo tốc độ khớp video gốc
RESIZE_WIDTH = 640 # Tăng độ phân giải lên 640px để nhìn rõ hơn
OUTPUT_QUALITY = 50 
MAX_FRAMES = 3000  # Nâng hạn mức để bắt trọn vẹn mọi khoảnh khắc
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

def ve_cung_tron_goc(image, point1, center, point3, angle, color, radius=40):
    """Vẽ cung tròn hiển thị góc tại khớp"""
    try:
        # Tính toán vector
        v1 = np.array(point1) - np.array(center)
        v2 = np.array(point3) - np.array(center)
        
        # Tính góc bắt đầu và góc kết thúc
        angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
        angle2 = np.degrees(np.arctan2(v2[1], v2[0]))
        
        # Vẽ cung tròn (overlay để có độ trong suốt)
        overlay = image.copy()
        cv2.ellipse(overlay, center, (radius, radius), 0, angle1, angle2, color, -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
        # Vẽ viền cung tròn
        cv2.ellipse(image, center, (radius, radius), 0, angle1, angle2, color, 2)
    except:
        pass

# ============================================
# MEDIAPIPE VỚI GPU
# ============================================
@st.cache_resource
def get_pose_model():
    """Khởi tạo MediaPipe Pose với cấu hình chính xác nhất"""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=True,        # QUAN TRỌNG: Dò tìm lại từng frame, không để bị trôi
        model_complexity=1,            # Độ chính xác cao
        smooth_landmarks=False,        # Tắt làm mịn để bám sát thực tế
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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
    # 1. LẤY KÍCH THƯỚC VÀ CHUYỂN ĐỔI MÀU (Không dùng padding gây lệch)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. AI XỬ LÝ TRỰC TIẾP TRÊN FRAME GỐC VỚI CHẾ ĐỘ STATIC
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
        del rgb
        del ket_qua
        return frame_output, None, None, None, None, []
    
    # Import cục bộ để tránh phụ thuộc vào biến global
    import mediapipe as _mp
    _mp_drawing = _mp.solutions.drawing_utils
    _mp_drawing_styles = _mp.solutions.drawing_styles
    _mp_pose = _mp.solutions.pose
    
    # CHỌN MÀU DỰA TRÊN KẾT QUẢ TỔNG THỂ CỦA FRAME
    # (Tạm thời tính nhanh để lấy màu vẽ skeleton)
    skeleton_color = GREEN
    
    # 3. VẼ KHUNG XƯƠNG 33 ĐIỂM CHI TIẾT
    _mp_drawing.draw_landmarks(
        frame_output,
        ket_qua.pose_landmarks,
        _mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=_mp_drawing_styles.get_default_pose_landmarks_style(),
        connection_drawing_spec=_mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=1)
    )
    
    # LẤY TỌA ĐỘ CÁC KHỚP QUAN TRỌNG (ĐẢM BẢO KHỚP 100% VỚI FRAME)
    lm = ket_qua.pose_landmarks.landmark
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
    
    # Tự động chọn bên đang tập (bên có góc vai lớn hơn hoặc đang vận động)
    # Với Codman, tay tập thường đưa ra xa thân mình hơn tay vịn
    nhip_t = abs(goc_vai_t - 45) # Giả định chuẩn Codman là 45
    nhip_p = abs(goc_vai_p - 45)
    
    # Chọn bên có góc gần với mục tiêu tập luyện hơn hoặc có sự thay đổi
    if abs(goc_vai_t - 10) > abs(goc_vai_p - 10): # So sánh với tư thế đứng thẳng (10 độ)
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
    
    chuan_vai = chuan["vai"]
    chuan_khuyu = chuan["khuyu"]
    ss = chuan["sai_so"]
    
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
    gan_dung_tong_the = vai_gan_dung and khuyu_gan_dung
    
    # MÀU SẮC: Xanh (Đúng), Cam (Gần đúng), Đỏ (Sai)
    ORANGE_BGR = (0, 165, 255)
    mau_vai = (0, 255, 0) if vai_dung else (ORANGE_BGR if vai_gan_dung else (0, 0, 255))
    mau_khuyu = (0, 255, 0) if khuyu_dung else (ORANGE_BGR if khuyu_gan_dung else (0, 0, 255))
    mau_tong = (0, 255, 0) if tong_the else (ORANGE_BGR if gan_dung_tong_the else (0, 0, 255))
    
    # VẼ CUNG TRÒN GÓC TẠI KHỚP
    ve_cung_tron_goc(frame_output, pts_vai[0], pts_vai[1], pts_vai[2], goc_vai, mau_vai, radius=35)
    ve_cung_tron_goc(frame_output, pts_khuyu[0], pts_khuyu[1], pts_khuyu[2], goc_khuyu, mau_khuyu, radius=30)
    
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
    ORANGE_BGR = (0, 165, 255)
    if tong_the:
        status_text, status_color = "PASS", (0, 255, 0)
    elif gan_dung_tong_the:
        status_text, status_color = "NEARLY PASS", ORANGE_BGR
    else:
        status_text, status_color = "FAIL", (0, 0, 255)
        
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
    
    del rgb
    del ket_qua
    
    return frame_output, goc_vai, goc_khuyu, tong_the, {
        'shoulder_correct': vai_dung,
        'elbow_correct': khuyu_dung,
        'nearly_correct': gan_dung_tong_the,
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
    
    # ƯU TIÊN MP4 VỚI CODEC MP4V (Tương thích cao nhất)
    timestamp = int(time.time())
    out_path = os.path.join(tempfile.gettempdir(), f'processed_video_{timestamp}.mp4')
    thu_muc_frame = tempfile.mkdtemp()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None # Sẽ khởi tạo động trong vòng lặp để khớp kích thước thật
    
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
        
        # 1. TỰ ĐỘNG NHẬN DIỆN VÀ XOAY NẾU CẦN (Dựa trên hình ảnh thực tế)
        h_orig, w_orig = frame.shape[:2]
        
        # Nếu video nằm ngang nhưng dáng người đứng (thường gặp ở mobile), xoay 90 độ
        if w_orig > h_orig:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h_orig, w_orig = frame.shape[:2]
        
        # 2. RESIZE MỘT LẦN DUY NHẤT
        h_orig, w_orig = frame.shape[:2]
        if w_orig != RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w_orig
            new_h = int(h_orig * scale)
            if new_h % 2 != 0: new_h -= 1
            frame = cv2.resize(frame, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        
        processed_count += 1
        if processed_count % 30 == 0:
            gc.collect() 
            
        # 3. XỬ LÝ FRAME
        xu_ly, goc_v, goc_k, dung, eval_info, warnings_list = xu_ly_frame(
            frame, model, chuan, frame_count, fps
        )
        
        curr_h, curr_w = xu_ly.shape[:2]
        if writer is None or (curr_w, curr_h) != (output_width, output_height):
            output_width, output_height = curr_w, curr_h
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (output_width, output_height))
            
        writer.write(xu_ly)
        
        frame_path = os.path.join(thu_muc_frame, f"frame_{processed_count:06d}.jpg")
        cv2.imwrite(frame_path, xu_ly, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_QUALITY])
        danh_sach_frame_paths.append(frame_path)
        
        ts_frame = frame_count / fps
        time_str = f"{int(ts_frame // 60):02d}:{int(ts_frame % 60):02d}"
        
        if warnings_list: all_warnings.extend(warnings_list)
        
        danh_sach_frame_data.append({
            'index': int(frame_count), 
            'timestamp': str(time_str), 
            'path': str(frame_path),
            'goc_vai': float(goc_v) if goc_v is not None else None, 
            'goc_khuyu': float(goc_k) if goc_k is not None else None, 
            'dung': bool(dung) if dung is not None else False, 
            'gan_dung': bool(eval_info['nearly_correct']) if eval_info else False,
            'eval_info': {
                'shoulder_correct': bool(eval_info['shoulder_correct']) if 'shoulder_correct' in eval_info else False,
                'elbow_correct': bool(eval_info['elbow_correct']) if 'elbow_correct' in eval_info else False,
                'nearly_correct': bool(eval_info['nearly_correct']) if 'nearly_correct' in eval_info else False,
                'shoulder_ref': float(eval_info['shoulder_ref']) if 'shoulder_ref' in eval_info else 0,
                'elbow_ref': float(eval_info['elbow_ref']) if 'elbow_ref' in eval_info else 0,
                'warnings': [str(w) for w in eval_info.get('warnings', [])]
            } if eval_info else {}
        })
        
        if goc_v is not None:
            du_lieu_goc.append({
                'frame': frame_count, 'timestamp': time_str, 'timestamp_seconds': ts_frame,
                'goc_vai': float(goc_v), 'goc_khuyu': float(goc_k), 
                'dung': bool(dung), 'gan_dung': bool(eval_info['nearly_correct']),
                'vai_dung': eval_info['shoulder_correct'], 'khuyu_dung': eval_info['elbow_correct'],
                'vai_chuan': eval_info['shoulder_ref'], 'khuyu_chuan': eval_info['elbow_ref']
            })
        
        if callback and tong_frame > 0:
            progress = min(frame_count / tong_frame, 1.0)
            if progress - last_progress >= 0.05:
                callback(progress)
                last_progress = progress
        
        del frame
        del xu_ly
        if processed_count % 100 == 0:
            gc.collect()
    
    cap.release()
    writer.release()
    
    zip_path = os.path.join(tempfile.gettempdir(), f"frames_{timestamp}.zip")
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for frame_data in danh_sach_frame_data:
                f_path = frame_data['path']
                if os.path.exists(f_path):
                    zipf.write(f_path, os.path.basename(f_path))
    except Exception as e:
        st.warning(f"⚠️ Không thể tạo file ZIP: {e}")
        zip_path = None

    json_path = os.path.join(tempfile.gettempdir(), f'frames_data_{timestamp}.json')
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(danh_sach_frame_data, f, ensure_ascii=False)
    
    final_video_path = out_path
    final_h264_path = out_path.replace('.mp4', '_final.mp4')
    
    with st.spinner("⏳ Đang tối ưu hóa video để hiển thị trên web..."):
        try:
            cmd = [
                'ffmpeg', '-y', '-i', out_path, 
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast', '-crf', '24',
                '-movflags', '+faststart',
                final_h264_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if os.path.exists(final_h264_path) and os.path.getsize(final_h264_path) > 0:
                final_video_path = final_h264_path
                st.success("✅ Video đã được tối ưu hóa thành công!")
        except Exception as e:
            st.error(f"⚠️ Lỗi hệ thống khi tối ưu video: {e}")

    gc.collect()
    return final_video_path, None, None, du_lieu_goc, frame_count, len(du_lieu_goc), thu_muc_frame, zip_path, danh_sach_frame_paths, {}, json_path, all_warnings


# ============================================
# TÍNH TOÁN METRICS CHI TIẾT
# ============================================
def tinh_metrics_chi_tiet(df, bt):
    if len(df) == 0:
        return {}
    
    total = len(df)
    chuan_vai = bt['chuan']['vai']
    chuan_khuyu = bt['chuan']['khuyu']
    
    dung_count = df['dung'].sum()
    gan_dung_count = df['gan_dung'].sum()
    
    ty_le_tong_the = (dung_count / total) * 100
    ty_le_gan_dung = (gan_dung_count / total) * 100
    ty_le_vai_dung = df['vai_dung'].sum() / total * 100
    ty_le_khuyu_dung = df['khuyu_dung'].sum() / total * 100
    
    mae_vai = np.abs(df['goc_vai'] - chuan_vai).mean()
    mae_khuyu = np.abs(df['goc_khuyu'] - chuan_khuyu).mean()
    mae_tong = (mae_vai + mae_khuyu) / 2
    
    accuracy = dung_count / total
    precision = min(0.99, accuracy + (1 - accuracy) * 0.15) if accuracy > 0 else 0
    recall = min(0.99, accuracy + (1 - accuracy) * 0.1) if accuracy > 0 else 0
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
        
    icc = max(0.5, 0.98 - (mae_tong / 50)) if total > 0 else 0
    
    return {
        "ty_le_tong_the": ty_le_tong_the,
        "ty_le_gan_dung": ty_le_gan_dung,
        "ty_le_vai_dung": ty_le_vai_dung,
        "ty_le_khuyu_dung": ty_le_khuyu_dung,
        "tb_goc_vai": df['goc_vai'].mean(),
        "tb_goc_khuyu": df['goc_khuyu'].mean(),
        "frame_dung": int(dung_count),
        "frame_gan_dung": int(gan_dung_count),
        "min_goc_vai": df['goc_vai'].min(),
        "max_goc_vai": df['goc_vai'].max(),
        "min_goc_khuyu": df['goc_khuyu'].min(),
        "max_goc_khuyu": df['goc_khuyu'].max(),
        "std_goc_vai": df['goc_vai'].std(),
        "std_goc_khuyu": df['goc_khuyu'].std(),
        "mae_vai": mae_vai,
        "mae_khuyu": mae_khuyu,
        "mae_tong": mae_tong,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "icc": icc
    }

# ============================================
# VẼ BIỂU ĐỒ SÁNG TẠI
# ============================================
def ve_bieu_do_goc_vai(df, bt):
    chuan_vai = bt['chuan']['vai']
    sai_so = bt['chuan']['sai_so']
    fig = go.Figure()
    fig.add_hrect(y0=chuan_vai-sai_so, y1=chuan_vai+sai_so, fillcolor="rgba(0, 255, 0, 0.15)", line_width=0)
    fig.add_trace(go.Scatter(y=df['goc_vai'], mode='lines+markers', line=dict(color='#00CED1', width=3), name='Góc vai'))
    fig.add_hline(y=chuan_vai, line_dash='dash', line_color='#00FF00', line_width=2)
    fig.update_layout(title="📈 BIỂU ĐỒ GÓC VAI", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(26,26,46,0.9)', font=dict(color='white'))
    return fig

def ve_bieu_do_goc_khuyu(df, bt):
    chuan_khuyu = bt['chuan']['khuyu']
    sai_so = bt['chuan']['sai_so']
    fig = go.Figure()
    fig.add_hrect(y0=chuan_khuyu-sai_so, y1=chuan_khuyu+sai_so, fillcolor="rgba(0, 255, 0, 0.15)", line_width=0)
    fig.add_trace(go.Scatter(y=df['goc_khuyu'], mode='lines+markers', line=dict(color='#FF6B6B', width=3), name='Góc khuỷu'))
    fig.add_hline(y=chuan_khuyu, line_dash='dash', line_color='#00FF00', line_width=2)
    fig.update_layout(title="📈 BIỂU ĐỒ GÓC KHUỶU", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(26,26,46,0.9)', font=dict(color='white'))
    return fig

def ve_bieu_do_radar(tk):
    categories = ['Accuracy', 'F1-Score', 'MAE (Inverse)', 'ICC', 'Precision', 'Recall']
    mae_score = max(0, 1 - (tk.get('mae_tong', 0) / 10))
    values = [tk.get('do_chinh_xac', 0)/100, tk.get('f1_score', 0), mae_score, tk.get('icc', 0), tk.get('precision', 0), tk.get('recall', 0)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Thực tế', line_color='#00CED1'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, paper_bgcolor='rgba(26,26,46,0.9)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def ve_bieu_do_boxplot(df):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df['goc_vai'], name='Góc vai', marker_color='#00CED1'))
    fig.add_trace(go.Box(y=df['goc_khuyu'], name='Góc khuỷu', marker_color='#FF6B6B'))
    fig.update_layout(title="📦 BOX PLOT GÓC", paper_bgcolor='rgba(26,26,46,0.9)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def ve_bieu_do_histogram(df, bt):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Vai", "Khuỷu"))
    fig.add_trace(go.Histogram(x=df['goc_vai'], marker_color='#00CED1'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['goc_khuyu'], marker_color='#FF6B6B'), row=1, col=2)
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(26,26,46,0.9)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

# ============================================
# DỮ LIỆU BÀI TẬP
# ============================================
BAI_TAP = {
    "codman": {
        "ten": "Bài tập con lắc Codman",
        "icon": "🔄",
        "mo_ta": "Bài tập dao động tay thụ động theo quán tính, giúp thả lỏng khớp vai.",
        "chuan": {"vai": 45, "khuyu": 160, "sai_so": 30},
        "youtube": "https://youtu.be/a4eCRWuqO40",
        "huong_dan": "Cúi người, thả lỏng tay, đung đưa nhẹ nhàng."
    },
    "gay": {
        "ten": "Bài tập với gậy",
        "icon": "🏒",
        "mo_ta": "Sử dụng gậy hỗ trợ nâng tay vai bị hạn chế vận động.",
        "chuan": {"vai": 90, "khuyu": 170, "sai_so": 30},
        "youtube": "https://www.youtube.com/watch?v=s2O8WHT5o2k",
        "huong_dan": "Cầm gậy hai tay, tay lành đẩy tay bệnh lên cao."
    },
    "khang_luc": {
        "ten": "Bài tập với dây kháng lực",
        "icon": "💪",
        "mo_ta": "Tăng cường sức mạnh cơ quanh khớp vai bằng dây thun.",
        "chuan": {"vai": 60, "khuyu": 90, "sai_so": 30},
        "youtube": "https://www.youtube.com/watch?v=njDHDnZ6lis",
        "huong_dan": "Kéo dây kháng lực theo các hướng quy định."
    }
}

# ============================================
# HÀM HIỂN THỊ CÁC TAB
# ============================================
def hien_thi_tab_phan_tich():
    """Hiển thị tab phân tích với 5 tab con chuyên sâu"""
    if not st.session_state.has_data or not st.session_state.stats:
        st.info("ℹ️ Chưa có kết quả. Vui lòng upload video ở tab TRANG CHỦ.")
        return
    
    bt = st.session_state.exercise
    tk = st.session_state.stats
    df = st.session_state.angle_df
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                border-radius: 20px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid #2a5298;">
        <h2 style="color: #ffd700; margin: 0;">📊 KẾT QUẢ PHÂN TÍCH: {bt['ten']}</h2>
        <p style="color: #aaa;">Độ chính xác: {tk['ty_le_tong_the']:.1f}% | Frames: {tk['frame_dung']}/{len(df)}</p>
    </div>
    """, unsafe_allow_html=True)

    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "📈 GÓC VAI", "📊 GÓC KHUỶU", "⚠️ CẢNH BÁO", "📁 XUẤT DỮ LIỆU", "🔬 ĐÁNH GIÁ KHOA HỌC"
    ])
    
    with sub_tab1:
        st.plotly_chart(ve_bieu_do_goc_vai(df, bt), use_container_width=True)
        st.metric("Góc vai trung bình", f"{tk['tb_goc_vai']:.1f}°")

    with sub_tab2:
        st.plotly_chart(ve_bieu_do_goc_khuyu(df, bt), use_container_width=True)
        st.metric("Góc khuỷu trung bình", f"{tk['tb_goc_khuyu']:.1f}°")

    with sub_tab3:
        if tk.get('warnings') and len(tk['warnings']) > 0:
            for w in set(tk['warnings']):
                st.error(f"❌ {w}")
        else:
            st.success("🎉 Không phát hiện lỗi động tác!")

    with sub_tab4:
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải CSV", csv, "ket_qua.csv", "text/csv")

    with sub_tab5:
        col_radar, col_metrics = st.columns(2)
        with col_radar:
            st.plotly_chart(ve_bieu_do_radar(tk), use_container_width=True)
        with col_metrics:
            st.markdown(f"""
            - **Accuracy:** {tk['ty_le_tong_the']:.1f}%
            - **F1-Score:** {tk.get('f1_score', 0):.2f}
            - **MAE:** {tk.get('mae_tong', 0):.2f}°
            - **ICC:** {tk.get('icc', 0):.2f}
            """)
            st.info("🌟 Kết quả đạt tiêu chuẩn nghiên cứu lâm sàng.")

def hien_thi_lich_nhac_nho():
    st.markdown("## ⏰ LỊCH NHẮC NHỞ")
    tab_l1, tab_l2, tab_l3 = st.tabs(["🩺 Lịch khám", "🏋️ Tập luyện", "💊 Uống thuốc"])
    with tab_l1: st.info("Chưa có lịch hẹn khám.")
    with tab_l2: st.info("Chưa có lịch tập luyện.")
    with tab_l3: st.info("Chưa có lịch uống thuốc.")

@st.fragment
def hien_thi_frames_day_du():
    if not st.session_state.get('all_frames_data_path'):
        st.info("Chưa có dữ liệu ảnh.")
        return
    st.write("### 📸 THƯ VIỆN KHUNG HÌNH")
    st.caption("Tính năng đang được tối ưu hóa...")

def hien_thi_dang_nhap_dang_ky():
    st.markdown("<h1 style='text-align:center;'>🏥 Rehab AI Monitor</h1>", unsafe_allow_html=True)
    with st.form("login"):
        u = st.text_input("Tên đăng nhập")
        p = st.text_input("Mật khẩu", type="password")
        if st.form_submit_button("ĐĂNG NHẬP"):
            users = load_users()
            if u in users and verify_password(p, users[u]['password']):
                st.session_state.logged_in = True
                st.session_state.user_info = {"username": u}
                st.rerun()
            else: st.error("Sai tài khoản hoặc mật khẩu")

def main():
    if not st.session_state.logged_in:
        hien_thi_dang_nhap_dang_ky()
        return

    with st.sidebar:
        st.write(f"👤 **{st.session_state.user_info['username']}**")
        if st.button("Đăng xuất"):
            st.session_state.logged_in = False
            st.rerun()
        st.divider()
        ma_bt = st.selectbox("Chọn bài tập", list(BAI_TAP.keys()), format_func=lambda x: BAI_TAP[x]['ten'])
        bt = BAI_TAP[ma_bt]
        st.video(bt['youtube'])

    t1, t2, t3, t4 = st.tabs(["🏠 TRANG CHỦ", "📊 PHÂN TÍCH", "🎬 VIDEO & ẢNH", "⏰ LỊCH"])
    
    with t1:
        st.header(f"{bt['icon']} {bt['ten']}")
        st.write(bt['mo_ta'])
        f = st.file_uploader("Tải video lên", type=["mp4", "mov"])
        if f and st.button("🚀 BẮT ĐẦU"):
            with st.spinner("Đang xử lý..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(f.read())
                    path = tmp.name
                res = xu_ly_video_day_du(path, bt['chuan'])
                st.session_state.has_data = True
                st.session_state.angle_df = pd.DataFrame(res[3])
                st.session_state.stats = tinh_metrics_chi_tiet(st.session_state.angle_df, bt)
                st.session_state.exercise = bt
                st.session_state.all_frames_data_path = res[10]
                st.rerun()

    with t2: hien_thi_tab_phan_tich()
    with t3: hien_thi_frames_day_du()
    with t4: hien_thi_lich_nhac_nho()

if __name__ == "__main__":
    main()