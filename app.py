# -*- coding: utf-8 -*-
# Trigger HF Sync: 2026-05-29
import os
import sys

import math
import json
import base64
import secrets
import logging as _logging
from app_startup import (
    app_startup,
    configure_logging_filters,
    configure_streamlit_page_if_running,
    get_compute_thread_count,
    load_startup_config,
)

_compute_threads = get_compute_thread_count()
MAX_FFMPEG_THREADS = int(os.environ.get("MAX_FFMPEG_THREADS", "2"))


def _chan_log_fragment_spam():
    configure_logging_filters(wrap_streams=True)

import streamlit as st

_PAGE_CONFIGURED_EARLY = configure_streamlit_page_if_running(st)


class _LazyCV2:
    """Lazy-load OpenCV chỉ khi lần đầu tiên gọi cv2.anything() — tránh chậm cold start."""
    __slots__ = ("_mod",)
    def __init__(self): object.__setattr__(self, "_mod", None)
    def _load(self):
        import cv2 as _m
        # Giới hạn thread OpenCV — tránh chiếm hết core khi vẽ/resize frame ở Pass 2,
        # giữ cho luồng UI Streamlit luôn còn CPU để phản hồi (không bị đơ).
        try:
            _m.setNumThreads(_compute_threads)
        except Exception:
            pass
        object.__setattr__(self, "_mod", _m)
        return _m
    def __getattr__(self, k): return getattr(object.__getattribute__(self,"_mod") or self._load(), k)
    def __setattr__(self, k, v):
        m = object.__getattribute__(self,"_mod") or self._load()
        setattr(m, k, v)
cv2 = _LazyCV2()

import numpy as np
import pandas as pd
import tempfile
import time
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import warnings
import zipfile
from io import BytesIO
import subprocess
import hashlib
import gc
import html
from types import SimpleNamespace

try:
    from utils.pose_classifier_utils import (
        apply_classifier_to_dataframe,
        create_pose_classifier_predictor,
        draw_ml_badge,
        draw_rule_badge,
        ensure_classifier_ready,
        format_ml_display,
        get_pose_classifier_status,
        merge_ml_metrics,
        refresh_saved_frame_labels,
        reprocess_videos_with_classifier,
        train_pose_classifier,
    )
    POSE_CLASSIFIER_IMPORT_ERROR = None
except Exception as _pose_classifier_import_error:
    apply_classifier_to_dataframe = None
    create_pose_classifier_predictor = None
    draw_ml_badge = None
    draw_rule_badge = None
    ensure_classifier_ready = None
    format_ml_display = None
    get_pose_classifier_status = None
    merge_ml_metrics = None
    refresh_saved_frame_labels = None
    reprocess_videos_with_classifier = None
    train_pose_classifier = None
    POSE_CLASSIFIER_IMPORT_ERROR = _pose_classifier_import_error

from utils.reference_utils import (
    NEAR_ERROR_MULTIPLIER,
    PHASE_ERROR,
    PHASE_ERROR_DEFAULT,
    PHASE_UI_LABELS,
    PHASE_UI_SHORT,
    detect_motion_subtype,
    find_closest_reference_pose,
    get_phase_error_for_segment,
    load_reference_poses,
    normalize_phase_selection,
    phase_frame_label,
    resolve_reference_file,
)
from utils.checkpoint_utils import (
    CHECKPOINT_INTERVAL_PASS2,
    assemble_video_from_jpgs,
    build_config_hash,
    checkpoint_ui_progress,
    clear_checkpoint,
    deserialize_pass1_item,
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    serialize_pass1_item,
)
from cloud.hf_sync import (
    HF_JSON_DOWNLOAD_FILES,
    HF_MODEL_ARTIFACT_FILES,
    HfPathPolicy,
    data_allowed_roots,
    dataset_rel_path_from_local,
    dataset_file_exists,
    download_dataset_file,
    download_dataset_file_with_progress as hf_download_dataset_file_with_progress,
    download_dataset_file_via_http,
    ensure_dataset_repo,
    hf_download_target_for_rel_path,
    hf_min_size_for_path,
    hf_repo_info,
    hf_token_fingerprint,
    hf_upload_rel_path_for_local,
    is_hf_auth_error,
    is_hf_library_error,
    is_hf_not_found_error,
    list_dataset_files,
    upload_dataset_file,
    verify_dataset_via_http,
)
from auth.accounts import (
    find_user_key,
    find_user_key_by_email,
    find_user_uniqueness_issues,
    normalize_auth_text,
    roles_match,
)
from auth.passwords import (
    password_record_update,
    verify_password_record,
)
from auth.permissions import (
    actor_can_access_patient as permission_actor_can_access_patient,
    patient_display_label_for_actor,
    require_actor_role,
    researcher_view_records_for_actor,
    scope_patient_usernames_for_actor as permission_scope_patient_usernames_for_actor,
    scope_records_for_actor as permission_scope_records_for_actor,
)
from auth.sessions import (
    bump_global_session_version,
    get_global_session_version,
    session_is_current,
)
from frontend.api_client import (
    FrontendApiClient,
    FrontendApiConfig,
    FrontendApiError,
)
from models.schemas import (
    ADMIN_ROLE,
    DOCTOR_ROLE,
    PATIENT_ROLE,
    RESEARCHER_ROLE,
)
from storage.app_json import (
    format_schema_issue_lines,
    normalize_app_json,
    read_app_json,
    update_app_json,
    write_app_json,
)
from ui.admin import render_admin_sidebar, render_admin_tab
from ui.admin_pages import render_admin_home_page, render_admin_management_page
from ui.doctor import render_doctor_sidebar, render_doctor_tab
from ui.doctor_forms import (
    render_doctor_evaluation_form as render_doctor_evaluation_form_page,
    render_latest_results_and_history as render_latest_results_and_history_page,
    render_selected_results_tab as render_selected_results_tab_page,
)
from ui.navigation import (
    render_tab_selector,
    sync_active_tab_state,
    tab_titles_for_role,
)
from ui.patient import render_patient_sidebar, render_patient_tab
from ui.researcher import render_researcher_sidebar, render_researcher_tab
from ui.reminders import render_reminders_page
from ui.research_forms import render_research_form_page
from ui.styles import inject_base_css
from ui.frames_viewer import render_frames_full as render_frames_full_page
from ui.analysis_tab import (
    render_analysis_tab as render_analysis_tab_page,
    render_deep_analysis_area as render_deep_analysis_area_page,
)
from ui.video_list import render_video_list_fragment as render_video_list_fragment_page
from ui.layout import (
    _hien_thi_header_chinh,
    hien_thi_footer_chung,
)
from ui.static_pages import (
    hien_thi_tab_cong_nghe,
    hien_thi_tab_huong_dan,
    hien_thi_tab_kien_thuc_phcn,
    hien_thi_tab_lien_he,
    hien_thi_tab_nckh,
    hien_thi_tab_nckh_va_thanh_vien_ncv,
    hien_thi_tab_thanh_vien,
    hien_thi_tab_thong_tin_nghien_cuu,
    hien_thi_tab_thong_tin_tong_hop,
    hien_thi_tab_thong_tin_tong_hop_benh_nhan,
)
from utils.path_security import (
    PathSecurityError,
    normalize_relative_path,
    path_is_within,
    relative_to_allowed_root,
    safe_data_path,
)
from video.validation import (
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_UPLOAD_VIDEO_EXTENSIONS,
    sanitize_filename,
    validate_upload_metadata,
    validate_video_file_for_processing as validate_video_file_for_processing_core,
)
from video.io import (
    build_async_h264_command,
    build_background_upload_h264_command,
    build_cut_segment_command,
    build_ffmpeg_version_command,
    build_frame_extract_command,
    build_h264_transcode_command,
    build_mov_to_mp4_command,
    build_upload_h264_command,
    ffprobe_video_codecs,
    ffprobe_video_duration_text,
    ffprobe_video_has_readable_duration,
    final_h264_path as video_final_h264_path,
    is_non_playable_video_artifact,
    mov_to_mp4_path,
    safe_extract_frames_zip,
    temp_h264_path,
    video_fallback_paths_for,
)
from video.jobs import (
    AnalysisJobRegistry,
    start_background_analysis,
)
from video.serving import (
    ALLOWED_VIDEO_EXTENSIONS,
    allowed_media_file_path,
    build_video_media_url,
    cleanup_media_tokens,
    is_allowed_video_origin,
    media_token_from_request_path,
    path_is_within,
    register_media_token,
    resolve_media_token,
    safe_realpath,
    video_media_allowed_roots,
)


def safe_html(value, max_length=None):
    """Escape untrusted values before inserting them into unsafe_allow_html templates."""
    text = "" if value is None else str(value)
    if max_length is not None and len(text) > max_length:
        text = text[:max_length] + "..."
    return html.escape(text, quote=True)


def safe_attr(value, max_length=None):
    return safe_html(value, max_length=max_length)


def get_clean_rel_path(path):
    """Lấy đường dẫn tương đối sạch của file đối với DATA_DIR,
    độc lập với hệ điều hành và việc path là tuyệt đối hay tương đối."""
    if not path:
        return ""
    try:
        return _dataset_rel_path_from_local(path)
    except Exception:
        p = str(path).replace("\\", "/")
        for folder in ["patient_uploads", "processed_results"]:
            idx = p.find(folder)
            if idx != -1:
                try:
                    return normalize_relative_path(p[idx:])
                except PathSecurityError:
                    return ""
        try:
            return normalize_relative_path(os.path.basename(str(path)))
        except PathSecurityError:
            return ""


def _la_duong_dan_video_gia(path):
    """Chan artifact frame/CSV/ZIP bi nham thanh video can tai/phat."""
    return is_non_playable_video_artifact(path)


def get_final_h264_path(video_path):
    """Trả về đường dẫn tệp H264 đích (_f.mp4) tương ứng một cách chuẩn xác, độc lập với định dạng/cú pháp phần mở rộng gốc."""
    return video_final_h264_path(video_path)


def video_fallback_paths(file_path):
    """Các đường dẫn video có thể tồn tại trên Dataset (H.264 _f.mp4 hoặc bản gốc .mp4)."""
    return video_fallback_paths_for(file_path, local_frame_path_resolver=get_local_frame_path)


def _is_scratch_video_path(path):
    """File tạm transcode — không phát được (màn đen/xám trên trình duyệt)."""
    if not path:
        return False
    low = str(path).replace("\\", "/").lower()
    return any(
        tag in low
        for tag in ("_ftmp.mp4", "_ttmp.mp4", "_ffmp.mp4", ".ftmp.mp4", "/transcode_error")
    )


def _strip_to_original_upload(path):
    """Đưa path về file upload gốc, bỏ hậu tố transcode tạm."""
    if not path:
        return path
    p = str(path)
    for suffix in ("_ftmp.mp4", "_ttmp.mp4", "_ffmp.mp4", "_f.mp4"):
        if p.endswith(suffix):
            return p[: -len(suffix)] + ".mp4"
    return p


def video_raw_only_paths(file_path):
    """Chỉ video gốc BN upload — không fallback sang processed/_f/_ftmp."""
    if not file_path:
        return []
    try:
        norm = get_local_frame_path(file_path) or file_path
    except Exception:
        norm = file_path
    candidates = []
    for p in (norm, _strip_to_original_upload(norm)):
        if p and not _is_scratch_video_path(p):
            candidates.append(p)
        base, ext = os.path.splitext(_strip_to_original_upload(p or ""))
        if base and ext.lower() == ".mp4":
            mov = base + ".mov"
            if not _is_scratch_video_path(mov):
                candidates.append(get_local_frame_path(mov) or mov)
    seen, out = set(), []
    for p in candidates:
        if p and p not in seen and not _is_scratch_video_path(p):
            seen.add(p)
            out.append(p)
    return out


def _tim_video_upload_goc(v):
    """Tìm file upload gốc trong patient_uploads theo BN + tên video."""
    uname = (v or {}).get("username") or ""
    vname = (v or {}).get("video_name") or ""
    if not uname:
        return None
    stem = os.path.splitext(os.path.basename(vname))[0] if vname else ""
    if not os.path.isdir(UPLOAD_DIR):
        return None
    best = None
    best_mtime = 0
    try:
        for fn in os.listdir(UPLOAD_DIR):
            low = fn.lower()
            if not low.endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            if _is_scratch_video_path(fn):
                continue
            if uname not in fn:
                continue
            if stem and stem not in fn and os.path.splitext(vname)[0] not in fn:
                continue
            fp = os.path.join(UPLOAD_DIR, fn)
            try:
                mt = os.path.getmtime(fp)
            except Exception:
                mt = 0
            if mt >= best_mtime:
                best_mtime = mt
                best = fp
    except Exception:
        return None
    return best


def _valid_raw_video_local(path):
    if not path or _is_scratch_video_path(path):
        return False
    if not is_local_file_ready(path):
        return False
    try:
        mtime, size = os.path.getmtime(path), os.path.getsize(path)
        return _check_video_valid_cached(path, mtime, size)
    except Exception:
        return is_local_file_ready(path)


def find_ready_local_video(file_path, min_size=5 * 1024):
    """Trả về đường dẫn video local hợp lệ đầu tiên trong danh sách fallback."""
    best_fallback = None
    for p in video_fallback_paths(file_path):
        if is_local_file_ready(p, min_size=min_size):
            try:
                mtime, size = os.path.getmtime(p), os.path.getsize(p)
                if _check_video_valid_cached(p, mtime, size):
                    return p
                # File exists với kích thước đủ nhưng validation chưa chắc chắn
                # (H.264 thường báo 0 FRAME_COUNT) — giữ làm fallback
                if best_fallback is None:
                    best_fallback = p
            except Exception:
                if best_fallback is None:
                    best_fallback = p
    return best_fallback


def sync_transcode_to_h264(src_path, dst_path=None, audio_path=None, timeout=1800, on_tick=None):
    """Chuyển video sang H.264 MP4 (faststart). Ghi file tạm rồi đổi tên atomic để tránh file hỏng."""
    if not src_path or not os.path.exists(src_path):
        return None
    if dst_path is None:
        dst_path = get_final_h264_path(src_path)
    has_audio_mux = bool(audio_path and os.path.exists(audio_path))
    if os.path.exists(dst_path) and not has_audio_mux:
        try:
            mtime, size = os.path.getmtime(dst_path), os.path.getsize(dst_path)
            if _check_video_valid_cached(dst_path, mtime, size):
                v_codec, _ = get_video_codec(dst_path)
                if v_codec == 'h264':
                    return dst_path
        except Exception:
            pass
    elif has_audio_mux and os.path.exists(dst_path):
        # Luôn re-mux khi có audio mới — tránh dùng cache _f.mp4 không có tiếng
        try:
            os.remove(dst_path)
        except Exception:
            pass
    tmp_dst = temp_h264_path(dst_path)
    for f_clean in (dst_path, tmp_dst):
        if os.path.exists(f_clean):
            try:
                os.remove(f_clean)
            except Exception:
                pass
    cmd = build_h264_transcode_command(
        src_path,
        tmp_dst,
        audio_path=audio_path,
        audio_exists=has_audio_mux,
        ffmpeg_threads=MAX_FFMPEG_THREADS,
    )
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + timeout
        last_tick = time.time()
        while process.poll() is None:
            if time.time() > deadline:
                process.kill()
                print("[Transcode] FFmpeg timeout")
                if os.path.exists(tmp_dst):
                    try:
                        os.remove(tmp_dst)
                    except Exception:
                        pass
                return None
            if on_tick and time.time() - last_tick >= 2.0:
                try:
                    on_tick()
                except Exception:
                    pass
                last_tick = time.time()
            time.sleep(0.5)
        stderr_out = ""
        try:
            _, stderr_out = process.communicate(timeout=10)
        except Exception:
            pass
        if process.returncode != 0:
            print(f"[Transcode] FFmpeg fail ({process.returncode}): {(stderr_out or '')[-800:]}")
            if os.path.exists(tmp_dst):
                try:
                    os.remove(tmp_dst)
                except Exception:
                    pass
            return None
        if not os.path.exists(tmp_dst) or os.path.getsize(tmp_dst) < 5 * 1024:
            return None
        mtime_f, size_f = os.path.getmtime(tmp_dst), os.path.getsize(tmp_dst)
        if not _check_video_valid_cached(tmp_dst, mtime_f, size_f):
            try:
                os.remove(tmp_dst)
            except Exception:
                pass
            return None
        os.replace(tmp_dst, dst_path)
        return dst_path
    except Exception as transcode_err:
        print(f"[Transcode] Error: {transcode_err}")
        if os.path.exists(tmp_dst):
            try:
                os.remove(tmp_dst)
            except Exception:
                pass
        return None


def resolve_playback_video_path(video_path, sync_transcode=False):
    """Trả về đường dẫn video phát/tải được (ưu tiên H.264 _f.mp4 hợp lệ)."""
    if not video_path:
        return video_path
    final_h264 = get_final_h264_path(video_path)
    if os.path.exists(final_h264):
        try:
            mtime, size = os.path.getmtime(final_h264), os.path.getsize(final_h264)
            if _check_video_valid_cached(final_h264, mtime, size):
                return final_h264
        except Exception:
            pass
    raw_path = video_path.replace('_f.mp4', '.mp4') if video_path.endswith('_f.mp4') else video_path
    if sync_transcode:
        src = raw_path if os.path.exists(raw_path) else video_path
        if not os.path.exists(src):
            try:
                ensure_local_file(src)
            except Exception:
                pass
        if os.path.exists(src):
            out = sync_transcode_to_h264(src, final_h264)
            if out:
                return out
    if os.path.exists(video_path) and video_path.lower().endswith('.mp4'):
        try:
            v_codec, _ = get_video_codec(video_path)
            if v_codec == 'h264':
                mtime, size = os.path.getmtime(video_path), os.path.getsize(video_path)
                if _check_video_valid_cached(video_path, mtime, size):
                    return video_path
        except Exception:
            pass
    return video_path


def _chuan_hoa_ten_video(name):
    """Chuẩn hóa tên file video để khớp .mp4/.mov và khoảng trắng thừa."""
    import re
    s = re.sub(r"\s+", " ", str(name or "").strip())
    low = s.lower()
    for ext in (".mov", ".mp4", ".avi", ".mkv", ".webm"):
        if low.endswith(ext):
            s = s[: -len(ext)].strip()
            break
    return s.lower()


def _normalize_video_key(username, video_name, exercise):
    return (
        str(username or "").strip(),
        _chuan_hoa_ten_video(video_name),
        str(exercise or "").strip(),
    )


def _format_vn_time(time_str, default="N/A"):
    """Định dạng thống nhất: HH:MM - dd/mm/YYYY."""
    dt = _parse_vn_datetime(time_str)
    if dt:
        return dt.strftime("%H:%M - %d/%m/%Y")
    return str(time_str).strip() if time_str else default


def _lay_epoch_tu_processed(path):
    """Trích epoch từ đường dẫn processed_1234567890."""
    if not path:
        return None
    import re
    m = re.search(r"processed_(\d+)", str(path))
    return int(m.group(1)) if m else None


def _lay_thoi_gian_phan_tich_on_dinh(v, ai_eval=None):
    """Thời gian phân tích AI ổn định — không lẫn với thời gian upload."""
    if ai_eval and ai_eval.get("time"):
        return _format_vn_time(ai_eval.get("time"))
    proc_ts = _lay_epoch_tu_processed(v.get("processed_path") if v else None)
    if proc_ts:
        try:
            return datetime.fromtimestamp(proc_ts).strftime("%H:%M - %d/%m/%Y")
        except (OSError, OverflowError, ValueError):
            pass
    return None


def _lay_thoi_gian_phan_tich_moi_nhat_bn(evals, username):
    """Thời gian phân tích AI mới nhất của một BN (định dạng thống nhất)."""
    best_t = datetime.min
    best_str = None
    for e in evals or []:
        if e.get("doctor_username") != "AI_Researcher":
            continue
        if e.get("patient_username") != username:
            continue
        t = _parse_vn_datetime(e.get("time")) or datetime.min
        if t >= best_t:
            best_t = t
            best_str = _format_vn_time(e.get("time"))
    return best_str


def _tom_tat_benh_nhan_tu_video(video_list, ai_eval_lookup, ai_eval_by_exercise):
    """Gộp theo BN: tên, số video, thời gian phân tích gần nhất."""
    by_user = {}
    for v in video_list or []:
        u = v.get("username")
        if not u:
            continue
        ev_key = _normalize_video_key(u, v.get("video_name"), v.get("exercise"))
        ai_eval = ai_eval_lookup.get(ev_key) or ai_eval_by_exercise.get((u, v.get("exercise")))
        t_str = _lay_thoi_gian_phan_tich_on_dinh(v, ai_eval)
        t_dt = _parse_vn_datetime(t_str) if t_str else datetime.min
        row = by_user.get(u)
        if not row:
            by_user[u] = {
                "username": u,
                "full_name": v.get("full_name") or u,
                "video_count": 1,
                "last_analysis": t_str,
                "last_dt": t_dt,
            }
            continue
        row["video_count"] += 1
        if t_dt >= row["last_dt"]:
            row["last_dt"] = t_dt
            row["last_analysis"] = t_str
    return sorted(
        by_user.values(),
        key=lambda x: x.get("last_dt") or datetime.min,
        reverse=True,
    )


def _dedup_evaluations(evals):
    """Giữ bản đánh giá mới nhất theo (BN, video, bài tập, người đánh giá)."""
    best = {}
    for e in evals or []:
        key = (
            e.get("patient_username"),
            _chuan_hoa_ten_video(e.get("video_name")),
            e.get("exercise"),
            e.get("doctor_username"),
        )
        t_new = _parse_vn_datetime(e.get("time")) or datetime.min
        if key not in best:
            best[key] = e
            continue
        t_old = _parse_vn_datetime(best[key].get("time")) or datetime.min
        if t_new >= t_old:
            best[key] = e
    result = list(best.values())
    result.sort(
        key=lambda x: _parse_vn_datetime(x.get("time")) or datetime.min,
        reverse=True,
    )
    return result


BN_NGHIEN_CUU = (
    "Cao Thị Thường",
    "Hoàng Hạnh Nguyên",
    "Nguyễn Thị Nga",
    "Vũ Thị Hòa",
)
BAI_NGHIEN_CUU = (
    "Bài tập con lắc Codman",
    "Bài tập với gậy (Pulley Exercise)",
)


def _slot_nghien_cuu_key(username, exercise):
    return (str(username or "").strip(), str(exercise or "").strip())


def _lay_khoa_video_da_danh_gia_bac_si(evals, patient_username=None):
    """8 slot nghiên cứu (4 BN × 2 bài tập) đã có nhận xét bác sĩ/KTV."""
    slots = set()
    for pu in BN_NGHIEN_CUU:
        if patient_username and pu != patient_username:
            continue
        for ex in BAI_NGHIEN_CUU:
            if _lay_eval_moi_nhat_theo_bai_tap(evals, pu, ex):
                slots.add(_slot_nghien_cuu_key(pu, ex))
    return slots


def _loc_bo_trung_video_danh_sach(videos):
    """Mỗi BN + video + bài tập chỉ giữ một bản ghi."""
    seen = set()
    out = []
    for v in videos or []:
        key = _normalize_video_key(v.get("username"), v.get("video_name"), v.get("exercise"))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _lay_thoi_gian_upload_video(v):
    """Thời gian BN upload mới nhất — ưu tiên video_list.time, rồi parse tên file."""
    t_json = _format_vn_time(v.get("time"), default="")
    if t_json and t_json != "N/A":
        return t_json
    t = _parse_upload_time_from_filename(v.get("video_path") or v.get("video_name"))
    return t or "N/A"


def _lay_eval_moi_nhat_theo_bai_tap(evals, username, exercise, doctor_username=None):
    """Lấy đánh giá mới nhất theo BN + bài tập (bỏ qua lệch tên file)."""
    best = None
    best_t = datetime.min
    for e in evals or []:
        if e.get("patient_username") != username:
            continue
        if exercise and e.get("exercise") != exercise:
            continue
        if doctor_username is not None and e.get("doctor_username") != doctor_username:
            continue
        if doctor_username is None and e.get("doctor_username") == "AI_Researcher":
            continue
        t = _parse_vn_datetime(e.get("time")) or datetime.min
        if t >= best_t:
            best_t = t
            best = e
    return best


def _lay_danh_gia_cho_video(selected_v, evals=None):
    """Lấy đánh giá NCV (AI) và Bác sĩ mới nhất — khớp BN+bài tập, ưu tiên đúng tên file."""
    if not selected_v:
        return None, None
    evals = evals if evals is not None else _dedup_evaluations(load_data(EVALUATIONS_FILE))
    pu = selected_v.get("username") or selected_v.get("patient_username")
    ex = selected_v.get("exercise")
    vn = selected_v.get("video_name")
    ai_eval = _lay_eval_moi_nhat_theo_bai_tap(evals, pu, ex, doctor_username="AI_Researcher")
    doc_eval = _lay_eval_moi_nhat_theo_bai_tap(evals, pu, ex)
    if vn:
        sel_key = _normalize_video_key(pu, vn, ex)
        for e in evals:
            if _normalize_video_key(
                e.get("patient_username"), e.get("video_name"), e.get("exercise")
            ) != sel_key:
                continue
            if e.get("doctor_username") == "AI_Researcher":
                ai_eval = e
            else:
                doc_eval = e
    return ai_eval, doc_eval


def _lay_do_chinh_xac_hien_thi(v, ai_eval=None):
    """Độ chính xác hiển thị — ưu tiên báo cáo AI mới nhất, không dùng do_chinh_xac cũ trong video_list."""
    if ai_eval and ai_eval.get("ai_accuracy") is not None:
        try:
            return float(ai_eval["ai_accuracy"])
        except (TypeError, ValueError):
            pass
    metrics = v.get("metrics") if isinstance(v.get("metrics"), dict) else {}
    mg2 = metrics.get("metrics_g2")
    if isinstance(mg2, dict) and mg2.get("do_chinh_xac") is not None:
        try:
            return float(mg2["do_chinh_xac"])
        except (TypeError, ValueError):
            pass
    for fld in ("do_chinh_xac", "ty_le_tong_the"):
        if metrics.get(fld) is not None:
            try:
                return float(metrics[fld])
            except (TypeError, ValueError):
                pass
    try:
        return float(v.get("accuracy") or 0)
    except (TypeError, ValueError):
        return 0.0


def _ap_dung_ket_qua_moi_nhat_vao_video(v, ai_eval=None):
    """Gắn kết quả AI mới nhất vào bản ghi video — giữ nguyên thời gian upload."""
    if not v:
        return v
    out = dict(v)
    upload_time = out.get("time")
    if ai_eval:
        acc = ai_eval.get("ai_accuracy")
        metrics = dict(out.get("metrics") or {}) if isinstance(out.get("metrics"), dict) else {}
        if acc is not None:
            out["accuracy"] = acc
            metrics["do_chinh_xac"] = acc
            if metrics.get("ty_le_tong_the") is not None:
                metrics["ty_le_tong_the"] = acc
        for g, fld in (("g1", "ai_accuracy_g1"), ("g2", "ai_accuracy_g2"), ("g3", "ai_accuracy_g3")):
            val = ai_eval.get(fld)
            if val is None:
                continue
            mg = metrics.get(f"metrics_{g}")
            if not isinstance(mg, dict):
                mg = {}
            mg["do_chinh_xac"] = val
            metrics[f"metrics_{g}"] = mg
        if metrics:
            out["metrics"] = metrics
    if upload_time is not None:
        out["time"] = upload_time
    return out


def _chon_video_moi_hon(a, b):
    """Chọn bản ghi video có thời gian upload mới hơn."""
    t_a = _parse_vn_datetime(_lay_thoi_gian_upload_video(a)) or datetime.min
    t_b = _parse_vn_datetime(_lay_thoi_gian_upload_video(b)) or datetime.min
    if t_a >= t_b:
        return dict(a)
    return dict(b)


def _lay_thoi_gian_phan_tich_hien_thi(v, ai_eval=None):
    """Thời gian hiển thị mục đã phân tích — chỉ từ AI eval / processed, không dùng upload."""
    t = _lay_thoi_gian_phan_tich_on_dinh(v, ai_eval)
    return t if t else "Chưa phân tích"


def _lay_trang_thai_video_danh_sach(v, ai_eval=None, doc_eval=None, user_role=None):
    """Nhãn trạng thái danh sách video — thời gian khớp tab Kết quả đánh giá."""
    if doc_eval:
        return f"Đã đánh giá ({_format_vn_time(doc_eval.get('time'), default='N/A')})"
    if ai_eval and ai_eval.get("time"):
        return f"Đã phân tích ({_format_vn_time(ai_eval.get('time'), default='N/A')})"
    if v.get("status") == "Đã phân tích":
        t_on_dinh = _lay_thoi_gian_phan_tich_on_dinh(v, ai_eval)
        return f"Đã phân tích ({t_on_dinh or 'N/A'})"
    if user_role == "Bác sĩ / KTV PHCN":
        return "Đang chờ bác sĩ đánh giá"
    return v.get("status") or "Chờ xử lý"


def _tao_ban_ghi_video_tu_danh_gia(doc_eval, ai_eval, users=None):
    """Tạo bản ghi video_list từ đánh giá khi video_list.json thiếu hoặc lệch khóa."""
    users = users if users is not None else load_users()
    uname = doc_eval.get("patient_username") or ""
    vname = doc_eval.get("video_name") or ""
    ex = doc_eval.get("exercise") or _exercise_tu_ten_file(vname)
    fn = users.get(uname, {}).get("full_name", uname) if isinstance(users, dict) else uname
    vp = _tim_upload_theo_video_name(uname, vname)
    return {
        "username": uname,
        "full_name": fn or uname,
        "video_name": vname,
        "exercise": ex,
        "accuracy": (ai_eval or {}).get("ai_accuracy") or 0,
        "time": _lay_thoi_gian_upload_video({"video_path": vp, "video_name": vname, "time": None}),
        "video_path": vp,
        "processed_path": None,
        "status": "Đã phân tích" if ai_eval else "Đã đánh giá (bác sĩ)",
    }


def _lay_video_nghien_cuu_chinh_thuc(video_list, evals=None):
    """Luôn trả đủ 8 video (4 BN × 2 bài tập) đã có đánh giá bác sĩ — khóa theo BN+bài tập."""
    evals = _dedup_evaluations(evals if evals is not None else load_data(EVALUATIONS_FILE))
    slots = [
        _slot_nghien_cuu_key(pu, ex)
        for pu in BN_NGHIEN_CUU
        for ex in BAI_NGHIEN_CUU
        if _lay_eval_moi_nhat_theo_bai_tap(evals, pu, ex)
    ]
    if not slots:
        return []

    valid_slots = {_slot_nghien_cuu_key(pu, ex) for pu in BN_NGHIEN_CUU for ex in BAI_NGHIEN_CUU}
    vlist_by_slot = {}
    for v in video_list or []:
        if _la_ban_ghi_video_mo_co(v):
            continue
        sk = _slot_nghien_cuu_key(v.get("username"), v.get("exercise"))
        if sk in valid_slots:
            if sk in vlist_by_slot:
                vlist_by_slot[sk] = _chon_video_moi_hon(v, vlist_by_slot[sk])
            else:
                vlist_by_slot[sk] = dict(v)

    out = []
    users = load_users()
    for pu, ex in slots:
        sk = (pu, ex)
        doc_eval = _lay_eval_moi_nhat_theo_bai_tap(evals, pu, ex)
        ai_eval = _lay_eval_moi_nhat_theo_bai_tap(evals, pu, ex, doctor_username="AI_Researcher")
        if sk in vlist_by_slot:
            out.append(_ap_dung_ket_qua_moi_nhat_vao_video(vlist_by_slot[sk], ai_eval))
            continue
        if doc_eval:
            rec = _tao_ban_ghi_video_tu_danh_gia(doc_eval, ai_eval, users=users)
            out.append(_ap_dung_ket_qua_moi_nhat_vao_video(rec, ai_eval))

    out.sort(
        key=lambda x: _parse_vn_datetime(_lay_thoi_gian_upload_video(x)) or datetime.min,
        reverse=True,
    )
    return out


@st.cache_data(show_spinner=False)
def _evals_dedup_cached(e_mtime):
    try:
        evals = _load_data_cached(EVALUATIONS_FILE, e_mtime)
    except Exception:
        evals = []
    return _dedup_evaluations(evals)


@st.cache_data(show_spinner=False)
def _video_nghien_cuu_cached(v_mtime, e_mtime):
    evals = _evals_dedup_cached(e_mtime)
    vlist = _load_video_list_core(v_mtime, e_mtime) or []
    return _lay_video_nghien_cuu_chinh_thuc(vlist, evals)


def _mtimes_video_eval():
    try:
        v_mtime = os.path.getmtime(VIDEOS_FILE) if os.path.exists(VIDEOS_FILE) else 0
    except Exception:
        v_mtime = 0
    try:
        e_mtime = os.path.getmtime(EVALUATIONS_FILE) if os.path.exists(EVALUATIONS_FILE) else 0
    except Exception:
        e_mtime = 0
    return v_mtime, e_mtime


def _frontend_api_enabled():
    return bool(FRONTEND_API_CONFIG.enabled)


def _frontend_api_token():
    return st.session_state.get("backend_access_token")


def _frontend_api_client():
    return FrontendApiClient(FRONTEND_API_CONFIG, token=_frontend_api_token())


def _frontend_api_set_error(message):
    st.session_state["_frontend_api_last_error"] = str(message)


def _frontend_api_clear_error():
    st.session_state.pop("_frontend_api_last_error", None)


def _frontend_api_videos():
    if not _frontend_api_enabled() or not _frontend_api_token():
        return None
    if (st.session_state.get("user_info") or {}).get("role") == RESEARCHER_ROLE:
        return None
    try:
        client = _frontend_api_client()
        videos = client.videos()
        evals = _dedup_evaluations(client.evaluations())
        videos = _lay_video_nghien_cuu_chinh_thuc(videos, evals)
        _frontend_api_clear_error()
        return videos
    except FrontendApiError as exc:
        _frontend_api_set_error(exc)
        return None


def load_danh_sach_video_nghien_cuu():
    """8 video nghiên cứu — cache theo mtime JSON (nhanh khi chuyển tab)."""
    api_videos = _frontend_api_videos()
    if api_videos is not None:
        return api_videos
    v_mtime, e_mtime = _mtimes_video_eval()
    videos = _video_nghien_cuu_cached(v_mtime, e_mtime) or []
    return scope_records_for_current_actor(videos)


def _tim_video_cho_progress(video_path):
    """Tìm video theo path — chỉ quét 8 video nghiên cứu, không load toàn bộ video_list."""
    if not video_path:
        return None
    cur = st.session_state.get("current_eval_video")
    if cur and cur.get("video_path") == video_path:
        return cur
    for v in load_danh_sach_video_nghien_cuu():
        if v.get("video_path") == video_path:
            return v
    return None


def _thong_ke_video_nghien_cuu():
    """Thống kê chỉ trên 8 video nghiên cứu — không đếm toàn bộ video_list."""
    evals_db = _evals_dedup_cached(_mtimes_video_eval()[1])
    v_research = load_danh_sach_video_nghien_cuu()
    total = len(v_research)
    pending = 0
    acc_vals = []
    for v in v_research:
        pu, ex = v.get("username"), v.get("exercise")
        ai_eval = _lay_eval_moi_nhat_theo_bai_tap(evals_db, pu, ex, doctor_username="AI_Researcher")
        has_result = bool(v.get("metrics")) or bool(ai_eval)
        if not has_result:
            pending += 1
        acc_vals.append(_lay_do_chinh_xac_hien_thi(v, ai_eval))
    avg_acc = sum(acc_vals) / len(acc_vals) if acc_vals else 0.0
    return total, pending, avg_acc


# --- OPTIMIZED CACHING FOR FASTER PAGE LOADS ---
@st.cache_data(show_spinner=False)
def _check_video_valid_cached(path, mtime, size):
    if not os.path.exists(path) or os.path.getsize(path) < 5 * 1024:
        return False
    # Sử dụng ffprobe để kiểm tra tính toàn vẹn của video (đọc được duration = video chuẩn, không bị lỗi nửa chừng)
    if ffprobe_video_has_readable_duration(path):
        return True
    # Dự phòng: dùng OpenCV — grab() thay cho FRAME_COUNT vì H.264 thường báo 0 frames
    try:
        cap_check = cv2.VideoCapture(path)
        if cap_check.isOpened() and cap_check.grab():
            cap_check.release()
            return True
        cap_check.release()
    except:
        pass
    return False

@st.cache_data(show_spinner=False)
def get_video_fps_cached(path, mtime, size):
    try:
        cap_test = cv2.VideoCapture(path)
        fps = int(cap_test.get(cv2.CAP_PROP_FPS)) or 15
        cap_test.release()
        return fps
    except:
        return 15

@st.cache_data(max_entries=80, show_spinner=False)
def read_csv_cached(path, mtime, size):
    """Đọc CSV có cache theo mtime/size để biểu đồ phân tích mở lại nhanh hơn."""
    return pd.read_csv(path)

def read_analysis_csv_fast(path):
    """Đọc CSV phân tích nhanh: cache nếu file local đã sẵn sàng."""
    if not path or not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        return read_csv_cached(path, mtime, size)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

# Các cột cần cho hiển thị biểu đồ/chỉ số (BỎ 132 cột tọa độ pt0..pt32 để đọc nhanh hơn nhiều)
DISPLAY_CSV_COLS = [
    "frame", "timestamp", "timestamp_seconds",
    "goc_vai", "goc_khuyu",
    "goc_vai_trai", "goc_khuyu_trai", "goc_vai_phai", "goc_khuyu_phai",
    "dung", "gan_dung", "vai_dung", "khuyu_dung", "vai_chuan", "khuyu_chuan",
    "ml_label", "ml_label_text", "ml_score", "dung_ml", "ml_dung", "ml_gan_dung",
]

@st.cache_data(max_entries=80, show_spinner=False)
def read_display_csv_cached(path, mtime, size):
    """Chỉ đọc các cột cần để hiển thị (bỏ cột tọa độ landmark) -> nhanh hơn nhiều với CSV lớn."""
    try:
        header = pd.read_csv(path, nrows=0)
        cols = [c for c in DISPLAY_CSV_COLS if c in header.columns]
        if not cols:
            return pd.read_csv(path)
        return pd.read_csv(path, usecols=cols)
    except Exception:
        return pd.read_csv(path)

def read_display_csv_fast(path):
    """Đọc CSV cho phần XEM kết quả: chỉ cột cần thiết, có cache theo mtime/size."""
    if not path or not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        return read_display_csv_cached(path, mtime, size)
    except Exception:
        try:
            return read_analysis_csv_fast(path)
        except Exception:
            return None


# --- THUMBNAIL GENERATOR ---
def get_thumbnail(path, width=320):
    """Tạo thumbnail nhẹ để load web nhanh"""
    if not os.path.exists(path):
        return None
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        aspect = h / w
        new_h = int(width * aspect)
        img_res = cv2.resize(img, (width, new_h))
        # Chuyển BGR sang RGB cho Streamlit
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        return img_res
    except:
        return None

@st.cache_data(max_entries=2000, show_spinner=False)
def get_cached_frame_b64(path, width, jpeg_quality):
    """Đọc ảnh, resize, và mã hóa base64 có cache để tăng tốc tải trang cực kỳ nhanh"""
    if not os.path.exists(path):
        return ""
    try:
        img = cv2.imread(path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        aspect = h / w
        new_h = int(width * aspect)
        img_res = cv2.resize(img, (width, new_h))
        # Encode trực tiếp từ BGR sang JPEG, không cần convert RGB để tối ưu CPU
        _, buffer = cv2.imencode('.jpg', img_res, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return base64.b64encode(buffer).decode()
    except:
        # Fallback đọc base64 trực tiếp
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except:
            return ""

def get_base64_image(path):
    """Fallback: Chuyển ảnh sang base64 nếu load trực tiếp lỗi"""
    try:
        with open(path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    except:
        return None

@st.cache_data(max_entries=50, show_spinner=False)
def get_video_base64_cached(path, mtime, size):
    """Đọc và mã hóa video sang Base64 có cache để phát trực tiếp cực kỳ mượt mà trên Cloud"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

@st.cache_data(max_entries=500, show_spinner=False)
def _get_video_codec_cached(path, mtime, size):
    """Cache kết quả ffprobe theo (path, mtime, size) để tránh gọi lại subprocess."""
    return ffprobe_video_codecs(path)

def get_video_codec(path):
    """Sử dụng ffprobe để lấy thông tin codec video và audio nhanh chóng (có cache)."""
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        return _get_video_codec_cached(path, mtime, size)
    except:
        pass
    return None, None


def video_has_audio_track(path):
    """True nếu file video có track âm thanh."""
    if not path or not os.path.exists(path):
        return False
    try:
        _, a_codec = get_video_codec(path)
        return bool(a_codec)
    except Exception:
        return False

@st.cache_data(max_entries=300, show_spinner=False)
def _get_playable_path_fast(video_path, mtime, size):
    """Cache nhanh đường dẫn playable của video: trả về kết quả đường dẫn cuối cùng mà không có side-effect.
    Lần đầu tiên sẽ chạy full kiểm tra, các lần sau chỉ tra cache O(1)."""
    # 1. Nếu đường dẫn đã là _f.mp4 và file tồn tại hợp lệ → phát thẳng
    if video_path.endswith('_f.mp4') and os.path.exists(video_path) and size >= 5 * 1024:
        return video_path
    # 2. Kiểm tra file _f.mp4 tương ứng có tồn tại hợp lệ không
    final_h264 = get_final_h264_path(video_path)
    if os.path.exists(final_h264):
        try:
            if os.path.getsize(final_h264) >= 5 * 1024:
                return final_h264
        except:
            pass
    # 3. Kiểm tra codec: nếu đã là h264 → phát trực tiếp, không cần convert
    if video_path.endswith('.mp4') and os.path.exists(video_path) and size >= 5 * 1024:
        v_codec, _ = _get_video_codec_cached(video_path, mtime, size)
        if v_codec == 'h264':
            return video_path
    # 4. Chưa biết codec hoặc cần convert → báo là cần gọi ensure_playable_video
    return None  # None = cần xử lý đầy đủ

import threading

if '_transcoding_jobs' not in globals():
    _transcoding_jobs = set()
if '_transcoding_lock' not in globals():
    _transcoding_lock = threading.Lock()

def ensure_playable_video(video_path):
    """Đảm bảo video có định dạng H.264 mượt mà (đuôi _f.mp4) để chơi được trên trình duyệt.
    Nếu file _f.mp4 chưa có hoặc bị lỗi (0 byte, quá nhỏ), tự động tải và chuyển đổi từ file gốc dưới nền bất đồng bộ (không block UI)."""
    if not video_path:
        return video_path

    # Xác định file H264 đích mong muốn
    final_h264 = get_final_h264_path(video_path)

    # Nếu video_path đã là file H264 transcode và tồn tại hợp lệ cục bộ → dùng ngay nếu qua kiểm tra tính toàn vẹn
    if video_path.endswith('_f.mp4'):
        if os.path.exists(video_path) and os.path.getsize(video_path) >= 5 * 1024:
            try:
                mtime = os.path.getmtime(video_path)
                size = os.path.getsize(video_path)
                if _check_video_valid_cached(video_path, mtime, size):
                    return video_path
            except:
                pass
    else:
        # Nếu video gốc thô đã có sẵn cục bộ và là định dạng H264 → phát trực tiếp luôn
        if os.path.exists(video_path) and os.path.getsize(video_path) >= 5 * 1024:
            if video_path.endswith('.mp4'):
                try:
                    # Tra cứu cache codec cực nhanh
                    v_codec, _ = get_video_codec(video_path)
                    if v_codec == 'h264':
                        return video_path
                except:
                    pass

    # Nếu đã tồn tại file H264 đích hợp lệ cục bộ → dùng ngay
    is_valid_h264 = False
    if os.path.exists(final_h264) and os.path.getsize(final_h264) > 5 * 1024:
        try:
            mtime = os.path.getmtime(final_h264)
            size = os.path.getsize(final_h264)
            is_valid_h264 = _check_video_valid_cached(final_h264, mtime, size)
        except:
            pass

    if is_valid_h264:
        return final_h264

    # Khởi chạy tải và convert bất đồng bộ hoàn toàn dưới nền để tránh chặn luồng UI
    with _transcoding_lock:
        if final_h264 in _transcoding_jobs:
            return video_path
        _transcoding_jobs.add(final_h264)

    def _async_download_and_transcode():
        nonlocal video_path
        try:
            # 1. PHỤC HỒI / TẢI VIDEO GỐC DƯỚI NỀN (NẾU CHƯA CÓ)
            if video_path.endswith('_f.mp4'):
                is_corrupted = False
                if os.path.exists(video_path):
                    try:
                        mtime = os.path.getmtime(video_path)
                        size = os.path.getsize(video_path)
                        if os.path.getsize(video_path) < 5 * 1024 or not _check_video_valid_cached(video_path, mtime, size):
                            is_corrupted = True
                    except:
                        is_corrupted = True
                else:
                    success_dl = ensure_local_file(video_path)
                    if not success_dl or os.path.getsize(video_path) < 5 * 1024:
                        is_corrupted = True

                if is_corrupted:
                    possible_orig_paths = []
                    base_without_f = video_path.replace('_f.mp4', '')
                    for ext in ['.mp4', '.mov', '.MOV', '.avi', '.mkv']:
                        possible_orig_paths.append(base_without_f + ext)
                    orig_recovered_path = None
                    for p_orig in possible_orig_paths:
                        if os.path.exists(p_orig) or ensure_local_file(p_orig):
                            orig_recovered_path = p_orig
                            break
                    if orig_recovered_path:
                        if os.path.exists(video_path):
                            try: os.remove(video_path)
                            except: pass
                        video_path = orig_recovered_path
            else:
                ensure_local_file(video_path)

            # Kiểm tra xem video gốc thô đã hợp lệ chưa
            if not os.path.exists(video_path) or os.path.getsize(video_path) < 5 * 1024:
                print(f"[Async Video] Không thể tải/phát hiện video gốc hợp lệ {video_path}")
                return
            try:
                mtime_src = os.path.getmtime(video_path)
                size_src = os.path.getsize(video_path)
                if not _check_video_valid_cached(video_path, mtime_src, size_src):
                    print(f"[Async Video] Bo qua transcode — file loi/moov atom: {video_path}")
                    return
            except Exception:
                print(f"[Async Video] Bo qua transcode — khong kiem tra duoc file: {video_path}")
                return

            # 2. XÓA CẢ FILE TẠM, FILE H264 CŨ VÀ LOG LỖI CŨ
            # Dùng đuôi _ftmp.mp4 (không phải .mp4.tmp) để ffmpeg nhận đúng container MP4
            tmp_h264 = final_h264.replace('_f.mp4', '_ftmp.mp4')
            error_log_path = os.path.join(os.path.dirname(final_h264), "transcode_error.txt")
            for f_clean in [final_h264, tmp_h264, error_log_path]:
                if os.path.exists(f_clean):
                    try: os.remove(f_clean)
                    except: pass

            # 3. TRÍCH CODEC & TRANSCODE — ghi vào file TẠM _ftmp.mp4 trước
            # Sau khi xong mới đổi tên → _f.mp4 KHÔNG BAO GIỜ bị nửa vời (moov atom not found)
            v_codec, a_codec = get_video_codec(video_path)
            cmd = build_async_h264_command(
                video_path,
                tmp_h264,
                has_audio=bool(a_codec),
                ffmpeg_threads=MAX_FFMPEG_THREADS,
            )

            print(f"[Async Video] Đang convert {video_path} sang H.264 (file tạm: {tmp_h264})...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1800)

            if result.returncode != 0:
                print("[Async Video] FFmpeg failed:", result.returncode, result.stderr[-500:])
                try:
                    error_log_path = os.path.join(os.path.dirname(final_h264), "transcode_error.txt")
                    with open(error_log_path, "w", encoding="utf-8") as f_err:
                        f_err.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f_err.write(f"Cmd: {' '.join(cmd)}\n")
                        f_err.write(f"Exit Code: {result.returncode}\n")
                        f_err.write(f"Stderr:\n{result.stderr}\n")
                    push_file_to_hf_async(error_log_path)
                except:
                    pass
                # Xóa file tạm nếu ffmpeg fail
                if os.path.exists(tmp_h264):
                    try: os.remove(tmp_h264)
                    except: pass
                return

            # Kiểm tra file tạm hợp lệ trước khi đổi tên
            if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 5 * 1024:
                mtime_f = os.path.getmtime(tmp_h264)
                size_f = os.path.getsize(tmp_h264)
                if not _check_video_valid_cached(tmp_h264, mtime_f, size_f):
                    print("[Async Video] File tạm không hợp lệ sau ffmpeg. Xóa...")
                    try:
                        error_log_path = os.path.join(os.path.dirname(final_h264), "transcode_error.txt")
                        with open(error_log_path, "w", encoding="utf-8") as f_err:
                            f_err.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f_err.write("Error: Output file failed integrity check after ffmpeg.\n")
                        push_file_to_hf_async(error_log_path)
                    except:
                        pass
                    try: os.remove(tmp_h264)
                    except: pass
                    return

                # ✅ File tạm hợp lệ → đổi tên ATOMIC sang _f.mp4
                try:
                    os.replace(tmp_h264, final_h264)
                    print(f"[Async Video] ✅ Convert thành công: {final_h264}")
                except Exception as rename_err:
                    print(f"[Async Video] Lỗi đổi tên: {rename_err}")
                    return

                # Xóa log lỗi cũ
                try:
                    error_log_path = os.path.join(os.path.dirname(final_h264), "transcode_error.txt")
                    if os.path.exists(error_log_path):
                        os.remove(error_log_path)
                except:
                    pass

                # Cập nhật database
                try:
                    video_list = load_data(VIDEOS_FILE)
                    updated = False
                    for vid in video_list:
                        if vid.get('processed_path') == video_path:
                            vid['processed_path'] = final_h264
                            updated = True
                        # Không ghi đè video_path — giữ file upload gốc BN trong DB
                    if updated:
                        save_data(VIDEOS_FILE, video_list)
                        print("[Async Video] Đã cập nhật database video_list.json")
                except Exception as db_err:
                    print(f"[Async Video] Lỗi cập nhật database: {db_err}")

                push_file_to_hf_async(final_h264)
            else:
                # File tạm không tồn tại hoặc quá nhỏ
                if os.path.exists(tmp_h264):
                    try: os.remove(tmp_h264)
                    except: pass
        except Exception as err:
            print(f"[Async Video] Lỗi trong tiến trình chạy nền: {err}")
            try:
                error_log_path = os.path.join(os.path.dirname(final_h264), "transcode_error.txt")
                with open(error_log_path, "w", encoding="utf-8") as f_err:
                    f_err.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f_err.write(f"Exception: {str(err)}\n")
                push_file_to_hf_async(error_log_path)
            except:
                pass
            # Xóa cả file tạm và file đích nếu có
            for f_clean in [final_h264, final_h264 + ".tmp"]:
                if os.path.exists(f_clean):
                    try: os.remove(f_clean)
                    except: pass
        finally:
            with _transcoding_lock:
                _transcoding_jobs.discard(final_h264)

    threading.Thread(target=_async_download_and_transcode, daemon=True).start()
    return video_path


# ============================================================
# VIDEO HTTP STREAMING SERVER
# Phục vụ video qua HTTP Range Requests thực sự — browser chỉ tải
# đúng đoạn đang cần, không cần đợi load toàn bộ file.
# ============================================================
_video_http_server_port = None
_video_http_server_roots = {}
_video_http_media_tokens = {}
_video_http_media_lock = None
_VIDEO_HTTP_TOKEN_TTL_SECONDS = int(os.environ.get("REHAB_MEDIA_TOKEN_TTL_SECONDS", "3600"))


def _safe_realpath(path):
    return safe_realpath(path)


def _path_is_within(child, parent):
    return path_is_within(child, parent)


def _video_media_allowed_roots():
    """Only expose known media folders, never the repository root."""
    return video_media_allowed_roots(
        data_dir=globals().get("DATA_DIR", "."),
        upload_dir=globals().get("UPLOAD_DIR"),
        processed_dir=globals().get("PROCESSED_DIR"),
    )


def _video_media_lock():
    global _video_http_media_lock
    if _video_http_media_lock is None:
        import threading
        _video_http_media_lock = threading.RLock()
    return _video_http_media_lock


def _video_media_token_cleanup(now=None):
    cleanup_media_tokens(_video_http_media_tokens, now=now)


def _allowed_media_file_path(path):
    return allowed_media_file_path(path, _video_media_allowed_roots(), allowed_extensions=ALLOWED_VIDEO_EXTENSIONS)


def _register_video_media_token(path):
    with _video_media_lock():
        return register_media_token(
            _video_http_media_tokens,
            path,
            _video_media_allowed_roots(),
            ttl_seconds=_VIDEO_HTTP_TOKEN_TTL_SECONDS,
        )


def _resolve_video_media_token(token):
    with _video_media_lock():
        return resolve_media_token(_video_http_media_tokens, token, _video_media_allowed_roots())


def _is_allowed_video_origin(origin):
    return is_allowed_video_origin(origin)

def _start_video_http_server():
    """Khởi động 1 lần duy nhất một HTTP server nhẹ để stream video file."""
    global _video_http_server_port, _video_http_server_roots
    if _video_http_server_port is not None:
        return _video_http_server_port

    import http.server
    import socketserver
    import threading

    _video_http_server_roots = _video_media_allowed_roots()
    if not _video_http_server_roots:
        return None

    class _RangeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # tắt log tràn console

        def _guess_type(self, path):
            try:
                import mimetypes
                return mimetypes.guess_type(path)[0] or 'video/mp4'
            except Exception:
                return 'video/mp4'

        def _reject(self, code=404, reason="not_found"):
            try:
                _logging.getLogger(__name__).warning("[VideoServer] rejected media request: %s", reason)
            except Exception:
                pass
            self.send_error(code)

        def _resolve_request_path(self):
            try:
                token = media_token_from_request_path(self.path)
                if not token:
                    return None
                return _resolve_video_media_token(token)
            except Exception:
                return None

        def do_GET(self):
            import re
            path = self._resolve_request_path()
            if not path:
                self._reject(404, "invalid_or_expired_token")
                return

            range_header = self.headers.get('Range')
            if not range_header:
                return self._send_full_file(path)

            match = re.match(r'bytes=(\d+)-(\d*)', range_header)
            if not match:
                return self._send_full_file(path)

            start = int(match.group(1))
            end_str = match.group(2)

            file_size = os.path.getsize(path)
            end = int(end_str) if end_str else file_size - 1
            if start >= file_size:
                self.send_error(416, "Requested Range Not Satisfiable")
                return

            if end >= file_size:
                end = file_size - 1

            content_length = end - start + 1

            try:
                self.send_response(206)
                ctype = self._guess_type(path)
                self.send_header('Content-Type', ctype or 'video/mp4')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', str(content_length))
                self.end_headers()

                with open(path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    buffer_size = 64 * 1024
                    while remaining > 0:
                        chunk_size = min(buffer_size, remaining)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except Exception:
                pass

        def do_HEAD(self):
            path = self._resolve_request_path()
            if not path:
                self._reject(404, "invalid_or_expired_token")
                return
            try:
                self.send_response(200)
                self.send_header('Content-Type', self._guess_type(path))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(os.path.getsize(path)))
                self.end_headers()
            except Exception:
                self._reject(404, "head_failed")

        def _send_full_file(self, path):
            try:
                self.send_response(200)
                self.send_header('Content-Type', self._guess_type(path))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(os.path.getsize(path)))
                self.end_headers()
                with open(path, 'rb') as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except Exception:
                pass

        def end_headers(self):
            origin = self.headers.get("Origin")
            if _is_allowed_video_origin(origin):
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.send_header('Cache-Control', 'private, max-age=300')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

    # Tìm cổng trống bắt đầu từ 8765
    port = 8765
    for attempt_port in range(8765, 8800):
        try:
            server = socketserver.ThreadingTCPServer(('127.0.0.1', attempt_port), _RangeHandler)
            server.allow_reuse_address = True
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            _video_http_server_port = attempt_port
            print(f'[VideoServer] Đang phục vụ video tại http://127.0.0.1:{attempt_port}')
            break
        except OSError:
            continue

    return _video_http_server_port


def _get_video_server_url(video_path):
    """Return a short-lived local media URL for an already-authorized local video file."""
    port = _start_video_http_server()
    token = _register_video_media_token(video_path)
    if port is None or token is None:
        return None
    return build_video_media_url(port, token, video_path)


def get_playable_local_copy(target_path):
    """Tạo bản sao tạm thời của video trong /tmp (có quyền ghi trên mọi môi trường Cloud) để st.video phát qua bytes."""
    if not target_path or not os.path.exists(target_path):
        return None
    try:
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "rehab_videos")
        os.makedirs(temp_dir, exist_ok=True)

        h = hashlib.md5(target_path.encode()).hexdigest()[:10]
        ext = os.path.splitext(target_path)[1] or ".mp4"
        filename = f"{h}{ext}"
        dest_path = os.path.join(temp_dir, filename)

        # Nếu chưa copy hoặc file nguồn mới hơn -> copy sang
        src_mtime = os.path.getmtime(target_path)
        if not os.path.exists(dest_path) or src_mtime > os.path.getmtime(dest_path):
            import shutil
            shutil.copy2(target_path, dest_path)

            # Tự động dọn dẹp để tránh đầy đĩa (chỉ giữ tối đa 15 file gần nhất)
            try:
                all_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
                if len(all_files) > 15:
                    all_files.sort(key=os.path.getmtime)
                    for f in all_files[:-15]:
                        try: os.remove(f)
                        except: pass
            except:
                pass

        return dest_path  # Trả về đường dẫn tuyệt đối trong /tmp
    except Exception as e:
        print(f"[TempCopy] Lỗi sao chép file: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def check_cloud_file_exists(url):
    """Kiểm tra nhanh xem file có tồn tại trên Cloud (Hugging Face) không bằng HTTP HEAD request (có cache)"""
    if not url:
        return False
    try:
        rel_path = get_clean_rel_path(url)
        return dataset_file_exists(rel_path, token=HF_TOKEN, dataset_id=HF_DATASET_ID)
    except Exception:
        return False


def _hf_dataset_resolve_urls(video_path, prefer_raw=False):
    """Tokenized client-side HF URLs are disabled; use server-side downloads."""
    return None, None


def _prefetch_video_quiet(video_path):
    """Tải video về local dưới nền — không chặn UI phát stream."""
    if not video_path:
        return

    def _video_prefetch_worker():
        try:
            for p in video_fallback_paths(video_path):
                ensure_local_file(p, quiet=True, try_fallbacks=True)
        except Exception:
            pass

    threading.Thread(target=_video_prefetch_worker, daemon=True).start()


def _render_video_html5_iframe(sources_html, comp_key, height=520, footer_html=""):
    """Phát video HTML5 — preload metadata để hiện khung hình nhanh."""
    import streamlit.components.v1 as _stcomp
    foot = safe_html(footer_html or "", max_length=220)
    vid_id = (comp_key or "vp").replace(" ", "_")
    msg_id = vid_id + "_msg"
    _stcomp.html(
        f"""
<!DOCTYPE html><html><head>
<style>
  body{{margin:0;padding:0;background:transparent;overflow:hidden;}}
  video{{width:100%;height:auto;max-height:{height}px;border-radius:8px;display:block;background:#111;object-fit:contain;}}
  .vf{{color:#aaa;font-size:0.72rem;margin-top:4px;text-align:right;font-family:sans-serif;}}
  #_{msg_id}{{display:none;width:100%;height:{height}px;background:#1a1a2e;border-radius:8px;
    align-items:center;justify-content:center;flex-direction:column;color:#aaa;font-family:sans-serif;font-size:0.85rem;text-align:center;}}
</style>
</head><body>
<video id="{vid_id}" controls preload="auto" playsinline>
  {sources_html}
  Trình duyệt không hỗ trợ video HTML5.
</video>
<div id="_{msg_id}" style="display:none;width:100%;height:{height}px;background:#1a1a2e;border-radius:8px;
  display:none;align-items:center;justify-content:center;flex-direction:column;color:#aaa;font-family:sans-serif;font-size:0.85rem;text-align:center;">
  <div style="font-size:2rem;margin-bottom:8px;">🎬</div>
  <div>Video đang tải từ Cloud...</div>
  <div style="font-size:0.75rem;margin-top:4px;color:#666;">Bấm 🔄 trên trang để thử lại nếu không hiện</div>
</div>
<script>
(function() {{
  var v = document.getElementById("{vid_id}");
  var msg = document.getElementById("_{msg_id}");
  if (!v) return;
  var idx = 0;
  var sources = v.querySelectorAll("source");
  function showFallback() {{
    if (msg) {{ v.style.display="none"; msg.style.display="flex"; }}
  }}
  function tryNext() {{
    idx += 1;
    if (idx < sources.length) {{
      v.src = sources[idx].src;
      v.load();
      v.play().catch(function(){{}});
    }} else {{
      showFallback();
    }}
  }}
  v.addEventListener("error", tryNext);
  setTimeout(function() {{
    if (v.readyState === 0 && v.networkState === 3) showFallback();
  }}, 8000);
}})();
</script>
{f'<div class="vf">{foot}</div>' if foot else ''}
</body></html>
""",
        height=height + (22 if foot else 0),
    )


def _is_hf_runtime():
    """Chạy trên Hugging Face Space (/data persistent volume)."""
    return bool(HF_SPACE_ID or os.environ.get("SPACE_ID") or os.path.exists("/data"))


def _try_render_cloud_video_stream(video_path, key_hint="", optimistic=False, prefer_raw=False):
    """Do not stream private HF media directly to the browser."""
    return False


def _render_video_static_iframe(target_path, video_key=None):
    """Phát file local qua media server đã giới hạn root — không đọc hết file vào RAM."""
    if not target_path or not os.path.exists(target_path):
        return False
    try:
        # Kiểm tra tính tương thích của codec và container với trình duyệt
        v_codec, _ = get_video_codec(target_path)
        if v_codec:
            is_compatible = (v_codec == 'h264' and target_path.lower().endswith('.mp4'))
            if not is_compatible:
                return False

        path_hash = hashlib.md5(target_path.encode()).hexdigest()[:10]
        video_key = video_key or f"st_vid_comp_{path_hash}"
        media_url = _get_video_server_url(target_path)
        if not media_url:
            return False

        iframe_height = 520
        try:
            cap_info = cv2.VideoCapture(target_path)
            if cap_info.isOpened():
                v_w = cap_info.get(cv2.CAP_PROP_FRAME_WIDTH)
                v_h = cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap_info.release()
                if v_w > 0 and v_h > 0:
                    iframe_height = int((v_h / v_w) * 640)
                    iframe_height = max(300, min(iframe_height, 720))
        except Exception:
            pass

        _render_video_html5_iframe(
            f'<source src="{media_url}" type="video/mp4">',
            video_key,
            height=iframe_height,
            footer_html=f"📁 Local — {os.path.basename(target_path)}",
        )
        return True
    except Exception as static_err:
        print(f"[render_video] static iframe fail: {static_err}")
        return False


def _render_video_streamlit_native(target_path, allow_large=False):
    """Phát video qua st.video — HF Space cần st.video (static/ iframe hay bị đen)."""
    if not target_path or not os.path.exists(target_path):
        return False
    try:
        mtime = os.path.getmtime(target_path)
        size = os.path.getsize(target_path)
        if size < 5 * 1024 or not _check_video_valid_cached(target_path, mtime, size):
            return False
        if size > 6 * 1024 * 1024 and not (allow_large or _is_hf_runtime()):
            return False

        # Kiểm tra tính tương thích của codec và container với trình duyệt
        v_codec, _ = get_video_codec(target_path)
        if v_codec:
            is_compatible = (v_codec == 'h264' and target_path.lower().endswith('.mp4'))
            if not is_compatible:
                return False

        # Đọc video thành bytes để phát qua Streamlit (giải quyết lỗi Range Request/màn đen trên HF Space)
        with open(target_path, "rb") as f:
            st.video(f.read(), format="video/mp4")
        return True
    except Exception as native_err:
        print(f"[render_video] st.video fail: {native_err}")
        return False


def dam_bao_tai_video_phan_tich(processed_path, allow_sync_transcode=False):
    """Tải video phân tích về local — không transcode đồng bộ khi chỉ cần phát."""
    if not processed_path or _la_duong_dan_video_gia(processed_path):
        return None
    ready = find_ready_local_video(processed_path)
    if ready:
        pb = resolve_playback_video_path(ready)
        if pb and is_local_file_ready(pb):
            return pb
        return ready
    if ensure_local_file(processed_path, quiet=True, try_fallbacks=True):
        ready = find_ready_local_video(processed_path)
        if ready:
            pb = resolve_playback_video_path(ready, sync_transcode=allow_sync_transcode)
            if pb and is_local_file_ready(pb):
                return pb
            return ready
    return None


def nap_phien_benh_nhan_vao_session(selected_v):
    """Nạp nhẹ session khi chọn phiên tập — không tải CSV/video nặng (tải lazy ở từng tab)."""
    if not selected_v:
        return
    vid_key = f"{selected_v.get('username')}|{selected_v.get('video_name')}|{selected_v.get('exercise')}"
    if st.session_state.get("_patient_session_key") == vid_key:
        return
    st.session_state._patient_session_key = vid_key
    st.session_state.current_eval_video = selected_v
    st.session_state.stats = selected_v.get('metrics')
    st.session_state.processed_video_path = selected_v.get('processed_path')
    st.session_state.all_frames_data_path = selected_v.get('all_frames_data_path')
    st.session_state.uploaded_file_name = selected_v.get('video_name')
    st.session_state.frames_zip = _frames_zip_path_from_video(selected_v)
    st.session_state.has_data = True
    st.session_state.view_old_analysis = True
    st.session_state.reanalyze_triggered = False
    ex_name = selected_v.get('exercise', 'codman')
    ex_base = next((BAI_TAP[k] for k in BAI_TAP if BAI_TAP[k]['ten'] == ex_name), BAI_TAP['codman'])
    st.session_state.exercise = ex_base.copy()
    if 'sai_so' in selected_v:
        st.session_state.exercise['chuan'] = ex_base['chuan'].copy()
        st.session_state.exercise['chuan']['sai_so'] = selected_v['sai_so']
    st.session_state.angle_df = None
    st.session_state.current_df_csv_path = selected_v.get('df_path')
    st.session_state.pop("_ncv_analysis_loaded_key", None)


def _lam_moi_ban_ghi_video_tu_db(v):
    """Lấy bản ghi video mới nhất từ DB (đường dẫn/metrics có thể đã cập nhật)."""
    if not v:
        return v
    u, vn, ex = v.get("username"), v.get("video_name"), v.get("exercise")
    fallback = None
    for cand in load_danh_sach_video_nghien_cuu():
        if cand.get("username") != u:
            continue
        if ex and cand.get("exercise") == ex:
            return cand
        if vn and cand.get("video_name") == vn:
            fallback = cand
    if fallback:
        return fallback
    return v


def _tim_video_phan_tich_moi_nhat():
    """Video nghiên cứu đã phân tích gần nhất."""
    analyzed = _danh_sach_video_phan_tich_sap_xep()
    return analyzed[0] if analyzed else None


def _tim_video_co_du_lieu_tai_duoc(preferred=None, only_preferred=False):
    """Thử lần lượt các video đã phân tích — chọn bản tải được CSV/JSON."""
    candidates = []
    seen_slots = set()
    if preferred:
        p = _lam_moi_ban_ghi_video_tu_db(preferred)
        if p:
            candidates.append(p)
            seen_slots.add(_slot_nghien_cuu_key(p.get("username"), p.get("exercise")))
    if only_preferred:
        for v in candidates:
            if not v or not v.get("metrics"):
                continue
            df, src = _nap_angle_df_tu_video(v)
            if df is not None:
                return v, df, src
        return None, None, None
    for v in _danh_sach_video_phan_tich_sap_xep():
        sk = _slot_nghien_cuu_key(v.get("username"), v.get("exercise"))
        if sk not in seen_slots:
            seen_slots.add(sk)
            candidates.append(v)
    for v in candidates:
        if not v or not v.get("metrics"):
            continue
        df, src = _nap_angle_df_tu_video(v)
        if df is not None:
            return v, df, src
    return None, None, None


def _duong_dan_csv_candidates(v):
    """Danh sách đường dẫn CSV có thể chứa dữ liệu biểu đồ."""
    if not v:
        return []
    candidates = []
    df_path = v.get("df_path")
    if df_path:
        candidates.append(get_local_frame_path(df_path) or df_path)
        candidates.append(df_path)
    proc = v.get("processed_path") or v.get("video_path") or ""
    import re
    m = re.search(r"processed_(\d+)", str(proc))
    if m:
        ts = m.group(1)
        for suffix in ("_f_data.csv", "_data.csv"):
            candidates.append(os.path.join(PROCESSED_DIR, f"processed_{ts}{suffix}"))
    seen, out = set(), []
    for p in candidates:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _artifact_timestamps_from_video(v):
    """Lay cac timestamp processed_* lien quan den mot ban ghi ket qua da luu."""
    if not v:
        return []
    import re
    stamps = []
    for key in (
        "processed_path", "df_path", "all_frames_data_path",
        "frames_zip", "frames_zip_path", "video_path",
    ):
        p = v.get(key)
        if not p:
            continue
        text = str(p)
        for m in re.finditer(r"processed_(\d+)", text):
            stamps.append(m.group(1))
        for m in re.finditer(r"(?:^|[/\\])f_(\d+)\.json", text):
            stamps.append(m.group(1))
    seen, out = set(), []
    for ts in stamps:
        if ts and ts not in seen:
            seen.add(ts)
            out.append(ts)
    return out


def _duong_dan_frames_json_candidates(v):
    """Danh sách đường dẫn JSON khung xương (fallback khi CSV không có trên Cloud)."""
    if not v:
        return []
    candidates = []
    frames_path = v.get("all_frames_data_path")
    if frames_path:
        candidates.append(get_local_frame_path(frames_path) or frames_path)
        candidates.append(frames_path)
    for ts in _artifact_timestamps_from_video(v):
        candidates.append(os.path.join(PROCESSED_DIR, f"f_{ts}.json"))
    seen, out = set(), []
    for p in candidates:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _frames_zip_from_processed_path(processed_path):
    if not processed_path:
        return ""
    proc = get_local_frame_path(processed_path) or processed_path
    import re
    m = re.search(r"processed_(\d+)", str(proc))
    if m:
        return os.path.join(PROCESSED_DIR, f"processed_{m.group(1)}_frames.zip")
    if str(proc).lower().endswith(".mp4"):
        return str(proc)[:-4] + "_frames.zip"
    return ""


def _frames_zip_path_from_video(v):
    if not v:
        return ""
    processed_zip = _frames_zip_from_processed_path(v.get("processed_path"))
    if processed_zip:
        import re
        processed_ts = re.search(r"processed_(\d+)", str(processed_zip))
        for key in ("frames_zip", "frames_zip_path"):
            raw = v.get(key)
            if not raw:
                continue
            raw_ts = re.search(r"processed_(\d+)", str(raw))
            if processed_ts and raw_ts and processed_ts.group(1) != raw_ts.group(1):
                print(
                    "[VideoMetadata] frames_zip timestamp mismatch; "
                    f"using processed_path-derived ZIP for {v.get('video_name') or v.get('video_path')}"
                )
                break
        return processed_zip
    for ts in _artifact_timestamps_from_video(v):
        if ts:
            return os.path.join(PROCESSED_DIR, f"processed_{ts}_frames.zip")
    for key in ("frames_zip", "frames_zip_path"):
        p = v.get(key)
        if p:
            return get_local_frame_path(p) or p
    return _frames_zip_from_processed_path(v.get("processed_path") or v.get("video_path"))


def _tim_duong_dan_csv_tu_video(v):
    """Tìm và tải file CSV góc khớp — từ df_path hoặc processed_* timestamp."""
    for p in _duong_dan_csv_candidates(v):
        if is_local_file_ready(p, min_size=80):
            return p
    for p in _duong_dan_csv_candidates(v):
        if ensure_local_file(p, quiet=True, try_fallbacks=False):
            lp = get_local_frame_path(p) or p
            if is_local_file_ready(lp, min_size=80):
                return lp
    return None


def _dataframe_tu_frames_json(json_path):
    """Dựng DataFrame biểu đồ từ file f_*.json khi CSV không tải được."""
    frames = load_all_frames_data_cached(json_path)
    if not frames:
        return None
    try:
        df = pd.DataFrame(frames)
        if df.empty:
            return None
        if "frame" not in df.columns and "index" in df.columns:
            df["frame"] = df["index"]
        if "timestamp_seconds" not in df.columns and "frame" in df.columns:
            df["timestamp_seconds"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0) / 30.0
        for col in ("goc_vai", "goc_khuyu"):
            if col not in df.columns:
                return None
        return df
    except Exception:
        return None


def _nap_angle_df_tu_video(v):
    """Nạp angle_df: ưu tiên CSV, fallback JSON khung xương trên Dataset."""
    global _hf_last_download_error
    csv_path = _tim_duong_dan_csv_tu_video(v)
    if csv_path:
        try:
            df = read_display_csv_fast(csv_path)
            if df is not None and len(df) > 0:
                return df, csv_path
        except Exception:
            pass
    for jp in _duong_dan_frames_json_candidates(v):
        # Chỉ đọc file nếu đã có sẵn local — không block render bằng download đồng bộ.
        # Background job (_bat_dau_tai_day_du_song_song) sẽ tải file về và fragment sẽ tự refresh.
        jp = get_local_frame_path(jp) or jp
        if not is_local_file_ready(jp, min_size=2):
            continue
        df = _dataframe_tu_frames_json(jp)
        if df is not None and len(df) > 0:
            _hf_last_download_error = None
            return df, jp
    tried = _duong_dan_csv_candidates(v)[:1] or _duong_dan_frames_json_candidates(v)[:1]
    if tried:
        rel = get_clean_rel_path(tried[0])
        if _hf_last_download_error and "404" not in str(_hf_last_download_error):
            pass
        elif rel:
            _hf_last_download_error = (
                f"Chưa có trên Dataset: `{rel}` — thử tải lại sau hoặc chạy phân tích mới."
            )
    return None, None


def _danh_sach_video_phan_tich_sap_xep():
    """8 video đã phân tích, mới nhất trước."""
    vlist = load_danh_sach_video_nghien_cuu()
    analyzed = [v for v in vlist if isinstance(v.get("metrics"), dict) and v.get("metrics")]
    if not analyzed:
        return []
    evals = _dedup_evaluations(load_data(EVALUATIONS_FILE))

    def _sort_key(v):
        ai = _lay_eval_moi_nhat_theo_bai_tap(
            evals, v.get("username"), v.get("exercise"), doctor_username="AI_Researcher"
        )
        t = _parse_vn_datetime(ai.get("time") if ai else None)
        if t:
            return t
        proc_ts = _lay_epoch_tu_processed(v.get("processed_path"))
        if proc_ts:
            try:
                return datetime.fromtimestamp(proc_ts)
            except (OSError, OverflowError, ValueError):
                pass
        return _parse_vn_datetime(v.get("time")) or datetime.min

    analyzed.sort(key=_sort_key, reverse=True)
    return analyzed


def _slot_video_phan_tich(v):
    if not v:
        return None
    return _slot_nghien_cuu_key(v.get("username"), v.get("exercise"))


def _session_phan_tich_khop_video(v):
    """Session đang nạp đúng BN+bài tập — tránh hiển thị biểu đồ/frames của người khác."""
    if not v:
        return False
    if st.session_state.get("_ncv_analysis_loaded_key") != _slot_video_phan_tich(v):
        return False
    return bool(
        st.session_state.get("has_data")
        and st.session_state.get("stats")
        and st.session_state.get("angle_df") is not None
    )


def _xoa_session_phan_tich():
    """Xóa cache phân tích trong session khi chuyển sang video/BN khác."""
    for k in (
        "has_data", "stats", "angle_df", "processed_video_path",
        "current_df_csv_path", "all_frames_data_path", "frames_zip",
        "_ncv_analysis_loaded_key", "all_frames_paths", "all_frames_data",
        "video_ready", "frames_ready", "frames_loaded", "temp_frames_dir",
    ):
        st.session_state.pop(k, None)
    _xoa_cache_hien_thi_ket_qua()


def _gan_khoa_session_phan_tich(v):
    key = _slot_video_phan_tich(v)
    if key:
        st.session_state["_ncv_analysis_loaded_key"] = key


def tu_dong_nap_ket_qua_phan_tich_gan_nhat(v=None, force=False):
    """Tự động nạp kết quả phân tích gần nhất: metrics, CSV/JSON biểu đồ, video, frames."""
    cur = st.session_state.get("current_eval_video") or {}
    target = (_lam_moi_ban_ghi_video_tu_db(v) or v) if v else cur
    if not force and _session_phan_tich_khop_video(target):
        return True

    preferred = v or cur or None
    if preferred:
        _dong_bo_video_list_day_du_tu_hf()
        preferred, _ = tai_tep_phan_tich_tu_hf(preferred)
        st.session_state.current_eval_video = preferred
    pref_slot = _slot_video_phan_tich(
        _lam_moi_ban_ghi_video_tu_db(preferred) or preferred
    ) if preferred else None
    found_v, pre_df, pre_src = _tim_video_co_du_lieu_tai_duoc(
        preferred, only_preferred=bool(preferred)
    )
    if not found_v:
        found_v = _lam_moi_ban_ghi_video_tu_db(preferred) if preferred else _tim_video_phan_tich_moi_nhat()
    if not found_v or not found_v.get("metrics"):
        return False
    if pref_slot and _slot_video_phan_tich(found_v) != pref_slot:
        return False

    ok = khoi_phuc_ket_qua_cu(found_v, tai_day_du=True)
    if pre_df is not None and st.session_state.get("angle_df") is None:
        st.session_state.angle_df = pre_df
        if pre_src:
            st.session_state.current_df_csv_path = pre_src
    if ok and st.session_state.get("stats") and st.session_state.get("angle_df") is not None:
        _gan_khoa_session_phan_tich(found_v)
        return True
    return False


def _xoa_cache_hien_thi_ket_qua():
    """Xóa cache UI để tải lại biểu đồ / video / frames không bị dữ liệu cũ."""
    for k in (
        "segment_bounds", "last_processed_video_for_bounds", "all_frames_paths",
        "all_frames_data", "video_ready", "frames_ready", "frames_loaded", "temp_frames_dir",
    ):
        st.session_state.pop(k, None)
    try:
        load_all_frames_data_cached.clear()
    except Exception:
        pass


def khoi_phuc_ket_qua_cu(v, tai_csv=True, tai_day_du=False):
    """Khôi phục kết quả phân tích đã lưu vào session — dùng khi hủy phân tích mới / xem bản cũ."""
    v = _lam_moi_ban_ghi_video_tu_db(v)
    if not v or not v.get("metrics"):
        return False
    _xoa_cache_hien_thi_ket_qua()
    st.session_state.current_eval_video = v
    st.session_state.reanalyze_triggered = False
    st.session_state.view_old_analysis = True
    st.session_state.stats = v["metrics"]
    st.session_state.has_data = True
    st.session_state.processed_video_path = v.get("processed_path", v.get("video_path"))
    st.session_state.uploaded_file_name = v.get("video_name", "Video đã lưu")
    st.session_state.all_frames_data_path = v.get("all_frames_data_path")
    st.session_state.frames_zip = _frames_zip_path_from_video(v)
    st.session_state.current_df_csv_path = v.get("df_path")
    ex_base = next((BAI_TAP[k] for k in BAI_TAP if BAI_TAP[k]["ten"] == v.get("exercise")), BAI_TAP["codman"])
    st.session_state.exercise = ex_base.copy()
    if "sai_so" in v:
        st.session_state.exercise["chuan"] = ex_base["chuan"].copy()
        st.session_state.exercise["chuan"]["sai_so"] = v["sai_so"]
    if tai_csv or tai_day_du:
        df_loaded, src_loaded = _nap_angle_df_tu_video(v)
        if df_loaded is not None:
            st.session_state.angle_df = df_loaded
            st.session_state.current_df_csv_path = src_loaded
            v_upd = dict(v)
            if src_loaded and str(src_loaded).lower().endswith(".csv"):
                v_upd["df_path"] = src_loaded
            if src_loaded and str(src_loaded).lower().endswith(".json"):
                v_upd["all_frames_data_path"] = src_loaded
            st.session_state.current_eval_video = v_upd
        else:
            st.session_state.angle_df = None
    if tai_day_du:
        proc = v.get("processed_path") or v.get("video_path")
        if proc:
            ensure_local_file(proc, try_fallbacks=True)
            dam_bao_tai_video_phan_tich(proc)
        frames_json = v.get("all_frames_data_path")
        if frames_json:
            ensure_local_file(frames_json, try_fallbacks=False)
        fz = _frames_zip_path_from_video(v)
        if fz:
            ensure_local_file(fz, try_fallbacks=False)
        if proc:
            check_and_extract_frames_zip(proc)
    if st.session_state.get("angle_df") is not None:
        _gan_khoa_session_phan_tich(st.session_state.get("current_eval_video") or v)
        return True
    if tai_csv or tai_day_du:
        return False
    return bool(st.session_state.get("stats"))


def _gan_session_ket_qua_tu_video(v):
    """Gắn đầy đủ metadata kết quả đã lưu vào session (biểu đồ + video + frames)."""
    v = _lam_moi_ban_ghi_video_tu_db(v) or v
    st.session_state.current_eval_video = v
    st.session_state.stats = v.get("metrics")
    st.session_state.has_data = bool(v.get("metrics"))
    st.session_state.view_old_analysis = True
    st.session_state.processed_video_path = v.get("processed_path", v.get("video_path"))
    st.session_state.uploaded_file_name = v.get("video_name", "Video đã lưu")
    st.session_state.all_frames_data_path = v.get("all_frames_data_path")
    st.session_state.frames_zip = _frames_zip_path_from_video(v)
    st.session_state.current_df_csv_path = v.get("df_path")
    ex_base = next((BAI_TAP[k] for k in BAI_TAP if BAI_TAP[k]["ten"] == v.get("exercise")), BAI_TAP["codman"])
    st.session_state.exercise = ex_base.copy()
    if "sai_so" in v:
        st.session_state.exercise["chuan"] = ex_base["chuan"].copy()
        st.session_state.exercise["chuan"]["sai_so"] = v["sai_so"]
    return v


def _dong_bo_metadata_frames_vao_session(v=None, download=False):
    """Bo sung JSON/ZIP frames cho session khi ket qua cu co bieu do nhung metadata frame bi thieu."""
    src = v or st.session_state.get("current_eval_video") or {}
    src = _lam_moi_ban_ghi_video_tu_db(src) or src
    if not src:
        return src

    sess_v = st.session_state.get("current_eval_video") or {}
    same_slot = _slot_video_phan_tich(sess_v) == _slot_video_phan_tich(src)
    cur = dict(sess_v if same_slot else src)
    for key in ("processed_path", "video_path", "df_path", "metrics", "exercise", "video_name", "username", "full_name"):
        if src.get(key) and not cur.get(key):
            cur[key] = src.get(key)

    if src.get("all_frames_data_path") and not cur.get("all_frames_data_path"):
        cur["all_frames_data_path"] = src.get("all_frames_data_path")
    if src.get("df_path") and not cur.get("df_path"):
        cur["df_path"] = src.get("df_path")

    frames_path = (st.session_state.get("all_frames_data_path") if same_slot else None) or cur.get("all_frames_data_path")
    if not frames_path:
        for cand in _duong_dan_frames_json_candidates(cur):
            lp = get_local_frame_path(cand) or cand
            if is_local_file_ready(lp, min_size=2) or (download and ensure_local_file(cand, quiet=True, try_fallbacks=False)):
                frames_path = lp if is_local_file_ready(lp, min_size=2) else cand
                cur["all_frames_data_path"] = frames_path
                break

    if frames_path:
        st.session_state.all_frames_data_path = frames_path
        cur["all_frames_data_path"] = frames_path

    zip_path = _frames_zip_path_from_video(cur)
    if zip_path:
        st.session_state.frames_zip = zip_path
        cur["frames_zip"] = zip_path
        cur["frames_zip_path"] = zip_path
        if download:
            ensure_local_file(zip_path, quiet=True, try_fallbacks=False)

    proc = cur.get("processed_path") or st.session_state.get("processed_video_path")
    if proc:
        st.session_state.processed_video_path = proc
        if download:
            ensure_local_file(proc, quiet=True, try_fallbacks=True)

    st.session_state.current_eval_video = cur
    return cur


def tai_bieu_do_va_frames_tu_hf(v):
    """Tải CSV biểu đồ + JSON khung xương từ HF (không tải video/zip — tránh đơ nút)."""
    if not v:
        return v, False
    v = _lam_moi_ban_ghi_video_tu_db(v)
    if not v.get("df_path") and not v.get("all_frames_data_path"):
        _dong_bo_video_list_day_du_tu_hf(force=False)
        v = _lam_moi_ban_ghi_video_tu_db(v)
    paths = []
    for key in ("df_path", "all_frames_data_path"):
        p = v.get(key)
        if p:
            paths.append(p)
    for p in _duong_dan_csv_candidates(v):
        paths.append(p)
    for p in _duong_dan_frames_json_candidates(v):
        paths.append(p)
    seen, got = set(), False
    for p in paths:
        rel = get_clean_rel_path(p)
        if not rel or rel in seen:
            continue
        seen.add(rel)
        low = rel.lower()
        if any(x in low for x in (".mp4", ".mov", ".zip", "patient_uploads")):
            continue
        if ensure_local_file(p, quiet=True, try_fallbacks=False):
            got = True
    return _lam_moi_ban_ghi_video_tu_db(v), got


def tai_csv_bieu_do_tu_hf(v):
    """Alias — tải CSV + JSON frames."""
    return tai_bieu_do_va_frames_tu_hf(v)


_bg_load_jobs = {}


def _slot_tai_key(v):
    sk = _slot_video_phan_tich(v)
    if sk:
        return sk
    vp = (v or {}).get("video_path") or ""
    return hashlib.md5(vp.encode()).hexdigest()[:12] if vp else "default"


def _trang_thai_tai_media(v):
    """Kiểm tra biểu đồ / JSON frames / video / zip đã sẵn sàng local chưa."""
    v = _lam_moi_ban_ghi_video_tu_db(v) or v
    v = _dong_bo_metadata_frames_vao_session(v, download=False) or v
    proc = v.get("processed_path") or v.get("video_path")
    fj = v.get("all_frames_data_path") or st.session_state.get("all_frames_data_path")
    fz = _frames_zip_path_from_video(v) or st.session_state.get("frames_zip")
    csv_ok = st.session_state.get("angle_df") is not None
    if not csv_ok:
        for p in _duong_dan_csv_candidates(v) + ([v.get("df_path")] if v.get("df_path") else []):
            if p and is_local_file_ready(get_local_frame_path(p) or p, min_size=80):
                csv_ok = True
                break
    json_ok = False
    if fj:
        lp = get_local_frame_path(fj) or fj
        json_ok = is_local_file_ready(lp, min_size=2)
    vid_ok = bool(proc and is_local_file_ready(proc, min_size=50 * 1024))
    zip_ok = bool(fz and is_local_file_ready(get_local_frame_path(fz) or fz, min_size=1024)) or vid_ok
    job = _bg_load_jobs.get(_slot_tai_key(v), {})
    return {
        "csv": csv_ok,
        "json": json_ok,
        "video": vid_ok,
        "zip": zip_ok,
        "running": bool(job.get("running")),
    }


def _bat_dau_tai_day_du_song_song(v):
    """Tải song song CSV + JSON + video + ZIP — không block UI."""
    key = _slot_tai_key(v)
    job = _bg_load_jobs.get(key, {})
    if job.get("running"):
        return key
    _bg_load_jobs[key] = {
        "csv": False, "json": False, "video": False, "zip": False, "running": True,
    }
    st.session_state["_media_load_slot"] = key

    def _media_prefetch_worker(_v=v, _key=key):
        try:
            v_local = _lam_moi_ban_ghi_video_tu_db(_v) or _v
            _dong_bo_video_list_day_du_tu_hf(force=False)
            v_local = _lam_moi_ban_ghi_video_tu_db(v_local) or v_local

            csv_paths = list(dict.fromkeys(
                [p for p in ([v_local.get("df_path")] + _duong_dan_csv_candidates(v_local)) if p]
            ))
            for p in csv_paths:
                if ensure_local_file(p, quiet=True, try_fallbacks=False):
                    _bg_load_jobs[_key]["csv"] = True
                    break

            json_paths = list(dict.fromkeys(
                [p for p in ([v_local.get("all_frames_data_path")] + _duong_dan_frames_json_candidates(v_local)) if p]
            ))
            for p in json_paths:
                rel = get_clean_rel_path(p).lower()
                if any(x in rel for x in (".mp4", ".mov", ".zip", "patient_uploads")):
                    continue
                if ensure_local_file(p, quiet=True, try_fallbacks=False):
                    _bg_load_jobs[_key]["json"] = True
                    break

            proc = v_local.get("processed_path") or v_local.get("video_path")
            if proc and ensure_local_file(proc, try_fallbacks=True, quiet=True):
                dam_bao_tai_video_phan_tich(proc)
                _bg_load_jobs[_key]["video"] = True

            fz = _frames_zip_path_from_video(v_local)
            if fz and ensure_local_file(fz, quiet=True, try_fallbacks=False):
                _bg_load_jobs[_key]["zip"] = True
            if proc:
                check_and_extract_frames_zip(proc)
        except Exception as err:
            print(f"[ParallelLoad] {err}")
        finally:
            if _key in _bg_load_jobs:
                _bg_load_jobs[_key]["running"] = False

    threading.Thread(target=_media_prefetch_worker, daemon=True).start()
    return key


def _tai_video_frames_phan_tich_nen(v):
    """Tương thích cũ — chuyển sang tải song song."""
    _bat_dau_tai_day_du_song_song(v)


def _ap_dung_angle_df_tu_video(v):
    """Nạp angle_df từ file local/Cloud vào session."""
    df_loaded, src_loaded = _nap_angle_df_tu_video(v)
    if df_loaded is None:
        return False
    st.session_state.angle_df = df_loaded
    if src_loaded:
        st.session_state.current_df_csv_path = src_loaded
        v_upd = dict(_lam_moi_ban_ghi_video_tu_db(v) or v)
        if str(src_loaded).lower().endswith(".csv"):
            v_upd["df_path"] = src_loaded
        if str(src_loaded).lower().endswith(".json"):
            v_upd["all_frames_data_path"] = src_loaded
            st.session_state.all_frames_data_path = src_loaded
        st.session_state.current_eval_video = v_upd
    return True


def _nap_bieu_do_nhanh_tu_cloud(v, giu_phan_tich_moi=False):
    """Nạp kết quả đã lưu: biểu đồ ngay + video/frames tải song song liên tục."""
    v = _lam_moi_ban_ghi_video_tu_db(v)
    if not v or not v.get("metrics"):
        return False, v

    v = _gan_session_ket_qua_tu_video(v)
    _ap_dung_angle_df_tu_video(v)
    if st.session_state.get("angle_df") is not None:
        _gan_khoa_session_phan_tich(st.session_state.get("current_eval_video") or v)
    if not giu_phan_tich_moi:
        st.session_state.reanalyze_triggered = False
    _bat_dau_tai_day_du_song_song(v)
    return bool(st.session_state.get("angle_df") is not None), v


def _fragment_tien_do_tai_media(v, key_suffix=""):
    """Fragment 1s — cập nhật tiến độ tải Cloud + tự hiện biểu đồ khi CSV sẵn sàng."""
    vp = (v or {}).get("video_path")
    prog = read_progress(vp) if vp else None
    dang_phan_tich = bool(prog and prog.get("status") == "processing")
    status = _trang_thai_tai_media(v) if v else {}
    can_poll = (
        st.session_state.get("_media_load_slot")
        or status.get("running")
        or not all((status.get("csv"), status.get("json"), status.get("video"), status.get("zip")))
        or dang_phan_tich
    )
    interval = timedelta(seconds=1) if can_poll else None

    def _poll():
        if not v or not v.get("metrics"):
            return
        status = _trang_thai_tai_media(v)
        icons = (
            f"{'✅' if status['csv'] else '⏳'} Biểu đồ · "
            f"{'✅' if status['json'] else '⏳'} Frames · "
            f"{'✅' if status['video'] else '⏳'} Video · "
            f"{'✅' if status['zip'] or not _frames_zip_path_from_video(v) else '⏳'} ZIP"
        )
        if status["running"] or not all((status["csv"], status["json"], status["video"], status["zip"])):
            st.caption(f"☁️ **Đang tải từ Cloud:** {icons}")
        elif status["csv"] and status["json"] and status["video"] and status["zip"]:
            st.caption(f"✅ **Đã tải đủ:** {icons} — mở tab **🎬 VIDEO & ẢNH FRAME**.")

        if st.session_state.get("angle_df") is None and status["csv"]:
            if _ap_dung_angle_df_tu_video(v):
                st.session_state.view_old_analysis = True
                _gan_khoa_session_phan_tich(st.session_state.get("current_eval_video") or v)
                _lam_moi_giao_dien_sau_nut()

    _poll()


def _quay_lai_ket_qua_cu_da_luu(v, rerun=False):
    """Nạp kết quả đã lưu đầy đủ — biểu đồ ngay, video/frames tải song song."""
    global _hf_last_download_error
    _hf_last_download_error = None
    v = _lam_moi_ban_ghi_video_tu_db(v)
    if not v or not v.get("metrics"):
        st.error("❌ Không tìm thấy kết quả cũ cho video này.")
        thong_bao_loi_tai_hf()
        return False

    vp = v.get("video_path")
    prog = read_progress(vp) if vp else None
    dang_chay = bool(prog and prog.get("status") == "processing")

    st.session_state.view_old_analysis = True
    st.session_state.pop("_ncv_analysis_loaded_key", None)
    if not dang_chay:
        st.session_state.reanalyze_triggered = False
        st.session_state.pop("_analysis_started_this_session", None)

    ok, v = _nap_bieu_do_nhanh_tu_cloud(v, giu_phan_tich_moi=dang_chay)
    if not ok:
        if HF_TOKEN and HF_DATASET_ID:
            dong_bo_json_cau_hinh_tu_hf(force_files=frozenset({"video_list.json"}))
            _xoa_cache_sau_dong_bo_json(["video_list.json"])
        v = _gan_session_ket_qua_tu_video(_lam_moi_ban_ghi_video_tu_db(v) or v)
        _bat_dau_tai_day_du_song_song(v)
        ok = _ap_dung_angle_df_tu_video(v)
        if ok:
            _gan_khoa_session_phan_tich(st.session_state.get("current_eval_video") or v)

    if ok and st.session_state.get("angle_df") is not None:
        msg = "✅ Biểu đồ sẵn sàng! Video + frames đang tải liên tục từ Cloud..."
        if dang_chay:
            msg += " (phân tích mới vẫn chạy bên phải)"
        st.toast(msg, icon="📊")
        st.session_state._pending_chart_refresh = True
        _lam_moi_giao_dien_sau_nut()
        return True
    if st.session_state.get("stats"):
        st.warning("⚠️ Đang tải CSV/JSON từ Cloud — chờ vài giây, tiến độ hiện phía trên.")
        _bat_dau_tai_day_du_song_song(v)
        st.session_state.view_old_analysis = True
        _lam_moi_giao_dien_sau_nut()
        return False
    st.error("❌ Không tìm thấy kết quả cũ cho video này.")
    thong_bao_loi_tai_hf()
    return False


def _hien_thi_hang_video_va_tien_do(v, key_suffix, is_processing=False):
    """Hàng 2 cột: video gốc (trái) + luồng phân tích 4 bước (phải) — theo screenshot 3."""
    col_v1, col_v2 = st.columns([1.0, 1.0])
    with col_v1:
        video_name = v.get("original_filename") or v.get("video_path", "")
        hien_thi_video_goc_fragment(v, key_suffix, video_name=video_name)
    with col_v2:
        hien_thi_khu_vuc_phan_tich_chuyen_sau_fragment(v, key_suffix)


def _hien_thi_thong_bao_che_do_phan_tich_moi():
    st.info(
        "🔬 **Chế độ phân tích mới** — MediaPipe 33 landmarks, đối chiếu YouTube (REF), "
        "huấn luyện/nạp ML Classifier. **Bạn có thể chuyển sang tab khác** trong lúc chờ; "
        "kết quả sẽ **tự hiển thị biểu đồ** khi hoàn tất."
    )


def hien_thi_nut_tai_lai_va_phan_tich_moi(v_re, key_suffix=""):
    """Nút thao tác nhanh: chạy phân tích mới (hiển thị loading ngay khi bấm)."""
    if not v_re:
        st.warning("⚠️ Chưa chọn video. Vào danh sách bệnh nhân, chọn video rồi bấm **Phân tích**.")
        return
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "📂 Tải lại kết quả đã lưu",
            key=f"btn_reload_saved_{key_suffix}",
            type="secondary",
            use_container_width=True,
        ):
            _quay_lai_ket_qua_cu_da_luu(v_re, rerun=False)
    with c2:
        if st.button(
            "🚀 Chạy phân tích mới",
            key=f"btn_new_analysis_{key_suffix}",
            type="primary",
            use_container_width=True,
        ):
            _bat_che_do_cuu_ho_hf(v_re.get("video_path"))
            _xu_ly_ket_qua_khoi_dong_phan_tich(khoi_dong_phan_tich_lai_video(v_re, auto_start=True))


def render_video(video_path, check_h264=True, prefer_raw=False):
    """Hiển thị video: ưu tiên HTTP Range Request server (local) để phát ngay lập tức.
    prefer_raw=True: phát video gốc BN upload (danh sách video), không dùng processed/_f.mp4."""
    if not video_path:
        st.error("❌ File video không tồn tại hoặc đường dẫn trống.")
        return

    # Hiển thị thông báo nếu hệ thống đang tối ưu hóa định dạng ở nền
    if not prefer_raw:
        try:
            final_h264 = get_final_h264_path(video_path)
            if '_transcoding_jobs' in globals() and final_h264 in _transcoding_jobs:
                st.info("🔄 Hệ thống đang nén và tối ưu hóa định dạng video H.264 dưới nền để phát mượt mà trên trình duyệt. Vui lòng chờ 1-2 phút và tải lại trang...")
        except:
            pass

    # URL trực tiếp (YouTube, HF, ...)
    if isinstance(video_path, str) and (video_path.startswith('http://') or video_path.startswith('https://')):
        try:
            import streamlit.components.v1 as _stcomp
            import hashlib
            url_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
            video_src = safe_attr(video_path, max_length=2000)
            _stcomp.html(f"""
<!DOCTYPE html><html><head>
<style>
  body{{margin:0;padding:0;background:transparent;overflow:hidden;}}
  video{{width:100%;height:auto;max-height:520px;border-radius:8px;display:block;background:#000;object-fit:contain;}}
</style>
</head><body>
<video id="vp" controls preload="auto" playsinline>
  <source src="{video_src}" type="video/mp4">
  Trình duyệt không hỗ trợ video HTML5.
</video>
</body></html>
""", height=255)
        except Exception as e:
            st.error(f'⚠️ Lỗi hiển thị video: {e}')
        return

    if not prefer_raw:
        ensure_playable_video(video_path)
    _prefetch_video_quiet(video_path)

    # Video gốc BN: tải local + st.video trước (iframe Cloud hay bị màn đen với file tạm / codec lạ)
    if prefer_raw:
        for p in video_raw_only_paths(video_path):
            if _is_scratch_video_path(p):
                continue
            # Không gọi ensure_local_file — tránh download 750MB trong render loop mỗi 3s
            if _valid_raw_video_local(p) and _render_video_streamlit_native(p, allow_large=True):
                st.caption(f"📤 Video gốc BN — {os.path.basename(p)}")
                return True
        # Fallback khi raw không có local: thử _f.mp4 (H.264 đã tải về khi chạy phân tích)
        _h264_local = get_final_h264_path(_strip_to_original_upload(video_path))
        if _h264_local and _h264_local != video_path and not _is_scratch_video_path(_h264_local):
            if _valid_raw_video_local(_h264_local) and _render_video_streamlit_native(_h264_local, allow_large=True):
                st.caption(f"📤 Video gốc BN — {os.path.basename(_h264_local)}")
                return True

    # Ưu tiên phát local qua st.video() — hỗ trợ Range request, không bị màn đen trên HF Space
    if not prefer_raw:
        _local_first = find_ready_local_video(video_path)
        if not _local_first:
            # H.264 chưa xong — thử phát MP4 gốc (vẫn tốt hơn màn đen)
            _raw_local = get_local_frame_path(video_path) if isinstance(video_path, str) else None
            if _raw_local and os.path.exists(_raw_local) and os.path.getsize(_raw_local) > 0:
                _local_first = _raw_local
        if _local_first:
            # st.video() stream đúng với Range request; static iframe bị đen trên HF Space
            if _render_video_streamlit_native(_local_first, allow_large=True):
                return
            # MP4V (OpenCV codec) bị từ chối bởi _render_video_streamlit_native — phát thẳng với st.video()
            # Áp dụng cho mọi trường hợp file local tồn tại (không chỉ raw fallback)
            if os.path.exists(_local_first) and os.path.getsize(_local_first) > 5 * 1024:
                try:
                    with open(_local_first, "rb") as _rf:
                        st.video(_rf.read(), format="video/mp4")
                    st.caption(f"📹 {os.path.basename(_local_first)} — bấm **Chuẩn bị video H.264** để cải thiện chất lượng phát")
                    return
                except Exception:
                    pass

    # HF Space: stream Cloud khi chưa có bản local hợp lệ
    if _is_hf_runtime() and HF_TOKEN and HF_DATASET_ID:
        # prefer_raw: dùng optimistic=False để không render iframe hỏng khi file chưa có trên HF Dataset
        _cloud_optimistic = not prefer_raw
        if _try_render_cloud_video_stream(video_path, key_hint="hf_first", optimistic=_cloud_optimistic, prefer_raw=prefer_raw):
            return True
        # Fallback Cloud: khi raw không trên HF Dataset, stream _f.mp4 (H.264) thay thế
        if prefer_raw:
            _h264_cloud = get_final_h264_path(_strip_to_original_upload(video_path))
            if _h264_cloud and _h264_cloud != video_path:
                if _try_render_cloud_video_stream(_h264_cloud, key_hint="hf_h264_fb", optimistic=False, prefer_raw=False):
                    return True
            # Không tìm được video gốc ở đâu — hiện placeholder rõ ràng thay vì iframe hỏng
            try:
                _vprog = read_progress(video_path)
                _vstat = _vprog.get("status") if _vprog else None
            except Exception:
                _vstat = None
            if _vstat == "error":
                st.warning(
                    "⚠️ **Video gốc BN không còn trên server** — "
                    "vui lòng gắn lại hoặc tải lên video bên dưới."
                )
            else:
                st.info(
                    "⏳ **Video gốc BN chưa sẵn sàng** — sẽ tự hiện sau khi phân tích hoàn tất "
                    "và file được đồng bộ lên Cloud."
                )
            return False

    local_ready = None
    if prefer_raw:
        for p in video_raw_only_paths(video_path):
            if _is_scratch_video_path(p):
                continue
            if is_local_file_ready(p):
                local_ready = p
                break
    else:
        local_ready = find_ready_local_video(video_path)
    if not local_ready:
        if _try_render_cloud_video_stream(video_path, optimistic=True, prefer_raw=prefer_raw):
            return

    final_h264 = get_final_h264_path(video_path) if not prefer_raw else video_path
    is_local_h264 = False
    if os.path.exists(final_h264) and os.path.getsize(final_h264) >= 5 * 1024:
        try:
            mtime = os.path.getmtime(final_h264)
            size = os.path.getsize(final_h264)
            is_local_h264 = _check_video_valid_cached(final_h264, mtime, size)
        except:
            pass

    is_local_raw = False
    if os.path.exists(video_path) and os.path.getsize(video_path) >= 5 * 1024:
        try:
            mtime = os.path.getmtime(video_path)
            size = os.path.getsize(video_path)
            is_local_raw = _check_video_valid_cached(video_path, mtime, size)
        except:
            pass

    # Xác định đường dẫn thực tế phát (sau khi đã thử fallback _f / .mp4)
    target_path = None
    if prefer_raw:
        for p in video_raw_only_paths(video_path):
            if is_local_file_ready(p):
                target_path = p
                break
        if not target_path and is_local_raw:
            target_path = video_path
    else:
        ready_any = find_ready_local_video(video_path)
        if ready_any:
            h264_ready = get_final_h264_path(ready_any)
            if is_local_file_ready(h264_ready):
                target_path = h264_ready
            else:
                target_path = ensure_playable_video(ready_any) or ready_any
        elif is_local_h264:
            target_path = final_h264
        elif is_local_raw:
            target_path = ensure_playable_video(video_path)

    # 1. TRƯỜNG HỢP 1: Có sẵn file cục bộ (local)
    if target_path:
        # Nếu file raw chưa có H264 và là định dạng không tương thích,
        # chỉ tải server-side nếu có thể; không render URL riêng tư ra trình duyệt.
        if target_path == video_path and is_local_raw and not is_local_h264:
            v_codec = None
            try:
                v_codec, _ = get_video_codec(video_path)
            except:
                pass

            # Kiểm tra xem video gốc có tương thích trực tiếp với trình duyệt hay không (phải là h264 MP4)
            is_compatible = (v_codec == 'h264' and video_path.lower().endswith('.mp4'))
            if not is_compatible:
                # Trước Phase 1B proxy media, không stream URL riêng tư chứa token ra trình duyệt.
                if HF_TOKEN and HF_DATASET_ID:
                    try:
                        _rel = get_clean_rel_path(video_path)
                        _min_size = _hf_min_size_for_path(_rel)
                        _local_copy = _hf_download_dataset_file(_rel, quiet=True, min_size=_min_size)
                        if _local_copy and is_local_file_ready(_local_copy, min_size=_min_size):
                            st.info("🔄 Video đã được tải về cục bộ. Hệ thống đang tối ưu H.264 để phát an toàn.")
                            return
                    except Exception:
                        pass  # fallthrough to static serving below

                # Nếu không có cloud URL, hiện thông báo
                import hashlib as _hashlib
                safe_btn_key = f"reload_btn_{_hashlib.md5(video_path.encode()).hexdigest()[:8]}"
                st.warning("⏳ **Hệ thống đang nén video sang H.264. Vui lòng đợi 1-2 phút rồi nhấn F5.**")
                if st.button("🔄 Tải lại trang (F5)", key=safe_btn_key):
                    st.rerun()
                return

        if _try_render_cloud_video_stream(video_path, key_hint="local_fallback", optimistic=True, prefer_raw=prefer_raw):
            return
        if _render_video_streamlit_native(target_path, allow_large=True):
            return
        if not _is_hf_runtime() and _render_video_static_iframe(target_path):
            return
        st.warning(
            "⚠️ Không phát được video trực tiếp. "
            + ("Thử F5 hoặc kiểm tra file upload trên Dataset." if prefer_raw else
               "Bấm **📥 Tải video Tất cả (H.264)** bên dưới hoặc thử F5 sau vài giây.")
        )
        return

    if _try_render_cloud_video_stream(video_path, key_hint="fallback", optimistic=True, prefer_raw=prefer_raw):
        return

    st.warning("⚠️ File video đang được xử lý dưới nền hoặc không khả dụng.")

import threading
import queue
import gc
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

# MEDIAPIPE sẽ được load lazily khi cần xử lý video
mp_pose = None
mp_drawing = None
mp_drawing_styles = None

def setup_mediapipe_resources():
    """Thiết lập thư mục tài nguyên ảo để ghi đè mô hình của MediaPipe trên server cloud (read-only)"""
    import os
    import sys
    import shutil
    import tempfile
    import urllib.request

    # Chỉ chạy trên Linux (môi trường Streamlit Cloud) nơi venv bị read-only
    if sys.platform == "win32":
        return True

    try:
        import mediapipe as mp
        import mediapipe.python.solutions.download_utils as download_utils
        import mediapipe.python._framework_bindings.resource_util as resource_util
        mp_package_dir = os.path.dirname(mp.__file__)

        # Thư mục chứa tài nguyên ảo
        tmp_root = os.path.join(tempfile.gettempdir(), "mediapipe_virtual_resources")
        tmp_mp_dir = os.path.join(tmp_root, "mediapipe")

        # Chặn ghi đè set_resource_dir của SolutionBase bằng cách monkey-patching nó
        original_set_resource_dir = resource_util.set_resource_dir
        def custom_set_resource_dir(path):
            original_set_resource_dir(tmp_root)
        resource_util.set_resource_dir = custom_set_resource_dir

        # Nếu đã thiết lập rồi thì bỏ qua
        heavy_model_path = os.path.join(tmp_root, "mediapipe", "modules", "pose_landmark", "pose_landmark_heavy.tflite")
        if os.path.exists(heavy_model_path) and os.path.exists(os.path.join(tmp_mp_dir, "graphs")):
            resource_util.set_resource_dir(tmp_root)
            # Áp dụng monkey-patch download_oss_model để tránh download thừa
            def custom_download_oss_model(model_path: str):
                return
            download_utils.download_oss_model = custom_download_oss_model
            return True

        os.makedirs(tmp_mp_dir, exist_ok=True)

        # Link toàn bộ cấu trúc thư mục từ site-packages sang tmp_mp_dir
        for item in os.listdir(mp_package_dir):
            if item == "__pycache__":
                continue
            src_path = os.path.join(mp_package_dir, item)
            dst_path = os.path.join(tmp_mp_dir, item)

            if item == "modules":
                os.makedirs(dst_path, exist_ok=True)
                src_modules_dir = os.path.join(mp_package_dir, "modules")
                for sub_item in os.listdir(src_modules_dir):
                    src_sub = os.path.join(src_modules_dir, sub_item)
                    dst_sub = os.path.join(dst_path, sub_item)
                    if sub_item == "pose_landmark":
                        os.makedirs(dst_sub, exist_ok=True)
                        src_pose_dir = os.path.join(src_modules_dir, "pose_landmark")
                        for file_item in os.listdir(src_pose_dir):
                            if file_item == "__pycache__":
                                continue
                            src_file = os.path.join(src_pose_dir, file_item)
                            dst_file = os.path.join(dst_sub, file_item)
                            if os.path.isdir(src_file):
                                continue
                            if os.path.exists(dst_file) or os.path.islink(dst_file):
                                continue
                            os.symlink(src_file, dst_file)
                    else:
                        if os.path.exists(dst_sub) or os.path.islink(dst_sub):
                            continue
                        os.symlink(src_sub, dst_sub)
            else:
                if os.path.exists(dst_path) or os.path.islink(dst_path):
                    continue
                os.symlink(src_path, dst_path)

        # Đặt resource dir của MediaPipe sang thư mục ảo
        resource_util.set_resource_dir(tmp_root)

        # Monkey-patch download_oss_model để chuyển hướng tải mô hình sang thư mục ảo
        def custom_download_oss_model(model_path: str):
            virtual_file_path = os.path.join(tmp_root, model_path)
            if os.path.exists(virtual_file_path):
                return

            gcs_url = "https://storage.googleapis.com/mediapipe-assets/" + model_path.split('/')[-1]
            os.makedirs(os.path.dirname(virtual_file_path), exist_ok=True)
            with urllib.request.urlopen(gcs_url) as response, open(virtual_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        download_utils.download_oss_model = custom_download_oss_model
        return True
    except Exception as e:
        st.warning(f"⚠️ Không thể thiết lập thư mục tài nguyên ảo cho MediaPipe: {e}")
        return False

def init_mediapipe():
    """Load MediaPipe chỉ khi cần thiết (lazy import)"""
    global mp_pose, mp_drawing, mp_drawing_styles
    if mp_pose is None:
        try:
            # Thiết lập tài nguyên ảo trước khi import solutions
            setup_mediapipe_resources()

            import mediapipe as mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            return True
        except Exception as e:
            st.error(f"🚨 Không thể khởi tạo MediaPipe: {e}")
            return False
    return True

# ============================================
# HỖ TRỢ MÚI GIỜ VIỆT NAM (ICT - UTC+7)
# ============================================
def get_vn_now():
    """Lấy thời gian hiện tại theo múi giờ Việt Nam (ICT - UTC+7)"""
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))

# ============================================
# QUẢN LÝ NGƯỜI DÙNG & BẢO MẬT
# ============================================
# --- HỖ TRỢ LƯU TRỮ BỀN VỮNG TRÊN HUGGING FACE SPACES (PERSISTENT STORAGE) ---
import shutil

DATA_DIR = "."
# Phát hiện /data một cách an toàn - tránh xung đột với hf-mount
try:
    _data_ok = (
        os.path.isdir("/data") and
        os.access("/data", os.W_OK) and
        os.access("/data", os.R_OK)
    )
    if _data_ok:
        DATA_DIR = "/data"
except Exception:
    _data_ok = False

DB_DIR = "database" if DATA_DIR == "." else DATA_DIR

USER_DATA_FILE = os.path.join(DB_DIR, "users.json")
SESSION_STATE_FILE = os.path.join(DB_DIR, "session_state.json")
SYMPTOMS_FILE = os.path.join(DB_DIR, "patient_symptoms.json")
EVALUATIONS_FILE = os.path.join(DB_DIR, "doctor_evaluations.json")
REMINDERS_FILE = os.path.join(DB_DIR, "schedules.json")
VIDEOS_FILE = os.path.join(DB_DIR, "video_list.json")
RESEARCH_DATA_FILE = os.path.join(DB_DIR, "research_data.json")
HISTORY_FILE = os.path.join(DB_DIR, "lich_su_tap_luyen.json")
FEEDBACK_FILE = os.path.join(DB_DIR, "phan_hoi.json")
UPLOAD_DIR = os.path.join(DATA_DIR, "patient_uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_results")
RUNTIME_DIRS = (DB_DIR, UPLOAD_DIR, PROCESSED_DIR)


_HF_JSON_DOWNLOAD_FILES = set(HF_JSON_DOWNLOAD_FILES)
_HF_MODEL_ARTIFACT_FILES = set(HF_MODEL_ARTIFACT_FILES)


def _hf_path_policy():
    return HfPathPolicy(
        data_dir=DATA_DIR,
        upload_dir=UPLOAD_DIR,
        processed_dir=PROCESSED_DIR,
        db_dir=DB_DIR,
        json_files=frozenset(_HF_JSON_DOWNLOAD_FILES),
        model_artifact_files=frozenset(_HF_MODEL_ARTIFACT_FILES),
    )


def _data_allowed_roots():
    return data_allowed_roots(_hf_path_policy())


def _dataset_rel_path_from_local(path):
    return dataset_rel_path_from_local(path, _hf_path_policy())


def _hf_download_target_for_rel_path(rel_path):
    return hf_download_target_for_rel_path(rel_path, _hf_path_policy())


def _hf_upload_rel_path_for_local(local_path):
    return hf_upload_rel_path_for_local(local_path, _hf_path_policy())


EXTRACTED_FRAMES_DIR = "extracted_frames"
OUTPUT_VIDEOS_DIR = "output_videos"


# --- TỰ ĐỘNG ĐỒNG BỘ DỮ LIỆU SANG HUGGING FACE DATASET (MIỄN PHÍ - BỀN VỮNG) ---
import threading

def _get_secret(key, default=""):
    val = os.environ.get(key, "").strip()
    if not val:
        try:
            val = (st.secrets.get(key) or "").strip()
        except Exception:
            pass
    return val or default

HF_TOKEN = _get_secret("HF_TOKEN") or None
HF_SPACE_ID = (_get_secret("HF_SPACE_ID") or _get_secret("SPACE_ID")).strip() or None
HF_DATASET_ID = _get_secret("HF_DATASET_ID") or (f"{HF_SPACE_ID}-data" if HF_SPACE_ID else None)
FRONTEND_API_CONFIG = FrontendApiConfig.from_env()
ALLOW_NETWORK_TTS = _get_secret("ALLOW_NETWORK_TTS", "false").lower() in {"1", "true", "yes"}
WEBRTC_STUN_URLS = [
    url.strip()
    for url in _get_secret("WEBRTC_STUN_URLS", "").split(",")
    if url.strip()
]

_hf_dataset_access_cache = {"ok": None, "msg": None, "fp": None}
_hf_last_download_error = None


def _hf_min_size_for_path(path):
    """Ngưỡng kích thước tối thiểu theo loại file — CSV/JSON nhỏ vẫn hợp lệ."""
    return hf_min_size_for_path(path)


def _hf_token_fingerprint():
    return hf_token_fingerprint(HF_TOKEN, HF_DATASET_ID)


def _lam_sach_cache_khi_doi_hf_token():
    """Xóa cờ auto-restore khi HF_TOKEN / HF_DATASET_ID thay đổi để cho phép tải lại dữ liệu cũ."""
    fp = _hf_token_fingerprint()
    if st.session_state.get("_hf_fp") == fp:
        return
    for k in list(st.session_state.keys()):
        if k.startswith("_auto_restored_"):
            st.session_state.pop(k, None)
    st.session_state["_hf_fp"] = fp
    global _hf_dataset_access_cache
    _hf_dataset_access_cache = {"ok": None, "msg": None, "fp": None}


def _hf_la_loi_thu_vien(err_text):
    return is_hf_library_error(err_text)


def _hf_verify_dataset_via_http():
    """Kiểm tra token + Dataset bằng HTTP (không cần huggingface_hub)."""
    return verify_dataset_via_http(HF_TOKEN, HF_DATASET_ID)


def _hf_download_via_http(rel_path, min_size=80, quiet=False):
    """Tải file Dataset qua HTTP — dự phòng khi huggingface_hub lỗi phiên bản."""
    global _hf_last_download_error
    target, err = download_dataset_file_via_http(
        rel_path,
        token=HF_TOKEN,
        dataset_id=HF_DATASET_ID,
        policy=_hf_path_policy(),
        min_size=min_size,
    )
    _hf_last_download_error = err
    if err and not quiet:
        silent_errors = (
            "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID.",
            "Token không có quyền tải file từ Dataset.",
        )
        if err == "Đường dẫn cloud không hợp lệ.":
            print(f"[HF Sync] Reject unsafe dataset path {rel_path}: {err}")
        elif not (err in silent_errors or str(err).startswith("Chưa có trên Dataset:") or "kích thước không hợp lệ" in str(err)):
            print(f"[HF Sync] HTTP fallback loi {rel_path}: {err}")
    return target


def kiem_tra_quyen_hf_dataset(force=False):
    """Kiểm tra token hiện tại có đọc được Dataset lưu trữ dữ liệu cũ hay không."""
    global _hf_dataset_access_cache
    fp = _hf_token_fingerprint()
    if (
        not force
        and _hf_dataset_access_cache.get("fp") == fp
        and _hf_dataset_access_cache.get("ok") is not None
    ):
        return _hf_dataset_access_cache["ok"], _hf_dataset_access_cache.get("msg")

    if not HF_TOKEN:
        msg = "Chưa cấu hình HF_TOKEN trong Space Secrets."
        _hf_dataset_access_cache = {"ok": False, "msg": msg, "fp": fp}
        return False, msg
    if not HF_DATASET_ID:
        msg = "Chưa cấu hình HF_DATASET_ID."
        _hf_dataset_access_cache = {"ok": False, "msg": msg, "fp": fp}
        return False, msg

    ok_repo, hub_err = hf_repo_info(HF_TOKEN, HF_DATASET_ID)
    if ok_repo:
        _hf_dataset_access_cache = {"ok": True, "msg": None, "fp": fp}
        return True, None

    err = str(hub_err or "").lower()
    if _hf_la_loi_thu_vien(err):
        ok_http, msg_http = _hf_verify_dataset_via_http()
        if ok_http:
            _hf_dataset_access_cache = {
                "ok": True,
                "msg": "Đồng bộ qua HTTP (huggingface_hub trên Space cần rebuild).",
                "fp": fp,
            }
            return True, _hf_dataset_access_cache["msg"]
        _hf_dataset_access_cache = {"ok": False, "msg": msg_http, "fp": fp}
        return False, msg_http
    if is_hf_auth_error(err):
        msg = (
            f"Token không có quyền đọc Dataset `{HF_DATASET_ID}`. "
            "Hãy dùng token có phạm vi tối thiểu hoặc thêm tài khoản chạy app làm collaborator."
        )
    elif is_hf_not_found_error(err):
        msg = (
            f"Không tìm thấy Dataset `{HF_DATASET_ID}`. "
            "Kiểm tra biến HF_DATASET_ID trong env hoặc Streamlit secrets."
        )
    else:
        ok_http, msg_http = _hf_verify_dataset_via_http()
        if ok_http:
            _hf_dataset_access_cache = {"ok": True, "msg": None, "fp": fp}
            return True, None
        msg = f"Không kết nối được Dataset: {hub_err}"
    _hf_dataset_access_cache = {"ok": False, "msg": msg, "fp": fp}
    return False, msg


def thong_bao_loi_tai_hf():
    """Thông báo lỗi thân thiện khi không tải được file từ Hugging Face Dataset."""
    ok, msg = kiem_tra_quyen_hf_dataset()
    if not ok and msg:
        st.error(f"🔐 **Lỗi đồng bộ Cloud:** {msg}")
        return
    if _hf_last_download_error:
        st.warning(f"☁️ **Không tải được file phân tích từ Cloud:** {_hf_last_download_error}")
    elif ok:
        st.info(
            "☁️ Cloud Sync đã kết nối nhưng file CSV/JSON của **video đang chọn** chưa tải được từ Dataset. "
            "Bấm **Tải lại kết quả đã lưu** để thử tải lại từ Cloud, hoặc chạy **Phân tích mới** nếu file chưa từng upload lên Dataset."
        )

def khoi_tao_dong_bo_hf():
    """Tải tất cả dữ liệu từ Hugging Face Dataset về đĩa khi khởi động (chạy trong background thread - an toàn với hf-mount)"""
    if not HF_TOKEN or not HF_DATASET_ID:
        # Kể cả không có HF_TOKEN, vẫn sao chép file mặc định sang /data nếu cần
        if DATA_DIR == "/data":
            _files_to_persist = [
                "patient_symptoms.json", "doctor_evaluations.json",
                "schedules.json", "video_list.json", "research_data.json",
                "lich_su_tap_luyen.json", "phan_hoi.json"
            ]
            for _f in _files_to_persist:
                _dst = os.path.join(DATA_DIR, _f)
                _src = os.path.join("database", _f)
                if not os.path.exists(_dst) and os.path.exists(_src):
                    try:
                        shutil.copy2(_src, _dst)
                    except:
                        pass
        return

    try:
        # 1. Tạo repo dataset riêng tư nếu chưa tồn tại (không block luồng nếu Token thiếu quyền create_repo)
        ok_create, create_err = ensure_dataset_repo(HF_TOKEN, HF_DATASET_ID)
        if not ok_create and create_err:
            print(f"[HF Sync] Bỏ qua lỗi tạo repo (có thể do Token thiếu quyền create, nhưng repo đã tồn tại): {create_err}")

        # 2. Tải các file cấu hình về máy
        files_to_download = [
            "patient_symptoms.json",
            "doctor_evaluations.json",
            "schedules.json",
            "video_list.json",
            "research_data.json",
            "lich_su_tap_luyen.json",
            "phan_hoi.json"
        ]

        for f_name in files_to_download:
            try:
                if _hf_download_dataset_file(f_name, quiet=True, min_size=2):
                    print(f"[HF Sync] Đã tải về: {f_name}")
            except Exception:
                pass

        # 3. Không tải hàng loạt patient_uploads/processed_results lúc khởi động.
        # Các file video/frames/CSV lớn được lazy-load qua ensure_local_file() khi thật sự cần,
        # giúp đăng nhập và mở kết quả cũ nhanh hơn rất nhiều trên HF Space.
    except Exception as e:
        print(f"[HF Sync] Lỗi khởi động đồng bộ: {e}")

_hf_upload_queue = []
_hf_upload_queue_lock = threading.Lock()
_hf_upload_worker_started = False
_hf_upload_backoff_until = 0.0
_hf_last_upload_at = 0.0
_hf_min_upload_interval = 45.0
_hf_rate_limit_logged = False


def _hf_start_upload_worker():
    global _hf_upload_worker_started
    if _hf_upload_worker_started:
        return
    _hf_upload_worker_started = True

    def _hf_upload_queue_worker():
        global _hf_last_upload_at, _hf_upload_backoff_until, _hf_rate_limit_logged
        while True:
            try:
                now = time.time()
                if now < _hf_upload_backoff_until:
                    time.sleep(15)
                    continue
                item = None
                with _hf_upload_queue_lock:
                    if _hf_upload_queue:
                        _hf_upload_queue.sort(key=lambda x: x[0])
                        item = _hf_upload_queue.pop(0)
                if not item:
                    time.sleep(2)
                    continue
                _, local_path = item
                if not local_path or not os.path.exists(local_path):
                    continue
                wait = _hf_min_upload_interval - (time.time() - _hf_last_upload_at)
                if wait > 0:
                    time.sleep(wait)
                try:
                    rel_path, upload_err = upload_dataset_file(
                        local_path,
                        token=HF_TOKEN,
                        dataset_id=HF_DATASET_ID,
                        policy=_hf_path_policy(),
                    )
                    if upload_err:
                        raise RuntimeError(upload_err)
                    _hf_last_upload_at = time.time()
                    _hf_rate_limit_logged = False
                    print(f"[HF Sync] Đã đẩy lên Dataset: {rel_path}")
                except Exception as e:
                    err = str(e).lower()
                    if any(x in err for x in ("429", "too many requests", "rate limit", "exceeded")):
                        _hf_upload_backoff_until = time.time() + 3600
                        if not _hf_rate_limit_logged:
                            print(
                                "[HF Sync] Đã chạm giới hạn 128 commit/giờ của Hugging Face — "
                                "tạm dừng đẩy Dataset ~1 giờ. Phân tích vẫn chạy bình thường trên Space."
                            )
                            _hf_rate_limit_logged = True
                    else:
                        print(f"[HF Sync] Lỗi đẩy file {local_path}: {e}")
            except Exception as loop_err:
                print(f"[HF Sync] Worker lỗi: {loop_err}")
                time.sleep(5)

    threading.Thread(target=_hf_upload_queue_worker, daemon=True).start()


def push_file_to_hf_async(local_path, priority=5):
    """Xếp hàng đẩy file lên HF Dataset — tối đa ~1 commit/45s, tránh lỗi 429."""
    global _hf_upload_queue
    if not HF_TOKEN or not HF_DATASET_ID or not local_path:
        return
    try:
        _hf_upload_rel_path_for_local(local_path)
    except PathSecurityError as exc:
        print(f"[HF Sync] Bo qua upload path khong duoc phep: {exc}")
        return
    _hf_start_upload_worker()
    norm = os.path.normpath(local_path)
    with _hf_upload_queue_lock:
        _hf_upload_queue = [x for x in _hf_upload_queue if x[1] != norm]
        _hf_upload_queue.append((int(priority), norm))

def _hf_download_dataset_file(rel_path, quiet=False, min_size=None):
    """Tải một file từ HF Dataset về DATA_DIR. Trả về đường dẫn local nếu thành công."""
    global _hf_last_download_error
    if not (HF_TOKEN and HF_DATASET_ID and rel_path):
        _hf_last_download_error = "Chưa cấu hình HF_TOKEN hoặc HF_DATASET_ID."
        return None
    try:
        _target, _local_dir, rel_norm = _hf_download_target_for_rel_path(rel_path)
    except PathSecurityError as exc:
        _hf_last_download_error = "Đường dẫn cloud không hợp lệ."
        if not quiet:
            print(f"[HF Sync] Reject unsafe dataset path {rel_path}: {exc}")
        return None
    if min_size is None:
        min_size = _hf_min_size_for_path(rel_norm)
    got, err, hub_err = download_dataset_file(
        rel_norm,
        token=HF_TOKEN,
        dataset_id=HF_DATASET_ID,
        policy=_hf_path_policy(),
        min_size=min_size,
    )
    if got:
        _hf_last_download_error = None
        return got
    _hf_last_download_error = err
    err_text = str(err or "")
    hub_text = str(hub_err or "")
    if err == "Token không có quyền tải file từ Dataset.":
        _, msg = kiem_tra_quyen_hf_dataset(force=True)
        _hf_last_download_error = msg or err
        if not quiet:
            print(f"[HF Sync] Token khong du quyen tai {rel_norm}: {hub_err or err}")
    elif _hf_la_loi_thu_vien(hub_text):
        if not quiet:
            print(f"[HF Sync] huggingface_hub loi phien ban, da thu HTTP: {hub_err}")
    elif err_text.startswith("Chưa có trên Dataset:"):
        if not quiet:
            print(f"[HF Sync] Chua co tren Dataset: {rel_norm}")
    elif err and not quiet:
        print(f"[HF Sync] loi tai {rel_norm}: {err}")
    return None


def ensure_local_file(file_path, quiet=False, try_fallbacks=True):
    """Đảm bảo file tồn tại cục bộ. Thử _f.mp4 rồi .mp4 gốc — không báo lỗi đỏ khi _f chua upload."""
    global _hf_last_download_error
    if not file_path:
        return False
    _low_fp = str(file_path).lower()
    if _low_fp.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")) and _la_duong_dan_video_gia(file_path):
        return False

    paths = video_fallback_paths(file_path) if try_fallbacks else [file_path]
    safe_paths = []
    for fp in paths:
        try:
            rel_fp = get_clean_rel_path(fp)
            safe_fp = safe_data_path(rel_fp, [UPLOAD_DIR, PROCESSED_DIR], base_dir=DATA_DIR)
            if safe_fp not in safe_paths:
                safe_paths.append(safe_fp)
        except PathSecurityError as exc:
            if not quiet:
                print(f"[HF Sync] Reject unsafe local candidate {fp}: {exc}")
        except Exception as exc:
            if not quiet:
                print(f"[HF Sync] Khong resolve duoc local candidate {fp}: {exc}")
    paths = safe_paths
    if not paths:
        _hf_last_download_error = "Đường dẫn file không nằm trong thư mục dữ liệu được phép."
        return False

    for fp in paths:
        if is_local_file_ready(fp):
            return True
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    if not (HF_TOKEN and HF_DATASET_ID):
        _hf_last_download_error = "Chưa cấu hình HF_TOKEN — không thể tải dữ liệu cũ từ Cloud."
        return False

    for fp in paths:
        rel_path = get_clean_rel_path(fp)
        min_sz = _hf_min_size_for_path(rel_path)
        got = _hf_download_dataset_file(rel_path, quiet=True, min_size=min_sz)
        if got and is_local_file_ready(got, min_size=min_sz):
            return True
        if is_local_file_ready(fp, min_size=min_sz):
            return True

    if not quiet:
        print(f"[HF Sync] Khong tim thay tren Dataset (da thu fallback): {file_path}")
    return False

def get_local_frame_path(stored_path):
    """Chuyển đổi đường dẫn frame được lưu trữ (có thể là Windows/Linux/Tuyệt đối)
    thành đường dẫn chính xác và hợp lệ trên OS hiện tại dưới DATA_DIR."""
    if not stored_path:
        return ""
    rel_path = get_clean_rel_path(stored_path)
    if not rel_path:
        return ""
    try:
        return safe_data_path(rel_path, [UPLOAD_DIR, PROCESSED_DIR], base_dir=DATA_DIR)
    except PathSecurityError:
        return ""

def is_local_file_ready(file_path, min_size=5 * 1024):
    """Kiểm tra file local có sẵn mà không tải từ cloud."""
    return bool(file_path and os.path.exists(file_path) and os.path.getsize(file_path) >= min_size)

def _safe_extract_frames_zip(zip_path, frames_dir):
    return safe_extract_frames_zip(zip_path, frames_dir)

def check_and_extract_frames_zip(processed_video_path):
    """Kiểm tra và tự động tải/giải nén file ZIP chứa các frame ảnh nếu thư mục ảnh chưa có.
    Hàm này giúp tải một lần duy nhất tất cả frames cực nhanh từ Cloud về thay vì trích xuất từng frame từ video."""
    if not processed_video_path:
        return

    filename = os.path.basename(processed_video_path)
    import re
    match = re.search(r'processed_(\d+)', filename)
    if not match:
        return
    timestamp = match.group(1)

    frames_dir = os.path.join(PROCESSED_DIR, f"processed_{timestamp}_frames")
    zip_path = os.path.join(PROCESSED_DIR, f"processed_{timestamp}_frames.zip")

    # Nếu thư mục frames đã tồn tại cục bộ và chứa ảnh, không cần làm gì thêm
    if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
        try:
            if sum(1 for _ in zip(range(6), os.scandir(frames_dir))) > 5:
                return
        except OSError as exc:
            print(f"[Frames Extract] Khong doc duoc thu muc frames {frames_dir}: {exc}")

    # Đảm bảo file ZIP tồn tại cục bộ (nếu chưa có, tự tải về từ Hugging Face Dataset)
    if not (os.path.exists(zip_path) and os.path.getsize(zip_path) >= 5 * 1024):
        ensure_local_file(zip_path, try_fallbacks=False)

    # Nếu đã có file ZIP cục bộ, tiến hành giải nén ra thư mục frames_dir
    if os.path.exists(zip_path) and os.path.getsize(zip_path) >= 5 * 1024:
        try:
            os.makedirs(frames_dir, exist_ok=True)
            _safe_extract_frames_zip(zip_path, frames_dir)
            print(f"[Frames Extract] Giải nén thành công {os.path.basename(zip_path)} vào {frames_dir}")
        except Exception as e:
            print(f"[Frames Extract] Lỗi giải nén ZIP: {e}")

@st.cache_data(show_spinner=False)
def _load_data_cached(file_path, mtime):
    result = read_app_json(file_path)
    for line in format_schema_issue_lines(file_path, result.issues):
        print(line)
    return result.data

def load_data(file_path):
    if os.path.exists(file_path):
        try:
            mtime = os.path.getmtime(file_path)
            data = _load_data_cached(file_path, mtime)
            result = normalize_app_json(file_path, data)
            if result.changed:
                write_app_json(file_path, result.data)
                _load_data_cached.clear()
                return result.data
            return data
        except OSError as err:
            print(f"[Data] Khong the doc metadata {file_path}: {err}")
    result = read_app_json(file_path)
    for line in format_schema_issue_lines(file_path, result.issues):
        print(line)
    return result.data

def save_data(file_path, data):
    result = write_app_json(file_path, data)
    for line in format_schema_issue_lines(file_path, result.issues):
        print(line)
    data = result.data
    if not result.ok:
        print(f"[Data] Khong the ghi {file_path}")
        try:
            st.warning("⚠️ File dữ liệu đang bị khóa tạm thời. Vui lòng bấm lại sau vài giây.")
        except Exception:
            pass
        return False
    try:
        _load_data_cached.clear()
        _load_video_list_core.clear()
        _video_nghien_cuu_cached.clear()
        _evals_dedup_cached.clear()
    except Exception:
        pass
    # Tự động đẩy file dữ liệu lên Hugging Face Dataset
    push_file_to_hf_async(file_path)
    return True


def get_current_actor():
    info = st.session_state.get("user_info") or {}
    return {
        "username": info.get("username") or "anonymous",
        "role": info.get("role") or "anonymous",
        "full_name": info.get("full_name") or info.get("username") or "anonymous",
    }


def write_audit_log(actor, role, action, target, result, metadata=None):
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        entry = {
            "time": get_vn_now().isoformat(),
            "actor": actor,
            "role": role,
            "action": action,
            "target": str(target),
            "result": result,
            "metadata": metadata or {},
        }
        with open(os.path.join(DB_DIR, "audit_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[Audit] Khong the ghi audit log {action}/{target}: {exc}")


def _clear_authenticated_session():
    for key in list(st.session_state.keys()):
        if key != "theme":
            del st.session_state[key]
    st.session_state.logged_in = False
    st.session_state.user_info = None


def revoke_all_sessions(actor=None, reason="manual"):
    actor_info = actor or get_current_actor()
    new_version = bump_global_session_version(
        SESSION_STATE_FILE,
        actor=actor_info.get("username", "system"),
        reason=reason,
    )
    write_audit_log(
        actor_info.get("username", "system"),
        actor_info.get("role", "system"),
        "session_revoke_all",
        "global",
        "success",
        {"global_session_version": new_version, "reason": reason},
    )
    return new_version


def _current_session_is_valid():
    info = st.session_state.get("user_info") or {}
    return session_is_current(SESSION_STATE_FILE, info.get("session_version"))


def require_role(*allowed_roles, action=None, target=None):
    actor = get_current_actor()
    try:
        return require_actor_role(actor, allowed_roles)
    except PermissionError:
        write_audit_log(actor["username"], actor["role"], action or "permission_check", target or "", "denied")
        raise PermissionError("Bạn không có quyền thực hiện thao tác này.")


def require_patient_scope(patient_username, *, action=None):
    actor = get_current_actor()
    if not patient_username:
        write_audit_log(actor["username"], actor["role"], action or "patient_scope", "", "denied_missing_patient")
        raise PermissionError("Thiếu bệnh nhân đích cho thao tác này.")
    if not permission_actor_can_access_patient(actor, patient_username, load_users()):
        write_audit_log(actor["username"], actor["role"], action or "patient_scope", patient_username, "denied_out_of_scope")
        raise PermissionError("Bạn không có quyền thao tác trên bệnh nhân này.")
    return actor


def scope_records_for_current_actor(records, users=None):
    """Filter patient-scoped records at data-access call sites."""
    actor = get_current_actor()
    users = users if users is not None else load_users()
    return permission_scope_records_for_actor(records or [], actor, users)


def scope_patient_usernames_for_current_actor(usernames, users=None):
    actor = get_current_actor()
    users = users if users is not None else load_users()
    return permission_scope_patient_usernames_for_actor(usernames, actor, users)


def current_actor_can_access_patient(patient_username, users=None):
    actor = get_current_actor()
    users = users if users is not None else load_users()
    return permission_actor_can_access_patient(actor, patient_username, users)


def researcher_view_records(records):
    actor = get_current_actor()
    return researcher_view_records_for_actor(records or [], actor)


def patient_display_label(record, *, include_username=True):
    """Return a role-aware patient label without exposing PII to researchers."""
    actor = get_current_actor()
    return patient_display_label_for_actor(record, actor, include_username=include_username)


def create_backup_before_destructive(action, files):
    actor = get_current_actor()
    stamp = get_vn_now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backups", "destructive", f"{stamp}_{action}")
    os.makedirs(backup_dir, exist_ok=True)
    backed_up = []
    for src in files or []:
        if not src or not os.path.exists(src):
            continue
        try:
            src_real = os.path.realpath(os.path.abspath(src))
            base = os.path.basename(src_real.rstrip(os.sep)) or "backup"
            dst = os.path.join(backup_dir, base)
            if os.path.isdir(src_real):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src_real, dst)
            else:
                shutil.copy2(src_real, dst)
            backed_up.append(src_real)
        except Exception as exc:
            write_audit_log(actor["username"], actor["role"], action, src, "backup_failed", {"error": str(exc)})
            raise
    write_audit_log(actor["username"], actor["role"], action, backup_dir, "backup_created", {"files": backed_up})
    return backup_dir


def _remove_files_in_dir(folder):
    if not folder or not os.path.isdir(folder):
        return
    root = os.path.realpath(os.path.abspath(folder))
    for name in os.listdir(root):
        target = os.path.realpath(os.path.abspath(os.path.join(root, name)))
        try:
            if os.path.commonpath([target, root]) != root:
                continue
            if os.path.isfile(target) or os.path.islink(target):
                os.remove(target)
        except OSError as exc:
            write_audit_log("system", "system", "remove_file", target, "failed", {"error": str(exc)})


HF_JSON_CONFIG_FILES = [
    "patient_symptoms.json", "doctor_evaluations.json",
    "schedules.json", "video_list.json", "research_data.json",
    "lich_su_tap_luyen.json", "phan_hoi.json",
]


def _json_dst_mtime(f_name):
    dst = os.path.join(DB_DIR, f_name)
    try:
        return os.path.getmtime(dst) if os.path.exists(dst) and os.path.getsize(dst) > 2 else 0
    except Exception:
        return 0


def _xoa_cache_sau_dong_bo_json(files):
    """Chỉ xóa cache JSON khi file thực sự thay đổi sau đồng bộ HF."""
    try:
        _load_data_cached.clear()
        if any(f == "video_list.json" for f in files):
            _load_video_list_core.clear()
            _video_nghien_cuu_cached.clear()
        if "doctor_evaluations.json" in files:
            _evals_dedup_cached.clear()
    except Exception:
        pass


def _dong_bo_video_list_day_du_tu_hf(force=False):
    """Làm mới video_list.json từ HF — lấy df_path / metrics đầy đủ cho 8 video đã phân tích."""
    if not (HF_TOKEN and HF_DATASET_ID):
        return False
    try:
        if not force and st.session_state.get("_video_list_full_sync"):
            return True
    except Exception:
        pass
    dst = os.path.join(DB_DIR, "video_list.json")
    local_ok = os.path.exists(dst) and os.path.getsize(dst) > 2
    if force or not local_ok:
        dong_bo_json_cau_hinh_tu_hf(force_files=frozenset({"video_list.json"}))
        _xoa_cache_sau_dong_bo_json(["video_list.json"])
    try:
        st.session_state["_video_list_full_sync"] = True
    except Exception:
        pass
    return True


def tai_tep_phan_tich_tu_hf(v):
    """Tải CSV/JSON/video/frames phân tích từ HF Dataset cho đúng video đang chọn."""
    if not v:
        return v, False
    v = _lam_moi_ban_ghi_video_tu_db(v)
    if not v.get("df_path") and not v.get("all_frames_data_path"):
        _dong_bo_video_list_day_du_tu_hf(force=True)
        v = _lam_moi_ban_ghi_video_tu_db(v)
    paths = []
    for key in ("df_path", "all_frames_data_path", "processed_path", "frames_zip", "frames_zip_path"):
        p = v.get(key)
        if p:
            paths.append(p)
    inferred_zip = _frames_zip_path_from_video(v)
    if inferred_zip:
        paths.append(inferred_zip)
    for p in _duong_dan_csv_candidates(v) + _duong_dan_frames_json_candidates(v):
        paths.append(p)
    seen, got = set(), False
    for p in paths:
        rel = get_clean_rel_path(p)
        if not rel or rel in seen:
            continue
        seen.add(rel)
        if ensure_local_file(p, quiet=True):
            got = True
    return _lam_moi_ban_ghi_video_tu_db(v), got


def dong_bo_hf_json_nhe_tab(files):
    """Chỉ tải JSON từ HF khi file local chưa có — tránh chậm mỗi lần chuyển tab."""
    if not (HF_TOKEN and HF_DATASET_ID) or not files:
        return False
    missing = [f for f in files if _json_dst_mtime(f) <= 0]
    if not missing:
        return False
    before = {f: _json_dst_mtime(f) for f in missing}
    dong_bo_json_cau_hinh_tu_hf(force_files=frozenset(missing))
    after = {f: _json_dst_mtime(f) for f in missing}
    if before != after:
        _xoa_cache_sau_dong_bo_json(missing)
    return True


def dong_bo_json_cau_hinh_tu_hf(force_files=None):
    """Tải đồng bộ JSON cấu hình từ HF Dataset (video_list.json có thể < 5KB)."""
    force_set = set(force_files or [])
    for f_name in HF_JSON_CONFIG_FILES:
        dst = os.path.join(DB_DIR, f_name)
        if f_name not in force_set and os.path.exists(dst) and os.path.getsize(dst) > 2:
            continue
        got = _hf_download_dataset_file(f_name, quiet=True, min_size=2)
        if got:
            try:
                os.makedirs(DB_DIR, exist_ok=True)
                shutil.copy2(got, dst)
                print(f"[HF Sync] Da dong bo JSON: {f_name} -> {dst}")
            except Exception:
                print(f"[HF Sync] Da dong bo JSON: {f_name}")
            continue
        for src in (os.path.join(DATA_DIR, f_name), os.path.join("database", f_name), f_name):
            if not os.path.exists(dst) and os.path.exists(src) and os.path.getsize(src) > 2:
                try:
                    os.makedirs(DB_DIR, exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"[HF Sync] Copy JSON tu repo: {src} -> {dst}")
                except Exception:
                    pass
                break


def _exercise_tu_ten_file(name):
    n = str(name or "").lower()
    if any(k in n for k in ["gậy", "gay", "pulley", "stick"]):
        return "Bài tập với gậy (Pulley Exercise)"
    if any(k in n for k in ["dây", "day", "theraband", "band", "kháng"]):
        return "Bài tập với dây kháng lực (Theraband)"
    return "Bài tập con lắc Codman"


def _tim_upload_theo_video_name(username, video_name):
    if not video_name or not os.path.isdir(UPLOAD_DIR):
        return None
    base = os.path.splitext(os.path.basename(video_name))[0].lower()
    for fn in os.listdir(UPLOAD_DIR):
        if base in fn.lower():
            if not username or fn.startswith(f"{username}_"):
                fp = os.path.join(UPLOAD_DIR, fn)
                if os.path.getsize(fp) >= 5 * 1024:
                    return fp
    return None


def khoi_phuc_video_list_tu_tep():
    """Khôi phục danh sách video từ evaluations, CSV đã phân tích, progress success và patient_uploads."""
    import glob
    import re
    seen = set()
    out = []
    users = load_users()
    evals = load_data(EVALUATIONS_FILE)

    def _add(rec):
        key = (rec.get("username"), rec.get("video_name"), rec.get("exercise"))
        if not rec.get("video_name") or key in seen:
            return
        seen.add(key)
        out.append(rec)

    for e in evals:
        uname = e.get("patient_username") or ""
        vname = e.get("video_name") or ""
        if not vname:
            continue
        is_ai = e.get("doctor_username") == "AI_Researcher"
        ex = e.get("exercise") or _exercise_tu_ten_file(vname)
        fn = users.get(uname, {}).get("full_name", uname) if isinstance(users, dict) else uname
        vp = _tim_upload_theo_video_name(uname, vname)
        _add({
            "username": uname,
            "full_name": fn or uname,
            "video_name": vname,
            "exercise": ex,
            "accuracy": (e.get("ai_accuracy") or 0) if is_ai else 0,
            "time": e.get("time", "N/A"),
            "video_path": vp,
            "processed_path": None,
            "status": "Đã phân tích" if is_ai else "Đã đánh giá (bác sĩ)",
            "df_path": None,
            "all_frames_data_path": None,
        })

    if os.path.isdir(PROCESSED_DIR):
        for prog_fn in glob.glob(os.path.join(PROCESSED_DIR, "progress_*.json")):
            try:
                with open(prog_fn, "r", encoding="utf-8") as pf:
                    pdata = json.load(pf)
            except Exception:
                continue
            rec = _video_entry_from_progress(pdata)
            if rec:
                _add(rec)

        for csv_f in glob.glob(os.path.join(PROCESSED_DIR, "processed_*_f_data.csv")):
            m = re.search(r"processed_(\d+)_f_data\.csv$", os.path.basename(csv_f))
            if not m:
                continue
            ts = m.group(1)
            proc_f = os.path.join(PROCESSED_DIR, f"processed_{ts}_f.mp4")
            proc_raw = os.path.join(PROCESSED_DIR, f"processed_{ts}.mp4")
            proc_path = proc_f if os.path.exists(proc_f) else proc_raw
            json_f = os.path.join(PROCESSED_DIR, f"f_{ts}.json")
            matched = None
            for item in out:
                if item.get("df_path") == csv_f:
                    matched = item
                    break
                pp = item.get("processed_path") or ""
                if ts in str(pp):
                    item.setdefault("df_path", csv_f)
                    item.setdefault("all_frames_data_path", json_f if os.path.exists(json_f) else None)
                    item.setdefault("processed_path", proc_path if os.path.exists(proc_path) else pp)
                    matched = item
                    break
            if matched:
                continue
            # Không tạo bản ghi mồ côi — dong_bo_video_list_tu_processed sẽ gắn path cho bản ghi thật
            continue

    if os.path.isdir(UPLOAD_DIR):
        for fn in os.listdir(UPLOAD_DIR):
            if not fn.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            fp = os.path.join(UPLOAD_DIR, fn)
            if os.path.getsize(fp) < 5 * 1024:
                continue
            parts = fn.split("_", 1)
            uname = parts[0] if parts else "unknown"
            orig_name = fn
            if "_" in fn:
                orig_name = fn.split("_", 2)[-1] if len(fn.split("_")) >= 3 else fn
            fn_user = users.get(uname, {}).get("full_name", uname) if isinstance(users, dict) else uname
            _add({
                "username": uname,
                "full_name": fn_user,
                "video_name": orig_name,
                "exercise": _exercise_tu_ten_file(orig_name),
                "accuracy": 0,
                "time": get_vn_now().strftime("%H:%M - %d/%m/%Y"),
                "video_path": fp,
                "processed_path": None,
                "status": "Chờ NCV phân tích",
            })

    return out


def _parse_upload_time_from_filename(path_or_name):
    """Trích thời gian upload từ tên file dạng ..._YYYYMMDD_HHMMSS_..."""
    import re
    m = re.search(r"_(\d{8})_(\d{6})_", str(path_or_name or ""))
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.strftime("%H:%M - %d/%m/%Y")
    except Exception:
        return None


def _lich_su_entry_key(entry):
    """Khóa duy nhất theo BN + video + bài tập + thời gian — giữ mọi lần phân tích."""
    return (
        entry.get("username") or "",
        entry.get("video_name") or "",
        entry.get("bai_tap") or entry.get("exercise") or "",
        entry.get("ngay") or "",
    )


def _parse_vn_datetime(time_str):
    """Parse thời gian VN: 'HH:MM - dd/mm/YYYY' hoặc 'YYYY-MM-DD HH:MM:SS'."""
    if not time_str:
        return None
    s = str(time_str).strip()
    for fmt in ("%H:%M - %d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _ngay_trong_lich_su(time_str):
    """Trả về phần ngày dd/mm/YYYY để so khớp giữa các định dạng."""
    dt = _parse_vn_datetime(time_str)
    if dt:
        return dt.strftime("%d/%m/%Y")
    import re
    m = re.search(r"(\d{2}/\d{2}/\d{4})", str(time_str or ""))
    return m.group(1) if m else ""


def lay_danh_gia_ai_benh_nhan(username, video_name=None, exercise=None):
    """Lấy tất cả báo cáo AI của BN, mới nhất trước."""
    if not username:
        return []
    evals = load_data(EVALUATIONS_FILE) or []
    ai = [
        e for e in evals
        if e.get("doctor_username") == "AI_Researcher"
        and e.get("patient_username") == username
    ]
    if video_name:
        ai = [
            e for e in ai
            if e.get("video_name") == video_name
            or video_name in (e.get("video_name") or "")
        ]
    if exercise:
        ai = [e for e in ai if e.get("exercise") == exercise]
    ai.sort(
        key=lambda e: _parse_vn_datetime(e.get("time")) or datetime.min,
        reverse=True,
    )
    return _dedup_evaluations(ai)


def khoi_phuc_lich_su_tu_danh_gia(history=None):
    """Khôi phục lịch sử từ doctor_evaluations + điền username còn thiếu."""
    if history is None:
        history = load_data(HISTORY_FILE) or []
    evals = load_data(EVALUATIONS_FILE) or []
    ai_evals = [e for e in evals if e.get("doctor_username") == "AI_Researcher"]
    users_db = load_users()
    by_key = {_lich_su_entry_key(h): h for h in history if isinstance(h, dict)}
    changed = False

    for h in history:
        if not isinstance(h, dict) or h.get("username"):
            continue
        h_acc = round(float(h.get("accuracy") or 0), 1)
        h_bt = h.get("bai_tap") or h.get("exercise") or ""
        h_ngay = h.get("ngay") or ""
        h_day = _ngay_trong_lich_su(h_ngay)
        best_e = None
        for e in ai_evals:
            e_acc = round(float(e.get("ai_accuracy") or 0), 1)
            e_day = _ngay_trong_lich_su(e.get("time"))
            if e.get("exercise") != h_bt:
                continue
            if abs(e_acc - h_acc) > 0.5:
                continue
            if h_day and e_day and h_day != e_day:
                continue
            best_e = e
            break
        if best_e:
            uname = best_e.get("patient_username") or ""
            h["username"] = uname
            h["full_name"] = users_db.get(uname, {}).get("full_name", uname)
            h["video_name"] = best_e.get("video_name") or h.get("video_name") or ""
            h["bai_tap"] = best_e.get("exercise") or h_bt
            if best_e.get("time"):
                h["ngay"] = best_e.get("time")
            changed = True

    for e in ai_evals:
        uname = e.get("patient_username") or ""
        ngay = e.get("time") or ""
        if not uname or not ngay:
            continue
        entry = {
            "ngay": ngay,
            "username": uname,
            "full_name": users_db.get(uname, {}).get("full_name", uname),
            "video_name": e.get("video_name") or "",
            "bai_tap": e.get("exercise") or "",
            "accuracy": round(float(e.get("ai_accuracy") or 0), 1),
            "f1": None,
            "thoi_gian_tap": None,
            "source": "ai_eval",
        }
        key = _lich_su_entry_key(entry)
        if key not in by_key:
            history.append(entry)
            by_key[key] = entry
            changed = True

    if changed:
        save_data(HISTORY_FILE, history)
        print(f"[LichSu] Khoi phuc {len(history)} buoi tap tu danh gia AI")
    return history


def sap_xep_lich_su_theo_thoi_gian(history):
    """Sắp xếp lịch sử — mới nhất lên đầu."""
    return sorted(
        [h for h in history if isinstance(h, dict)],
        key=lambda h: _parse_vn_datetime(h.get("ngay")) or datetime.min,
        reverse=True,
    )


def nap_ket_qua_ai_vao_session(ai_eval):
    """Nạp phiên phân tích cũ từ bản ghi AI evaluation vào session."""
    if not ai_eval:
        return False
    all_vids = load_data(VIDEOS_FILE) or []
    v = next(
        (
            x for x in all_vids
            if x.get("username") == ai_eval.get("patient_username")
            and (
                x.get("video_name") == ai_eval.get("video_name")
                or ai_eval.get("video_name", "") in (x.get("video_name") or "")
            )
        ),
        None,
    )
    if v:
        nap_phien_benh_nhan_vao_session(v)
        st.session_state.view_old_analysis = True
        st.session_state.reanalyze_triggered = False
        df_path = v.get("df_path")
        if df_path:
            ensure_local_file(df_path)
            if os.path.exists(df_path):
                try:
                    st.session_state.angle_df = read_display_csv_fast(df_path)
                except Exception:
                    pass
        return True

    st.session_state.current_eval_video = {
        "username": ai_eval.get("patient_username"),
        "full_name": ai_eval.get("patient_username"),
        "video_name": ai_eval.get("video_name"),
        "exercise": ai_eval.get("exercise"),
    }
    st.session_state.view_old_analysis = True
    st.session_state.reanalyze_triggered = False
    st.session_state.has_data = False
    st.session_state.stats = None
    st.session_state.angle_df = None
    return False


def hien_thi_ket_qua_gan_nhat_va_lich_su(
    username,
    video_name=None,
    exercise=None,
    selected_v=None,
    key_suffix="",
    chi_nhan_xet=False,
):
    """Compatibility wrapper for latest result/history UI now owned by ui.doctor_forms."""
    render_latest_results_and_history_page(
        _build_ui_tab_dependencies(),
        username,
        video_name=video_name,
        exercise=exercise,
        selected_v=selected_v,
        key_suffix=key_suffix,
        chi_nhan_xet=chi_nhan_xet,
    )



def dong_bo_lich_su_tu_video_list(video_list=None):
    """Đồng bộ lich_su_tap_luyen.json từ video_list — mỗi BN + thời gian phân tích xong."""
    vlist = video_list if video_list is not None else load_data(VIDEOS_FILE)
    history = load_data(HISTORY_FILE) or []
    if not vlist:
        history = khoi_phuc_lich_su_tu_danh_gia(history)
        return sap_xep_lich_su_theo_thoi_gian(history)
    by_key = {_lich_su_entry_key(h): h for h in history if isinstance(h, dict)}
    changed = False
    for v in vlist:
        if v.get("status") != "Đã phân tích":
            continue
        uname = v.get("username") or ""
        vname = v.get("video_name") or ""
        ex = v.get("exercise") or ""
        if not vname or not ex:
            continue
        metrics = v.get("metrics") if isinstance(v.get("metrics"), dict) else {}
        ngay = v.get("time") or _parse_upload_time_from_filename(v.get("video_path") or vname)
        if not ngay:
            continue
        entry = {
            "ngay": ngay,
            "username": uname,
            "full_name": v.get("full_name") or uname,
            "video_name": vname,
            "bai_tap": ex,
            "accuracy": round(float(v.get("accuracy") or metrics.get("do_chinh_xac") or metrics.get("ty_le_tong_the") or 0), 1),
            "f1": round(float(metrics.get("f1_score") or 0), 2),
            "thoi_gian_tap": round(float(metrics.get("thoi_gian_xu_ly") or 0), 1) if metrics.get("thoi_gian_xu_ly") else None,
        }
        key = _lich_su_entry_key(entry)
        if key in by_key:
            existing = by_key[key]
            for fld, val in entry.items():
                if val is not None and (not existing.get(fld) or fld in ("ngay", "accuracy", "f1")):
                    existing[fld] = val
            changed = True
        else:
            history.append(entry)
            by_key[key] = entry
            changed = True
    if changed:
        save_data(HISTORY_FILE, history)
        print(f"[LichSu] Dong bo {len(history)} buoi tap tu video_list")
    history = khoi_phuc_lich_su_tu_danh_gia(history)
    return sap_xep_lich_su_theo_thoi_gian(history)


def _ensure_videos_file_exists():
    """Đảm bảo video_list.json tồn tại — fallback từ repo root hoặc database/."""
    if os.path.exists(VIDEOS_FILE) and os.path.getsize(VIDEOS_FILE) > 2:
        return
    for alt in ("video_list.json", os.path.join("database", "video_list.json")):
        if os.path.exists(alt) and os.path.getsize(alt) > 2:
            try:
                shutil.copy2(alt, VIDEOS_FILE)
                print(f"[VideoList] Copy tu {alt} -> {VIDEOS_FILE}")
                return
            except Exception:
                pass


def _ensure_evaluations_file_exists():
    """Đảm bảo doctor_evaluations.json tồn tại trong database/."""
    if os.path.exists(EVALUATIONS_FILE) and os.path.getsize(EVALUATIONS_FILE) > 2:
        return
    for alt in ("doctor_evaluations.json", os.path.join("database", "doctor_evaluations.json")):
        if os.path.exists(alt) and os.path.getsize(alt) > 2:
            try:
                os.makedirs(DB_DIR, exist_ok=True)
                shutil.copy2(alt, EVALUATIONS_FILE)
                print(f"[Evaluations] Copy tu {alt} -> {EVALUATIONS_FILE}")
                return
            except Exception:
                pass


def _video_entry_from_progress(pdata):
    """Chuyển progress_*.json (success) thành bản ghi video_list."""
    if pdata.get("status") != "success":
        return None
    res = pdata.get("result") or {}
    uname = pdata.get("username") or ""
    vname = pdata.get("video_name") or ""
    if not vname:
        return None
    users = load_users()
    meta = pdata.get("job_meta") or {}
    ex = meta.get("exercise_name") or res.get("exercise", {}).get("ten") or _exercise_tu_ten_file(vname)
    fn = meta.get("full_name") or (users.get(uname, {}).get("full_name") if isinstance(users, dict) else uname)
    stats = res.get("stats") or {}
    job_time = meta.get("started_at") or pdata.get("updated_at")
    time_str = get_vn_now().strftime("%H:%M - %d/%m/%Y")
    if job_time:
        try:
            if isinstance(job_time, (int, float)):
                time_str = datetime.fromtimestamp(job_time).strftime("%H:%M - %d/%m/%Y")
            elif isinstance(job_time, str) and job_time.strip():
                time_str = job_time
        except Exception:
            pass
    return {
        "username": uname,
        "full_name": fn or uname,
        "video_name": vname,
        "exercise": ex if isinstance(ex, str) else str(ex),
        "accuracy": stats.get("do_chinh_xac") or stats.get("ty_le_tong_the") or 0,
        "time": time_str,
        "video_path": pdata.get("video_path") or _tim_upload_theo_video_name(uname, vname),
        "processed_path": res.get("processed_video_path"),
        "metrics": stats,
        "df_path": res.get("df_path"),
        "all_frames_data_path": res.get("all_frames_data_path"),
        "frames_zip": res.get("frames_zip"),
        "frames_zip_path": res.get("frames_zip") or res.get("frames_zip_path"),
        "status": "Đã phân tích",
    }


def _merge_video_lists_union(base_list, extra_list):
    """Gộp hai danh sách video — ưu tiên bản upload/kết quả mới nhất."""
    by_key = {}
    for x in base_list or []:
        key = (x.get("username"), x.get("video_name"), x.get("exercise"))
        if x.get("video_name"):
            by_key[key] = x
    for rec in extra_list or []:
        key = (rec.get("username"), rec.get("video_name"), rec.get("exercise"))
        if not rec.get("video_name"):
            continue
        if key not in by_key:
            by_key[key] = rec
            continue
        existing = by_key[key]
        t_rec = _parse_vn_datetime(_lay_thoi_gian_upload_video(rec)) or datetime.min
        t_exist = _parse_vn_datetime(_lay_thoi_gian_upload_video(existing)) or datetime.min
        if t_rec >= t_exist:
            newer, older = rec, existing
        else:
            newer, older = existing, rec
        merged = dict(newer)
        for fld in ("video_path", "processed_path", "df_path", "all_frames_data_path", "metrics", "frames_zip", "frames_zip_path"):
            if not merged.get(fld) and older.get(fld):
                merged[fld] = older[fld]
        if t_rec >= t_exist:
            for fld in ("metrics", "processed_path", "df_path", "all_frames_data_path", "frames_zip", "frames_zip_path", "accuracy", "status"):
                if rec.get(fld) is not None:
                    merged[fld] = rec[fld]
        elif existing.get("metrics") and not merged.get("metrics"):
            merged["metrics"] = existing["metrics"]
            if existing.get("accuracy"):
                merged["accuracy"] = existing["accuracy"]
        if rec.get("status") == "Đã phân tích" or existing.get("status") == "Đã phân tích":
            merged["status"] = "Đã phân tích"
        by_key[key] = merged
    return list(by_key.values())


def _merge_missing_from_progress(video_list):
    """Bổ sung video đã phân tích xong (progress success) nhưng chưa có trong video_list."""
    import glob
    if not os.path.isdir(PROCESSED_DIR):
        return video_list
    seen = {(x.get("username"), x.get("video_name"), x.get("exercise")) for x in (video_list or [])}
    extras = []
    for prog_fn in glob.glob(os.path.join(PROCESSED_DIR, "progress_*.json")):
        try:
            with open(prog_fn, "r", encoding="utf-8") as pf:
                pdata = json.load(pf)
        except Exception:
            continue
        rec = _video_entry_from_progress(pdata)
        if not rec:
            continue
        key = (rec.get("username"), rec.get("video_name"), rec.get("exercise"))
        if key in seen:
            continue
        seen.add(key)
        extras.append(rec)
    if not extras:
        return video_list
    merged = _merge_video_lists_union(video_list, extras)
    print(f"[VideoList] Bo sung {len(extras)} video tu progress_*.json")
    return merged


def tai_lai_video_list_tu_cloud():
    """Tải lại video_list + evaluations từ HF và khôi phục từ progress/CSV/upload."""
    _ensure_videos_file_exists()
    _ensure_evaluations_file_exists()
    dong_bo_json_cau_hinh_tu_hf(force_files=frozenset({"video_list.json", "doctor_evaluations.json"}))
    try:
        _load_data_cached.clear()
    except Exception:
        pass
    lst = load_video_list_an_toan(sync_processed=True)
    recovered = khoi_phuc_video_list_tu_tep()
    if recovered:
        lst = _merge_video_lists_union(lst, _merge_video_list_with_evals(recovered))
        lst = _merge_missing_from_progress(lst)
        lst = dong_bo_video_list_tu_processed(lst or [])
    if lst:
        save_data(VIDEOS_FILE, lst)
        print(f"[VideoList] Tai lai tu Cloud: {len(lst)} video")
        try:
            dong_bo_lich_su_tu_video_list(lst)
        except Exception:
            pass
    return lst or []


def _merge_video_list_with_evals(video_list):
    """Bổ sung video từ doctor_evaluations (bác sĩ + NCV) — giữ cả đánh giá cũ và cập nhật AI mới."""
    base = list(video_list or [])
    seen = {(x.get("username"), x.get("video_name"), x.get("exercise")) for x in base}
    users = load_users()
    evals = load_data(EVALUATIONS_FILE)
    by_key = {}
    for x in base:
        by_key[(x.get("username"), x.get("video_name"), x.get("exercise"))] = x

    def _upsert(rec):
        key = (rec.get("username"), rec.get("video_name"), rec.get("exercise"))
        if not rec.get("video_name"):
            return
        if key in by_key:
            existing = by_key[key]
            for fld in ("video_path", "processed_path", "df_path", "all_frames_data_path", "metrics", "frames_zip", "frames_zip_path"):
                if not existing.get(fld) and rec.get(fld):
                    existing[fld] = rec[fld]
            if rec.get("accuracy") and (not existing.get("accuracy") or float(rec.get("accuracy") or 0) > float(existing.get("accuracy") or 0)):
                existing["accuracy"] = rec["accuracy"]
            if rec.get("status") == "Đã phân tích":
                existing["status"] = "Đã phân tích"
        else:
            base.append(rec)
            by_key[key] = rec
            seen.add(key)

    for e in evals:
        uname = e.get("patient_username") or ""
        vname = e.get("video_name") or ""
        ex = e.get("exercise") or _exercise_tu_ten_file(vname)
        if not vname:
            continue
        fn = users.get(uname, {}).get("full_name", uname) if isinstance(users, dict) else uname
        is_ai = e.get("doctor_username") == "AI_Researcher"
        vp = _tim_upload_theo_video_name(uname, vname)
        upload_time = _lay_thoi_gian_upload_video({"video_path": vp, "video_name": vname})
        _upsert({
            "username": uname,
            "full_name": fn or uname,
            "video_name": vname,
            "exercise": ex,
            "accuracy": e.get("ai_accuracy") if is_ai else 0,
            "time": upload_time,
            "video_path": vp,
            "processed_path": None,
            "status": "Đã phân tích" if is_ai else "Đã đánh giá (bác sĩ)",
            "df_path": None,
            "all_frames_data_path": None,
        })

    return base


def _resolve_video_display_path(raw_path, processed_path, prefer_processed=False):
    """Chuẩn hóa đường dẫn video để hiển thị — ưu tiên processed khi đã phân tích."""
    candidates = []
    if prefer_processed:
        candidates = [processed_path, raw_path]
    else:
        raw_lp = get_local_frame_path(raw_path) if raw_path else None
        proc_lp = get_local_frame_path(processed_path) if processed_path else None
        raw_ok = raw_lp and find_ready_local_video(raw_lp)
        proc_ok = proc_lp and find_ready_local_video(proc_lp)
        if processed_path and (proc_ok or not raw_ok):
            candidates = [processed_path, raw_path]
        else:
            candidates = [raw_path, processed_path]
    for p in candidates:
        if not p:
            continue
        lp = get_local_frame_path(p) or p
        if lp:
            return lp
    return None


def _lay_duong_dan_video_hien_thi(v):
    """Đường dẫn phát video — ưu tiên bản processed nếu video đã phân tích."""
    prefer = v.get("status") == "Đã phân tích"
    return _resolve_video_display_path(
        v.get("video_path"),
        v.get("processed_path"),
        prefer_processed=prefer,
    )


def _lay_duong_dan_video_tho(v):
    """Video gốc BN đã upload — dùng trong danh sách video (không hiển thị bản processed)."""
    if not v:
        return None
    candidates = []
    raw = v.get("video_path")
    if raw and not _is_scratch_video_path(raw):
        candidates.append(get_local_frame_path(raw) or raw)
    stripped = _strip_to_original_upload(raw or "")
    if stripped and stripped not in candidates:
        candidates.append(get_local_frame_path(stripped) or stripped)
    found = _tim_video_upload_goc(v)
    if found:
        candidates.insert(0, found)
    for c in candidates:
        if c and not _is_scratch_video_path(c):
            return c
    return None


def _dam_bao_video_san_sang_play(path, prefer_raw=False, video_record=None):
    """Tự động tải video từ Cloud/local — không cần nút thủ công."""
    if not path and video_record:
        path = _lay_duong_dan_video_tho(video_record)
    if not path:
        return None
    if not prefer_raw and _la_duong_dan_video_gia(path):
        return None
    if prefer_raw:
        search_paths = list(video_raw_only_paths(path))
        if video_record:
            alt = _lay_duong_dan_video_tho(video_record)
            if alt and alt not in search_paths:
                search_paths.insert(0, alt)
        for candidate in search_paths:
            if _is_scratch_video_path(candidate):
                continue
            ensure_local_file(candidate, quiet=True, try_fallbacks=False)
            if _valid_raw_video_local(candidate):
                return candidate
        for candidate in search_paths:
            if _is_scratch_video_path(candidate):
                continue
            ensure_local_file(candidate, quiet=True, try_fallbacks=False)
            if is_local_file_ready(candidate):
                return candidate
        return _strip_to_original_upload(path)
    ready = find_ready_local_video(path)
    if ready:
        pb = resolve_playback_video_path(ready)
        return pb if pb and is_local_file_ready(pb) else ready
    for candidate in video_fallback_paths(path):
        ensure_local_file(candidate, quiet=True, try_fallbacks=True)
    ready = find_ready_local_video(path)
    if ready:
        pb = resolve_playback_video_path(ready)
        return pb if pb and is_local_file_ready(pb) else ready
    return path


def _video_mo_duoc_opencv(path):
    if not path or not is_local_file_ready(path):
        return False
    try:
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened()
        if ok:
            ok = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0 or cap.read()[0]
        cap.release()
        return bool(ok)
    except Exception:
        return False


def _dam_bao_video_cho_phan_tich(video_path, username=None, video_name=None):
    """Tải + xác thực file video trước khi MediaPipe/OpenCV phân tích."""
    vrec = {
        "username": username,
        "video_name": video_name,
        "video_path": video_path,
    }
    search_paths = []
    canon = _lay_duong_dan_video_tho(vrec)
    if canon:
        search_paths.append(canon)
    if video_path:
        for p in video_raw_only_paths(video_path):
            if p not in search_paths:
                search_paths.append(p)
        for p in video_fallback_paths(video_path):
            if p not in search_paths and not _is_scratch_video_path(p):
                search_paths.append(p)
    for candidate in search_paths:
        if _is_scratch_video_path(candidate):
            continue
        ensure_local_file(candidate, quiet=True, try_fallbacks=False)
        if _video_mo_duoc_opencv(candidate):
            return candidate
        h264 = get_final_h264_path(_strip_to_original_upload(candidate))
        if h264 and not _is_scratch_video_path(h264) and h264 != candidate:
            ensure_local_file(h264, quiet=True, try_fallbacks=False)
            if _video_mo_duoc_opencv(h264):
                return h264
    return None


def _la_ban_ghi_video_mo_co(v):
    """Bản ghi khôi phục tạm — ẩn khỏi danh sách chính."""
    if v.get("full_name") == "Bệnh nhân (khôi phục)":
        return True
    vn = str(v.get("video_name") or "")
    if v.get("username") in (None, "", "unknown") and vn.startswith("video_processed_"):
        return True
    return False


def dong_bo_video_list_tu_processed(video_list):
    """Tự cập nhật video_list.json từ processed_results, progress và patient_uploads."""
    import glob
    import re

    if video_list is None:
        video_list = []
    changed = False

    progress_index = {}
    if os.path.isdir(PROCESSED_DIR):
        for prog_fn in glob.glob(os.path.join(PROCESSED_DIR, "progress_*.json")):
            try:
                with open(prog_fn, "r", encoding="utf-8") as pf:
                    pdata = json.load(pf)
            except Exception:
                continue
            if pdata.get("status") != "success":
                continue
            uname = pdata.get("username") or ""
            vname = pdata.get("video_name") or ""
            if vname:
                progress_index[(uname, vname)] = pdata

    for v in video_list:
        uname = v.get("username") or ""
        vname = v.get("video_name") or ""

        vp = _tim_upload_theo_video_name(uname, vname)
        if vp and v.get("video_path") != vp:
            v["video_path"] = vp
            changed = True

        pdata = progress_index.get((uname, vname))
        if pdata:
            rec = _video_entry_from_progress(pdata)
            if rec:
                for fld in (
                    "processed_path", "df_path", "all_frames_data_path",
                    "metrics", "frames_zip", "frames_zip_path", "accuracy", "status", "video_path",
                ):
                    new_val = rec.get(fld)
                    if new_val and v.get(fld) != new_val:
                        if fld == "video_path" and v.get("video_path"):
                            continue
                        v[fld] = new_val
                        changed = True

        if v.get("status") == "Đã phân tích" and not v.get("processed_path"):
            pp = v.get("df_path") or ""
            m = re.search(r"processed_(\d+)_f_data\.csv$", os.path.basename(str(pp)))
            if m:
                ts = m.group(1)
                proc_f = os.path.join(PROCESSED_DIR, f"processed_{ts}_f.mp4")
                proc_raw = os.path.join(PROCESSED_DIR, f"processed_{ts}.mp4")
                proc_path = proc_f if os.path.exists(proc_f) else proc_raw
                if os.path.exists(proc_path):
                    v["processed_path"] = proc_path
                    changed = True

        # Tự động điền các trường null từ processed_path (timestamp)
        pp = v.get("processed_path") or ""
        m_ts = re.search(r"processed_(\d+)", os.path.basename(str(pp)))
        if m_ts:
            ts = m_ts.group(1)
            if not v.get("frames_zip"):
                v["frames_zip"] = os.path.join(PROCESSED_DIR, f"processed_{ts}_frames.zip")
                changed = True
            if not v.get("frames_zip_path"):
                v["frames_zip_path"] = v.get("frames_zip") or os.path.join(PROCESSED_DIR, f"processed_{ts}_frames.zip")
                changed = True
            if not v.get("all_frames_data_path"):
                v["all_frames_data_path"] = os.path.join(PROCESSED_DIR, f"f_{ts}.json")
                changed = True
            if not v.get("df_path"):
                v["df_path"] = os.path.join(PROCESSED_DIR, f"processed_{ts}_f_data.csv")
                changed = True

    proper_sigs = set()
    for v in video_list:
        if not _la_ban_ghi_video_mo_co(v):
            proper_sigs.add((
                v.get("video_name"),
                v.get("exercise"),
                v.get("processed_path") or v.get("df_path"),
            ))

    cleaned = []
    for v in video_list:
        if _la_ban_ghi_video_mo_co(v):
            sig = (
                v.get("video_name"),
                v.get("exercise"),
                v.get("processed_path") or v.get("df_path"),
            )
            if sig in proper_sigs or str(v.get("video_name", "")).startswith("video_processed_"):
                changed = True
                continue
        cleaned.append(v)

    if len(cleaned) != len(video_list):
        changed = True
        video_list = cleaned

    if changed:
        save_data(VIDEOS_FILE, video_list)
        print(f"[VideoList] Dong bo tu processed — {len(video_list)} muc")
    return video_list


@st.cache_data(show_spinner=False)
def _load_video_list_core(v_mtime, e_mtime):
    """Nạp video_list + merge eval — cache theo mtime file JSON."""
    lst = _merge_video_list_with_evals(load_data(VIDEOS_FILE))
    if lst:
        return lst
    recovered = khoi_phuc_video_list_tu_tep()
    if recovered:
        recovered = _merge_video_list_with_evals(recovered)
        save_data(VIDEOS_FILE, recovered)
        print(f"[VideoList] Da khoi phuc {len(recovered)} video vao video_list.json")
    return recovered or []


def load_video_list_an_toan(sync_processed=False):
    """Nạp video_list — mặc định chỉ đọc (nhanh); sync_processed=True chỉ dùng khi khởi động nền."""
    try:
        v_mtime = os.path.getmtime(VIDEOS_FILE) if os.path.exists(VIDEOS_FILE) else 0
    except Exception:
        v_mtime = 0
    try:
        e_mtime = os.path.getmtime(EVALUATIONS_FILE) if os.path.exists(EVALUATIONS_FILE) else 0
    except Exception:
        e_mtime = 0
    lst = _load_video_list_core(v_mtime, e_mtime) or []
    if sync_processed and lst:
        lst = _merge_missing_from_progress(lst)
        lst = dong_bo_video_list_tu_processed(lst)
    return lst


def lay_do_chinh_xac_ai_chuan(selected_v):
    if not selected_v:
        return None
    evals = load_data(EVALUATIONS_FILE)
    v_name = selected_v.get('video_name')
    ex_name = selected_v.get('exercise')
    if not v_name or not ex_name:
        return None
    ai_evals = [e for e in evals if e.get('doctor_username') == "AI_Researcher"
                and e.get('video_name') == v_name
                and e.get('exercise') == ex_name]
    if ai_evals:
        return ai_evals[-1].get('ai_accuracy')
    return selected_v.get('accuracy')

@st.cache_data(show_spinner=False)
def _get_cached_users_dict(mtime):
    users = load_data(USER_DATA_FILE)
    if not isinstance(users, dict):
        users = {}
    if not users:
        users = _build_bootstrap_users()
        save_data(USER_DATA_FILE, users)

    # Đảm bảo các user cũ có role mặc định là Bệnh nhân
    for username in users:
        if "role" not in users[username]:
            users[username]["role"] = "Bệnh nhân"

    _detect_user_uniqueness_issues(users)
    return users

def load_users():
    mtime = 0.0
    if os.path.exists(USER_DATA_FILE):
        mtime = os.path.getmtime(USER_DATA_FILE)
    return _get_cached_users_dict(mtime)

def save_users(users):
    save_data(USER_DATA_FILE, users)
    try:
        _get_cached_users_dict.clear()
    except Exception:
        pass

def verify_password(password, hashed):
    return verify_password_record(password, {"password": hashed}).ok


def _password_update_fields(password, *, must_change_password=None):
    return password_record_update(
        password,
        updated_at=get_vn_now().isoformat(),
        must_change_password=must_change_password,
    )


def _set_user_password(user_record, password, *, must_change_password=None):
    user_record.update(_password_update_fields(password, must_change_password=must_change_password))
    return user_record


def _build_bootstrap_users():
    bootstrap_password = _get_secret("BOOTSTRAP_ADMIN_PASSWORD") or secrets.token_urlsafe(48)
    now = get_vn_now().isoformat()
    return {
        "admin": {
            **password_record_update(bootstrap_password, updated_at=now, must_change_password=True),
            "full_name": "System Administrator",
            "role": "Quản trị viên",
            "email": "",
            "created_at": now,
            "seeded": True,
        }
    }


def _detect_user_uniqueness_issues(users):
    issues = find_user_uniqueness_issues(users)
    if issues:
        print(f"[Auth] User database uniqueness issues detected: {', '.join(issues)}")
    return issues

def _normalize_auth_text(value):
    """Chuẩn hóa chuỗi đăng nhập để tránh lệch dấu Unicode/khoảng trắng sau F5 hoặc copy-paste."""
    return normalize_auth_text(value)


def _auth_lookup_key(users, username):
    """Tìm username theo cách mềm hơn nhưng không match theo full_name để tránh collision."""
    return find_user_key(users, username)


def _auth_lookup_email_key(users, email):
    return find_user_key_by_email(users, email)


def _roles_match(stored_role, selected_role):
    return roles_match(stored_role, selected_role)


def _verify_auth_password(username_key, password, user_record):
    return verify_password_record(password, user_record).ok


def _rehash_password_if_needed(users, username_key, password):
    user_record = (users or {}).get(username_key)
    verification = verify_password_record(password, user_record)
    if not verification.ok or not verification.needs_rehash:
        return verification.ok
    _set_user_password(user_record, password)
    save_users(users)
    return True


def _query_param_text(name):
    try:
        value = st.query_params.get(name, "")
    except Exception:
        value = ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return _normalize_auth_text(value)

def don_dep_file_tam():
    """Dọn dẹp file tạm cũ trong /tmp để ngăn OOM khi chạy nhiều phân tích liên tiếp"""
    try:
        import glob
        import time as _time
        tmp_dir = tempfile.gettempdir()
        current_time = _time.time()
        max_age = 1800  # Xóa file cũ hơn 30 phút

        # Các pattern file tạm của hệ thống
        patterns = [
            os.path.join(tmp_dir, 'processed_*.mp4'),
            os.path.join(tmp_dir, 'f_*.json'),
            os.path.join(tmp_dir, '*_data.csv'),
            os.path.join(tmp_dir, '*_audio.wav'),
        ]

        deleted_count = 0
        for pattern in patterns:
            for fpath in glob.glob(pattern):
                try:
                    if current_time - os.path.getmtime(fpath) > max_age:
                        os.unlink(fpath)
                        deleted_count += 1
                except OSError as exc:
                    print(f"[Cleanup] Khong xoa duoc file tam {fpath}: {exc}")

        if deleted_count > 0:
            print(f"[Cleanup] Da xoa {deleted_count} file tam cu khoi {tmp_dir}")
    except Exception as e:
        print(f"[Cleanup] Loi don file tam: {e}")

def thuc_hien_khoi_tao_he_thong_mot_lan():
    """Chạy startup có kiểm soát, idempotent và có config bật/tắt side effect."""
    config = load_startup_config()

    def _init_nen():
        """JSON/HF sync + merge video — chạy nền, không block đăng nhập."""
        _ensure_videos_file_exists()
        _ensure_evaluations_file_exists()
        dong_bo_json_cau_hinh_tu_hf()
        try:
            _load_video_list_core.clear()
        except Exception:
            pass
        lst = load_video_list_an_toan(sync_processed=True)
        try:
            dong_bo_lich_su_tu_video_list(lst)
        except Exception as hist_sync_err:
            print(f"[LichSu] Loi dong bo khoi dong: {hist_sync_err}")
        khoi_tao_dong_bo_hf()

    def _auto_transcode_all_hevc():
        """Transcode HEVC → H.264 nền — không quét patient_uploads (tránh đơ CPU + lỗi moov)."""
        try:
            if _scan_progress_by_status("processing"):
                print("[AutoTranscode] Co job phan tich dang chay — bo qua transcode khoi dong")
                return
            video_list = load_data(VIDEOS_FILE)
            print(f"[AutoTranscode] Scan {len(video_list)} muc — chi processed_results, toi da 2 file")
            done = 0
            for vid in video_list:
                if done >= 2:
                    break
                vpath = vid.get("processed_path") or ""
                if not vpath or "patient_uploads" in vpath.replace("\\", "/"):
                    continue
                if not os.path.exists(vpath):
                    ensure_local_file(vpath, quiet=True)
                if not os.path.exists(vpath) or os.path.getsize(vpath) < 5 * 1024:
                    continue
                try:
                    mtime = os.path.getmtime(vpath)
                    size = os.path.getsize(vpath)
                    if not _check_video_valid_cached(vpath, mtime, size):
                        print(f"[AutoTranscode] Bo qua file loi/moov: {os.path.basename(vpath)}")
                        continue
                except Exception:
                    continue
                final_h264 = get_final_h264_path(vpath)
                if os.path.exists(final_h264) and os.path.getsize(final_h264) > 5 * 1024:
                    try:
                        mtime = os.path.getmtime(final_h264)
                        size = os.path.getsize(final_h264)
                        if _check_video_valid_cached(final_h264, mtime, size):
                            continue
                    except Exception:
                        pass
                try:
                    v_codec, _ = get_video_codec(vpath)
                    if v_codec and v_codec != "h264":
                        print(f"[AutoTranscode] Transcode: {os.path.basename(vpath)} ({v_codec})")
                        ensure_playable_video(vpath)
                        done += 1
                        time.sleep(3)
                except Exception as e:
                    print(f"[AutoTranscode] Loi: {e}")
        except Exception as e:
            print(f"[AutoTranscode] Loi toan cuc: {e}")

    def _khoi_phuc_nen_sau_boot():
        try:
            _chay_khoi_phuc_phan_tich_sau_deploy()
        except Exception as boot_resume_err:
            print(f"[Resume] Loi khoi phuc dong bo luc boot: {boot_resume_err}")

    def _resume_and_watch_analysis_jobs():
        """Theo dõi job bị crash/OOM sau khi Space đã chạy."""
        while True:
            time.sleep(max(5.0, config.resume_watcher_interval_seconds))
            try:
                n2 = khoi_phuc_job_phan_tich_sau_deploy(cold_start=False)
                if n2:
                    print(f"[Resume] Khoi dong lai {n2} job phan tich bi gian doan")
            except Exception as poll_err:
                print(f"[Resume] Loi poll job: {poll_err}")

    return app_startup(
        st=st,
        config=config,
        dirs=RUNTIME_DIRS,
        boot_sync_job=_init_nen,
        cleanup_job=don_dep_file_tam,
        auto_transcode_job=_auto_transcode_all_hevc,
        auto_resume_job=_khoi_phuc_nen_sau_boot,
        resume_watcher_job=_resume_and_watch_analysis_jobs,
    )

def _xoa_widget_dang_nhap_sau_rerun():
    """Dọn các widget login ở đầu rerun, trước khi form đăng nhập được render lại."""
    if not st.session_state.pop("_clear_login_widgets_next_run", False):
        return
    for _k in (
        "login_u", "login_p", "login_role_main", "theme_toggle_login",
        "forgot_password_mode", "change_password_mode",
        "f_u", "f_e", "f_p1", "f_p2",
        "cp_u_v2", "cp_old_v2", "cp_new_v2", "cp_conf_v2",
    ):
        st.session_state.pop(_k, None)


def _rerun_toan_bo_app():
    """Rerun toàn app — st.rerun() thuần, tránh scope gây màn trắng trên HF Space."""
    st.rerun()


def _lam_moi_giao_dien_sau_nut():
    """Sau bấm nút — full app rerun (scope='app') để làm mới UI ổn định."""
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()


def _hoan_tat_dang_nhap(username, user_record):
    """Gom xử lý sau đăng nhập để tránh sidebar đã login nhưng body còn form cũ."""
    role = user_record.get('role', 'Bệnh nhân')
    session_version = get_global_session_version(SESSION_STATE_FILE)
    st.session_state.logged_in = True
    st.session_state.user_info = {
        "username": username,
        "full_name": user_record.get('full_name', username),
        "email": user_record.get('email'),
        "role": role,
        "session_version": session_version,
    }
    st.session_state.show_login_dialog = False
    st.session_state.active_tab = "🏠 TRANG CHỦ"
    st.session_state.pop("active_tab_widget", None)
    st.session_state._need_home_sync = True
    st.session_state._clear_login_widgets_next_run = True


def _hoan_tat_dang_nhap_api(login_response):
    user = login_response.get("user") if isinstance(login_response, dict) else {}
    if not isinstance(user, dict) or not user.get("username"):
        raise FrontendApiError("backend login response missing user")
    st.session_state.backend_access_token = login_response.get("access_token")
    st.session_state.logged_in = True
    st.session_state.user_info = {
        "username": user.get("username"),
        "full_name": user.get("full_name") or user.get("username"),
        "email": user.get("email"),
        "role": user.get("role") or "Bệnh nhân",
        "session_version": get_global_session_version(SESSION_STATE_FILE),
        "auth_backend": "api",
    }
    st.session_state.show_login_dialog = False
    st.session_state.active_tab = "🏠 TRANG CHỦ"
    st.session_state.pop("active_tab_widget", None)
    st.session_state._need_home_sync = False
    st.session_state._clear_login_widgets_next_run = True
    _frontend_api_clear_error()


def _initialize_session_defaults():
    """Khởi tạo session state khi app thật sự chạy, không chạy lúc import module."""
    defaults = {
        "logged_in": False,
        "user_info": None,
        "forgot_password_mode": False,
        "show_login_dialog": False,
        "processed_video_path": None,
        "theme": "dark",
        "has_data": False,
        "ncv_model_type": "MediaPipe Heavy",
        "ncv_skip_frames": 0,
        "view_old_analysis": False,
        "angle_df": None,
        "stats": None,
        "frames_zip": None,
        "exercise": None,
        "output_video_path": None,
        "output_video_bytes": None,
        "processing": False,
        "processing_error": False,
        "current_upload_key": None,
        "all_frames_paths": [],
        "temp_video_file": None,
        "video_ready": False,
        "frames_ready": False,
        "frames_loaded": False,
        "current_page": 1,
        "uploaded_file_name": None,
        "processed_video_bytes": None,
        "processing_progress": 0,
        "processing_status": "",
        "all_frames_data": [],
        "processing_result": None,
        "appointments": [],
        "exercise_reminders": [],
        "medication_reminders": [],
        "reminder_id_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = queue.Queue()
    if "ncv_resize_width" not in st.session_state:
        st.session_state.ncv_resize_width = 480 if _is_hf_runtime() else 720


def _handle_google_identity_login():
    """Kiểm tra đăng nhập Google khi runtime UI đã được khởi tạo."""
    if st.session_state.get("logged_in"):
        return
    try:
        user_detected = None
        # 1. Kiểm tra chuẩn chính thức mới nhất (st.user)
        if hasattr(st, 'user') and st.user and getattr(st.user, 'email', None):
            user_detected = st.user
        # 2. Kiểm tra chuẩn experimental cũ hơn
        elif hasattr(st, 'experimental_user') and st.experimental_user and getattr(st.experimental_user, 'email', None):
            user_detected = st.experimental_user

        if user_detected and user_detected.email:
            users_for_google = load_users()
            google_key = _auth_lookup_email_key(users_for_google, user_detected.email)
            if google_key:
                google_record = users_for_google[google_key]
                _hoan_tat_dang_nhap(google_key, google_record)
                st.session_state.user_info["auth_type"] = "google"
                if 'auth_initiated' in st.session_state:
                    del st.session_state['auth_initiated']
                st.rerun()
            else:
                st.warning("Tài khoản Google này chưa được quản trị viên cấp quyền trong hệ thống.")
    except Exception as e:
        # st.error(f"Lỗi nhận diện Google: {e}") # Debug nếu cần
        pass


def _inject_base_css_once():
    inject_base_css(st)


def initialize_session_runtime():
    """Khởi tạo phần phụ thuộc Streamlit session sau khi app_startup đã chạy."""
    _initialize_session_defaults()
    _lam_sach_cache_khi_doi_hf_token()
    _xoa_widget_dang_nhap_sau_rerun()
    _handle_google_identity_login()
    _inject_base_css_once()


MAX_FILE_SIZE_MB = MAX_UPLOAD_SIZE_MB
def validate_uploaded_video_file(uploaded_file):
    """Reject unsafe uploads before reading the full buffer into memory."""
    ok, msg = validate_upload_metadata(uploaded_file)
    return ok, msg


def validate_video_file_for_processing(file_path):
    """Run ffprobe before transcoding or saving a submitted video."""
    return validate_video_file_for_processing_core(file_path)

# ============================================
# CẤU HÌNH XỬ LÝ - TỐI ƯU ĐỘ CHÍNH XÁC CAO
# ============================================
SKIP_FRAMES = 0    # Mặc định: Xử lý mọi khung hình
RESIZE_WIDTH = 720 # Mặc định độ phân giải HD (720p) để trích xuất sắc nét và chuẩn xác nhất
OUTPUT_QUALITY = 50
MAX_FRAMES = 100000  # Hỗ trợ tối đa 100000 frame (~55 phút @ 30fps)
THUMBNAIL_QUALITY = 80
THUMBNAIL_WIDTH = 320

# ============================================
# HÀM CHUYỂN ĐỔI MOV SANG MP4
# ============================================
def convert_mov_to_mp4(input_path):
    output_path = mov_to_mp4_path(input_path)
    try:
        result = subprocess.run(build_ffmpeg_version_command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return input_path

        subprocess.run(
            build_mov_to_mp4_command(input_path, output_path),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return input_path
    except Exception as e:
        print(f"Lỗi chuyển đổi MOV: {e}")
        return input_path

# ============================================
# HÀM HIỂN THỊ TAB: THEO DÕI TIẾN TRIỂN (MỚI)
# ============================================
def hien_thi_tab_tien_trien():
    """Thiết kế Tab Tiến triển sử dụng DỮ LIỆU THẬT từ lịch sử tập luyện"""
    st.markdown("### 📈 THEO DÕI TIẾN TRIỂN THỜI GIAN THỰC")

    history_data = dong_bo_lich_su_tu_video_list()

    if not history_data:
        st.info("ℹ️ Hiện chưa có dữ liệu thực tế. Hãy tải video và phân tích để bắt đầu theo dõi tiến triển.")
        # Hiển thị demo nếu chưa có dữ liệu
        st.markdown("---")
        st.markdown("#### 🔍 BẢN DEMO (Dữ liệu mẫu)")
        # ... (giữ lại một phần giao diện demo để tab không bị trống)
        z_data = [[1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.5, 1, 1, 1]]
        fig_heat = ff.create_annotated_heatmap(z=z_data, x=['T2','T3','T4','T5','T6','T7','CN'], y=['W1','W2','W3','W4'],
                                              colorscale='Viridis', showscale=False)
        fig_heat.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True, theme=None)
    else:
        # CHẾ ĐỘ DỮ LIỆU THẬT
        patient_opts = {}
        for h in history_data:
            u = h.get("username") or ""
            fn = h.get("full_name") or u or "Không rõ"
            if u or fn:
                patient_opts[u or fn] = f"👤 {fn}" + (f" ({u})" if u else "")
        filter_patient = st.selectbox(
            "Lọc theo bệnh nhân:",
            ["-- Tất cả --"] + sorted(patient_opts.values(), key=str.lower),
            key="filter_tien_trien_patient",
        )
        if filter_patient != "-- Tất cả --":
            sel_u = next((k for k, v in patient_opts.items() if v == filter_patient), None)
            history_data = [
                h for h in history_data
                if h.get("username") == sel_u
                or h.get("full_name") == sel_u
                or patient_opts.get(h.get("username"), "") == filter_patient
            ]

        df_hist = pd.DataFrame(history_data)

        # 1. Chỉ số tổng hợp thực tế
        c1, c2, c3 = st.columns(3)
        avg_acc = df_hist['accuracy'].mean()
        max_acc = df_hist['accuracy'].max()
        total_sessions = len(df_hist)

        with c1:
            st.metric("🎯 Độ chính xác TB", f"{avg_acc:.1f}%")
        with c2:
            st.metric("🏆 Kỷ lục đạt được", f"{max_acc:.1f}%")
        with c3:
            st.metric("🎬 Tổng số buổi tập", f"{total_sessions}")

        st.markdown("---")

        # 2. Biểu đồ tiến triển thực tế
        st.markdown("#### 📉 BIỂU ĐỒ TĂNG TRƯỞNG HIỆU SUẤT")
        fig_real = px.line(df_hist, x='ngay', y='accuracy', color='bai_tap', markers=True,
                          title="Sự thay đổi độ chính xác qua các lần tập")
        fig_real.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_real, use_container_width=True, theme=None)

        # 3. Bảng lịch sử chi tiết
        st.markdown("#### 📑 NHẬT KÝ TẬP LUYỆN CHI TIẾT")
        show_cols = [c for c in ['ngay', 'full_name', 'bai_tap', 'accuracy', 'f1', 'thoi_gian_tap'] if c in df_hist.columns]
        df_show = df_hist[show_cols].copy()
        if 'full_name' in df_show.columns:
            df_show.rename(columns={'full_name': 'Bệnh nhân', 'ngay': 'Thời gian tập xong', 'bai_tap': 'Bài tập'}, inplace=True)
        if 'accuracy' in df_show.columns:
            df_show['accuracy'] = df_show['accuracy'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(df_show, use_container_width=True)

        # 4. Nút xóa lịch sử (để làm mới nếu cần)
        if st.button("🗑️ Xóa toàn bộ lịch sử", type="secondary"):
            save_data(HISTORY_FILE, [])
            st.rerun()

# ============================================
# HÀM HIỂN THỊ TAB: HƯỚNG DẪN SỬ DỤNG (MỚI)
# ============================================

# ============================================
# HÀM HIỂN THỊ TAB: PHẢN HỒI (MỚI)
# ============================================
def hien_thi_tab_phan_hoi():
    """Giao diện cộng đồng: Góp ý và hiển thị bình luận công khai"""
    st.markdown("### 💬 CỘNG ĐỒNG REHAB-AI: GÓP Ý & THẢO LUẬN")

    feedback_file = FEEDBACK_FILE

    # Tải danh sách phản hồi hiện có
    comments = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
        except: pass

    col_f1, col_f2 = st.columns([1, 1.2])

    with col_f1:
        st.markdown("#### 📮 Để lại ý kiến của bạn")
        with st.form("feedback_form", clear_on_submit=True):
            default_name = st.session_state.user_info.get('username', '') if st.session_state.user_info else ''
            user_name = st.text_input("Tên của bạn", value=default_name)
            user_msg = st.text_area("Nội dung góp ý/thảo luận")
            submitted = st.form_submit_button("Gửi bình luận", width="stretch", type="primary")

            if submitted:
                if user_name and user_msg:
                    try:
                        require_role(PATIENT_ROLE, DOCTOR_ROLE, RESEARCHER_ROLE, ADMIN_ROLE, action="create_feedback", target="feedback")
                        new_comment = {
                            "name": user_name,
                            "message": user_msg,
                            "time": get_vn_now().strftime("%H:%M - %d/%m/%Y")
                        }
                        comments.insert(0, new_comment) # Đưa bình luận mới lên đầu
                        save_data(FEEDBACK_FILE, comments)
                        st.balloons()
                        st.success("Cảm ơn bạn! Bình luận đã được đăng công khai.")
                        st.rerun()
                    except PermissionError as exc:
                        st.error(str(exc))
                else:
                    st.warning("⚠️ Vui lòng nhập đầy đủ tên và nội dung.")

        st.markdown("#### 📞 Thông tin hỗ trợ kỹ thuật")
        is_light = st.session_state.theme == 'light'
        box_bg = "rgba(0,114,255,0.05)" if is_light else "rgba(255,255,255,0.05)"
        st.markdown(f"""
        <div style="background: {box_bg}; padding: 1.2rem; border-radius: 15px; border: 1px solid #2a5298;">
            <p>📧 <b>Email:</b> 2211090031@studenthuph.edu.vn</p>
            <p>🏫 <b>Đơn vị:</b> Khoa KHDL Y sinh - HUPH</p>
            <p>📍 <b>Vị trí:</b> 1A Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        st.markdown(f"#### 🗨️ Thảo luận gần đây ({len(comments)})")

        if not comments:
            st.info("Chưa có bình luận nào. Hãy là người đầu tiên để lại ý kiến!")
        else:
            is_light = st.session_state.theme == 'light'
            item_bg = "rgba(0,0,0,0.03)" if is_light else "rgba(255,255,255,0.08)"
            item_text = "#333" if is_light else "#ccc"
            name_color = "#0072ff" if is_light else "#ffd700"
            # Hiển thị danh sách bình luận dưới dạng thẻ
            for c in comments[:20]: # Hiển thị 20 bình luận mới nhất
                c_name = safe_html(c.get('name', 'Ẩn danh'), max_length=80)
                c_time = safe_html(c.get('time', ''), max_length=40)
                c_message = safe_html(c.get('message', ''), max_length=1000)
                st.markdown(f"""
                <div style="background: {item_bg}; padding: 1rem; border-radius: 12px; margin-bottom: 10px; border-left: 4px solid #00CED1;">
                    <div style="display: flex; justify-content: space-between;">
                        <b style="color: {name_color};">👤 {c_name}</b>
                        <span style="color: #666; font-size: 0.8rem;">{c_time}</span>
                    </div>
                    <p style="color: {item_text}; margin-top: 5px; font-size: 0.95rem;">{c_message}</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================
# LỚP XỬ LÝ VIDEO REAL-TIME (WEBRTC) — import lazy trong hien_thi_tab_realtime
# ============================================
def hien_thi_tab_realtime(bai_tap):
    """Xử lý Camera trực tiếp qua Trình duyệt (WebRTC)"""
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

    class PoseProcessor(VideoProcessorBase):
        def __init__(self):
            init_mediapipe()
            self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.bai_tap = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w, _ = img.shape

            # Xử lý với MediaPipe
            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                ve_khung_xuong_custom(img, results.pose_landmarks, active_side="LEFT", mau_tong=(0, 255, 0), scale_factor=w/640.0)

                try:
                    landmarks = results.pose_landmarks.landmark
                    # Lấy tọa độ các khớp (Bên trái)
                    vai = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                    khuyu = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                    co_tay = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
                    hong = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]

                    g_vai = tinh_goc(hong, vai, khuyu)
                    g_khuyu = tinh_goc(vai, khuyu, co_tay)

                    # Hiển thị
                    cv2.putText(img, f"VAI: {int(g_vai)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, f"KHUYU: {int(g_khuyu)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Cảnh báo (Nếu có bài tập chuẩn)
                    if self.bai_tap:
                        if abs(g_vai - self.bai_tap['chuan']['vai']) > self.bai_tap['chuan']['sai_so']:
                            cv2.putText(img, "⚠️ SAI TU THE VAI!", (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                except Exception:
                    pass

            import av
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    st.markdown("### 📹 TẬP LUYỆN TRỰC TIẾP VỚI AI (REAL-TIME)")
    st.info("💡 Trình duyệt sẽ yêu cầu quyền Camera. Hãy nhấn 'Allow' để bắt đầu.")

    ice_servers = [{"urls": WEBRTC_STUN_URLS}] if WEBRTC_STUN_URLS else []
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})
    if not WEBRTC_STUN_URLS:
        st.caption("WebRTC đang chạy không cấu hình STUN ngoài. Đặt WEBRTC_STUN_URLS nếu môi trường triển khai cần NAT traversal.")

    col1, col2 = st.columns([1.3, 1])

    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>🎯 BÀI TẬP: {bai_tap['ten']}</h4>
            <p>🔄 <b>Trạng thái:</b> So sánh thời gian thực</p>
            <p>📽️ <b>Mục tiêu:</b> Khớp với Video mẫu</p>
            <hr>
            <p style="font-size: 0.8rem; color: #aaa;">Hệ thống sẽ vẽ khung xương và đối chiếu chuyển động của bạn với video chuẩn theo từng giây.</p>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        webrtc_ctx = webrtc_streamer(
            key="rehab-pose",
            video_processor_factory=PoseProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.bai_tap = bai_tap


# ============================================
# HÀM HIỂN THỊ TAB 8: KIẾN THỨC PHCN
# ============================================


# ============================================
# HÀM HIỂN THỊ TAB 7: THÔNG TIN & CÔNG NGHỆ
# ============================================


def hien_thi_tab_phan_tich_va_video_ncv():
    """Gộp tab Phân tích và Video cho Nghiên cứu viên"""
    st.markdown("## 🔬 PHÂN TÍCH CHUYÊN SÂU & DỮ LIỆU KHUNG XƯƠNG")

    _dong_bo_video_list_day_du_tu_hf(force=False)

    v_cur = _lam_moi_ban_ghi_video_tu_db(
        st.session_state.get("current_eval_video") or _tim_video_phan_tich_moi_nhat()
    )
    if v_cur:
        st.session_state.current_eval_video = v_cur
        vp = v_cur.get("video_path")
        if vp and finalize_background_analysis_if_ready(vp):
            v_cur = _lam_moi_ban_ghi_video_tu_db(st.session_state.current_eval_video) or v_cur
            st.session_state.current_eval_video = v_cur
            st.toast("✅ Phân tích xong! Đang hiển thị kết quả...", icon="🎉")
            st.session_state._pending_chart_refresh = True
        slot_cur = _slot_video_phan_tich(v_cur)
        if (
            slot_cur
            and st.session_state.get("_ncv_analysis_loaded_key")
            and st.session_state.get("_ncv_analysis_loaded_key") != slot_cur
        ):
            _xoa_session_phan_tich()
        need_load = (
            v_cur.get("metrics")
            and not (
                st.session_state.get("reanalyze_triggered", False)
                and not st.session_state.get("view_old_analysis", False)
            )
            and (
                not _session_phan_tich_khop_video(v_cur)
                or st.session_state.get("angle_df") is None
            )
        )
        if need_load:
            with st.spinner(
                f"📥 Đang nạp kết quả: {v_cur.get('full_name')} — {v_cur.get('exercise')}..."
            ):
                _nap_bieu_do_nhanh_tu_cloud(v_cur)
                st.session_state.view_old_analysis = True
        v_cur = _lam_moi_ban_ghi_video_tu_db(st.session_state.current_eval_video or v_cur)
    if v_cur:
        st.info(
            f"📌 Đang xem phân tích: **{v_cur.get('full_name', 'N/A')}** — "
            f"**{v_cur.get('exercise', 'N/A')}** · `{v_cur.get('video_name', '')}`"
        )
    if v_cur and v_cur.get("username"):
        if st.session_state.get("reanalyze_triggered") and v_cur.get("video_path"):
            prog = read_progress(v_cur["video_path"])
            if prog and prog.get("status") == "processing":
                p_pct = prog.get("progress", 0.0) * 100
                st.info(
                    f"🔄 **Đang chạy phân tích mới** — tiến độ **{p_pct:.0f}%**. "
                    "Mở tab **📊 BIỂU ĐỒ PHÂN TÍCH** bên dưới để xem chi tiết và video gốc."
                )
        st.markdown("---")

    ncv_sub_list = ["📊 BIỂU ĐỒ PHÂN TÍCH", "🎬 VIDEO & ẢNH FRAME"]
    if st.session_state.get("ncv_sub_tab") not in ncv_sub_list:
        st.session_state.ncv_sub_tab = ncv_sub_list[0]
    ncv_sub = st.segmented_control(
        "Sub menu NCV",
        options=ncv_sub_list,
        default=st.session_state.ncv_sub_tab,
        key="ncv_sub_tab_widget",
        label_visibility="collapsed",
    )
    if ncv_sub:
        st.session_state.ncv_sub_tab = ncv_sub
    else:
        ncv_sub = st.session_state.ncv_sub_tab
    if ncv_sub == "📊 BIỂU ĐỒ PHÂN TÍCH":
        hien_thi_tab_phan_tich(key_suffix="ncv_combined_tab")
    else:
        _dam_bao_du_lieu_video_frames_truoc_hien_thi(v_cur)
        hien_thi_frames_day_du(key_suffix="ncv_combined_video_tab")

def _dam_bao_du_lieu_video_frames_truoc_hien_thi(v=None):
    """Nạp metadata + khởi động tải song song trước tab VIDEO — không block UI."""
    v = _lam_moi_ban_ghi_video_tu_db(v or st.session_state.get("current_eval_video"))
    if not v or not v.get("metrics"):
        return
    _gan_session_ket_qua_tu_video(v)
    if st.session_state.get("angle_df") is None:
        _ap_dung_angle_df_tu_video(v)
    _bat_dau_tai_day_du_song_song(v)


def hien_thi_tab_danh_gia_tong_hop_benh_nhan():
    """Gộp tab Kết quả đánh giá cho Bệnh nhân (Chỉ hiện kết quả bác sĩ/ncv)"""
    # Xóa header thừa ở đây vì tab đã có tiêu đề
    hien_thi_ket_qua_cho_benh_nhan()


def hien_thi_tab_danh_gia_va_nckh_bac_si():
    render_doctor_tab("📊 QUẢN LÝ ĐÁNH GIÁ & NCKH", _build_ui_tab_dependencies())


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
def get_pose_model(model_type="MediaPipe Heavy", min_confidence=0.5):
    """Khởi tạo MediaPipe Pose với cấu hình linh hoạt"""
    # pyrefly: ignore [missing-import]
    # Kích hoạt thiết lập tài nguyên ảo và monkey-patching
    setup_mediapipe_resources()

    import mediapipe as mp
    mp_pose = mp.solutions.pose

    complexity = 1
    if "Lite" in model_type: complexity = 0
    elif "Heavy" in model_type: complexity = 2

    try:
        return mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
    except Exception as e:
        if complexity == 2:
            st.warning(f"⚠️ Lỗi khởi tạo MediaPipe Heavy ({e}). Tự động chuyển sang mô hình MediaPipe Full.")
            try:
                return mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=min_confidence,
                    min_tracking_confidence=min_confidence
                )
            except Exception as e2:
                st.warning("⚠️ Không thể tải mô hình MediaPipe Full. Đang chuyển sang mô hình MediaPipe Lite.")
                return mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    smooth_landmarks=True,
                    min_detection_confidence=min_confidence,
                    min_tracking_confidence=min_confidence
                )
        elif complexity == 1:
            st.warning("⚠️ Không thể tải mô hình MediaPipe Full. Đang chuyển sang mô hình MediaPipe Lite.")
            return mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence
            )
        else:
            raise e

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


def ve_khung_xuong_custom(frame_output, current_landmarks, active_side=None, mau_tong=(0, 255, 0), scale_factor=1.0):
    """Vẽ khung xương 33 điểm tùy chỉnh chuyên nghiệp, đảm bảo hiển thị 100% cả hai bên chân"""
    h, w = frame_output.shape[:2]
    lm = current_landmarks.landmark
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(33)]

    line_thickness = max(3, int(3 * scale_factor))
    circle_rad = max(4, int(4.5 * scale_factor))

    # Định nghĩa các liên kết xương vẽ thủ công
    LIEN_KET_TRAI = [
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Tay trái
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31)             # Chân trái
    ]
    LIEN_KET_PHAI = [
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Tay phải
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)             # Chân phải
    ]
    LIEN_KET_THAN = [
        (11, 12), (11, 23), (12, 24), (23, 24)                       # Thân mình
    ]
    LIEN_KET_MAT = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),              # Mắt, mũi
        (3, 7), (6, 8), (9, 10)                                      # Tai, miệng
    ]

    # 1. Vẽ các đường nối (connections) trước
    # Đường nối thân mình
    for start_idx, end_idx in LIEN_KET_THAN:
        cv2.line(frame_output, pts[start_idx], pts[end_idx], (180, 180, 180), line_thickness)

    # Đường nối đầu/mặt
    for start_idx, end_idx in LIEN_KET_MAT:
        cv2.line(frame_output, pts[start_idx], pts[end_idx], (200, 200, 200), max(1, line_thickness - 1))

    # Đường nối bên trái
    color_trai = mau_tong if active_side in ["LEFT", "BOTH"] else (180, 180, 180)
    for start_idx, end_idx in LIEN_KET_TRAI:
        cv2.line(frame_output, pts[start_idx], pts[end_idx], color_trai, line_thickness)

    # Đường nối bên phải
    color_phai = mau_tong if active_side in ["RIGHT", "BOTH"] else (180, 180, 180)
    for start_idx, end_idx in LIEN_KET_PHAI:
        cv2.line(frame_output, pts[start_idx], pts[end_idx], color_phai, line_thickness)

    # 2. Vẽ các điểm khớp (joints)
    # Khớp mặt
    for i in range(11):
        cv2.circle(frame_output, pts[i], max(2, circle_rad - 1), (255, 255, 255), -1)

    # Khớp thân và tứ chi
    for i in range(11, 33):
        is_left = i in [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        is_active = (active_side == "BOTH") or (active_side == "LEFT" and is_left) or (active_side == "RIGHT" and not is_left)

        # Màu sắc khớp: Vàng sáng cho bên active, Trắng cho bên inactive
        color_khop = (0, 235, 255) if is_active else (240, 240, 240)
        cv2.circle(frame_output, pts[i], circle_rad, color_khop, -1)


# ============================================
# XỬ LÝ FRAME - CẢI THIỆN BOX THÔNG TIN
# ============================================================
def _build_processing_dependencies():
    """Build dependencies for core processing helpers."""
    return SimpleNamespace(
        **{k: v for k, v in globals().items() if not k.startswith("__")}
    )


def xu_ly_frame(frame, model, chuan, frame_idx, fps=30, dynamic_chuan=None, active_side=None, last_pose_landmarks=None, precomputed_landmarks=None, exercise_name="codman"):
    """Compatibility wrapper for video.processing.xu_ly_frame."""
    from video.processing import xu_ly_frame as _impl
    return _impl(_build_processing_dependencies(), frame=frame, model=model, chuan=chuan, frame_idx=frame_idx, fps=fps, dynamic_chuan=dynamic_chuan, active_side=active_side, last_pose_landmarks=last_pose_landmarks, precomputed_landmarks=precomputed_landmarks, exercise_name=exercise_name)


def ve_nhan_ml_classifier(frame_output, ml_info, scale_factor=1.0):
    """Draw the trained classifier output on a processed frame."""
    if draw_ml_badge:
        return draw_ml_badge(frame_output, ml_info, scale_factor=scale_factor)
    return frame_output


def ve_nhan_rule_classifier(frame_output, dung, gan_dung, scale_factor=1.0):
    """Draw YouTube reference comparison label (PASS / NEARLY / FAIL) on frame."""
    if draw_rule_badge:
        return draw_rule_badge(frame_output, dung, gan_dung, scale_factor=scale_factor)
    return frame_output

def ensure_voice_files(force_voice=False):
    import os
    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
    if not os.path.exists(sounds_dir):
        os.makedirs(sounds_dir, exist_ok=True)

    _VOICE_FILES = ("dung.mp3", "gan_dung.mp3", "sai.mp3")
    _BEEP_FREQ = {"dung.mp3": 880, "gan_dung.mp3": 660, "sai.mp3": 440}
    _TTS_TEXT = {"dung.mp3": "Đúng", "gan_dung.mp3": "Gần đúng", "sai.mp3": "Sai"}
    marker_path = os.path.join(sounds_dir, ".voice_tts_ok")

    def _bundle_ready():
        if not os.path.exists(marker_path):
            return False
        return all(
            os.path.exists(os.path.join(sounds_dir, f)) and os.path.getsize(os.path.join(sounds_dir, f)) > 0
            for f in _VOICE_FILES
        )

    if not force_voice and _bundle_ready():
        return sounds_dir

    any_tts = False
    for filename in _VOICE_FILES:
        filepath = os.path.join(sounds_dir, filename)
        need_tts = force_voice or not _bundle_ready()

        saved = False
        if need_tts and ALLOW_NETWORK_TTS:
            try:
                from gtts import gTTS
                tts = gTTS(text=_TTS_TEXT[filename], lang='vi')
                tts.save(filepath)
                saved = os.path.exists(filepath) and os.path.getsize(filepath) > 0
                if saved:
                    any_tts = True
                    print(f"[Audio] gTTS voice: {filename}")
            except Exception as _gte:
                print(f"[Audio] gTTS fail cho {filename}: {_gte}")
        elif need_tts and not ALLOW_NETWORK_TTS:
            print("[Audio] Network TTS disabled; using local beep fallback. Set ALLOW_NETWORK_TTS=true to enable gTTS.")

        if not saved and os.path.exists(filepath) and os.path.getsize(filepath) > 0 and not force_voice:
            saved = True

        if not saved:
            try:
                freq = _BEEP_FREQ[filename]
                subprocess.run(
                    ['ffmpeg', '-y', '-f', 'lavfi',
                     '-i', f'sine=frequency={freq}:duration=0.6',
                     '-ar', '44100', '-ac', '1', filepath],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10
                )
                saved = os.path.exists(filepath) and os.path.getsize(filepath) > 0
                if saved:
                    print(f"[Audio] fallback beep cho {filename} ({freq}Hz)")
            except Exception as _ffe:
                print(f"[Audio] ffmpeg beep fail cho {filename}: {_ffe}")

    if any_tts:
        try:
            with open(marker_path, "w", encoding="utf-8") as mf:
                mf.write("gtts\n")
        except Exception:
            pass
    elif _bundle_ready():
        pass
    else:
        try:
            if os.path.exists(marker_path):
                os.remove(marker_path)
        except Exception:
            pass

    missing = [
        f for f in _VOICE_FILES
        if not os.path.exists(os.path.join(sounds_dir, f))
        or os.path.getsize(os.path.join(sounds_dir, f)) == 0
    ]
    if missing:
        print(f"[Audio] Van thieu file am thanh: {missing}")

    return sounds_dir


# ============================================
# HÀM TẠO ZIP THEO YÊU CẦU (LAZY ZIP - TRÁNH OOM)
# ============================================
def create_zip_of_frames(frame_data_list, processed_video_path=None):
    """Nén tất cả các frame ảnh thành file ZIP.
    Nếu file ảnh bị thiếu, tự động giải nén/trích xuất trực tiếp từ video đã xử lý trên fly để đảm bảo đầy đủ.
    """
    import zipfile
    import tempfile
    import time
    import os

    if not frame_data_list:
        return None

    timestamp = int(time.time())
    zip_path = os.path.join(tempfile.gettempdir(), f"frames_{timestamp}.zip")

    cap = None
    if processed_video_path and os.path.exists(processed_video_path):
        try:
            cap = cv2.VideoCapture(processed_video_path)
        except Exception as e:
            print("Lỗi mở video phục hồi frame khi tạo ZIP:", e)
            cap = None

    try:
        # Nếu frame_data_list là list of strings (để tương thích ngược)
        if len(frame_data_list) > 0 and isinstance(frame_data_list[0], str):
            paths = frame_data_list
            frame_data_list = []
            for idx, p in enumerate(paths):
                try:
                    f_name = os.path.basename(p)
                    f_idx_str = ''.join(c for c in f_name if c.isdigit())
                    f_idx = int(f_idx_str) if f_idx_str else idx + 1
                except:
                    f_idx = idx + 1
                frame_data_list.append({'index': f_idx, 'path': p})

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for idx, f_data in enumerate(frame_data_list):
                if not isinstance(f_data, dict):
                    continue
                f_path = get_local_frame_path(f_data.get('path'))
                if not f_path:
                    continue

                # Phục hồi ảnh nếu thiếu và có video nguồn
                if not os.path.exists(f_path) and cap and cap.isOpened():
                    try:
                        os.makedirs(os.path.dirname(f_path), exist_ok=True)
                        f_idx = idx
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ret, frame_img = cap.read()
                        if ret:
                            cv2.imwrite(f_path, frame_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    except Exception as err:
                        print(f"Lỗi khôi phục frame {f_data.get('index')} khi nén ZIP: {err}")

                # Nén vào file ZIP
                if os.path.exists(f_path):
                    z.write(f_path, os.path.basename(f_path))
        return zip_path
    except Exception as e:
        print("Lỗi tạo file ZIP robust:", e)
        return None
    finally:
        if cap:
            cap.release()

# ============================================
# XỬ LÝ VIDEO
# ============================================
def dong_bo_va_chuan_hoa_exercise(username, video_name, video_path, original_exercise):
    # 1. Nhận diện động tác thực tế
    video_filename_clean = os.path.basename(video_path or video_name or '').lower()
    exercise_clean = str(original_exercise or '').lower()

    ref_name_detected = "codman"
    if "codman" in video_filename_clean:
        ref_name_detected = "codman"
    elif any(kw in video_filename_clean for kw in ["gậy", "gay", "pulley", "stick"]):
        ref_name_detected = "gay"
    elif any(kw in video_filename_clean for kw in ["dây", "day", "kháng lực", "khang", "theraband", "band"]):
        ref_name_detected = "day"
    else:
        if "codman" in exercise_clean:
            ref_name_detected = "codman"
        elif any(kw in exercise_clean for kw in ["gậy", "gay", "pulley", "stick"]):
            ref_name_detected = "gay"
        elif any(kw in exercise_clean for kw in ["dây", "day", "kháng lực", "khang", "theraband", "band"]):
            ref_name_detected = "day"

    correct_ex_name = "Bài tập con lắc Codman"
    if ref_name_detected == "gay":
        correct_ex_name = "Bài tập với gậy (Pulley Exercise)"
    elif ref_name_detected == "day":
        correct_ex_name = "Bài tập với dây kháng lực (Theraband)"

    # 2. Đồng bộ trong video_list.json
    video_list = load_data(VIDEOS_FILE)
    updated_vid = False
    for v in video_list:
        if v.get('username') == username and (v.get('video_name') == video_name or v.get('video_path') == video_path):
            v['exercise'] = correct_ex_name
            updated_vid = True
    if updated_vid:
        save_data(VIDEOS_FILE, video_list)

    # 3. Đồng bộ trong doctor_evaluations.json
    evals = load_data(EVALUATIONS_FILE)
    updated_eval = False
    for e in evals:
        if e.get('patient_username') == username and (e.get('video_name') == video_name or (video_name and video_name in e.get('video_name', ''))):
            e['exercise'] = correct_ex_name
            updated_eval = True
    if updated_eval:
        save_data(EVALUATIONS_FILE, evals)

    # 4. Đồng bộ trong patient_symptoms.json
    symptoms = load_data(SYMPTOMS_FILE)
    updated_symp = False
    for s in symptoms:
        if s.get('username') == username and s.get('exercise') == original_exercise:
            s['exercise'] = correct_ex_name
            updated_symp = True
    if updated_symp:
        save_data(SYMPTOMS_FILE, symptoms)

    return correct_ex_name

def xu_ly_video_day_du(duong_dan_video, chuan, callback=None, model_type="MediaPipe Heavy", min_confidence=0.5, exercise_name="codman", skip_step=None, resize_width=None, force_train_classifier=False, checkpoint_video_path=None):
    """Compatibility wrapper for video.processing.xu_ly_video_day_du."""
    from video.processing import xu_ly_video_day_du as _impl
    return _impl(_build_processing_dependencies(), duong_dan_video=duong_dan_video, chuan=chuan, callback=callback, model_type=model_type, min_confidence=min_confidence, exercise_name=exercise_name, skip_step=skip_step, resize_width=resize_width, force_train_classifier=force_train_classifier, checkpoint_video_path=checkpoint_video_path)

# =====================================================================
# BACKGROUND VIDEO ANALYSIS ENGINE (XỬ LÝ VIDEO DƯỚI NỀN BẤT ĐỒNG BỘ)
# =====================================================================
import threading
import hashlib
import traceback

_db_lock = threading.Lock()

# Số video phân tích chạy SONG SONG. HF Space mặc định 1 (Gậy + Heavy: chạy từng video).
# Ghi đè: biến môi trường MAX_CONCURRENT_ANALYSIS=2
_hf_default_concurrent = "1" if (os.environ.get("HF_SPACE_ID") or os.environ.get("SPACE_ID") or os.path.exists("/data")) else "4"
try:
    MAX_CONCURRENT_ANALYSIS = max(1, min(8, int(os.environ.get("MAX_CONCURRENT_ANALYSIS", _hf_default_concurrent))))
except (TypeError, ValueError):
    MAX_CONCURRENT_ANALYSIS = 1 if _hf_default_concurrent == "1" else 4
JOB_ORPHAN_SECONDS = 90  # Không có heartbeat trong 90s → coi job bị gián đoạn, tự khởi động lại

_analysis_registry = AnalysisJobRegistry(
    MAX_CONCURRENT_ANALYSIS,
    orphan_seconds=JOB_ORPHAN_SECONDS,
)
_running_threads = _analysis_registry.running_threads
_cancel_flags = _analysis_registry.cancel_flags   # video_path -> threading.Event(); set() = yêu cầu thread dừng
_analysis_slots = _analysis_registry.slots

def doc_lock_save_data(file_path, handle_fn):
    """
    Hàm tiện ích giúp đọc, xử lý và ghi lại file JSON một cách thread-safe sử dụng _db_lock
    """
    with _db_lock:
        update_app_json(file_path, handle_fn)
        try:
            _load_data_cached.clear()
            _load_video_list_core.clear()
            _video_nghien_cuu_cached.clear()
            _evals_dedup_cached.clear()
        except Exception:
            pass
        push_file_to_hf_async(file_path)

PROGRESS_STALE_SECONDS = 28800  # 8 giờ — video nặng (Heavy/720p >5000 frames) có thể mất 3-5h, cần đủ thời gian

def get_progress_file(video_path):
    """Trả về đường dẫn file progress JSON tương ứng với video_path"""
    if not video_path:
        return ""
    clean_p = video_path.replace("\\", "/")
    h = hashlib.md5(clean_p.encode('utf-8')).hexdigest()
    return os.path.join(PROCESSED_DIR, f"progress_{h}.json")

def _load_progress_file(video_path):
    """Đọc file progress thuần — không xóa, không side-effect."""
    p_file = get_progress_file(video_path)
    if not p_file or not os.path.exists(p_file):
        return None
    try:
        with open(p_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


_analysis_registry.configure_progress_loader(
    _load_progress_file,
    orphan_seconds=JOB_ORPHAN_SECONDS,
)


def _tien_do_phan_tich_hien_tai(video_path, ckpt=None):
    """Lấy % / elapsed / start_time tốt nhất — giữ tiến độ sau push HF / redeploy Space."""
    if ckpt is None and video_path:
        ckpt = load_checkpoint(get_checkpoint_path(video_path, PROCESSED_DIR))
    if ckpt and ckpt.get("pass1_data"):
        ui_prog, ui_msg = checkpoint_ui_progress(ckpt)
        return {
            "progress": ui_prog,
            "elapsed": float(ckpt.get("elapsed") or 0),
            "start_time": ckpt.get("start_time") or time.time(),
            "status_msg": ui_msg,
        }
    existing = _load_progress_file(video_path) or {}
    if existing.get("status") == "processing":
        try:
            prog = min(max(float(existing.get("progress") or 0), 0.01), 0.99)
        except (TypeError, ValueError):
            prog = 0.01
        return {
            "progress": prog,
            "elapsed": float(existing.get("elapsed") or 0),
            "start_time": existing.get("start_time") or time.time(),
            "status_msg": existing.get("status_msg") or "🔄 Tiếp tục phân tích sau khi Space khởi động lại...",
        }
    return {
        "progress": 0.01,
        "elapsed": 0.0,
        "start_time": time.time(),
        "status_msg": "🚀 Đang chuẩn bị phân tích...",
    }

def read_progress(video_path):
    """Đọc tiến trình từ đĩa. Không xóa job đang chạy khi Streamlit reload (thread in-memory mất)."""
    data = _load_progress_file(video_path)
    if not data:
        return None
    if data.get("status") == "error":
        err_msg = data.get("error_msg", "")
        if "final_h264" in err_msg or "referenced before assignment" in err_msg:
            p_file = get_progress_file(video_path)
            try:
                if p_file and os.path.exists(p_file):
                    os.remove(p_file)
            except Exception:
                pass
            return None
    if data.get("status") == "processing":
        p_file = get_progress_file(video_path)
        try:
            mtime = os.path.getmtime(p_file) if p_file and os.path.exists(p_file) else 0
            heartbeat = float(data.get("heartbeat") or data.get("start_time") or mtime)
            if time.time() - max(mtime, heartbeat) > PROGRESS_STALE_SECONDS:
                print(f"[Progress] Job qua han 2h, bo qua hien thi: {video_path}")
                return None
        except Exception:
            pass
    return data

_last_progress_hf_push = {}
_resume_phan_tich_lock = threading.Lock()
_resume_phan_tich_done = False


def _day_progress_checkpoint_len_hf(video_path, p_file=None, force=False, progress=None, status=None):
    """Đẩy progress + checkpoint lên HF Dataset — giãn cách để không vượt 128 commit/giờ."""
    if not (HF_TOKEN and HF_DATASET_ID):
        return
    key = video_path or p_file or ""
    if not key:
        return
    now = time.time()
    on_space = bool(HF_SPACE_ID or os.path.exists("/data"))
    throttle = 300 if on_space else 600
    last = _last_progress_hf_push.get(key, {})
    prog = float(progress if progress is not None else last.get("prog", 0.0) or 0.0)
    st = status or last.get("status") or ""
    terminal = st in ("success", "error")
    prog_jump = abs(prog - float(last.get("prog", 0.0) or 0.0)) >= 0.10
    status_changed = bool(st and st != last.get("status"))

    if not force and not terminal:
        if (now - float(last.get("t", 0) or 0)) < throttle and not prog_jump and not status_changed:
            return

    _last_progress_hf_push[key] = {"t": now, "prog": prog, "status": st}

    prio = 0 if terminal else (1 if force else 4)
    if p_file and os.path.exists(p_file):
        push_file_to_hf_async(p_file, priority=prio)
    if video_path:
        ckpt = get_checkpoint_path(video_path, PROCESSED_DIR)
        if ckpt and os.path.exists(ckpt) and os.path.getsize(ckpt) > 100:
            if force or terminal or prog_jump:
                push_file_to_hf_async(ckpt, priority=prio + 1)


def _tai_trang_thai_phan_tich_tu_hf(force=False):
    """Tải progress/checkpoint đang chạy từ HF Dataset sau khi Space redeploy."""
    if not (HF_TOKEN and HF_DATASET_ID):
        return 0
    restored = 0
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        files, list_err = list_dataset_files(HF_TOKEN, HF_DATASET_ID)
        if list_err:
            raise RuntimeError(list_err)
        for rel in files:
            rel_norm = rel.replace("\\", "/")
            if not rel_norm.startswith("processed_results/"):
                continue
            base = os.path.basename(rel_norm)
            dst = os.path.join(PROCESSED_DIR, base)
            need = force
            if base.startswith("progress_") and base.endswith(".json"):
                # JSON nhỏ — luôn tải lại sau deploy để giữ đúng % hiển thị
                need = True
            elif base.startswith("checkpoint_") and base.endswith(".pkl.gz"):
                # Dọn sạch nếu dst là thư mục từ lần download thất bại trước đó
                if os.path.isdir(dst):
                    try:
                        import shutil as _shutil2
                        _shutil2.rmtree(dst, ignore_errors=True)
                    except Exception:
                        pass
                if os.path.isfile(dst) and os.path.getsize(dst) > 100 and load_checkpoint(dst):
                    need = False
                else:
                    need = force or not os.path.isfile(dst) or os.path.getsize(dst) < 100
            if need and _hf_download_dataset_file(rel_norm, quiet=True, min_size=2):
                if base.startswith("checkpoint_") and base.endswith(".pkl.gz"):
                    if load_checkpoint(dst) is None:
                        try:
                            if os.path.exists(dst):
                                import shutil as _shutil3
                                (_shutil3.rmtree if os.path.isdir(dst) else os.remove)(dst)
                        except Exception:
                            pass
                        continue
                restored += 1
    except Exception as e:
        print(f"[HF Resume] Loi tai trang thai phan tich: {e}")
    if restored:
        print(f"[HF Resume] Da tai {restored} file progress/checkpoint tu Dataset")
    return restored


def _chay_khoi_phuc_phan_tich_sau_deploy():
    """Tải progress từ Dataset + khởi động lại thread — gọi một lần khi Space boot."""
    global _resume_phan_tich_done
    with _resume_phan_tich_lock:
        if _resume_phan_tich_done:
            return 0
        try:
            _analysis_slots._purge_dead()
            _tai_trang_thai_phan_tich_tu_hf(force=True)
        except Exception as hf_restore_err:
            print(f"[HF Resume] Loi tai progress tu Dataset: {hf_restore_err}")
        # Dong bo video_list.json MOI NHAT tu HF TRUOC khi resume. Neu bo qua buoc nay,
        # guard "_video_da_co_ket_qua_luu" doc ban video_list.json cu (seed git, thieu
        # metrics moi luu) -> khong nhan ra video DA co ket qua -> resume chay lai phan
        # tich thua (dung CPU, lam web do va nhay %). Co metrics -> guard danh dau
        # success va bo qua, khong chay lai.
        try:
            dong_bo_json_cau_hinh_tu_hf(force_files=frozenset({"video_list.json"}))
            _xoa_cache_sau_dong_bo_json(["video_list.json"])
        except Exception as sync_err:
            print(f"[HF Resume] Loi dong bo video_list.json truoc resume: {sync_err}")
        try:
            n = khoi_phuc_job_phan_tich_sau_deploy(cold_start=True)
            if n:
                print(f"[Resume] Da khoi dong lai {n} job phan tich sau khoi dong Space")
        except Exception as resume_err:
            print(f"[Resume] Loi khoi phuc job: {resume_err}")
            n = 0
        _resume_phan_tich_done = True
        return n


def write_progress(video_path, status, username="", video_name="", progress=0.0, elapsed=0.0, start_time=None, error_msg="", result=None, status_msg="", job_meta=None):
    """Ghi thông tin tiến trình xuống đĩa"""
    p_file = get_progress_file(video_path)
    if not p_file:
        return
    existing = _load_progress_file(video_path) or {}
    merged_meta = {**(existing.get("job_meta") or {}), **(job_meta or {})}
    data = {
        "video_path": video_path,
        "username": username,
        "video_name": video_name,
        "status": status,
        "progress": progress,
        "elapsed": elapsed,
        "start_time": start_time if start_time is not None else (existing.get("start_time") or time.time()),
        "heartbeat": time.time(),
        "error_msg": error_msg,
        "result": result,
        "status_msg": status_msg
    }
    if merged_meta:
        data["job_meta"] = merged_meta
    try:
        with open(p_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        _day_progress_checkpoint_len_hf(
            video_path, p_file, progress=progress, status=status
        )
    except Exception as e:
        print(f"Lỗi ghi progress file: {e}")

def clear_analysis_progress(video_path):
    """Xóa file progress cũ để tránh nạp nhầm kết quả phân tích trước đó."""
    if not video_path:
        return
    p_file = get_progress_file(video_path)
    try:
        if p_file and os.path.exists(p_file):
            os.remove(p_file)
    except Exception as exc:
        print(f"[Progress] Khong xoa duoc progress file: {exc}")

def _video_da_co_ket_qua_luu(v):
    if not v:
        return False
    has_metrics = bool(v.get("metrics"))
    try:
        acc_ok = float(v.get("accuracy") or 0) > 0
    except Exception:
        acc_ok = False
    status_txt = str(v.get("status") or "").lower()
    status_ok = "phân tích" in status_txt or "phÃ¢n tÃ­ch" in status_txt
    has_artifact = bool(v.get("df_path") or v.get("processed_path") or v.get("all_frames_data_path"))
    return has_metrics or acc_ok or (status_ok and has_artifact)


def _tim_video_da_hoan_tat_cho_job(job):
    vp = (job or {}).get("video_path") or ""
    uname = (job or {}).get("username") or ""
    vname = (job or {}).get("video_name") or ""
    names = {os.path.basename(str(x or "")).lower() for x in (vp, vname)}
    candidates = []
    try:
        candidates.extend(load_data(VIDEOS_FILE) or [])
    except Exception:
        pass
    try:
        candidates.extend(load_danh_sach_video_nghien_cuu() or [])
    except Exception:
        pass
    seen = set()
    for v in candidates:
        key = (v.get("username"), v.get("video_name"), v.get("exercise"), v.get("video_path"))
        if key in seen:
            continue
        seen.add(key)
        if not _video_da_co_ket_qua_luu(v):
            continue
        paths = {str(v.get(k) or "") for k in ("video_path", "processed_path")}
        if vp and vp in paths:
            return v
        if uname and v.get("username") == uname and vname and v.get("video_name") == vname:
            return v
        cand_names = {os.path.basename(p).lower() for p in paths if p}
        cand_names.add(os.path.basename(str(v.get("video_name") or "")).lower())
        if names and cand_names and names.intersection(cand_names):
            return v
    return None


def _dong_progress_neu_da_co_ket_qua_luu(job):
    v_done = _tim_video_da_hoan_tat_cho_job(job)
    vp = (job or {}).get("video_path") or (v_done or {}).get("video_path")
    if not (v_done and vp):
        return False
    fz = _frames_zip_path_from_video(v_done)
    result_data = {
        "stats": v_done.get("metrics") or {},
        "processed_video_path": v_done.get("processed_path"),
        "df_path": v_done.get("df_path"),
        "all_frames_data_path": v_done.get("all_frames_data_path"),
        "exercise": v_done.get("exercise"),
        "frames_zip": fz,
        "frames_zip_path": fz,
    }
    write_progress(
        vp,
        "success",
        username=(job or {}).get("username") or v_done.get("username"),
        video_name=(job or {}).get("video_name") or v_done.get("video_name"),
        progress=1.0,
        elapsed=float((job or {}).get("elapsed") or 0),
        start_time=(job or {}).get("start_time"),
        result=result_data,
        status_msg="✅ Đã có kết quả đã lưu, không cần chạy lại.",
        job_meta={
            "full_name": ((job or {}).get("job_meta") or {}).get("full_name") or v_done.get("full_name"),
            "exercise_name": v_done.get("exercise"),
        },
    )
    try:
        clear_checkpoint(get_checkpoint_path(vp, PROCESSED_DIR))
    except Exception:
        pass
    print(f"[Resume] Bo qua resume vi da co ket qua luu: {v_done.get('video_name') or os.path.basename(vp)}")
    return True


def clear_all_progress_files():
    """Xóa toàn bộ file tiến trình (progress_*.json) để làm mới — không còn job nào hiển thị 'đang tải'.
    Trả về số file đã xóa."""
    require_role(RESEARCHER_ROLE, ADMIN_ROLE, action="clear_progress", target="processed_results")
    removed = 0
    if not os.path.exists(PROCESSED_DIR):
        return 0
    try:
        for fn in os.listdir(PROCESSED_DIR):
            if fn.startswith("progress_") and fn.endswith(".json"):
                try:
                    os.remove(os.path.join(PROCESSED_DIR, fn))
                    removed += 1
                except Exception:
                    pass
    except Exception as e:
        print(f"[Reset] Loi xoa progress files: {e}")
    return removed


def _don_dep_thread_phan_tich(video_path):
    """Gỡ thread phân tích đã chết khỏi registry."""
    _analysis_registry.cleanup_dead_thread(video_path)


def _xoa_cache_h264_video(video_path):
    """Xóa bản H.264 _f.mp4 để buộc tạo lại (kèm âm thanh sau phân tích mới)."""
    if not video_path:
        return
    for p in {video_path, get_final_h264_path(video_path)}:
        if p and p.endswith("_f.mp4") and os.path.exists(p):
            try:
                os.remove(p)
                print(f"[Reanalyze] Da xoa cache H264: {p}")
            except Exception as exc:
                print(f"[Reanalyze] Khong xoa duoc {p}: {exc}")


def _chuan_bi_phan_tich_lai(video_path, v=None):
    """Xóa checkpoint và cache H.264 cũ trước khi chạy phân tích mới."""
    if video_path:
        clear_checkpoint(get_checkpoint_path(video_path, PROCESSED_DIR))
    if v:
        proc = v.get("processed_path")
        if proc:
            _xoa_cache_h264_video(proc)


def khoi_dong_phan_tich_lai_video(v, auto_start=True):
    """
    Chuẩn bị và khởi chạy phân tích lại: MediaPipe 33 điểm + REF YouTube + ML Classifier.
    Giữ kết quả đã lưu trên màn hình nếu video đã có metrics — phân tích mới chạy nền.
    """
    if not v:
        return False
    v = _lam_moi_ban_ghi_video_tu_db(v) or v
    video_path = v.get("video_path")
    co_ket_qua_cu = bool(v.get("metrics"))
    clear_analysis_progress(video_path)
    if video_path:
        done_key = f"_bg_done_{hashlib.md5(video_path.encode()).hexdigest()}"
        st.session_state.pop(done_key, None)
    st.session_state.reanalyze_triggered = True
    st.session_state["_analysis_started_this_session"] = True
    st.session_state.current_eval_video = v
    # Luôn xóa dữ liệu cũ khi bắt đầu phân tích mới — người dùng thấy màn hình tiến độ ngay
    st.session_state.view_old_analysis = False
    st.session_state.has_data = False
    st.session_state.stats = None
    st.session_state.angle_df = None
    st.session_state.processed_video_path = None
    st.session_state.current_df_csv_path = None
    st.session_state.all_frames_data_path = None
    st.session_state.frames_zip = None
    st.session_state.pop("_ncv_analysis_loaded_key", None)
    _ = co_ket_qua_cu  # không dùng nữa nhưng giữ để tránh lỗi reference

    if not video_path:
        return {"started": False, "reason": "no_video"}
    if not auto_start:
        return {"started": True, "reason": "prepared"}

    _chuan_bi_phan_tich_lai(video_path, v)
    ncv_gd = st.session_state.get("ncv_giai_doan", PHASE_UI_LABELS["g2"])
    # Uu tien thiet lap CUU-HO (video dai tren HF) do _bat_che_do_cuu_ho_hf dat qua key
    # rieng — tranh phai ghi de widget-key (gay StreamlitAPIException). Pop ngay de khong
    # dinh sang lan phan tich binh thuong sau do.
    _force_model = st.session_state.pop("_ncv_force_model", None)
    _force_resize = st.session_state.pop("_ncv_force_resize", None)
    _force_skip = st.session_state.pop("_ncv_force_skip", None)
    return bat_dau_phan_tich_background(
        video_path=video_path,
        username=v.get("username"),
        full_name=v.get("full_name"),
        video_name=v.get("video_name"),
        exercise_name=v.get("exercise"),
        giai_doan=ncv_gd,
        model_type=_force_model or st.session_state.get("ncv_model_type", "MediaPipe Heavy"),
        confidence=st.session_state.get("ncv_confidence", 0.5),
        skip_step=_force_skip if _force_skip is not None else st.session_state.get("ncv_skip_frames", 0),
        resize_width=_force_resize or st.session_state.get("ncv_resize_width", 720),
        force_train_classifier=True,
        force_restart=True,
    )

def find_progress_by_video_info(username, video_name):
    """Tìm thông tin tiến trình background cho cặp username và video_name"""
    if not os.path.exists(PROCESSED_DIR):
        return None, None
    for fn in os.listdir(PROCESSED_DIR):
        if fn.startswith("progress_") and fn.endswith(".json"):
            p_file = os.path.join(PROCESSED_DIR, fn)
            try:
                with open(p_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("username") == username and data.get("video_name") == video_name:
                    return data.get("video_path"), data
            except:
                pass
    return None, None

def check_and_populate_background_result(video_path):
    """
    Kiểm tra xem video có tiến trình background hoàn tất thành công hay không.
    Nếu thành công, nạp kết quả vào session_state và xóa file progress để reset trạng thái.
    """
    prog = read_progress(video_path)
    if prog and prog.get("status") == "success":
        result = prog.get("result", {})
        if result:
            st.session_state.stats = result.get("stats")
            st.session_state.has_data = True

            # Đọc DataFrame từ CSV
            df_path = result.get("df_path")
            if df_path and os.path.exists(df_path):
                try:
                    st.session_state.angle_df = read_display_csv_fast(df_path)
                except Exception as e:
                    print("Lỗi đọc CSV trong check_and_populate:", e)
                    st.session_state.angle_df = None
            else:
                st.session_state.angle_df = None

            st.session_state.processed_video_path = result.get("processed_video_path")
            st.session_state.all_frames_data_path = result.get("all_frames_data_path")
            st.session_state.exercise = result.get("exercise")
            st.session_state.current_df_csv_path = df_path
            st.session_state.frames_zip = (
                result.get("frames_zip")
                or result.get("frames_zip_path")
                or _frames_zip_from_processed_path(result.get("processed_video_path"))
            )
            st.session_state.temp_frames_dir = result.get("temp_frames_dir")
            st.session_state.reanalyze_triggered = False
            st.session_state.view_old_analysis = True

            # Cập nhật st.session_state.current_eval_video để đồng bộ trạng thái phân tích
            try:
                all_vids = load_data(VIDEOS_FILE)
                updated_v = next((vid for vid in all_vids if vid.get('video_path') == video_path), None)
                if updated_v:
                    st.session_state.current_eval_video = updated_v
                else:
                    # Fallback dict nếu chưa cập nhật kịp vào file database
                    st.session_state.current_eval_video = {
                        "username": prog.get("username"),
                        "full_name": prog.get("full_name", "Bệnh nhân"),
                        "video_name": prog.get("video_name"),
                        "exercise": result.get("exercise", {}).get("ten", "codman") if isinstance(result.get("exercise"), dict) else "Bài tập",
                        "accuracy": result.get("stats", {}).get("do_chinh_xac", 0.0) if isinstance(result.get("stats"), dict) else 0.0,
                        "time": get_vn_now().strftime("%H:%M - %d/%m/%Y"),
                        "video_path": video_path,
                        "processed_path": result.get("processed_video_path"),
                        "metrics": result.get("stats"),
                        "df_path": df_path,
                        "all_frames_data_path": result.get("all_frames_data_path"),
                        "status": "Đã phân tích"
                    }
            except Exception as e_sync:
                print("Lỗi đồng bộ current_eval_video:", e_sync)

            v_sync = st.session_state.get("current_eval_video")
            if v_sync:
                _gan_khoa_session_phan_tich(v_sync)
                # CSV local mất (Space restart) — tải ngay từ HF Dataset để biểu đồ hiện nhanh
                if st.session_state.get("angle_df") is None and df_path and not os.path.exists(df_path):
                    try:
                        _bat_dau_tai_day_du_song_song(v_sync)
                    except Exception:
                        pass

            # Xóa progress file sau khi đã nạp kết quả
            p_file = get_progress_file(video_path)
            try:
                if os.path.exists(p_file):
                    os.remove(p_file)
            except:
                pass
            return True
    return False

def finalize_background_analysis_if_ready(video_path):
    """
    Nạp kết quả phân tích background khi đã xong.
    Trả về True nếu vừa nạp xong (cần rerun một lần để cập nhật UI).
    """
    if not video_path:
        return False
    prog = read_progress(video_path)
    if not prog or prog.get("status") != "success":
        return False
    done_key = f"_bg_done_{hashlib.md5(video_path.encode()).hexdigest()}"
    if st.session_state.get(done_key) and st.session_state.get("has_data"):
        clear_analysis_progress(video_path)
        return False
    loaded = check_and_populate_background_result(video_path)
    if loaded:
        st.session_state[done_key] = True
        st.session_state.reanalyze_triggered = False
        st.session_state.view_old_analysis = True
        return True
    return False

def poll_background_analysis_complete():
    """Nạp kết quả phân tích nền nếu đã xong; rerun ngay để hiện biểu đồ tự động."""
    v = st.session_state.get("current_eval_video")
    if not v:
        return
    video_path = v.get("video_path")
    if video_path and finalize_background_analysis_if_ready(video_path):
        st.rerun()

def _scan_progress_by_status(*statuses):
    """Quét file progress_*.json theo danh sách trạng thái."""
    items = []
    if not os.path.exists(PROCESSED_DIR):
        return items
    seen_paths = set()
    try:
        for fn in sorted(os.listdir(PROCESSED_DIR)):
            if not (fn.startswith("progress_") and fn.endswith(".json")):
                continue
            try:
                with open(os.path.join(PROCESSED_DIR, fn), 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            if not data or data.get("status") not in statuses:
                continue
            vp_key = data.get("video_path") or fn
            if vp_key in seen_paths:
                continue
            seen_paths.add(vp_key)
            items.append(data)
    except Exception as scan_err:
        print(f"[Jobs] Loi quet progress: {scan_err}")
    return items

def liet_ke_jobs_dang_chay():
    """Đọc tất cả file progress trên đĩa để lấy danh sách video đang trích xuất.
    Nhờ tiến trình ghi ra đĩa dùng chung nên mở link ở thiết bị/nền tảng nào cũng thấy."""
    return _scan_progress_by_status("processing")

def liet_ke_jobs_vua_xong():
    """Video vừa phân tích xong (status=success), chờ người dùng mở xem kết quả."""
    return _scan_progress_by_status("success")


def _thread_dang_chay_thuc_su(video_path=None):
    """True chỉ khi có thread Python thực sự đang alive trong _running_threads.
    Không bao giờ dựa vào progress file — tránh auto-refresh do file cũ/crash."""
    return _analysis_registry.is_thread_running(video_path)


def _co_job_dang_chay():
    # Chỉ trigger auto-refresh khi thread thực sự chạy (không dựa progress file)
    if _thread_dang_chay_thuc_su():
        return True
    # Fallback: jobs mới xong chờ xem kết quả (success)
    return bool(liet_ke_jobs_vua_xong())


def _interval_theo_doi_jobs():
    """Chỉ auto-refresh panel job khi thực sự có tiến trình — tránh rerun vô ích."""
    return timedelta(seconds=4) if _co_job_dang_chay() else None


def _noi_dung_jobs_dang_chay(key_suffix=""):
    """Nội dung panel theo dõi job phân tích (gọi từ fragment)."""
    jobs = liet_ke_jobs_dang_chay()
    done_jobs = liet_ke_jobs_vua_xong()
    if not jobs and not done_jobs:
        return
    if jobs:
        st.markdown(f"#### 🔄 Đang phân tích **{len(jobs)}** video (chạy nền — tối đa {MAX_CONCURRENT_ANALYSIS} song song)")
        st.caption(
            f"Tiến trình lưu trên đĩa + **checkpoint**: push Git/HF hoặc crash → tự chạy lại sau ~{JOB_ORPHAN_SECONDS}s, "
            f"**tiếp tục từ Bước 2** nếu Bước 1 đã xong. "
            f"Bài **Gậy** dùng **MediaPipe Heavy** — chạy **{MAX_CONCURRENT_ANALYSIS}** video/lúc (video tiếp theo xếp hàng)."
        )
    if done_jobs:
        st.success(f"✅ **{len(done_jobs)}** video đã phân tích xong — bấm **Xem kết quả** để mở (Codman / Gậy / ...).")
        for done_idx, job in enumerate(done_jobs):
            vname = job.get("video_name", "Video")
            vp = job.get("video_path", "")
            if st.button(f"📊 Xem kết quả: {vname}", key=f"view_done_{key_suffix}_{done_idx}_{hashlib.md5((vp or str(done_idx)).encode()).hexdigest()[:8]}", use_container_width=True):
                if vp and finalize_background_analysis_if_ready(vp):
                    st.rerun()
                elif vp:
                    check_and_populate_background_result(vp)
                    st.rerun()
    for job_idx, job in enumerate(jobs):
        vp = job.get("video_path", "")
        vname = job.get("video_name", "Video")
        prog = job.get("progress", 0.0)
        try:
            prog = min(max(float(prog), 0.0), 1.0)
        except (TypeError, ValueError):
            prog = 0.0
        msg = job.get("status_msg", "")
        elapsed = job.get("elapsed", 0.0)
        c1, c2 = st.columns([3.2, 1])
        with c1:
            try:
                st.progress(prog, text=f"🎬 {vname} — {prog*100:.1f}% · ⏱️ {elapsed:.0f}s · {msg}")
            except TypeError:
                st.progress(prog)
                st.caption(f"🎬 {vname} — {prog*100:.0f}% · {msg}")
        with c2:
            is_current = st.session_state.get("current_eval_video", {}).get("video_path") == vp
            if is_current:
                st.caption("👁️ Đang xem")
            elif st.button("👁️ Theo dõi", key=f"track_job_{key_suffix}_{job_idx}_{hashlib.md5((vp or str(job_idx)).encode()).hexdigest()[:8]}", use_container_width=True):
                vid = None
                try:
                    vid = _tim_video_cho_progress(vp)
                except Exception:
                    vid = None
                if not vid:
                    vid = {
                        "video_path": vp,
                        "username": job.get("username"),
                        "video_name": vname,
                        "full_name": job.get("full_name", "Bệnh nhân"),
                        "exercise": "codman",
                    }
                st.session_state.current_eval_video = vid
                st.session_state.reanalyze_triggered = True
                st.session_state.view_old_analysis = False
                st.session_state.pop("_ncv_analysis_loaded_key", None)
                st.rerun()
    st.markdown("---")


def hien_thi_jobs_dang_chay_fragment(key_suffix=""):
    """Panel theo dõi các video đang trích xuất khung xương — dùng chung cho mọi thiết bị/phiên."""
    # run_every phải là số/timedelta/None (không truyền callable) — đánh giá mỗi lần rerun script.
    def _job_status_fragment():
        _noi_dung_jobs_dang_chay(key_suffix)

    _job_status_fragment()


def hien_thi_tien_trinh_background(video_path):
    """Hiển thị giao diện tiến trình chạy nền"""
    prog = read_progress(video_path)
    if not prog:
        return False

    status = prog.get("status")
    if status == "processing":
        p_val = prog.get("progress", 0.0)
        elapsed = prog.get("elapsed", 0.0)

        # Thiết kế card UI hiện đại, sang trọng theo tiêu chuẩn Web App
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 20px; backdrop-filter: blur(10px);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <span style="color: #60a5fa; font-weight: 600; font-size: 1.1rem;">⚙️ Đang xử lý AI trong nền...</span>
                <span style="color: #94a3b8; font-size: 0.9rem;">⏱️ Đang chạy: {elapsed:.1f}s</span>
            </div>
            <div style="color: #e2e8f0; font-size: 0.95rem; margin-bottom: 15px;">
                Hệ thống đang thực hiện trích xuất khung xương và phân tích các chỉ số tập luyện lâm sàng.
                Bạn có thể an tâm <b>tắt trình duyệt, chuyển tab</b> hoặc làm việc khác. Tiến trình sẽ tự chạy đến khi hoàn tất.
            </div>
        </div>
        """, unsafe_allow_html=True)

        _hien_thi_progress_hai_pass(
            prog,
            status_msg=prog.get("status_msg", ""),
            elapsed_text=f"⏱️ {elapsed:.1f}s",
            show_total=True,
        )

        # Cho phép hủy và xem kết quả cũ nếu có kết quả cũ
        try:
            v_re = _tim_video_cho_progress(video_path)
            if v_re and v_re.get('metrics'):
                if st.button("⬅️ Quay lại xem kết quả cũ đã lưu", key=f"btn_cancel_processing_{hashlib.md5(video_path.encode()).hexdigest()}", type="secondary", use_container_width=True):
                    _quay_lai_ket_qua_cu_da_luu(v_re)
        except:
            pass
        return True

    elif status == "error":
        err_msg = prog.get("error_msg", "Lỗi không xác định")
        st.error(f"❌ Phân tích thất bại: {err_msg}")

        # Nút retry để xóa progress file
        if st.button("🔄 THỬ LẠI PHÂN TÍCH", type="primary", key=f"btn_retry_bg_{hashlib.md5(video_path.encode()).hexdigest()}"):
            p_file = get_progress_file(video_path)
            try:
                if os.path.exists(p_file):
                    os.remove(p_file)
            except:
                pass
            st.rerun()
        return True

    return False

def finalize_and_refresh_analysis(video_path):
    """Nạp kết quả phân tích nền khi xong và refresh UI để hiện biểu đồ ngay."""
    if not video_path:
        return
    if finalize_background_analysis_if_ready(video_path):
        v = _lam_moi_ban_ghi_video_tu_db(st.session_state.get("current_eval_video"))
        if v:
            st.session_state.current_eval_video = v
            _gan_khoa_session_phan_tich(v)
        st.toast("✅ Phân tích hoàn tất! Đang hiển thị kết quả...", icon="🎉")
        st.session_state._pending_chart_refresh = True

def hien_thi_tien_trinh_background_small(video_path):
    """Hiển thị tiến trình chạy nền nhỏ gọn bên trong cột phải (không reload toàn trang)"""
    def _background_progress_fragment():
        _noi_dung_tien_trinh_background_small(video_path)

    _background_progress_fragment()


def _noi_dung_tien_trinh_background_small(video_path):
    prog = read_progress(video_path)
    if not prog:
        st.write("Đang khởi động...")
        return

    status = prog.get("status")
    if status == "processing":
        p_val = prog.get("progress", 0.0)
        elapsed = prog.get("elapsed", 0.0)

        # Thêm spinner icon và hiệu ứng xoay CSS để tạo hiệu ứng đang chạy trực quan
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: bold; color: #60a5fa;">⚙️ Tiến độ phân tích <span class="spinner-icon-small">🔄</span></span>
            <span style="font-size: 0.85rem; color: #888;">⏱️ Đang chạy: {elapsed:.1f}s</span>
        </div>
        <style>
            @keyframes spin-small {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .spinner-icon-small {{
                display: inline-block;
                animation: spin-small 2s linear infinite;
                margin-left: 5px;
            }}
        </style>
        """, unsafe_allow_html=True)

        _hien_thi_progress_hai_pass(
            prog,
            status_msg=prog.get("status_msg", ""),
            elapsed_text=f"⏱️ {elapsed:.1f}s",
            show_total=True,
        )

        # Mách nước tối ưu hóa tốc độ
        st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.05); border: 1px solid rgba(255, 215, 0, 0.2); border-radius: 8px; padding: 10px; margin-top: 10px;">
            <span style="color: #ffd700; font-size: 0.85rem; font-weight: bold;">💡 Mẹo tăng tốc:</span>
            <span style="color: #ccc; font-size: 0.85rem;">Bạn có thể chỉnh <b>"Tốc độ xử lý"</b> ở sidebar bên trái thành <b>"Nhanh (Bỏ qua 2 hoặc 4 frame)"</b> để rút ngắn thời gian phân tích gấp 3-5 lần!</span>
        </div>
        """, unsafe_allow_html=True)

        # Cho phép hủy và xem kết quả cũ nếu có kết quả cũ
        try:
            v_re = _tim_video_cho_progress(video_path)
            if v_re and v_re.get('metrics'):
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                if st.button("⬅️ Quay lại xem kết quả cũ đã lưu", key=f"btn_cancel_proc_small_{hashlib.md5(video_path.encode()).hexdigest()}", type="secondary", use_container_width=True):
                    _quay_lai_ket_qua_cu_da_luu(v_re)
        except:
            pass
    elif status == "success":
        finalize_and_refresh_analysis(video_path)
    elif status == "error":
        err_msg = prog.get("error_msg", "Lỗi không xác định")
        st.error(f"❌ Phân tích thất bại: {err_msg}")
        if st.button("🔄 THỬ LẠI PHÂN TÍCH", type="primary", key=f"btn_retry_bg_small_{hashlib.md5(video_path.encode()).hexdigest()}"):
            p_file = get_progress_file(video_path)
            try:
                if os.path.exists(p_file):
                    os.remove(p_file)
            except:
                pass
            st.rerun()

def hien_thi_tien_trinh_background_home_fragment(video_path):
    """Hiển thị giao diện tiến trình chạy nền ở màn hình trang chủ (không reload toàn trang)"""
    def _home_progress_fragment():
        _noi_dung_tien_trinh_background_home(video_path)

    _home_progress_fragment()


def _noi_dung_tien_trinh_background_home(video_path):
    prog = read_progress(video_path)
    if not prog:
        st.write("Đang khởi động...")
        return

    status = prog.get("status")
    if status == "processing":
        p_val = prog.get("progress", 0.0)
        elapsed = prog.get("elapsed", 0.0)

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 20px; backdrop-filter: blur(10px);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <span style="color: #60a5fa; font-weight: 600; font-size: 1.1rem;">⚙️ Đang xử lý AI trong nền... <span class="spinner-icon">🔄</span></span>
                <span style="color: #94a3b8; font-size: 0.9rem;">⏱️ Đang chạy: {elapsed:.1f}s</span>
            </div>
            <div style="color: #e2e8f0; font-size: 0.95rem; margin-bottom: 15px;">
                Hệ thống đang thực hiện trích xuất khung xương và phân tích các chỉ số tập luyện lâm sàng.
                Bạn có thể an tâm <b>tắt trình duyệt, chuyển tab</b> hoặc làm việc khác. Tiến trình sẽ tự chạy đến khi hoàn tất.
            </div>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .spinner-icon {{
                display: inline-block;
                animation: spin 2s linear infinite;
                margin-left: 5px;
            }}
        </style>
        """, unsafe_allow_html=True)

        st.progress(p_val)
        st.info(f"🔄 Tiến độ tổng thể: {p_val*100:.1f}%")

        # Mách nước tối ưu hóa tốc độ
        st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.05); border: 1px solid rgba(255, 215, 0, 0.2); border-radius: 8px; padding: 10px; margin-top: 10px;">
            <span style="color: #ffd700; font-size: 0.85rem; font-weight: bold;">💡 Mẹo tăng tốc:</span>
            <span style="color: #ccc; font-size: 0.85rem;">Bạn có thể chỉnh <b>"Tốc độ xử lý"</b> ở sidebar bên trái thành <b>"Nhanh (Bỏ qua 2 hoặc 4 frame)"</b> để rút ngắn thời gian phân tích gấp 3-5 lần!</span>
        </div>
        """, unsafe_allow_html=True)

        # Cho phép hủy và xem kết quả cũ nếu có kết quả cũ
        try:
            v_re = _tim_video_cho_progress(video_path)
            if v_re and v_re.get('metrics'):
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                if st.button("⬅️ Quay lại xem kết quả cũ đã lưu", key=f"btn_cancel_proc_home_{hashlib.md5(video_path.encode()).hexdigest()}", type="secondary", use_container_width=True):
                    _quay_lai_ket_qua_cu_da_luu(v_re)
        except:
            pass
    elif status == "success":
        finalize_and_refresh_analysis(video_path)
    elif status == "error":
        err_msg = prog.get("error_msg", "Lỗi không xác định")
        st.error(f"❌ Phân tích thất bại: {err_msg}")
        if st.button("🔄 THỬ LẠI PHÂN TÍCH", type="primary", key=f"btn_retry_bg_home_{hashlib.md5(video_path.encode()).hexdigest()}"):
            p_file = get_progress_file(video_path)
            try:
                if os.path.exists(p_file):
                    os.remove(p_file)
            except:
                pass
            st.rerun()

def _hien_thi_gan_lai_video_ui(v, video_path, key_suffix):
    """UI gắn lại / tải lên video thay thế khi file gốc BN không còn trên server hoặc Cloud."""
    # Kiểm tra _f.mp4 có sẵn trên HF Dataset không (cached HEAD request)
    _has_cloud_h264 = False
    _cloud_h264_path = get_final_h264_path(_strip_to_original_upload(video_path))
    if _is_hf_runtime() and HF_TOKEN and HF_DATASET_ID and _cloud_h264_path and _cloud_h264_path != video_path:
        try:
            _, _u = _hf_dataset_resolve_urls(_cloud_h264_path, prefer_raw=False)
            _has_cloud_h264 = bool(_u and check_cloud_file_exists(_u))
        except Exception:
            pass

    with st.expander("📎 Gắn lại / tải lên video thay thế", expanded=True):
        if _has_cloud_h264:
            st.success(
                f"✅ **Phát hiện file H.264 trên Cloud Dataset** ({os.path.basename(_cloud_h264_path)}) — "
                "phân tích có thể tự tải về và chạy lại."
            )
            st.caption("Bấm **THỬ LẠI PHÂN TÍCH** bên phải để tải về và phân tích ngay.")
            st.divider()

        st.markdown("**Tải lên video gốc từ máy tính:**")
        _new_file = st.file_uploader(
            "Chọn file video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key=f"file_relink_{key_suffix}",
            label_visibility="collapsed",
        )
        if _new_file is not None:
            st.caption(f"Đã chọn: **{_new_file.name}** ({_new_file.size // 1024 // 1024} MB)")
            _new_ok, _new_msg = validate_uploaded_video_file(_new_file)
            if not _new_ok:
                st.error(f"🚫 {_new_msg}")
            if _new_ok and st.button(
                "✅ Gắn lại video này vào hồ sơ BN",
                type="primary",
                key=f"btn_confirm_relink_{key_suffix}",
                use_container_width=True,
            ):
                with st.spinner("Đang lưu video..."):
                    _saved = False
                    # Normalize path (Windows backslash → absolute Linux path)
                    _save_path = get_local_frame_path(video_path) or os.path.normpath(
                        os.path.join(DATA_DIR, video_path.replace("\\", "/"))
                    )
                    try:
                        # Ưu tiên: ghi đè lại đúng video_path cũ (không cần cập nhật DB)
                        _save_dir = os.path.dirname(_save_path)
                        if _save_dir:
                            os.makedirs(_save_dir, exist_ok=True)
                        _safe_new_name = sanitize_filename(_new_file.name, fallback="replacement.mp4")
                        _tmp_save_path = _save_path + ".upload_tmp" + (os.path.splitext(_safe_new_name)[1].lower() or ".mp4")
                        with open(_tmp_save_path, "wb") as _wf:
                            _wf.write(_new_file.getbuffer())
                        _probe_ok, _probe_msg = validate_video_file_for_processing(_tmp_save_path)
                        if not _probe_ok:
                            try:
                                os.remove(_tmp_save_path)
                            except Exception:
                                pass
                            raise ValueError(_probe_msg)
                        os.replace(_tmp_save_path, _save_path)
                        push_file_to_hf_async(_save_path, priority=1)
                        _saved = True
                    except Exception:
                        pass

                    if not _saved:
                        # Fallback: lưu vào path mới và cập nhật DB
                        try:
                            _ts = time.strftime("%Y%m%d_%H%M%S")
                            _pn = (v.get("full_name") or "BN").replace("/", "_")
                            _ex = (v.get("exercise") or "tap").replace("/", "_")
                            _new_name = f"{_pn}_{_ts}_{_pn} - {_ex}.mp4"
                            _save_path = os.path.join(UPLOAD_DIR, _new_name)
                            os.makedirs(UPLOAD_DIR, exist_ok=True)
                            _safe_new_name = sanitize_filename(_new_file.name, fallback="replacement.mp4")
                            _tmp_save_path = _save_path + ".upload_tmp" + (os.path.splitext(_safe_new_name)[1].lower() or ".mp4")
                            with open(_tmp_save_path, "wb") as _wf:
                                _wf.write(_new_file.getbuffer())
                            _probe_ok, _probe_msg = validate_video_file_for_processing(_tmp_save_path)
                            if not _probe_ok:
                                try:
                                    os.remove(_tmp_save_path)
                                except Exception:
                                    pass
                                raise ValueError(_probe_msg)
                            os.replace(_tmp_save_path, _save_path)
                            # Cập nhật video_path trong DB
                            _vlist = load_data(VIDEOS_FILE)
                            for _rec in _vlist:
                                if _rec.get("video_path") == video_path:
                                    _rec["video_path"] = _save_path
                                    break
                            save_data(VIDEOS_FILE, _vlist)
                            push_file_to_hf_async(VIDEOS_FILE)
                            push_file_to_hf_async(_save_path, priority=1)
                            _saved = True
                        except Exception as _err2:
                            st.error(f"❌ Không thể lưu video: {_err2}")

                if _saved:
                    st.success("✅ Video đã được gắn lại thành công! Đang làm mới trang...")
                    time.sleep(0.5)
                    _lam_moi_giao_dien_sau_nut()


def hien_thi_video_goc_fragment(video_or_v, key_suffix, video_name=""):
    """Hiển thị/ẩn video gốc trong fragment riêng -> bấm nút không làm rerun cả trang,
    nhờ vậy phần trích xuất khung xương bên cạnh KHÔNG bị tải lại từ đầu."""
    if isinstance(video_or_v, dict):
        vrec = video_or_v
        video_path = _lay_duong_dan_video_tho(vrec)
        video_name = vrec.get("video_name") or video_name
    else:
        vrec = {"video_path": video_or_v, "video_name": video_name}
        video_path = video_or_v
    show_key = f"show_src_video_{key_suffix}"
    # Mặc định luôn hiển thị video (True nếu chưa set)
    if show_key not in st.session_state:
        st.session_state[show_key] = True
    if st.session_state[show_key]:
        if video_path:
            # render_video tự xử lý mọi fallback: local → HF Cloud stream → cảnh báo
            st.caption(f"🎬 Video gốc BN — {video_name or ''}")
            _vid_ok = render_video(video_path, check_h264=False, prefer_raw=True)
            if not _vid_ok:
                _hien_thi_gan_lai_video_ui(vrec, video_path, key_suffix)
        else:
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.5);border:1px dashed rgba(148,163,184,0.3);border-radius:12px;padding:28px;text-align:center;">
                <div style="font-size:2.2rem;margin-bottom:8px;">🎬</div>
                <div style="color:#94a3b8;font-size:0.9rem;">Không tìm thấy video — tải lên lại để tiếp tục</div>
            </div>""", unsafe_allow_html=True)
        if st.button("🙈 Ẩn video gốc", key=f"btn_hide_src_video_{key_suffix}", use_container_width=True):
            st.session_state[show_key] = False
            st.rerun()
    else:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 12px; padding: 18px;">
            <div style="font-weight: 700; color: #e2e8f0; margin-bottom: 6px;">🎬 Video gốc đã chọn</div>
            <div style="color: #94a3b8; font-size: 0.88rem;">{safe_html(video_name or 'Video bệnh nhân', max_length=180)}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👁️ Xem video gốc", key=f"btn_show_src_video_{key_suffix}", use_container_width=True):
            st.session_state[show_key] = True
            st.rerun()

def _tim_duong_dan_video_phan_tich_hien_tai(v, video_path, prog_data=None):
    """Tìm file video phân tích có thể phát: checkpoint đang ghi, kết quả mới, hoặc bản đã lưu."""
    candidates = []
    if video_path:
        try:
            ckpt = load_checkpoint(get_checkpoint_path(video_path, PROCESSED_DIR))
            if ckpt and ckpt.get("out_path"):
                candidates.append(ckpt["out_path"])
        except Exception:
            pass
    if prog_data:
        result = prog_data.get("result") or {}
        proc = result.get("processed_video_path")
        if proc:
            candidates.append(proc)
    ss_proc = st.session_state.get("processed_video_path")
    if ss_proc:
        candidates.append(ss_proc)
    if v:
        db_proc = v.get("processed_path")
        if db_proc:
            candidates.append(db_proc)
    seen = set()
    for raw in candidates:
        if not raw or raw in seen:
            continue
        seen.add(raw)

        is_processing = prog_data and prog_data.get("status") == "processing"
        if is_processing:
            # Nếu đang trong quá trình phân tích, chỉ dùng file đã sẵn sàng cục bộ, tránh gọi hàm tải mạng đồng bộ gây đơ web
            ready = find_ready_local_video(raw)
            if ready:
                pb = resolve_playback_video_path(ready)
                play = pb if (pb and is_local_file_ready(pb) and os.path.getsize(pb) > 5 * 1024) else ready
                if play and is_local_file_ready(play) and os.path.getsize(play) > 5 * 1024:
                    return play
            continue

        play = dam_bao_tai_video_phan_tich(raw, allow_sync_transcode=False)
        if play and is_local_file_ready(play) and os.path.getsize(play) > 5 * 1024:
            return play
        for fb in video_fallback_paths(raw):
            if is_local_file_ready(fb) and os.path.getsize(fb) > 5 * 1024:
                return fb
        play = _dam_bao_video_san_sang_play(raw, prefer_raw=False)
        if play and is_local_file_ready(play) and os.path.getsize(play) > 5 * 1024:
            return play
    return None


def _pass_progress_from_total(progress_value, status_msg="", status="processing"):
    """Tách progress tổng thành Pass 1/Pass 2 để UI luôn thấy cả hai pass lên 100%."""
    try:
        p = min(max(float(progress_value or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        p = 0.0
    if status == "success" or p >= 0.995:
        return 1.0, 1.0

    p1 = 0.0
    p2 = 0.0
    if p >= 0.45:
        p1 = 1.0
    elif p > 0.18:
        p1 = min(max((p - 0.18) / 0.27, 0.0), 1.0)

    if p >= 0.90:
        p2 = 1.0
    elif p > 0.50:
        p2 = min(max((p - 0.50) / 0.40, 0.0), 1.0)

    import re as _re_pass
    msg = status_msg or ""
    m1 = _re_pass.search(r"Bước 1/2.*?\((\d+(?:\.\d+)?)%\)", msg)
    m2 = _re_pass.search(r"Bước 2/2.*?\((\d+(?:\.\d+)?)%\)", msg)
    if m1:
        p1 = max(p1, min(float(m1.group(1)) / 100.0, 1.0))
    if m2:
        p1 = 1.0
        p2 = max(p2, min(float(m2.group(1)) / 100.0, 1.0))
    return p1, p2


def _hien_thi_progress_hai_pass(prog_data, status_msg="", elapsed_text="", show_total=True):
    """Render tổng tiến trình + hai pass 100% rõ ràng cho vùng phân tích."""
    prog_data = prog_data or {}
    status = prog_data.get("status", "processing")
    try:
        p_val = min(max(float(prog_data.get("progress", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        p_val = 0.0
    # Chong nhay LUI nho (vd 22% -> 19%) do nhieu fragment/luong cung doc-ghi progress
    # lech nhau: giu lai % cao nhat da hien cho moi video trong phien. Van cho phep
    # RESET that su (tut > 12%) khi phan tich chay lai tu dau.
    if status == "processing":
        _vp_key = str(prog_data.get("video_path") or "")
        if _vp_key:
            try:
                _store = st.session_state.setdefault("_prog_monotonic", {})
                _last = _store.get(_vp_key)
                if _last is not None and 0 < (_last - p_val) < 0.12:
                    p_val = _last
                _store[_vp_key] = p_val
            except Exception:
                pass
    msg = status_msg or prog_data.get("status_msg", "")
    p1, p2 = _pass_progress_from_total(p_val, msg, status=status)

    if show_total:
        try:
            st.progress(p_val, text=f"Tiến độ tổng thể: {p_val * 100:.1f}%")
        except TypeError:
            st.progress(p_val)
            st.caption(f"Tiến độ tổng thể: **{p_val * 100:.1f}%**")
    try:
        st.progress(p1, text=f"Pass 1 - Trích xuất khung xương: {p1 * 100:.1f}%")
        st.progress(p2, text=f"Pass 2 - Vẽ nhãn, video và frames: {p2 * 100:.1f}%")
    except TypeError:
        st.progress(p1)
        st.caption(f"Pass 1 - Trích xuất khung xương: **{p1 * 100:.1f}%**")
        st.progress(p2)
        st.caption(f"Pass 2 - Vẽ nhãn, video và frames: **{p2 * 100:.1f}%**")
    detail = f" | {msg}" if msg else ""
    elapsed = f" | {elapsed_text}" if elapsed_text else ""
    st.caption(f"🔄 Tổng **{p_val * 100:.1f}%** · Pass 1 **{p1 * 100:.1f}%** · Pass 2 **{p2 * 100:.1f}%**{elapsed}{detail}")
    return p1, p2


def _hien_thi_tien_do_phan_tich_compact(prog_data, v, key_suffix):
    """Thanh tiến độ gọn cho cột video phân tích."""
    if not prog_data:
        return False, False
    status = prog_data.get("status")
    video_path = v.get("video_path")
    if status == "error":
        st.error(f"❌ Phân tích thất bại: {prog_data.get('error_msg', 'Lỗi không xác định')}")
        if st.button(
            "🔄 THỬ LẠI PHÂN TÍCH",
            width="stretch",
            type="primary",
            key=f"btn_retry_preview_{key_suffix}",
        ):
            clear_analysis_progress(video_path)
            _bat_che_do_cuu_ho_hf(video_path)
            _xu_ly_ket_qua_khoi_dong_phan_tich(khoi_dong_phan_tich_lai_video(v, auto_start=True))
        return False, True
    if status == "success":
        if finalize_background_analysis_if_ready(video_path):
            _lam_moi_giao_dien_sau_nut()
        else:
            st.rerun()
        return False, False
    if status == "processing":
        p_val = prog_data.get("progress", 0.0)
        status_msg = prog_data.get("status_msg", "")
        start_t = prog_data.get("start_time")
        elapsed_live = (time.time() - float(start_t)) if start_t else prog_data.get("elapsed", 0.0)
        detail = f" — {status_msg}" if status_msg else ""
        _hien_thi_progress_hai_pass(
            prog_data,
            status_msg=status_msg,
            elapsed_text=f"⏱️ {elapsed_live:.1f}s",
            show_total=True,
        )
        return True, False
    return False, False


def hien_thi_video_phan_tich_preview_fragment(v, key_suffix):
    """Cột phải: tiến độ + phát video phân tích (checkpoint hoặc bản đã lưu)."""
    video_path = v["video_path"]

    def _render_preview():
        prog_data = read_progress(video_path)
        is_processing, is_error = _hien_thi_tien_do_phan_tich_compact(prog_data, v, key_suffix)
        if is_error:
            return
        play_path = _tim_duong_dan_video_phan_tich_hien_tai(v, video_path, prog_data)
        if play_path:
            if is_processing:
                st.caption("🔄 Video phân tích đang được tạo — tự cập nhật theo tiến độ.")
            render_video(play_path, check_h264=True, prefer_raw=False)
        elif is_processing:
            st.info("⏳ Đang tạo video phân tích (khung xương, nhãn REF/ML, âm thanh Đúng/Sai/Gần đúng)...")
        elif v.get("processed_path"):
            st.warning("⚠️ Chưa tải được video phân tích từ Cloud — thử tải lại trang sau vài giây.")
        else:
            st.info("Video phân tích sẽ hiển thị ở đây khi quá trình hoàn tất.")

    _render_preview()


def _interval_khu_vuc_phan_tich(video_path):
    """Auto-refresh khi thread đang chạy hoặc progress file vẫn là 'processing'.
    Dừng refresh khi status=='success' — kết quả đã hiển thị, không cần tiếp.
    Stall detection (_STALL_SECONDS=180) sẽ hiện cảnh báo nếu thread thực sự đã chết.
    Dùng 2s thay vì 1s: 4 fragment cùng refresh 1s → server HF ngập request, nút bị bỏ qua."""
    if not video_path:
        return None
    if _thread_dang_chay_thuc_su(video_path):
        return timedelta(seconds=2.0)
    # reanalyze_triggered: vừa bấm nút, thread chưa kịp ghi progress → vẫn refresh để bắt kịp
    if st.session_state.get("reanalyze_triggered"):
        return timedelta(seconds=1.5)
    prog = read_progress(video_path)
    if not prog:
        return None
    status = prog.get("status")
    if status == "processing":
        return timedelta(seconds=2.0)
    return None


def _interval_tien_trinh_background(video_path):
    """Auto-refresh tiến trình khi thread đang chạy hoặc status còn 'processing'.
    Dừng refresh khi status=='success' để tránh fragment warning spam."""
    if not video_path:
        return None
    if _thread_dang_chay_thuc_su(video_path):
        return timedelta(seconds=3.0)
    prog = read_progress(video_path)
    if not prog:
        return None
    status = prog.get("status")
    if status == "processing":
        return timedelta(seconds=3.0)
    return None


def hien_thi_khu_vuc_phan_tich_chuyen_sau_fragment(v, key_suffix):
    """Compatibility wrapper for the deep analysis/progress area."""
    render_deep_analysis_area_page(_build_ui_tab_dependencies(), v, key_suffix)


def download_file_with_progress(file_path, write_progress_fn, start_t, username, video_name):
    """Tải file từ Hugging Face Dataset có cập nhật tiến độ (progress bar) từng chunk"""
    if not file_path:
        return False

    # Chuẩn hóa path: video_list.json có thể lưu backslash Windows ('.\\patient_uploads\\...')
    # Trên Linux (HF Spaces), os.path.dirname không nhận '\\' → trả về '' → makedirs lỗi.
    file_path = get_local_frame_path(file_path) or os.path.normpath(
        os.path.join(DATA_DIR, file_path.replace("\\", "/"))
    )

    # Xóa file cũ lỗi nếu có
    if os.path.exists(file_path):
        try: os.remove(file_path)
        except: pass

    if not (HF_TOKEN and HF_DATASET_ID):
        return False

    try:
        rel_path = get_clean_rel_path(file_path)
        last_pct_update = -1

        def _progress(downloaded, total_size):
            nonlocal last_pct_update
            if total_size <= 0:
                return
            dl_pct = downloaded / total_size
            # Tải video chiếm tiến trình từ 12% đến 18%
            prog_val = 0.12 + dl_pct * 0.06
            percent = int(dl_pct * 100)
            # Giảm tần suất cập nhật I/O tiến độ xuống đĩa
            if percent != last_pct_update:
                elapsed = time.time() - start_t
                write_progress_fn(
                    file_path, "processing",
                    username=username, video_name=video_name,
                    progress=prog_val, elapsed=elapsed, start_time=start_t,
                    status_msg=f"⬇️ Đang tải video từ Cloud: {downloaded/(1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB ({percent}%)"
                )
                last_pct_update = percent

        got, err = hf_download_dataset_file_with_progress(
            rel_path,
            file_path,
            token=HF_TOKEN,
            dataset_id=HF_DATASET_ID,
            progress_callback=_progress,
            min_size=5 * 1024,
        )
        if err:
            print(f"[Download Progress] Lỗi khi tải {rel_path}: {err}")
            return False

        return bool(got and os.path.exists(file_path) and os.path.getsize(file_path) >= 5 * 1024)
    except Exception as e:
        print(f"[Download Progress] Lỗi khi tải file {file_path}: {e}")
        return False

@st.cache_data(show_spinner=False)
def get_video_frame_count_cached(path, mtime, size):
    """Số khung hình + FPS (cache theo mtime/size)."""
    try:
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
        cap.release()
        return frames, fps
    except Exception:
        return 0, 15.0


def lay_so_khung_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return 0, 15.0
    try:
        mtime = os.path.getmtime(video_path)
        size = os.path.getsize(video_path)
        return get_video_frame_count_cached(video_path, mtime, size)
    except Exception:
        return 0, 15.0


def la_bai_tap_gay(exercise_name):
    ex = str(exercise_name or "").lower()
    return any(k in ex for k in ["gậy", "gay", "pulley", "stick"])


def tinh_tham_so_toc_do_phan_tich(video_path, exercise_name, model_type, skip_step, resize_width):
    """Tự động tối ưu tốc độ xử lý cho video dài.

    Khi NCV để mặc định (skip=0, resize=720), hàm này tự chỉnh:
    - Video > 3000 frames (~100s@30fps): skip=1 (2× nhanh hơn)
    - Video > 6000 frames (~200s@30fps): skip=2 (3× nhanh hơn), resize=480
    - Video > 12000 frames (~400s@30fps): skip=3 (4× nhanh hơn), resize=480
    Nếu NCV đã chọn skip > 0 hoặc resize ≠ 720 ở sidebar → giữ nguyên.
    """
    try:
        frames, fps = lay_so_khung_video(video_path)
        duration = (frames / fps) if fps > 0 else 0.0
    except Exception:
        frames, fps, duration = 0, 30.0, 0.0

    # Chỉ tự điều chỉnh khi NCV dùng mặc định (skip=0 và resize=720)
    user_chose_skip = (skip_step is not None and int(skip_step) > 0)
    user_chose_res = (resize_width is not None and int(resize_width) != 720)

    on_hf = _is_hf_runtime()
    if not user_chose_skip and frames > 0:
        if on_hf:
            # Giu day du frame; HF chi ha resolution/model neu can, khong tu ep bo frame.
            if frames > 9000 or duration > 300:
                skip_step = 0
            elif frames > 6000 or duration > 210:
                skip_step = 0
            elif frames > 3000 or duration > 105:
                skip_step = 0
            else:
                skip_step = 0
        else:
            if frames > 12000 or duration > 420:
                skip_step = 0
            elif frames > 6000 or duration > 210:
                skip_step = 0
            elif frames > 3000 or duration > 105:
                skip_step = 0
            else:
                skip_step = 0  # Video ngắn — giữ mọi frame

    if not user_chose_res and frames > 0:
        if on_hf and (frames > 3000 or duration > 105):
            resize_width = 480
        elif frames > 6000 or duration > 210:
            resize_width = 480  # 480p đủ chính xác cho MediaPipe, nhanh hơn ~2×

    # Ước tính thời gian sau tối ưu
    eff_skip = int(skip_step or 0)
    eff_frames = max(1, frames // (eff_skip + 1)) if frames > 0 else 0
    eff_res = int(resize_width or 720)
    if frames > 3000:
        print(
            f"[Analysis] Video dai ({frames} frames, {duration:.0f}s) — "
            f"Tu dong toi uu: skip={eff_skip}, resize={eff_res}p, "
            f"xu ly ~{eff_frames} frames (giam {100 - eff_frames * 100 // max(frames, 1):.0f}%)"
        )
    return model_type, skip_step, resize_width


def _checkpoint_qua_nang_cho_hf(video_path, ckpt):
    if not (_is_hf_runtime() and ckpt and video_path):
        return False
    try:
        frames, fps = lay_so_khung_video(video_path)
        duration = frames / fps if fps > 0 else 0
    except Exception:
        frames, duration = 0, 0
    if frames <= 6000 and duration <= 210:
        return False
    try:
        ckpt_skip = int(ckpt.get("skip_step") or 0)
    except Exception:
        ckpt_skip = 0
    try:
        ckpt_resize = int(ckpt.get("resize_width") or 720)
    except Exception:
        ckpt_resize = 720
    return ckpt_resize > 480


def _bat_che_do_cuu_ho_hf(video_path):
    if not _is_hf_runtime():
        return False
    try:
        frames, fps = lay_so_khung_video(video_path)
        duration = frames / fps if fps > 0 else 0
    except Exception:
        frames, duration = 0, 0
    if frames <= 3000 and duration <= 105:
        return False
    _skip = 0
    # KHONG ghi thang vao widget-key (ncv_model_type / ncv_resize_width / ncv_skip_frames):
    # cac selectbox sidebar DA duoc tao trong lan chay nay -> Streamlit nem
    # StreamlitAPIException "cannot be modified after the widget ... is instantiated".
    # Luu vao key override RIENG; khoi_dong_phan_tich_lai_video (luon goi NGAY sau ham nay)
    # se uu tien doc cac key nay.
    st.session_state["_ncv_force_model"] = "MediaPipe Lite"
    st.session_state["_ncv_force_resize"] = 480
    st.session_state["_ncv_force_skip"] = _skip
    return True


def tim_video_trong_db(video_path):
    if not video_path:
        return {}
    try:
        for v in load_data(VIDEOS_FILE):
            if v.get("video_path") == video_path:
                return v
    except Exception:
        pass
    return {}


def job_phan_tich_bi_gian_doan(video_path):
    """Thread đã chết hoặc heartbeat quá cũ — cần khởi động lại."""
    if not video_path:
        return False
    if _video_da_co_ket_qua_luu(tim_video_trong_db(video_path) or _tim_video_cho_progress(video_path)):
        return False
    if video_path in _running_threads and _running_threads[video_path].is_alive():
        return False
    prog = _load_progress_file(video_path)
    if not prog or prog.get("status") != "processing":
        return False
    hb = float(prog.get("heartbeat") or prog.get("start_time") or 0)
    return (time.time() - hb) >= JOB_ORPHAN_SECONDS


def khoi_phuc_job_phan_tich_sau_deploy(cold_start=False):
    """Khởi động lại job processing mất thread (sau deploy HF / crash / OOM)."""
    resumed = 0
    for job in liet_ke_jobs_dang_chay():
        vp = job.get("video_path")
        if not vp:
            continue
        if _dong_progress_neu_da_co_ket_qua_luu(job):
            continue
        # Kiểm tra heartbeat — nhưng ưu tiên checkpoint nếu có dữ liệu hợp lệ
        # Video Heavy/720p >5000 frames có thể cần 4-6h → không xóa nếu còn checkpoint
        try:
            _hb = float(job.get("heartbeat") or job.get("start_time") or 0)
            if _hb and (time.time() - _hb) > PROGRESS_STALE_SECONDS:
                # Kiểm tra checkpoint trước khi xóa — nếu còn Pass1 data thì resume thay vì bỏ
                _ckpt_check = load_checkpoint(get_checkpoint_path(vp, PROCESSED_DIR))
                if _checkpoint_qua_nang_cho_hf(vp, _ckpt_check):
                    print(f"[Resume] Xoa checkpoint qua nang tren HF de chay lai nhanh hon: {job.get('video_name')}")
                    clear_checkpoint(get_checkpoint_path(vp, PROCESSED_DIR))
                    clear_analysis_progress(vp)
                    continue
                if _ckpt_check and _ckpt_check.get("pass1_data"):
                    print(f"[Resume] Job heartbeat cu nhung co checkpoint hop le — se resume: {job.get('video_name')}")
                    # Reset heartbeat để tránh bị loop, nhưng KHÔNG xóa progress
                else:
                    print(f"[Resume] Bo qua job qua han ({(time.time()-_hb)/3600:.1f}h), khong co checkpoint: {job.get('video_name')}")
                    clear_analysis_progress(vp)
                    continue
        except Exception:
            pass
        if vp in _running_threads and _running_threads[vp].is_alive():
            continue
        if not cold_start and not job_phan_tich_bi_gian_doan(vp):
            continue
        try:
            ensure_local_file(vp)
        except Exception:
            pass
        if not os.path.exists(vp) or os.path.getsize(vp) < 1024:
            continue
        meta = job.get("job_meta") or {}
        db_v = tim_video_trong_db(vp)
        exercise = meta.get("exercise_name") or db_v.get("exercise") or "codman"
        model_type = meta.get("model_type") or "MediaPipe Heavy"
        confidence = meta.get("confidence", 0.5)
        skip_step = meta.get("skip_step")
        resize_width = meta.get("resize_width")
        giai_doan = meta.get("giai_doan") or PHASE_UI_LABELS.get("g2", "Giai đoạn 2")
        force_train = bool(meta.get("force_train_classifier", False))
        ckpt_resume = load_checkpoint(get_checkpoint_path(vp, PROCESSED_DIR))
        if _checkpoint_qua_nang_cho_hf(vp, ckpt_resume):
            print(f"[Resume] Bo checkpoint qua nang tren HF, chay lai che do nhanh: {job.get('video_name')}")
            clear_checkpoint(get_checkpoint_path(vp, PROCESSED_DIR))
            ckpt_resume = None
        if ckpt_resume and ckpt_resume.get("pass1_data"):
            model_type = ckpt_resume.get("model_type") or model_type
            skip_step = ckpt_resume.get("skip_step") if ckpt_resume.get("skip_step") is not None else skip_step
            resize_width = ckpt_resume.get("resize_width") or resize_width
            force_train = False
            ui_prog, ui_msg = checkpoint_ui_progress(ckpt_resume)
        else:
            model_type, skip_step, resize_width = tinh_tham_so_toc_do_phan_tich(
                vp, exercise, model_type, skip_step, resize_width
            )
            ui_prog = max(float(job.get("progress") or 0), 0.01)
            ui_msg = "🔄 Tiếp tục phân tích sau khi Space khởi động lại / job bị gián đoạn..."
        print(f"[Resume] Khoi dong lai phan tich: {job.get('video_name')} ({os.path.basename(vp)})")
        write_progress(
            vp, "processing",
            username=job.get("username") or db_v.get("username"),
            video_name=job.get("video_name") or db_v.get("video_name"),
            progress=ui_prog,
            elapsed=float(job.get("elapsed") or 0),
            start_time=job.get("start_time"),
            status_msg=ui_msg,
            job_meta={
                "full_name": meta.get("full_name") or db_v.get("full_name"),
                "exercise_name": exercise,
                "giai_doan": giai_doan,
                "model_type": model_type,
                "confidence": confidence,
                "skip_step": skip_step,
                "resize_width": resize_width,
                "force_train_classifier": force_train,
            },
        )
        bat_dau_phan_tich_background(
            video_path=vp,
            username=job.get("username") or db_v.get("username"),
            full_name=meta.get("full_name") or db_v.get("full_name"),
            video_name=job.get("video_name") or db_v.get("video_name"),
            exercise_name=exercise,
            giai_doan=giai_doan,
            model_type=model_type,
            confidence=confidence,
            skip_step=skip_step,
            resize_width=resize_width,
            force_train_classifier=force_train,
        )
        resumed += 1
    return resumed


def skip_step_theo_model(model_type, manual_skip=None):
    """
    Quy ước bỏ frame theo loại MediaPipe:
    - NCV chọn skip>0 ở sidebar: áp dụng cho mọi model (kể cả Heavy/Full).
    - Heavy/Full mặc định: lấy MỌI frame (skip=0).
    - Lite: tự bỏ frame (skip>=2) để xử lý nhanh.
    """
    try:
        ms = int(manual_skip) if manual_skip is not None else 0
    except (TypeError, ValueError):
        ms = 0
    if ms > 0:
        return ms
    mt = str(model_type or "")
    if "Lite" in mt:
        return max(ms, 2)
    return 0

def video_dang_phan_tich(video_path):
    """Video này đang có job phân tích thật sự chạy (thread sống hoặc heartbeat còn tươi)."""
    if not video_path:
        return False
    if _video_da_co_ket_qua_luu(tim_video_trong_db(video_path) or _tim_video_cho_progress(video_path)):
        return False
    if video_path in _running_threads and _running_threads[video_path].is_alive():
        return True
    prog = read_progress(video_path)
    if prog and prog.get("status") == "processing":
        hb = float(prog.get("heartbeat") or prog.get("start_time") or 0)
        if hb and (time.time() - hb) < JOB_ORPHAN_SECONDS:
            return True
    return False


def video_can_khoi_dong_phan_tich(v, only_pending=True):
    """True nếu video có thể đưa vào hàng đợi phân tích."""
    vp = v.get("video_path")
    if not vp or video_dang_phan_tich(vp):
        return False
    if not only_pending:
        return True
    has_metrics = bool(v.get("metrics"))
    acc = v.get("accuracy") or 0
    if has_metrics or (isinstance(acc, (int, float)) and float(acc) > 0) or v.get("status") == "Đã phân tích":
        return False
    return True


def bat_dau_phan_tich_hang_loat(video_records, only_pending=True, force_reanalyze=False):
    """Khởi chạy phân tích hàng loạt — tối đa MAX_CONCURRENT_ANALYSIS chạy song song, phần còn lại xếp hàng."""
    started = 0
    skipped = 0
    ncv_gd = st.session_state.get("ncv_giai_doan", PHASE_UI_LABELS["g2"])
    model_type_ncv = st.session_state.get("ncv_model_type", "MediaPipe Heavy")
    conf_ncv = st.session_state.get("ncv_confidence", 0.5)
    skip_step = st.session_state.get("ncv_skip_frames", 0)
    resize_width = st.session_state.get("ncv_resize_width", 720)
    for v in video_records:
        vp = v.get("video_path")
        if not vp:
            skipped += 1
            continue
        if video_dang_phan_tich(vp):
            skipped += 1
            continue
        if only_pending and not force_reanalyze and not video_can_khoi_dong_phan_tich(v, only_pending=True):
            skipped += 1
            continue
        if force_reanalyze:
            clear_analysis_progress(vp)
            done_key = f"_bg_done_{hashlib.md5(vp.encode()).hexdigest()}"
            st.session_state.pop(done_key, None)
        try:
            ensure_local_file(vp)
        except Exception:
            pass
        if not os.path.exists(vp) or os.path.getsize(vp) < 1024:
            skipped += 1
            continue
        bat_dau_phan_tich_background(
            video_path=vp,
            username=v.get("username"),
            full_name=v.get("full_name"),
            video_name=v.get("video_name"),
            exercise_name=v.get("exercise"),
            giai_doan=ncv_gd,
            model_type=model_type_ncv,
            confidence=conf_ncv,
            skip_step=skip_step,
            resize_width=resize_width,
            force_train_classifier=force_reanalyze,
        )
        started += 1
    return started, skipped


def _build_analysis_job_dependencies():
    """Build dependencies for background analysis jobs."""
    return SimpleNamespace(
        **{k: v for k, v in globals().items() if not k.startswith("__")}
    )

def bat_dau_phan_tich_background(
    video_path,
    username,
    full_name,
    video_name,
    exercise_name,
    giai_doan,
    model_type,
    confidence,
    temp_uploaded_path=None,
    skip_step=None,
    resize_width=None,
    force_train_classifier=False,
    force_restart=False,
):
    """Compatibility wrapper for background analysis jobs now owned by video.jobs."""
    return start_background_analysis(
        _build_analysis_job_dependencies(),
        video_path=video_path,
        username=username,
        full_name=full_name,
        video_name=video_name,
        exercise_name=exercise_name,
        giai_doan=giai_doan,
        model_type=model_type,
        confidence=confidence,
        temp_uploaded_path=temp_uploaded_path,
        skip_step=skip_step,
        resize_width=resize_width,
        force_train_classifier=force_train_classifier,
        force_restart=force_restart,
    )


def _xu_ly_ket_qua_khoi_dong_phan_tich(result):
    """Hiển thị toast/warning sau khi bấm nút phân tích."""
    if not isinstance(result, dict):
        result = {"started": bool(result), "reason": ""}
    if result.get("started"):
        st.toast("🚀 Đã khởi chạy phân tích mới — theo dõi tiến độ bên dưới!", icon="⚡")
        _lam_moi_giao_dien_sau_nut()
    elif result.get("reason") == "already_running":
        st.warning(
            "⏳ Video này **đang phân tích**. Chờ hoàn tất hoặc bấm "
            "**🧹 HỦY TẤT CẢ & LÀM MỚI** ở sidebar rồi thử lại."
        )
    elif result.get("reason") == "no_video":
        st.error("❌ Không khởi chạy được — thiếu đường dẫn video.")
    else:
        st.error("❌ Không khởi chạy được phân tích — kiểm tra log hoặc thử lại.")


def recalc_metrics(*args, **kwargs):
    """Compatibility wrapper for video.metrics.recalc_metrics."""
    from video.metrics import recalc_metrics as _impl
    return _impl(*args, **kwargs)

def gui_bao_cao_tong_hop_3_giai_doan():
    """Gửi báo cáo cho cả Bác sĩ & Bệnh nhân"""
    try:
        require_role(RESEARCHER_ROLE, ADMIN_ROLE, action="create_ai_report", target="doctor_evaluations")
    except PermissionError as exc:
        st.error(str(exc))
        return False
    v_meta = st.session_state.get('current_eval_video')
    if not v_meta:
        # Fallback cho trường hợp vừa mới phân tích video xong và chưa có current_eval_video trong session_state
        target_u = st.session_state.get('last_uploaded_patient_username', 'unknown')
        users_db = load_users()
        target_fn = users_db.get(target_u, {}).get('full_name', target_u)
        v_meta = {
            "username": target_u,
            "full_name": target_fn,
            "video_name": st.session_state.get('uploaded_file_name', 'N/A'),
            "exercise": st.session_state.get('exercise').get('ten', 'codman') if isinstance(st.session_state.get('exercise'), dict) else 'codman',
            "video_path": st.session_state.get('processed_video_path', '')
        }
    try:
        require_patient_scope(v_meta.get("username"), action="create_ai_report")
    except PermissionError as exc:
        st.error(str(exc))
        return False

    # Đồng bộ & chuẩn hóa exercise ngay lập tức cho các file DB khác
    correct_ex_name = dong_bo_va_chuan_hoa_exercise(
        username=v_meta['username'],
        video_name=v_meta.get('video_name'),
        video_path=v_meta.get('video_path'),
        original_exercise=v_meta.get('exercise')
    )
    v_meta['exercise'] = correct_ex_name
    if st.session_state.get('current_eval_video'):
        st.session_state.current_eval_video['exercise'] = correct_ex_name

    df = st.session_state.get('angle_df')
    # Nếu không có df trong session state, thử đọc từ file csv của video
    if df is None and v_meta.get('df_path') and os.path.exists(v_meta.get('df_path')):
        try:
            df = read_display_csv_fast(v_meta['df_path'])
        except:
            pass
    elif df is None and st.session_state.get('current_df_csv_path') and os.path.exists(st.session_state.get('current_df_csv_path')):
        try:
            df = read_display_csv_fast(st.session_state.get('current_df_csv_path'))
        except:
            pass

    if df is None or len(df) == 0:
        st.error("❌ Không thể nạp dữ liệu tọa độ chi tiết của video để phân tích.")
        return False

    is_gay_ex = any(kw in correct_ex_name.lower() for kw in ["gậy", "gay", "pulley", "stick"])
    if is_gay_ex:
        # Riêng bài tập gậy: chỉ có 1 giai đoạn tổng thể, không chia 3 giai đoạn
        ss_standard = 30 # Hoặc lấy từ v_meta.get('sai_so', 30)
        # Thử đọc sai_so từ video record
        video_list_temp = load_data(VIDEOS_FILE)
        for vx in video_list_temp:
            if vx.get('video_path') == v_meta.get('video_path') or (vx.get('video_name') == v_meta.get('video_name') and vx.get('username') == v_meta.get('username')):
                ss_standard = vx.get('sai_so', 30)
                break

        metrics_overall = recalc_metrics(df, ss_standard, correct_ex_name)
        acc_overall = metrics_overall['do_chinh_xac']
        clinical_res = "Đúng" if acc_overall >= 80 else ("Gần đúng" if acc_overall >= 50 else "Sai")

        evals = load_data(EVALUATIONS_FILE)
        evals.append({
            "patient_username": v_meta['username'],
            "doctor_username": "AI_Researcher",
            "video_name": v_meta.get('video_name', 'N/A'),
            "exercise": correct_ex_name,
            "ai_accuracy": round(float(acc_overall), 1),
            "ai_accuracy_g1": round(float(acc_overall), 1),
            "ai_accuracy_g2": round(float(acc_overall), 1),
            "ai_accuracy_g3": round(float(acc_overall), 1),
            "doctor_result": clinical_res,
            "errors": metrics_overall.get('warnings', []),
            "comments": (
                f"BÁO CÁO PHÂN TÍCH BÀI TẬP VỚI GẬY (TỔNG QUAN):\n"
                f"🏒 Độ chính xác: {acc_overall:.1f}% | Đúng: {metrics_overall['frame_dung']}/{metrics_overall['tong_frame_hop_le']} frames\n"
                f"🤖 AI đề xuất: " + ("Tập tốt, duy trì." if acc_overall >= 80 else ("Tập khá, lưu ý cùi chỏ." if acc_overall >= 50 else "Cần chuyên gia y tế hướng dẫn."))
            ),
            "plan": (
                f"Kế hoạch luyện tập đề xuất:\n"
                f"- Bài tập với gậy: Đạt {acc_overall:.1f}% - " + ("Đạt yêu cầu tự tập." if acc_overall >= 80 else "Cần rèn luyện thêm để giảm sai số.")
            ),
            "doctor_name": f"NCV: {st.session_state.user_info.get('full_name', 'Nghiên cứu viên')}",
            "time": get_vn_now().strftime("%H:%M - %d/%m/%Y"),
            "giai_doan": "Phân tích Tổng quan",
            "sai_so": ss_standard
        })
        save_data(EVALUATIONS_FILE, evals)

        # Cập nhật VIDEOS_FILE
        for v in video_list_temp:
            if v.get('video_path') == v_meta.get('video_path') or (v.get('video_name') == v_meta.get('video_name') and v.get('username') == v_meta.get('username')):
                v['accuracy'] = round(acc_overall, 1)
                v['status'] = "Đã phân tích"
                v['exercise'] = correct_ex_name

                # Để tương thích ngược với các giao diện mong chờ metrics_g1/g2/g3
                v['metrics'] = {
                    "do_chinh_xac": round(acc_overall, 1),
                    "ty_le_gan_dung": metrics_overall['ty_le_gan_dung'],
                    "frame_dung": metrics_overall['frame_dung'],
                    "frame_gan_dung": metrics_overall['frame_gan_dung'],
                    "tong_frame_hop_le": metrics_overall['tong_frame_hop_le'],
                    "tb_goc_vai": metrics_overall['tb_goc_vai'],
                    "tb_goc_khuyu": metrics_overall['tb_goc_khuyu'],
                    "warnings": metrics_overall.get('warnings', []),
                    "metrics_g1": metrics_overall,
                    "metrics_g2": metrics_overall,
                    "metrics_g3": metrics_overall
                }
        save_data(VIDEOS_FILE, video_list_temp)
        return True

    # Tính toán chỉ số cho cả 3 giai đoạn tương ứng với từng phân đoạn
    bounds = segment_frames(df)
    n0, n1, n2, n3 = bounds

    df_g1 = df.iloc[n0:n1]
    df_g2 = df.iloc[n1:n2]
    df_g3 = df.iloc[n2:n3]

    metrics_g1 = recalc_metrics(df_g1, PHASE_ERROR["g1"], correct_ex_name)
    metrics_g2 = recalc_metrics(df_g2, PHASE_ERROR["g2"], correct_ex_name)
    metrics_g3 = recalc_metrics(df_g3, PHASE_ERROR["g3"], correct_ex_name)

    acc_g1 = metrics_g1['do_chinh_xac']
    acc_g2 = metrics_g2['do_chinh_xac']
    acc_g3 = metrics_g3['do_chinh_xac']

    # Lấy G2 làm mốc chính để đánh giá lâm sàng chung
    clinical_res = "Đúng" if acc_g2 >= 85 else ("Gần đúng" if acc_g2 >= 60 else "Sai")

    # Lưu đánh giá vào doctor_evaluations.json
    evals = load_data(EVALUATIONS_FILE)

    # Tạo bản ghi đánh giá chi tiết
    evals.append({
        "patient_username": v_meta['username'],
        "doctor_username": "AI_Researcher",
        "video_name": v_meta.get('video_name', 'N/A'),
        "exercise": correct_ex_name,
        "ai_accuracy": round(float(acc_g2), 1),
        "ai_accuracy_g1": round(float(acc_g1), 1),
        "ai_accuracy_g2": round(float(acc_g2), 1),
        "ai_accuracy_g3": round(float(acc_g3), 1),
        "doctor_result": clinical_res,
        "errors": metrics_g2.get('warnings', []),
        "comments": (
            f"BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:\n"
            f"🌱 GĐ 1 (Khởi đầu - Sai số ±{PHASE_ERROR['g1']}°): {acc_g1:.1f}% | Đúng: {metrics_g1['frame_dung']}/{metrics_g1['tong_frame_hop_le']} frames\n"
            f"📈 GĐ 2 (Hồi phục - Sai số ±{PHASE_ERROR['g2']}°): {acc_g2:.1f}% | Đúng: {metrics_g2['frame_dung']}/{metrics_g2['tong_frame_hop_le']} frames\n"
            f"🎯 GĐ 3 (Chuẩn xác - Sai số ±{PHASE_ERROR['g3']}°): {acc_g3:.1f}% | Đúng: {metrics_g3['frame_dung']}/{metrics_g3['tong_frame_hop_le']} frames\n"
            f"🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn " +
            ("3" if acc_g3 >= 80 or acc_g2 >= 75 else ("2" if acc_g2 >= 50 else "1"))
        ),
        "plan": (
            f"Kế hoạch luyện tập đề xuất:\n"
            f"- GĐ1 (Sai số ±{PHASE_ERROR['g1']}°): Đạt {acc_g1:.1f}% - " + ("Đạt yêu cầu chuyển giai đoạn." if acc_g1 >= 75 else "Cần rèn luyện thêm.") + "\n"
            f"- GĐ2 (Sai số ±{PHASE_ERROR['g2']}°): Đạt {acc_g2:.1f}% - " + ("Đạt yêu cầu chuyển giai đoạn." if acc_g2 >= 70 else "Cần rèn luyện thêm.") + "\n"
            f"- GĐ3 (Sai số ±{PHASE_ERROR['g3']}°): Đạt {acc_g3:.1f}% - " + ("Ổn định khớp hoàn toàn." if acc_g3 >= 80 else "Khớp còn cứng hoặc lệch biên độ.")
        ),
        "doctor_name": f"NCV: {st.session_state.user_info.get('full_name', 'Nghiên cứu viên')}",
        "time": get_vn_now().strftime("%H:%M - %d/%m/%Y"),
        "giai_doan": "Phân tích 3 Giai đoạn",
        "sai_so": {
            "giai_doan_1": PHASE_ERROR["g1"],
            "giai_doan_2": PHASE_ERROR["g2"],
            "giai_doan_3": PHASE_ERROR["g3"],
        }
    })
    save_data(EVALUATIONS_FILE, evals)

    # Cập nhật thông tin trong danh sách video (VIDEOS_FILE) để Bác sĩ có thể thấy
    video_list = load_data(VIDEOS_FILE)
    for v in video_list:
        if v.get('video_path') == v_meta.get('video_path') or (v.get('video_name') == v_meta.get('video_name') and v.get('username') == v_meta.get('username')):
            # Tính trung bình có trọng số 3 giai đoạn (GĐ1:25%, GĐ2:40%, GĐ3:35%)
            _acc_tong_hop = round(acc_g1 * 0.25 + acc_g2 * 0.40 + acc_g3 * 0.35, 1)
            v['accuracy'] = _acc_tong_hop
            v['status'] = "Đã phân tích"
            v['exercise'] = correct_ex_name

            # Lưu stats và các giai đoạn vào metadata video
            v['metrics'] = {
                "do_chinh_xac": _acc_tong_hop,
                "ty_le_gan_dung": metrics_g2['ty_le_gan_dung'],
                "frame_dung": metrics_g2['frame_dung'],
                "frame_gan_dung": metrics_g2['frame_gan_dung'],
                "tong_frame_hop_le": metrics_g2['tong_frame_hop_le'],
                "tb_goc_vai": metrics_g2['tb_goc_vai'],
                "tb_goc_khuyu": metrics_g2['tb_goc_khuyu'],
                "warnings": metrics_g2.get('warnings', []),
                "metrics_g1": metrics_g1,
                "metrics_g2": metrics_g2,
                "metrics_g3": metrics_g3
            }
    save_data(VIDEOS_FILE, video_list)
    return True

def tinh_metrics_chi_tiet(*args, **kwargs):
    """Compatibility wrapper for video.metrics.tinh_metrics_chi_tiet."""
    from video.metrics import tinh_metrics_chi_tiet as _impl
    return _impl(*args, **kwargs)

# ============================================
# VẼ BIỂU ĐỒ SÁNG TẠO
# ============================================
def ve_bieu_do_goc_vai(df, bt, sai_so_override=None):
    """Vẽ biểu đồ góc vai với thiết kế đẹp mắt"""
    sai_so = sai_so_override if sai_so_override is not None else bt['chuan']['sai_so']

    # Lấy chuẩn trung bình để vẽ vùng nền
    c_vai = df['vai_chuan'].mean() if 'vai_chuan' in df.columns else 90

    fig = go.Figure()

    # Thêm vùng chuẩn (Dựa trên trung bình)
    fig.add_hrect(y0=c_vai-sai_so, y1=c_vai+sai_so,
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

    # Thêm đường chuẩn (Ưu tiên đường động từ YouTube)
    if 'vai_chuan' in df.columns:
        fig.add_trace(go.Scatter(
            y=df['vai_chuan'],
            mode='lines',
            line=dict(color='#00FF00', width=2, dash='dash'),
            name='Góc vai chuẩn (YouTube Động)'
        ))
    else:
        fig.add_hline(y=c_vai, line_dash='dash', line_color='#00FF00',
                     line_width=2, annotation_text=f"Chuẩn tĩnh: {c_vai}°",
                     annotation_position="top right")

    # Tô màu vùng ngoài chuẩn
    fig.add_hrect(y0=0, y1=c_vai-sai_so, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
    fig.add_hrect(y0=c_vai+sai_so, y1=180, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)

    is_light = st.session_state.theme == 'light'
    chart_text_color = '#333' if is_light else 'white'
    chart_grid_color = 'rgba(0,0,0,0.1)' if is_light else 'rgba(255,255,255,0.1)'
    chart_bg = 'rgba(255,255,255,1)' if is_light else 'rgba(26,26,46,0.9)'

    fig.update_layout(
        title=dict(
            text="<b>📈 BIỂU ĐỒ GÓC VAI THEO THỜI GIAN</b>",
            font=dict(size=20, color=chart_text_color, family='Arial Black'),
            x=0.5
        ),
        xaxis=dict(title=dict(text="<b>Số Frame</b>", font=dict(size=14, color=chart_text_color)), gridcolor=chart_grid_color),
        yaxis=dict(title=dict(text="<b>Góc (độ)</b>", font=dict(size=14, color=chart_text_color)), gridcolor=chart_grid_color,
                   range=[0, 180]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor=chart_bg,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)' if is_light else 'rgba(0,0,0,0.5)',
            bordercolor=chart_text_color,
            borderwidth=1,
            font=dict(color=chart_text_color, size=12)
        ),
        margin=dict(l=50, r=50, t=70, b=50)
    )

    return fig


def ve_bieu_do_goc_khuyu(df, bt, sai_so_override=None):
    """Vẽ biểu đồ góc khuỷu với thiết kế đẹp mắt"""
    sai_so = sai_so_override if sai_so_override is not None else bt['chuan']['sai_so']

    # Lấy chuẩn trung bình để vẽ vùng nền
    c_khuyu = df['khuyu_chuan'].mean() if 'khuyu_chuan' in df.columns else 170

    fig = go.Figure()

    # Thêm vùng chuẩn (Dựa trên trung bình)
    fig.add_hrect(y0=c_khuyu-sai_so, y1=c_khuyu+sai_so,
                  fillcolor="rgba(0, 255, 0, 0.15)", line_width=0,
                  annotation_text="Vùng chuẩn", annotation_position="top left")

    # Thêm đường góc bệnh nhân
    fig.add_trace(go.Scatter(
        y=df['goc_khuyu'],
        mode='lines+markers',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=4, color='#FF6B6B', symbol='circle'),
        name='Góc khuỷu bệnh nhân',
        hovertemplate='Frame: %{x}<br>Góc khuỷu: %{y:.1f}°<extra></extra>'
    ))

    # Thêm đường chuẩn (Ưu tiên đường động từ YouTube)
    if 'khuyu_chuan' in df.columns:
        fig.add_trace(go.Scatter(
            y=df['khuyu_chuan'],
            mode='lines',
            line=dict(color='#00FF00', width=2, dash='dash'),
            name='Góc khuỷu chuẩn (YouTube Động)'
        ))
    else:
        fig.add_hline(y=c_khuyu, line_dash='dash', line_color='#00FF00',
                     line_width=2, annotation_text=f"Chuẩn tĩnh: {c_khuyu}°",
                     annotation_position="top right")

    # Tô màu vùng ngoài chuẩn
    fig.add_hrect(y0=0, y1=c_khuyu-sai_so, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
    fig.add_hrect(y0=c_khuyu+sai_so, y1=180, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)

    is_light = st.session_state.theme == 'light'
    chart_text_color = '#333' if is_light else 'white'
    chart_grid_color = 'rgba(0,0,0,0.1)' if is_light else 'rgba(255,255,255,0.1)'
    chart_bg = 'rgba(255,255,255,1)' if is_light else 'rgba(26,26,46,0.9)'

    fig.update_layout(
        title=dict(
            text="<b>📈 BIỂU ĐỒ GÓC KHUỶU THEO THỜI GIAN</b>",
            font=dict(size=20, color=chart_text_color, family='Arial Black'),
            x=0.5
        ),
        xaxis=dict(title=dict(text="<b>Số Frame</b>", font=dict(size=14, color=chart_text_color)), gridcolor=chart_grid_color),
        yaxis=dict(title=dict(text="<b>Góc (độ)</b>", font=dict(size=14, color=chart_text_color)), gridcolor=chart_grid_color,
                   range=[0, 180]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor=chart_bg,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)' if is_light else 'rgba(0,0,0,0.5)',
            bordercolor=chart_text_color,
            borderwidth=1,
            font=dict(color=chart_text_color, size=12)
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

    is_light = st.session_state.theme == 'light'
    chart_text_color = '#333' if is_light else 'white'
    chart_grid_color = 'rgba(0,0,0,0.1)' if is_light else 'rgba(255,255,255,0.1)'
    chart_bg = 'rgba(255,255,255,1)' if is_light else 'rgba(26,26,46,0.9)'

    fig.update_layout(
        title=dict(
            text="<b>📊 PHÂN PHỐI GÓC KHỚP (HISTOGRAM)</b>",
            font=dict(size=20, color=chart_text_color, family='Arial Black'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor=chart_bg,
        showlegend=False,
        height=500,
        bargap=0.05
    )

    fig.update_xaxes(title=dict(text="<b>Góc (độ)</b>", font=dict(size=12, color=chart_text_color)),
                     gridcolor=chart_grid_color, row=1, col=1)
    fig.update_xaxes(title=dict(text="<b>Góc (độ)</b>", font=dict(size=12, color=chart_text_color)),
                     gridcolor=chart_grid_color, row=1, col=2)
    fig.update_yaxes(title=dict(text="<b>Tần suất</b>", font=dict(size=12, color=chart_text_color)),
                     gridcolor=chart_grid_color, row=1, col=1)

    return fig


def ve_bieu_do_tron_thong_ke(tk):
    """Vẽ biểu đồ tròn thống kê kết quả tập luyện (Pass/Nearly/Fail)"""
    labels = ['ĐÚNG (Pass)', 'GẦN ĐÚNG (Nearly)', 'SAI (Fail)']

    # Tính toán số lượng cho từng loại
    fail_count = tk['tong_frame_hop_le'] - tk['frame_dung'] - tk['frame_gan_dung']
    values = [tk['frame_dung'], tk['frame_gan_dung'], max(0, fail_count)]

    colors = ['#00FF00', '#FFA500', '#FF4444'] # Xanh, Cam, Đỏ

    is_light = st.session_state.theme == 'light'
    chart_text_color = '#333' if is_light else 'white'

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors, line=dict(color='#1a1a2e', width=2)),
        textinfo='percent+label',
        insidetextfont=dict(color='white' if not is_light else '#333'), # Ép màu chữ bên trong
        outsidetextfont=dict(color='white' if not is_light else '#333'), # Ép màu chữ bên ngoài
        hovertemplate="<b>%{label}</b><br>Số lượng: %{value} frames<br>Tỷ lệ: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title=dict(
            text="<b>📊 PHÂN BỔ KẾT QUẢ TẬP LUYỆN</b>",
            font=dict(size=18, color=chart_text_color),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=chart_text_color),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color=chart_text_color) # Ép màu chữ chú thích
        ),
        height=450,
        margin=dict(t=80, b=50, l=20, r=20)
    )
    return fig

def ve_bieu_do_boxplot_phan_loai_single(df, column='goc_vai', title="Góc Vai theo nhóm"):
    """Vẽ một biểu đồ Boxplot phân loại góc cho một khớp cụ thể"""
    plot_df = df.copy()
    def classify(row):
        if row['dung']: return 'ĐÚNG (Pass)'
        if row['gan_dung']: return 'GẦN ĐÚNG (Nearly)'
        return 'SAI (Fail)'

    plot_df['Phân loại'] = plot_df.apply(classify, axis=1)

    fig = go.Figure()
    colors = {'ĐÚNG (Pass)': '#00FF00', 'GẦN ĐÚNG (Nearly)': '#FFA500', 'SAI (Fail)': '#FF4444'}

    for label in ['ĐÚNG (Pass)', 'GẦN ĐÚNG (Nearly)', 'SAI (Fail)']:
        subset = plot_df[plot_df['Phân loại'] == label]
        if not subset.empty:
            fig.add_trace(go.Box(
                y=subset[column],
                name=label,
                marker_color=colors[label],
                boxmean='sd',
                showlegend=True
            ))

    is_light = st.session_state.theme == 'light'
    chart_text_color = '#333' if is_light else 'white'
    chart_grid_color = 'rgba(0,0,0,0.1)' if is_light else 'rgba(255,255,255,0.1)'

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color=chart_text_color),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=chart_text_color),
        height=450,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color=chart_text_color)
        ),
        margin=dict(t=60, b=150, l=20, r=20)
    )

    fig.update_yaxes(title_text="Góc (độ)", gridcolor=chart_grid_color)
    fig.update_xaxes(automargin=True, tickangle=0) # Automargin helps with spacing
    return fig

def ve_bieu_do_boxplot_phan_loai(df):
    """Giữ lại hàm cũ nhưng gọi hàm mới để tương thích ngược nếu cần, hoặc trả về list 2 fig"""
    fig_vai = ve_bieu_do_boxplot_phan_loai_single(df, 'goc_vai', "Góc Vai theo nhóm")
    fig_khuyu = ve_bieu_do_boxplot_phan_loai_single(df, 'goc_khuyu', "Góc Khuỷu theo nhóm")
    return fig_vai, fig_khuyu

def lay_nhan_dinh_lam_sang(goc_vai, goc_khuyu, bt, v_chuan=None, k_chuan=None, sai_so_override=None):
    """Cung cấp nhận định lâm sàng dựa trên lỗi phát hiện"""
    cv = v_chuan if v_chuan is not None else bt['chuan'].get('vai', 90)
    ck = k_chuan if k_chuan is not None else bt['chuan'].get('khuyu', 170)
    ss = sai_so_override if sai_so_override is not None else bt['chuan']['sai_so']

    nhan_dinh = []

    # Phân tích góc vai
    if goc_vai > cv + ss:
        nhan_dinh.append({
            "loai": "VAI - QUÁ BIÊN ĐỘ",
            "chi_so": f"{goc_vai:.1f}° > {cv+ss:.1f}°",
            "canh_bao": "Nguy cơ trật khớp vai hoặc tổn thương bao khớp phía trước.",
            "loi_khuyen": "Cần kiểm soát cơ delta tốt hơn, tránh vung tay quá đà."
        })
    elif goc_vai < cv - ss:
        nhan_dinh.append({
            "loai": "VAI - THIẾU BIÊN ĐỘ",
            "chi_so": f"{goc_vai:.1f}° < {cv-ss:.1f}°",
            "canh_bao": "Dấu hiệu của hội chứng đông cứng khớp vai hoặc đau do chạm (Impingement).",
            "loi_khuyen": "Thực hiện các bài tập kéo giãn nhẹ nhàng trước khi tập chính thức."
        })

    # Phân tích góc khuỷu
    if goc_khuyu > ck + ss:
        nhan_dinh.append({
            "loai": "KHUỶU - QUÁ DUỖI",
            "chi_so": f"{goc_khuyu:.1f}° > {ck+ss:.1f}°",
            "canh_bao": "Gây áp lực lên mỏm khuỷu và dây chằng bên trong.",
            "loi_khuyen": "Giữ khớp khuỷu hơi gập nhẹ (micro-bend) để bảo vệ khớp."
        })
    elif goc_khuyu < ck - ss:
        nhan_dinh.append({
            "loai": "KHUỶU - QUÁ GẬP",
            "chi_so": f"{goc_khuyu:.1f}° < {ck-ss:.1f}°",
            "canh_bao": "Căng cơ nhị đầu quá mức, nguy cơ viêm gân vùng khuỷu.",
            "loi_khuyen": "Thả lỏng cánh tay và tập trung vào cơ mục tiêu."
        })

    return nhan_dinh

def ve_bieu_do_radar(tk):
    """Vẽ biểu đồ Radar so sánh các chỉ số khoa học"""
    categories = [
        'Accuracy', 'F1-Score', 'MAE (Inverse)',
        'ICC', 'Precision', 'Recall'
    ]

    # Chuẩn hóa MAE: Càng thấp càng tốt, chúng ta đảo ngược để hiển thị trên Radar (max 10 độ)
    mae_score = max(0, 1 - (tk.get('mae_tong', 0) / 10))

    # Giá trị thực tế (0-1)
    values = [
        tk.get('do_chinh_xac', 0) / 100,
        tk.get('f1_score', 0),
        mae_score,
        tk.get('icc', 0),
        tk.get('precision', 0),
        tk.get('recall', 0)
    ]

    # Giá trị chuẩn (Research Target)
    targets = [0.90, 0.85, 0.5, 0.75, 0.85, 0.85] # MAE < 5 độ -> score > 0.5

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=targets,
        theta=categories,
        fill='toself',
        name='Mục tiêu (Target)',
        line_color='rgba(255, 215, 0, 0.5)',
        fillcolor='rgba(255, 215, 0, 0.1)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Thực tế (Actual)',
        line_color='#00CED1',
        fillcolor='rgba(0, 206, 209, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='white', size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='white', size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(26,26,46,0.9)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            font=dict(color='white')
        ),
        title=dict(
            text="<b>🔬 ĐÁNH GIÁ CHỈ SỐ KHOA HỌC (RADAR CHART)</b>",
            font=dict(size=18, color='white', family='Arial Black'),
            x=0.5,
            y=0.05
        ),
        margin=dict(l=80, r=80, t=100, b=80)
    )

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
        "chuan": {"sai_so": 30, "kieu": "dynamic"},
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
        - Khớp với biên độ vận động của Video YouTube chuẩn
        - Sai số tọa độ Euclidean thấp hơn ngưỡng cho phép
        - Bệnh nhân không có biểu hiện đau khi thực hiện (VAS < 3)
        - Thực hiện động tác mượt mà, đồng bộ với nhịp của video mẫu
        - Duy trì được nhịp thở đều đặn trong khi tập
        """
    },
    "gay": {
        "ten": "Bài tập với gậy (Pulley Exercise)",
        "icon": "🏒",
        "mo_ta": "Sử dụng gậy hoặc ròng rọc hỗ trợ nâng tay và xoay vai bị hạn chế vận động.",
        "chuan": {"sai_so": 30, "kieu": "dynamic"},
        "youtube": "https://www.youtube.com/watch?v=s2O8WHT5o2k",
        "thoi_gian": 45,
        "lan": 12,
        # THÊM DÒNG NÀY - BẢN NGẮN CHO HIỂN THỊ CHÍNH
        "huong_dan": "1. Cầm gậy bằng hai tay, tay lành cầm một đầu, tay bệnh cầm đầu kia\n2. Tay lành dùng lực đẩy gậy lên cao, kéo tay bệnh theo khớp với biên độ trong Video mẫu\n3. Giữ 5-10 giây ở tư thế cao nhất, hạ từ từ\n4. Thực hiện 10 lần mỗi động tác: nâng trước, xoay ngoài, xoay trong\n5. Thở ra khi nâng gậy lên, hít vào khi hạ xuống",
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
        - Nằm ngửa, tay bệnh gập khuỷu khớp với tư thế trong Video mẫu, cẳng tay hướng lên trần
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
        - Tuần 5-6: Thực hiện toàn bộ tầm vận động khớp với video chuẩn
        - Tuần 7-8: Thêm tạ nhẹ (0.5-1kg) nếu không đau
        """,
        "tieu_chi_danh_gia": """
        📊 **TIÊU CHÍ ĐÁNH GIÁ KẾT QUẢ:**
        - Khớp với biên độ nâng gậy và xoay vai của Video mẫu
        - Không có hiện tượng bù trừ (nghiêng người, nhún vai)
        - Bệnh nhân có thể tự thực hiện với mức độ trợ giúp tối thiểu
        - Cải thiện khả năng với tay lên cao (lấy đồ trên kệ, móc áo)
        - Đồng bộ thời gian thực với các mốc giây trong video hướng dẫn
        """
    },
    "khang_luc": {
        "ten": "Bài tập với dây kháng lực (Theraband Exercise)",
        "icon": "💪",
        "mo_ta": "Tăng cường sức mạnh cơ chóp xoay và cơ quanh khớp vai bằng dây thun kháng lực.",
        "chuan": {"sai_so": 30, "kieu": "dynamic"},
        "youtube": "https://www.youtube.com/watch?v=njDHDnZ6lis",
        "thoi_gian": 40,
        "lan": 15,
        # THÊM DÒNG NÀY - BẢN NGẮN CHO HIỂN THỊ CHÍNH
        "huong_dan": "1. Bắt đầu với dây kháng lực thấp nhất (màu vàng hoặc đỏ)\n2. Xoay vai ngoài: Nằm nghiêng, khuỷu gập 90°, xoay cẳng tay ra ngoài theo mẫu\n3. Xoay vai trong: Đứng hoặc nằm nghiêng, kéo dây vào trong áp sát bụng\n4. Dang vai: Đứng, dẫm dây dưới chân, dang tay sang ngang khớp với biên độ Video mẫu\n5. Gập vai: Đứng, nâng tay ra trước theo Video hướng dẫn\n6. Mỗi động tác 10-15 lần x 3 hiệp, nghỉ 30 giây giữa hiệp",
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
        - Tay bệnh: Gập khuỷu khớp với tư thế trong Video mẫu, đặt sát thân mình
        - Cố định dây: Buộc dây vào vật chắc chắn ngang thắt lưng
        - Động tác: Xoay cẳng tay ra ngoài, giữ khuỷu sát người
        - Giữ 2-3 giây ở cuối tầm, trở về chậm
        - Thực hiện 10-15 lần, 3 hiệp

        **3. ĐỘNG TÁC 2 - XOAY VAI TRONG:**
        - Tư thế: Đứng hoặc nằm nghiêng về phía tay bệnh
        - Tay bệnh: Gập khuỷu khớp với tư thế trong Video mẫu
        - Cố định dây: Buộc dây ở phía cùng bên
        - Động tác: Kéo dây vào trong, áp sát tay vào bụng
        - Thực hiện 10-15 lần, 3 hiệp

        **4. ĐỘNG TÁC 3 - DANG VAI (Abduction):**
        - Tư thế: Đứng, tay bệnh duỗi thẳng, dây dẫm dưới chân
        - Động tác: Dang tay sang ngang khớp với biên độ Video mẫu
        - Giữ 2 giây, hạ về chậm
        - Thực hiện 10-12 lần, 3 hiệp

        **5. ĐỘNG TÁC 4 - GẬP VAI (Flexion):**
        - Tư thế: Đứng, dây dẫm dưới chân
        - Động tác: Nâng tay ra trước theo biên độ Video chuẩn
        - Kiểm soát ở cả chiều lên và xuống
        - Thực hiện 10-12 lần, 3 hiệp

        **6. NGUYÊN TẮC TĂNG TIẾN:**
        - Tuần 1-2: 10 lần x 2 hiệp, nghỉ 30s, dây cấp độ 1
        - Tuần 3-4: 12 lần x 3 hiệp, tăng số lần trước
        - Tuần 5-6: 15 lần x 3 hiệp, tăng cấp độ dây nếu dễ dàng
        - Tuần 7-8: Thêm các bài tập phức hợp và tăng kháng lực

        **7. THEO DÕI CƯỜNG ĐỘ TẬP (RPE - Rating of Perceived Exertion):**
        - RPE 3-4: Hơi mệt, có thể trò chuyện trong khi tập
        - RPE 5-6: Mệt vừa, khó nói chuyện
        - RPE 7-8: Mệt nhiều - GIẢM CƯỜNG ĐỘ NGAY
        """,
        "tieu_chi_danh_gia": """
        📊 **TIÊU CHÍ ĐÁNH GIÁ KẾT QUẢ:**
        - Kháng lực phù hợp với biên độ vận động trong Video tham chiếu
        - Thực hiện đúng kỹ thuật, không bù trừ bằng cơ vai khác
        - Bệnh nhân có thể thực hiện 3 hiệp 15 lần với dây cấp độ phù hợp
        - Không đau trong và sau khi tập (VAS < 2)
        - Cải thiện sức mạnh (test cơ manual muscle testing tăng 1-2 cấp độ)
        - Tốc độ co duỗi khớp với nhịp đếm trong video chuẩn
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
# HÀM HIỂN THỊ TAB 2 - THIẾT KẾ LẠI
# ============================================
# ============================================
# HÀM HIỂN THỊ TAB 2 - THIẾT KẾ LẠI
# ============================================
def hien_thi_tab_phan_tich(key_suffix="", stats_ext=None, df_ext=None, exercise_ext=None):
    """Compatibility wrapper for analysis tab UI now owned by ui.analysis_tab."""
    render_analysis_tab_page(
        _build_ui_tab_dependencies(),
        key_suffix=key_suffix,
        stats_ext=stats_ext,
        df_ext=df_ext,
        exercise_ext=exercise_ext,
    )



# ==================== CÁC HÀM HỖ TRỢ VAI TRÒ MỚI ====================

def hien_thi_form_danh_gia_bac_si():
    """Compatibility wrapper for doctor evaluation form UI now owned by ui.doctor_forms."""
    render_doctor_evaluation_form_page(_build_ui_tab_dependencies())

def hien_thi_ket_qua_cho_benh_nhan(target_username=None):
    st.markdown("## 📊 KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP")

    dong_bo_hf_json_nhe_tab(["doctor_evaluations.json"])

    evals = _evals_dedup_cached(_mtimes_video_eval()[1])
    user_role = st.session_state.user_info.get('role')

    if target_username:
        if not current_actor_can_access_patient(target_username):
            st.error("Bạn không có quyền xem kết quả của bệnh nhân ngoài phạm vi phụ trách.")
            return
        my_evals = [e for e in evals if e.get('patient_username') == target_username]
        username = target_username
    else:
        if user_role == "Bệnh nhân":
            username = st.session_state.user_info['username']
            my_evals = [e for e in evals if e.get('patient_username') == username]
        elif user_role == "Quản trị viên":
            my_evals = evals
            username = None
        else:
            my_evals = scope_records_for_current_actor(evals)
            username = None

    # 1. CHỈ HIỂN THỊ VIDEO GỐC BN ĐÃ CÓ NHẬN XÉT BÁC SĨ/KTV (8 video nghiên cứu)
    all_vids = load_danh_sach_video_nghien_cuu()
    if user_role == "Bệnh nhân":
        p_username = username if username else st.session_state.user_info['username']
        my_history_vids = [v for v in all_vids if v.get("username") == p_username]
    else:
        my_history_vids = all_vids

    my_history_vids = sorted(
        my_history_vids,
        key=lambda x: _parse_vn_datetime(
            _lay_thoi_gian_phan_tich_on_dinh(
                x,
                _lay_eval_moi_nhat_theo_bai_tap(my_evals, x.get("username"), x.get("exercise"), doctor_username="AI_Researcher"),
            )
        ) or datetime.min,
        reverse=True,
    )

    # 2. XÁC ĐỊNH TRẠNG THÁI "CHỜ KẾT QUẢ" (FRESH SESSION)
    is_fresh_session = st.session_state.get('fresh_session', False)

    # AUTO-LOAD: nếu video vừa nộp đã có kết quả
    if is_fresh_session and st.session_state.get('active_video_name'):
        if my_history_vids and my_history_vids[0].get('video_name') == st.session_state.get('active_video_name'):
            st.session_state.fresh_session = False
            is_fresh_session = False
            st.session_state.active_video_name = None

    selected_v = None

    if not my_history_vids and not my_evals:
        st.info("🕒 Kết quả đánh giá chuyên môn đang được xử lý. Vui lòng quay lại sau khi Bác sĩ hoặc Nhóm Nghiên cứu hoàn tất đánh giá.")
        return

    if user_role == "Nghiên cứu viên":
        cur = st.session_state.get("current_eval_video")
        if cur and my_history_vids:
            cur_key = _slot_nghien_cuu_key(cur.get("username"), cur.get("exercise"))
            matched = next(
                (
                    v for v in my_history_vids
                    if _slot_nghien_cuu_key(v.get("username"), v.get("exercise")) == cur_key
                ),
                None,
            )
            if matched:
                st.session_state.current_eval_video = matched

    hien_thi_tab_ket_qua_da_chon(my_history_vids, my_evals, user_role, is_fresh_session)


def hien_thi_tab_ket_qua_da_chon(my_history_vids, my_evals, user_role, is_fresh_session=False):
    """Compatibility wrapper for selected-result tab UI now owned by ui.doctor_forms."""
    render_selected_results_tab_page(
        _build_ui_tab_dependencies(),
        my_history_vids,
        my_evals,
        user_role,
        is_fresh_session=is_fresh_session,
    )


def _hien_thi_khoi_nhan_xet_danh_gia(eval_data, accent_color, accent_bg, accent_border, default_source):
    """Hiển thị một khối nhận xét đánh giá (chỉ văn bản, không biểu đồ)."""
    if not eval_data:
        return
    is_light = st.session_state.theme == "light"
    card_bg = "rgba(255,255,255,1)" if is_light else "rgba(0,0,0,0.2)"
    text_muted = "#666" if is_light else "#aaa"
    text_main = "#222" if is_light else "#eee"
    source_name = safe_html(eval_data.get("doctor_name") or eval_data.get("doctor_username") or default_source, max_length=120)
    eval_time = safe_html(_format_vn_time(eval_data.get("time"), default="N/A"), max_length=60)
    exercise = safe_html(eval_data.get("exercise", "N/A"), max_length=120)
    result = safe_html(eval_data.get("doctor_result", "N/A"), max_length=80)
    comments = safe_html((eval_data.get("comments") or "").strip() or "Không có nhận xét.", max_length=2000)
    plan = (eval_data.get("plan") or "").strip()
    errors = [str(err) for err in eval_data.get("errors", []) if "WARNING" not in str(err).upper()]

    st.markdown(f"""
    <div style="background:{card_bg}; border:1px solid {accent_border}; border-left:5px solid {accent_color};
                border-radius:14px; padding:18px 20px; margin-bottom:12px;">
        <p style="margin:0 0 8px 0; color:{text_muted}; font-size:0.82rem;">
            🕒 {eval_time} · Bài tập: <b style="color:{text_main};">{exercise}</b>
        </p>
        <p style="margin:0 0 12px 0; color:{text_muted}; font-size:0.82rem;">
            Nguồn: <span style="color:{accent_color}; font-weight:700;">{source_name}</span>
        </p>
        <p style="margin:0 0 10px 0; font-size:0.78rem; color:{text_muted}; text-transform:uppercase; letter-spacing:0.5px;">
            Kết quả đánh giá
        </p>
        <p style="margin:0 0 14px 0; font-size:1.15rem; color:{accent_color}; font-weight:800;">
            {result}
        </p>
        <p style="margin:0 0 6px 0; font-size:0.78rem; color:{text_muted}; text-transform:uppercase; letter-spacing:0.5px;">
            Nhận xét
        </p>
        <p style="margin:0; font-size:0.95rem; color:{text_main}; white-space:pre-line; line-height:1.6;">
            {comments}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if plan:
        st.markdown(f"**Kế hoạch / Chỉ định:**\n\n{plan}")
    if errors:
        st.markdown(f"**Lỗi kỹ thuật ghi nhận:** {', '.join(errors)}")


def hien_thi_noi_dung_ket_qua(selected_v, my_evals):
    """Tab Kết quả đánh giá: chỉ nhận xét NCV/AI và Bác sĩ, tách riêng."""
    if not selected_v:
        st.info("👆 Hãy chọn một phiên tập từ danh sách bên trên để xem nhận xét chi tiết.")
        return

    ai_eval, doc_eval = _lay_danh_gia_cho_video(selected_v, my_evals)
    exercise = selected_v.get("exercise", "N/A")
    st.markdown(f"#### 📋 Phiên tập: **{exercise}**")
    st.markdown("---")

    st.markdown("### 🤖 Nhận xét NCV / Phân tích AI")
    if ai_eval:
        _hien_thi_khoi_nhan_xet_danh_gia(
            ai_eval,
            accent_color="#00CED1",
            accent_bg="rgba(0,206,209,0.08)",
            accent_border="rgba(0,206,209,0.35)",
            default_source="Nghiên cứu viên / Hệ thống AI",
        )
    else:
        st.info("Chưa có nhận xét phân tích AI cho bài tập này.")

    st.markdown("---")
    st.markdown("### 👨‍⚕️ Nhận xét Bác sĩ / KTV PHCN")
    if doc_eval:
        if doc_eval.get("comments_ncv"):
            st.markdown(
                f"**💬 Ghi chú nội bộ cho NCV:** "
                f"<span style='color:#ffd700;'>{safe_html(doc_eval.get('comments_ncv'), max_length=1000)}</span>",
                unsafe_allow_html=True,
            )
        _hien_thi_khoi_nhan_xet_danh_gia(
            doc_eval,
            accent_color="#ffd700",
            accent_bg="rgba(255,215,0,0.08)",
            accent_border="rgba(255,215,0,0.35)",
            default_source="Bác sĩ / KTV PHCN",
        )
    else:
        st.info("Chưa có đánh giá lâm sàng từ Bác sĩ / KTV cho bài tập này.")

    if not ai_eval and not doc_eval:
        st.warning(
            "Chưa tìm thấy nhận xét đánh giá cho phiên tập này. "
            "Vui lòng quay lại sau khi NCV hoặc Bác sĩ hoàn tất đánh giá."
        )


def hien_thi_tab_khai_bao_trieu_chung():
    st.markdown("## 🩺 KHAI BÁO TRIỆU CHỨNG & CẢM NHẬN")
    st.info("💡 Thông tin này sẽ được gửi trực tiếp cho Bác sĩ/KTV để hỗ trợ quá trình đánh giá và điều trị.")

    with st.form("patient_symptoms_form"):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Họ và tên", value=st.session_state.user_info.get('full_name', ''))
            age = st.number_input("Tuổi", 0, 120, 22)
        with col2:
            gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
            date = st.date_input("Ngày khai báo", get_vn_now())

        symptoms = st.text_area("Mô tả cảm giác đau hoặc khó khăn khi vận động:",
                              placeholder="VD: Đau nhói ở khớp vai khi giơ tay quá đầu, cứng khớp vào buổi sáng...",
                              height=150)

        muc_do_dau = st.select_slider("Mức độ đau hiện tại (VAS):",
                                     options=list(range(11)),
                                     value=3)

        submitted = st.form_submit_button("📤 GỬI THÔNG TIN CHO BÁC SĨ", width="stretch", type="primary")

        if submitted:
            if symptoms:
                try:
                    actor = require_role(PATIENT_ROLE, action="create_symptom", target=st.session_state.user_info['username'])
                    require_patient_scope(actor["username"], action="create_symptom")
                    data = load_data(SYMPTOMS_FILE)
                    data.append({
                        "username": st.session_state.user_info['username'],
                        "full_name": full_name,
                        "age": age,
                        "gender": gender,
                        "symptoms": symptoms,
                        "vas": muc_do_dau,
                        "time": get_vn_now().strftime("%H:%M - %d/%m/%Y")
                    })
                    save_data(SYMPTOMS_FILE, data)
                    st.success("✅ Đã gửi thông tin cho Bác sĩ thành công!")
                    st.balloons()
                except PermissionError as exc:
                    st.error(str(exc))
            else:
                st.warning("⚠️ Vui lòng nhập mô tả triệu chứng.")


# ============================================
# HÀM HIỂN THỊ PHIẾU ĐÁNH GIÁ NCKH (MỚI)
# ============================================


def segment_frames(*args, **kwargs):
    """Compatibility wrapper for video.metrics.segment_frames."""
    from video.metrics import segment_frames as _impl
    return _impl(*args, **kwargs)

def cut_video_segments(input_path, n1, n2, total_frames, fps_export=15):
    """
    Cắt video gốc thành 3 video tương ứng với 3 giai đoạn tập luyện
    bằng ffmpeg và transcode sang H.264 có faststart để trình duyệt load tức thì.
    """
    import subprocess
    import os

    input_path = get_local_frame_path(input_path)
    input_path = resolve_playback_video_path(input_path) or input_path

    # Sử dụng hậu tố _gX_f.mp4 để tránh bị cache tệp phân đoạn cũ bị lỗi codec/lag
    g1_path = input_path.replace('.mp4', '_g1_f.mp4')
    g2_path = input_path.replace('.mp4', '_g2_f.mp4')
    g3_path = input_path.replace('.mp4', '_g3_f.mp4')

    if (os.path.exists(g1_path) and os.path.getsize(g1_path) > 0 and
        os.path.exists(g2_path) and os.path.getsize(g2_path) > 0 and
        os.path.exists(g3_path) and os.path.getsize(g3_path) > 0):
        return g1_path, g2_path, g3_path

    t0 = 0.0
    t1 = n1 / fps_export
    t2 = n2 / fps_export
    t3 = total_frames / fps_export

    def _run_cut(start, end, out_p):
        dur = end - start
        try:
            # Transcode H.264 với preset ultrafast và +faststart để stream mượt mà trên browser
            cmd = build_cut_segment_command(input_path, out_p, start=start, duration=dur)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40)
        except Exception as e:
            print(f"Lỗi khi cắt phân đoạn {start} -> {end}: {e}")

    _run_cut(t0, t1, g1_path)
    _run_cut(t1, t2, g2_path)
    _run_cut(t2, t3, g3_path)

    return g1_path, g2_path, g3_path


@st.cache_data(show_spinner=False)
def load_all_frames_data_cached(path):
    if not path or not os.path.exists(path):
        return []
    import json
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

# ============================================
# HÀM HIỂN THỊ LỊCH FRAMES ĐẦY ĐỦ
# ============================================
def hien_thi_frames_day_du(key_suffix=""):
    """Compatibility wrapper for frame viewer UI now owned by ui.frames_viewer."""
    render_frames_full_page(_build_ui_tab_dependencies(), key_suffix=key_suffix)

# Callback xử lý đổi theme nhanh (Để ngoài hàm main để tránh lỗi WebSocket Cache)

def update_theme_callback():
    val = st.session_state.get('theme_toggle_top')
    if val is not None:
        st.session_state.theme = 'dark' if val else 'light'


# ============================================
# GIAO DIỆN ĐĂNG NHẬP / ĐĂNG KÝ
# ============================================
def hien_thi_dang_nhap_dang_ky():
    with st.sidebar:
        st.markdown("### 🛠️ CẤU HÌNH GIAO DIỆN")
        current_theme = st.session_state.get('theme', 'dark')
        t_label = "🌙 Chế độ Tối" if current_theme == 'dark' else "☀️ Chế độ Sáng"
        st.toggle(t_label, value=(current_theme == 'dark'),
                  key="theme_toggle_login",
                  on_change=lambda: st.session_state.update({"theme": "dark" if st.session_state.get("theme_toggle_login", True) else "light"}))
        st.markdown("---")

    # Định nghĩa màu sắc tiêu đề theo theme để tránh lỗi nền trắng chữ trắng
    is_light = st.session_state.get('theme') == 'light'
    header_color = "#ffffff" if not is_light else "#1a1a2e"
    sub_color = "#ffffff" if not is_light else "#333333"
    _hien_thi_header_chinh(header_color, sub_color, is_light=is_light, extra_style="margin-bottom: 1.5rem;")

    # Sử dụng cột để tạo khung hình vuông ở giữa màn hình
    _, col_mid, _ = st.columns([1, 1.8, 1])

    with col_mid:
        # Dùng container với border=True để tạo ô vuông bao quanh chuẩn web
        with st.container(border=True):
            # CHẾ ĐỘ QUÊN MẬT KHẨU
            if st.session_state.get('forgot_password_mode', False):
                st.markdown("### 🔄 KHÔI PHỤC MẬT KHẨU")
                st.warning("Tính năng tự đặt lại mật khẩu đang tạm tắt. Vui lòng liên hệ Quản trị viên để cấp mật khẩu tạm thời.")
                if st.button("Quay lại đăng nhập", width="stretch"):
                    st.session_state.forgot_password_mode = False
                    st.rerun()
                return

            # GIAO DIỆN CHÍNH (TABS) - role được lấy từ hồ sơ người dùng sau khi xác thực.

            tab_list = ["🔐 ĐĂNG NHẬP", "📋 ĐĂNG KÝ", "🚀 GOOGLE ID"]
            all_login_tabs = st.tabs(tab_list)
            t_map = {name: all_login_tabs[i] for i, name in enumerate(tab_list)}

            if "🔐 ĐĂNG NHẬP" in t_map:
                with t_map["🔐 ĐĂNG NHẬP"]:
                    # CHẾ ĐỘ ĐỔI MẬT KHẨU TRONG LOGIN
                    if st.session_state.get('change_password_mode', False):
                        st.markdown("### 🔑 THAY ĐỔI MẬT KHẨU")
                        st.info("💡 Điền thông tin bên dưới để cập nhật mật khẩu mới.")
                        with st.form("login_change_password_form_v2"):
                            cp_u = st.text_input("👤 Tên đăng nhập", key="cp_u_v2")
                            cp_old = st.text_input("🔒 Mật khẩu hiện tại", type="password", key="cp_old_v2")
                            cp_new = st.text_input("🆕 Mật khẩu mới", type="password", key="cp_new_v2")
                            cp_conf = st.text_input("✅ Xác nhận mật khẩu mới", type="password", key="cp_conf_v2")

                            c1, c2 = st.columns(2)
                            with c1:
                                if st.form_submit_button("💾 CẬP NHẬT", width="stretch"):
                                    users = load_users()
                                    cp_key = _auth_lookup_key(users, cp_u)
                                    if cp_key and _verify_auth_password(cp_key, cp_old, users[cp_key]):
                                        if cp_new == cp_conf and len(cp_new) >= 6:
                                            _set_user_password(users[cp_key], cp_new, must_change_password=False)
                                            save_users(users)
                                            st.success("✅ Thành công! Hãy đăng nhập lại.")
                                            st.session_state.change_password_mode = False
                                            st.rerun()
                                        else: st.error("❌ Mật khẩu không khớp hoặc quá ngắn.")
                                    else: st.error("❌ Thông tin không chính xác.")
                            with c2:
                                if st.form_submit_button("Hủy bỏ", width="stretch"):
                                    st.session_state.change_password_mode = False
                                    st.rerun()
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)
                        u = st.text_input("👤 Tên đăng nhập", placeholder="Nhập tên tài khoản", key="login_u")
                        p = st.text_input("🔑 Mật khẩu", type="password", placeholder="Nhập mật khẩu", key="login_p")

                        if st.button("🚀 ĐĂNG NHẬP NGAY", width="stretch", type="primary"):
                            if _frontend_api_enabled():
                                try:
                                    login_response = FrontendApiClient(FRONTEND_API_CONFIG).login(u, p)
                                    if (login_response.get("user") or {}).get("must_change_password"):
                                        st.warning("⚠️ Tài khoản này cần đổi mật khẩu trước khi tiếp tục.")
                                        st.session_state.change_password_mode = True
                                        st.rerun()
                                    _hoan_tat_dang_nhap_api(login_response)
                                    _rerun_toan_bo_app()
                                except FrontendApiError as exc:
                                    _frontend_api_set_error(exc)
                                    st.error("❌ Tài khoản hoặc mật khẩu không đúng")
                            else:
                                users = load_users()
                                u_key = _auth_lookup_key(users, u)
                                if u_key and _verify_auth_password(u_key, p, users[u_key]):
                                    if users[u_key].get("must_change_password"):
                                        st.warning("⚠️ Tài khoản này cần đổi mật khẩu trước khi tiếp tục.")
                                        st.session_state.change_password_mode = True
                                        st.rerun()
                                    else:
                                        _rehash_password_if_needed(users, u_key, p)
                                        _hoan_tat_dang_nhap(u_key, users[u_key])
                                        _rerun_toan_bo_app()
                                else:
                                    st.error("❌ Tài khoản hoặc mật khẩu không đúng")

                        if st.button("🔑 ĐỔI MẬT KHẨU", width="stretch", type="secondary"):
                            st.session_state.change_password_mode = True
                            st.rerun()

                        if st.button("❓ Bạn quên mật khẩu?", width="stretch", type="secondary"):
                            st.session_state.forgot_password_mode = True
                            st.rerun()

            if "📋 ĐĂNG KÝ" in t_map:
                with t_map["📋 ĐĂNG KÝ"]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    reg_name = st.text_input("📛 Họ và tên", placeholder="VD: Nguyễn Văn A", key="reg_n")
                    reg_u = st.text_input("👤 Tên đăng nhập *", placeholder="Chọn tên tài khoản", key="reg_u")
                    reg_e = st.text_input("📧 Email liên hệ *", placeholder="example@gmail.com", key="reg_e")
                    reg_p = st.text_input("🔑 Mật khẩu *", type="password", placeholder="Tối thiểu 6 ký tự", key="reg_p")
                    reg_cp = st.text_input("✅ Xác nhận mật khẩu *", type="password", placeholder="Nhập lại mật khẩu", key="reg_cp")
                    st.info("💡 Các tài khoản Bác sĩ và Nghiên cứu viên đã được khởi tạo theo danh sách. Để cấp thêm tài khoản mới, vui lòng liên hệ Quản trị viên.")
                    reg_role = "Bệnh nhân" # Gán mặc định không cần hiển thị

                    if st.button("🚀 ĐĂNG KÝ TRUY CẬP", width="stretch", type="primary"):
                        reg_u_clean = _normalize_auth_text(reg_u)
                        reg_e_clean = _normalize_auth_text(reg_e)
                        if not reg_u or not reg_e or len(reg_p) < 6:
                            st.warning("⚠️ Vui lòng điền đầy đủ các thông tin bắt buộc (*)")
                        elif reg_p != reg_cp:
                            st.error("❌ Mật khẩu xác nhận không khớp")
                        else:
                            users = load_users()
                            if _auth_lookup_key(users, reg_u_clean): st.error("❌ Tên đăng nhập này đã tồn tại")
                            else:
                                now = get_vn_now().isoformat()
                                users[reg_u_clean] = {
                                    **password_record_update(reg_p, updated_at=now, must_change_password=False),
                                    "email": reg_e_clean,
                                    "full_name": _normalize_auth_text(reg_name) or reg_u_clean,
                                    "role": reg_role,
                                    "created_at": now,
                                }
                                save_users(users)
                                st.success("🎉 Đăng ký thành công! Bạn có thể đăng nhập ngay.")

            if "🚀 GOOGLE ID" in t_map:
                with t_map["🚀 GOOGLE ID"]:
                    st.markdown("""
                    <div style="text-align: center; padding: 10px;">
                        <img src="https://www.gstatic.com/images/branding/product/1x/googleg_48dp.png" width="40" style="margin-bottom: 5px;">
                        <h5 style="color: white;">Đăng nhập nhanh</h5>
                        <p style="color: #888; font-size: 0.85rem;">Truy cập an toàn qua Google ID</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("🌐 TIẾP TỤC ĐĂNG NHẬP VỚI GOOGLE", width="stretch", type="primary"):
                        try:
                            st.session_state.auth_initiated = True
                            st.login("google")
                        except Exception as e:
                            st.error(f"⚠️ Lỗi Google: {e}")

# ============================================
# HÀM HIỂN TRỊ TAB QUẢN TRỊ VIÊN
# ============================================


# ============================================
# HÀM HIỂN THỊ TAB ĐỔI MẬT KHẨU
# ============================================
def hien_thi_tab_doi_mat_khau():
    st.markdown("## 🔑 THAY ĐỔI MẬT KHẨU")
    st.info("💡 Bạn nên đặt mật khẩu mạnh (bao gồm chữ, số và ký tự đặc biệt) để bảo vệ tài khoản.")

    with st.form("change_password_form"):
        old_p = st.text_input("🔒 Mật khẩu hiện tại", type="password")
        new_p = st.text_input("🆕 Mật khẩu mới", type="password")
        conf_p = st.text_input("✅ Xác nhận mật khẩu mới", type="password")

        if st.form_submit_button("💾 CẬP NHẬT MẬT KHẨU"):
            users = load_users()
            u = st.session_state.user_info['username']

            if _verify_auth_password(u, old_p, users.get(u)):
                if new_p == conf_p:
                    if len(new_p) >= 6:
                        _set_user_password(users[u], new_p, must_change_password=False)
                        save_users(users)
                        st.success("✅ Đã thay đổi mật khẩu thành công! Hãy ghi nhớ mật khẩu mới của bạn.")
                    else:
                        st.error("❌ Mật khẩu mới phải có ít nhất 6 ký tự.")
                else:
                    st.error("❌ Mật khẩu mới và mật khẩu xác nhận không khớp.")
            else:
                st.error("❌ Mật khẩu hiện tại không chính xác. Vui lòng thử lại.")


def delete_video_callback(video_name, username):
    try:
        actor = require_role(
            "Quản trị viên",
            "Bác sĩ / KTV PHCN",
            "Nghiên cứu viên",
            action="delete_video",
            target=f"{username}/{video_name}",
        )
    except PermissionError as exc:
        st.session_state.delete_success = str(exc)
        return

    if not current_actor_can_access_patient(username):
        write_audit_log(actor["username"], actor["role"], "delete_video", f"{username}/{video_name}", "denied_out_of_scope")
        st.session_state.delete_success = "Bạn không có quyền xóa video của bệnh nhân ngoài phạm vi phụ trách."
        return

    # Load video list
    video_list = load_data(VIDEOS_FILE)

    # Tìm video cần xóa
    target_idx = None
    for idx, v in enumerate(video_list):
        if v.get('video_name') == video_name and v.get('username') == username:
            target_idx = idx
            break

    if target_idx is not None:
        v = video_list[target_idx]
        files_to_backup = [
            VIDEOS_FILE,
            EVALUATIONS_FILE,
            v.get('video_path'),
            v.get('processed_path'),
            v.get('df_path'),
            v.get('all_frames_data_path'),
        ]
        try:
            create_backup_before_destructive("delete_video", files_to_backup)
        except Exception as exc:
            st.session_state.delete_success = f"Không thể backup trước khi xóa: {exc}"
            return

        # Xóa file thực tế
        for f_path in [v.get('video_path'), v.get('processed_path'), v.get('df_path'), v.get('all_frames_data_path')]:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except:
                    pass

        # CASCADE DELETE
        evals_all = load_data(EVALUATIONS_FILE)
        evals_filtered = [
            ev for ev in evals_all
            if not (
                ev.get('patient_username') == v.get('username')
                and ev.get('video_name') == v.get('video_name')
                and (not v.get('exercise') or ev.get('exercise') == v.get('exercise'))
            )
        ]
        save_data(EVALUATIONS_FILE, evals_filtered)

        # Xóa khỏi danh sách
        video_list.pop(target_idx)
        save_data(VIDEOS_FILE, video_list)
        write_audit_log(actor["username"], actor["role"], "delete_video", f"{username}/{video_name}", "success")
        st.session_state.delete_success = f"Đã xóa video: {v.get('video_name', 'Không rõ tên')}"


def _dong_bo_video_list_nen(force=False):
    """Đồng bộ video_list.json nền — không chặn render trang."""
    import threading

    def _video_list_background_sync_job():
        try:
            _dong_bo_video_list_day_du_tu_hf(force=force)
            _video_nghien_cuu_cached.clear()
            _load_video_list_core.clear()
        except Exception:
            pass

    threading.Thread(target=_video_list_background_sync_job, daemon=True).start()


def hien_thi_danh_sach_video_fragment(user_role):
    """Compatibility wrapper for video-list UI now owned by ui.video_list."""
    render_video_list_fragment_page(_build_ui_tab_dependencies(), user_role)


# ============================================
# MAIN - GIỮ NGUYÊN CẤU TRÚC TAB
# ============================================
def _lay_meta_tab_bac_si(selected_video):
    """Tra cứu nhanh AI eval + output video — tránh quét toàn bộ evals/frames mỗi lần load."""
    if not selected_video:
        return False, False
    evals = _evals_dedup_cached(_mtimes_video_eval()[1])
    pu, ex = selected_video.get("username"), selected_video.get("exercise")
    has_ai = _lay_eval_moi_nhat_theo_bai_tap(evals, pu, ex, doctor_username="AI_Researcher") is not None
    has_output = bool(
        selected_video.get("processed_path")
        or selected_video.get("all_frames_data_path")
        or selected_video.get("metrics")
    )
    if not has_output:
        vname = (selected_video.get("video_name") or "").rsplit(".", 1)[0]
        if vname:
            frames_path = os.path.join(EXTRACTED_FRAMES_DIR, vname)
            try:
                has_output = os.path.isdir(frames_path) and next(os.scandir(frames_path), None) is not None
            except OSError:
                has_output = False
    return has_ai, has_output


def _build_ui_tab_dependencies():
    deps = SimpleNamespace(
        st=st,
        pd=pd,
        BAI_TAP=BAI_TAP,
        DATA_DIR=DATA_DIR,
        DB_DIR=DB_DIR,
        PATIENT_ROLE=PATIENT_ROLE,
        DOCTOR_ROLE=DOCTOR_ROLE,
        RESEARCHER_ROLE=RESEARCHER_ROLE,
        ADMIN_ROLE=ADMIN_ROLE,
        EVALUATIONS_FILE=EVALUATIONS_FILE,
        HISTORY_FILE=HISTORY_FILE,
        PROCESSED_DIR=PROCESSED_DIR,
        REMINDERS_FILE=REMINDERS_FILE,
        RESEARCH_DATA_FILE=RESEARCH_DATA_FILE,
        SESSION_STATE_FILE=SESSION_STATE_FILE,
        USER_DATA_FILE=USER_DATA_FILE,
        SYMPTOMS_FILE=SYMPTOMS_FILE,
        VIDEOS_FILE=VIDEOS_FILE,
        UPLOAD_DIR=UPLOAD_DIR,
        HF_TOKEN=HF_TOKEN,
        HF_DATASET_ID=HF_DATASET_ID,
        PHASE_ERROR=PHASE_ERROR,
        PHASE_UI_LABELS=PHASE_UI_LABELS,
        POSE_CLASSIFIER_IMPORT_ERROR=POSE_CLASSIFIER_IMPORT_ERROR,
        cv2=cv2,
        MAX_FILE_SIZE_MB=MAX_FILE_SIZE_MB,
        MAX_FFMPEG_THREADS=MAX_FFMPEG_THREADS,
        MAX_CONCURRENT_ANALYSIS=MAX_CONCURRENT_ANALYSIS,
        safe_html=safe_html,
        render_video=render_video,
        require_role=require_role,
        require_patient_scope=require_patient_scope,
        load_data=load_data,
        load_users=load_users,
        save_users=save_users,
        password_record_update=password_record_update,
        write_audit_log=write_audit_log,
        create_backup_before_destructive=create_backup_before_destructive,
        get_global_session_version=get_global_session_version,
        revoke_all_sessions=revoke_all_sessions,
        parse_upload_time_from_filename=_parse_upload_time_from_filename,
        remove_files_in_dir=_remove_files_in_dir,
        save_data=save_data,
        get_vn_now=get_vn_now,
        current_actor_can_access_patient=current_actor_can_access_patient,
        sanitize_filename=sanitize_filename,
        validate_uploaded_video_file=validate_uploaded_video_file,
        validate_video_file_for_processing=validate_video_file_for_processing,
        get_video_codec=get_video_codec,
        build_upload_h264_command=build_upload_h264_command,
        push_file_to_hf_async=push_file_to_hf_async,
        scope_records_for_current_actor=scope_records_for_current_actor,
        scope_patient_usernames_for_current_actor=scope_patient_usernames_for_current_actor,
        researcher_view_records=researcher_view_records,
        format_vn_time=_format_vn_time,
        render_patient_results=hien_thi_ket_qua_cho_benh_nhan,
        render_symptoms_tab=hien_thi_tab_khai_bao_trieu_chung,
        render_patient_info=hien_thi_tab_thong_tin_tong_hop_benh_nhan,
        render_general_info=hien_thi_tab_thong_tin_tong_hop,
        render_contact=hien_thi_tab_lien_he,
        render_feedback=hien_thi_tab_phan_hoi,
        load_research_videos=load_danh_sach_video_nghien_cuu,
        mtimes_video_eval=_mtimes_video_eval,
        evals_dedup_cached=_evals_dedup_cached,
        dedup_evaluations=_dedup_evaluations,
        read_display_csv_fast=read_display_csv_fast,
        ensure_local_file=ensure_local_file,
        get_local_frame_path=get_local_frame_path,
        frames_zip_from_processed_path=_frames_zip_from_processed_path,
        sync_transcode_to_h264=sync_transcode_to_h264,
        build_frame_extract_command=build_frame_extract_command,
        format_ml_display=format_ml_display,
        frames_zip_path_from_video=_frames_zip_path_from_video,
        find_latest_analyzed_video=_tim_video_phan_tich_moi_nhat,
        sync_frame_metadata_to_session=_dong_bo_metadata_frames_vao_session,
        analysis_session_matches_video=_session_phan_tich_khop_video,
        auto_load_latest_analysis_result=tu_dong_nap_ket_qua_phan_tich_gan_nhat,
        is_local_file_ready=is_local_file_ready,
        check_and_extract_frames_zip=check_and_extract_frames_zip,
        load_all_frames_data_cached=load_all_frames_data_cached,
        recalc_metrics=recalc_metrics,
        standard_ai_accuracy=lay_do_chinh_xac_ai_chuan,
        resolve_playback_video_path=resolve_playback_video_path,
        ensure_playable_video=ensure_playable_video,
        prefetch_video_quiet=_prefetch_video_quiet,
        video_has_audio_track=video_has_audio_track,
        segment_frames=segment_frames,
        get_video_fps_cached=get_video_fps_cached,
        cut_video_segments=cut_video_segments,
        create_zip_of_frames=create_zip_of_frames,
        ensure_analysis_video_downloaded=dam_bao_tai_video_phan_tich,
        render_reload_and_reanalyze_button=hien_thi_nut_tai_lai_va_phan_tich_moi,
        send_three_stage_report=gui_bao_cao_tong_hop_3_giai_doan,
        start_parallel_full_download=_bat_dau_tai_day_du_song_song,
        render_media_download_progress_fragment=_fragment_tien_do_tai_media,
        mark_analysis_session_key=_gan_khoa_session_phan_tich,
        apply_video_result_to_session=_gan_session_ket_qua_tu_video,
        get_hf_last_download_error=lambda: _hf_last_download_error,
        render_video_progress_row=_hien_thi_hang_video_va_tien_do,
        render_new_analysis_mode_notice=_hien_thi_thong_bao_che_do_phan_tich_moi,
        refresh_ui_after_button=_lam_moi_giao_dien_sau_nut,
        get_video_evaluations=_lay_danh_gia_cho_video,
        return_to_saved_results=_quay_lai_ket_qua_cu_da_luu,
        thread_is_really_running=_thread_dang_chay_thuc_su,
        finalize_background_analysis_if_ready=finalize_background_analysis_if_ready,
        get_pose_classifier_status=get_pose_classifier_status,
        check_hf_dataset_access=kiem_tra_quyen_hf_dataset,
        clinical_insights=lay_nhan_dinh_lam_sang,
        reprocess_videos_with_classifier=reprocess_videos_with_classifier,
        train_pose_classifier=train_pose_classifier,
        render_classification_boxplot=ve_bieu_do_boxplot_phan_loai,
        render_elbow_angle_chart=ve_bieu_do_goc_khuyu,
        render_shoulder_angle_chart=ve_bieu_do_goc_vai,
        render_histogram_chart=ve_bieu_do_histogram,
        render_radar_chart=ve_bieu_do_radar,
        render_pie_stats_chart=ve_bieu_do_tron_thong_ke,
        enable_hf_rescue_mode=_bat_che_do_cuu_ho_hf,
        cancel_flags=_cancel_flags,
        render_two_pass_progress=_hien_thi_progress_hai_pass,
        is_hf_runtime=_is_hf_runtime,
        running_threads=_running_threads,
        handle_analysis_start_result=_xu_ly_ket_qua_khoi_dong_phan_tich,
        clear_analysis_progress=clear_analysis_progress,
        restart_video_analysis=khoi_dong_phan_tich_lai_video,
        read_progress=read_progress,
        write_progress=write_progress,
        show_hf_download_error=thong_bao_loi_tai_hf,
        render_researcher_analysis_video=hien_thi_tab_phan_tich_va_video_ncv,
        render_research_profile_team=hien_thi_tab_nckh_va_thanh_vien_ncv,
        render_research_topic=hien_thi_tab_nckh,
        render_research_info=hien_thi_tab_thong_tin_nghien_cuu,
        render_team=hien_thi_tab_thanh_vien,
        render_change_password=hien_thi_tab_doi_mat_khau,
        render_technology=hien_thi_tab_cong_nghe,
        sync_video_list_background=_dong_bo_video_list_nen,
        reload_video_list_from_cloud=tai_lai_video_list_tu_cloud,
        normalize_video_key=_normalize_video_key,
        parse_vn_datetime=_parse_vn_datetime,
        patient_summary_from_video=_tom_tat_benh_nhan_tu_video,
        latest_eval_by_exercise=_lay_eval_moi_nhat_theo_bai_tap,
        is_placeholder_video_record=_la_ban_ghi_video_mo_co,
        stable_analysis_time=_lay_thoi_gian_phan_tich_on_dinh,
        can_start_analysis=video_can_khoi_dong_phan_tich,
        list_running_jobs=liet_ke_jobs_dang_chay,
        start_batch_analysis=bat_dau_phan_tich_hang_loat,
        raw_video_path=_lay_duong_dan_video_tho,
        get_final_h264_path=get_final_h264_path,
        find_ready_local_video=find_ready_local_video,
        ensure_video_ready_to_play=_dam_bao_video_san_sang_play,
        is_scratch_video_path=_is_scratch_video_path,
        video_list_status=_lay_trang_thai_video_danh_sach,
        upload_time_for_video=_lay_thoi_gian_upload_video,
        patient_display_label=patient_display_label,
        display_accuracy=_lay_do_chinh_xac_hien_thi,
        ffprobe_video_duration_text=ffprobe_video_duration_text,
        get_clean_rel_path=get_clean_rel_path,
        refresh_video_record_from_db=_lam_moi_ban_ghi_video_tu_db,
        video_is_analyzing=video_dang_phan_tich,
        load_chart_fast_from_cloud=_nap_bieu_do_nhanh_tu_cloud,
        analysis_slot_key=_slot_video_phan_tich,
        clear_analysis_session=_xoa_session_phan_tich,
        delete_video_callback=delete_video_callback,
        get_patient_ai_evaluations=lay_danh_gia_ai_benh_nhan,
        load_ai_result_into_session=nap_ket_qua_ai_vao_session,
        render_selected_result_content=hien_thi_noi_dung_ket_qua,
    )
    deps.render_video_list_fragment = (
        lambda user_role, video_list_preloaded=None: render_video_list_fragment_page(
            deps,
            user_role,
            video_list_preloaded=video_list_preloaded,
        )
    )
    deps.render_frames_full = (
        lambda key_suffix="": render_frames_full_page(deps, key_suffix=key_suffix)
    )
    deps.render_analysis_tab = (
        lambda key_suffix="", stats_ext=None, df_ext=None, exercise_ext=None: render_analysis_tab_page(
            deps,
            key_suffix=key_suffix,
            stats_ext=stats_ext,
            df_ext=df_ext,
            exercise_ext=exercise_ext,
        )
    )
    deps.render_doctor_eval_form = lambda: render_doctor_evaluation_form_page(deps)
    deps.render_latest_results_and_history = (
        lambda username, video_name=None, exercise=None, selected_v=None, key_suffix="", chi_nhan_xet=False:
            render_latest_results_and_history_page(
                deps,
                username,
                video_name=video_name,
                exercise=exercise,
                selected_v=selected_v,
                key_suffix=key_suffix,
                chi_nhan_xet=chi_nhan_xet,
            )
    )
    deps.render_selected_results_tab = (
        lambda my_history_vids, my_evals, user_role, is_fresh_session=False:
            render_selected_results_tab_page(
                deps,
                my_history_vids,
                my_evals,
                user_role,
                is_fresh_session=is_fresh_session,
            )
    )
    deps.render_reminders = lambda: render_reminders_page(deps)
    deps.render_research_form = lambda: render_research_form_page(deps)
    deps.render_admin_home = lambda: render_admin_home_page(deps)
    deps.render_admin_management = lambda: render_admin_management_page(deps)
    return deps


def _render_common_tab(selected_tab, user_role, deps):
    common_routes = {
        "📖 HƯỚNG DẪN": lambda: hien_thi_tab_huong_dan(role=user_role),
        "🏥 KIẾN THỨC PHCN": hien_thi_tab_kien_thuc_phcn,
        "🔑 ĐỔI MẬT KHẨU": hien_thi_tab_doi_mat_khau,
        "🌐 CÔNG NGHỆ": hien_thi_tab_cong_nghe,
        "📚 ĐỀ TÀI NCKH": hien_thi_tab_nckh,
        "📄 THÔNG TIN NGHIÊN CỨU": hien_thi_tab_thong_tin_nghien_cuu,
        "👥 THÀNH VIÊN": hien_thi_tab_thanh_vien,
        "📄 PHIẾU NCKH": deps.render_research_form,
    }
    handler = common_routes.get(selected_tab)
    if handler:
        handler()
        return True
    return False


def _render_main_tab_content(tab_titles, user_role):
    selected_tab = render_tab_selector(st, tab_titles)
    deps = _build_ui_tab_dependencies()

    if _render_common_tab(selected_tab, user_role, deps):
        return
    if user_role == "Quản trị viên":
        render_admin_tab(selected_tab, deps)
    elif user_role == "Bác sĩ / KTV PHCN":
        render_doctor_tab(selected_tab, deps)
    elif user_role == "Bệnh nhân":
        render_patient_tab(selected_tab, deps)
    elif user_role == "Nghiên cứu viên":
        render_researcher_tab(selected_tab, deps)

def main():
    # Do not force browser reloads after F5 on HF Spaces.
    # Streamlit owns reconnect; manual location.reload() can loop into a blank app shell.
    thuc_hien_khoi_tao_he_thong_mot_lan()
    initialize_session_runtime()

    # Kiểm tra trạng thái đăng nhập ngay đầu hàm main
    if not st.session_state.get("logged_in") or not st.session_state.get("user_info"):
        if st.session_state.get("logged_in") and not st.session_state.get("user_info"):
            st.session_state.logged_in = False
        hien_thi_dang_nhap_dang_ky()
        return

    if not _current_session_is_valid():
        st.warning("Phiên đăng nhập đã hết hiệu lực. Vui lòng đăng nhập lại.")
        _clear_authenticated_session()
        hien_thi_dang_nhap_dang_ky()
        return

    # Nạp nhẹ kết quả phân tích nền đã hoàn tất (không rerun) -> hiện ngay khi tải trang
    poll_background_analysis_complete()

    # Đồng bộ nền ngay sau đăng nhập — hiện UI ngay, không chờ Cloud
    if st.session_state.pop("_need_home_sync", False):
        _dong_bo_video_list_nen(force=True)

    # Callback xử lý đổi theme nhanh
    def update_theme_callback():
        st.session_state.theme = 'dark' if st.session_state.get('theme_toggle_top', True) else 'light'

    # Chuyển các điều khiển hệ thống vào Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ HỆ THỐNG")

        # 1. Chế độ Sáng/Tối
        current_theme = st.session_state.get('theme', 'dark')
        label = "🌙 Chế độ Tối" if current_theme == 'dark' else "☀️ Chế độ Sáng"
        st.toggle(label, value=(current_theme == 'dark'),
                  key="theme_toggle_top",
                  on_change=update_theme_callback)

        # 2. Thông tin người dùng & Đăng xuất
        sidebar_username_html = safe_html(st.session_state.user_info.get('username', 'user'), max_length=80)
        st.markdown(f"""
        <div style="background: rgba(255, 215, 0, 0.1); padding: 15px; border-radius: 12px; border: 1px solid rgba(255, 215, 0, 0.3); margin-top: 10px; margin-bottom: 10px;">
            <div style="font-size: 0.8rem; color: #888;">Đang đăng nhập:</div>
            <div style="color: #ffd700; font-weight: bold; font-size: 1.1rem; margin-bottom: 10px;">👤 {sidebar_username_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Trạng thái đồng bộ Hugging Face Dataset (Đặc biệt quan trọng trên Space)
        if HF_SPACE_ID or os.path.exists("/data"):
            hf_ok, hf_msg = kiem_tra_quyen_hf_dataset()
            if hf_ok:
                sub = safe_html(hf_msg, max_length=180) if hf_msg else f"Dataset: <b>{safe_html(HF_DATASET_ID, max_length=120)}</b>"
                st.markdown(f"""
                <div style="background: rgba(46, 204, 113, 0.15); padding: 10px; border-radius: 8px; border: 1px solid rgba(46, 204, 113, 0.4); text-align: center; margin-top: 5px; margin-bottom: 15px;">
                    <span style="color: #2ecc71; font-weight: bold; font-size: 0.85rem;">💚 Cloud Sync: ĐÃ KÍCH HOẠT</span>
                    <p style="color: #aaa; font-size: 0.75rem; margin: 5px 0 0 0;">{sub}</p>
                </div>
                """, unsafe_allow_html=True)
            elif HF_TOKEN:
                lib_err = _hf_la_loi_thu_vien(hf_msg or "")
                sync_label = "THƯ VIỆN LỖI" if lib_err else "TOKEN LỖI"
                hf_msg_html = safe_html(hf_msg or 'Token không đọc được Dataset.', max_length=220)
                st.markdown(f"""
                <div style="background: rgba(241, 196, 15, 0.15); padding: 12px; border-radius: 8px; border: 1px solid rgba(241, 196, 15, 0.4); text-align: center; margin-top: 5px; margin-bottom: 15px;">
                    <span style="color: #f1c40f; font-weight: bold; font-size: 0.85rem;">⚠️ Cloud Sync: {sync_label}</span>
                    <p style="color: #ddd; font-size: 0.75rem; margin: 5px 0 0 0;">{hf_msg_html}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(231, 76, 60, 0.15); padding: 12px; border-radius: 8px; border: 1px solid rgba(231, 76, 60, 0.4); text-align: center; margin-top: 5px; margin-bottom: 15px;">
                    <span style="color: #e74c3c; font-weight: bold; font-size: 0.85rem;">⚠️ Cloud Sync: TẮT (NGUY HIỂM)</span>
                    <p style="color: #ddd; font-size: 0.75rem; margin: 5px 0 0 0;">Dữ liệu sẽ bị xóa sạch khi Space restart! Hãy cấu hình <b>HF_TOKEN</b> (loại Write) trong Space Secrets.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(52, 152, 219, 0.15); padding: 10px; border-radius: 8px; border: 1px solid rgba(52, 152, 219, 0.4); text-align: center; margin-top: 5px; margin-bottom: 15px;">
                <span style="color: #3498db; font-weight: bold; font-size: 0.85rem;">💾 Bộ nhớ: Local Storage</span>
                <p style="color: #aaa; font-size: 0.75rem; margin: 5px 0 0 0;">Đang chạy offline trên máy tính cá nhân.</p>
            </div>
            """, unsafe_allow_html=True)

        if FRONTEND_API_CONFIG.base_url:
            api_status = "ĐÃ BẬT" if _frontend_api_enabled() else "CHƯA BẬT"
            api_color = "#2ecc71" if _frontend_api_enabled() else "#3498db"
            api_error = st.session_state.get("_frontend_api_last_error")
            api_sub = (
                safe_html(api_error, max_length=180)
                if api_error
                else safe_html(FRONTEND_API_CONFIG.base_url, max_length=180)
            )
            st.markdown(
                f"""
                <div style="background: rgba(52, 152, 219, 0.10); padding: 10px; border-radius: 8px; border: 1px solid rgba(52, 152, 219, 0.35); text-align: center; margin-top: 5px; margin-bottom: 15px;">
                    <span style="color: {api_color}; font-weight: bold; font-size: 0.85rem;">🔌 Backend API: {api_status}</span>
                    <p style="color: #aaa; font-size: 0.75rem; margin: 5px 0 0 0;">{api_sub}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🚪 Đăng xuất hệ thống", width="stretch", key="logout_sidebar", type="secondary"):
            if st.session_state.get("backend_access_token") and _frontend_api_enabled():
                try:
                    _frontend_api_client().logout()
                except FrontendApiError as exc:
                    _frontend_api_set_error(exc)
            if st.session_state.user_info and st.session_state.user_info.get("auth_type") == "google":
                st.logout()
            st.query_params.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown("---")

    # TOP HEADER - Đã được tối ưu cho cả Light và Dark mode
    is_light = st.session_state.get('theme') == 'light'
    header_h1_color = "#ffffff" if not is_light else "#1a1a2e"
    header_p_color = "#ffffff" if not is_light else "#333333"
    _hien_thi_header_chinh(header_h1_color, header_p_color, show_badge=True, is_light=is_light)

    user_role = st.session_state.user_info.get('role', 'Bệnh nhân')

    # --- KHỞI TẠO MẶC ĐỊNH ĐỂ TRÁNH LỖI UNBOUNDLOCALERROR ---
    ma_bai_tap = list(BAI_TAP.keys())[0]
    bai_tap = BAI_TAP[ma_bai_tap]

    with st.sidebar:
        st.markdown(f"### 🎭 VAI TRÒ: {user_role.upper()}")

        if user_role == "Nghiên cứu viên":
            ncv_name_html = safe_html(st.session_state.user_info.get('full_name', 'Chuyên gia AI'), max_length=120)
            render_researcher_sidebar(
                st,
                researcher_name_html=ncv_name_html,
                phase_labels=PHASE_UI_LABELS,
                normalize_phase_selection=normalize_phase_selection,
                stats=_thong_ke_video_nghien_cuu(),
                clear_progress=clear_all_progress_files,
            )

        elif user_role == "Quản trị viên":
            admin_name_html = safe_html(st.session_state.user_info.get('full_name', 'Administrator'), max_length=120)
            render_admin_sidebar(
                st,
                admin_name_html=admin_name_html,
                lookup_user=lambda username: load_users().get(username),
            )

        else:
            if user_role == "Bác sĩ / KTV PHCN":
                doctor_name_html = safe_html(st.session_state.user_info.get('full_name', 'Bác sĩ / KTV'), max_length=120)
                # THỐNG KÊ NHANH CHO BÁC SĨ — dùng cache để tránh load 341KB mỗi rerun
                v_list = load_danh_sach_video_nghien_cuu()
                v_mtime, e_mtime = _mtimes_video_eval()
                evals_db_cached = _evals_dedup_cached(e_mtime)

                # Đếm O(n): build set các (username, video_name, exercise) đã được bác sĩ đánh giá
                evaluated_keys = {
                    (e.get('patient_username'), e.get('video_name'), e.get('exercise'))
                    for e in evals_db_cached
                    if e.get('doctor_username') and e.get('doctor_username') != "AI_Researcher"
                }
                pending_eval = sum(
                    1 for v in v_list
                    if (v.get('username'), v.get('video_name'), v.get('exercise')) not in evaluated_keys
                )
                total_patients = len(set(v.get('username') for v in v_list if v.get('username')))

                render_doctor_sidebar(
                    st,
                    doctor_name_html=doctor_name_html,
                    pending_eval=pending_eval,
                    total_patients=total_patients,
                )

            else: # Vai trò Bệnh nhân
                full_name = safe_html(st.session_state.user_info.get('full_name', 'Bệnh nhân'), max_length=120)
                render_patient_sidebar(st, full_name_html=full_name)


        st.markdown("---")
        st.markdown("**👨‍🏫 Giảng viên hướng dẫn 1 (Khoa học dữ liệu):** TS. Trần Hồng Việt 🎓")
        st.markdown("**👩‍🏫 Giảng viên hướng dẫn 2 (Lâm sàng):** Nguyễn Thị Thùy Chi 🎓")
        st.markdown("**🏥 Trường Đại học Y tế Công cộng**")
        st.markdown("**👩‍⚕️ Chủ nhiệm đề tài:** Đinh Lê Quỳnh Phương")

    # Tự động chọn video đầu tiên một lần — tránh load lại danh sách mỗi lần chuyển tab
    if user_role in ["Bác sĩ / KTV PHCN", "Nghiên cứu viên"]:
        if not st.session_state.get('current_eval_video') and not st.session_state.get('_default_video_picked'):
            all_vids = load_danh_sach_video_nghien_cuu()
            if all_vids:
                st.session_state.current_eval_video = all_vids[0]
            st.session_state._default_video_picked = True

    has_video_output = False
    if user_role == "Bác sĩ / KTV PHCN":
        selected_video_main = st.session_state.get('current_eval_video')
        # Cache kết quả _lay_meta_tab_bac_si theo video key — tránh scan disk mỗi rerun
        _vid_key_meta = (
            (selected_video_main or {}).get("username", ""),
            (selected_video_main or {}).get("video_name", ""),
        )
        if st.session_state.get("_meta_tab_vid_key") != _vid_key_meta:
            _, _has_out = _lay_meta_tab_bac_si(selected_video_main)
            st.session_state["_meta_tab_vid_key"] = _vid_key_meta
            st.session_state["_meta_tab_has_output"] = _has_out
        has_video_output = st.session_state.get("_meta_tab_has_output", False)
    tab_titles = tab_titles_for_role(user_role, has_video_output=bool(has_video_output))

    # Khởi tạo hoặc khôi phục active_tab (đồng bộ widget sau reload — tránh trang trống)
    sync_active_tab_state(st.session_state, tab_titles)

    try:
        _render_main_tab_content(tab_titles, user_role)
    except Exception as tab_err:
        st.error(f"💥 Lỗi hiển thị nội dung tab: {tab_err}")
        import traceback
        st.code(traceback.format_exc())

    # ==================== FOOTER CHUNG (LUÔN HIỆN Ở DƯỚI CÙNG) ====================
    st.markdown('<div id="rehab-footer-anchor"></div>', unsafe_allow_html=True)
    hien_thi_footer_chung()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"💥 Lỗi khởi động ứng dụng: {e}")
        import traceback
        st.code(traceback.format_exc())
        try:
            st.markdown('<div id="rehab-footer-anchor"></div>', unsafe_allow_html=True)
            hien_thi_footer_chung()
        except Exception:
            pass
