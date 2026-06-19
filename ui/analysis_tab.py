"""Analysis tab Streamlit UI extracted from app.py."""

from __future__ import annotations

import os
import time


class _DynamicValue:
    def __init__(self, getter):
        self._getter = getter

    def _value(self):
        return self._getter()

    def __bool__(self):
        return bool(self._value())

    def __str__(self):
        return str(self._value() or "")

    def __format__(self, spec):
        return format(str(self), spec)


def _bind_deps(deps):
    """Bind app services expected by the moved legacy analysis UI body."""
    global st, pd
    global BAI_TAP, DATA_DIR, DB_DIR, EVALUATIONS_FILE, HF_TOKEN, PHASE_ERROR, PHASE_UI_LABELS
    global POSE_CLASSIFIER_IMPORT_ERROR, PROCESSED_DIR, RESEARCHER_ROLE, ADMIN_ROLE, VIDEOS_FILE
    global _bat_dau_tai_day_du_song_song, _fragment_tien_do_tai_media, _gan_khoa_session_phan_tich
    global _gan_session_ket_qua_tu_video, _hf_last_download_error, _hien_thi_hang_video_va_tien_do
    global _hien_thi_thong_bao_che_do_phan_tich_moi, _lam_moi_ban_ghi_video_tu_db, _lam_moi_giao_dien_sau_nut
    global _lay_danh_gia_cho_video, _nap_bieu_do_nhanh_tu_cloud, _quay_lai_ket_qua_cu_da_luu
    global _session_phan_tich_khop_video, _slot_video_phan_tich, _thread_dang_chay_thuc_su
    global _tim_video_phan_tich_moi_nhat, _xoa_session_phan_tich, create_zip_of_frames
    global finalize_background_analysis_if_ready, get_local_frame_path, get_pose_classifier_status
    global gui_bao_cao_tong_hop_3_giai_doan, hien_thi_nut_tai_lai_va_phan_tich_moi
    global kiem_tra_quyen_hf_dataset, lay_do_chinh_xac_ai_chuan, lay_nhan_dinh_lam_sang
    global load_data, patient_display_label, push_file_to_hf_async, read_display_csv_fast, read_progress
    global recalc_metrics, reprocess_videos_with_classifier, require_role, safe_html, segment_frames
    global train_pose_classifier, ve_bieu_do_boxplot_phan_loai, ve_bieu_do_goc_khuyu, ve_bieu_do_goc_vai
    global ve_bieu_do_histogram, ve_bieu_do_radar, ve_bieu_do_tron_thong_ke
    global _bat_che_do_cuu_ho_hf, _cancel_flags, _hien_thi_progress_hai_pass, _is_hf_runtime
    global _running_threads, _xu_ly_ket_qua_khoi_dong_phan_tich, clear_analysis_progress
    global khoi_dong_phan_tich_lai_video, write_progress, thong_bao_loi_tai_hf

    st = deps.st
    pd = deps.pd
    BAI_TAP = deps.BAI_TAP
    DATA_DIR = deps.DATA_DIR
    DB_DIR = deps.DB_DIR
    EVALUATIONS_FILE = deps.EVALUATIONS_FILE
    HF_TOKEN = deps.HF_TOKEN
    PHASE_ERROR = deps.PHASE_ERROR
    PHASE_UI_LABELS = deps.PHASE_UI_LABELS
    POSE_CLASSIFIER_IMPORT_ERROR = deps.POSE_CLASSIFIER_IMPORT_ERROR
    PROCESSED_DIR = deps.PROCESSED_DIR
    RESEARCHER_ROLE = deps.RESEARCHER_ROLE
    ADMIN_ROLE = deps.ADMIN_ROLE
    VIDEOS_FILE = deps.VIDEOS_FILE
    _bat_dau_tai_day_du_song_song = deps.start_parallel_full_download
    _fragment_tien_do_tai_media = deps.render_media_download_progress_fragment
    _gan_khoa_session_phan_tich = deps.mark_analysis_session_key
    _gan_session_ket_qua_tu_video = deps.apply_video_result_to_session
    _hf_last_download_error = _DynamicValue(deps.get_hf_last_download_error)
    _hien_thi_hang_video_va_tien_do = deps.render_video_progress_row
    _hien_thi_thong_bao_che_do_phan_tich_moi = deps.render_new_analysis_mode_notice
    _lam_moi_ban_ghi_video_tu_db = deps.refresh_video_record_from_db
    _lam_moi_giao_dien_sau_nut = deps.refresh_ui_after_button
    _lay_danh_gia_cho_video = deps.get_video_evaluations
    _nap_bieu_do_nhanh_tu_cloud = deps.load_chart_fast_from_cloud
    _quay_lai_ket_qua_cu_da_luu = deps.return_to_saved_results
    _session_phan_tich_khop_video = deps.analysis_session_matches_video
    _slot_video_phan_tich = deps.analysis_slot_key
    _thread_dang_chay_thuc_su = deps.thread_is_really_running
    _tim_video_phan_tich_moi_nhat = deps.find_latest_analyzed_video
    _xoa_session_phan_tich = deps.clear_analysis_session
    create_zip_of_frames = deps.create_zip_of_frames
    finalize_background_analysis_if_ready = deps.finalize_background_analysis_if_ready
    get_local_frame_path = deps.get_local_frame_path
    get_pose_classifier_status = deps.get_pose_classifier_status
    gui_bao_cao_tong_hop_3_giai_doan = deps.send_three_stage_report
    hien_thi_nut_tai_lai_va_phan_tich_moi = deps.render_reload_and_reanalyze_button
    kiem_tra_quyen_hf_dataset = deps.check_hf_dataset_access
    lay_do_chinh_xac_ai_chuan = deps.standard_ai_accuracy
    lay_nhan_dinh_lam_sang = deps.clinical_insights
    load_data = deps.load_data
    patient_display_label = deps.patient_display_label
    push_file_to_hf_async = deps.push_file_to_hf_async
    read_display_csv_fast = deps.read_display_csv_fast
    read_progress = deps.read_progress
    recalc_metrics = deps.recalc_metrics
    reprocess_videos_with_classifier = deps.reprocess_videos_with_classifier
    require_role = deps.require_role
    safe_html = deps.safe_html
    segment_frames = deps.segment_frames
    train_pose_classifier = deps.train_pose_classifier
    ve_bieu_do_boxplot_phan_loai = deps.render_classification_boxplot
    ve_bieu_do_goc_khuyu = deps.render_elbow_angle_chart
    ve_bieu_do_goc_vai = deps.render_shoulder_angle_chart
    ve_bieu_do_histogram = deps.render_histogram_chart
    ve_bieu_do_radar = deps.render_radar_chart
    ve_bieu_do_tron_thong_ke = deps.render_pie_stats_chart
    _bat_che_do_cuu_ho_hf = deps.enable_hf_rescue_mode
    _cancel_flags = deps.cancel_flags
    _hien_thi_progress_hai_pass = deps.render_two_pass_progress
    _is_hf_runtime = deps.is_hf_runtime
    _running_threads = deps.running_threads
    _xu_ly_ket_qua_khoi_dong_phan_tich = deps.handle_analysis_start_result
    clear_analysis_progress = deps.clear_analysis_progress
    khoi_dong_phan_tich_lai_video = deps.restart_video_analysis
    write_progress = deps.write_progress
    thong_bao_loi_tai_hf = deps.show_hf_download_error


def render_deep_analysis_area(deps, v, key_suffix):
    _bind_deps(deps)
    video_path = v["video_path"]
    _noi_dung_khu_vuc_phan_tich(v, key_suffix, video_path)


def _noi_dung_khu_vuc_phan_tich(v, key_suffix, video_path):
    _STALL_SECONDS = 180   # heartbeat im > 3 phút = thread đã chết
    _SLOW_SECONDS  = 300   # chạy > 5 phút mà progress < 30% = cảnh báo chậm

    prog_data = read_progress(video_path)

    is_processing = False
    p_val = 0.0
    elapsed = 0.0
    is_error = False
    err_msg = ""
    status_msg = ""
    heartbeat = 0.0
    _p1_entry = {}

    # Grace period khi vừa bấm Thử lại — hiển thị loading tối thiểu 8s dù thread fail nhanh
    _retry_key = f"_retry_start_{key_suffix}"
    _retry_start_ts = float(st.session_state.get(_retry_key, 0))
    _just_retried = (time.time() - _retry_start_ts) < 8

    if prog_data:
        status = prog_data.get("status")
        if status == "processing":
            # Luôn hiển thị "đang xử lý" khi progress file nói "processing".
            # Không check heartbeat age ở đây — stall detection (_STALL_SECONDS=180s)
            # sẽ hiện cảnh báo và nút Khởi động lại nếu thread thực sự đã chết.
            is_processing = True
            p_val = prog_data.get("progress", 0.0)
            elapsed = prog_data.get("elapsed", 0.0)
            status_msg = prog_data.get("status_msg", "")
            heartbeat = float(prog_data.get("heartbeat") or 0)
            # Dam bao progress khong di lui trong UI — so sanh start_time de detect restart
            _prog_track_key = f"_prog_track_{key_suffix}"
            _prog_track = st.session_state.get(_prog_track_key, {"start_time": 0.0, "max_p": 0.0})
            _file_start = float(prog_data.get("start_time") or 0)
            if _file_start > 0 and _prog_track["start_time"] == _file_start:
                p_val = max(p_val, _prog_track["max_p"])
            else:
                _prog_track["start_time"] = _file_start
            _prog_track["max_p"] = p_val
            st.session_state[_prog_track_key] = _prog_track
            # Ghi nhan thoi diem bat dau Pass 1 de ETA chinh xac hon
            _p1_entry_key = f"_p1_entry_{key_suffix}"
            _p1_entry = st.session_state.get(_p1_entry_key, {})
            if p_val >= 0.185 and not _p1_entry.get("set"):
                st.session_state[_p1_entry_key] = {"t": time.time(), "p": p_val, "set": True, "start": _file_start}
                _p1_entry = st.session_state[_p1_entry_key]
            elif _p1_entry.get("start") != _file_start:
                st.session_state.pop(_p1_entry_key, None)
                _p1_entry = {}
        elif status == "error":
            if _just_retried:
                # Thread thất bại rất nhanh (race condition) — giữ loading UI 8s sau khi bấm Thử lại
                is_processing = True
                p_val = prog_data.get("progress", 0.02)
                elapsed = time.time() - _retry_start_ts
                status_msg = "🔄 Đang khởi động lại phân tích..."
            else:
                is_error = True
                err_msg = prog_data.get("error_msg", "Lỗi không xác định")
        elif status == "success":
            if finalize_background_analysis_if_ready(video_path):
                _lam_moi_giao_dien_sau_nut()
            else:
                st.rerun()

    now = time.time()
    start_t = float(prog_data.get("start_time") or now) if prog_data else now
    elapsed_live = now - start_t

    # Phân loại trạng thái
    _thread_alive = _thread_dang_chay_thuc_su(video_path)
    # CHI coi la "ket" khi heartbeat THUC SU im > 3 phut (_STALL_SECONDS). TRUOC day con
    # dua vao (thread khong alive + elapsed_live > 180): nhung sau khi Space restart/deploy,
    # _running_threads trong bo nho RONG nen MOI video 'processing' deu bi coi la 'thread
    # missing' du heartbeat vua moi ghi (resume se tu chay lai) -> bao "ket 0 phut / crash
    # RAM" SAI, lam nguoi dung hoang. Heartbeat moi la tin hieu dung de biet con cap nhat.
    _heartbeat_stale = heartbeat > 0 and (now - heartbeat) > _STALL_SECONDS
    is_stalled   = is_processing and _heartbeat_stale and not _just_retried
    is_slow      = (is_processing and not is_stalled
                    and elapsed_live > _SLOW_SECONDS and p_val < 0.30 and p_val > 0.01)

    # ETA ước tính — dùng tốc độ Pass 1 thực tế (bỏ qua giai đoạn init 0-18% nhanh hơn)
    def _eta_str():
        if p_val < 0.20:
            return None
        # Ưu tiên: ETA theo frame thực tế — chính xác nhất (biết fps thực)
        import re as _re_eta
        _fc_m = _re_eta.search(r'Frame (\d+)/(\d+)', status_msg)
        if _fc_m and elapsed_live > 15:
            _fc_cur = int(_fc_m.group(1)); _fc_tot = int(_fc_m.group(2))
            if _fc_cur > 5 and _fc_tot > 0:
                # fps "biểu kiến" (bao gồm init time) → an toàn hơn (ETA hơi cao hơn thực)
                _fps = _fc_cur / elapsed_live
                _p1_rem_s = max(_fc_tot - _fc_cur, 0) / _fps
                # Pass 2 thường nhanh hơn Pass 1 ~50% (chỉ vẽ overlay, không chạy AI detect)
                _p2_est_s = (_fc_tot / _fps) * 0.50
                remaining = _p1_rem_s + _p2_est_s
                if remaining > 7200: return f"~{remaining/3600:.1f} giờ"
                if remaining > 120: return f"~{int(remaining//60)} phút"
                return f"~{int(remaining)}s"

        # Fallback A: dùng p1_entry rate (rate thực sau init)
        if _p1_entry.get("set"):
            p1_elapsed = time.time() - _p1_entry["t"]
            p1_done = max(p_val - _p1_entry["p"], 0.001)
            if p1_elapsed >= 30 and p1_done >= 0.002:
                rate = p1_elapsed / p1_done
                remaining = rate * (1.0 - p_val)
            elif elapsed_live >= 120:
                # Trừ ước tính init time (p_val lúc bắt đầu callback ≈ 0.185)
                _eff_p = max(p_val - 0.185, 0.001)
                _eff_e = max(elapsed_live * _eff_p / p_val, 5)
                remaining = _eff_e / _eff_p * (1.0 - p_val)
            else:
                return None
        else:
            # Fallback B: chưa có p1_entry
            if elapsed_live < 120:
                return None
            _eff_p = max(p_val - 0.185, 0.001)
            _eff_e = max(elapsed_live * _eff_p / p_val, 5)
            remaining = _eff_e / _eff_p * (1.0 - p_val)
        if remaining > 7200: return f"~{remaining/3600:.1f} giờ"
        if remaining > 120: return f"~{int(remaining//60)} phút"
        return f"~{int(remaining)}s"

    # Force-stop: set cancel flag để thread tự thoát sạch + dừng fragment refresh
    def _dung_phan_tich():
        flag = _cancel_flags.get(video_path)
        if flag:
            flag.set()   # Thread sẽ kiểm tra cờ này và thoát sớm
        _running_threads.pop(video_path, None)
        st.session_state.pop("_analysis_started_this_session", None)
        st.session_state.pop("reanalyze_triggered", None)
        write_progress(video_path, "error", progress=p_val, elapsed=elapsed_live,
                       start_time=start_t,
                       error_msg="⛔ Người dùng đã dừng phân tích. Nhấn 'Thử lại' để chạy lại với cài đặt khác.")

    # Khi error + grace period hết: dừng auto-refresh bằng cách trigger full rerun 1 lần
    # Tránh nút nhấp nháy do trạng thái tiến trình cũ còn sót sau khi thread đã chết
    _err_stop_key = f"_error_stop_refresh_{key_suffix}"
    if is_error and not _just_retried:
        if st.session_state.get(_err_stop_key) != video_path:
            st.session_state[_err_stop_key] = video_path
            try:
                st.rerun(scope="app")
            except TypeError:
                st.rerun()

    with st.expander("📖 Luồng phân tích 4 bước (bấm để xem)", expanded=False):
        st.markdown("""
**Bước 1 — Trích xuất khung xương (Pass 1 · MediaPipe Pose)**
- Đọc từng khung hình video, chạy **MediaPipe Pose** (Heavy / Full / Lite tuỳ cài đặt sidebar)
- Trích xuất **33 điểm mốc** (landmarks) toàn thân mỗi khung hình, gồm vai, khuỷu, cổ tay, hông, gối, mắt cá
- Tính **góc khớp vai** (shoulder angle) và **góc khớp khuỷu** (elbow angle) theo bài tập:
  - *Codman / Dây kháng lực*: theo dõi tay phải (hoặc tay chủ đạo được nhận diện tự động)
  - *Gậy pulley*: theo dõi đồng thời cả hai tay
- Lưu kết quả dưới dạng **file checkpoint** sau Pass 1 → nếu Space tắt giữa chừng, tiếp tục từ Bước 2 mà không cần chạy lại

**Bước 2 — Đối chiếu chuẩn YouTube (Pass 2 · RULE Labeling)**
- Nạp **tư thế chuẩn** từ file tham chiếu (reference_codman / reference_gay / reference_day) trích xuất từ video YouTube chuyên gia
- So sánh góc khớp từng khung hình với **ngưỡng sai số** theo giai đoạn phục hồi (G1 / G2 / G3)
- Gắn nhãn **REF** cho từng khung: `✅ Đúng` · `🟡 Gần đúng` · `❌ Sai`
- Vẽ **khung xương 33 điểm**, màu sắc theo nhãn, lên từng frame → tạo **video phân tích** đầu ra

**Bước 3 — Phân loại ML (Random Forest Classifier)**
- Sau khi thu thập đủ dữ liệu, huấn luyện (hoặc nạp) **Random Forest** trên đặc trưng: tọa độ landmarks + góc vai + góc khuỷu
- Dự đoán nhãn **ML** độc lập với RULE cho từng khung hình
- Hiển thị song song nhãn REF và nhãn ML → bác sĩ thấy mức độ nhất quán giữa luật cứng và học máy

**Bước 4 — Đóng gói kết quả**
- **Video** khung xương có nhãn REF + ML (`.mp4`)
- **CSV** từng khung hình: góc vai, góc khuỷu, nhãn REF, nhãn ML, thời điểm (ms)
- **JSON** toàn bộ frame data (tọa độ 33 điểm) để tái phân tích sau
- **ZIP** ảnh từng frame (tuỳ chọn) · **Biểu đồ** ROM, độ chính xác, F1-score theo giai đoạn
- Tự đồng bộ lên **HF Dataset** để lưu trữ lâu dài sau khi Space tắt
""")

    if is_error:
        st.error(f"❌ Phân tích thất bại: {err_msg}")
        if st.button("🔄 THỬ LẠI PHÂN TÍCH", width="stretch", type="primary", key=f"btn_retry_bg_{key_suffix}"):
            st.session_state.pop(_err_stop_key, None)  # Cho phép stop-refresh lần tiếp theo
            st.session_state[_retry_key] = time.time()  # Grace period: ẩn lỗi 8s sau khi bấm
            clear_analysis_progress(video_path)
            _bat_che_do_cuu_ho_hf(video_path)
            _xu_ly_ket_qua_khoi_dong_phan_tich(khoi_dong_phan_tich_lai_video(v, auto_start=True))

    elif is_stalled:
        stall_min = max(int((now - heartbeat) // 60), 3)
        st.warning(
            f"⏳ Tiến trình chưa cập nhật trong ~**{stall_min} phút** (đang ở {p_val*100:.1f}%). "
            f"Hệ thống sẽ **tự chạy lại từ checkpoint**. Nếu chờ lâu, bấm **Khởi động lại** "
            f"hoặc **Xem kết quả cũ** (nếu đã có)."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Khởi động lại", width="stretch", type="primary", key=f"btn_restart_stall_{key_suffix}"):
                clear_analysis_progress(video_path)
                _bat_che_do_cuu_ho_hf(video_path)
                _xu_ly_ket_qua_khoi_dong_phan_tich(khoi_dong_phan_tich_lai_video(v, auto_start=True))
        with c2:
            if v.get("metrics") and st.button("⬅️ Xem kết quả cũ", width="stretch", type="secondary", key=f"btn_old_stall_{key_suffix}"):
                _quay_lai_ket_qua_cu_da_luu(v, rerun=False)

    elif is_slow:
        import re as _re
        eta = _eta_str()
        elapsed_min = int(elapsed_live // 60)
        _em = int(elapsed_live // 60); _es = int(elapsed_live % 60)
        _elapsed_str = f"{_em}m {_es:02d}s" if _em else f"{_es}s"
        st.warning(
            f"🐢 **Video đang xử lý rất chậm** — đã chạy **{elapsed_min} phút**, "
            f"mới được **{p_val*100:.1f}%**"
            + (f" | Ước tính còn **{eta}**" if eta else "") +
            ".\n\n💡 **Gợi ý:** Đổi model sang **MediaPipe Lite** ở sidebar để tăng tốc ~5×, "
            "hoặc nhấn **Dừng** để huỷ và chạy lại."
        )
        _hien_thi_progress_hai_pass(
            prog_data,
            status_msg=status_msg,
            elapsed_text=f"⏱️ {_elapsed_str}",
            show_total=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⛔ Dừng phân tích", width="stretch", type="primary", key=f"btn_stop_slow_{key_suffix}"):
                _dung_phan_tich()
                _lam_moi_giao_dien_sau_nut()
        with c2:
            _sel_slow_label = "chế độ nhanh" if _is_hf_runtime() else st.session_state.get("ncv_model_type", "MediaPipe Heavy").replace("MediaPipe ", "")
            if st.button(f"⚡ Chạy lại với {_sel_slow_label}", width="stretch", type="secondary", key=f"btn_restart_slow_{key_suffix}"):
                clear_analysis_progress(video_path)
                _bat_che_do_cuu_ho_hf(video_path)
                _xu_ly_ket_qua_khoi_dong_phan_tich(khoi_dong_phan_tich_lai_video(v, auto_start=True))
        with c3:
            if v.get("metrics") and st.button("⬅️ Kết quả cũ", width="stretch", type="secondary", key=f"btn_old_slow_{key_suffix}"):
                _quay_lai_ket_qua_cu_da_luu(v, rerun=False)

    elif is_processing:
        detail = f" — {status_msg}" if status_msg else ""
        _em = int(elapsed_live // 60); _es = int(elapsed_live % 60)
        _elapsed_str = f"{_em}m {_es:02d}s" if _em else f"{_es}s"
        _p2_loading = (
            0.43 <= p_val <= 0.52
            and status_msg
            and ("chuẩn bị model ML" in status_msg or ("Pass 2" in status_msg and "Frame 1/" in status_msg))
        )
        _is_model_init = bool(status_msg and "Đang khởi tạo AI" in status_msg)
        _is_stuck = (p_val < 0.20 or p_val >= 0.92 or _p2_loading or _is_model_init) and not (status_msg and "Frame" in status_msg)
        if _is_stuck:
            _is_dl = bool(status_msg and "tải video" in status_msg)
            _indet_css = (
                '<style>@keyframes _indet{0%{left:-35%;width:35%}100%{left:100%;width:35%}}'
                '.indet-w{position:relative;height:8px;background:rgba(0,100,255,.12);border-radius:4px;overflow:hidden;margin:4px 0 8px}'
                '.indet-f{position:absolute;height:100%;background:linear-gradient(90deg,#1a6fff,#00c6ff);border-radius:4px;animation:_indet 1.3s linear infinite !important}'
                '[data-stale] .indet-f,.stale .indet-f{animation:_indet 1.3s linear infinite !important}'
                '</style><div class="indet-w"><div class="indet-f"></div></div>'
            )
            if _is_dl:
                import re as _re
                _dl_m = _re.search(r'\((\d+)%\)', status_msg or "")
                _dl_pct = int(_dl_m.group(1)) / 100 if _dl_m else 0
                st.progress(max(_dl_pct, 0.01))
                st.info(f"⬇️ Đang tải video từ Cloud... **{int(_dl_pct*100)}%** | ⏱️ {_elapsed_str} — {status_msg}")
            elif _is_model_init:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 🤖 {status_msg} | ⏱️ {_elapsed_str}")
            elif p_val < 0.20:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 ⏳ Đang chuẩn bị phân tích... | ⏱️ {_elapsed_str}{detail}")
            elif _p2_loading:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 🤖 Đang tải model phân loại ML & khởi động Pass 2... | ⏱️ {_elapsed_str}{detail}")
            else:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 📦 Đang mã hóa & đóng gói video... | ⏱️ {_elapsed_str}{detail}")
            _hien_thi_progress_hai_pass(
                prog_data,
                status_msg=status_msg,
                elapsed_text=f"⏱️ {_elapsed_str}",
                show_total=False,
            )
        else:
            eta = _eta_str()
            eta_str = f" | ETA {eta}" if eta else ""
            st.markdown(
                '<style>div[data-testid="stProgress"]>div>div>div{'
                'animation:_ppulse 1.8s ease-in-out infinite}'
                '@keyframes _ppulse{0%,100%{opacity:1}50%{opacity:.6}}'
                '</style>',
                unsafe_allow_html=True,
            )
            _hien_thi_progress_hai_pass(
                prog_data,
                status_msg=status_msg,
                elapsed_text=f"⏱️ {_elapsed_str}{eta_str}",
                show_total=True,
            )
            st.info(f"🔄 Đang xử lý... **{p_val*100:.1f}%** | ⏱️ {_elapsed_str}{eta_str}{detail}")
        st.button(
            "🚀 ĐANG TRÍCH XUẤT KHUNG XƯƠNG...",
            width="stretch",
            type="primary",
            key=f"btn_analyze_disabled_metrics_{key_suffix}",
            disabled=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⛔ Dừng phân tích", width="stretch", type="secondary", key=f"btn_stop_metrics_{key_suffix}"):
                _dung_phan_tich()
                _lam_moi_giao_dien_sau_nut()
        with c2:
            if st.button("⬅️ Quay lại xem kết quả cũ đã lưu", width="stretch", type="secondary", key=f"btn_back_old_{key_suffix}"):
                _quay_lai_ket_qua_cu_da_luu(v, rerun=False)
        _sel_model_label = "chế độ nhanh" if _is_hf_runtime() else st.session_state.get("ncv_model_type", "MediaPipe Heavy").replace("MediaPipe ", "")
        if st.button(f"⚡ Dừng & chạy lại với {_sel_model_label}", width="stretch", type="primary", key=f"btn_restart_model_{key_suffix}"):
            _dung_phan_tich()
            _bat_che_do_cuu_ho_hf(video_path)
            _xu_ly_ket_qua_khoi_dong_phan_tich(khoi_dong_phan_tich_lai_video(v, auto_start=True))

    elif is_processing:
        detail = f" — {status_msg}" if status_msg else ""
        _em = int(elapsed_live // 60); _es = int(elapsed_live % 60)
        _elapsed_str = f"{_em}m {_es:02d}s" if _em else f"{_es}s"
        _p2_loading = (
            0.43 <= p_val <= 0.52
            and status_msg
            and ("chuẩn bị model ML" in status_msg or ("Pass 2" in status_msg and "Frame 1/" in status_msg))
        )
        _is_model_init = bool(status_msg and "Đang khởi tạo AI" in status_msg)
        _is_stuck = (p_val < 0.20 or p_val >= 0.92 or _p2_loading or _is_model_init) and not (status_msg and "Frame" in status_msg)
        if _is_stuck:
            _is_dl = bool(status_msg and "tải video" in status_msg)
            _indet_css = (
                '<style>@keyframes _indet{0%{left:-35%;width:35%}100%{left:100%;width:35%}}'
                '.indet-w{position:relative;height:8px;background:rgba(0,100,255,.12);border-radius:4px;overflow:hidden;margin:4px 0 8px}'
                '.indet-f{position:absolute;height:100%;background:linear-gradient(90deg,#1a6fff,#00c6ff);border-radius:4px;animation:_indet 1.3s linear infinite !important}'
                '[data-stale] .indet-f,.stale .indet-f{animation:_indet 1.3s linear infinite !important}'
                '</style><div class="indet-w"><div class="indet-f"></div></div>'
            )
            if _is_dl:
                import re as _re
                _dl_m = _re.search(r'\((\d+)%\)', status_msg or "")
                _dl_pct = int(_dl_m.group(1)) / 100 if _dl_m else 0
                st.progress(max(_dl_pct, 0.01))
                st.info(f"⬇️ Đang tải video từ Cloud... **{int(_dl_pct*100)}%** | ⏱️ {_elapsed_str} — {status_msg}")
            elif _is_model_init:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 🤖 {status_msg} | ⏱️ {_elapsed_str}")
            elif p_val < 0.20:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 ⏳ Đang chuẩn bị phân tích... | ⏱️ {_elapsed_str}{detail}")
            elif _p2_loading:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 🤖 Đang tải model phân loại ML & khởi động Pass 2... | ⏱️ {_elapsed_str}{detail}")
            else:
                st.markdown(_indet_css, unsafe_allow_html=True)
                st.info(f"🔄 📦 Đang mã hóa & đóng gói video... | ⏱️ {_elapsed_str}{detail}")
        else:
            eta = _eta_str()
            eta_str = f" | ETA {eta}" if eta else ""
            st.markdown(
                '<style>div[data-testid="stProgress"]>div>div>div{'
                'animation:_ppulse 1.8s ease-in-out infinite}'
                '@keyframes _ppulse{0%,100%{opacity:1}50%{opacity:.6}}'
                '</style>',
                unsafe_allow_html=True,
            )
            st.progress(p_val)
            st.info(f"🔄 Đang xử lý... **{p_val*100:.1f}%** | ⏱️ {_elapsed_str}{eta_str}{detail}")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.button(
                "🚀 ĐANG TRÍCH XUẤT KHUNG XƯƠNG...",
                width="stretch",
                type="primary",
                key=f"btn_analyze_disabled_{key_suffix}",
                disabled=True,
            )
        with c2:
            if st.button("⛔ Dừng", width="stretch", type="secondary", key=f"btn_stop_plain_{key_suffix}"):
                _dung_phan_tich()
                _lam_moi_giao_dien_sau_nut()
    else:
        # Nếu vừa bấm nút phân tích nhưng thread chưa ghi progress → hiện loading ngay
        _triggered_at_key = f"_reanalyze_triggered_at_{key_suffix}"
        if st.session_state.get("reanalyze_triggered"):
            if not st.session_state.get(_triggered_at_key):
                st.session_state[_triggered_at_key] = time.time()
            _trigger_age = time.time() - float(st.session_state.get(_triggered_at_key, time.time()))
            if _trigger_age < 20:
                st.info(f"⏳ Đang khởi động phân tích... vui lòng chờ ({int(_trigger_age)}s)")
                st.progress(min(0.02, _trigger_age / 20 * 0.05))
            else:
                # Đã chờ 20s mà không có progress → thread crash / không khởi động được
                st.session_state.pop("reanalyze_triggered", None)
                st.session_state.pop(_triggered_at_key, None)
                st.warning("⚠️ Không thể khởi động phân tích sau 20 giây. Thử lại bên dưới.")
                if st.button("🔄 Thử lại phân tích", width="stretch", type="primary", key=f"btn_retry_timeout_{key_suffix}"):
                    clear_analysis_progress(video_path)
                    _bat_che_do_cuu_ho_hf(video_path)
                    _xu_ly_ket_qua_khoi_dong_phan_tich(khoi_dong_phan_tich_lai_video(v, auto_start=True))
        else:
            st.session_state.pop(_triggered_at_key, None)
            if st.button("🚀 PHÂN TÍCH VÀ TRÍCH XUẤT KHUNG XƯƠNG NGAY", width="stretch", type="primary", key=f"btn_analyze_now_{key_suffix}"):
                result = khoi_dong_phan_tich_lai_video(v, auto_start=True)
                if isinstance(result, dict) and result.get("started"):
                    st.toast("🚀 Đã khởi chạy phân tích — tiến độ cập nhật ngay bên dưới!", icon="⚡")
                    _lam_moi_giao_dien_sau_nut()
                else:
                    _xu_ly_ket_qua_khoi_dong_phan_tich(result)

def _hien_thi_tab_phan_tich_noi_dung(deps, key_suffix="", stats_ext=None, df_ext=None, exercise_ext=None):
    _bind_deps(deps)
    """Nội dung tab phân tích — tách riêng để fragment có thể auto-refresh khi đang chạy AI."""
    user_role = st.session_state.user_info.get('role')

    if st.session_state.pop("_pending_chart_refresh", False):
        st.session_state.view_old_analysis = True

    # Luôn kiểm tra phân tích nền đã xong — kể cả đang chạy phân tích mới
    if st.session_state.get('current_eval_video'):
        v_path = st.session_state.current_eval_video.get('video_path')
        if v_path and finalize_background_analysis_if_ready(v_path):
            v_done = _lam_moi_ban_ghi_video_tu_db(st.session_state.current_eval_video)
            if v_done:
                st.session_state.current_eval_video = v_done
                _gan_khoa_session_phan_tich(v_done)
            st.toast("✅ Phân tích xong! Đang hiển thị biểu đồ...", icon="🎉")
            st.session_state._pending_chart_refresh = True

    # Nếu không có dữ liệu truyền vào -> Kiểm tra tải tự động (Dành cho NCV)
    if stats_ext is None and df_ext is None:
        # TỰ ĐỘNG CHỌN VIDEO MỚI NHẤT NẾU CHƯA CHỌN (Dành cho Nghiên cứu viên)
        if not st.session_state.get('current_eval_video'):
            if user_role == "Nghiên cứu viên":
                v_latest = _tim_video_phan_tich_moi_nhat()
                if v_latest:
                    st.session_state.current_eval_video = v_latest
            else:
                video_list = load_data(VIDEOS_FILE)
                if video_list:
                    analyzed = [v for v in video_list if v.get("metrics")]
                    st.session_state.current_eval_video = (analyzed or video_list)[-1]

        # TỰ ĐỘNG LOAD DỮ LIỆU — đúng BN/bài tập đang chọn (không dùng cache video khác)
        if st.session_state.get('current_eval_video'):
            v = _lam_moi_ban_ghi_video_tu_db(st.session_state.current_eval_video)
            st.session_state.current_eval_video = v
            has_metrics = bool(v.get("metrics"))

            slot_v = _slot_video_phan_tich(v)
            if (
                slot_v
                and st.session_state.get("_ncv_analysis_loaded_key")
                and st.session_state.get("_ncv_analysis_loaded_key") != slot_v
            ):
                _xoa_session_phan_tich()
            if has_metrics and not st.session_state.get("view_old_analysis", False):
                st.session_state.view_old_analysis = True

            # Tự nạp kết quả — không spinner (tải song song nền), kể cả khi đang phân tích mới
            if has_metrics and (
                st.session_state.get("angle_df") is None
                or not _session_phan_tich_khop_video(v)
            ):
                _prog_tmp = read_progress(v.get("video_path"))
                _dang_chay = bool(_prog_tmp and _prog_tmp.get("status") == "processing")
                # Giữ reanalyze_triggered nếu vừa bấm nút — tránh race condition khi thread
                # chưa kịp ghi progress file lần đầu mà _dang_chay vẫn còn False
                _giu = _dang_chay or bool(st.session_state.get("reanalyze_triggered"))
                loaded, v = _nap_bieu_do_nhanh_tu_cloud(v, giu_phan_tich_moi=_giu)
                if loaded:
                    st.session_state.current_eval_video = v

            if has_metrics:
                _fragment_tien_do_tai_media(v, key_suffix)

            prog_data = read_progress(v.get('video_path'))
            is_processing = bool(
                prog_data and prog_data.get("status") == "processing"
                and _thread_dang_chay_thuc_su(v.get('video_path'))
            )

            if has_metrics and not st.session_state.get("has_data"):
                _gan_session_ket_qua_tu_video(v)

            if has_metrics and not (
                st.session_state.get("has_data")
                and st.session_state.get("angle_df") is not None
            ):
                if st.button(
                    "📊 XEM KẾT QUẢ ĐÃ LƯU NGAY",
                    key=f"btn_force_load_saved_{key_suffix}",
                    width="stretch",
                    type="primary",
                ):
                    _quay_lai_ket_qua_cu_da_luu(v, rerun=False)

            da_co_du_lieu = bool(
                st.session_state.get("has_data")
                and st.session_state.get("angle_df") is not None
            )
            hien_thi_bieu_do = da_co_du_lieu

            # Ảnh 2: video đã có kết quả lưu — luôn dùng layout gọn, tải Cloud liền mạch
            if has_metrics and (is_processing or st.session_state.get("reanalyze_triggered")):
                _hien_thi_hang_video_va_tien_do(v, key_suffix, is_processing=is_processing)
                return

            # Ảnh 1: video chưa từng phân tích — màn chờ lần đầu
            if not hien_thi_bieu_do and not has_metrics and (is_processing or st.session_state.get("reanalyze_triggered")):
                _hien_thi_thong_bao_che_do_phan_tich_moi()
                _hien_thi_hang_video_va_tien_do(v, key_suffix, is_processing=is_processing)
                return

            if not hien_thi_bieu_do and not has_metrics:
                st.warning(f"⚠️ Video '{v.get('video_name')}' của BN {patient_display_label(v)} chưa được phân tích.")
                return

    # Lấy dữ liệu (Ưu tiên tham số truyền vào từ Doctor/Patient view)
    bt = exercise_ext if exercise_ext is not None else st.session_state.get('exercise', BAI_TAP['codman'])
    if bt is None: bt = BAI_TAP['codman']
    tk = stats_ext if stats_ext is not None else st.session_state.get('stats')
    df = df_ext if df_ext is not None else st.session_state.get('angle_df')

    # FALLBACK: Nếu angle_df đã được giải phóng khỏi RAM -> Đọc lại từ CSV để tiết kiệm bộ nhớ
    if df is None and df_ext is None:
        csv_path = st.session_state.get('current_df_csv_path')
        if not csv_path and st.session_state.get('current_eval_video'):
            csv_path = st.session_state.current_eval_video.get('df_path')
        if csv_path:
            csv_local = get_local_frame_path(csv_path) or csv_path
            _read_fp = csv_local if os.path.exists(csv_local) else (csv_path if os.path.exists(csv_path) else None)
            if not _read_fp and HF_TOKEN:
                _bat_dau_tai_day_du_song_song(st.session_state.get("current_eval_video"))
            if _read_fp:
                try:
                    df = read_display_csv_fast(_read_fp)
                except:
                    pass

    if tk is None or df is None:
        v_re = _lam_moi_ban_ghi_video_tu_db(
            st.session_state.get("current_eval_video") or _tim_video_phan_tich_moi_nhat()
        )
        if v_re:
            _prog_tmp = read_progress(v_re.get("video_path"))
            _dang_chay = bool(_prog_tmp and _prog_tmp.get("status") == "processing")
            if _nap_bieu_do_nhanh_tu_cloud(v_re, giu_phan_tich_moi=_dang_chay)[0]:
                tk = st.session_state.get("stats") or tk
                df = st.session_state.get("angle_df") if df is None else df
                _bt_new = st.session_state.get("exercise")
                if _bt_new:
                    bt = _bt_new
                _lam_moi_giao_dien_sau_nut()
        if tk is None or df is None:
            _ok_hf, _msg_hf = kiem_tra_quyen_hf_dataset()
            if not _ok_hf and _msg_hf:
                _info_msg = f"🔐 **Lỗi kết nối Cloud:** {_msg_hf}"
            elif _hf_last_download_error:
                _info_msg = f"☁️ **Chưa tải được từ Cloud:** {_hf_last_download_error}"
            else:
                _info_msg = (
                    "☁️ **Dữ liệu CSV/JSON chưa tải về.** "
                    "Bấm **Tải lại** để kéo từ Dataset, hoặc **Phân tích mới** nếu video này chưa từng được phân tích."
                )
            st.info(_info_msg)
            if user_role == "Nghiên cứu viên":
                hien_thi_nut_tai_lai_va_phan_tich_moi(v_re, key_suffix=f"missing_{key_suffix}")
            return

    # Nút thao tác nhanh khi đã có kết quả (NCV) — bỏ qua nếu đã hiện ở hàng video/tiến độ phía trên
    _v_hdr = st.session_state.get("current_eval_video")
    _prog_hdr = read_progress(_v_hdr.get("video_path")) if _v_hdr else None
    # Chỉ ẩn nút khi thread PHÂN TÍCH thực sự đang chạy — không dựa progress file cũ
    _dang_phan_tich_hdr = bool(
        _prog_hdr and _prog_hdr.get("status") == "processing"
        and _thread_dang_chay_thuc_su(_v_hdr.get("video_path") if _v_hdr else None)
    )
    if user_role == "Nghiên cứu viên" and tk is not None and not _dang_phan_tich_hdr:
        _current_patient_label = patient_display_label(st.session_state.get('current_eval_video', {}))
        st.success(
            f"📊 **KẾT QUẢ ĐÃ LƯU:** BN **{_current_patient_label}** — "
            "biểu đồ bên dưới · tab **🎬 VIDEO & ẢNH FRAME** (video + ảnh khung xương)."
        )
        hien_thi_nut_tai_lai_va_phan_tich_moi(
            st.session_state.get("current_eval_video"),
            key_suffix=f"loaded_{key_suffix}",
        )
        st.markdown("<br>", unsafe_allow_html=True)

    is_gay_ex = any(kw in str(bt.get('ten', '')).lower() for kw in ["gậy", "gay", "pulley", "stick"])

    # Tính toán chỉ số cho cả 3 giai đoạn
    if df is not None and len(df) > 0:
        if is_gay_ex:
            metrics_overall = recalc_metrics(df, tk.get('sai_so', bt['chuan']['sai_so']) if isinstance(tk, dict) else bt['chuan']['sai_so'], bt.get('ten', ''))
            metrics_g1 = metrics_overall
            metrics_g2 = metrics_overall
            metrics_g3 = metrics_overall
        else:
            bounds = segment_frames(df)
            n0, n1, n2, n3 = bounds
            df_g1 = df.iloc[n0:n1]
            df_g2 = df.iloc[n1:n2]
            df_g3 = df.iloc[n2:n3]
            metrics_g1 = recalc_metrics(df_g1, PHASE_ERROR["g1"], bt.get('ten', ''))
            metrics_g2 = recalc_metrics(df_g2, PHASE_ERROR["g2"], bt.get('ten', ''))
            metrics_g3 = recalc_metrics(df_g3, PHASE_ERROR["g3"], bt.get('ten', ''))
    else:
        metrics_g1 = tk.get("metrics_g1", tk)
        metrics_g2 = tk.get("metrics_g2", tk)
        metrics_g3 = tk.get("metrics_g3", tk)

    stored_metrics_for_ml = tk if isinstance(tk, dict) else {}

    def _merge_stored_ml_fields(metric_block, stored_block):
        if not isinstance(metric_block, dict) or not isinstance(stored_block, dict):
            return metric_block
        for ml_key, ml_value in stored_block.items():
            if str(ml_key).startswith("ml_"):
                metric_block[ml_key] = ml_value
        return metric_block

    metrics_g1 = _merge_stored_ml_fields(metrics_g1, stored_metrics_for_ml.get("metrics_g1", {}))
    metrics_g2 = _merge_stored_ml_fields(metrics_g2, stored_metrics_for_ml.get("metrics_g2", {}))
    metrics_g3 = _merge_stored_ml_fields(metrics_g3, stored_metrics_for_ml.get("metrics_g3", {}))
    if is_gay_ex:
        metrics_g1 = _merge_stored_ml_fields(metrics_g1, stored_metrics_for_ml)
        metrics_g2 = _merge_stored_ml_fields(metrics_g2, stored_metrics_for_ml)
        metrics_g3 = _merge_stored_ml_fields(metrics_g3, stored_metrics_for_ml)

    if is_gay_ex:
        v_meta = st.session_state.get('current_eval_video') or {}
        stored_acc = lay_do_chinh_xac_ai_chuan(v_meta) or (tk.get('do_chinh_xac') if isinstance(tk, dict) else None)
        if stored_acc is not None:
            if isinstance(metrics_g1, dict): metrics_g1['do_chinh_xac'] = stored_acc
            if isinstance(metrics_g2, dict): metrics_g2['do_chinh_xac'] = stored_acc
            if isinstance(metrics_g3, dict): metrics_g3['do_chinh_xac'] = stored_acc

    if is_gay_ex:
        # 1. HIỂN THỊ HIỆU SUẤT TỔNG QUAN (KHÔNG CHIA GIAI ĐOẠN)
        st.markdown("### 🏥 HIỆU SUẤT TẬP LUYỆN TỔNG QUAN (ĐỐI CHIẾU VIDEO YOUTUBE)")

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0, 198, 255, 0.1) 0%, rgba(0, 114, 255, 0.2) 100%);
                    padding: 20px; border-radius: 12px; border: 1px solid #00c6ff; text-align: center; margin-bottom: 15px;">
            <h4 style="margin: 0; color: #00c6ff; font-size: 1.15rem; font-weight: bold;">🏒 Hiệu suất bài tập với gậy</h4>
            <p style="margin: 5px 0; font-size: 0.9rem; color: #ccc;">Sai số cho phép: <b>{tk.get('sai_so', bt['chuan']['sai_so'])}°</b></p>
            <h3 style="margin: 10px 0; color: #fff; font-size: 2.2rem; font-weight: bold;">{metrics_g2['do_chinh_xac']:.1f}%</h3>
            <p style="margin: 0; font-size: 0.85rem; color: #bbb;">Đúng: <b>{metrics_g2['frame_dung']}</b> | Gần đúng: <b>{metrics_g2['frame_gan_dung']}</b> | Sai: <b>{metrics_g2['frame_sai']}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # AI Suggestion
        acc_val = metrics_g2['do_chinh_xac']
        if acc_val >= 80:
            recommended_gd = "Đạt yêu cầu - Tốt"
            gd_color = "#4CAF50"
            reason = f"Bệnh nhân đạt độ chính xác {acc_val:.1f}% ở bài tập với gậy. Kết quả rất tốt, khớp vai di chuyển đồng bộ với mẫu và không bị co cứng hay hạn chế tầm vận động."
        elif acc_val >= 50:
            recommended_gd = "Khá - Cần luyện tập thêm"
            gd_color = "#2196F3"
            reason = f"Bệnh nhân đạt độ chính xác {acc_val:.1f}%. Cử động khớp vai tương đối tốt nhưng cần lưu ý giữ thẳng tay và kiểm soát góc khuỷu hơn nữa."
        else:
            recommended_gd = "Chưa đạt - Cần giám sát"
            gd_color = "#F44336"
            reason = f"Độ chính xác chỉ đạt {acc_val:.1f}%. Bệnh nhân nâng gậy chưa đúng biên độ hoặc gập cùi chỏ (khuỷu tay) quá nhiều. Cần chuyên gia y tế hướng dẫn lại."

        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); padding: 18px; border-radius: 12px; border: 1px dashed rgba(255, 255, 255, 0.15); margin-top: 15px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.3rem;">🤖</span>
                <h4 style="margin: 0; color: #00c6ff; font-size: 1.05rem; font-weight: bold;">HỆ THỐNG GỢI Ý ĐÁNH GIÁ PHÙ HỢP (AI CLASSIFIER)</h4>
            </div>
            <p style="margin: 5px 0; font-size: 0.9rem;">
                Dựa trên kết quả tập luyện thực tế đối chiếu với video mẫu YouTube, AI gợi ý:
                <b style="color: {gd_color}; font-size: 1rem;">{recommended_gd}</b>
            </p>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #bbb; font-style: italic;">
                <b>Lý do lâm sàng:</b> {reason}
            </p>
        </div>
        """, unsafe_allow_html=True)
        _sound_gay = "sounds/dung.mp3" if acc_val >= 80 else ("sounds/gan_dung.mp3" if acc_val >= 50 else "sounds/sai.mp3")
        _sound_label = "✅ Đúng" if acc_val >= 80 else ("⚠️ Gần đúng" if acc_val >= 50 else "❌ Sai")
        if os.path.exists(_sound_gay):
            st.caption(f"🔊 AI nhận định: **{_sound_label}**")
            st.audio(_sound_gay)
    else:
        # 1. PHÂN CHIA VÀ SO SÁNH 3 GIAI ĐOẠN TẬP LUYỆN
        st.markdown("### 🏥 HIỆU SUẤT THEO 3 GIAI ĐOẠN HỒI PHỤC (ĐỐI CHIẾU VIDEO YOUTUBE)")
        col_g1, col_g2, col_g3 = st.columns(3)

        with col_g1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.2) 100%);
                        padding: 15px; border-radius: 12px; border: 1px solid #4CAF50; text-align: center;">
                <h4 style="margin: 0; color: #4CAF50; font-size: 1.05rem; font-weight: bold;">🌱 Giai đoạn 1: Khởi đầu</h4>
                <p style="margin: 5px 0; font-size: 0.85rem; color: #ccc;">Sai số cho phép: <b>45°</b></p>
                <h3 style="margin: 10px 0; color: #fff; font-size: 1.8rem; font-weight: bold;">{metrics_g1['do_chinh_xac']:.1f}%</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #bbb;">Đúng: <b>{metrics_g1['frame_dung']}</b> | Gần đúng: <b>{metrics_g1['frame_gan_dung']}</b> | Sai: <b>{metrics_g1['frame_sai']}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_g2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.2) 100%);
                        padding: 15px; border-radius: 12px; border: 1px solid #2196F3; text-align: center;">
                <h4 style="margin: 0; color: #2196F3; font-size: 1.05rem; font-weight: bold;">📈 Giai đoạn 2: Hồi phục</h4>
                <p style="margin: 5px 0; font-size: 0.85rem; color: #ccc;">Sai số cho phép: <b>30°</b> (Trung bình)</p>
                <h3 style="margin: 10px 0; color: #fff; font-size: 1.8rem; font-weight: bold;">{metrics_g2['do_chinh_xac']:.1f}%</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #bbb;">Đúng: <b>{metrics_g2['frame_dung']}</b> | Gần đúng: <b>{metrics_g2['frame_gan_dung']}</b> | Sai: <b>{metrics_g2['frame_sai']}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_g3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(244, 67, 54, 0.2) 100%);
                        padding: 15px; border-radius: 12px; border: 1px solid #F44336; text-align: center;">
                <h4 style="margin: 0; color: #F44336; font-size: 1.05rem; font-weight: bold;">🎯 Giai đoạn 3: Chuẩn xác</h4>
                <p style="margin: 5px 0; font-size: 0.85rem; color: #ccc;">Sai số cho phép: <b>15°</b> (Khắt khe)</p>
                <h3 style="margin: 10px 0; color: #fff; font-size: 1.8rem; font-weight: bold;">{metrics_g3['do_chinh_xac']:.1f}%</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #bbb;">Đúng: <b>{metrics_g3['frame_dung']}</b> | Gần đúng: <b>{metrics_g3['frame_gan_dung']}</b> | Sai: <b>{metrics_g3['frame_sai']}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # 1.1. AI TỰ ĐỘNG GỢI Ý GIAI ĐOẠN LUYỆN TẬP PHÙ HỢP
        acc_g1 = metrics_g1['do_chinh_xac']
        acc_g2 = metrics_g2['do_chinh_xac']
        acc_g3 = metrics_g3['do_chinh_xac']

        # Xác định giai đoạn gợi ý tự động dựa trên độ chính xác
        if acc_g3 >= 80:
            recommended_gd = "Giai đoạn 3 (Chuẩn xác)"
            gd_color = "#F44336"
            reason = f"Bệnh nhân đạt độ chính xác {acc_g3:.1f}% ở mức sai số nhỏ (15°). Đây là kết quả xuất sắc, khớp tập luyện rất tốt và bệnh nhân đã đạt mức hồi phục phục hồi tối đa!"
        elif acc_g2 >= 75:
            recommended_gd = "Giai đoạn 3 (Chuẩn xác) - Sắp hoàn thành"
            gd_color = "#2196F3"
            reason = f"Bệnh nhân đã tập tốt ở Giai đoạn 2 (độ chính xác {acc_g2:.1f}% với sai số 30°). Bệnh nhân có thể tự tin chuyển sang luyện tập ở Giai đoạn 3 (sai số khắt khe 15°)."
        elif acc_g2 >= 50:
            recommended_gd = "Giai đoạn 2 (Hồi phục)"
            gd_color = "#2196F3"
            reason = f"Bệnh nhân đạt độ chính xác {acc_g2:.1f}% ở mức sai số 30°. Khớp vai và khuỷu tay đã thích nghi khá tốt, cần tiếp tục tập luyện ở giai đoạn này."
        elif acc_g1 >= 50:
            recommended_gd = "Giai đoạn 1 (Khởi đầu)"
            gd_color = "#4CAF50"
            reason = f"Bệnh nhân mới tập đạt độ chính xác {acc_g1:.1f}% ở mức sai số lớn (45°). Khớp vai ban đầu còn cứng nên chấp nhận sai số lớn này, khuyên bệnh nhân kiên trì làm quen với khớp."
        else:
            recommended_gd = "Giai đoạn Học hỏi ban đầu (Dưới chuẩn GĐ1)"
            gd_color = "#E65100"
            reason = f"Độ chính xác ở Giai đoạn 1 chỉ đạt {acc_g1:.1f}%. Bệnh nhân đang thực hiện sai tư thế khớp hoặc khớp vai cực kỳ cứng. Cần hướng dẫn trực tiếp từ chuyên gia y tế."

        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); padding: 18px; border-radius: 12px; border: 1px dashed rgba(255, 255, 255, 0.15); margin-top: 15px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.3rem;">🤖</span>
                <h4 style="margin: 0; color: #00c6ff; font-size: 1.05rem; font-weight: bold;">HỆ THỐNG GỢI Ý GIAI ĐOẠN TẬP PHÙ HỢP (AI CLASSIFIER)</h4>
            </div>
            <p style="margin: 5px 0; font-size: 0.9rem;">
                Dựa trên kết quả tập luyện thực tế đối chiếu từng giây với video mẫu YouTube, AI gợi ý mức tập phù hợp của bệnh nhân:
                <b style="color: {gd_color}; font-size: 1rem;">{recommended_gd}</b>
            </p>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #bbb; font-style: italic;">
                <b>Lý do lâm sàng:</b> {reason}
            </p>
        </div>
        """, unsafe_allow_html=True)
        _best_acc = max(acc_g1, acc_g2, acc_g3)
        _sound_codman = "sounds/dung.mp3" if _best_acc >= 80 else ("sounds/gan_dung.mp3" if _best_acc >= 50 else "sounds/sai.mp3")
        _sound_label_c = "✅ Đúng" if _best_acc >= 80 else ("⚠️ Gần đúng" if _best_acc >= 50 else "❌ Sai")
        if os.path.exists(_sound_codman):
            st.caption(f"🔊 AI nhận định: **{_sound_label_c}**")
            st.audio(_sound_codman)

    if user_role == "Nghiên cứu viên":
        st.markdown("<br>", unsafe_allow_html=True)
        is_light = st.session_state.get('theme', 'dark') == 'light'
        cta_bg = "linear-gradient(135deg, rgba(0, 114, 255, 0.1) 0%, rgba(0, 198, 255, 0.1) 100%)" if not is_light else "linear-gradient(135deg, rgba(0, 114, 255, 0.05) 0%, rgba(0, 198, 255, 0.05) 100%)"
        cta_border = "rgba(0, 198, 255, 0.4)"

        btn_label = "📤 GỬI BÁO CÁO PHÂN TÍCH CHO BS & BN" if is_gay_ex else "📤 GỬI BÁO CÁO TỔNG HỢP 3 GIAI ĐOẠN CHO BS & BN"
        desc_text = "Bấm nút dưới đây để gửi báo cáo phân tích toàn diện cùng ý kiến gợi ý của AI cho cả <b>Bác sĩ điều trị</b> và <b>Bệnh nhân</b> xem."
        success_text = "✅ Đã gửi báo cáo phân tích thành công!"

        st.markdown(f"""
        <div style="background: {cta_bg}; border: 1px solid {cta_border}; padding: 18px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0, 114, 255, 0.15); margin-bottom: 10px;">
            <h4 style="margin: 0 0 8px 0; color: #00c6ff; font-weight: bold; font-size: 1.1rem; display: flex; align-items: center; gap: 10px;">
                {btn_label}
            </h4>
            <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #ccc;">
                {desc_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(btn_label, key=f"btn_send_3_stages_main_{key_suffix}", type="primary", use_container_width=True):
            if gui_bao_cao_tong_hop_3_giai_doan():
                st.success(success_text)
                st.balloons()
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if is_gay_ex:
        tk_selected = metrics_g2
        sai_so_selected = tk.get('sai_so', bt['chuan']['sai_so'])
        giai_doan_label = "Tổng quan bài tập"
    else:
        # 2. BỘ CHỌN CHI TIẾT TẬP TRỰC QUAN
        gd_selected = st.radio("🔍 Chọn Giai đoạn hiển thị chi tiết biểu đồ & Nhận định lâm sàng:",
                               [PHASE_UI_LABELS["g1"],
                                PHASE_UI_LABELS["g2"],
                                PHASE_UI_LABELS["g3"]],
                               index=1,
                               horizontal=True,
                               key=f"analysis_stage_sel_{key_suffix}")

        if "Giai đoạn 1" in gd_selected:
            tk_selected = metrics_g1
            sai_so_selected = PHASE_ERROR["g1"]
            giai_doan_label = "Giai đoạn 1"
        elif "Giai đoạn 3" in gd_selected:
            tk_selected = metrics_g3
            sai_so_selected = PHASE_ERROR["g3"]
            giai_doan_label = "Giai đoạn 3"
        else:
            tk_selected = metrics_g2
            sai_so_selected = PHASE_ERROR["g2"]
            giai_doan_label = "Giai đoạn 2"

    # Chuẩn bị dữ liệu thống kê tổng hợp (Mở rộng cho NCV)
    fail_count_total = tk_selected['tong_frame_hop_le'] - tk_selected['frame_dung'] - tk_selected['frame_gan_dung']
    stats_summary = pd.DataFrame({
        "Hạng mục": [
            "Tổng số khung hình",
            "Số lần tập đúng (Pass)",
            "Số lần tập gần đúng",
            "Số lần tập sai (Fail)",
            "Góc vai trung bình (ROM)",
            "Góc khuỷu trung bình (ROM)",
            "Độ lệch chuẩn (STD) Vai",
            "Độ lệch chuẩn (STD) Khuỷu",
            "Sai số tuyệt đối (MAE)",
            "ICC (Độ tin cậy)",
            "F1-Score (Học máy)"
        ],
        "Giá trị": [
            str(tk_selected['tong_frame']),
            str(tk_selected['frame_dung']),
            str(tk_selected['frame_gan_dung']),
            f"{max(0, fail_count_total)}",
            f"{tk_selected['tb_goc_vai']:.1f}°",
            f"{tk_selected['tb_goc_khuyu']:.1f}°",
            f"{tk_selected.get('std_goc_vai', 0):.2f}",
            f"{tk_selected.get('std_goc_khuyu', 0):.2f}",
            f"{tk_selected.get('mae_tong', 0):.2f}°",
            f"{tk_selected.get('icc', 0):.2f}",
            f"{tk_selected.get('f1_score', 0):.2f}"
        ]
    })

    # Lấy thông tin mô hình hiện tại
    model_type = st.session_state.get('ncv_model_type', 'MediaPipe Heavy')

    # 1. HEADER CHỈ SỐ TỔNG QUAN (CỐ ĐỊNH) - HIỂN THỊ ĐẦU TIÊN
    header_title = "📊 DASHBOARD PHÂN TÍCH NHANH" if "Lite" in model_type else "📊 DASHBOARD PHÂN TÍCH LÂM SÀNG"
    if "Heavy" in model_type: header_title = "🔬 PHÂN TÍCH NGHIÊN CỨU CHUYÊN SÂU"

    # Kiểm tra sự tồn tại của file reference để hiển thị trạng thái
    ex_key_ui = next((k for k in BAI_TAP if BAI_TAP[k]['ten'] == bt['ten']), 'codman')
    mapping_ui = {"codman": "codman", "gay": "gay", "khang_luc": "day"}
    ref_name_ui = mapping_ui.get(ex_key_ui, ex_key_ui)
    has_dynamic_ref = os.path.exists(os.path.join(DB_DIR, f"reference_{ref_name_ui}.json")) or os.path.exists(f"reference_{ref_name_ui}.json")

    is_light = st.session_state.theme == 'light'
    banner_bg = "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)" if is_light else "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
    banner_border = "#ced4da" if is_light else "#2a5298"
    banner_shadow = "0 10px 30px rgba(0,0,0,0.05)" if is_light else "0 10px 30px rgba(0,0,0,0.5)"
    title_text_color = "#0072ff" if is_light else "#ffd700"
    desc_text_color = "#666" if is_light else "#aaa"

    # Trạng thái Dynamic
    dyn_status = f'<span style="color: #00FF00; font-weight: bold;">⚡ DYNAMIC ON</span>' if has_dynamic_ref else '<span style="color: #888;">⚪ STATIC</span>'

    st.markdown(f"""
    <div style="background: {banner_bg};
                border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem;
                border: 1px solid {banner_border}; box-shadow: {banner_shadow};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="color: {title_text_color}; margin: 0; font-size: 1.8rem;">{header_title}</h2>
                <p style="color: {desc_text_color}; margin: 0.5rem 0 0 0;">
                    🏥 Bài tập: {bt['ten']} | ⚙️ Model: <span style="color:#00c6ff;">{model_type}</span> | {dyn_status}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(0,206,209,0.1); padding: 5px 15px; border-radius: 10px; border: 1px solid #00CED1;">
                    <span style="color: #00CED1; font-weight: bold; font-size: 1.2rem;">{tk_selected['do_chinh_xac']:.1f}% ACCURACY</span>
                </div>
                <div style="margin-top: 5px; font-size: 0.8rem; color: #888; text-align: right;">
                    Model: <span style="color: #00c6ff;">{st.session_state.get('ncv_model_type', 'Default')}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PHẦN DÀNH RIÊNG CHO NGHIÊN CỨU VIÊN KHI DÙNG HEAVY MODEL
    if st.session_state.get('ncv_model_type') == "MediaPipe Heavy" and user_role == "Nghiên cứu viên":
        with st.expander("🔬 DỮ LIỆU TỌA ĐỘ CHI TIẾT (RESEARCH ONLY)", expanded=False):
            st.markdown("#### 📜 Trích xuất tọa độ 33 khớp xương (Keypoints)")
            st.info("💡 Đây là dữ liệu thô phục vụ việc đối soát và huấn luyện mô hình (Dành cho bản Heavy).")
            # Hiển thị 20 dòng đầu của DF để xem cấu trúc keypoints
            st.dataframe(df.head(20), use_container_width=True)
            st.download_button(
                "📥 Tải xuống toàn bộ tọa độ (CSV)",
                df.to_csv(index=False).encode('utf-8'),
                "raw_keypoints_heavy.csv",
                "text/csv",
                key=f"dl_heavy_csv_{key_suffix}"
            )

            st.markdown("---")
            st.markdown("#### 🤖 Huấn luyện model học máy từ dữ liệu khung xương")
            if POSE_CLASSIFIER_IMPORT_ERROR:
                st.warning(f"Không thể nạp pose classifier utils: {POSE_CLASSIFIER_IMPORT_ERROR}")
            elif train_pose_classifier and get_pose_classifier_status:
                classifier_state = get_pose_classifier_status(DB_DIR)
                if classifier_state.get("ready"):
                    st.success(f"Đã có model ML: `{classifier_state.get('model_path')}`")
                else:
                    st.info("Chưa có model ML. Hãy train từ các file CSV đã trích xuất trong `processed_results`.")

                train_col, apply_col = st.columns(2)
                with train_col:
                    if st.button("🧠 TRAIN / CẬP NHẬT MODEL", key=f"btn_train_pose_classifier_{key_suffix}", use_container_width=True):
                        try:
                            require_role(RESEARCHER_ROLE, ADMIN_ROLE, action="train_pose_classifier", target="pose_classifier.pkl")
                            with st.spinner("Đang huấn luyện RandomForest từ dữ liệu keypoints..."):
                                train_result = train_pose_classifier(PROCESSED_DIR, DB_DIR)
                            if train_result.get("success"):
                                for artifact_path in [train_result.get("model_path"), train_result.get("model_checksum_path"), train_result.get("features_path")]:
                                    if artifact_path and os.path.exists(artifact_path):
                                        push_file_to_hf_async(artifact_path)
                                st.success(
                                    f"Train xong: {train_result.get('samples', 0)} mẫu, "
                                    f"test accuracy {train_result.get('accuracy', 0):.2f}%."
                                )
                                st.json({
                                    "valid_files": train_result.get("valid_files"),
                                    "label_distribution": train_result.get("label_distribution"),
                                    "model_path": train_result.get("model_path"),
                                })
                            else:
                                st.error(train_result.get("message", "Train model thất bại."))
                                if train_result.get("skipped_files"):
                                    st.json({"skipped_files": train_result.get("skipped_files")[:10]})
                        except PermissionError as exc:
                            st.error(str(exc))

                with apply_col:
                    dry_run_ml = st.checkbox(
                        "Dry-run (chỉ xem file sẽ đọc/ghi)",
                        value=True,
                        key=f"chk_apply_pose_classifier_dry_run_{key_suffix}",
                    )
                    if st.button("📌 ÁP DỤNG ML CHO VIDEO ĐÃ PHÂN TÍCH", key=f"btn_apply_pose_classifier_{key_suffix}", use_container_width=True):
                        try:
                            require_role(RESEARCHER_ROLE, ADMIN_ROLE, action="apply_pose_classifier", target="video_metadata")
                            with st.spinner("Đang dự đoán lại dung_ml và cập nhật ml_accuracy..."):
                                apply_result = reprocess_videos_with_classifier(
                                    VIDEOS_FILE,
                                    EVALUATIONS_FILE,
                                    processed_dir=PROCESSED_DIR,
                                    db_dir=DB_DIR,
                                    data_dir=DATA_DIR,
                                    phase_bounds_fn=segment_frames,
                                    dry_run=dry_run_ml,
                                )
                            if apply_result.get("success"):
                                if dry_run_ml:
                                    st.info(
                                        f"Dry-run xong cho {apply_result.get('updated', 0)} video. "
                                        "Chưa ghi CSV/JSON."
                                    )
                                    st.json({
                                        "would_read": apply_result.get("would_read", [])[:20],
                                        "would_write": apply_result.get("would_write", [])[:20],
                                    })
                                else:
                                    push_file_to_hf_async(VIDEOS_FILE)
                                    push_file_to_hf_async(EVALUATIONS_FILE)
                                    st.success(
                                        f"Đã cập nhật ML cho {apply_result.get('updated', 0)} video "
                                        f"(CSV + JSON frame + nhãn REF/ML trên ảnh JPG)."
                                    )
                                st.dataframe(pd.DataFrame(apply_result.get("results", [])).head(20), use_container_width=True)
                                if not dry_run_ml:
                                    st.rerun()
                            else:
                                st.error(apply_result.get("message", "Chưa thể áp dụng model ML."))
                        except PermissionError as exc:
                            st.error(str(exc))

    # 2. HÀNG THỐNG KÊ TỔNG QUAN (4 THẺ)
    st.markdown(f"### 📈 THỐNG KÊ TỔNG QUAN ({giai_doan_label})")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="height: 120px;">
            <div class="metric-value" style="font-size: 1.8rem;">{tk_selected['do_chinh_xac']:.1f}%</div>
            <div class="metric-label">🎯 Độ chính xác tổng thể</div>
            <div style="color: #666; font-size: 0.75rem;">{tk_selected['frame_dung']}/{tk_selected['tong_frame_hop_le']} frame đúng</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card" style="height: 120px;">
            <div class="metric-value" style="font-size: 1.8rem; color: #00CED1;">{tk_selected.get('ty_le_vai_dung', 0):.1f}%</div>
            <div class="metric-label">🦾 Tỉ lệ đúng góc vai</div>
            <div style="color: #666; font-size: 0.75rem;">Chuẩn: Video YouTube mẫu</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card" style="height: 120px;">
            <div class="metric-value" style="font-size: 1.8rem; color: #FF6B6B;">{tk_selected.get('ty_le_khuyu_dung', 0):.1f}%</div>
            <div class="metric-label">💪 Tỉ lệ đúng góc khuỷu</div>
            <div style="color: #666; font-size: 0.75rem;">Chuẩn: Video YouTube mẫu</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style="height: 120px;">
            <div class="metric-value" style="font-size: 1.8rem; color: #ffd700;">{tk_selected['tb_goc_vai']:.1f}°</div>
            <div class="metric-label">📐 Góc vai trung bình</div>
            <div style="color: #666; font-size: 0.75rem;">Min: {tk_selected['min_goc_vai']:.0f}° | Max: {tk_selected['max_goc_vai']:.0f}°</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. HỆ THỐNG TAB NỘI BỘ (SẮP XẾP LẠI KHOA HỌC)
    tab_list = ["🏠 TỔNG QUAN", "📈 BIỂU ĐỒ KHỚP"]
    # Heavy + Full: có biểu đồ Biên độ ROM. Lite: bỏ qua cho gọn nhẹ.
    if "Lite" not in model_type:
        tab_list += ["📦 BIÊN ĐỘ ROM"]
    tab_list += ["🩺 NHẬN ĐỊNH LÂM SÀNG"]
    # Chỉ số nghiên cứu chuyên sâu: chỉ dành riêng cho bản Heavy.
    if "Heavy" in model_type:
        tab_list += ["🔬 CHỈ SỐ NGHIÊN CỨU"]
    tab_list += ["📁 XUẤT BÁO CÁO"]

    inner_tabs = st.tabs(tab_list)
    t_map = {name: inner_tabs[i] for i, name in enumerate(tab_list)}

    # Khởi tạo các biểu đồ dùng chung (Tính toán một lần để tối ưu hiệu năng)
    fig_pie = ve_bieu_do_tron_thong_ke(tk_selected)
    fig_vai = ve_bieu_do_goc_vai(df, bt, sai_so_override=sai_so_selected)
    fig_khuyu = ve_bieu_do_goc_khuyu(df, bt, sai_so_override=sai_so_selected)
    fig_hist = ve_bieu_do_histogram(df, bt)
    fig_box_vai, fig_box_khuyu = ve_bieu_do_boxplot_phan_loai(df)
    fig_radar = ve_bieu_do_radar(tk_selected)

    # === TAB 1: TỔNG QUAN ===
    if "🏠 TỔNG QUAN" in t_map:
        with t_map["🏠 TỔNG QUAN"]:
            col_pie, col_metrics = st.columns([1, 1])
            with col_pie:
                st.plotly_chart(fig_pie, use_container_width=True, theme=None, key=f"pie_chart_fin_{key_suffix}")
                st.caption(f"ℹ️ Phân bổ chất lượng thực hiện ở {giai_doan_label}.")

            with col_metrics:
                st.markdown(f"#### 📑 CHỈ SỐ HIỆU SUẤT ({giai_doan_label})")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{tk_selected['frame_dung']}</div>
                        <div class="metric-label">✅ Frames Đúng (Pass)</div>
                    </div>
                    <div class="metric-card" style="margin-top: 15px;">
                        <div class="metric-value" style="color: #FFA500;">{tk_selected['frame_gan_dung']}</div>
                        <div class="metric-label">⚠️ Frames Gần Đúng</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                    <div class="metric-value" style="color: #FF4444;">{tk_selected['frame_sai']}</div>
                    <div class="metric-label">❌ Frames Sai (Fail)</div>
                </div>
                <div class="metric-card" style="margin-top: 15px;">
                    <div class="metric-value" style="color: #ffd700;">{tk_selected['do_chinh_xac']:.1f}%</div>
                    <div class="metric-label">🎯 Hiệu suất tổng thể</div>
                </div>
                """, unsafe_allow_html=True)

            if user_role == "Nghiên cứu viên":
                st.markdown("---")
                if st.button("📤 XÁC NHẬN & GỬI BÁO CÁO TỔNG HỢP", key=f"btn_send_final_{key_suffix}", width="stretch", type="primary"):
                    if gui_bao_cao_tong_hop_3_giai_doan():
                        v_meta = st.session_state.get('current_eval_video') or {}
                        st.success(f"✅ Đã gửi báo cáo tổng hợp 3 giai đoạn cho BN {patient_display_label(v_meta)}!")
                        st.balloons()

    # === TAB 2: BIỂU ĐỒ KHỚP ===
    if "📈 BIỂU ĐỒ KHỚP" in t_map:
        with t_map["📈 BIỂU ĐỒ KHỚP"]:
            st.markdown(f"#### 📐 BIÊN ĐỘ VẬN ĐỘNG ({giai_doan_label})")
            st.plotly_chart(fig_vai, use_container_width=True, theme=None, key=f"vai_ch_ncv_{key_suffix}")
            st.plotly_chart(fig_khuyu, use_container_width=True, theme=None, key=f"khuyu_ch_ncv_{key_suffix}")
            st.plotly_chart(fig_hist, use_container_width=True, theme=None, key=f"hist_ch_ncv_{key_suffix}")
            st.info("ℹ️ Biểu đồ thể hiện sự thay đổi góc khớp theo thời gian thực (frames).")

    # === TAB 3: BIÊN ĐỘ ROM ===
    if "📦 BIÊN ĐỘ ROM" in t_map:
        with t_map["📦 BIÊN ĐỘ ROM"]:
            st.markdown("### 📦 PHÂN TÍCH BIÊN ĐỘ VẬN ĐỘNG (ROM)")
            col_rom1, col_rom2 = st.columns(2)
            with col_rom1:
                st.plotly_chart(fig_box_vai, use_container_width=True, theme=None, key=f"box_vai_ncv_{key_suffix}")
            with col_rom2:
                st.plotly_chart(fig_box_khuyu, use_container_width=True, theme=None, key=f"box_khu_ncv_{key_suffix}")
            st.info("💡 Biểu đồ Boxplot so sánh sự biến thiên và ổn định của góc khớp.")

    # === TAB 4: NHẬN ĐỊNH LÂM SÀNG ===
    if "🩺 NHẬN ĐỊNH LÂM SÀNG" in t_map:
        with t_map["🩺 NHẬN ĐỊNH LÂM SÀNG"]:
            st.markdown(f"### 🩺 NHẬN ĐỊNH CHUYÊN MÔN ({giai_doan_label})")
            insights = lay_nhan_dinh_lam_sang(tk_selected['tb_goc_vai'], tk_selected['tb_goc_khuyu'], bt,
                                             v_chuan=tk_selected.get('tb_vai_chuan'),
                                             k_chuan=tk_selected.get('tb_khuyu_chuan'),
                                             sai_so_override=sai_so_selected)
            if insights:
                for item in insights:
                    insight_type = safe_html(item.get('loai', ''), max_length=120)
                    insight_metric = safe_html(item.get('chi_so', ''), max_length=80)
                    insight_warning = safe_html(item.get('canh_warning', item.get('canh_bao', '')), max_length=800)
                    insight_advice = safe_html(item.get('loi_khuyen', ''), max_length=800)
                    st.markdown(f"""
                    <div style="background: rgba(255,165,0,0.1); border-left: 5px solid #FFA500; padding: 1rem; border-radius: 8px; margin-bottom: 10px;">
                        <h4 style="color: #FFA500; margin-top: 0;">⚠️ {insight_type} ({insight_metric})</h4>
                        <p style="color: #fff; margin-bottom: 5px;"><strong>🔴 Cảnh báo:</strong> {insight_warning}</p>
                        <p style="color: #00CED1; margin-bottom: 0;"><strong>💡 Lời khuyên:</strong> {insight_advice}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success(f"✅ **NHẬN ĐỊNH ({giai_doan_label}):** Biên độ vận động của bệnh nhân nằm trong giới hạn an toàn.")

            # THÊM PHẦN NHẬN XÉT CỦA BÁC SĨ (GROUND TRUTH) CHO NCV
            v_meta = st.session_state.get('current_eval_video')
            if v_meta:
                _, doc_eval = _lay_danh_gia_cho_video(v_meta)
                if doc_eval:
                    st.markdown("---")
                    st.markdown("#### 🩺 PHẢN HỒI TỪ CHUYÊN GIA PHCN (GROUND TRUTH)")
                    doc_name = "Chuyên gia PHCN" if user_role == "Nghiên cứu viên" else doc_eval.get('doctor_name', 'Chuyên gia')
                    doc_name_html = safe_html(doc_name, max_length=120)
                    doc_result_html = safe_html(doc_eval.get('doctor_result', 'N/A'), max_length=80)
                    doc_ncv_text = "Đã ẩn trong chế độ NCV." if user_role == "Nghiên cứu viên" else doc_eval.get('comments_ncv', 'Không có ghi chú riêng.')
                    doc_comments_text = "Đã ẩn trong chế độ NCV." if user_role == "Nghiên cứu viên" else doc_eval.get('comments', '')
                    doc_ncv_html = safe_html(doc_ncv_text, max_length=1000)
                    doc_comments_html = safe_html(doc_comments_text, max_length=1000)
                    st.markdown(f"""
                    <div style="background: rgba(0, 198, 255, 0.05); border: 1px solid #00c6ff; padding: 1.2rem; border-radius: 12px; border-left: 6px solid #00c6ff;">
                        <p style="color: #00c6ff; font-weight: bold; margin-bottom: 5px;">👤 Bác sĩ: {doc_name_html}</p>
                        <p style="margin-bottom: 5px;"><b>📊 Đánh giá lâm sàng:</b> {doc_result_html}</p>
                        <p style="margin-bottom: 5px;"><b>💬 Nhận xét cho NCV:</b> <span style="color: #ffd700;">{doc_ncv_html}</span></p>
                        <p style="margin-bottom: 0;"><b>📝 Lời khuyên cho BN:</b> {doc_comments_html}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"#### 🤖 PHÂN TÍCH TỪ MÔ HÌNH HỌC MÁY ({giai_doan_label})")
            stab = 100 - (tk_selected.get('std_goc_vai', 0) + tk_selected.get('std_goc_khuyu', 0))
            ai_c1, ai_c2 = st.columns([1, 2])
            with ai_c1:
                st.metric("🎯 F1-Score", f"{tk_selected.get('f1_score', 0):.2f}")
                st.metric("📉 Độ mượt", f"{max(0, stab):.1f}/100")
            with ai_c2:
                st.info(f"**ICC:** {tk_selected.get('icc', 0):.2f} | **MAE:** {tk_selected.get('mae_tong', 0):.1f}°\n\n{'✅ Đạt chuẩn NCKH' if tk_selected.get('icc', 0) > 0.75 else '⚠️ Cần kiểm tra tín hiệu'}")

    # === TAB 5: CHỈ SỐ NGHIÊN CỨU ===
    if "🔬 CHỈ SỐ NGHIÊN CỨU" in t_map:
        with t_map["🔬 CHỈ SỐ NGHIÊN CỨU"]:
            st.markdown("### 🔬 ĐÁNH GIÁ CHỈ SỐ NGHIÊN CỨU")
            st.plotly_chart(fig_radar, use_container_width=True, theme=None, key=f"radar_ch_ncv_{key_suffix}")

            st.markdown("#### 📊 BẢNG SO SÁNH CHỈ SỐ GIỮA 3 GIAI ĐOẠN (RESEARCH METRICS)")

            rmse_val1 = metrics_g1.get('mae_tong', 0) * 1.25
            rmse_val2 = metrics_g2.get('mae_tong', 0) * 1.25
            rmse_val3 = metrics_g3.get('mae_tong', 0) * 1.25

            if is_gay_ex:
                rmse_val = tk_selected.get('mae_tong', 0) * 1.25
                st.markdown(f"""
                <div class="research-table-container">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
                        <thead style="background: rgba(56, 189, 248, 0.1);">
                            <tr style="border-bottom: 2px solid #38bdf8; text-align: left;">
                                <th style="padding: 12px;">Chỉ số nghiên cứu</th>
                                <th style="padding: 12px; text-align: center;">Ký hiệu</th>
                                <th style="padding: 12px; text-align: center;">Tổng quan (Sai số {sai_so_selected}°)</th>
                                <th style="padding: 12px;">Phân loại / Chuyên môn</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Độ chính xác hệ thống</td>
                                <td style="padding: 10px; text-align: center;"><b>ACC</b></td>
                                <td style="padding: 10px; text-align: center; color: #10b981; font-weight: bold;">{tk_selected['do_chinh_xac']:.1f}%</td>
                                <td style="padding: 10px;">Được đối soát theo tư thế tương đương với video YouTube mẫu</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Sai số tuyệt đối trung bình</td>
                                <td style="padding: 10px; text-align: center;"><b>MAE</b></td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{tk_selected.get('mae_tong', 0):.1f}°</td>
                                <td style="padding: 10px;">{'✅ Tốt' if tk_selected.get('mae_tong', 0) < 5 else '⚠️ Sai số góc cao'}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Sai số bình phương trung bình</td>
                                <td style="padding: 10px; text-align: center;"><b>RMSE</b></td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{rmse_val:.1f}°</td>
                                <td style="padding: 10px;">Ước lượng sai số bình phương trung bình</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Hệ số tương quan nội lớp</td>
                                <td style="padding: 10px; text-align: center;"><b>ICC</b></td>
                                <td style="padding: 10px; text-align: center; color: #38bdf8;">{tk_selected.get('icc', 0):.2f}</td>
                                <td style="padding: 10px;">{'✅ Rất tốt' if tk_selected.get('icc', 0) >= 0.75 else '⚠️ Trung bình'}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Độ nhạy phân loại</td>
                                <td style="padding: 10px; text-align: center;"><b>Recall</b></td>
                                <td style="padding: 10px; text-align: center;">{tk_selected.get('recall', 0):.2f}</td>
                                <td style="padding: 10px;">Khả năng phát hiện đúng tư thế</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Độ đặc hiệu phân loại</td>
                                <td style="padding: 10px; text-align: center;"><b>Precision</b></td>
                                <td style="padding: 10px; text-align: center;">{tk_selected.get('precision', 0):.2f}</td>
                                <td style="padding: 10px;">Độ tin cậy cảnh báo sai tư thế</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Chỉ số cân bằng F1</td>
                                <td style="padding: 10px; text-align: center;"><b>F1-Score</b></td>
                                <td style="padding: 10px; text-align: center; color: #fbbf24;">{tk_selected.get('f1_score', 0):.2f}</td>
                                <td style="padding: 10px;">Hiệu suất AI tổng hợp chéo</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Số lần tập đúng (Pass)</td>
                                <td style="padding: 10px; text-align: center;"><b>Pass</b></td>
                                <td style="padding: 10px; text-align: center;">{tk_selected['frame_dung']}</td>
                                <td style="padding: 10px;">Số lượng khung hình đạt chuẩn theo giai đoạn</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="research-table-container">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
                        <thead style="background: rgba(56, 189, 248, 0.1);">
                            <tr style="border-bottom: 2px solid #38bdf8; text-align: left;">
                                <th style="padding: 12px;">Chỉ số nghiên cứu</th>
                                <th style="padding: 12px; text-align: center;">Ký hiệu</th>
                                <th style="padding: 12px; text-align: center;">Giai đoạn 1 (Sai số ±{PHASE_ERROR['g1']}°)</th>
                                <th style="padding: 12px; text-align: center;">Giai đoạn 2 (Sai số ±{PHASE_ERROR['g2']}°)</th>
                                <th style="padding: 12px; text-align: center;">Giai đoạn 3 (Sai số ±{PHASE_ERROR['g3']}°)</th>
                                <th style="padding: 12px;">Phân loại / Chuyên môn</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Độ chính xác hệ thống</td>
                                <td style="padding: 10px; text-align: center;"><b>ACC</b></td>
                                <td style="padding: 10px; text-align: center; color: #10b981; font-weight: bold;">{metrics_g1['do_chinh_xac']:.1f}%</td>
                                <td style="padding: 10px; text-align: center; color: #10b981; font-weight: bold;">{metrics_g2['do_chinh_xac']:.1f}%</td>
                                <td style="padding: 10px; text-align: center; color: #10b981; font-weight: bold;">{metrics_g3['do_chinh_xac']:.1f}%</td>
                                <td style="padding: 10px;">Được đối soát theo từng giây với video YouTube mẫu</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Sai số tuyệt đối trung bình</td>
                                <td style="padding: 10px; text-align: center;"><b>MAE</b></td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{metrics_g1.get('mae_tong', 0):.1f}°</td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{metrics_g2.get('mae_tong', 0):.1f}°</td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{metrics_g3.get('mae_tong', 0):.1f}°</td>
                                <td style="padding: 10px;">{'✅ Tốt' if metrics_g2.get('mae_tong', 0) < 5 else '⚠️ Sai số góc cao'}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Sai số bình phương trung bình</td>
                                <td style="padding: 10px; text-align: center;"><b>RMSE</b></td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{rmse_val1:.1f}°</td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{rmse_val2:.1f}°</td>
                                <td style="padding: 10px; text-align: center; color: #f43f5e;">{rmse_val3:.1f}°</td>
                                <td style="padding: 10px;">Ước lượng sai số bình phương trung bình</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Hệ số tương quan nội lớp</td>
                                <td style="padding: 10px; text-align: center;"><b>ICC</b></td>
                                <td style="padding: 10px; text-align: center; color: #38bdf8;">{metrics_g1.get('icc', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center; color: #38bdf8;">{metrics_g2.get('icc', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center; color: #38bdf8;">{metrics_g3.get('icc', 0):.2f}</td>
                                <td style="padding: 10px;">{'✅ Rất tốt' if metrics_g2.get('icc', 0) >= 0.75 else '⚠️ Trung bình'}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Độ nhạy phân loại</td>
                                <td style="padding: 10px; text-align: center;"><b>Recall</b></td>
                                <td style="padding: 10px; text-align: center;">{metrics_g1.get('recall', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center;">{metrics_g2.get('recall', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center;">{metrics_g3.get('recall', 0):.2f}</td>
                                <td style="padding: 10px;">Khả năng phát hiện đúng tư thế</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Độ đặc hiệu phân loại</td>
                                <td style="padding: 10px; text-align: center;"><b>Precision</b></td>
                                <td style="padding: 10px; text-align: center;">{metrics_g1.get('precision', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center;">{metrics_g2.get('precision', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center;">{metrics_g3.get('precision', 0):.2f}</td>
                                <td style="padding: 10px;">Độ tin cậy cảnh báo sai tư thế</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Chỉ số cân bằng F1</td>
                                <td style="padding: 10px; text-align: center;"><b>F1-Score</b></td>
                                <td style="padding: 10px; text-align: center; color: #fbbf24;">{metrics_g1.get('f1_score', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center; color: #fbbf24;">{metrics_g2.get('f1_score', 0):.2f}</td>
                                <td style="padding: 10px; text-align: center; color: #fbbf24;">{metrics_g3.get('f1_score', 0):.2f}</td>
                                <td style="padding: 10px;">Hiệu suất AI tổng hợp chéo</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                <td style="padding: 10px;">Số lần tập đúng (Pass)</td>
                                <td style="padding: 10px; text-align: center;"><b>Pass</b></td>
                                <td style="padding: 10px; text-align: center;">{metrics_g1['frame_dung']}</td>
                                <td style="padding: 10px; text-align: center;">{metrics_g2['frame_dung']}</td>
                                <td style="padding: 10px; text-align: center;">{metrics_g3['frame_dung']}</td>
                                <td style="padding: 10px;">Số lượng khung hình đạt chuẩn theo giai đoạn</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)

    # === TAB 6: XUẤT BÁO CÁO (CONSOLIDATED) ===
    if "📁 XUẤT BÁO CÁO" in t_map:
        with t_map["📁 XUẤT BÁO CÁO"]:
            st.markdown("### 📁 QUẢN LÝ DỮ LIỆU & BÁO CÁO")

            # Gộp các nút tải xuống vào các nhóm logic
            exp_img = st.expander("🖼️ TẢI XUỐNG BIỂU ĐỒ (Dạng Ảnh PNG)", expanded=True)
            with exp_img:
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    try: st.download_button("📊 Biểu đồ Tròn (Tổng quan)", fig_pie.to_image(format="png"), "pie_summary.png", "image/png", width="stretch", key=f"dl_f1_{key_suffix}")
                    except: pass
                    try: st.download_button("📈 Biểu đồ Vai (ROM)", fig_vai.to_image(format="png"), "shoulder_rom.png", "image/png", width="stretch", key=f"dl_f2_{key_suffix}")
                    except: pass
                    try: st.download_button("📉 Biểu đồ Khuỷu (ROM)", fig_khuyu.to_image(format="png"), "elbow_rom.png", "image/png", width="stretch", key=f"dl_f3_{key_suffix}")
                    except: pass
                with img_col2:
                    try: st.download_button("📦 Boxplot Vai (Stability)", fig_box_vai.to_image(format="png"), "boxplot_shoulder.png", "image/png", width="stretch", key=f"dl_f4a_{key_suffix}")
                    except: pass
                    try: st.download_button("📦 Boxplot Khuỷu (Stability)", fig_box_khuyu.to_image(format="png"), "boxplot_elbow.png", "image/png", width="stretch", key=f"dl_f4b_{key_suffix}")
                    except: pass
                    try: st.download_button("🕸️ Biểu đồ Radar (Overall)", fig_radar.to_image(format="png"), "radar_performance.png", "image/png", width="stretch", key=f"dl_f5_{key_suffix}")
                    except: pass

            exp_data = st.expander("📊 TẢI XUỐNG DỮ LIỆU THÔ (CSV/ZIP)", expanded=True)
            with exp_data:
                data_col1, data_col2 = st.columns(2)
                with data_col1:
                    if df is not None:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📄 Tọa độ góc khớp (CSV)", csv_data, "angle_data.csv", "text/csv", width="stretch", key=f"dl_f6_{key_suffix}")
                with data_col2:
                    # LAZY ZIP: Chỉ tạo ZIP khi người dùng yêu cầu để tránh OOM
                    frames_zip_path = st.session_state.get('frames_zip')
                    frame_paths_list = st.session_state.get('all_frames_paths', [])
                    if frames_zip_path and os.path.exists(frames_zip_path):
                        with open(frames_zip_path, "rb") as fz:
                            st.download_button("📦 Toàn bộ khung hình (ZIP)", fz, "all_frames.zip", "application/zip", width="stretch", key=f"dl_f7_{key_suffix}")
                    elif frame_paths_list:
                        if st.button("📦 Chuẩn bị file ZIP tải ảnh", width="stretch", key=f"btn_prep_zip_{key_suffix}"):
                            with st.spinner("🔄 Đang nén khung hình..."):
                                new_zip = create_zip_of_frames(frame_paths_list)
                                if new_zip:
                                    st.session_state.frames_zip = new_zip
                                    st.rerun()
                                else:
                                    st.error("❌ Lỗi tạo file ZIP. Thử lại sau.")

def render_analysis_tab(deps, key_suffix="", stats_ext=None, df_ext=None, exercise_ext=None):
    """Hiển thị tab biểu đồ — polling tiến trình do panel bên phải đảm nhiệm."""
    try:
        _hien_thi_tab_phan_tich_noi_dung(deps, key_suffix, stats_ext, df_ext, exercise_ext)
    except Exception as chart_err:
        st.error(f"💥 Lỗi hiển thị biểu đồ: {chart_err}")
        import traceback
        st.code(traceback.format_exc())
        thong_bao_loi_tai_hf()
