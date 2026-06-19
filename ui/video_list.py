"""Video list Streamlit UI extracted from app.py."""

from __future__ import annotations

import os
from datetime import datetime, timedelta


def _bind_deps(deps):
    """Bind app services expected by the moved legacy UI body."""
    global st, _evals_dedup_cached, _mtimes_video_eval, load_danh_sach_video_nghien_cuu
    global HF_TOKEN, HF_DATASET_ID, _dong_bo_video_list_nen, tai_lai_video_list_tu_cloud
    global _normalize_video_key, _parse_vn_datetime, _tom_tat_benh_nhan_tu_video, safe_html
    global _lay_eval_moi_nhat_theo_bai_tap, _la_ban_ghi_video_mo_co, _lay_thoi_gian_phan_tich_on_dinh
    global video_can_khoi_dong_phan_tich, liet_ke_jobs_dang_chay, MAX_CONCURRENT_ANALYSIS, bat_dau_phan_tich_hang_loat
    global _lay_duong_dan_video_tho, get_final_h264_path, find_ready_local_video
    global _dam_bao_video_san_sang_play, _is_scratch_video_path, render_video
    global _lay_trang_thai_video_danh_sach, _lay_thoi_gian_upload_video, patient_display_label, PHASE_ERROR
    global _lay_do_chinh_xac_hien_thi, _format_vn_time, get_video_codec, ffprobe_video_duration_text
    global PROCESSED_DIR, get_clean_rel_path, _lam_moi_ban_ghi_video_tu_db, video_dang_phan_tich
    global _nap_bieu_do_nhanh_tu_cloud, _slot_video_phan_tich, _xoa_session_phan_tich, delete_video_callback

    st = deps.st
    _evals_dedup_cached = deps.evals_dedup_cached
    _mtimes_video_eval = deps.mtimes_video_eval
    load_danh_sach_video_nghien_cuu = deps.load_research_videos
    HF_TOKEN = deps.HF_TOKEN
    HF_DATASET_ID = deps.HF_DATASET_ID
    _dong_bo_video_list_nen = deps.sync_video_list_background
    tai_lai_video_list_tu_cloud = deps.reload_video_list_from_cloud
    _normalize_video_key = deps.normalize_video_key
    _parse_vn_datetime = deps.parse_vn_datetime
    _tom_tat_benh_nhan_tu_video = deps.patient_summary_from_video
    safe_html = deps.safe_html
    _lay_eval_moi_nhat_theo_bai_tap = deps.latest_eval_by_exercise
    _la_ban_ghi_video_mo_co = deps.is_placeholder_video_record
    _lay_thoi_gian_phan_tich_on_dinh = deps.stable_analysis_time
    video_can_khoi_dong_phan_tich = deps.can_start_analysis
    liet_ke_jobs_dang_chay = deps.list_running_jobs
    MAX_CONCURRENT_ANALYSIS = deps.MAX_CONCURRENT_ANALYSIS
    bat_dau_phan_tich_hang_loat = deps.start_batch_analysis
    _lay_duong_dan_video_tho = deps.raw_video_path
    get_final_h264_path = deps.get_final_h264_path
    find_ready_local_video = deps.find_ready_local_video
    _dam_bao_video_san_sang_play = deps.ensure_video_ready_to_play
    _is_scratch_video_path = deps.is_scratch_video_path
    render_video = deps.render_video
    _lay_trang_thai_video_danh_sach = deps.video_list_status
    _lay_thoi_gian_upload_video = deps.upload_time_for_video
    patient_display_label = deps.patient_display_label
    PHASE_ERROR = deps.PHASE_ERROR
    _lay_do_chinh_xac_hien_thi = deps.display_accuracy
    _format_vn_time = deps.format_vn_time
    get_video_codec = deps.get_video_codec
    ffprobe_video_duration_text = deps.ffprobe_video_duration_text
    PROCESSED_DIR = deps.PROCESSED_DIR
    get_clean_rel_path = deps.get_clean_rel_path
    _lam_moi_ban_ghi_video_tu_db = deps.refresh_video_record_from_db
    video_dang_phan_tich = deps.video_is_analyzing
    _nap_bieu_do_nhanh_tu_cloud = deps.load_chart_fast_from_cloud
    _slot_video_phan_tich = deps.analysis_slot_key
    _xoa_session_phan_tich = deps.clear_analysis_session
    delete_video_callback = deps.delete_video_callback


def reset_vid_list_page():
    st.session_state.vid_list_page = 0


def _chuan_hoa_widget_loc_video(key, options, default):
    """Sau F5 session/widget có thể lệch — xóa key nếu giá trị không còn trong options."""
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    if key not in st.session_state:
        st.session_state[key] = default

def render_video_list_fragment(deps, user_role, video_list_preloaded=None):
    """Danh sách video/BN — tự refresh khi đang đồng bộ Cloud sau F5."""
    _bind_deps(deps)
    # Pre-check để set interval=5s ngay từ đầu nếu list trống — tránh tạo fragment với interval=None
    # rồi không tự refresh khi _bg_video_list_sync được set bên trong fragment body.
    _pre = video_list_preloaded if video_list_preloaded is not None else load_danh_sach_video_nghien_cuu()
    _syncing = st.session_state.get("_bg_video_list_sync") or (not _pre and bool(HF_TOKEN and HF_DATASET_ID))
    interval = timedelta(seconds=5) if _syncing else None

    def _body():
        _noi_dung_danh_sach_video_fragment(
            user_role,
            video_list_preloaded=None if st.session_state.get("_bg_video_list_sync") else _pre,
        )

    _body()


def _noi_dung_danh_sach_video_fragment(user_role, video_list_preloaded=None):
    evals_db = _evals_dedup_cached(_mtimes_video_eval()[1])
    video_list = video_list_preloaded if video_list_preloaded is not None else load_danh_sach_video_nghien_cuu()

    # Mở link bookmark / F5: đồng bộ Cloud nền nếu danh sách trống (không chặn UI)
    if not video_list and (HF_TOKEN and HF_DATASET_ID):
        if not st.session_state.get("_bg_video_list_sync"):
            st.session_state._bg_video_list_sync = True
            _dong_bo_video_list_nen(force=True)
        st.caption("☁️ Đang đồng bộ danh sách video từ Cloud — vui lòng chờ vài giây...")
        video_list = load_danh_sach_video_nghien_cuu()
        if video_list:
            st.session_state.pop("_bg_video_list_sync", None)

    if st.session_state.get('delete_success'):
        st.toast(f"🗑️ {st.session_state.delete_success}", icon="✅")
        st.session_state.delete_success = None

    if not video_list:
        # Khi đang sync Cloud thì không hiển thị "📭 chưa có video" — tránh hiện cả 2 message cùng lúc
        if not st.session_state.get("_bg_video_list_sync"):
            st.info("📭 Hiện chưa có video nào được gửi đến.")
            if st.button("🔄 Tải lại danh sách từ Cloud / khôi phục", key="btn_reload_video_list", use_container_width=True):
                with st.spinner("Đang tải danh sách từ Cloud..."):
                    tai_lai_video_list_tu_cloud()
                st.rerun()
    else:
        st.caption(
            f"📋 Hiển thị **{len(video_list)} video nghiên cứu** "
            "(4 bệnh nhân × 2 bài tập) đã có đánh giá bác sĩ/KTV."
        )

        # --- Tối ưu: xây dựng lookup dict O(1) thay vì O(n) linear scan trong vòng lặp ---
        ai_eval_lookup = {}
        ai_eval_by_exercise = {}
        doc_eval_lookup = {}
        doc_eval_by_exercise = {}
        for e in evals_db:
            key = _normalize_video_key(e.get('patient_username'), e.get('video_name'), e.get('exercise'))
            pu, ex = e.get('patient_username'), e.get('exercise')
            if e.get('doctor_username') == "AI_Researcher":
                ai_eval_lookup[key] = e
                prev = ai_eval_by_exercise.get((pu, ex))
                if not prev or (_parse_vn_datetime(e.get('time')) or datetime.min) >= (_parse_vn_datetime(prev.get('time')) or datetime.min):
                    ai_eval_by_exercise[(pu, ex)] = e
            else:
                doc_eval_lookup[key] = e
                prev = doc_eval_by_exercise.get((pu, ex))
                if not prev or (_parse_vn_datetime(e.get('time')) or datetime.min) >= (_parse_vn_datetime(prev.get('time')) or datetime.min):
                    doc_eval_by_exercise[(pu, ex)] = e

        patient_summary = _tom_tat_benh_nhan_tu_video(video_list, ai_eval_lookup, ai_eval_by_exercise)
        if patient_summary:
            st.markdown("##### 👥 DANH SÁCH BỆNH NHÂN — THỜI GIAN PHÂN TÍCH GẦN NHẤT")
            for row in patient_summary:
                last_t = safe_html(row.get("last_analysis") or "Chưa phân tích", max_length=80)
                full_name_html = safe_html(row.get("full_name"), max_length=120)
                video_count = int(row.get("video_count") or 0)
                st.markdown(
                    f"<div style='background:rgba(0,198,255,0.06);border:1px solid rgba(0,198,255,0.2);"
                    f"border-left:4px solid #00c6ff;border-radius:10px;padding:10px 14px;margin-bottom:8px;'>"
                    f"<b>👤 {full_name_html}</b> "
                    f"<span style='color:#888;font-size:0.85rem;'>({video_count} video)</span><br>"
                    f"<span style='color:#00c6ff;font-size:0.95rem;'>🕒 Phân tích gần nhất: <b>{last_t}</b></span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # --- BỘ LỌC DANH SÁCH VIDEO ---
        st.markdown("##### 🔍 BỘ LỌC DANH SÁCH")

        # Lấy danh sách bệnh nhân duy nhất có video
        patient_options = {}
        for v in video_list:
            u = v.get('username')
            fn = v.get('full_name') or u
            if u:
                patient_options[u] = f"👤 {fn} ({u})"

        sorted_patients = sorted(patient_options.items(), key=lambda item: item[1].lower())
        patient_list_opts = ["-- Tất cả bệnh nhân --"] + [item[1] for item in sorted_patients]
        patient_lookup = {item[1]: item[0] for item in sorted_patients}

        status_list_opts = ["-- Tất cả trạng thái --", "Đã đánh giá", "Đang chờ bác sĩ đánh giá"]
        _chuan_hoa_widget_loc_video("filter_video_patient", patient_list_opts, "-- Tất cả bệnh nhân --")
        _chuan_hoa_widget_loc_video("filter_video_status", status_list_opts, "-- Tất cả trạng thái --")
        if st.session_state.get("vid_list_page", 0) < 0:
            st.session_state.vid_list_page = 0

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_patient_opt = st.selectbox(
                "Lọc theo bệnh nhân:",
                patient_list_opts,
                key="filter_video_patient",
                on_change=reset_vid_list_page,
                label_visibility="collapsed"
            )
        with col_f2:
            selected_status_opt = st.selectbox(
                "Lọc theo trạng thái đánh giá:",
                status_list_opts,
                key="filter_video_status",
                on_change=reset_vid_list_page,
                label_visibility="collapsed"
            )

        def _video_co_danh_gia_bac_si(v):
            pu, ex = v.get("username"), v.get("exercise")
            ev_key = _normalize_video_key(pu, v.get("video_name"), ex)
            return (
                ev_key in doc_eval_lookup
                or (pu, ex) in doc_eval_by_exercise
                or _lay_eval_moi_nhat_theo_bai_tap(evals_db, pu, ex) is not None
            )

        # Tiến hành lọc danh sách video
        filtered_videos = []
        for v in video_list:
            if _la_ban_ghi_video_mo_co(v):
                continue
            if selected_patient_opt != "-- Tất cả bệnh nhân --":
                target_username = patient_lookup.get(selected_patient_opt)
                if not target_username or v.get("username") != target_username:
                    continue
            if selected_status_opt != "-- Tất cả trạng thái --":
                has_doc_eval = _video_co_danh_gia_bac_si(v)
                if selected_status_opt == "Đã đánh giá" and not has_doc_eval:
                    continue
                if selected_status_opt == "Đang chờ bác sĩ đánh giá" and has_doc_eval:
                    continue
            filtered_videos.append(v)

        def _sort_key_by_analysis(v_item):
            ev_k = _normalize_video_key(v_item.get("username"), v_item.get("video_name"), v_item.get("exercise"))
            ai_e = ai_eval_lookup.get(ev_k) or ai_eval_by_exercise.get((v_item.get("username"), v_item.get("exercise")))
            t_s = _lay_thoi_gian_phan_tich_on_dinh(v_item, ai_e)
            return _parse_vn_datetime(t_s) if t_s else datetime.min

        filtered_videos.sort(key=_sort_key_by_analysis, reverse=True)

        if (
            not filtered_videos
            and video_list
            and selected_patient_opt == "-- Tất cả bệnh nhân --"
            and selected_status_opt == "-- Tất cả trạng thái --"
        ):
            filtered_videos = [v for v in video_list if not _la_ban_ghi_video_mo_co(v)]

        if not filtered_videos:
            st.info("ℹ️ Không tìm thấy video nào khớp với điều kiện lọc.")
            if (
                video_list
                and selected_patient_opt == "-- Tất cả bệnh nhân --"
                and selected_status_opt == "-- Tất cả trạng thái --"
                and not st.session_state.get("_vid_filter_heal_rerun")
            ):
                for _fk in ("filter_video_patient", "filter_video_status", "vid_list_page"):
                    st.session_state.pop(_fk, None)
                st.session_state._vid_filter_heal_rerun = True
                st.rerun()
        else:
            st.session_state.pop("_vid_filter_heal_rerun", None)
            page_videos = []
            if user_role == "Nghiên cứu viên":
                pending_batch = [v for v in filtered_videos if video_can_khoi_dong_phan_tich(v, only_pending=True)]
                running_n = len(liet_ke_jobs_dang_chay())
                st.markdown("##### ⚡ Phân tích hàng loạt (chạy nền)")
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if pending_batch:
                        if st.button(
                            f"🚀 Phân tích TẤT CẢ chưa xong ({len(pending_batch)} video · {MAX_CONCURRENT_ANALYSIS} song song)",
                            key="btn_batch_analyze_pending",
                            type="primary",
                            use_container_width=True,
                        ):
                            n_start, n_skip = bat_dau_phan_tich_hang_loat(pending_batch, only_pending=True)
                            st.toast(f"Đã khởi chạy {n_start} video (bỏ qua {n_skip}). Đang chạy nền: {running_n + n_start}.", icon="🚀")
                            st.rerun()
                    else:
                        st.caption("✅ Không còn video chưa phân tích trong bộ lọc hiện tại.")
                with b_col2:
                    if st.button(
                        f"🔁 Chạy lại AI TẤT CẢ trong bộ lọc ({len(filtered_videos)} video · {MAX_CONCURRENT_ANALYSIS} song song)",
                        key="btn_batch_reanalyze_all",
                        use_container_width=True,
                    ):
                        n_start, n_skip = bat_dau_phan_tich_hang_loat(filtered_videos, only_pending=False, force_reanalyze=True)
                        st.toast(f"Đã xếp hàng chạy lại {n_start} video (bỏ qua {n_skip}).", icon="🔁")
                        st.rerun()
                if running_n or pending_batch:
                    st.caption(
                        f"Tối đa **{MAX_CONCURRENT_ANALYSIS}** video chạy cùng lúc; video tiếp theo tự xếp hàng. "
                        "Theo dõi tiến độ ở tab **🔬 PHÂN TÍCH & TRÍCH XUẤT DỮ LIỆU**."
                    )
                st.markdown("---")

            # --- Pagination: chỉ render 10 video/trang để tránh render quá nhiều expander ---
            PAGE_SIZE = 10
            total_videos = len(filtered_videos)
            total_pages = max(1, (total_videos + PAGE_SIZE - 1) // PAGE_SIZE)

            if 'vid_list_page' not in st.session_state:
                st.session_state.vid_list_page = 0
            # Đảm bảo trang hiện tại không vượt quá tổng số trang
            if st.session_state.vid_list_page >= total_pages:
                st.session_state.vid_list_page = total_pages - 1

            # Thanh điều hướng trang
            if total_pages > 1:
                pg_c1, pg_c2, pg_c3 = st.columns([1, 3, 1])
                with pg_c1:
                    if st.button("◀ Trang trước", disabled=(st.session_state.vid_list_page == 0), key="vid_pg_prev"):
                        st.session_state.vid_list_page -= 1
                        st.rerun()
                with pg_c2:
                    st.markdown(
                        f"<div style='text-align:center; padding:6px; color:#aaa;'>Trang {st.session_state.vid_list_page + 1} / {total_pages} "
                        f"({total_videos} video)</div>",
                        unsafe_allow_html=True
                    )
                with pg_c3:
                    if st.button("Trang sau ▶", disabled=(st.session_state.vid_list_page >= total_pages - 1), key="vid_pg_next"):
                        st.session_state.vid_list_page += 1
                        st.rerun()

            start_idx = st.session_state.vid_list_page * PAGE_SIZE
            page_videos = list(enumerate(filtered_videos))[start_idx: start_idx + PAGE_SIZE]
            if not page_videos and total_videos > 0:
                st.session_state.vid_list_page = 0
                page_videos = list(enumerate(filtered_videos))[0:PAGE_SIZE]

            st.caption(f"📌 Đang hiển thị **{len(page_videos)}** / **{total_videos}** video trong bộ lọc.")

            for idx, v in page_videos:
                col_list1, col_list2 = st.columns([12, 1])
                with col_list1:
                    processed_path = v.get('processed_path')
                    raw_path = v.get('video_path')
                    active_display_path = _lay_duong_dan_video_tho(v)
                    final_h264 = get_final_h264_path(raw_path) if raw_path else ""

                    def is_valid_local_file(path):
                        if path and os.path.exists(path):
                            try:
                                size = os.path.getsize(path)
                                return size >= 5 * 1024
                            except:
                                pass
                        return False

                    local_exists = bool(active_display_path and is_valid_local_file(
                        find_ready_local_video(active_display_path) or active_display_path
                    ))

                    # Tra cứu O(1) từ dict đã build sẵn
                    ev_key = _normalize_video_key(v.get('username'), v.get('video_name'), v.get('exercise'))
                    ai_eval = ai_eval_lookup.get(ev_key) or ai_eval_by_exercise.get((v.get('username'), v.get('exercise')))
                    doc_eval = doc_eval_lookup.get(ev_key) or doc_eval_by_exercise.get((v.get('username'), v.get('exercise')))
                    v_has_ai = ai_eval is not None

                    display_status = _lay_trang_thai_video_danh_sach(v, ai_eval, doc_eval, user_role)
                    analysis_time = _lay_thoi_gian_phan_tich_on_dinh(v, ai_eval) or "Chưa phân tích"
                    upload_time = _lay_thoi_gian_upload_video(v)
                    patient_label = patient_display_label(v)
                    with st.expander(
                        f"👤 {patient_label} — {v.get('exercise', 'N/A')} | "
                        f"🕒 Phân tích: {analysis_time} | 📤 Upload: {upload_time} | {display_status}"
                    ):
                        # Tỷ lệ cột [1.3, 1.0] để nới rộng video hiển thị vừa vặn hơn
                        col_v1, col_v2 = st.columns([1.3, 1.0])
                        with col_v1:
                            show_vid_key = f"show_video_{v.get('username')}_{v.get('video_name')}_{idx}"
                            if st.session_state.get(show_vid_key):
                                if active_display_path:
                                    with st.spinner("📥 Đang tải video gốc..."):
                                        play_path = _dam_bao_video_san_sang_play(
                                            active_display_path, prefer_raw=True, video_record=v
                                        )
                                        if play_path and not _is_scratch_video_path(play_path):
                                            render_video(play_path, check_h264=False, prefer_raw=True)
                                        elif _is_scratch_video_path(play_path):
                                            st.error(
                                                "❌ Đang trỏ nhầm file tạm transcode (_ftmp). "
                                                "Nhấn F5 hoặc liên hệ NCV kiểm tra file upload trên Dataset."
                                            )
                                        else:
                                            st.error("❌ Không tìm thấy file video gốc. Vui lòng thử lại sau vài giây.")
                                else:
                                    st.error("❌ Chưa có đường dẫn video upload cho mục này.")
                                if st.button("⏸️ Ẩn video", key=f"hide_vid_btn_{idx}", use_container_width=True):
                                    st.session_state[show_vid_key] = False
                                    st.rerun()
                            else:
                                st.info("ℹ️ Nhấp bên dưới để xem **video gốc** bệnh nhân đã upload (không phải bản trích xuất AI).")
                                if st.button("▶️ Xem video gốc", key=f"play_vid_btn_{idx}", type="primary", use_container_width=True):
                                    st.session_state[show_vid_key] = True
                                    st.rerun()
                        with col_v2:
                            st.write(f"**Người tập:** {patient_label}")
                            is_gay_ex = any(kw in str(v.get('exercise', '')).lower() for kw in ["gậy", "gay", "pulley", "stick"])

                            if user_role == "Bác sĩ / KTV PHCN" and not v_has_ai:
                                st.write("**Độ chính xác AI:** ⏳ Chờ NCV phân tích")
                            else:
                                # Lấy accuracy mới nhất từ evals hoặc video và hiển thị chi tiết theo 3 giai đoạn
                                ai_eval_record = ai_eval

                                metrics_v = v.get('metrics', {}) if isinstance(v.get('metrics'), dict) else {}
                                acc_g1 = metrics_v.get('metrics_g1', {}).get('do_chinh_xac') if isinstance(metrics_v.get('metrics_g1'), dict) else None
                                acc_g2 = metrics_v.get('metrics_g2', {}).get('do_chinh_xac') if isinstance(metrics_v.get('metrics_g2'), dict) else None
                                acc_g3 = metrics_v.get('metrics_g3', {}).get('do_chinh_xac') if isinstance(metrics_v.get('metrics_g3'), dict) else None

                                if ai_eval_record:
                                    acc_g1 = ai_eval_record.get('ai_accuracy_g1', acc_g1)
                                    acc_g2 = ai_eval_record.get('ai_accuracy_g2', acc_g2)
                                    acc_g3 = ai_eval_record.get('ai_accuracy_g3', acc_g3)

                                if acc_g1 is not None and acc_g2 is not None and acc_g3 is not None and not is_gay_ex:
                                    st.write("**Độ chính xác AI theo 3 giai đoạn:**")
                                    st.markdown(
                                        f"<ul style='margin: 0 0 10px 10px; padding: 0; list-style-type: none;'>"
                                        f"<li style='margin-bottom:3px;'>🌱 Giai đoạn 1 (ss±{PHASE_ERROR['g1']}°): <b style='color:#22c55e;'>{acc_g1:.1f}%</b></li>"
                                        f"<li style='margin-bottom:3px;'>📈 Giai đoạn 2 (ss±{PHASE_ERROR['g2']}°): <b style='color:#eab308;'>{acc_g2:.1f}%</b></li>"
                                        f"<li style='margin-bottom:3px;'>🎯 Giai đoạn 3 (ss±{PHASE_ERROR['g3']}°): <b style='color:#ef4444;'>{acc_g3:.1f}%</b></li>"
                                        f"</ul>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    acc_val = _lay_do_chinh_xac_hien_thi(v, ai_eval_record)
                                    acc_text = f"{acc_val:.1f}%" if isinstance(acc_val, (int, float)) and acc_val > 0 else ("Chưa phân tích" if acc_val == 0 else f"{acc_val}%")
                                    st.write(f"**Độ chính xác AI:** {acc_text}")

                            st.write(f"**Trạng thái:** {display_status}")
                            # Thời gian phân tích AI — badge màu nổi bật
                            if analysis_time and analysis_time != "Chưa phân tích":
                                analysis_time_html = safe_html(analysis_time, max_length=80)
                                st.markdown(
                                    f"**🤖 Phân tích lần cuối:**<br>"
                                    f"<span style='background:rgba(0,198,255,0.15); color:#00c6ff; "
                                    f"padding:3px 10px; border-radius:8px; font-size:0.9rem; "
                                    f"border:1px solid rgba(0,198,255,0.4); font-weight:bold;'>"
                                    f"🕒 {analysis_time_html}</span>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    "**🤖 Phân tích lần cuối:** "
                                    "<span style='color:#888; font-size:0.85rem;'>⏳ Chưa phân tích</span>",
                                    unsafe_allow_html=True
                                )
                            st.caption(f"📤 Upload: {upload_time}")

                            # Khối chẩn đoán thông tin file (chỉ hiển thị cho bác sĩ/NCV để debug)
                            if user_role in ["Bác sĩ / KTV PHCN", "Quản trị viên"]:
                                with st.popover("🔍 Kiểm tra tệp tin (Debug)"):
                                    st.markdown(f"**Tệp hiển thị:** `{active_display_path or '(chưa có — chỉ có metadata đánh giá)'}`")
                                    st.write(f"- Video gốc BN: `{raw_path or '(n/a)'}`")
                                    st.write(f"- Video processed: `{processed_path or '(n/a)'}`")
                                    st.write(f"- Tồn tại cục bộ: {'✅ Có' if local_exists else '☁️ Sẽ stream/tải từ Cloud khi xem'}")
                                    deep_check = st.button("🔬 Quét sâu codec/ffprobe", key=f"btn_deep_file_check_{idx}")
                                    if active_display_path and os.path.exists(active_display_path):
                                        st.write(f"- Kích thước tệp: `{os.path.getsize(active_display_path)/(1024*1024):.2f} MB`")
                                    if deep_check and active_display_path and os.path.exists(active_display_path):
                                        try:
                                            v_codec, a_codec = get_video_codec(active_display_path)
                                            st.write(f"- Codec: `{v_codec} / {a_codec}`")
                                            dur, ffprobe_error = ffprobe_video_duration_text(active_display_path)
                                            st.write(f"- Thời lượng ffprobe: `{dur if dur else 'Không xác định'} giây`")
                                            if ffprobe_error:
                                                st.error(f"Lỗi ffprobe: {ffprobe_error}")
                                        except Exception as e:
                                            st.write(f"- Lỗi quét ffprobe: `{e}`")

                                    st.markdown(f"**Tệp nén H.264:** `{final_h264 or '(n/a)'}`")
                                    h264_exists = bool(final_h264 and os.path.exists(final_h264) and os.path.getsize(final_h264) >= 5 * 1024)
                                    st.write(f"- Tồn tại cục bộ và hợp lệ: {'✅ Có' if h264_exists else '❌ Không'}")
                                    if final_h264 and os.path.exists(final_h264):
                                        st.write(f"- Kích thước tệp: `{os.path.getsize(final_h264)/(1024*1024):.2f} MB`")
                                    if deep_check and final_h264 and os.path.exists(final_h264):
                                        try:
                                            v_codec_h, a_codec_h = get_video_codec(final_h264)
                                            st.write(f"- Codec H264: `{v_codec_h} / {a_codec_h}`")
                                            dur_h, ffprobe_error_h = ffprobe_video_duration_text(final_h264)
                                            st.write(f"- Thời lượng ffprobe H264: `{dur_h if dur_h else 'Không xác định'} giây`")
                                            if ffprobe_error_h:
                                                st.error(f"Lỗi ffprobe H264: {ffprobe_error_h}")
                                        except Exception as e_h:
                                            st.write(f"- Lỗi quét ffprobe H264: `{e_h}`")

                                    error_log_path = os.path.join(os.path.dirname(final_h264 or PROCESSED_DIR), "transcode_error.txt")
                                    if final_h264 and os.path.exists(error_log_path):
                                        st.warning("⚠️ Phát hiện log lỗi nén gần nhất:")
                                        try:
                                            import hashlib as _hl_log
                                            _log_key = f"ffmpeg_err_{_hl_log.md5(final_h264.encode()).hexdigest()[:8]}"
                                            with open(error_log_path, "r", encoding="utf-8") as f_err:
                                                st.text_area("Chi tiết lỗi ffmpeg:", value=f_err.read(), height=150, key=_log_key)
                                        except Exception as e_log:
                                            st.write(f"Không thể đọc log lỗi: {e_log}")

                                    st.markdown("**Trạng thái Cloud:**")
                                    if HF_TOKEN and HF_DATASET_ID and active_display_path:
                                        rel_path = get_clean_rel_path(active_display_path)
                                        st.write(f"- Dataset: `{HF_DATASET_ID}`")
                                        st.write(f"- Tệp: `{rel_path}`")
                                    else:
                                        st.write("- Chưa cấu hình Cloud Dataset.")

                            # HIỂN THỊ ĐÁNH GIÁ CỦA BÁC SĨ (GROUND TRUTH) CHO NCV
                            if doc_eval:
                                eval_time_formatted = _format_vn_time(doc_eval.get('time'), default='N/A')
                                with st.expander("🩺 ĐÁNH GIÁ CHUYÊN MÔN (GROUND TRUTH)", expanded=True):
                                    if user_role == "Nghiên cứu viên":
                                        st.success(f"**Nguồn:** Chuyên gia PHCN | 🕒 **Thời gian đánh giá:** {eval_time_formatted}")
                                    else:
                                        st.success(f"**Bác sĩ:** {doc_eval.get('doctor_name', 'Bác sĩ')} | 🕒 **Thời gian đánh giá:** {eval_time_formatted}")
                                    st.write(f"**Kết quả:** {doc_eval.get('doctor_result', 'N/A')}")
                                    if user_role != "Nghiên cứu viên" and doc_eval.get('comments_ncv'):
                                        st.markdown(
                                            "<div style='background: rgba(0,198,255,0.1); padding: 10px; border-radius: 5px; "
                                            f"border-left: 3px solid #00c6ff;'><b>💬 Ghi chú cho NCV:</b> {safe_html(doc_eval.get('comments_ncv'), max_length=1000)}</div>",
                                            unsafe_allow_html=True,
                                        )
                                    if user_role != "Nghiên cứu viên":
                                        st.write(f"**Nhận xét cho BN:** {doc_eval.get('comments', '')}")
                                    st.write(f"**Kế hoạch:** {doc_eval.get('plan', 'N/A')}")
                            elif user_role == "Nghiên cứu viên":
                                st.warning("⏳ Đang chờ Bác sĩ / KTV đánh giá chuyên môn.")

                            # Đổi nhãn nút theo vai trò
                            eval_btn_label = "📝 Đánh giá của chuyên môn PHCN" if user_role == "Bác sĩ / KTV PHCN" else "📝 Phân tích và trích xuất khung xương AI"
                            if st.button(eval_btn_label, key=f"eval_btn_{idx}", width="stretch"):
                                v = _lam_moi_ban_ghi_video_tu_db(v)
                                st.session_state.current_eval_video = v
                                vp = v.get("video_path")
                                has_metrics = bool(v.get("metrics"))
                                dang_chay = bool(vp and video_dang_phan_tich(vp))

                                if user_role == "Nghiên cứu viên":
                                    # Không xóa session / không spinner block — nạp song song, chuyển tab ngay
                                    if has_metrics or dang_chay:
                                        st.session_state.view_old_analysis = True
                                        # Luôn đặt reanalyze_triggered=True để PHÂN TÍCH tab hiện
                                        # giao diện phân tích (nút bắt đầu / tiến độ) thay vì chỉ kết quả cũ
                                        st.session_state.reanalyze_triggered = True
                                        _nap_bieu_do_nhanh_tu_cloud(v, giu_phan_tich_moi=dang_chay)
                                        st.session_state._pending_chart_refresh = True
                                        if dang_chay:
                                            st.toast(
                                                "🔄 Video đang phân tích nền — mở tab xem tiến độ + kết quả đã lưu...",
                                                icon="⏳",
                                            )
                                        else:
                                            st.toast(
                                                "🔬 Chuyển sang tab Phân tích — bấm nút để chạy lại hoặc xem kết quả cũ.",
                                                icon="🧭",
                                            )
                                    else:
                                        slot_moi = _slot_video_phan_tich(v)
                                        slot_cu = st.session_state.get("_ncv_analysis_loaded_key")
                                        if slot_moi and slot_cu and slot_moi != slot_cu:
                                            _xoa_session_phan_tich()
                                        st.session_state.view_old_analysis = False
                                        st.session_state.reanalyze_triggered = False
                                        st.toast(
                                            "🔬 Video chưa có kết quả — sang tab Phân tích và bấm **Chạy phân tích mới**.",
                                            icon="🧭",
                                        )
                                else:
                                    st.session_state.reanalyze_triggered = False
                                    st.session_state.view_old_analysis = bool(v.get("metrics"))
                                    if user_role == "Bác sĩ / KTV PHCN":
                                        st.toast("🚀 Đang chuyển sang tab 📊 QUẢN LÝ ĐÁNH GIÁ & NCKH...", icon="🔄")
                                    else:
                                        st.toast("🚀 Đang chuyển tab...", icon="🔄")

                                if user_role == "Bác sĩ / KTV PHCN":
                                    st.session_state.trigger_tab_switch = "📊 QUẢN LÝ ĐÁNH GIÁ & NCKH"
                                elif user_role == "Nghiên cứu viên":
                                    st.session_state.trigger_tab_switch = "🔬 PHÂN TÍCH & TRÍCH XUẤT DỮ LIỆU"
                                # scope="app" cần thiết vì button này nằm trong fragment:
                                # st.rerun() không scope → chỉ rerun fragment → tab switch không chạy
                                st.rerun(scope="app")

                        delete_key = f"{v.get('username')}|{v.get('video_name')}|{idx}"
                        if st.button("🗑️ Yêu cầu xóa video", key=f"del_video_{idx}", width="stretch"):
                            st.session_state["_pending_delete_video"] = delete_key
                        if st.session_state.get("_pending_delete_video") == delete_key:
                            st.warning("Xác nhận xóa video này. Hệ thống sẽ backup trước khi xóa.")
                            confirm_delete = st.checkbox("Tôi hiểu thao tác này sẽ xóa video và đánh giá liên quan.", key=f"confirm_delete_video_{idx}")
                            if st.button("Xác nhận xóa video", key=f"confirm_del_video_{idx}", width="stretch", disabled=not confirm_delete):
                                delete_video_callback(v.get('video_name'), v.get('username'))
                                st.session_state.pop("_pending_delete_video", None)
                with col_list2:
                    if st.button("❌", key=f"quick_x_video_{idx}", help="Yêu cầu xóa"):
                        st.session_state["_pending_delete_video"] = f"{v.get('username')}|{v.get('video_name')}|{idx}"
