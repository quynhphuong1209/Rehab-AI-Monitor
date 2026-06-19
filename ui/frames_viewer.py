"""Frame gallery UI helpers extracted from app.py."""

from __future__ import annotations

import base64
import os
import subprocess


def render_frame_grid(
    deps,
    indices_list,
    frame_data_list,
    quality_mode_val,
    tab_threshold,
    tab_key,
    key_suffix_val,
    *,
    processed_video_path,
    frame_phase_status,
):
    import math
    st = deps.st
    cv2 = deps.cv2
    format_ml_display = deps.format_ml_display
    get_local_frame_path = deps.get_local_frame_path
    frames_zip_from_processed_path = deps.frames_zip_from_processed_path
    ensure_local_file = deps.ensure_local_file
    get_final_h264_path = deps.get_final_h264_path
    sync_transcode_to_h264 = deps.sync_transcode_to_h264
    build_frame_extract_command = deps.build_frame_extract_command
    page_key = f"fp_{tab_key}_{key_suffix_val}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    # Cấu hình màu sắc động thích ứng theo chế độ Sáng/Tối (Light/Dark Mode)
    is_light = st.session_state.get('theme') == 'light'
    card_bg = "#ffffff"
    card_border = "#00c6ff"
    card_hover_border = "#0072ff"
    card_text = "#1a1a2e"
    card_text_muted = "#555555"
    img_bg = "#f1f3f5"
    card_shadow = "0 6px 16px rgba(0, 198, 255, 0.15)"

    if not is_light:
        card_bg = "#1a1a2e"
        card_border = "#2d2d44"
        card_text = "#ffffff"
        card_text_muted = "#aaa"
        img_bg = "#0a0a16"
        card_shadow = "0 4px 6px rgba(0, 0, 0, 0.15)"

    # Style động chỉ còn màu theo theme; layout frame nằm trong assets/styles.css.
    st.markdown(f"""
    <style>
    .frame-card {{
        background-color: {card_bg} !important;
        border: 1.5px solid {card_border} !important;
        box-shadow: {card_shadow} !important;
        color: {card_text} !important;
    }}
    .frame-card:hover {{
        border-color: {card_hover_border} !important;
        box-shadow: 0 6px 12px rgba(0, 114, 255, 0.25) !important;
    }}
    .frame-card-index {{
        color: {card_text_muted} !important;
    }}
    .frame-card-img-wrapper {{
        background-color: {img_bg} !important;
    }}
    .frame-card-footer {{
        color: {card_text} !important;
    }}
    .frame-card-footer span {{
        color: {card_text} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    rc1, rc2, rc3, rc4 = st.columns([1.5, 1.5, 2.0, 0.6])
    with rc1:
        fpp_option = st.selectbox("📄 Số/Trang", [12, 24, 36, 48, 96, "Tất cả"], index=3, key=f"fpp_{tab_key}_{key_suffix_val}")
        fpp = 999999 if fpp_option == "Tất cả" else int(fpp_option)
    with rc2:
        grid_cols = st.selectbox("🗂️ Số cột", [1, 2, 3, 4], index=3, key=f"fcols_{tab_key}_{key_suffix_val}")
    with rc3:
        sub_filter = st.selectbox("🔍 Lọc thêm", ["Tất cả", "PASS", "NEAR", "FAIL"], key=f"fsub_{tab_key}_{key_suffix_val}")
    with rc4:
        st.write("")
        st.write("")
        if st.button("🔄", width="stretch", key=f"fref_{tab_key}_{key_suffix_val}"):
            st.rerun()

    # Áp dụng sub_filter
    if sub_filter == "PASS":
        indices_list = [i for i in indices_list if frame_phase_status(frame_data_list[i], tab_threshold) == "PASS"]
    elif sub_filter == "NEAR":
        indices_list = [i for i in indices_list if frame_phase_status(frame_data_list[i], tab_threshold) == "NEAR"]
    elif sub_filter == "FAIL":
        indices_list = [i for i in indices_list if frame_phase_status(frame_data_list[i], tab_threshold) == "FAIL"]

    total_f = len(indices_list)
    total_p = max(1, (total_f + fpp - 1) // fpp)
    if st.session_state[page_key] > total_p:
        st.session_state[page_key] = total_p

    # Đếm PASS/NEAR/FAIL theo ngưỡng giai đoạn
    cnt_pass = sum(1 for i in indices_list if frame_phase_status(frame_data_list[i], tab_threshold) == "PASS")
    cnt_near = sum(1 for i in indices_list if frame_phase_status(frame_data_list[i], tab_threshold) == "NEAR")
    cnt_fail = total_f - cnt_pass - cnt_near

    # Thẻ thống kê
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("📊 Tổng frames", total_f)
    sc2.metric("✅ PASS", cnt_pass, f"{cnt_pass/total_f*100:.0f}%" if total_f > 0 else "0%")
    sc3.metric("⚠️ NEAR", cnt_near, f"{cnt_near/total_f*100:.0f}%" if total_f > 0 else "0%")
    sc4.metric("❌ FAIL", cnt_fail, f"{cnt_fail/total_f*100:.0f}%" if total_f > 0 else "0%")

    st.caption(f"Đang xem trang {st.session_state[page_key]}/{total_p} ({total_f} frames)")
    nc1, nc2, nc3 = st.columns([1, 2, 1])
    def _prev(pk=page_key):
        if st.session_state[pk] > 1: st.session_state[pk] -= 1
    def _next(pk=page_key, tp=total_p):
        if st.session_state[pk] < tp: st.session_state[pk] += 1
    with nc1:
        st.button("◀ Trước", key=f"pp_{tab_key}_{key_suffix_val}", width='stretch', on_click=_prev)
    with nc2:
        st.number_input("Trang", min_value=1, max_value=total_p, key=page_key, label_visibility="collapsed")
    with nc3:
        st.button("Sau ▶", key=f"pn_{tab_key}_{key_suffix_val}", width='stretch', on_click=_next)

    if total_f == 0:
        st.info("ℹ️ Không có frame nào trong bộ lọc này.")
        return

    s_idx = (st.session_state[page_key] - 1) * fpp
    e_idx = min(s_idx + fpp, total_f)
    page_inds = indices_list[s_idx:e_idx]

    # Xác định ZIP và đọc TOÀN BỘ frames trang hiện tại từ ZIP một lần (tránh mở ZIP N lần)
    has_zip = False
    zip_path_for_check = ""
    # Thử cả hai nguồn ZIP: từ video path và từ session state frames_zip
    _zip_candidates = []
    if processed_video_path:
        _zip_candidates.append(get_local_frame_path(frames_zip_from_processed_path(processed_video_path)))
    _fz_alt = get_local_frame_path(st.session_state.get('frames_zip') or "")
    if _fz_alt and _fz_alt not in _zip_candidates:
        _zip_candidates.append(_fz_alt)
    # Nếu ZIP chưa có local → thử tải về từ HF Dataset (1 lần, try_fallbacks=False để không bỏ ZIP)
    for _zc in _zip_candidates:
        if _zc and not (os.path.exists(_zc) and os.path.getsize(_zc) > 1024):
            try:
                with st.spinner("📥 Đang tải file ảnh frames từ Cloud..."):
                    ensure_local_file(_zc, quiet=True, try_fallbacks=False)
            except Exception:
                pass
            break
    for _zc in _zip_candidates:
        if _zc and os.path.exists(_zc) and os.path.getsize(_zc) > 1024:
            zip_path_for_check = _zc
            has_zip = True
            break

    # ── DEBUG: hiển thị trạng thái ZIP / Video để chẩn đoán ──────────────────
    _zip_dbg = zip_path_for_check or (_zip_candidates[0] if _zip_candidates else "N/A")
    _zip_sz = os.path.getsize(_zip_dbg) if (_zip_dbg and os.path.exists(_zip_dbg)) else -1
    _vid_dbg = processed_video_path or "N/A"
    _vid_sz = os.path.getsize(_vid_dbg) if (_vid_dbg and os.path.exists(_vid_dbg)) else -1
    with st.expander("🔍 Debug thông tin frames (bấm để mở)", expanded=False):
        st.caption(
            f"**ZIP:** `{_zip_dbg}` — tồn tại: {os.path.exists(_zip_dbg) if _zip_dbg != 'N/A' else False}, "
            f"kích thước: {_zip_sz/1024/1024:.1f} MB | has_zip={has_zip}\n\n"
            f"**Video:** `{_vid_dbg}` — tồn tại: {os.path.exists(_vid_dbg) if _vid_dbg != 'N/A' else False}, "
            f"kích thước: {_vid_sz/1024/1024:.1f} MB"
        )
        if has_zip and zip_path_for_check:
            try:
                import zipfile as _zmod
                with _zmod.ZipFile(zip_path_for_check, 'r') as _zt:
                    _zcount = len(_zt.namelist())
                st.caption(f"Số frame trong ZIP: **{_zcount}** / {total_f} total")
            except Exception as _ze:
                st.caption(f"⚠️ Không đọc được ZIP: {_ze}")
    # ── END DEBUG ──────────────────────────────────────────────────────────────

    # Đọc trước base64 tất cả frames trang hiện tại từ ZIP (1 lần mở, đọc nhiều)
    zip_b64_cache = {}
    if has_zip:
        try:
            import zipfile
            with zipfile.ZipFile(zip_path_for_check, 'r') as _z_page:
                _names_in_zip = set(_z_page.namelist())
                for _pi in page_inds:
                    _fp = get_local_frame_path(frame_data_list[_pi].get('path', ''))
                    _fn = os.path.basename(_fp) if _fp else ''
                    if _fn and _fn in _names_in_zip:
                        try:
                            zip_b64_cache[_fn] = base64.b64encode(_z_page.read(_fn)).decode('utf-8')
                        except Exception:
                            pass
        except Exception as _ze:
            print(f"[Frame Gallery] Lỗi đọc ZIP: {_ze}")

    def _is_image_missing_or_invalid(img_p):
        fn = os.path.basename(img_p) if img_p else ''
        if fn and fn in zip_b64_cache:
            return False
        if not img_p or not os.path.exists(img_p):
            return True
        try:
            return os.path.getsize(img_p) < 5 * 1024
        except:
            return True

    # Phục hồi frame từ video nếu còn thiếu sau khi đã thử ZIP
    any_missing = any(_is_image_missing_or_invalid(get_local_frame_path(frame_data_list[idx].get('path', ''))) for idx in page_inds)
    cap_recover = None
    _recover_vid_path = None
    if any_missing and processed_video_path:
        # Ưu tiên H.264 (_f.mp4) vì dễ decode hơn MP4V gốc
        _h264_path = get_final_h264_path(processed_video_path)
        _vid_for_recovery = (
            _h264_path if (_h264_path and os.path.exists(_h264_path) and os.path.getsize(_h264_path) > 0)
            else (processed_video_path if (processed_video_path and os.path.exists(processed_video_path) and os.path.getsize(processed_video_path) > 0)
                  else None)
        )
        if _vid_for_recovery:
            _recover_vid_path = _vid_for_recovery  # luôn giữ để ffmpeg fallback dùng
            try:
                cap_recover = cv2.VideoCapture(_vid_for_recovery)
                if not cap_recover.isOpened():
                    cap_recover.release()
                    cap_recover = None
            except Exception as e:
                print("[Frame Recovery] Lỗi mở video phục hồi frame:", e)
                cap_recover = None
        # Nếu OpenCV không mở được (MP4V không hỗ trợ) → transcode sang H.264 trước
        if not cap_recover and _vid_for_recovery and not (_h264_path and os.path.exists(_h264_path) and os.path.getsize(_h264_path) > 5 * 1024):
            try:
                _h264_done = sync_transcode_to_h264(_vid_for_recovery)
                if _h264_done:
                    cap_recover = cv2.VideoCapture(_h264_done)
                    if not cap_recover.isOpened():
                        cap_recover.release()
                        cap_recover = None
                    else:
                        _recover_vid_path = _h264_done
            except Exception as _tc_err:
                print("[Frame Recovery] Lỗi transcode để recovery:", _tc_err)

    for orig_idx in page_inds:
        f_data = frame_data_list[orig_idx]
        f_path = get_local_frame_path(f_data.get('path'))

        # Khôi phục ảnh từ video nếu thiếu (dùng đúng index frame trong video)
        if f_path and _is_image_missing_or_invalid(f_path):
            f_idx = f_data.get('index', orig_idx)
            _recovered = False
            # Thử OpenCV trước
            if cap_recover and cap_recover.isOpened():
                try:
                    os.makedirs(os.path.dirname(f_path), exist_ok=True)
                    cap_recover.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame_img = cap_recover.read()
                    if ret:
                        cv2.imwrite(f_path, frame_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        _recovered = os.path.exists(f_path)
                except Exception as e:
                    print(f"[Frame Recovery] Lỗi trích xuất frame {orig_idx}: {e}")
            # Fallback: dùng ffmpeg nếu OpenCV thất bại
            if not _recovered and _recover_vid_path and os.path.exists(_recover_vid_path):
                try:
                    os.makedirs(os.path.dirname(f_path), exist_ok=True)
                    fps_v = 30.0
                    try:
                        _cap_tmp = cv2.VideoCapture(_recover_vid_path)
                        if _cap_tmp.isOpened():
                            fps_v = _cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
                        _cap_tmp.release()
                    except Exception:
                        pass
                    t_sec = f_idx / max(fps_v, 1.0)
                    subprocess.run(
                        build_frame_extract_command(_recover_vid_path, f_path, timestamp=t_sec),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
                    )
                except Exception as e_ff:
                    print(f"[Frame Recovery ffmpeg] frame {orig_idx}: {e_ff}")

    if cap_recover:
        cap_recover.release()

    # Gộp toàn bộ trang thành 1 st.markdown() → 1 ForwardMsg duy nhất, tránh "Cached ForwardMsg MISS"
    grid_parts = [f'<div style="display:grid;grid-template-columns:repeat({grid_cols},1fr);gap:12px;">']
    any_missing_img = False

    for i, orig_idx in enumerate(page_inds):
        f_data = frame_data_list[orig_idx]
        f_path = get_local_frame_path(f_data.get('path'))

        phase_st = frame_phase_status(f_data, tab_threshold)
        color = "#22c55e" if phase_st == "PASS" else ("#f59e0b" if phase_st == "NEAR" else "#ef4444")
        bg_alpha = "rgba(34,197,94,0.12)" if phase_st == "PASS" else ("rgba(245,158,11,0.12)" if phase_st == "NEAR" else "rgba(239,68,68,0.12)")

        ml_label = f_data.get('ml_label_text')
        ml_badge_html = ""
        ml_footer_html = ""
        if ml_label:
            if format_ml_display:
                ml_disp = format_ml_display(f_data)
                ml_label_text = ml_disp.get("label_vi") or str(ml_label)
                ml_key = ml_label_text.strip().lower()
            else:
                ml_label_text = str(ml_label)
                ml_key = ml_label_text.strip().lower()
                ml_disp = {"badge_text": ml_label_text, "footer_text": f"ML: {ml_label_text}", "prob_text": ""}
            ml_color = "#22c55e" if ("đúng" in ml_key and "gần" not in ml_key) or ("dung" in ml_key and "gan" not in ml_key) else ("#f59e0b" if "gần" in ml_key or "gan" in ml_key else "#ef4444")
            badge_text = ml_disp.get("badge_text") or ml_label_text
            footer_text = ml_disp.get("footer_text") or f"ML: {ml_label_text}"
            prob_text = ml_disp.get("prob_text") or ""
            ml_badge_html = f'<span class="frame-card-badge" style="background:{ml_color}1f;color:{ml_color};border-color:{ml_color}40;">ML · {badge_text}</span>'
            ml_footer_html = f'<div class="frame-card-row"><span>Model ML:</span><span style="color:{ml_color};font-weight:bold;">{footer_text}</span></div>'
            if prob_text:
                ml_footer_html += f'<div class="frame-card-row"><span>Xác suất 3 lớp:</span><span style="font-size:0.72rem;">{prob_text}</span></div>'

        gv = f_data.get('goc_vai', 0) or 0
        gk = f_data.get('goc_khuyu', 0) or 0
        eval_inf = f_data.get('eval_info', {})
        cv_ref = eval_inf.get('shoulder_ref', 90)
        ck_ref = eval_inf.get('elbow_ref', 170)
        diff_v = abs(gv - cv_ref)
        diff_k = abs(gk - ck_ref)

        # Lấy base64: ưu tiên cache từ ZIP, sau đó đọc từ file
        f_name = os.path.basename(f_path) if f_path else ''
        b64_data = zip_b64_cache.get(f_name, '')
        if not b64_data and f_path and os.path.exists(f_path) and os.path.getsize(f_path) >= 5 * 1024:
            try:
                with open(f_path, "rb") as img_file:
                    b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            except:
                pass

        if b64_data:
            grid_parts.append(
                f'<div class="frame-card">'
                f'<div class="frame-card-header">'
                f'<span class="frame-card-index">#{f_data.get("index")}</span>'
                f'<span class="frame-card-badges">'
                f'<span class="frame-card-badge" style="background:{bg_alpha};color:{color};border-color:{color}40;">{phase_st}</span>'
                f'{ml_badge_html}'
                f'</span></div>'
                f'<div class="frame-card-img-wrapper">'
                f'<img class="frame-card-img" src="data:image/jpeg;base64,{b64_data}" />'
                f'</div>'
                f'<div class="frame-card-footer">'
                f'<div class="frame-card-row"><span>Vai: <b>{gv:.0f}°</b> / {cv_ref:.0f}°</span>'
                f'<span style="color:{color};font-weight:bold;">Δ {diff_v:.1f}°</span></div>'
                f'<div class="frame-card-row"><span>Khuỷu: <b>{gk:.0f}°</b> / {ck_ref:.0f}°</span>'
                f'<span style="color:{color};font-weight:bold;">Δ {diff_k:.1f}°</span></div>'
                f'{ml_footer_html}'
                f'</div></div>'
            )
        else:
            any_missing_img = True
            grid_parts.append(
                f'<div class="frame-card" style="display:flex;align-items:center;justify-content:center;min-height:120px;">'
                f'<span style="color:#ef4444;font-size:0.85rem;">⚠ Ảnh lỗi #{f_data.get("index","?")}</span>'
                f'</div>'
            )

    grid_parts.append('</div>')
    st.markdown(''.join(grid_parts), unsafe_allow_html=True)
    if any_missing_img:
        st.caption("⚠ Một số ảnh chưa tải được — thử tải lại trang hoặc chạy lại phân tích.")

def render_frames_full(deps, key_suffix=""):
    """Hiển thị video khung xương + ảnh frame đã phân tích."""
    st = deps.st
    try:
        _noi_dung_frames_day_du(deps, key_suffix)
    except Exception as frames_err:
        st.error(f"❌ Không thể hiển thị video/ảnh frame: {frames_err}")
        v_err = st.session_state.get("current_eval_video")
        if v_err:
            deps.render_reload_and_reanalyze_button(v_err, key_suffix=f"frames_err_{key_suffix}")
        import traceback
        st.caption(traceback.format_exc())


def _noi_dung_frames_day_du(deps, key_suffix=""):

    st = deps.st
    PHASE_ERROR = deps.PHASE_ERROR
    safe_html = deps.safe_html
    render_video = deps.render_video
    _tim_video_phan_tich_moi_nhat = deps.find_latest_analyzed_video
    _dong_bo_metadata_frames_vao_session = deps.sync_frame_metadata_to_session
    _session_phan_tich_khop_video = deps.analysis_session_matches_video
    tu_dong_nap_ket_qua_phan_tich_gan_nhat = deps.auto_load_latest_analysis_result
    get_local_frame_path = deps.get_local_frame_path
    is_local_file_ready = deps.is_local_file_ready
    ensure_local_file = deps.ensure_local_file
    check_and_extract_frames_zip = deps.check_and_extract_frames_zip
    _frames_zip_path_from_video = deps.frames_zip_path_from_video
    _frames_zip_from_processed_path = deps.frames_zip_from_processed_path
    load_all_frames_data_cached = deps.load_all_frames_data_cached
    recalc_metrics = deps.recalc_metrics
    lay_do_chinh_xac_ai_chuan = deps.standard_ai_accuracy
    find_ready_local_video = deps.find_ready_local_video
    resolve_playback_video_path = deps.resolve_playback_video_path
    ensure_playable_video = deps.ensure_playable_video
    _prefetch_video_quiet = deps.prefetch_video_quiet
    video_has_audio_track = deps.video_has_audio_track
    segment_frames = deps.segment_frames
    get_video_fps_cached = deps.get_video_fps_cached
    cut_video_segments = deps.cut_video_segments
    get_final_h264_path = deps.get_final_h264_path
    sync_transcode_to_h264 = deps.sync_transcode_to_h264
    create_zip_of_frames = deps.create_zip_of_frames
    dam_bao_tai_video_phan_tich = deps.ensure_analysis_video_downloaded
    hien_thi_nut_tai_lai_va_phan_tich_moi = deps.render_reload_and_reanalyze_button
    gui_bao_cao_tong_hop_3_giai_doan = deps.send_three_stage_report
    user_role = st.session_state.user_info.get('role')
    ex_obj = st.session_state.get('exercise')
    exercise_name = ex_obj.get('ten', '') if isinstance(ex_obj, dict) else ''
    is_gay_ex = any(kw in str(exercise_name).lower() or kw in str(st.session_state.get('current_eval_video', {}).get('exercise', '')).lower() for kw in ["gậy", "gay", "pulley", "stick"])

    v_frames = st.session_state.get("current_eval_video") or _tim_video_phan_tich_moi_nhat()
    if v_frames:
        v_frames = _dong_bo_metadata_frames_vao_session(v_frames, download=False) or v_frames
    if v_frames and not _session_phan_tich_khop_video(v_frames):
        with st.spinner(
            f"📥 Đang tải khung xương: {v_frames.get('full_name')} — {v_frames.get('exercise')}..."
        ):
            tu_dong_nap_ket_qua_phan_tich_gan_nhat(v_frames, force=False)

    all_frames_data_path = get_local_frame_path(st.session_state.get('all_frames_data_path'))
    if not all_frames_data_path:
        v_frames = _dong_bo_metadata_frames_vao_session(v_frames, download=True) if v_frames else v_frames
        all_frames_data_path = get_local_frame_path(st.session_state.get('all_frames_data_path'))
    if not all_frames_data_path:
        st.info("📭 Không có dữ liệu khung hình để hiển thị.")
        if st.session_state.get("current_eval_video"):
            hien_thi_nut_tai_lai_va_phan_tich_moi(st.session_state.current_eval_video, key_suffix=f"frames_empty_{key_suffix}")
        return

    if not is_local_file_ready(all_frames_data_path):
        with st.spinner("📥 Đang tải video và ảnh frame từ Cloud..."):
            ensure_local_file(all_frames_data_path)
            proc = st.session_state.get("processed_video_path")
            if proc:
                ensure_local_file(proc, try_fallbacks=True)
                check_and_extract_frames_zip(proc)
            fz = _frames_zip_path_from_video(st.session_state.get("current_eval_video") or {}) or st.session_state.get("frames_zip")
            if fz:
                ensure_local_file(get_local_frame_path(fz) or fz)
        if not is_local_file_ready(all_frames_data_path):
            st.info("📭 Dữ liệu frames chưa có sẵn local. Bấm **Tải lại kết quả đã lưu** ở trên để tải đầy đủ.")
            if st.button("⚡ Tải nhanh dữ liệu frames", key=f"btn_lazy_frames_data_{key_suffix}", use_container_width=True):
                with st.spinner("📥 Đang tải dữ liệu frames..."):
                    if ensure_local_file(all_frames_data_path):
                        proc = st.session_state.get("processed_video_path")
                        if proc:
                            check_and_extract_frames_zip(proc)
                        st.rerun()
                    else:
                        st.error("Không tải được dữ liệu frames từ Cloud.")
            if st.session_state.get("current_eval_video"):
                hien_thi_nut_tai_lai_va_phan_tich_moi(st.session_state.current_eval_video, key_suffix=f"frames_retry_{key_suffix}")
            return

    all_frames_data = load_all_frames_data_cached(all_frames_data_path)

    total_frames = len(all_frames_data)
    if total_frames == 0:
        st.warning("⚠️ Dữ liệu khung hình trống. Vui lòng phân tích lại video.")
        return

    # Kiểm tra sớm: nếu frame images không có local và ZIP cũng chưa có → tải từ HF Dataset
    _proc_fp = get_local_frame_path(st.session_state.get("processed_video_path") or "")
    _fz_sess = get_local_frame_path(st.session_state.get("frames_zip") or "")
    _zip_from_proc = _frames_zip_from_processed_path(_proc_fp)
    _zip_local = (_zip_from_proc if (_zip_from_proc and os.path.exists(_zip_from_proc))
                  else (_fz_sess if (_fz_sess and os.path.exists(_fz_sess)) else ""))
    _first_frame_path = get_local_frame_path((all_frames_data[0] if all_frames_data else {}).get("path", ""))
    _vid_ok = (_proc_fp and os.path.exists(_proc_fp) and os.path.getsize(_proc_fp) > 0)
    _frames_ready = (
        (_first_frame_path and os.path.exists(_first_frame_path) and os.path.getsize(_first_frame_path) >= 5 * 1024)
        or bool(_zip_local)
        or _vid_ok   # video tồn tại local → frame recovery trong _render_frame_grid xử lý
    )
    if not _frames_ready:
        # Chưa có gì local — thử tải từ HF Dataset
        _zip_to_dl = _zip_from_proc or _fz_sess
        _got_new_zip = False
        _got_new_vid = False
        with st.spinner("📥 Đang tải ảnh frames từ Cloud..."):
            if _zip_to_dl and not (os.path.exists(_zip_to_dl) and os.path.getsize(_zip_to_dl) >= 5 * 1024):
                if ensure_local_file(_zip_to_dl, try_fallbacks=False):
                    _got_new_zip = os.path.exists(_zip_to_dl)
            if _proc_fp and not os.path.exists(_proc_fp):
                if ensure_local_file(_proc_fp, try_fallbacks=True):
                    _got_new_vid = os.path.exists(_proc_fp)
                    if _got_new_vid:
                        check_and_extract_frames_zip(_proc_fp)
        _zip_now = (_zip_from_proc if (_zip_from_proc and os.path.exists(_zip_from_proc))
                    else (_fz_sess if (_fz_sess and os.path.exists(_fz_sess)) else ""))
        _first_ok_now = (_first_frame_path and os.path.exists(_first_frame_path) and os.path.getsize(_first_frame_path) >= 5 * 1024)
        if _zip_now or _first_ok_now:
            st.rerun()
        elif _got_new_vid and os.path.exists(_proc_fp):
            st.rerun()
        else:
            st.info("⏳ Ảnh frames chưa có trên Cloud. Chạy lại phân tích trên hệ thống này để tạo ảnh.")
            if st.session_state.get("current_eval_video"):
                hien_thi_nut_tai_lai_va_phan_tich_moi(
                    st.session_state.current_eval_video, key_suffix=f"noframes_{key_suffix}"
                )
            return

    pass_count = sum(1 for f in all_frames_data if f.get('dung'))
    nearly_count = sum(1 for f in all_frames_data if f.get('gan_dung') and not f.get('dung'))
    fail_count = total_frames - pass_count - nearly_count

    tk = st.session_state.get('stats') or {}

    if is_gay_ex:
        df = st.session_state.get('angle_df')
        ss_val = tk.get('sai_so', 30) if isinstance(tk, dict) else 30
        if df is not None and len(df) > 0:
            m_overall = recalc_metrics(df, ss_val, exercise_name)
        else:
            m_overall = tk

        if isinstance(m_overall, dict) and 'frame_dung' in m_overall:
            pass_count = int(m_overall.get('frame_dung', 0))
            nearly_count = int(m_overall.get('frame_gan_dung', 0))
            fail_count = int(m_overall.get('frame_sai', 0))
    filename = st.session_state.get('uploaded_file_name') or os.path.basename(st.session_state.get('processed_video_path', '') or 'Video hệ thống')
    filename_html = safe_html(filename, max_length=180)
    v_meta = st.session_state.get('current_eval_video') or {}
    ai_acc = lay_do_chinh_xac_ai_chuan(v_meta) or (tk.get('do_chinh_xac', 0.0) if isinstance(tk, dict) else 0.0)

    # Lấy thông số 3 giai đoạn để hiển thị chi tiết
    metrics_g1 = tk.get('metrics_g1', {}) if isinstance(tk, dict) else {}
    metrics_g2 = tk.get('metrics_g2', {}) if isinstance(tk, dict) else {}
    metrics_g3 = tk.get('metrics_g3', {}) if isinstance(tk, dict) else {}

    acc_g1 = metrics_g1.get('do_chinh_xac', 0.0) if isinstance(metrics_g1, dict) else 0.0
    acc_g2 = metrics_g2.get('do_chinh_xac', 0.0) if isinstance(metrics_g2, dict) else 0.0
    acc_g3 = metrics_g3.get('do_chinh_xac', 0.0) if isinstance(metrics_g3, dict) else 0.0
    raw_video_path = st.session_state.get('processed_video_path')
    processed_video_path = get_local_frame_path(raw_video_path) or raw_video_path
    playback_video_path = None
    if processed_video_path:
        local_ready = find_ready_local_video(processed_video_path)
        if local_ready:
            playback_video_path = (
                resolve_playback_video_path(local_ready)
                or local_ready
            )
        else:
            playback_video_path = processed_video_path
            ensure_playable_video(processed_video_path)
            _prefetch_video_quiet(processed_video_path)
    frames_zip = get_local_frame_path(st.session_state.get('frames_zip'))
    has_video = bool(processed_video_path)

    # 0. HIỂN THỊ VIDEO ĐÃ PHÂN TÍCH
    st.markdown("### 🎬 VIDEO ĐÃ PHÂN TÍCH")
    st.caption(
        "🔊 Giọng nói AI **Đúng / Gần đúng / Sai** được trộn vào video khi bấm **Chạy phân tích mới** "
        "(mặc định: Heavy · 720p · mọi frame)."
    )
    _play_check = playback_video_path or processed_video_path
    if (
        has_video
        and _play_check
        and os.path.exists(_play_check)
        and not video_has_audio_track(_play_check)
    ):
        st.info(
            "ℹ️ Video này chưa có giọng nói phản hồi (bản phân tích cũ hoặc chỉ có tiếng beep). "
            "Bấm **🚀 Chạy phân tích mới** để tạo lại video có giọng nói AI."
        )
    if total_frames >= 3:
        _idx_sample = [int((f or {}).get("index") or (i + 1)) for i, f in enumerate(all_frames_data[:12])]
        _gaps = [b - a for a, b in zip(_idx_sample, _idx_sample[1:]) if b - a > 1]
        if _gaps and max(_gaps, default=1) > 1:
            st.info(
                "ℹ️ Khung hình đang **nhảy số** (vd. #1, #4, #7) — phân tích cũ đã bỏ frame. "
                "Bấm **🚀 Chạy phân tích mới** với mặc định **Mọi frame** để có đủ #1, #2, #3..."
            )

    # Khung video và thông tin
    v_col1, v_col2 = st.columns([1.3, 1.0], gap='large')
    with v_col1:
        if has_video:
            # Tự động tính toán phân đoạn thông minh
            if 'segment_bounds' not in st.session_state or st.session_state.get('last_processed_video_for_bounds') != processed_video_path:
                st.session_state.segment_bounds = segment_frames(all_frames_data)
                st.session_state.last_processed_video_for_bounds = processed_video_path

            n0, n1, n2, n3 = st.session_state.segment_bounds

            # Lựa chọn ngang bằng st.radio thay cho st.tabs để tăng tối đa tốc độ hiển thị
            giai_doan_options = [
                "📋 Video Tất cả",
                f"🟢 Video G1 (Lượt 1: {n1 - n0} F)",
                f"🟡 Video G2 (Lượt 2 lặp lại: {n2 - n1} F)",
                f"🔴 Video G3 (Lượt 3 lặp lại: {n3 - n2} F)"
            ]

            if is_gay_ex:
                sel_giai_doan = "📋 Video Tất cả"
            else:
                sel_giai_doan = st.radio(
                    "Chọn phân đoạn video hiển thị:",
                    options=giai_doan_options,
                    horizontal=True,
                    key=f"sel_giai_doan_{key_suffix}"
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            g1_v_path = g2_v_path = g3_v_path = None
            if sel_giai_doan != "📋 Video Tất cả":
                # Chỉ cắt video giai đoạn khi thật sự được chọn, tránh chậm khi mở tab.
                fps_export = 15
                try:
                    mtime = os.path.getmtime(processed_video_path)
                    size = os.path.getsize(processed_video_path)
                    fps_export = get_video_fps_cached(processed_video_path, mtime, size)
                except Exception:
                    pass
                g1_v_path, g2_v_path, g3_v_path = cut_video_segments(processed_video_path, n1, n2, total_frames, fps_export)

            if sel_giai_doan == "📋 Video Tất cả":
                # Nếu _f.mp4 chưa tồn tại nhưng video gốc MP4V có sẵn → tự động transcode để phát được trên trình duyệt
                _play_target = playback_video_path or processed_video_path
                if _play_target:
                    _h264_target = get_final_h264_path(_play_target)
                    if _h264_target and not (os.path.exists(_h264_target) and os.path.getsize(_h264_target) > 5 * 1024):
                        _raw_src = _play_target.replace('_f.mp4', '.mp4') if _play_target.endswith('_f.mp4') else _play_target
                        if _raw_src and os.path.exists(_raw_src) and os.path.getsize(_raw_src) > 5 * 1024:
                            with st.spinner("⏳ Đang chuẩn bị video H.264 để hiển thị (chạy một lần)..."):
                                _h264_done = sync_transcode_to_h264(_raw_src)
                            if _h264_done:
                                playback_video_path = _h264_done
                render_video(playback_video_path or processed_video_path)
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    dl_ready_path = st.session_state.get(f"dl_h264_ready_{key_suffix}") or playback_video_path
                    if dl_ready_path and dl_ready_path.endswith('_f.mp4') and os.path.exists(dl_ready_path):
                        dl_name = os.path.splitext(filename)[0] + "_phan_tich.mp4" if filename else "processed_video_full.mp4"
                        with open(dl_ready_path, "rb") as f:
                            st.download_button("📥 Tải video Tất cả (H.264)", f, dl_name, "video/mp4", width="stretch", key=f"dl_v_all_{key_suffix}")
                    else:
                        if st.button("📥 Chuẩn bị video H.264 để tải", width="stretch", key=f"btn_prep_dl_all_{key_suffix}"):
                            with st.spinner("Đang chuyển sang H.264 (mở được trên Windows/điện thoại)..."):
                                ready = resolve_playback_video_path(processed_video_path, sync_transcode=True)
                                if ready and os.path.exists(ready):
                                    st.session_state[f"dl_h264_ready_{key_suffix}"] = ready
                                    st.rerun()
                                else:
                                    st.error("Không chuyển được video. Thử phân tích lại hoặc liên hệ hỗ trợ.")
                with d_col2:
                    frames_zip = st.session_state.get('frames_zip')
                    if frames_zip and os.path.exists(frames_zip):
                        with open(frames_zip, "rb") as fzip:
                            st.download_button("📦 Tải tất cả frames (ZIP)", fzip, "all_frames.zip", "application/zip", width="stretch", key=f"dl_zip_main_{key_suffix}")
                    else:
                        if st.button("📦 Chuẩn bị file ZIP tải ảnh", width="stretch", key=f"btn_prep_zip_main_{key_suffix}"):
                            with st.spinner("🔄 Đang nén khung hình..."):
                                new_zip_main = create_zip_of_frames(all_frames_data, processed_video_path)
                                if new_zip_main:
                                    st.session_state.frames_zip = new_zip_main
                                    st.rerun()
                                else:
                                    st.error("❌ Lỗi tạo file ZIP. Thử lại sau.")

            elif sel_giai_doan == giai_doan_options[1]:
                if os.path.exists(g1_v_path) and os.path.getsize(g1_v_path) > 0:
                    render_video(g1_v_path)
                    dg1_col1, dg1_col2 = st.columns(2)
                    with dg1_col1:
                        with open(g1_v_path, "rb") as f:
                            st.download_button("📥 Tải video Giai đoạn 1", f, "processed_video_g1.mp4", "video/mp4", width="stretch", key=f"dl_v_g1_{key_suffix}")
                    with dg1_col2:
                        frames_zip_g1 = st.session_state.get(f'frames_zip_g1_{key_suffix}')
                        if frames_zip_g1 and os.path.exists(frames_zip_g1):
                            with open(frames_zip_g1, "rb") as fzip:
                                st.download_button("📦 Tải ảnh Giai đoạn 1 (ZIP)", fzip, "frames_g1.zip", "application/zip", width="stretch", key=f"dl_zip_g1_{key_suffix}")
                        else:
                            if st.button("📦 Chuẩn bị file ZIP ảnh G1", width="stretch", key=f"btn_prep_zip_g1_{key_suffix}"):
                                with st.spinner("🔄 Đang nén khung hình G1..."):
                                    new_zip = create_zip_of_frames(all_frames_data[n0:n1], processed_video_path)
                                    if new_zip:
                                        st.session_state[f'frames_zip_g1_{key_suffix}'] = new_zip
                                        st.rerun()
                                    else:
                                        st.error("❌ Lỗi tạo file ZIP. Thử lại sau.")
                else:
                    st.info("ℹ️ Không tìm thấy video Giai đoạn 1 hoặc lỗi cắt phân đoạn.")

            elif sel_giai_doan == giai_doan_options[2]:
                if os.path.exists(g2_v_path) and os.path.getsize(g2_v_path) > 0:
                    render_video(g2_v_path)
                    dg2_col1, dg2_col2 = st.columns(2)
                    with dg2_col1:
                        with open(g2_v_path, "rb") as f:
                            st.download_button("📥 Tải video Giai đoạn 2", f, "processed_video_g2.mp4", "video/mp4", width="stretch", key=f"dl_v_g2_{key_suffix}")
                    with dg2_col2:
                        frames_zip_g2 = st.session_state.get(f'frames_zip_g2_{key_suffix}')
                        if frames_zip_g2 and os.path.exists(frames_zip_g2):
                            with open(frames_zip_g2, "rb") as fzip:
                                st.download_button("📦 Tải ảnh Giai đoạn 2 (ZIP)", fzip, "frames_g2.zip", "application/zip", width="stretch", key=f"dl_zip_g2_{key_suffix}")
                        else:
                            if st.button("📦 Chuẩn bị file ZIP ảnh G2", width="stretch", key=f"btn_prep_zip_g2_{key_suffix}"):
                                with st.spinner("🔄 Đang nén khung hình G2..."):
                                    new_zip = create_zip_of_frames(all_frames_data[n1:n2], processed_video_path)
                                    if new_zip:
                                        st.session_state[f'frames_zip_g2_{key_suffix}'] = new_zip
                                        st.rerun()
                                    else:
                                        st.error("❌ Lỗi tạo file ZIP. Thử lại sau.")
                else:
                    st.info("ℹ️ Không tìm thấy video Giai đoạn 2 hoặc lỗi cắt phân đoạn.")

            elif sel_giai_doan == giai_doan_options[3]:
                if os.path.exists(g3_v_path) and os.path.getsize(g3_v_path) > 0:
                    render_video(g3_v_path)
                    dg3_col1, dg3_col2 = st.columns(2)
                    with dg3_col1:
                        with open(g3_v_path, "rb") as f:
                            st.download_button("📥 Tải video Giai đoạn 3", f, "processed_video_g3.mp4", "video/mp4", width="stretch", key=f"dl_v_g3_{key_suffix}")
                    with dg3_col2:
                        frames_zip_g3 = st.session_state.get(f'frames_zip_g3_{key_suffix}')
                        if frames_zip_g3 and os.path.exists(frames_zip_g3):
                            with open(frames_zip_g3, "rb") as fzip:
                                st.download_button("📦 Tải ảnh Giai đoạn 3 (ZIP)", fzip, "frames_g3.zip", "application/zip", width="stretch", key=f"dl_zip_g3_{key_suffix}")
                        else:
                            if st.button("📦 Chuẩn bị file ZIP ảnh G3", width="stretch", key=f"btn_prep_zip_g3_{key_suffix}"):
                                with st.spinner("🔄 Đang nén khung hình G3..."):
                                    new_zip = create_zip_of_frames(all_frames_data[n2:n3], processed_video_path)
                                    if new_zip:
                                        st.session_state[f'frames_zip_g3_{key_suffix}'] = new_zip
                                        st.rerun()
                                    else:
                                        st.error("❌ Lỗi tạo file ZIP. Thử lại sau.")
                else:
                    st.info("ℹ️ Không tìm thấy video Giai đoạn 3 hoặc lỗi cắt phân đoạn.")
        else:
            st.info("ℹ️ Video trích xuất chưa có sẵn local. Bấm nút dưới để tải khi cần xem video.")
            if st.button("⚡ Tải video khung xương", key=f"btn_lazy_processed_video_{key_suffix}", use_container_width=True):
                with st.spinner("📥 Đang tải video khung xương..."):
                    got = dam_bao_tai_video_phan_tich(processed_video_path)
                    if got:
                        st.rerun()
                    else:
                        st.warning("Chưa có file video trên Cloud. Bảng số liệu và biểu đồ bên cạnh vẫn xem được.")

    with v_col2:
        is_light = st.session_state.theme == 'light'
        v_stats_bg = "#ffffff" if is_light else "rgba(15, 23, 42, 0.6)"
        v_stats_border = "#eee" if is_light else "rgba(100, 116, 139, 0.2)"
        v_stats_text = "#000000" if is_light else "#ffffff"

        # Tạo phần HTML hiển thị độ chính xác
        if (acc_g1 > 0 or acc_g2 > 0 or acc_g3 > 0) and not is_gay_ex:
            accuracy_html = (
                f"<div style='margin-bottom:10px;'>"
                f"<b>Độ chính xác 3 giai đoạn:</b>"
                f"<ul style='margin: 5px 0 0 10px; padding: 0; list-style-type: none;'>"
                f"<li style='margin-bottom:3px;'>🌱 GĐ 1 (ss±{PHASE_ERROR['g1']}°): <b style='color:#22c55e;'>{acc_g1:.1f}%</b></li>"
                f"<li style='margin-bottom:3px;'>📈 GĐ 2 (ss±{PHASE_ERROR['g2']}°): <b style='color:#eab308;'>{acc_g2:.1f}%</b></li>"
                f"<li style='margin-bottom:3px;'>🎯 GĐ 3 (ss±{PHASE_ERROR['g3']}°): <b style='color:#ef4444;'>{acc_g3:.1f}%</b></li>"
                f"</ul>"
                f"</div>"
            )
        else:
            accuracy_html = f"<div style='margin-bottom:10px;'><b>Độ chính xác:</b> <span style='color:#22c55e; font-size:1.2rem; font-weight:bold;'>{ai_acc:.1f}%</span></div>"

        st.markdown(f"""<div style='background: {v_stats_bg}; border: 1px solid {v_stats_border}; border-radius: 16px; padding: 20px; color: {v_stats_text}; box-shadow: 0 4px 15px rgba(0,0,0,{"0.05" if is_light else "0.3"});'>
<h4 style='color:#38bdf8; margin-top:0;'>📊 Thông số Video</h4>
<div style='margin-bottom:10px;'><b>Tên:</b> {filename_html}</div>
{accuracy_html}
<div style='margin-bottom:10px;'><b>Thời lượng:</b> {total_frames} frames</div>
<hr style='opacity:0.1; margin:15px 0;'>
<div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
<span>✅ PASS:</span> <b>{pass_count}</b>
</div>
<div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
<span>⚠️ NEARLY:</span> <b>{nearly_count}</b>
</div>
<div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
<span>❌ FAIL:</span> <b>{fail_count}</b>
</div>
<hr style='opacity:0.15; margin:10px 0;'>
<div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
<span>👤 Không nhận dạng BN:</span> <b style='color:#f97316;'>{sum(1 for f in all_frames_data if f.get('goc_vai') is None and f.get('goc_vai_trai') is None and f.get('goc_vai_phai') is None)}</b>
</div>
<div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
<span>🚫 Lọc người lạ:</span> <b style='color:#a855f7;'>{sum(1 for f in all_frames_data if f.get('filtered_stranger', False))}</b>
</div>
<div style='font-size:0.75rem; color:#94a3b8; margin-top:4px;'>ℹ️ AI chỉ theo dõi bệnh nhân đầu tiên phát hiện</div>
</div>""", unsafe_allow_html=True)

        if user_role == "Nghi\u00ean c\u1ee9u vi\u00ean":
            # Hi\u1ec3n th\u1ecb t\u00f3m t\u1eaft \u00e2m thanh AI
            st.write("")
            _n_dung_audio = sum(1 for f in all_frames_data if f.get('dung'))
            _n_gan_audio = sum(1 for f in all_frames_data if f.get('gan_dung') and not f.get('dung'))
            _n_sai_audio = total_frames - _n_dung_audio - _n_gan_audio
            _pct_d = _n_dung_audio / total_frames * 100 if total_frames > 0 else 0
            _pct_g = _n_gan_audio / total_frames * 100 if total_frames > 0 else 0
            _pct_s = _n_sai_audio / total_frames * 100 if total_frames > 0 else 0
            _play_path_audio = playback_video_path or processed_video_path
            _has_audio_track = bool(
                _play_path_audio and os.path.exists(_play_path_audio)
                and video_has_audio_track(_play_path_audio)
            )
            _audio_badge = (
                "**:green[Co giong AI]**" if _has_audio_track
                else "**:orange[Chua co giong AI]**"
            )
            st.markdown(f"**:blue[Phan hoi am thanh AI]** {_audio_badge}")
            a1, a2, a3 = st.columns(3)
            a1.metric(":green[Dung]", _n_dung_audio, f"{_pct_d:.0f}%")
            a2.metric(":orange[Gan dung]", _n_gan_audio, f"{_pct_g:.0f}%")
            a3.metric(":red[Sai]", _n_sai_audio, f"{_pct_s:.0f}%")
            if not _has_audio_track:
                st.caption("Bam **Chay phan tich moi** de nhung giong AI vao video.")
            st.write("")
            # Nut chay lai phan tich moi
            _v_for_reanalyze = st.session_state.get("current_eval_video")
            if _v_for_reanalyze:
                hien_thi_nut_tai_lai_va_phan_tich_moi(_v_for_reanalyze, key_suffix=f"video_col2_{key_suffix}")
            st.write("")
            btn_label = "GUI BAO CAO PHAN TICH CHO BS & BN" if is_gay_ex else "GUI BAO CAO TONG HOP 3 GIAI DOAN CHO BS & BN"
            if st.button(f"📤 {btn_label}", key=f"btn_send_ncv_3_stages_{key_suffix}", use_container_width=True, type="primary"):
                if gui_bao_cao_tong_hop_3_giai_doan():
                    v_meta = st.session_state.get('current_eval_video') or {}
                    pname = v_meta.get('full_name', 'Benh nhan')
                    msg = f"Da gui bao cao phan tich cho BN {pname}!" if is_gay_ex else f"Da gui bao cao tong hop 3 giai doan cho BN {pname}!"
                    st.success(f"✅ {msg}")
                    st.balloons()
                    st.rerun()

    st.markdown("---")


    # ================================================================
    # PHẦN HIỂN THỊ KHUNG HÌNH TRÍCH XUẤT (NCV)
    # ================================================================
    if is_gay_ex:
        st.markdown("### 📷 KHUNG HÌNH TRÍCH XUẤT")
    else:
        st.markdown("### 📷 KHUNG HÌNH TRÍCH XUẤT — PHÂN LOẠI THEO 3 GIAI ĐOẠN")

    with st.expander("📖 Giải thích nhãn REF (PASS) và ML (%) — bấm để xem", expanded=False):
        st.markdown(
            """
**Mỗi khung hình có 2 nhãn độc lập:**

| Nhãn | Cách chấm | Ý nghĩa |
|------|-----------|---------|
| **PASS / NEAR / FAIL** (REF) | So góc vai & khuỷu với **tư thế gần nhất** trong video mẫu YouTube (không khớp từng giây) | **PASS**: Δ vai **và** Δ khuỷu ≤ **sai số cho phép** (G1 **±45°**, G2 **±30°**, G3 **±15°**) · **NEAR**: Δ ≤ sai số × **1.5** · **FAIL**: vượt NEAR |
| **ML · Đúng / Gần đúng / Sai** | Mô hình RandomForest học từ dữ liệu các video đã phân tích | Chọn **1 trong 3 lớp** có xác suất cao nhất (không dùng ngưỡng % cố định kiểu 80/60) |

**Con số % bên cạnh ML** = **độ tin cậy vào đúng nhãn ML đang hiển thị** (ví dụ *Gần đúng · tin cậy 42%* = mô hình 42% chắc frame thuộc lớp *Gần đúng*).

| Mức tin cậy ML | Ý nghĩa khi đọc kết quả |
|----------------|-------------------------|
| **≥ 70%** | Tin cậy cao — có thể tham khảo mạnh |
| **50–69%** | Tin cậy vừa — nên xem kèm nhãn REF và góc Δ |
| **< 50%** | Không chắc chắn — mô hình phân vân giữa các lớp |

**Ví dụ ảnh của bạn:** `PASS` (REF: góc đạt ngưỡng) nhưng `ML · Gần đúng · tin cậy 33%` (ML thấy tư thế gần đúng và **không chắc** — dưới 50%).

Dòng **Xác suất 3 lớp** (nếu có): tổng ~100%, cho biết mô hình phân bố giữa Sai / Gần đúng / Đúng.
            """
        )

    # Hàm helper tính G1/G2/G3 status cho một frame_data
    def _frame_phase_status(f_data, threshold):
        """Tính PASS/NEAR/FAIL cho frame theo ngưỡng sai số threshold"""
        if threshold is None:
            idx = f_data.get('index', 1) - 1
            if 'segment_bounds' in st.session_state and st.session_state.segment_bounds:
                n0, n1, n2, n3 = st.session_state.segment_bounds
                if n0 <= idx < n1:
                    threshold = PHASE_ERROR["g1"]
                elif n1 <= idx < n2:
                    threshold = PHASE_ERROR["g2"]
                elif n2 <= idx < n3:
                    threshold = PHASE_ERROR["g3"]
                else:
                    threshold = PHASE_ERROR["g2"]
            else:
                threshold = PHASE_ERROR["g2"]

        eval_info = f_data.get('eval_info', {})
        cv = eval_info.get('shoulder_ref', 90)
        ck = eval_info.get('elbow_ref', 170)

        if is_gay_ex:
            vt = f_data.get('goc_vai_trai')
            vp = f_data.get('goc_vai_phai')
            kt = f_data.get('goc_khuyu_trai')
            kp = f_data.get('goc_khuyu_phai')
            if vt is None or vp is None or kt is None or kp is None:
                return "FAIL"
            vd = (abs(vt - cv) <= threshold) and (abs(vp - cv) <= threshold)
            kd = (abs(kt - ck) <= threshold) and (abs(kp - ck) <= threshold)
            vn = (abs(vt - cv) <= threshold * 1.5) and (abs(vp - cv) <= threshold * 1.5)
            kn = (abs(kt - ck) <= threshold * 1.5) and (abs(kp - ck) <= threshold * 1.5)
            if vd and kd:
                return "PASS"
            elif vn and kn:
                return "NEAR"
            return "FAIL"
        else:
            goc_v = f_data.get('goc_vai')
            goc_k = f_data.get('goc_khuyu')
            if goc_v is None or goc_k is None:
                return "FAIL"
            vd = abs(goc_v - cv) <= threshold
            kd = abs(goc_k - ck) <= threshold
            vn = abs(goc_v - cv) <= threshold * 1.5
            kn = abs(goc_k - ck) <= threshold * 1.5
            if vd and kd:
                return "PASS"
            elif vn and kn:
                return "NEAR"
            return "FAIL"

    # Hàm helper render grid HTML frames
    def _render_frame_grid(indices_list, frame_data_list, quality_mode_val, tab_threshold, tab_key, key_suffix_val):
        render_frame_grid(
            deps,
            indices_list,
            frame_data_list,
            quality_mode_val,
            tab_threshold,
            tab_key,
            key_suffix_val,
            processed_video_path=processed_video_path,
            frame_phase_status=_frame_phase_status,
        )

    if is_gay_ex:
        st.info("📋 **Bài tập với gậy:** Đánh giá động thái khớp vai & khuỷu theo tư thế chuẩn tương đương.")
        ss_chuan = tk.get('sai_so', 30)
        _render_frame_grid(list(range(len(all_frames_data))), all_frames_data, None, ss_chuan, "all", key_suffix)
    else:
        # Lấy ranh giới phân đoạn đã tính toán ở trên
        if 'segment_bounds' not in st.session_state or st.session_state.get('last_processed_video_for_bounds') != processed_video_path:
            st.session_state.segment_bounds = segment_frames(all_frames_data)
            st.session_state.last_processed_video_for_bounds = processed_video_path

        n0, n1, n2, n3 = st.session_state.segment_bounds

        # Tất cả frame indices
        all_indices = list(range(len(all_frames_data)))

        # Phân chia chỉ số theo từng phân đoạn giai đoạn
        g1_indices = all_indices[n0:n1]
        g2_indices = all_indices[n1:n2]
        g3_indices = all_indices[n2:n3]

        # Tính trước số frame pass cho từng giai đoạn để hiển thị trên tiêu đề tab
        def _count_pass_segment(indices_list, threshold):
            return sum(1 for i in indices_list if _frame_phase_status(all_frames_data[i], threshold) == "PASS")

        g1_pass = _count_pass_segment(g1_indices, 45)
        g2_pass = _count_pass_segment(g2_indices, 30)
        g3_pass = _count_pass_segment(g3_indices, 15)

        tab_all, tab_g1, tab_g2, tab_g3 = st.tabs([
            f"📋 Tất cả ({total_frames})",
            f"🟢 G1 (Lượt 1: {len(g1_indices)} frames | {g1_pass} PASS)",
            f"🟡 G2 (Lượt 2 lặp: {len(g2_indices)} frames | {g2_pass} PASS)",
            f"🔴 G3 (Lượt 3 lặp: {len(g3_indices)} frames | {g3_pass} PASS)",
        ])

        with tab_all:
            st.caption("Hiển thị tất cả khung hình. Badge màu theo **giai đoạn mặc định** bạn đã chọn trước khi phân tích.")
            _render_frame_grid(all_indices, all_frames_data, None, None, "all", key_suffix)

        with tab_g1:
            st.info(f"🟢 **Giai đoạn 1 — Khởi đầu (Sai số ±{PHASE_ERROR['g1']}°):** Chỉ hiển thị các khung hình thuộc **Lượt tập 1**. Badge **PASS** = lệch chuẩn ≤ {PHASE_ERROR['g1']}°.")
            _render_frame_grid(g1_indices, all_frames_data, None, PHASE_ERROR["g1"], "g1", key_suffix)

        with tab_g2:
            st.info(f"🟡 **Giai đoạn 2 — Hồi phục (Sai số ±{PHASE_ERROR['g2']}°):** Chỉ hiển thị các khung hình thuộc **Lượt lặp lại lần 2**. Badge **PASS** = lệch chuẩn ≤ {PHASE_ERROR['g2']}°.")
            _render_frame_grid(g2_indices, all_frames_data, None, PHASE_ERROR["g2"], "g2", key_suffix)

        with tab_g3:
            st.info(f"🔴 **Giai đoạn 3 — Chuẩn xác (Sai số ±{PHASE_ERROR['g3']}°):** Chỉ hiển thị các khung hình thuộc **Lượt lặp lại lần 3**. Badge **PASS** = lệch chuẩn ≤ {PHASE_ERROR['g3']}°.")
            _render_frame_grid(g3_indices, all_frames_data, None, PHASE_ERROR["g3"], "g3", key_suffix)

    st.write("")  # Final spacer
