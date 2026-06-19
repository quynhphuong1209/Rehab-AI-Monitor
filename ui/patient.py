"""Patient-specific Streamlit UI panels."""

from __future__ import annotations

import gc
import os
import shutil
import subprocess


def render_patient_sidebar(st, *, full_name_html: str) -> None:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(0, 198, 255, 0.08) 0%, rgba(0, 114, 255, 0.08) 100%);
                padding: 14px; border-radius: 12px; border: 1px solid rgba(0, 198, 255, 0.2); margin-bottom: 15px;">
        <p style="margin:0; font-weight:bold; color:#00c6ff; font-size: 1rem;">🏥 Xin chào, {full_name_html}!</p>
        <p style="margin:4px 0 0; font-size:0.8rem; color:#888;">Bệnh nhân - Hệ thống PHCN AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📚 HƯỚNG DẪN SỬ DỤNG")
    st.markdown("""
    <div style="font-size: 0.88rem; line-height: 1.7;">
    <p>👉 Hệ thống hỗ trợ bạn qua các Tab sau:</p>
    <p>🏠 <b>TRANG CHỦ</b><br>
    <span style="color:#aaa; font-size:0.8rem;">Khai báo thông tin, triệu chứng, chọn bài tập và tải video tập luyện lên cho Bác sĩ.</span></p>
    <p>📊 <b>KẾT QUẢ ĐÁNH GIÁ</b><br>
    <span style="color:#aaa; font-size:0.8rem;">Xem nhận xét của Bác sĩ/KTV và kết quả phân tích AI về chuyển động của bạn.</span></p>
    <p>⏰ <b>LỊCH NHẮC NHỞ</b><br>
    <span style="color:#aaa; font-size:0.8rem;">Xem lịch tái khám và các nhắc nhở tập luyện hàng ngày.</span></p>
    <p>📚 <b>THÔNG TIN</b><br>
    <span style="color:#aaa; font-size:0.8rem;">Tìm hiểu về bài tập phục hồi chức năng vai và các kiến thức y tế hữu ích.</span></p>
    <p>📞 <b>LIÊN HỆ</b><br>
    <span style="color:#aaa; font-size:0.8rem;">Thông tin liên hệ với Bác sĩ/KTV khi cần hỗ trợ khẩn cấp.</span></p>
    </div>
    """, unsafe_allow_html=True)


def _reset_large_video_session_state(st) -> None:
    old_video_path = st.session_state.get("processed_video_path")
    if old_video_path and os.path.exists(old_video_path):
        try:
            os.unlink(old_video_path)
            old_h264 = (
                old_video_path.replace("_f.mp4", ".mp4")
                if old_video_path.endswith("_f.mp4")
                else old_video_path.replace(".mp4", "_f.mp4")
            )
            if os.path.exists(old_h264):
                os.unlink(old_h264)
        except Exception:
            pass

    old_csv_path = st.session_state.get("current_df_csv_path")
    if old_csv_path and os.path.exists(old_csv_path):
        try:
            os.unlink(old_csv_path)
        except Exception:
            pass

    old_frames_dir = st.session_state.get("temp_frames_dir")
    if old_frames_dir and os.path.exists(old_frames_dir):
        try:
            shutil.rmtree(old_frames_dir, ignore_errors=True)
        except Exception:
            pass

    keys_to_clear = [
        "has_data",
        "stats",
        "angle_df",
        "processed_video_path",
        "all_frames_paths",
        "all_frames_data",
        "all_frames_data_path",
        "output_video_bytes",
        "processed_video_bytes",
        "frames_zip",
        "current_df_csv_path",
        "temp_frames_dir",
        "temp_video_file",
        "video_ready",
        "frames_ready",
        "frames_loaded",
        "current_page",
        "processing_result",
        "processing_progress",
        "processing_status",
        "exercise",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = [] if key in ("all_frames_paths", "all_frames_data") else None

    st.session_state.has_data = False
    st.session_state.video_ready = False
    st.session_state.frames_ready = False
    st.session_state.frames_loaded = False
    st.session_state.current_page = 1
    st.session_state.processing_progress = 0
    st.session_state.processing_status = ""
    gc.collect()
    st.toast("🧹 Đã giải phóng bộ nhớ từ phân tích trước.", icon="💾")


def _render_patient_intake_and_exercise(st, deps) -> dict:
    bai_tap_map = deps.BAI_TAP
    st.markdown("## 📝 THÔNG TIN KHÁM & TẬP LUYỆN")

    with st.container(border=True):
        st.markdown("### 📋 THÔNG TIN NGƯỜI DÙNG")
        bn_col1, bn_col2 = st.columns(2)
        with bn_col1:
            ten_nguoi_dung = st.text_input(
                "Họ và tên (*)",
                value=st.session_state.user_info.get("full_name", ""),
                placeholder="VD: Nguyễn Văn A",
                key="bn_tab_ten",
            )
            tuoi = st.number_input("Tuổi (*)", 0, 120, 22, key="bn_tab_tuoi")
        with bn_col2:
            ma_nguoi_dung = st.text_input(
                "Mã số định danh (*)",
                placeholder="VD: BN0001",
                key="bn_tab_ma",
            )
            gioi_tinh = st.selectbox("Giới tính (*)", ["", "Nam", "Nữ"], key="bn_tab_gt")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### 🩺 KHAI BÁO TRIỆU CHỨNG")
        s_desc = st.text_area(
            "Mô tả cảm giác đau:",
            placeholder="VD: Đau nhói ở khớp vai khi nâng tay lên cao...",
            height=100,
            key="bn_tab_desc",
        )
        s_vas = st.select_slider(
            "📊 Mức độ đau (VAS 0-10):",
            options=list(range(11)),
            value=3,
            key="bn_tab_vas",
        )
        vas_labels = {
            0: "Đ 0: Không đau",
            1: "Đ 1-3: Đau nhẹ",
            4: "Đ 4-6: Đau vừa",
            7: "Đ 7-9: Đau nặng",
            10: "Đ 10: Đau dữ dội",
        }
        closest = min(vas_labels, key=lambda x: abs(x - s_vas))
        st.caption(f"💡 {vas_labels[closest]}")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### 🎯 CHỌN BÀI TẬP VÀ XEM HƯỚNG DẪN")
        ma_bai_tap = st.selectbox(
            "🎯 Chọn bài tập",
            list(bai_tap_map.keys()),
            format_func=lambda x: f"{bai_tap_map[x]['icon']} {bai_tap_map[x]['ten']}",
            key="bn_tab_bt",
        )
        bai_tap = bai_tap_map[ma_bai_tap]

        is_light = st.session_state.theme == "light"
        info_bg = "rgba(255, 255, 255, 1)" if is_light else "rgba(255, 255, 255, 0.04)"
        info_border = "#eee" if is_light else "rgba(255, 255, 255, 0.1)"
        info_text = "#000" if is_light else "#fff"

        ex_col1, ex_col2 = st.columns([3, 2])
        with ex_col1:
            exercise_icon_html = deps.safe_html(bai_tap.get("icon", ""), max_length=20)
            exercise_name_html = deps.safe_html(bai_tap.get("ten", ""), max_length=160)
            exercise_desc_html = deps.safe_html(bai_tap.get("mo_ta", ""), max_length=800)
            exercise_duration_html = deps.safe_html(bai_tap.get("thoi_gian", ""), max_length=30)
            exercise_reps_html = deps.safe_html(bai_tap.get("lan", ""), max_length=30)
            st.markdown(f"""
            <div class="info-box" style="background: {info_bg}; border: 1px solid {info_border}; color: {info_text}; padding: 15px; border-radius: 10px;">
                <h3 style="margin-top:0;">{exercise_icon_html} {exercise_name_html}</h3>
                <p>{exercise_desc_html}</p>
                <div style="display: flex; gap: 20px; font-size: 0.9rem; opacity: 0.8;">
                    <span>⏱️ <b>Thời gian:</b> {exercise_duration_html}s/lần</span>
                    <span>🔄 <b>Số lần:</b> {exercise_reps_html} lần/ngày</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📖 HƯỚNG DẪN TẬP LUYỆN", expanded=True):
                st.markdown(bai_tap["huong_dan"])
            with st.expander("✨ LỢI ÍCH CỦA BÀI TẬP", expanded=False):
                for loi_ich in bai_tap["loi_ich"]:
                    st.markdown(f"- {loi_ich}")

        with ex_col2:
            card_bg = "#ffffff" if is_light else "rgba(26,26,46,0.8)"
            st.markdown(f"""
            <div class="custom-card" style="background: {card_bg}; padding: 15px; border-radius: 10px; border: 1px solid {info_border};">
                <h4 style="color:{'#0072ff' if is_light else '#fff'}; margin-top:0;">🎯 ĐỐI CHIẾU VIDEO CHUẨN</h4>
                <p style="color:#00CED1; margin-bottom:8px; font-size:0.9rem;">⚡ Hệ thống tự động so sánh chuyển động của bạn với <b>Video chuẩn</b>.</p>
                <p style="color:#FF6B6B; margin-bottom:10px; font-size:0.9rem;">📊 Độ chính xác dựa trên sai số Euclidean và biên độ khớp.</p>
                <div style="font-size:0.85rem; border-top:1px solid {info_border}; padding-top:10px;">
                    <p style="margin-bottom:5px;">✅ <b>Đạt:</b> Chuyển động khớp với video mẫu.</p>
                    <p style="margin-bottom:0;">❌ <b>Cần cải thiện:</b> Động tác sai lệch.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if "video_guide" in bai_tap:
                st.markdown("### 🎬 VIDEO HƯỚNG DẪN")
                deps.render_video(bai_tap["video_guide"])
            elif bai_tap.get("youtube"):
                st.markdown("### 📺 VIDEO YOUTUBE THAM KHẢO")
                st.video(bai_tap["youtube"])
            if "video_guide" in bai_tap and bai_tap.get("youtube"):
                st.markdown("### 📺 VIDEO YOUTUBE THAM KHẢO")
                st.video(bai_tap["youtube"])

    st.markdown("---")
    if st.button("📤 GỬi THÔNG TIN CHO BÁC SĨ/KTV VÀ NCV", type="primary", width="stretch"):
        if ten_nguoi_dung and ma_nguoi_dung and gioi_tinh != "" and s_desc and ma_bai_tap:
            try:
                actor = deps.require_role(
                    deps.PATIENT_ROLE,
                    action="create_symptom",
                    target=st.session_state.user_info["username"],
                )
                deps.require_patient_scope(actor["username"], action="create_symptom")
                s_data = deps.load_data(deps.SYMPTOMS_FILE)
                s_data.append({
                    "username": st.session_state.user_info["username"],
                    "full_name": ten_nguoi_dung,
                    "patient_id": ma_nguoi_dung,
                    "age": tuoi,
                    "gender": gioi_tinh,
                    "exercise": bai_tap_map[ma_bai_tap]["ten"],
                    "symptoms": s_desc,
                    "vas": s_vas,
                    "time": deps.get_vn_now().strftime("%H:%M - %d/%m/%Y"),
                })
                deps.save_data(deps.SYMPTOMS_FILE, s_data)
                st.success("✅ Đã gửi thông tin đầy đủ cho BÁC SĨ - KTV và NCV thành công!")
                st.balloons()
            except PermissionError as exc:
                st.error(str(exc))
        else:
            st.warning("⚠️ Vui lòng điền đầy đủ các thông tin: Họ tên, Mã định danh, Giới tính, Bài tập và Mô tả triệu chứng.")

    return {
        "ten_nguoi_dung": ten_nguoi_dung,
        "bai_tap": bai_tap,
    }


def _persist_patient_uploaded_video(st, deps, file_upload, upload_state: dict) -> None:
    save_dir = deps.UPLOAD_DIR
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass

    timestamp = deps.get_vn_now().strftime("%Y%m%d_%H%M%S")
    safe_upload_name = deps.sanitize_filename(file_upload.name, fallback="video.mp4")
    base_name, _ = os.path.splitext(safe_upload_name)
    orig_ext = os.path.splitext(safe_upload_name)[1].lower() or ".mp4"
    safe_username = deps.sanitize_filename(st.session_state.user_info["username"], fallback="user")
    filename = f"{safe_username}_{timestamp}_{base_name}{orig_ext}"
    file_path = os.path.join(save_dir, filename)

    temp_uploaded_path = file_path + "_temp" + orig_ext
    with open(temp_uploaded_path, "wb") as handle:
        handle.write(file_upload.getbuffer())
    probe_ok, probe_msg = deps.validate_video_file_for_processing(temp_uploaded_path)
    if not probe_ok:
        try:
            os.remove(temp_uploaded_path)
        except Exception:
            pass
        st.error(f"🚫 {probe_msg}")
        st.stop()

    v_codec = None
    a_codec = None
    try:
        v_codec, a_codec = deps.get_video_codec(temp_uploaded_path)
    except Exception:
        pass

    is_h264_mp4 = v_codec == "h264" and orig_ext == ".mp4"
    if is_h264_mp4:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        os.rename(temp_uploaded_path, file_path)
        print(f"[Upload Optimization] Video {file_upload.name} đã là H.264 MP4, lưu trực tiếp không cần convert.")
    else:
        file_path_mp4 = file_path.rsplit(".", 1)[0] + ".mp4"
        try:
            cmd = deps.build_upload_h264_command(
                temp_uploaded_path,
                file_path_mp4,
                has_audio=bool(a_codec),
                ffmpeg_threads=deps.MAX_FFMPEG_THREADS,
            )
            result_compress = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
            )
            if (
                result_compress.returncode == 0
                and os.path.exists(file_path_mp4)
                and os.path.getsize(file_path_mp4) > 1024
            ):
                try:
                    os.remove(temp_uploaded_path)
                except Exception:
                    pass
                file_path = file_path_mp4
                print(f"[Upload Optimization] Đã convert thành công {file_upload.name} sang H.264 MP4.")
            else:
                if os.path.exists(file_path_mp4):
                    try:
                        os.remove(file_path_mp4)
                    except Exception:
                        pass
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                os.rename(temp_uploaded_path, file_path)
        except Exception as compress_err:
            print(f"[Compress Upload] Lỗi nén video: {compress_err}")
            if os.path.exists(temp_uploaded_path):
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                os.rename(temp_uploaded_path, file_path)

    deps.push_file_to_hf_async(file_path)

    video_list = deps.load_data(deps.VIDEOS_FILE)
    bai_tap = upload_state["bai_tap"]
    video_list.append({
        "username": st.session_state.user_info["username"],
        "full_name": upload_state["ten_nguoi_dung"],
        "video_name": file_upload.name,
        "original_filename": file_upload.name,
        "stored_filename": os.path.basename(file_path),
        "exercise": bai_tap["ten"],
        "accuracy": 0,
        "time": deps.get_vn_now().strftime("%H:%M - %d/%m/%Y"),
        "video_path": file_path,
        "processed_path": None,
        "status": "Chờ NCV phân tích",
    })
    deps.save_data(deps.VIDEOS_FILE, video_list)
    st.success("✅ Đã gửi video cho BÁC SĨ - KTV và NCV thành công! Chuyên gia sẽ xem và đánh giá bài tập của bạn.")
    st.balloons()

    st.session_state.active_video_name = file_upload.name
    st.session_state.fresh_session = True
    st.session_state.has_data = False
    st.rerun()


def _render_patient_video_upload(st, deps, upload_state: dict) -> None:
    st.markdown("---")
    if "uploader_id" not in st.session_state:
        st.session_state.uploader_id = 0

    st.markdown("### 📤 TẢI LÊN VIDEO TẬP LUYỆN")
    st.info(f"📁 Hỗ trợ upload file tối đa {deps.MAX_FILE_SIZE_MB}MB (MP4, MOV, AVI, MKV)")
    file_upload = st.file_uploader(
        "Tải lên video của bạn để gửi cho Bác sĩ/NCV",
        type=["mp4", "mov", "avi", "mkv", "MP4", "MOV"],
        help=f"Dung lượng tối đa {deps.MAX_FILE_SIZE_MB}MB",
        key=f"video_uploader_v{st.session_state.uploader_id}",
    )

    upload_ok = False
    upload_msg = "Video chưa được kiểm tra hoặc đang trong phiên xử lý khác."
    if file_upload is not None and not st.session_state.processing:
        if st.session_state.get("uploaded_file_name") != file_upload.name:
            _reset_large_video_session_state(st)
        upload_ok, upload_msg = deps.validate_uploaded_video_file(file_upload)
        if upload_ok:
            st.success(f"✅ Đã chọn file: {file_upload.name} ({file_upload.size / (1024 * 1024):.2f} MB)")
        else:
            st.error(f"🚫 {upload_msg}")

    if st.button("📤 GỬI VIDEO CHO BÁC SĨ - KTV VÀ NCV", width="stretch", type="primary"):
        try:
            actor = deps.require_role(
                deps.PATIENT_ROLE,
                action="upload_video",
                target=st.session_state.user_info.get("username"),
            )
            deps.require_patient_scope(actor["username"], action="upload_video")
        except PermissionError as exc:
            st.error(str(exc))
            st.stop()

        if file_upload is None:
            st.error("🚨 Bạn chưa chọn video hoặc file video không hợp lệ. Vui lòng tải lên video trước khi gửi!")
        elif not upload_ok:
            st.error(f"🚫 {upload_msg}")
        else:
            _persist_patient_uploaded_video(st, deps, file_upload, upload_state)

    if st.session_state.processing:
        st.warning("⏳ Đang xử lý video, vui lòng chờ...")
        if st.button("❌ Hủy xử lý", width="stretch"):
            st.session_state.processing = False
            st.rerun()
    elif st.session_state.has_data:
        st.success("✅ Đã có kết quả phân tích! Hãy xem các tab PHÂN TÍCH và VIDEO & ẢNH.")
        st.session_state.processing = False


def _render_patient_research_flow(st) -> None:
    st.markdown("---")
    st.markdown("<h3 style='color: #00c6ff; text-align: center; margin-bottom: 25px;'>⚙️ QUY TRÌNH XỬ LÝ DỮ LIỆU NCKH</h3>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .step-container {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .step-box {
        flex: 1;
        min-width: 200px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 198, 255, 0.3);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: all 0.3s;
        border-top: 3px solid #00c6ff;
    }
    .step-box:hover {
        transform: translateY(-5px);
        background: rgba(0, 198, 255, 0.08);
        border-color: #00c6ff;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    .step-num {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .step-txt-title {
        color: #00c6ff;
        font-weight: bold;
        font-size: 1rem;
        margin-bottom: 8px;
        display: block;
    }
    .step-txt-desc {
        color: #aaa;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("1", "📸 GHI HÌNH", "Camera đặt ngang vai (90°), tối thiểu 30 FPS, đủ ánh sáng."),
        ("2", "⚙️ TRÍCH XUẤT", "Sử dụng MediaPipe Heavy trích xuất 33 điểm Landmarks."),
        ("3", "📊 PHÂN TÍCH", "Tính toán Vector góc Vai/Khuỷu và làm mượt dữ liệu."),
        ("4", "💾 LƯU TRỮ", "Số hóa dữ liệu sang JSON/CSV phục vụ báo cáo NCKH."),
    ]
    for col, (num, title, desc) in zip((c1, c2, c3, c4), cards):
        with col:
            st.markdown(f"""
            <div class="step-box">
                <div class="step-num">{num}</div>
                <span class="step-txt-title">{title}</span>
                <p class="step-txt-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def render_patient_home(st, deps) -> None:
    upload_state = _render_patient_intake_and_exercise(st, deps)
    _render_patient_video_upload(st, deps, upload_state)
    _render_patient_research_flow(st)


def render_patient_tab(selected_tab: str, deps) -> None:
    st = deps.st
    if selected_tab == "🏠 TRANG CHỦ":
        render_patient_home(st, deps)
    elif selected_tab in ("📊 KẾT QUẢ", "📊 KẾT QUẢ ĐÁNH GIÁ"):
        deps.render_patient_results()
    elif selected_tab == "🩺 KHAI BÁO TRIỆU CHỨNG":
        deps.render_symptoms_tab()
    elif selected_tab == "⏰ LỊCH NHẮC NHỞ":
        deps.render_reminders()
    elif selected_tab == "📚 THÔNG TIN TỔNG HỢP":
        deps.render_patient_info()
    elif selected_tab == "📞 THÔNG TIN LIÊN HỆ":
        deps.render_contact()
    elif selected_tab == "💬 PHẢN HỒI":
        deps.render_feedback()
    elif selected_tab == "📄 PHIẾU NCKH":
        deps.render_research_form()
