"""Researcher/NCV-specific Streamlit UI panels."""

from __future__ import annotations

from collections.abc import Callable

from ui.doctor import render_clinician_home


def _selected_index(options, current, default_index=0) -> int:
    try:
        return options.index(current)
    except ValueError:
        return default_index


def render_researcher_sidebar(
    st,
    *,
    researcher_name_html: str,
    phase_labels: dict[str, str],
    normalize_phase_selection: Callable[[object], str],
    stats: tuple[int, int, float],
    clear_progress: Callable[[], int],
) -> None:
    st.markdown("### 🔬 THÔNG TIN CHUYÊN GIA")
    st.markdown(f"""
    <div class="custom-card" style="padding: 10px; border-left: 5px solid #00c6ff; background: rgba(0, 198, 255, 0.05);">
        <p style="margin:0; font-weight:bold; color:#00c6ff;">👤 {researcher_name_html}</p>
        <p style="margin:0; font-size:0.8rem; color:#888;">Trường Đại học Y tế Công cộng</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ CẤU HÌNH AI & TỐC ĐỘ")
    st.slider("Độ tự tin tối thiểu (Confidence)", 0.0, 1.0, 0.5, key="ncv_confidence", help="Ngưỡng để AI chấp nhận một điểm khớp xương.")
    st.selectbox(
        "Tốc độ xử lý",
        options=[0, 1, 2, 4],
        index=_selected_index([0, 1, 2, 4], st.session_state.get("ncv_skip_frames", 0)),
        format_func=lambda x: "Tự động (theo độ dài video)" if x == 0 else f"Nhanh (Bỏ qua {x} frame)",
        key="ncv_skip_frames",
        help="0 = Tự động tối ưu theo độ dài video (video >100s tự bỏ frame). Chọn giá trị khác để ghi đè.",
    )
    st.selectbox(
        "Độ phân giải video (Video Quality)",
        options=[480, 720, 1080],
        index=_selected_index([480, 720, 1080], st.session_state.get("ncv_resize_width", 720), default_index=1),
        format_func=lambda x: "480p (Tốc độ tối ưu)" if x == 480 else ("720p (HD - Chuẩn sắc nét)" if x == 720 else "1080p (Full HD - Cực kỳ chuẩn xác)"),
        key="ncv_resize_width",
        help="Độ phân giải càng cao thì vẽ khung xương càng sắc nét và bám sát khớp bệnh nhân hơn.",
    )
    st.slider("Độ nhạy chuyển động (Sensitivity)", 0.0, 1.0, 0.7, key="ncv_sensitivity", help="Ảnh hưởng đến việc tính toán vận tốc khớp.")
    if "ncv_giai_doan" in st.session_state:
        st.session_state.ncv_giai_doan = normalize_phase_selection(st.session_state.ncv_giai_doan)
    st.selectbox(
        "🌱 Giai đoạn tập bệnh nhân (Mặc định video):",
        options=[phase_labels["g1"], phase_labels["g2"], phase_labels["g3"]],
        index=1,
        key="ncv_giai_doan",
        help="Điều chỉnh ngưỡng sai số để vẽ khung xương và phát âm thanh phản hồi trực tiếp khi xử lý video.",
    )

    st.markdown("### 📊 THỐNG KÊ HỆ THỐNG")
    total_vids, pending_ai, avg_acc = stats
    st.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);">
        <p style="margin:0; font-size:0.85rem; color: #aaa;">📁 Video chờ xử lý: <b style="color: #00c6ff;">{pending_ai}</b></p>
        <p style="margin:5px 0; font-size:0.85rem; color: #aaa;">🎯 Accuracy TB: <b style="color: #00ff00;">{avg_acc:.1f}%</b></p>
        <p style="margin:0; font-size:0.85rem; color: #aaa;">📚 Tổng dữ liệu: <b style="color: #ffd700;">{total_vids} Video</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 CHỌN MÔ HÌNH")
    model_options = ["MediaPipe Heavy", "MediaPipe Full", "MediaPipe Lite"]
    st.selectbox(
        "Mô hình Pose",
        options=model_options,
        index=_selected_index(model_options, st.session_state.get("ncv_model_type", "MediaPipe Heavy")),
        key="ncv_model_type",
        help=(
            "Heavy (Complexity 2): chính xác nhất — mặc định. "
            "Full (Complexity 1): cân bằng tốc độ/chính xác. "
            "Lite (Complexity 0): nhanh nhất."
        ),
    )

    st.markdown("### 🧹 LÀM MỚI TIẾN TRÌNH")
    st.caption("Hủy tất cả tiến trình đang chạy/đang chờ để bắt đầu phân tích lại từ đầu.")
    if st.button("🧹 HỦY TẤT CẢ & LÀM MỚI", key="sidebar_reset_progress", use_container_width=True, type="secondary"):
        try:
            n_removed = clear_progress()
            for key in ("reanalyze_triggered", "view_old_analysis", "has_data", "stats", "angle_df", "current_eval_video"):
                st.session_state.pop(key, None)
            st.toast(f"🧹 Đã làm mới — xóa {n_removed} tiến trình. Bạn có thể tải/phân tích lại từ đầu.", icon="✅")
            st.rerun()
        except PermissionError as exc:
            st.error(str(exc))


def render_researcher_tab(selected_tab: str, deps) -> None:
    st = deps.st
    if selected_tab == "🏠 TRANG CHỦ":
        render_clinician_home(st, deps, user_role="Nghiên cứu viên")
    elif selected_tab in ("📊 KẾT QUẢ", "📊 KẾT QUẢ ĐÁNH GIÁ"):
        deps.render_patient_results()
    elif selected_tab == "🔬 PHÂN TÍCH & TRÍCH XUẤT DỮ LIỆU":
        deps.render_researcher_analysis_video()
    elif selected_tab == "📚 THÔNG TIN TỔNG HỢP":
        deps.render_general_info("Nghiên cứu viên")
    elif selected_tab == "👥 HỒ SƠ ĐỀ TÀI & ĐỘI NGŨ CHUYÊN GIA":
        deps.render_research_profile_team()
    elif selected_tab == "📚 ĐỀ TÀI NCKH":
        deps.render_research_topic()
    elif selected_tab == "📄 THÔNG TIN NGHIÊN CỨU":
        deps.render_research_info()
    elif selected_tab == "👥 THÀNH VIÊN":
        deps.render_team()
    elif selected_tab == "💬 PHẢN HỒI":
        deps.render_feedback()
