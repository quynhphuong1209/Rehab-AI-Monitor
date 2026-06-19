"""Doctor/KTV-specific Streamlit UI panels."""

from __future__ import annotations

import os


def render_doctor_sidebar(
    st,
    *,
    doctor_name_html: str,
    pending_eval: int,
    total_patients: int,
) -> None:
    st.markdown("### 🩺 HỒ SƠ CHUYÊN GIA")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(0, 198, 255, 0.1) 0%, rgba(0, 114, 255, 0.1) 100%);
                padding: 15px; border-radius: 12px; border: 1px solid rgba(0, 198, 255, 0.2); margin-bottom: 10px;">
        <p style="margin:0; font-weight:bold; color:#00c6ff; font-size: 1.05rem;">👨‍⚕️ {doctor_name_html}</p>
        <p style="margin:0; font-size:0.8rem; color:#888; margin-top: 4px;">Chuyên gia Phục hồi chức năng</p>
        <hr style="margin: 10px 0; border: 0; border-top: 1px solid rgba(0, 198, 255, 0.2);">
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #aaa;">
            <span>Cơ sở:</span>
            <span style="color: #fff;">ĐH Y tế Công cộng</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display: flex; gap: 8px; margin-bottom: 20px;">
        <div style="flex:1; background: rgba(255,255,255,0.03); padding: 12px 8px; border-radius: 10px; text-align: center; border: 1px solid rgba(255,255,255,0.05);">
            <p style="margin:0; font-size: 0.65rem; color: #888; font-weight: bold;">CHỜ ĐÁNH GIÁ</p>
            <p style="margin:5px 0 0; font-size: 1.3rem; font-weight: bold; color: #ff4b4b;">{pending_eval}</p>
        </div>
        <div style="flex:1; background: rgba(255,255,255,0.03); padding: 12px 8px; border-radius: 10px; text-align: center; border: 1px solid rgba(255,255,255,0.05);">
            <p style="margin:0; font-size: 0.65rem; color: #888; font-weight: bold;">TỔNG BỆNH NHÂN</p>
            <p style="margin:5px 0 0; font-size: 1.3rem; font-weight: bold; color: #00c6ff;">{total_patients}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_clinician_symptom_list(st, deps, *, user_role: str) -> None:
    st.markdown("### 👥 DANH SÁCH TRIỆU CHỨNG BN MỚI NHẤT")
    symptoms_data = deps.scope_records_for_current_actor(deps.load_data(deps.SYMPTOMS_FILE))
    symptoms_view = deps.researcher_view_records(symptoms_data)
    if not symptoms_view:
        st.info("ℹ️ Hiện chưa có thông tin khai báo triệu chứng mới từ bệnh nhân.")
        return

    grouped_symptoms = {}
    for item in symptoms_view:
        key = item.get("patient_id") or item.get("subject_code") or item.get("full_name") or item.get("username")
        if key not in grouped_symptoms:
            grouped_symptoms[key] = {
                "full_name": item.get("full_name") or item.get("subject_code") or item.get("username", "Không rõ"),
                "patient_id": item.get("patient_id") or item.get("subject_code") or "N/A",
                "age": item.get("age", "N/A"),
                "gender": item.get("gender", "N/A"),
                "symptoms": item.get("symptoms", ""),
                "vas": item.get("vas", "N/A"),
                "time": item.get("time", ""),
                "exercises": [item.get("exercise", "N/A")],
            }
        else:
            exercise = item.get("exercise", "N/A")
            if exercise not in grouped_symptoms[key]["exercises"]:
                grouped_symptoms[key]["exercises"].append(exercise)
            grouped_symptoms[key]["time"] = item.get("time", grouped_symptoms[key]["time"])
            grouped_symptoms[key]["vas"] = item.get("vas", grouped_symptoms[key]["vas"])

    display_list = list(reversed(list(grouped_symptoms.values())))[:4]
    symp_cols = st.columns(3)
    for index, symptom in enumerate(display_list):
        with symp_cols[index % 3]:
            with st.container(border=True):
                st.markdown(f"**👤 {symptom.get('full_name', 'Không rõ')}**")
                st.caption(f"🕒 {deps.format_vn_time(symptom.get('time'), default='N/A')}")
                st.write(f"**Đau (VAS):** {symptom.get('vas', 'N/A')}/10")
                with st.expander("Chi tiết triệu chứng"):
                    st.write(f"**Tuổi:** {symptom.get('age', 'N/A')} | **Mã:** {symptom.get('patient_id', 'N/A')}")
                    st.write(f"**Bài tập đã chọn:** {', '.join(symptom.get('exercises', []))}")
                    if symptom.get("symptoms"):
                        st.info(symptom.get("symptoms"))
                    if user_role == "Bác sĩ / KTV PHCN" and st.button("Xóa thông báo", key=f"del_symp_main_{index}"):
                        symptom_id = symptom.get("patient_id")
                        symptom_name = symptom.get("full_name")
                        all_symptoms = deps.load_data(deps.SYMPTOMS_FILE)
                        new_symptoms = [
                            item
                            for item in all_symptoms
                            if item.get("patient_id") != symptom_id and item.get("full_name") != symptom_name
                        ]
                        deps.save_data(deps.SYMPTOMS_FILE, new_symptoms)
                        st.rerun()


def render_clinician_video_list(st, deps, *, user_role: str) -> None:
    st.markdown("---")
    st.markdown("### 🎬 DANH SÁCH VIDEO BỆNH NHÂN ĐÃ QUAY")
    deps.render_video_list_fragment(user_role)


def render_clinician_home(st, deps, *, user_role: str) -> None:
    render_clinician_symptom_list(st, deps, user_role=user_role)
    render_clinician_video_list(st, deps, user_role=user_role)


def _find_video_for_selected(deps, selected_video):
    all_vids = deps.load_research_videos()
    return next(
        (
            video
            for video in all_vids
            if video.get("username") == selected_video.get("username")
            and (
                video.get("video_name") == selected_video.get("video_name")
                or selected_video.get("video_name", "") in video.get("video_name", "")
            )
        ),
        None,
    )


def _ai_report_has_been_sent(deps, video_data) -> bool:
    _, eval_mtime = deps.mtimes_video_eval()
    evals = deps.evals_dedup_cached(eval_mtime)
    return any(
        eval_item.get("doctor_username") == "AI_Researcher"
        and eval_item.get("patient_username") == video_data.get("username")
        and (
            eval_item.get("video_name") == video_data.get("video_name")
            or video_data.get("video_name", "") in eval_item.get("video_name", "")
        )
        for eval_item in evals
    )


def _populate_ai_session_state(st, deps, video_data) -> None:
    st.session_state.stats = video_data["metrics"]
    st.session_state.processed_video_path = video_data.get("processed_path")
    st.session_state.all_frames_data_path = video_data.get("all_frames_data_path")
    st.session_state.uploaded_file_name = video_data.get("video_name")
    st.session_state.has_data = True

    exercise_name = video_data.get("exercise", "codman")
    exercise_base = next(
        (deps.BAI_TAP[key] for key in deps.BAI_TAP if deps.BAI_TAP[key]["ten"] == exercise_name),
        deps.BAI_TAP["codman"],
    )
    st.session_state.exercise = exercise_base.copy()
    if "sai_so" in video_data:
        st.session_state.exercise["chuan"] = exercise_base["chuan"].copy()
        st.session_state.exercise["chuan"]["sai_so"] = video_data["sai_so"]
    if video_data.get("df_path") and os.path.exists(video_data["df_path"]):
        try:
            st.session_state.angle_df = deps.read_display_csv_fast(video_data["df_path"])
        except Exception:
            pass


def render_doctor_ai_results(st, deps) -> None:
    selected_video = st.session_state.get("current_eval_video")
    if not selected_video:
        st.info("ℹ️ Vui lòng chọn một video bệnh nhân ở TRANG CHỦ để xem kết quả AI.")
        return

    video_data = _find_video_for_selected(deps, selected_video)
    if not video_data or not video_data.get("metrics"):
        st.warning("🕒 Nghiên cứu viên chưa thực hiện phân tích AI cho video này.")
        return

    if not _ai_report_has_been_sent(deps, video_data):
        st.warning("🕒 Nghiên cứu viên đã thực hiện phân tích nhưng CHƯA BẤM GỬI báo cáo chính thức.")
        return

    _populate_ai_session_state(st, deps, video_data)
    st.markdown("## 📊 KẾT QUẢ PHÂN TÍCH AI TỪ NGHIÊN CỨU VIÊN")
    chart_tab, video_tab = st.tabs(["📊 BIỂU ĐỒ CHI TIẾT", "🎬 VIDEO & XƯƠNG TRÍCH XUẤT"])
    with chart_tab:
        deps.render_analysis_tab(key_suffix="doc_ai_tab")
    with video_tab:
        deps.render_frames_full(key_suffix="doc_ai_tab")


def render_doctor_video_assets(st, deps) -> None:
    selected_video = st.session_state.get("current_eval_video")
    if not selected_video:
        st.info("ℹ️ Vui lòng chọn một video bệnh nhân ở TRANG CHỦ để xem video trích xuất.")
        return

    video_data = _find_video_for_selected(deps, selected_video)
    if not video_data or not video_data.get("metrics"):
        st.warning("🕒 Nghiên cứu viên chưa thực hiện phân tích AI cho video này.")
        return
    if not _ai_report_has_been_sent(deps, video_data):
        st.warning("🕒 Nghiên cứu viên đã thực hiện phân tích nhưng CHƯA BẤM GỬI báo cáo chính thức.")
        return

    _populate_ai_session_state(st, deps, video_data)
    deps.render_frames_full(key_suffix="doc_ai_top_video_tab")


def _selected_video_has_ai_report(st, deps):
    selected_video = st.session_state.get("current_eval_video")
    if not selected_video:
        return selected_video, False
    _, eval_mtime = deps.mtimes_video_eval()
    evals = deps.evals_dedup_cached(eval_mtime)
    has_ai = any(
        eval_item.get("doctor_username") == "AI_Researcher"
        and eval_item.get("patient_username") == selected_video.get("username")
        and (
            eval_item.get("video_name") == selected_video.get("video_name")
            or selected_video.get("video_name", "") in eval_item.get("video_name", "")
        )
        for eval_item in evals
    )
    return selected_video, has_ai


def _find_exact_video_for_selected(st, deps, selected_video):
    video_list = deps.load_data(deps.VIDEOS_FILE)
    return next(
        (
            video
            for video in video_list
            if video.get("username") == selected_video.get("username")
            and video.get("video_name") == selected_video.get("video_name")
        ),
        None,
    )


def _sync_doc_ai_video_session(st, deps, video_data) -> None:
    st.session_state.stats = video_data.get("metrics")
    st.session_state.processed_video_path = video_data.get("processed_path")
    st.session_state.all_frames_data_path = video_data.get("all_frames_data_path")
    st.session_state.uploaded_file_name = video_data.get("video_name")
    st.session_state.frames_zip = deps.frames_zip_path_from_video(video_data)


def render_doctor_combined_eval_and_research(st, deps) -> None:
    """Doctor/KTV combined tab: clinical evaluation, NCKH form, AI result, frames."""
    st.markdown("## 📊 QUẢN LÝ ĐÁNH GIÁ LÂM SÀNG & DỮ LIỆU NCKH")

    selected_video, has_ai = _selected_video_has_ai_report(st, deps)
    tab_list = ["📝 ĐÁNH GIÁ PHCN", "📄 PHIẾU NCKH", "🔬 KẾT QUẢ TỪ NCV (AI)", "🎬 VIDEO & HÌNH ẢNH"]
    if st.session_state.get("doc_sub_tab") not in tab_list:
        st.session_state.doc_sub_tab = tab_list[0]
    selected_sub = st.segmented_control(
        "Sub menu bác sĩ",
        options=tab_list,
        default=st.session_state.doc_sub_tab,
        key="doc_sub_tab_widget",
        label_visibility="collapsed",
    )
    if selected_sub:
        st.session_state.doc_sub_tab = selected_sub
    else:
        selected_sub = st.session_state.doc_sub_tab

    if selected_sub == "📝 ĐÁNH GIÁ PHCN":
        deps.render_doctor_eval_form()
    elif selected_sub == "📄 PHIẾU NCKH":
        deps.render_research_form()
    elif selected_sub == "🔬 KẾT QUẢ TỪ NCV (AI)":
        if not selected_video:
            st.info("ℹ️ Vui lòng chọn một video bệnh nhân ở TRANG CHỦ để xem kết quả AI.")
        elif not has_ai:
            st.warning("🕒 Nghiên cứu viên chưa thực hiện phân tích AI hoặc chưa gửi báo cáo cho video này.")
        else:
            video_ai = _find_exact_video_for_selected(st, deps, selected_video)
            if not video_ai:
                st.warning("⚠️ Không tìm thấy dữ liệu video AI tương ứng.")
                return
            _sync_doc_ai_video_session(st, deps, video_ai)
            if not video_ai.get("metrics"):
                st.warning("⚠️ NCV đã gửi báo cáo nhưng dữ liệu biểu đồ chi tiết chưa được đồng bộ hoặc bị lỗi file.")
                return

            df_ncv = None
            df_path_ncv = video_ai.get("df_path")
            if df_path_ncv:
                deps.ensure_local_file(df_path_ncv)
                if os.path.exists(df_path_ncv):
                    try:
                        df_ncv = deps.read_display_csv_fast(df_path_ncv)
                    except Exception:
                        pass
            exercise_ai = next(
                (deps.BAI_TAP[key] for key in deps.BAI_TAP if deps.BAI_TAP[key]["ten"] == video_ai.get("exercise")),
                deps.BAI_TAP["codman"],
            )
            deps.render_analysis_tab(
                key_suffix="doc_view_ncv_sub",
                stats_ext=video_ai["metrics"],
                df_ext=df_ncv,
                exercise_ext=exercise_ai,
            )
    elif selected_sub == "🎬 VIDEO & HÌNH ẢNH":
        if not selected_video:
            st.info("ℹ️ Vui lòng chọn một video bệnh nhân ở TRANG CHỦ để xem video trích xuất.")
        elif not has_ai:
            st.warning("🕒 Nghiên cứu viên chưa thực hiện phân tích AI hoặc chưa gửi báo cáo cho video này.")
        else:
            video_ai = _find_exact_video_for_selected(st, deps, selected_video)
            if video_ai:
                _sync_doc_ai_video_session(st, deps, video_ai)
                deps.render_frames_full(key_suffix="doc_view_ncv_vid")
            else:
                st.warning("⚠️ Không tìm thấy dữ liệu video AI tương ứng.")


def render_doctor_tab(selected_tab: str, deps) -> None:
    st = deps.st
    if selected_tab == "🏠 TRANG CHỦ":
        render_clinician_home(st, deps, user_role="Bác sĩ / KTV PHCN")
    elif selected_tab == "📊 QUẢN LÝ ĐÁNH GIÁ & NCKH":
        render_doctor_combined_eval_and_research(st, deps)
    elif selected_tab == "📝 ĐÁNH GIÁ PHCN":
        deps.render_doctor_eval_form()
    elif selected_tab == "📊 KẾT QUẢ AI":
        render_doctor_ai_results(st, deps)
    elif selected_tab == "🎬 VIDEO & ẢNH":
        render_doctor_video_assets(st, deps)
    elif selected_tab == "⏰ LỊCH NHẮC NHỞ":
        deps.render_reminders()
    elif selected_tab == "📚 THÔNG TIN TỔNG HỢP":
        deps.render_general_info("Bác sĩ / KTV PHCN")
    elif selected_tab == "📞 THÔNG TIN LIÊN HỆ":
        deps.render_contact()
    elif selected_tab == "👥 HỒ SƠ ĐỀ TÀI & ĐỘI NGŨ CHUYÊN GIA":
        deps.render_research_profile_team()
    elif selected_tab == "💬 PHẢN HỒI":
        deps.render_feedback()
