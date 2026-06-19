"""Research evaluation form page."""
from __future__ import annotations


def render_research_form_page(deps):
    st = deps.st
    pd = deps.pd
    ADMIN_ROLE = deps.ADMIN_ROLE
    DOCTOR_ROLE = deps.DOCTOR_ROLE
    EVALUATIONS_FILE = deps.EVALUATIONS_FILE
    PATIENT_ROLE = deps.PATIENT_ROLE
    RESEARCHER_ROLE = deps.RESEARCHER_ROLE
    RESEARCH_DATA_FILE = deps.RESEARCH_DATA_FILE
    SYMPTOMS_FILE = deps.SYMPTOMS_FILE
    current_actor_can_access_patient = deps.current_actor_can_access_patient
    get_vn_now = deps.get_vn_now
    load_data = deps.load_data
    require_patient_scope = deps.require_patient_scope
    require_role = deps.require_role
    researcher_view_records = deps.researcher_view_records
    save_data = deps.save_data
    scope_records_for_current_actor = deps.scope_records_for_current_actor
    write_audit_log = deps.write_audit_log

    st.markdown("## 📄 PHIẾU ĐÁNH GIÁ KỸ THUẬT TẬP LUYỆN")
    st.markdown("*(Bộ công cụ thu thập dữ liệu Nghiên cứu khoa học)*")
    st.info("💡 Phiếu này dùng để thu thập dữ liệu phục vụ nghiên cứu mô hình trí tuệ nhân tạo (AI) trong nhận diện động tác phục hồi chức năng.")

    user_role = st.session_state.user_info.get('role', 'Bệnh nhân')
    selected_video = st.session_state.get('current_eval_video')
    if selected_video and not current_actor_can_access_patient(selected_video.get('username')):
        st.warning("Video đang chọn nằm ngoài phạm vi phụ trách nên đã được bỏ khỏi phiếu NCKH.")
        selected_video = None
        st.session_state.pop('current_eval_video', None)

    # Lấy đánh giá lâm sàng hiện tại nếu có để điền sẵn vào phần IV
    existing_eval = None
    if selected_video:
        evals_db = load_data(EVALUATIONS_FILE)
        existing_eval = next((e for e in evals_db if
                             e.get('patient_username') == selected_video.get('username') and
                             e.get('video_name') == selected_video.get('video_name') and
                             e.get('doctor_username') != "AI_Researcher"), None)

    # Giá trị mặc định cho Phần IV (Ground Truth)
    options_result = ["Đúng", "Sai", "Gần đúng"]
    default_res_idx = 0
    if existing_eval and existing_eval.get('doctor_result') in options_result:
        default_res_idx = options_result.index(existing_eval.get('doctor_result'))

    options_plan = ["Tiếp tục", "Chuyển bài", "Khám lại"]
    default_plan_idx = 0
    if existing_eval and existing_eval.get('plan') in options_plan:
        default_plan_idx = options_plan.index(existing_eval.get('plan'))

    default_errors = []
    if existing_eval and isinstance(existing_eval.get('errors'), list):
        default_errors = existing_eval.get('errors', [])

    default_comment = ""
    if existing_eval:
        default_comment = existing_eval.get('comments', '')

    # --- LOGIC TỰ ĐỘNG ĐIỀN THÔNG TIN TỪ KHAI BÁO CỦA BN ---
    symptoms_data = scope_records_for_current_actor(load_data(SYMPTOMS_FILE))

    # Xác định BN mục tiêu: Nếu có video thì lấy theo video, nếu không lấy BN mới nhất gửi triệu chứng
    if user_role in ["Bác sĩ / KTV PHCN", "Nghiên cứu viên"]:
        if selected_video:
            patient_username = selected_video.get('username')
        else:
            patient_username = symptoms_data[-1].get('username') if symptoms_data else ""
    else:
        patient_username = st.session_state.user_info['username']

    # Lấy bản ghi mới nhất của BN này để auto-fill
    p_record = next((s for s in reversed(symptoms_data) if s.get('username') == patient_username), None)

    # Giá trị mặc định hoặc từ record
    d_full_name = p_record.get('full_name', '') if p_record else ""
    d_sub_code = p_record.get('patient_id', patient_username) if p_record else patient_username
    d_age = p_record.get('age', 25) if p_record else 25
    d_gender_idx = 0
    if p_record:
        d_gender_idx = 0 if "Nam" in p_record.get('gender', 'Nam') else 1

    d_pain_idx = 0
    d_severity_idx = 0
    if p_record:
        vas = p_record.get('vas', 0)
        if vas <= 3: d_pain_idx, d_severity_idx = 0, 0
        elif vas <= 6: d_pain_idx, d_severity_idx = 1, 1
        else: d_pain_idx, d_severity_idx = 2, 2

    with st.form("research_evaluation_form_v2"):
        # I. THÔNG TIN CHUNG
        st.markdown("### I. THÔNG TIN CHUNG VÀ ĐẶC ĐIỂM LÂM SÀNG")
        col1, col2 = st.columns(2)
        with col1:
            interviewer = st.text_input("Họ và tên người phỏng vấn:", value=d_full_name)
            interview_date = st.date_input("Ngày phỏng vấn:", value=get_vn_now())
            subject_code = st.text_input("Mã đối tượng (Mã BN):", value=d_sub_code)
            age = st.number_input("Tuổi:", min_value=0, max_value=120, value=d_age)
            gender = st.radio("Giới tính:", ["Nam (1)", "Nữ (2)"], horizontal=True, index=d_gender_idx)
            region = st.radio("Khu vực:", ["Nội thành (1)", "Ngoại thành (2)"], horizontal=True)
        with col2:
            job = st.selectbox("Nghề nghiệp:", [
                "Nông dân (1)", "Công nhân (2)", "Cán bộ - viên chức (3)",
                "Buôn bán (4)", "Nội trợ (5)", "Lao động tự do (6)", "Nghỉ hưu (7)",
                "Không có nghề nghiệp cụ thể (8)"
            ])
            education = st.selectbox("Trình độ học vấn:", [
                "Mù chữ (1)", "Tiểu học (2)", "Trung học cơ sở (3)",
                "Trung học phổ thông (4)", "Cao đẳng – đại học (5)", "Không rõ (6)"
            ])
            department = st.radio("Khoa điều trị:", ["Khoa PHCN – Y học cổ truyền (1)", "Khác (99)"], horizontal=True)
            treatment_type = st.radio("Hình thức điều trị:", ["Nội trú (1)", "Ngoại trú (2)"], horizontal=True)
            st.markdown("[🔍 Tra cứu danh mục mã ICD-10 (Bộ Y tế)](https://icd.kcb.vn/icd-10/icd10)")
            diagnosis = st.radio("Chẩn đoán:", [
                "Viêm quanh khớp vai thể giả liệt (ICD-10: M75.1)",
                "Viêm quanh khớp vai thể đông cứng (ICD-10: M75.0)",
                "Viêm quanh khớp vai thể đơn thuần (ICD-10: M75.8)",
                "Viêm quanh khớp cấp (ICD-10: M75.3 / M75.5)",
                "Viêm quanh khớp vai (P) (ICD-10: M75)"
            ])
            lesion_side = st.radio("Vị trí vai tổn thương:", ["Vai trái (1)", "Vai phải (2)", "Cả hai vai (3)"], horizontal=True)
            duration = st.radio("Thời gian mắc bệnh:", ["< 1 tháng (1)", "1 – 3 tháng (2)", ">= 3 tháng (3)"], horizontal=True)

        # II. THÔNG TIN PHỤC HỒI
        st.markdown("### II. THÔNG TIN PHỤC HỒI")
        col3, col4 = st.columns(2)
        with col3:
            training_side = st.radio("Bên tập luyện:", ["Vai trái", "Vai phải", "Cả hai vai"], horizontal=True)
            pain_level = st.radio("Mức độ đau (VAS 0–10):", ["Nhẹ (0–3)", "Trung bình (4–6)", "Nặng (7–10)"], horizontal=True, index=d_pain_idx)
        with col4:
            disease_severity = st.radio("Mức độ bệnh:", ["Nhẹ", "Trung bình", "Nặng"], horizontal=True, index=d_severity_idx)

        # III. NỘI DUNG TẬP LUYỆN
        st.markdown("### III. NỘI DUNG TẬP LUYỆN ĐƯỢC GHI HÌNH")
        exercise = selected_video.get('exercise') if selected_video else "Bài tập con lắc Codman"
        st.markdown(f"**Bài tập được ghi hình:** {exercise}")
        exercise_list = [exercise]

        # IV. ĐÁNH GIÁ KỸ THUẬT (GROUND TRUTH)
        st.markdown("### IV. ĐÁNH GIÁ KỸ THUẬT ĐỘNG TÁC (GROUND TRUTH)")
        if user_role == "Bệnh nhân":
            st.info("💡 Phần này sẽ do Bác sĩ / KTV PHCN hoặc Nghiên cứu viên đánh giá sau khi xem video.")

        col5, col6 = st.columns(2)
        with col5:
            general_result = st.radio("Kết quả:", ["Đúng", "Sai", "Gần đúng"], index=default_res_idx, horizontal=True)
            plan = st.radio("Chỉ định:", ["Tiếp tục", "Chuyển bài", "Khám lại"], index=default_plan_idx, horizontal=True)
        with col6:
            errors = st.multiselect("Lỗi sai:", ["Vị trí tay chưa đúng", "Biên độ chưa đạt", "Tốc độ quá nhanh/chậm", "Sai tư thế thân người"], default=default_errors)
        specialist_comment = st.text_area("Nhận xét chuyên môn của Bác sĩ/KTV PHCN:", value=default_comment)

        # V. THÔNG TIN VIDEO
        st.markdown("### V. THÔNG TIN DỮ LIỆU VIDEO")
        col7, col8 = st.columns(2)
        with col7:
            video_code = st.text_input("Mã video:", value=selected_video.get('video_name') if selected_video else "")
            recording_device = st.radio("Thiết bị ghi hình:", ["Điện thoại (1)", "Webcam (2)", "Khác (3)"], horizontal=True)
        with col8:
            recording_angle = st.radio("Góc quay:", ["Chính diện (1)", "Bên trái (2)", "Bên phải (3)"], horizontal=True)
            camera_distance = st.text_input("Khoảng cách camera (m):", value="2.5")

        # VI. XÁC NHẬN
        st.markdown("---")
        st.write("### VI. XÁC NHẬN")
        confirm = st.checkbox("Tôi xác nhận các thông tin trên là chính xác và phục vụ cho mục đích nghiên cứu khoa học.")

        btn_label = "🚀 LƯU & GỬI PHIẾU NCKH CHO BỆNH NHÂN & NCV" if user_role != "Bệnh nhân" else "🚀 GỬI THÔNG TIN KHAI BÁO"
        submitted = st.form_submit_button(btn_label, width="stretch", type="primary")

        if submitted:
            if not confirm:
                st.error("⚠️ Vui lòng tick chọn xác nhận trước khi lưu phiếu.")
            else:
                try:
                    require_role(PATIENT_ROLE, DOCTOR_ROLE, RESEARCHER_ROLE, ADMIN_ROLE, action="create_research_record", target=patient_username)
                    require_patient_scope(patient_username, action="create_research_record")
                    research_data = load_data(RESEARCH_DATA_FILE)
                    entry = {
                        "interviewer": interviewer,
                        "interview_date": str(interview_date),
                        "subject_code": subject_code,
                        "age": age,
                        "gender": gender,
                        "region": region,
                        "job": job,
                        "education": education,
                        "department": department,
                        "treatment_type": treatment_type,
                        "diagnosis": diagnosis,
                        "lesion_side": lesion_side,
                        "duration": duration,
                        "training_side": training_side,
                        "pain_level": pain_level,
                        "disease_severity": disease_severity,
                        "exercises": exercise_list,
                        "general_result": general_result,
                        "errors": errors,
                        "plan": plan,
                        "specialist_comment": specialist_comment,
                        "video_code": video_code,
                        "recording_device": recording_device,
                        "recording_angle": recording_angle,
                        "camera_distance": camera_distance,
                        "patient_username": patient_username, # Liên kết với tài khoản BN
                        "submitted_by": st.session_state.user_info['username'],
                        "role": user_role,
                        "timestamp": get_vn_now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    research_data.append(entry)
                    save_data(RESEARCH_DATA_FILE, research_data)
                    write_audit_log(st.session_state.user_info['username'], user_role, "create_research_record", patient_username, "success")
                    st.success("✅ Đã lưu và gửi phiếu đánh giá nghiên cứu cho Bệnh nhân & NCV thành công!")
                    st.balloons()
                    st.rerun()
                except PermissionError as exc:
                    st.error(str(exc))

    # --- PHẦN HIỂN THỊ LỊCH SỬ (VIEWER) ---
    st.markdown("---")
    st.markdown("### 📜 LỊCH SỬ PHIẾU ĐÁNH GIÁ NCKH")

    all_research_data = load_data(RESEARCH_DATA_FILE)
    if not isinstance(all_research_data, list): all_research_data = []

    # Phân quyền xem dữ liệu
    username = st.session_state.user_info['username']
    if user_role == "Bệnh nhân":
        # Bệnh nhân thấy phiếu dựa trên username tài khoản hoặc mã đối tượng
        display_list = [d for d in all_research_data if d.get('patient_username') == username or d.get('subject_code') == username]
    elif user_role == "Quản trị viên":
        display_list = all_research_data
    else:
        display_list = scope_records_for_current_actor(all_research_data)

    display_list = researcher_view_records(display_list)
    export_research_data = researcher_view_records(scope_records_for_current_actor(all_research_data)) if user_role == "Nghiên cứu viên" else display_list

    if not display_list:
        st.info("📭 Chưa có bản ghi dữ liệu nghiên cứu nào được lưu.")
    else:
        # Nút xuất dữ liệu cho NCV/Bác sĩ
        if user_role in ["Bác sĩ / KTV PHCN", "Nghiên cứu viên", "Quản trị viên"]:
            c_exp1, c_exp2 = st.columns([1, 4])
            with c_exp1:
                df_export = pd.DataFrame(export_research_data)
                csv = df_export.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📊 Xuất Excel (CSV)",
                    data=csv,
                    file_name=f"research_data_{get_vn_now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width="stretch"
                )

        # Hiển thị danh sách phiếu
        for i, item in enumerate(reversed(display_list)):
            col_h_main, col_h_del = st.columns([12, 1])
            with col_h_main:
                exercises_str = ", ".join(item.get('exercises', []))
                exercises_display = f" - Động tác: {exercises_str}" if exercises_str else ""
                with st.expander(f"📅 Phiếu ngày {item.get('timestamp', 'N/A')} - BN: {item.get('subject_code', 'N/A')}{exercises_display} - KQ: {item.get('general_result', 'N/A')}"):
                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        st.markdown("**📌 Thông tin chung**")
                        st.write(f"- Người PV: {item.get('interviewer')}")
                        st.write(f"- Ngày PV: {item.get('interview_date')}")
                        st.write(f"- Tuổi/Giới: {item.get('age')}/{item.get('gender')}")
                        st.write(f"- Khu vực: {item.get('region')}")
                    with col_i2:
                        st.markdown("**🩺 Lâm sàng & Tập luyện**")
                        st.write(f"- Chẩn đoán: {item.get('diagnosis')}")
                        st.write(f"- Thời gian bệnh: {item.get('duration')}")
                        st.write(f"- Bài tập: {', '.join(item.get('exercises', []))}")
                        st.write(f"- Đau (VAS): {item.get('pain_level')}")
                    with col_i3:
                        st.markdown("**📊 Đánh giá chuyên môn**")
                        st.write(f"- Kết quả: {item.get('general_result')}")
                        if item.get('errors'):
                            st.write(f"- Lỗi sai: {', '.join(item.get('errors'))}")
                        if item.get('plan'):
                            st.write(f"- Chỉ định: {item.get('plan')}")
                        if item.get('correct_reps') is not None and item.get('total_reps') is not None:
                            st.write(f"- Số lần Đúng/Tổng: {item.get('correct_reps')}/{item.get('total_reps')}")
                        st.info(f"**Nhận xét:** {item.get('specialist_comment')}")

                    if item.get('video_code'):
                        st.caption(f"🎬 Mã video: {item.get('video_code')} | Thiết bị: {item.get('recording_device')} | Góc: {item.get('recording_angle')}")

            with col_h_del:
                if st.button("❌", key=f"del_res_{i}", help="Xóa bản ghi này"):
                    try:
                        require_role(DOCTOR_ROLE, RESEARCHER_ROLE, ADMIN_ROLE, action="delete_research_record", target=item.get('timestamp'))
                        require_patient_scope(item.get("patient_username") or item.get("subject_code"), action="delete_research_record")
                        # Tìm index thực tế trong all_research_data để xóa
                        actual_idx = -1
                        for idx, d in enumerate(all_research_data):
                            if d.get('timestamp') == item.get('timestamp') and d.get('subject_code') == item.get('subject_code'):
                                actual_idx = idx
                                break

                        if actual_idx != -1:
                            all_research_data.pop(actual_idx)
                            save_data(RESEARCH_DATA_FILE, all_research_data)
                            write_audit_log(st.session_state.user_info['username'], user_role, "delete_research_record", item.get('timestamp'), "success")
                            st.success("Đã xóa!")
                            st.rerun()
                    except PermissionError as exc:
                        st.error(str(exc))
