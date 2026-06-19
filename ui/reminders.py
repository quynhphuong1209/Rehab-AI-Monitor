"""Reminder and schedule Streamlit page."""
from __future__ import annotations


def render_reminders_page(deps):
    st = deps.st
    time = deps.time
    BAI_TAP = deps.BAI_TAP
    DOCTOR_ROLE = deps.DOCTOR_ROLE
    REMINDERS_FILE = deps.REMINDERS_FILE
    VIDEOS_FILE = deps.VIDEOS_FILE
    get_vn_now = deps.get_vn_now
    load_data = deps.load_data
    load_users = deps.load_users
    require_patient_scope = deps.require_patient_scope
    require_role = deps.require_role
    safe_html = deps.safe_html
    save_data = deps.save_data
    scope_patient_usernames_for_current_actor = deps.scope_patient_usernames_for_current_actor
    scope_records_for_current_actor = deps.scope_records_for_current_actor
    write_audit_log = deps.write_audit_log

    """Hiển thị lịch nhắc nhở chi tiết (Persistent & Role-based)"""
    st.markdown("## ⏰ LỊCH NHẮC NHỞ & HẸN KHÁM")

    user_info = st.session_state.user_info
    user_role = user_info.get('role', 'Bệnh nhân')
    username = user_info.get('username')

    # LOAD DATA TỪ FILE
    schedules = load_data(REMINDERS_FILE)
    users = load_users()
    if not isinstance(schedules, list): schedules = []

    # FILTER DATA
    if user_role == "Bệnh nhân":
        # Lọc theo username (không phân biệt hoa thường, xóa khoảng trắng)
        target_uname = username.strip().lower()
        display_schedules = [s for s in schedules if s.get('patient_username', '').strip().lower() == target_uname]
    elif user_role == "Quản trị viên":
        display_schedules = schedules
    else:
        display_schedules = scope_records_for_current_actor(schedules, users=users)

    current_time = get_vn_now()
    is_light = st.session_state.theme == 'light'
    m_bg = "white" if is_light else "rgba(255,255,255,0.05)"
    m_border = "1px solid #eee" if is_light else "1px solid rgba(255,255,255,0.1)"
    m_text = "#0072ff" if is_light else "#ffd700"
    m_label = "#666" if is_light else "#aaa"

    # Màu sắc cho các thẻ (Cards)
    card_bg = "#ffffff" if is_light else "rgba(26,26,46,0.8)"
    card_text = "#000000" if is_light else "#ffffff"
    card_border = "1px solid #ddd" if is_light else "none"

    # Màu nhấn cho từng loại
    color_app = "#0072ff" if is_light else "#ffd700"  # Xanh dương đậm cho light, Vàng cho dark
    color_ex = "#008B8B" if is_light else "#00CED1"   # Cyan đậm cho light, Cyan sáng cho dark
    color_med = "#D32F2F" if is_light else "#FF6B6B"  # Đỏ đậm cho light, Đỏ nhạt cho dark

    col1, col2, col3, col4 = st.columns(4)

    day_mapping = {
        "Monday": "Thứ Hai",
        "Tuesday": "Thứ Ba",
        "Wednesday": "Thứ Tư",
        "Thursday": "Thứ Năm",
        "Friday": "Thứ Sáu",
        "Saturday": "Thứ Bảy",
        "Sunday": "Chủ Nhật"
    }
    english_day = current_time.strftime("%A")
    vietnamese_day = day_mapping.get(english_day, english_day)

    metrics_data = [
        ("📅 Hôm nay", current_time.strftime("%d/%m/%Y")),
        ("⏰ Hiện tại", current_time.strftime("%H:%M:%S")),
        ("📆 Thứ", vietnamese_day),
        ("📊 Tổng lịch", len(display_schedules))
    ]

    cols = [col1, col2, col3, col4]
    for i, (label, val) in enumerate(metrics_data):
        with cols[i]:
            if "Hiện tại" in label:
                val_html = f'<div id="vietnam-live-time">{val}</div>'
            else:
                val_html = f'<div>{val}</div>'
            st.markdown(f"""
                <div style="background: {m_bg}; border: {m_border}; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <div style="color: {m_label}; font-size: 0.9rem; font-weight: 500; margin-bottom: 5px;">{label}</div>
                    <div style="color: {m_text}; font-size: 1.8rem; font-weight: 800;">{val_html}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if user_role == "Bệnh nhân":
        st.caption(f"👤 Tài khoản: `{username}`")
    else:
        st.caption(f"🔍 Hệ thống hiện có tổng cộng {len(schedules)} bản ghi. Đang quản lý với tư cách: `{user_info.get('full_name', username)}`")

    tab_list = ["🩺 Lịch hẹn khám", "🏋️ Lịch tập luyện", "💊 Lịch uống thuốc"]
    if user_role == "Bác sĩ / KTV PHCN":
        tab_list.append("➕ Thêm mới")

    all_lich_tabs = st.tabs(tab_list)

    # Phân loại schedules
    apps = [s for s in display_schedules if s.get('type') == "appointment"]
    exercises = [s for s in display_schedules if s.get('type') == "exercise"]
    meds = [s for s in display_schedules if s.get('type') == "medication"]

    with all_lich_tabs[0]:
        st.subheader("🩺 Lịch hẹn với bác sĩ")
        if not apps:
            st.info("📭 Không có lịch hẹn nào.")
        else:
            for app in apps:
                col1, col2 = st.columns([4, 1])
                with col1:
                    app_title = safe_html(app.get('title', 'Lịch hẹn'), max_length=200)
                    app_datetime = safe_html(app.get('datetime', 'N/A'), max_length=80)
                    app_doctor = safe_html(app.get('doctor_name', 'Hệ thống'), max_length=120)
                    app_patient_html = (
                        "👤 <b>Bệnh nhân:</b> " + safe_html(app.get('patient_name', 'Chưa rõ'), max_length=120) + "<br>"
                        if user_role != "Bệnh nhân" else ""
                    )
                    app_notes_html = (
                        "📝 <b>Ghi chú:</b> " + safe_html(app.get('notes'), max_length=1000) + "<br>"
                        if app.get('notes') else ""
                    )
                    st.markdown(f"""<div style="background: {card_bg}; color: {card_text}; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; border-left: 5px solid {color_app}; border: {card_border}; border-left: 6px solid {color_app}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
<strong style="color: {color_app}; font-size: 1.15rem; display: block; margin-bottom: 8px;">📌 {app_title}</strong>
<div style="line-height: 1.6; font-size: 0.95rem;">
🕒 <b>Thời gian:</b> {app_datetime}<br>
👨‍⚕️ <b>Bác sĩ:</b> {app_doctor}<br>
{app_patient_html}
{app_notes_html}
</div>
<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(128,128,128,0.1); color: {color_app}; font-size: 0.85rem; font-weight: 500;">
{"🟢 Đã gửi đến bệnh nhân" if user_role != "Bệnh nhân" else "📩 Đã nhận từ bác sĩ"}
</div>
</div>""", unsafe_allow_html=True)
                with col2:
                    if user_role == "Bác sĩ / KTV PHCN":
                        app_id = app.get('id')
                        if st.button("🗑️", key=f"del_app_{app_id}"):
                            try:
                                require_role(DOCTOR_ROLE, action="delete_schedule", target=app_id)
                                require_patient_scope(app.get("patient_username"), action="delete_schedule")
                                schedules = [s for s in schedules if s.get('id') != app_id]
                                save_data(REMINDERS_FILE, schedules)
                                write_audit_log(username, user_role, "delete_schedule", app_id, "success")
                                st.rerun()
                            except PermissionError as exc:
                                st.error(str(exc))

    with all_lich_tabs[1]:
        st.subheader("🏋️ Lịch tập luyện")
        if not exercises:
            st.info("📭 Không có lịch tập nào.")
        else:
            for ex in exercises:
                col1, col2 = st.columns([4, 1])
                with col1:
                    ex_name = safe_html(ex.get('exercise_name', 'Bài tập'), max_length=200)
                    ex_datetime = safe_html(ex.get('datetime', 'N/A'), max_length=80)
                    ex_frequency = safe_html(ex.get('frequency', 'Một lần'), max_length=80)
                    ex_doctor = safe_html(ex.get('doctor_name', 'Hệ thống'), max_length=120)
                    ex_patient_html = (
                        "👤 <b>Bệnh nhân:</b> " + safe_html(ex.get('patient_name', 'Chưa rõ'), max_length=120) + "<br>"
                        if user_role != "Bệnh nhân" else ""
                    )
                    ex_notes_html = (
                        "📝 <b>Ghi chú:</b> " + safe_html(ex.get('notes'), max_length=1000) + "<br>"
                        if ex.get('notes') else ""
                    )
                    st.markdown(f"""<div style="background: {card_bg}; color: {card_text}; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; border-left: 5px solid {color_ex}; border: {card_border}; border-left: 6px solid {color_ex}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
<strong style="color: {color_ex}; font-size: 1.15rem; display: block; margin-bottom: 8px;">💪 {ex_name}</strong>
<div style="line-height: 1.6; font-size: 0.95rem;">
🕒 <b>Thời gian:</b> {ex_datetime}<br>
🔁 <b>Tần suất:</b> {ex_frequency}<br>
👨‍⚕️ <b>Chỉ định bởi:</b> {ex_doctor}<br>
{ex_patient_html}
{ex_notes_html}
</div>
<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(128,128,128,0.1); color: {color_ex}; font-size: 0.85rem; font-weight: 500;">
{"🟢 Đã gửi đến bệnh nhân" if user_role != "Bệnh nhân" else "📩 Đã nhận từ bác sĩ"}
</div>
</div>""", unsafe_allow_html=True)
                with col2:
                    if user_role == "Bác sĩ / KTV PHCN":
                        ex_id = ex.get('id')
                        if st.button("🗑️", key=f"del_ex_{ex_id}"):
                            try:
                                require_role(DOCTOR_ROLE, action="delete_schedule", target=ex_id)
                                require_patient_scope(ex.get("patient_username"), action="delete_schedule")
                                schedules = [s for s in schedules if s.get('id') != ex_id]
                                save_data(REMINDERS_FILE, schedules)
                                write_audit_log(username, user_role, "delete_schedule", ex_id, "success")
                                st.rerun()
                            except PermissionError as exc:
                                st.error(str(exc))

    with all_lich_tabs[2]:
        st.subheader("💊 Lịch uống thuốc")
        if not meds:
            st.info("📭 Không có lịch uống thuốc nào.")
        else:
            for med in meds:
                col1, col2 = st.columns([4, 1])
                with col1:
                    med_name_html = safe_html(med.get('medication_name', 'Thuốc'), max_length=200)
                    med_datetime = safe_html(med.get('datetime', 'N/A'), max_length=80)
                    med_dosage = safe_html(med.get('dosage', 'Theo chỉ định'), max_length=120)
                    med_doctor = safe_html(med.get('doctor_name', 'Hệ thống'), max_length=120)
                    med_patient_html = (
                        "👤 <b>Bệnh nhân:</b> " + safe_html(med.get('patient_name', 'Chưa rõ'), max_length=120) + "<br>"
                        if user_role != "Bệnh nhân" else ""
                    )
                    med_notes_html = (
                        "📝 <b>Ghi chú:</b> " + safe_html(med.get('notes'), max_length=1000) + "<br>"
                        if med.get('notes') else ""
                    )
                    st.markdown(f"""<div style="background: {card_bg}; color: {card_text}; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; border-left: 5px solid {color_med}; border: {card_border}; border-left: 6px solid {color_med}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
<strong style="color: {color_med}; font-size: 1.15rem; display: block; margin-bottom: 8px;">💊 {med_name_html}</strong>
<div style="line-height: 1.6; font-size: 0.95rem;">
🕒 <b>Thời gian:</b> {med_datetime}<br>
💊 <b>Liều:</b> {med_dosage}<br>
👨‍⚕️ <b>Bác sĩ kê đơn:</b> {med_doctor}<br>
{med_patient_html}
{med_notes_html}
</div>
<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(128,128,128,0.1); color: {color_med}; font-size: 0.85rem; font-weight: 500;">
{"🟢 Đã gửi đến bệnh nhân" if user_role != "Bệnh nhân" else "📩 Đã nhận từ bác sĩ"}
</div>
</div>""", unsafe_allow_html=True)
                with col2:
                    if user_role == "Bác sĩ / KTV PHCN":
                        med_id = med.get('id')
                        if st.button("🗑️", key=f"del_med_{med_id}"):
                            try:
                                require_role(DOCTOR_ROLE, action="delete_schedule", target=med_id)
                                require_patient_scope(med.get("patient_username"), action="delete_schedule")
                                schedules = [s for s in schedules if s.get('id') != med_id]
                                save_data(REMINDERS_FILE, schedules)
                                write_audit_log(username, user_role, "delete_schedule", med_id, "success")
                                st.rerun()
                            except PermissionError as exc:
                                st.error(str(exc))

    if user_role == "Bác sĩ / KTV PHCN":
        with all_lich_tabs[3]:
            st.subheader("➕ Thêm lịch nhắc nhở mới")

            # 1. Tổng hợp danh sách bệnh nhân từ cả users.json và video_list.json
            current_users = load_users()
            patients_from_db = [u for u, info in current_users.items() if info.get('role') == 'Bệnh nhân']

            videos = load_data(VIDEOS_FILE)
            patients_from_videos = [v.get('username') for v in videos if v.get('username')]

            all_patient_usernames = list(set(patients_from_db + patients_from_videos))
            allowed_patients = scope_patient_usernames_for_current_actor(all_patient_usernames, users=current_users)
            all_patient_usernames = [u for u in all_patient_usernames if u in allowed_patients]

            # 2. Xây dựng ánh xạ tên đầy đủ để tránh KeyError cho tài khoản Google
            patient_names = {}
            for u in all_patient_usernames:
                if u in current_users:
                    patient_names[u] = current_users[u].get('full_name', u)
                else:
                    for v in videos:
                        if v.get('username') == u and v.get('full_name'):
                            patient_names[u] = v.get('full_name')
                            break
                    if u not in patient_names:
                        patient_names[u] = u

            all_patients = sorted(all_patient_usernames, key=lambda x: patient_names.get(x, x).lower())

            if not all_patients:
                st.warning("⚠️ Hệ thống hiện chưa có bệnh nhân nào.")
                return

            # 3. Tự động chọn bệnh nhân từ video đang được chọn ở TRANG CHỦ (nếu có)
            current_eval = st.session_state.get('current_eval_video')
            default_index = 0
            if current_eval:
                selected_patient_from_video = current_eval.get('username')
                if selected_patient_from_video in all_patients:
                    default_index = all_patients.index(selected_patient_from_video)
                    patient_name = safe_html(patient_names.get(selected_patient_from_video, selected_patient_from_video), max_length=120)
                    patient_username_html = safe_html(selected_patient_from_video, max_length=80)
                    st.markdown(f"""
                    <div style="background: rgba(0, 198, 255, 0.1); padding: 15px; border-radius: 12px; border-left: 5px solid #00c6ff; margin-bottom: 20px;">
                        <p style="margin:0; color:#888; font-size:0.8rem;">👤 BỆNH NHÂN TỪ VIDEO ĐANG CHỌN:</p>
                        <h4 style="margin:5px 0; color:#00c6ff;">{patient_name}</h4>
                        <p style="margin:0; font-size:0.85rem; color:#aaa;">Tài khoản: {patient_username_html}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💡 Bạn có thể chọn bất kỳ bệnh nhân nào bên dưới để thêm lịch nhắc nhở mới.")

            selected_patient = st.selectbox(
                "Xác nhận bệnh nhân:",
                all_patients,
                index=default_index,
                format_func=lambda x: f"🌟 {patient_names.get(x, x)} ({x})"
            )

            loai = st.radio("Chọn loại:", ["Lịch hẹn khám", "Lịch tập luyện", "Lịch uống thuốc"], horizontal=True)

            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Ngày", min_value=get_vn_now().date())
            with col2:
                time_input = st.time_input("Giờ")

            if loai == "Lịch hẹn khám":
                title = st.text_input("Tiêu đề", placeholder="VD: Khám lại khớp vai")
                notes = st.text_area("Ghi chú")
                if st.button("➕ Thêm lịch hẹn", key="add_appointment_btn", type="primary", width="stretch"):
                    if not title:
                        st.error("⚠️ Vui lòng nhập tiêu đề lịch hẹn!")
                    elif not selected_patient:
                        st.error("⚠️ Vui lòng chọn bệnh nhân!")
                    else:
                        try:
                            require_role(DOCTOR_ROLE, action="create_schedule", target=selected_patient)
                            require_patient_scope(selected_patient, action="create_schedule")
                            new_item = {
                                'id': int(time.time() * 1000),
                                'type': 'appointment',
                                'title': title,
                                'datetime': f"{date} {time_input}",
                                'notes': notes,
                                'patient_username': selected_patient,
                                'patient_name': patient_names.get(selected_patient, selected_patient),
                                'doctor_username': username,
                                'doctor_name': user_info.get('full_name', username)
                            }
                            schedules.append(new_item)
                            save_data(REMINDERS_FILE, schedules)
                            write_audit_log(username, user_role, "create_schedule", selected_patient, "success", {"type": "appointment"})
                            st.success(f"✅ Đã thêm lịch hẹn cho {patient_names.get(selected_patient, selected_patient)}!")
                            st.toast(f"✅ Đã thêm lịch hẹn thành công!", icon="🩺")
                            time.sleep(1.5)
                            st.rerun()
                        except PermissionError as exc:
                            st.error(str(exc))

            elif loai == "Lịch tập luyện":
                exercise = st.selectbox("Bài tập", list(BAI_TAP.keys()), format_func=lambda x: BAI_TAP[x]['ten'])
                frequency = st.selectbox("Tần suất", ["Một lần", "Hàng ngày", "Thứ 2-4-6", "Thứ 3-5-7"])
                notes = st.text_area("Ghi chú")
                if st.button("➕ Thêm lịch tập", key="add_exercise_btn", type="primary", width="stretch"):
                    if not selected_patient:
                        st.error("⚠️ Vui lòng chọn bệnh nhân!")
                    else:
                        try:
                            require_role(DOCTOR_ROLE, action="create_schedule", target=selected_patient)
                            require_patient_scope(selected_patient, action="create_schedule")
                            new_item = {
                                'id': int(time.time() * 1000),
                                'type': 'exercise',
                                'exercise_name': BAI_TAP[exercise]['ten'],
                                'datetime': f"{date} {time_input}",
                                'frequency': frequency,
                                'notes': notes,
                                'patient_username': selected_patient,
                                'patient_name': patient_names.get(selected_patient, selected_patient),
                                'doctor_username': username,
                                'doctor_name': user_info.get('full_name', username)
                            }
                            schedules.append(new_item)
                            save_data(REMINDERS_FILE, schedules)
                            write_audit_log(username, user_role, "create_schedule", selected_patient, "success", {"type": "exercise"})
                            st.success(f"✅ Đã thêm lịch tập cho {patient_names.get(selected_patient, selected_patient)}!")
                            st.toast(f"✅ Đã thêm lịch tập thành công!", icon="🏋️")
                            time.sleep(1.5)
                            st.rerun()
                        except PermissionError as exc:
                            st.error(str(exc))

            else:
                med_name = st.text_input("Tên thuốc")
                dosage = st.text_input("Liều lượng")
                notes = st.text_area("Ghi chú")
                if st.button("➕ Thêm lịch uống thuốc", key="add_medication_btn", type="primary", width="stretch"):
                    if not med_name:
                        st.error("⚠️ Vui lòng nhập tên thuốc!")
                    elif not selected_patient:
                        st.error("⚠️ Vui lòng chọn bệnh nhân!")
                    else:
                        try:
                            require_role(DOCTOR_ROLE, action="create_schedule", target=selected_patient)
                            require_patient_scope(selected_patient, action="create_schedule")
                            new_item = {
                                'id': int(time.time() * 1000),
                                'type': 'medication',
                                'medication_name': med_name,
                                'dosage': dosage,
                                'datetime': f"{date} {time_input}",
                                'notes': notes,
                                'taken': False,
                                'patient_username': selected_patient,
                                'patient_name': patient_names.get(selected_patient, selected_patient),
                                'doctor_username': username,
                                'doctor_name': user_info.get('full_name', username)
                            }
                            schedules.append(new_item)
                            save_data(REMINDERS_FILE, schedules)
                            write_audit_log(username, user_role, "create_schedule", selected_patient, "success", {"type": "medication"})
                            st.success(f"✅ Đã thêm lịch uống thuốc cho {patient_names.get(selected_patient, selected_patient)}!")
                            st.toast(f"✅ Đã thêm lịch uống thuốc thành công!", icon="💊")
                            time.sleep(1.5)
                            st.rerun()
                        except PermissionError as exc:
                            st.error(str(exc))
