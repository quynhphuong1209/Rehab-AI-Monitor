"""Admin dashboard and management pages."""
from __future__ import annotations


def render_admin_management_page(deps):
    st = deps.st
    pd = deps.pd
    px = deps.px
    os = deps.os
    datetime = deps.datetime
    EVALUATIONS_FILE = deps.EVALUATIONS_FILE
    HISTORY_FILE = deps.HISTORY_FILE
    PROCESSED_DIR = deps.PROCESSED_DIR
    REMINDERS_FILE = deps.REMINDERS_FILE
    SESSION_STATE_FILE = deps.SESSION_STATE_FILE
    SYMPTOMS_FILE = deps.SYMPTOMS_FILE
    UPLOAD_DIR = deps.UPLOAD_DIR
    USER_DATA_FILE = deps.USER_DATA_FILE
    VIDEOS_FILE = deps.VIDEOS_FILE
    _format_vn_time = deps.format_vn_time
    _parse_upload_time_from_filename = deps.parse_upload_time_from_filename
    _remove_files_in_dir = deps.remove_files_in_dir
    create_backup_before_destructive = deps.create_backup_before_destructive
    get_global_session_version = deps.get_global_session_version
    get_vn_now = deps.get_vn_now
    load_data = deps.load_data
    load_users = deps.load_users
    password_record_update = deps.password_record_update
    require_role = deps.require_role
    revoke_all_sessions = deps.revoke_all_sessions
    save_data = deps.save_data
    save_users = deps.save_users
    write_audit_log = deps.write_audit_log

    st.markdown("## 🛠️ QUẢN TRỊ VIÊN: QUẢN LÝ HỆ THỐNG")

    users = load_users()

    # THỐNG KÊ NHANH
    total_users = len(users)
    doctors = len([u for u in users.values() if u.get('role') == "Bác sĩ / KTV PHCN"])
    researchers = len([u for u in users.values() if u.get('role') == "Nghiên cứu viên"])
    patients = len([u for u in users.values() if u.get('role') == "Bệnh nhân"])

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_users}</div>
            <div class="metric-label">Tổng người dùng</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #FFD700;">{doctors}</div>
            <div class="metric-label">Bác sĩ / KTV</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #00c6ff;">{researchers}</div>
            <div class="metric-label">Nghiên cứu viên</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #2ecc71;">{patients}</div>
            <div class="metric-label">Bệnh nhân</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_u1, tab_u2, tab_u3, tab_u4 = st.tabs(["👥 NGƯỜI DÙNG", "➕ TẠO TÀI KHOẢN", "📊 NHẬT KÝ HOẠT ĐỘNG", "🧹 DỌN DẸP"])

    with tab_u1:
        st.markdown("### 👥 Quản lý tài khoản")
        df_users = []
        for u, data in users.items():
            df_users.append({
                "Tên đăng nhập": u,
                "Họ tên": data.get("full_name", "N/A"),
                "Vai trò": data.get("role", "N/A"),
                "Email": data.get("email", "N/A"),
                "Ngày tạo": data.get("created_at", "N/A")[:10] if data.get("created_at") else "N/A"
            })

        # Hiển thị bảng với tìm kiếm
        df_display = pd.DataFrame(df_users)
        search_q = st.text_input("🔍 Tìm kiếm người dùng:", placeholder="Nhập tên hoặc username...", key="search_user_admin")
        if search_q:
            df_display = df_display[df_display.apply(lambda row: search_q.lower() in str(row).lower(), axis=1)]

        st.dataframe(df_display, use_container_width=True, height=400)

        st.markdown("---")
        st.markdown("### 🗑️ Xóa tài khoản")
        cols_del = st.columns([3, 1])
        with cols_del[0]:
            u_to_del = st.selectbox("Chọn tài khoản muốn xóa (Lưu ý: Không thể hoàn tác):", [u for u in users if u != "admin"], key="del_user_sel")
        with cols_del[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ YÊU CẦU XÓA", type="secondary", width="stretch", key="request_delete_user"):
                st.session_state["_pending_delete_user"] = u_to_del
        pending_delete_user = st.session_state.get("_pending_delete_user")
        if pending_delete_user:
            st.warning(f"Nhập chính xác tên tài khoản để xác nhận xóa: {pending_delete_user}")
            confirm_user = st.text_input("Xác nhận xóa tài khoản", key="confirm_delete_user")
            if st.button("Xác nhận xóa tài khoản", type="primary", width="stretch", key="confirm_delete_user_btn"):
                try:
                    actor = require_role("Quản trị viên", action="delete_user", target=pending_delete_user)
                    if confirm_user != pending_delete_user:
                        st.error("Chuỗi xác nhận không khớp.")
                    elif pending_delete_user in users and pending_delete_user != "admin":
                        create_backup_before_destructive("delete_user", [USER_DATA_FILE])
                        del users[pending_delete_user]
                        save_users(users)
                        write_audit_log(actor["username"], actor["role"], "delete_user", pending_delete_user, "success")
                        st.session_state.pop("_pending_delete_user", None)
                        st.success(f"✅ Đã xóa tài khoản '{pending_delete_user}'")
                        st.rerun()
                except PermissionError as exc:
                    st.error(str(exc))

    with tab_u2:
        st.markdown("### ➕ Cấp tài khoản mới")
        st.info("💡 Chỉ Admin mới có quyền tạo tài khoản cho Bác sĩ và Nghiên cứu viên.")

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            with st.form("admin_create_user"):
                new_u = st.text_input("👤 Tên đăng nhập *")
                new_n = st.text_input("📛 Họ và tên")
                new_e = st.text_input("📧 Email")
                new_p = st.text_input("🔑 Mật khẩu *", type="password")
                new_r = st.selectbox("🎭 Vai trò", ["Bác sĩ / KTV PHCN", "Nghiên cứu viên", "Quản trị viên"])

                if st.form_submit_button("🚀 TẠO TÀI KHOẢN", width="stretch"):
                    if new_u and new_p:
                        if new_u in users:
                            st.error("❌ Tên đăng nhập này đã được sử dụng!")
                        else:
                            now = get_vn_now().isoformat()
                            users[new_u] = {
                                **password_record_update(new_p, updated_at=now, must_change_password=True),
                                "full_name": new_n,
                                "role": new_r,
                                "email": new_e,
                                "created_at": now,
                            }
                            save_users(users)
                            st.success(f"✅ Đã tạo thành công tài khoản cho {new_n}!")
                            st.rerun()
                    else:
                        st.warning("⚠️ Vui lòng không bỏ trống Tên đăng nhập và Mật khẩu.")

        with col_f2:
            st.markdown("""
            <div class="info-box">
                <h4>📌 HƯỚNG DẪN ADMIN</h4>
                <p>1. <b>Tên đăng nhập:</b> Nên viết liền không dấu (VD: bacsi_an).</p>
                <p>2. <b>Mật khẩu:</b> Cung cấp mật khẩu tạm thời cho người dùng, sau đó yêu cầu họ đổi mật khẩu ở Tab "Đổi mật khẩu".</p>
                <p>3. <b>Vai trò:</b>
                    <ul>
                        <li><b>Bác sĩ:</b> Có quyền xem video BN, đánh giá lâm sàng.</li>
                        <li><b>Nghiên cứu viên:</b> Có quyền chạy AI phân tích xương.</li>
                    </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab_u3:
        st.markdown("### 📊 NHẬT KÝ HOẠT ĐỘNG HỆ THỐNG")
        st.info("💡 Bảng dưới đây tổng hợp tất cả các hoạt động của Bệnh nhân, Bác sĩ và NCV theo mốc thời gian.")

        # Load dữ liệu
        v_list = load_data(VIDEOS_FILE)
        e_list = load_data(EVALUATIONS_FILE)
        s_list = load_data(SYMPTOMS_FILE)

        # Tạo danh sách hoạt động tổng hợp
        all_activities = []

        # 1. Bệnh nhân Upload / Phân tích xong
        for v in v_list:
            upload_t = _parse_upload_time_from_filename(v.get("video_path") or v.get("video_name"))
            if upload_t:
                all_activities.append({
                    "Thời gian": upload_t,
                    "Người thực hiện": v.get('full_name', v.get('username', 'N/A')),
                    "Vai trò": "Bệnh nhân",
                    "Hành động": "📤 Upload Video",
                    "Chi tiết": f"Bài tập: {v.get('exercise')} | File: {v.get('video_name')}"
                })
            if v.get("status") == "Đã phân tích" and v.get("time"):
                all_activities.append({
                    "Thời gian": _format_vn_time(v.get("time"), default="N/A"),
                    "Người thực hiện": v.get('full_name', v.get('username', 'N/A')),
                    "Vai trò": "Nghiên cứu viên",
                    "Hành động": "✅ Phân tích AI xong",
                    "Chi tiết": f"BN: {v.get('username')} | {v.get('exercise')} | Acc: {v.get('accuracy', 0)}%"
                })

        # 2. Bác sĩ & NCV Đánh giá
        for e in e_list:
            is_ai = e.get('doctor_username') == "AI_Researcher"
            role = "Nghiên cứu viên" if is_ai else "Bác sĩ / KTV"
            action = "🤖 NCV Gửi kết quả AI" if is_ai else "👨‍⚕️ Bác sĩ Đánh giá"
            details = f"BN: {e.get('patient_username')} | KQ: {e.get('doctor_result')}"
            if is_ai: details += f" | AI Acc: {e.get('ai_accuracy')}%"

            all_activities.append({
                "Thời gian": e.get('time', 'N/A'),
                "Người thực hiện": e.get('doctor_name', e.get('doctor_username', 'N/A')),
                "Vai trò": role,
                "Hành động": action,
                "Chi tiết": details
            })

        # 3. Bệnh nhân gửi triệu chứng
        for s in s_list:
            all_activities.append({
                "Thời gian": s.get('time', 'N/A'),
                "Người thực hiện": s.get('full_name', s.get('username', 'N/A')),
                "Vai trò": "Bệnh nhân",
                "Hành động": "🩺 Gửi Triệu chứng (VAS)",
                "Chi tiết": f"Mức độ đau: {s.get('vas')}/10 | {s.get('symptoms')[:30]}..."
            })

        if not all_activities:
            st.info("📭 Hiện chưa có hoạt động nào được ghi nhận.")
        else:
            # Sắp xếp theo thời gian mới nhất (cần xử lý format thời gian vì nó không đồng nhất)
            # Thử parse thời gian để sort chuẩn hơn
            def parse_vn_time(t_str):
                try:
                    # Format 1: "H:M - d/m/Y"
                    return datetime.strptime(t_str, "%H:%M - %d/%m/%Y")
                except:
                    try:
                        # Format 2: "Y-m-d H:M:S" (từ đánh giá bác sĩ)
                        return datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                    except:
                        return datetime.min

            all_activities.sort(key=lambda x: parse_vn_time(x['Thời gian']), reverse=True)

            df_act = pd.DataFrame(all_activities)

            # Filter nhanh
            f_role = st.multiselect("Lọc theo vai trò:", ["Bệnh nhân", "Bác sĩ / KTV", "Nghiên cứu viên"], default=["Bệnh nhân", "Bác sĩ / KTV", "Nghiên cứu viên"])
            if f_role:
                df_act = df_act[df_act["Vai trò"].isin(f_role)]

            st.dataframe(df_act, use_container_width=True, height=500)

            # Nút xuất log
            csv_log = df_act.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Xuất nhật ký hoạt động (CSV)", csv_log, "system_log.csv", "text/csv")

    with tab_u4:
        st.markdown("### 🧹 Dọn dẹp dữ liệu hệ thống")
        st.warning("⚠️ CẢNH BÁO: Thao tác này sẽ xóa vĩnh viễn dữ liệu. Hãy cẩn thận!")

        st.markdown("#### 🔐 Phiên đăng nhập")
        st.caption(f"Session version hiện tại: `{get_global_session_version(SESSION_STATE_FILE)}`")
        if st.button("Thu hồi tất cả phiên đăng nhập", width="stretch", key="btn_revoke_all_sessions"):
            try:
                actor = require_role("Quản trị viên", action="session_revoke_all", target="global")
                new_version = revoke_all_sessions(actor, reason="admin_manual_revoke")
                st.success(f"Đã thu hồi các phiên cũ. Version mới: {new_version}.")
            except PermissionError as exc:
                st.error(str(exc))

        cleanup_actions = {
            "clear_evaluations_symptoms": {
                "label": "🗑️ XÓA TẤT CẢ LỊCH SỬ ĐÁNH GIÁ & TRIỆU CHỨNG",
                "confirm": "XOA DANH GIA",
                "files": [EVALUATIONS_FILE, SYMPTOMS_FILE],
            },
            "clear_reminders": {
                "label": "🗑️ XÓA TẤT CẢ LỊCH NHẮC NHỞ",
                "confirm": "XOA LICH",
                "files": [REMINDERS_FILE],
            },
            "clear_videos": {
                "label": "🗑️ XÓA DANH SÁCH VIDEO & FILE TẠM",
                "confirm": "XOA VIDEO",
                "files": [VIDEOS_FILE, UPLOAD_DIR, PROCESSED_DIR],
            },
            "reset_all": {
                "label": "💥 RESET TOÀN BỘ HỆ THỐNG (CLEAR ALL)",
                "confirm": "RESET TOAN BO",
                "files": [EVALUATIONS_FILE, SYMPTOMS_FILE, REMINDERS_FILE, VIDEOS_FILE, HISTORY_FILE, UPLOAD_DIR, PROCESSED_DIR],
            },
        }

        def _request_cleanup(action_key):
            st.session_state["_pending_cleanup_action"] = action_key

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.button(
                cleanup_actions["clear_evaluations_symptoms"]["label"],
                width="stretch",
                key="request_clear_evals_symptoms",
                on_click=_request_cleanup,
                args=("clear_evaluations_symptoms",),
            )

            st.button(
                cleanup_actions["clear_reminders"]["label"],
                width="stretch",
                key="request_clear_reminders",
                on_click=_request_cleanup,
                args=("clear_reminders",),
            )

        with col_c2:
            st.button(
                cleanup_actions["clear_videos"]["label"],
                width="stretch",
                key="request_clear_videos",
                on_click=_request_cleanup,
                args=("clear_videos",),
            )

            st.button(
                cleanup_actions["reset_all"]["label"],
                type="primary",
                width="stretch",
                key="request_reset_all",
                on_click=_request_cleanup,
                args=("reset_all",),
            )

        pending_cleanup = st.session_state.get("_pending_cleanup_action")
        if pending_cleanup in cleanup_actions:
            action = cleanup_actions[pending_cleanup]
            st.error(f"Đang chờ xác nhận: {action['label']}")
            confirm_text = st.text_input(
                f"Nhập `{action['confirm']}` để xác nhận",
                key="cleanup_confirm_text",
            )
            if st.button("Xác nhận thao tác xóa", type="primary", width="stretch", key="confirm_cleanup_btn"):
                try:
                    actor = require_role("Quản trị viên", action=pending_cleanup, target=action["label"])
                    if confirm_text != action["confirm"]:
                        st.error("Chuỗi xác nhận không khớp.")
                    else:
                        create_backup_before_destructive(pending_cleanup, action["files"])
                        if pending_cleanup in ("clear_evaluations_symptoms", "reset_all"):
                            save_data(EVALUATIONS_FILE, [])
                            save_data(SYMPTOMS_FILE, [])
                        if pending_cleanup in ("clear_reminders", "reset_all"):
                            save_data(REMINDERS_FILE, [])
                        if pending_cleanup in ("clear_videos", "reset_all"):
                            save_data(VIDEOS_FILE, [])
                            _remove_files_in_dir(UPLOAD_DIR)
                            _remove_files_in_dir(PROCESSED_DIR)
                        if pending_cleanup == "reset_all" and os.path.exists(HISTORY_FILE):
                            save_data(HISTORY_FILE, [])
                            revoke_all_sessions(actor, reason="admin_reset_all")
                            for key in list(st.session_state.keys()):
                                if key not in ['logged_in', 'user_info', 'theme']:
                                    del st.session_state[key]
                        write_audit_log(actor["username"], actor["role"], pending_cleanup, action["label"], "success")
                        st.session_state.pop("_pending_cleanup_action", None)
                        st.success("✅ Đã thực hiện thao tác sau khi backup và ghi audit.")
                        st.rerun()
                except PermissionError as exc:
                    st.error(str(exc))


def render_admin_home_page(deps):
    st = deps.st
    pd = deps.pd
    px = deps.px
    os = deps.os
    datetime = deps.datetime
    EVALUATIONS_FILE = deps.EVALUATIONS_FILE
    HISTORY_FILE = deps.HISTORY_FILE
    PROCESSED_DIR = deps.PROCESSED_DIR
    REMINDERS_FILE = deps.REMINDERS_FILE
    SESSION_STATE_FILE = deps.SESSION_STATE_FILE
    SYMPTOMS_FILE = deps.SYMPTOMS_FILE
    UPLOAD_DIR = deps.UPLOAD_DIR
    USER_DATA_FILE = deps.USER_DATA_FILE
    VIDEOS_FILE = deps.VIDEOS_FILE
    _format_vn_time = deps.format_vn_time
    _parse_upload_time_from_filename = deps.parse_upload_time_from_filename
    _remove_files_in_dir = deps.remove_files_in_dir
    create_backup_before_destructive = deps.create_backup_before_destructive
    get_global_session_version = deps.get_global_session_version
    get_vn_now = deps.get_vn_now
    load_data = deps.load_data
    load_users = deps.load_users
    password_record_update = deps.password_record_update
    require_role = deps.require_role
    revoke_all_sessions = deps.revoke_all_sessions
    save_data = deps.save_data
    save_users = deps.save_users
    write_audit_log = deps.write_audit_log

    """Trang chủ chuyên biệt dành cho Quản trị viên (Admin Dashboard)"""
    st.markdown("## 📊 HỆ THỐNG QUẢN TRỊ TỔNG THỂ")

    # Load dữ liệu
    users = load_users()
    v_list = load_data(VIDEOS_FILE)
    e_list = load_data(EVALUATIONS_FILE)
    s_list = load_data(SYMPTOMS_FILE)

    # Tính toán chỉ số
    total_users = len(users)
    patients = len([u for u in users.values() if u.get('role') == 'Bệnh nhân'])
    doctors = len([u for u in users.values() if u.get('role') == 'Bác sĩ / KTV PHCN'])
    ncvs = len([u for u in users.values() if u.get('role') == 'Nghiên cứu viên'])

    total_vids = len(v_list)
    total_evals = len(e_list)

    # Tính tổng số frame AI đã xử lý (nếu có)
    total_frames = 0
    for v in v_list:
        if v.get('metrics') and isinstance(v['metrics'], dict):
            total_frames += v['metrics'].get('tong_frame', 0)

    # Hiển thị các Metric Card cao cấp
    is_light = st.session_state.theme == 'light'
    card_bg = "rgba(255, 255, 255, 1)" if is_light else "rgba(255, 255, 255, 0.05)"

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""<div class="metric-card" style="background:{card_bg};"><div class="metric-label">👥 Người dùng</div><div class="metric-value">{total_users}</div><div style="font-size:0.8rem; color:#888;">{patients} BN | {doctors} BS</div></div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""<div class="metric-card" style="background:{card_bg};"><div class="metric-label">🎬 Tổng Video</div><div class="metric-value">{total_vids}</div><div style="font-size:0.8rem; color:#888;">Video đã nhận</div></div>""", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""<div class="metric-card" style="background:{card_bg};"><div class="metric-label">📝 Đánh giá</div><div class="metric-value">{total_evals}</div><div style="font-size:0.8rem; color:#888;">Bản ghi chuyên môn</div></div>""", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""<div class="metric-card" style="background:{card_bg};"><div class="metric-label">⚡ Frames Xử lý</div><div class="metric-value">{total_frames:,}</div><div style="font-size:0.8rem; color:#888;">Dữ liệu qua AI</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Biểu đồ thống kê
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📈 Mức độ phổ biến của Bài tập")
        if v_list:
            ex_data = [v.get('exercise') for v in v_list if v.get('exercise')]
            if ex_data:
                ex_counts = pd.Series(ex_data).value_counts().reset_index()
                ex_counts.columns = ['Bài tập', 'Số lượt']
                fig_ex = px.bar(ex_counts, x='Bài tập', y='Số lượt', color='Bài tập',
                               color_discrete_sequence=px.colors.qualitative.Pastel,
                               template="plotly_dark" if not is_light else "plotly_white")
                fig_ex.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=350, showlegend=False)
                st.plotly_chart(fig_ex, use_container_width=True, theme=None)
            else:
                st.info("Chưa có dữ liệu bài tập.")
        else:
            st.info("Chưa có dữ liệu video.")

    with c2:
        st.markdown("### 📊 Cơ cấu Vai trò Hệ thống")
        role_data = [u.get('role') for u in users.values()]
        role_counts = pd.Series(role_data).value_counts().reset_index()
        role_counts.columns = ['Vai trò', 'Số lượng']
        fig_role = px.pie(role_counts, values='Số lượng', names='Vai trò', hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         template="plotly_dark" if not is_light else "plotly_white")
        fig_role.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=350)
        st.plotly_chart(fig_role, use_container_width=True, theme=None)

    st.markdown("---")
    st.markdown("### 📊 Bảng thống kê chi tiết kết quả phân tích & đánh giá")
    if v_list:
        table_rows = []
        # Tải dữ liệu và xây dựng bảng lookup tối ưu cho đánh giá
        ai_evals_dict = {}
        doc_evals_dict = {}
        for e in e_list:
            key = (e.get('patient_username'), e.get('video_name'), e.get('exercise'))
            if e.get('doctor_username') == "AI_Researcher":
                ai_evals_dict[key] = e
            else:
                doc_evals_dict[key] = e

        # Xây dựng bảng lookup tối ưu cho triệu chứng (s_list)
        symptoms_dict = {}
        symptoms_by_user = {}
        for s in s_list:
            s_username = s.get('username')
            s_exercise = s.get('exercise')
            symptoms_dict[(s_username, s_exercise)] = s
            symptoms_by_user[s_username] = s

        for v in v_list:
            v_username = v.get('username')
            v_key = (v_username, v.get('video_name'), v.get('exercise'))
            ai_eval = ai_evals_dict.get(v_key)
            doc_eval = doc_evals_dict.get(v_key)

            # Tra cứu thông tin triệu chứng lâm sàng
            # Ưu tiên theo bài tập cụ thể, nếu không có thì lấy lần khai báo triệu chứng gần nhất của bệnh nhân đó
            symp = symptoms_dict.get((v_username, v.get('exercise'))) or symptoms_by_user.get(v_username)
            if symp:
                patient_id = symp.get('patient_id', 'N/A')
                age = symp.get('age', 'N/A')
                gender = symp.get('gender', 'N/A')
                desc = symp.get('symptoms', '').strip()
                vas = symp.get('vas', 'N/A')

                demographics = f"{age} tuổi / {gender}"
                symptom_summary = f"{desc} (VAS: {vas}/10)" if desc else f"Đau mức {vas}/10 (Không mô tả thêm)"
            else:
                patient_id = "N/A"
                demographics = "Chưa khai báo"
                symptom_summary = "Chưa khai báo"

            # Thống kê frame hình
            metrics = v.get('metrics', {}) if isinstance(v.get('metrics'), dict) else {}
            if v.get('status') == "Đã phân tích" and metrics:
                tong_frame = metrics.get('tong_frame_hop_le', metrics.get('tong_frame', 0))
                if not tong_frame:
                    tong_frame = metrics.get('tong_frame', 0)
                frame_dung = metrics.get('frame_dung', 0)
                frame_gan_dung = metrics.get('frame_gan_dung', 0)
                frame_sai = max(0, tong_frame - frame_dung - frame_gan_dung)

                tong_str = str(tong_frame)
                dung_str = str(frame_dung)
                gan_str = str(frame_gan_dung)
                sai_str = str(frame_sai)
            else:
                tong_str = "Chờ xử lý"
                dung_str = "-"
                gan_str = "-"
                sai_str = "-"

            # Đánh giá AI
            if ai_eval:
                ai_accuracy = ai_eval.get('ai_accuracy', 0)
                ai_res = ai_eval.get('doctor_result', 'N/A')
                ai_comment = f"{ai_accuracy:.1f}% ({ai_res})"
            else:
                ai_comment = "Chờ phân tích"

            # Nhận xét Bác sĩ
            if doc_eval:
                doc_res = doc_eval.get('doctor_result', 'N/A')
                doc_text = doc_eval.get('comments', '')
                doc_comment = f"{doc_res}: {doc_text}"
            else:
                doc_comment = "Chờ bác sĩ đánh giá"

            table_rows.append({
                "Bệnh nhân": v.get('full_name', 'N/A'),
                "Tài khoản": v_username,
                "Mã BN": patient_id,
                "Tuổi/GT": demographics,
                "Triệu chứng khai báo": symptom_summary,
                "Bài tập": v.get('exercise', 'N/A'),
                "Thời gian": v.get('time', 'N/A'),
                "Tổng Frames": tong_str,
                "Frames Đúng": dung_str,
                "Frames Gần Đúng": gan_str,
                "Frames Sai": sai_str,
                "Đánh giá AI": ai_comment,
                "Nhận xét của Bác sĩ": doc_comment
            })

        df_stats = pd.DataFrame(table_rows)
        st.dataframe(df_stats, use_container_width=True, height=400)
    else:
        st.info("Chưa có dữ liệu video để thống kê.")

    # Thống kê hoạt động theo thời gian (giả lập hoặc từ logs)
    st.markdown("### 🕒 Lưu lượng hoạt động gần đây")
    if v_list or e_list or s_list:
        # Lấy 10 hoạt động mới nhất
        recent_acts = []
        for v in v_list[-5:]: recent_acts.append({"Time": v.get('time'), "Event": "📤 Video Upload", "User": v.get('username')})
        for e in e_list[-5:]: recent_acts.append({"Time": e.get('time'), "Event": "📝 Evaluation", "User": e.get('doctor_username')})

        # Sắp xếp thô theo chuỗi thời gian
        recent_acts.sort(key=lambda x: str(x['Time']), reverse=True)
        df_recent = pd.DataFrame(recent_acts[:8])
        st.table(df_recent)
    else:
        st.info("Chưa có hoạt động nào.")
