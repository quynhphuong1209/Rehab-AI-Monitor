"""Doctor evaluation form UI extracted from app.py."""

from __future__ import annotations


def _bind_deps(deps):
    """Bind app services expected by the moved doctor form UI body."""
    global st, pd, DOCTOR_ROLE, EVALUATIONS_FILE
    global _dedup_evaluations, _format_vn_time, _lay_danh_gia_cho_video
    global _lay_eval_moi_nhat_theo_bai_tap, _lay_thoi_gian_phan_tich_on_dinh
    global current_actor_can_access_patient, get_vn_now, lay_danh_gia_ai_benh_nhan
    global load_data, nap_ket_qua_ai_vao_session, patient_display_label, render_video
    global require_patient_scope, require_role, safe_html, save_data, scope_records_for_current_actor
    global write_audit_log, hien_thi_noi_dung_ket_qua

    st = deps.st
    pd = deps.pd
    DOCTOR_ROLE = deps.DOCTOR_ROLE
    EVALUATIONS_FILE = deps.EVALUATIONS_FILE
    _dedup_evaluations = deps.dedup_evaluations
    _format_vn_time = deps.format_vn_time
    _lay_danh_gia_cho_video = deps.get_video_evaluations
    _lay_eval_moi_nhat_theo_bai_tap = deps.latest_eval_by_exercise
    _lay_thoi_gian_phan_tich_on_dinh = deps.stable_analysis_time
    current_actor_can_access_patient = deps.current_actor_can_access_patient
    get_vn_now = deps.get_vn_now
    lay_danh_gia_ai_benh_nhan = deps.get_patient_ai_evaluations
    load_data = deps.load_data
    nap_ket_qua_ai_vao_session = deps.load_ai_result_into_session
    patient_display_label = deps.patient_display_label
    render_video = deps.render_video
    require_patient_scope = deps.require_patient_scope
    require_role = deps.require_role
    safe_html = deps.safe_html
    save_data = deps.save_data
    scope_records_for_current_actor = deps.scope_records_for_current_actor
    write_audit_log = deps.write_audit_log
    hien_thi_noi_dung_ket_qua = deps.render_selected_result_content


def render_latest_results_and_history(deps,
    username,
    video_name=None,
    exercise=None,
    selected_v=None,
    key_suffix="",
    chi_nhan_xet=False,
):
    _bind_deps(deps)
    """Hiển thị kết quả gần nhất theo đúng BN + bài tập đang xem."""
    if not username and not selected_v:
        return
    evals = _dedup_evaluations(load_data(EVALUATIONS_FILE))
    if selected_v is None:
        selected_v = {
            "username": username,
            "patient_username": username,
            "video_name": video_name,
            "exercise": exercise,
        }
    pu = selected_v.get("username") or selected_v.get("patient_username") or username
    ex_cur = selected_v.get("exercise") or exercise
    ai_eval, doc_eval = _lay_danh_gia_cho_video(selected_v, evals)
    if doc_eval:
        latest_doc = doc_eval
        t_doc = safe_html(_format_vn_time(latest_doc.get("time"), default="N/A"))
        ex_doc = safe_html(latest_doc.get('exercise', 'N/A'))
        result_doc = safe_html(latest_doc.get('doctor_result', 'N/A'))
        doc_comment = safe_html(latest_doc.get('comments', ''), max_length=200)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,215,0,0.12) 0%, rgba(255,165,0,0.08) 100%);
            border: 1px solid rgba(255,215,0,0.35); border-left: 5px solid #ffd700; border-radius: 14px;
            padding: 18px 20px; margin-bottom: 16px;">
            <p style="margin:0 0 6px 0; font-size:0.8rem; color:#888; text-transform:uppercase; letter-spacing:1px;">
                👨‍⚕️ Đánh giá Bác sĩ / KTV gần nhất
            </p>
            <p style="margin:0; font-size:1.05rem; color:#fff; font-weight:600;">
                🕒 {t_doc} — {ex_doc}
            </p>
            <p style="margin:6px 0 0; font-size:0.95rem; color:#ffd700;">
                Kết quả: <b>{result_doc}</b>
            </p>
            <p style="margin:6px 0 0; font-size:0.88rem; color:#ccc;">
                {doc_comment}
            </p>
        </div>
        """, unsafe_allow_html=True)

    vn_cur = selected_v.get("video_name") or video_name
    ai_history = lay_danh_gia_ai_benh_nhan(pu, vn_cur, exercise=ex_cur)
    if ai_eval and (not ai_history or ai_history[0] is not ai_eval):
        ai_history = [ai_eval] + [e for e in ai_history if e is not ai_eval]
    if not ai_history:
        return

    latest = ai_history[0]
    verdict = safe_html(latest.get("doctor_result", "N/A"))
    t_latest = safe_html(_format_vn_time(latest.get("time"), default="N/A"))
    ex_latest = safe_html(latest.get("exercise", "N/A"))
    ai_comment = safe_html(latest.get("comments") or "", max_length=200)

    if chi_nhan_xet:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,198,255,0.12) 0%, rgba(0,114,255,0.08) 100%);
            border: 1px solid rgba(0,198,255,0.35); border-left: 5px solid #00c6ff; border-radius: 14px;
            padding: 18px 20px; margin-bottom: 16px;">
            <p style="margin:0 0 6px 0; font-size:0.8rem; color:#888; text-transform:uppercase; letter-spacing:1px;">
                🤖 Nhận xét NCV / AI gần nhất
            </p>
            <p style="margin:0; font-size:1.05rem; color:#fff; font-weight:600;">
                🕒 {t_latest} — {ex_latest}
            </p>
            <p style="margin:6px 0 0; font-size:0.95rem; color:#00c6ff;">
                Kết quả: <b>{verdict}</b>
            </p>
            <p style="margin:6px 0 0; font-size:0.88rem; color:#ccc;">
                {ai_comment}
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    acc = latest.get("ai_accuracy", 0)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(0,198,255,0.12) 0%, rgba(0,114,255,0.08) 100%);
        border: 1px solid rgba(0,198,255,0.35); border-left: 5px solid #00c6ff; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 16px;">
        <p style="margin:0 0 6px 0; font-size:0.8rem; color:#888; text-transform:uppercase; letter-spacing:1px;">
            📌 Kết quả gần đây nhất
        </p>
        <p style="margin:0; font-size:1.05rem; color:#fff; font-weight:600;">
            🕒 {t_latest} — {ex_latest}
        </p>
        <p style="margin:6px 0 0; font-size:0.95rem; color:#00c6ff;">
            {verdict} · Độ chính xác AI: <b>{acc}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if len(ai_history) <= 1:
        return

    st.markdown("#### 📜 XEM LẠI KẾT QUẢ PHÂN TÍCH TRƯỚC ĐÓ")

    def _eval_label(e):
        acc_e = e.get("ai_accuracy", 0)
        return (
            f"🕒 {_format_vn_time(e.get('time'), default='N/A')} — {e.get('exercise', 'N/A')} "
            f"({e.get('doctor_result', 'N/A')}: {acc_e}%)"
        )

    hist_opts = [{"label": _eval_label(e), "val": e} for e in ai_history]
    picked = st.selectbox(
        "Chọn phiên phân tích:",
        hist_opts,
        format_func=lambda x: x["label"],
        key=f"ncv_ai_history_{key_suffix}_{username}",
    )
    if st.button(
        "📂 TẢI KẾT QUẢ ĐÃ CHỌN",
        key=f"btn_load_ai_hist_{key_suffix}_{username}",
        type="secondary",
        use_container_width=True,
    ):
        nap_ket_qua_ai_vao_session(picked["val"])
        st.toast("✅ Đã tải kết quả phân tích đã chọn!", icon="📂")
        st.rerun()

def render_doctor_evaluation_form(deps):
    _bind_deps(deps)
    st.markdown("## 📝 PHIẾU ĐÁNH GIÁ KỸ THUẬT ĐỘNG TÁC (GROUND TRUTH)")

    selected_video = st.session_state.get('current_eval_video')
    evals = load_data(EVALUATIONS_FILE)
    evals_visible = scope_records_for_current_actor(evals)
    my_history = [e for e in evals if e.get('doctor_username') == st.session_state.user_info['username']]

    if not selected_video:
        st.info("💡 Chọn một video ở TRANG CHỦ để bắt đầu đánh giá mới. Danh sách các đánh giá cũ hiển thị ở phía dưới.")
    elif not current_actor_can_access_patient(selected_video.get("username")):
        st.error("Bạn không có quyền đánh giá video của bệnh nhân ngoài phạm vi phụ trách.")
        selected_video = None
    else:
        existing_eval = next((e for e in my_history if
                             e.get('patient_username') == selected_video.get('username') and
                             e.get('video_name') == selected_video.get('video_name')), None)

        if existing_eval and not st.session_state.get('re_eval_mode'):
            st.success(f"✅ BẠN ĐÃ ĐÁNH GIÁ VIDEO: {selected_video.get('full_name', selected_video.get('username', 'Không rõ'))}")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Kết quả", existing_eval.get('doctor_result', 'N/A'))
                st.write(f"**Thời gian:** {existing_eval.get('time', 'N/A')}")
            with c2:
                st.info(f"**Nhận xét cho BN:** {existing_eval.get('comments', '')}")
                st.warning(f"**Ghi chú cho NCV:** {existing_eval.get('comments_ncv', 'N/A')}")

            if st.button("🔄 Đánh giá lại bài tập này"):
                st.session_state.re_eval_mode = True
                st.rerun()
            st.markdown("---")

        else:
            if st.session_state.get('re_eval_mode'):
                if st.button("⬅️ Quay lại xem tóm tắt"):
                    st.session_state.re_eval_mode = False
                    st.rerun()

            st.markdown(f"#### 🎬 Đang đánh giá: {selected_video.get('full_name', selected_video.get('username', 'Không rõ'))} - {selected_video.get('exercise', 'N/A')}")

            v_to_render = selected_video.get('processed_path') if (selected_video.get('status') == "Đã phân tích" and selected_video.get('processed_path')) else selected_video.get('video_path')
            if v_to_render:
                render_video(v_to_render, check_h264=(selected_video.get('status') == "Đã phân tích"))

            with st.form("doctor_eval_form_final_v_fixed"):
                col1, col2 = st.columns(2)
                with col1:
                    k_qua = st.radio("Kết quả:", ["Đúng", "Sai", "Gần đúng"])
                with col2:
                    l_sai = st.multiselect("Lỗi sai:", ["Vị trí tay chưa đúng", "Biên độ chưa đạt", "Tốc độ quá nhanh/chậm", "Sai tư thế thân người"])

                n_xet = st.text_area("Nhận xét cho BN:", height=80)
                n_xet_ncv = st.text_area("Ghi chú cho NCV:", height=80)
                k_hoach = st.radio("Chỉ định:", ["Tiếp tục", "Chuyển bài", "Khám lại"])

                submitted = st.form_submit_button("🚀 GỬI ĐÁNH GIÁ", type="primary")

            if submitted:
                try:
                    require_role(DOCTOR_ROLE, action="create_evaluation", target=selected_video.get('video_name'))
                    require_patient_scope(selected_video.get('username'), action="create_evaluation")
                    new_e = {
                        "patient_username": selected_video.get('username'),
                        "doctor_username": st.session_state.user_info['username'],
                        "doctor_name": st.session_state.user_info.get('full_name', st.session_state.user_info['username']),
                        "video_name": selected_video.get('video_name'),
                        "exercise": selected_video.get('exercise'),
                        "doctor_result": k_qua,
                        "errors": l_sai,
                        "comments": n_xet,
                        "comments_ncv": n_xet_ncv,
                        "plan": k_hoach,
                        "time": get_vn_now().strftime("%H:%M - %d/%m/%Y")
                    }
                    evals = [e for e in evals if not (e.get('patient_username') == new_e['patient_username'] and e.get('video_name') == new_e['video_name'] and e.get('exercise') == new_e['exercise'] and e.get('doctor_username') == new_e['doctor_username'])]
                    evals.append(new_e)
                    save_data(EVALUATIONS_FILE, evals)
                    write_audit_log(st.session_state.user_info['username'], DOCTOR_ROLE, "create_evaluation", selected_video.get('video_name'), "success")
                    st.session_state.re_eval_mode = False
                    st.success("✅ Gửi thành công!")
                    st.rerun()
                except PermissionError as exc:
                    st.error(str(exc))


    # 2. PHẦN NHẬT KÝ LỊCH SỬ (DƯỚI CÙNG - LUÔN HIỆN)
    # Hiển thị TẤT CẢ đánh giá từ bác sĩ/KTV (không phải AI_Researcher)
    st.markdown("---")
    st.markdown("### 📜 NHẬT KÝ ĐÁNH GIÁ LÂM SÀNG")

    all_doctor_history = [
        e for e in evals_visible
        if e.get('doctor_username') not in (None, "", "AI_Researcher")
    ]
    all_doctor_history = list(reversed(all_doctor_history))  # Mới nhất lên đầu

    if not all_doctor_history:
        st.info("📭 Chưa có bản ghi đánh giá lâm sàng nào từ Bác sĩ / KTV PHCN.")
    else:
        user_role = st.session_state.user_info.get('role', 'Bác sĩ / KTV PHCN')
        if user_role in ["Bác sĩ / KTV PHCN", "Nghiên cứu viên", "Quản trị viên"]:
            c_exp_doc1, c_exp_doc2 = st.columns([1, 4])
            with c_exp_doc1:
                df_export_doc = pd.DataFrame(all_doctor_history)
                csv_doc = df_export_doc.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📊 Xuất Excel (CSV)",
                    data=csv_doc,
                    file_name=f"clinical_evaluations_{get_vn_now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="btn_export_clinical_evals",
                    width="stretch"
                )
        # --- Bộ lọc nhanh ---
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
        with filter_col1:
            all_patients_hist = sorted(set(h.get('patient_username', '') for h in all_doctor_history if h.get('patient_username')))
            filter_patient = st.selectbox("🔍 Lọc theo bệnh nhân:", ["-- Tất cả --"] + all_patients_hist, key="filter_doc_hist_patient")
        with filter_col2:
            filter_result = st.selectbox("📊 Lọc theo kết quả:", ["-- Tất cả --", "Đúng", "Gần đúng", "Sai"], key="filter_doc_hist_result")
        with filter_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            show_only_mine = st.toggle("👤 Chỉ của tôi", value=False, key="filter_doc_hist_mine")

        # Áp dụng bộ lọc
        filtered_history = all_doctor_history
        if show_only_mine:
            filtered_history = [h for h in filtered_history if h.get('doctor_username') == st.session_state.user_info['username']]
        if filter_patient != "-- Tất cả --":
            filtered_history = [h for h in filtered_history if h.get('patient_username') == filter_patient]
        if filter_result != "-- Tất cả --":
            filtered_history = [h for h in filtered_history if h.get('doctor_result') == filter_result]

        # Badge màu theo kết quả
        def result_badge_doc(result):
            color_map = {"Đúng": "#2ecc71", "Gần đúng": "#f39c12", "Sai": "#e74c3c"}
            color = color_map.get(result, "#95a5a6")
            return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:bold;">{safe_html(result, max_length=40)}</span>'

        st.caption(f"Hiển thị **{len(filtered_history)}** / {len(all_doctor_history)} bản ghi đánh giá lâm sàng")

        for i, h in enumerate(filtered_history):
            is_mine = h.get('doctor_username') == st.session_state.user_info['username']
            doc_label = h.get('doctor_name') or h.get('doctor_username', 'N/A')
            doc_label_html = safe_html(doc_label, max_length=120)
            mine_tag = " 👤" if is_mine else ""

            eval_time_formatted = _format_vn_time(h.get('time'), default='N/A')

            col_main_h, col_del_h = st.columns([12, 1])
            with col_main_h:
                exercise_name = h.get('exercise', 'N/A')
                expander_label = f"🕒 {eval_time_formatted} | BN: {h.get('patient_username', 'N/A')} | Động tác: {exercise_name} | BS: {doc_label}{mine_tag} | KQ: {h.get('doctor_result', '')}"
                with st.expander(expander_label):
                    st.markdown(
                        f"**Kết quả:** {result_badge_doc(h.get('doctor_result', ''))} &nbsp;&nbsp;"
                        f"**Bác sĩ/KTV:** `{doc_label_html}` &nbsp;&nbsp;"
                        f"**Thời gian:** `{safe_html(eval_time_formatted, max_length=60)}`",
                        unsafe_allow_html=True
                    )
                    col_h1, col_h2 = st.columns(2)
                    with col_h1:
                        st.write(f"**Bài tập:** {h.get('exercise', 'N/A')}")
                        st.write(f"**Bệnh nhân:** {h.get('patient_username', 'N/A')}")
                        if h.get('errors'):
                            st.write(f"**Lỗi:** {', '.join(h.get('errors', []))}")
                        st.write(f"**Chỉ định:** {h.get('plan', 'N/A')}")
                    with col_h2:
                        if h.get('comments'):
                            st.success(f"**Nhận xét BN:** {h.get('comments')}")
                        if h.get('comments_ncv'):
                            st.info(f"**Ghi chú NCV:** {h.get('comments_ncv')}")
            with col_del_h:
                st.write("")  # Căn chỉnh nút
                if is_mine:
                    if st.button("❌", key=f"del_doc_h_{i}", help="Xóa bản ghi đánh giá này (chỉ đánh giá của bạn)"):
                        try:
                            require_role(DOCTOR_ROLE, action="delete_evaluation", target=h.get('video_name'))
                            require_patient_scope(h.get('patient_username'), action="delete_evaluation")
                            all_evals = load_data(EVALUATIONS_FILE)
                            all_evals = [e for e in all_evals if not (
                                e.get('time') == h.get('time') and
                                e.get('patient_username') == h.get('patient_username') and
                                e.get('doctor_username') == st.session_state.user_info['username']
                            )]
                            save_data(EVALUATIONS_FILE, all_evals)
                            write_audit_log(st.session_state.user_info['username'], DOCTOR_ROLE, "delete_evaluation", h.get('video_name'), "success")
                            st.success("Đã xóa bản ghi!")
                            st.rerun()
                        except PermissionError as exc:
                            st.error(str(exc))
                else:
                    st.markdown("<span title='Bạn không thể xóa đánh giá của bác sĩ khác' style='color:#555;font-size:1.1rem;cursor:default;'>🔒</span>", unsafe_allow_html=True)

def render_selected_results_tab(deps, my_history_vids, my_evals, user_role, is_fresh_session=False):
    _bind_deps(deps)
    """Fragment: chọn phiên tập + tab kết quả — chỉ reload vùng này (nhanh cho bệnh nhân)."""
    selected_v = None

    p_username_hist = None
    if my_history_vids:
        p_username_hist = my_history_vids[0].get("username")
    elif my_evals:
        p_username_hist = my_evals[0].get("patient_username")
    if p_username_hist and user_role in ("Bệnh nhân", "Bác sĩ / KTV PHCN", "Nghiên cứu viên"):
        v_ctx = st.session_state.get("current_eval_video") or (my_history_vids[0] if my_history_vids else None)
        render_latest_results_and_history(
            deps,
            p_username_hist,
            video_name=v_ctx.get("video_name") if v_ctx else None,
            exercise=v_ctx.get("exercise") if v_ctx else None,
            selected_v=v_ctx,
            key_suffix=f"pat_hist_{user_role}",
            chi_nhan_xet=True,
        )

    if my_history_vids:
        if is_fresh_session:
            current_selection = st.session_state.get('patient_history_selector_global')
            is_viewing_history = current_selection is not None and current_selection.get('val') is not None
            if not is_viewing_history:
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(0,114,255,0.08) 0%, rgba(0,198,255,0.08) 100%);
                    border: 1px solid rgba(0,198,255,0.3); border-left: 5px solid #00c6ff; border-radius: 16px;
                    padding: 28px 24px; text-align: center; margin: 0 0 20px 0;">
                    <div style="font-size: 3rem; margin-bottom: 12px;">⏳</div>
                    <h3 style="color: #00c6ff; margin: 0 0 10px 0; font-size: 1.3rem;">
                        Đang chờ Nghiên cứu viên gửi kết quả bài tập mới
                    </h3>
                    <p style="color: #aaa; margin: 0; font-size: 0.95rem; line-height: 1.6;">
                        Bạn đã gửi video tập luyện. Nghiên cứu viên (NCV) đang phân tích và sẽ gửi kết quả AI cho bạn sớm nhất có thể.<br>
                        <span style="color: #ffd700;">💡 Trong lúc chờ, bạn có thể xem lại lịch sử tập luyện bên dưới.</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("### 📅 XEM LẠI LỊCH SỬ TẬP LUYỆN")

        if user_role == "Bệnh nhân":
            def _hist_label(v):
                ai_e = _lay_eval_moi_nhat_theo_bai_tap(
                    my_evals, v.get('username'), v.get('exercise'), doctor_username="AI_Researcher"
                )
                doc_e = _lay_eval_moi_nhat_theo_bai_tap(
                    my_evals, v.get('username'), v.get('exercise')
                )
                t_show = _lay_thoi_gian_phan_tich_on_dinh(v, ai_e) or "Chưa phân tích"
                parts = [f"🕒 {t_show} - Bài: {v.get('exercise')}"]
                if ai_e and ai_e.get("doctor_result"):
                    parts.append(f"AI: {ai_e.get('doctor_result')}")
                if doc_e and doc_e.get("doctor_result"):
                    parts.append(f"BS: {doc_e.get('doctor_result')}")
                return " · ".join(parts)
            history_opts = [{"label": "--- Đang chờ kết quả mới (Ẩn lịch sử) ---", "val": None}] + [{"label": _hist_label(v), "val": v} for v in my_history_vids]
        else:
            history_opts = [{"label": "--- Chọn một phiên tập để xem ---", "val": None}] + [
                {
                    "label": (
                        f"🕒 {_lay_thoi_gian_phan_tich_on_dinh(v, _lay_eval_moi_nhat_theo_bai_tap(my_evals, v.get('username'), v.get('exercise'), doctor_username='AI_Researcher')) or 'Chưa phân tích'} "
                        f"- {patient_display_label(v, include_username=False)} - {v.get('exercise')}"
                    ),
                    "val": v,
                }
                for v in my_history_vids
            ]

        selected_opt = st.selectbox(
            "Lựa chọn phiên tập:",
            history_opts,
            format_func=lambda x: x["label"],
            key="patient_history_selector_global"
        )
        selected_v = selected_opt["val"]

        if selected_v:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 LÀM MỚI (QUAY LẠI CHỜ KẾT QUẢ)", width="stretch", type="secondary"):
                del st.session_state['patient_history_selector_global']
                st.session_state.pop("_patient_session_key", None)
                st.rerun()
        else:
            selected_v = my_history_vids[0]
            st.markdown("---")
            if st.button("🔄 LÀM MỚI ĐỂ TẬP BÀI KHÁC", width="stretch", type="primary", key="btn_lam_moi_bn_global"):
                for key in ['has_data', 'stats', 'angle_df', 'processed_video_path',
                            'current_df_csv_path', 'uploaded_file_name', 'all_frames_data_path',
                            'processing', 'temp_folder', 'zip_data', 'frame_paths', 'active_video_name',
                            'patient_history_selector_global', '_patient_session_key']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.fresh_session = True
                st.session_state.uploader_id = st.session_state.get('uploader_id', 0) + 1
                st.rerun()

    if not selected_v:
        return

    hien_thi_noi_dung_ket_qua(selected_v, my_evals)
