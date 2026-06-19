"""Role-aware tab navigation helpers for the Streamlit app."""

from __future__ import annotations

from collections.abc import MutableMapping


ADMIN_TABS = [
    "🏠 TRANG CHỦ",
    "🛠️ QUẢN TRỊ VIÊN",
    "📚 THÔNG TIN TỔNG HỢP",
    "👥 HỒ SƠ ĐỀ TÀI & ĐỘI NGŨ CHUYÊN GIA",
    "💬 PHẢN HỒI",
]

DOCTOR_BASE_TABS = [
    "🏠 TRANG CHỦ",
    "📊 QUẢN LÝ ĐÁNH GIÁ & NCKH",
]

DOCTOR_TAIL_TABS = [
    "⏰ LỊCH NHẮC NHỞ",
    "📚 THÔNG TIN TỔNG HỢP",
    "👥 HỒ SƠ ĐỀ TÀI & ĐỘI NGŨ CHUYÊN GIA",
    "📞 THÔNG TIN LIÊN HỆ",
    "💬 PHẢN HỒI",
]

PATIENT_TABS = [
    "🏠 TRANG CHỦ",
    "📊 KẾT QUẢ ĐÁNH GIÁ",
    "⏰ LỊCH NHẮC NHỞ",
    "📚 THÔNG TIN TỔNG HỢP",
    "📞 THÔNG TIN LIÊN HỆ",
    "💬 PHẢN HỒI",
]

RESEARCHER_TABS = [
    "🏠 TRANG CHỦ",
    "📊 KẾT QUẢ ĐÁNH GIÁ",
    "🔬 PHÂN TÍCH & TRÍCH XUẤT DỮ LIỆU",
    "📚 THÔNG TIN TỔNG HỢP",
    "👥 HỒ SƠ ĐỀ TÀI & ĐỘI NGŨ CHUYÊN GIA",
    "💬 PHẢN HỒI",
]


def tab_titles_for_role(user_role: str, *, has_video_output: bool = False) -> list[str]:
    if user_role == "Quản trị viên":
        return list(ADMIN_TABS)
    if user_role == "Bác sĩ / KTV PHCN":
        tabs = list(DOCTOR_BASE_TABS)
        if has_video_output:
            tabs.append("🎬 VIDEO & ẢNH")
        tabs.extend(DOCTOR_TAIL_TABS)
        return tabs
    if user_role == "Bệnh nhân":
        return list(PATIENT_TABS)
    return list(RESEARCHER_TABS)


def sync_active_tab_state(
    session_state: MutableMapping[str, object],
    tab_titles: list[str],
    *,
    active_key: str = "active_tab",
    widget_key: str = "active_tab_widget",
) -> str:
    if not tab_titles:
        session_state.pop(active_key, None)
        session_state.pop(widget_key, None)
        return ""
    if session_state.get(active_key) not in tab_titles:
        session_state[active_key] = tab_titles[0]
    if session_state.get(widget_key) not in tab_titles:
        session_state.pop(widget_key, None)
        session_state[active_key] = tab_titles[0]
    return str(session_state[active_key])


def consume_tab_switch(
    session_state: MutableMapping[str, object],
    tab_titles: list[str],
    *,
    trigger_key: str = "trigger_tab_switch",
    active_key: str = "active_tab",
    widget_key: str = "active_tab_widget",
) -> str | None:
    tab_target = session_state.pop(trigger_key, None)
    if tab_target and tab_target in tab_titles:
        session_state[active_key] = tab_target
        session_state[widget_key] = tab_target
        return str(tab_target)
    return None


def render_tab_selector(st, tab_titles: list[str], *, caption: bool = True) -> str:
    session_state = st.session_state
    consume_tab_switch(session_state, tab_titles)
    active_tab = sync_active_tab_state(session_state, tab_titles)

    seg_kwargs = {}
    if "active_tab_widget" not in session_state:
        seg_kwargs["default"] = active_tab

    selected_tab = st.segmented_control(
        label="Menu điều hướng",
        options=tab_titles,
        selection_mode="single",
        key="active_tab_widget",
        label_visibility="collapsed",
        **seg_kwargs,
    )

    if selected_tab:
        session_state.active_tab = selected_tab
    else:
        selected_tab = session_state.active_tab

    if caption:
        st.caption(f"📍 Đang xem: **{selected_tab}**")
    return str(selected_tab)
