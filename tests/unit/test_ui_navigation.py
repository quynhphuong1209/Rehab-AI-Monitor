from ui.navigation import (
    consume_tab_switch,
    sync_active_tab_state,
    tab_titles_for_role,
)


def test_tab_titles_for_roles():
    assert tab_titles_for_role("Quản trị viên")[1] == "🛠️ QUẢN TRỊ VIÊN"
    assert "🎬 VIDEO & ẢNH" not in tab_titles_for_role("Bác sĩ / KTV PHCN")
    assert "🎬 VIDEO & ẢNH" in tab_titles_for_role("Bác sĩ / KTV PHCN", has_video_output=True)
    assert tab_titles_for_role("Bệnh nhân") == [
        "🏠 TRANG CHỦ",
        "📊 KẾT QUẢ ĐÁNH GIÁ",
        "⏰ LỊCH NHẮC NHỞ",
        "📚 THÔNG TIN TỔNG HỢP",
        "📞 THÔNG TIN LIÊN HỆ",
        "💬 PHẢN HỒI",
    ]
    assert "🔬 PHÂN TÍCH & TRÍCH XUẤT DỮ LIỆU" in tab_titles_for_role("Nghiên cứu viên")


def test_sync_active_tab_state_resets_invalid_values():
    state = {"active_tab": "missing", "active_tab_widget": "missing"}
    active = sync_active_tab_state(state, ["home", "settings"])

    assert active == "home"
    assert state["active_tab"] == "home"
    assert "active_tab_widget" not in state


def test_consume_tab_switch_updates_active_and_widget_state():
    state = {"trigger_tab_switch": "settings", "active_tab": "home"}

    selected = consume_tab_switch(state, ["home", "settings"])

    assert selected == "settings"
    assert state["active_tab"] == "settings"
    assert state["active_tab_widget"] == "settings"
    assert "trigger_tab_switch" not in state


def test_consume_tab_switch_ignores_invalid_target():
    state = {"trigger_tab_switch": "missing", "active_tab": "home"}

    selected = consume_tab_switch(state, ["home", "settings"])

    assert selected is None
    assert state["active_tab"] == "home"
    assert "active_tab_widget" not in state
    assert "trigger_tab_switch" not in state
