"""Admin-specific Streamlit UI panels."""

from __future__ import annotations

from collections.abc import Callable


def render_admin_sidebar(
    st,
    *,
    admin_name_html: str,
    lookup_user: Callable[[str], dict | None],
) -> None:
    st.markdown("### 👑 QUẢN TRỊ HỆ THỐNG")
    st.markdown(f"""
    <div style="background: rgba(255, 215, 0, 0.05); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 215, 0, 0.2); margin-bottom: 15px;">
        <p style="margin:0; font-weight:bold; color:#ffd700;">👤 {admin_name_html}</p>
        <p style="margin:0; font-size:0.8rem; color:#888;">Quyền hạn tối cao (Super User)</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    **Chức năng các Tab quản trị:**
    1. **🏠 TRANG CHỦ**: Dashboard thống kê, biểu đồ và chỉ số hiệu suất hệ thống.
    2. **🛠️ QUẢN TRỊ**: Cấp tài khoản mới, xóa người dùng và reset database.
    3. **📊 NHẬT KÝ**: Xem log hoạt động chi tiết của tất cả người dùng.
    4. **📖 HƯỚNG DẪN**: Quản lý tài liệu và video hướng dẫn sử dụng.
    5. **🏥 KIẾN THỨC**: Thư viện nội dung chuyên môn về PHCN vai.
    6. **🌐 CÔNG NGHỆ**: Thông số kỹ thuật về hạ tầng AI và Computer Vision.
    """)

    st.markdown("### 🔍 TRA CỨU NHANH")
    query = st.text_input("Tìm kiếm Username", placeholder="VD: patient01")
    if not query:
        return
    user = lookup_user(query)
    if user:
        st.success(f"Tìm thấy: {user.get('full_name')} ({user.get('role')})")
    else:
        st.error("Không tìm thấy người dùng.")


def render_admin_home(st, deps) -> None:
    deps.render_admin_home()


def render_admin_tab(selected_tab: str, deps) -> None:
    st = deps.st
    if selected_tab == "🏠 TRANG CHỦ":
        render_admin_home(st, deps)
    elif selected_tab == "🛠️ QUẢN TRỊ VIÊN":
        deps.render_admin_management()
    elif selected_tab == "📚 THÔNG TIN TỔNG HỢP":
        deps.render_general_info("Quản trị viên")
    elif selected_tab == "👥 HỒ SƠ ĐỀ TÀI & ĐỘI NGŨ CHUYÊN GIA":
        deps.render_research_profile_team()
    elif selected_tab == "💬 PHẢN HỒI":
        deps.render_feedback()
    elif selected_tab == "🔑 ĐỔI MẬT KHẨU":
        deps.render_change_password()
    elif selected_tab == "🌐 CÔNG NGHỆ":
        deps.render_technology()
