"""Static/common Streamlit pages extracted from app.py."""

from __future__ import annotations

import streamlit as st


def hien_thi_tab_huong_dan(role="Bệnh nhân"):
    """Hướng dẫn sử dụng hệ thống tùy biến theo vai trò"""
    # Xóa header thừa

    if role == "Bệnh nhân":
        steps = [
            ("1️⃣ Chuẩn bị không gian", "Đứng cách camera 2-3 mét, đảm bảo ánh sáng đủ tốt và thấy rõ toàn thân."),
            ("2️⃣ Chọn bài tập", "Tại TRANG CHỦ, chọn động tác cần tập (Vai, Khuỷu...) để hệ thống chuẩn bị bộ lọc tương ứng."),
            ("3️⃣ Upload Video", "Tải file video tập luyện lên. Hệ thống hỗ trợ MP4, MOV."),
            ("4️⃣ Xem kết quả", "Chờ Nghiên cứu viên phân tích và xem nhận xét chi tiết của Bác sĩ tại tab KẾT QUẢ ĐÁNH GIÁ."),
            ("5️⃣ Đặt lịch nhắc nhở", "Sử dụng tab LỊCH NHẮC NHỞ để không bỏ lỡ các buổi tập luyện tiếp theo.")
        ]
    elif role == "Bác sĩ / KTV PHCN":
        steps = [
            ("1️⃣ Tiếp nhận Video", "Xem danh sách video bệnh nhân gửi đến tại TRANG CHỦ."),
            ("2️⃣ Đánh giá lâm sàng", "Sử dụng tab QUẢN LÝ ĐÁNH GIÁ để điền kết quả dựa trên chuyên môn của bạn."),
            ("3️⃣ Tham khảo AI", "Xem sub-tab KẾT QUẢ TỪ NCV (nếu có) để có thêm dữ liệu khách quan về góc khớp."),
            ("4️⃣ Phản hồi cho BN", "Nhấn Gửi kết quả để bệnh nhân nhận được lời khuyên và phác đồ điều trị.")
        ]
    elif role == "Quản trị viên":
        steps = [
            ("1️⃣ Quản lý tài khoản", "Thêm mới hoặc khóa tài khoản của Bác sĩ, NCV và Bệnh nhân."),
            ("2️⃣ Giám sát hệ thống", "Theo dõi lưu lượng video và tính ổn định của server."),
            ("3️⃣ Cấu hình tham số", "Điều chỉnh các ngưỡng cảnh báo góc khớp chuẩn cho toàn hệ thống.")
        ]
    else: # Nghiên cứu viên (NCV)
        steps = [
            ("1️⃣ Trích xuất dữ liệu", "Sử dụng công cụ AI để trích xuất khung xương từ video của bệnh nhân."),
            ("2️⃣ Kiểm định Metrics", "Kiểm tra các chỉ số MAE, ICC, F1-Score để đảm bảo độ chính xác của mô hình."),
            ("3️⃣ Xuất báo cáo", "Tải xuống dữ liệu CSV hoặc ảnh biểu đồ cho mục đích viết bài báo khoa học."),
            ("4️⃣ Chuyển tiếp", "Gửi kết quả phân tích AI để Bác sĩ có cơ sở đưa ra đánh giá lâm sàng.")
        ]

    for title, desc in steps:
        with st.expander(title, expanded=True):
            st.write(desc)

    st.warning("⚠️ **Lưu ý:** Không nên mặc quần áo quá rộng hoặc quá tối màu để hệ thống nhận diện khớp chính xác nhất.")


def hien_thi_tab_kien_thuc_phcn():
    """Thiết kế Tab 8 về kiến thức y khoa Phục hồi chức năng"""
    # Cấu hình màu sắc theo Theme
    is_light = st.session_state.theme == 'light'
    bg_gradient = "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)" if is_light else "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"
    text_color = "#1a1a2e" if is_light else "#fff"
    sub_color = "#0072ff" if is_light else "#00CED1"
    border_color = "#0072ff" if is_light else "#00CED1"

    st.markdown(f"""
    <div style="background: {bg_gradient};
                padding: 2rem; border-radius: 20px; text-align: center;
                border: 1px solid {border_color}; box-shadow: 0 10px 30px rgba(0, 206, 209, 0.1);
                margin-bottom: 2rem;">
        <h1 style="color: {text_color}; margin: 0; font-size: 2rem;">🏥 KIẾN THỨC PHỤC HỒI CHỨC NĂNG</h1>
        <p style="color: {sub_color}; font-weight: bold; margin-top: 0.5rem;">
            Nền tảng y khoa cho sự phục hồi toàn diện
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. 4 TRỤ CỘT Y TẾ
    st.markdown("### 🏛️ 4 TRỤ CỘT CỦA Y TẾ HIỆN ĐẠI")
    cols = st.columns(4)
    pillar_data = [
        ("🛡️", "Phòng bệnh", "Ngăn ngừa nguy cơ"),
        ("💊", "Điều trị", "Xử lý cấp tính"),
        ("🦾", "PHCN", "Khôi phục chức năng"),
        ("🌟", "Nâng cao sức khỏe", "Tối ưu thể chất")
    ]
    for i, (icon, title, desc) in enumerate(pillar_data):
        with cols[i]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 15px; text-align: center; border: 1px solid #333;">
                <div style="font-size: 2.5rem;">{icon}</div>
                <h5 style="color: #00CED1; margin: 10px 0;">{title}</h5>
                <p style="color: #888; font-size: 0.8rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. LỢI ÍCH VÀ QUY TRÌNH
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 💎 LỢI ÍCH CỦA PHCN")
        st.markdown("""
        *   **Hồi phục tối đa:** Khôi phục những chức năng bị suy yếu do chấn thương, đột quỵ.
        *   **Ngăn ngừa biến chứng:** Tránh teo cơ, cứng khớp và loét tì đè.
        *   **Hòa nhập xã hội:** Giúp người bệnh tự lập trong ăn uống, vệ sinh và đi lại (ADL).
        *   **Giảm gánh nặng y tế:** Rút ngắn thời gian nằm viện và giảm chi phí chăm sóc dài hạn.
        """)
        st.success("💡 PHCN giúp người bệnh chuyển từ trạng thái 'Được chăm sóc' sang 'Tự lực'.")

    with col_right:
        st.markdown("### 📑 QUY TRÌNH PHCN CHUẨN")
        with st.expander("Bước 1: Lượng giá chức năng", expanded=True):
            st.write("Bác sĩ khám và đánh giá mức độ tổn thương, tầm vận động khớp.")
        with st.expander("Bước 2: Lập kế hoạch điều trị"):
            st.write("Thiết lập bài tập chuyên biệt (Vật lý trị liệu, Vận động trị liệu).")
        with st.expander("Bước 3: Thực hiện & Theo dõi"):
            st.write("Kỹ thuật viên hướng dẫn tập luyện và điều chỉnh theo tiến độ.")
        with st.expander("Bước 4: Đánh giá & Duy trì"):
            st.write("Kiểm tra kết quả và hướng dẫn bệnh nhân tự tập luyện tại nhà.")

    # 3. KÊNH THÔNG TIN THAM KHẢO
    st.markdown("---")
    st.info("""
    📚 **Nguồn tham khảo chính thống:**
    *   [Cục Quản lý Khám, chữa bệnh - Bộ Y tế Việt Nam (KCB.VN)](https://kcb.vn/van-ban/huong-dan-quy-trinh-ky-thuat-chuyen-nganh-phuc-hoi-chuc-nang)
    *   [Tổ chức Y tế Thế giới (WHO) - Rehabilitation Topics](https://www.who.int/news-room/fact-sheets/detail/rehabilitation)
    *   [Chiến lược Phục hồi chức năng 2030 (WHO)](https://www.who.int/initiatives/rehabilitation-2030)
    """)


def hien_thi_tab_cong_nghe():
    """Thiết kế Tab 7 với phong cách công nghệ cao cấp"""

    # Cấu hình màu sắc theo Theme
    is_light = st.session_state.theme == 'light'
    bg_gradient = "linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%)" if is_light else "linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)"
    text_color = "#000000" if is_light else "#ffd700"
    sub_color = "#0072ff" if is_light else "#00CED1"
    border_color = "#0072ff" if is_light else "#ffd700"
    shadow = "rgba(0, 114, 255, 0.1)" if is_light else "rgba(255, 215, 0, 0.1)"

    # 1. HEADER CHƯƠNG TRÌNH
    st.markdown(f"""
    <div style="background: {bg_gradient};
                padding: 2.5rem; border-radius: 25px; text-align: center;
                border: 1px solid {border_color}; box-shadow: 0 15px 35px {shadow};
                margin-bottom: 2rem;">
        <h1 style="color: {text_color}; margin: 0; font-size: 2.2rem; letter-spacing: 2px;">🌐 HỆ SINH THÁI CÔNG NGHỆ Y TẾ</h1>
        <p style="color: {sub_color}; font-weight: bold; margin-top: 0.5rem; font-size: 1.1rem;">
            Sự kết hợp hoàn hảo giữa Phục hồi chức năng và Trí tuệ nhân tạo (AI)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. PHẦN 1: PHỤC HỒI CHỨC NĂNG 4.0
    st.markdown("### 🏥 PHỤC HỒI CHỨC NĂNG TỪ XA (TELEREHABILITATION)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card" style="height: 250px; border-top: 4px solid #00CED1;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🌍</div>
            <h4 style="color: #fff;">Tiếp cận toàn cầu</h4>
            <p style="color: #aaa; font-size: 0.9rem;">
                Theo tiêu chuẩn WHO 2022, Telerehab giúp bệnh nhân ở vùng sâu tiếp cận y tế chất lượng cao mà không cần di chuyển.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="height: 250px; border-top: 4px solid #ffd700;">
            <div style="font-size: 3rem; margin-bottom: 10px;">📉</div>
            <h4 style="color: #fff;">Tối ưu chi phí</h4>
            <p style="color: #aaa; font-size: 0.9rem;">
                Giảm 40% chi phí điều trị nội trú nhờ duy trì chương trình tập luyện tại nhà được giám sát tự động qua AI.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="height: 250px; border-top: 4px solid #FF6B6B;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
            <h4 style="color: #fff;">Cá nhân hóa</h4>
            <p style="color: #aaa; font-size: 0.9rem;">
                Dữ liệu từ cảm biến AI giúp bác sĩ điều chỉnh phác đồ theo từng milimet biên độ vận động của bệnh nhân.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. PHẦN 2: CÔNG NGHỆ MEDIAPIPE
    col_text, col_img = st.columns([1.2, 1])

    with col_text:
        st.markdown("### 🤖 CỐT LÕI AI: GOOGLE MEDIAPIPE")
        st.markdown("""
        Hệ thống sử dụng kiến trúc **BlazePose** mạnh mẽ nhất từ Google Research, mang lại khả năng theo dõi cơ thể người với độ chính xác cấp độ nghiên cứu.

        #### ✨ Các tính năng ưu việt:
        *   **33 Body Landmarks:** Theo dõi toàn diện từ khuôn mặt, tay chân đến tư thế cột sống.
        *   **Real-time Inference:** Xử lý hơn 30 khung hình/giây ngay trên trình duyệt, không cần máy chủ mạnh.
        *   **BlazePose Topology:** Khả năng nhận diện hướng của khớp vai và khuỷu tay trong không gian 3D, vượt xa các thuật toán Pose truyền thống.
        *   **Robustness:** Hoạt động ổn định trong nhiều điều kiện ánh sáng và trang phục khác nhau.
        """)

        st.info("💡 **Bạn có biết?** MediaPipe Pose được sử dụng trong các ứng dụng Fitness hàng đầu thế giới để chấm điểm động tác Yoga và Gym tự động.")

    with col_img:
        st.markdown("""
        <div style="background: rgba(0,206,209,0.05); padding: 20px; border-radius: 20px; border: 1px dashed #00CED1; text-align: center;">
            <h4 style="color: #00CED1;">BLAZEPOSE LANDMARKS MAP</h4>
            <img src="https://mediapipe.dev/images/mobile/pose_tracking_full_body_landmarks.png" style="width: 100%; border-radius: 10px; margin-top: 10px;">
            <p style="color: #888; font-size: 0.8rem; margin-top: 10px;">Sơ đồ 33 điểm mốc được AI trích xuất thời gian thực</p>
        </div>
        """, unsafe_allow_html=True)

    # 4. FOOTER THÔNG TIN
    st.markdown("""
    <div style="margin-top: 3rem; padding: 1.5rem; background: rgba(255,215,0,0.05); border-radius: 15px; text-align: center;">
        <p style="color: #aaa; font-style: italic;">
            "Công nghệ không thay thế bác sĩ, nhưng bác sĩ sử dụng công nghệ sẽ thay thế những bác sĩ không sử dụng."
        </p>
        <p style="color: #ffd700; font-weight: bold; margin-top: 0.5rem;">— Rehab AI Monitor Team —</p>
    </div>
    """, unsafe_allow_html=True)


def hien_thi_tab_thong_tin_tong_hop(role):
    """Gộp các tab Hướng dẫn, Kiến thức và Công nghệ cho Nghiên cứu viên/Bác sĩ"""
    st.markdown("## 📚 TỔNG HỢP THÔNG TIN & HƯỚNG DẪN")
    st.info("💡 Đây là khu vực tổng hợp các tài liệu hướng dẫn, kiến thức chuyên môn và công nghệ cốt lõi của hệ thống Rehab-AI-Monitor.")

    # Sử dụng sub-tabs để gộp 3 nội dung
    sub_tab_titles = ["📖 HƯỚNG DẪN SỬ DỤNG", "🏥 KIẾN THỨC PHCN", "🌐 CÔNG NGHỆ AI"]
    st_sub_tabs = st.tabs(sub_tab_titles)

    with st_sub_tabs[0]:
        hien_thi_tab_huong_dan(role=role)
    with st_sub_tabs[1]:
        hien_thi_tab_kien_thuc_phcn()
    with st_sub_tabs[2]:
        hien_thi_tab_cong_nghe()


def hien_thi_tab_nckh_va_thanh_vien_ncv():
    """Gộp tab Đề tài NCKH và Thành viên cho Nghiên cứu viên"""
    st.markdown("## 👥 THÔNG TIN ĐỀ TÀI & ĐỘI NGŨ NGHIÊN CỨU")

    sub_tabs = st.tabs(["📚 NỘI DUNG ĐỀ TÀI", "👥 THÀNH VIÊN DỰ ÁN"])

    with sub_tabs[0]:
        hien_thi_tab_nckh()
    with sub_tabs[1]:
        hien_thi_tab_thanh_vien()


def hien_thi_tab_thong_tin_tong_hop_benh_nhan():
    """Tab Thông tin tổng hợp cho Bệnh nhân"""
    t1, t2 = st.tabs(["📄 THÔNG TIN NGHIÊN CỨU", "📖 HƯỚNG DẪN SỬ DỤNG"])
    with t1:
        hien_thi_tab_thong_tin_nghien_cuu()
    with t2:
        hien_thi_tab_huong_dan(role="Bệnh nhân")


def hien_thi_tab_lien_he():
    """Giao diện liên hệ xịn xò (Premium Design)"""

    # Header xịn
    st.markdown("""
        <div style="text-align: center; padding: 10px 20px 30px 20px; margin-bottom: 10px;">
            <h1 style="color: #00c6ff; font-family: 'Outfit', sans-serif; text-shadow: 2px 2px 10px rgba(0,198,255,0.3);">📞 THÔNG TIN LIÊN HỆ KHẨN CẤP</h1>
            <p style="color: #aaa; font-style: italic; font-size: 1.1rem;">Hệ thống luôn sẵn sàng hỗ trợ bạn trong quá trình nghiên cứu và tập luyện.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(0, 198, 255, 0.4); border-radius: 20px; padding: 30px; min-height: 480px; position: relative; overflow: hidden; box-shadow: 0 15px 35px rgba(0,0,0,0.4); backdrop-filter: blur(10px);">
            <div style="position: absolute; top: -50px; right: -50px; width: 150px; height: 150px; background: rgba(0, 198, 255, 0.1); border-radius: 50%;"></div>
            <h2 style="color: #00c6ff; margin-bottom: 30px; border-bottom: 3px solid #00c6ff; padding-bottom: 15px; display: flex; align-items: center;">
                <span style="margin-right: 15px; font-size: 2rem;">👩‍🔬</span> Nghiên cứu viên chính
            </h2>
            <div style="margin-bottom: 20px;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Họ và tên</p>
                <p style="font-size: 1.4rem; font-weight: bold; color: white; font-family: 'Outfit', sans-serif;">Đinh Lê Quỳnh Phương</p>
            </div>
            <div style="margin-bottom: 20px;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Đơn vị công tác</p>
                <p style="font-size: 1.1rem; color: #ccc;">Trường Đại học Y tế Công cộng - Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
            </div>
            <div style="margin-bottom: 20px;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Email liên hệ</p>
                <p style="font-size: 1.2rem;"><a href="mailto:2211090031@studenthuph.edu.vn" style="color: #00c6ff; text-decoration: none; border-bottom: 1px dashed #00c6ff;">2211090031@studenthuph.edu.vn</a></p>
            </div>
            <div style="margin-top: 30px; padding: 15px; background: rgba(0, 198, 255, 0.1); border-radius: 12px; border-left: 5px solid #00c6ff;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px;">Số điện thoại khẩn cấp</p>
                <p style="font-size: 1.6rem; font-weight: bold; color: #00c6ff; margin: 0;">0382665916</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 215, 0, 0.4); border-radius: 20px; padding: 30px; min-height: 480px; position: relative; overflow: hidden; box-shadow: 0 15px 35px rgba(0,0,0,0.4); backdrop-filter: blur(10px);">
            <div style="position: absolute; top: -50px; right: -50px; width: 150px; height: 150px; background: rgba(255, 215, 0, 0.1); border-radius: 50%;"></div>
            <h2 style="color: #ffd700; margin-bottom: 30px; border-bottom: 3px solid #ffd700; padding-bottom: 15px; display: flex; align-items: center;">
                <span style="margin-right: 15px; font-size: 2rem;">⚖️</span> Hội đồng đạo đức
            </h2>
            <div style="margin-bottom: 20px;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Tên cơ quan</p>
                <p style="font-size: 1.4rem; font-weight: bold; color: white; font-family: 'Outfit', sans-serif;">HĐĐĐ Trường ĐH Y tế Công cộng</p>
            </div>
            <div style="margin-bottom: 20px;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Địa chỉ trụ sở</p>
                <p style="font-size: 1.1rem; color: #ccc;">Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
            </div>
            <div style="margin-bottom: 20px;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Email hỗ trợ</p>
                <p style="font-size: 1.2rem;"><a href="mailto:irb@huph.edu.vn" style="color: #ffd700; text-decoration: none; border-bottom: 1px dashed #ffd700;">irb@huph.edu.vn</a></p>
            </div>
            <div style="margin-top: 30px; padding: 15px; background: rgba(255, 215, 0, 0.1); border-radius: 12px; border-left: 5px solid #ffd700;">
                <p style="color: #888; font-size: 1rem; margin-bottom: 5px;">Đường dây nóng</p>
                <p style="font-size: 1.6rem; font-weight: bold; color: #ffd700; margin: 0;">024 62663024</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Thêm mục Bản đồ & Địa chỉ Bệnh viện
    st.markdown("""<style>
.map-container {
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
}
.map-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0, 198, 255, 0.15) !important;
    border-color: rgba(0, 198, 255, 0.6) !important;
}
.map-btn {
    display: inline-block;
    background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
    color: white !important;
    padding: 12px 24px;
    border-radius: 12px;
    text-decoration: none !important;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3);
}
.map-btn:hover {
    background: linear-gradient(135deg, #00d2ff 0%, #0080ff 100%);
    box-shadow: 0 6px 20px rgba(0, 198, 255, 0.5);
    transform: scale(1.02);
}
</style>
<div class="map-container" style="margin-top: 35px; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(0, 198, 255, 0.3); border-radius: 20px; padding: 30px; box-shadow: 0 15px 35px rgba(0,0,0,0.4); backdrop-filter: blur(10px);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 25px; border-bottom: 3px solid #00c6ff; padding-bottom: 15px;">
<h2 style="color: #00c6ff; margin: 0; display: flex; align-items: center; font-family: 'Outfit', sans-serif;">
<span style="margin-right: 15px; font-size: 2rem;">📍</span> VỊ TRÍ BỆNH VIỆN ĐA KHOA PHẠM NGỌC THẠCH
</h2>
<a class="map-btn" href="https://www.google.com/maps/place/B%E1%BB%87nh+vi%E1%BB%87n+%C4%91a+khoa+Ph%E1%BA%A1m+Ng%E1%BB%8Dc+Th%E1%BA%A1ch/@21.0821035,105.7766556,17z/data=!3m1!4b1!4m6!3m5!1s0x313455002cadccfd:0xf42e13275632d6dc!8m2!3d21.0820985!4d105.7792305!16s%2Fg%2F11wbfdswkr?entry=ttu" target="_blank">
🗺️ Xem trên Google Maps
</a>
</div>
<div style="margin-bottom: 25px;">
<p style="color: #888; font-size: 0.95rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Địa chỉ bệnh viện</p>
<p style="font-size: 1.25rem; color: #fff; font-weight: 500; font-family: 'Outfit', sans-serif; margin: 0;">
Số 1A, Đường Đức Thắng, Phường Đông Ngạc, Quận Bắc Từ Liêm, Hà Nội
</p>
</div>
<div style="width: 100%; border-radius: 15px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
<iframe src="https://maps.google.com/maps?q=21.0820985,105.7792305&z=16&output=embed" width="100%" height="400" style="border:0; display: block;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
</div>
</div>""", unsafe_allow_html=True)


def hien_thi_tab_nckh():
    is_light = st.session_state.theme == 'light'
    bg_gradient = "linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%)" if is_light else "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
    text_color = "#000" if is_light else "white"
    sub_color = "#0072ff" if is_light else "#ffd700"
    border_color = "#0072ff" if is_light else "#2a5298"

    st.markdown(f"""
    <div style="background: {bg_gradient}; padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center; border: 1px solid {border_color};">
        <h2 style="color: {text_color}; margin: 0;">📚 ĐỀ TÀI NGHIÊN CỨU KHOA HỌC</h2>
        <p style="color: {sub_color}; font-size: 1.1rem; margin-top: 0.5rem;">Phát triển Mô hình thử nghiệm giám sát tập luyện Phục hồi chức năng từ xa</p>
        <p style="color: {"#333" if is_light else "#ccc"};">Dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision)</p>
        <p style="color: {"#666" if is_light else "#aaa"}; font-size: 0.9rem;">Bệnh viện Đa khoa Phạm Ngọc Thạch - Trường Đại học Y tế Công cộng (2025-2026)</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 ĐẶT VẤN ĐỀ", expanded=True):
        st.markdown("""
        Trong những năm gần đây, cùng với sự gia tăng của các bệnh lý cơ xương khớp, chấn thương thể thao và đột quỵ, nhu cầu phục hồi chức năng (PHCN) trên toàn thế giới ngày càng tăng cao.

        Theo Tổ chức Y tế Thế giới (WHO), hiện có khoảng 2,4 tỷ người cần ít nhất một hình thức phục hồi chức năng, chiếm gần một phần ba dân số toàn cầu. Tại Việt Nam, theo Hội Phục hồi chức năng Việt Nam (2023), có khoảng 7,06% dân số từ 2 tuổi trở lên là người khuyết tật, trong đó phần lớn cần được can thiệp PHCN.

        Mặc dù nhu cầu PHCN lớn, song năng lực cung cấp dịch vụ này tại Việt Nam vẫn còn hạn chế. Trung bình 10.000 người dân chỉ có 0,25 nhân viên phục hồi chức năng, thấp hơn đáng kể so với khuyến nghị của WHO là 0,5-1 người/10.000 dân. Thực tế này khiến nhiều bệnh nhân phải tự tập luyện tại nhà sau khi xuất viện mà thiếu sự giám sát chuyên môn.

        Xuất phát từ thực tiễn trên, nhóm nghiên cứu quyết định thực hiện đề tài: **"Phát triển Mô hình thử nghiệm giám sát tập luyện Phục hồi chức năng từ xa dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision)"**.
        """)

    with st.expander("🎯 MỤC TIÊU NGHIÊN CỨU", expanded=True):
        st.markdown("""
        **Mục tiêu 1:** Xây dựng mô hình nhận diện và đánh giá 3 bài tập phục hồi chức năng cho bệnh nhân viêm quanh khớp vai, bao gồm:
        - Bài tập con lắc Codman
        - Bài tập với gậy
        - Bài tập với dây kháng lực

        **Mục tiêu 2:** So sánh độ chính xác của mô hình với đánh giá thủ công trên một tập dữ liệu nhỏ.
        """)

    with st.expander("🔬 ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU", expanded=True):
        st.markdown("""
        **Đối tượng nghiên cứu:** 05 bệnh nhân viêm quanh khớp vai + nhóm chuyên gia PHCN tại Khoa Phục hồi chức năng, Bệnh viện Đa khoa Phạm Ngọc Thạch.

        **Thiết kế nghiên cứu:** Nghiên cứu định lượng, phát triển mô hình học máy.

        **Công nghệ sử dụng:**
        - MediaPipe Pose Estimation cho ước lượng tư thế
        - Python và các thư viện xử lý ảnh (OpenCV, NumPy, Pandas)
        - Streamlit cho giao diện người dùng
        - Plotly cho trực quan hóa dữ liệu

        **Cỡ mẫu dự kiến:** 500-1000 chuỗi chuyển động.
        """)

    with st.expander("📊 KẾT QUẢ DỰ KIẾN", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Độ chính xác (Accuracy)", "≥ 90%")
            st.metric("F1-Score", "≥ 0.85")
        with col2:
            st.metric("Sai số MAE", "< 5°")
            st.metric("Hệ số ICC", "≥ 0.75")
        with col3:
            st.metric("Precision", "≥ 0.85")
            st.metric("Recall", "≥ 0.85")

    with st.expander("🎁 ĐÓNG GÓP CỦA ĐỀ TÀI", expanded=True):
        st.markdown("""
        - **Về khoa học và đào tạo:** Xây dựng mô hình nhận diện động tác PHCN, tạo bộ dữ liệu chuẩn hóa, là tài liệu thực hành cho sinh viên ngành Khoa học dữ liệu y sinh.
        - **Về phát triển kinh tế:** Giảm chi phí đi lại, giảm tải cho nhân viên y tế, tối ưu nguồn lực bệnh viện.
        - **Về xã hội:** Tăng khả năng tiếp cận dịch vụ PHCN, thúc đẩy chuyển đổi số y tế, xây dựng hệ thống chăm sóc sức khỏe thông minh.
        """)

    with st.expander("📚 TÀI LIỆU THAM KHẢO", expanded=False):
        st.markdown("""
        1. WHO. Rehabilitation 2030: A call for action.
        2. Cieza A, et al. Global estimates of the need for rehabilitation. Lancet. 2021.
        3. Lugaresi C, et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv. 2019.
        4. Cao Z, et al. OpenPose: Realtime Multi-Person 2D Pose Estimation. arXiv. 2019.
        5. Hellstén T, et al. Reliability and validity of computer vision-based markerless human pose estimation. Healthc Technol Lett. 2025.
        6. Ino T, et al. Validity and Reliability of OpenPose-Based Motion Analysis. J Sports Sci Med. 2024.
        7. Aguilar-Ortega R, et al. UCO Physical Rehabilitation: New Dataset and Study. Sensors. 2023.
        8. Nguyễn Thị Ngọc Lan, et al. Thực trạng nhu cầu phục hồi chức năng tại Việt Nam. Tạp chí Y học Việt Nam. 2024.
        """)


def hien_thi_tab_thong_tin_nghien_cuu():
    is_light = st.session_state.theme == 'light'
    card_bg = "#f8f9fa" if is_light else "rgba(255, 255, 255, 0.05)"
    text_color = "#333" if is_light else "#ccc"
    accent_color = "#00c6ff"

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: {accent_color}; text-transform: uppercase; margin-bottom: 0.5rem;">Trang thông tin nghiên cứu</h2>
        <h4 style="color: {text_color}; line-height: 1.4;">PHÁT TRIỂN MÔ HÌNH THỬ NGHIỆM GIÁM SÁT TẬP LUYỆN PHỤC HỒI CHỨC NĂNG TỪ XA DỰA TRÊN TRÍ TUỆ NHÂN TẠO VÀ THỊ GIÁC MÁY TÍNH TẠI BỆNH VIỆN ĐA KHOA PHẠM NGỌC THẠCH - TRƯỜNG ĐẠI HỌC Y TẾ CÔNG CỘNG (2025–2026)</h4>
        <p style="color: {accent_color}; font-weight: bold; font-size: 1.1rem; margin-top: 1rem;">🎯 Dành cho đối tượng: Người bệnh viêm quanh khớp vai điều trị tại Khoa Phục hồi chức năng</p>
    </div>
    """, unsafe_allow_html=True)

    # Sử dụng các expander để trình bày nội dung chuyên nghiệp
    with st.expander("1. GIỚI THIỆU VỀ NGHIÊN CỨU", expanded=True):
        st.markdown(f"""
        <div style="padding: 10px; border-left: 4px solid {accent_color};">
            <p>Nghiên cứu này nhằm thử nghiệm một hệ thống giúp theo dõi việc tập luyện phục hồi chức năng khớp vai bằng camera và máy tính. Hệ thống sẽ giúp ghi nhận và phân tích các động tác tập luyện của người bệnh.</p>
            <p>Đối tượng tham gia là người bệnh được chẩn đoán viêm quanh khớp vai đang điều trị tại Khoa Phục hồi chức năng – Bệnh viện Đa khoa Phạm Ngọc Thạch. Mục tiêu của nghiên cứu là đánh giá xem hệ thống có thể nhận biết và đánh giá đúng các động tác tập luyện hay không, từ đó hướng tới việc hỗ trợ theo dõi tập luyện từ xa trong tương lai.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("2. QUY TRÌNH NGHIÊN CỨU", expanded=False):
        st.markdown(f"""
        <p>Nghiên cứu được thực hiện từ tháng 12 năm 2025 đến tháng 7 năm 2026 tại Khoa Phục hồi chức năng – Bệnh viện Đa khoa Phạm Ngọc Thạch. Người tham gia là người bệnh viêm quanh khớp vai đang được chỉ định tập phục hồi chức năng. Dự kiến có khoảng 05 người bệnh tham gia.</p>
        <p>Người tham gia cần có khả năng thực hiện các bài tập theo hướng dẫn và đồng ý tham gia nghiên cứu. Những trường hợp không đủ điều kiện sức khỏe hoặc không thể phối hợp trong quá trình thực hiện sẽ không được tham gia.</p>
        <p>Trong quá trình tham gia, người bệnh sẽ thực hiện các bài tập phục hồi chức năng khớp vai theo hướng dẫn của nhân viên y tế, bao gồm bài tập con lắc, bài tập với gậy và bài tập với dây kháng lực. Quá trình tập luyện sẽ được ghi hình bằng thiết bị điện tử. Thông tin thu thập bao gồm video ghi lại quá trình tập luyện và một số thông tin cơ bản liên quan đến việc thực hiện động tác. Các video này sẽ được sử dụng để đánh giá mức độ chính xác của động tác và so sánh với nhận định của nhân viên y tế. Kết quả đánh giá sẽ được thông báo lại cho người bệnh để biết và điều chỉnh cách tập nếu cần.</p>
        <p>Mỗi lần tham gia kéo dài khoảng 5–10 phút và không làm ảnh hưởng đến thời gian điều trị thông thường của người bệnh.</p>
        """, unsafe_allow_html=True)

    with st.expander("3. NGUY CƠ CÓ THỂ XẢY RA", expanded=False):
        st.warning("⚠️ Người bệnh có thể cảm thấy mệt, đau cơ nhẹ hoặc căng cơ khi thực hiện các bài tập. Việc ghi hình có thể khiến một số người cảm thấy không thoải mái.")
        st.info("💡 Để giảm thiểu các nguy cơ này, người bệnh luôn có nhân viên y tế theo dõi. Dữ liệu (video) sẽ được mã hóa và bảo mật tuyệt đối.")

    with st.expander("4. QUYỀN LỢI CỦA NGƯỜI THAM GIA", expanded=False):
        st.success("✅ Người tham gia không phải trả bất kỳ chi phí nào. Được nhân viên y tế hướng dẫn và theo dõi tập luyện để đảm bảo an toàn và đúng kỹ thuật.")

    with st.expander("5. BẢO MẬT VÀ LƯU TRỮ THÔNG TIN", expanded=False):
        st.markdown("""
        Toàn bộ thông tin và dữ liệu thu thập được bảo mật theo quy định. Dữ liệu được mã hóa và lưu trữ trong hệ thống an toàn; chỉ các thành viên được phân công mới có quyền truy cập. Thông tin cá nhân sẽ không được tiết lộ khi công bố kết quả.
        """)

    with st.expander("6. TÍNH CHẤT TÌNH NGUYỆN", expanded=False):
        st.markdown("""
        Việc tham gia hoàn toàn tự nguyện. Người bệnh có quyền từ chối hoặc rút khỏi nghiên cứu bất cứ lúc nào mà không cần nêu lý do. Quyết định này không ảnh hưởng đến việc điều trị tại bệnh viện.
        """)

    with st.expander("7. HÌNH THỨC CÔNG BỐ THÔNG TIN", expanded=False):
        st.markdown("""
        Kết quả có thể được sử dụng cho mục đích học tập, báo cáo khoa học hoặc hội thảo. Mọi thông tin cá nhân đều được bảo mật tuyệt đối.
        """)

    # Thông tin liên hệ dạng thẻ
    st.markdown("### 📞 THÔNG TIN LIÊN HỆ")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"""
        <div class="custom-card" style="background: {card_bg}; padding: 15px; border-radius: 12px; border: 1px solid {accent_color}; border-top: 5px solid {accent_color};">
            <h4 style="margin-top:0; color:{accent_color};">Nghiên cứu viên chính</h4>
            <p style="margin:5px 0;"><b>Họ tên:</b> Đinh Lê Quỳnh Phương</p>
            <p style="margin:5px 0;"><b>Địa chỉ:</b> Trường Đại học Y tế Công cộng - Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
            <p style="margin:5px 0;"><b>Email:</b> 2211090031@studenthuph.edu.vn</p>
            <p style="margin:5px 0;"><b>SĐT:</b> 0382665916</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown(f"""
        <div class="custom-card" style="background: {card_bg}; padding: 15px; border-radius: 12px; border: 1px solid #ff4b4b; border-top: 5px solid #ff4b4b;">
            <h4 style="margin-top:0; color:#ff4b4b;">Hội đồng đạo đức</h4>
            <p style="margin:5px 0;"><b>Tên:</b> HĐĐĐ Trường ĐH Y tế Công cộng</p>
            <p style="margin:5px 0;"><b>Địa chỉ:</b> Trường Đại học Y tế Công cộng - Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
            <p style="margin:5px 0;"><b>Email:</b> irb@huph.edu.vn</p>
            <p style="margin:5px 0;"><b>SĐT:</b> 024 62663024</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎁 ĐÓNG GÓP CỦA ĐỀ TÀI", expanded=True):
        st.markdown("""
        **- Về khoa học và đào tạo:** Xây dựng mô hình nhận diện động tác PHCN, tạo bộ dữ liệu chuẩn hóa, là tài liệu thực hành cho sinh viên ngành Khoa học dữ liệu y sinh.

        **- Về phát triển kinh tế:** Giảm chi phí đi lại, giảm tải cho nhân viên y tế, tối ưu nguồn lực bệnh viện.

        **- Về xã hội:** Tăng khả năng tiếp cận dịch vụ PHCN, thúc đẩy chuyển đổi số y tế, xây dựng hệ thống chăm sóc sức khỏe thông minh.
        """)

    with st.expander("📚 TÀI LIỆU THAM KHẢO", expanded=False):
        st.markdown("""
        1. WHO. Rehabilitation 2030: A call for action.
        2. Cieza A, et al. Global estimates of the need for rehabilitation. Lancet. 2021.
        3. Lugaresi C, et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv. 2019.
        4. Cao Z, et al. OpenPose: Realtime Multi-Person 2D Pose Estimation. arXiv. 2019.
        5. Hellstén T, et al. Reliability and validity of computer vision-based markerless human pose estimation. Healthc Technol Lett. 2025.
        6. Ino T, et al. Validity and Reliability of OpenPose-Based Motion Analysis. J Sports Sci Med. 2024.
        7. Aguilar-Ortega R, et al. UCO Physical Rehabilitation: New Dataset and Study. Sensors. 2023.
        8. Nguyễn Thị Ngọc Lan, et al. Thực trạng nhu cầu phục hồi chức năng tại Việt Nam. Tạp chí Y học Việt Nam. 2024.
        """)


def hien_thi_tab_thanh_vien():
    st.markdown("### 👨‍🏫 GIẢNG VIÊN HƯỚNG DẪN")
    gv_col1, gv_col2 = st.columns(2)
    with gv_col1:
        st.markdown("""
        <div class="lecturer-card">
            <div class="lecturer-name">TS. Trần Hồng Việt 🎓</div>
            <p style="color: #ccc; margin-top: 0.5rem;">Giảng viên hướng dẫn Khoa học Dữ Liệu</p>
            <p style="color: #aaa; font-size: 0.9rem;">Trường Đại học Y tế Công cộng</p>
            <a href="mailto:thviet79@gmail.com" style="text-decoration:none; color:#00CED1; font-size:0.9rem;">📧 thviet79@gmail.com</a>
        </div>
        """, unsafe_allow_html=True)
    with gv_col2:
        st.markdown("""
        <div class="lecturer-card" style="border-color: #00CED1;">
            <div class="lecturer-name" style="color: #00CED1;">Cô Nguyễn Thị Thùy Chi 🎓</div>
            <p style="color: #ccc; margin-top: 0.5rem;">Giảng viên hướng dẫn Lâm Sàng</p>
            <p style="color: #aaa; font-size: 0.9rem;">Trường Đại học Y tế Công cộng</p>
            <a href="mailto:chi.ntt@huph.edu.vn" style="text-decoration:none; color:#00CED1; font-size:0.9rem;">📧 chi.ntt@huph.edu.vn</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 👩‍⚕️ CHỦ NHIỆM ĐỀ TÀI")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="member-card" style="border-color: #ffd700; border: 2px solid #ffd700;">
            <div class="member-name">Đinh Lê Quỳnh Phương 🛡️</div>
            <div class="member-role">⭐ Chủ nhiệm đề tài ⭐</div>
            <div class="member-id">MSSV: 2211090031</div>
            <a href="mailto:2211090031@studenthuph.edu.vn" style="text-decoration:none; color:#0072ff; font-size:0.85rem;">📧 2211090031@studenthuph.edu.vn</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👥 THÀNH VIÊN NGHIÊN CỨU")
    thanh_vien = [
        ("Kim Mạnh Hưng 🛡️", "Thành viên", "CNCQ KHDL1-1A", "2211090016", "2211090016@studenthuph.edu.vn"),
        ("Nguyễn Hải An 🛡️", "Thành viên", "CNCQ KHDL1-1A", "2211090001", "2211090001@studenthuph.edu.vn"),
        ("Phan Vân Anh 🛡️", "Thành viên", "CNCQ KHDL1-1A", "2211090004", "2211090004@studenthuph.edu.vn"),
        ("Nguyễn Thị Thanh Nga 🛡️", "Thành viên", "CNCQ KHDL1-1A", "2211090027", "2211090027@studenthuph.edu.vn"),
        ("Nguyễn Thị Thơm 🛡️", "Thành viên nghiên cứu", "CNCQ KTPHCN3-1A", "2216030122", "2216030122@studenthuph.edu.vn"),
        ("Nguyễn Thị Thu Hương 🛡️", "Thành viên nghiên cứu", "CNCQ YTCC22-1A", "2317010071", "2317010071@studenthuph.edu.vn"),
    ]

    # Hiển thị grid 3 cột cho 6 thành viên
    for i in range(0, len(thanh_vien), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(thanh_vien):
                ten, vai_tro, lop, mssv, email = thanh_vien[i+j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="member-card">
                        <div class="member-name">{ten}</div>
                        <div class="member-role">{vai_tro}</div>
                        <div class="member-class">{lop}</div>
                        <div class="member-id">MSSV: {mssv}</div>
                        <a href="mailto:{email}" style="text-decoration:none; color:#00CED1; font-size:0.8rem;">📧 {email}</a>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏥 ĐƠN VỊ PHỐI HỢP")
    is_light = st.session_state.theme == 'light'
    partner_bg = "#ffffff" if is_light else "rgba(26,26,46,0.8)"
    partner_text = "#333" if is_light else "#ccc"
    partner_title = "#0072ff" if is_light else "#ffd700"

    st.markdown(f"""
    <div style="background: {partner_bg}; border-radius: 16px; padding: 1.5rem; text-align: center; border: 1px solid #2a5298;">
        <p style="color: {partner_title}; font-weight: bold;">Bệnh viện Đa khoa Phạm Ngọc Thạch</p>
        <p style="color: {partner_text};">Khoa Phục hồi chức năng</p>
        <p style="color: {"#666" if is_light else "#aaa"}; font-size: 0.9rem;">Địa chỉ: 1A Đ. Đức Thắng, Đông Ngạc, Hà Nội</p>
    </div>
    """, unsafe_allow_html=True)
