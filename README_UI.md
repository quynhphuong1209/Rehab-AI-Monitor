# 🎨 Tài liệu Kiến trúc Giao diện (UI/UX) - Rehab AI Monitor

Tài liệu này mô tả chi tiết về cấu trúc mã nguồn, thiết kế giao diện (UI) và trải nghiệm người dùng (UX) của hệ thống **Rehab AI Monitor**. Hệ thống được xây dựng bằng **Streamlit** kết hợp với **Custom CSS/HTML** để mang lại giao diện hiện đại, chuyên nghiệp và thân thiện với người dùng y tế.

---

## 1. Tổng quan Công nghệ UI
- **Framework chính:** Streamlit (Python).
- **Tùy biến Giao diện:** Sử dụng `st.markdown(..., unsafe_allow_html=True)` để nhúng mã CSS tĩnh, tạo các hiệu ứng bo góc, đổ bóng và gradient.
- **Biểu đồ Tương tác:** Sử dụng **Plotly** (`plotly.graph_objects`, `plotly.express`) để vẽ các biểu đồ chuyên sâu, hỗ trợ zoom/pan ngay trên trình duyệt.
- **Tối ưu Hiển thị:** Thay vì load toàn bộ dữ liệu vào RAM (gây crash OOM), UI áp dụng chiến lược **Lazy Loading** và **Phân trang (Pagination)**.

---

## 2. Cấu trúc Tabs (Điều hướng chính)
Toàn bộ ứng dụng được gói gọn trong 6 Tabs chính để điều hướng mượt mà không cần chuyển trang:

### 🏠 Tab 1: Trang Chủ (Home & Input)
- **Khu vực Sidebar:** Nơi bác sĩ/kỹ thuật viên nhập thông tin bệnh nhân (Tên, Mã số, Tuổi, Chỉ số VAS).
- **Chọn Bài Tập:** Danh sách các bài tập (ví dụ: Codman, Gậy) với hướng dẫn chi tiết và video minh họa (nhúng qua `st.video`).
- **Upload Video:** Khu vực kéo thả file. 
  - *Điểm nhấn kỹ thuật:* Sử dụng cơ chế đọc luồng (Chunk Streaming 1MB/lần) để tránh tràn bộ nhớ RAM khi tải video lớn. Cập nhật tiến trình qua `st.progress`.

### 📊 Tab 2: Phân Tích (Dashboard)
Giao diện phân tích được chia thành 4 Sub-tabs nhỏ để không làm rối mắt người xem:
1. **Tổng Quan:** Các thẻ chỉ số (Metric Cards) hiển thị độ chính xác tổng thể, số frame đúng/sai. Sử dụng thiết kế bo tròn với gradient hiện đại.
2. **Biểu Đồ Chi Tiết:** Tích hợp Plotly vẽ biểu đồ đường (Line chart) theo dõi góc vai/khuỷu theo thời gian.
3. **Cảnh Báo Lỗi:** Hiển thị biểu đồ tròn (Pie chart) thống kê các lỗi tư thế phổ biến (ví dụ: tay quá thấp, khuỷu tay cong).
4. **Xuất Dữ Liệu:** Cung cấp bảng dữ liệu thô (Dataframe) và các nút `st.download_button` tải file `.csv` hoặc `.zip` biểu đồ.

### 🎬 Tab 3: Video & Ảnh (Gallery)
- **Video Playback:** Phát lại video sau khi đã vẽ khung xương bằng `st.video()`.
- **Bộ Sưu Tập Khung Hình (Gallery):**
  - *Thuật toán Phân trang:* Hiển thị tối đa 24 frames trên một trang. Chuyển trang thông qua các nút `st.button` cập nhật `st.session_state`.
  - *Tối ưu RAM:* Đọc ảnh trực tiếp từ đường dẫn đĩa (Disk Path) thông qua `st.image()`, loại bỏ hoàn toàn việc mã hóa Base64 để tiết kiệm RAM.
  - *Trực quan hóa:* Box thông tin dưới mỗi ảnh sẽ có màu Xanh (PASS) hoặc Đỏ (FAIL), có viền sáng giúp dễ nhận biết lỗi.

### ⏰ Tab 4: Lịch Nhắc Nhở
- Hiển thị ngày giờ hiện tại trực quan.
- Bảng lịch trình tập luyện dự kiến cho bệnh nhân bằng Markdown Table.

### 📚 Tab 5 & 6: Thông tin NCKH & Đội ngũ
- Trình bày thông tin dự án, bối cảnh y tế bằng các khối `st.expander` (Accordion) giúp tiết kiệm không gian.
- Khối giới thiệu thành viên sử dụng thiết kế thẻ (Card Design) CSS.

---

## 3. Các Điểm Nhấn Kỹ Thuật (UI/UX Highlights)

### 3.1. Hệ thống Custom CSS Tích Hợp
Code UI áp dụng một đoạn CSS toàn cục (`<style>`) đặt ở đầu `app.py`:
- **Chủ đề Tối (Dark Theme):** `linear-gradient(135deg, #0a0a0a 0%, #0f0f1a 50%, #1a1a2e 100%)`.
- **Thẻ Metric (Metric Cards):** Các hộp hiển thị thông số được thiết kế với màu nền trong suốt, viền sáng (`border: 1px solid rgba(255,255,255,0.1)`).
- **Hover Effects:** Các nút bấm (`stButton`) và ảnh thumbnail có hiệu ứng `transform: scale(1.02)` khi di chuột qua, tạo cảm giác app "sống động".

### 3.2. Quản Lý Trạng Thái (State Management)
Giao diện Streamlit có nhược điểm là tải lại toàn trang khi tương tác. Để giữ UI mượt mà:
- Sử dụng `st.session_state` để ghi nhớ kết quả phân tích (`has_data`, `angle_df`), giúp người dùng chuyển đổi qua lại giữa các Tab mà không bị mất dữ liệu hay phải chờ phân tích lại.
- Trạng thái phân trang (`current_page`) được lưu kỹ trong Session để người dùng xem thư viện ảnh mượt mà.

### 3.3. Tối ưu Hiệu năng Render (Tránh sập Web)
- Khác với giao diện Web thông thường, môi trường 1GB RAM bắt buộc UI không được ôm quá nhiều dữ liệu. 
- Mọi thao tác lưu trữ ảnh/biểu đồ đều được xử lý dưới dạng file tạm (Tempfile) và Streamlit chỉ đóng vai trò là ống dẫn (Stream) nội dung lên giao diện. File ZIP chỉ được tải khi người dùng thực sự ấn nút thay vì tạo sẵn.

---

*Tài liệu này giúp đội ngũ phát triển dễ dàng bảo trì và mở rộng giao diện hệ thống trong tương lai.*
