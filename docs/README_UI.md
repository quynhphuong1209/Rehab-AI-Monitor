# 🎨 Tài liệu Kiến trúc Giao diện (UI/UX) - Rehab AI Monitor (v3.2 Updated)

Tài liệu này mô tả chi tiết về cấu trúc mã nguồn, thiết kế giao diện (UI) và trải nghiệm người dùng (UX) của hệ thống **Rehab AI Monitor**. Hệ thống áp dụng phong cách **Clinical Aesthetics** (Thẩm mỹ Lâm sàng) kết hợp với công nghệ **Glassmorphism** và **Custom CSS** cao cấp.

---

## 1. Tổng quan Công nghệ UI
- **Framework chính:** Streamlit (Python).
- **Thiết kế:** Custom CSS nhúng qua Markdown (`unsafe_allow_html=True`) với font chữ chủ đạo là **Times New Roman** chuẩn học thuật và y tế.
- **Tương tác:** 
    - **WebRTC:** Xử lý luồng Camera thời gian thực.
    - **JavaScript Injection:** Thực hiện tự động hóa điều hướng UI (Auto-Tab Switching).
    - **Plotly Engine:** Trực quan hóa dữ liệu lâm sàng và AI đa chiều.
- **Tối ưu:** Cơ chế đồng bộ Theme (Light/Dark Mode) đảm bảo không có artifacts và độ tương phản chuẩn y tế. Hệ thống ép font chữ **Times New Roman** thông qua CSS Inject để tạo môi trường học thuật chuyên nghiệp.

---

## 2. Cấu trúc Giao diện & Trải nghiệm (Finalized)
Ứng dụng được thiết kế tối ưu hóa theo mô hình Role-based UI (Giao diện theo vai trò):

### 🏥 Thẩm mỹ Lâm sàng (Clinical Aesthetics)
- **Typography:** Sử dụng font chữ 'Times New Roman' cho toàn bộ hệ thống (tiêu đề, nội dung, footer), mang lại cảm giác tin cậy, chuyên nghiệp trong môi trường y tế.
- **Theme Sync:** Hệ thống tự động điều chỉnh màu sắc Input, Card, Sidebar và các thông báo (Success/Info/Warning) khi chuyển đổi giữa Light và Dark mode, đảm bảo tính nhất quán tuyệt đối.

### 📱 Tối ưu hóa Di động (Mobile-First Optimization)
- **Standardized Tabs:** Hệ thống Tab được tái thiết kế bằng CSS cấp cao để đảm bảo hiển thị hoàn hảo trên màn hình dọc của smartphone. 
- **No Clipping:** Sử dụng `min-width: fit-content` và `white-space: nowrap` để ngăn chặn việc cắt chữ hoặc nén tiêu đề Tab.
- **Horizontal Scrolling:** Cho phép người dùng vuốt ngang mượt mà để truy cập tất cả các tính năng mà không làm vỡ bố cục ứng dụng.

### 🏠 Trang Chủ & Dashboard
- **Thiết kế:** Bố cục card-based với các thẻ thông tin bài tập (Thời gian, Số lần, Thông số chuẩn).
- **Sidebar Phẳng (Flattened Sidebar):** Loại bỏ hoàn toàn các container lồng nhau để tạo luồng thao tác phẳng mật độ cao, giúp tối ưu diện tích và giảm thiểu thao tác menu (Thông tin BN -> Chọn bài tập -> Khai báo triệu chứng).
- **Admin Log:** Bổ sung giao diện Nhật ký hoạt động hợp nhất cho Quản trị viên, hỗ trợ lọc và xuất dữ liệu CSV chuyên nghiệp.

### 🩺 Đạo đức & Thông tin Nghiên cứu (Ethics & Consent UI)
- **Bioethics Panel:** Tích hợp Trang thông tin nghiên cứu (PIS) chuyên nghiệp dưới dạng các hộp co giãn linh hoạt (Expander) (Giới thiệu, Quy trình, Nguy cơ, Quyền lợi, Bảo mật, và Đóng góp đề tài).
- **Contact Cards:** Thẻ thông tin liên hệ được thiết kế nổi bật, phân chia màu sắc trực quan: màu xanh dương chủ đạo cho Nghiên cứu viên chính (NCV) và màu đỏ/cam cảnh báo cho Hội đồng Đạo đức (IRB) để người bệnh dễ dàng liên hệ.

### 🩺 Đánh Giá Chuyên Môn (Bác sĩ/KTV)
- **Workflow Tối ưu:** Bác sĩ chọn video BN từ danh sách -> Hệ thống tự động kích hoạt JavaScript để chuyển đổi sang Tab Đánh giá -> Nhận xét lâm sàng.
- **Hợp nhất Dữ liệu:** Hiển thị song song kết quả khai báo của BN và kết quả phân tích AI để Bác sĩ đưa ra chỉ định chính xác nhất.

### 📊 Phân Tích Kỹ Thuật (Nghiên cứu viên)
- **NCV Dashboard:** Truy cập sâu vào cấu hình mô hình (Confidence, Skip Frames), xem tọa độ khớp và trích xuất dữ liệu frame.
- **Biểu đồ trực quan hóa đa chiều (Multi-Chart Analytics):**
  - **ROM Trend & Distribution:** Vẽ biểu đồ đường thay đổi góc khớp và biểu đồ hộp (Boxplot) để đánh giá độ biến thiên cũng như tính ổn định của chuyển động.
  - **Radar Chart:** Trực quan hóa cùng lúc 7 chỉ số chất lượng bài tập (ACC, MAE, RMSE, ICC, F1-Score, Precision, Recall).
- **Stage-based Table:** Bảng đối sánh 3 giai đoạn PHCN hiển thị song song giúp NCV dễ dàng phát hiện mức độ hoàn thành bài tập ở từng ngưỡng sai số khác nhau ($45^\circ$, $30^\circ$, $15^\circ$).
- **Đồng bộ Ground Truth:** Hiển thị thẻ nhận xét và phác đồ điều trị của Bác sĩ dưới dạng thẻ viền sáng màu xanh dương nhạt (Light blue border) trực tiếp trên giao diện của NCV để phục vụ đối soát thuật toán.

### 📁 Quản lý & Xuất báo cáo (Consolidated Export Tab)
- **Multi-format Downloads:** 
  - Cho phép xuất hình ảnh PNG trực tiếp cho mọi biểu đồ (Pie chart, ROM vai, ROM khuỷu, Boxplots, Radar chart).
  - Tải file CSV chứa toàn bộ dữ liệu thô tọa độ góc khớp theo khung hình (Frame ID).
- **Lazy ZIP Engine:** Cơ chế tạo ZIP danh sách khung hình y tế một cách lười (chỉ nén khi người dùng click yêu cầu) nhằm tránh tình trạng tràn RAM (Out of Memory - OOM) của máy chủ khi xử lý video dung lượng lớn.

---

## 3. Các Điểm Nhấn Kỹ Thuật UI
- **Persistence UI:** Lưu trạng thái đăng nhập và thông tin phiên làm việc qua Session State và JSON.
- **Performance Aesthetics:** Sử dụng CSS Transitions cho các hiệu ứng hover, Glassmorphism cho Sidebar và các khối container.
- **Robust Path Resolution:** Hệ thống giải quyết đường dẫn đa tầng được thiết kế riêng để tự động dò tìm các tệp tin logo base64 và file tham chiếu chuẩn trong các thư mục `assets/` và `database/`, hạn chế tối đa lỗi thiếu file trên Cloud.
- **Responsive Layout:** Tối ưu hóa hiển thị trên mọi kích thước màn hình, từ clinical tablet đến smartphone cá nhân của bệnh nhân.

---

## 4. Hướng dẫn Thay đổi Theme
Hệ thống tích hợp nút gạt (Toggle) ngay tại thanh điều hướng trên cùng và màn hình đăng nhập:
- **Dark Mode:** Phù hợp cho việc phân tích video và biểu đồ AI (giảm mỏi mắt).
- **Light Mode:** Phù hợp cho môi trường văn phòng bác sĩ và bệnh nhân sử dụng ban ngày (độ tương phản cao).

---

## 5. Cấu trúc thư mục được tinh chỉnh
Hệ thống UI hoạt động đồng bộ với cấu trúc thư mục được quy hoạch ngăn nắp:
- `assets/`: Lưu trữ logo cơ quan và khoa Khoa học dữ liệu HUPH.
- `database/`: Chứa các file dữ liệu cấu hình người dùng, lịch sử tập luyện và góc khớp chuẩn của 3 bài tập.
- `docs/`: Chứa các file tài liệu hướng dẫn kỹ thuật, kiến trúc giao diện, và đề cương nghiên cứu khoa học.
- `scripts/`: Chứa các script dọn dẹp và bảo trì hệ thống định kỳ.

---

© 2025-2026 Rehab AI Monitor Team.

