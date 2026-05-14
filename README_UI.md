# 🎨 Tài liệu Kiến trúc Giao diện (UI/UX) - Rehab AI Monitor (v3.1 Updated)

Tài liệu này mô tả chi tiết về cấu trúc mã nguồn, thiết kế giao diện (UI) và trải nghiệm người dùng (UX) của hệ thống **Rehab AI Monitor**. Hệ thống áp dụng phong cách **Clinical Aesthetics** (Thẩm mỹ Lâm sàng) kết hợp với công nghệ **Glassmorphism** và **Custom CSS** cao cấp.

---

## 1. Tổng quan Công nghệ UI
- **Framework chính:** Streamlit (Python).
- **Thiết kế:** Custom CSS nhúng qua Markdown (`unsafe_allow_html=True`) với font chữ chủ đạo là **Times New Roman**.
- **Tương tác:** 
    - **WebRTC:** Xử lý luồng Camera thời gian thực.
    - **JavaScript Injection:** Thực hiện tự động hóa UI (Auto-Tab Switching).
    - **Plotly Engine:** Trực quan hóa dữ liệu lâm sàng và AI.
- **Tối ưu:** Cơ chế đồng bộ Theme (Light/Dark Mode) đảm bảo không có artifacts và độ tương phản chuẩn y tế. Hệ thống ép font chữ **Times New Roman** thông qua CSS Inject để tạo môi trường học thuật và lâm sàng chuyên nghiệp.

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
- **Sidebar Phẳng:** Loại bỏ các container lồng nhau để tạo luồng thao tác phẳng (Thông tin BN -> Chọn bài tập -> Khai báo triệu chứng).

### 🩺 Đánh Giá Chuyên Môn (Bác sĩ/KTV)
- **Workflow Tối ưu:** Bác sĩ chọn video BN từ danh sách -> Hệ thống tự động chuyển sang Tab Đánh giá -> Nhận xét lâm sàng.
- **Hợp nhất Dữ liệu:** Hiển thị song song kết quả khai báo của BN và kết quả phân tích AI để Bác sĩ đưa ra chỉ định chính xác nhất.

### 📊 Phân Tích Kỹ Thuật (Nghiên cứu viên)
- **NCV Dashboard:** Truy cập sâu vào cấu hình mô hình (Confidence, Skip Frames), xem tọa độ khớp và trích xuất dữ liệu frame.
- **Unified Results View:** NCV xem được cả nhận xét của Bác sĩ và kết quả AI trong cùng một màn hình tinh gọn để đối chiếu dữ liệu nghiên cứu mà không bị chồng chéo tính năng.
- **Export:** Công cụ xuất CSV và ZIP frame được tích hợp sẵn để phục vụ báo cáo khoa học.

---

## 3. Các Điểm Nhấn Kỹ Thuật UI
- **Persistence UI:** Lưu trạng thái đăng nhập và thông tin phiên làm việc qua Session State và JSON.
- **Performance Aesthetics:** Sử dụng CSS Transitions cho các hiệu ứng hover, Glassmorphism cho Sidebar và các khối container.
- **Responsive Layout:** Tối ưu hóa hiển thị trên mọi kích thước màn hình, từ clinical tablet đến smartphone cá nhân của bệnh nhân.

---

## 4. Hướng dẫn Thay đổi Theme
Hệ thống tích hợp nút gạt (Toggle) ngay tại thanh điều hướng trên cùng và màn hình đăng nhập:
- **Dark Mode:** Phù hợp cho việc phân tích video và biểu đồ AI (giảm mỏi mắt).
- **Light Mode:** Phù hợp cho môi trường văn phòng bác sĩ và bệnh nhân sử dụng ban ngày (độ tương phản cao).

---
© 2025-2026 Rehab AI Monitor Team.
