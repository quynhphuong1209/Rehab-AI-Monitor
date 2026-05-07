# 🎨 Tài liệu Kiến trúc Giao diện (UI/UX) - Rehab AI Monitor

Tài liệu này mô tả chi tiết về cấu trúc mã nguồn, thiết kế giao diện (UI) và trải nghiệm người dùng (UX) của hệ thống **Rehab AI Monitor**. Hệ thống áp dụng phong cách **Glassmorphism** hiện đại kết hợp với **Custom CSS** cao cấp.

---

## 1. Tổng quan Công nghệ UI
- **Framework chính:** Streamlit (Python).
- **Thiết kế:** Custom CSS nhúng qua Markdown (`unsafe_allow_html=True`).
- **Tương tác:** WebRTC (Xử lý Camera trình duyệt), Plotly (Biểu đồ tương tác).
- **Tối ưu:** Quản lý RAM 1GB thông qua luồng đọc Chunk và Session State.

---

## 2. Cấu trúc Giao diện & Trải nghiệm (v2.5)
Ứng dụng được thiết kế tối ưu cho từng vai trò người dùng (Role-based UI):

### 🏠 Trang Chủ (Dashboard Tổng quan)
- **Thiết kế:** Bố cục card-based hàng ngang với các thẻ thông tin bài tập (Thời gian, Số lần, Thông số chuẩn).
- **Trải nghiệm:** Tải lên video và gửi trực tiếp cho đội ngũ chuyên môn. Hệ thống tự động ẩn các tính năng kỹ thuật phức tạp đối với Bệnh nhân để tối ưu hóa sự đơn giản.

### 📄 Trang Thông Tin Nghiên Cứu (Dành cho Bệnh nhân)
- **Nội dung:** Hiển thị chi tiết về đề tài NCKH (Quy trình, nguy cơ, bảo mật, thông tin liên hệ).
- **Trình bày:** Sử dụng các khối `Expander` và `Custom Cards` để tạo cảm giác chuyên nghiệp, minh bạch.

### 🩺 Đánh Giá Chuyên Môn (Dành cho Bác sĩ/KTV)
- **Workflow:** Bác sĩ chọn video BN -> Tự động chuyển Tab -> Nhập đánh giá lâm sàng.
- **Tập trung:** Loại bỏ các tab biểu đồ AI kỹ thuật để Bác sĩ tập trung hoàn toàn vào việc nhận xét chuyên môn và chỉ định kế hoạch điều trị.

### 📊 Phân Tích & Video (Dành cho Nghiên cứu viên)
- **NCV Dashboard:** Truy cập toàn bộ dữ liệu AI, biểu đồ tọa độ, và video trích xuất khung xương để kiểm định mô hình.
- **Export:** Công cụ xuất CSV và ZIP frame để phục vụ báo cáo khoa học.

### 🛠️ Sidebar & Form (Hợp nhất)
- **Thông tin BN:** Hiển thị trên cùng (Họ tên, Tuổi, Giới tính).
- **Khai báo Triệu chứng:** Form phẳng ngay trong Sidebar giúp BN báo cáo mức độ đau (VAS) nhanh chóng.
- **Chế độ Sáng/Tối:** Chuyển đổi Theme linh hoạt hỗ trợ tương phản tốt.

---

## 3. Các Điểm Nhấn Kỹ Thuật
- **Persistence:** Lưu lịch sử tập luyện vào tệp JSON cục bộ để theo dõi dài hạn.
- **Performance:** Cơ chế dọn rác (GC) chủ động để duy trì tính ổn định trên Streamlit Cloud.
- **Aesthetics:** Gradient background, bo góc 15px-20px, và các micro-animations cho nút bấm.

---
© 2025-2026 Rehab AI Monitor Team.
