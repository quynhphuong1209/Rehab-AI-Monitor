# 🎨 Tài liệu Kiến trúc Giao diện (UI/UX) - Rehab AI Monitor

Tài liệu này mô tả chi tiết về cấu trúc mã nguồn, thiết kế giao diện (UI) và trải nghiệm người dùng (UX) của hệ thống **Rehab AI Monitor**. Hệ thống áp dụng phong cách **Glassmorphism** hiện đại kết hợp với **Custom CSS** cao cấp.

---

## 1. Tổng quan Công nghệ UI
- **Framework chính:** Streamlit (Python).
- **Thiết kế:** Custom CSS nhúng qua Markdown (`unsafe_allow_html=True`).
- **Tương tác:** WebRTC (Xử lý Camera trình duyệt), Plotly (Biểu đồ tương tác).
- **Tối ưu:** Quản lý RAM 1GB thông qua luồng đọc Chunk và Session State.

---

## 2. Cấu trúc 12 Tabs (Tiến trình người dùng)
Ứng dụng được tổ chức theo quy trình lâm sàng logic:

### 🏠 Tab 1: Trang Chủ
- Nhập thông tin bệnh nhân và lựa chọn bài tập PHCN.
- Hiển thị hướng dẫn video và hình ảnh bài tập chuẩn.

### 📹 Tab 2: Trực Tiếp (Real-time Analysis)
- **Công nghệ:** WebRTC Streamer.
- **Trải nghiệm:** Bệnh nhân bật Camera và tập luyện. Hệ thống vẽ khung xương và tính góc trực tiếp.
- **Phản hồi:** Cảnh báo "SAI TƯ THẾ" hiện ngay trên khung hình nếu độ lệch vượt ngưỡng cho phép.

### 📊 Tab 3: Phân Tích (Dashboard)
- Dashboard chỉ số chi tiết sau khi phân tích video.
- **Metric Cards:** 4 thẻ chỉ số cao cấp (Accuracy, F1-Score, ICC, Stability).
- **AI Insights:** Nhận định chuyên môn tự động từ mô hình học máy.

### 🎬 Tab 4: Video & Ảnh (Gallery)
- Phát lại video đã xử lý với đầy đủ skeleton và angle arcs.
- Thư viện ảnh (Gallery) phân trang để xem lại các lỗi sai cụ thể.

### ⏰ Tab 5: Lịch Nhắc Nhở
- Quản lý lịch trình tập luyện, nhắc uống thuốc và hẹn tái khám.

### 📈 Tab 6: Tiến Triển (Long-term Progress)
- **Dữ liệu thật:** Tự động lấy dữ liệu từ các buổi tập đã thực hiện.
- **Trực quan:** Biểu đồ đường xu hướng hồi phục và bảng nhật ký chi tiết.

### 📖 Tab 7-12: Tài nguyên & Thông tin
- **Hướng dẫn:** 5 bước cơ bản để làm quen với hệ thống.
- **Kiến thức PHCN:** 4 trụ cột y khoa hiện đại và tài liệu Bộ Y tế/WHO.
- **Công nghệ:** Giải thích về MediaPipe và thuật toán AI.
- **Đề tài NCKH:** Toàn văn báo cáo nghiên cứu và kết quả dự kiến.
- **Thành viên:** Thông tin đội ngũ thực hiện và chuyên gia lâm sàng.
- **Phản hồi:** Kênh hỗ trợ kỹ thuật và đóng góp ý kiến.

---

## 3. Các Điểm Nhấn Kỹ Thuật
- **Persistence:** Lưu lịch sử tập luyện vào tệp JSON cục bộ để theo dõi dài hạn.
- **Performance:** Cơ chế dọn rác (GC) chủ động để duy trì tính ổn định trên Streamlit Cloud.
- **Aesthetics:** Gradient background, bo góc 15px-20px, và các micro-animations cho nút bấm.

---
© 2025-2026 Rehab AI Monitor Team.
