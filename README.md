# 🏥 Rehab AI Monitor

**Hệ thống giám sát tập luyện Phục hồi chức năng từ xa dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rehab-ai-monitor.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Giới thiệu đề tài
Đây là sản phẩm thuộc **Đề tài Nghiên cứu Khoa học cấp Trường (2025-2026)** tại trường **Đại học Y tế Công cộng**, phối hợp cùng **Bệnh viện Đa khoa Phạm Ngọc Thạch**. 

Hệ thống giúp tự động hóa việc giám sát và đánh giá các bài tập phục hồi chức năng (PHCN) cho bệnh nhân viêm quanh khớp vai, đảm bảo bệnh nhân tập luyện đúng kỹ thuật ngay cả khi ở nhà.

## ✨ Tính năng nổi bật
- 🦾 **Phân tích khung xương AI:** Sử dụng MediaPipe Pose (Model Full) cho độ chính xác cao.
- 📐 **Đo góc sinh học trực tiếp:** Tự động tính toán góc vai và góc khuỷu tay theo thời gian thực.
- 🎨 **Trực quan hóa sinh động:** Vẽ cung tròn (angle arcs) và tô màu cảnh báo đúng/sai ngay trên video.
- 📊 **Báo cáo chi tiết:** Xuất dữ liệu dưới dạng biểu đồ Plotly chuyên nghiệp và file CSV.
- ⏰ **Lịch nhắc nhở thông minh:** Quản lý lịch tập, lịch uống thuốc và hẹn khám cho bệnh nhân.
- 🎬 **Tối ưu hóa Web:** Chuyển đổi video sang chuẩn H.264 giúp xem mượt mà trên mọi trình duyệt.

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.10+
- **Thị giác máy tính:** MediaPipe, OpenCV
- **Giao diện:** Streamlit (với Custom CSS/JS cao cấp)
- **Xử lý dữ liệu:** Pandas, NumPy
- **Trực quan hóa:** Plotly, Matplotlib
- **Hệ thống:** FFmpeg (Xử lý hậu kỳ video)

## 🚀 Hướng dẫn cài đặt (Local)

1. **Clone repository:**
   ```bash
   git clone https://github.com/quynhphuong1209/Rehab-AI-Monitor.git
   cd Rehab-AI-Monitor
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Cài đặt FFmpeg:**
   - Windows: Tải bản build tại [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
   - Linux: `sudo apt-get install ffmpeg`

4. **Chạy ứng dụng:**
   ```bash
   streamlit run app.py
   ```

## 👩‍⚕️ Nhóm nghiên cứu
- **Giảng viên hướng dẫn:** TS. Trần Hồng Việt
- **Chủ nhiệm đề tài:** Đinh Lê Quỳnh Phương
- **Thành viên:** Kim Mạnh Hưng, Nguyễn Hải An, Phan Vân Anh, Nguyễn Thị Thanh Nga, Nguyễn Thị Thơm, Nguyễn Thị Thu Hương

---
© 2025-2026 Rehab AI Monitor Team - Đại học Y tế Công cộng.

