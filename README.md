# 🏥 Rehab AI Monitor (Clinical Ecosystem)

**Hệ thống giám sát tập luyện Phục hồi chức năng từ xa dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính - Quy mô 12 Tab chuyên nghiệp.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rehab-ai-monitor.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Giới thiệu đề tài
Đây là sản phẩm thuộc **Đề tài Nghiên cứu Khoa học cấp Trường (2025-2026)** tại trường **Đại học Y tế Công cộng**, phối hợp cùng **Bệnh viện Đa khoa Phạm Ngọc Thạch**. 

Hệ thống giúp tự động hóa việc giám sát và đánh giá các bài tập phục hồi chức năng (PHCN) cho bệnh nhân viêm quanh khớp vai, đảm bảo bệnh nhân tập luyện đúng kỹ thuật ngay cả khi ở nhà.

## ✨ Tính năng nổi bật (v2.5 Modernized)
- 💎 **Giao diện Cao cấp:** Thiết kế card-based hiện đại, bố cục hàng ngang chuyên nghiệp với hiệu ứng hover và Glassmorphism.
- 🩺 **Luồng liên lạc trực tiếp:** Bệnh nhân gửi video và triệu chứng (VAS) trực tiếp từ Sidebar đến Bác sĩ, KTV và NCV.
- 📄 **Thông tin Nghiên cứu:** Trang thông tin chi tiết dành riêng cho bệnh nhân tham gia NCKH (Quy trình, đạo đức, bảo mật).
- 🚀 **Điều hướng thông minh:** Tự động chuyển Tab khi chọn video để đánh giá (Auto-Tab Switching).
- 🦾 **Phân tích khung xương AI:** Sử dụng MediaPipe Pose Estimation cho độ chính xác cao nhất.
- ⏰ **Lịch nhắc nhở & Hướng dẫn:** Hệ thống quản lý lịch trình và tài liệu hướng dẫn bài tập đa phương tiện.

## 🗺️ Cấu trúc Tab Điều hướng (Role-based)
Hệ thống tự động thay đổi cấu trúc dựa trên vai trò người dùng:
- **Bệnh nhân:** Tập trung vào tập luyện, xem kết quả, khai báo triệu chứng và thông tin nghiên cứu.
- **Bác sĩ / KTV:** Chuyên sâu vào danh sách video BN, đánh giá chuyên môn lâm sàng và theo dõi triệu chứng.
- **Nghiên cứu viên:** Phân tích tọa độ AI, xuất dữ liệu CSV, quản lý video thô và thông số kỹ thuật mô hình.
- **Quản trị viên:** Quản lý tài khoản, cấu hình hệ thống và cơ sở dữ liệu.

## 🛠️ Công nghệ sử dụng
- **AI Core:** MediaPipe, OpenCV, WebRTC (Streamlit-webrtc)
- **Framework:** Streamlit (Custom CSS/JS)
- **Data:** Pandas, NumPy, JSON Persistence
- **Visualization:** Plotly, Figure Factory (Heatmaps)

## 🚀 Chạy ứng dụng
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍⚕️ Nhóm nghiên cứu
- **Giảng viên hướng dẫn:** TS. Trần Hồng Việt
- **Chủ nhiệm đề tài:** Đinh Lê Quỳnh Phương
- **Thành viên nhóm NCKH:** Kim Mạnh Hưng, Nguyễn Hải An, Nguyễn Thị Thanh Nga, Phan Vân Anh, Nguyễn Thị Thơm, Nguyễn Thị Thu Hương
- **Đơn vị:** Đại học Y tế Công cộng - Bệnh viện Đa khoa Phạm Ngọc Thạch.

---
© 2025-2026 Rehab AI Monitor Team.

