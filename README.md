# 🏥 Rehab AI Monitor (Clinical Ecosystem)

**Hệ thống giám sát tập luyện Phục hồi chức năng từ xa dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính - Quy mô 12 Tab chuyên nghiệp.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rehab-ai-monitor.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Giới thiệu đề tài
Đây là sản phẩm thuộc **Đề tài Nghiên cứu Khoa học cấp Trường (2025-2026)** tại trường **Đại học Y tế Công cộng**, phối hợp cùng **Bệnh viện Đa khoa Phạm Ngọc Thạch**. 

Hệ thống giúp tự động hóa việc giám sát và đánh giá các bài tập phục hồi chức năng (PHCN) cho bệnh nhân viêm quanh khớp vai, đảm bảo bệnh nhân tập luyện đúng kỹ thuật ngay cả khi ở nhà.

## ✨ Tính năng nổi bật (New Updates)
- 📹 **Tập luyện Real-time (WebRTC):** Phân tích trực tiếp qua Webcam với phản hồi lỗi tức thì (⚠️ SAI TƯ THẾ).
- 📈 **Theo dõi tiến triển (Long-term Tracking):** Lưu trữ lịch sử tập luyện thực tế và vẽ biểu đồ xu hướng hồi phục.
- 🦾 **Phân tích khung xương AI:** Sử dụng MediaPipe Pose (Model Full) cho độ chính xác cao.
- 📐 **Đo góc sinh học:** Tự động tính toán góc vai và góc khuỷu tay theo thời gian thực.
- 📊 **Báo cáo chuyên sâu:** Xuất dữ liệu dưới dạng biểu đồ Plotly và file CSV cho nghiên cứu khoa học.
- ⏰ **Lịch nhắc nhở:** Quản lý lịch tập, lịch uống thuốc và hẹn khám cho bệnh nhân.

## 🗺️ Cấu trúc 12 Tab Điều hướng
Hệ thống được tổ chức theo quy trình lâm sàng chuẩn:
1. **Trang chủ:** Tiếp nhận thông tin & Hướng dẫn bài tập.
2. **📹 Trực tiếp:** Tập luyện thực tế với phản hồi AI ngay lập tức.
3. **📊 Phân tích:** Dashboard chỉ số chuyên sâu sau khi tập.
4. **🎬 Video & Ảnh:** Xem lại video đã vẽ khung xương và Gallery lỗi.
5. **⏰ Lịch nhắc nhở:** Quản lý thời gian biểu tập luyện.
6. **📈 Tiến triển:** Biểu đồ lịch sử tập luyện thực tế qua các ngày.
7. **Hướng dẫn, Kiến thức PHCN, Công nghệ, Đề tài NCKH, Thành viên, Phản hồi.**

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

