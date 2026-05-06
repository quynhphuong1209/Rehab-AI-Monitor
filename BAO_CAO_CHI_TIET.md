# BÁO CÁO CHI TIẾT MÃ NGUỒN VÀ THUẬT TOÁN 
**HỆ THỐNG GIÁM SÁT TẬP LUYỆN PHỤC HỒI CHỨC NĂNG (REHAB AI MONITOR)**

---

## 1. TỔNG QUAN HỆ THỐNG
**Rehab AI Monitor** là một hệ sinh thái giám sát tập luyện ứng dụng Trí tuệ nhân tạo (AI) và Thị giác máy tính để đánh giá các bài tập phục hồi chức năng (PHCN).

Hệ thống đã phát triển từ mô hình phân tích video tĩnh thành một **Nền tảng Giám sát Thời gian thực** với 12 phân khu chức năng (Tabs), hỗ trợ bệnh nhân từ giai đoạn tập luyện trực tiếp đến theo dõi tiến triển dài hạn.

---

## 2. CHI TIẾT CÁC CÔNG NGHỆ ĐỘT PHÁ

### 2.1. Phân tích Real-time qua WebRTC
Hệ thống tích hợp thư viện `streamlit-webrtc` để thiết lập luồng truyền tải video bảo mật giữa trình duyệt của bệnh nhân và máy chủ AI. 
- **Pose Processor:** Một lớp xử lý tùy chỉnh (Custom VideoProcessor) chạy trên luồng riêng biệt để tính toán tọa độ khớp và vẽ skeleton với độ trễ cực thấp.
- **Instant Feedback:** Thuật toán so sánh góc (Angular Comparison) được thực hiện ngay trong vòng lặp frame của WebRTC, cho phép hiển thị cảnh báo lỗi tức thì trên màn hình camera.

### 2.2. Cơ chế Lưu trữ và Theo dõi Tiến triển (Persistence)
Khác với các ứng dụng Streamlit thông thường bị mất dữ liệu khi tải lại trang, hệ thống đã triển khai cơ chế **Persistent Logging**:
- **JSON Database:** Mọi kết quả phân tích (Accuracy, F1-Score, Bài tập) được tự động nối vào tệp `lich_su_tap_luyen.json`.
- **Longitudinal Tracking:** Tab "Tiến triển" sử dụng dữ liệu từ tệp này để vẽ biểu đồ tăng trưởng hiệu suất, giúp bác sĩ đánh giá tốc độ hồi phục của bệnh nhân qua nhiều tuần.

---

## 3. PHÂN TÍCH LUỒNG XỬ LÝ DỮ LIỆU
1. **Input:** Nhận luồng video từ Webcam (Real-time) hoặc Video tải lên (Offline).
2. **AI Processing:** MediaPipe Pose Estimation trích xuất 33 điểm khớp.
3. **Kinematic Analysis:** Tính toán góc Vai và góc Khuỷu bằng hàm `tinh_goc` (Lượng giác học).
4. **Evaluation:** So sánh với biên độ chuẩn (ROM) và gán nhãn Đúng/Sai.
5. **Reporting:** Trực quan hóa qua Plotly Dashboard và ghi nhật ký vào cơ sở dữ liệu JSON.

---

## 4. TỐI ƯU HÓA HỆ THỐNG (OOM PREVENTION)
Đảm bảo tính ổn định trên môi trường Cloud 1GB RAM:
- **Chunked Streaming:** Đọc/Ghi video theo từng khối 1MB.
- **Active Garbage Collection:** Gọi `gc.collect()` định kỳ để giải phóng bộ nhớ RAM từ các biến C++ của OpenCV.
- **Pagination Gallery:** Chỉ tải ảnh khi người dùng thực sự xem qua cơ chế phân trang.

---

## 5. KẾT QUẢ ĐẠT ĐƯỢC
- **Quy mô:** Hệ thống 12 Tab hoàn chỉnh, bao quát mọi khía cạnh từ lâm sàng đến nghiên cứu.
- **Độ chính xác:** Đạt mức Accuracy ≥ 90% trong các thử nghiệm nội bộ.
- **UX/UI:** Giao diện Glassmorphism chuyên nghiệp, tối ưu cho cả nhân viên y tế và bệnh nhân.

---
© 2025-2026 Nhóm Nghiên cứu Rehab AI Monitor.
 giúp bác sĩ nhận diện sự phân tán trong biên độ dao động tay của bệnh nhân.
- **Trải nghiệm phân trang (Pagination):** Hỗ trợ duyệt qua hàng ngàn tấm ảnh phân tích lỗi mà không làm "treo" trình duyệt nhờ thuật toán cắt lát danh sách (List Slicing) trực tiếp bằng Python.
- **Xuất báo cáo:** Chức năng xuất dữ liệu dưới dạng tệp `CSV` (phục vụ nghiên cứu thống kê bằng SPSS/R) và lưu toàn bộ biểu đồ bằng thư viện `kaleido`.

---

## 6. HƯỚNG PHÁT TRIỂN & MỞ RỘNG (FUTURE WORK)
- **Tích hợp Camera Real-time:** Bổ sung tính năng WebRTC để giám sát trực tiếp qua webcam thay vì chỉ upload video.
- **Nhận diện tự động bài tập (Action Recognition):** Dùng mô hình LSTM hoặc TimeSformer để hệ thống tự động biết bệnh nhân đang tập bài gì mà không cần chọn tay.
- **Xử lý bất đồng bộ (Celery/Redis):** Tách tác vụ phân tích video khỏi tiến trình chính của Streamlit để phục vụ nhiều bệnh nhân cùng lúc mà không sợ nghẽn cổ chai.
