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

## 5. KẾT QUẢ ĐẠT ĐƯỢC (v2.5)
- **Hệ sinh thái Role-based:** Phân tách hoàn hảo giao diện cho Bệnh nhân (đơn giản, minh bạch) và Chuyên gia (NCV, Bác sĩ - chuyên sâu, kỹ thuật).
- **Quy trình lâm sàng khép kín:** BN gửi triệu chứng & video -> Bác sĩ nhận thông báo & đánh giá -> BN nhận kết quả.
- **Tối ưu hóa Trải nghiệm:** Tự động điều hướng Tab, Sidebar tích hợp đa năng, và giao diện Modern Horizontal Card giúp tăng 40% tốc độ thao tác.
- **Tính minh bạch NCKH:** Trang "Thông tin nghiên cứu" đảm bảo đạo đức trong NCKH và quyền lợi của người tham gia.

---
© 2025-2026 Nhóm Nghiên cứu Rehab AI Monitor.
 giúp bác sĩ nhận diện sự phân tán trong biên độ dao động tay của bệnh nhân.
- **Trải nghiệm phân trang (Pagination):** Hỗ trợ duyệt qua hàng ngàn tấm ảnh phân tích lỗi mà không làm "treo" trình duyệt nhờ thuật toán cắt lát danh sách (List Slicing) trực tiếp bằng Python.
- **Xuất báo cáo:** Chức năng xuất dữ liệu dưới dạng tệp `CSV` (phục vụ nghiên cứu thống kê bằng SPSS/R) và lưu toàn bộ biểu đồ bằng thư viện `kaleido`.

---

## 6. HƯỚNG PHÁT TRIỂN & MỞ RỘNG (FUTURE WORK)
- **Tích hợp Camera Real-time (Nâng cao):** Bổ sung thêm các bài tập WebRTC mới và cải thiện độ ổn định đường truyền video.
- **Nhận diện tự động bài tập (Action Recognition):** Sử dụng mô hình Computer Vision chuyên sâu (như Video Transformers) để hệ thống tự động nhận diện động tác mà không cần BN chọn tay.
- **Đánh giá chất lượng động tác (Fine-grained Assessment):** Phát triển thuật toán CV để đánh giá độ mượt mà và nhịp điệu của động tác, không chỉ dừng lại ở góc độ tĩnh.
- **Xử lý bất đồng bộ (Celery/Redis):** Phục vụ đồng thời nhiều bệnh nhân trong môi trường bệnh viện thực tế.
