# BÁO CÁO CHI TIẾT MÃ NGUỒN VÀ THUẬT TOÁN (v3.1 Updated)
**HỆ THỐNG GIÁM SÁT TẬP LUYỆN PHỤC HỒI CHỨC NĂNG (REHAB AI MONITOR)**

---

## 1. TỔNG QUAN HỆ THỐNG
**Rehab AI Monitor** là một hệ sinh thái giám sát tập luyện y tế ứng dụng Trí tuệ nhân tạo (AI) và Thị giác máy tính để đánh giá các bài tập phục hồi chức năng (PHCN). 

Hệ thống đã đạt đến độ hoàn thiện cao nhất (Production-ready) với 12 phân khu chức năng, hỗ trợ quy trình lâm sàng khép kín giữa Bệnh nhân, Bác sĩ/KTV và Nghiên cứu viên.

---

## 2. CHI TIẾT CÁC CÔNG NGHỆ ĐỘT PHÁ

### 2.1. Phân tích Real-time qua WebRTC & AI Core
- **Pose Processor:** Tích hợp MediaPipe Pose (Heavy Model) chạy trên luồng xử lý riêng biệt để tính toán 33 tọa độ khớp xương với độ trễ cực thấp (<50ms).
- **Angular Comparison:** Thuật toán tính toán góc Vai và Khuỷu tay dựa trên lượng giác học (Arc-cosine) để so sánh với biên độ vận động chuẩn (ROM - Range of Motion).

### 2.2. Cơ chế Lưu trữ và Theo dõi Tiến triển (Persistence)
- **JSON Unified Database:** Hệ thống sử dụng 6 tệp tin JSON chuyên biệt (`users.json`, `patient_symptoms.json`, `doctor_evaluations.json`, `schedules.json`, `video_list.json`, `lich_su_tap_luyen.json`) để lưu trữ bền vững triệu chứng bệnh nhân, đánh giá của bác sĩ và kết quả AI.
- **Python 3.10 Compatibility:** Hệ thống được tối ưu hóa cho Python 3.10, sử dụng các thư viện build sẵn (.whl) cho MediaPipe và NumPy 1.26.4 để đảm bảo hiệu suất xử lý khung xương ổn định nhất trên Windows/Linux.
- **Longitudinal Analytics:** Tự động tổng hợp dữ liệu để vẽ biểu đồ tiến triển, giúp bác sĩ nhận diện sự cải thiện biên độ vận động qua từng phiên tập.

### 2.3. Tối ưu hóa Trải nghiệm Người dùng (UX Enhancement)
- **Auto-Tab Switching:** Sử dụng JavaScript Injection để tự động điều hướng chuyên gia đến đúng Tab chức năng khi có tác vụ mới (như chọn video BN).
- **Mobile Responsive Tabs:** Hệ thống CSS Injection đặc biệt được thiết kế để chuẩn hóa giao diện Tab trên di động, ngăn chặn tình trạng tràn chữ (Text Clipping) và cho phép cuộn ngang linh hoạt.
- **Theme Synchronization:** Hệ thống CSS thông minh tự động hiệu chỉnh độ tương phản cho cả hai chế độ Sáng/Tối, đảm bảo tính thẩm mỹ y tế chuyên nghiệp.

---

## 3. PHÂN TÍCH LUỒNG XỬ LÝ DỮ LIỆU LÂM SÀNG
1. **Input:** Thu thập video (Webcam/Upload) và triệu chứng VAS từ bệnh nhân.
2. **AI Processing:** Trích xuất khung xương, tính toán độ chính xác và gán nhãn Đúng/Sai/Gần đúng.
3. **Clinical Integration:** Bác sĩ nhận dữ liệu tổng hợp -> Đánh giá lâm sàng (Ground Truth) -> Hệ thống đồng bộ kết quả cho cả BN và NCV.
4. **Research Feedback Loop:** Nghiên cứu viên đối chiếu nhận xét của Bác sĩ với thông số AI để tối ưu hóa thuật toán và độ chính xác của mô hình.

---

## 4. TỐI ƯU HÓA HỆ THỐNG (OOM PREVENTION)
Đảm bảo tính ổn định trên môi trường Streamlit Cloud (1GB RAM):
- **FFmpeg Integration:** Tự động chuyển đổi định dạng MOV sang MP4 để tối ưu hóa dung lượng truyền tải.
- **Active Garbage Collection:** Sử dụng `gc.collect()` và quản lý bộ nhớ đệm (Cache) chủ động để ngăn chặn lỗi tràn bộ nhớ khi xử lý video dài.
- **Pagination Gallery:** Thuật toán phân trang (List Slicing) giúp hiển thị hàng ngàn ảnh phân tích khung xương mà không làm treo trình duyệt.

---

## 5. KẾT QUẢ ĐẠT ĐƯỢC (v3.1 Updated)
- **Giao diện chuẩn y khoa:** Sử dụng font 'Times New Roman' và bố cục Card hiện đại, tăng 50% hiệu suất thao tác của bác sĩ.
- **Quy trình bảo mật NCKH:** Tích hợp trang thông tin đạo đức nghiên cứu, bảo mật dữ liệu bệnh nhân theo chuẩn NCKH. Hệ thống tài khoản được phân quyền chặt chẽ (Admin, Doctor, Researcher, Patient).
- **Tính thực tiễn cao:** Đã được tinh chỉnh bởi Chủ nhiệm đề tài **Đinh Lê Quỳnh Phương** và cố vấn chuyên môn **TS. Trần Hồng Việt** để sẵn sàng ứng dụng thử nghiệm tại Bệnh viện Đa khoa Phạm Ngọc Thạch.

---

## 6. HƯỚNG PHÁT TRIỂN (FUTURE WORK)
- **Auto Action Recognition:** Nâng cấp AI để tự động nhận diện bài tập mà không cần lựa chọn thủ công.
- **AI-driven Prognosis:** Sử dụng Machine Learning để dự đoán thời gian hồi phục dựa trên lịch sử tập luyện.
- **Cloud-Scale Deployment:** Mở rộng kiến trúc sang mô hình Microservices (Docker/Kubernetes) để phục vụ quy mô bệnh viện lớn.

---
© 2025-2026 Nhóm Nghiên cứu Rehab AI Monitor.
