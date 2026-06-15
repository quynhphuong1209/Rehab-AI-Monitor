# Tài liệu Cấu trúc Cơ sở dữ liệu JSON (database/)

Dự án sử dụng cơ sở dữ liệu dạng tệp phẳng JSON (Flat-file JSON Database) được lưu trữ thống nhất trong thư mục này để lưu trữ thông tin người dùng, kết quả phân tích AI và cấu hình hệ thống.

## Các tệp tin dữ liệu chính

### 1. [users.json](file:///d:/Rehab-AI-Monitor-main/database/users.json)
* **Tác dụng**: Lưu thông tin tài khoản người dùng (bao gồm bệnh nhân, bác sĩ, admin).
* **Vận hành**: Chứa thông tin về username, tên đầy đủ, vai trò (role), email, mã số sinh viên/bệnh nhân và mật khẩu băm bảo mật (hash).

### 2. [video_list.json](file:///d:/Rehab-AI-Monitor-main/database/video_list.json)
* **Tác dụng**: Danh sách các video tập luyện mà bệnh nhân đã tải lên kèm thông tin trạng thái phân tích.
* **Vận hành**: Chứa đường dẫn tệp video gốc, tệp video đã xử lý vẽ xương khớp, nhãn bài tập, và điểm số đánh giá chuyển động tổng quát.

### 3. [doctor_evaluations.json](file:///d:/Rehab-AI-Monitor-main/database/doctor_evaluations.json)
* **Tác dụng**: Chứa các nhận xét, đánh giá chuyên môn từ bác sĩ hoặc kỹ thuật viên phục hồi chức năng đối với từng bài tập.
* **Vận hành**: Đồng bộ thời gian thực qua giao diện bác sĩ trong ứng dụng chính.

### 4. [patient_symptoms.json](file:///d:/Rehab-AI-Monitor-main/database/patient_symptoms.json)
* **Tác dụng**: Báo cáo triệu chứng ban đầu và mức độ đau (VAS) của bệnh nhân.

### 5. [lich_su_tap_luyen.json](file:///d:/Rehab-AI-Monitor-main/database/lich_su_tap_luyen.json)
* **Tác dụng**: Lưu lại lịch sử toàn bộ các phiên tập luyện phục hồi chức năng của người bệnh theo thời gian.

### 6. Các tệp tin tư thế mẫu (Reference Poses)
* **[reference_codman.json](file:///d:/Rehab-AI-Monitor-main/database/reference_codman.json)**
* **[reference_day.json](file:///d:/Rehab-AI-Monitor-main/database/reference_day.json)**
* **[reference_gay.json](file:///d:/Rehab-AI-Monitor-main/database/reference_gay.json)**
* **Tác dụng**: Lưu dữ liệu tọa độ các khớp xương chuẩn đối với các bài tập như bài tập Codman, bài tập dây thun, bài tập gậy gỗ. Được sử dụng bởi `reference_utils.py` làm cơ sở đối chiếu góc gập.

### 7. [pose_classifier_features.json](file:///d:/Rehab-AI-Monitor-main/database/pose_classifier_features.json)
* **Tác dụng**: Đặc trưng (features) trích xuất phục vụ cho huấn luyện bộ phân loại tư thế.

### 8. [schedules.json](file:///d:/Rehab-AI-Monitor-main/database/schedules.json) & [research_data.json](file:///d:/Rehab-AI-Monitor-main/database/research_data.json)
* **Tác dụng**: Lưu trữ lịch nhắc tập luyện của bệnh nhân và cấu hình nghiên cứu thử nghiệm lâm sàng.

## Cách thức lưu trữ & Sao lưu
- Ứng dụng Python tương tác với các tệp này thông qua các thao tác đọc ghi file JSON chuẩn (`json.load` và `json.dump`).
- Việc thực thi đồng bộ lên Hugging Face Dataset hoặc sao lưu được quản lý thông qua các tập lệnh đồng bộ tự động.
