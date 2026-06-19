# CẤU TRÚC VÀ DỮ LIỆU ĐÁNH GIÁ CỦA BÁC SĨ / KTV PHCN (PORTAL CLINICAL & RESEARCH)

Tài liệu này chi tiết hóa cách thức truy cập, các trường thông tin trong biểu mẫu đánh giá, cấu trúc lưu trữ dữ liệu dưới cơ sở dữ liệu (JSON) và số liệu lâm sàng hiện tại của Bác sĩ / KTV PHCN trên hệ thống **Rehab-AI-Monitor**.

---

## 1. HƯỚNG DẪN ĐĂNG NHẬP VÀ VAI TRÒ BÁC SĨ / KTV PHCN

### 1.1. Tài khoản truy cập chuyên gia
Tài khoản chuyên gia y tế phải được Quản trị viên cấp trong hệ thống. Mật khẩu tạm thời không được ghi trong tài liệu hoặc commit vào repository.
*   **Tên đăng nhập (Username):** do Quản trị viên cấp.
*   **Mật khẩu (Password):** cấp ngoài repository, bắt buộc đổi ở lần đăng nhập đầu tiên nếu là mật khẩu tạm.
*   **Vai trò hiển thị (Role):** Bác sĩ / KTV PHCN.

### 1.2. Cách thức chọn bệnh nhân và video
1.  Bác sĩ/KTV đăng nhập bằng tài khoản đã được Quản trị viên cấp.
2.  Tại tab **🏠 TRANG CHỦ**, danh sách bệnh nhân và các video bài tập đã nộp sẽ hiển thị.
3.  Bác sĩ chọn bệnh nhân và click vào video mong muốn đánh giá. Video này sẽ được tải vào bộ nhớ tạm của phiên làm việc (`st.session_state.current_eval_video`).
4.  Bác sĩ chuyển sang tab **📊 QUẢN LÝ ĐÁNH GIÁ & NCKH** để thực hiện ghi nhận thông tin.

---

## 2. CHI TIẾT SUBTAB: 📝 ĐÁNH GIÁ PHCN (GROUND TRUTH)

Giao diện này dùng để Bác sĩ/KTV PHCN ghi nhận nhanh chẩn đoán kỹ thuật động tác làm cơ sở đối chiếu (Ground Truth) với mô hình AI.

### 2.1. Các trường thông tin trên giao diện (Streamlit Form)
| STT | Tên trường trên giao diện | Loại điều khiển | Giá trị lựa chọn | Ý nghĩa lâm sàng |
| :---: | :--- | :---: | :--- | :--- |
| 1 | **Kết quả** | Radio | `Đúng`, `Gần đúng`, `Sai` | Đánh giá tổng quan mức độ hoàn thành động tác |
| 2 | **Lỗi sai** | Multiselect | `Vị trí tay chưa đúng`, `Biên độ chưa đạt`, `Tốc độ quá nhanh/chậm`, `Sai tư thế thân người` | Phân loại lỗi kỹ thuật trong quá trình tập |
| 3 | **Nhận xét cho BN** | Text Area | Nhập tự do | Lời khuyên trực tiếp gửi cho bệnh nhân |
| 4 | **Ghi chú cho NCV** | Text Area | Nhập tự do | Góp ý chuyên môn gửi cho nhóm Nghiên cứu viên |
| 5 | **Chỉ định** | Radio | `Tiếp tục`, `Chuyển bài`, `Khám lại` | Phác đồ hướng dẫn tiếp theo cho bệnh nhân |

### 2.2. Cấu trúc lưu trữ trong cơ sở dữ liệu (`database/doctor_evaluations.json`)
Mỗi lần Bác sĩ gửi đánh giá, một bản ghi sẽ được lưu vào file JSON này với cấu trúc:
```json
{
    "patient_username": "Tên tài khoản bệnh nhân",
    "doctor_username": "doctor1",
    "doctor_name": "Doctor 1",
    "video_name": "Tên_file_video_da_gui.mp4",
    "exercise": "Tên bài tập",
    "doctor_result": "Đúng / Gần đúng / Sai",
    "errors": [
        "Vị trí tay chưa đúng",
        "Biên độ chưa đạt"
    ],
    "comments": "Nhận xét cho bệnh nhân...",
    "comments_ncv": "Ghi chú cho nghiên cứu viên...",
    "plan": "Tiếp tục / Chuyển bài / Khám lại",
    "time": "YYYY-MM-DD HH:MM:SS"
}
```

---

## 3. CHI TIẾT SUBTAB: 📄 PHIẾU NCKH (BỘ CÔNG CỤ THU THẬP DỮ LIỆU ĐỀ TÀI)

Biểu mẫu này được thiết kế theo cấu trúc Đề tài Nghiên cứu Khoa học để thu thập dịch tễ, bệnh lý lâm sàng và dữ liệu video đồng bộ phục vụ xây dựng mô hình AI.

### 3.1. Các phần và trường thông tin chi tiết

#### I. THÔNG TIN CHUNG VÀ ĐẶC ĐIỂM LÂM SÀNG
*   **Họ và tên người phỏng vấn (`interviewer`):** Text input (Mặc định tự động điền tên đầy đủ của Bác sĩ đang đăng nhập).
*   **Ngày phỏng vấn (`interview_date`):** Date input (Mặc định ngày hiện tại).
*   **Mã đối tượng (Mã BN) (`subject_code`):** Text input (Tự động điền mã số bệnh nhân từ hệ thống).
*   **Tuổi (`age`):** Number input (Tự động lấy từ hồ sơ bệnh nhân).
*   **Giới tính (`gender`):** Radio `Nam (1)`, `Nữ (2)` (Tự động đồng bộ từ triệu chứng).
*   **Khu vực (`region`):** Radio `Nội thành (1)`, `Ngoại thành (2)`.
*   **Nghề nghiệp (`job`):** Selectbox:
    1.  *Nông dân (1)*
    2.  *Công nhân (2)*
    3.  *Cán bộ - viên chức (3)*
    4.  *Buôn bán (4)*
    5.  *Nội trợ (5)*
    6.  *Lao động tự do (6)*
    7.  *Nghỉ hưu (7)*
    8.  *Không có nghề nghiệp cụ thể (8)*
*   **Trình độ học vấn (`education`):** Selectbox:
    1.  *Mù chữ (1)*
    2.  *Tiểu học (2)*
    3.  *Trung học cơ sở (3)*
    4.  *Trung học phổ thông (4)*
    5.  *Cao đẳng – đại học (5)*
    6.  *Không rõ (6)*
*   **Khoa điều trị (`department`):** Radio `Khoa PHCN – Y học cổ truyền (1)`, `Khác (99)`.
*   **Hình thức điều trị (`treatment_type`):** Radio `Nội trú (1)`, `Ngoại trú (2)`.
*   **Chẩn đoán bệnh học (`diagnosis`):** Radio (mã hóa chuẩn ICD-10):
    *   *Viêm quanh khớp vai thể giả liệt (ICD-10: M75.1)*
    *   *Viêm quanh khớp vai thể đông cứng (ICD-10: M75.0)*
    *   *Viêm quanh khớp vai thể đơn thuần (ICD-10: M75.8)*
    *   *Viêm quanh khớp cấp (ICD-10: M75.3 / M75.5)*
    *   *Viêm quanh khớp vai (P) (ICD-10: M75)*
*   **Vị trí vai tổn thương (`lesion_side`):** Radio `Vai trái (1)`, `Vai phải (2)`, `Cả hai vai (3)`.
*   **Thời gian mắc bệnh (`duration`):** Radio `< 1 tháng (1)`, `1 – 3 tháng (2)`, `\>= 3 tháng (3)`.

#### II. THÔNG TIN PHỤC HỒI
*   **Bên tập luyện (`training_side`):** Radio `Vai trái`, `Vai phải`, `Cả hai vai`.
*   **Mức độ đau (VAS 0–10) (`pain_level`):** Radio `Nhẹ (0–3)`, `Trung bình (4–6)`, `Nặng (7–10)` (Tự động ánh xạ từ điểm đau VAS do bệnh nhân tự khai báo).
*   **Mức độ bệnh (`disease_severity`):** Radio `Nhẹ`, `Trung bình`, `Nặng`.

#### III. NỘI DUNG TẬP LUYỆN ĐƯỢC GHI HÌNH
*   **Bài tập được ghi hình (`exercises`):** Văn bản hiển thị tự động lấy từ video đang chọn (ví dụ: `Bài tập con lắc Codman` hoặc `Bài tập với gậy (Pulley Exercise)`).

#### IV. ĐÁNH GIÁ KỸ THUẬT ĐỘNG TÁC (GROUND TRUTH)
*(Tự động kế thừa dữ liệu nếu bác sĩ đã điền ở tab Đánh giá PHCN để tránh nhập liệu 2 lần)*
*   **Kết quả (`general_result`):** Radio `Đúng`, `Sai`, `Gần đúng`.
*   **Chỉ định (`plan`):** Radio `Tiếp tục`, `Chuyển bài`, `Khám lại`.
*   **Lỗi sai (`errors`):** Multiselect lỗi kỹ thuật động tác.
*   **Nhận xét chuyên môn của Bác sĩ/KTV PHCN (`specialist_comment`):** Text area.

#### V. THÔNG TIN DỮ LIỆU VIDEO
*   **Mã video (`video_code`):** Text input (Lấy theo tên file video được tải lên).
*   **Thiết bị ghi hình (`recording_device`):** Radio `Điện thoại (1)`, `Webcam (2)`, `Khác (3)`.
*   **Góc quay (`recording_angle`):** Radio `Chính diện (1)`, `Bên trái (2)`, `Bên phải (3)`.
*   **Khoảng cách camera (m) (`camera_distance`):** Text input (Mặc định: `2.5` mét).

#### VI. XÁC NHẬN
*   **Hộp kiểm xác nhận (`confirm`):** Bác sĩ tích chọn "Tôi xác nhận các thông tin trên là chính xác và phục vụ cho mục đích nghiên cứu khoa học."

### 3.2. Cấu trúc lưu trữ dữ liệu NCKH (`database/research_data.json`)
Khi nhấn gửi phiếu NCKH, hệ thống ghi nhận thông tin dưới dạng đối tượng JSON đầy đủ:
```json
{
    "interviewer": "Tên người phỏng vấn",
    "interview_date": "YYYY-MM-DD",
    "subject_code": "Mã bệnh nhân",
    "age": 39,
    "gender": "Nam (1) / Nữ (2)",
    "region": "Nội thành (1) / Ngoại thành (2)",
    "job": "Nghề nghiệp (Mã)",
    "education": "Học vấn (Mã)",
    "department": "Khoa (Mã)",
    "treatment_type": "Hình thức (Mã)",
    "diagnosis": "Chẩn đoán bệnh học (Mã ICD-10)",
    "lesion_side": "Vị trí vai tổn thương (Mã)",
    "duration": "Thời gian mắc bệnh (Mã)",
    "training_side": "Bên tập luyện",
    "pain_level": "Nhẹ (0–3) / Trung bình (4–6) / Nặng (7–10)",
    "disease_severity": "Nhẹ / Trung bình / Nặng",
    "exercises": ["Tên bài tập"],
    "general_result": "Đúng / Sai / Gần đúng",
    "errors": ["Lỗi kỹ thuật"],
    "plan": "Tiếp tục / Chuyển bài / Khám lại",
    "specialist_comment": "Nhận xét chuyên môn...",
    "video_code": "Tên_video.mp4",
    "recording_device": "Điện thoại (1) / Webcam (2) / Khác (3)",
    "recording_angle": "Chính diện (1) / Bên trái (2) / Bên phải (3)",
    "camera_distance": "2.5",
    "patient_username": "Tên tài khoản bệnh nhân",
    "submitted_by": "doctor1",
    "role": "Bác sĩ / KTV PHCN",
    "timestamp": "YYYY-MM-DD HH:MM:SS"
}
```

---

## 4. TỔNG HỢP SỐ LIỆU ĐÁNH GIÁ LÂM SÀNG HIỆN TẠI TRONG CƠ SỞ DỮ LIỆU

### 4.1. Thống kê tổng quan bệnh nhân (Dữ liệu triệu chứng)
Cơ sở dữ liệu triệu chứng bệnh nhân (`database/patient_symptoms.json`) là dữ liệu runtime nhạy cảm và không được đưa vào tài liệu công khai. Khi cần minh họa, chỉ dùng fixture đã ẩn danh:

| Mã đối tượng | Tuổi | Giới tính | Điểm đau (VAS) | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| `SUBJ-0001` | Nhóm tuổi | Ẩn danh | Khoảng điểm | Mô tả lâm sàng đã tổng quát hóa |
| `SUBJ-0002` | Nhóm tuổi | Ẩn danh | Khoảng điểm | Mô tả lâm sàng đã tổng quát hóa |

### 4.2. Trạng thái các bản ghi đánh giá lâm sàng hiện tại
*   **Số lượng bản ghi gợi ý của AI/NCV:** **26 bản ghi** gợi ý tự động đã được lập trong `database/doctor_evaluations.json` cho 4 bệnh nhân trên (phục vụ đối sánh).
*   **Số lượng bản ghi đánh giá độc lập của Bác sĩ / KTV:** Phụ thuộc dữ liệu runtime nội bộ. Không ghi username chuyên gia thật hoặc mật khẩu mẫu trong tài liệu công khai.
*   **Bản ghi phiếu NCKH (`database/research_data.json`):** Hiện tại đang là danh sách rỗng (`[]`), chờ dữ liệu khảo sát lâm sàng thực tế được Bác sĩ/KTV điền khi tiến hành phỏng vấn bệnh nhân.
