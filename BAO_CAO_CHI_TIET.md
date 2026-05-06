# BÁO CÁO CHI TIẾT MÃ NGUỒN VÀ THUẬT TOÁN 
**HỆ THỐNG GIÁM SÁT TẬP LUYỆN PHỤC HỒI CHỨC NĂNG (REHAB AI MONITOR)**

---

## 1. TỔNG QUAN HỆ THỐNG
**Rehab AI Monitor** là một hệ thống web ứng dụng Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision) để giám sát, đánh giá và chấm điểm tự động các bài tập phục hồi chức năng (PHCN) tại nhà cho bệnh nhân (ví dụ: hội chứng chóp xoay, đông cứng khớp vai). 

Hệ thống được phát triển trên ngôn ngữ **Python**, triển khai giao diện qua **Streamlit** và sử dụng lõi AI **MediaPipe** của Google để nhận diện khung xương người (Pose Estimation) theo thời gian thực hoặc qua video tải lên.

---

## 2. PHÂN TÍCH LUỒNG XỬ LÝ DỮ LIỆU (DATA PIPELINE)

Hệ thống tuân thủ một quy trình xử lý tuyến tính và khép kín:
1. **Input (Đầu vào):** Nhận video tập luyện từ bệnh nhân qua giao diện. Hỗ trợ nhiều định dạng (MP4, MOV).
2. **Pre-processing (Tiền xử lý):** Nén kích thước video (`RESIZE_WIDTH = 540`) để giảm tải tính toán; tự động chuyển đổi chuẩn mã hóa (MOV sang MP4 bằng FFmpeg).
3. **AI Processing (Trích xuất đặc trưng AI):** Đưa từng khung hình (frame) qua mạng neural MediaPipe Pose để trích xuất tọa độ 3D của 33 điểm khớp trên cơ thể.
4. **Kinematic Analysis (Phân tích động học):** Tính toán các góc khớp (góc vai, góc khuỷu) từ tọa độ AI, so sánh với biên độ chuẩn (Range of Motion - ROM) của từng bài tập do bác sĩ thiết lập.
5. **Output (Đầu ra):** Sinh ra video đã vẽ skeleton (khung xương), trích xuất hình ảnh báo lỗi, và tạo các biểu đồ thống kê bằng Plotly.

---

## 3. CHI TIẾT CÁC THUẬT TOÁN CỐT LÕI

### 3.1. Thuật toán tính góc khớp (Trigonometry)
Hàm `tinh_goc(a, b, c)` là trái tim của hệ thống phân tích. Sử dụng lượng giác học hàm `numpy.arctan2` để tính góc nội tiếp tạo bởi 3 điểm (VD: Hông - Vai - Khuỷu tay).
- **Công thức:** `radians = arctan2(c.y - b.y, c.x - b.x) - arctan2(a.y - b.y, a.x - b.x)`
- Góc được chuyển đổi từ radian sang độ (`numpy.abs(radians * 180.0 / numpy.pi)`). Đảm bảo góc luôn nằm trong khoảng 0° - 180°.

### 3.2. Thuật toán đánh giá Động tác (Rule-based Evaluation)
Hệ thống linh hoạt nhận diện tay đang tập bằng cách so sánh góc vai hai bên (tay nào có độ dịch chuyển khỏi trục cơ thể lớn hơn sẽ được ưu tiên).
- Với mỗi frame, góc thực tế được so sánh với `chuan['vai']` và `chuan['khuyu']`.
- Nếu độ lệch (absolute error) nằm trong khoảng `sai_so` (tolerance margin), frame đó được đánh giá là **PASS (Xanh)**. Ngược lại là **FAIL (Đỏ)**.
- Hàm `get_warning_message` sẽ cụ thể hóa lỗi (VD: "Tay quá thấp", "Khuỷu tay quá cong").

---

## 4. CƠ CHẾ TỐI ƯU HÓA TÀI NGUYÊN BỘ NHỚ (OOM PREVENTION)
Do hệ thống được triển khai trên Streamlit Cloud (giới hạn 1GB RAM), việc xử lý video hàng ngàn khung hình sinh ra lượng rác bộ nhớ khổng lồ. Mã nguồn đã được áp dụng các kỹ thuật tối ưu cấp cao:

1. **Chunked Streaming Upload:** Không dùng `getvalue()` đọc toàn bộ video vào RAM. Thay vào đó, video được đọc từng block 1MB (`read(1024*1024)`) và ghi tuần tự xuống ổ cứng mây.
2. **Quản lý Vòng đời Đối tượng C++ (Memory Deallocation):** Các biến `rgb` và kết quả `ket_qua` (chứa con trỏ C++ của MediaPipe/OpenCV) được ép xóa thủ công bằng lệnh `del` ngay trong vòng lặp khung hình, tránh hiện tượng rò rỉ bộ nhớ (Memory Leak).
3. **Thu gom rác chủ động:** Gọi `gc.collect()` mỗi 100 khung hình.
4. **Hủy bỏ mã hóa Base64:** Chuyển hoàn toàn giao diện bộ sưu tập (Gallery) sang việc load ảnh trực tiếp từ đường dẫn đĩa (Disk Path) bằng `st.image()`, giảm thiểu gánh nặng lưu trữ string trong `st.session_state`.
5. **Loại bỏ File ZIP ẩn:** Hủy việc nén ảnh liên tục trên ổ cứng tạm (thường được map vào RAM/tmpfs trên cloud) giúp giải phóng ~300MB tài nguyên tức thời.

---

## 5. THIẾT KẾ GIAO DIỆN VÀ TRẢI NGHIỆM NGƯỜI DÙNG (UI/UX)

Hệ thống cung cấp một bảng điều khiển (Dashboard) y tế chuyên nghiệp chia thành 6 Tab:
- **Tùy biến CSS:** Ứng dụng Dark Theme (Giao diện tối) chuẩn y tế kết hợp hiệu ứng gradient, viền nổi và bóng đổ.
- **Biểu đồ Plotly:** Tích hợp các biểu đồ phân phối (Histogram) và biểu đồ hộp (Boxplot) tương tác, giúp bác sĩ nhận diện sự phân tán trong biên độ dao động tay của bệnh nhân.
- **Trải nghiệm phân trang (Pagination):** Hỗ trợ duyệt qua hàng ngàn tấm ảnh phân tích lỗi mà không làm "treo" trình duyệt nhờ thuật toán cắt lát danh sách (List Slicing) trực tiếp bằng Python.
- **Xuất báo cáo:** Chức năng xuất dữ liệu dưới dạng tệp `CSV` (phục vụ nghiên cứu thống kê bằng SPSS/R) và lưu toàn bộ biểu đồ bằng thư viện `kaleido`.

---

## 6. HƯỚNG PHÁT TRIỂN & MỞ RỘNG (FUTURE WORK)
- **Tích hợp Camera Real-time:** Bổ sung tính năng WebRTC để giám sát trực tiếp qua webcam thay vì chỉ upload video.
- **Nhận diện tự động bài tập (Action Recognition):** Dùng mô hình LSTM hoặc TimeSformer để hệ thống tự động biết bệnh nhân đang tập bài gì mà không cần chọn tay.
- **Xử lý bất đồng bộ (Celery/Redis):** Tách tác vụ phân tích video khỏi tiến trình chính của Streamlit để phục vụ nhiều bệnh nhân cùng lúc mà không sợ nghẽn cổ chai.
