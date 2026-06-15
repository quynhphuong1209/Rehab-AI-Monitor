# Kiến trúc Thư mục Tiện ích (utils/)

Thư mục này chứa các mô-đun Python hỗ trợ đắc lực cho hoạt động tính toán, xử lý và lưu trữ dữ liệu tư thế của ứng dụng Rehab-AI-Monitor.

## Các tệp tin thành phần và tác dụng

### 1. [checkpoint_utils.py](file:///d:/Rehab-AI-Monitor-main/utils/checkpoint_utils.py)
* **Tác dụng**: Quản lý việc lưu trữ và khôi phục trạng thái xử lý video (checkpoint). Khi người dùng phân tích các video dài, nếu có sự cố xảy ra hoặc quá trình xử lý bị ngắt quãng, module này đảm bảo tiến trình có thể tiếp tục từ vị trí đã dừng mà không phải chạy lại từ đầu.
* **Vận hành**:
  - Tự động serialization trạng thái của từng khung hình (frame) bằng JSON/pickle.
  - Sử dụng hàm `save_checkpoint()` định kỳ để lưu trữ và `load_checkpoint()` khi khởi động hoặc tiếp tục tiến trình.

### 2. [pose_classifier_utils.py](file:///d:/Rehab-AI-Monitor-main/utils/pose_classifier_utils.py)
* **Tác dụng**: Chứa các thuật toán phân loại tư thế sử dụng Machine Learning (ML) và hệ chuyên gia dựa trên luật (Rule-based). Đây là hạt nhân phân tích chuyển động để đánh giá xem bệnh nhân tập đúng hay sai.
* **Vận hành**:
  - Trích xuất đặc trưng từ tọa độ các khớp xương của MediaPipe.
  - Huấn luyện mô hình phân loại tư thế thông qua hàm `train_pose_classifier()`.
  - Dự đoán nhãn chuyển động thời gian thực bằng `create_pose_classifier_predictor()`.

### 3. [pose_classfier_untils.py](file:///d:/Rehab-AI-Monitor-main/utils/pose_classfier_untils.py)
* **Tác dụng**: Tệp tin bao bọc tương thích (compatibility wrapper). Do trong phiên bản cũ có sự nhầm lẫn chính tả ở tên tệp (`untils` thay vì `utils`), tệp này được giữ lại để tránh làm hỏng các lệnh gọi từ các đoạn script cũ.
* **Vận hành**: Import toàn bộ nội dung từ `pose_classifier_utils.py` để chuyển tiếp lời gọi hàm.

### 4. [reference_utils.py](file:///d:/Rehab-AI-Monitor-main/utils/reference_utils.py)
* **Tác dụng**: Cung cấp các hàm so sánh tư thế của người tập với các tư thế chuẩn (reference poses) được lưu trong hệ thống.
* **Vận hành**:
  - Tính toán khoảng cách tọa độ, góc gập các khớp.
  - Xác định pha tập luyện (Ví dụ: pha chuẩn bị, pha thực hiện, pha kết thúc) bằng hàm `detect_motion_subtype()`.
  - Xác định tư thế chuẩn gần nhất bằng hàm `find_closest_reference_pose()`.

## Cách thức tích hợp
Thư mục này được tự động thêm vào đường dẫn hệ thống (`sys.path`) ở dòng đầu tiên của `app.py`. Do đó, ứng dụng chính có thể gọi trực tiếp:
```python
from pose_classifier_utils import ...
```
mà không cần thay đổi cấu trúc mã nguồn phức tạp.
