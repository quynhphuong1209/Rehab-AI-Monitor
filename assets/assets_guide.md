# Hướng dẫn Thư mục Tài nguyên Hình ảnh (assets/)

Thư mục này lưu trữ các tài nguyên tĩnh dưới dạng hình ảnh được nhúng vào giao diện người dùng (UI/UX) của ứng dụng Rehab-AI-Monitor.

## Các tệp tin thành phần và tác dụng

### 1. [abc1.png](file:///d:/Rehab-AI-Monitor-main/assets/abc1.png)
* **Tác dụng**: Logo của trường đại học/đơn vị nghiên cứu.
* **Vận hành**: Được mã hóa sang định dạng Base64 thông qua hàm `get_school_logo_base64()` trong `app.py` và nhúng trực tiếp vào thanh bên (Sidebar) hoặc chân trang (Footer) của giao diện web Streamlit để hiển thị thông tin bản quyền và đơn vị chủ quản.

### 2. [logo_data_science_sm.png](file:///d:/Rehab-AI-Monitor-main/assets/logo_data_science_sm.png)
* **Tác dụng**: Logo của Khoa Khoa học dữ liệu (Data Science).
* **Vận hành**: Được gọi thông qua URL Github tĩnh hoặc Base64 tương ứng (`get_data_science_logo_base64()`) và hiển thị cùng với thông tin về đề tài nghiên cứu khoa học của nhóm tác giả trong phần giới thiệu ứng dụng.

## Lưu ý khi thay đổi hình ảnh
- Nếu bạn muốn cập nhật logo mới, hãy ghi đè tệp tin mới với cùng định dạng tên và phần mở rộng (`abc1.png` và `logo_data_science_sm.png`).
- Ứng dụng Streamlit sẽ tự động nạp lại hình ảnh mới sau khi bạn tải lại trang.
