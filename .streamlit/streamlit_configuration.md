# Cấu hình Ứng dụng Streamlit (.streamlit/)

Thư mục này chứa các tệp tin cấu hình hệ thống và cấu hình bảo mật dành riêng cho ứng dụng Streamlit (Web framework chính của Rehab-AI-Monitor).

## Chi tiết các tệp tin và chức năng

### 1. [config.toml](file:///d:/Rehab-AI-Monitor-main/.streamlit/config.toml)
* **Tác dụng**: Cấu hình hoạt động của máy chủ Streamlit và tùy biến giao diện web.
* **Tham số quan trọng**:
  - Giao diện chủ đề (theme): Tông màu chủ đạo (primary color), màu nền (background color), và phông chữ hiển thị.
  - Cấu hình máy chủ: Giới hạn dung lượng tải lên tối đa (`maxUploadSize` thường đặt lớn để tải được các video HD của bệnh nhân).
  - Bật/tắt chế độ giám sát tự động tải lại file khi code thay đổi (`runOnSave`).

### 2. [secrets.toml](file:///d:/Rehab-AI-Monitor-main/.streamlit/secrets.toml)
* **Tác dụng**: Lưu trữ các biến môi trường và thông tin bảo mật nhạy cảm (Credentials) ở môi trường local.
* **Tham số quan trọng**:
  - `HF_TOKEN`: Mã thông báo (Access token) dùng để xác thực quyền ghi dữ liệu lên Hugging Face Dataset của nhóm nghiên cứu.
* **Lưu ý bảo mật**: Tệp tin này chứa khóa bí mật, tuyệt đối **không được đẩy lên Git công khai** (đã được cấu hình chặn trong tệp `.gitignore`).

## Cách thiết lập
Khi deploy lên Hugging Face Spaces, nội dung của `secrets.toml` sẽ được khai báo thông qua phần "Repository Secrets" trong bảng cài đặt của Space thay vì đọc file trực tiếp.
