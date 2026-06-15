# Danh mục Tệp tin Gỡ lỗi và Tạm thời (debug_files/)

Thư mục này đóng vai trò như một phân vùng lưu trữ các tệp tin log, kết quả tìm kiếm font hệ thống, tệp tin trình bày phụ trợ và các bản sao lưu cơ sở dữ liệu cũ để tránh gây rác thư mục gốc của dự án.

## Chi tiết các tệp tin và nguồn gốc

### 1. Các bản sao lưu dữ liệu JSON cũ
* **[doctor_evaluations.json](file:///d:/Rehab-AI-Monitor-main/debug_files/doctor_evaluations.json)**
* **[users.json](file:///d:/Rehab-AI-Monitor-main/debug_files/users.json)**
* **[video_list.json](file:///d:/Rehab-AI-Monitor-main/debug_files/video_list.json)**
* **Tác dụng**: Đây là các bản sao dữ liệu phát sinh tại thư mục gốc từ các phiên bản chạy thử cũ hoặc các tiến trình đồng bộ dữ liệu từ Hugging Face.
* **Vận hành**: Các tệp tin này không được ứng dụng sử dụng trực tiếp nữa. Dữ liệu chạy thực tế đã được chuyển hoàn toàn vào thư mục [database/](file:///d:/Rehab-AI-Monitor-main/database/). Chúng được lưu ở đây nhằm mục đích đối chiếu lịch sử gỡ lỗi khi cần thiết.

### 2. Tệp tin HTML phụ trợ
* **[gesture_presentation.html](file:///d:/Rehab-AI-Monitor-main/debug_files/gesture_presentation.html)**
* **Tác dụng**: Slide HTML độc lập biểu diễn mô phỏng nhận diện cử chỉ bằng JavaScript.
* **Vận hành**: Có thể mở trực tiếp bằng trình duyệt để xem minh họa chuyển động mẫu độc lập với máy chủ Python.

### 3. Tệp tin text liên quan đến phông chữ (Font Search)
* **[search_font.txt](file:///d:/Rehab-AI-Monitor-main/debug_files/search_font.txt)**, **[search_font_after.txt](file:///d:/Rehab-AI-Monitor-main/debug_files/search_font_after.txt)**
* **[search_results.txt](file:///d:/Rehab-AI-Monitor-main/debug_files/search_results.txt)**, **[search_results2.txt](file:///d:/Rehab-AI-Monitor-main/debug_files/search_results2.txt)**
* **Tác dụng**: Lưu kết quả quét font chữ Tiếng Việt trên môi trường máy chủ (đặc biệt hữu ích khi cấu hình chạy trên môi trường Linux headless của Hugging Face Spaces để tránh lỗi hiển thị nhãn OpenCV).
* **Vận hành**: Sử dụng làm tài liệu tham khảo cho lập trình viên khi cần kiểm tra xem hệ thống đích có hỗ trợ font `Be Vietnam Pro` hoặc các font tương tự hay không.
