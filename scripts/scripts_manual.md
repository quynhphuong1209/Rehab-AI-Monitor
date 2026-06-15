# Sổ tay Vận hành các Tập lệnh Hệ thống (scripts/)

Thư mục này chứa các tập lệnh tự động hóa phục vụ cho việc huấn luyện mô hình, kiểm tra dữ liệu, trích xuất báo cáo kết quả và đồng bộ dữ liệu với máy chủ Hugging Face.

## Chi tiết các tập lệnh và cách vận hành

### 1. Đồng bộ và Quản lý dữ liệu
* **[sync_from_hf.py](file:///d:/Rehab-AI-Monitor-main/scripts/sync_from_hf.py)**
  - *Tác dụng*: Đồng bộ dữ liệu bệnh nhân và đánh giá của bác sĩ từ Hugging Face Dataset về local.
  - *Cách chạy*: `python scripts/sync_from_hf.py --token <HF_TOKEN>`
* **[sync_data_and_report.py](file:///d:/Rehab-AI-Monitor-main/scripts/sync_data_and_report.py)**
  - *Tác dụng*: Tự động kéo dữ liệu mới, chạy phân tích và cập nhật các bảng báo cáo khoa học.
* **[reset_data.py](file:///d:/Rehab-AI-Monitor-main/scripts/reset_data.py)**
  - *Tác dụng*: Khôi phục dữ liệu hệ thống về trạng thái ban đầu của nhà phát triển (xóa dữ liệu tạm, lịch sử kiểm thử).

### 2. Huấn luyện Mô hình Học máy (AI Pipeline)
* **[train_classifier.py](file:///d:/Rehab-AI-Monitor-main/scripts/train_classifier.py)**
  - *Tác dụng*: Huấn luyện mô hình bộ phân loại tư thế (pose classifier) dựa trên các đặc trưng khớp.
* **[run_ml_pipeline.py](file:///d:/Rehab-AI-Monitor-main/scripts/run_ml_pipeline.py)**
  - *Tác dụng*: Chạy đường ống huấn luyện và lưu mô hình trực tiếp từ đầu đến cuối.
* **[reprocess_all.py](file:///d:/Rehab-AI-Monitor-main/scripts/reprocess_all.py)**
  - *Tác dụng*: Chạy lại toàn bộ video trong cơ sở dữ liệu để cập nhật nhãn phân loại mới sau khi cập nhật mô hình ML.

### 3. Trích xuất và Kiểm định số liệu báo cáo
* **[export_report_metrics.py](file:///d:/Rehab-AI-Monitor-main/scripts/export_report_metrics.py)** & **[export_rom_charts_data.py](file:///d:/Rehab-AI-Monitor-main/scripts/export_rom_charts_data.py)**
  - *Tác dụng*: Trích xuất bảng số liệu đo đạc biên độ khớp vai, khuỷu tay và thời gian thực hiện pha của các bài tập phục vụ nghiên cứu lâm sàng.
* **[verify_report_numbers.py](file:///d:/Rehab-AI-Monitor-main/scripts/verify_report_numbers.py)**
  - *Tác dụng*: Tự động xác thực chéo các con số thống kê trong báo cáo đảm bảo tính khoa học chính xác 100%.

### 4. Trích xuất dữ liệu YouTube Chuẩn
* **[extract_youtube_reference.py](file:///d:/Rehab-AI-Monitor-main/scripts/extract_youtube_reference.py)**
  - *Tác dụng*: Tải và phân tích tọa độ khớp từ các video bài tập vật lý trị liệu chuẩn trên Youtube để sinh file JSON dữ liệu tham chiếu bài tập.

### 5. Tạo âm thanh giọng nói và Sửa lỗi thư viện
* **[generate_voice_sounds.py](file:///d:/Rehab-AI-Monitor-main/scripts/generate_voice_sounds.py)**
  - *Tác dụng*: Sử dụng Google Text-to-Speech sinh các file âm thanh giọng nói hướng dẫn tập luyện tiếng Việt.
* **[fix_plotly_v2.py](file:///d:/Rehab-AI-Monitor-main/scripts/fix_plotly_v2.py)**
  - *Tác dụng*: Bản vá sửa lỗi tương thích biểu đồ Plotly trên giao diện Web Streamlit.

### 6. File thực thi đẩy mã nguồn lên Git
* **[push_code.bat](file:///d:/Rehab-AI-Monitor-main/scripts/push_code.bat)** & **[push_to_git.bat](file:///d:/Rehab-AI-Monitor-main/scripts/push_to_git.bat)**
  - *Tác dụng*: Kịch bản batch chạy trên Windows CMD để tự động đẩy mã nguồn lên Github và Hugging Face.
