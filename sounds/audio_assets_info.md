# Hướng dẫn Thư mục Âm thanh Phản hồi (sounds/)

Thư mục này lưu trữ các tệp tin âm thanh giọng nói tiếng Việt (.mp3) được sử dụng để phản hồi trực tiếp (audio feedback) cho bệnh nhân trong quá trình luyện tập vật lý trị liệu phục hồi chức năng.

## Các tệp tin thành phần và tác dụng

### 1. [dung.mp3](file:///d:/Rehab-AI-Monitor-main/sounds/dung.mp3)
* **Nội dung phát âm**: "Tốt lắm, bạn đã thực hiện đúng tư thế."
* **Tác dụng**: Phát ra tín hiệu khích lệ khi bộ engine AI nhận diện góc gập khớp nằm hoàn toàn trong phạm vi đúng (hoặc bộ phân loại ML dự đoán nhãn đạt yêu cầu).

### 2. [gan_dung.mp3](file:///d:/Rehab-AI-Monitor-main/sounds/gan_dung.mp3)
* **Nội dung phát âm**: "Tư thế của bạn gần đúng, hãy cố gắng thêm một chút nữa."
* **Tác dụng**: Phát ra khi tư thế bệnh nhân gần đạt chuẩn nhưng có sai số nhỏ (Ví dụ: chưa đạt biên độ ROM tối thiểu nhưng đã vượt qua ngưỡng lỗi nghiêm trọng).

### 3. [sai.mp3](file:///d:/Rehab-AI-Monitor-main/sounds/sai.mp3)
* **Nội dung phát âm**: "Tư thế chưa đúng, vui lòng điều chỉnh lại cánh tay."
* **Tác dụng**: Phát ra để cảnh báo bệnh nhân khi tập sai tư thế (lệch góc khớp quá nhiều hoặc thực hiện chuyển động sai kỹ thuật), giúp tránh chấn thương thứ cấp.

## Vận hành hệ thống
- Các tệp tin này được nạp động trong `app.py` và phát ra bằng trình phát âm thanh HTML5 của Streamlit (`st.audio()`) khi người dùng bật chế độ "Hỗ trợ giọng nói".
- Tệp tin ẩn `.voice_tts_ok` được ghi tự động bởi tập lệnh `generate_voice_sounds.py` để đánh dấu rằng các tệp âm thanh đã được khởi tạo thành công và sẵn sàng sử dụng.
