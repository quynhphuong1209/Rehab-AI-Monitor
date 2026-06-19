# Đánh Giá Dự Án Rehab AI Monitor

## Tổng Quan
Dự án này là một prototype nghiên cứu khá công phu, có giá trị trình diễn và nghiên cứu rõ ràng: từ bệnh nhân upload video, AI phân tích tư thế, bác sĩ đánh giá, NCV xuất số liệu, đến admin quản trị.

Tuy nhiên, nếu đánh giá theo chuẩn sản phẩm dùng thật trong y tế, dự án chưa sẵn sàng production. Điểm nghẽn lớn nhất nằm ở bảo mật, riêng tư dữ liệu, kiến trúc monolith và độ tin cậy vận hành.

## Điểm Mạnh
- Luồng nghiệp vụ đầy đủ theo vai trò: bệnh nhân, bác sĩ/KTV, nghiên cứu viên, quản trị viên.
- AI/CV có nền tảng hợp lý: MediaPipe Pose, tính góc vai/khuỷu, so khớp reference, chia 3 giai đoạn PHCN.
- Có nhiều cơ chế thực dụng cho HF Spaces: transcode H.264, cache, checkpoint, resume job, sync Hugging Face Dataset.
- Bộ tài liệu dày và có chiều sâu nghiên cứu.
- Có script kiểm định và xuất số liệu nghiên cứu, giúp báo cáo khoa học có tính đối soát.

## Điểm Yếu Chính
- `app.py` quá lớn, ôm gần như toàn bộ frontend, backend, AI, auth, sync và video processing.
- Cơ sở dữ liệu JSON tiện cho demo nhưng yếu cho dữ liệu y tế thật: thiếu schema, migration, transaction và audit.
- Bảo mật là rủi ro lớn: login qua query params, tài khoản mặc định, hash SHA-256 đơn giản, nhiều HTML `unsafe_allow_html=True`, token HF có nguy cơ lộ.
- Repo có dấu hiệu chứa dữ liệu định danh bệnh nhân/người dùng.
- Upload video cho phép dung lượng rất lớn, dễ gây OOM hoặc treo app.
- Chưa thấy test tự động đáng kể cho các luồng quan trọng.

## Đánh Giá Theo Khía Cạnh
- Ý tưởng và giá trị ứng dụng: 8/10
- Mức hoàn thiện prototype: 8/10
- Kiến trúc kỹ thuật: 5/10
- Bảo mật và quyền riêng tư: 2/10
- Khả năng bảo trì: 4/10
- Giá trị nghiên cứu: 7/10

## Kết Luận
Dự án tốt ở vai trò mô hình thử nghiệm nghiên cứu: nhiều công sức, có luồng end-to-end, có ý nghĩa lâm sàng và có khả năng trình diễn mạnh.

Để thành hệ thống dùng thật, nên ưu tiên:
1. Khóa bảo mật và quyền riêng tư.
2. Tách `app.py` thành các module nhỏ.
3. Thêm test cho metrics, auth, role access và JSON read/write.
4. Chuyển dữ liệu runtime sang storage an toàn hơn hoặc ít nhất có schema, lock, backup, audit.
5. Chuẩn hóa pipeline nghiên cứu để tách bạch dữ liệu demo, dữ liệu thật và dữ liệu đã ẩn danh.
