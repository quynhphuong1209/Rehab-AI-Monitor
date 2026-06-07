# BÁO CÁO KẾT QUẢ AI VÀ ĐÁNH GIÁ CỦA CHUYÊN GIA (CLINICAL & AI RESEARCH DATA)
Tài liệu này tổng hợp toàn bộ dữ liệu chạy thử nghiệm của mô hình AI (Nghiên cứu viên) và các đánh giá lâm sàng từ Chuyên gia (Bác sĩ/Kỹ thuật viên) tại Bệnh viện Đa khoa Phạm Ngọc Thạch. Dữ liệu này dùng để đưa vào mô hình ngôn ngữ lớn (như Claude hoặc GPT) để làm báo cáo tóm tắt đề tài.

---

## 👤 BỆNH NHÂN: Hoàng Hạnh Nguyên
Tổng số video bài tập: 2

### 🎬 Bài tập 1: Bài tập con lắc Codman
- **Tên video file:** `Hoàng Hạnh Nguyên - Codman.mp4`
- **Thời gian tập:** 10:53 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 97.6% | 12.87° | 0.72 | 0.98 | 503 |
| GĐ 2: Hồi phục (30°) | 94.8% | 10.78° | 0.76 | 0.95 | 674 |
| GĐ 3: Chuẩn xác (15°) | 41.9% | 23.84° | 0.50 | 0.49 | 497 |

#### 🩺 Đánh giá từ Chuyên gia lâm sàng (Ground Truth):
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 19:20 - 02/06/2026)
- **Kết quả đánh giá lâm sàng:** `Gần đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 83.2% | Đúng: 738/887 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 67.8% | Đúng: 892/1315 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 23.2% | Đúng: 266/1146 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 2
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 83.2% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 67.8% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 23.2% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 11:35 - 03/06/2026)
- **Kết quả đánh giá lâm sàng:** `Gần đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 83.2% | Đúng: 738/887 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 67.8% | Đúng: 892/1315 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 23.2% | Đúng: 266/1146 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 2
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 83.2% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 67.8% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 23.2% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 10:54 - 04/06/2026)
- **Kết quả đánh giá lâm sàng:** `Đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 97.6% | Đúng: 491/503 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 94.8% | Đúng: 639/674 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 41.9% | Đúng: 205/489 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 3
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 97.6% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 94.8% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ3 (Sai số 15°): Đạt 41.9% - Khớp còn cứng hoặc lệch biên độ.

---

### 🎬 Bài tập 2: Bài tập với gậy (Pulley Exercise)
- **Tên video file:** `Hoàng Hạnh Nguyên - Bài tập với gậy.mp4`
- **Thời gian tập:** 14:49 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 48.3% | 33.66° | 0.50 | 0.55 | 11774 |
| GĐ 2: Hồi phục (30°) | 48.3% | 33.66° | 0.50 | 0.55 | 11774 |
| GĐ 3: Chuẩn xác (15°) | 48.3% | 33.66° | 0.50 | 0.55 | 11774 |

#### 🩺 Đánh giá từ Chuyên gia lâm sàng (Ground Truth):
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 23:57 - 02/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 26.2% | Đúng: 846/3229 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 19.5% | Đúng: 885/4539 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 3.2% | Đúng: 129/4006 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 1
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 26.2% - Cần rèn luyện thêm.
  > - GĐ2 (Sai số 30°): Đạt 19.5% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 3.2% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 23:58 - 02/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 26.2% | Đúng: 846/3229 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 19.5% | Đúng: 885/4539 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 3.2% | Đúng: 129/4006 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 1
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 26.2% - Cần rèn luyện thêm.
  > - GĐ2 (Sai số 30°): Đạt 19.5% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 3.2% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 23:50 - 05/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO PHÂN TÍCH BÀI TẬP VỚI GẬY (TỔNG QUAN):
  > 🏒 Độ chính xác: 48.3% | Đúng: 749/1550 frames
  > 🤖 AI đề xuất: Cần chuyên gia y tế hướng dẫn.
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - Bài tập với gậy: Đạt 48.3% - Cần rèn luyện thêm để giảm sai số.

---

## 👤 BỆNH NHÂN: Nguyễn Thị Nga
Tổng số video bài tập: 2

### 🎬 Bài tập 1: Bài tập con lắc Codman
- **Tên video file:** `Nguyễn Thị Nga - Codman.mp4`
- **Thời gian tập:** 12:07 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 96.5% | 12.90° | 0.72 | 0.97 | 738 |
| GĐ 2: Hồi phục (30°) | 92.6% | 9.06° | 0.80 | 0.94 | 1031 |
| GĐ 3: Chuẩn xác (15°) | 48.5% | 13.69° | 0.71 | 0.55 | 859 |

#### 🩺 Đánh giá từ Chuyên gia lâm sàng (Ground Truth):
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 00:20 - 03/06/2026)
- **Kết quả đánh giá lâm sàng:** `Gần đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 78.7% | Đúng: 581/738 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 64.0% | Đúng: 701/1096 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 28.8% | Đúng: 229/794 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 2
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 78.7% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 64.0% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 28.8% - Khớp còn cứng hoặc lệch biên độ.

---

### 🎬 Bài tập 2: Bài tập với gậy (Pulley Exercise)
- **Tên video file:** `Nguyễn Thị Nga - Bài tập với gậy.mp4`
- **Thời gian tập:** 13:37 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 32.7% | 27.39° | 0.50 | 0.41 | 4535 |
| GĐ 2: Hồi phục (30°) | 32.7% | 27.39° | 0.50 | 0.41 | 4535 |
| GĐ 3: Chuẩn xác (15°) | 32.7% | 27.39° | 0.50 | 0.41 | 4535 |

#### 🩺 Đánh giá từ Chuyên gia lâm sàng (Ground Truth):
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 13:38 - 04/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO PHÂN TÍCH BÀI TẬP VỚI GẬY (TỔNG QUAN):
  > 🏒 Độ chính xác: 32.7% | Đúng: 1485/4535 frames
  > 🤖 AI đề xuất: Cần chuyên gia y tế hướng dẫn.
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - Bài tập với gậy: Đạt 32.7% - Cần rèn luyện thêm để giảm sai số.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 13:39 - 04/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO PHÂN TÍCH BÀI TẬP VỚI GẬY (TỔNG QUAN):
  > 🏒 Độ chính xác: 32.7% | Đúng: 1485/4535 frames
  > 🤖 AI đề xuất: Cần chuyên gia y tế hướng dẫn.
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - Bài tập với gậy: Đạt 32.7% - Cần rèn luyện thêm để giảm sai số.

---

## 👤 BỆNH NHÂN: Vũ Thị Hòa
Tổng số video bài tập: 2

### 🎬 Bài tập 1: Bài tập con lắc Codman
- **Tên video file:** `Vũ Thị Hoà - Codman.mp4`
- **Thời gian tập:** 12:22 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 99.9% | 11.40° | 0.75 | 0.99 | 797 |
| GĐ 2: Hồi phục (30°) | 100.0% | 10.90° | 0.76 | 0.99 | 1135 |
| GĐ 3: Chuẩn xác (15°) | 31.6% | 13.49° | 0.71 | 0.40 | 808 |

#### 🩺 Đánh giá từ Chuyên gia lâm sàng (Ground Truth):
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 00:59 - 03/06/2026)
- **Kết quả đánh giá lâm sàng:** `Gần đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 82.3% | Đúng: 656/797 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 67.0% | Đúng: 760/1135 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 14.0% | Đúng: 113/808 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 2
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 82.3% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 67.0% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 14.0% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 12:25 - 04/06/2026)
- **Kết quả đánh giá lâm sàng:** `Đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 99.9% | Đúng: 796/797 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 100.0% | Đúng: 1135/1135 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 31.6% | Đúng: 255/808 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 3
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 99.9% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 100.0% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ3 (Sai số 15°): Đạt 31.6% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 12:27 - 04/06/2026)
- **Kết quả đánh giá lâm sàng:** `Đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 99.9% | Đúng: 796/797 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 100.0% | Đúng: 1135/1135 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 31.6% | Đúng: 255/808 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 3
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 99.9% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 100.0% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ3 (Sai số 15°): Đạt 31.6% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 12:34 - 04/06/2026)
- **Kết quả đánh giá lâm sàng:** `Đúng`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 99.9% | Đúng: 796/797 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 100.0% | Đúng: 1135/1135 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 31.6% | Đúng: 255/808 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 3
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 99.9% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ2 (Sai số 30°): Đạt 100.0% - Đạt yêu cầu chuyển giai đoạn.
  > - GĐ3 (Sai số 15°): Đạt 31.6% - Khớp còn cứng hoặc lệch biên độ.

---

### 🎬 Bài tập 2: Bài tập với gậy (Pulley Exercise)
- **Tên video file:** `Vũ Thị Hoà - Bài tập với gậy.mp4`
- **Thời gian tập:** 14:51 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 24.4% | 29.97° | 0.50 | 0.34 | 9174 |
| GĐ 2: Hồi phục (30°) | 24.4% | 29.97° | 0.50 | 0.34 | 9174 |
| GĐ 3: Chuẩn xác (15°) | 24.4% | 29.97° | 0.50 | 0.34 | 9174 |

#### 🩺 Đánh giá từ Chuyên gia lâm sàng (Ground Truth):
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 02:07 - 03/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 14.1% | Đúng: 420/2977 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 18.8% | Đúng: 691/3677 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 7.5% | Đúng: 190/2520 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 1
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 14.1% - Cần rèn luyện thêm.
  > - GĐ2 (Sai số 30°): Đạt 18.8% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 7.5% - Khớp còn cứng hoặc lệch biên độ.
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 02:08 - 03/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO TỔNG HỢP NCV - ĐẦY ĐỦ 3 GIAI ĐOẠN:
  > 🌱 GĐ 1 (Khởi đầu - Sai số 45°): 14.1% | Đúng: 420/2977 frames
  > 📈 GĐ 2 (Hồi phục - Sai số 30°): 18.8% | Đúng: 691/3677 frames
  > 🎯 GĐ 3 (Chuẩn xác - Sai số 15°): 7.5% | Đúng: 190/2520 frames
  > 🤖 AI đề xuất: Phù hợp tập luyện ở giai đoạn 1
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - GĐ1 (Sai số 45°): Đạt 14.1% - Cần rèn luyện thêm.
  > - GĐ2 (Sai số 30°): Đạt 18.8% - Cần rèn luyện thêm.
  > - GĐ3 (Sai số 15°): Đạt 7.5% - Khớp còn cứng hoặc lệch biên độ.

---

## 👤 BỆNH NHÂN: Cao Thị Thường
Tổng số video bài tập: 2

### 🎬 Bài tập 1: Bài tập con lắc Codman
- **Tên video file:** `Cao Thị Thường -  Codman.mov`
- **Thời gian tập:** 12:32 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 53.9% | 26.44° | 0.50 | 0.60 | 804 |
| GĐ 2: Hồi phục (30°) | 32.3% | 24.88° | 0.50 | 0.41 | 1148 |
| GĐ 3: Chuẩn xác (15°) | 17.9% | 21.66° | 0.55 | 0.28 | 811 |

⚠️ *Chưa có đánh giá lâm sàng từ Bác sĩ cho video này.*

---

### 🎬 Bài tập 2: Bài tập với gậy (Pulley Exercise)
- **Tên video file:** `Cao Thị Thường - Bài tập với gậy.mov`
- **Thời gian tập:** 10:31 - 04/06/2026
- **Trạng thái phân tích AI:** Đã phân tích
- **Cấu hình phân tích:** Giai đoạn 2: Hồi phục (Sai số vừa - 30°) (Sai số cho phép: 30°)

#### 🤖 Chỉ số Phân tích AI (NCV):
| Giai đoạn phân tích | Độ chính xác (ACC) | Sai số trung bình (MAE) | Hệ số đồng thuận (ICC) | Điểm F1 (F1-Score) | Tổng số Frames |
| --- | --- | --- | --- | --- | --- |
| GĐ 1: Khởi đầu (45°) | 38.5% | 26.48° | 0.50 | 0.46 | 5386 |
| GĐ 2: Hồi phục (30°) | 38.5% | 26.48° | 0.50 | 0.46 | 5386 |
| GĐ 3: Chuẩn xác (15°) | 38.5% | 26.48° | 0.50 | 0.46 | 5386 |

#### 🩺 Đánh giá từ Chuyên gia lâm sàng (Ground Truth):
##### 👤 Người đánh giá: NCV: Nghiên cứu viên (Thời gian: 10:41 - 04/06/2026)
- **Kết quả đánh giá lâm sàng:** `Sai`
- **Nhận xét chuyên môn:**
  > BÁO CÁO PHÂN TÍCH BÀI TẬP VỚI GẬY (TỔNG QUAN):
  > 🏒 Độ chính xác: 38.5% | Đúng: 2076/5386 frames
  > 🤖 AI đề xuất: Cần chuyên gia y tế hướng dẫn.
- **Phác đồ đề xuất tiếp theo:**
  > Kế hoạch luyện tập đề xuất:
  > - Bài tập với gậy: Đạt 38.5% - Cần rèn luyện thêm để giảm sai số.

---
