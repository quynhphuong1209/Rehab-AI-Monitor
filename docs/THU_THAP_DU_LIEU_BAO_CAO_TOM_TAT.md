# THU THẬP DỮ LIỆU NGHIÊN CỨU — PHỤC VỤ BÁO CÁO TÓM TẮT KHOA HỌC

*Nguồn: `database/video_list.json`, `doctor_evaluations.json`, `patient_symptoms.json`, `research_data.json`*  
*Đối chiếu tự động: `scripts/export_report_metrics.py` → `docs/_VERIFIED_METRICS.txt`, `docs/_PHASE_METRICS_TABLES.txt`; `scripts/export_rom_charts_data.py` → `docs/_ROM_CHARTS_DATA.txt` (11/06/2026)*  
*AI phân tích lần mới nhất: 04–10/06/2026; đánh giá Bác sĩ PHCN + phiếu NCKH: 08/06/2026*

> **Lưu ý số liệu:** Mỗi video lưu **metrics lần chạy mới nhất** (top-level) và có thể còn **metrics_g1/g2/g3** từ lần chạy cũ. Báo cáo này ưu tiên **top-level** khi có nhiều khung hơn (vd. Cao Codman: **34,3% / 949–2763** chứ không phải g2 cũ 32,3% / 371–1148). Biểu đồ **Boxplot** trong app vẽ từ CSV (`goc_vai`, `goc_khuyu` × nhóm Đúng/Gần đúng/Sai) — số min/max/std bên dưới lấy từ cùng lần phân tích đó.

---

## PHẦN A — QUY ĐỊNH TRÌNH BÀY (copy sang Microsoft Word)

| Yêu cầu | Thiết lập Word |
| --- | --- |
| Khổ giấy | A4 |
| Bảng mã | Unicode TCVN 6909:2001 |
| Font | Times New Roman |
| Cỡ chữ nội dung | 12 |
| Cách dòng | Đơn (Single) |
| Lề trái | 3 cm |
| Lề phải | 2 cm |
| Lề trên | 2,5 cm |
| Lề dưới | 2,5 cm |
| **Tên báo cáo** | Cỡ **14**, **IN HOA**, **đậm**, **căn giữa**, tối đa **20 chữ** |
| **Tác giả / nhóm tác giả** | Cỡ 12, căn giữa, ghi chức danh + đơn vị trong ngoặc |
| **Thầy/Cô hướng dẫn** | Tối đa 2 người: học hàm, họ tên, đơn vị |
| **Tóm tắt** | Một đoạn văn, **≤ 200 từ**, không tách mục |
| **Từ khóa** | Tối đa **6** từ/cụm từ |

---

## PHẦN B — BẢN BÁO CÁO TÓM TẮT (sẵn sàng đánh máy)

<div align="center">

### **ỨNG DỤNG AI GIÁM SÁT TẬP LUYỆN PHCN KHỚP VAI TỪ XA**

*(14 pt, in hoa, đậm, căn giữa — 12 chữ)*

**Nhóm tác giả:** Đinh Lê Quỳnh Phương (Chủ nhiệm đề tài, sinh viên ngành Khoa học dữ liệu, lớp CNCQ KHDL1-1A); Kim Mạnh Hưng, Nguyễn Hải An, Phan Vân Anh, Nguyễn Thị Thanh Nga (sinh viên ngành Khoa học dữ liệu, lớp CNCQ KHDL1-1A); Nguyễn Thị Thơm (sinh viên ngành Kỹ thuật Phục hồi chức năng, lớp CNCQ KTPHCN3-1A); Nguyễn Thị Thu Hương (sinh viên ngành Y tế công cộng, lớp CNCQ YTCC22-1A) — Trường Đại học Y tế Công cộng

**Thầy/Cô hướng dẫn:** TS. Trần Hồng Việt (Trường Đại học Y tế Công cộng); ThS. Nguyễn Thị Thùy Chi (Trường Đại học Y tế Công cộng)

</div>

**Tóm tắt:** Người bệnh phục hồi chức năng (PHCN) tự tập tại nhà thiếu giám sát chuyên môn nên dễ sai động tác, giảm hiệu quả điều trị. Nghiên cứu nhằm xây dựng mô hình thị giác máy tính đánh giá động tác PHCN khớp vai và đối chiếu với đánh giá của chuyên gia. Đối tượng gồm 04 bệnh nhân nữ (39–71 tuổi) viêm quanh khớp vai (ICD-10: M75) tại Bệnh viện Đa khoa Phạm Ngọc Thạch; phương pháp thu thập 08 video smartphone (bài tập con lắc Codman và bài tập với gậy), trích xuất 33 điểm mốc MediaPipe Pose, tính góc vai–khuỷu và phân loại đúng/sai theo ba ngưỡng sai số ±45°, ±30°, ±15° tương ứng ba giai đoạn phục hồi; đối chiếu song song với đánh giá độc lập của bác sĩ/KTV PHCN và phiếu NCKH (≈ 24.100 khung hình). Kết quả: bài tập Codman đạt độ chính xác 34,3–100% (ngưỡng ±30°), bài tập với gậy 18,8–48,3%; AI và bác sĩ thống nhất xu hướng (Codman tốt hơn gậy) nhưng AI chưa phát hiện lỗi bù trừ tư thế thân người (5/8 video). Kết luận: mô hình marker-less khả thi giám sát biên độ vận động khớp vai từ xa, cần bổ sung tiêu chí tư thế thân người và mở rộng cỡ mẫu.

*(≈ 198 từ)*

**Từ khóa:** Phục hồi chức năng từ xa; trí tuệ nhân tạo; thị giác máy tính; ước lượng tư thế; khớp vai; MediaPipe Pose.

---

## PHẦN C — TỔNG HỢP DỮ LIỆU NGHIÊN CỨU

### C.1. Đối tượng và quy mô mẫu

| Chỉ số | Giá trị |
| --- | --- |
| Số bệnh nhân | 4 (100% nữ) |
| Độ tuổi | 39; 55; 57; 71 (TB ≈ 55,5) |
| Chẩn đoán | Viêm quanh khớp vai (ICD-10: M75) |
| Mức đau VAS (khai báo) | 6–8/10 |
| Số video bài tập | 8 (4 Codman + 4 với gậy) |
| Tổng khung hình phân tích AI | **24.106** khung hợp lệ (tổng `tong_frame_hop_le` 8 video) |
| Đánh giá AI (NCV) | 8/8 video |
| Đánh giá Bác sĩ PHCN | 8/8 video (doctor1, 08/06/2026) |
| Phiếu NCKH | 8/8 phiếu (08/06/2026) |
| Phiếu khai báo triệu chứng BN | 8/8 (mỗi BN × 2 bài tập) |

### C.2. Thông tin bệnh nhân và triệu chứng khai báo

| BN | Mã | Tuổi | VAS | Triệu chứng / đặc điểm lâm sàng nổi bật |
| --- | --- | --- | --- | --- |
| Hoàng Hạnh Nguyên | 25009284 | 39 | 6 | Đau vai (P) nhiều tháng, tăng khi vận động, đau đêm; chưa điều trị |
| Nguyễn Thị Nga | 25007938 | 55 | 8 | Đau vai (P) vài tháng; Jobe test (+); đã VLTL–Đông y |
| Vũ Thị Hòa | 26002558 | 57 | 6 | Đau hai vai; tê bì cánh tay; hạn chế xoay trong/ngoài vai (P) |
| Cao Thị Thường | 26001385 | 71 | 6 | Đau vai (P) nhiều tháng, đau đêm; tiền sử đau dạ dày |

### C.3. Đối chiếu AI – Bác sĩ PHCN (kết quả chính)

*Ngưỡng đánh giá AI chính: Giai đoạn 2 (±30°)*

| Bệnh nhân | Bài tập | ACC AI (±30°) | Khung đúng/tổng | Kết luận AI | Kết luận BS | Lỗi BS ghi nhận |
| --- | --- | --- | --- | --- | --- | --- |
| Hoàng Hạnh Nguyên | Codman | 94,8% | 639/674 | Đúng | Gần đúng | Sai tư thế thân người |
| Nguyễn Thị Nga | Codman | 64,0% | 2084/2628 | Gần đúng | Gần đúng | Sai tư thế thân người |
| Vũ Thị Hòa | Codman | 100,0% | 1135/1135 | Đúng | Sai | Sai tư thế thân người |
| Cao Thị Thường | Codman | 34,3% | 949/2763 | Sai | Gần đúng | Sai tư thế thân người |
| Hoàng Hạnh Nguyên | Với gậy | 48,3% | 749/1550 | Sai | Gần đúng | Vị trí tay chưa đúng |
| Nguyễn Thị Nga | Với gậy | 32,7% | 1485/4535 | Sai | Gần đúng | Sai tư thế thân người |
| Vũ Thị Hòa | Với gậy | 18,8% | 1326/5435 | Sai | Gần đúng | Biên độ chưa đạt |
| Cao Thị Thường | Với gậy | 38,5% | 2076/5386 | Sai | Sai | Vị trí tay chưa đúng |

**Nhận xét đối chiếu:**
- Codman: ACC ±30° từ **34,3%** đến 100% (TB ≈ **73,3%**)
- Với gậy: ACC ±30° từ 18,8% đến 48,3% (TB ≈ 34,6%)
- Lỗi thân người do BS ghi: **5/8 video** — AI chưa phát hiện được
- Trường hợp điển hình: Vũ Thị Hòa – Codman ACC 100% nhưng BS kết luận **Sai** (tư thế thân người)

### C.4. Bảng chỉ số AI đầy đủ — Codman (trung bình 4 BN, 3 giai đoạn)

*RMSE = MAE × 1,25 (cùng công thức hiển thị trong app). GĐ2: ưu tiên metrics top-level khi lần chạy mới hơn `metrics_g2` (vd. Cao Thường 34,3% / 949 khung).*

| Ký hiệu | Giai đoạn 1 (±45°) | Giai đoạn 2 (±30°) | Giai đoạn 3 (±15°) | Phân loại / Chuyên môn |
| --- | --- | --- | --- | --- |
| Độ chính xác hệ thống (ACC) | 82,5% | **73,3%** | 30,0% | Đối soát từng giây với video YouTube mẫu |
| Sai số tuyệt đối trung bình (MAE) | 15,9° | 14,4° | 18,2° | Tốt ở G1–G2; G3 khắt khe hơn |
| Sai số bình phương trung bình (RMSE) | 19,9° | 18,1° | 22,7° | RMSE = MAE × 1,25 |
| Hệ số tương quan nội lớp (ICC) | 0,67 | 0,69 | 0,62 | Khá tốt ở G1–G2 |
| Độ nhạy phân loại (Recall) | 0,88 | 0,79 | 0,41 | Giảm mạnh ở G3 (ngưỡng ±15°) |
| Độ đặc hiệu phân loại (Precision) | 0,89 | 0,80 | 0,45 | Tin cậy cảnh báo sai tư thế |
| Chỉ số cân bằng F1 (F1-Score) | 0,88 | 0,80 | 0,43 | Hiệu suất AI tổng hợp |
| Số lần tập đúng — Pass (khung) | 2.432 | 3.678 | 1.022 | Tổng khung đạt chuẩn theo từng giai đoạn |

### C.4b. Codman — chi tiết từng bệnh nhân (3 giai đoạn)

**Hoàng Hạnh Nguyên**

| Chỉ số | G1 (±45°) | G2 (±30°) | G3 (±15°) |
| --- | --- | --- | --- |
| ACC | 97,6% | 94,8% | 41,9% |
| MAE / RMSE | 12,9° / 16,1° | 10,8° / 13,5° | 23,8° / 29,8° |
| ICC / F1 | 0,72 / 0,98 | 0,76 / 0,95 | 0,50 / 0,49 |
| Recall / Precision | 0,98 / 0,98 | 0,95 / 0,96 | 0,48 / 0,51 |
| Pass | 491 | 639 | 205 |

**Nguyễn Thị Nga**

| Chỉ số | G1 (±45°) | G2 (±30°) | G3 (±15°) |
| --- | --- | --- | --- |
| ACC | 78,7% | 64,0% | 28,8% |
| MAE / RMSE | 12,9° / 16,1° | 11,7° / 14,6° | 13,7° / 17,1° |
| ICC / F1 | 0,72 / 0,97 | 0,75 / 0,82 | 0,71 / 0,55 |
| Recall / Precision | 0,97 / 0,97 | 0,81 / 0,82 | 0,54 / 0,56 |
| Pass | 712 | 955 | 417 |

**Vũ Thị Hòa**

| Chỉ số | G1 (±45°) | G2 (±30°) | G3 (±15°) |
| --- | --- | --- | --- |
| ACC | 99,9% | 100,0% | 31,6% |
| MAE / RMSE | 11,4° / 14,2° | 10,9° / 13,6° | 13,5° / 16,9° |
| ICC / F1 | 0,75 / 0,99 | 0,76 / 0,99 | 0,71 / 0,40 |
| Recall / Precision | 0,99 / 0,99 | 0,99 / 0,99 | 0,38 / 0,42 |
| Pass | 796 | 1.135 | 255 |

**Cao Thị Thường** *(GĐ2 = metrics lần chạy mới nhất toàn video)*

| Chỉ số | G1 (±45°) | G2 (±30°) | G3 (±15°) |
| --- | --- | --- | --- |
| ACC | 53,9% | **34,3%** | 17,9% |
| MAE / RMSE | 26,4° / 33,1° | 24,4° / 30,5° | 21,7° / 27,1° |
| ICC / F1 | 0,50 / 0,60 | 0,50 / 0,42 | 0,55 / 0,28 |
| Recall / Precision | 0,58 / 0,61 | 0,41 / 0,44 | 0,26 / 0,30 |
| Pass | 433 | **949** | 145 |

### C.4c. Bảng chỉ số AI đầy đủ — Bài tập với gậy (trung bình 4 BN)

*Bài gậy chủ yếu phân tích tổng quan ±30°; Vũ Thị Hòa có tách 3 giai đoạn riêng (14,1% / 18,8% / 7,5%).*

| Ký hiệu | G1 (±45°) | G2 (±30°) | G3 (±15°) | Ghi chú |
| --- | --- | --- | --- | --- |
| ACC | 33,4% | **34,6%** | 31,8% | TB ngưỡng chính ±30° = 34,6% |
| MAE | 29,4° | 29,4° | 29,4° | |
| RMSE | 36,7° | 36,7° | 36,7° | MAE × 1,25 |
| ICC | 0,50 | 0,50 | 0,50 | |
| Recall | 0,42 | 0,42 | 0,42 | |
| Precision | 0,46 | 0,46 | 0,46 | |
| F1-Score | 0,44 | 0,44 | 0,44 | |
| Pass | 5.636 | 5.636 | 5.636 | Tổng khung đúng (±30°) |

### C.4d. Gậy — chi tiết từng bệnh nhân

| BN | ACC (±30°) | MAE | RMSE | ICC | F1 | Pass / Tổng | G1 / G2 / G3 *(nếu có)* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hoàng Hạnh Nguyên | 48,3% | 33,7° | 42,1° | 0,50 | 0,55 | 749 / 1.550 | 48,3% / 48,3% / 48,3% |
| Nguyễn Thị Nga | 32,7% | 27,4° | 34,2° | 0,50 | 0,41 | 1.485 / 4.535 | 32,7% / 32,7% / 32,7% |
| Vũ Thị Hòa | 18,8% | 30,0° | 37,5° | 0,50 | 0,34 | 1.326 / 5.435 | **14,1% / 18,8% / 7,5%** |
| Cao Thị Thường | 38,5% | 26,5° | 33,1° | 0,50 | 0,46 | 2.076 / 5.386 | 38,5% / 38,5% / 38,5% |

### C.4e. Tóm tắt nhanh (GĐ2 ±30°) — góc khớp & khung hình

| BN | Bài tập | ACC | MAE (°) | F1 | ICC | Góc vai TB | Góc khuỷu TB | Pass / Tổng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hoàng Hạnh Nguyên | Codman | 94,8% | 10,8 | 0,95 | 0,76 | 44,5° | 161,9° | 639 / 674 |
| Hoàng Hạnh Nguyên | Gậy | 48,3% | 33,7 | 0,55 | 0,50 | 21,1° | 114,4° | 749 / 1.550 |
| Nguyễn Thị Nga | Codman | 64,0% | 11,7 | 0,82 | 0,75 | 44,4° | 161,4° | 2.084 / 2.628 |
| Nguyễn Thị Nga | Gậy | 32,7% | 27,4 | 0,41 | 0,50 | 52,0° | 138,5° | 1.485 / 4.535 |
| Vũ Thị Hòa | Codman | 100,0% | 10,9 | 0,99 | 0,76 | 25,6° | 163,9° | 1.135 / 1.135 |
| Vũ Thị Hòa | Gậy | 18,8% | 30,0 | 0,34 | 0,50 | 51,7° | 131,7° | 1.326 / 5.435 |
| Cao Thị Thường | Codman | 34,3% | 24,4 | 0,42 | 0,50 | 84,7° | 153,6° | 949 / 2.763 |
| Cao Thị Thường | Gậy | 38,5% | 26,5 | 0,46 | 0,50 | 39,6° | 136,7° | 2.076 / 5.386 |

### C.5. Phân tích 3 giai đoạn — Bài tập Codman (ACC %)

| Bệnh nhân | GĐ1 (±45°) | GĐ2 (±30°) | GĐ3 (±15°) | AI đề xuất |
| --- | --- | --- | --- | --- |
| Hoàng Hạnh Nguyên | 97,6% | 94,8% | 41,9% | Giai đoạn 3 |
| Nguyễn Thị Nga | 78,7% | 64,0% | 28,8% | Giai đoạn 2 |
| Vũ Thị Hòa | 99,9% | 100,0% | 31,6% | Giai đoạn 3 |
| Cao Thị Thường | 53,9% | **34,3%** | 17,9% | Giai đoạn 1 |
| **Trung bình 4 BN** | **82,5%** | **73,3%** | **30,0%** | — |

### C.5b. Số liệu Boxplot (phân phối góc — phục vụ mô tả biến thiên)

*Boxplot app nhóm theo Đúng / Gần đúng / Sai; bảng dưới là min–max–độ lệch chuẩn (std) từ cùng lần phân tích (GĐ2 Codman hoặc tổng quan Gậy).*

**Lệnh tự lấy từ dữ liệu web (không cần mở app):**

```bash
python scripts/export_rom_charts_data.py
```

- Đọc **8 video nghiên cứu** từ `video_list.json` (cùng logic tab NCV).
- Xuất **ROM 3 giai đoạn** (Codman) + **tổng quan / 3 GĐ** (Gậy; Hòa: 14,1% / 18,8% / 7,5%) → `docs/_ROM_CHARTS_DATA.txt` + `.json`.
- Tùy chọn boxplot theo nhóm từ CSV: `python scripts/export_rom_charts_data.py --hf --csv` (cần `HF_TOKEN` hoặc CSV trong `processed_results/`). **PowerShell:** `$env:HF_TOKEN = "hf_xxx"` (không dùng `set` như CMD).

| BN | Bài tập | Khớp | TB (°) | Min (°) | Max (°) | Std (°) |
| --- | --- | --- | --- | --- | --- | --- |
| Hoàng Hạnh Nguyên | Codman | Vai | 44,5 | 0,0 | 94,1 | 24,8 |
| Hoàng Hạnh Nguyên | Codman | Khuỷu | 161,9 | 5,3 | 179,9 | 23,6 |
| Nguyễn Thị Nga | Codman | Vai | 44,4 | 0,0 | 126,8 | 27,0 |
| Nguyễn Thị Nga | Codman | Khuỷu | 161,4 | 0,2 | 179,9 | 18,4 |
| Vũ Thị Hòa | Codman | Vai | 25,6 | 0,0 | 64,9 | 19,6 |
| Vũ Thị Hòa | Codman | Khuỷu | 163,9 | — | — | 3,4 |
| Cao Thị Thường | Codman | Vai | 84,7 | 2,3 | 179,8 | 27,8 |
| Cao Thị Thường | Codman | Khuỷu | 153,6 | 13,0 | 180,0 | 25,6 |
| Cao Thị Thường | Gậy | Vai | 39,6 | 0,0 | 161,5 | 37,9 |
| Vũ Thị Hòa | Gậy | Vai | 51,7 | 0,0 | 170,2 | 51,2 |

### C.6. Phiếu NCKH — Thông tin thu thập (08/06/2026)

| BN | Video | Chẩn đoán | Vai tập | Thời gian bệnh | Mức đau | Góc quay | K/c camera (m) | Kết quả BS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hoàng Hạnh Nguyên | Codman | M75 (P) | Vai trái | ≥ 3 tháng | 4–6 | Chính diện | 3,0 | Gần đúng |
| Hoàng Hạnh Nguyên | Gậy | M75 (P) | Cả hai vai | ≥ 3 tháng | 4–6 | Chính diện | 3,5 | Gần đúng |
| Nguyễn Thị Nga | Codman | M75 (P) | Vai phải | ≥ 3 tháng | 4–6 | Chính diện | 3,0 | Gần đúng |
| Nguyễn Thị Nga | Gậy | M75 (P) | Vai phải | ≥ 3 tháng | 4–6 | Chính diện | 3,0 | Gần đúng |
| Vũ Thị Hòa | Codman | M75 (P) | Cả hai vai | ≥ 3 tháng | 4–6 | Chính diện | 2,5 | Sai |
| Vũ Thị Hòa | Gậy | M75 (P) | Cả hai vai | ≥ 3 tháng | 4–6 | Chính diện | 2,5 | Gần đúng |
| Cao Thị Thường | Codman | M75 (P) | Vai phải | ≥ 3 tháng | 4–6 | Chính diện | 3,0 | Gần đúng |
| Cao Thị Thường | Gậy | M75 (P) | Vai phải | ≥ 3 tháng | 4–6 | Chính diện | 4,0 | Sai |

### C.7. Trích xuất video & khung xương (kỹ thuật)

| Hạng mục | Mô tả |
| --- | --- |
| Công nghệ | MediaPipe Pose — 33 landmarks |
| Góc tính toán | Góc khớp vai, góc khớp khuỷu (vector tích vô hướng) |
| Chuẩn tham chiếu | JSON góc chuẩn từ video YouTube (REF) — Codman, gậy, dây |
| Phân loại | Rule-based (ngưỡng góc) + ML Classifier (Random Forest) |
| Đầu ra mỗi video | CSV góc khớp (`processed_*_f_data.csv`); JSON toàn frame (`f_*.json`); video overlay khung xương (`processed_*_f.mp4`); zip ảnh frame |
| Tổng file phân tích | 8 CSV + 8 JSON + 8 video processed (lưu Cloud Dataset) |

---

## PHẦN D — CHI TIẾT TỪNG BỆNH NHÂN (copy vào báo cáo đầy đủ)

### D.1. Hoàng Hạnh Nguyên (39 tuổi, VAS 6)

**Codman**
- AI: ACC 94,8% (GĐ2); MAE 10,8°; F1 0,95; ICC 0,76 → **Đúng**
- BS: **Gần đúng** — lỗi: sai tư thế thân người; nhận xét: cần gập thân song song mặt sàn, tay còn lại bám ghế/vật cố định

**Với gậy**
- AI: ACC 48,3%; MAE 33,7°; F1 0,55 → **Sai**
- BS: **Gần đúng** — lỗi: vị trí tay chưa đúng; nhận xét: xoay trong/ngoài cánh tay cần áp sát thân mình

### D.2. Nguyễn Thị Nga (55 tuổi, VAS 8)

**Codman**
- AI: ACC 64,0% (2084/2628 khung); MAE 11,7°; F1 0,82; ICC 0,75 → **Gần đúng**
- BS: **Gần đúng** — lỗi: sai tư thế thân người

**Với gậy**
- AI: ACC 32,7%; MAE 27,4°; F1 0,41 → **Sai**
- BS: **Gần đúng** — lỗi: sai tư thế thân người

### D.3. Vũ Thị Hòa (57 tuổi, VAS 6)

**Codman**
- AI: ACC 100,0%; MAE 10,9°; F1 0,99; ICC 0,76 → **Đúng**
- BS: **Sai** — lỗi: sai tư thế thân người *(phát hiện quan trọng: AI–BS lệch)*

**Với gậy**
- AI: ACC 18,8%; MAE 30,0°; F1 0,34 → **Sai**
- BS: **Gần đúng** — lỗi: biên độ chưa đạt

### D.4. Cao Thị Thường (71 tuổi, VAS 6)

**Codman**
- AI: ACC **34,3%** (949/2763 khung); MAE 24,4°; F1 0,42; vai đúng 48,1% / khuỷu đúng 78,4%; cảnh báo: khuỷu gập quá mức → **Sai**
- BS: **Gần đúng** — lỗi: sai tư thế thân người; nhận xét: hai chân rộng ngang vai, không chân trước chân sau

**Với gậy**
- AI: ACC 38,5%; MAE 26,5°; F1 0,46 → **Sai**
- BS: **Sai** — lỗi: vị trí tay chưa đúng

---

## PHẦN E — KẾT LUẬN CHO BÁO CÁO ĐẦY ĐỦ

1. **Khả thi kỹ thuật:** Mô hình marker-less (MediaPipe + ngưỡng góc 3 giai đoạn) giám sát được biên độ vai/khuỷu trên video smartphone.
2. **Giá trị lâm sàng:** Phân tầng ±45°/±30°/±15° giúp đề xuất giai đoạn tập phù hợp từng BN.
3. **Độ đồng quy AI–BS:** Thống nhất xu hướng Codman > gậy; lệch ở tiêu chí tư thế thân người.
4. **Hạn chế:** Cỡ mẫu nhỏ (4 BN); chưa giám sát compensatory movement; bài gậy độ chính xác thấp.

---

## PHẦN F — FILE NGUỒN TRONG HỆ THỐNG

| File | Nội dung |
| --- | --- |
| `database/patient_symptoms.json` | Khai báo triệu chứng, VAS, tiền sử |
| `database/doctor_evaluations.json` | Đánh giá AI (NCV) + Bác sĩ PHCN |
| `database/research_data.json` | Phiếu NCKH (08 phiếu) |
| `database/video_list.json` | Metrics, đường dẫn CSV/JSON/video |
| `docs/KET_QUA_NCV_VA_BAC_SI.md` | Báo cáo chi tiết từng BN |
| `docs/BAO_CAO_TOM_TAT_PHUC_VU_ABSTRACT.md` | Bản thảo abstract trước đó |
| `docs/_VERIFIED_METRICS.txt` | Bảng đối chiếu tự động (chạy lại khi cập nhật JSON) |
| `docs/_PHASE_METRICS_TABLES.txt` | Bảng ACC/MAE/RMSE/ICC/F1/Pass theo 3 giai đoạn (Codman + Gậy) |
| `docs/_ROM_CHARTS_DATA.txt` | ROM + boxplot tóm tắt 8 video (Codman 3 GĐ, Gậy) |
| `docs/_ROM_CHARTS_DATA.json` | Cùng dữ liệu dạng JSON (dùng cho biểu đồ/script khác) |
| `scripts/export_report_metrics.py` | Script xuất chỉ số ACC/MAE/RMSE/ICC/F1/Pass |
| `scripts/export_rom_charts_data.py` | Script xuất ROM + boxplot từ metrics web |

**Chạy cả hai script (chỉ số + biểu đồ/ROM):**

```bash
python scripts/export_report_metrics.py
python scripts/export_rom_charts_data.py
```

*Ảnh PNG biểu đồ: tab **Phân tích** trên web, hoặc thêm `--hf --csv` để lấy median/Q1/Q3 theo nhóm Đúng/Gần đúng/Sai từ CSV trên HF Dataset.*
