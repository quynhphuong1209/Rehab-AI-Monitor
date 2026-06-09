# Hướng dẫn đọc nhãn REF (PASS) và ML (%) trên khung hình

Tài liệu này giải thích cách đọc **hai nhãn độc lập** hiển thị trên mỗi khung hình trích xuất trong Rehab AI Monitor: nhãn **REF** (rule-based, so góc với video chuẩn YouTube) và nhãn **ML** (mô hình RandomForest học từ dữ liệu đã phân tích).

---

## 1. Tổng quan: Mỗi khung hình có 2 nhãn

| Nhãn trên ảnh | Tên gọi | Cách chấm |
|---------------|---------|-----------|
| **PASS / NEAR / FAIL** | **REF** (Reference / Rule) | So góc vai & khuỷu với video chuẩn YouTube |
| **ML · Đúng / Gần đúng / Sai · tin cậy X%** | **ML** (Machine Learning) | Mô hình RandomForest dự đoán từ góc khớp + tọa độ 8 khớp quan trọng |

Hai nhãn **độc lập** — có thể khác nhau trên cùng một khung hình. Ví dụ: REF = **PASS** nhưng ML = **Gần đúng · tin cậy 33%**.

---

## 2. Nhãn REF (PASS / NEAR / FAIL)

### 2.1. Góc chuẩn lấy từ video YouTube mẫu — **khớp theo tư thế, không theo từng giây**

Video YouTube mẫu (Codman, gậy, dây kháng lực) thường dài và gồm **nhiều động tác** (giơ tay, đung trước–sau, sang ngang, **xoay vòng tròn bàn tay**…). Hệ thống **không** bắt video bệnh nhân khớp lần lượt từng giây 0 s → 1 s → 2 s với video mẫu.

**Cách máy làm thực tế:**

1. **Trích xuất trước** bản đồ góc chuẩn từ video YouTube (MediaPipe) → lưu trong `database/reference_<bài_tập>.json`  
   Mỗi tư thế mẫu gồm: `time`, `vai`, `khuyu` và (với gậy) `vai_trai`, `khuyu_trai`, `vai_phai`, `khuyu_phai`, kèm `exercise_id` (1/2/3) và `motion_type`.  
   Chạy lại trích xuất: `python scripts/extract_youtube_reference.py codman gay`

2. **Ưu tiên đoạn động tác liên quan xoay / con lắc** (không lấy cả video tuần tự):
   - **Codman:** tập trung đoạn **xoay vòng tròn bàn tay / con lắc** (tay thả lỏng, vai–khuỷu dao động theo vòng cung). File `reference_codman.json` mô tả chu kỳ góc vai ~30°↔60°, khuỷu ~171°↔175° trong pha xoay — **không** dùng phần mở đầu/giới thiệu của video YouTube.
   - **Bài tập gậy (Pulley):** tương tự — bản chuẩn lấy từ các pha **nâng gậy, xoay vai ngoài/trong** có biên độ rõ trên video mẫu (`reference_gay.json`), không ép khớp theo thứ tự thời gian tuyệt đối.

3. **Nhận dạng động tác con (1/2/3)** từ góc hiện tại — ví dụ bệnh nhân **dơ gậy cao** → khớp mẫu động tác 1 (flexion) trong `reference_gay.json`; Codman xoay vòng → động tác 3 (circular), chỉ so **tay phải**.

4. **Khi phân tích từng frame bệnh nhân**, máy tìm **tư thế mẫu gần nhất** trong JSON (theo góc, không theo giây):

   ```
   Frame bệnh nhân: góc vai hiện tại = 52°
   → Tìm trong reference_codman.json dòng có vai gần 52° nhất (vd. vai = 51°)
   → Lấy cặp góc chuẩn vai/khuỷu của dòng đó để so sánh
   ```

   Logic trong mã nguồn (`app.py` → `xu_ly_frame`):

   - Codman & gậy: chọn mẫu có **|vai_mẫu − vai_bệnh_nhân|** nhỏ nhất
   - Dây kháng lực: chọn mẫu có **|khuỷu_mẫu − khuỷu_bệnh_nhân|** nhỏ nhất

5. **Codman:** chỉ so góc **tay phải** (vai phải + khuỷu phải).  
   **Gậy:** so **cả hai bên** — cả trái và phải đều phải đạt ngưỡng.

**Tóm lại:** Video mẫu YouTube cung cấp **thư viện tư thế chuẩn**; mỗi frame bệnh nhân được đối chiếu với **tư thế tương đồng nhất** trong thư viện đó — giống như tìm “điểm trên vòng tròn xoay tay” có góc gần nhất, thay vì bắt đúng giây thứ mấy trong video.

| Video mẫu YouTube | File chuẩn | Đoạn ưu tiên trích góc |
|-------------------|------------|-------------------------|
| Codman — [youtu.be/a4eCRWuqO40](https://youtu.be/a4eCRWuqO40) | `reference_codman.json` | BT1 đung trước-sau · BT2 sang ngang · BT3 xoay vòng (tay phải) |
| Gậy — [youtube.com/watch?v=s2O8WHT5o2k](https://www.youtube.com/watch?v=s2O8WHT5o2k) | `reference_gay.json` | BT1 dơ gậy cao · BT2 xoay ngoài · BT3 xoay trong (hai bên) |
| Dây kháng lực — [youtube.com/watch?v=njDHDnZ6lis](https://www.youtube.com/watch?v=njDHDnZ6lis) | `reference_day.json` | Các pha xoay/dang theo mẫu hướng dẫn |

### 2.2. Cách tính sai số (Δ)

Sau khi đã chọn được tư thế mẫu gần nhất:

- **Vai (shoulder):** Δ vai = |góc vai đo − góc vai chuẩn|
- **Khuỷu (elbow):** Δ khuỷu = |góc khuỷu đo − góc khuỷu chuẩn|

**Cả vai và khuỷu** phải thỏa điều kiện mới được xếp vào một nhóm.

### 2.3. Ngưỡng phân loại REF

| Nhãn REF | Màu | Điều kiện (cả Δ vai **và** Δ khuỷu) |
|----------|-----|--------------------------------------|
| **PASS** | Xanh | Δ ≤ **sai số cho phép** của giai đoạn đang xem |
| **NEAR** | Cam | Δ ≤ sai số × **1,5** (chưa đạt PASS) |
| **FAIL** | Đỏ | Δ **vượt** sai số × 1,5 |

**Sai số cơ bản của bài tập** (trong cấu hình `chuan.sai_so`): **±30°** — dùng khi gán nhãn đúng/gần đúng trong pipeline xử lý.

### 2.4. Ba giai đoạn PHCN (G1 / G2 / G3) — là **mức sai số cho phép**, không phải “góc 45°”

G1, G2, G3 **không** có nghĩa là “bài tập ở góc 45° / 30° / 15°”. Đây là **ba mức độ chặt** khi đánh giá nghiên cứu theo tiến trình PHCN — mỗi giai đoạn cho phép sai số Δ khác nhau:

| Giai đoạn | Tên trên UI | **Sai số cho phép (PASS)** | **Sai số NEAR** (× 1,5) | Ý nghĩa lâm sàng |
|-----------|-------------|----------------------------|--------------------------|------------------|
| **G1** | Khởi đầu | **±45°** | ±67,5° | Giai đoạn sớm — yêu cầu lỏng hơn |
| **G2** | Hồi phục | **±30°** | ±45° | Giai đoạn giữa — mặc định thường dùng |
| **G3** | Chuẩn xác | **±15°** | ±22,5° | Giai đoạn cuối — yêu cầu chặt nhất |

Video bệnh nhân được **tự chia 3 đoạn thời gian** (theo chu kỳ góc khớp — điểm “đáy” con lắc) rồi áp dụng lần lượt mức sai số G1 → G2 → G3 cho từng đoạn.

> **Ví dụ đọc kết quả:** Frame có Δ vai = 36° · Δ khuỷu = 32°  
> - Xem ở **G1 (sai số ±45°)** → **PASS**  
> - Xem ở **G2 (sai số ±30°)** → **NEAR** (36° > 30° nhưng ≤ 45°)  
> - Xem ở **G3 (sai số ±15°)** → **FAIL**

### 2.5. Bài tập với gậy (Stick / Pulley)

Đánh giá **cả hai bên** (trái và phải): vai trái, vai phải, khuỷu trái, khuỷu phải đều phải nằm trong ngưỡng sai số tương ứng. Góc chuẩn vẫn lấy bằng cách **khớp tư thế gần nhất** trong `reference_gay.json`, không theo từng giây video YouTube.

### 2.6. Dòng số dưới mỗi khung hình (REF)

```
Vai: 96° / 60° | Δ 36.2°
Khuỷu: 139° / 171° | Δ 32.1°
```

| Thành phần | Ý nghĩa |
|------------|---------|
| **96°** | Góc vai đo được từ MediaPipe |
| **60°** | Góc vai chuẩn (video YouTube) |
| **Δ 36.2°** | Sai số so với chuẩn |

---

## 3. Nhãn ML (Đúng / Gần đúng / Sai)

### 3.1. Mô hình dùng gì?

- **Thuật toán:** RandomForest Classifier (200 cây)
- **Đầu vào mỗi frame:** góc vai, góc khuỷu + tọa độ 8 khớp (2 vai, 2 khuỷu, 2 cổ tay, 2 hông)
- **Huấn luyện:** từ các file CSV trong `processed_results/`, nhãn gốc lấy từ cột `dung` và `gan_dung` (rule REF)

### 3.2. Ba lớp phân loại ML

| Nhãn ML | Mã lớp | Ý nghĩa |
|---------|--------|---------|
| **Sai** | 0 | Động tác sai |
| **Gần đúng** | 1 | Động tác gần đúng |
| **Đúng** | 2 | Động tác đúng |

### 3.3. ML **không** dùng ngưỡng % cố định kiểu “≥ 80% = Đúng”

Cách gán nhãn ML:

1. Mô hình tính **3 xác suất**: P(Sai), P(Gần đúng), P(Đúng) — tổng ≈ 100%
2. Chọn lớp có **xác suất cao nhất** → đó là nhãn ML hiển thị
3. Con số **%** kèm theo = **độ tin cậy vào đúng nhãn ML đang hiển thị** (`ml_confidence`)

> **Lưu ý:** % **không** phải “% đúng động tác” hay “% giống video chuẩn”. Đó là mức tin cậy thống kê của mô hình vào nhãn vừa dự đoán.

### 3.4. Bảng đọc mức tin cậy ML

| Tin cậy ML | Ý nghĩa khi đọc kết quả |
|------------|-------------------------|
| **≥ 70%** | **Tin cậy cao** — có thể tham khảo mạnh |
| **50–69%** | **Tin cậy vừa** — nên xem kèm nhãn REF và Δ góc |
| **< 50%** | **Không chắc chắn** — mô hình phân vân giữa các lớp |

### 3.5. Định dạng hiển thị trên UI

```
ML · Gần đúng · tin cậy 42%
```

Dòng phụ (khi có dữ liệu đầy đủ):

```
Xác suất 3 lớp: Sai 18% · Gần đúng 42% · Đúng 40%
```

---

## 4. Ví dụ thực tế

### Ví dụ A — REF PASS, ML Gần đúng · tin cậy 33%

| Thành phần | Giá trị | Giải thích |
|------------|---------|------------|
| REF | **PASS** | Δ vai & Δ khuỷu ≤ **sai số G1 (±45°)** |
| ML | **Gần đúng · tin cậy 33%** | Mô hình chọn lớp “Gần đúng” nhưng chỉ 33% tin — dưới 50% → **không chắc chắn** |
| Δ vai / Δ khuỷu | 36,2° / 32,1° | REF đạt ở G1; ML vẫn thấy tư thế tổng thể “gần đúng” hơn “đúng hoàn toàn” |

### Ví dụ B — REF và ML cùng hướng

| REF | ML | Ý nghĩa |
|-----|-----|---------|
| PASS | Đúng · tin cậy 85% | Góc đạt chuẩn, ML cũng tin mạnh là đúng |
| NEAR | Gần đúng · tin cậy 62% | Cả hai đồng thuận “gần đúng” |
| FAIL | Sai · tin cậy 78% | Góc lệch nhiều, ML cũng phân loại sai |

---

## 5. So sánh nhanh REF vs ML

| Tiêu chí | REF (PASS/NEAR/FAIL) | ML (Đúng/Gần đúng/Sai) |
|----------|----------------------|------------------------|
| **Cơ sở** | Góc chuẩn YouTube | Học từ dữ liệu nhiều video |
| **Ngưỡng** | Sai số cho phép G1/G2/G3 (±45° / ±30° / ±15°) | Không có ngưỡng % cố định |
| **% hiển thị** | Không có | Tin cậy vào nhãn ML |
| **Mục đích** | Đối chiếu chuẩn lâm sàng | Gợi ý bổ sung từ kinh nghiệm dữ liệu |

**Khuyến nghị:** Luôn đọc **REF + Δ góc** trước; dùng **ML + tin cậy %** như tham khảo thứ hai, đặc biệt khi tin cậy ≥ 70%.

---

## 6. Cập nhật dữ liệu ML trên video cũ

Video phân tích **trước** bản cập nhật hiển thị ML có thể thiếu dòng **Xác suất 3 lớp**. Để cập nhật:

1. Bấm **Chạy lại phân tích AI**, hoặc
2. Bấm **ÁP DỤNG ML CHO VIDEO ĐÃ PHÂN TÍCH** trong tab kết quả

---

## 7. Liên quan trong mã nguồn

| Thành phần | File |
|------------|------|
| Logic REF (PASS/NEAR/FAIL) | `app.py` — `xu_ly_frame()`, khớp `dynamic_chuan` theo góc gần nhất |
| Khớp tư thế YouTube & nhận dạng BT1/2/3 | `reference_utils.py` — `detect_motion_subtype()`, `filter_reference_poses()`, `find_closest_reference_pose()` |
| Trích xuất góc từ video YouTube mẫu | `scripts/extract_youtube_reference.py` |
| Hằng số sai số G1/G2/G3 | `reference_utils.py` — `PHASE_ERROR`, `PHASE_UI_LABELS` |
| File góc chuẩn YouTube | `database/reference_codman.json`, `reference_gay.json`, `reference_day.json` |
| Chia 3 giai đoạn video bệnh nhân | `app.py` — `segment_frames()` |
| Huấn luyện & dự đoán ML | `pose_classifier_utils.py` |
| Hiển thị badge trên frame | `pose_classifier_utils.py` — `format_ml_display()`, `draw_ml_badge()` |
| Giải thích ngắn trên UI | `app.py` — expander “📖 Giải thích nhãn REF (PASS) và ML (%)” |

---

*Tài liệu cập nhật: tháng 6/2026 — Rehab AI Monitor*
