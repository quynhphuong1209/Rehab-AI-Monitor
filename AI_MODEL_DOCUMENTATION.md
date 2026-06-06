# BÁO CÁO CHI TIẾT VỀ MÔ HÌNH AI & PHƯƠNG PHÁP TÍNH TOÁN SINH HỌC
## HỆ THỐNG GIÁM SÁT TẬP LUYỆN PHỤC HỒI CHỨC NĂNG (REHAB AI MONITOR)

Tài liệu này giải thích chi tiết về mô hình AI nhận diện khung xương (**MediaPipe Pose**), cấu trúc các điểm mốc sinh học, thuật toán toán học để tính góc khớp vai/khuỷu và các chỉ số lượng hóa chất lượng bài tập phục hồi chức năng (PHCN).

---

## 1. MÔ HÌNH AI NHẬN DIỆN KHUNG XƯƠNG (MEDIAPIPE POSE)

Hệ thống sử dụng giải pháp **MediaPipe Pose** được phát triển bởi Google để theo dõi và trích xuất tọa độ khớp xương của bệnh nhân từ video theo thời gian thực.

```
                  11 (Vai trái)     12 (Vai phải)
                       *               *
                      / \             / \
                     /   \           /   \
         (Khuỷu L) 13 *   * 15      14 *   * 16 (Khuỷu R)
                    /       \        /       \
         (Cổ tay L) 15       * 17    16       * 18 (Cổ tay R)
                    |                 |
                  23 *               * 24 (Hông R)
                 (Hông L)
```

### 1.1. Cấu trúc 33 Điểm mốc Khớp (Pose Landmarks Topology)
MediaPipe Pose trả về tọa độ của **33 điểm mốc khớp xương** trong không gian 3D dạng $(x, y, z)$. Trong bài tập PHCN khớp vai, hệ thống tập trung khai thác các điểm mốc cốt lõi sau:
* **Khớp vai (Shoulder):** Mốc số **11** (Vai trái) và **12** (Vai phải).
* **Khớp khuỷu tay (Elbow):** Mốc số **13** (Khuỷu trái) và **14** (Khuỷu phải).
* **Khớp cổ tay (Wrist):** Mốc số **15** (Cổ tay trái) và **16** (Cổ tay phải).
* **Khớp hông (Hip):** Mốc số **23** (Hông trái) và **24** (Hông phải) - Dùng làm điểm tựa thẳng đứng để tính độ gập/dạng của cánh tay so với thân người.

### 1.2. Các cấp độ phức tạp của Mô hình (Model Complexity)
MediaPipe Pose cung cấp 3 tùy chọn mô hình thông qua tham số `model_complexity`, được tối ưu hóa riêng biệt:
* **Lite (Complexity 0):** Tốc độ xử lý nhanh nhất, tiêu tốn rất ít tài nguyên, thích hợp cho thiết bị di động cấu hình thấp hoặc trình duyệt web trực tiếp nhưng độ chính xác ở các khớp bị che khuất kém hơn.
* **Full (Complexity 1):** Cân bằng hoàn hảo giữa hiệu năng xử lý (FPS) và độ chính xác của tọa độ khớp.
* **Heavy (Complexity 2 - Mặc định):** Mô hình có số lượng tham số lớn nhất. Nó sử dụng các bộ lọc tích chập sâu để tự động ước lượng vị trí khớp xương bị che khuất (occlusion resolution) dựa trên cấu trúc giải phẫu người tập. Đây là mô hình được lựa chọn để đảm bảo tính chuẩn xác y khoa.

---

## 2. THUẬT TOÁN TÍNH TOÁN GÓC KHỚP SINH HỌC (BIOMECHANICAL ANGLES)

Để đánh giá biên độ vận động (ROM - Range of Motion) của khớp vai và khuỷu tay, hệ thống áp dụng các phép toán hình học vector trong không gian.

### 2.1. Phương pháp Tính góc giữa 3 Điểm khớp (Angle Between 3 Points)
Giả sử chúng ta cần tính góc tại khớp vai tạo bởi 3 điểm: Hông ($A$), Vai ($B$) và Khuỷu tay ($C$).

1. **Xác định tọa độ:**
   * $A = (x_a, y_a, z_a)$ (Hông)
   * $B = (x_b, y_b, z_b)$ (Vai)
   * $C = (x_c, y_c, z_c)$ (Khuỷu)
2. **Tính toán hai Vector liên kết:**
   * Vector từ Vai đến Hông: $\vec{u} = \vec{BA} = (x_a - x_b, y_a - y_b, z_a - z_b)$
   * Vector từ Vai đến Khuỷu: $\vec{v} = \vec{BC} = (x_c - x_b, y_c - y_b, z_c - z_b)$
3. **Áp dụng công thức Tích vô hướng (Dot Product):**
   * Tích vô hướng: $\vec{u} \cdot \vec{v} = (u_x \cdot v_x) + (u_y \cdot v_y) + (u_z \cdot v_z)$
   * Độ dài vector $\vec{u}$: $\|\vec{u}\| = \sqrt{u_x^2 + u_y^2 + u_z^2}$
   * Độ dài vector $\vec{v}$: $\|\vec{v}\| = \sqrt{v_x^2 + v_y^2 + v_z^2}$
4. **Tính góc $\theta$ (Đơn vị: Độ):**
   $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \cdot \|\vec{v}\|}\right) \times \frac{180}{\pi}$$

Thuật toán này được thực thi bằng thư viện **NumPy** để tối ưu hóa xử lý mảng tốc độ cao.

---

## 3. PHƯƠNG PHÁP ĐÁNH GIÁ CHẤT LƯỢNG LÂM SÀNG (ROM CLINICAL EVALUATION)

Độ chính xác của bài tập được lượng hóa bằng cách so sánh góc khớp thực tế của bệnh nhân với chuỗi dữ liệu góc chuẩn (Reference Trajectory) qua từng thời điểm.

### 3.1. Phân chia Ngưỡng sai số theo Giai đoạn PHCN (Dynamic Thresholds)
Hệ thống cho phép điều chỉnh ngưỡng sai số góc khớp tùy theo giai đoạn phục hồi của bệnh nhân:
* **Giai đoạn 1: Khởi đầu (Ngưỡng lệch ≤ 45°):** Phù hợp với bệnh nhân mới bắt đầu tập luyện, cơ vai còn cứng và đau. Chấp nhận sai số lớn.
* **Giai đoạn 2: Hồi phục (Ngưỡng lệch ≤ 30°):** Áp dụng cho bệnh nhân đã tập được 2-4 tuần, biên độ vận động cải thiện rõ rệt.
* **Giai đoạn 3: Chuẩn xác (Ngưỡng lệch ≤ 15°):** Áp dụng cho giai đoạn cuối chuẩn bị xuất viện. Đòi hỏi cử động chuẩn xác như người khỏe mạnh.

### 3.2. Tiêu chuẩn Phân loại Trạng thái Khung hình (Frame Status Classification)
Tại mỗi frame hình y tế, hệ thống phân loại trạng thái như sau:
* **ĐÚNG (PASS / CORRECT):** Khi sai số góc khớp vai và khớp khuỷu nhỏ hơn hoặc bằng ngưỡng của giai đoạn tập được cấu hình (ví dụ: $\Delta \theta \leq 15^\circ$ ở Giai đoạn 3).
* **GẦN ĐÚNG (NEAR PASS):** Khi sai số góc vượt quá ngưỡng cấu hình nhưng nhỏ hơn ngưỡng an toàn mở rộng (thường là $\text{Ngưỡng} + 15^\circ$).
* **SAI (FAIL / INCORRECT):** Khi cử động lệch chuẩn nghiêm trọng vượt quá tất cả các ngưỡng an toàn.

---

## 4. CÁC CHỈ SỐ THỐNG KÊ KHOA HỌC (RESEARCH METRICS)

Để phục vụ công tác Nghiên cứu khoa học (NCKH) của các Bác sĩ và Nghiên cứu viên, hệ thống tính toán các chỉ số thống kê toán học sâu:

### 4.1. Sai số tuyệt đối trung bình (MAE - Mean Absolute Error)
Đo lường độ lệch góc trung bình của cả phiên tập so với mẫu chuẩn:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\theta_{\text{patient}, i} - \theta_{\text{reference}, i}|$$
*Ý nghĩa:* MAE càng nhỏ thể hiện động tác của bệnh nhân càng bám sát quỹ đạo mẫu chuẩn.

### 4.2. Hệ số tương quan nội lớp (ICC - Intraclass Correlation Coefficient)
Đánh giá mức độ đồng thuận giải phẫu giữa quỹ đạo chuyển động của bệnh nhân và chuyên gia mẫu:
$$\text{ICC} = \frac{\text{Var}(\text{Between-subjects})}{\text{Var}(\text{Between-subjects}) + \text{Var}(\text{Within-subjects})}$$
*Ý nghĩa:* Chỉ số ICC tiến gần về $1.0$ ($> 0.75$) chứng minh bài tập có độ tin cậy và sự nhất quán động học cao, đạt tiêu chuẩn y khoa quốc tế.

### 4.3. Các chỉ số Precision, Recall và F1-Score
* **Precision (Độ chính xác):** Tỷ lệ số khung hình AI dự đoán bệnh nhân tập "Đúng" trên thực tế bác sĩ đánh giá đạt chuẩn.
* **Recall (Độ nhạy):** Tỷ lệ số khung hình tập đạt chuẩn thực tế được AI nhận diện thành công.
* **F1-Score (Điểm F1):** Trung bình điều hòa giữa Precision và Recall để lượng hóa hiệu suất tổng thể của mô hình phân loại:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
