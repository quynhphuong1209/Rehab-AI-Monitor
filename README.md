---
title: Rehab AI Monitor 2026
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.57.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🏥 Rehab AI Monitor (Clinical Ecosystem)

**Hệ thống giám sát tập luyện Phục hồi chức năng từ xa dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính - Giải pháp Clinical-Grade chuyên nghiệp.**

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue?style=for-the-badge)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Giới thiệu đề tài & Đặt vấn đề (Introduction & Rationale)

### Đặt vấn đề (Problem Statement)
Trong những năm gần đây, cùng với sự gia tăng của các bệnh lý cơ xương khớp, chấn thương thể thao và đột quỵ, nhu cầu phục hồi chức năng (PHCN) trên toàn thế giới ngày càng tăng cao. Theo Tổ chức Y tế Thế giới (WHO), hiện có khoảng 2,4 tỷ người cần ít nhất một hình thức phục hồi chức năng, chiếm gần một phần ba dân số toàn cầu (1, 2). Tại Việt Nam, theo Hội Phục hồi chức năng Việt Nam (2023), có khoảng 7,06% dân số từ 2 tuổi trở lên là người khuyết tật, trong đó phần lớn cần được can thiệp PHCN để cải thiện chức năng và tái hòa nhập cộng đồng. Đồng thời, tỷ lệ người cao tuổi chiếm 11,9% dân số và đang tăng nhanh, kéo theo sự gia tăng các bệnh lý thoái hóa xương khớp, rối loạn vận động và bệnh lý thần kinh (3). Mặc dù nhu cầu PHCN lớn, song năng lực cung cấp dịch vụ này tại Việt Nam vẫn còn hạn chế. Theo thống kê của Bộ Y tế (2023), trung bình 10.000 người dân chỉ có 0,25 nhân viên phục hồi chức năng, thấp hơn đáng kể so với khuyến nghị của WHO là 0,5–1 người/10.000 dân (4). Ngoài ra, chỉ khoảng 40% người bệnh có khả năng tiếp cận đầy đủ dịch vụ PHCN do hạn chế về nhân lực, cơ sở vật chất và điều kiện địa lý (5). Thực tế này khiến nhiều bệnh nhân phải tự tập luyện tại nhà sau khi xuất viện mà thiếu sự giám sát chuyên môn, dẫn đến nguy cơ tập sai động tác, giảm hiệu quả điều trị và kéo dài thời gian hồi phục.

Trước thực trạng đó, việc ứng dụng công nghệ Trí tuệ nhân tạo (Artificial Intelligence – AI) và Thị giác máy tính (Computer Vision – CV) vào giám sát tập luyện phục hồi chức năng từ xa được xem là xu hướng tất yếu. Trên thế giới, nhiều hệ thống AI hỗ trợ PHCN đã được thử nghiệm hoặc triển khai tại các quốc gia như Hoa Kỳ, Nhật Bản, Hàn Quốc với kết quả tích cực. Nghiên cứu của Ali Abedi và cộng sự (2024) cho thấy việc tích hợp AI vào chương trình phục hồi từ xa giúp nâng cao chất lượng đánh giá bài tập và cá nhân hóa phác đồ điều trị, góp phần cải thiện kết quả lâm sàng so với phương pháp truyền thống (6). Tại Việt Nam, một số đơn vị tiên phong như Trung tâm ASINA đã triển khai ứng dụng AI trong phục hồi cơ xương khớp, giúp bệnh nhân tập luyện từ xa một cách hiệu quả và tiện lợi (7). Bên cạnh đó, Bệnh viện C Đà Nẵng cũng đã tích hợp AI và công nghệ thực tế ảo (Virtual Reality – VR) vào quy trình điều trị, mang lại chất lượng sống tốt hơn cho hàng trăm bệnh nhân (8). Tuy nhiên, hiện nay chưa có nhiều hệ thống trong nước tích hợp đầy đủ khả năng nhận diện tư thế vận động theo thời gian thực, phản hồi trực quan, đồng thời lưu trữ và phân tích dữ liệu tập luyện phục vụ cho việc theo dõi tiến trình phục hồi của bác sĩ. Vì vậy, việc phát triển một nền tảng ứng dụng thông minh có khả năng giám sát, hỗ trợ và kết nối giữa bệnh nhân – bác sĩ – kỹ thuật viên là nhu cầu cấp thiết trong bối cảnh chăm sóc sức khỏe từ xa ngày càng được chú trọng. 

Tại khoa Phục hồi chức năng Bệnh viện Đa khoa Phạm Ngọc Thạch, nhu cầu theo dõi và hỗ trợ người bệnh luyện tập ngày càng tăng, đặc biệt với các trường hợp luyện tập lâu dài tại nhà. Tuy nhiên, hiện nay việc giám sát chủ yếu thực hiện trực tiếp tại bệnh viện, khi về nhà người bệnh tự tập theo video hoặc tài liệu hướng dẫn mà không có sự kiểm soát chuyên môn. Điều này dẫn đến nguy cơ tập sai động tác, giảm hiệu quả điều trị và khó theo dõi tiến trình phục hồi. Tại bệnh viện hiện nay vẫn chưa có nghiên cứu hay hệ thống nào ứng dụng Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision) để giám sát tập luyện từ xa khiến việc thu thập dữ liệu, đánh giá kết quả và cải tiến phác đồ điều trị còn hạn chế. Xuất phát từ thực tiễn trên, nhóm nghiên cứu chúng tôi quyết định thực hiện đề tài: **“Phát triển mô hình thử nghiệm giám sát tập luyện Phục hồi chức năng từ xa dựa trên Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision) tại Bệnh viện Đa khoa Phạm Ngọc Thạch – Trường Đại học Y tế Công cộng (2025–2026)”**.

### 🎯 Mục tiêu nghiên cứu (Research Objectives)
*   **Mục tiêu 1:** Xây dựng mô hình nhận diện và đánh giá 2-3 động tác phục hồi chức năng cơ bản (ví dụ: giơ tay ngang vai, co gối, xoay cổ tay) bằng công nghệ thị giác máy tính (pose estimation).
*   **Mục tiêu 2:** So sánh độ chính xác của mô hình với đánh giá thủ công (ví dụ: góc khớp, số lần lặp) trên một tập dữ liệu nhỏ (do nhóm tự quay hoặc dùng dữ liệu mở).


## ✨ Tính năng nổi bật (v3.2 Updated)
- 💎 **Thẩm mỹ Lâm sàng:** Giao diện sử dụng font chữ 'Times New Roman' chuẩn mực, thiết kế card-based hiện đại với hiệu ứng Glassmorphism.
- 🌓 **Đồng bộ Theme:** Hỗ trợ hoàn hảo chế độ Sáng (Light) và Tối (Dark) với sự chuyển đổi mượt mà, không lỗi tương phản.
- 📱 **Mobile-First Optimization:** Hệ thống Tab được tối ưu hóa toàn diện cho di động, đảm bảo chữ không bị tràn, hiển thị đầy đủ và hỗ trợ cuộn ngang chuyên nghiệp.
- 🩺 **Luồng liên lạc khép kín:** Bệnh nhân khai báo triệu chứng (VAS) -> Chuyên gia nhận xét lâm sàng -> Kết nối kết quả AI.
- 🚀 **Điều hướng Auto-Tab:** Tự động chuyển Tab thông minh bằng JavaScript khi chọn video để đánh giá, tối ưu hóa thao tác người dùng.
- 📊 **Phân tích Đa chiều (Plotly Analytics):**
  - **ROM Trend & Boxplot:** Đánh giá xu hướng góc khớp và độ biến động chuyển động qua từng phiên.
  - **Radar Chart (7 Chỉ số AI):** Lượng hóa hiệu suất mô hình qua 7 tham số cốt lõi: Accuracy, MAE, RMSE, ICC, F1-Score, Precision, Recall.
- 🦾 **Phân tích 3 Giai đoạn PHCN:** Bảng đối sánh kết quả tự động tại các ngưỡng sai số góc khớp $\pm 45°$, $\pm 30°$, và $\pm 15°$.
- 📁 **Xuất báo cáo Hợp nhất & Lazy ZIP:**
  - Xuất dữ liệu tọa độ CSV và biểu đồ dạng PNG trực tiếp.
  - Tải file ZIP ảnh phân tích bằng cơ chế "lười" (chỉ nén khi click), giúp chống lỗi tràn bộ nhớ (OOM).
- 🩺 **Đạo đức & Thông tin Nghiên cứu:** Bioethics Panel hiển thị thông tin PIS và các thẻ liên hệ chuyên biệt cho NCV và Hội đồng Đạo đức (IRB).
- 📱 **Sidebar Phẳng (Flattened):** Cấu trúc Sidebar mật độ cao, truy cập nhanh thông tin bệnh nhân và khai báo triệu chứng.

## 🗺️ Cấu trúc Tab Điều hướng (Role-based)
Hệ thống tự động thay đổi cấu trúc dựa trên vai trò người dùng:
- **Bệnh nhân:** Tập luyện (Xem video mẫu, upload video tập, xem kết quả), Khai báo triệu chứng & VAS, Xem phác đồ của bác sĩ, Lịch nhắc nhở (Schedules), Đạo đức & Thông tin nghiên cứu (Consent).
- **Bác sĩ / KTV:** Quản lý bệnh nhân, Giao diện quản lý & Phê duyệt video (Trình xem video kép, JavaScript Auto-Tab), Bộ đánh giá lâm sàng chuyên môn (Ground Truth Entry), Quản lý phác đồ.
- **Nghiên cứu viên:** Cấu hình tham số mô hình AI, Phân tích sâu & Trích xuất tọa độ (Xuất CSV/JSON), Phân tích đa chiều (ROM Trend, Boxplot, Radar Chart), Bảng đối sánh 3 giai đoạn PHCN, Đồng bộ Ground Truth từ Bác sĩ.
- **Quản trị viên:** Bộ Metric Cards tổng quan, Biểu đồ thống kê trực quan (Cơ cấu vai trò, bài tập phổ biến), Bảng quản trị cốt lõi (hợp nhất mọi thông tin bệnh nhân, AI, bác sĩ), Nhật ký hoạt động toàn hệ thống (Admin Log - Xuất CSV), Dọn dẹp & Reset hệ thống.

<!-- CLINICAL_FINDINGS_START -->

## Báo cáo nghiên cứu

Báo cáo lâm sàng chi tiết đã được gỡ khỏi README để tránh đưa dữ liệu nhạy cảm, đường dẫn đăng nhập, hoặc nội dung bệnh án vào tài liệu công khai.

Các báo cáo sinh tự động phải được ghi vào `docs/generated/`, rà soát PII thủ công trước khi commit, và không được chứa token, email, tên thật bệnh nhân, bệnh sử chi tiết, hoặc liên kết đăng nhập.

<!-- CLINICAL_FINDINGS_END -->

## 🏗️ Kiến trúc hệ thống (Architecture Overview)

Hệ thống được thiết kế theo mô hình kiến trúc phân lớp tối ưu hiệu năng chạy trên các nền tảng đám mây CPU-only (như Hugging Face Spaces). Dưới đây là sơ đồ và luồng hoạt động chi tiết:

### Sơ đồ luồng hoạt động (Data & Control Flow)

```mermaid
graph TD
    %% Khai báo Style cho các khối
    classDef client fill:#00c6ff,stroke:#333,stroke-width:1px,color:#000;
    classDef logic fill:#ffd700,stroke:#333,stroke-width:1px,color:#000;
    classDef ai fill:#00ff87,stroke:#333,stroke-width:1px,color:#000;
    classDef data fill:#ff4757,stroke:#333,stroke-width:1px,color:#fff;

    %% Các thành phần hệ thống
    Patient[Bệnh nhân: Tập & Khai báo VAS]:::client
    Doctor[Bác sĩ: Quản lý & Đánh giá]:::client
    Researcher[Nghiên cứu viên: Phân tích & AI]:::client
    
    UI[Giao diện Web Streamlit - Custom CSS & JS Engine]:::logic
    
    Pass1[Pass 1: Trích xuất landmarks và tính toán góc gốc]:::ai
    MP[MediaPipe Pose - Heavy / Full / Lite]:::ai
    Seg[Phân đoạn cử động - np.convolve & valleys finder]:::logic
    
    Pass2[Pass 2: Vẽ khung xương động & Tính sai số động]:::ai
    Pydub[Trộn âm thanh phản hồi VAS - Pydub]:::logic
    FFmpeg[Đóng gói video H.264 - FFmpeg ultrafast]:::logic
    
    Heal[Cơ chế Auto-Healing: Sửa lỗi video cũ mp4v sang H.264]:::logic
    
    JSON[Cơ sở dữ liệu: JSON & CSV Logs]:::data
    HF[Đồng bộ bất đồng bộ lên Hugging Face Dataset]:::data

    %% Các liên kết luồng
    Patient --> UI
    Doctor --> UI
    Researcher --> UI
    
    UI --> Pass1
    Pass1 --> MP
    MP --> Seg
    Seg --> Pass2
    Pass2 --> Pydub
    Pass2 --> FFmpeg
    
    UI --> Heal
    Heal --> FFmpeg
    
    FFmpeg --> JSON
    JSON --> HF
```

### Các thành phần chính trong kiến trúc:

1. **Luồng xử lý Video 2-Pass tối ưu bộ nhớ:**
   * **Pass 1 (Trích xuất dữ liệu thô):** Đọc từng khung hình video từ `cv2.VideoCapture`, chuẩn hóa kích thước (resize) và xoay chiều phù hợp. MediaPipe Pose chạy trên ảnh RGB để lấy 33 điểm landmarks, sau đó tính toán góc vai và góc khuỷu mà không vẽ hoặc ghi file nhằm tiết kiệm RAM tối đa.
   * **Phân đoạn Giai đoạn tự động (Segmentation):** Áp dụng bộ lọc mượt tích chập (`np.convolve`) lên chuỗi tín hiệu góc khớp để khử nhiễu. Thuật toán tìm điểm cực tiểu (valleys) để chia video bệnh nhân thành 3 giai đoạn cử động (Giai đoạn 1 bắt đầu giơ tay, Giai đoạn 2 dạng sai số vừa, Giai đoạn 3 chuẩn xác dần).
   * **Pass 2 (Vẽ đè & Gộp đa phương tiện):** Sử dụng landmarks đã trích xuất ở Pass 1 để vẽ khung xương động, vòng cung góc khớp trực tiếp lên frame. Sai số động được áp dụng theo phân đoạn (GĐ1: 45°, GĐ2: 30°, GĐ3: 15°).
   
2. **Hệ thống phản hồi âm thanh & Đóng gói Video:**
   * **Voice Feedback Engine:** Trích xuất các khoảnh khắc chuyển đổi trạng thái (Đúng, Gần đúng, Sai). Sử dụng `pydub` để nối ghép các file âm thanh chỉ dẫn. Hệ thống tự động giới hạn tối đa 40 sự kiện âm thanh để tránh tràn RAM (Out of Memory - OOM).
   * **FFmpeg H.264 Transcoding:** Sử dụng bộ mã hóa `libx264` cùng với cấu hình `-preset ultrafast` và `-crf 24` để nén video thô `mp4v` thành định dạng H.264 chuẩn web, đảm bảo video hiển thị mượt mà trên mọi thiết bị di động mà không bị lag/buffering.

3. **Cơ chế Tự sửa lỗi thông minh (Auto-Healing Engine):**
   * Tích hợp trực tiếp vào hàm `render_video`. Khi phát hiện người dùng tải lại kết quả của các phiên tập cũ có video định dạng `mp4v` không chơi được, hệ thống sẽ tự động kích hoạt `ffmpeg` ngầm để chuyển đổi sang H.264 chuẩn, đồng thời tự động cập nhật lại cơ sở dữ liệu `video_list.json` mà không làm gián đoạn trải nghiệm của người dùng.

4. **Đồng bộ hóa dữ liệu đám mây bất đồng bộ (Async Cloud Sync):**
   * Sử dụng luồng chạy nền (`threading.Thread` độc lập) để tải dữ liệu CSV tọa độ và các file video thành phẩm lên Hugging Face Dataset. Cơ chế này giúp giữ cho luồng giao diện (UI) chính của Streamlit luôn mượt mà, không bị khóa cứng (blocking) khi truyền tải file lớn.

## 🤖 Hướng dẫn cấu hình & Lựa chọn mô hình AI (Model Configurations)

Để tối ưu hóa độ chính xác hoặc tốc độ xử lý tùy theo năng lực phần cứng (đặc biệt khi chạy trên các môi trường CPU Cloud hạn chế như Hugging Face Spaces), Nghiên cứu viên có thể tùy chỉnh cấu hình các tham số phân tích AI trực tiếp tại **Sidebar bên trái** trước khi nhấn phân tích:

### 1. Phân loại mô hình AI (Model Type)
Hệ thống tích hợp 3 phiên bản mô hình Pose Estimation từ **MediaPipe**:
* **MediaPipe Heavy (Khuyến nghị cho lâm sàng):** Có độ chính xác cao nhất về định vị các điểm landmarks khớp vai/khuỷu tay, giảm thiểu tối đa hiện tượng rung/trượt tọa độ do góc quay camera. Phù hợp nhất cho việc đánh giá lâm sàng cần độ tin cậy tuyệt đối.
* **MediaPipe Full (Tiêu chuẩn):** Cân bằng tốt giữa tốc độ xử lý và độ chính xác, thích hợp khi kiểm tra nhanh.
* **MediaPipe Lite (Siêu nhẹ):** Tối ưu hóa tối đa về hiệu năng CPU. Phù hợp nhất khi chạy thử nghiệm nhanh hoặc trên các dòng máy tính/thiết bị có cấu hình yếu.

### 2. Các tham số tối ưu hiệu năng chạy nền
* **Tốc độ xử lý (Skip Frames):**
  * **0 (Mặc định)**: Quét và phân tích toàn bộ khung hình trong video (độ chính xác cao nhất).
  * **2** hoặc **4**: Bỏ qua 2 hoặc 4 khung hình trong mỗi bước quét. Giúp tăng tốc độ xử lý của mô hình AI gấp **3 - 5 lần** (rút ngắn thời gian xử lý video dài xuống còn vài chục giây) mà vẫn đảm bảo giữ nguyên được các điểm cực trị lâm sàng.
* **Độ phân giải đầu vào (Resize Width):**
  * Hỗ trợ nén chiều rộng khung hình đầu vào về mức `360px` hoặc `720px` trước khi nạp dữ liệu vào mô hình AI. Giúp giảm tải đáng kể dung lượng bộ nhớ RAM tiêu thụ và tránh lỗi tràn RAM (OOM - Out of Memory) trên máy chủ Cloud CPU.
* **Ngưỡng tin cậy (Confidence Threshold):**
  * Đặt mức tối thiểu (mặc định `0.5`) để lọc bỏ các khung hình bị che khuất hoặc các điểm khớp nhận diện kém tự tin, đảm bảo dữ liệu vẽ biểu đồ góc khớp sạch nhất.

## 📁 Cấu trúc thư mục dự án (Directory Structure)

Dưới đây là sơ đồ phân loại toàn bộ tệp tin trong dự án giúp bạn dễ dàng chủ động quản lý và bảo trì:

```
Rehab-AI-Monitor/
│
├── 🌐 Chương trình chạy Web chính
│   ├── app.py                     # File chạy chính (Frontend Streamlit + Backend Python)
│   └── .streamlit/
│       └── config.toml            # Cấu hình cổng mạng, theme, tối ưu hóa của Streamlit
│
├── 🖼️ Tài nguyên hình ảnh & Logo (Thư mục assets/)
│   └── assets/
│       ├── abc1.png                   # Logo Đại học Y tế Công cộng (HUPH)
│       └── logo_data_science_sm.png   # Logo khoa Khoa học dữ liệu HUPH
│
├── 📂 Thư mục tài liệu hướng dẫn & Báo cáo (docs/)
│   ├── README_UI.md               # Tài liệu thuyết minh chi tiết về thiết kế giao diện UI/UX
│   ├── BAO_CAO_CHI_TIET.md        # Báo cáo chuyên sâu về mã nguồn, giải thuật lâm sàng & RAM
│   ├── TECHNICAL_DOCUMENTATION.md # Tài liệu kỹ thuật phân tích sâu cấu trúc Front-End & Back-End
│   └── AI_MODEL_DOCUMENTATION.md  # Tài liệu giải thích mô hình AI, công thức toán lý thuyết góc khớp
│
├── 📝 Hướng dẫn khởi chạy chính
│   └── README.md                  # Hướng dẫn chung về cách cài đặt và chạy dự án
│
├── 💾 Cơ sở dữ liệu JSON (Thư mục database/)
│   └── database/
│       ├── users.json                 # Danh sách tài khoản người dùng và mật khẩu băm bảo mật
│       ├── patient_symptoms.json      # Triệu chứng lâm sàng và mức độ đau VAS của bệnh nhân
│       ├── doctor_evaluations.json    # Chẩn đoán lâm sàng (Ground Truth) và nhận xét của Bác sĩ
│       ├── schedules.json             # Lịch nhắc nhở luyện tập của bệnh nhân
│       ├── video_list.json            # Quản lý siêu dữ liệu video, kết quả phân tích góc và sai số AI
│       ├── lich_su_tap_luyen.json     # Lịch sử và tiến trình tập luyện của bệnh nhân
│       ├── reference_codman.json      # Dữ liệu góc chuẩn cho bài tập Codman Pendulum
│       ├── reference_gay.json         # Dữ liệu góc chuẩn cho bài tập gậy khớp vai
│       └── reference_day.json         # Dữ liệu góc chuẩn cho bài tập dây kháng lực
│
├── 📂 Thư mục chứa dữ liệu Media
│   ├── patient_uploads/           # Nơi lưu trữ video gốc do bệnh nhân tải lên
│   └── processed_results/         # Nơi lưu kết quả video/ảnh đã vẽ khung xương khớp từ AI
│
├── ⚙️ Cấu hình môi trường & Deploy
│   ├── requirements.txt           # Danh sách thư viện Python cần cài đặt (numpy, mediapipe...)
│   ├── packages.txt               # Thư viện hệ thống cài cho Linux khi deploy lên cloud (ffmpeg...)
│   ├── Dockerfile                 # Cấu hình Container để chạy ứng dụng tự động
│   └── runtime.txt                # Khai báo phiên bản Python chạy trên Cloud (Python 3.10)
│
└── 🛠️ Công cụ & Batch Scripts hỗ trợ (Thư mục scripts/)
    └── scripts/
        ├── reset_data.py              # Script dọn dẹp sạch sẽ video rác và reset cơ sở dữ liệu
        ├── fix_plotly_v2.py           # Script nhỏ sửa lỗi hiển thị của biểu đồ Plotly
        ├── push_code.bat              # Batch script trên Windows dùng để lưu nhanh code lên GitHub
        └── push_to_git.bat            # Batch script đẩy code dự phòng lên GitHub
```

## 🛠️ Công nghệ sử dụng
- **AI Core:** MediaPipe (Pose), OpenCV, FFmpeg (Xử lý đa định dạng video MOV/MP4)
- **Runtime:** Python 3.10 (Khuyến nghị để đảm bảo tương thích MediaPipe & Docker)
- **Framework:** Streamlit (Custom CSS/JS & WebRTC)
- **Data:** Pandas, NumPy, JSON Persistence (Lưu trữ bền vững)
- **Visualization:** Plotly Professional Charts (Heatmaps, Progress Charts)

## 🚀 Chạy ứng dụng
```bash
# Yêu cầu Python 3.10
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍🏫 Nhóm thực hiện & Hướng dẫn
- **Giảng viên hướng dẫn:** 
  1. Chuyên gia khoa học dữ liệu
  2. Chuyên gia lâm sàng
- **Chủ nhiệm đề tài:** Nhóm nghiên cứu Rehab AI Monitor
- **Thành viên nhóm nghiên cứu:** Danh sách thành viên được quản lý trong hồ sơ nội bộ, không đưa vào README công khai.
- **Đơn vị phối hợp:** Đại học Y tế Công cộng - Bệnh viện Đa khoa Phạm Ngọc Thạch.

---
© 2025-2026 Rehab AI Monitor Team.

