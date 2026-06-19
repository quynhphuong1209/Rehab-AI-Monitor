# Báo cáo rà soát bổ sung — Bugs & Vấn đề còn tồn tại (N01–N20)

Ngày rà soát: 17/06/2026  
Phạm vi bổ sung: Xác minh lại tất cả F01–F40 từ báo cáo gốc + quét sâu thêm các vùng chưa được đề cập, bao gồm: HTTP server nội bộ, `.gitignore`/data exposure, `users.json`, path traversal, upload flow chi tiết, locking pattern, gTTS/WebRTC, sync script PII, ffmpeg Popen, và các vấn đề mới phát hiện.

---

## 1. Tóm tắt đánh giá báo cáo gốc

Tất cả **F01–F40** đã được xác nhận **còn tồn tại** trong codebase hiện tại. Không có lỗi nào đã được fix.

Số liệu bổ sung đo được trong lần rà soát này:

| Metric | Giá trị |
|---|---|
| `unsafe_allow_html=True` | **139 lần** (báo cáo cũ ước "nhiều") |
| `except:` trần | **105** (khớp báo cáo cũ) |
| `except Exception:` | **131** (báo cáo cũ nói 217 — có thể tính cả nhánh lồng nhau) |
| `threading.Thread(...)` | **12 chỗ** rải rác |
| `save_data(VIDEOS_FILE, ...)` không có lock | **13 chỗ** |
| `save_data(EVALUATIONS_FILE, ...)` không có lock | **8 chỗ** |
| `doc_lock_save_data(VIDEOS_FILE, ...)` | **2 chỗ** (dùng đúng) |
| `html.escape()` trong toàn repo | **0** |
| `logging` được dùng thực sự | **5 dòng** (chủ yếu suppress Streamlit warning) |

---

## 2. Mức độ ưu tiên — Bugs bổ sung

| ID | Mức | Vấn đề |
|---|---|---|
| N01 | **Critical** | HTTP server phục vụ toàn project root + CORS wildcard |
| N02 | **Critical** | `users.json` commit tên/hash bệnh nhân thật vào git |
| N07 | **Critical** | `.gitignore` exceptions cho phép commit toàn bộ dữ liệu lâm sàng |
| N20 | **Critical** | Script tự ghi link login bypass vào README |
| N03 | **High** | Tất cả bệnh nhân dùng chung email NCV → password reset takeover |
| N04 | **High** | `_auth_lookup_key()` fuzzy matching gây username collision |
| N05 | **High** | Upload không reject sớm trước `getbuffer()` → OOM |
| N06 | **High** | `ffmpeg -threads 0` không giới hạn CPU |
| N08 | **High** | Tên thật bệnh nhân hard-code trong script |
| N09 | **High** | Dữ liệu lâm sàng thật hard-code trong Python source |
| N11 | **High** | 13/15 `save_data(VIDEOS_FILE)` không có lock |
| N13 | **High** | Popen ffmpeg không có timeout mặc định |
| N17 | **High** | `except: pass` trong 100+ chỗ IO quan trọng |
| N10 | **Medium/High** | HTTP server không có auth token |
| N12 | **Medium** | gTTS gọi Internet trong background thread |
| N14 | **Medium** | Không giới hạn đăng ký tài khoản mới |
| N15 | **Medium** | HF_DATASET_ID hard-code fallback |
| N16 | **Medium** | `while True` thread không có stop condition |
| N18 | **Medium** | `schedules.json` vẫn là `{}` (F13 chưa fix) |
| N19 | **Medium** | WebRTC không rõ bảo mật/privacy |

---

## 3. Phát hiện chi tiết

### N01. HTTP server nội bộ phục vụ toàn bộ project root — path traversal và CORS wildcard

**Mức độ: Critical/High**  
**Vị trí:** `app.py:1556–1671`

App khởi động một `ThreadingTCPServer` (`SimpleHTTPRequestHandler`) để stream video qua range request. Server này:

- Bind tại `127.0.0.1:{8765–8800}`, nhưng được gọi từ `<video src>` trong trình duyệt — tức là browser gửi request qua iframe Streamlit.
- Gửi header `Access-Control-Allow-Origin: *` (dòng 1629), cho phép bất kỳ origin nào đọc response.
- `serve_root = os.path.abspath(".")` — phục vụ **toàn bộ thư mục project**, bao gồm `database/`, `utils/`, `.streamlit/`, `app.py`, `requirements.txt`, ...
- Guard path traversal (`rel.startswith('..')`) chỉ giới hạn ở `rel.count('..') > 3` — **ít nhất 3 bậc `../` vẫn được phục vụ**.
- `SimpleHTTPRequestHandler.translate_path()` đã normalize `../`, nhưng không ngăn chặn đầy đủ nếu có symlink hoặc path encoded.

**Tác động:**

- Bất kỳ trang web nào có thể fetch `http://127.0.0.1:876x/database/users.json` nếu biết cổng, và nhờ CORS `*`, browser sẽ không chặn.
- Lộ `users.json` (chứa SHA-256 hash password), `doctor_evaluations.json`, `patient_symptoms.json`.
- Trong môi trường HF Space, server bind `127.0.0.1` nên không bị expose ra ngoài trực tiếp — nhưng nếu có SSRF, hoặc khi run local, rủi ro rất cao.

**Khuyến nghị:**

- Whitelist chỉ các thư mục được phục vụ: `patient_uploads/`, `processed_results/`.
- Kiểm tra `os.path.realpath(path).startswith(serve_root)` trước khi mở file.
- Bỏ `Access-Control-Allow-Origin: *`; chỉ cho phép `http://localhost:*`.
- Xem xét dùng Streamlit static serving chính thức thay vì tự viết HTTP server.

---

### N02. `users.json` commit vào repo chứa SHA-256 hash của mật khẩu thật bệnh nhân

**Mức độ: Critical/Privacy**  
**Vị trí:** `database/users.json:132–159`

File `users.json` trong repo hiện chứa **4 bệnh nhân thật** (Hoàng Hạnh Nguyên, Nguyễn Thị Nga, Vũ Thị Hòa, Cao Thị Thường) với hash SHA-256 password và **đều dùng chung email `2211090031@studenthuph.edu.vn`**. Đây là dữ liệu y tế thật, không phải demo.

Ngoài ra, `.gitignore` có rule `!database/users.json` — tức là file này **được commit vào git** và có thể đã lên GitHub/HF.

**Tác động:**

- Hash SHA-256 của mật khẩu bệnh nhân có thể bị crack offline (rainbow table hoặc dictionary attack vì không có salt).
- Tên thật của 4 bệnh nhân + email bị commit vào repo — vi phạm quyền riêng tư nghiêm trọng.
- Tất cả tài khoản bệnh nhân này đang dùng email của một người (NCV) — dữ liệu gian lận hoặc sai.

**Khuyến nghị:**

- Ngay lập tức đổi password bệnh nhân và revoke các hash đã lộ.
- Xóa `!database/users.json` khỏi `.gitignore`, thêm `database/users.json` vào `.gitignore`.
- Thêm `database/` (trừ schema/reference) vào `.gitignore` production.
- Chạy `git filter-repo` hoặc `git bfg` để xóa lịch sử commit.
- Dùng dữ liệu giả (fake) cho demo/dev.

---

### N03. Tất cả bệnh nhân trong `users.json` dùng chung một email của NCV

**Mức độ: High/Data Integrity**  
**Vị trí:** `database/users.json:133, 141, 148, 155`

```json
"Hoàng Hạnh Nguyên": { "email": "2211090031@studenthuph.edu.vn" },
"Nguyễn Thị Nga":    { "email": "2211090031@studenthuph.edu.vn" },
"Vũ Thị Hòa":        { "email": "2211090031@studenthuph.edu.vn" },
"Cao Thị Thường":     { "email": "2211090031@studenthuph.edu.vn" }
```

`2211090031@studenthuph.edu.vn` là email của NCV Đinh Lê Quỳnh Phương.

**Tác động:**

- Password reset flow (F03) sẽ cho phép NCV reset password của cả 4 bệnh nhân vì email khớp.
- Bất kỳ ai biết email NCV này đều có thể chiếm toàn bộ tài khoản bệnh nhân.
- Dữ liệu nghiên cứu không hợp lệ nếu bệnh nhân không có email riêng.

**Khuyến nghị:**

- Gán email thật hoặc placeholder riêng cho từng bệnh nhân (hoặc để trống).
- Phân biệt rõ email người dùng và email liên lạc nghiên cứu.
- Kiểm tra tính duy nhất của email khi tạo/reset tài khoản.

---

### N04. `_auth_lookup_key()` tìm kiếm case-insensitive và theo `full_name` — rủi ro username collision

**Mức độ: High**  
**Vị trí:** `app.py:5039–5053`

Hàm tra cứu username khi đăng nhập chấp nhận:
1. Exact match key.
2. Casefold match với key.
3. Casefold match với `full_name` của user.

Điều này nghĩa là nếu một user có full_name trùng với username của user khác (hoặc gần giống khi casefold), có thể login vào sai tài khoản. Ví dụ: username `ADMIN` sẽ match `admin`. Username `Doctor 1` (full_name) sẽ match `doctor1` (key).

**Tác động:**

- Nếu attacker tạo tài khoản với full_name giống username admin/NCV, có thể gây confusion trong tra cứu.
- Với dữ liệu người dùng Unicode (tên tiếng Việt), casefold có thể hoạt động không đúng với một số chuỗi.

**Khuyến nghị:**

- Chỉ match theo key chính xác sau khi normalize Unicode.
- Tách `full_name` lookup khỏi auth lookup — full_name chỉ dùng để hiển thị.

---

### N05. File upload ghi vào `getbuffer()` trước khi kiểm tra kích thước — OOM không được ngăn chặn sớm

**Mức độ: High**  
**Vị trí:** `app.py:19203, 19233, 19286`

Ở dòng 19203, app chỉ hiển thị thông báo file size sau khi file đã được uploader nhận. Nhưng không có reject sớm trước `getbuffer()`:

```python
# Dòng 19286 — ghi toàn bộ buffer vào disk
f.write(file_upload.getbuffer())
```

`file_upload.size` được đọc ở dòng 19203 để hiển thị, nhưng **không có guard `if file_upload.size > MAX_FILE_SIZE_BYTES: return`** trước khi gọi `getbuffer()`.

**Tác động:**

- File 10 GB hợp lệ theo config hiện tại sẽ bị đọc vào RAM hoàn toàn trước khi ghi xuống disk.
- Streamlit worker OOM, trắng trang hoặc crash Space.

**Khuyến nghị:**

- Thêm `if file_upload.size > MAX_FILE_SIZE_BYTES: st.error(...); st.stop()` ngay khi `file_upload is not None`.
- Giảm `MAX_FILE_SIZE_MB` xuống 200–500 MB thực tế.
- Xem xét dùng chunked write thay vì `getbuffer()`.

---

### N06. `ffmpeg -threads 0` được dùng trong nhiều chỗ — không giới hạn CPU

**Mức độ: High**  
**Vị trí:** `app.py:401, 19320`

Lệnh ffmpeg trong cả transcode khi phân tích lẫn convert upload đều dùng `-threads 0` (auto = dùng tất cả CPU core). Trong môi trường HF Space 2 vCPU, điều này làm server đơ nếu có nhiều job đồng thời.

**Tác động:**

- Một upload lớn + một phân tích nền có thể chiếm 100% CPU.
- Các request khác (kể cả auth, load trang) bị stall.

**Khuyến nghị:**

- Đặt `-threads 2` hoặc `-threads 1` cho transcode nền.
- Queue ffmpeg jobs, không chạy song song không giới hạn.

---

### N07. `.gitignore` có rule `!database/users.json`, `!database/video_list.json`... — dữ liệu nhạy cảm được commit có chủ đích

**Mức độ: Critical/Privacy**  
**Vị trí:** `.gitignore:17–27`

```gitignore
*.json
!database/video_list.json
!database/doctor_evaluations.json
!database/users.json
!database/patient_symptoms.json
!database/lich_su_tap_luyen.json
!database/schedules.json
!database/research_data.json
```

Tất cả các file JSON chứa dữ liệu lâm sàng nhạy cảm đang được **chủ động commit vào git**. Kết hợp với F40 (thiếu ranh giới PII), đây là vi phạm quyền riêng tư nghiêm trọng.

**Tác động:**

- `doctor_evaluations.json` chứa đánh giá lâm sàng đã được commit — có thể tìm thấy trong lịch sử git.
- `patient_symptoms.json` chứa triệu chứng bệnh nhân thật.
- Nếu repo public (HF Space), dữ liệu bệnh nhân có thể đã bị lộ từ trước.

**Khuyến nghị:**

- Xóa tất cả các `!database/*.json` exceptions khỏi `.gitignore`.
- Chỉ giữ `database/reference_*.json` và `database/database_schema.md` trong git.
- Chạy `git filter-repo` để xóa lịch sử.
- Thêm pre-commit hook kiểm tra không commit file JSON chứa PII.

---

### N08. `sync_data_and_report.py` hard-code tên thật bệnh nhân trong hàm `anonymize_name()`

**Mức độ: High/Privacy**  
**Vị trí:** `scripts/sync_data_and_report.py:249–262`

```python
mapping = {
    "Hoàng Hạnh Nguyên": "Bệnh nhân 1 (BN1)",
    "Nguyễn Thị Nga":    "Bệnh nhân 2 (BN2)",
    "Vũ Thị Hòa":        "Bệnh nhân 3 (BN3)",
    "Vũ Thị Hoà":        "Bệnh nhân 3 (BN3)",
    "Cao Thị Thường":    "Bệnh nhân 4 (BN4)"
}
```

Tên thật của 4 bệnh nhân được hard-code trong script công khai trong repo.

**Tác động:**

- Dù script có mục đích anonymize, tên thật vẫn hiện diện trong source code.
- Nếu repo public, tên bệnh nhân bị lộ hoàn toàn.
- Mapping này còn xuất hiện ở `clean_text_names()` (dòng 264–276).

**Khuyến nghị:**

- Không hard-code tên thật trong bất kỳ script nào.
- Lưu mapping anonymize trong file config ngoài repo (env, secret vault).
- Thay thế bằng ID nội bộ từ đầu.

---

### N09. `generate_report()` trong `sync_data_and_report.py` hard-code dữ liệu lâm sàng thật trong string Python

**Mức độ: High/Privacy**  
**Vị trí:** `scripts/sync_data_and_report.py:83–242`

Hàm `generate_report()` là một chuỗi Python dài hàng trăm dòng chứa:

- Bảng chỉ số lâm sàng chi tiết của 4 bệnh nhân.
- Mô tả bệnh sử, triệu chứng, nghiệm pháp lâm sàng.
- Nhận định chuyên môn và kế hoạch điều trị.

Toàn bộ nội dung này được hard-code trong source code (không phải đọc từ database), sau đó ghi vào README.md.

**Tác động:**

- Dữ liệu y tế nhạy cảm không thể bị xóa chỉ bằng cách xóa file JSON — nó nằm trong Python source.
- README.md (đã có dữ liệu lâm sàng) đã được commit vào git công khai.
- Không thể cập nhật nếu dữ liệu thay đổi trừ khi sửa code thủ công.

**Khuyến nghị:**

- Tách dữ liệu lâm sàng ra file riêng ngoài source code.
- Report nên được generate từ database, không phải hard-code.
- Xem xét xóa hoàn toàn báo cáo lâm sàng khỏi README và script công khai.

---

### N10. HTTP server nội bộ không có authentication — bất kỳ process nào trên cùng host đều truy cập được

**Mức độ: Medium/High**  
**Vị trí:** `app.py:1637`

```python
server = socketserver.ThreadingTCPServer(('127.0.0.1', attempt_port), _RangeHandler)
```

Server bind `127.0.0.1` (loopback), nhưng không có token, session check, hay bất kỳ authentication nào. Bất kỳ process nào chạy trên cùng máy/container đều có thể fetch file từ `http://127.0.0.1:876x/...`.

**Tác động:**

- Trong môi trường multi-tenant (chia sẻ container), các Space khác có thể fetch dữ liệu.
- Kết hợp với N01 (CORS `*`), browser có thể fetch nếu biết port.
- Port scan từ `8765` đến `8800` rất dễ dàng.

**Khuyến nghị:**

- Thêm random token trong query string của URL video, kiểm tra token trên server.
- Hoặc dùng Streamlit static serving chính thức (`enableStaticServing=true` đã bật) thay vì tự viết server.

---

### N11. `save_data(VIDEOS_FILE, ...)` không dùng lock ở 13/15 chỗ gọi — race condition dữ liệu video

**Mức độ: High**  
**Vị trí:** `app.py:4890, 19355+, và 11 chỗ khác`

Codebase có `doc_lock_save_data()` nhưng chỉ được dùng tại **2 chỗ** trong toàn bộ app. 13 chỗ còn lại gọi `save_data(VIDEOS_FILE, ...)` trực tiếp, và 8 chỗ gọi `save_data(EVALUATIONS_FILE, ...)` không có lock.

Kết hợp với 12 background threads (F37), rủi ro race condition rất cao, đặc biệt khi:

- Thread sync HF ghi video list.
- Thread transcode cập nhật status.
- User upload video mới.

**Tác động:**

- Mất bản ghi video hoặc evaluation do ghi đè.
- JSON bị truncate giữa chừng (dù có temp + `os.replace`, nhưng race ở bước đọc-modify).

**Khuyến nghị:**

- Chuẩn hóa **tất cả** ghi JSON qua `doc_lock_save_data()` hoặc một helper lock tập trung.
- Đặc biệt quan trọng với `VIDEOS_FILE` và `EVALUATIONS_FILE`.

---

### N12. `gTTS` gọi network ra ngoài Internet trong quá trình xử lý video — phụ thuộc bên ngoài không kiểm soát

**Mức độ: Medium**  
**Vị trí:** `app.py:8675–8706`

gTTS (Google Text-to-Speech) gọi API Google khi tạo audio hướng dẫn. Điều này:

- Tạo ra network request ra Internet trong quá trình phân tích video (background thread).
- Phụ thuộc vào API Google — nếu rate limit hoặc lỗi mạng, audio tạo thất bại.
- Có thể gửi text hướng dẫn tập luyện ra Google (dù không nhạy cảm, nhưng là data leak về tính năng app).

**Tác động:**

- Phân tích video thất bại âm thầm nếu gTTS lỗi (nhưng `except` rộng che khuất).
- Latency tăng, timeout có thể gây treo thread.

**Khuyến nghị:**

- Cache audio file sau khi tạo lần đầu (đã có `sounds/` directory — nên tận dụng tốt hơn).
- Dùng audio tĩnh pre-generated thay vì gọi gTTS mỗi lần.
- Nếu giữ gTTS, thêm timeout và retry rõ ràng.

---

### N13. `ffmpeg Popen` không đặt timeout cứng trong hàm transcode chính

**Mức độ: Medium/High**  
**Vị trí:** `app.py:406–430`

Hàm transcode video (re-mux + audio) dùng `subprocess.Popen` với vòng loop kiểm tra `process.poll()`. Timeout được truyền vào qua tham số của hàm gọi — nhưng không có giá trị mặc định an toàn, và một số nơi gọi không truyền timeout.

```python
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
deadline = time.time() + timeout  # timeout từ caller — có thể None hoặc rất lớn
```

**Tác động:**

- Daemon thread treo vĩnh viễn, chiếm CPU/RAM.
- Accumulate nếu nhiều video được xử lý liên tiếp.
- Khó debug vì thread là daemon (không có stack trace khi app rerun).

**Khuyến nghị:**

- Đặt timeout mặc định (ví dụ 600 giây) trong hàm transcode.
- Dùng `subprocess.run(..., timeout=X)` thay vì Popen thủ công khi không cần streaming.
- Kill process con khi thread bị cancel.

---

### N14. Không có giới hạn số lần đăng ký tài khoản mới — spam/flood tài khoản bệnh nhân

**Mức độ: Medium**  
**Vị trí:** `app.py:17770`

Luồng đăng ký tài khoản mới (Bệnh nhân) không có:
- Rate limit theo IP.
- Captcha.
- Giới hạn số tài khoản trên mỗi email.
- Email verification.

**Tác động:**

- Attacker có thể tạo hàng nghìn tài khoản bệnh nhân, làm phình `users.json` và `video_list.json`.
- Pollute dữ liệu nghiên cứu.

**Khuyến nghị:**

- Thêm email verification trước khi kích hoạt tài khoản.
- Giới hạn 1 tài khoản per email.
- Rate limit tạo tài khoản theo session/IP.

---

### N15. `HF_DATASET_ID` bị hard-code fallback trong source code

**Mức độ: Medium**  
**Vị trí:** `app.py:3269`, `scripts/sync_data_and_report.py:21`

```python
HF_DATASET_ID = _get_secret("HF_DATASET_ID") or (f"{HF_SPACE_ID}-data" if HF_SPACE_ID else
    "quynhphuong1209/Rehab-AI-Monitor-2026-data")
```

Nếu secret `HF_DATASET_ID` không được set, app fallback về dataset ID hard-code của tác giả.

**Tác động:**

- Ai fork repo sẽ vô tình kết nối về dataset gốc của tác giả.
- Nếu deploy fork mà không set `HF_DATASET_ID`, dữ liệu của họ có thể ghi vào dataset gốc (nếu có token write).

**Khuyến nghị:**

- Không hard-code Dataset ID — fail rõ ràng nếu `HF_DATASET_ID` không được cấu hình.
- Thêm startup check: `if not HF_DATASET_ID: st.error("HF_DATASET_ID chưa được cấu hình"); st.stop()`.

---

### N16. `_resume_and_watch_analysis_jobs()` chạy `while True` vĩnh viễn — không có cơ chế dừng

**Mức độ: Medium**  
**Vị trí:** `app.py:5197–5208`

```python
def _resume_and_watch_analysis_jobs():
    while True:
        time.sleep(120)
        try:
            n2 = khoi_phuc_job_phan_tich_sau_deploy(cold_start=False)
            ...
```

Thread daemon này chạy `while True` không có điều kiện dừng, không có cơ chế signal stop, không có exponential backoff khi lỗi liên tiếp.

**Tác động:**

- Nếu hàm gây side effect hoặc I/O lỗi lặp đi lặp lại, thread tiếp tục loop vô thời hạn.
- Khó cancel khi admin reset hệ thống.

**Khuyến nghị:**

- Thêm `stop_event = threading.Event()` và kiểm tra trong loop.
- Thêm exponential backoff khi có lỗi liên tiếp.
- Đăng ký thread trong một registry để admin có thể dừng từ UI.

---

### N17. `except: pass` trong nhiều hàm IO quan trọng — lỗi bị nuốt hoàn toàn

**Mức độ: High/Maintainability**  
**Vị trí:** `app.py:19159, 19167, 19174, 19300–19302, 19329–19330, 19336–19338, 18036–18041` và **105 chỗ** khác

Rất nhiều `except:` trần dùng `pass` trong các nhánh xóa file, rename, ghi JSON — không có log, không có alert. Ví dụ tiêu biểu trong upload flow:

```python
try: os.remove(file_path)
except: pass

try: os.remove(file_path_mp4)
except: pass
```

Nếu OS raise `PermissionError` hoặc `IsADirectoryError`, lỗi bị bỏ qua, logic tiếp tục với giả định đã xóa thành công.

**Tác động:**

- File cũ không bị xóa nhưng code nghĩ đã xóa → trùng file, metadata sai.
- Lỗi nghiêm trọng (disk full, permission denied) không hiện ra cho admin.
- Khó điều tra khi bệnh nhân báo lỗi.

**Khuyến nghị:**

- Tối thiểu `except Exception as e: print(f"[Error] {context}: {e}")` trong mọi IO operation quan trọng.
- Không dùng bare `except:` — ít nhất `except OSError`.

---

### N18. `schedules.json` vẫn là `{}` — F13 chưa được fix

**Mức độ: Medium**  
**Vị trí:** `database/schedules.json`

Đây là F13 trong báo cáo gốc — vẫn còn tồn tại. File hiện là `{}` (2 bytes), trong khi code mong list. Dù có fallback `if not isinstance(schedules, list): schedules = []`, file không được tự sửa → mỗi lần khởi động reset về `[]` mà không lưu.

**Khuyến nghị:**

- Đổi nội dung `database/schedules.json` thành `[]`.
- Thêm auto-migrate khi load: nếu load ra không phải list thì save lại `[]` ngay.

---

### N19. `WebRTC streamer` tích hợp nhưng không rõ trạng thái bảo mật và quyền camera

**Mức độ: Medium**  
**Vị trí:** `app.py:7476, 7538`

App tích hợp `streamlit_webrtc` cho tính năng camera trực tiếp. Không rõ:

- Có validate stream từ client không.
- Có rate limit số session WebRTC không.
- Dữ liệu frame có được lưu trữ không (privacy risk).
- ICE server/STUN/TURN config có mặc định Google STUN không (leak IP người dùng ra Google).

**Khuyến nghị:**

- Kiểm tra RTC configuration — không dùng public STUN servers cho app y tế.
- Rõ ràng về việc frame camera có được lưu không.
- Thêm consent dialog trước khi bật camera.

---

### N20. `generate_report()` trong `sync_data_and_report.py` tự ghi link login bypass vào README

**Mức độ: Critical**  
**Vị trí:** `scripts/sync_data_and_report.py:87`

Trong `generate_report()`, hard-coded string ở dòng 87:

```
https://quynhphuong1209-rehab-ai-monitor-2026.hf.space/?logged_in_user=2211090031&logged_in_role=Nghiên+cứu+viên
```

Link này vừa là exploit path (F01/F22) vừa sẽ tiếp tục được ghi vào README mỗi lần script chạy — kể cả sau khi đã xóa thủ công khỏi README.

**Tác động:**

- Fix README thủ công sẽ bị script ghi đè lại khi sync.
- Link login bypass sẽ liên tục tái xuất hiện.

**Khuyến nghị:**

- Xóa URL login bypass khỏi `generate_report()` ngay lập tức.
- Không cho script tự cập nhật README với URL chứa auth param.

---

## 4. Hành động khẩn cấp (trước khi tiếp tục deploy/commit)

> [!CAUTION]
> Các hành động dưới đây cần thực hiện **trước khi push commit tiếp theo** hoặc **trước khi share link app**.

1. **Xóa tên thật bệnh nhân** khỏi `sync_data_and_report.py` và chạy `git bfg --replace-text` để xóa khỏi lịch sử git.
2. **Thêm `database/users.json` vào `.gitignore`** — bỏ exception `!database/users.json` hiện tại.
3. **Xóa URL login bypass** trong `generate_report()` tại dòng 87 của `sync_data_and_report.py`.
4. **Thêm path guard** vào `_RangeHandler` (`app.py:1587`): kiểm tra `os.path.realpath(path).startswith(abs_serve_root)` trước khi mở file.
5. **Bỏ `Access-Control-Allow-Origin: *`** trong HTTP server nội bộ (`app.py:1629`).
6. **Thêm reject sớm** `if file_upload.size > MAX_BYTES: st.error(); st.stop()` trước `getbuffer()` (`app.py:19286`).
7. **Fix email bệnh nhân** trong `users.json` — không dùng email NCV cho bệnh nhân.

---

## 5. Liên kết với báo cáo gốc

Báo cáo này là **phần bổ sung** cho [FRONTEND_CODE_REVIEW_BUG_REPORT.md](./FRONTEND_CODE_REVIEW_BUG_REPORT.md) (F01–F40). Hai tài liệu cần đọc cùng nhau để có bức tranh đầy đủ về tình trạng bảo mật và chất lượng codebase.

Tổng hợp:

| Nguồn | Số vấn đề | Critical | High | Medium |
|---|---|---|---|---|
| Báo cáo gốc (F01–F40) | 40 | 5 | 12 | 14 |
| Báo cáo bổ sung (N01–N20) | 20 | 4 | 9 | 7 |
| **Tổng cộng** | **60** | **9** | **21** | **21** |
