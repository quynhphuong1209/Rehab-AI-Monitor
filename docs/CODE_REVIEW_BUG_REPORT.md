# Báo cáo rà soát bugs và vấn đề tồn tại trong codebase

Ngày rà soát: 17/06/2026  
Phạm vi: đọc codebase hiện tại, tập trung kỹ vào frontend Streamlit trong `app.py`, các luồng đăng nhập, render giao diện, upload video, danh sách video, lịch nhắc, admin, đồng bộ Hugging Face và dữ liệu JSON.

## 1. Tóm tắt điều hành

Codebase là một ứng dụng Streamlit gần như monolith, phần lớn logic frontend, backend nhẹ, xử lý video, đồng bộ cloud và quản trị dữ liệu đều nằm trong `app.py` (~1 MB). Kiểm tra cú pháp Python bằng `python -m py_compile app.py utils\reference_utils.py utils\pose_classifier_utils.py utils\checkpoint_utils.py ...` không phát hiện lỗi syntax.

Tuy nhiên có nhiều vấn đề runtime, bảo mật và UX đang tồn tại. Nghiêm trọng nhất là:

- Có thể đăng nhập chỉ bằng query string `logged_in_user` + `logged_in_role`, không cần mật khẩu.
- Danh sách tài khoản quyền cao và mật khẩu mặc định được hard-code, đồng thời bị ghi đè lại khi load users.
- Frontend render nhiều HTML tự ghép bằng f-string với `unsafe_allow_html=True`, trong đó có dữ liệu do người dùng/bác sĩ nhập.
- URL Hugging Face Dataset chứa `HF_TOKEN` được nhúng vào HTML/video source và hiển thị trong debug popover.
- Upload video cho phép tới 10 GB và dùng `file_upload.getbuffer()`, dễ gây OOM/đơ UI.
- Các thao tác xóa/reset dữ liệu hệ thống không có confirm flow đủ mạnh.
- Docker tắt CORS/XSRF protection.

## 2. Mức độ ưu tiên

| Mức | Số lỗi | Nhóm |
| --- | ---: | --- |
| Critical | 5 | Auth bypass, hard-coded credentials, token leak, XSS, destructive admin sau khi bypass |
| High | 7 | Password reset yếu, upload OOM, CORS/XSRF off, delete thiếu kiểm tra quyền/confirm, path/file handling |
| Medium | 9 | Lỗi schema JSON, KeyError tiềm ẩn, race condition dữ liệu, CSS/UX frontend |
| Low | 5 | Maintainability, logging/debug, cấu trúc monolith, duplicated UI |

### 2.1. Phân loại mã lỗi theo lớp hệ thống

Một số lỗi chạm nhiều lớp khác nhau; bảng dưới đây gán theo lớp chi phối chính để dễ nhìn nhanh phạm vi ảnh hưởng.

| Mục | Mã lỗi chính | Số lỗi | Ghi chú |
| --- | --- | ---: | --- |
| Frontend/UI | F05, F06, F10, F15, F16, F17, F19, F24, F35 | 9 | HTML/CSS/JS render, UX, debug UI và rò rỉ dữ liệu qua giao diện |
| Backend/Auth/Logic | F01, F02, F03, F04, F08, F09, F11, F14, F18, F20, F21, F23, F25, F26, F27, F28, F33 | 17 | Luồng đăng nhập, quyền, validation, xử lý file và callback nghiệp vụ |
| Database/Storage | F12, F13, F38, F40 | 4 | JSON flat-file, schema, sync và ranh giới dữ liệu PII |
| Model/ML | F30 | 1 | Load checkpoint/model không an toàn |
| DevOps/Operations | F07, F29, F31, F32, F34 | 5 | Docker, dependency, test, script vận hành/destructive |
| Documentation/Reporting | F22, F39 | 2 | README/script sinh báo cáo và link nhạy cảm |
| Architecture/Maintainability | F36, F37 | 2 | Trộn layer, side effects và concurrency |
| Tổng |  | 40 |  |

## 3. Phát hiện chi tiết

### F01. Auth bypass qua query parameters

Mức độ: **Critical**  
Vị trí: `app.py:5217-5238`, `app.py:5290-5291`, `README.md:62`, `scripts/sync_data_and_report.py:87`

Hiện tại app khôi phục phiên đăng nhập như sau:

- Nếu URL có `logged_in_user` và `logged_in_role`, app gọi `load_users()`.
- Nếu username tồn tại và role khớp, app set `st.session_state.logged_in = True`.
- Không kiểm tra password, session token, chữ ký HMAC, expiry, nonce, hay server-side session.

Trong `_hoan_tat_dang_nhap()`, app còn tự ghi:

```python
st.query_params["logged_in_user"] = username
st.query_params["logged_in_role"] = role
```

Tác động:

- Bất kỳ ai biết hoặc đoán username có thể truy cập vai trò tương ứng bằng URL.
- README đang chứa ví dụ link đăng nhập sẵn với `logged_in_user=2211090031&logged_in_role=Nghiên cứu viên`, vô tình biến bypass thành tài liệu công khai.
- Khi kết hợp với hard-coded admin/NCV accounts, toàn bộ dữ liệu bệnh nhân, video, đánh giá và reset hệ thống có thể bị truy cập trái phép.

Khuyến nghị:

- Xóa hoàn toàn cơ chế login bằng query params.
- Nếu cần deep-link, chỉ lưu `next`/tab target trong query params, không lưu identity.
- Dùng server-side session/token ký HMAC có expiry, hoặc rely hoàn toàn vào Streamlit auth/OIDC.
- Xóa các link chứa `logged_in_user` trong README và scripts/docs.

### F02. Tài khoản quyền cao và mật khẩu mặc định bị hard-code, bị reset mỗi lần load

Mức độ: **Critical**  
Vị trí: `app.py:4941-5002`, `database/users.json:1-160`

`_get_cached_users_dict()` luôn tạo `predefined` gồm admin, bác sĩ, nghiên cứu viên với mật khẩu mặc định, sau đó ghi đè vào `users`:

```python
for u, data in predefined.items():
    users[u] = data
```

Ví dụ mật khẩu mặc định trong code:

- `admin` -> `admin123@`
- bác sĩ `doctor1..doctor5` -> `bs123@`
- nhiều tài khoản NCV -> `ncv123@`
- một tài khoản admin khác -> mật khẩu hard-coded riêng

Tác động:

- Đổi mật khẩu cho các tài khoản predefined có thể không bền vững vì load users sẽ ghi đè lại.
- Ai đọc repo là biết mật khẩu mặc định của tài khoản quyền cao.
- Kết hợp với F01, attacker không cần password; nếu F01 được fix nhưng F02 còn, vẫn có rủi ro takeover.

Khuyến nghị:

- Không hard-code password trong source.
- Chỉ seed tài khoản một lần khi database rỗng, không ghi đè tài khoản hiện có.
- Bắt buộc đổi mật khẩu lần đầu, rotate toàn bộ mật khẩu mặc định.
- Lưu password bằng thuật toán có salt và cost như `bcrypt`/`argon2`, không dùng SHA-256 trần.

### F03. Password reset chỉ cần username + email, không có token hoặc rate limit

Mức độ: **High**  
Vị trí: `app.py:17652-17672`

Luồng quên mật khẩu cho phép đổi mật khẩu nếu nhập đúng username và email lưu trong JSON. Không có email confirmation, one-time token, expiry, rate limit, captcha, audit log hay khóa tạm khi thử sai nhiều lần.

Tác động:

- Email của nhiều tài khoản nghiên cứu viên nằm ngay trong source/default users; username cũng dễ đoán.
- Attacker có thể reset tài khoản bệnh nhân/bác sĩ nếu biết email.
- Không có dấu vết đầy đủ để điều tra.

Khuyến nghị:

- Tắt reset password trong app cho tới khi có email token thật.
- Thêm reset token ngẫu nhiên, expiry ngắn, lưu hash token.
- Rate limit theo IP/username và ghi audit log.

### F04. Google login tự nhận bất kỳ `st.user.email` là bệnh nhân, không ràng buộc trạng thái bắt đầu auth

Mức độ: **Medium/High**  
Vị trí: `app.py:5300-5325`, `app.py:17789-17793`

Nếu `st.user.email` tồn tại, app tự set logged_in với role `Bệnh nhân`. Không thấy kiểm tra `auth_initiated`, allowlist domain/email, hay mapping vào `users.json`.

Tác động:

- Tài khoản Google bất kỳ có thể trở thành bệnh nhân mới trong session.
- Username lấy từ display name hoặc prefix email, có thể lệch với dữ liệu bệnh nhân hiện có.
- Dễ gây lỗi dữ liệu: lịch nhắc/video/symptoms phụ thuộc username.

Khuyến nghị:

- Chỉ chấp nhận Google login sau callback hợp lệ.
- Map email vào user record có sẵn hoặc tạo record rõ ràng sau đăng ký.
- Cho phép cấu hình allowlist domain nếu cần.

### F05. Token Hugging Face bị nhúng vào frontend và debug UI

Mức độ: **Critical**  
Vị trí: `app.py:1726-1735`, `app.py:2870-2876`, `app.py:11632-11640`, `app.py:18770-18775`

Code tạo URL dạng:

```python
https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/{rel_path}?token={HF_TOKEN}
```

URL này được:

- đặt trong `<source src="...">` của video HTML;
- ghi ra debug popover bằng `st.write`;
- có thể xuất hiện trong DOM, browser devtools, referrer/log/proxy.

Tác động:

- Lộ token có thể cho phép đọc/ghi Dataset tùy scope.
- Bác sĩ/NCV hoặc ai vào được UI có thể copy token.
- Nếu kết hợp F01, token cloud có thể bị lộ cho người ngoài.

Khuyến nghị:

- Không đưa token vào URL phía client.
- Proxy file qua backend/server hoặc dùng signed URL ngắn hạn nếu nền tảng hỗ trợ.
- Với Hugging Face, ưu tiên private dataset access ở server-side; frontend chỉ nhận stream từ app.
- Xóa debug popover hiển thị URL chứa token; nếu cần debug, mask token.

### F06. XSS/HTML injection do `unsafe_allow_html=True` với dữ liệu người dùng

Mức độ: **Critical**  
Vị trí tiêu biểu: `app.py:18785-18786`, `app.py:14935-14939`, `app.py:15856-15860`, `app.py:16050-16128`, `app.py:17561`, nhiều vị trí `unsafe_allow_html=True`

Ứng dụng dùng rất nhiều `st.markdown(..., unsafe_allow_html=True)` và tự ghép HTML bằng f-string. Một số field có nguồn từ user/bác sĩ/dữ liệu JSON được đưa thẳng vào HTML:

- `doc_eval['comments_ncv']`
- `doc_eval['comments']`, `doctor_result`, `plan`
- lịch nhắc: `title`, `notes`, `exercise_name`, `medication_name`
- tên người dùng, tên bệnh nhân, tên video ở một số khối HTML

Ví dụ:

```python
st.markdown(
    f"<div ...><b>Ghi chú cho NCV:</b> {doc_eval['comments_ncv']}</div>",
    unsafe_allow_html=True
)
```

Tác động:

- Người dùng có quyền nhập nội dung có thể chèn HTML/script/style vào UI của bác sĩ/NCV/admin.
- Trong context Streamlit, script injection trong markdown có thể bị hạn chế tùy sanitizer, nhưng HTML/style injection vẫn đủ để phá UI, che nút, giả giao diện hoặc lừa thao tác.
- Nếu token URL xuất hiện trong DOM, XSS làm rủi ro token leak nặng hơn.

Khuyến nghị:

- Mặc định dùng `st.write`, `st.text`, `st.info`, `st.caption` cho nội dung user-generated.
- Nếu bắt buộc render HTML, escape bằng `html.escape()` mọi field động.
- Tách template HTML chỉ cho dữ liệu nội bộ đã kiểm soát.
- Thêm helper `safe_html(value)` và review toàn bộ `unsafe_allow_html=True`.

### F07. Docker tắt CORS và XSRF protection

Mức độ: **High**  
Vị trí: `Dockerfile:39`

Docker command:

```json
"--server.enableCORS=false", "--server.enableXsrfProtection=false"
```

Tác động:

- Tắt lớp bảo vệ mặc định của Streamlit.
- Khi app có các nút ghi/xóa dữ liệu và auth yếu, rủi ro cross-site/embedded interaction tăng lên.

Khuyến nghị:

- Bật lại XSRF protection.
- Chỉ tắt CORS nếu có lý do deploy cụ thể và đã kiểm soát bằng reverse proxy.
- Tách cấu hình local/dev khỏi production.

### F08. Upload video đặt max 10 GB và đọc toàn bộ file vào memory

Mức độ: **High**  
Vị trí: `.streamlit/config.toml:2`, `Dockerfile:39`, `app.py:7160`, `app.py:19131-19136`, `app.py:19283-19286`

Cấu hình:

- `maxUploadSize = 10000`
- `MAX_FILE_SIZE_MB = 10000`
- UI báo hỗ trợ tối đa 10000 MB.

Khi lưu file:

```python
f.write(file_upload.getbuffer())
```

Tác động:

- `getbuffer()` giữ toàn bộ file upload trong RAM/session.
- File vài GB có thể làm Streamlit worker OOM, trắng trang hoặc restart Space.
- Sau upload còn chạy `ffmpeg` với `-threads 0`, dễ chiếm toàn bộ CPU.

Khuyến nghị:

- Giảm max upload xuống mức thực tế, ví dụ 100-300 MB.
- Kiểm tra `file_upload.size` trước khi đọc buffer; reject sớm.
- Nếu cần file lớn, dùng direct upload đến object storage/HF Dataset qua signed upload flow, không đi qua RAM Streamlit.
- Giới hạn ffmpeg threads, timeout, queue và số job đồng thời.

### F09. Không kiểm tra file type thực sự, chỉ dựa vào extension/uploader type

Mức độ: **High**  
Vị trí: `app.py:19131-19136`, `app.py:19288-19327`

Uploader chỉ lọc extension. File sau đó được đưa vào `ffprobe`/`ffmpeg`. Không có MIME sniffing, magic bytes validation, giới hạn duration/resolution, hoặc chặn file giả dạng.

Tác động:

- File không phải video có thể làm ffmpeg treo/lỗi lâu.
- Video quá dài/độ phân giải quá lớn làm pipeline đơ hoặc OOM.
- Tăng rủi ro xử lý input độc hại qua thư viện native.

Khuyến nghị:

- Dùng `ffprobe` trước, lấy duration/streams/resolution/codec.
- Reject nếu duration, width/height, frame count vượt ngưỡng.
- Dùng MIME/magic bytes và transcode trong sandbox/job worker riêng.

### F10. Nút xóa/reset dữ liệu không có xác nhận hai bước

Mức độ: **High**  
Vị trí: `app.py:18014-18059`, `app.py:18848-18851`

Admin tab có các nút xóa toàn bộ:

- xóa lịch sử đánh giá và triệu chứng;
- xóa lịch nhắc;
- xóa danh sách video và file tạm;
- reset toàn bộ hệ thống.

Danh sách video cũng có nút xóa từng video và nút quick delete. Các thao tác này chạy ngay sau một click.

Tác động:

- Click nhầm là mất dữ liệu.
- Với F01, người ngoài có thể vào vai trò admin/NCV/bác sĩ và xóa dữ liệu nếu đoán đúng tài khoản/role.

Khuyến nghị:

- Thêm confirm dialog hoặc text confirmation bắt nhập tên tài khoản/mã xác nhận.
- Phân quyền rõ: chỉ admin xóa toàn hệ thống, bác sĩ không xóa video nếu không được cấp.
- Trước khi xóa, backup snapshot JSON + file list.
- Ghi audit log bắt buộc.

### F11. `delete_video_callback()` dễ KeyError và thiếu kiểm tra quyền

Mức độ: **High**  
Vị trí: `app.py:18282-18312`

Hàm xóa video lọc evaluations bằng:

```python
ev['patient_username'] == v['username'] and ev['video_name'] == v['video_name']
```

Nếu một bản ghi evaluation thiếu key, hàm sẽ KeyError. Đồng thời callback không tự kiểm tra role/quyền, phụ thuộc vào nơi gọi UI.

Tác động:

- Dữ liệu JSON không đồng nhất có thể làm xóa video crash giữa chừng.
- Nếu callback bị gọi từ nơi không mong muốn, thiếu guard quyền ở business logic.
- Cascade delete không phân biệt exercise, có thể xóa nhiều evaluation cùng video_name của cùng bệnh nhân.

Khuyến nghị:

- Dùng `ev.get(...)`.
- Kiểm tra role trong callback: chỉ admin/NCV hoặc owner được phép.
- Filter theo `(patient_username, video_name, exercise)` nếu muốn xóa đúng một record.
- Bọc transaction mềm: backup trước, ghi sau khi mọi bước hợp lệ.

### F12. Cơ sở dữ liệu JSON dạng flat-file có nguy cơ race/lost update

Mức độ: **Medium/High**  
Vị trí: `app.py:3762-3814`, `app.py:9902-9910`, nhiều nơi `load_data()` -> mutate -> `save_data()`

`save_data()` ghi atomic bằng temp + `os.replace`, đây là điểm tốt. Nhưng nhiều luồng thao tác theo pattern:

1. `load_data(file)`
2. mutate list/dict
3. `save_data(file, data)`

Không phải mọi chỗ đều dùng `doc_lock_save_data()`. App cũng có nhiều background threads: sync HF, upload worker, analysis jobs.

Tác động:

- Hai người cùng cập nhật có thể làm mất update của nhau.
- Background sync từ HF có thể ghi đè JSON local trong lúc UI đang ghi.
- Cache `st.cache_data` có thể khiến UI đọc dữ liệu cũ nếu invalidation lệch.

Khuyến nghị:

- Chuẩn hóa mọi ghi JSON qua một helper lock/transaction.
- Dùng version/updated_at và merge theo key.
- Với dữ liệu có nhiều người dùng, cân nhắc SQLite/Postgres thay JSON.

### F13. `schedules.json` từng seed là object `{}` trong khi code mong list

Mức độ: **Low/History**  
Vị trí: `database/schedules.json`, `app.py:15929-15931`, `app.py:16044-16046`

Trạng thái sau khi quét lại working tree ngày 17/06/2026: `database/schedules.json` hiện là `[]`, nên claim "hiện là `{}`" đã lỗi thời. Tuy vậy mục này vẫn nên giữ như ghi chú schema vì code vẫn có nhánh tự vệ nếu dữ liệu bị sync/seed sai:

```python
schedules = load_data(REMINDERS_FILE)
if not isinstance(schedules, list): schedules = []
```

Tác động:

- Không còn là lỗi nóng trong working tree hiện tại.
- Nếu HF sync/script cũ đưa `{}` trở lại, các script/đồng bộ khác vẫn có thể hiểu nhầm type.
- Cần schema validation để tránh dữ liệu lịch quay lại trạng thái sai.

Khuyến nghị:

- Đổi seed thành `[]`.
- Thêm schema validation khi load JSON.

### F14. Nhiều chỗ truy cập dict bằng `[]` với dữ liệu JSON không đảm bảo schema

Mức độ: **Medium**  
Vị trí tiêu biểu: `app.py:16044-16046`, `app.py:16050-16128`, `app.py:18306`, `app.py:18625`, `app.py:18784-18788`, `app.py:18965-18999`

Ví dụ:

- `s['type']`
- `app['title']`, `app['datetime']`
- `doc_eval['doctor_result']`, `doc_eval['comments']`, `doc_eval['plan']`
- `v['full_name']`, `v['exercise']`

Tác động:

- Một record JSON thiếu field sẽ làm tab crash.
- Dữ liệu từ Google login/HF sync/upload cũ có khả năng không đủ field.

Khuyến nghị:

- Dùng `.get()` với default và validate schema khi load.
- Viết migration/normalizer cho từng JSON file.
- Test với bản ghi thiếu field.

### F15. Frontend có CSS global quá mạnh, dễ phá layout và trạng thái cảnh báo

Mức độ: **Medium**  
Vị trí: `app.py:13949-14205`, `app.py:17312-17318`, nhiều block CSS khác

Các vấn đề:

- `* { font-family: ... !important; }` áp lên toàn app, kể cả Plotly/components.
- `.stButton > button` ép mọi button cùng gradient, bo tròn, hover scale, kể cả nút xóa/reset nguy hiểm.
- Ảnh frame hover `transform: scale(2.2)` + `z-index: 99999` + overflow visible nhiều container.
- Có nhiều block CSS lặp lại ở các tab, khó kiểm soát specificity.

Tác động:

- Nút destructive không có visual hierarchy rõ.
- Hover ảnh có thể che controls/tabs/nút xóa, đặc biệt trên màn nhỏ.
- UI dễ vỡ khi Streamlit đổi DOM/testid.

Khuyến nghị:

- Scope CSS theo container/class cụ thể, tránh selector global.
- Thiết kế variant riêng cho destructive actions.
- Thay hover zoom bằng modal/lightbox hoặc popover click-to-open.
- Gom CSS thành module/hàm riêng, giảm lặp.

### F16. Dùng JS can thiệp DOM parent của Streamlit

Mức độ: **Medium**  
Vị trí: `app.py:5339-5387`, `app.py:18863-18888`, `app.py:16000-16029`

App dùng JavaScript để click tab, bind wheel scroll, update clock bằng cách truy cập `window.parent.document`.

Tác động:

- Dễ vỡ theo phiên bản Streamlit.
- Có thể bị chặn bởi sandbox/CSP trong môi trường deploy.
- MutationObserver/polling có thể gây hành vi khó debug.

Khuyến nghị:

- Ưu tiên state-driven navigation bằng `st.session_state` và `st.segmented_control`.
- Nếu cần JS, cô lập trong component riêng và tránh thao tác parent DOM.

### F17. Debug popover lộ thông tin nội bộ cho bác sĩ/NCV

Mức độ: **Medium/High**  
Vị trí: `app.py:18712-18775`

Popover "Kiểm tra tệp tin (Debug)" hiển thị:

- đường dẫn file local/server;
- codec, duration, lỗi ffmpeg;
- URL Cloud có token.

Tác động:

- Lộ cấu trúc server/path dữ liệu.
- Lộ token như F05.
- Người dùng không cần debug có thể copy thông tin nhạy cảm.

Khuyến nghị:

- Chỉ hiển thị debug khi `DEBUG=true` và role admin.
- Mask đường dẫn/token.
- Không render log ffmpeg nguyên văn nếu có thể chứa path/token.

### F18. Phân quyền frontend/backend chưa tách biệt rõ

Mức độ: **High**  
Vị trí: nhiều hàm callback: `delete_video_callback`, `hien_thi_tab_quan_tri_vien`, `hien_thi_lich_nhac_nho`, luồng đánh giá bác sĩ/NCV

Quyền chủ yếu được kiểm soát bằng việc có render button/tab hay không. Business logic phía hàm ghi/xóa ít tự guard lại role.

Tác động:

- Khi có auth bypass hoặc bug state, các hành động nguy hiểm dễ bị gọi.
- Khó audit đúng vai trò nào được làm gì.

Khuyến nghị:

- Tạo helper `require_role(...)` dùng trong mọi callback/handler trước khi mutate dữ liệu.
- Tách permission matrix: Bệnh nhân, Bác sĩ/KTV, NCV, Admin.
- Log actor/action/target/time/result cho mọi mutation.

### F19. Màn login/role select có UX gây nhầm và dễ leak role

Mức độ: **Medium**  
Vị trí: `app.py:17679-17737`

Người dùng phải chọn role trước khi login. Nếu đúng username/password nhưng sai role, app báo "tài khoản này không có quyền truy cập với vai trò X".

Tác động:

- Role enumeration: attacker biết tài khoản thuộc role nào.
- UX rườm rà vì role vốn có thể lấy từ user record sau khi xác thực.

Khuyến nghị:

- Bỏ role select khỏi login.
- Xác thực username/password trước, sau đó điều hướng theo role trong database.

### F20. Hash mật khẩu SHA-256 không salt/cost

Mức độ: **High**  
Vị trí: `app.py:5025-5029`, `database/users.json`

`hash_password()` dùng:

```python
hashlib.sha256(password.encode()).hexdigest()
```

Tác động:

- Nếu `users.json` lộ, hash dễ bị dictionary/bruteforce.
- Mật khẩu mặc định càng dễ bị crack.

Khuyến nghị:

- Dùng `argon2-cffi` hoặc `bcrypt`.
- Migrate hash cũ khi user đăng nhập thành công.

### F21. `load_users()` luôn thêm predefined accounts nhưng không persist rõ ràng

Mức độ: **Medium**
Vị trí: `app.py:4941-5018`

`load_users()` trả về dict đã merge predefined, nhưng không nhất thiết ghi lại ngay vào `users.json` trừ khi save_users được gọi ở nơi khác.

Tác động:

- UI/admin thấy user tồn tại nhưng file source có thể khác.
- Sync HF/local có thể lệch trạng thái tài khoản.
- Đổi/xóa tài khoản predefined có hành vi khó dự đoán vì lần load sau lại thêm/ghi đè.

Khuyến nghị:

- Seed/migrate explicit một lần.
- Tách "default seed data" khỏi runtime loader.

### F22. README/docs chứa link đăng nhập vai trò NCV

Mức độ: **Critical**  
Vị trí: `README.md:62`, `scripts/sync_data_and_report.py:87`

README public chứa URL có `logged_in_user` và `logged_in_role`. Đây không chỉ là tài liệu mà còn là exploit path nếu F01 còn tồn tại.

Khuyến nghị:

- Xóa toàn bộ query login khỏi docs.
- Nếu cần demo link, dùng link app gốc và hướng dẫn login hợp lệ.

### F23. Admin reset không logout hoặc revoke session/token của các client khác

Mức độ: **Medium**  
Vị trí: `app.py:18045-18059`

Reset chỉ xóa JSON/session hiện tại, không có session store chung để revoke các session đang mở ở browser khác.

Tác động:

- Người dùng đã login vẫn có thể thao tác trên session state cũ cho tới rerun/refresh.
- Sau reset dữ liệu, các thao tác ghi lại có thể tái tạo dữ liệu cũ từ session.

Khuyến nghị:

- Có `global_session_version` trong database; tăng khi reset.
- Mỗi request/rerun so version, mismatch thì logout.

### F24. Dữ liệu bệnh nhân/triệu chứng hiển thị rộng cho bác sĩ và NCV

Mức độ: **Medium/Privacy**  
Vị trí: `app.py:18956-19006`, `app.py:18135-18233`

Trang chủ bác sĩ/NCV hiển thị danh sách triệu chứng bệnh nhân mới nhất. Admin dashboard gom thông tin triệu chứng/video/eval.

Tác động:

- Chưa có phân quyền theo bác sĩ phụ trách.
- Dữ liệu sức khỏe nhạy cảm hiển thị cho mọi tài khoản cùng role.

Khuyến nghị:

- Thêm assignment doctor-patient hoặc project/team scope.
- NCV nên xem dữ liệu đã pseudonymize nếu không cần danh tính thật.

### F25. Tên file upload có thể chứa ký tự bất lợi

Mức độ: **Medium**  
Vị trí: `app.py:19280-19286`

Tên file lưu:

```python
filename = f"{username}_{timestamp}_{base_name}{orig_ext}"
```

`base_name` lấy từ `file_upload.name`, chưa thấy sanitize ký tự đặc biệt/độ dài/reserved names.

Tác động:

- Path rất dài hoặc ký tự lạ có thể làm lỗi Windows/Linux/HF Dataset.
- Có nguy cơ path traversal nếu upstream không normalize filename đủ tốt.

Khuyến nghị:

- Dùng whitelist `[a-zA-Z0-9._-]`, normalize unicode, giới hạn độ dài.
- Lưu original filename riêng trong metadata.

## 4. Vấn đề frontend/UX nổi bật

1. **Giao diện phụ thuộc rất nhiều vào HTML/CSS thủ công**  
   Streamlit DOM thay đổi theo version sẽ làm selector gãy. Nên đóng gói style và hạn chế selector `data-testid`.

2. **Nút nguy hiểm chưa khác biệt đủ**  
   CSS global khiến nút reset/xóa trông giống nút primary thông thường. Với dữ liệu y tế, cần confirm rõ hơn.

3. **Hover zoom frame có thể che UI**  
   `scale(2.2)` + `z-index:99999` làm trải nghiệm khó kiểm soát trên mobile/desktop nhỏ.

4. **Debug thông tin kỹ thuật nằm trong UI production**  
   Người dùng role chuyên môn thấy path, codec, URL, log. Đây là UX không cần thiết và là security smell.

5. **Login UX yêu cầu chọn role trước**  
   Vừa gây nhầm, vừa hỗ trợ dò role. App nên tự xác định role sau khi auth.

6. **Nhiều text/HTML dùng emoji và uppercase dài trong button/tab**  
   Trên mobile dễ tràn, dù đã có scroll ngang. Cần test viewport thật bằng Playwright.

## 5. Kiểm tra đã chạy

- `python -m py_compile app.py utils\reference_utils.py utils\pose_classifier_utils.py utils\checkpoint_utils.py scripts\verify_report_numbers.py scripts\sync_from_hf.py`  
  Kết quả: không báo lỗi cú pháp.

- Kiểm tra JSON chính:
  - `database/users.json`: object, 24 users.
  - `database/video_list.json`: array, 8 records.
  - `database/doctor_evaluations.json`: array, 34 records.
  - `database/patient_symptoms.json`: array, 8 records.
  - `database/schedules.json`: list rỗng `[]`.
  - `database/research_data.json`: array, 8 records.
  - `database/lich_su_tap_luyen.json`: array, 45 records.

## 6. Backlog sửa đề xuất

### Cần làm ngay trước khi public/demo

1. Xóa login qua query params (`logged_in_user`, `logged_in_role`) và xóa các link đó trong README/docs.
2. Rotate toàn bộ mật khẩu mặc định; bỏ hard-code password; bỏ ghi đè predefined users mỗi lần load.
3. Không đưa `HF_TOKEN` ra frontend; mask toàn bộ debug URL.
4. Escape hoặc bỏ `unsafe_allow_html=True` ở mọi nơi có dữ liệu người dùng.
5. Bật lại XSRF/CORS protection trong production.
6. Giảm max upload size, reject file lớn trước khi `getbuffer()`.
7. Thêm confirm hai bước cho reset/xóa dữ liệu.

### Nên làm trong sprint kế tiếp

1. Đổi password hashing sang Argon2/bcrypt.
2. Chuẩn hóa schema JSON và migration cho `schedules.json`.
3. Tạo permission guard dùng chung cho mọi callback mutate dữ liệu.
4. Thêm audit log cho login, reset password, upload, đánh giá, xóa, reset hệ thống.
5. Tách CSS/frontend components khỏi `app.py`.
6. Viết smoke tests cho các role chính: bệnh nhân upload, NCV phân tích, bác sĩ đánh giá, admin reset.

## 7. Rà soát bổ sung: kích thước `app.py` và vấn đề kiến trúc

### 7.1. `app.py` có quá lớn không?

Có. Với codebase hiện tại, `app.py` đã vượt xa ngưỡng hợp lý cho một module ứng dụng.

Số liệu đo được:

- Kích thước file: **1,054,073 bytes** (~1.05 MB).
- Số dòng theo `splitlines`: **19,978 dòng**.
- Số dòng theo `Measure-Object`: **18,674 dòng**.
- Số function: **395**.
- Số class: **5**.
- Số import statement: **152**.
- Số top-level statement: **503**.
- Số `except:` trần: **104**.
- Số `except Exception:` đúng mẫu: **136**; mọi biến thể `except Exception...`: **225**.
- Số hàm Streamlit fragment: **9**.
- Số hàm cache Streamlit: **18**.

Các hàm dài nhất:

| Hàm | Dòng | Độ dài |
| --- | ---: | ---: |
| `xu_ly_video_day_du` | `app.py:8861-9819` | 959 dòng |
| `_hien_thi_tab_phan_tich_noi_dung` | `app.py:14211-15159` | 949 dòng |
| `_noi_dung_frames_day_du` | `app.py:16713-17616` | 904 dòng |
| `_inject_base_css_once` | `app.py:5392-6216` | 825 dòng |
| `_render_main_tab_content` | `app.py:18915-19624` | 710 dòng |
| `bat_dau_phan_tich_background` | `app.py:12039-12626` | 588 dòng |
| `_noi_dung_danh_sach_video_fragment` | `app.py:18360-18851` | 492 dòng |
| `xu_ly_frame` | `app.py:8231-8629` | 399 dòng |
| `hien_thi_lich_nhac_nho` | `app.py:15920-16278` | 359 dòng |
| `main` | `app.py:19628-19964` | 337 dòng |

Tác động:

- Khó review: một thay đổi nhỏ có thể ảnh hưởng auth, UI, video, cache, cloud sync cùng lúc.
- Khó test: không có ranh giới rõ giữa pure logic và Streamlit UI.
- Khó debug state: nhiều hàm đọc/ghi `st.session_state` trực tiếp.
- Khó tách quyền: callback xóa/sửa dữ liệu nằm xen với render UI.
- Cold start và rerun dễ chậm vì nhiều side effect top-level.

Khuyến nghị tách module:

- `auth.py`: login, password hashing, session, role guards.
- `storage/json_store.py`: load/save JSON, schema validation, locking, migration.
- `cloud/hf_sync.py`: Hugging Face download/upload, token handling, retry/backoff.
- `video/io.py`: upload, path sanitize, ffprobe, transcode, video serving.
- `analysis/pipeline.py`: MediaPipe, checkpoint, frame extraction, metrics.
- `ui/`: từng role/tab thành file riêng: `patient.py`, `doctor.py`, `researcher.py`, `admin.py`.
- `ui/styles.py` hoặc CSS file riêng: gom CSS và giảm selector global.
- `models/schemas.py`: dataclass/Pydantic schema cho users/videos/evaluations/schedules.

### F26. Quá nhiều `except` rộng làm nuốt lỗi thật

Mức độ: **High/Maintainability**  
Vị trí: toàn `app.py`, đo lại ngày 17/06/2026 được **104** `except:` trần, **136** `except Exception:` đúng mẫu, và **225** mọi biến thể `except Exception...`.

Tác động:

- Lỗi dữ liệu, lỗi quyền, lỗi path, lỗi network có thể bị bỏ qua, UI vẫn hiện trạng thái sai.
- Khó xác định nguyên nhân khi bệnh nhân/bác sĩ báo lỗi.
- Một số lỗi bảo mật có thể bị che khuất, ví dụ xóa file thất bại nhưng metadata đã bị xóa.

Khuyến nghị:

- Chỉ catch exception cụ thể: `FileNotFoundError`, `JSONDecodeError`, `PermissionError`, `requests.RequestException`.
- Log có cấu trúc kèm context: actor, file, video, action.
- Không dùng `pass` trong nhánh lỗi ghi/xóa dữ liệu quan trọng.

### F27. Side effects lớn ở top-level và trong boot làm app khó dự đoán

Mức độ: **High**  
Vị trí: `app.py:5108-5209`, `app.py:6276-6803`, `app.py:6806-7158`, các khối CSS top-level.

`thuc_hien_khoi_tao_he_thong_mot_lan()` chạy background threads khi app khởi động: sync HF, merge video list, transcode nền, resume jobs. Ngoài ra nhiều CSS/theme block chạy ở top-level theo `st.session_state`.

Tác động:

- Import/run app có thể sinh thread và mutate file.
- Khó viết unit test vì import module đã có side effect.
- Rerun Streamlit có thể gặp race giữa UI render và sync nền.

Khuyến nghị:

- Đưa side effects vào service startup có kiểm soát.
- Tách pure functions ra module không phụ thuộc Streamlit.
- Với background jobs, có registry/queue rõ ràng thay vì nhiều thread rải rác.

### F28. Có function name trùng/lồng nhau gây khó debug stack trace

Mức độ: **Medium**  
Vị trí đo được:

- `update_theme_callback`: `app.py:17620`, `app.py:19648`.
- `_worker`: `app.py:2483`, `app.py:3549`.
- `_frag`: `app.py:10604`, `app.py:10684`, `app.py:10761`.
- `custom_download_oss_model`: `app.py:2967`, `app.py:3013`.
- `__init__`: nhiều class.

Tác động:

- Stack trace/log chỉ hiện `_worker` hoặc `_frag`, khó biết thread/fragment nào lỗi.
- Dễ sửa nhầm hàm cùng tên.

Khuyến nghị:

- Đặt tên theo domain: `_hf_upload_worker`, `_media_prefetch_worker`, `_job_status_fragment`.
- Tránh định nghĩa callback lồng nhau nếu cần debug/trace.

### F29. Runtime tự cài package bằng pip trong script

Mức độ: **Medium/High**  
Vị trí: `scripts/sync_data_and_report.py:6-17`, `scripts/extract_youtube_reference.py:124-129`

Script tự chạy:

- `pip install huggingface_hub`
- `pip install yt-dlp -q`

Tác động:

- Build/runtime không reproducible.
- Có thể fail ở môi trường offline/read-only.
- Có rủi ro supply chain nếu package version không pin.

Khuyến nghị:

- Đưa dependency vào `requirements.txt` với version pin hoặc optional requirements.
- Script chỉ báo thiếu dependency và thoát với hướng dẫn.

### F30. Dùng `pickle.load` và `joblib.load` với file local có thể bị thay thế

Mức độ: **High/Security**  
Vị trí: `utils/checkpoint_utils.py:137`, `utils/checkpoint_utils.py:172`, `utils/pose_classifier_utils.py:398`

Checkpoint dùng pickle gzip; model classifier dùng joblib. Cả hai định dạng đều có thể thực thi code khi load nếu file bị thay thế độc hại.

Tác động:

- Nếu attacker ghi được file vào `processed_results`/`database`, có thể dẫn tới code execution khi app resume job/load model.
- Rủi ro tăng vì app có upload/sync cloud và nhiều path fallback.

Khuyến nghị:

- Không load pickle/joblib từ vị trí người dùng hoặc cloud không tin cậy.
- Ký/checksum file checkpoint/model.
- Với checkpoint, ưu tiên JSON/Parquet/npz thuần dữ liệu.
- Với model, cân nhắc `skops` hoặc format an toàn hơn, hoặc ít nhất verify hash.

### F31. Không có test suite trong repo

Mức độ: **High/Maintainability**  
Vị trí: repo không có `tests/`, `test_*.py`, pytest config.

Tác động:

- Các lỗi auth/data/UI regression khó phát hiện trước deploy.
- Refactor `app.py` rất rủi ro vì không có safety net.

Khuyến nghị:

- Bắt đầu bằng tests nhỏ:
  - auth: không được login bằng query params.
  - storage: load/save JSON không mất update.
  - schema: schedules/users/videos/evaluations validate được.
  - video path sanitize.
  - metrics/recalc với dataframe mẫu.
- Thêm smoke tests Playwright/Streamlit cho 4 role chính.

### F32. Requirements pin chưa đủ chặt, dễ vỡ theo version

Mức độ: **Medium**  
Vị trí: `requirements.txt`

Nhiều dependency không pin version: `pandas`, `plotly`, `Pillow`, `pydub`, `gTTS`, `scikit-learn`, `joblib`, `opencv-python-headless`, `streamlit-webrtc`, `tornado`, `aiortc`, `av`.

Tác động:

- Một update upstream có thể làm app vỡ DOM/CSS, video codec, Plotly image export, hoặc ML serialization.
- Khó tái hiện bug giữa local và HF Space.

Khuyến nghị:

- Pin version đầy đủ hoặc dùng lock file.
- Tách `requirements-dev.txt` và `requirements-prod.txt`.
- Có CI cài từ lock file và chạy smoke tests.

### F33. Wrapper file sai chính tả duy trì import wildcard

Mức độ: **Low/Medium**  
Vị trí: `utils/pose_classfier_untils.py`

File wrapper:

```python
from pose_classifier_utils import *  # noqa: F401,F403
```

Tác động:

- Duy trì typo `classfier_untils`.
- `import *` làm namespace khó kiểm soát.
- Người mới dễ import nhầm module cũ.

Khuyến nghị:

- Deprecate wrapper, tìm toàn repo xem còn import không.
- Nếu cần backward compatibility, export explicit symbols và ghi warning.

### F34. `scripts/reset_data.py` là destructive script, không có confirm/dry-run

Mức độ: **High/Operational**  
Vị trí: `scripts/reset_data.py`

Script xóa/ghi rỗng nhiều JSON và `shutil.rmtree()` thư mục media ngay khi chạy, không hỏi xác nhận.

Tác động:

- Chạy nhầm là mất dữ liệu.
- Không backup trước reset.

Khuyến nghị:

- Thêm `--yes`, `--dry-run`, backup timestamped.
- In rõ workspace target trước khi xóa.

### F35. CSS rất lớn nằm trong Python string

Mức độ: **Medium/Frontend**  
Vị trí: `_inject_base_css_once()` `app.py:5392-6216`, dark/light blocks `app.py:6276-7158`, global CSS `app.py:13949-14205`

Tác động:

- Không có lint/format/minify CSS.
- Selector lặp và `!important` dày đặc làm sửa UI khó dự đoán.
- Python diff rất ồn khi chỉnh style.

Khuyến nghị:

- Chuyển CSS ra `assets/styles.css`.
- Dùng theme variables/classes rõ ràng.
- Giảm `!important`, tránh selector `*`, `div`, `span` global.

### F36. `app.py` trộn quá nhiều layer trong cùng hàm

Mức độ: **High/Maintainability**  
Vị trí tiêu biểu: `_render_main_tab_content`, `_hien_thi_tab_phan_tich_noi_dung`, `xu_ly_video_day_du`, `bat_dau_phan_tich_background`.

Một hàm thường vừa:

- đọc/ghi JSON;
- mutate `st.session_state`;
- render UI;
- gọi cloud sync;
- xử lý file/video;
- tính metrics;
- điều hướng tab/rerun.

Tác động:

- Khó test từng hành vi.
- Dễ tạo race condition và bug do rerun.
- Không có contract rõ giữa UI và business logic.

Khuyến nghị:

- Tách command handlers khỏi render functions.
- Hàm render chỉ đọc view model; mutation đi qua service layer.

### F37. Background threads và Streamlit state/cache có nguy cơ race

Mức độ: **High**  
Vị trí: nhiều `threading.Thread(...)`: `app.py:1544`, `1766`, `2524`, `3601`, `5128`, `5182`, `5195`, `5208`, `12623`, `18327`.

Tác động:

- Thread nền ghi file trong lúc UI load/cache.
- Streamlit không đảm bảo mọi thao tác state/cache an toàn từ thread nền.
- Các hàm clear cache gọi rải rác sau save/sync có thể lệch nhịp.

Khuyến nghị:

- Dùng queue/job table thay vì thread ad hoc.
- Mọi cập nhật dữ liệu đi qua lock/transaction.
- Không gọi Streamlit API trong thread nền.

### F38. Script đồng bộ có thể ghi dữ liệu ra cả `database/` và root

Mức độ: **Medium**  
Vị trí: `scripts/sync_data_and_report.py:54-66`

Sau khi tải file vào `database`, script copy thêm ra root:

```python
shutil.copy2(local_db_path, file_name)
```

Tác động:

- Dễ có hai nguồn dữ liệu khác nhau (`database/video_list.json` và `video_list.json` ở root).
- Người đọc code/script có thể dùng nhầm file.

Khuyến nghị:

- Chỉ dùng một data directory.
- Nếu cần backward compatibility, tạo symlink rõ hoặc bỏ root copy.

### F39. `README.md` bị script tự cập nhật báo cáo lâm sàng

Mức độ: **Medium/Operational**  
Vị trí: `scripts/sync_data_and_report.py:278-305`, `README.md`

Script tạo report rồi ghi vào README theo marker. README hiện chứa lượng lớn dữ liệu/báo cáo lâm sàng.

Tác động:

- README vừa là tài liệu kỹ thuật vừa là output dữ liệu, dễ tạo diff lớn.
- Có nguy cơ commit dữ liệu nhạy cảm hoặc link login vào repo.

Khuyến nghị:

- Chuyển báo cáo sinh tự động sang `docs/generated/`.
- README chỉ tóm tắt và link tới file phù hợp.
- Thêm kiểm tra không cho commit query login/token/PII.

### F40. Thiếu ranh giới dữ liệu y tế/PII

Mức độ: **High/Privacy**  
Vị trí: `database/*.json`, `README.md`, `docs/*`, UI admin/doctor/NCV.

Repo chứa tên, email, triệu chứng, bệnh sử, đánh giá lâm sàng và link demo. Một số báo cáo docs/README mô tả bệnh nhân cụ thể.

Tác động:

- Nếu repo public hoặc share rộng, có rủi ro lộ thông tin sức khỏe.
- Dữ liệu nghiên cứu nên pseudonymize/anonymize rõ ràng.

Khuyến nghị:

- Tách demo data khỏi real clinical data.
- Pseudonymize bệnh nhân trong repo.
- Thêm `.gitignore`/push guard cho database runtime nếu có dữ liệu thật.
- Tạo policy export báo cáo không chứa PII trừ khi có mục đích rõ.

## 8. Lớp thẩm định bổ sung từ báo cáo N01-N20

Sau khi có báo cáo rà soát bổ sung `docs/SUPPLEMENTARY_BUG_REPORT.md`, đã đối chiếu lại các nhận xét N01-N20 với codebase hiện tại. Mục tiêu của lớp thẩm định này là kiểm tra claim nào được xác nhận bằng code, claim nào cần chỉnh câu chữ, và claim nào đã lỗi thời so với working tree hiện tại.

Kết luận chung: phần lớn phát hiện N01-N20 là đúng hoặc đúng về bản chất, đặc biệt ở các nhóm auth bypass, PII/clinical data exposure, HTTP video server, upload/ffmpeg, JSON race condition và script tự sinh lại nội dung nhạy cảm. Tuy nhiên có một số điểm cần ghi chú lại để báo cáo chính xác hơn.

### 8.1. Xếp hạng mức độ nghiêm trọng N01-N20 sau thẩm định

| ID | Mức sau thẩm định | Lý do xếp hạng |
| --- | --- | --- |
| N02 | **Critical** | `users.json` chứa PII và hash mật khẩu SHA-256 của bệnh nhân; cần xử lý như dữ liệu đã lộ nếu repo/app từng được share. |
| N07 | **Critical** | `.gitignore` đang cho phép nhiều JSON runtime nhạy cảm đi vào repo; khuếch đại rủi ro PII/clinical data exposure. |
| N20 | **Critical** | Script có thể sinh lại link login bypass vào README, làm fix thủ công không bền và trực tiếp củng cố F01/F22. |
| N01 | **High/Critical** | Server video phục vụ project root, CORS wildcard, không token; Critical nếu môi trường cho browser/process khác đọc được loopback hoặc có file nhạy cảm trong root. |
| N09 | **High/Critical** | Báo cáo lâm sàng hard-code trong source/README; Critical nếu là dữ liệu thật hoặc repo public. |
| N03 | **High** | Nhiều bệnh nhân dùng chung email NCV, kết hợp password reset yếu có thể takeover tài khoản bệnh nhân. |
| N05 | **High** | Upload dùng `getbuffer()` với max 10 GB, rủi ro OOM/crash worker. |
| N06 | **High** | ffmpeg `-threads 0` có thể chiếm toàn bộ CPU trong môi trường ít vCPU. |
| N08 | **High** | Tên thật bệnh nhân hard-code trong script anonymize, làm pseudonymization mất tác dụng. |
| N11 | **High** | Nhiều ghi `VIDEOS_FILE`/`EVALUATIONS_FILE` không đi qua lock, có nguy cơ lost update/race condition. |
| N13 | **High/Medium** | Một số `Popen` ffmpeg có nguy cơ treo; cần hiệu chỉnh vì hàm chính `sync_transcode_to_h264()` đã có timeout mặc định. |
| N17 | **High** | Nhiều `except:`/`except Exception:` nuốt lỗi IO, làm sai trạng thái dữ liệu và khó điều tra sự cố. |
| N04 | **Medium/High** | Auth lookup match theo key/full_name casefold gây collision/confusion; nặng hơn nếu kết hợp đăng ký tự do hoặc reset mật khẩu. |
| N10 | **Medium/High** | HTTP server loopback không auth/token; là phần của N01, mức phụ thuộc môi trường deploy. |
| N14 | **Medium/High** | Đăng ký không rate limit/email verification; rủi ro spam/pollute dữ liệu, chưa phải exploit đã chứng minh. |
| N15 | **Medium/High** | Fallback `HF_DATASET_ID` hard-code có thể làm fork ghi/đọc nhầm dataset khi thiếu cấu hình. |
| N12 | **Medium** | gTTS gọi Internet trong runtime/background, gây phụ thuộc ngoài và rủi ro privacy/latency. |
| N16 | **Medium** | Thread `while True` không stop/backoff đầy đủ, tăng rủi ro vận hành và khó shutdown sạch. |
| N19 | **Medium** | WebRTC dùng Google STUN và chưa thấy policy rõ về consent/privacy/session limit. |
| N18 | **Outdated/Low hiện tại** | Working tree hiện có `database/schedules.json` là `[]`; giữ như ghi chú lịch sử schema, không xếp vào backlog nóng. |

Nếu chỉ chọn một nhóm làm ngay, ưu tiên thực tế là: **N02/N07/N20/N01/N09**, sau đó đến **N03/N05/N06/N11/N17**.

### 8.2. Các phát hiện bổ sung được xác nhận mạnh

| Nhóm | Kết luận thẩm định | Liên hệ với Fxx |
| --- | --- | --- |
| N01/N10 | HTTP video server tự mở tại `127.0.0.1:{8765-8800}`, phục vụ từ project root qua `SimpleHTTPRequestHandler`, có `Access-Control-Allow-Origin: *` và không có auth/token. Đây là bề mặt rò rỉ dữ liệu nội bộ rất đáng ưu tiên. Claim "path traversal" nên diễn đạt là nguy cơ/guard yếu, vì handler vẫn dùng `translate_path()`; lỗi chắc chắn hơn là scope phục vụ quá rộng + CORS wildcard. | Bổ sung cho F05, F12, F40 |
| N02/N03/N07 | `database/users.json` chứa tên/email/hash mật khẩu SHA-256 của bệnh nhân; `.gitignore` có exception cho nhiều JSON runtime nhạy cảm. Đây là rủi ro PII/clinical data rất rõ. | Củng cố F02, F20, F40 |
| N04 | `_auth_lookup_key()` match mềm theo key casefold và `full_name`, có rủi ro collision/confusion khi đăng nhập hoặc reset mật khẩu. | Bổ sung cho F03, F19 |
| N05/N06 | Upload vẫn dùng `file_upload.getbuffer()` sau khi chọn file, max upload vẫn 10 GB, và nhiều lệnh ffmpeg dùng `-threads 0`. Rủi ro OOM/CPU starvation là thực tế. | Củng cố F08, F09 |
| N08/N09/N20 | `scripts/sync_data_and_report.py` hard-code mapping tên thật, hard-code báo cáo lâm sàng, hard-code link login bypass và có thể ghi lại nội dung đó vào README. Fix README thủ công sẽ không bền nếu script còn giữ template này. | Củng cố F01, F22, F39, F40 |
| N11/N16/N17 | Có nhiều pattern `load_data()` -> mutate -> `save_data()` không qua lock, 12 chỗ tạo `threading.Thread(...)`, một thread `while True`, và nhiều `except` rộng. Race/lost update và lỗi bị nuốt là rủi ro thật. | Củng cố F12, F26, F37 |
| N12/N19 | gTTS gọi Google trong runtime và WebRTC dùng Google STUN (`stun:stun.l.google.com:19302`). Đây là phụ thuộc network/privacy cần kiểm soát cho ứng dụng y tế. | Bổ sung cho F24, F40 |
| N15 | `HF_DATASET_ID` có fallback hard-code về dataset của tác giả. Fork/deploy thiếu biến môi trường có thể vô tình trỏ về dataset không mong muốn. | Bổ sung cho F05, F38 |

### 8.3. Các điểm cần hiệu chỉnh trong báo cáo bổ sung

- **N18 đã lỗi thời trong working tree hiện tại:** `database/schedules.json` đang là `[]`, không còn là `{}`. F13 vẫn là lịch sử/rủi ro schema đáng nhắc nếu file từng được seed sai, nhưng không nên ghi rằng trạng thái hiện tại vẫn là `{}`.
- **N13 cần diễn đạt lại:** `sync_transcode_to_h264()` có timeout mặc định `1800` giây, nên câu "không có timeout mặc định" không đúng cho hàm này. Tuy vậy vẫn có nhánh `subprocess.Popen()` khác trong background/upload flow không thấy timeout cứng, nên vấn đề "Popen ffmpeg có thể treo" vẫn còn nhưng cần chỉ đúng vị trí.
- **Các câu "đã commit vào git" cần xác minh lịch sử repo public:** checkout hiện tại không có `.git` riêng trong thư mục project; git root đang là `D:\AI20K` và thư mục `Rehab-AI-Monitor-main/Rehab-AI-Monitor-main` hiện là untracked trong repo cha. Vì vậy từ môi trường này chỉ xác nhận được rằng file/rule nhạy cảm tồn tại trong working tree, chưa đủ để kết luận chắc chắn đã được commit/push lên GitHub hoặc Hugging Face.
- **N01 không nên khẳng định path traversal tuyệt đối nếu chưa có test khai thác:** claim chắc chắn là server phục vụ project root quá rộng, có CORS wildcard và không auth. Path traversal/symlink/encoded-path nên để là rủi ro cần kiểm thử thêm.

### 8.4. Số liệu đối chiếu bổ sung

Các số liệu sau được đo lại trên `app.py` trong working tree hiện tại:

| Metric | Kết quả đo lại |
| --- | ---: |
| `unsafe_allow_html=True` | 139 |
| bare `except:` | 104 |
| `except Exception:` đúng mẫu | 136 |
| mọi biến thể `except Exception...` | 225 |
| `threading.Thread(...)` | 12 |

Các số liệu này củng cố nhận định chính của báo cáo gốc: codebase đang có bề mặt HTML tự ghép rất rộng, xử lý lỗi quá lỏng, nhiều thread nền, và nhiều luồng ghi JSON cần lock/transaction tập trung.

### 8.5. Ưu tiên xử lý sau lớp thẩm định thứ hai

Lớp thẩm định N01-N20 không thay thế F01-F40, mà làm rõ thứ tự ưu tiên:

1. Gỡ auth bypass qua query params và xóa mọi link `logged_in_user`/`logged_in_role` trong README/script.
2. Loại bỏ dữ liệu bệnh nhân thật khỏi repo, README, script và JSON runtime; rotate mật khẩu/hash đã lộ; xác minh lịch sử repo public nếu đã từng push.
3. Thu hẹp hoặc thay thế HTTP video server: chỉ phục vụ thư mục media cần thiết, bỏ CORS wildcard, thêm path realpath guard và token/session check.
4. Không đưa `HF_TOKEN` hoặc URL có token ra frontend/debug UI; bỏ fallback dataset ID hard-code.
5. Giảm max upload, reject file lớn trước mọi `getbuffer()`, giới hạn ffmpeg thread/timeout/concurrency.
6. Chuẩn hóa mọi ghi JSON qua helper lock/transaction và giảm `except: pass` ở các thao tác IO quan trọng.

## 9. Rà soát bổ sung lần 3: các vấn đề khác với F01-F40 và N01-N20

Sau khi đối chiếu lại `CODE_REVIEW_BUG_REPORT.md` với working tree hiện tại, phát hiện thêm một số vấn đề chưa được gọi tên rõ trong F01-F40 hoặc lớp N01-N20. Các mục dưới đây không thay thế các lỗi cũ, mà là backlog bổ sung cần xử lý để tránh bỏ sót khi cleanup dữ liệu và gia cố storage/media pipeline.

### R01. `debug_files/` còn chứa bản sao dữ liệu nhạy cảm

Mức độ: **High/Critical**  
Vị trí: `debug_files/users.json:1`, `debug_files/doctor_evaluations.json:1`, `debug_files/video_list.json:1`, `debug_files/debug_files_manifest.md:7-12`

Report cũ đã nêu rủi ro PII/clinical data trong `database/*.json`, README và docs, nhưng chưa chỉ rõ rằng `debug_files/` cũng đang giữ bản sao dữ liệu cũ. `debug_files/users.json` chứa tên/email/hash mật khẩu; `debug_files/doctor_evaluations.json` chứa đánh giá lâm sàng; manifest còn mô tả đây là "bản sao dữ liệu JSON cũ".

Tác động:

- Nếu chỉ xóa/pseudonymize `database/` thì dữ liệu nhạy cảm vẫn còn trong repo qua `debug_files/`.
- Các bản sao debug có thể bị dùng nhầm làm nguồn dữ liệu hoặc bị commit/share cùng codebase.

Khuyến nghị:

- Xóa hoặc anonymize toàn bộ `debug_files/*.json` chứa dữ liệu người dùng/bệnh nhân.
- Đưa `debug_files/*.json` vào `.gitignore`, chỉ giữ manifest hoặc dữ liệu mẫu đã sanitize.
- Nếu repo từng public/push, rà lịch sử git giống quy trình xử lý PII ở F40/N02/N07.

### R02. Path từ JSON/HF có thể điều khiển local write/delete ngoài vùng dữ liệu

Mức độ: **High**  
Vị trí: `app.py:196-206`, `app.py:3353-3384`, `app.py:3615-3666`, `app.py:3669-3701`, `app.py:11616-11679`

Các helper như `get_clean_rel_path()`, `_hf_download_via_http()`, `_hf_download_dataset_file()`, `ensure_local_file()` và `download_file_with_progress()` dùng path lấy từ `video_list.json`/session/HF metadata để dựng đường dẫn local. Code hiện chưa có guard kiểu `realpath(target).startswith(realpath(DATA_DIR))` trước khi ghi hoặc xóa file.

Tác động:

- Nếu record JSON/HF bị nhiễm path chứa `../` hoặc path bất thường, app có thể ghi/xóa file ngoài `DATA_DIR`.
- Đây là nhánh rủi ro khác với N01: N01 nói về HTTP server phục vụ project root; R02 nói về dữ liệu đồng bộ từ cloud điều khiển thao tác filesystem local.

Khuyến nghị:

- Thêm helper duy nhất `safe_data_path(rel_path, allowed_roots)` dùng `Path.resolve()`/`os.path.realpath()` để enforce containment.
- Reject mọi path absolute, path có `..`, hoặc path không nằm trong `patient_uploads/`, `processed_results/`, nhóm JSON được phép.
- Không xóa file cũ trong `download_file_with_progress()` nếu path chưa qua containment guard.

### R03. Giải nén ZIP frames không giới hạn số file/kích thước

Mức độ: **High/Medium**  
Vị trí: `app.py:3715-3752`

`check_and_extract_frames_zip()` tải hoặc dùng ZIP frames rồi gọi `zip_ref.extractall(frames_dir)` trực tiếp. Hiện chưa kiểm tra `ZipInfo.filename`, số lượng entry, tổng uncompressed size, extension file, hoặc quota thư mục đích.

Tác động:

- ZIP lớn hoặc zip bomb có thể làm đầy disk, làm Space/app treo.
- ZIP chứa tên file bất thường có thể tạo file ngoài ý muốn nếu thư viện/phiên bản không bảo vệ đầy đủ hoặc có symlink/metadata đặc biệt.
- Rủi ro này độc lập với F08/N05 upload OOM vì nguồn ZIP có thể đến từ HF Dataset/lazy download.

Khuyến nghị:

- Trước khi extract, duyệt `infolist()` và giới hạn tổng `file_size`, số entry, từng entry size.
- Chỉ cho phép file ảnh với basename hợp lệ, không path separator, không absolute path, không `..`.
- Extract từng file thủ công vào `frames_dir` sau khi resolve path đích và kiểm tra containment.

### R04. Metadata `frames_zip_path` sai timestamp từng làm tải/hiển thị nhầm frames

Mức độ: **Low/Medium hiện tại**  
Vị trí: `debug_files/video_list.json:9-113`, `database/video_list.json:9-113`, `app.py:2202-2216`

Trạng thái sau khi quét lại working tree ngày 17/06/2026: bản ghi đầu tiên trong `database/video_list.json` đã có `frames_zip_path` đúng là `/data/processed_results/processed_1780544672_frames.zip`. Tuy nhiên bản sao cũ trong `debug_files/video_list.json` vẫn còn `frames_zip_path` trỏ tới `processed_1780393267_frames.zip`. Trong code, `_frames_zip_path_from_video()` vẫn ưu tiên `frames_zip`/`frames_zip_path` trước khi suy ra từ `processed_path`.

Tác động:

- Luồng chính dùng `database/video_list.json` không còn thể hiện lỗi mẫu đã nêu.
- Nếu `debug_files/video_list.json` được dùng nhầm để restore/test, UI có thể tải ZIP frames sai video hoặc báo thiếu frames dù dữ liệu đúng tồn tại.
- Rủi ro còn lại nằm ở thiếu validator metadata và bản sao debug chưa được dọn/sanitize.

Khuyến nghị:

- Sửa hoặc xóa bản sao sai trong `debug_files/video_list.json`.
- Khi load video record, nếu timestamp trong `frames_zip_path` khác timestamp trong `processed_path`/`df_path`/`all_frames_data_path`, tự ưu tiên path suy ra từ `processed_path` và log cảnh báo.
- Thêm script validation metadata cho `video_list.json`.

### R05. `sync_from_hf.py` có thể ghi đè toàn bộ `users.json`

Mức độ: **Medium/High**  
Vị trí: `scripts/sync_from_hf.py:27-36`, `scripts/sync_from_hf.py:181-189`, `scripts/sync_from_hf.py:260-269`

`sync_from_hf.py` mặc định đồng bộ cả `users.json`. Với dữ liệu dạng dict, script ghi thẳng remote vào local thay vì merge có kiểm soát. Điều này khác với F38 ở chỗ rủi ro trực tiếp nằm ở rollback/ghi đè tài khoản, role và password hash local.

Tác động:

- Nếu admin vừa đổi mật khẩu/role local, chạy sync có thể đưa dữ liệu cũ từ HF về và làm mất thay đổi.
- Nếu HF Dataset bị sai hoặc cũ, toàn bộ auth DB local bị rollback.

Khuyến nghị:

- Không sync `users.json` mặc định; yêu cầu flag rõ ràng như `--include-users`.
- Với `users.json`, merge theo username và không ghi đè password/role nếu local mới hơn.
- Thêm backup timestamped thay vì chỉ `.bak` một phiên bản.

### R06. ML reprocess có thể ghi ra ngoài workspace qua resolver path quá rộng

Mức độ: **Medium**  
Vị trí: `utils/pose_classifier_utils.py:646-672`, `utils/pose_classifier_utils.py:855-947`

`resolve_local_path()` thử nhiều candidate, trong đó có `os.path.abspath(clean)` và `os.path.abspath(os.path.join(processed_dir, basename))`, rồi pipeline `reprocess_videos_with_classifier()` ghi đè CSV, frame JSON, ảnh frame và file evaluation/video list. Path đầu vào lấy từ JSON metadata.

Tác động:

- Nếu JSON metadata chứa path ngoài vùng dự án nhưng file tồn tại, script có thể đọc/ghi nhầm file ngoài workspace.
- Các thao tác ML batch có thể sửa nhiều file mà không có dry-run hoặc containment guard.

Khuyến nghị:

- Giới hạn resolver vào `data_dir`, `processed_dir`, `db_dir` sau khi resolve realpath.
- Thêm `--dry-run` cho pipeline apply/reprocess.
- Log danh sách file sẽ ghi trước khi thực thi, đặc biệt với CSV/frame JSON.

### 9.1. Ưu tiên xử lý cho nhóm R01-R06

1. Xóa/anonymize `debug_files/*.json` và cập nhật `.gitignore`.
2. Viết helper path containment dùng chung cho mọi download/sync/resolve path.
3. Thay `extractall()` bằng giải nén có validate từng entry.
4. Dọn metadata `frames_zip_path` sai trong `debug_files/video_list.json` và thêm validator cho `video_list.json`.
5. Đổi `sync_from_hf.py` để không sync `users.json` mặc định.
6. Siết `resolve_local_path()` trong `utils/pose_classifier_utils.py` và thêm dry-run cho reprocess.

## 10. Cập nhật sau quét lại working tree ngày 17/06/2026

Đã quét lại codebase hiện tại và đối chiếu với các mục F01-F40, N01-N20 và R01-R06. Kết luận ngắn: phần lớn lỗi nghiêm trọng vẫn còn đúng, đặc biệt nhóm auth, token, upload, PII, JSON race và destructive actions. Chưa thấy fix đáng kể cho các mục Critical chính.

### 10.1. Các phát hiện vẫn khớp mạnh

| Nhóm | Trạng thái hiện tại | Bằng chứng chính |
| --- | --- | --- |
| Auth bypass | Vẫn còn | `app.py:5376-5399` đọc `logged_in_user`/`logged_in_role`; `app.py:5441-5452` ghi identity vào query params. |
| Hard-coded credentials | Vẫn còn | `app.py:5100-5162` định nghĩa predefined users và ghi đè vào `users`. |
| Password reset yếu | Vẫn còn | `app.py:17890-17911` reset bằng username + email. |
| Google login auto-trust | Vẫn còn | `app.py:5461-5485` tự set role `Bệnh nhân` cho mọi `st.user.email`. |
| HF token leak | Vẫn còn | `app.py:1771-1796`, `app.py:19003-19008`, và nhánh video source quanh `app.py:3010-3022`. |
| Unsafe HTML | Vẫn còn rộng | 139 lần `unsafe_allow_html=True`; ví dụ `doc_eval['comments_ncv']` tại `app.py:19013-19021`. |
| CORS/XSRF off, upload 10 GB | Vẫn còn | `Dockerfile:39`, `.streamlit/config.toml:2`. |
| Upload OOM | Vẫn còn | `file_upload.getbuffer()` tại `app.py:19464-19467`, `app.py:19516-19519`. |
| Delete/reset một click | Vẫn còn | Admin destructive actions tại `app.py:18252-18299`. |
| `delete_video_callback()` thiếu guard | Vẫn còn | `app.py:18521-18550` chưa kiểm tra quyền và vẫn dùng `ev[...]`. |
| HTTP video server rộng | Vẫn còn | `app.py:1603-1689` phục vụ project root, CORS `*`, không auth. |
| PII/runtime JSON trong repo | Vẫn còn | `database/*.json`, `debug_files/*.json`, `.gitignore` whitelist JSON runtime. |
| `sync_from_hf.py` ghi đè users | Vẫn còn | `scripts/sync_from_hf.py:27-36`, `scripts/sync_from_hf.py:181-189`, `scripts/sync_from_hf.py:260-269`. |
| Script sinh lại link login | Vẫn còn | `scripts/sync_data_and_report.py:83-88`, `scripts/sync_data_and_report.py:278-305`. |

### 10.2. Các điểm đã thay đổi so với nội dung cũ

- `database/schedules.json` hiện là `[]`; F13/N18 chỉ nên xem là ghi chú lịch sử schema, không còn là lỗi nóng ở working tree hiện tại.
- `database/video_list.json` bản ghi đầu tiên đã sửa `frames_zip_path`; R04 chỉ còn đúng với `debug_files/video_list.json` và rủi ro thiếu validator.
- Không thấy JSON runtime ở root hiện tại; F38 vẫn đúng dưới dạng rủi ro tái tạo vì `scripts/sync_data_and_report.py` có logic copy từ `database/` ra root khi chạy.
- Các câu khẳng định "đã commit/push" vẫn cần kiểm chứng bằng lịch sử remote/public; từ working tree chỉ xác nhận dữ liệu/rule nhạy cảm đang tồn tại trong repo local.

### 10.3. Số liệu đo lại

| Metric | Kết quả |
| --- | ---: |
| `unsafe_allow_html=True` trong `app.py` | 139 |
| bare `except:` trong `app.py` | 104 |
| `except Exception:` đúng mẫu trong `app.py` | 136 |
| mọi biến thể `except Exception...` trong `app.py` | 225 |
| `threading.Thread(...)` trong `app.py` | 12 |
| `save_data(...)` trong `app.py` | 43 |
| `doc_lock_save_data(...)` trong `app.py` | 3 |

Kiểm tra cú pháp `python -m py_compile` cho `app.py`, các file `utils` chính và các script chính đều pass. Có một `SyntaxWarning` trong `scripts/sync_data_and_report.py` về escape `\|`, chưa phải blocker runtime.

## 11. Kết luận

Ứng dụng có nhiều nỗ lực xử lý thực tế cho video, cache, HF sync và workflow lâm sàng, nhưng bề mặt frontend/auth đang là điểm yếu lớn nhất. Trước khi triển khai thật hoặc chia sẻ link public, nên ưu tiên xử lý auth bypass, token leak, hard-coded credentials và unsafe HTML. Đây là các lỗi có khả năng gây lộ dữ liệu hoặc mất dữ liệu cao hơn nhiều so với các lỗi giao diện thông thường.

Sau rà soát bổ sung và lớp thẩm định thứ hai, `app.py` cũng nên được xem là một vấn đề kiến trúc cấp cao: file hiện quá lớn, quá nhiều trách nhiệm và quá nhiều side effect để tiếp tục phát triển an toàn. Nên refactor theo từng lát nhỏ có test bảo vệ, bắt đầu từ auth/storage/cloud/video, rồi mới tách các tab UI.
