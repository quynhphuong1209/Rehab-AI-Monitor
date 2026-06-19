# Bản Đối Chiếu: Frontend Bug Report vs Code Hiện Tại

Ngày đối chiếu: 17/06/2026

Phạm vi:
- `docs/FRONTEND_CODE_REVIEW_BUG_REPORT.md`
- `docs/SUPPLEMENTARY_BUG_REPORT.md`
- `app.py`, `Dockerfile`, `.streamlit/config.toml`, `scripts/`, `database/`

Mục tiêu:
- Đối chiếu từng vấn đề trong báo cáo với code hiện tại.
- Ghi rõ mục nào khớp hoàn toàn, mục nào cần chỉnh câu chữ, và mục nào đã lỗi thời.

## Cách đọc

- `Khớp hoàn toàn`: code hiện tại xác nhận đúng bản chất vấn đề trong báo cáo.
- `Khớp một phần / cần chỉnh câu`: code xác nhận rủi ro, nhưng cách diễn đạt trong báo cáo nên tinh chỉnh.
- `Outdated / hạ mức`: code hiện tại không còn ủng hộ mức độ/định nghĩa ban đầu.

## Kết luận Nhanh

- 9/9 mục `F01-F09` trong báo cáo chính khớp với code hiện tại.
- 19/20 mục `N01-N20` trong phần thẩm định bổ sung khớp hoặc khớp về bản chất.
- Mục duy nhất nên hạ mức rõ ràng là `N18` vì `database/schedules.json` hiện là `[]`.
- Một số mục nên chỉnh câu chữ để chính xác hơn: `N01`, `N11`, `N15`, `N19`.

## Đối Chiếu F01-F09

| Mục | Trạng thái | Bằng chứng code | Ghi chú |
| --- | --- | --- | --- |
| F01 | Khớp hoàn toàn | `app.py:5217-5291`, `README.md:62` | Login có thể khôi phục bằng `logged_in_user` + `logged_in_role`; `_hoan_tat_dang_nhap()` còn ghi ngược identity vào URL. |
| F02 | Khớp hoàn toàn | `app.py:4942-5002`, `database/users.json` | `_get_cached_users_dict()` seed và ghi đè tài khoản predefined với mật khẩu mặc định. |
| F03 | Khớp hoàn toàn | `app.py:17652-17672` | Quên mật khẩu chỉ cần username + email, chưa có token/expiry/rate limit. |
| F04 | Khớp hoàn toàn | `app.py:5304-5325` | `st.user.email`/`experimental_user.email` được auto-map thành bệnh nhân. |
| F05 | Khớp hoàn toàn | `app.py:1726-1735`, `app.py:11632-11640`, `app.py:18770-18775` | HF token đang nằm trong URL client và cả debug popover. |
| F06 | Khớp hoàn toàn | `app.py:14935-14939`, `app.py:15856-15860`, `app.py:18785-18786`, nhiều vị trí `unsafe_allow_html=True` | Field động như `comments_ncv`, `comments`, `doctor_result`, `plan` được render trực tiếp vào HTML. |
| F07 | Khớp hoàn toàn | `Dockerfile:39` | Runtime tắt `enableCORS` và `enableXsrfProtection`. |
| F08 | Khớp hoàn toàn | `.streamlit/config.toml:2`, `app.py:7160`, `app.py:19131-19233`, `app.py:19286-19327` | Upload max 10 GB và dùng `getbuffer()` để ghi toàn bộ file vào memory. |
| F09 | Khớp hoàn toàn | `app.py:19131-19327` | Chỉ lọc theo extension/uploader type, chưa có kiểm tra file type thực sự. |

## Đối Chiếu N01-N20

| Mục | Trạng thái | Bằng chứng code | Ghi chú |
| --- | --- | --- | --- |
| N01 | Khớp một phần / cần chỉnh câu | `app.py:1556-1637` | Chắc chắn có vấn đề vì server phục vụ cả project root và bật `Access-Control-Allow-Origin: *`; cụm “path traversal” nên để là rủi ro cần test thêm, không nên khẳng định quá mạnh nếu chưa khai thác được. |
| N02 | Khớp hoàn toàn | `database/users.json` | File chứa PII và hash mật khẩu bệnh nhân trong repo. |
| N03 | Khớp hoàn toàn | `database/users.json` | Nhiều bệnh nhân dùng chung một email của NCV, đúng như báo cáo. |
| N04 | Khớp hoàn toàn | `app.py:5039-5053` | Lookup auth match mềm theo key và `full_name`, có rủi ro collision/confusion. |
| N05 | Khớp hoàn toàn | `app.py:19131-19286` | File upload bị nhận vào rồi mới xét kích thước; `getbuffer()` vẫn giữ toàn bộ nội dung trong RAM. |
| N06 | Khớp hoàn toàn | `app.py:401, 1439, 12207, 19320` | `ffmpeg -threads 0` xuất hiện ở nhiều nhánh, có thể chiếm toàn bộ CPU. |
| N07 | Khớp hoàn toàn | `.gitignore:17-23` | `.gitignore` đang unignore nhiều JSON runtime nhạy cảm. |
| N08 | Khớp hoàn toàn | `scripts/sync_data_and_report.py:249-272` | Tên thật bệnh nhân được hard-code trong mapping anonymize. |
| N09 | Khớp hoàn toàn | `scripts/sync_data_and_report.py:84-272` | `generate_report()` hard-code dữ liệu lâm sàng thật và ghi thẳng vào README. |
| N10 | Khớp hoàn toàn | `app.py:1556-1637` | Đây là hệ quả của N01: HTTP server loopback không có auth/token. |
| N11 | Khớp một phần | `app.py:3788, 8837, 9902, 10907, 12932, 13015, 15503, 18307, 19367` | Có `doc_lock_save_data()` nhưng nhiều đường ghi vẫn đi thẳng qua `save_data()`. Rủi ro race/lost update là thật, nhưng không phải mọi write đều unsafe. |
| N12 | Khớp hoàn toàn | `app.py:8675-8680` | `gTTS` gọi ra Internet trong runtime, đúng như báo cáo. |
| N13 | Khớp hoàn toàn | `app.py:12199-12224` | Nhánh transcode chính dùng `subprocess.Popen(...)` không đặt timeout cứng. |
| N14 | Khớp về bản chất | `app.py:17767-17787` | Form đăng ký không thấy giới hạn số lần, captcha, hay xác minh email; đây là rủi ro thiết kế chứ chưa phải exploit động đã chứng minh. |
| N15 | Khớp, nhưng có nuance | `app.py:3267`, `scripts/sync_from_hf.py:23` | Có fallback hard-coded cho `HF_DATASET_ID`; trong `app.py` còn có nhánh suy ra từ `SPACE_ID` chứ không chỉ một giá trị duy nhất. |
| N16 | Khớp hoàn toàn | `app.py:3551-3601`, `app.py:5197-5208` | Có các vòng `while True` chạy nền để watch jobs và upload worker. |
| N17 | Khớp hoàn toàn | `app.py:1397, 1701, 19223, 19302`, `scripts/sync_from_hf.py:52, 59, 127, 238` | `except: pass`/`except Exception:` nuốt lỗi ở nhiều nhánh IO quan trọng. |
| N18 | Outdated / hạ mức | `database/schedules.json` | Trong working tree hiện tại file này là `[]`, nên phần này không còn là hot issue như báo cáo mô tả trước đó. |
| N19 | Khớp về bản chất | `app.py:7475-7538` | WebRTC streamer có Google STUN, nhưng rủi ro chính nằm ở policy/privacy/consent hơn là bug khai thác trực tiếp. |
| N20 | Khớp hoàn toàn | `scripts/sync_data_and_report.py:87`, `README.md:62` | Script vẫn tự sinh lại link login bypass vào README, củng cố lại F01. |

## Nhận Xét Ngắn

- Báo cáo chính xác ở phần lớn các điểm security/runtime.
- `N01` nên giữ nhưng chỉnh câu chữ để tập trung vào “phục vụ quá rộng + CORS wildcard + loopback exposure”.
- `N11`, `N15`, `N19` đúng về bản chất nhưng nên diễn đạt mềm hơn để không quá đà.
- `N18` là mục duy nhất nên coi là lịch sử hoặc low priority trong working tree hiện tại.

## Tài liệu liên quan

- `docs/FRONTEND_CODE_REVIEW_BUG_REPORT.md`
- `docs/SUPPLEMENTARY_BUG_REPORT.md`
- `sonet_bugs_review.md`
