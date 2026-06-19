# Tổng Hợp Vấn Đề Tìm Được

Ngày ghi nhận: 17/06/2026

Phạm vi:
- Quét code thực tế trong `app.py`, `Dockerfile`, `.streamlit/config.toml`, `scripts/`, `database/`
- Đối chiếu với các báo cáo hiện có trong `docs/FRONTEND_CODE_REVIEW_BUG_REPORT.md` và `docs/SUPPLEMENTARY_BUG_REPORT.md`

Mục tiêu:
- Lưu lại các vấn đề đã xác minh được trong codebase hiện tại.
- Tách riêng bản tổng hợp này khỏi các báo cáo đối chiếu và tài liệu kiến trúc.

## Tóm tắt

Các vấn đề nổi bật nhất vẫn là:
- Auth bypass qua query parameters.
- Tài khoản quyền cao và mật khẩu mặc định hard-code.
- Rò rỉ `HF_TOKEN` ra client/debug UI.
- XSS/HTML injection qua `unsafe_allow_html=True`.
- Upload video quá lớn, có nguy cơ OOM.
- Dữ liệu nhạy cảm/PII xuất hiện trong `database/`, README và hard-code trong script; cần xác minh lịch sử repo public để kết luận chắc chắn đã từng bị commit/push.

Ghi chú về phạm vi: file này là bản **gom nhóm ưu tiên**, không phải danh sách đầy đủ từng mã issue. Hai báo cáo gốc/bổ sung đang có 60 mã (`F01-F40` và `N01-N20`); sau thẩm định, `N18` đang outdated trong working tree hiện tại, nên backlog theo mã còn khoảng 59 mục cần xem xét. Bảng dưới gom các mục trùng/đồng nguồn thành 14 nhóm vấn đề nổi bật.

## Danh Sách Vấn Đề

| Mức độ | Vấn đề | Bằng chứng chính | Ảnh hưởng |
| --- | --- | --- | --- |
| Critical | Auth bypass qua query params | `app.py:5217-5291`, `README.md:62` | Có thể khôi phục session chỉ bằng `logged_in_user` + `logged_in_role`, không cần mật khẩu. |
| Critical | Mật khẩu/tài khoản mặc định hard-code | `app.py:4942-5002`, `database/users.json` | Seed lại tài khoản quyền cao mỗi lần load, mật khẩu đã lộ trong source. |
| High | Reset mật khẩu quá yếu | `app.py:17652-17672` | Chỉ cần username + email là đổi được mật khẩu, không có token/expiry/rate limit. |
| High | Google login auto-trust `st.user.email` | `app.py:5304-5325` | Bất kỳ account Google nào cũng có thể được nhận như bệnh nhân nếu có email hợp lệ. |
| Critical | HF token bị nhúng vào client | `app.py:1726-1735`, `app.py:11632-11640`, `app.py:18770-18775` | Token có thể lộ qua HTML, devtools, log hoặc referrer. |
| Critical | HTML injection/XSS do `unsafe_allow_html=True` | `app.py:14935-14939`, `app.py:15856-15860`, `app.py:18785-18786` | Nội dung người dùng/bác sĩ có thể phá UI hoặc lừa thao tác. |
| High | HTTP video server phục vụ quá rộng | `app.py:1556-1637` | Server phục vụ cả project root và bật `Access-Control-Allow-Origin: *`. |
| High | Upload video dễ OOM | `.streamlit/config.toml:2`, `app.py:19131-19286` | Max upload 10 GB và dùng `getbuffer()` khiến worker dễ hết RAM. |
| High | ffmpeg chiếm nhiều CPU / thiếu timeout cứng | `app.py:401, 1439, 12199-12224, 19320` | `-threads 0` và `Popen` ở nhánh transcode có thể làm UI chậm/đơ. |
| Critical | PII / clinical data xuất hiện trong working tree | `database/users.json`, `.gitignore:17-23`, `scripts/sync_data_and_report.py:84-272`, `README.md:62` | Tên thật, email, hash mật khẩu, mô tả lâm sàng xuất hiện trong file dự án; nếu đã public/push thì cần xử lý như dữ liệu lộ. |
| High | Ghi JSON có nguy cơ race/lost update | `app.py:3788`, `app.py:9902`, nhiều chỗ `save_data(...)` | Một số luồng vẫn ghi thẳng file mà không đi qua lock. |
| Medium | gTTS gọi Internet trong runtime | `app.py:8675-8680` | Phụ thuộc dịch vụ bên ngoài, ảnh hưởng privacy và độ ổn định. |
| Medium | WebRTC dùng STUN công khai | `app.py:7475-7538` | Có ràng buộc privacy/consent cần được làm rõ. |
| Medium | `HF_DATASET_ID` có fallback hard-code | `app.py:3269`, `scripts/sync_from_hf.py:23` | Dễ trỏ nhầm dataset nếu thiếu cấu hình môi trường. |

## Ghi Chú Về Mức Độ

- Những mục `Critical` đều có thể dẫn đến lộ dữ liệu hoặc chiếm quyền nếu bị kết hợp cùng nhau.
- Những mục `High` là rủi ro vận hành hoặc bảo mật đủ lớn để ưu tiên xử lý sớm.
- Những mục `Medium` chủ yếu là rủi ro kiến trúc, privacy hoặc maintainability.

## Ưu Tiên Sửa

1. Xóa auth bypass qua query params.
2. Bỏ hard-code tài khoản/mật khẩu và chuyển sang seed một lần hoặc cơ chế auth thật.
3. Không đưa `HF_TOKEN` ra client.
4. Escape toàn bộ nội dung động đang render bằng `unsafe_allow_html=True`.
5. Giảm max upload và tránh `getbuffer()` cho file lớn.
6. Rà lại toàn bộ JSON write path để tránh race/lost update.
7. Dọn PII/clinical data khỏi repo và khỏi README/script sinh tài liệu.

## Tài Liệu Liên Quan

- `docs/FRONTEND_CODE_REVIEW_BUG_REPORT.md`
- `docs/SUPPLEMENTARY_BUG_REPORT.md`
- `docs/DOI_CHIEU_FRONTEND_BUG_REPORT.md`
