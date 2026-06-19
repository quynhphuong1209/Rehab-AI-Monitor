# Đề xuất thứ tự sửa chữa React/Backend Roadmap

Nguồn: đối chiếu `docs/REACT_BACKEND_MIGRATION_ROADMAP.md` với code backend/frontend hiện tại.

Mục tiêu của kế hoạch này là sửa các điểm lệch theo thứ tự giảm rủi ro: bảo vệ dữ liệu lâm sàng trước, sau đó mới hoàn thiện tài liệu, media artifact, biểu mẫu và UX.

## Thứ tự ưu tiên

### 1. Siết an toàn thao tác xóa clinical data

Ưu tiên cao nhất vì các thao tác này đụng trực tiếp dữ liệu y tế và dữ liệu bệnh nhân.

- [x] Thêm confirm text ở UI cho xóa `evaluation`, `schedule`, `research-record`.
- [x] Backend yêu cầu confirm token/text trước khi thực hiện xóa.
- [x] Backup JSON trước khi ghi/xóa dữ liệu.
- [x] Ghi audit log sau mỗi thao tác xóa.
- [x] Thêm unit tests cho permission, confirm, audit và backup.

### 2. Cập nhật roadmap cho đúng trạng thái thật

Giảm lệch kỳ vọng giữa tài liệu và implementation hiện tại.

- [x] Phase H ghi rõ hiện có tooling JSON -> SQLite, runtime vẫn đọc JSON.
- [x] Ghi rõ chưa có Postgres/runtime repository switch nếu chưa triển khai.
- [x] Chuyển mục "download ảnh biểu đồ" vào backlog nếu chưa có endpoint/artifact tương ứng.
- [x] Ghi chú các phần đã có một phần nhưng chưa đủ workflow riêng, ví dụ báo cáo AI chính thức.

### 3. Hoàn thiện media download còn thiếu

Roadmap có nhắc download gồm ảnh biểu đồ, nhưng code hiện mới có preview chart và các artifact phân tích tiêu chuẩn.

- [x] Quyết định giữ hay bỏ requirement tải ảnh biểu đồ.
- [ ] Nếu giữ: thêm export chart dưới dạng PNG hoặc SVG.
- [ ] Thêm endpoint/artifact metadata cho chart image.
- [ ] Cập nhật UI artifact list và tests.

Quyết định 2026-06-20: đưa tải ảnh biểu đồ vào backlog. Hiện backend/React đã có preview chart từ JSON endpoint và download artifact nguồn (CSV/JSON/ZIP/video), nhưng chưa có artifact/endpoint PNG/SVG riêng.

### 4. Nâng form triệu chứng bệnh nhân

Form hiện tại vẫn là bản basic, chưa đủ các trường backlog đã nêu.

- [x] Thêm đau trước/sau tập.
- [x] Thêm vị trí đau.
- [x] Thêm đau khi nghỉ/vận động.
- [x] Thêm giới hạn vận động.
- [x] Thêm ghi chú tự do.
- [x] Cho phép link tới video/buổi tập liên quan.
- [x] Cập nhật API, UI, bảng hiển thị và tests.

### 5. Nâng nhật ký đánh giá bác sĩ

Sau khi destructive flow đã an toàn, có thể nâng UX khai thác dữ liệu.

- [x] Thêm filter theo bệnh nhân.
- [x] Thêm filter theo bài tập.
- [x] Thêm filter theo kết quả.
- [x] Thêm filter theo khoảng ngày.
- [x] Thêm export CSV.
- [x] Bảo đảm filter/export không lộ dữ liệu ngoài quyền truy cập.

### 6. Hoàn thiện lịch nhắc

Nên làm sau evaluation filters vì cùng nhóm UX quản lý dữ liệu lâm sàng.

- [x] Thêm filter theo loại lịch.
- [x] Thêm trạng thái hoàn thành/quá hạn.
- [x] Thêm view ngày/tuần nếu cần cho workflow bác sĩ.
- [x] Thêm export/print lịch cho bệnh nhân nếu phù hợp.

### 7. Cải thiện phiếu NCKH/ground-truth

Phần này có giá trị nghiên cứu nhưng nên đứng sau clinical safety và các filter vận hành.

- [x] Auto-fill từ video/evaluation khi có dữ liệu nguồn đáng tin cậy.
- [x] Validate field bắt buộc.
- [x] Thêm nhật ký chỉnh sửa nếu cần truy vết ground-truth.
- [x] Làm rõ quyền sửa/xem giữa bác sĩ, NCV và admin.

### 8. Quyết định storage production

Không nên làm Postgres nửa vời nếu chưa có nhu cầu deploy nhiều instance.

- [ ] Chọn hướng runtime chuyển sang SQLite hoặc Postgres.
- [ ] Nếu chỉ một instance: ưu tiên SQLite trước để giảm độ phức tạp.
- [ ] Nếu multi-instance/concurrent writes: thiết kế Postgres repository layer.
- [ ] Thêm migration, backup, rollback và tests trước khi đổi runtime.

### 9. Chạy full gate sau từng cụm sửa

Không gom quá nhiều thay đổi rồi mới test, vì các phần này chạm cả backend, UI và dữ liệu.

- [ ] Chạy `pytest tests/unit` sau các thay đổi backend.
- [ ] Chạy `cd web && npm run lint && npm run build` sau các thay đổi frontend.
- [ ] Chạy Playwright smoke khi chạm flow UI chính.
- [ ] Cập nhật roadmap/checklist sau mỗi cụm đã merge.

## Gợi ý chia cụm triển khai

### Cụm 1: Safety

- Ưu tiên: mục 1.
- Kết quả mong muốn: mọi thao tác xóa dữ liệu lâm sàng đều có confirm, backup và audit.

### Cụm 2: Documentation Parity

- Ưu tiên: mục 2 và quyết định của mục 3.
- Kết quả mong muốn: roadmap không còn hứa quá implementation hiện tại.

### Cụm 3: Clinical UX

- Ưu tiên: mục 4, 5, 6.
- Kết quả mong muốn: form triệu chứng, nhật ký đánh giá và lịch nhắc đủ dùng hơn cho workflow thật.

### Cụm 4: Research & Storage

- Ưu tiên: mục 7 và 8.
- Kết quả mong muốn: ground-truth rõ ràng hơn và có quyết định storage production trước khi mở rộng tiếp.

## Nguyên tắc thực hiện

- Sửa dữ liệu y tế theo hướng an toàn trước, tiện lợi sau.
- Mọi delete/bulk action phải có confirm, backup và audit.
- Không đánh dấu roadmap là xong nếu code mới có route nền tảng nhưng chưa có workflow/UI/test tương ứng.
- Không chuyển Postgres/SQLite runtime nếu chưa có test bảo vệ migration và rollback.
