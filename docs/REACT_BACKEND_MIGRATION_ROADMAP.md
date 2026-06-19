# Lo trinh migration React/Backend

Ngay cap nhat: 2026-06-19

Tai lieu nay tong hop lo trinh tiep theo cho viec dua cac workflow tu Streamlit sang React + backend API rieng. Muc tieu la khong rewrite big-bang: moi buoc phai co API, UI, test va smoke rieng.

## Nguyen tac uu tien

- Uu tien workflow da co backend nen tang truoc, de tang gia tri UI nhanh va it rui ro.
- Khong lam lai phan da co: backend hien da co upload, media endpoint, analysis job, artifacts, CRUD clinical co ban va admin user co ban.
- Moi slice moi can co it nhat mot verify: unit backend, `npm run lint`, `npm run build`, hoac Playwright smoke neu cham UI.
- Cac thao tac nhay cam nhu xoa/reset/revoke/export phai co auth guard, audit log va backup neu co ghi/xoa du lieu.

## Hien trang da co

### Backend API da co

- Auth: login, register, me, change password, logout.
- Patient data:
  - `GET /patients`
  - `GET /videos`
  - `GET /symptoms`
  - `POST /symptoms`
- Upload/media:
  - `POST /videos/upload`
  - `GET /videos/media/{stored_filename}`
- Analysis:
  - `POST /videos/{stored_filename}/analysis-jobs`
  - `GET /videos/{stored_filename}/analysis-jobs/latest`
  - `GET /videos/{stored_filename}/analysis-artifacts`
  - `GET /videos/{stored_filename}/analysis-artifacts/{artifact_kind}`
- Clinical records:
  - `GET/POST/DELETE /evaluations`
  - `GET/POST/DELETE /schedules`
  - `GET/POST/DELETE /research-records`
- Admin co ban:
  - `GET/POST /admin/users`
  - `DELETE /admin/users/{username}`

### React UI da co

- Login/register patient.
- Dashboard theo role.
- Benh nhan khai bao trieu chung co ban.
- Benh nhan upload video.
- Xem video upload qua backend media endpoint.
- Bang video, benh nhan, trieu chung, lich nhac, research records, users.
- Bac si/admin tao/xoa evaluation, schedule, research record.
- Admin tao/xoa user co ban.
- NCV/admin start analysis job va poll progress.
- Xem manifest/download artifact phan tich.
- Playwright smoke patient login/dashboard/video/schedule.

## Khoang trong thuc su con lai

### 1. Benh nhan

- Form khai bao trieu chung can day du hon:
  - dau truoc/sau tap;
  - vi tri dau;
  - dau khi nghi/van dong;
  - gioi han van dong;
  - ghi chu tu do;
  - lien ket voi buoi tap/video neu co.
- Chon bai tap va xem video huong dan/YouTube tham khao.
- Xem ket qua chi tiet:
  - ket qua AI;
  - nhan xet bac si;
  - ke hoach tiep theo;
  - lich su tap luyen theo timeline;
  - video da phan tich va artifact lien quan.

### 2. Bac si/KTV

- Nhat ky danh gia nang cao:
  - filter theo benh nhan, bai tap, ket qua, khoang ngay;
  - export CSV;
  - confirm truoc khi xoa.
- Man hinh ket qua AI chi tiet de phuc vu lam sang:
  - metrics;
  - bieu do;
  - video da phan tich;
  - frames PASS/NEAR/FAIL;
  - comment/ket luan de gui lai benh nhan.
- Phiếu NCKH/ground-truth can UX tot hon:
  - auto-fill tu video/evaluation;
  - validate field bat buoc;
  - nhat ky chinh sua.

### 3. Nghien cuu vien

- Chon benh nhan/video/bai tap/model trong React.
- Chay lai voi model khac.
- Cancel/retry/rerun analysis job.
- Theo doi tien trinh 4 buoc ro rang:
  - validate/transcode;
  - MediaPipe pass 1;
  - overlay/export;
  - artifact/persist.
- Train/cap nhat pose classifier qua backend.
- Ap dung ML cho video da phan tich.
- Dong bo/keo ket qua tu Hugging Face Dataset.
- Gui bao cao AI cho bac si va benh nhan.

### 4. Video/Media

- Gallery frames trong React.
- Doc ZIP frames hoac JSON frames qua backend.
- Phan trang frames.
- Loc frame theo PASS/NEAR/FAIL.
- Preview CSV/bieu do trong UI, khong chi download.
- Media token tam thoi cho backend React moi, tuong duong muc an toan Streamlit da harden.
- Download/artifact hien co gom:
  - video goc;
  - video da phan tich;
  - frames ZIP;
  - CSV;
  - JSON frames.
- Backlog media con lai:
  - anh bieu do PNG/SVG chua co artifact/endpoint rieng; hien React chi preview chart tu JSON endpoint va download du lieu nguon.

### 5. Lich nhac

- React da co tao/xoa lich cho bac si/admin, nhung can hoan thien:
  - filter theo loai lich;
  - trang thai hoan thanh/qua han;
  - view lich theo ngay/tuan;
  - export/print lich cho benh nhan.

### 6. Admin

- Admin panel van con toi thieu, can bo sung:
  - xem/export audit log;
  - revoke all sessions;
  - khoa/mo khoa user;
  - reset password bat buoc doi mat khau;
  - cleanup/reset du lieu theo tung nhom;
  - backup truoc thao tac destructive;
  - trang thai storage/artifact/model.

### 7. Trang noi dung

- Huong dan su dung.
- Kien thuc PHCN.
- Cong nghe AI.
- Thong tin de tai.
- Thanh vien/lien he/phan hoi.

## Lo trinh uu tien

## Phase A - Ket qua chi tiet cho benh nhan va bac si

Trang thai: Da trien khai slice dau tien ngay 2026-06-19.

Muc tieu: bien artifact API da co thanh man hinh ket qua co gia tri thuc te.

Viec can lam:

- [x] Tao subpanel `Ket qua chi tiet` trong tab Video.
- [x] Gom ket qua theo video qua `GET /videos/{stored_filename}/results`:
  - thong tin video;
  - evaluation cua bac si;
  - analysis job moi nhat;
  - metrics;
  - artifact available/download.
- [x] Hien thi ket qua cho benh nhan bang ngon ngu de hieu.
- [x] Hien thi ket qua cho bac si voi chi tiet lam sang hon.
- [x] Them empty/error/loading states co ban.
- [x] Them e2e smoke:
  - patient xem ket qua cua minh;
  - doctor scope duoc verify bang backend unit test.

Tieu chi xong:

- [x] Benh nhan khong xem duoc video/ket qua ngoai scope.
- [x] Bac si chi xem benh nhan duoc gan.
- [x] `npm run lint`, `npm run build`, backend unit va Playwright smoke pass trong slice.

## Phase B - Frame gallery va bieu do

Trang thai: Da trien khai ngay 2026-06-19.

Muc tieu: React xem duoc frames/bieu do thay vi chi tai artifact.

Viec can lam:

- [x] Backend endpoint doc manifest frame:
  - tu `all_frames_data_path`;
  - tu `frames_zip_path`;
  - tra pagination metadata.
- [x] Backend endpoint lay frame image an toan theo token/scope.
- [x] React gallery:
  - phan trang;
  - loc PASS/NEAR/FAIL;
  - hien goc/label neu co.
- [x] Xem frame lon/modal.
- [x] Preview chart tu CSV/JSON:
  - summary metrics;
  - chart nhe trong React.

Tieu chi xong:

- [x] ZIP/path traversal bi reject.
- [x] Gallery khong tai toan bo ZIP lon mot luc.
- [x] E2E smoke co fixture artifact nho.
- [x] Preview chart dung scope video va gioi han kich thuoc/so dong artifact.

## Phase B+ - Parity Streamlit cho ket qua va gallery

Trang thai: Da trien khai ngay 2026-06-19.

Muc tieu: dua React/backend gan hon voi gia tri lam sang cua Streamlit ma khong port nguyen cac workaround UI/session/HF vao frontend.

Nguyen tac:

- Port cac tinh nang nguoi dung can de doc ket qua lam sang/nghien cuu.
- Khong dua debug expander, recovery OpenCV/ffmpeg, HF lazy download hay train/apply ML truc tiep vao React trong phase nay.
- Cac xu ly nang neu can phai nam o backend API/job rieng.

Viec can lam:

- [x] Gate bac si xem ket qua AI giong Streamlit:
  - chi hien AI detail khi NCV da gui bao cao chinh thuc;
  - patient van xem summary phu hop neu co ket qua duoc phep;
  - backend tra trang thai `report_sent`/`report_status` theo video.
- [x] Frame gallery theo giai doan:
  - tinh/tra segment G1/G2/G3 tu JSON frames bang logic tuong duong `segment_frames`;
  - loc theo `ALL`, `G1`, `G2`, `G3`, `PASS`, `NEAR`, `FAIL`;
  - dung nguong sai so G1/G2/G3 `45/30/15` khi tinh PASS/NEAR/FAIL theo giai doan.
- [x] Hien thi REF + ML badge neu artifact co du lieu:
  - `ml_label`, `ml_label_text`, `ml_confidence`, `ml_probabilities`;
  - giai thich ngan gon y nghia REF va ML confidence trong React;
  - khong train/apply model trong man hinh ket qua.
- [x] Chart parity toi thieu:
  - summary theo G1/G2/G3;
  - chart goc vai/khuyu theo phase/filter;
  - neu co metrics giai doan trong artifact thi hien `accuracy`, `MAE`, `F1`, `ICC` theo phase.
- [x] Export/parity download:
  - giu download artifact hien co;
  - chart preview tra JSON qua endpoint rieng va co the dung lam chart data preview;
  - chua cat video G1/G2/G3 trong frontend neu backend chua co API rieng.

Tieu chi xong:

- [x] Bac si khong thay AI detail truoc khi NCV gui bao cao.
- [x] Frame G1/G2/G3 trong React cho ket qua dem PASS/NEAR/FAIL khop fixture Streamlit.
- [x] ML badge hien dung khi co field ML va degrade sach khi artifact cu khong co ML.
- [x] Chart preview khong tra qua 180 diem moi series va khong doc artifact vuot gioi han.
- [x] Co backend unit tests cho segment/filter/report gate va Playwright smoke cho G1/G2/G3 + ML badge.

## Phase C - Analysis job nang cao

Trang thai: Da trien khai job lifecycle va queue nhe ngay 2026-06-19.

Muc tieu: NCV dieu khien job AI nhu mot workflow that, khong chi start/poll.

Viec can lam:

- Them API:
  - [x] cancel job;
  - [x] retry job;
  - [x] rerun voi model khac;
  - [x] list job history theo video.
- Them model selection trong React:
  - [x] MediaPipe Heavy/Full/Lite;
  - [x] skip step;
  - [x] resize width;
  - [x] confidence.
- [x] Hien thi tien trinh 4 buoc.
- [x] Tach runner dai han ra background job queue nhe neu can.

Tieu chi xong:

- [x] Cancel khong lam hong progress file.
- [x] Retry/rerun tao job moi co metadata ro.
- [x] Job loi hien error message co ich, khong leak path/token nhay cam.
- [x] Neu bat AI runner that trong production, can queue/worker rieng thay vi thread process noi bo.

## Phase D - Pose classifier va ML workflow

Trang thai: Da trien khai slice train/apply backend + React dau tien ngay 2026-06-19.

Muc tieu: dua train/apply classifier tu Streamlit/script sang backend/React.

Viec can lam:

- [x] Backend API train pose classifier.
- [x] Backend API xem status model/checksum.
- [x] Backend API apply classifier cho video da phan tich.
- [x] React UI cho NCV:
  - [x] train;
  - [x] dry-run;
  - [x] apply;
  - [x] xem ket qua gan nhan ML.

Tieu chi xong:

- [x] Model pickle/joblib van can checksum sidecar.
- [x] Train/apply co dry-run.
- [x] Khong block request lau neu job nang.

## Phase E - Hugging Face sync/report

Trang thai: Da trien khai slice guarded sync/report backend + React dau tien ngay 2026-06-19.

Muc tieu: dua sync/report vao backend co guard thay vi thao tac thu cong trong Streamlit.

Viec can lam:

- [x] API sync metadata/artifacts tu HF Dataset.
- [x] API upload artifact hop le len HF Dataset.
- [x] API tao report/export sanitize.
- [x] UI NCV/admin:
  - [x] sync status;
  - [x] dry-run;
  - [x] conflict summary;
  - [x] download report.

Tieu chi xong:

- [x] Mac dinh khong sync `users.json`.
- [x] Token khong bao gio tra ve frontend.
- [x] Moi sync co audit log.

## Phase F - Admin van hanh

Trang thai: Da trien khai audit/session/user ops va cleanup/reset theo nhom ngay 2026-06-19.

Muc tieu: React thay the cac tac vu quan tri quan trong cua Streamlit.

Viec can lam:

- [x] Audit log API/UI.
- [x] Revoke all sessions API/UI.
- [x] Lock/unlock user.
- [x] Reset password voi `must_change_password`.
- [x] Cleanup/reset tung nhom du lieu:
  - [x] evaluations;
  - [x] symptoms;
  - [x] schedules;
  - [x] videos/temp files;
  - [x] processed artifacts.
- [x] Backup truoc destructive action user ops.
- [x] Backup truoc cleanup/reset tung nhom du lieu.

Tieu chi xong:

- [x] Revoke all sessions va delete user can confirm text.
- [x] Cleanup/reset destructive action can confirm text.
- [x] Co audit log gom actor, action, target, timestamp.
- [x] Co test permission admin-only cho audit/revoke/lock/reset.

## Phase G - Static/info pages

Trang thai: Da trien khai React info workspace + feedback backend ngay 2026-06-19.

Muc tieu: React co day du cac trang noi dung dang con o Streamlit/docs.

Viec can lam:

- [x] Huong dan su dung theo role.
- [x] Kien thuc PHCN.
- [x] Cong nghe AI.
- [x] Thong tin de tai/thanh vien/lien he.
- [x] Form phan hoi neu can backend luu.

Tieu chi xong:

- [x] Noi dung khong chua PII/default password/secrets.
- [x] Responsive mobile/desktop.
- [x] Khong lam thanh landing page marketing; giu trong workspace/app shell.

## Phase H - Production gate

Trang thai: Da trien khai CI/release gate, migration dry-run/apply/rollback va checklist production ngay 2026-06-19. Runtime van doc JSON; SQLite migration hien la gate/tooling truoc khi switch repository storage.

Ghi chu parity storage: hien co tooling JSON -> SQLite va test dry-run/apply/rollback. Backend runtime van dung `JsonRepository` doc/ghi JSON; chua co Postgres runtime repository switch va chua doi storage production sang SQLite/Postgres.

Muc tieu: truoc deploy that, co gate tu dong va migration data ben vung.

Viec can lam:

- [x] CI:
  - [x] Python unit tests;
  - [x] backend compile;
  - [x] `npm run lint`;
  - [x] `npm run build`;
  - [x] Playwright smoke.
- [x] Tooling migration JSON sang SQLite.
- [ ] Runtime repository switch sang SQLite hoac Postgres khi co quyet dinh deploy production.
- [x] Background job system cho AI.
- [x] Privacy/compliance checklist.
- [x] Release checklist.

Tieu chi xong:

- [x] Khong deploy neu unit/build/e2e smoke fail.
- [x] Co backup/migration rollback.
- [x] Co config production rieng cho secrets/CORS/storage.

## Thu tu lam gan nhat

1. Phase A: Ket qua chi tiet cho benh nhan va bac si.
2. Phase B: Frame gallery va preview bieu do.
3. Phase B+: Parity Streamlit cho report gate, G1/G2/G3, ML badge va chart theo phase.
4. Phase C: Cancel/retry/rerun analysis job va chon model.
5. Phase F: Admin revoke/audit/reset vi lien quan van hanh an toan.
6. Phase D/E: Pose classifier va HF sync khi analysis UI da on dinh.
7. Phase G/H: Static pages va production gate.

## Ghi chu ve khoang trong backend da cap nhat

Danh sach cu co cac muc nhu "chua co upload API", "chua co media API", "chua co analysis job API", "chua co CRUD symptoms/schedules/evaluations/research". Cac muc nay hien da duoc lấp o muc co ban. Viec con lai khong phai la tao API tu dau, ma la lam day du workflow, UX, export, job lifecycle, media gallery, admin operations va production gates.
