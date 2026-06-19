# React Frontend

Frontend React/Vite moi, dung song song voi Streamlit trong giai doan tach frontend/backend.

## Chay local

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd web
npm install
npm run dev
```

Mo `http://127.0.0.1:5173`.

## Smoke test

Chay smoke React + backend bang Playwright:

```powershell
cd web
npx playwright install chromium
npm run e2e:smoke
```

Smoke test tao database tam trong `scratch/web-e2e-smoke/`, chay backend tren
`8010`, frontend tren `5183`, dang nhap qua UI, kiem tra dashboard benh nhan
va mo panel ket qua chi tiet/frame gallery trong tab Video, tab Thong tin va form phan hoi.

Production gate day du:

```powershell
cd ..
.\.venv\Scripts\python.exe -m pytest tests\unit
.\.venv\Scripts\python.exe scripts\migrate_json_to_sqlite.py --repo-root . --dry-run
cd web
npm run lint
npm run build
npm run e2e:smoke
```

## Design

Giao dien dung token tu `DESIGN.md`:

- `#F8FAFC` cho nen workspace.
- `#FFFFFF` + border `#E2E8F0` + radius `8px` cho panel/card.
- `#0284C7` cho CTA va active/focus.
- Grid spacing 8/16/24/32px.

## Pham vi hien tai

- Login va dang ky benh nhan bang backend API.
- Benh nhan co form khai bao trieu chung trong tab `Trieu chung`, luu qua backend API.
- Benh nhan co form upload video trong tab `Video`, luu file va metadata qua backend API.
- Tab `Video` co the xem/an video upload bang authenticated media endpoint cua backend.
- Tab `Video` co cot tien do phan tich; admin/nghien cuu vien co the tao job, cac role duoc scope co the poll tien do, va dashboard tu reload mot lan khi job `success`.
- Tab `Video` cho admin/nghien cuu vien chon model MediaPipe Heavy/Full/Lite, skip step, resize width va confidence; co nut `Chay`, `Rerun`, `Retry`, `Huy` va `History`.
- Job AI hien thi 4 buoc tien trinh: validate/transcode, MediaPipe pass 1, overlay/export, artifact/persist.
- Tab `Video` co panel `Pose classifier ML` cho admin/nghien cuu vien xem checksum/model status, train/dry-run va apply/dry-run classifier cho video da phan tich.
- Tab `Video` co panel `Ket qua chi tiet` goi `/videos/{stored_filename}/results`, gom tom tat theo vai tro, nhan xet bac si, metrics AI, timeline va artifact download.
- Panel ket qua ton trong report gate: bac si/KTV chi thay AI detail khi NCV da gui bao cao chinh thuc; patient van thay summary phu hop.
- Panel ket qua co frame gallery goi `/videos/{stored_filename}/analysis-frames`, loc ALL/G1/G2/G3/PASS/NEAR/FAIL, hien REF threshold va ML badge neu artifact co ML, tai anh tung frame theo trang va xem frame lon trong modal.
- Panel ket qua co preview bieu do goi `/videos/{stored_filename}/analysis-chart`, ve SVG nhe tu CSV `df_path` hoac fallback JSON frame, kem summary G1/G2/G3 va phase metrics.
- Workspace doc du lieu theo vai tro:
  - Video va danh gia.
  - Ho so benh nhan.
  - Khai bao trieu chung.
  - Lich nhac cho admin/bac si/benh nhan.
  - Du lieu nghien cuu cho admin/nghien cuu vien.
- Tab `Dữ liệu NCKH/Phiếu nghiên cứu` co panel Hugging Face sync cho admin/nghien cuu vien: status, dry-run metadata sync, upload artifact va report sanitize.
- Tab `Thong tin` co huong dan theo role, kien thuc PHCN, cong nghe AI, thong tin de tai/doi ngu va form phan hoi luu qua backend; admin/nghien cuu vien xem duoc phan hoi gan day.
- Tab `Nguoi dung` cho admin tao/xoa tai khoan, khoa/mo khoa, reset password bat buoc doi mat khau, thu hoi phien theo user/toan bo, reset tung nhom du lieu co confirm/backup/audit va xem audit log gan nhat.
- Sidebar tab va metric cards cap nhat tu backend API.
- Dang ky self-service chi tao role `Benh nhan`; cac role nhay cam do admin cap.
- Backend job da validate/transcode H.264, chay qua queue worker nhe va co hook AI runner de cap nhat `video_list.json`; runner phan tich AI/MediaPipe that chua bat mac dinh.
