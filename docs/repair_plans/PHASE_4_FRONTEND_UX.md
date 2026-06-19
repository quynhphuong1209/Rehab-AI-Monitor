# Phase 4 Plan - Frontend va UX Improvement

Muc tieu: lam giao dien ben hon, giam CSS/JS fragile va chuan hoa schema hien thi sau khi cac van de bao mat/storage da duoc xu ly.

## Nguyen tac rieng cua Phase 4

- UI cleanup khong duoc lam suy yeu permission guard Phase 2.
- CSS/JS tach dan theo khu vuc, khong thay doi toan bo visual mot lan.
- Moi du lieu render ra UI van phai escape/validate.
- Uu tien on dinh thao tac lap lai cua bac si, benh nhan, NCV, admin.

## 4.1 - Schema validation cho JSON

Lien quan loi: F13, F14

File can tao/sua:

- `models/schemas.py`
- `app.py`
- `database/database_schema.md`
- Migration script neu can

Thu tu thuc hien:

1. Chon dataclass hoac Pydantic. Neu muon it dependency, bat dau bang dataclass + validator functions.
2. Dinh nghia schema cho:
   - User
   - Video
   - Evaluation
   - Schedule
   - PatientSymptom
3. Khi load JSON:
   - Validate type root.
   - Fill default cho optional fields.
   - Reject/log record hong nghiem trong.
4. Doi access `dict["key"]` sang `.get()` hoac object normalized.
5. Tao migration idempotent:
   - Backup truoc.
   - Add missing fields.
   - Normalize date/time.
6. Cap nhat `database_schema.md`.

Lenh verify:

```powershell
rg -n "\\[[\"'][A-Za-z0-9_]+[\"']\\]" app.py
python -m py_compile app.py models\schemas.py
pytest tests\test_schema.py
```

Tieu chi xong:

- Record JSON thieu field khong lam crash tab.
- Schema docs khop code.

## 4.2 - Tach CSS ra file rieng

Lien quan loi: F15, F35

File can tao/sua:

- `assets/styles.css`
- `assets/theme_dark.css`
- `assets/theme_light.css`
- `app.py`

Thu tu thuc hien:

1. Gom cac CSS block lap lai tu `app.py` vao assets.
2. Tao helper load CSS co cache, vi du `load_css_file(path)`.
3. Scope selector theo class/container rieng, giam:
   - `*`
   - `div`
   - `span`
   - `.stButton > button` global
   - `!important`
4. Tao variant destructive button ro visual.
5. Thay hover zoom frame `scale(2.2)` bang modal/lightbox hoac click-to-open.
6. Kiem tra mobile/desktop khong overlap controls.

Lenh verify:

```powershell
rg -n "<style|unsafe_allow_html=True|!important|scale\\(2\\.2\\)|\\.stButton > button|\\* \\{" app.py assets
python -m py_compile app.py
```

Tieu chi xong:

- CSS chinh nam trong assets.
- Nut destructive co visual khac nut thuong.
- Hover frame khong che nut/controls.

## 4.3 - Loai bo JS DOM hacks

Lien quan loi: F16

File can sua:

- `app.py`
- Co the tao Streamlit component rieng neu that su can

Vung code hien tai:

- JS click tab qua `window.parent.document`
- JS clock
- wheel/MutationObserver hacks

Thu tu thuc hien:

1. Thay navigation bang `st.session_state`.
2. Thay JS clock bang server-side render hoac component co boundary ro.
3. Neu can JS:
   - Co lap vao component rieng.
   - Khong truy cap `window.parent.document`.
   - Khong phu thuoc DOM/testid Streamlit noi bo.
4. Xoa JS khong con can.

Lenh verify:

```powershell
rg -n "window\\.parent|document\\.querySelector|MutationObserver|setInterval|components\\.html" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Navigation khong dua vao click DOM.
- App it vo khi Streamlit doi DOM.

## 4.4 - Phan quyen bac si-benh nhan va NCV pseudonymized

Lien quan loi: F24, N12, N19

File can sua:

- `app.py`
- `models/schemas.py`
- `database/users.json` hoac migration sanitize

Thu tu thuc hien:

1. Them truong assignment:
   - user bac si co danh sach patient IDs.
   - patient co assigned doctor/team neu can.
2. Filter data theo actor:
   - Benh nhan: chi xem minh.
   - Bac si/KTV: chi xem benh nhan phu trach.
   - NCV: chi xem pseudonymized data.
   - Admin: xem day du.
3. Ap dung filter o data access, khong chi UI.
4. gTTS:
   - Ghi ro privacy note.
   - Cho phep disable network TTS trong config.
   - Cache audio/local assets neu co.
5. WebRTC/STUN:
   - Document policy.
   - Them consent/session limit neu dung.

Lenh verify:

```powershell
rg -n "assigned|doctor|patient|NCV|Nghiên cứu viên|gTTS|stun:stun\\.l\\.google\\.com" app.py models docs
python -m py_compile app.py
```

Tieu chi xong:

- Bac si khong xem duoc benh nhan ngoai pham vi.
- NCV khong thay PII neu khong duoc phep.
- Network privacy dependencies co config/policy.

## 4.5 - Cleanup data sync/report script

Lien quan loi: F38

File can sua:

- `scripts/sync_data_and_report.py`
- `README.md`
- `docs/generated/`

Thu tu thuc hien:

1. Script chi doc/ghi trong mot data directory ro.
2. Bo logic copy JSON runtime ra root.
3. README chi chua summary sanitize va link docs.
4. Generated reports mac dinh vao `docs/generated/`.
5. Them `--dry-run` neu script co ghi file.

Lenh verify:

```powershell
rg -n "copy|shutil|README\\.md|users\\.json|doctor_evaluations\\.json|video_list\\.json" scripts\sync_data_and_report.py
python -m py_compile scripts\sync_data_and_report.py
```

Tieu chi xong:

- Script khong tao JSON runtime o root.
- README khong bi ghi de bang du lieu nhay cam.

## 4.6 - Dat ten ham/thread ro rang

Lien quan loi: F28

File can sua:

- `app.py`

Thu tu thuc hien:

1. Tim nested function/callback ten chung:
   - `_worker`
   - `_frag`
   - `_render`
   - `_callback`
2. Rename theo vai tro:
   - `_hf_upload_worker`
   - `_media_prefetch_worker`
   - `_job_status_fragment`
3. Doi ten theo tung vung nho, compile sau moi batch.
4. Khong doi behavior.

Lenh verify:

```powershell
rg -n "def _worker|def _frag|threading\\.Thread\\(" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Thread/callback quan trong co ten debug duoc.
- Khong thay doi logic.

## Definition of Done Phase 4

- JSON schema validation lam UI khong crash voi record thieu field.
- CSS chinh duoc tach, scoped hon, destructive UI ro hon.
- JS DOM hacks duoc thay bang state/component co boundary.
- Doctor-patient scoping va NCV pseudonymization duoc enforce.
- Sync/report script khong tai tao PII/root JSON.
- Naming giup debug thread/callback de hon.

