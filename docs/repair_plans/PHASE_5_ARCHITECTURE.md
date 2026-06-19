# Phase 5 Plan - Tach Monolith `app.py`

Muc tieu: giam rui ro phat trien lau dai bang cach tach `app.py` theo modules. Phase nay chi nen bat dau khi Phase 3 co test suite va Phase 4 da giam cac UI/schema fragile lon.

## Nguyen tac rieng cua Phase 5

- Khong tach toan bo `app.py` mot lan.
- Moi slice phai co test hoac smoke check truoc va sau.
- Giu behavior nguoi dung nhu cu trong moi slice.
- Module moi khong duoc import Streamlit neu la domain/storage/auth pure logic.
- Top-level import khong duoc tu khoi dong thread, sync cloud, download file, migrate data.

## Kien truc muc tieu

```text
app.py
auth/
  __init__.py
  passwords.py
  sessions.py
  permissions.py
storage/
  __init__.py
  json_store.py
  migrations.py
  schemas.py
cloud/
  __init__.py
  hf_sync.py
video/
  __init__.py
  io.py
  processing.py
  serving.py
ui/
  __init__.py
  patient.py
  doctor.py
  researcher.py
  admin.py
  styles.py
```

Neu repo muon toi thieu file, co the bat dau bang `auth.py`, `cloud/hf_sync.py`, `video/io.py`, `storage/json_store.py`, roi tach folder sau.

## 5.1 - Slice Auth

File can tao/sua:

- `auth/passwords.py`
- `auth/sessions.py`
- `auth/permissions.py`
- `app.py`
- `tests/test_auth.py`
- `tests/test_password.py`

Pham vi di chuyen:

- Hash/verify/migration password.
- Login state/session version.
- `require_role()`.
- Permission matrix.
- Auth lookup.

Thu tu thuc hien:

1. Copy helper pure auth ra module moi.
2. Viet test cho module moi.
3. Trong `app.py`, thay implementation bang import.
4. Chay test auth/password.
5. Xoa code cu chi khi import moi da on.

Tieu chi xong:

- `app.py` khong con logic hash password chi tiet.
- Permission guard import tu auth module.
- Test pass.

## 5.2 - Slice Storage

File can tao/sua:

- `storage/json_store.py`
- `storage/migrations.py`
- `storage/schemas.py` hoac `models/schemas.py`
- `app.py`
- `tests/test_storage.py`
- `tests/test_schema.py`

Pham vi di chuyen:

- `load_data()`, `save_data()`, `doc_lock_save_data()`.
- Atomic write/file lock/update_json.
- Schema normalize/migration.

Thu tu thuc hien:

1. Dam bao `storage/json_store.py` da co tu Phase 3.
2. Chuyen wrapper cu trong `app.py` sang call module.
3. Tach constants file path neu can.
4. Chay tests.
5. Xoa duplicate helper cu.

Tieu chi xong:

- Moi JSON IO di qua `storage`.
- `app.py` khong con nhieu implementation IO.

## 5.3 - Slice Cloud/HF Sync

File can tao/sua:

- `cloud/hf_sync.py`
- `app.py`
- `scripts/sync_from_hf.py` neu muon dung chung logic
- Tests/smoke cho path/token

Pham vi di chuyen:

- HF config reading.
- HF upload/download.
- Retry/backoff.
- Token masking.
- Dataset path validation.

Thu tu thuc hien:

1. Tao API nho:
   - `download_dataset_file(rel_path, target_path)`.
   - `upload_dataset_file(local_path, rel_path)`.
   - `list_dataset_files()`.
2. Token chi nam trong module cloud.
3. `app.py` khong tu tao URL HF client-side.
4. Script co the import module nay de tranh duplicate.
5. Chay compile va smoke.

Tieu chi xong:

- `app.py` khong con request HF thap cap rai rac.
- Token handling tap trung.

## 5.4 - Slice Video

File can tao/sua:

- `video/io.py`
- `video/processing.py`
- `video/serving.py`
- `app.py`
- `tests/test_video.py`

Pham vi di chuyen:

- Filename sanitize.
- Path containment media.
- ffprobe/ffmpeg/transcode.
- ZIP frames extract.
- HTTP/range serving.

Thu tu thuc hien:

1. Di chuyen helper pure path/validation truoc.
2. Di chuyen ffprobe/ffmpeg wrappers.
3. Di chuyen ZIP extract.
4. Di chuyen media serving neu co test thu cong.
5. UI `app.py` chi goi service function.

Tieu chi xong:

- Video IO/processing khong nam rai rac trong UI code.
- Path guard va ZIP tests pass.

## 5.5 - Slice UI theo role/tab

File can tao/sua:

- `ui/patient.py`
- `ui/doctor.py`
- `ui/researcher.py`
- `ui/admin.py`
- `ui/styles.py`
- `app.py`

Pham vi di chuyen:

- Render tab/panel theo role.
- CSS load helper.
- UI-only helpers.

Thu tu thuc hien:

1. Chon mot role it phu thuoc nhat de tach truoc.
2. Di chuyen render function + UI helper lien quan.
3. Truyen dependencies vao function, tranh import vong:
   - current_user.
   - services.
   - storage functions.
4. `app.py` lam orchestrator route theo role.
5. Chay smoke thu cong role do.
6. Lap lai cho role tiep theo.

Tieu chi xong:

- `app.py` giam manh dong render UI.
- Moi UI module chi phu trach role/tab cua no.

## 5.6 - Tach side effects khoi boot sequence

Trang thai: **Da hoan thanh** (2026-06-18)

File can sua:

- `app.py`
- Modules moi

Thu tu thuc hien:

1. Tim top-level side effects:
   - start thread.
   - auto sync HF.
   - create dirs/write files.
   - load model/checkpoint.
2. Tao `app_startup()` idempotent.
3. Moi side effect phai:
   - co guard run-once.
   - co config enable/disable.
   - co timeout/backoff neu network.
4. Import module khong duoc tao side effect.

Lenh verify:

```powershell
rg -n "threading\\.Thread\\(|while True|start\\(\\)|upload_file|download|os\\.makedirs|save_data\\(" app.py auth storage cloud video ui
python -m py_compile app.py auth\*.py storage\*.py cloud\*.py video\*.py ui\*.py
pytest tests/
```

Tieu chi xong:

- Import module khong tu khoi dong job/network/write.
- Startup co kiem soat.
- Da them `app_startup.py` de gom process/env/logging/page config, tao thu muc runtime va thread boot vao mot luong startup co guard/config.
- Da dua khoi tao `st.session_state`, Google identity va CSS injection vao runtime init thay vi chay top-level khi import.

## Definition of Done Phase 5

- `app.py` chu yeu la setup Streamlit, startup, route role/tab.
- Auth/storage/cloud/video co module rieng va tests.
- UI role duoc tach theo slice.
- Khong co side effect nguy hiem o import time.
- Test suite pass sau moi slice.
