# Phase 2 Plan - Stability va Access Control

Muc tieu: chong OOM/mat du lieu/truy cap sai quyen. Phase nay bien cac guard bao mat thanh co che tap trung: upload validation, path containment, ZIP safe extraction, permission guard, destructive confirm va session revocation.

## Nguyen tac rieng cua Phase 2

- Moi mutation phai co role guard trong business logic, khong chi an nut UI.
- Moi write/delete filesystem phai qua path containment.
- Moi destructive action phai co backup va audit log.
- Khong refactor UI lon neu chua can.

## 2.1 - Giam upload size va validate video

Lien quan loi: F08, F09, N05, N06, N13

File can sua:

- `.streamlit/config.toml`
- `Dockerfile`
- `app.py`

Vung code hien tai:

- `file_upload.getbuffer()` quanh `app.py:19464-19467`, `app.py:19516-19519`
- `MAX_FILE_SIZE_MB`
- ffmpeg/ffprobe helpers

Thu tu thuc hien:

1. Doi `maxUploadSize` production ve 300 MB trong `.streamlit/config.toml` va Dockerfile.
2. Tao constant duy nhat `MAX_UPLOAD_SIZE_MB = 300`.
3. Truoc khi goi `getbuffer()`, kiem tra `file_upload.size`.
4. Reject som neu:
   - size qua gioi han.
   - extension khong trong allowlist.
   - MIME/magic bytes khong phai video.
5. Chay `ffprobe` truoc transcode, validate:
   - duration toi da.
   - resolution toi da.
   - co video stream.
   - codec/container hop le.
6. Gioi han ffmpeg `-threads 2`, khong dung `-threads 0`.
7. Them timeout cho moi `subprocess.Popen` ffmpeg/ffprobe.

Lenh verify:

```powershell
rg -n "maxUploadSize|MAX_FILE_SIZE_MB|MAX_UPLOAD_SIZE_MB|getbuffer\\(|-threads 0|subprocess\\.Popen" app.py Dockerfile .streamlit
python -m py_compile app.py
```

Tieu chi xong:

- File >300MB bi reject truoc khi doc buffer.
- File khong phai video bi reject truoc pipeline.
- ffmpeg khong chiem tat ca CPU va khong treo vo han.

## 2.2 - Bat lai CORS/XSRF trong production

Lien quan loi: F07

File can sua:

- `Dockerfile`
- `.streamlit/config.toml`
- Co the tao `.streamlit/config.dev.toml`

Thu tu thuc hien:

1. Xoa `--server.enableCORS=false`.
2. Xoa `--server.enableXsrfProtection=false`.
3. Tach config dev neu local can override.
4. Test app start trong container/local.

Lenh verify:

```powershell
rg -n "enableCORS=false|enableXsrfProtection=false" Dockerfile .streamlit
```

Tieu chi xong:

- Docker production khong tat CORS/XSRF.

## 2.3 - Path containment guard cho download/sync/resolve

Lien quan loi: R02

File can sua:

- `app.py`
- Co the tao helper `utils/path_security.py` neu can

Vung code hien tai:

- `get_clean_rel_path()`
- `_hf_download_via_http()`
- `_hf_download_dataset_file()`
- `ensure_local_file()`
- `download_file_with_progress()`

Thu tu thuc hien:

1. Tao helper duy nhat `safe_data_path(rel_path, allowed_roots, *, must_exist=False)`.
2. Helper phai:
   - Reject path absolute tu input.
   - Reject `..`.
   - Normalize slash.
   - Resolve target bang `Path.resolve()`.
   - Enforce target nam trong mot allowed root da resolve.
3. Dinh nghia allowed roots:
   - `patient_uploads/`
   - `processed_results/`
   - `database/` chi cho file JSON duoc phep.
   - cache media neu co.
4. Ap dung helper truoc moi write/delete/download.
5. Khong xoa file cu trong helper download neu target chua qua guard.
6. Log warning cho path bi reject, khong log token.

Lenh verify:

```powershell
rg -n "get_clean_rel_path|_hf_download_via_http|_hf_download_dataset_file|ensure_local_file|download_file_with_progress|os\\.remove|os\\.replace|open\\(" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Input `../database/users.json` bi reject o moi download/resolve path.
- Absolute path ngoai workspace bi reject.
- Cac write/delete quan trong di qua containment guard.

## 2.4 - ZIP frames extract an toan

Lien quan loi: R03

File can sua:

- `app.py`

Vung code hien tai:

- `check_and_extract_frames_zip()`
- `zip_ref.extractall(frames_dir)`

Thu tu thuc hien:

1. Xoa `extractall()`.
2. Duyet `ZipInfo` truoc khi extract.
3. Gioi han:
   - so entry toi da.
   - tong uncompressed size.
   - size moi entry.
4. Chi cho phep file anh: `.jpg`, `.jpeg`, `.png`, `.webp` neu can.
5. Filename phai la basename hop le, khong separator, khong absolute, khong `..`.
6. Resolve path dich va check nam trong `frames_dir`.
7. Extract tung file bang stream copy, khong trust metadata symlink.

Lenh verify:

```powershell
rg -n "extractall|ZipInfo|infolist|check_and_extract_frames_zip" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Zip bomb/file count explosion bi reject.
- ZIP khong the ghi ra ngoai `frames_dir`.

## 2.5 - `sync_from_hf.py` khong ghi de users mac dinh

Lien quan loi: R05

File can sua:

- `scripts/sync_from_hf.py`

Thu tu thuc hien:

1. Loai `users.json` khoi `SYNC_FILES` mac dinh.
2. Them flag `--include-users` neu that su can sync auth DB.
3. Neu sync users:
   - Backup timestamped truoc khi ghi.
   - Merge theo username.
   - Khong ghi de `password`, `role`, `hash_version`, `must_change_password` neu local moi hon hoac khong co xac nhan.
4. Backup file theo timestamp, khong chi `.bak` mot phien ban.
5. Dry-run phai cho biet `users.json` se bi bo qua neu khong co flag.

Lenh verify:

```powershell
python -m py_compile scripts\sync_from_hf.py
python scripts\sync_from_hf.py --dry-run
rg -n "users\\.json|include-users|\\.bak" scripts\sync_from_hf.py
```

Tieu chi xong:

- Chay mac dinh khong dong bo `users.json`.
- Muon sync users phai co flag ro.
- Backup timestamped hoat dong.

## 2.6 - Confirm flow, backup va audit log cho destructive actions

Lien quan loi: F10, F11, F34

File can sua:

- `app.py`
- `scripts/reset_data.py`
- Co the tao `database/audit_log.jsonl`

Vung code hien tai:

- Admin destructive actions quanh `app.py:18252-18299`
- `delete_video_callback()` quanh `app.py:18521-18550`

Thu tu thuc hien:

1. Tao helper `create_backup_before_destructive(action, files)`.
2. Tao helper `write_audit_log(actor, role, action, target, result, metadata=None)`.
3. Them confirm 2 buoc:
   - Click lan 1 set pending action.
   - Lan 2 yeu cau nhap text confirm hoac checkbox + button rieng.
4. `delete_video_callback()`:
   - Tu kiem tra role.
   - Dung `.get()` thay `[]`.
   - Filter evaluation theo `(patient_username, video_name, exercise)` neu co.
   - Backup truoc khi xoa.
5. `scripts/reset_data.py`:
   - Them `--dry-run`.
   - Them `--yes`.
   - Backup timestamped.
   - Khong chay destructive neu thieu `--yes`.

Lenh verify:

```powershell
rg -n "delete_video_callback|reset|audit|backup|dry-run|--yes" app.py scripts\reset_data.py
python -m py_compile app.py scripts\reset_data.py
```

Tieu chi xong:

- Khong co xoa/reset chi bang mot click.
- Moi destructive action co backup va audit log.
- Callback business logic co guard quyen.

## 2.7 - Permission guard tap trung

Lien quan loi: F18, F11

File can sua:

- `app.py`
- Co the tach `auth.py` nho neu khong gay refactor lon

Thu tu thuc hien:

1. Tao `get_current_actor()` doc `st.session_state`.
2. Tao `require_role(*allowed_roles, action=None, target=None)`:
   - Tra ve actor neu hop le.
   - Raise/return loi ro neu khong co quyen.
   - Ghi audit denied.
3. Dinh nghia permission matrix trong code comment/doc gan helper.
4. Ap dung cho:
   - upload.
   - evaluate.
   - delete video.
   - reset system.
   - edit schedules.
   - sync/admin operations.
5. UI van an nut theo role, nhung business logic bat buoc check lai.

Lenh verify:

```powershell
rg -n "require_role|get_current_actor|delete_video_callback|save_data\\(|upload|reset" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Moi callback ghi/xoa du lieu co role guard.
- Denied actions duoc audit.

## 2.8 - Admin session revocation

Lien quan loi: F23

File can sua:

- `app.py`
- `database/session_state.json` hoac storage tuong duong

Thu tu thuc hien:

1. Tao `global_session_version` trong storage.
2. Khi login thanh cong, luu version vao session.
3. Moi rerun so session version voi global.
4. Khi admin reset/revoke/security incident, tang global version.
5. Mismatch thi clear session va force logout.

Lenh verify:

```powershell
rg -n "session_version|global_session_version|logout|logged_in" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Admin revoke/reset lam tat ca session cu bi logout.

## 2.9 - Sua reset password, Google login, pickle/joblib va login UX

Lien quan loi: F03, N04, F04, N14, F30, F19

File can sua:

- `app.py`
- `utils/checkpoint_utils.py`
- `utils/pose_classifier_utils.py`

Thu tu goi y:

1. Reset password:
   - Tam tat trong app neu chua co email token that.
   - Hoac them token ngau nhien, expiry 15 phut, hash token, rate limit.
   - Sua `_auth_lookup_key()` de khong match collision theo `full_name` casefold.
2. Google login:
   - Chi chap nhan callback OIDC hop le.
   - Map email vao user record co san.
   - Khong auto role `Bệnh nhân` cho moi email.
   - Them allowlist domain neu can.
3. Pickle/joblib:
   - Verify checksum truoc `pickle.load`/`joblib.load`.
   - Khong load tu user upload/cloud sync chua verify.
4. Login UX:
   - Bo chon role truoc login.
   - Xac thuc username/password truoc, redirect theo role DB.
   - Loi login khong tiet lo role/account existence.

Lenh verify:

```powershell
rg -n "_auth_lookup_key|reset|Google|st\\.user|pickle\\.load|joblib\\.load|hashlib|checksum" app.py utils
python -m py_compile app.py utils\checkpoint_utils.py utils\pose_classifier_utils.py
```

Tieu chi xong:

- Reset password khong con dua vao username+email don thuan.
- Google login khong tu cap role cho email bat ky.
- Load pickle/joblib co checksum.
- Login khong can chon role truoc.

## Definition of Done Phase 2

- Upload co size/type/probe validation va timeout.
- Production khong tat CORS/XSRF.
- Path/ZIP operations co containment va quota.
- `sync_from_hf.py` khong ghi de users mac dinh.
- Destructive actions co confirm, backup, audit.
- Moi mutation co permission guard.
- Session revoke hoat dong.
- Auth flows phu tro da harden.

