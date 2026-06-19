# Phase 3 Plan - Data Integrity va Quality

Muc tieu: tang do tin cay du lieu va tao safety net cho refactor. Phase nay tap trung vao password hashing, JSON storage lock, exception logging, validators, tests va dependency hygiene.

## Nguyen tac rieng cua Phase 3

- Sau Phase 2 moi refactor storage rong hon.
- Moi migration phai idempotent va co backup.
- Test can dung fixture sanitize, khong dung PII thật.
- Khong doi schema mot lan lon neu khong co migration.

## 3.1 - Doi password hashing sang Argon2/Bcrypt

Lien quan loi: F20

File can sua:

- `app.py`
- `requirements.txt`
- Co the tao `auth/passwords.py` neu da san sang tach nho

Thu tu thuc hien:

1. Chon Argon2 (`argon2-cffi`) hoac bcrypt. Uu tien Argon2.
2. Them dependency vao requirements.
3. Tao helper:
   - `hash_password_v2(password)`.
   - `verify_password(password, user_record)`.
   - `needs_password_rehash(user_record)`.
4. Giu verify SHA-256 cu chi de migration khi login thanh cong.
5. Sau login thanh cong voi hash cu:
   - Rehash bang Argon2.
   - Set `hash_version: "argon2"`.
   - Set `updated_at`.
6. Password moi/reset moi chi dung hash moi.

Lenh verify:

```powershell
rg -n "sha256|hash_password|verify_password|hash_version|argon2|bcrypt" app.py requirements.txt
python -m py_compile app.py
```

Tieu chi xong:

- Password moi dung Argon2/Bcrypt.
- Hash SHA-256 cu duoc migrate dan khi login thanh cong.
- Khong luu plaintext password.

## 3.2 - Chuan hoa JSON read/write voi lock

Lien quan loi: F12, N11

File can sua:

- Tao `storage/json_store.py`
- `app.py`
- Cac script ghi JSON neu lien quan

Thu tu thuc hien:

1. Tao `storage/json_store.py` gom:
   - `read_json(path, default)`.
   - `write_json(path, data)`.
   - `update_json(path, update_fn)`.
   - file lock.
   - atomic write temp + replace.
   - backup optional.
2. Migrate dan cac ghi quan trong truoc:
   - `VIDEOS_FILE`
   - `EVALUATIONS_FILE`
   - `USER_DATA_FILE`
   - `REMINDERS_FILE`
3. Thay pattern `load_data -> mutate -> save_data` bang `update_json`.
4. Them `updated_at` va optional `version` cho records khi ghi.
5. Dam bao background HF sync khong ghi de local update:
   - merge theo key.
   - so sanh updated_at.
6. Sau khi migrate, giu wrapper `load_data`/`save_data` goi vao json_store de giam diff.

Lenh verify:

```powershell
rg -n "save_data\\(|doc_lock_save_data\\(|load_data\\(" app.py
python -m py_compile app.py storage\json_store.py
```

Tieu chi xong:

- Moi ghi JSON quan trong di qua helper lock.
- Khong con lost update de thay o cac callback chinh.
- Write atomic va co log loi.

## 3.3 - Giam exception nuot loi

Lien quan loi: F26, N17

File can sua:

- `app.py`
- `scripts/*.py`
- `utils/*.py`

Thu tu thuc hien:

1. Uu tien cac vung IO:
   - read/write JSON.
   - delete file.
   - HF sync.
   - ffmpeg.
   - auth/session.
2. Thay `except:` bang exception cu the:
   - `FileNotFoundError`
   - `json.JSONDecodeError`
   - `PermissionError`
   - `requests.RequestException`
   - `subprocess.TimeoutExpired`
3. Khong dung `pass` trong loi ghi/xoa du lieu.
4. Them logger co context:
   - actor.
   - action.
   - target/file.
   - result.
5. Loi user-facing phai ngan gon, log moi chua stack trace.

Lenh verify:

```powershell
rg -n "except:\\s*$|except Exception:\\s*pass|except Exception:\\s*$" app.py scripts utils
python -m py_compile app.py scripts\sync_from_hf.py scripts\reset_data.py utils\pose_classifier_utils.py
```

Tieu chi xong:

- Bare `except:` trong `app.py` giam xuong muc rat thap, muc tieu <10.
- Cac IO error quan trong duoc log co context.

## 3.4 - Metadata video validator va frames_zip_path

Lien quan loi: R04

File can sua:

- `app.py`
- Tao `scripts/validate_video_metadata.py`
- `debug_files/video_list.json` neu con giu fixture sanitize

Thu tu thuc hien:

1. Trong `_frames_zip_path_from_video()`, neu `frames_zip_path` co timestamp khac `processed_path`, uu tien suy ra tu `processed_path` va log warning.
2. Tao script validator:
   - Doc `database/video_list.json`.
   - Check path fields timestamp consistency.
   - Check file path nam trong allowed roots.
   - Check required keys.
3. Script co `--fix` optional, mac dinh chi report.
4. Xoa/sua ban sao sai trong `debug_files/video_list.json` theo Phase 1B.

Lenh verify:

```powershell
python -m py_compile app.py scripts\validate_video_metadata.py
python scripts\validate_video_metadata.py --dry-run
```

Tieu chi xong:

- Khong con record tro frames ZIP sai timestamp.
- Validator chay duoc trong CI/smoke.

## 3.5 - Siết ML reprocess path va dry-run

Lien quan loi: R06

File can sua:

- `utils/pose_classifier_utils.py`
- `scripts/reprocess_all.py`
- `scripts/run_ml_pipeline.py`

Thu tu thuc hien:

1. Sua `resolve_local_path()` de chi resolve trong:
   - `data_dir`
   - `processed_dir`
   - `db_dir`
2. Reject path ngoai allowed roots sau `realpath`.
3. Them `dry_run` vao `reprocess_videos_with_classifier()`.
4. Khi dry-run:
   - In danh sach file se doc.
   - In danh sach file se ghi.
   - Khong ghi CSV/JSON/image/evaluation.
5. Script CLI truyen `--dry-run`.

Lenh verify:

```powershell
rg -n "resolve_local_path|reprocess_videos_with_classifier|dry_run|realpath|resolve" utils\pose_classifier_utils.py scripts
python -m py_compile utils\pose_classifier_utils.py scripts\reprocess_all.py scripts\run_ml_pipeline.py
```

Tieu chi xong:

- ML reprocess khong doc/ghi ngoai workspace.
- `--dry-run` hoat dong.

## 3.6 - Them test suite co ban

Lien quan loi: F31

File can tao:

- `tests/test_auth.py`
- `tests/test_storage.py`
- `tests/test_schema.py`
- `tests/test_video.py`
- `tests/test_password.py`
- `pytest.ini` hoac config tuong duong

Thu tu thuc hien:

1. Them pytest dependency vao dev requirements.
2. Viet fixture temp dir, data sanitize.
3. Test auth:
   - query params khong login.
   - role guard deny/allow.
4. Test storage:
   - write atomic.
   - update_json khong lost update co ban.
   - invalid JSON fallback/log.
5. Test schema:
   - users/videos/evaluations/schedules missing fields duoc normalize hoac reject ro.
6. Test video:
   - path containment reject `..`.
   - filename sanitize.
   - ZIP invalid entry reject.
7. Test password:
   - Argon2 verify.
   - SHA-256 migration.

Lenh verify:

```powershell
pytest tests/
```

Tieu chi xong:

- Test suite pass local.
- Test khong dung PII thật.

## 3.7 - Pin dependencies va tach requirements

Lien quan loi: F32, F29

File can sua:

- `requirements.txt`
- Tao `requirements-prod.txt`
- Tao `requirements-dev.txt`
- Scripts co runtime pip install

Thu tu thuc hien:

1. Pin version cac package runtime quan trong.
2. Tach dev tools vao `requirements-dev.txt`.
3. `requirements.txt` co the include prod hoac giu single entry tuy deploy.
4. Xoa runtime `pip install` trong scripts; neu thieu dependency, in huong dan cai dat.
5. Kiem tra compatibility Streamlit/HF/Mediapipe.

Lenh verify:

```powershell
rg -n "pip install|subprocess.*pip|python -m pip" scripts utils app.py
python -m pip install -r requirements.txt --dry-run
```

Tieu chi xong:

- Install reproducible hon.
- Scripts khong tu cai package runtime.

## 3.8 - Sanitize filename upload

Lien quan loi: F25

File can sua:

- `app.py`

Thu tu thuc hien:

1. Tao `sanitize_filename(original_name)`:
   - normalize unicode.
   - lay basename.
   - whitelist `[a-zA-Z0-9._-]`.
   - gioi han 200 chars.
   - them fallback name neu rong.
2. Luu original filename trong metadata JSON.
3. Moi path upload dung sanitized name hoac generated ID.
4. Ket hop path containment Phase 2.

Lenh verify:

```powershell
rg -n "sanitize_filename|file_upload\\.name|os\\.path\\.join\\(.*file_upload" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Filename nguy hiem khong tao path nguy hiem.
- Original name chi la metadata escaped khi render.

## 3.9 - Cleanup scripts destructive va runtime pip

Lien quan loi: F29, F33, F34

File can sua:

- `scripts/reset_data.py`
- `scripts/sync_data_and_report.py`
- `scripts/extract_youtube_reference.py`
- `utils/pose_classfier_untils.py`

Thu tu thuc hien:

1. Confirm `reset_data.py` da co `--yes`, `--dry-run`, backup.
2. Xoa runtime pip install o scripts.
3. Sua SyntaxWarning escape trong `sync_data_and_report.py`.
4. Deprecate typo file `pose_classfier_untils.py`:
   - Giu shim import neu co code cu dung.
   - Huong sang `pose_classifier_utils.py`.

Lenh verify:

```powershell
python -m py_compile scripts\reset_data.py scripts\sync_data_and_report.py scripts\extract_youtube_reference.py utils\pose_classfier_untils.py
rg -n "pip install|SyntaxWarning|pose_classfier" scripts utils
```

Tieu chi xong:

- Scripts an toan hon va khong cai dependency luc runtime.

## Definition of Done Phase 3

- Password hash moi dung Argon2/Bcrypt va co migration.
- JSON write quan trong co lock/atomic/update helper.
- Exception IO quan trong khong bi nuot.
- Metadata/video/ML path co validators/dry-run.
- Test suite co ban pass.
- Dependencies/script hygiene san sang cho refactor Phase 5.

