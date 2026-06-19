# Test Plan - Rehab-AI-Monitor

Muc tieu: dinh nghia cach chay test theo tung phase sua code, tu smoke checks toi pytest/CI. Test plan nay di kem `MAIN_PLAN.md` va phai duoc cap nhat khi them module/test moi.

## Nguyen tac test

1. Test khong duoc dung PII that, token that, password that, email that cua benh nhan.
2. Moi bug security duoc fix phai co it nhat mot regression check: pytest neu co the, neu chua thi smoke grep/command.
3. Moi task sua code phai chay test nho nhat lien quan truoc, sau do chay phase test gate.
4. Neu test can fixture JSON, tao fixture sanitize trong `tests/fixtures/`, khong doc truc tiep `database/*.json` production.
5. Test destructive action phai dung temp directory va dry-run, khong ghi/xoa data that.
6. Neu test fail, dung phase hien tai de sua fail lien quan; khong tiep tuc phase tiep theo khi gate fail.

## Test layers

| Layer | Khi nao chay | Muc dich | Lenh chinh |
| --- | --- | --- | --- |
| L0 - Static smoke | Moi task, truoc/sau sua code | Bat syntax/import/search pattern nguy hiem | `py_compile`, `rg` |
| L1 - Security grep | Phase 0-2 va truoc release | Dam bao khong tai tao bypass/token/PII | `rg` pattern |
| L2 - Unit tests | Tu Phase 3 tro di | Test helper auth/storage/video/path/schema | `pytest tests/unit` |
| L3 - Integration tests | Tu Phase 3/4 tro di | Test flow lien module voi temp data | `pytest tests/integration` |
| L4 - Streamlit smoke | Sau phase co UI/auth/upload | App boot va role flow co ban | `streamlit run` + manual/Playwright |
| L5 - CI/release gates | Phase 6 | Chan regression tren PR/deploy | GitHub Actions |

## Environment test

Khuyen nghi tao moi truong rieng:

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Neu `requirements-dev.txt` chua ton tai, Phase 3 se tao. Tam thoi co the cai rieng:

```powershell
python -m pip install pytest pytest-cov
```

Bien moi truong test:

```powershell
$env:APP_ENV = "test"
$env:HF_TOKEN = ""
$env:HF_DATASET_ID = ""
```

Quy tac:

- Test khong phu thuoc network mac dinh.
- Test HF/cloud phai mock HTTP/HfApi.
- Test video/ffmpeg dung file mau nho hoac mock subprocess.

## L0 - Static smoke commands

Chay truoc/sau moi task sua code:

```powershell
python -m py_compile app.py utils\reference_utils.py utils\pose_classifier_utils.py utils\checkpoint_utils.py scripts\sync_from_hf.py scripts\sync_data_and_report.py scripts\reset_data.py
```

Kiem tra pattern nong:

```powershell
rg -n "logged_in_user|logged_in_role" app.py scripts README.md docs
rg -n "\\?token=|token=\\{HF_TOKEN\\}|HF_TOKEN.*st\\.|st\\..*HF_TOKEN" app.py scripts README.md
rg -n "admin123|bs123|ncv123" app.py database scripts docs
rg -n "file_upload\\.getbuffer\\(|extractall\\(|--server.enableCORS=false|--server.enableXsrfProtection=false" app.py Dockerfile .streamlit scripts
rg -n "except:\\s*$|except Exception:\\s*pass" app.py scripts utils
```

Ghi chu:

- Mot so `rg` co the van co ket qua trong docs audit lich su. Gate security chi fail khi ket qua nam trong app/script/runtime docs co kha nang tai tao exploit.
- Pattern `HF_TOKEN` duoc phep xuat hien trong code server-side, khong duoc xuat hien trong URL/frontend/debug output.

## L1 - Security regression grep

Chay sau Phase 1A, 1B, 2 va truoc moi release:

```powershell
rg -n "logged_in_user|logged_in_role|\\?logged_in_user=" app.py scripts README.md
rg -n "\\?token=|token=\\{HF_TOKEN\\}|cloud_url.*HF_TOKEN" app.py
rg -n "admin123|bs123|ncv123|password.*admin|password.*doctor" app.py scripts database docs
rg -n "@gmail\\.com|@yahoo\\.com|full_name|patient_name|doctor_result|comments_ncv" README.md docs/generated scripts database debug_files
rg -n "Access-Control-Allow-Origin.*\\*|SimpleHTTPRequestHandler|extractall\\(" app.py
```

Tieu chi pass:

- Khong con auth bypass runtime.
- Khong con token trong URL client-side.
- Khong con default plaintext password.
- Khong con PII trong README/generated/debug fixtures.
- Khong con CORS wildcard/path server rong/ZIP extractall nguy hiem.

## L2 - Unit test suite muc tieu

Tao cac nhom test trong Phase 3:

```text
tests/
  unit/
    test_auth.py
    test_password.py
    test_permissions.py
    test_storage_json_store.py
    test_path_security.py
    test_video_validation.py
    test_zip_extract.py
    test_schema.py
  fixtures/
    users_sanitized.json
    video_list_sanitized.json
    doctor_evaluations_sanitized.json
```

Lenh:

```powershell
pytest tests/unit -q
pytest tests/unit --maxfail=1
pytest tests/unit --cov=auth --cov=storage --cov=video --cov=models
```

Test bat buoc:

| File test | Case can co |
| --- | --- |
| `test_auth.py` | query params khong login; lookup khong collision full_name/casefold |
| `test_password.py` | hash Argon2 verify; migrate SHA-256 sau login thanh cong; sai password fail |
| `test_permissions.py` | patient/doctor/NCV/admin allow-deny matrix |
| `test_storage_json_store.py` | atomic write; update_json co lock; invalid JSON duoc handle/log |
| `test_path_security.py` | reject `..`; reject absolute path; reject symlink out of root |
| `test_video_validation.py` | reject over-size; reject non-video; ffprobe timeout handled |
| `test_zip_extract.py` | reject zip-slip; reject zip bomb; allow image basename hop le |
| `test_schema.py` | normalize missing optional fields; reject root type sai |

## L3 - Integration test suite muc tieu

Tao trong Phase 3/4:

```text
tests/
  integration/
    test_login_flow.py
    test_video_delete_flow.py
    test_upload_pipeline.py
    test_hf_sync.py
    test_reset_data_script.py
```

Lenh:

```powershell
pytest tests/integration -q
```

Case bat buoc:

1. Login username/password hop le set session role tu DB, khong can chon role truoc.
2. Login bang query params bi bo qua.
3. Delete video:
   - user khong co quyen bi deny.
   - admin xoa co backup va audit log.
   - evaluation filter khong KeyError khi thieu field.
4. Upload:
   - file qua size bi reject truoc `getbuffer()`.
   - filename nguy hiem duoc sanitize.
   - path output nam trong allowed roots.
5. HF sync:
   - mac dinh khong sync `users.json`.
   - `--include-users` moi xu ly users.
   - backup timestamped.
6. Reset script:
   - khong `--yes` thi khong xoa.
   - `--dry-run` khong ghi file.

## L4 - Streamlit/UI smoke

Dung sau cac phase co thay doi UI/auth/upload:

```powershell
streamlit run app.py --server.port 8501 --server.headless true
```

Manual smoke checklist:

1. App boot khong crash.
2. Trang login hien thi.
3. Login sai hien loi chung, khong lo role/account enumeration.
4. Login tung role bang fixture/sandbox account:
   - Benh nhan.
   - Bac si/KTV.
   - NCV.
   - Admin.
5. Moi role chi thay tab/action dung quyen.
6. Debug UI khong hien token/path nhay cam.
7. Upload file invalid bi reject ro rang.
8. Destructive action can confirm 2 buoc.
9. XSS payload trong field text hien nhu text, khong pha layout.
10. Video render duoc qua local/proxy ma network/DOM khong co `HF_TOKEN`.

Playwright/automation muc tieu Phase 6:

```powershell
pytest tests/e2e -q
```

Test e2e nen dung seed data sanitize va mock cloud.

## Test gates theo phase

| Phase | Test phai chay | Gate pass |
| --- | --- | --- |
| Phase 0 | L0 baseline + py_compile | Co baseline va compile status |
| Phase 1A | L0 + L1 auth/token/README grep | Khong con bypass runtime, README/script khong sinh bypass |
| Phase 1B | L0 + L1 + manual XSS/token/video server smoke | Token khong ra frontend, PII/debug files da xu ly, XSS user data escaped |
| Phase 2 | L0 + L1 + focused unit neu da co + manual upload/delete/path smoke | Upload/path/ZIP/permission/destructive guards hoat dong |
| Phase 3 | L0 + L1 + L2 + L3 core | `pytest` pass cho auth/storage/schema/video/password |
| Phase 4 | L0 + L2/L3 lien quan + L4 manual UI smoke | UI role/schema/CSS/JS khong regression |
| Phase 5 | Full L0-L4 sau moi slice | Refactor module khong doi behavior |
| Phase 6 | Full test + CI + e2e smoke | PR/deploy bi chan neu test/security fail |

## Coverage muc tieu

Coverage khong phai muc tieu duy nhat, nhung dung de do do an toan refactor:

| Module | Muc tieu toi thieu |
| --- | ---: |
| `auth/` | 80% |
| `storage/` | 85% |
| `video/` path/validation/ZIP | 80% |
| `models/` schema | 80% |
| `cloud/` | 70%, chu yeu mock network |
| UI modules | Smoke/e2e thay vi unit coverage cao |

## Bao cao ket qua test

Moi PR/task nen ghi:

```text
Tests run:
- python -m py_compile ...
- rg ...
- pytest tests/unit -q

Results:
- Pass/fail
- Failures neu co
- Tests skipped va ly do

Residual risk:
- Phan chua test duoc
- Manual check can lam
```

## Khi test fail

Quy trinh:

1. Xac dinh fail do code moi, fixture, hay test cu da sai.
2. Neu fail do code moi, sua trong cung task.
3. Neu fail do test cu sai voi spec moi, cap nhat test va ghi ly do.
4. Neu fail do moi truong thieu dependency/network, mock hoac tach test thanh optional.
5. Khong bo qua test security gate de sang phase tiep theo.

