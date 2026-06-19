# Main Plan Sua Code - Rehab-AI-Monitor

Nguon dau vao:

- `docs/ROADMAP_SUA_CHUA.md`
- `docs/CODE_REVIEW_BUG_REPORT.md`
- Doi chieu nhanh working tree ngay 17/06/2026

Muc tieu cua bo plan nay la bien roadmap thanh cac ke hoach sua code co the giao cho AI/coder theo tung phase. Moi phase co file rieng, task ro dau vao, file can sua, thu tu thuc hien, tieu chi xong va lenh verify.

## Cau truc file

| Thu tu | Phase | File plan | Muc tieu |
| --- | --- | --- | --- |
| 0 | Containment va baseline | `PHASE_0_CONTAINMENT.md` | Tra loi gate public exposure, backup, lay baseline truoc khi sua |
| 1A | Chan ro ri ngay | `PHASE_1A_CRITICAL_CONTAINMENT.md` | Xoa auth bypass, tat lo token/debug, loai PII khoi README/script |
| 1B | Security hardening | `PHASE_1B_SECURITY_HARDENING.md` | Seed user an toan, proxy token, harden video server, XSS, PII purge |
| 2 | Stability va access control | `PHASE_2_ACCESS_STABILITY.md` | Upload, CORS/XSRF, path guard, ZIP safe extract, permission, session revoke |
| 3 | Data integrity va quality | `PHASE_3_DATA_QUALITY.md` | Password hashing, JSON lock, exception logging, tests, deps, script cleanup |
| 4 | Frontend va UX | `PHASE_4_FRONTEND_UX.md` | Schema validation, CSS/JS cleanup, doctor-patient scoping, UX cleanup |
| 5 | Architecture refactor | `PHASE_5_ARCHITECTURE.md` | Tach `app.py` thanh modules theo slice co test |
| 6 | Production-ready | `PHASE_6_PRODUCTION_READY.md` | DB migration, CI/CD, job system, privacy/compliance |
| Test | Test plan xuyen phase | `TEST_PLAN.md` | Smoke, security grep, pytest, UI smoke va CI gates |

## Nguyen tac thuc hien chung

1. Khong sua lan man ngoai phase hien tai.
2. Moi task nen thanh mot change set nho, co verify rieng.
3. Phase 0 den Phase 2 uu tien giam rui ro bao mat va mat du lieu, khong refactor kien truc lon.
4. Neu mot task can chon giua "fix nhanh an toan" va "thiet ke dep", chon fix nhanh an toan truoc, de refactor sang Phase 5.
5. Moi noi ghi/xoa du lieu phai co backup, guard quyen, log va test hoac smoke check.
6. Khong dua `HF_TOKEN`, mat khau, email benh nhan, ten that benh nhan vao README, log UI, DOM, link, screenshot hoac file generated.
7. Khong load du lieu tu path do JSON/HF/user dieu khien neu chua qua containment guard.
8. Khi sua `app.py`, doc vung code lien quan truoc va sau toi thieu 80 dong de tranh pha side effect Streamlit.
9. Sau moi task sua code, chay test toi thieu theo `TEST_PLAN.md`; sau moi phase, chay phase test gate tuong ung.

## Quy tac cho AI codegen

Moi lan giao viec cho AI nen kem contract ngan:

```text
Chi sua cac file duoc liet ke trong task.
Khong doi UI/flow khong lien quan.
Khong viet lai app.py toan bo.
Khong them dependency neu task khong yeu cau.
Sau khi sua, chay cac lenh verify trong task va bao cao ket qua.
Neu phat hien du lieu nhay cam moi, dung lai va ghi vao findings, khong dua vao output.
```

## Thu tu gate bat buoc

| Gate | Dieu kien can dat | Phase tiep theo |
| --- | --- | --- |
| G0 | Da tra loi repo/app tung public chua, da backup `database/*.json`, co baseline grep, L0 smoke status | Phase 1A |
| G1A | Khong con login bang query params, README/script khong sinh link bypass, debug token tat, L1 auth/token grep pass | Phase 1B |
| G1B | Token khong ra frontend, user seed khong ghi de, PII/debug files duoc xu ly, unsafe HTML user-generated duoc escape, manual XSS/token smoke pass | Phase 2 |
| G2 | Upload duoc validate, path/ZIP safe, destructive action co confirm, mutation co role guard, focused smoke pass | Phase 3 |
| G3 | Password hashing moi, JSON write co lock, test suite co nhom auth/storage/video, exception IO duoc log, pytest core pass | Phase 4 va Phase 5 |
| G4 | UI/CSS/JS bot fragile, schema validation co migration, UI smoke pass | Phase 5 |
| G5 | `app.py` chi con orchestrator lon vua phai, module co test, full regression pass | Phase 6 |

## Ke hoach tong theo phase

### Phase 0 - Containment va Verification

File: `PHASE_0_CONTAINMENT.md`

Lam truoc khi sua code:

- Xac nhan public exposure.
- Backup JSON runtime.
- Chay baseline `rg` va compile smoke.
- Ghi lai ket qua vao `docs/repair_plans/PHASE_0_BASELINE.md` neu can audit.

Khong sua logic app trong phase nay, tru truong hop can tao file baseline.

### Phase 1A - Chan ro ri ngay

File: `PHASE_1A_CRITICAL_CONTAINMENT.md`

Thu tu:

1. Xoa auth bypass `logged_in_user` / `logged_in_role`.
2. Sua `README.md` va `scripts/sync_data_and_report.py` de khong con link/template bypass.
3. Tat debug popover/HTML/URL co token.
4. Bo fallback dataset hard-code nguy hiem.
5. Dua bao cao generated sang `docs/generated/` va loai PII khoi README/script.

Day la phase "stop bleeding", khong xay he thong auth moi.

### Phase 1B - Security hardening

File: `PHASE_1B_SECURITY_HARDENING.md`

Thu tu:

1. Thay hard-coded credentials bang seed-once va `must_change_password`.
2. Build server-side HF access path/proxy va thu hep HTTP video server.
3. Them `safe_html()` va audit tat ca HTML co du lieu dong.
4. Pseudonymize/xoa PII trong `database/`, `debug_files/`, README generated.
5. Sua `.gitignore` va push guard.

Phase nay can lam theo batch nho vi co 139 vi tri `unsafe_allow_html=True`.

### Phase 2 - Stability va Access Control

File: `PHASE_2_ACCESS_STABILITY.md`

Thu tu:

1. Giam upload size va validate file truoc khi doc buffer.
2. Bat lai CORS/XSRF trong production.
3. Them path containment helper va ap dung vao download/sync/resolve.
4. Thay ZIP `extractall()` bang extract tung entry co quota.
5. Sua `sync_from_hf.py` khong sync `users.json` mac dinh.
6. Them confirm flow, backup va audit log cho destructive actions.
7. Them permission guard tap trung.
8. Them session revocation.
9. Sua reset password, Google login, pickle/joblib load va login UX.

Day la phase chong mat du lieu va truy cap sai quyen.

### Phase 3 - Data Integrity va Quality

File: `PHASE_3_DATA_QUALITY.md`

Thu tu:

1. Doi password hashing sang Argon2/Bcrypt co migration.
2. Tao `storage/json_store.py` co lock va atomic write.
3. Giam bare `except:` o cac IO path quan trong.
4. Them validator metadata video.
5. Siết ML reprocess path va dry-run.
6. Them test suite co ban.
7. Pin dependencies va tach requirements dev/prod.
8. Sanitize filename upload.
9. Cleanup scripts destructive/runtime pip.

Phase nay tao safety net cho refactor lon.

### Phase 4 - Frontend va UX Improvement

File: `PHASE_4_FRONTEND_UX.md`

Thu tu:

1. Tao schema models va migration data.
2. Tach CSS ra assets rieng, giam selector global.
3. Loai JS DOM hacks.
4. Phan quyen bac si-benh nhan va NCV pseudonymized.
5. Cleanup sync report script.
6. Rename ham/thread ro nghia.

Phase nay lam UI ben hon sau khi bao mat/storage da on hon.

### Phase 5 - Architecture Refactor

File: `PHASE_5_ARCHITECTURE.md`

Thu tu slice:

1. `auth/` hoac `auth.py`
2. `storage/json_store.py`
3. `cloud/hf_sync.py`
4. `video/io.py`
5. `ui/patient.py`, `ui/doctor.py`, `ui/researcher.py`, `ui/admin.py`
6. `app_startup()` cho side effects

Khong refactor toan bo `app.py` mot lan. Moi slice phai pass test lien quan.

### Phase 6 - Production-ready

File: `PHASE_6_PRODUCTION_READY.md`

Thu tu:

1. Migration JSON sang SQLite/Postgres.
2. CI/CD lint/test/build/deploy.
3. Background job system.
4. Privacy/compliance workflow.

Phase nay can quyet dinh san pham va moi truong deploy truoc khi code.

### Test plan xuyen phase

File: `TEST_PLAN.md`

Dung cho moi phase:

1. L0 static smoke: `py_compile` va `rg` pattern nong.
2. L1 security grep: auth bypass, token URL, plaintext password, PII, CORS/ZIP/server patterns.
3. L2 unit tests: auth, password, permissions, storage, path, video, ZIP, schema.
4. L3 integration tests: login, delete video, upload, HF sync, reset script.
5. L4 Streamlit/UI smoke: boot app, role flow, debug token, upload invalid, destructive confirm, XSS payload.
6. L5 CI/release gates: full test, e2e smoke, secret/security scan.

## Lenh verify chung

Chay sau moi phase neu co the:

```powershell
python -m py_compile app.py utils\reference_utils.py utils\pose_classifier_utils.py utils\checkpoint_utils.py scripts\sync_from_hf.py scripts\sync_data_and_report.py
rg "logged_in_user|logged_in_role" app.py scripts README.md
rg "HF_TOKEN.*token=|token=\\{HF_TOKEN\\}|\\?token=\\{HF_TOKEN\\}" app.py scripts README.md
rg -c "unsafe_allow_html=True" app.py
rg -n "except:\\s*$|except Exception:\\s*pass" app.py
```

Khi test suite da co:

```powershell
pytest tests/
```

Chi tiet lenh theo tung phase nam trong `TEST_PLAN.md`.

## Bao cao tien do

Moi task khi xong nen cap nhat:

- File da sua.
- Hanh vi thay doi.
- Lenh da chay va ket qua.
- Rui ro con lai.
- Task tiep theo nen lam.
