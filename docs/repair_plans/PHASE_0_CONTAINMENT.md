# Phase 0 Plan - Containment va Verification

Muc tieu: khong chua bug ngay lap tuc, ma tao an toan truoc khi sua. Phase nay tra loi gate public exposure, backup du lieu runtime va lay baseline de cac phase sau co the so sanh.

## Nguyen tac rieng cua Phase 0

- Khong sua logic `app.py`.
- Khong xoa/chinh sua `database/*.json` hoac `debug_files/*.json`.
- Neu phat hien app/repo da public, uu tien rotate secrets va quy trinh incident truoc khi tiep tuc.
- Moi output baseline khong duoc chua token, mat khau plaintext, email/ten that benh nhan.

## P0.1 - Gate public exposure

File lien quan:

- `.git/`
- `.gitignore`
- `README.md`
- `docs/ROADMAP_SUA_CHUA.md`
- Hugging Face Space/Dataset settings ben ngoai repo

Viec can lam:

1. Hoi/ghi nhan cau tra loi: repo/app da tung push GitHub public hoac deploy HF Spaces public chua?
2. Neu da public hoac khong chac:
   - Rotate HF token.
   - Invalidate/reset password/hash da lo.
   - Kiem tra git history bang `git log`, `git ls-files`, BFG/git-filter-repo sau khi co quyet dinh team.
   - Kiem tra HF commit/cache/access logs neu co.
   - Dong bang public share/demo link den het Phase 1B.
3. Neu chua public:
   - Tiep tuc backup va baseline.

Lenh goi y:

```powershell
git status --short
git remote -v
git log --oneline -n 20
git ls-files database debug_files README.md scripts
```

Tieu chi xong:

- Co cau tra loi ro: `Da public`, `Chua public`, hoac `Khong chac -> xu ly nhu da public`.
- Neu public/khong chac, da co ticket/ghi chu rotate secrets va xoa history truoc khi code tiep.

## P0.2 - Backup database runtime

File lien quan:

- `database/*.json`
- `debug_files/*.json`

Viec can lam:

1. Tao thu muc backup timestamp local, vi du `backups/pre_repair_YYYYMMDD_HHMMSS/`.
2. Copy nguyen trang `database/` va `debug_files/`.
3. Khong commit backup nay.
4. Kiem tra `.gitignore` co chan `backups/`; neu chua, ghi note de Phase 1B sua.

Lenh goi y:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force "backups/pre_repair_$stamp" | Out-Null
Copy-Item -Recurse -Force database "backups/pre_repair_$stamp/database"
Copy-Item -Recurse -Force debug_files "backups/pre_repair_$stamp/debug_files"
```

Tieu chi xong:

- Backup ton tai va doc duoc.
- Backup khong nam trong danh sach staged commit.

## P0.3 - Baseline security grep

File lien quan:

- `app.py`
- `README.md`
- `scripts/*.py`
- `.gitignore`
- `.streamlit/config.toml`
- `Dockerfile`

Lenh baseline:

```powershell
rg -n "logged_in_user|logged_in_role" app.py scripts README.md docs
rg -n "HF_TOKEN|HF_DATASET_ID|token=\\{HF_TOKEN\\}|\\?token=" app.py scripts README.md
rg -c "unsafe_allow_html=True" app.py
rg -n "except:\\s*$" app.py
rg -n "except Exception:\\s*pass" app.py
rg -n "file_upload\\.getbuffer|extractall|SimpleHTTPRequestHandler|--server.enableCORS=false|--server.enableXsrfProtection=false" app.py Dockerfile .streamlit scripts
```

Viec can lam:

1. Chay lenh tren.
2. Luu so lieu tong hop vao ghi chu local hoac `docs/repair_plans/PHASE_0_BASELINE.md` neu team muon track bang repo.
3. Khong paste token hoac PII vao file baseline.

Tieu chi xong:

- Co baseline cho auth bypass, token leak, unsafe HTML, broad exception, upload, ZIP, CORS/XSRF.

## P0.4 - Smoke compile hien tai

Lenh:

```powershell
python -m py_compile app.py utils\reference_utils.py utils\pose_classifier_utils.py utils\checkpoint_utils.py scripts\sync_from_hf.py scripts\sync_data_and_report.py scripts\reset_data.py
```

Viec can lam:

1. Chay compile.
2. Neu loi syntax, ghi lai file/line, nhung khong sua trong Phase 0 tru khi loi chan moi phase sau.
3. Neu chi co warning, ghi note de Phase 3/4 cleanup.

Tieu chi xong:

- Biet trang thai compile truoc khi sua.

## P0.5 - Kiem tra du lieu va gitignore

File lien quan:

- `.gitignore`
- `database/schedules.json`
- `database/video_list.json`
- `debug_files/video_list.json`

Viec can lam:

1. Xac nhan `database/schedules.json` la list `[]` hoac list records.
2. Xac nhan `.gitignore` hien dang whitelist nhieu runtime JSON de dua vao Phase 1B.
3. Xac nhan `debug_files/*.json` co phai ban sao PII can xu ly trong Phase 1B.

Lenh goi y:

```powershell
Get-Content -Raw database/schedules.json
Get-Content -Raw .gitignore
rg -n "email|full_name|comments|doctor_result|patient|password" database debug_files
```

Tieu chi xong:

- Co danh sach data file nhay cam can xu ly.
- Khong co thay doi code.

## Definition of Done Phase 0

- Gate public exposure da duoc tra loi.
- Backup local da tao.
- Baseline grep va compile da ghi nhan.
- Biet `.gitignore` va data file nao can sua o Phase 1B.
- San sang vao Phase 1A.

