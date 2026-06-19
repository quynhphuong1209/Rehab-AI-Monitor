# Phase 1B Plan - Security Hardening

Muc tieu: sau khi da chan ro ri nhanh, xay lai nhung diem bao mat can nen tang dung: user seed, token proxy/server-side access, HTTP video server, HTML escaping va PII purge.

## Nguyen tac rieng cua Phase 1B

- Lam theo batch nho, moi batch co verify.
- Khong refactor `app.py` thanh nhieu module trong phase nay, tru helper nho that su can.
- Khong dua PII mau vao test fixture.
- Bat dau bang auth/user, sau do den token/media, sau cung den XSS/PII cleanup.

## 1B.1 - Xoa hard-coded credentials va chuyen sang seed-once

Lien quan loi: F02, F21, N03

File can sua:

- `app.py`
- `database/users.json` hoac seed file moi da sanitize
- Co the tao `database/seed_users.example.json`

Vung code hien tai:

- `_get_cached_users_dict(mtime)` quanh `app.py:5100-5162`

Thu tu thuc hien:

1. Tach predefined users khoi code runtime.
2. Neu can seed, chi seed khi `database/users.json` rong hoac khong ton tai.
3. Khong ghi de user da ton tai khi `load_users()`/cached users chay.
4. Khong co plaintext password trong source.
5. User seed moi phai co:
   - password hash tam thoi duoc tao tu secret/runtime, khong hard-code.
   - `must_change_password: true`.
   - `created_at`, `updated_at`, `hash_version`.
6. Tach email NCV/patient khong dung chung de tranh takeover reset password.
7. Them validation phat hien username/email duplicate.

Lenh verify:

```powershell
rg -n "admin123|bs123|ncv123|password.*@|plain" app.py database scripts
python -m py_compile app.py
```

Tieu chi xong:

- Khong con plaintext default password trong source.
- `load_users()` khong ghi de password/role/email user hien co.
- Tai khoan seed moi bi bat doi mat khau.

Can tranh:

- Khong sinh password random roi in ra log/README.
- Khong tao seed moi moi lan app rerun.

## 1B.2 - Server-side HF access va proxy video

Lien quan loi: F05, F17, N15

File can sua:

- `app.py`
- Co the them helper nho trong `cloud/` neu khong anh huong lon

Vung code hien tai:

- `_hf_download_via_http()`
- `_hf_download_dataset_file()`
- Cac noi tao URL `https://huggingface.co/datasets/...?...token=...`
- Cac noi render video source trong `app.py`

Thu tu thuc hien:

1. Dinh nghia mot duong server-side duy nhat de lay file HF:
   - Token chi nam trong request server-to-HF.
   - Frontend chi nhan file local da cache hoac response stream khong chua token.
2. Uu tien cach it thay doi:
   - Download file ve cache local trong `DATA_DIR` qua HF header `Authorization`.
   - Render video tu local safe media server da harden o task 1B.3.
3. Neu lam proxy endpoint rieng, endpoint phai:
   - Kiem tra session/role.
   - Kiem tra path containment.
   - Khong log token.
   - Co range support neu video can seek.
4. Xoa tat ca code tao cloud URL co `?token=`.
5. Them fallback UI ro neu file chua tai duoc, khong hien token.

Lenh verify:

```powershell
rg -n "\\?token=|token=\\{HF_TOKEN\\}|HF_TOKEN.*quote|cloud_url" app.py
python -m py_compile app.py
```

Tieu chi xong:

- DOM/Network khong thay `HF_TOKEN`.
- Video van phat duoc tu local/proxy neu user co quyen.
- Loi download hien thong bao sach, khong ro token/path nhay cam.

Can tranh:

- Khong dua token vao signed URL client neu HF khong ho tro URL ngan han that su.
- Khong dung query param token noi bo de bao ve video.

## 1B.3 - Harden HTTP video server

Lien quan loi: N01, N10, F40

File can sua:

- `app.py`

Vung code hien tai:

- `_RangeHandler(http.server.SimpleHTTPRequestHandler)` quanh `app.py:1603-1689`

Thu tu thuc hien:

1. Gioi han document root chi vao thu muc media can phuc vu:
   - `patient_uploads/`
   - `processed_results/`
   - Hoac cache media da duoc phep
2. Dung `Path.resolve()`/`os.path.realpath()` de guard containment.
3. Reject:
   - absolute path tu request.
   - path co `..`.
   - symlink tro ra ngoai allowed roots.
   - extension khong phai media duoc phep.
4. Bo `Access-Control-Allow-Origin: *`; neu can, chi allow origin app hien tai.
5. Them session/token check phia server neu co cach lay session hop le. Neu chua, it nhat randomize per-session media URL va expire ngan, sau Phase 2 se noi voi permission guard.
6. Log request bi reject o muc warning, khong log PII/day du path neu khong can.

Lenh verify:

```powershell
rg -n "SimpleHTTPRequestHandler|Access-Control-Allow-Origin|translate_path" app.py
python -m py_compile app.py
```

Test thu cong:

- Request file media hop le phai thanh cong.
- Request `../database/users.json` phai bi reject.
- Request file ngoai media root phai bi reject.

Tieu chi xong:

- Server khong phuc vu project root.
- CORS wildcard bi loai bo.
- Path traversal/symlink ra ngoai root bi chan.

## 1B.4 - XSS/HTML injection audit

Lien quan loi: F06

File can sua:

- `app.py`

Pham vi:

- 139 vi tri `unsafe_allow_html=True`
- Uu tien field user-generated: comments, title, notes, exercise_name, medication_name, doctor_result, plan, full_name, video_name

Thu tu thuc hien:

1. Them helper nho:
   - `safe_html(value)`: `html.escape(str(value or ""), quote=True)`.
   - `safe_attr(value)` neu gia tri vao attribute.
2. Audit theo batch:
   - Batch A: doctor evaluation/comments/plan/result.
   - Batch B: schedules/reminders.
   - Batch C: user name/patient name/video name.
   - Batch D: cards/metrics chi dung data tinh toan noi bo.
3. Voi noi khong can custom HTML, doi sang `st.write`, `st.text`, `st.info`, `st.caption`.
4. Voi HTML bat buoc giu, escape moi field dong.
5. Them test/smoke bang payload:
   - `<script>alert(1)</script>`
   - `<img src=x onerror=alert(1)>`
   - `<style>body{display:none}</style>`

Lenh verify:

```powershell
rg -n "unsafe_allow_html=True" app.py
rg -n "comments|comments_ncv|doctor_result|plan|title|notes|full_name|video_name" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Moi du lieu user-generated dua vao HTML deu qua `safe_html()`/`safe_attr()` hoac khong render HTML.
- Payload HTML khong pha UI va khong duoc thuc thi/render nhu markup.

Can tranh:

- Khong escape toan bo template HTML lam vo layout; chi escape bien dong.
- Khong dung regex thay the mù tat ca f-string.

## 1B.5 - PII purge trong database/debug_files va gitignore

Lien quan loi: F40, N07, N08, R01

File can sua:

- `database/*.json`
- `debug_files/*.json`
- `.gitignore`
- `.push-guard`
- Co the them `database/*.example.json`

Thu tu thuc hien:

1. Phan loai file JSON:
   - Runtime sensitive: users, symptoms, evaluations, schedules, research data, video list.
   - Reference non-sensitive: `reference_*.json`, `pose_classifier_features.json` neu khong chua PII.
2. Pseudonymize hoac xoa du lieu benh nhan that trong `database/`.
3. Xoa hoac sanitize `debug_files/*.json`; giu `debug_files_manifest.md` neu da sanitize.
4. Sua `.gitignore`:
   - Ignore `database/*.json` runtime.
   - Ignore `debug_files/*.json`.
   - Chi whitelist seed/example/reference da sanitize.
5. Them/cap nhat push guard/pre-commit:
   - Chan token pattern `hf_`.
   - Chan email that.
   - Chan `logged_in_user`.
   - Chan default password pattern.
6. Neu repo da public, dung BFG/git-filter-repo theo quy trinh rieng, khong lam tuy tien trong task code.

Lenh verify:

```powershell
git status --short
git check-ignore -v database/users.json debug_files/users.json
rg -n "hf_[A-Za-z0-9]|logged_in_user|admin123|bs123|ncv123|@gmail\\.com|@|full_name" database debug_files README.md scripts
```

Tieu chi xong:

- Runtime JSON nhay cam khong con duoc track moi theo `.gitignore`.
- Debug JSON nhay cam da xoa/sanitize.
- Repo working tree khong con PII ro rang trong docs/script.

Can tranh:

- Khong xoa du lieu production neu chua co backup Phase 0.
- Khong commit backup PII.

## Definition of Done Phase 1B

- User seed khong hard-code password va khong ghi de user hien co.
- `HF_TOKEN` khong ra frontend/debug.
- HTTP video server chi phuc vu media allowed roots.
- User-generated HTML da escape o cac surface nguy hiem.
- PII/debug_files/gitignore da xu ly.

