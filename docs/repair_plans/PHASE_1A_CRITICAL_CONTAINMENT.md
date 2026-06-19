# Phase 1A Plan - Chan Ro Ri Ngay

Muc tieu: dong cac cua ro ri dang mo ma khong can refactor lon. Phase nay tap trung vao auth bypass, token/debug leak, PII trong README/script va fallback cau hinh nguy hiem.

## Nguyen tac rieng cua Phase 1A

- Khong xay auth framework moi.
- Khong doi schema user neu chua can, de Phase 1B xu ly seed/password.
- Khong sua hang loat `unsafe_allow_html=True` trong phase nay, tru noi lam lo token/PII truc tiep.
- Moi thay doi phai verify bang `rg`.

## 1A.1 - Xoa auth bypass qua query params

Lien quan loi: F01, F22, N20

File can sua:

- `app.py`
- `README.md`
- `scripts/sync_data_and_report.py`
- Cac docs neu chua link bypass kha dung

Vung code hien tai:

- `app.py` quanh `_khoi_phuc_dang_nhap_tu_query_params()` va `_hoan_tat_dang_nhap()`
- `app.py` co comment quanh cuoi file ve link `?logged_in_user=...`
- `README.md` co link HF Space kem query params
- `scripts/sync_data_and_report.py` co template sinh lai link nay

Thu tu thuc hien:

1. Trong `app.py`, xoa hoac vo hieu hoa hoan toan logic doc `logged_in_user` va `logged_in_role` tu `st.query_params`.
2. Trong `_hoan_tat_dang_nhap()`, khong ghi identity vao query params nua.
3. Neu can giu deep-link, chi cho phep tham so trung tinh nhu `tab`, `page`, `next`, va validate whitelist.
4. Cap nhat comment/log lien quan de khong huong dan login bang link.
5. Trong `README.md`, xoa link co `logged_in_user`/`logged_in_role`; thay bang link Space khong identity hoac mo ta chung.
6. Trong `scripts/sync_data_and_report.py`, sua template de khong sinh lai link bypass vao README.
7. Quet toan repo de dam bao khong con link bypass kha dung.

Lenh verify:

```powershell
rg -n "logged_in_user|logged_in_role" app.py scripts README.md
rg -n "\\?logged_in_user=|&logged_in_role=" .
python -m py_compile app.py scripts\sync_data_and_report.py
```

Tieu chi xong:

- `rg "logged_in_user|logged_in_role" app.py scripts README.md` khong con ket qua.
- Dang nhap chi xay ra sau khi credential/OIDC hop le.
- README/script khong the sinh lai URL bypass.

Can tranh:

- Khong thay bypass bang token query param moi.
- Khong tin vao role do client/query string gui len.

## 1A.2 - Tat debug UI lam lo token va URL cloud

Lien quan loi: F05, F17, N02

File can sua:

- `app.py`

Vung code hien tai:

- Cac noi tao URL `...?token={HF_TOKEN}` cho video/debug.
- Debug popover quanh khu vuc danh sach video/bac si/NCV.

Thu tu thuc hien:

1. Tim tat ca noi render hoac write URL co token ra UI.
2. Xoa debug popover neu khong bat buoc.
3. Neu can giu debug tam thoi, guard bang ca 3 dieu kien:
   - `DEBUG=true` tu env/secrets.
   - `role == "Admin"`.
   - Token/path duoc mask, khong render URL day du.
4. Khong dua token vao `st.markdown`, `st.video`, HTML `<source>`, log UI hay exception message.
5. Phase 1A duoc phep tam thoi an nut/feature video cloud neu chua co proxy Phase 1B.

Lenh verify:

```powershell
rg -n "token=\\{HF_TOKEN\\}|\\?token=|HF_TOKEN.*st\\.|st\\..*HF_TOKEN|cloud_url" app.py
python -m py_compile app.py
```

Tieu chi xong:

- Inspect code khong con render URL co token ra frontend.
- Debug token khong xuat hien trong DOM/UI theo luong thu cong.

Can tranh:

- Khong mask token bang CSS/HTML ma van de gia tri that trong DOM.
- Khong ghi token vao session_state de UI in ra.

## 1A.3 - Xoa fallback `HF_DATASET_ID` hard-code nguy hiem

Lien quan loi: N15, F05, F38

File can sua:

- `app.py`
- Co the them docs cau hinh trong `.streamlit/streamlit_configuration.md`

Thu tu thuc hien:

1. Doi khoi tao `HF_DATASET_ID` de chi lay tu env/secrets.
2. Neu thieu `HF_DATASET_ID`, set `None` va hien canh bao cau hinh ro rang trong UI/admin area.
3. Khong tu fallback ve dataset cua tac gia khi deploy/fork.
4. Dam bao cac helper HF xu ly `None` bang loi ro, khong crash.

Lenh verify:

```powershell
rg -n "Rehab-AI-Monitor-2026-data|quynhphuong1209" app.py scripts README.md .streamlit
python -m py_compile app.py
```

Tieu chi xong:

- `app.py` khong fallback dataset ID hard-code.
- Khi thieu config, app bao loi cau hinh thay vi doc/ghi nham dataset.

Can tranh:

- Khong hard-code dataset moi trong source.
- Script doc lap co the giu default tam thoi neu co flag ro, nhung app production khong duoc.

## 1A.4 - Loai PII khoi README va script sinh bao cao

Lien quan loi: F39, N08, N09, N20

File can sua:

- `README.md`
- `scripts/sync_data_and_report.py`
- Tao `docs/generated/` neu can

Thu tu thuc hien:

1. Trong README, thay noi dung bao cao co ten that/benh su/email/clinical details bang summary da sanitize.
2. Trong `scripts/sync_data_and_report.py`, bo mapping ten that va bao cao lam san hard-code.
3. Script khong ghi de README bang du lieu runtime/PII.
4. Neu can generated report, ghi vao `docs/generated/` va sanitize mac dinh.
5. Them note: generated reports khong duoc commit neu chua review PII.

Lenh verify:

```powershell
rg -n "logged_in_user|logged_in_role" README.md scripts\sync_data_and_report.py
rg -n "email|gmail|benh|triệu chứng|bệnh sử|full_name" README.md scripts\sync_data_and_report.py
python -m py_compile scripts\sync_data_and_report.py
```

Tieu chi xong:

- README khong con PII/link bypass.
- Script khong the tao lai README co PII/link bypass.
- Generated output mac dinh nam ngoai README.

Can tranh:

- Khong chi sua README ma quen template script.
- Khong thay ten that bang pseudonym neu mapping pseudonym van de trong repo.

## 1A.5 - Rotate secrets va passwords da lo

Day la task van hanh, khong chi code.

Viec can lam:

1. Revoke HF token cu, tao token moi co scope toi thieu.
2. Cap nhat secret tren HF Spaces/local `.streamlit/secrets.toml` ngoai repo.
3. Reset toan bo mat khau mac dinh da lo.
4. Neu repo da public, invalidate hash cu trong `database/users.json` va bat buoc doi mat khau lan dau.

Tieu chi xong:

- Token cu khong con dung duoc.
- Khong con mat khau mac dinh trong van hanh.
- Ket qua rotate khong duoc commit vao repo.

## Definition of Done Phase 1A

- Auth bypass query params bi xoa.
- README/script khong con link bypass hoac PII ro rang.
- Token/debug URL khong render ra frontend.
- App khong fallback sang dataset hard-code.
- Secrets/passwords da duoc rotate neu da public hoac khong chac.

