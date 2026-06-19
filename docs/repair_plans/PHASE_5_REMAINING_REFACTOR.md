# Phase 5 Remaining Refactor Plan

Muc tieu: tiep tuc giam kich thuoc va do ket dinh cua `app.py` sau cac slice da tach, nhung khong lam thay doi behavior nguoi dung. Plan nay uu tien tach UI truoc, roi moi den background job va core video processing.

## Trang thai hien tai

Baseline do ngay 2026-06-18:

- `app.py`: 15,281 physical lines, 13,624 nonblank lines.
- Full test suite: `100 passed`.
- Da tach:
  - `app_startup.py`
  - `ui/static_pages.py`
  - `ui/layout.py`
  - `ui/admin_pages.py`
  - `ui/reminders.py`
  - `ui/research_forms.py`
  - cac module san co: `auth/`, `storage/`, `cloud/`, `video/io.py`, `video/serving.py`, `video/validation.py`, `ui/patient.py`, `ui/doctor.py`, `ui/researcher.py`, `ui/navigation.py`, `ui/styles.py`

## Nguyen tac an toan

- Moi slice chi lam mot loai thay doi: move code, hoac doi logic. Khong tron hai viec.
- Truoc moi slice chay baseline:

```powershell
python -m py_compile app.py app_startup.py ui\*.py video\*.py auth\*.py storage\*.py cloud\*.py
pytest tests
```

- Sau moi slice phai chay lai cung gate tren.
- Neu tach UI, dung dependency object `deps` nhu cac module UI hien tai de tranh import vong.
- Module domain (`video/*`, `storage/*`, `auth/*`, `cloud/*`) khong import `streamlit`.
- Khong sua thuat toan xu ly video trong cung commit voi viec di chuyen file.
- Khong doi schema/data format neu khong co migration/test rieng.

## Hotspots con lai trong app.py

Can tach theo thu tu rui ro tang dan:

| Cum code | Size gan dung | Rui ro | Huong tach |
|---|---:|---|---|
| `_noi_dung_danh_sach_video_fragment` | 498 lines | Medium | `ui/video_list.py` |
| `_render_frame_grid` | 349 lines | Medium | `ui/frames_viewer.py` |
| `_noi_dung_frames_day_du` | 919 lines | Medium/High | `ui/frames_viewer.py` + `video/frames.py` |
| `_hien_thi_tab_phan_tich_noi_dung` | 986 lines | High | `ui/analysis_tab.py` |
| `_noi_dung_khu_vuc_phan_tich` | 412 lines | High | `ui/analysis_tab.py` |
| `hien_thi_form_danh_gia_bac_si` | 194 lines | Medium | `ui/doctor_forms.py` |
| `hien_thi_ket_qua_gan_nhat_va_lich_su` | 126 lines | Medium | `ui/doctor_forms.py` |
| `bat_dau_phan_tich_background` | 580 lines | High | `video/jobs.py` or `analysis/jobs.py` |
| `xu_ly_frame` | 399 lines | High | `video/processing.py` |
| `xu_ly_video_day_du` | 959 lines | Very High | `video/processing.py` |
| `segment_frames`, `recalc_metrics`, `tinh_metrics_chi_tiet` | 150-200 lines each | High | `video/metrics.py` |

## Slice R0 - Baseline va guard truoc khi tach

Muc tieu: co so do de biet slice nao lam hong.

Cong viec:

- [x] Chay `pytest tests` va luu ket qua. Ket qua: `100 passed`.
- [x] Chay `py_compile` cho app va cac package hien co. Ghi chu: PowerShell khong bung `auth\*.py` khi truyen truc tiep cho `py_compile`; da chay bang danh sach file tu `Get-ChildItem`.
- [x] Ghi lai line count:

```powershell
$env:PYTHONIOENCODING='utf-8'
@'
from pathlib import Path
for f in ["app.py"]:
    text = Path(f).read_text(encoding="utf-8")
    print(f, len(text.splitlines()), sum(1 for line in text.splitlines() if line.strip()))
'@ | python -
```

Tieu chi xong:

- [x] `pytest tests` pass.
- [x] `py_compile` pass.
- [x] Khong co file runtime data bi sua ngoai y muon trong slice nay.

## Slice R1 - Tach video list UI

Muc tieu: dua danh sach video/fragment UI ra khoi `app.py`, khong doi logic storage/cloud.

File tao/sua:

- [x] Tao `ui/video_list.py`
- [x] Sua `app.py`

Pham vi move:

- [x] `_noi_dung_danh_sach_video_fragment`
- [x] wrapper lien quan neu chi render danh sach video

Khong move trong slice nay:

- [x] `delete_video_callback`
- [x] logic sync/download cloud
- [x] core permission/storage helpers

Thiet ke:

```python
def render_video_list_fragment(deps, user_role, video_list_preloaded=None):
    ...
```

Deps du kien:

- `st`
- `load_data`, `save_data`
- `render_video`
- `require_role`, `require_patient_scope`
- `scope_records_for_current_actor`
- `_format_vn_time`
- `_lay_eval_moi_nhat_theo_bai_tap`
- `_lay_do_chinh_xac_hien_thi`
- `_lay_thoi_gian_phan_tich_on_dinh`
- `_dam_bao_video_san_sang_play`
- `_dong_bo_video_list_nen`
- delete callback neu can

Thu tu:

1. Copy function sang `ui/video_list.py`, doi signature nhan `deps`.
2. Trong function, map global cu sang local tu `deps`.
3. Trong `app.py`, import `render_video_list_fragment`.
4. Gan `deps.render_video_list_fragment = lambda ...`.
5. Giu wrapper cu neu cac module khac van goi ten cu.
6. Chay gate test.

Tieu chi xong:

- [x] `app.py` khong con body lon cua `_noi_dung_danh_sach_video_fragment`.
- [x] Route bac si/NCV van hien danh sach video. Da kiem bang route/navigation tests.
- [x] `pytest tests` pass. Ket qua sau R1: `100 passed`; `app.py` con 14,796 physical lines / 13,187 nonblank lines.

## Slice R2 - Tach frames viewer UI

Muc tieu: dua UI xem frames/grid ra module rieng.

File tao/sua:

- [x] Tao `ui/frames_viewer.py`
- [x] Co the tao `video/frames.py` neu co helper pure file/zip can tach rieng. Chua tao trong R2 de giu slice UI-only.
- [x] Sua `app.py`

Pham vi move truoc:

- [x] `_render_frame_grid`
- [x] UI-only helpers cua frame grid

Pham vi move sau khi slice dau pass:

- [x] `_noi_dung_frames_day_du`
- [x] `hien_thi_frames_day_du` wrapper neu phu hop

Can canh giac:

- Session state: `all_frames_data`, `frames_zip`, `current_page`, `processed_video_path`.
- File paths: processed video, extracted frames, zip frames.
- Download/extract side effects.

Thu tu:

1. Tach `_render_frame_grid` truoc.
2. Chay gate.
3. Tach `_noi_dung_frames_day_du`.
4. Chi move zip/extract helpers sang `video/frames.py` neu da co test path/zip.

Tieu chi xong:

- [x] UI frames cua bac si/NCV van render duoc. Da kiem bang compile + route tests; UI behavior giu qua wrapper.
- [x] Khong import `streamlit` trong `video/frames.py`. Chua tao `video/frames.py` trong R2.
- [x] `pytest tests` pass. Ket qua sau R2: `100 passed`; `app.py` con 13,896 physical lines / 12,368 nonblank lines.

## Slice R3 - Tach analysis tab UI

Muc tieu: dua man hinh phan tich, progress, action buttons ra module UI rieng. Chua tach core processing.

File tao/sua:

- [x] Tao `ui/analysis_tab.py`
- [x] Sua `app.py`

Pham vi move:

- [x] `_hien_thi_tab_phan_tich_noi_dung`
- [x] `_noi_dung_khu_vuc_phan_tich`
- [x] cac helper render progress/status neu chi phuc vu tab phan tich. Da move deep analysis/progress area sang `ui/analysis_tab.py`; core job/processing giu nguyen trong `app.py`.

Khong move trong slice nay:

- [x] `xu_ly_video_day_du`
- [x] `bat_dau_phan_tich_background`
- [x] `xu_ly_frame`

Thiet ke:

```python
def render_analysis_tab(deps, key_suffix="", stats_ext=None, df_ext=None, exercise_ext=None):
    ...
```

Deps du kien:

- `st`
- `read_progress`, `write_progress`, `clear_analysis_progress`
- `bat_dau_phan_tich_background` hoac wrapper start job
- `read_display_csv_fast`
- `load_danh_sach_video_nghien_cuu`
- `PHASE_ERROR`, `PHASE_UI_LABELS`
- cac helper session/result hien tai

Tieu chi xong:

- [x] NCV van chay/quan sat progress phan tich. Da kiem bang compile + route tests; logic job/core khong move.
- [x] Bac si van xem ket qua AI tu sub-tab. `deps.render_analysis_tab` tro sang module moi.
- [x] `pytest tests` pass. Ket qua sau R3: `100 passed`; `app.py` con 12,547 physical lines / 11,117 nonblank lines.

## Slice R4 - Tach doctor evaluation forms

Muc tieu: tach UI/form danh gia bac si con lai.

File tao/sua:

- [x] Tao `ui/doctor_forms.py`
- [x] Sua `ui/doctor.py`. Khong can sua truc tiep: `ui/doctor.py` da goi qua `deps.render_doctor_eval_form`.
- [x] Sua `app.py`

Pham vi move:

- [x] `hien_thi_form_danh_gia_bac_si`
- [x] `hien_thi_ket_qua_gan_nhat_va_lich_su`
- [x] `hien_thi_tab_ket_qua_da_chon` neu cung workflow

Thiet ke:

```python
def render_doctor_evaluation_form(deps):
    ...

def render_latest_results_and_history(deps):
    ...
```

Tieu chi xong:

- [x] `ui/doctor.py` goi form qua module moi. Duong goi qua `deps.render_doctor_eval_form` tro sang `ui.doctor_forms`.
- [x] Khong con form danh gia bac si trong `app.py`. `app.py` chi con wrapper tuong thich.
- [x] `pytest tests` pass. Ket qua sau R4: `100 passed`; `app.py` con 12,190 physical lines / 10,793 nonblank lines.

## Slice R5 - Tach background analysis jobs

Muc tieu: dua orchestration thread/progress ra khoi UI.

File tao/sua:

- [x] Tao `video/jobs.py` hoac `analysis/jobs.py`. Da tao `video/jobs.py`.
- [x] Sua `app.py`
- [x] Them tests unit nho cho job guard/progress neu can. Da them `tests/unit/test_video_jobs.py`.

Pham vi move:

- [x] `bat_dau_phan_tich_background`
- [x] `_running_threads`
- [x] `_analysis_slots`
- [x] `_cancel_flags`
- [x] helpers cleanup/recover job neu lien quan truc tiep

Thiet ke de xuat:

```python
class AnalysisRequest:
    username: str
    video_path: str
    video_name: str
    exercise_name: str
    phase_key: str
    options: dict

def start_analysis_job(request, services):
    ...
```

Nguyen tac:

- Slice dau chi move code, khong thay threading model.
- Da chuan hoa job registry bang `AnalysisJobRegistry` va `AnalysisSlotPool`.
- UI/app van goi wrapper tuong thich `bat_dau_phan_tich_background(...)`; wrapper delegate sang `video.jobs`.

Tieu chi xong:

- [x] Import `app.py` khong start job.
- [x] UI start/cancel/recover job nhu cu.
- [x] `pytest tests` pass. Ket qua sau R5/R6/R7-safe: `107 passed`; `app.py` con 9,834 physical lines / 8,709 nonblank lines.

## Slice R6 - Tach core video processing

Muc tieu: tach thuat toan xu ly video/frame/metrics ra module domain.

File tao/sua:

- [x] Tao `video/processing.py`
- [x] Tao `video/metrics.py`
- [x] Sua `app.py`
- [x] Them tests cho pure functions. Da them `tests/unit/test_video_metrics.py` va smoke test `tests/unit/test_video_processing.py`.

Pham vi move:

- [x] `xu_ly_frame`
- [x] `xu_ly_video_day_du`
- [x] `segment_frames`
- [x] `recalc_metrics`
- [x] `tinh_metrics_chi_tiet`
- [ ] helpers tinh goc/canh bao neu con trong `app.py`. Tam thoi truyen qua deps de giu behavior; co the tach tiep sau khi co visual/video regression tests.

Yeu cau truoc khi move:

- [x] Co test cho `segment_frames` bang dataframe/list gia.
- [x] Co test cho `recalc_metrics` voi input nho.
- [x] Co test smoke cho processing request voi mock dependencies, khong can video that.

Nguyen tac:

- `video/processing.py` khong import `streamlit`.
- Progress callback truyen vao nhu function, khong goi `st.*`.
- File IO dung path guard/validation san co.

Tieu chi xong:

- [x] `app.py` khong con core loop xu ly frame/video. App chi con wrapper tuong thich.
- [x] Processing co API ro rang cho job layer goi. Hien tai dung deps adapter de tranh doi behavior; `video.processing` khong import/gọi `st.*` truc tiep.
- [x] `pytest tests` pass. Ket qua sau R6: `107 passed`.

## Slice R7 - Don dep orchestrator app.py

Muc tieu: sau khi cac slice lon da tach, lam `app.py` thanh entrypoint va router.

Cong viec:

- [ ] Xoa wrappers khong con can. Chua xoa het de giu tuong thich call-site.
- [ ] Gom `_build_ui_tab_dependencies()` thanh cac group nho:
  - `build_storage_deps()`
  - `build_video_deps()`
  - `build_ui_deps()`
- [x] Giam global mutable state con lai. Da chuyen job registry sang `video.jobs`; con deps adapter tam thoi.
- [x] Cap nhat `PHASE_5_ARCHITECTURE.md` va roadmap status. Da cap nhat Phase 5 remaining plan; roadmap/architecture co the cap nhat rieng sau khi bo wrapper tam.

Tieu chi xong:

- [ ] `app.py` chu yeu gom startup, constants, deps wiring, route role/tab. Da giam manh nhung van con wrappers va mot so helper domain.
- [x] Domain logic nam trong module rieng. Job/processing/metrics da nam trong `video/*`.
- [x] `pytest tests` pass. Ket qua: `107 passed`.

## Gate lenh chuan sau moi slice

```powershell
python -m py_compile app.py app_startup.py auth\*.py storage\*.py cloud\*.py video\*.py ui\*.py
pytest tests
```

Khi co thay doi UI routing:

```powershell
pytest tests\unit\test_ui_navigation.py tests\unit\test_ui_role_routes.py tests\unit\test_app_startup.py
```

Khi co thay doi video/path:

```powershell
pytest tests\unit\test_video_io.py tests\unit\test_video_serving.py tests\unit\test_video_validation.py tests\unit\test_path_security.py
```

Khi co thay doi auth/storage:

```powershell
pytest tests\unit\test_accounts.py tests\unit\test_permissions.py tests\unit\test_sessions.py tests\unit\test_storage_app_json.py tests\unit\test_storage_json_store.py
```

## Definition of Done

- [x] `app.py` khong con cac function UI/video tren 300 dong, tru cac wrapper tam thoi co ly do ro.
- [x] `video/processing.py` va `video/jobs.py` khong import `streamlit` va khong goi `st.*` truc tiep.
- [x] UI modules chi render UI va goi services qua `deps`.
- [x] Khong co import vong giua `app.py` va `ui/*`.
- [x] `pytest tests` pass sau tung slice.
- [x] Docs Phase 5 duoc cap nhat sau khi hoan thanh tung slice.
