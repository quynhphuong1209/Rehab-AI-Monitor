# Backend API

Backend API rieng cho du an, tach khoi Streamlit frontend theo tung buoc.

Chay backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Endpoint hien co:

- `GET /health`
- `POST /auth/login`
- `POST /auth/register`
- `GET /auth/me`
- `POST /auth/logout`
- `GET /patients`
- `GET /videos`
- `GET /videos/media/{stored_filename}`
- `POST /videos/upload`
- `POST /videos/{stored_filename}/analysis-jobs`
- `GET /videos/{stored_filename}/analysis-jobs/latest`
- `GET /videos/{stored_filename}/analysis-jobs/history`
- `POST /videos/{stored_filename}/analysis-jobs/cancel`
- `POST /videos/{stored_filename}/analysis-jobs/retry`
- `POST /videos/{stored_filename}/analysis-jobs/rerun`
- `GET /videos/{stored_filename}/results`
- `GET /videos/{stored_filename}/analysis-frames`
- `GET /videos/{stored_filename}/analysis-frames/{image_id}`
- `GET /videos/{stored_filename}/analysis-chart`
- `GET /videos/{stored_filename}/analysis-artifacts`
- `GET /videos/{stored_filename}/analysis-artifacts/{artifact_kind}`
- `GET /evaluations`
- `GET /symptoms`
- `POST /symptoms`
- `GET /schedules`
- `GET /research-records`

Vi du dang nhap:

```powershell
$login = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/auth/login `
  -ContentType 'application/json' `
  -Body '{"username":"admin","password":"your-password"}'

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/videos `
  -Headers @{ Authorization = "Bearer $($login.access_token)" }
```

Vi du dang ky benh nhan:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/auth/register `
  -ContentType 'application/json' `
  -Body '{"username":"patient01","full_name":"Patient One","email":"patient01@example.test","password":"patientpass","confirm_password":"patientpass"}'
```

`/auth/register` chi tao tai khoan `Benh nhan`. Tai khoan Bac si, Nghien cuu vien va Quan tri vien can duoc cap boi admin.

Vi du benh nhan khai bao trieu chung:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/symptoms `
  -Headers @{ Authorization = "Bearer $($login.access_token)" } `
  -ContentType 'application/json' `
  -Body '{"full_name":"Patient One","patient_id":"BN001","age":40,"gender":"Nu","exercise":"Codman","symptoms":"Dau vai khi nang tay","vas":5}'
```

Vi du benh nhan upload video:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/videos/upload `
  -Headers @{ Authorization = "Bearer $($login.access_token)" } `
  -Form @{ full_name = "Patient One"; exercise = "Codman"; file = Get-Item ".\sample.mp4" }
```

Vi du xem video da upload qua backend:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/videos/media/patient01_clip.mp4" `
  -Headers @{ Authorization = "Bearer $($login.access_token)" } `
  -OutFile ".\preview.mp4"
```

`/videos/media/{stored_filename}` chi tra file nam trong media root cho phep va thuoc video record actor duoc xem theo role/patient scope.

Vi du tao job phan tich cho video:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/videos/patient01_clip.mp4/analysis-jobs" `
  -Headers @{ Authorization = "Bearer $($login.access_token)" } `
  -ContentType 'application/json' `
  -Body '{"model_type":"MediaPipe Heavy","skip_step":0,"resize_width":720,"min_confidence":0.5}'

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/videos/patient01_clip.mp4/analysis-jobs/latest" `
  -Headers @{ Authorization = "Bearer $($login.access_token)" }
```

`POST /videos/{stored_filename}/analysis-jobs` hien chi cho `Nghien cuu vien` va `Quan tri vien`. Endpoint ghi progress JSON theo convention `processed_results/progress_<md5(video_path)>.json`; runner backend hien kiem tra ffprobe, transcode sang MP4/H.264 khi can. Mac dinh job dung o trang thai `ready_for_ai_worker`. Dat `REHAB_BACKEND_ENABLE_AI_RUNNER=1` de backend gan MediaPipe runner opt-in, goi `video.processing.xu_ly_video_day_du`, ghi CSV/metrics/processed video va cap nhat `video_list.json` khi thanh cong.

Lifecycle job nang cao:

- `GET /analysis-jobs/history`: tra toi da 50 lan chay gan nhat theo video.
- `POST /analysis-jobs/cancel`: danh dau job dang chay la `canceled` va giu progress/history hop le.
- `POST /analysis-jobs/retry`: tao `run_id` moi voi options cua lan chay gan nhat.
- `POST /analysis-jobs/rerun`: tao `run_id` moi voi options trong body, ho tro `model_type`, `skip_step`, `resize_width`, `min_confidence`.

`job_id` van la md5 cua `video_path` de latest/poll on dinh theo video. Moi lan start/retry/rerun co `run_id` rieng, `job_meta.action`, `job_meta.options` va `steps` bon buoc: validate/transcode, MediaPipe pass 1, overlay/export, artifact/persist.

Vi du xem ket qua chi tiet theo video:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/videos/patient01_clip.mp4/results" `
  -Headers @{ Authorization = "Bearer $($login.access_token)" }
```

`GET /videos/{stored_filename}/results` gom video, danh gia bac si, latest analysis job, metrics, artifact manifest va timeline nho. Endpoint van dung patient scope cua video: benh nhan chi xem video cua minh, bac si/KTV chi xem benh nhan duoc gan, admin xem all, NCV duoc pseudonymize theo response shaping hien co. Bac si/KTV chi nhan AI detail khi co ban ghi bao cao chinh thuc tu `AI_Researcher`; response luon tra `report_sent`, `report_status` va `ai_detail_allowed`.

Vi du xem gallery frame theo trang:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/videos/patient01_clip.mp4/analysis-frames?page=1&page_size=12&label=G2" `
  -Headers @{ Authorization = "Bearer $($login.access_token)" }
```

`GET /videos/{stored_filename}/analysis-frames` doc metadata tu `all_frames_data_path`, kiem ZIP `frames_zip_path` neu co anh, tra pagination va summary PASS/NEAR/FAIL. Filter ho tro `ALL`, `G1`, `G2`, `G3`, `PASS`, `NEAR`, `FAIL`; backend tinh segment bang logic `video.metrics.segment_frames` va nguong REF G1/G2/G3 la 45/30/15 do. Anh frame duoc tai rieng qua `/analysis-frames/{image_id}`; backend khong tra path local va khong doc toan bo ZIP cho gallery. Neu frame co `ml_label_text`, `ml_confidence`, `ml_probabilities`, response tra badge ML de React hien thi.

Preview bieu do nhe cho React:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/videos/patient01_clip.mp4/analysis-chart" `
  -Headers @{ Authorization = "Bearer $($login.access_token)" }
```

`GET /videos/{stored_filename}/analysis-chart` uu tien doc CSV tu `df_path`, fallback sang `all_frames_data_path`, gioi han kich thuoc/so dong, downsample toi da 180 diem va tra summary PASS/NEAR/FAIL cung series goc vai/khuyu de frontend ve SVG chart. Query `label=G1|G2|G3|PASS|NEAR|FAIL` loc cung logic voi gallery; response co `phase_summary`, `segment_bounds` va `phase_metrics` neu metadata video co `metrics_g1/g2/g3`.

Frontend Streamlit hien van chay doc lap:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Bat frontend goi backend API theo tung phan:

```powershell
$env:REHAB_BACKEND_URL="http://127.0.0.1:8000"
$env:REHAB_FRONTEND_USE_BACKEND="1"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Neu khong dat `REHAB_FRONTEND_USE_BACKEND=1`, frontend van dung flow JSON local hien co.

Giai doan hien tai:

- Backend doc JSON qua `backend.repository`, khong import `app.py`.
- Auth dung password verifier hien co trong `auth.passwords`.
- Dang ky benh nhan tu phuc vu dung Argon2 va ghi JSON qua locked storage helper.
- Benh nhan co the tao khai bao trieu chung qua API; record duoc gan username theo bearer token.
- Benh nhan co the upload video qua multipart API; file luu vao `patient_uploads/`, metadata ghi `video_list.json`.
- Backend co endpoint media de phuc vu video upload theo bearer token va scope actor.
- Backend co contract job progress cho phan tich video; React co the start/poll/cancel/retry/rerun job, backend da validate/transcode H.264 va co MediaPipe runner opt-in (`REHAB_BACKEND_ENABLE_AI_RUNNER=1`) de persist metrics/processed_path vao `video_list.json` khi thanh cong.
- Backend co endpoint ket qua chi tiet `/videos/{stored_filename}/results` de React gom evaluation/latest job/metrics/artifact/timeline theo video, kem gate report cho bac si truoc khi NCV gui bao cao.
- Backend co endpoint gallery frame `/videos/{stored_filename}/analysis-frames` voi pagination, filter G1/G2/G3/PASS/NEAR/FAIL, ML badge va image endpoint rieng.
- Admin API co `/admin/audit-log`, lock/unlock user, reset password bat doi mat khau, revoke session theo user/toan bo; user ops co audit log va backup `users.json` truoc thao tac ghi/xoa.
- Scope theo role/patient dung `auth.permissions`.
- `backend.access` gom response shaping va pseudonymize cho NCV.
- Frontend co `frontend.api_client` va opt-in qua `REHAB_FRONTEND_USE_BACKEND=1`.
- Worker AI/MediaPipe that da co adapter backend opt-in, chua bat mac dinh de tranh thay doi runtime khi chua smoke test tren video that.

AI runner env tuy chon:

- `REHAB_BACKEND_ENABLE_AI_RUNNER=1`: bat backend MediaPipe runner.
- `REHAB_BACKEND_AI_MODEL_TYPE`: mac dinh `MediaPipe Heavy`.
- `REHAB_BACKEND_AI_MIN_CONFIDENCE`: mac dinh `0.5`.
- `REHAB_BACKEND_AI_SKIP_STEP`: mac dinh `0`.
- `REHAB_BACKEND_AI_RESIZE_WIDTH`: mac dinh `720`.
- `REHAB_BACKEND_AI_ENABLE_POSE_CLASSIFIER=1`: bat classifier phu neu dependency/model san sang.
