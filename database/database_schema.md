# JSON Database Schema

Ứng dụng vẫn dùng JSON flat-file trong `database/` hoặc `/data` trên HF Spaces. Tất cả file runtime nên được đọc/ghi qua `load_data()` / `save_data()` trong app hoặc `storage/json_store.py`; dữ liệu được normalize bằng `models/schemas.py` trước khi render UI.

## users.json

Root type: object keyed by username.

Required/effective fields per user:

- `username`: string, trùng key.
- `password`: password hash.
- `hash_version`: ví dụ `argon2`.
- `role`: `Bệnh nhân`, `Bác sĩ / KTV PHCN`, `Nghiên cứu viên`, hoặc `Quản trị viên`.
- `full_name`: display name.
- `email`: optional.
- `must_change_password`: boolean.
- `assigned_patient_usernames`: list username bệnh nhân do bác sĩ/KTV phụ trách.
- `assigned_doctor_username`: optional username bác sĩ phụ trách bệnh nhân.
- `team_usernames`: optional list cho nhóm chăm sóc.
- `active`, `created_at`, `updated_at`: metadata.

## video_list.json

Root type: list.

Core fields:

- `username`, `full_name`: bệnh nhân sở hữu video.
- `video_name`, `original_filename`, `stored_filename`.
- `exercise`.
- `video_path`, `processed_path`, `df_path`, `frames_zip_path`, `all_frames_data_path`.
- `accuracy`, `metrics`, `status`, `time`.

UI access is scoped by current actor. Patients see self records, doctors/KTV see assigned patients, researchers see research workflow data, admins see all.

## doctor_evaluations.json

Root type: list.

Fields:

- `patient_username`, `doctor_username`, `doctor_name`.
- `video_name`, `exercise`.
- `doctor_result`, `errors`, `comments`, `comments_ncv`, `plan`.
- `time`.

Records with missing optional fields are filled with defaults. Broken rows missing both `patient_username` and `video_name` are skipped by schema normalization.

## schedules.json

Root type: list.

Fields:

- `id`, `type`: `appointment`, `exercise`, or `medication`.
- `patient_username`, `patient_name`.
- `doctor_username`, `doctor_name`.
- `title`, `datetime`, `notes`.
- Type-specific: `exercise_name`, `frequency`, `medication_name`, `dosage`, `taken`.

Schedule UI is patient-scoped using the same assignment rules as videos/evaluations.

## patient_symptoms.json

Root type: list.

Fields:

- `username`, `full_name`, `patient_id`.
- `age`, `gender`, `symptoms`, `vas`.
- `exercise`, `exercises`, `time`.

Missing `patient_id` is normalized from `username`.

## research_data.json

Root type: list.

Fields:

- `patient_username`, `subject_code`.
- `interviewer`, `interview_date`, `timestamp`.
- `age`, `gender`, `diagnosis`, `duration`, `training_side`, `pain_level`, `disease_severity`.
- `exercises`, `general_result`, `errors`, `plan`, `specialist_comment`.
- `video_code`, `recording_device`, `recording_angle`, `camera_distance`.
- `submitted_by`, `role`.

NCV/researcher views and exports use pseudonymized records by default: direct identifiers and clinical free-text notes are removed before display/export.

## lich_su_tap_luyen.json

Root type: list.

Fields:

- `username`, `full_name`.
- `bai_tap`, `accuracy`, `ngay`, `thoi_gian_tap`.
- Optional AI/metrics fields may be present.

## processed_results/progress_*.json

Root type: object.

Fields:

- `job_id`: md5 of normalized `video_path`.
- `run_id`: unique id for each start/retry/rerun attempt on the video.
- `video_path`, `username`, `video_name`, `exercise`.
- `status`: `processing`, `ready_for_ai_worker`, `success`, `error`, or `canceled`.
- `progress`: number from 0 to 1.
- `elapsed`, `start_time`, `heartbeat`, `updated_at`.
- `status_msg`, `error_msg`.
- `result`: optional object when analysis finishes or when video is ready for the next stage. For `ready_for_ai_worker`, it can include `analysis_input_path`, `transcoded`, `source_path`, `video_codec`, and `audio_codec`. For `success`, backend AI runners should include fields that can update `video_list.json`, such as `processed_path` or `processed_video_path`, `metrics` or `stats`, `df_path`, `all_frames_data_path`, `frames_zip_path`, `accuracy`, `sai_so`, and `giai_doan`.
- `job_meta`: optional object, including backend request metadata such as `requested_by`, `action`, and public `options` (`model_type`, `skip_step`, `resize_width`, `min_confidence`).
- `steps`: four workflow steps for UI status: validate/transcode, MediaPipe pass 1, overlay/export, artifact/persist.

Backend job endpoints and legacy Streamlit progress use the same file convention so the worker can share progress state. Backend API supports an injectable AI runner hook. Without that hook, jobs stop at `ready_for_ai_worker`. With `REHAB_BACKEND_ENABLE_AI_RUNNER=1`, the backend MediaPipe runner calls `video.processing.xu_ly_video_day_du`; a successful result is persisted back into `video_list.json`. History for each video is stored as `processed_results/analysis_job_history_<md5(video_path)>.json` and keeps the latest run entries for `GET /videos/{stored_filename}/analysis-jobs/history`.

## Backend `/videos/{stored_filename}/results`

Root type: object.

Fields:

- `video`: scoped video record with `stored_filename` and display `video_name`.
- `evaluation`: latest matching doctor evaluation for the video, or `null`. Patient responses omit internal `comments_ncv`.
- `latest_job`: latest progress object from `processed_results/progress_*.json`, or `null`.
- `metrics`: metrics from `video_list.json`, falling back to job result metrics/stats.
- `artifacts`: manifest items for processed video, CSV, frames JSON and frames ZIP.
- `report_sent`, `report_status`, `ai_detail_allowed`: report gate fields. For doctors, AI detail is hidden until an `AI_Researcher` evaluation/report matches the video.
- `phase_metrics`: normalized G1/G2/G3 metrics when available from `metrics.metrics_g1`, `metrics.metrics_g2`, `metrics.metrics_g3`.
- `summary`: patient-friendly summary fields used by React.
- `timeline`: compact events from video upload, symptom declaration, doctor evaluation, AI job and matching schedule.

This response is generated through the same video visibility check as media/artifact endpoints. It must not be assembled client-side from unscoped records.

## Backend `/videos/{stored_filename}/analysis-frames`

Root type: object.

Fields:

- `items`: page of normalized frame metadata.
- `summary`: total PASS/NEAR/FAIL counts from `all_frames_data_path`, plus `phases.G1/G2/G3` counts.
- `pagination`: `page`, `page_size`, `total`, `total_pages`.
- `filter`: `ALL`, `G1`, `G2`, `G3`, `PASS`, `NEAR`, or `FAIL`.
- `segment_bounds`: `[0, n1, n2, total]` from `video.metrics.segment_frames`.
- `phase_ranges`: start/end/threshold metadata for G1/G2/G3. Thresholds are 45/30/15 degrees.
- `sources`: booleans for `frames_json` and `frames_zip`.

Each item includes `index`, `timestamp`, `label`, `phase`, `phase_threshold`, `image_id`, `has_image`, shoulder/elbow angles, reference angles and deltas when available. If the artifact includes `ml_label`, `ml_label_text`, `ml_confidence` or `ml_probabilities`, item `ml` is returned for the React badge. `image_id` is opaque (`frame:<index>`); frontend must request `/videos/{stored_filename}/analysis-frames/{image_id}` instead of using local paths. Backend validates frame JSON, ZIP names, image extensions and patient scope before serving image bytes.

## Backend `/videos/{stored_filename}/analysis-chart`

Root type: object.

Fields:

- `source`: `csv`, `frames-json`, or `none`; CSV from `df_path` is preferred over frame JSON.
- `filter`: same label filter vocabulary as frame gallery.
- `total_rows`, `filtered_rows`, and `sampled_rows`: original row count, filtered row count and returned chart point count.
- `columns`: angle/reference series present in the payload.
- `summary`: per-series min/max/avg plus PASS/NEAR/FAIL counts.
- `phase_summary`: PASS/NEAR/FAIL counts for G1/G2/G3.
- `phase_metrics`: normalized per-phase `accuracy`, `mae`, `f1`, `icc` when available.
- `metrics`: video metrics copied from scoped video metadata when present.
- `series`: downsampled chart points with `index`, `frame`, `timestamp`, `label`, `phase`, `goc_vai`, `goc_khuyu`, `vai_chuan`, and `khuyu_chuan`.

The endpoint uses the same video visibility and artifact path checks as the download endpoints. Source files are size/row limited before parsing and downsampled before being returned to React.

## Migration

Run an idempotent dry-run first:

```powershell
python -m models.migrate_json --data-dir database --dry-run
```

Apply migration with automatic backups:

```powershell
python -m models.migrate_json --data-dir database
```

## Privacy Configuration

- `ALLOW_NETWORK_TTS=false` by default. When false, audio feedback uses local beep fallback instead of calling gTTS.
- `WEBRTC_STUN_URLS` is empty by default. Set comma-separated STUN/TURN URLs only when the deployment policy permits external WebRTC traversal services.
