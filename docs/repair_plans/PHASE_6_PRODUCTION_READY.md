# Phase 6 Plan - Production-ready

Muc tieu: dua ung dung tu prototype Streamlit + JSON flat-file sang nen tang van hanh co transaction, CI/CD, job queue va quy trinh privacy/compliance. Phase nay can quyet dinh moi truong deploy truoc khi code.

## Nguyen tac rieng cua Phase 6

- Khong bat dau migration DB neu Phase 5 chua tach storage layer.
- Moi migration du lieu phai co backup, dry-run, rollback plan.
- Security/compliance khong chi la code: can quy trinh rotate, audit, retention va incident response.

## 6.1 - Chuyen JSON sang SQLite/Postgres

File/module lien quan:

- `storage/`
- `models/`
- `database/`
- Migration scripts moi

Lua chon:

- SQLite neu deploy single-instance, it concurrent users.
- Postgres neu co multi-user, multi-instance, audit/compliance nghiem tuc.

Thu tu thuc hien:

1. Chon DB va driver.
2. Dinh nghia schema:
   - users
   - sessions/session_versions
   - videos
   - evaluations
   - schedules
   - symptoms
   - audit_logs
   - background_jobs
3. Tao migration tool:
   - `json -> db --dry-run`.
   - backup JSON truoc.
   - validate count/checksum sau import.
4. Tao repository layer thay cho JSON store:
   - `UserRepository`
   - `VideoRepository`
   - `EvaluationRepository`
5. Giu compatibility read-only voi JSON trong mot thoi gian neu can.
6. Them transaction cho destructive/multi-file operations.

Tieu chi xong:

- App chay tu DB, khong can ghi runtime JSON cho du lieu chinh.
- Migration co dry-run va report.
- Transaction bao ve update lien quan.

## 6.2 - CI/CD pipeline

File can tao/sua:

- `.github/workflows/ci.yml`
- `requirements-dev.txt`
- Docker build config
- Pre-commit config neu dung

Thu tu thuc hien:

1. CI tren pull request:
   - install deps.
   - `python -m py_compile`.
   - `pytest tests/`.
   - lint/format check neu team chon.
   - security scan dependency.
2. Secret scanning:
   - token `hf_`.
   - query bypass params.
   - email/PII patterns.
3. Docker build smoke.
4. Deploy staging rieng production.
5. Manual approval truoc production.

Tieu chi xong:

- PR khong pass neu test/security scan fail.
- Build Docker duoc kiem tra truoc deploy.

## 6.3 - Background job system

Lien quan loi: F27, F37, N16

Module lien quan:

- `jobs/`
- `video/processing.py`
- `cloud/hf_sync.py`

Lua chon:

- Simple in-process queue cho single-instance.
- RQ/Celery cho production co Redis.
- HF Spaces co gioi han rieng, can chon theo deploy thuc te.

Thu tu thuc hien:

1. Liet ke tat ca background thread hien co.
2. Tao job model:
   - id.
   - type.
   - status.
   - progress.
   - actor.
   - created_at/updated_at.
   - error.
3. Tao worker co stop event, timeout, retry/backoff.
4. UI chi poll job status, khong spawn thread tuy tien.
5. Graceful shutdown.
6. Health check job queue.

Tieu chi xong:

- Khong con thread ad hoc khong stop/backoff.
- Moi job co status va audit.

## 6.4 - Privacy va compliance workflow

Pham vi:

- PII classification.
- Consent.
- Audit trail.
- Retention.
- Incident response.
- Data export cho research.

Thu tu thuc hien:

1. Data classification:
   - public.
   - internal.
   - patient PII.
   - clinical sensitive.
   - secrets.
2. Consent model:
   - patient consent version.
   - allowed processing.
   - allowed research export.
3. Pseudonymization pipeline:
   - stable pseudonym ID.
   - mapping luu rieng, access admin only.
   - export NCV khong chua PII.
4. Audit trail:
   - login/logout.
   - view PII.
   - upload/download.
   - evaluation changes.
   - destructive actions.
5. Retention:
   - delete/archive policy.
   - backup retention.
6. Incident response:
   - rotate token.
   - revoke sessions.
   - history purge.
   - notification checklist.

Tieu chi xong:

- Co policy va code enforce cho PII/research export.
- Audit log du dung de dieu tra.
- Incident checklist duoc tai lieu hoa.

## 6.5 - Observability va ops

Thu tu thuc hien:

1. Structured logging theo request/job/action.
2. Error dashboard hoac log aggregation theo moi truong.
3. Health checks:
   - DB reachable.
   - HF reachable neu bat sync.
   - disk quota.
   - job queue lag.
4. Metrics:
   - upload failures.
   - transcode duration.
   - sync failures.
   - auth failures/rate limit.
5. Backup restore drill dinh ky.

Tieu chi xong:

- Van hanh co the phat hien loi truoc khi user bao.
- Restore backup da duoc test.

## Definition of Done Phase 6

- Du lieu chinh nam trong DB co transaction.
- CI/CD chan regression va secrets.
- Background jobs co queue/status/shutdown.
- Privacy/compliance workflow co code va tai lieu.
- Observability va backup restore san sang cho production.

