# JSON To SQLite Migration

The current runtime still reads JSON. `scripts/migrate_json_to_sqlite.py` is a
production-gate migration tool so the team can validate counts, create backups,
and rehearse rollback before switching repository storage to SQLite.

## Dry Run

```powershell
.\.venv\Scripts\python.exe scripts\migrate_json_to_sqlite.py --repo-root . --dry-run
```

Dry-run validates known JSON files through the existing schema normalizers and
prints a JSON report with table counts and schema issues. It does not create a
SQLite file.

## Apply

```powershell
.\.venv\Scripts\python.exe scripts\migrate_json_to_sqlite.py `
  --repo-root . `
  --sqlite-path database\rehab_monitor.sqlite3
```

Apply creates `backups/sqlite_migration/<timestamp>/` before writing the SQLite
database. The script imports each JSON record into `runtime_records` with the
source file, stable record key, payload JSON, checksum, and migration timestamp.

## Rollback

```powershell
.\.venv\Scripts\python.exe scripts\migrate_json_to_sqlite.py --repo-root . --rollback
```

Rollback restores JSON files from the latest migration backup. To use a specific
snapshot, pass `--backup-dir backups\sqlite_migration\<timestamp>`.

## Scope

Included JSON files:

- `users.json`
- `video_list.json`
- `doctor_evaluations.json`
- `patient_symptoms.json`
- `schedules.json`
- `research_data.json`
- `user_feedback.json`
- `audit_log.json`

This tool does not yet change application runtime storage. That switch should be
a later repository-layer change after staging restore drills pass.
