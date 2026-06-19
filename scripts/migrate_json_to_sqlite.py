"""Migrate runtime JSON data into a single SQLite database.

The app still runs on JSON today; this tool is the Phase H production gate for a
durable migration path. It validates JSON through the existing schema layer,
supports dry-run, backs up source JSON before apply, and can restore the last
backup snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import normalize_json_data
from storage.json_store import read_json


JSON_TABLES = {
    "users": "users.json",
    "videos": "video_list.json",
    "evaluations": "doctor_evaluations.json",
    "symptoms": "patient_symptoms.json",
    "schedules": "schedules.json",
    "research_records": "research_data.json",
    "feedback": "user_feedback.json",
    "audit_log": "audit_log.json",
}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS migration_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_records (
    table_name TEXT NOT NULL,
    record_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_index INTEGER NOT NULL,
    payload_sha256 TEXT NOT NULL,
    migrated_at TEXT NOT NULL,
    PRIMARY KEY (table_name, record_key)
);

CREATE INDEX IF NOT EXISTS idx_runtime_records_table ON runtime_records(table_name);
CREATE INDEX IF NOT EXISTS idx_runtime_records_source ON runtime_records(source_file);
"""


def _json_default(path: Path) -> Any:
    return {} if path.name == "users.json" else []


def _stable_record_key(table_name: str, source_key: str, record: dict[str, Any], index: int) -> str:
    for key in ("id", "username", "job_id", "run_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    identity = {
        "table": table_name,
        "source_key": source_key,
        "patient": record.get("patient_username") or record.get("username") or record.get("subject_code"),
        "video": record.get("video_name") or record.get("stored_filename") or record.get("video_code"),
        "time": record.get("timestamp") or record.get("created_at") or record.get("time") or record.get("datetime"),
        "index": index,
    }
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return f"{table_name}:{digest}"


def _records_from_json(table_name: str, path: Path) -> tuple[list[tuple[str, dict[str, Any], int]], list[str]]:
    default = _json_default(path)
    raw = read_json(path, default)
    normalized = normalize_json_data(path, raw)
    issues = [f"{issue.severity}: {issue.path}: {issue.message}" for issue in normalized.issues]
    data = normalized.data
    records: list[tuple[str, dict[str, Any], int]] = []
    if isinstance(data, dict):
        for index, (source_key, value) in enumerate(sorted(data.items())):
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("username", source_key)
                records.append((_stable_record_key(table_name, source_key, item, index), item, index))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            if isinstance(value, dict):
                records.append((_stable_record_key(table_name, "", value, index), value, index))
    return records, issues


def _ensure_backup(repo_root: Path, database_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = repo_root / "backups" / "sqlite_migration" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for filename in JSON_TABLES.values():
        source = database_dir / filename
        if source.exists():
            shutil.copy2(source, backup_dir / filename)
    return backup_dir


def _latest_backup(repo_root: Path) -> Path | None:
    root = repo_root / "backups" / "sqlite_migration"
    if not root.exists():
        return None
    backups = sorted([item for item in root.iterdir() if item.is_dir()])
    return backups[-1] if backups else None


def _restore_backup(repo_root: Path, database_dir: Path, backup_dir: Path | None = None) -> dict[str, Any]:
    source_dir = backup_dir or _latest_backup(repo_root)
    if source_dir is None or not source_dir.exists():
        raise FileNotFoundError("No sqlite migration backup found.")
    restored = []
    for filename in JSON_TABLES.values():
        source = source_dir / filename
        if source.exists():
            database_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, database_dir / filename)
            restored.append(filename)
    return {"backup_dir": str(source_dir), "restored": restored, "count": len(restored)}


def _prepare_connection(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.executescript(SCHEMA_SQL)
    return conn


def migrate(repo_root: Path, database_dir: Path, sqlite_path: Path, *, dry_run: bool, backup: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "database_dir": str(database_dir),
        "sqlite_path": str(sqlite_path),
        "tables": {},
        "issues": {},
        "backup_dir": "",
    }
    collected: dict[str, list[tuple[str, dict[str, Any], int, str]]] = {}
    total = 0
    for table_name, filename in JSON_TABLES.items():
        source = database_dir / filename
        records, issues = _records_from_json(table_name, source)
        collected[table_name] = [(key, item, index, filename) for key, item, index in records]
        report["tables"][table_name] = {"source_file": filename, "records": len(records), "exists": source.exists()}
        if issues:
            report["issues"][table_name] = issues
        total += len(records)
    report["total_records"] = total
    if dry_run:
        return report

    if backup:
        report["backup_dir"] = str(_ensure_backup(repo_root, database_dir))
    sqlite_backup = sqlite_path.with_suffix(sqlite_path.suffix + ".bak")
    if sqlite_path.exists():
        shutil.copy2(sqlite_path, sqlite_backup)
    migrated_at = datetime.now().isoformat(timespec="seconds")
    try:
        with _prepare_connection(sqlite_path) as conn:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM runtime_records")
            for table_name, records in collected.items():
                for record_key, payload, index, filename in records:
                    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO runtime_records
                            (table_name, record_key, payload, source_file, source_index, payload_sha256, migrated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            table_name,
                            record_key,
                            payload_text,
                            filename,
                            index,
                            hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
                            migrated_at,
                        ),
                    )
            conn.execute("INSERT OR REPLACE INTO migration_metadata (key, value) VALUES (?, ?)", ("migrated_at", migrated_at))
            conn.execute("INSERT OR REPLACE INTO migration_metadata (key, value) VALUES (?, ?)", ("source_database_dir", str(database_dir)))
            conn.commit()
    except Exception:
        if sqlite_backup.exists():
            shutil.copy2(sqlite_backup, sqlite_path)
        raise
    finally:
        if sqlite_backup.exists():
            sqlite_backup.unlink()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate Rehab AI Monitor JSON data to SQLite.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--database-dir", default="", help="JSON database directory. Defaults to <repo-root>/database.")
    parser.add_argument("--sqlite-path", default="", help="Output SQLite path. Defaults to <database-dir>/rehab_monitor.sqlite3.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only; do not write SQLite.")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup JSON before apply.")
    parser.add_argument("--rollback", action="store_true", help="Restore JSON files from latest migration backup and exit.")
    parser.add_argument("--backup-dir", default="", help="Specific backup directory for rollback.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    database_dir = Path(args.database_dir).resolve() if args.database_dir else repo_root / "database"
    sqlite_path = Path(args.sqlite_path).resolve() if args.sqlite_path else database_dir / "rehab_monitor.sqlite3"
    try:
        if args.rollback:
            report = _restore_backup(repo_root, database_dir, Path(args.backup_dir).resolve() if args.backup_dir else None)
        else:
            report = migrate(repo_root, database_dir, sqlite_path, dry_run=args.dry_run, backup=not args.no_backup)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
