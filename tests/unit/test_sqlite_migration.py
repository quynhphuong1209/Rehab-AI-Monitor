import json
import sqlite3

from scripts.migrate_json_to_sqlite import main, migrate


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_sqlite_migration_dry_run_reports_counts_without_writing(tmp_path):
    db = tmp_path / "database"
    _write_json(db / "users.json", {"patient01": {"role": "Bệnh nhân", "full_name": "Patient One"}})
    _write_json(db / "video_list.json", [{"username": "patient01", "video_name": "clip.mp4"}])

    report = migrate(tmp_path, db, db / "rehab_monitor.sqlite3", dry_run=True, backup=True)

    assert report["dry_run"] is True
    assert report["tables"]["users"]["records"] == 1
    assert report["tables"]["videos"]["records"] == 1
    assert report["total_records"] == 2
    assert not (db / "rehab_monitor.sqlite3").exists()


def test_sqlite_migration_apply_creates_backup_and_can_rollback(tmp_path):
    db = tmp_path / "database"
    sqlite_path = db / "rehab_monitor.sqlite3"
    users_path = db / "users.json"
    _write_json(users_path, {"patient01": {"role": "Bệnh nhân", "full_name": "Patient One"}})
    _write_json(db / "patient_symptoms.json", [{"username": "patient01", "symptoms": "pain"}])

    report = migrate(tmp_path, db, sqlite_path, dry_run=False, backup=True)

    assert sqlite_path.exists()
    assert report["backup_dir"]
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute("SELECT table_name, COUNT(*) FROM runtime_records GROUP BY table_name").fetchall()
    assert ("users", 1) in rows
    assert ("symptoms", 1) in rows

    _write_json(users_path, {"changed": {"role": "Bệnh nhân"}})
    result = main(["--repo-root", str(tmp_path), "--rollback"])

    assert result == 0
    restored = json.loads(users_path.read_text(encoding="utf-8"))
    assert "patient01" in restored
