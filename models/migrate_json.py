"""Idempotent JSON schema migration CLI."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from models.schemas import SCHEMA_NORMALIZERS, normalize_json_data
from storage.json_store import read_json, write_json


def migrate_file(path: Path, backup_dir: Path | None = None, dry_run: bool = False):
    default = {} if path.name == "users.json" else []
    data = read_json(path, default)
    result = normalize_json_data(path, data)
    backup_path = None
    if result.changed and not dry_run:
        if path.exists():
            backup_target_dir = backup_dir or path.parent / "schema_backups"
            backup_target_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_target_dir / f"{path.name}.bak-{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(path, backup_path)
        write_json(path, result.data)
    return result, backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize JSON database files to the current schema.")
    parser.add_argument("--data-dir", default="database", help="Directory containing JSON database files.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    changed = 0
    for filename in SCHEMA_NORMALIZERS:
        path = data_dir / filename
        result, backup_path = migrate_file(path, dry_run=args.dry_run)
        if result.changed:
            changed += 1
            suffix = "dry-run" if args.dry_run else f"backup={backup_path}"
            print(f"{filename}: normalized ({suffix})")
        for issue in result.issues:
            print(f"{filename}: {issue.severity}: {issue.path}: {issue.message}")
    print(f"Schema migration complete. Files needing changes: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

