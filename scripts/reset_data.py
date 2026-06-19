import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.json_store import write_json


DB_FILES = [
    "doctor_evaluations.json",
    "video_list.json",
    "patient_symptoms.json",
    "schedules.json",
    "lich_su_tap_luyen.json",
]
MEDIA_FOLDERS = ["patient_uploads", "temp_frames"]


def repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "scripts" else script_dir


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_within_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(resolved), str(root_resolved)]) != str(root_resolved):
        raise ValueError(f"Refuse path outside repo root: {resolved}")
    return resolved


def backup_path(path: Path, backup_root: Path, root: Path, stamp: str) -> Path:
    rel = path.resolve().relative_to(root.resolve())
    return backup_root / stamp / rel


def backup_file(path: Path, backup_root: Path, root: Path, stamp: str) -> None:
    if not path.exists():
        return
    dst = backup_path(path, backup_root, root, stamp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def backup_directory(path: Path, backup_root: Path, root: Path, stamp: str) -> None:
    if not path.exists():
        return
    dst = backup_path(path, backup_root, root, stamp)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(path, dst)


def reset_data(dry_run: bool, yes: bool) -> int:
    root = repo_root()
    db_dir = ensure_within_root(root / "database", root)
    backup_root = ensure_within_root(root / "backups" / "reset_data", root)
    stamp = timestamp()

    print(f"Repo root : {root}")
    print(f"Mode      : {'DRY-RUN' if dry_run else 'WRITE'}")
    print(f"Confirmed : {'yes' if yes else 'no'}")

    if not dry_run and not yes:
        print("Refusing destructive reset without --yes. Use --dry-run to preview.")
        return 2

    targets = [db_dir / name for name in DB_FILES]
    folders = [root / name for name in MEDIA_FOLDERS]
    for target in targets + folders:
        ensure_within_root(target, root)

    for f_path in targets:
        print(f"Reset JSON: {f_path}")
        if dry_run:
            continue
        db_dir.mkdir(parents=True, exist_ok=True)
        backup_file(f_path, backup_root, root, stamp)
        if not write_json(f_path, [], indent=2):
            raise OSError(f"Failed to reset JSON file: {f_path}")

    for folder_path in folders:
        print(f"Reset folder: {folder_path}")
        if dry_run:
            continue
        backup_directory(folder_path, backup_root, root, stamp)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        print(f"Backup saved under: {backup_root / stamp}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset runtime JSON/media with backup and confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Preview targets without writing or deleting.")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive reset.")
    args = parser.parse_args()
    return reset_data(dry_run=args.dry_run, yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
