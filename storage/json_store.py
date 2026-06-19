"""Locked JSON read/write helpers with atomic replace.

The app is still JSON-backed for now, so this module centralizes the bits that
matter most: per-file locks, temp-file writes, optional backups, and update
callbacks that avoid read-modify-write races inside a single process.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


def _lock_for(path: str | os.PathLike[str]) -> threading.RLock:
    key = str(Path(path).resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def _clone_default(default: Any) -> Any:
    return copy.deepcopy(default)


def _backup_path(path: Path, backup_dir: Path | None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target_dir = backup_dir or path.parent
    return target_dir / f"{path.name}.bak-{stamp}-{os.getpid()}"


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return _clone_default(default)

    with _lock_for(file_path):
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            logger.exception("invalid JSON in %s", file_path)
            return _clone_default(default)
        except OSError:
            logger.exception("failed to read JSON %s", file_path)
            return _clone_default(default)


def write_json(
    path: str | os.PathLike[str],
    data: Any,
    *,
    backup: bool = False,
    backup_dir: str | os.PathLike[str] | None = None,
    indent: int = 4,
) -> bool:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with _lock_for(file_path):
        tmp_name = None
        try:
            if backup and file_path.exists():
                target_backup_dir = Path(backup_dir) if backup_dir else None
                if target_backup_dir:
                    target_backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, _backup_path(file_path, target_backup_dir))

            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(file_path.parent),
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                json.dump(data, handle, ensure_ascii=False, indent=indent)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, file_path)
            return True
        except OSError:
            logger.exception("failed to write JSON %s", file_path)
            return False
        finally:
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    logger.warning("failed to remove temp JSON %s", tmp_name, exc_info=True)


def update_json(
    path: str | os.PathLike[str],
    update_fn: Callable[[Any], Any],
    *,
    default: Any = None,
    backup: bool = False,
    backup_dir: str | os.PathLike[str] | None = None,
) -> Any:
    file_path = Path(path)
    with _lock_for(file_path):
        current = read_json(file_path, default)
        updated = update_fn(current)
        if updated is None:
            updated = current
        if not write_json(file_path, updated, backup=backup, backup_dir=backup_dir):
            raise OSError(f"failed to update JSON: {file_path}")
        return updated

