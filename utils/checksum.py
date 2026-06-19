"""SHA-256 sidecar helpers for local binary artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import IO


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: IO[bytes], *, chunk_size: int = 1024 * 1024) -> str:
    pos = handle.tell()
    try:
        handle.seek(0)
        digest = hashlib.sha256()
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        handle.seek(pos)


def checksum_sidecar_path(path: str | os.PathLike[str]) -> str:
    return f"{path}.sha256"


def write_sha256_sidecar(path: str | os.PathLike[str]) -> str:
    file_path = Path(path)
    digest = sha256_file(file_path)
    sidecar = Path(checksum_sidecar_path(file_path))
    sidecar.write_text(f"{digest}  {file_path.name}\n", encoding="utf-8")
    return str(sidecar)


def read_sha256_sidecar(path: str | os.PathLike[str]) -> str | None:
    sidecar = Path(checksum_sidecar_path(path))
    if not sidecar.exists():
        return None
    text = sidecar.read_text(encoding="utf-8").strip()
    if not text:
        return None
    digest = text.split()[0].lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return None
    return digest


def verify_sha256_sidecar(path: str | os.PathLike[str], *, required: bool = True) -> bool:
    expected = read_sha256_sidecar(path)
    if not expected:
        return not required
    if not Path(path).exists():
        return False
    return sha256_file(path).lower() == expected
