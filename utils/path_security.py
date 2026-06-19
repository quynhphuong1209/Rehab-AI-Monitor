"""Path containment helpers for user/cloud-controlled paths."""

from __future__ import annotations

import os
from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when a path cannot be safely resolved inside allowed roots."""


def normalize_relative_path(value: str | os.PathLike[str]) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        raise PathSecurityError("empty path")
    if "://" in text or text.startswith("//"):
        raise PathSecurityError("URL paths are not allowed")
    drive, _ = os.path.splitdrive(text)
    if drive or text.startswith("/"):
        raise PathSecurityError("absolute paths are not allowed")

    parts: list[str] = []
    for part in text.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise PathSecurityError("parent traversal is not allowed")
        if "\x00" in part:
            raise PathSecurityError("NUL byte is not allowed")
        parts.append(part)
    if not parts:
        raise PathSecurityError("empty path")
    return "/".join(parts)


def path_is_within(child: str | os.PathLike[str], parent: str | os.PathLike[str]) -> bool:
    try:
        child_real = Path(child).expanduser().resolve()
        parent_real = Path(parent).expanduser().resolve()
        return child_real == parent_real or parent_real in child_real.parents
    except (OSError, RuntimeError, ValueError):
        return False


def safe_data_path(
    path_value: str | os.PathLike[str],
    allowed_roots: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...],
    *,
    base_dir: str | os.PathLike[str] | None = None,
    allow_absolute: bool = False,
    must_exist: bool = False,
) -> str:
    if not allowed_roots:
        raise PathSecurityError("no allowed roots configured")

    raw = Path(str(path_value or ""))
    if raw.is_absolute():
        if not allow_absolute:
            raise PathSecurityError("absolute paths are not allowed")
        candidate = raw
    else:
        rel = normalize_relative_path(str(path_value or ""))
        candidate = Path(base_dir or ".") / rel

    try:
        resolved = candidate.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PathSecurityError(f"cannot resolve path: {exc}") from exc

    if must_exist and not resolved.exists():
        raise PathSecurityError("path does not exist")

    for root in allowed_roots:
        try:
            root_resolved = Path(root).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return str(resolved)

    raise PathSecurityError("path escapes allowed roots")


def relative_to_allowed_root(
    path_value: str | os.PathLike[str],
    allowed_roots: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...],
) -> str:
    try:
        resolved = Path(path_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PathSecurityError(f"cannot resolve path: {exc}") from exc

    for root in allowed_roots:
        try:
            root_resolved = Path(root).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved.relative_to(root_resolved).as_posix()

    raise PathSecurityError("path escapes allowed roots")
