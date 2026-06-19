"""Schema-aware JSON helpers for app runtime files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from models.schemas import ValidationIssue, normalize_json_data
from storage.json_store import read_json, update_json, write_json


@dataclass(frozen=True)
class AppJsonResult:
    data: Any
    issues: list[ValidationIssue]
    changed: bool = False
    ok: bool = True


def default_for_json_path(file_path: str | Path) -> Any:
    # Preserve the legacy app behavior while centralizing the rule.
    return {} if "users" in str(file_path) else []


def normalize_app_json(file_path: str | Path, data: Any) -> AppJsonResult:
    result = normalize_json_data(file_path, data)
    return AppJsonResult(result.data, result.issues, result.changed, True)


def read_app_json(file_path: str | Path, default: Any = None) -> AppJsonResult:
    if default is None:
        default = default_for_json_path(file_path)
    data = read_json(file_path, default)
    return normalize_app_json(file_path, data)


def write_app_json(file_path: str | Path, data: Any) -> AppJsonResult:
    result = normalize_app_json(file_path, data)
    ok = write_json(file_path, result.data)
    return AppJsonResult(result.data, result.issues, result.changed, ok)


def update_app_json(
    file_path: str | Path,
    update_fn: Callable[[Any], Any],
    *,
    default: Any = None,
) -> Any:
    if default is None:
        default = default_for_json_path(file_path)

    def _normalized_update(current: Any) -> Any:
        updated = update_fn(current)
        if updated is None:
            updated = current
        return normalize_app_json(file_path, updated).data

    return update_json(file_path, _normalized_update, default=default)


def format_schema_issue_lines(file_path: str | Path, issues: list[ValidationIssue]) -> list[str]:
    name = Path(file_path).name
    return [f"[Schema] {name}: {issue.severity}: {issue.path}: {issue.message}" for issue in issues]
