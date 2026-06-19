"""JSON-backed backend repository.

This is the first backend boundary: API code reads data through this module
instead of importing Streamlit's `app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from storage.app_json import read_app_json

from backend.config import BackendConfig


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class JsonRepository:
    config: BackendConfig

    def users(self) -> dict[str, Any]:
        return _as_dict(read_app_json(self.config.users_file, default={}).data)

    def videos(self) -> list[dict[str, Any]]:
        return _as_list(read_app_json(self.config.videos_file, default=[]).data)

    def evaluations(self) -> list[dict[str, Any]]:
        return _as_list(read_app_json(self.config.evaluations_file, default=[]).data)

    def symptoms(self) -> list[dict[str, Any]]:
        return _as_list(read_app_json(self.config.symptoms_file, default=[]).data)

    def schedules(self) -> list[dict[str, Any]]:
        return _as_list(read_app_json(self.config.schedules_file, default=[]).data)

    def research_records(self) -> list[dict[str, Any]]:
        return _as_list(read_app_json(self.config.research_file, default=[]).data)

    def feedback(self) -> list[dict[str, Any]]:
        return _as_list(read_app_json(self.config.feedback_file, default=[]).data)

    def audit_log(self) -> list[dict[str, Any]]:
        return _as_list(read_app_json(self.config.audit_log_file, default=[]).data)
