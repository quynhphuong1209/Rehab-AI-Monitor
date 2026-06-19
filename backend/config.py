"""Backend runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class BackendConfig:
    repo_root: Path
    database_dir: Path
    cors_origins: tuple[str, ...] = ()
    enable_ai_runner: bool = False
    ai_model_type: str = "MediaPipe Heavy"
    ai_min_confidence: float = 0.5
    ai_skip_step: int | None = 0
    ai_resize_width: int | None = 720
    ai_force_train_classifier: bool = False
    ai_enable_pose_classifier: bool = False
    ai_ffmpeg_threads: int = 2

    @classmethod
    def from_env(cls) -> "BackendConfig":
        repo_root = Path(os.getenv("REHAB_REPO_ROOT", Path.cwd())).resolve()
        database_dir = Path(os.getenv("REHAB_DATABASE_DIR", repo_root / "database")).resolve()
        origins_raw = os.getenv(
            "REHAB_BACKEND_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        )
        cors_origins = tuple(origin.strip().rstrip("/") for origin in origins_raw.split(",") if origin.strip())
        return cls(
            repo_root=repo_root,
            database_dir=database_dir,
            cors_origins=cors_origins,
            enable_ai_runner=_env_bool("REHAB_BACKEND_ENABLE_AI_RUNNER", False),
            ai_model_type=os.getenv("REHAB_BACKEND_AI_MODEL_TYPE", "MediaPipe Heavy").strip()
            or "MediaPipe Heavy",
            ai_min_confidence=max(0.1, min(0.95, _env_float("REHAB_BACKEND_AI_MIN_CONFIDENCE", 0.5))),
            ai_skip_step=_env_int("REHAB_BACKEND_AI_SKIP_STEP", 0),
            ai_resize_width=_env_int("REHAB_BACKEND_AI_RESIZE_WIDTH", 720),
            ai_force_train_classifier=_env_bool("REHAB_BACKEND_AI_FORCE_TRAIN_CLASSIFIER", False),
            ai_enable_pose_classifier=_env_bool("REHAB_BACKEND_AI_ENABLE_POSE_CLASSIFIER", False),
            ai_ffmpeg_threads=max(1, _env_int("REHAB_BACKEND_AI_FFMPEG_THREADS", _env_int("MAX_FFMPEG_THREADS", 2)) or 2),
        )

    @property
    def users_file(self) -> Path:
        return self.database_dir / "users.json"

    @property
    def videos_file(self) -> Path:
        return self.database_dir / "video_list.json"

    @property
    def upload_dir(self) -> Path:
        return self.repo_root / "patient_uploads"

    @property
    def processed_dir(self) -> Path:
        return self.repo_root / "processed_results"

    @property
    def evaluations_file(self) -> Path:
        return self.database_dir / "doctor_evaluations.json"

    @property
    def symptoms_file(self) -> Path:
        return self.database_dir / "patient_symptoms.json"

    @property
    def schedules_file(self) -> Path:
        return self.database_dir / "schedules.json"

    @property
    def research_file(self) -> Path:
        return self.database_dir / "research_data.json"

    @property
    def audit_log_file(self) -> Path:
        return self.database_dir / "audit_log.json"

    @property
    def session_state_file(self) -> Path:
        return self.database_dir / "session_state.json"
