"""Controlled startup helpers for the Streamlit app.

Keep import-time behavior small and predictable. Runtime setup that mutates the
process, Streamlit, or the filesystem should go through these idempotent helpers.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


class _SuppressFragmentWarning(logging.Filter):
    def filter(self, record):
        try:
            return "does not exist anymore" not in record.getMessage()
        except Exception:
            return True


class _FilterFragmentStream:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        try:
            if "does not exist anymore" in str(text):
                return 0
        except Exception:
            pass
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


@dataclass(frozen=True)
class StartupConfig:
    enable_stream_filters: bool = True
    enable_warnings_filter: bool = True
    create_dirs: bool = True
    enable_boot_sync: bool = True
    enable_temp_cleanup: bool = True
    enable_auto_transcode: bool = False
    enable_auto_resume: bool = False
    enable_resume_watcher: bool = False
    boot_sync_delay_seconds: float = 0.0
    auto_transcode_delay_seconds: float = 20.0
    auto_resume_delay_seconds: float = 15.0
    resume_watcher_interval_seconds: float = 120.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_startup_config() -> StartupConfig:
    """Read side-effect toggles from environment variables."""
    return StartupConfig(
        enable_stream_filters=_env_flag("REHAB_ENABLE_STREAM_FILTERS", True),
        enable_warnings_filter=_env_flag("REHAB_ENABLE_WARNINGS_FILTER", True),
        create_dirs=_env_flag("REHAB_CREATE_DIRS_ON_STARTUP", True),
        enable_boot_sync=_env_flag("REHAB_BOOT_SYNC_ENABLED", True),
        enable_temp_cleanup=_env_flag("REHAB_TEMP_CLEANUP_ON_BOOT", True),
        enable_auto_transcode=_env_flag("REHAB_AUTO_TRANSCODE_ON_BOOT", False),
        enable_auto_resume=_env_flag("REHAB_AUTO_RESUME_ON_BOOT", False),
        enable_resume_watcher=_env_flag("REHAB_AUTO_RESUME_WATCHER", False),
        boot_sync_delay_seconds=float(os.getenv("REHAB_BOOT_SYNC_DELAY_SECONDS", "0") or 0),
        auto_transcode_delay_seconds=float(os.getenv("REHAB_AUTO_TRANSCODE_DELAY_SECONDS", "20") or 20),
        auto_resume_delay_seconds=float(os.getenv("REHAB_AUTO_RESUME_DELAY_SECONDS", "15") or 15),
        resume_watcher_interval_seconds=float(os.getenv("REHAB_RESUME_WATCHER_INTERVAL_SECONDS", "120") or 120),
    )


_process_setup_lock = threading.Lock()
_process_setup_done = False
_logging_filter = _SuppressFragmentWarning()
_fragment_streams_wrapped = False


def get_compute_thread_count(cpu_count_fn: Callable[[], int | None] | None = None) -> int:
    """Calculate native compute threads without mutating process state."""
    if cpu_count_fn is None:
        cpu_count_fn = os.cpu_count
    try:
        cpu_total = cpu_count_fn() or 2
    except Exception:
        cpu_total = 2
    return max(1, cpu_total - 1)


def configure_process_environment(cpu_count_fn: Callable[[], int | None] | None = None) -> int:
    """Apply process env defaults once and return the compute thread count."""
    global _process_setup_done
    with _process_setup_lock:
        compute_threads = get_compute_thread_count(cpu_count_fn)
        if not _process_setup_done:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "0")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                os.environ.setdefault(name, str(compute_threads))
            _process_setup_done = True
        return compute_threads


def configure_logging_filters(*, wrap_streams: bool = True) -> None:
    """Suppress noisy Streamlit fragment lifecycle messages."""
    global _fragment_streams_wrapped
    try:
        import streamlit.runtime.app_session  # noqa: F401
    except Exception:
        pass
    try:
        logging.getLogger("streamlit.runtime.app_session").setLevel(logging.WARNING)
    except Exception:
        pass
    try:
        names = ["", "streamlit"] + [
            n
            for n in list(logging.root.manager.loggerDict)
            if isinstance(n, str) and n.startswith("streamlit")
        ]
        seen_handlers = set()
        for name in names:
            logger = logging.getLogger(name)
            try:
                logger.addFilter(_logging_filter)
            except Exception:
                pass
            for handler in list(getattr(logger, "handlers", []) or []):
                if id(handler) in seen_handlers:
                    continue
                seen_handlers.add(id(handler))
                try:
                    handler.addFilter(_logging_filter)
                except Exception:
                    pass
    except Exception:
        pass
    if wrap_streams and not _fragment_streams_wrapped:
        try:
            sys.stderr = _FilterFragmentStream(sys.stderr)
            sys.stdout = _FilterFragmentStream(sys.stdout)
            _fragment_streams_wrapped = True
        except Exception:
            pass


_streamlit_page_configured = False


def configure_streamlit_page(st) -> None:
    """Call st.set_page_config once per process."""
    global _streamlit_page_configured
    if _streamlit_page_configured:
        return
    st.set_page_config(
        page_title="Hệ thống giám sát tập PHCN từ xa - Đề tài NCKH",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _streamlit_page_configured = True


def has_streamlit_script_context() -> bool:
    """Return True when code is executing inside `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def configure_streamlit_page_if_running(st) -> bool:
    """Configure page early only for real Streamlit runtime executions."""
    if not has_streamlit_script_context():
        return False
    configure_streamlit_page(st)
    return True


def ensure_runtime_dirs(paths: Iterable[str | os.PathLike[str]]) -> None:
    for raw_path in paths:
        if not raw_path:
            continue
        try:
            Path(raw_path).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


_thread_once_lock = threading.Lock()
_started_thread_keys: set[str] = set()


def start_thread_once(key: str, target: Callable[[], None], *, delay_seconds: float = 0.0) -> bool:
    """Start a daemon thread at most once per process."""
    with _thread_once_lock:
        if key in _started_thread_keys:
            return False
        _started_thread_keys.add(key)

    def _runner():
        if delay_seconds > 0:
            import time

            time.sleep(delay_seconds)
        target()

    threading.Thread(target=_runner, daemon=True, name=f"rehab-{key}").start()
    return True


_app_startup_lock = threading.Lock()
_app_startup_done = False


def app_startup(
    *,
    st=None,
    config: StartupConfig | None = None,
    dirs: Iterable[str | os.PathLike[str]] = (),
    boot_sync_job: Callable[[], None] | None = None,
    cleanup_job: Callable[[], None] | None = None,
    auto_transcode_job: Callable[[], None] | None = None,
    auto_resume_job: Callable[[], None] | None = None,
    resume_watcher_job: Callable[[], None] | None = None,
) -> bool:
    """Run controlled app startup side effects once.

    Returns True when this call performed startup work, False when it was already
    completed in this process.
    """
    global _app_startup_done
    if config is None:
        config = load_startup_config()

    with _app_startup_lock:
        if _app_startup_done:
            return False
        _app_startup_done = True

    configure_process_environment()
    if config.enable_warnings_filter:
        warnings.filterwarnings("ignore")
    if config.enable_stream_filters:
        configure_logging_filters(wrap_streams=True)
    if st is not None:
        configure_streamlit_page(st)
        if config.enable_stream_filters:
            configure_logging_filters(wrap_streams=True)
    if config.create_dirs:
        ensure_runtime_dirs(dirs)
    if config.enable_boot_sync and boot_sync_job is not None:
        start_thread_once("boot-sync", boot_sync_job, delay_seconds=config.boot_sync_delay_seconds)
    if config.enable_temp_cleanup and cleanup_job is not None:
        cleanup_job()
    if config.enable_auto_transcode and auto_transcode_job is not None:
        start_thread_once(
            "auto-transcode",
            auto_transcode_job,
            delay_seconds=config.auto_transcode_delay_seconds,
        )
    if config.enable_auto_resume and auto_resume_job is not None:
        start_thread_once("auto-resume", auto_resume_job, delay_seconds=config.auto_resume_delay_seconds)
    if config.enable_resume_watcher and resume_watcher_job is not None:
        start_thread_once("resume-watcher", resume_watcher_job)
    return True
