import threading
import time

from video.jobs import AnalysisJobRegistry, start_analysis_job


def test_start_analysis_job_rejects_duplicate_live_thread():
    registry = AnalysisJobRegistry(1)
    stop = threading.Event()
    first_started = threading.Event()

    def hold():
        first_started.set()
        stop.wait(2)

    result = start_analysis_job(
        registry=registry,
        video_path="video-a.mp4",
        video_name="video-a",
        target=hold,
    )
    assert result == {"started": True, "reason": ""}
    assert first_started.wait(1)

    duplicate = start_analysis_job(
        registry=registry,
        video_path="video-a.mp4",
        video_name="video-a",
        target=lambda: None,
    )
    assert duplicate == {"started": False, "reason": "already_running"}

    stop.set()
    registry.running_threads["video-a.mp4"].join(timeout=1)


def test_force_restart_sets_previous_cancel_flag():
    registry = AnalysisJobRegistry(1)
    stop = threading.Event()

    def hold():
        stop.wait(2)

    assert start_analysis_job(
        registry=registry,
        video_path="video-a.mp4",
        video_name="video-a",
        target=hold,
    )["started"]
    old_flag = registry.cancel_flags["video-a.mp4"]

    restarted = start_analysis_job(
        registry=registry,
        video_path="video-a.mp4",
        video_name="video-a",
        target=lambda: None,
        force_restart=True,
    )

    assert restarted == {"started": True, "reason": ""}
    assert old_flag.is_set()
    assert registry.cancel_flags["video-a.mp4"] is not old_flag
    stop.set()


def test_slot_pool_releases_dead_thread_holder():
    registry = AnalysisJobRegistry(
        1,
        load_progress_fn=lambda _: {
            "status": "processing",
            "heartbeat": time.time(),
            "video_name": "video-a",
        },
    )
    registry.running_threads["video-a.mp4"] = threading.Thread(target=lambda: None)

    assert registry.slots.try_acquire("video-a.mp4", timeout=0.2)
    assert registry.slots.try_acquire("video-b.mp4", timeout=0.3)
    assert "video-a.mp4" not in registry.slots._holders
