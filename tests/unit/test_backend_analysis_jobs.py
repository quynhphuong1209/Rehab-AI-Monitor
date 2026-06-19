import json
import threading
import time
from types import SimpleNamespace

import pytest

from backend.ai_runner import BackendAIOptions, BackendMediaPipeAIRunner
from backend.analysis_jobs import AnalysisJobRequest, BackendAnalysisJobs


def _request(video_path):
    return AnalysisJobRequest(
        actor_username="researcher",
        username="patient01",
        video_name="clip.mov",
        video_path=str(video_path),
        exercise="Codman",
        options={},
    )


def test_validate_transcode_runner_marks_existing_h264_ready(tmp_path):
    video_path = tmp_path / "patient_uploads" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        if "-show_entries" in cmd:
            return SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264"}]}),
            stderr="",
        )

    jobs = BackendAnalysisJobs(repo_root=tmp_path, upload_dir=tmp_path / "patient_uploads", command_runner=runner)
    updates = []

    result = jobs._validate_transcode_runner(_request(video_path), lambda **kwargs: updates.append(kwargs))

    assert result["status"] == "ready_for_ai_worker"
    assert result["result"]["transcoded"] is False
    assert result["result"]["analysis_input_path"] == str(video_path)
    assert any(update["progress"] == 0.18 for update in updates)
    assert not any(cmd[0] == "ffmpeg" for cmd in calls)


def test_validate_transcode_runner_transcodes_non_h264_video(tmp_path):
    video_path = tmp_path / "patient_uploads" / "clip.mov"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "ffmpeg":
            output_path = cmd[-1]
            with open(output_path, "wb") as handle:
                handle.write(b"h264")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "-show_entries" in cmd:
            return SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264"} if cmd[-1].endswith("_f.mp4") else {"codec_type": "video", "codec_name": "mpeg4"}]}),
            stderr="",
        )

    jobs = BackendAnalysisJobs(repo_root=tmp_path, upload_dir=tmp_path / "patient_uploads", command_runner=runner)
    updates = []

    result = jobs._validate_transcode_runner(_request(video_path), lambda **kwargs: updates.append(kwargs))

    assert result["status"] == "ready_for_ai_worker"
    assert result["result"]["transcoded"] is True
    assert result["result"]["analysis_input_path"].endswith("clip_f.mp4")
    assert (tmp_path / "patient_uploads" / "clip_f.mp4").read_bytes() == b"h264"
    assert any(cmd[0] == "ffmpeg" for cmd in calls)
    assert any(update["progress"] == 0.34 for update in updates)


def test_start_runs_injected_ai_runner_after_video_is_ready(tmp_path):
    video_path = tmp_path / "patient_uploads" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    handled = []

    def command_runner(cmd, **kwargs):
        if "-show_entries" in cmd:
            return SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264"}]}),
            stderr="",
        )

    def ai_runner(request, analysis_input_path, progress):
        progress(status="processing", progress=0.66, status_msg="AI đang đọc khung hình.")
        return {
            "status": "success",
            "progress": 1.0,
            "status_msg": "AI đã phân tích xong.",
            "result": {
                "processed_path": "processed_results/processed_clip.mp4",
                "metrics": {"do_chinh_xac": 91.25},
                "df_path": "processed_results/clip_data.csv",
                "analysis_input_path": analysis_input_path,
            },
        }

    jobs = BackendAnalysisJobs(
        repo_root=tmp_path,
        upload_dir=tmp_path / "patient_uploads",
        command_runner=command_runner,
        ai_runner=ai_runner,
        result_handler=lambda request, result: handled.append((request, result)),
    )

    start = jobs.start(_request(video_path))

    assert start["started"] is True
    deadline = time.time() + 2
    while jobs.is_running(str(video_path)) and time.time() < deadline:
        time.sleep(0.01)
    progress = jobs.read_progress(str(video_path))
    assert progress["status"] == "success"
    assert progress["progress"] == 1.0
    assert progress["result"]["metrics"]["do_chinh_xac"] == 91.25
    assert handled
    assert handled[0][0].options["analysis_input_path"] == str(video_path)


def test_cancel_marks_running_job_and_writes_history(tmp_path):
    video_path = tmp_path / "patient_uploads" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    runner_started = threading.Event()
    release_runner = threading.Event()

    def slow_runner(request, progress):
        runner_started.set()
        progress(status="processing", progress=0.12, status_msg="Đang chờ worker.")
        release_runner.wait(1)
        progress(status="processing", progress=0.50, status_msg="Không nên tới đây.")
        return {"status": "ready_for_ai_worker", "progress": 0.55, "result": {"analysis_input_path": str(video_path)}}

    jobs = BackendAnalysisJobs(
        repo_root=tmp_path,
        upload_dir=tmp_path / "patient_uploads",
        runner=slow_runner,
    )

    start = jobs.start(_request(video_path))
    assert start["started"] is True
    assert runner_started.wait(1)

    canceled = jobs.cancel(_request(video_path), canceled_by="researcher")
    release_runner.set()
    deadline = time.time() + 2
    while jobs.is_running(str(video_path)) and time.time() < deadline:
        time.sleep(0.01)

    assert canceled["ok"] is True
    progress = jobs.read_progress(str(video_path))
    assert progress["status"] == "canceled"
    assert progress["job_meta"]["canceled_by"] == "researcher"
    history = jobs.read_history(str(video_path))
    assert len(history) == 1
    assert history[0]["run_id"] == progress["run_id"]
    assert history[0]["status"] == "canceled"


def test_rerun_creates_new_run_id_and_preserves_public_options(tmp_path):
    video_path = tmp_path / "patient_uploads" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")

    def command_runner(cmd, **kwargs):
        if "-show_entries" in cmd:
            return SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264"}]}),
            stderr="",
        )

    jobs = BackendAnalysisJobs(
        repo_root=tmp_path,
        upload_dir=tmp_path / "patient_uploads",
        command_runner=command_runner,
    )
    first = _request(video_path)
    second = AnalysisJobRequest(
        actor_username=first.actor_username,
        username=first.username,
        video_name=first.video_name,
        video_path=first.video_path,
        exercise=first.exercise,
        options={
            "model_type": "MediaPipe Lite",
            "min_confidence": 0.7,
            "skip_step": 2,
            "resize_width": 480,
            "media_path": str(video_path),
        },
        action="rerun",
    )

    assert jobs.start(first)["started"] is True
    deadline = time.time() + 2
    while jobs.is_running(str(video_path)) and time.time() < deadline:
        time.sleep(0.01)
    first_run_id = jobs.read_progress(str(video_path))["run_id"]

    assert jobs.start(second)["started"] is True
    deadline = time.time() + 2
    while jobs.is_running(str(video_path)) and time.time() < deadline:
        time.sleep(0.01)
    progress = jobs.read_progress(str(video_path))
    history = jobs.read_history(str(video_path))

    assert progress["run_id"] != first_run_id
    assert progress["job_meta"]["action"] == "rerun"
    assert progress["job_meta"]["options"] == {
        "model_type": "MediaPipe Lite",
        "min_confidence": 0.7,
        "skip_step": 2,
        "resize_width": 480,
    }
    assert len(history) == 2


def test_backend_mediapipe_ai_runner_wraps_processing_result(tmp_path):
    pd = pytest.importorskip("pandas")
    assert pd is not None

    input_path = tmp_path / "patient_uploads" / "clip.mp4"
    input_path.parent.mkdir()
    input_path.write_bytes(b"video")
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    updates = []

    def fake_processing(deps, **kwargs):
        assert deps.DB_DIR == str(tmp_path / "database")
        assert kwargs["duong_dan_video"] == str(input_path)
        output_path = processed_dir / "processed_1_f.mp4"
        output_path.write_bytes(b"processed")
        kwargs["callback"](0.5, frame_count=1, total_frames=3)
        rows = [
            {
                "frame": 1,
                "goc_vai": 88.0,
                "goc_khuyu": 168.0,
                "goc_vai_phai": 88.0,
                "goc_khuyu_phai": 168.0,
                "dung": True,
                "gan_dung": False,
                "vai_dung": True,
                "khuyu_dung": True,
                "vai_chuan": 90.0,
                "khuyu_chuan": 170.0,
            },
            {
                "frame": 2,
                "goc_vai": 94.0,
                "goc_khuyu": 166.0,
                "goc_vai_phai": 94.0,
                "goc_khuyu_phai": 166.0,
                "dung": True,
                "gan_dung": False,
                "vai_dung": True,
                "khuyu_dung": True,
                "vai_chuan": 90.0,
                "khuyu_chuan": 170.0,
            },
            {
                "frame": 3,
                "goc_vai": 130.0,
                "goc_khuyu": 130.0,
                "goc_vai_phai": 130.0,
                "goc_khuyu_phai": 130.0,
                "dung": False,
                "gan_dung": False,
                "vai_dung": False,
                "khuyu_dung": False,
                "vai_chuan": 90.0,
                "khuyu_chuan": 170.0,
            },
        ]
        return (
            str(output_path),
            "codman",
            None,
            rows,
            3,
            3,
            str(processed_dir / "processed_1_frames"),
            str(processed_dir / "processed_1_f_frames.zip"),
            [],
            {},
            str(processed_dir / "f_1.json"),
            ["test-warning"],
        )

    runner = BackendMediaPipeAIRunner(
        repo_root=tmp_path,
        database_dir=tmp_path / "database",
        processed_dir=processed_dir,
        options=BackendAIOptions(enable_pose_classifier=False),
        processing_fn=fake_processing,
    )

    result = runner(
        _request(input_path),
        str(input_path),
        lambda **kwargs: updates.append(kwargs),
    )

    assert result["status"] == "success"
    payload = result["result"]
    assert payload["processed_path"].endswith("processed_1_f.mp4")
    assert payload["df_path"].endswith("processed_1_f_data.csv")
    assert (processed_dir / "processed_1_f_data.csv").exists()
    assert payload["metrics"]["do_chinh_xac"] > 0
    assert payload["warnings"] == ["test-warning"]
    assert any(update["progress"] > 0.42 for update in updates)
