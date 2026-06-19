import zipfile
from types import SimpleNamespace

import pytest

from video.io import (
    build_async_h264_command,
    build_background_upload_h264_command,
    build_cut_segment_command,
    build_ffmpeg_version_command,
    build_frame_extract_command,
    build_h264_transcode_command,
    build_mov_to_mp4_command,
    build_upload_h264_command,
    ffprobe_video_codecs,
    ffprobe_video_duration_text,
    ffprobe_video_has_readable_duration,
    final_h264_path,
    is_non_playable_video_artifact,
    mov_to_mp4_path,
    safe_extract_frames_zip,
    temp_h264_path,
    video_fallback_paths_for,
)


def _make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)


def test_safe_extract_frames_zip_allows_image_basenames(tmp_path):
    zip_path = tmp_path / "frames.zip"
    out_dir = tmp_path / "frames"
    _make_zip(zip_path, [("frame_001.jpg", b"jpg"), ("frame_002.png", b"png")])

    summary = safe_extract_frames_zip(zip_path, out_dir)

    assert summary.count == 2
    assert summary.total_bytes == 6
    assert (out_dir / "frame_001.jpg").read_bytes() == b"jpg"
    assert (out_dir / "frame_002.png").read_bytes() == b"png"


@pytest.mark.parametrize("entry_name", ["../frame.jpg", "nested/frame.jpg", r"nested\frame.jpg"])
def test_safe_extract_frames_zip_rejects_zip_slip_and_nested_paths(tmp_path, entry_name):
    zip_path = tmp_path / "bad.zip"
    _make_zip(zip_path, [(entry_name, b"x")])

    with pytest.raises(ValueError, match="đường dẫn frame không hợp lệ"):
        safe_extract_frames_zip(zip_path, tmp_path / "frames")


def test_safe_extract_frames_zip_rejects_non_image_entries(tmp_path):
    zip_path = tmp_path / "bad.zip"
    _make_zip(zip_path, [("payload.txt", b"x")])

    with pytest.raises(ValueError, match="không phải ảnh frame"):
        safe_extract_frames_zip(zip_path, tmp_path / "frames")


def test_safe_extract_frames_zip_enforces_size_quota(tmp_path):
    zip_path = tmp_path / "bad.zip"
    _make_zip(zip_path, [("frame.jpg", b"12345")])

    with pytest.raises(ValueError, match="vượt quá dung lượng"):
        safe_extract_frames_zip(zip_path, tmp_path / "frames", max_entry_bytes=4)

    with pytest.raises(ValueError, match="tổng dung lượng"):
        safe_extract_frames_zip(zip_path, tmp_path / "frames", max_total_bytes=4)


def test_video_path_helpers_reject_non_playable_artifacts():
    assert is_non_playable_video_artifact("processed_results/f_123.json")
    assert is_non_playable_video_artifact("processed_results/processed_123_frames.mp4")
    assert not is_non_playable_video_artifact("processed_results/processed_123.mp4")


def test_final_h264_path_preserves_existing_h264_suffix():
    assert final_h264_path("clip.mov") == "clip_f.mp4"
    assert final_h264_path("clip_f.mp4") == "clip_f.mp4"
    assert final_h264_path("frames.zip") == ""


def test_video_fallback_paths_for_prefers_h264_then_original():
    assert video_fallback_paths_for("clip.mov") == ["clip_f.mp4", "clip.mov"]
    assert video_fallback_paths_for("clip_f.mp4") == ["clip_f.mp4", "clip.mp4"]
    assert video_fallback_paths_for("clip_ffmp.mp4") == ["clip_ffmp.mp4", "clip.mp4"]
    assert video_fallback_paths_for("frames.zip") == []


def test_video_fallback_paths_for_uses_optional_resolver():
    assert video_fallback_paths_for("cloud/path", local_frame_path_resolver=lambda _: "clip.mp4") == [
        "clip_f.mp4",
        "clip.mp4",
    ]


def test_ffprobe_video_has_readable_duration_uses_runner():
    def runner(cmd, **kwargs):
        assert cmd[0] == "ffprobe"
        return SimpleNamespace(returncode=0, stdout="12.34\n", stderr="")

    assert ffprobe_video_has_readable_duration("clip.mp4", runner=runner)


def test_ffprobe_video_has_readable_duration_rejects_bad_output():
    assert not ffprobe_video_has_readable_duration(
        "clip.mp4",
        runner=lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout="not-a-number", stderr=""),
    )
    assert not ffprobe_video_has_readable_duration(
        "clip.mp4",
        runner=lambda cmd, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad"),
    )


def test_ffprobe_video_codecs_parses_video_and_audio_streams():
    stdout = '{"streams":[{"codec_type":"video","codec_name":"h264"},{"codec_type":"audio","codec_name":"aac"}]}'

    codecs = ffprobe_video_codecs(
        "clip.mp4",
        runner=lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    assert codecs == ("h264", "aac")


def test_ffprobe_video_codecs_handles_invalid_json():
    assert ffprobe_video_codecs(
        "clip.mp4",
        runner=lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout="{bad", stderr=""),
    ) == (None, None)


def test_temp_h264_path_uses_ftmp_suffix():
    assert temp_h264_path("clip_f.mp4") == "clip_ftmp.mp4"
    assert temp_h264_path("clip.mp4") == "clip.mp4.ftmp.mp4"


def test_build_h264_transcode_command_without_audio():
    cmd = build_h264_transcode_command("in.mov", "out_ftmp.mp4", audio_exists=False, ffmpeg_threads=3)

    assert cmd[:4] == ["ffmpeg", "-y", "-i", "in.mov"]
    assert "-an" in cmd
    assert "-map" in cmd
    assert cmd[-1] == "out_ftmp.mp4"
    assert cmd[cmd.index("-threads") + 1] == "3"


def test_build_h264_transcode_command_with_external_audio():
    cmd = build_h264_transcode_command(
        "in.mov",
        "out_ftmp.mp4",
        audio_path="audio.wav",
        audio_exists=True,
    )

    assert cmd[4:6] == ["-i", "audio.wav"]
    assert "-shortest" in cmd
    assert "-an" not in cmd


def test_build_upload_h264_command_toggles_audio():
    with_audio = build_upload_h264_command("in.mov", "out.mp4", has_audio=True, ffmpeg_threads=2)
    no_audio = build_upload_h264_command("in.mov", "out.mp4", has_audio=False, ffmpeg_threads=2)

    assert "-c:a" in with_audio
    assert "-an" in no_audio
    assert with_audio[-1] == "out.mp4"


def test_build_async_h264_command_keeps_faststart_and_mp4_format():
    cmd = build_async_h264_command("in.mov", "out_ftmp.mp4", has_audio=True, ffmpeg_threads=4)

    assert cmd[:4] == ["ffmpeg", "-y", "-i", "in.mov"]
    assert cmd[cmd.index("-vf") + 1] == r"scale=-2:min(480\,ih)"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert cmd[cmd.index("-threads") + 1] == "4"
    assert cmd[-3:] == ["-f", "mp4", "out_ftmp.mp4"]
    assert "-c:a" in cmd


def test_build_async_h264_command_can_disable_audio():
    cmd = build_async_h264_command("in.mov", "out_ftmp.mp4", has_audio=False)

    assert "-an" in cmd
    assert "-c:a" not in cmd


def test_build_background_upload_h264_command_uses_720p_profile():
    cmd = build_background_upload_h264_command("in.mov", "out.mp4", ffmpeg_threads=5)

    assert cmd[:4] == ["ffmpeg", "-y", "-i", "in.mov"]
    assert cmd[cmd.index("-vf") + 1] == "scale=-2:720"
    assert cmd[cmd.index("-threads") + 1] == "5"
    assert cmd[-1] == "out.mp4"


def test_mov_to_mp4_path_and_command():
    assert mov_to_mp4_path("clip.MOV") == "clip.mp4"
    assert build_ffmpeg_version_command() == ["ffmpeg", "-version"]

    cmd = build_mov_to_mp4_command("clip.mov")

    assert cmd[:3] == ["ffmpeg", "-i", "clip.mov"]
    assert cmd[-1] == "clip.mp4"
    assert cmd[cmd.index("-crf") + 1] == "23"


def test_build_cut_segment_command_formats_times():
    cmd = build_cut_segment_command("in.mp4", "g1.mp4", start=1 / 3, duration=2 / 3)

    assert cmd[:6] == ["ffmpeg", "-y", "-ss", "0.333", "-t", "0.667"]
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert cmd[-1] == "g1.mp4"


def test_build_frame_extract_command_formats_timestamp_and_quality():
    cmd = build_frame_extract_command("in.mp4", "frame.jpg", timestamp=1 / 30, quality=7)

    assert cmd == [
        "ffmpeg",
        "-y",
        "-ss",
        "0.0333",
        "-i",
        "in.mp4",
        "-vframes",
        "1",
        "-q:v",
        "7",
        "frame.jpg",
    ]


def test_ffprobe_video_duration_text_reports_duration_or_error():
    ok = ffprobe_video_duration_text(
        "clip.mp4",
        runner=lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout="9.5\n", stderr=""),
    )
    bad = ffprobe_video_duration_text(
        "clip.mp4",
        runner=lambda cmd, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )

    assert ok == ("9.5", None)
    assert bad == ("", "boom")
