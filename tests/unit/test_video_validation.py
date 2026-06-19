from video.validation import sanitize_filename, upload_video_magic_matches


def test_sanitize_filename_removes_path_and_unsafe_chars():
    assert sanitize_filename("../../Benh nhan 01?.MP4") == "Benh_nhan_01_.mp4"


def test_sanitize_filename_has_fallback_and_length_limit():
    assert sanitize_filename("///", fallback="video.mp4") == "video.mp4"
    assert len(sanitize_filename("a" * 260 + ".mp4")) <= 200


def test_video_magic_checks_known_headers():
    assert upload_video_magic_matches(".mp4", b"\x00\x00\x00\x18ftypmp42")
    assert upload_video_magic_matches(".avi", b"RIFFxxxxAVI ")
    assert not upload_video_magic_matches(".mp4", b"MZ executable")

