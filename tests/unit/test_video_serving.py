from video.serving import (
    allowed_media_file_path,
    build_video_media_url,
    is_allowed_video_origin,
    media_token_from_request_path,
    register_media_token,
    resolve_media_token,
    video_media_allowed_roots,
)


def test_video_media_allowed_roots_only_returns_existing_dirs(tmp_path):
    upload_root = tmp_path / "patient_uploads"
    processed_root = tmp_path / "processed_results"
    upload_root.mkdir()

    roots = video_media_allowed_roots(data_dir=tmp_path, processed_dir=processed_root)

    assert roots == {"uploads": str(upload_root.resolve())}


def test_allowed_media_file_path_requires_video_inside_allowed_root(tmp_path):
    root = tmp_path / "patient_uploads"
    root.mkdir()
    video = root / "clip.mp4"
    video.write_bytes(b"video")
    non_video = root / "clip.txt"
    non_video.write_text("nope", encoding="utf-8")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")

    roots = {"uploads": str(root.resolve())}

    assert allowed_media_file_path(video, roots) == str(video.resolve())
    assert allowed_media_file_path(non_video, roots) is None
    assert allowed_media_file_path(outside, roots) is None


def test_media_token_register_resolve_and_expire(tmp_path):
    root = tmp_path / "patient_uploads"
    root.mkdir()
    video = root / "clip.mp4"
    video.write_bytes(b"video")
    roots = {"uploads": str(root.resolve())}
    tokens = {"old": {"path": str(video), "expires_at": 9.0}}

    token = register_media_token(
        tokens,
        video,
        roots,
        ttl_seconds=5,
        now=10.0,
        token_factory=lambda n: f"tok{n}",
    )

    assert token == "tok32"
    assert "old" not in tokens
    assert resolve_media_token(tokens, token, roots, now=11.0) == str(video.resolve())
    assert resolve_media_token(tokens, token, roots, now=16.0) is None


def test_media_token_rejects_path_after_file_moves_outside_allowed_roots(tmp_path):
    root = tmp_path / "patient_uploads"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    tokens = {"tok": {"path": str(outside), "expires_at": 99.0}}

    assert resolve_media_token(tokens, "tok", {"uploads": str(root.resolve())}, now=10.0) is None
    assert "tok" not in tokens


def test_is_allowed_video_origin_allows_only_local_http_hosts():
    assert is_allowed_video_origin("http://127.0.0.1:8765")
    assert is_allowed_video_origin("https://localhost:8765")
    assert not is_allowed_video_origin("https://example.com")
    assert not is_allowed_video_origin("file://127.0.0.1/video.mp4")


def test_media_token_from_request_path_validates_shape_and_filename():
    assert media_token_from_request_path("/_media/abc/clip.mp4") == "abc"
    assert media_token_from_request_path("/_media/abc/nested/clip.mp4") == "abc"
    assert media_token_from_request_path("/media/abc/clip.mp4") is None
    assert media_token_from_request_path("/_media/abc/../clip.mp4") is None


def test_build_video_media_url_quotes_filename():
    assert build_video_media_url(8765, "tok", "a clip.mp4") == "http://127.0.0.1:8765/_media/tok/a%20clip.mp4"
    assert build_video_media_url(None, "tok", "clip.mp4") is None
