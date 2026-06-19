import pytest

from cloud.hf_sync import (
    HfPathPolicy,
    dataset_rel_path_from_local,
    dataset_file_exists,
    download_dataset_file,
    download_dataset_file_bytes,
    download_dataset_file_with_progress,
    download_dataset_file_via_http,
    ensure_dataset_repo,
    hf_auth_headers,
    hf_dataset_file_url,
    hf_download_target_for_rel_path,
    hf_download_status_error,
    hf_min_size_for_path,
    hf_repo_info,
    hf_token_fingerprint,
    hf_upload_rel_path_for_local,
    hf_verify_dataset_status_message,
    is_hf_auth_error,
    is_hf_library_error,
    is_hf_not_found_error,
    list_dataset_files,
    upload_dataset_file,
    verify_dataset_via_http,
)
from utils.path_security import PathSecurityError


class _FakeResponse:
    def __init__(self, status_code=200, chunks=None, headers=None, raise_error=None):
        self.status_code = status_code
        self._chunks = chunks if chunks is not None else [b"{}"]
        self.headers = headers if headers is not None else {"content-length": str(sum(len(c) for c in self._chunks))}
        self._raise_error = raise_error
        self.content = b"".join(self._chunks)

    def iter_content(self, chunk_size=1):
        yield from self._chunks

    def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error


def _policy(tmp_path):
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "patient_uploads"
    processed_dir = data_dir / "processed_results"
    db_dir = tmp_path / "database"
    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    db_dir.mkdir()
    return HfPathPolicy(
        data_dir=str(data_dir),
        upload_dir=str(upload_dir),
        processed_dir=str(processed_dir),
        db_dir=str(db_dir),
    )


def test_dataset_rel_path_from_local_handles_media_paths(tmp_path):
    policy = _policy(tmp_path)
    local = tmp_path / "data" / "patient_uploads" / "clip.mp4"

    assert dataset_rel_path_from_local(local, policy) == "patient_uploads/clip.mp4"


def test_hf_download_target_allows_json_and_media(tmp_path):
    policy = _policy(tmp_path)

    json_target, json_dir, json_rel = hf_download_target_for_rel_path("video_list.json", policy)
    media_target, media_dir, media_rel = hf_download_target_for_rel_path("processed_results/out.mp4", policy)

    assert json_target.endswith("database\\video_list.json") or json_target.endswith("database/video_list.json")
    assert json_dir == policy.db_dir
    assert json_rel == "video_list.json"
    assert media_target.endswith("data\\processed_results\\out.mp4") or media_target.endswith("data/processed_results/out.mp4")
    assert media_dir == policy.data_dir
    assert media_rel == "processed_results/out.mp4"


def test_hf_upload_rel_path_rejects_users_json_and_outside_paths(tmp_path):
    policy = _policy(tmp_path)
    users = tmp_path / "database" / "users.json"
    users.write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(PathSecurityError):
        hf_upload_rel_path_for_local(users, policy)
    with pytest.raises(PathSecurityError):
        hf_upload_rel_path_for_local(outside, policy)


def test_hf_upload_rel_path_allows_model_artifact(tmp_path):
    policy = _policy(tmp_path)
    model = tmp_path / "database" / "pose_classifier.pkl.sha256"
    model.write_text("abc", encoding="utf-8")

    assert hf_upload_rel_path_for_local(model, policy) == "pose_classifier.pkl.sha256"


def test_hf_min_size_and_fingerprint_are_stable():
    assert hf_min_size_for_path("metrics.csv") == 80
    assert hf_min_size_for_path("video_list.json") == 2
    assert hf_min_size_for_path("clip.mp4") == 5 * 1024
    assert hf_token_fingerprint("token", "owner/dataset") == hf_token_fingerprint("token", "owner/dataset")


def test_hf_url_headers_and_status_messages():
    assert hf_auth_headers("abc") == {"Authorization": "Bearer abc"}
    assert hf_dataset_file_url("owner/dataset", "processed_results/a b.mp4").endswith(
        "/processed_results/a%20b.mp4"
    )
    assert "không có quyền" in hf_verify_dataset_status_message(403, "owner/dataset")
    assert "Không tìm thấy" in hf_verify_dataset_status_message(404, "owner/dataset")
    assert hf_verify_dataset_status_message(200, "owner/dataset") is None
    assert hf_download_status_error(401, "video_list.json") == "Token không có quyền tải file từ Dataset."
    assert hf_download_status_error(404, "video_list.json") == "Chưa có trên Dataset: `video_list.json`"


def test_hf_error_classifiers():
    assert is_hf_library_error("ImportError: cannot import name HfApi")
    assert is_hf_auth_error("403 forbidden")
    assert is_hf_not_found_error("Entry not found")


def test_hf_repo_info_and_create_use_injected_api():
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("init", token))

        def repo_info(self, **kwargs):
            calls.append(("repo_info", kwargs))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

    assert hf_repo_info("token", "owner/dataset", api_factory=FakeApi) == (True, None)
    assert ensure_dataset_repo("token", "owner/dataset", api_factory=FakeApi) == (True, None)
    assert ("repo_info", {"repo_id": "owner/dataset", "repo_type": "dataset"}) in calls
    assert (
        "create_repo",
        {"repo_id": "owner/dataset", "repo_type": "dataset", "private": True, "exist_ok": True},
    ) in calls


def test_list_dataset_files_uses_injected_function():
    calls = []

    def fake_list(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return ["video_list.json"]

    files, err = list_dataset_files("token", "owner/dataset", list_repo_files_fn=fake_list)

    assert err is None
    assert files == ["video_list.json"]
    assert calls == [("owner/dataset", {"repo_type": "dataset", "token": "token"})]


def test_upload_dataset_file_uses_policy_and_injected_api(tmp_path):
    policy = _policy(tmp_path)
    local = tmp_path / "database" / "video_list.json"
    local.write_text("[]", encoding="utf-8")
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("init", token))

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs))

    rel, err = upload_dataset_file(
        local,
        token="token",
        dataset_id="owner/dataset",
        policy=policy,
        api_factory=FakeApi,
    )

    assert err is None
    assert rel == "video_list.json"
    assert calls[-1][1]["path_in_repo"] == "video_list.json"


def test_download_dataset_file_uses_hub_then_http_fallback(tmp_path):
    policy = _policy(tmp_path)
    calls = []

    def fake_hub_download(**kwargs):
        calls.append(("hub", kwargs))
        raise RuntimeError("cannot import name HfApi")

    def fake_get(url, **kwargs):
        calls.append(("http", url, kwargs))
        return _FakeResponse(200, chunks=[b"[]"])

    path, err, hub_err = download_dataset_file(
        "video_list.json",
        token="token",
        dataset_id="owner/dataset",
        policy=policy,
        min_size=2,
        hf_hub_download_fn=fake_hub_download,
        request_get=fake_get,
    )

    assert err is None
    assert "cannot import" in hub_err
    assert path.endswith("video_list.json")
    assert calls[0][0] == "hub"
    assert calls[1][0] == "http"


def test_download_dataset_file_bytes_and_exists_use_http_helpers():
    raw, err = download_dataset_file_bytes(
        "video_list.json",
        token="token",
        dataset_id="owner/dataset",
        request_get=lambda *args, **kwargs: _FakeResponse(200, chunks=[b"[]"]),
    )

    assert err is None
    assert raw == b"[]"
    assert dataset_file_exists(
        "video_list.json",
        token="token",
        dataset_id="owner/dataset",
        request_head=lambda *args, **kwargs: _FakeResponse(200, chunks=[]),
    )


def test_download_dataset_file_with_progress_writes_target(tmp_path):
    target = tmp_path / "video_list.json"
    progress = []

    path, err = download_dataset_file_with_progress(
        "video_list.json",
        target,
        token="token",
        dataset_id="owner/dataset",
        min_size=2,
        progress_callback=lambda downloaded, total: progress.append((downloaded, total)),
        request_get=lambda *args, **kwargs: _FakeResponse(200, chunks=[b"[", b"]"]),
    )

    assert err is None
    assert path == str(target)
    assert target.read_bytes() == b"[]"
    assert progress[-1] == (2, 2)


def test_verify_dataset_via_http_uses_injected_request():
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(200, chunks=[b"{}"])

    ok, msg = verify_dataset_via_http("token", "owner/dataset", request_get=fake_get)

    assert ok
    assert msg is None
    assert calls[0][0].endswith("/video_list.json")
    assert calls[0][1]["headers"] == {"Authorization": "Bearer token"}


def test_verify_dataset_via_http_reports_empty_probe_file():
    ok, msg = verify_dataset_via_http(
        "token",
        "owner/dataset",
        request_get=lambda *args, **kwargs: _FakeResponse(200, chunks=[b""], headers={"content-length": "0"}),
    )

    assert not ok
    assert "trống" in msg


def test_download_dataset_file_via_http_writes_allowed_file(tmp_path):
    policy = _policy(tmp_path)

    path, err = download_dataset_file_via_http(
        "video_list.json",
        token="token",
        dataset_id="owner/dataset",
        policy=policy,
        min_size=2,
        request_get=lambda *args, **kwargs: _FakeResponse(200, chunks=[b"[]"]),
    )

    assert err is None
    assert path.endswith("video_list.json")
    assert (tmp_path / "database" / "video_list.json").read_bytes() == b"[]"


def test_download_dataset_file_via_http_rejects_unsafe_rel_path(tmp_path):
    policy = _policy(tmp_path)

    path, err = download_dataset_file_via_http(
        "../users.json",
        token="token",
        dataset_id="owner/dataset",
        policy=policy,
        request_get=lambda *args, **kwargs: _FakeResponse(200, chunks=[b"{}"]),
    )

    assert path is None
    assert err == "Đường dẫn cloud không hợp lệ."
