from storage.app_json import (
    default_for_json_path,
    format_schema_issue_lines,
    read_app_json,
    update_app_json,
    write_app_json,
)
from storage.json_store import read_json


def test_default_for_json_path_preserves_users_object_default():
    assert default_for_json_path("database/users.json") == {}
    assert default_for_json_path("database/video_list.json") == []


def test_write_app_json_normalizes_video_records(tmp_path):
    path = tmp_path / "video_list.json"

    result = write_app_json(path, [{"username": "bn01", "video_name": "clip.mp4"}])

    assert result.ok
    saved = read_json(path, [])
    assert saved[0]["full_name"] == "bn01"
    assert saved[0]["metrics"] == {}


def test_read_app_json_reports_schema_issues(tmp_path):
    path = tmp_path / "video_list.json"
    path.write_text("{}", encoding="utf-8")

    result = read_app_json(path)

    assert result.data == []
    assert result.changed
    assert format_schema_issue_lines(path, result.issues)[0].startswith("[Schema] video_list.json:")


def test_update_app_json_normalizes_mutated_data(tmp_path):
    path = tmp_path / "video_list.json"

    def add_video(data):
        data.append({"username": "bn01", "video_name": "clip.mp4"})

    update_app_json(path, add_video)

    saved = read_json(path, [])
    assert saved[0]["full_name"] == "bn01"
