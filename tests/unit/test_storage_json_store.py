import json

from storage.json_store import read_json, update_json, write_json


def test_write_json_is_atomic_and_readable(tmp_path):
    path = tmp_path / "data.json"

    assert write_json(path, {"items": [1, 2, 3]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"items": [1, 2, 3]}
    assert read_json(path, {}) == {"items": [1, 2, 3]}
    assert not list(tmp_path.glob("*.tmp"))


def test_update_json_uses_default_and_persists(tmp_path):
    path = tmp_path / "counter.json"

    update_json(path, lambda data: {"count": data["count"] + 1}, default={"count": 0})

    assert read_json(path, {}) == {"count": 1}


def test_invalid_json_returns_default(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    assert read_json(path, {"fallback": True}) == {"fallback": True}

