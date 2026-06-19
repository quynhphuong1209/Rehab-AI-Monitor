from pathlib import Path

import pytest

from utils.path_security import PathSecurityError, normalize_relative_path, safe_data_path


def test_normalize_relative_path_rejects_traversal_and_absolute():
    with pytest.raises(PathSecurityError):
        normalize_relative_path("../database/users.json")
    with pytest.raises(PathSecurityError):
        normalize_relative_path("/data/users.json")


def test_safe_data_path_accepts_path_inside_allowed_root(tmp_path):
    upload_root = tmp_path / "patient_uploads"
    upload_root.mkdir()

    resolved = safe_data_path("patient_uploads/demo.mp4", [upload_root], base_dir=tmp_path)

    assert resolved == str((upload_root / "demo.mp4").resolve())


def test_safe_data_path_rejects_escape_from_allowed_root(tmp_path):
    upload_root = tmp_path / "patient_uploads"
    upload_root.mkdir()

    with pytest.raises(PathSecurityError):
        safe_data_path("database/users.json", [upload_root], base_dir=tmp_path)
