from pathlib import Path

from utils.pose_classifier_utils import resolve_local_path


def test_resolve_local_path_rejects_file_outside_allowed_roots(tmp_path):
    data_dir = tmp_path / "data"
    processed_dir = tmp_path / "processed"
    db_dir = tmp_path / "db"
    outside = tmp_path / "outside.csv"
    for folder in (data_dir, processed_dir, db_dir):
        folder.mkdir()
    outside.write_text("x" * 200, encoding="utf-8")

    assert resolve_local_path(str(outside), str(data_dir), str(processed_dir), str(db_dir)) is None


def test_resolve_local_path_accepts_processed_file(tmp_path):
    data_dir = tmp_path / "data"
    processed_dir = tmp_path / "processed"
    db_dir = tmp_path / "db"
    for folder in (data_dir, processed_dir, db_dir):
        folder.mkdir()
    csv_path = processed_dir / "processed_123_f_data.csv"
    csv_path.write_text("x" * 200, encoding="utf-8")

    resolved = resolve_local_path(csv_path.name, str(data_dir), str(processed_dir), str(db_dir))

    assert resolved == str(csv_path.resolve())

