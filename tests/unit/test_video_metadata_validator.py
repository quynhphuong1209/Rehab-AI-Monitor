from scripts.validate_video_metadata import validate_records


def test_video_metadata_validator_detects_and_fixes_zip_timestamp_mismatch():
    records = [
        {
            "username": "user01",
            "video_name": "video.mp4",
            "video_path": "patient_uploads/user01_video.mp4",
            "processed_path": "processed_results/processed_111_f.mp4",
            "frames_zip_path": "processed_results/processed_222_frames.zip",
            "status": "done",
        }
    ]

    findings, changed = validate_records(records, fix=True)

    assert changed
    assert findings
    assert "processed_111_frames.zip" in records[0]["frames_zip_path"]

