from utils.checkpoint_utils import load_checkpoint, save_checkpoint


def test_checkpoint_load_requires_valid_checksum_sidecar(tmp_path):
    checkpoint = tmp_path / "checkpoint.pkl.gz"
    payload = {"phase": "pass2", "pass1_data": [1], "pass2_processed_count": 1}

    assert save_checkpoint(str(checkpoint), payload)
    loaded = load_checkpoint(str(checkpoint))
    assert loaded["phase"] == "pass2"

    checkpoint.write_bytes(b"tampered")
    assert load_checkpoint(str(checkpoint), retries=1) is None
