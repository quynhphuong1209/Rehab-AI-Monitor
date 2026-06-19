from utils.checksum import verify_sha256_sidecar, write_sha256_sidecar


def test_sha256_sidecar_detects_tampering(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"trusted artifact")

    write_sha256_sidecar(artifact)
    assert verify_sha256_sidecar(artifact, required=True)

    artifact.write_bytes(b"tampered artifact")
    assert not verify_sha256_sidecar(artifact, required=True)
