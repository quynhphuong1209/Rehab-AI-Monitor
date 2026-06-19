from auth.passwords import (
    HASH_VERSION_ARGON2,
    HASH_VERSION_SHA256,
    hash_password_legacy_sha256,
    hash_password_v2,
    needs_password_rehash,
    password_record_update,
    verify_password_record,
)


def test_argon2_hash_verifies_and_wrong_password_fails():
    hashed = hash_password_v2("correct horse")
    record = {"password": hashed, "hash_version": HASH_VERSION_ARGON2}

    assert verify_password_record("correct horse", record).ok
    assert not verify_password_record("wrong", record).ok


def test_legacy_sha256_verifies_and_requests_rehash():
    record = {
        "password": hash_password_legacy_sha256("old-password"),
        "hash_version": HASH_VERSION_SHA256,
    }

    result = verify_password_record("old-password", record)

    assert result.ok
    assert result.needs_rehash
    assert needs_password_rehash(record)


def test_password_record_update_sets_argon2_metadata():
    fields = password_record_update("new-password", updated_at="2026-06-18T10:00:00", must_change_password=False)

    assert fields["hash_version"] == HASH_VERSION_ARGON2
    assert fields["updated_at"] == "2026-06-18T10:00:00"
    assert fields["must_change_password"] is False
    assert verify_password_record("new-password", fields).ok
