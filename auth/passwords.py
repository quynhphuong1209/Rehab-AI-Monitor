"""Password hashing and migration helpers.

New passwords use Argon2id. Legacy SHA-256 hashes are verified only so a
successful login can migrate the account to the stronger hash.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from typing import Any, Mapping

_original_platform_machine = platform.machine
try:
    platform.machine = lambda: "AMD64"
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
finally:
    platform.machine = _original_platform_machine


HASH_VERSION_ARGON2 = "argon2"
HASH_VERSION_SHA256 = "sha256"


def _build_password_hasher() -> PasswordHasher:
    original_machine = platform.machine
    try:
        platform.machine = lambda: "AMD64"
        return PasswordHasher(
            time_cost=2,
            memory_cost=65536,
            parallelism=2,
            hash_len=32,
            salt_len=16,
        )
    finally:
        platform.machine = original_machine


_HASHER = _build_password_hasher()


@dataclass(frozen=True)
class PasswordVerification:
    ok: bool
    needs_rehash: bool = False
    legacy_version: str | None = None


def hash_password_v2(password: str) -> str:
    if password is None:
        raise ValueError("password is required")
    return _HASHER.hash(str(password))


def password_record_update(
    password: str,
    *,
    updated_at: str | None = None,
    must_change_password: bool | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "password": hash_password_v2(password),
        "hash_version": HASH_VERSION_ARGON2,
    }
    if updated_at is not None:
        fields["updated_at"] = updated_at
    if must_change_password is not None:
        fields["must_change_password"] = bool(must_change_password)
    return fields


def hash_password_legacy_sha256(password: str) -> str:
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()


def is_legacy_sha256_hash(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


def verify_password_record(password: str, user_record: Mapping[str, Any] | None) -> PasswordVerification:
    if not user_record:
        return PasswordVerification(False)

    stored_hash = str(user_record.get("password") or "")
    version = str(user_record.get("hash_version") or "").lower()
    if not stored_hash:
        return PasswordVerification(False)

    if version == HASH_VERSION_ARGON2 or stored_hash.startswith("$argon2"):
        try:
            ok = _HASHER.verify(stored_hash, str(password))
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return PasswordVerification(False)
        return PasswordVerification(ok, needs_rehash=ok and _HASHER.check_needs_rehash(stored_hash))

    if version in {"", HASH_VERSION_SHA256} and is_legacy_sha256_hash(stored_hash):
        ok = hash_password_legacy_sha256(password) == stored_hash
        return PasswordVerification(ok, needs_rehash=ok, legacy_version=HASH_VERSION_SHA256)

    return PasswordVerification(False)


def needs_password_rehash(user_record: Mapping[str, Any] | None) -> bool:
    if not user_record:
        return False
    stored_hash = str(user_record.get("password") or "")
    version = str(user_record.get("hash_version") or "").lower()
    if version != HASH_VERSION_ARGON2 or not stored_hash.startswith("$argon2"):
        return bool(stored_hash)
    try:
        return _HASHER.check_needs_rehash(stored_hash)
    except (VerificationError, InvalidHashError):
        return True
