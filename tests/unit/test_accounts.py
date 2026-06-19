from auth.accounts import (
    find_user_key,
    find_user_key_by_email,
    find_user_uniqueness_issues,
    normalize_auth_text,
    roles_match,
)


def test_normalize_auth_text_trims_and_collapses_whitespace():
    assert normalize_auth_text("  user\t name  ") == "user name"
    assert normalize_auth_text(None) == ""


def test_find_user_key_matches_username_case_insensitive_but_not_full_name():
    users = {
        "Patient01": {"full_name": "Display Name", "email": "patient@example.test"},
    }

    assert find_user_key(users, " patient01 ") == "Patient01"
    assert find_user_key(users, "Display Name") is None


def test_find_user_key_by_email_matches_normalized_email():
    users = {
        "patient01": {"email": "Patient@Example.Test"},
    }

    assert find_user_key_by_email(users, " patient@example.test ") == "patient01"
    assert find_user_key_by_email(users, "") is None


def test_roles_match_uses_default_patient_role():
    assert roles_match("", "Bệnh nhân")
    assert not roles_match("Quản trị viên", "Bệnh nhân")


def test_find_user_uniqueness_issues_reports_case_and_email_duplicates():
    users = {
        "User01": {"email": "same@example.test"},
        "user01": {"email": "other@example.test"},
        "User02": {"email": " SAME@example.test "},
    }

    assert find_user_uniqueness_issues(users) == ["duplicate email", "duplicate username"]
