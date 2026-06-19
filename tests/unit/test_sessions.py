from auth.sessions import (
    bump_global_session_version,
    get_global_session_version,
    session_is_current,
)


def test_global_session_version_bump_invalidates_old_sessions(tmp_path):
    state_file = tmp_path / "session_state.json"

    original = get_global_session_version(str(state_file))
    assert session_is_current(str(state_file), original)

    bumped = bump_global_session_version(str(state_file), actor="admin", reason="test")

    assert bumped == original + 1
    assert not session_is_current(str(state_file), original)
    assert session_is_current(str(state_file), bumped)
