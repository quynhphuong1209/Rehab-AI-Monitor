import importlib
def _fresh_startup_module():
    import app_startup

    return importlib.reload(app_startup)


def test_compute_thread_count_is_pure(monkeypatch):
    startup = _fresh_startup_module()
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    assert startup.get_compute_thread_count(lambda: 4) == 3
    assert "OMP_NUM_THREADS" not in startup.os.environ


def test_configure_process_environment_sets_defaults_once(monkeypatch):
    startup = _fresh_startup_module()
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    threads = startup.configure_process_environment(lambda: 3)

    assert threads == 2
    assert startup.os.environ["OMP_NUM_THREADS"] == "2"


def test_app_startup_respects_disabled_side_effects(tmp_path):
    startup = _fresh_startup_module()
    calls = []
    target_dir = tmp_path / "runtime"
    config = startup.StartupConfig(
        create_dirs=False,
        enable_boot_sync=False,
        enable_temp_cleanup=False,
        enable_auto_transcode=False,
        enable_auto_resume=False,
        enable_resume_watcher=False,
        enable_stream_filters=False,
        enable_warnings_filter=False,
    )

    ran = startup.app_startup(
        config=config,
        dirs=[target_dir],
        boot_sync_job=lambda: calls.append("boot"),
        cleanup_job=lambda: calls.append("cleanup"),
    )

    assert ran is True
    assert calls == []
    assert not target_dir.exists()


def test_app_startup_is_idempotent_and_runs_enabled_jobs(tmp_path):
    startup = _fresh_startup_module()
    calls = []
    target_dir = tmp_path / "runtime"
    config = startup.StartupConfig(
        enable_stream_filters=False,
        enable_warnings_filter=False,
        enable_boot_sync=False,
        enable_temp_cleanup=True,
        create_dirs=True,
    )

    first = startup.app_startup(config=config, dirs=[target_dir], cleanup_job=lambda: calls.append("cleanup"))
    second = startup.app_startup(config=config, dirs=[target_dir], cleanup_job=lambda: calls.append("again"))

    assert first is True
    assert second is False
    assert calls == ["cleanup"]
    assert target_dir.is_dir()


def test_streamlit_page_config_only_runs_inside_script_context(monkeypatch):
    startup = _fresh_startup_module()
    calls = []

    class FakeSt:
        def set_page_config(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(startup, "has_streamlit_script_context", lambda: False)
    assert startup.configure_streamlit_page_if_running(FakeSt()) is False
    assert calls == []

    monkeypatch.setattr(startup, "has_streamlit_script_context", lambda: True)
    assert startup.configure_streamlit_page_if_running(FakeSt()) is True
    assert calls[0]["layout"] == "wide"
