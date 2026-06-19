import json
import subprocess
import sys
import time
import zipfile

from starlette.testclient import TestClient

from auth.passwords import password_record_update
from backend.config import BackendConfig
from backend.main import analysis_jobs, app, tokens
from backend.repository import JsonRepository
from models.schemas import ADMIN_ROLE, DOCTOR_ROLE, PATIENT_ROLE, RESEARCHER_ROLE


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _configure_tmp_backend(tmp_path, monkeypatch):
    db = tmp_path / "database"
    users = {
        "admin": {
            "username": "admin",
            "role": ADMIN_ROLE,
            "full_name": "Admin",
            "active": True,
            **password_record_update("secret"),
        },
        "doctor": {
            "username": "doctor",
            "role": DOCTOR_ROLE,
            "full_name": "Doctor",
            "active": True,
            "assigned_patient_usernames": ["patient01"],
            **password_record_update("doctorpass"),
        },
        "patient01": {
            "username": "patient01",
            "role": PATIENT_ROLE,
            "full_name": "Patient One",
            "email": "patient01@example.test",
            "active": True,
            "assigned_doctor_username": "doctor",
            **password_record_update("patientpass"),
        },
        "patient02": {
            "username": "patient02",
            "role": PATIENT_ROLE,
            "full_name": "Patient Two",
            "email": "patient02@example.test",
            "active": True,
            **password_record_update("patient2pass"),
        },
        "researcher": {
            "username": "researcher",
            "role": RESEARCHER_ROLE,
            "full_name": "Researcher",
            "active": True,
            **password_record_update("researchpass"),
        },
    }
    videos = [
        {"username": "patient01", "full_name": "Patient One", "video_name": "a.mp4", "exercise": "codman"},
        {"username": "patient02", "full_name": "Patient Two", "video_name": "b.mp4", "exercise": "codman"},
    ]
    evaluations = [
        {"patient_username": "patient01", "video_name": "a.mp4", "doctor_username": "doctor", "comments": "private"},
        {"patient_username": "patient02", "video_name": "b.mp4", "doctor_username": "doctor", "comments": "private"},
    ]
    symptoms = [
        {"username": "patient01", "full_name": "Patient One", "symptoms": "pain"},
        {"username": "patient02", "full_name": "Patient Two", "symptoms": "stiff"},
    ]
    schedules = [
        {"patient_username": "patient01", "title": "Exercise", "notes": "private note"},
        {"patient_username": "patient02", "title": "Visit", "notes": "private note"},
    ]
    research_records = [
        {"patient_username": "patient01", "general_result": "ok", "specialist_comment": "private"},
        {"patient_username": "patient02", "general_result": "ok", "specialist_comment": "private"},
    ]
    _write(db / "users.json", users)
    _write(db / "video_list.json", videos)
    _write(db / "doctor_evaluations.json", evaluations)
    _write(db / "patient_symptoms.json", symptoms)
    _write(db / "schedules.json", schedules)
    _write(db / "research_data.json", research_records)

    config = BackendConfig(repo_root=tmp_path, database_dir=db)
    monkeypatch.setattr("backend.main.repo", JsonRepository(config))
    analysis_jobs.configure(repo_root=tmp_path, upload_dir=tmp_path / "patient_uploads", processed_dir=tmp_path / "processed_results")
    analysis_jobs.runner = analysis_jobs._validate_transcode_runner
    analysis_jobs.ai_runner = None
    analysis_jobs.result_handler = None
    analysis_jobs.command_runner = subprocess.run
    analysis_jobs._running.clear()
    analysis_jobs._cancel_events.clear()
    tokens._tokens.clear()


def test_backend_sync_ai_runner_is_opt_in(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)

    from backend import main as backend_main

    backend_main._sync_analysis_jobs_config()
    assert backend_main.analysis_jobs.ai_runner is None

    enabled_config = BackendConfig(
        repo_root=tmp_path,
        database_dir=tmp_path / "database",
        enable_ai_runner=True,
    )
    monkeypatch.setattr("backend.main.repo", JsonRepository(enabled_config))

    backend_main._sync_analysis_jobs_config()

    assert getattr(backend_main.analysis_jobs.ai_runner, "is_backend_mediapipe_ai_runner", False) is True


def test_backend_health(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_backend_login_and_me(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)

    login = client.post("/auth/login", json={"username": "admin", "password": "secret"})

    assert login.status_code == 200
    body = login.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["username"] == "admin"
    assert "password" not in body["user"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["user"]["role"] == ADMIN_ROLE


def test_backend_rejects_bad_login_and_requires_auth(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)

    assert client.post("/auth/login", json={"username": "admin", "password": "bad"}).status_code == 401
    assert client.get("/videos").status_code == 401


def test_backend_registers_patient_account_and_logs_in(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/auth/register",
        json={
            "username": "newpatient",
            "full_name": "New Patient",
            "email": "newpatient@example.test",
            "password": "newpass1",
            "confirm_password": "newpass1",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["username"] == "newpatient"
    assert body["user"]["role"] == PATIENT_ROLE
    assert body["user"]["must_change_password"] is False
    assert "password" not in body["user"]

    users = JsonRepository(BackendConfig(repo_root=tmp_path, database_dir=tmp_path / "database")).users()
    assert users["newpatient"]["role"] == PATIENT_ROLE
    assert users["newpatient"]["hash_version"] == "argon2"
    assert users["newpatient"]["password"].startswith("$argon2")

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "newpatient"


def test_backend_register_rejects_duplicates_and_weak_payloads(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)

    duplicate_username = client.post(
        "/auth/register",
        json={
            "username": "patient01",
            "email": "unique@example.test",
            "password": "newpass1",
            "confirm_password": "newpass1",
        },
    )
    duplicate_email = client.post(
        "/auth/register",
        json={
            "username": "uniqueuser",
            "email": "patient01@example.test",
            "password": "newpass1",
            "confirm_password": "newpass1",
        },
    )
    weak_password = client.post(
        "/auth/register",
        json={
            "username": "weakuser",
            "email": "weak@example.test",
            "password": "123",
            "confirm_password": "123",
        },
    )
    mismatch = client.post(
        "/auth/register",
        json={
            "username": "mismatch",
            "email": "mismatch@example.test",
            "password": "newpass1",
            "confirm_password": "different",
        },
    )

    assert duplicate_username.status_code == 409
    assert duplicate_email.status_code == 409
    assert weak_password.status_code == 400
    assert mismatch.status_code == 400


def test_backend_scopes_video_and_evaluation_lists_for_doctor(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    videos = client.get("/videos", headers=headers).json()
    evaluations = client.get("/evaluations", headers=headers).json()

    assert videos["count"] == 1
    assert videos["items"][0]["username"] == "patient01"
    assert evaluations["count"] == 1
    assert evaluations["items"][0]["patient_username"] == "patient01"


def test_backend_patient_can_upload_video_and_append_metadata(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    video_bytes = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128

    response = client.post(
        "/videos/upload",
        headers=headers,
        data={"full_name": "Patient One", "exercise": "Codman"},
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
    )

    assert response.status_code == 201
    item = response.json()["item"]
    assert item["username"] == "patient01"
    assert item["video_name"] == "clip.mp4"
    assert item["original_filename"] == "clip.mp4"
    assert item["stored_filename"].startswith("patient01_")
    assert item["stored_filename"].endswith("_clip.mp4")
    assert item["video_path"].replace("\\", "/").startswith("patient_uploads/")
    assert item["exercise"] == "Codman"
    assert item["status"] == "Chờ NCV phân tích"

    stored = tmp_path / item["video_path"]
    assert stored.exists()
    assert stored.read_bytes() == video_bytes

    videos = JsonRepository(BackendConfig(repo_root=tmp_path, database_dir=tmp_path / "database")).videos()
    assert videos[-1]["stored_filename"] == item["stored_filename"]


def test_backend_upload_video_rejects_non_patient_and_bad_payload(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    video_bytes = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128

    doctor_response = client.post(
        "/videos/upload",
        headers={"Authorization": f"Bearer {doctor_login.json()['access_token']}"},
        data={"exercise": "Codman"},
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
    )
    missing_exercise = client.post(
        "/videos/upload",
        headers={"Authorization": f"Bearer {patient_login.json()['access_token']}"},
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
    )
    bad_header = client.post(
        "/videos/upload",
        headers={"Authorization": f"Bearer {patient_login.json()['access_token']}"},
        data={"exercise": "Codman"},
        files={"file": ("clip.mp4", b"MZ executable", "video/mp4")},
    )

    assert doctor_response.status_code == 403
    assert missing_exercise.status_code == 400
    assert bad_header.status_code == 400


def test_backend_serves_video_media_for_visible_actor(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    upload_dir.mkdir()
    video_bytes = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128
    (upload_dir / "patient01_clip.mp4").write_bytes(video_bytes)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.get("/videos/media/patient01_clip.mp4", headers=headers)

    assert response.status_code == 200
    assert response.content == video_bytes
    assert response.headers["content-type"].startswith("video/mp4")


def test_backend_video_media_rejects_out_of_scope_and_bad_filename(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    upload_dir.mkdir()
    (upload_dir / "patient02_clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient02",
                "full_name": "Patient Two",
                "video_name": "clip.mp4",
                "stored_filename": "patient02_clip.mp4",
                "video_path": "patient_uploads/patient02_clip.mp4",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    assert client.get("/videos/media/patient02_clip.mp4", headers=headers).status_code == 404
    assert client.get("/videos/media/..%5Cpatient02_clip.mp4", headers=headers).status_code == 400


def test_backend_researcher_can_start_analysis_job_and_patient_can_read_progress(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    upload_dir.mkdir()
    (upload_dir / "patient01_clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    researcher_login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})

    response = client.post(
        "/videos/patient01_clip.mp4/analysis-jobs",
        headers={"Authorization": f"Bearer {researcher_login.json()['access_token']}"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["started"] is True
    assert body["job"]["video_name"] == "clip.mp4"
    assert body["job"]["username"] == "patient01"
    assert body["job"]["status"] == "processing"

    latest = client.get(
        "/videos/patient01_clip.mp4/analysis-jobs/latest",
        headers={"Authorization": f"Bearer {patient_login.json()['access_token']}"},
    )
    assert latest.status_code == 200
    job = latest.json()["job"]
    assert job["job_id"] == body["job"]["job_id"]
    assert job["status"] in {"processing", "ready_for_ai_worker", "error"}
    assert (tmp_path / "processed_results" / f"progress_{job['job_id']}.json").exists()


def test_backend_analysis_job_success_updates_video_metadata(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    upload_dir.mkdir()
    (upload_dir / "patient01_clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "exercise": "Codman",
                "status": "Chờ NCV phân tích",
            }
        ],
    )

    def command_runner(cmd, **kwargs):
        if "-show_entries" in cmd:
            return type("Result", (), {"returncode": 0, "stdout": "12.5\n", "stderr": ""})()
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264"}]}),
                "stderr": "",
            },
        )()

    def ai_runner(request, analysis_input_path, progress):
        progress(status="processing", progress=0.72, status_msg="AI đang phân tích.")
        return {
            "status": "success",
            "progress": 1.0,
            "status_msg": "AI đã phân tích xong.",
            "result": {
                "processed_path": "processed_results/processed_patient01_clip.mp4",
                "metrics": {"do_chinh_xac": 88.8, "f1_score": 0.91},
                "df_path": "processed_results/patient01_clip_data.csv",
                "all_frames_data_path": "processed_results/patient01_clip_frames.json",
                "frames_zip_path": "processed_results/patient01_clip_frames.zip",
            },
        }

    monkeypatch.setattr(analysis_jobs, "command_runner", command_runner)
    monkeypatch.setattr(analysis_jobs, "ai_runner", ai_runner)
    client = TestClient(app)
    researcher_login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})

    response = client.post(
        "/videos/patient01_clip.mp4/analysis-jobs",
        headers={"Authorization": f"Bearer {researcher_login.json()['access_token']}"},
    )

    assert response.status_code == 202
    job_id = response.json()["job"]["job_id"]
    progress_path = tmp_path / "processed_results" / f"progress_{job_id}.json"
    deadline = time.time() + 2
    job = None
    while time.time() < deadline:
        if progress_path.exists():
            try:
                job = json.loads(progress_path.read_text(encoding="utf-8"))
            except PermissionError:
                time.sleep(0.01)
                continue
            if job.get("status") == "success":
                break
        time.sleep(0.01)

    assert job["status"] == "success"
    videos = JsonRepository(BackendConfig(repo_root=tmp_path, database_dir=tmp_path / "database")).videos()
    video = videos[0]
    assert video["status"] == "Đã phân tích"
    assert video["processed_path"] == "processed_results/processed_patient01_clip.mp4"
    assert video["df_path"] == "processed_results/patient01_clip_data.csv"
    assert video["frames_zip_path"] == "processed_results/patient01_clip_frames.zip"
    assert video["accuracy"] == 88.8
    assert video["metrics"]["f1_score"] == 0.91


def test_backend_analysis_job_start_requires_role_and_scope(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    upload_dir.mkdir()
    (upload_dir / "patient02_clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient02",
                "full_name": "Patient Two",
                "video_name": "clip.mp4",
                "stored_filename": "patient02_clip.mp4",
                "video_path": "patient_uploads/patient02_clip.mp4",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    patient_login = client.post("/auth/login", json={"username": "patient02", "password": "patient2pass"})
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})

    forbidden = client.post(
        "/videos/patient02_clip.mp4/analysis-jobs",
        headers={"Authorization": f"Bearer {patient_login.json()['access_token']}"},
    )
    out_of_scope = client.get(
        "/videos/patient02_clip.mp4/analysis-jobs/latest",
        headers={"Authorization": f"Bearer {doctor_login.json()['access_token']}"},
    )

    assert forbidden.status_code == 403
    assert out_of_scope.status_code == 404


def test_backend_analysis_job_lifecycle_history_retry_rerun_options(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    upload_dir.mkdir()
    (upload_dir / "patient01_clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "exercise": "Codman",
            }
        ],
    )

    def command_runner(cmd, **kwargs):
        if "-show_entries" in cmd:
            return type("Result", (), {"returncode": 0, "stdout": "12.5\n", "stderr": ""})()
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264"}]}),
                "stderr": "",
            },
        )()

    monkeypatch.setattr(analysis_jobs, "command_runner", command_runner)
    client = TestClient(app)
    researcher_login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})
    headers = {"Authorization": f"Bearer {researcher_login.json()['access_token']}"}

    rerun = client.post(
        "/videos/patient01_clip.mp4/analysis-jobs/rerun",
        headers=headers,
        json={
            "model_type": "MediaPipe Lite",
            "min_confidence": 0.7,
            "skip_step": 3,
            "resize_width": 640,
        },
    )

    assert rerun.status_code == 202
    first_run_id = rerun.json()["job"]["run_id"]
    deadline = time.time() + 2
    latest_job = None
    while time.time() < deadline:
        latest = client.get("/videos/patient01_clip.mp4/analysis-jobs/latest", headers=headers)
        latest_job = latest.json()["job"]
        if latest_job and latest_job["status"] == "ready_for_ai_worker":
            break
        time.sleep(0.01)

    assert latest_job["job_meta"]["action"] == "rerun"
    assert latest_job["job_meta"]["options"] == {
        "model_type": "MediaPipe Lite",
        "min_confidence": 0.7,
        "skip_step": 3,
        "resize_width": 640,
    }
    assert latest_job["steps"][0]["status"] == "done"

    retry = client.post("/videos/patient01_clip.mp4/analysis-jobs/retry", headers=headers)
    assert retry.status_code == 202
    second_run_id = retry.json()["job"]["run_id"]
    assert second_run_id != first_run_id
    deadline = time.time() + 2
    while time.time() < deadline:
        latest = client.get("/videos/patient01_clip.mp4/analysis-jobs/latest", headers=headers)
        latest_job = latest.json()["job"]
        if latest_job and latest_job["run_id"] == second_run_id and latest_job["status"] == "ready_for_ai_worker":
            break
        time.sleep(0.01)

    history = client.get("/videos/patient01_clip.mp4/analysis-jobs/history", headers=headers)
    assert history.status_code == 200
    body = history.json()
    assert body["count"] == 2
    assert [item["run_id"] for item in body["items"]] == [first_run_id, second_run_id]
    assert body["items"][1]["job_meta"]["action"] == "retry"
    assert body["items"][1]["job_meta"]["options"]["model_type"] == "MediaPipe Lite"


def test_backend_analysis_job_cancel_requires_role_and_marks_running_job(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    upload_dir.mkdir()
    (upload_dir / "patient01_clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "exercise": "Codman",
            }
        ],
    )
    started = False

    def slow_runner(request, progress):
        nonlocal started
        started = True
        progress(status="processing", progress=0.21, status_msg="Đang chuẩn bị.")
        time.sleep(0.2)
        progress(status="processing", progress=0.60, status_msg="Không nên hoàn tất.")
        return {"status": "ready_for_ai_worker", "progress": 0.65, "result": {"analysis_input_path": request.video_path}}

    monkeypatch.setattr(analysis_jobs, "runner", slow_runner)
    client = TestClient(app)
    researcher_login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    researcher_headers = {"Authorization": f"Bearer {researcher_login.json()['access_token']}"}
    patient_headers = {"Authorization": f"Bearer {patient_login.json()['access_token']}"}

    response = client.post("/videos/patient01_clip.mp4/analysis-jobs", headers=researcher_headers)
    assert response.status_code == 202
    deadline = time.time() + 1
    while not started and time.time() < deadline:
        time.sleep(0.01)

    forbidden = client.post("/videos/patient01_clip.mp4/analysis-jobs/cancel", headers=patient_headers)
    canceled = client.post("/videos/patient01_clip.mp4/analysis-jobs/cancel", headers=researcher_headers)

    assert forbidden.status_code == 403
    assert canceled.status_code == 200
    assert canceled.json()["job"]["status"] == "canceled"
    assert canceled.json()["job"]["job_meta"]["canceled_by"] == "researcher"


def test_backend_lists_and_downloads_analysis_artifacts_for_visible_video(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    upload_dir = tmp_path / "patient_uploads"
    processed_dir = tmp_path / "processed_results"
    upload_dir.mkdir()
    processed_dir.mkdir()
    (upload_dir / "patient01_clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128)
    processed_video = b"\x00\x00\x00\x18ftypmp42processed"
    csv_data = b"frame,goc_vai\n1,90\n"
    frames_data = b'[{"index":1,"dung":true}]'
    zip_data = b"PK\x03\x04frames"
    (processed_dir / "processed_patient01_clip.mp4").write_bytes(processed_video)
    (processed_dir / "patient01_clip_data.csv").write_bytes(csv_data)
    (processed_dir / "patient01_clip_frames.json").write_bytes(frames_data)
    (processed_dir / "patient01_clip_frames.zip").write_bytes(zip_data)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "processed_path": "processed_results/processed_patient01_clip.mp4",
                "df_path": "processed_results/patient01_clip_data.csv",
                "all_frames_data_path": "processed_results/patient01_clip_frames.json",
                "frames_zip_path": "processed_results/patient01_clip_frames.zip",
                "exercise": "Codman",
                "accuracy": 91.2,
                "metrics": {"do_chinh_xac": 91.2, "f1_score": 0.88},
                "status": "Đã phân tích",
            }
        ],
    )
    _write(
        tmp_path / "database" / "doctor_evaluations.json",
        [
            {
                "patient_username": "patient01",
                "video_name": "clip.mp4",
                "exercise": "Codman",
                "doctor_username": "AI_Researcher",
                "doctor_name": "NCV: Researcher",
                "doctor_result": "Đúng",
                "comments": "AI report sent",
                "plan": "Tiếp tục",
                "time": "2026-06-19T07:15:00Z",
            }
        ],
    )
    client = TestClient(app)
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    headers = {"Authorization": f"Bearer {doctor_login.json()['access_token']}"}

    manifest = client.get("/videos/patient01_clip.mp4/analysis-artifacts", headers=headers)

    assert manifest.status_code == 200
    body = manifest.json()
    assert body["metrics"]["f1_score"] == 0.88
    assert body["video"]["accuracy"] == 91.2
    assert {item["kind"] for item in body["items"]} == {"processed-video", "angle-csv", "frames-json", "frames-zip"}
    assert all(item["available"] for item in body["items"])

    csv_response = client.get("/videos/patient01_clip.mp4/analysis-artifacts/angle-csv", headers=headers)
    json_response = client.get("/videos/patient01_clip.mp4/analysis-artifacts/frames-json", headers=headers)
    video_response = client.get("/videos/patient01_clip.mp4/analysis-artifacts/processed-video", headers=headers)

    assert csv_response.status_code == 200
    assert csv_response.content == csv_data
    assert json_response.status_code == 200
    assert json_response.content == frames_data
    assert video_response.status_code == 200
    assert video_response.content == processed_video


def test_backend_analysis_artifacts_reject_out_of_scope_and_unknown_kind(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    (processed_dir / "patient02_clip_data.csv").write_text("x\n", encoding="utf-8")
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient02",
                "video_name": "clip.mp4",
                "stored_filename": "patient02_clip.mp4",
                "video_path": "patient_uploads/patient02_clip.mp4",
                "df_path": "processed_results/patient02_clip_data.csv",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    researcher_login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})

    out_of_scope = client.get(
        "/videos/patient02_clip.mp4/analysis-artifacts",
        headers={"Authorization": f"Bearer {doctor_login.json()['access_token']}"},
    )
    unknown = client.get(
        "/videos/patient02_clip.mp4/analysis-artifacts/nope",
        headers={"Authorization": f"Bearer {researcher_login.json()['access_token']}"},
    )

    assert out_of_scope.status_code == 404
    assert unknown.status_code == 404


def test_backend_analysis_chart_previews_csv_and_falls_back_to_frames_json(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    (processed_dir / "patient01_clip_data.csv").write_text(
        "frame,goc_vai,goc_khuyu,vai_chuan,khuyu_chuan,dung,gan_dung\n"
        "1,90,170,95,165,true,false\n"
        "2,110,150,95,165,false,true\n",
        encoding="utf-8",
    )
    _write(
        processed_dir / "patient01_json_only_frames.json",
        [
            {"index": 1, "goc_vai": 80, "goc_khuyu": 160, "dung": True},
            {"index": 2, "goc_vai": 130, "goc_khuyu": 120, "dung": False, "gan_dung": False},
        ],
    )
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "df_path": "processed_results/patient01_clip_data.csv",
                "exercise": "Codman",
                "metrics": {"do_chinh_xac": 91.2},
            },
            {
                "username": "patient01",
                "video_name": "json-only.mp4",
                "stored_filename": "patient01_json_only.mp4",
                "video_path": "patient_uploads/patient01_json_only.mp4",
                "all_frames_data_path": "processed_results/patient01_json_only_frames.json",
                "exercise": "Codman",
            },
        ],
    )
    _write(
        tmp_path / "database" / "doctor_evaluations.json",
        [
            {
                "patient_username": "patient01",
                "video_name": "clip.mp4",
                "exercise": "Codman",
                "doctor_username": "AI_Researcher",
                "doctor_name": "NCV: Researcher",
                "doctor_result": "Đúng",
                "comments": "AI report sent",
                "plan": "Tiếp tục",
                "time": "2026-06-19T07:15:00Z",
            },
            {
                "patient_username": "patient01",
                "video_name": "json-only.mp4",
                "exercise": "Codman",
                "doctor_username": "AI_Researcher",
                "doctor_name": "NCV: Researcher",
                "doctor_result": "Gần đúng",
                "comments": "AI report sent",
                "plan": "Tiếp tục",
                "time": "2026-06-19T07:20:00Z",
            },
        ],
    )
    client = TestClient(app)
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    headers = {"Authorization": f"Bearer {patient_login.json()['access_token']}"}

    csv_response = client.get("/videos/patient01_clip.mp4/analysis-chart", headers=headers)
    json_response = client.get("/videos/patient01_json_only.mp4/analysis-chart", headers=headers)

    assert csv_response.status_code == 200
    csv_body = csv_response.json()
    assert csv_body["source"] == "csv"
    assert csv_body["total_rows"] == 2
    assert csv_body["filtered_rows"] == 2
    assert csv_body["sampled_rows"] == 2
    assert csv_body["columns"] == ["goc_vai", "goc_khuyu", "vai_chuan", "khuyu_chuan"]
    assert csv_body["summary"]["series"]["goc_vai"]["avg"] == 100.0
    assert csv_body["summary"]["labels"] == {"total": 2, "PASS": 2, "NEAR": 0, "FAIL": 0}
    assert csv_body["phase_summary"]["phases"]["G1"]["threshold"] == 45.0
    assert csv_body["series"][0]["goc_khuyu"] == 170.0
    assert csv_body["metrics"]["do_chinh_xac"] == 91.2

    assert json_response.status_code == 200
    json_body = json_response.json()
    assert json_body["source"] == "frames-json"
    assert json_body["summary"]["labels"]["FAIL"] == 1
    assert json_body["series"][1]["goc_vai"] == 130.0


def test_backend_analysis_chart_rejects_out_of_scope(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    (processed_dir / "patient02_clip_data.csv").write_text("frame,goc_vai\n1,90\n", encoding="utf-8")
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient02",
                "video_name": "clip.mp4",
                "stored_filename": "patient02_clip.mp4",
                "video_path": "patient_uploads/patient02_clip.mp4",
                "df_path": "processed_results/patient02_clip_data.csv",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})

    response = client.get(
        "/videos/patient02_clip.mp4/analysis-chart",
        headers={"Authorization": f"Bearer {doctor_login.json()['access_token']}"},
    )

    assert response.status_code == 404


def test_backend_video_result_detail_combines_scoped_clinical_and_ai_data(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    (processed_dir / "patient01_clip_data.csv").write_text("frame,goc_vai\n1,90\n", encoding="utf-8")
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "df_path": "processed_results/patient01_clip_data.csv",
                "exercise": "Codman",
                "accuracy": 91.2,
                "metrics": {"do_chinh_xac": 91.2, "f1_score": 0.88},
                "status": "Đã phân tích",
            },
            {
                "username": "patient02",
                "full_name": "Patient Two",
                "video_name": "other.mp4",
                "stored_filename": "patient02_clip.mp4",
                "video_path": "patient_uploads/patient02_clip.mp4",
                "exercise": "Codman",
            },
        ],
    )
    _write(
        tmp_path / "database" / "doctor_evaluations.json",
        [
            {
                "patient_username": "patient01",
                "video_name": "clip.mp4",
                "exercise": "Codman",
                "doctor_username": "doctor",
                "doctor_name": "Doctor",
                "doctor_result": "Gần đúng",
                "comments": "Tập chậm hơn",
                "comments_ncv": "private research note",
                "plan": "Tiếp tục",
                "time": "2026-06-19T07:05:00Z",
            }
        ],
    )
    _write(
        tmp_path / "database" / "patient_symptoms.json",
        [
            {
                "username": "patient01",
                "video_name": "clip.mp4",
                "exercise": "Codman",
                "symptoms": "Đau nhẹ",
                "vas": 3,
                "time": "2026-06-19T07:00:00Z",
            }
        ],
    )
    client = TestClient(app)
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    patient_headers = {"Authorization": f"Bearer {patient_login.json()['access_token']}"}
    doctor_headers = {"Authorization": f"Bearer {doctor_login.json()['access_token']}"}

    patient_response = client.get("/videos/patient01_clip.mp4/results", headers=patient_headers)
    doctor_response = client.get("/videos/patient01_clip.mp4/results", headers=doctor_headers)
    patient_out_of_scope = client.get("/videos/patient02_clip.mp4/results", headers=patient_headers)
    doctor_out_of_scope = client.get("/videos/patient02_clip.mp4/results", headers=doctor_headers)

    assert patient_response.status_code == 200
    patient_body = patient_response.json()
    assert patient_body["summary"]["doctor_result"] == "Gần đúng"
    assert patient_body["summary"]["doctor_plan"] == "Tiếp tục"
    assert patient_body["metrics"]["f1_score"] == 0.88
    assert {item["kind"] for item in patient_body["artifacts"]} == {
        "processed-video",
        "angle-csv",
        "frames-json",
        "frames-zip",
    }
    assert any(item["kind"] == "symptom" for item in patient_body["timeline"])
    assert "comments_ncv" not in patient_body["evaluation"]

    assert doctor_response.status_code == 200
    assert doctor_response.json()["evaluation"]["comments_ncv"] == "private research note"
    assert patient_out_of_scope.status_code == 404
    assert doctor_out_of_scope.status_code == 404


def test_backend_doctor_ai_detail_is_gated_until_researcher_report_is_sent(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    (processed_dir / "patient01_clip_data.csv").write_text("frame,goc_vai,goc_khuyu\n1,90,170\n", encoding="utf-8")
    _write(
        processed_dir / "patient01_clip_frames.json",
        [{"index": 1, "goc_vai": 90, "goc_khuyu": 170, "path": "processed_results/f_000001.jpg"}],
    )
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "full_name": "Patient One",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "df_path": "processed_results/patient01_clip_data.csv",
                "all_frames_data_path": "processed_results/patient01_clip_frames.json",
                "exercise": "Codman",
                "accuracy": 91.2,
                "metrics": {"do_chinh_xac": 91.2, "f1_score": 0.88},
                "status": "Đã phân tích",
            },
        ],
    )
    _write(
        tmp_path / "database" / "doctor_evaluations.json",
        [
            {
                "patient_username": "patient01",
                "video_name": "clip.mp4",
                "exercise": "Codman",
                "doctor_username": "doctor",
                "doctor_name": "Doctor",
                "doctor_result": "Gần đúng",
                "comments": "Tập chậm hơn",
                "plan": "Tiếp tục",
                "time": "2026-06-19T07:05:00Z",
            }
        ],
    )
    client = TestClient(app)
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    headers = {"Authorization": f"Bearer {doctor_login.json()['access_token']}"}

    result = client.get("/videos/patient01_clip.mp4/results", headers=headers)
    chart = client.get("/videos/patient01_clip.mp4/analysis-chart", headers=headers)
    frames = client.get("/videos/patient01_clip.mp4/analysis-frames", headers=headers)

    assert result.status_code == 200
    body = result.json()
    assert body["report_sent"] is False
    assert body["ai_detail_allowed"] is False
    assert body["report_status"]["report_status"] == "blocked_for_doctor"
    assert body["metrics"] == {}
    assert body["artifacts"] == []
    assert body["summary"]["doctor_result"] == "Gần đúng"
    assert chart.status_code == 403
    assert frames.status_code == 403

    _write(
        tmp_path / "database" / "doctor_evaluations.json",
        [
            {
                "patient_username": "patient01",
                "video_name": "clip.mp4",
                "exercise": "Codman",
                "doctor_username": "AI_Researcher",
                "doctor_name": "NCV: Researcher",
                "doctor_result": "Đúng",
                "comments": "Báo cáo AI chính thức",
                "plan": "Tiếp tục",
                "time": "2026-06-19T07:15:00Z",
            }
        ],
    )

    opened = client.get("/videos/patient01_clip.mp4/results", headers=headers)
    assert opened.status_code == 200
    assert opened.json()["ai_detail_allowed"] is True
    assert opened.json()["metrics"]["f1_score"] == 0.88
    assert client.get("/videos/patient01_clip.mp4/analysis-chart", headers=headers).status_code == 200


def test_backend_lists_paginated_analysis_frames_and_serves_zip_image(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    frames_json = processed_dir / "patient01_clip_frames.json"
    frames_zip = processed_dir / "patient01_clip_frames.zip"
    jpg_bytes = b"\xff\xd8\xff\xe0frame-one\xff\xd9"
    png_bytes = b"\x89PNG\r\n\x1a\nframe-two"
    _write(
        frames_json,
        [
            {
                "index": 1,
                "timestamp": "00:01",
                "path": "processed_results/f_000001.jpg",
                "goc_vai": 90,
                "goc_khuyu": 170,
                "dung": True,
                "gan_dung": False,
                "eval_info": {"shoulder_ref": 90, "elbow_ref": 170},
            },
            {
                "index": 2,
                "timestamp": "00:02",
                "path": "processed_results/f_000002.png",
                "goc_vai": 125,
                "goc_khuyu": 135,
                "dung": False,
                "gan_dung": True,
                "ml_label": "near",
                "ml_label_text": "Gần đúng",
                "ml_confidence": 0.42,
                "ml_probabilities": {"Sai": 0.2, "Gần đúng": 0.42, "Đúng": 0.38},
            },
            {
                "index": 3,
                "timestamp": "00:03",
                "path": "processed_results/f_000003.jpg",
                "goc_vai": 140,
                "goc_khuyu": 120,
                "dung": False,
                "gan_dung": False,
            },
        ],
    )
    with zipfile.ZipFile(frames_zip, "w") as zf:
        zf.writestr("f_000001.jpg", jpg_bytes)
        zf.writestr("f_000002.png", png_bytes)
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient01",
                "video_name": "clip.mp4",
                "stored_filename": "patient01_clip.mp4",
                "video_path": "patient_uploads/patient01_clip.mp4",
                "all_frames_data_path": "processed_results/patient01_clip_frames.json",
                "frames_zip_path": "processed_results/patient01_clip_frames.zip",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    headers = {"Authorization": f"Bearer {patient_login.json()['access_token']}"}

    page = client.get(
        "/videos/patient01_clip.mp4/analysis-frames?page=1&page_size=2",
        headers=headers,
    )
    filtered = client.get(
        "/videos/patient01_clip.mp4/analysis-frames?label=PASS&page_size=12",
        headers=headers,
    )

    assert page.status_code == 200
    body = page.json()
    assert body["summary"]["total"] == 3
    assert body["summary"]["PASS"] == 1
    assert body["summary"]["NEAR"] == 1
    assert body["summary"]["FAIL"] == 1
    assert body["summary"]["phases"]["G2"]["NEAR"] == 1
    assert body["pagination"]["total"] == 3
    assert body["pagination"]["total_pages"] == 2
    assert [item["label"] for item in body["items"]] == ["PASS", "NEAR"]
    assert body["items"][0]["image_id"] == "frame:0"
    assert body["items"][1]["phase"] == "G2"
    assert body["items"][1]["phase_threshold"] == 30.0
    assert body["items"][1]["ml"]["label_text"] == "Gần đúng"
    assert body["items"][1]["ml"]["confidence"] == 0.42

    image = client.get(
        f"/videos/patient01_clip.mp4/analysis-frames/{body['items'][0]['image_id']}",
        headers=headers,
    )
    assert image.status_code == 200
    assert image.content == jpg_bytes
    assert image.headers["content-type"].startswith("image/jpeg")

    assert filtered.status_code == 200
    assert filtered.json()["pagination"]["total"] == 1
    assert filtered.json()["items"][0]["label"] == "PASS"

    g2 = client.get(
        "/videos/patient01_clip.mp4/analysis-frames?label=G2&page_size=12",
        headers=headers,
    )
    assert g2.status_code == 200
    assert g2.json()["pagination"]["total"] == 1
    assert g2.json()["items"][0]["label"] == "NEAR"


def test_backend_analysis_frames_reject_out_of_scope_and_unsafe_zip(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed_results"
    processed_dir.mkdir()
    _write(
        processed_dir / "patient02_clip_frames.json",
        [{"index": 1, "path": "processed_results/f_000001.jpg", "dung": True}],
    )
    with zipfile.ZipFile(processed_dir / "patient02_clip_frames.zip", "w") as zf:
        zf.writestr("../escape.jpg", b"bad")
    _write(
        tmp_path / "database" / "video_list.json",
        [
            {
                "username": "patient02",
                "video_name": "clip.mp4",
                "stored_filename": "patient02_clip.mp4",
                "video_path": "patient_uploads/patient02_clip.mp4",
                "all_frames_data_path": "processed_results/patient02_clip_frames.json",
                "frames_zip_path": "processed_results/patient02_clip_frames.zip",
                "exercise": "Codman",
            }
        ],
    )
    client = TestClient(app)
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    researcher_login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})

    out_of_scope = client.get(
        "/videos/patient02_clip.mp4/analysis-frames",
        headers={"Authorization": f"Bearer {patient_login.json()['access_token']}"},
    )
    unsafe_zip = client.get(
        "/videos/patient02_clip.mp4/analysis-frames",
        headers={"Authorization": f"Bearer {researcher_login.json()['access_token']}"},
    )

    assert out_of_scope.status_code == 404
    assert unsafe_zip.status_code == 400


def test_backend_lists_patients_without_passwords_and_scopes_by_role(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    token = login.json()["access_token"]

    patients = client.get("/patients", headers={"Authorization": f"Bearer {token}"}).json()

    assert patients["count"] == 1
    assert patients["items"][0]["username"] == "patient01"
    assert "password" not in patients["items"][0]
    assert "email" not in patients["items"][0]


def test_backend_scopes_patient_owned_collections(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/symptoms", headers=headers).json()["count"] == 1
    assert client.get("/schedules", headers=headers).json()["count"] == 1
    assert client.get("/research-records", headers=headers).status_code == 403


def test_backend_patient_can_create_symptom_record(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post(
        "/symptoms",
        headers=headers,
        json={
            "full_name": "Edited Name",
            "patient_id": "BN001",
            "age": 44,
            "gender": "Nữ",
            "exercise": "Codman",
            "symptoms": "Đau vai khi nâng tay",
            "vas": 8,
            "time": "10:30 - 19/06/2026",
        },
    )

    assert response.status_code == 201
    item = response.json()["item"]
    assert item["username"] == "patient01"
    assert item["full_name"] == "Edited Name"
    assert item["patient_id"] == "BN001"
    assert item["vas"] == 8
    assert item["exercise"] == "Codman"
    assert item["exercises"] == ["Codman"]

    listed = client.get("/symptoms", headers=headers).json()
    assert listed["count"] == 2
    assert listed["items"][-1]["symptoms"] == "Đau vai khi nâng tay"


def test_backend_create_symptom_rejects_non_patient_and_invalid_payload(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})

    doctor_response = client.post(
        "/symptoms",
        headers={"Authorization": f"Bearer {doctor_login.json()['access_token']}"},
        json={"exercise": "Codman", "symptoms": "pain"},
    )
    missing_symptoms = client.post(
        "/symptoms",
        headers={"Authorization": f"Bearer {patient_login.json()['access_token']}"},
        json={"exercise": "Codman"},
    )
    missing_exercise = client.post(
        "/symptoms",
        headers={"Authorization": f"Bearer {patient_login.json()['access_token']}"},
        json={"symptoms": "pain"},
    )

    assert doctor_response.status_code == 403
    assert missing_symptoms.status_code == 400
    assert missing_exercise.status_code == 400


def test_backend_doctor_can_create_and_delete_evaluation(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    created = client.post(
        "/evaluations",
        headers=headers,
        json={
            "patient_username": "patient01",
            "video_name": "a.mp4",
            "exercise": "codman",
            "doctor_result": "Gần đúng",
            "errors": ["Biên độ chưa đạt"],
            "comments": "Tập chậm hơn",
            "comments_ncv": "Cần xem lại G2",
            "plan": "Tiếp tục",
        },
    )

    assert created.status_code == 201
    item = created.json()["item"]
    assert item["patient_username"] == "patient01"
    assert item["doctor_username"] == "doctor"
    assert item["doctor_result"] == "Gần đúng"
    assert item["id"]

    listed = client.get("/evaluations", headers=headers).json()
    assert listed["count"] == 1
    assert listed["items"][0]["doctor_result"] == "Gần đúng"

    deleted = client.delete(f"/evaluations/{item['id']}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/evaluations", headers=headers).json()["count"] == 0


def test_backend_doctor_can_create_and_delete_schedule_for_assigned_patient(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    created = client.post(
        "/schedules",
        headers=headers,
        json={
            "patient_username": "patient01",
            "type": "exercise",
            "exercise_name": "Codman",
            "datetime": "2026-06-20 08:30",
            "frequency": "Hàng ngày",
            "notes": "Tập nhẹ",
        },
    )

    assert created.status_code == 201
    item = created.json()["item"]
    assert item["type"] == "exercise"
    assert item["patient_username"] == "patient01"
    assert item["doctor_username"] == "doctor"

    listed = client.get("/schedules", headers=headers).json()
    assert listed["count"] == 2

    deleted = client.delete(f"/schedules/{item['id']}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/schedules", headers=headers).json()["count"] == 1


def test_backend_researcher_can_create_and_delete_research_record(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    created = client.post(
        "/research-records",
        headers=headers,
        json={
            "patient_username": "patient01",
            "subject_code": "BN001",
            "exercise": "Codman",
            "general_result": "Đúng",
            "specialist_comment": "private note",
            "diagnosis": "M75",
        },
    )

    assert created.status_code == 201
    item = created.json()["item"]
    assert item["patient_username"].startswith("SUBJ-")
    assert "specialist_comment" not in item
    assert item["id"].startswith("res_")

    listed = client.get("/research-records", headers=headers).json()
    assert listed["count"] == 3
    assert "specialist_comment" not in listed["items"][-1]

    deleted = client.delete(f"/research-records/{item['id']}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/research-records", headers=headers).json()["count"] == 2


def test_backend_admin_can_manage_users_and_change_password(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    created = client.post(
        "/admin/users",
        headers=headers,
        json={
            "username": "newdoctor",
            "full_name": "New Doctor",
            "email": "newdoctor@example.test",
            "password": "doctor123",
            "role": DOCTOR_ROLE,
            "assigned_patient_usernames": ["patient01"],
        },
    )

    assert created.status_code == 201
    assert created.json()["item"]["username"] == "newdoctor"
    assert created.json()["item"]["must_change_password"] is True
    users = client.get("/admin/users", headers=headers).json()
    assert any(item["username"] == "newdoctor" for item in users["items"])

    changed = client.post(
        "/auth/change-password",
        headers=headers,
        json={"old_password": "secret", "new_password": "newsecret", "confirm_password": "newsecret"},
    )
    assert changed.status_code == 200
    assert changed.json()["user"]["must_change_password"] is False
    assert client.post("/auth/login", json={"username": "admin", "password": "newsecret"}).status_code == 200

    deleted = client.delete("/admin/users/newdoctor", headers=headers)
    assert deleted.status_code == 200


def test_backend_admin_user_ops_write_audit_and_revoke_sessions(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    admin_login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}
    patient_login = client.post("/auth/login", json={"username": "patient01", "password": "patientpass"})
    patient_token = patient_login.json()["access_token"]
    patient_headers = {"Authorization": f"Bearer {patient_token}"}

    reset = client.post(
        "/admin/users/patient01/reset-password",
        headers=admin_headers,
        json={"password": "resetpass", "confirm_password": "resetpass"},
    )
    assert reset.status_code == 200
    assert reset.json()["item"]["must_change_password"] is True
    assert reset.json()["revoked_sessions"] == 1
    assert client.get("/auth/me", headers=patient_headers).status_code == 401

    locked_login = client.post("/auth/login", json={"username": "patient01", "password": "resetpass"})
    assert locked_login.status_code == 200
    lock = client.post("/admin/users/patient01/active", headers=admin_headers, json={"active": False})
    assert lock.status_code == 200
    assert lock.json()["item"]["active"] is False
    assert lock.json()["revoked_sessions"] == 1
    assert client.post("/auth/login", json={"username": "patient01", "password": "resetpass"}).status_code == 401
    assert client.post("/admin/users/admin/active", headers=admin_headers, json={"active": False}).status_code == 400

    unlock = client.post("/admin/users/patient01/active", headers=admin_headers, json={"active": True})
    assert unlock.status_code == 200
    assert unlock.json()["item"]["active"] is True

    another_admin_login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    another_admin_headers = {"Authorization": f"Bearer {another_admin_login.json()['access_token']}"}
    revoked = client.post(
        "/admin/sessions/revoke",
        headers=admin_headers,
        json={"confirm": "REVOKE ALL SESSIONS", "reason": "test"},
    )
    assert revoked.status_code == 200
    assert revoked.json()["scope"] == "all"
    assert revoked.json()["revoked_sessions"] >= 2
    assert client.get("/auth/me", headers=another_admin_headers).status_code == 401

    relogin = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    audit_headers = {"Authorization": f"Bearer {relogin.json()['access_token']}"}
    audit = client.get("/admin/audit-log", headers=audit_headers)
    assert audit.status_code == 200
    actions = [item["action"] for item in audit.json()["items"]]
    assert "admin_reset_password" in actions
    assert "admin_set_user_active" in actions
    assert "admin_revoke_all_sessions" in actions
    assert (tmp_path / "backups" / "admin_ops").exists()


def test_backend_admin_ops_are_admin_only(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    doctor_login = client.post("/auth/login", json={"username": "doctor", "password": "doctorpass"})
    headers = {"Authorization": f"Bearer {doctor_login.json()['access_token']}"}

    assert client.get("/admin/audit-log", headers=headers).status_code == 403
    assert client.post("/admin/users/patient01/active", headers=headers, json={"active": False}).status_code == 403
    assert client.post(
        "/admin/users/patient01/reset-password",
        headers=headers,
        json={"password": "resetpass", "confirm_password": "resetpass"},
    ).status_code == 403
    assert client.post(
        "/admin/sessions/revoke",
        headers=headers,
        json={"confirm": "REVOKE ALL SESSIONS"},
    ).status_code == 403


def test_backend_researcher_gets_pseudonymized_records(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "researcher", "password": "researchpass"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    videos = client.get("/videos", headers=headers).json()
    patients = client.get("/patients", headers=headers).json()
    research = client.get("/research-records", headers=headers).json()
    schedules = client.get("/schedules", headers=headers)

    assert videos["count"] == 2
    assert videos["items"][0]["username"].startswith("SUBJ-")
    assert "full_name" not in videos["items"][0]
    assert patients["items"][0]["username"].startswith("SUBJ-")
    assert "email" not in patients["items"][0]
    assert "specialist_comment" not in research["items"][0]
    assert schedules.status_code == 403


def test_backend_logout_revokes_token(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_backend_import_does_not_load_streamlit_app():
    code = (
        "import sys;"
        "import backend.main;"
        "assert 'app' not in sys.modules;"
        "assert 'streamlit' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_backend_allows_configured_local_frontend_origin(tmp_path, monkeypatch):
    _configure_tmp_backend(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.options(
        "/videos",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
