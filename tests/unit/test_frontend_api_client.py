import httpx

from frontend.api_client import FrontendApiClient, FrontendApiConfig, FrontendApiError


def _client(handler, *, enabled=True, token=None):
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    config = FrontendApiConfig(base_url="http://backend.local", enabled=enabled)
    return FrontendApiClient(config, token=token, client=http_client)


def test_config_from_env_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("REHAB_BACKEND_URL", "http://127.0.0.1:8000/")
    monkeypatch.delenv("REHAB_FRONTEND_USE_BACKEND", raising=False)
    monkeypatch.delenv("REHAB_USE_BACKEND_API", raising=False)

    config = FrontendApiConfig.from_env()

    assert config.base_url == "http://127.0.0.1:8000"
    assert config.enabled is False

    monkeypatch.setenv("REHAB_FRONTEND_USE_BACKEND", "1")
    assert FrontendApiConfig.from_env().enabled is True


def test_login_posts_credentials_and_returns_user():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode("utf-8")
        return httpx.Response(200, json={"access_token": "tok", "user": {"username": "admin"}})

    client = _client(handler)

    response = client.login("admin", "secret")

    assert response["access_token"] == "tok"
    assert seen["url"] == "http://backend.local/auth/login"
    assert '"username":"admin"' in seen["body"]


def test_authorized_list_items_sends_bearer_token():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"items": [{"video_name": "a.mp4"}], "count": 1})

    client = _client(handler, token="abc")

    assert client.videos() == [{"video_name": "a.mp4"}]
    assert seen["auth"] == "Bearer abc"


def test_client_raises_on_backend_error_detail():
    def handler(_request):
        return httpx.Response(403, json={"detail": "denied"})

    client = _client(handler)

    try:
        client.videos()
    except FrontendApiError as exc:
        assert exc.status_code == 403
        assert "denied" in str(exc)
    else:
        raise AssertionError("expected FrontendApiError")


def test_disabled_client_refuses_requests():
    client = _client(lambda request: httpx.Response(200, json={}), enabled=False)

    try:
        client.health()
    except FrontendApiError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("expected FrontendApiError")
