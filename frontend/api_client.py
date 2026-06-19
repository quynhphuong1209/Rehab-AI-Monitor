"""Small HTTP client used by the Streamlit frontend to call the backend API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_BACKEND_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class FrontendApiConfig:
    base_url: str = ""
    enabled: bool = False
    timeout_seconds: float = DEFAULT_BACKEND_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "FrontendApiConfig":
        base_url = (
            os.getenv("REHAB_BACKEND_URL")
            or os.getenv("REHAB_API_BASE_URL")
            or ""
        ).strip()
        enabled_raw = (
            os.getenv("REHAB_FRONTEND_USE_BACKEND")
            or os.getenv("REHAB_USE_BACKEND_API")
            or ""
        ).strip().lower()
        enabled = enabled_raw in {"1", "true", "yes", "on"} and bool(base_url)
        timeout_raw = os.getenv("REHAB_BACKEND_TIMEOUT_SECONDS", "")
        try:
            timeout_seconds = float(timeout_raw) if timeout_raw else DEFAULT_BACKEND_TIMEOUT_SECONDS
        except ValueError:
            timeout_seconds = DEFAULT_BACKEND_TIMEOUT_SECONDS
        return cls(base_url=base_url.rstrip("/"), enabled=enabled, timeout_seconds=max(0.5, timeout_seconds))


@dataclass(frozen=True)
class FrontendApiError(Exception):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} ({self.status_code})"


class FrontendApiClient:
    def __init__(
        self,
        config: FrontendApiConfig,
        *,
        token: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.token = token
        self._client = client

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def with_token(self, token: str | None) -> "FrontendApiClient":
        return FrontendApiClient(self.config, token=token, client=self._client)

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not self.config.enabled:
            raise FrontendApiError("backend API is disabled")
        url = f"{self.config.base_url}/{path.lstrip('/')}"
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.update(self._headers())
        close_client = False
        client = self._client
        if client is None:
            client = httpx.Client(timeout=self.config.timeout_seconds)
            close_client = True
        try:
            response = client.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise FrontendApiError(f"backend API request failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                detail = str(body.get("detail") or "")
            except ValueError:
                detail = response.text[:200]
            raise FrontendApiError(detail or "backend API error", response.status_code)
        try:
            data = response.json()
        except ValueError as exc:
            raise FrontendApiError("backend API returned non-JSON response", response.status_code) from exc
        if not isinstance(data, dict):
            raise FrontendApiError("backend API returned unexpected JSON shape", response.status_code)
        return data

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/auth/login", json={"username": username, "password": password})

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    def logout(self) -> dict[str, Any]:
        return self._request("POST", "/auth/logout")

    def list_items(self, resource: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/{resource.strip('/')}")
        items = data.get("items")
        if not isinstance(items, list):
            raise FrontendApiError("backend API response missing items")
        return [item for item in items if isinstance(item, dict)]

    def videos(self) -> list[dict[str, Any]]:
        return self.list_items("videos")

    def evaluations(self) -> list[dict[str, Any]]:
        return self.list_items("evaluations")

    def patients(self) -> list[dict[str, Any]]:
        return self.list_items("patients")

    def symptoms(self) -> list[dict[str, Any]]:
        return self.list_items("symptoms")

    def schedules(self) -> list[dict[str, Any]]:
        return self.list_items("schedules")

    def research_records(self) -> list[dict[str, Any]]:
        return self.list_items("research-records")
