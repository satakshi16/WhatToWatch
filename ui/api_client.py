"""Small synchronous client for the WhatToWatch API (P1-13).

The UI talks to the backend only through HTTP: no `app.*` imports (enforced by a test).
Errors come back as RFC 9457 problem+json and are raised as `ApiError`.
The caller keeps the access token (in `st.session_state`) and passes it in.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

API_PREFIX = "/api/v1"
DEFAULT_API_BASE_URL = "http://localhost:8000"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
# Recommendations and seeding may fetch from TMDB / embed on the fly, so reads get room.
TIMEOUT = httpx.Timeout(60.0, connect=5.0)

JSON = dict[str, Any]


def _dotenv_value(path: Path, key: str) -> str | None:
    """Read one `KEY=value` from a .env file (inline `# comments` and quotes stripped)."""
    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip().removeprefix("export ").strip() != key:
            continue
        value = value.split(" #", 1)[0].strip().strip("'\"")
        return value or None
    return None


def api_base_url(env: Mapping[str, str] | None = None, env_file: Path = ENV_FILE) -> str:
    """`API_BASE_URL` from the environment, else `.env`, else the local default."""
    env = os.environ if env is None else env
    value = env.get("API_BASE_URL") or _dotenv_value(env_file, "API_BASE_URL")
    return (value or DEFAULT_API_BASE_URL).rstrip("/")


@dataclass
class ApiError(Exception):
    """A failed API call. `status == 0` means the API could not be reached."""

    status: int
    title: str
    detail: str | None = None
    errors: list[JSON] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message

    @property
    def unauthorized(self) -> bool:
        return self.status == 401

    @property
    def message(self) -> str:
        text = self.detail or self.title
        problems = [_describe_error(err) for err in self.errors]
        return " ".join([text, *problems]) if problems else text


def _describe_error(err: JSON) -> str:
    loc = [str(part) for part in err.get("loc", []) if part not in ("body", "query")]
    msg = str(err.get("msg", "invalid value"))
    return f"{'.'.join(loc)}: {msg}." if loc else f"{msg}."


def _error_from_response(response: httpx.Response) -> ApiError:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return ApiError(response.status_code, response.reason_phrase or "Request failed")
    errors = body.get("errors")
    return ApiError(
        status=response.status_code,
        title=str(body.get("title") or response.reason_phrase or "Request failed"),
        detail=body.get("detail"),
        errors=errors if isinstance(errors, list) else [],
    )


def connect(base_url: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """A pooled HTTP client for `base_url` (share it across Streamlit reruns)."""
    return httpx.Client(base_url=base_url + API_PREFIX, timeout=TIMEOUT, transport=transport)


class ApiClient:
    def __init__(self, http: httpx.Client, token: str | None = None) -> None:
        self._http = http
        self.token = token

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: JSON | None = None,
        params: Mapping[str, str | int] | None = None,
    ) -> JSON:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            response = self._http.request(method, path, json=json, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise ApiError(
                0,
                "API unreachable",
                f"Could not reach the WhatToWatch API at {self._http.base_url} "
                f"({type(exc).__name__}). Is `make api` running?",
            ) from exc
        if response.is_error:
            raise _error_from_response(response)
        body = response.json()
        if not isinstance(body, dict):
            raise ApiError(response.status_code, "Unexpected response from the API")
        return body

    # --- auth & profile (P1-06) ---
    def register(
        self, *, email: str, password: str, display_name: str, country: str, age_confirmed: bool
    ) -> JSON:
        return self._request(
            "POST",
            "/auth/register",
            json={
                "email": email,
                "password": password,
                "display_name": display_name,
                "country": country,
                "age_confirmed": age_confirmed,
            },
        )

    def login(self, *, email: str, password: str) -> JSON:
        return self._request("POST", "/auth/login", json={"email": email, "password": password})

    def me(self) -> JSON:
        return self._request("GET", "/me")

    def update_me(self, **fields: str) -> JSON:
        """`PATCH /me`; the response includes `removed_platforms` after a country change."""
        return self._request("PATCH", "/me", json=dict(fields))

    # --- preferences & providers (P1-07) ---
    def preferences(self) -> JSON:
        return self._request("GET", "/me/preferences")

    def save_preferences(self, preferences: JSON) -> JSON:
        return self._request("PUT", "/me/preferences", json=preferences)

    def providers(self, country: str) -> JSON:
        return self._request("GET", "/providers", params={"country": country})

    # --- seeding (P1-08, P1-09) ---
    def search_titles(self, query: str, type_: str = "all", limit: int = 10) -> JSON:
        return self._request(
            "GET", "/titles/search", params={"q": query, "type": type_, "limit": limit}
        )

    def submit_seeds(self, seeds: list[JSON]) -> JSON:
        refs = [{"tmdb_id": s["tmdb_id"], "type": s["type"]} for s in seeds]
        return self._request("POST", "/onboarding/seeds", json={"seeds": refs})

    def current_seeds(self) -> JSON:
        return self._request("GET", "/onboarding/seeds")

    # --- feedback (P1-09) ---
    def rate(self, tmdb_id: int, type_: str, stars: int) -> JSON:
        return self._request(
            "POST",
            "/interactions",
            json={"tmdb_id": tmdb_id, "type": type_, "kind": "rating", "value": stars},
        )

    def not_interested(self, tmdb_id: int, type_: str) -> JSON:
        return self._request(
            "POST",
            "/interactions",
            json={"tmdb_id": tmdb_id, "type": type_, "kind": "not_interested"},
        )

    # --- recommendations (P1-12) ---
    def recommendations(self, n: int = 10, exclude: Sequence[str] = ()) -> JSON:
        """`exclude`: `"movie:123"` / `"tv:45"` keys to leave out (sent as repeated params)."""
        params: dict[str, Any] = {"n": n}
        if exclude:
            params["exclude"] = list(exclude)
        return self._request("GET", "/recommendations", params=params)
