"""P1-13 `ui/api_client.py`: requests, auth header, problem+json errors, base URL."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ui.api_client import (
    DEFAULT_API_BASE_URL,
    ApiClient,
    ApiError,
    api_base_url,
    connect,
)

BASE = "http://api.test"
Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler, token: str | None = "tok") -> ApiClient:  # noqa: S107
    return ApiClient(connect(BASE, transport=httpx.MockTransport(handler)), token)


class Recorder:
    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response or httpx.Response(200, json={"ok": True})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def body(self) -> object:
        return json.loads(self.last.content)


def test_sends_bearer_token_under_api_prefix() -> None:
    rec = Recorder()
    assert make_client(rec).me() == {"ok": True}
    assert rec.last.method == "GET"
    assert str(rec.last.url) == f"{BASE}/api/v1/me"
    assert rec.last.headers["Authorization"] == "Bearer tok"


def test_no_auth_header_without_token() -> None:
    rec = Recorder()
    make_client(rec, token=None).login(email="a@x.com", password="secret123")
    assert "Authorization" not in rec.last.headers
    assert rec.body() == {"email": "a@x.com", "password": "secret123"}


def test_register_body() -> None:
    rec = Recorder()
    make_client(rec, None).register(
        email="a@x.com", password="pw123456", display_name="A", country="US", age_confirmed=True
    )
    assert rec.last.url.path == "/api/v1/auth/register"
    assert rec.body() == {
        "email": "a@x.com",
        "password": "pw123456",
        "display_name": "A",
        "country": "US",
        "age_confirmed": True,
    }


@pytest.mark.parametrize(
    ("call", "method", "path", "params", "body"),
    [
        (lambda c: c.update_me(country="KR"), "PATCH", "/me", {}, {"country": "KR"}),
        (lambda c: c.preferences(), "GET", "/me/preferences", {}, None),
        (
            lambda c: c.save_preferences({"languages": ["en"]}),
            "PUT",
            "/me/preferences",
            {},
            {"languages": ["en"]},
        ),
        (lambda c: c.providers("US"), "GET", "/providers", {"country": "US"}, None),
        (
            lambda c: c.search_titles("squid", "tv", 5),
            "GET",
            "/titles/search",
            {"q": "squid", "type": "tv", "limit": "5"},
            None,
        ),
        (
            lambda c: c.submit_seeds(
                [{"tmdb_id": 1, "type": "movie", "title": "X"}, {"tmdb_id": 2, "type": "tv"}]
            ),
            "POST",
            "/onboarding/seeds",
            {},
            {"seeds": [{"tmdb_id": 1, "type": "movie"}, {"tmdb_id": 2, "type": "tv"}]},
        ),
        (
            lambda c: c.rate(7, "movie", 4),
            "POST",
            "/interactions",
            {},
            {"tmdb_id": 7, "type": "movie", "kind": "rating", "value": 4},
        ),
        (
            lambda c: c.not_interested(7, "tv"),
            "POST",
            "/interactions",
            {},
            {"tmdb_id": 7, "type": "tv", "kind": "not_interested"},
        ),
        (lambda c: c.recommendations(10), "GET", "/recommendations", {"n": "10"}, None),
    ],
)
def test_endpoints(
    call: Callable[[ApiClient], object],
    method: str,
    path: str,
    params: dict[str, str],
    body: object,
) -> None:
    rec = Recorder()
    call(make_client(rec))
    assert rec.last.method == method
    assert rec.last.url.path == "/api/v1" + path
    assert dict(rec.last.url.params) == params
    if body is None:
        assert rec.last.content == b""
    else:
        assert rec.body() == body


def test_recommendations_sends_exclude_as_repeated_params() -> None:
    rec = Recorder()
    make_client(rec).recommendations(10, ["movie:1", "tv:2"])
    assert rec.last.url.params.multi_items() == [
        ("n", "10"),
        ("exclude", "movie:1"),
        ("exclude", "tv:2"),
    ]


def test_current_seeds() -> None:
    rec = Recorder()
    make_client(rec).current_seeds()
    assert (rec.last.method, rec.last.url.path) == ("GET", "/api/v1/onboarding/seeds")


def test_problem_json_becomes_api_error() -> None:
    problem = {
        "type": "about:blank",
        "title": "Unprocessable Content",
        "status": 422,
        "detail": "Request validation failed.",
        "errors": [{"loc": ["body", "platforms", 0], "msg": "unknown provider", "type": "x"}],
    }
    rec = Recorder(
        httpx.Response(422, json=problem, headers={"content-type": "application/problem+json"})
    )
    with pytest.raises(ApiError) as info:
        make_client(rec).save_preferences({"platforms": [99999]})
    err = info.value
    assert err.status == 422
    assert err.title == "Unprocessable Content"
    assert err.errors == problem["errors"]
    assert err.message == "Request validation failed. platforms.0: unknown provider."
    assert str(err) == err.message
    assert not err.unauthorized


def test_401_is_flagged_unauthorized() -> None:
    rec = Recorder(httpx.Response(401, json={"title": "Unauthorized", "status": 401}))
    with pytest.raises(ApiError) as info:
        make_client(rec).me()
    assert info.value.unauthorized
    assert info.value.message == "Unauthorized"


def test_non_json_error_body() -> None:
    rec = Recorder(httpx.Response(502, text="Bad gateway"))
    with pytest.raises(ApiError) as info:
        make_client(rec).me()
    assert info.value.status == 502
    assert info.value.message == "Bad Gateway"


def test_unreachable_api() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError) as info:
        make_client(boom).me()
    assert info.value.status == 0
    assert "make api" in info.value.message
    assert BASE in info.value.message


def test_non_object_success_body_is_an_error() -> None:
    rec = Recorder(httpx.Response(200, json=[1, 2]))
    with pytest.raises(ApiError):
        make_client(rec).me()


def test_base_url_from_env_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("API_BASE_URL=http://from-file:1\n")
    assert api_base_url({"API_BASE_URL": "http://from-env:2/"}, env_file) == "http://from-env:2"


def test_base_url_from_dotenv_strips_comments(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nAPP_ENV=local\nAPI_BASE_URL=http://localhost:9000  # used by UI\n"
    )
    assert api_base_url({}, env_file) == "http://localhost:9000"


def test_base_url_default(tmp_path: Path) -> None:
    assert api_base_url({}, tmp_path / "missing.env") == DEFAULT_API_BASE_URL
    empty = tmp_path / ".env"
    empty.write_text("API_BASE_URL=\n")
    assert api_base_url({}, empty) == DEFAULT_API_BASE_URL
