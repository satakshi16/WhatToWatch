from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.api.deps import AdminUser
from app.core.config import Settings
from app.core.security import TokenType, create_token, decode_token
from app.db.models import User, UserStatus

PROBLEM = "application/problem+json"
PASSWORD = "s3cret-passw0rd"  # pragma: allowlist secret


@pytest.fixture
def api(migrated_engine: Engine, client: TestClient) -> TestClient:
    """Client against a migrated DB."""
    return client


def _register(client: TestClient, **overrides: Any) -> Any:
    body: dict[str, Any] = {
        "email": "Priya@Example.com",
        "password": PASSWORD,
        "display_name": "Priya",
        "country": "IN",
        "age_confirmed": True,
        **overrides,
    }
    return client.post("/api/v1/auth/register", json=body)


def _login(client: TestClient, email: str = "priya@example.com", password: str = PASSWORD) -> Any:
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _db_user(engine: Engine, email: str = "priya@example.com") -> User:
    with Session(engine) as s:
        user = s.scalar(select(User).where(User.email == email))
        assert user is not None
        return user


# --- register ---


def test_register_creates_user_and_returns_tokens(
    api: TestClient, migrated_engine: Engine, settings: Settings
) -> None:
    resp = _register(api)
    assert resp.status_code == 201
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 1800
    user = body["user"]
    assert user["email"] == "priya@example.com"
    assert user["country"] == "IN"
    assert user["timezone"] == "Asia/Kolkata"  # derived from country
    assert user["is_admin"] is False
    assert user["status"] == "active"
    assert user["last_seen_at"] is not None
    assert PASSWORD not in resp.text
    assert "password_hash" not in resp.text
    claims = decode_token(settings, body["access_token"], TokenType.ACCESS)
    assert str(claims.sub) == user["id"]
    decode_token(settings, body["refresh_token"], TokenType.REFRESH)

    db_user = _db_user(migrated_engine)
    assert db_user.password_hash.startswith("$argon2id$")
    assert PASSWORD not in db_user.password_hash


def test_register_explicit_timezone_and_country_case(api: TestClient) -> None:
    resp = _register(api, country="us", timezone="America/Los_Angeles")
    assert resp.status_code == 201
    assert resp.json()["user"]["country"] == "US"
    assert resp.json()["user"]["timezone"] == "America/Los_Angeles"


def test_duplicate_email_case_insensitive_409(api: TestClient) -> None:
    assert _register(api, email="A@x.com").status_code == 201
    resp = _register(api, email="a@X.com")
    assert resp.status_code == 409
    assert resp.headers["content-type"] == PROBLEM
    body = resp.json()
    assert body["status"] == 409
    assert body["instance"] == "/api/v1/auth/register"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"password": "short7!"}, "password"),  # pragma: allowlist secret
        ({"email": "not-an-email"}, "email"),
        ({"country": "XX"}, "country"),
        ({"country": "USA"}, "country"),
        ({"timezone": "Mars/Base"}, "timezone"),
        ({"display_name": "   "}, "display_name"),
        ({"age_confirmed": False}, "age_confirmed"),
        ({"is_admin": True}, "is_admin"),  # extra fields forbidden
    ],
)
def test_register_validation_422(api: TestClient, overrides: dict[str, Any], field: str) -> None:
    resp = _register(api, **overrides)
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM
    locs = [e["loc"] for e in resp.json()["errors"]]
    assert ["body", field] in locs


def test_register_requires_age_confirmation(api: TestClient) -> None:
    body = {
        "email": "x@example.com",
        "password": PASSWORD,
        "display_name": "X",
        "country": "US",
    }
    resp = api.post("/api/v1/auth/register", json=body)
    assert resp.status_code == 422


def test_validation_errors_do_not_echo_password(api: TestClient) -> None:
    resp = _register(api, password="tiny")
    assert "tiny" not in resp.text


# --- login ---


def test_login_returns_both_tokens_and_access_works_on_me(
    api: TestClient, settings: Settings
) -> None:
    _register(api)
    resp = _login(api, email="  PRIYA@example.COM ")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"access_token", "refresh_token", "token_type", "expires_in"}
    decode_token(settings, body["refresh_token"], TokenType.REFRESH)
    me = api.get("/api/v1/me", headers=_auth(body["access_token"]))
    assert me.status_code == 200
    assert me.json()["email"] == "priya@example.com"


@pytest.mark.parametrize(
    ("email", "password"),
    [("priya@example.com", "wrong-password"), ("nobody@example.com", PASSWORD), ("bad", "x")],
)
def test_bad_credentials_generic_401(api: TestClient, email: str, password: str) -> None:
    _register(api)
    resp = _login(api, email=email, password=password)
    assert resp.status_code == 401
    assert resp.headers["content-type"] == PROBLEM
    assert resp.json()["detail"] == "Incorrect email or password."


def test_deleted_user_cannot_login_or_use_token(api: TestClient, migrated_engine: Engine) -> None:
    token = _register(api).json()["access_token"]
    with Session(migrated_engine) as s:
        user = s.scalar(select(User))
        assert user is not None
        user.status = UserStatus.DELETED
        s.commit()
    assert _login(api).status_code == 401
    assert api.get("/api/v1/me", headers=_auth(token)).status_code == 401


def test_login_updates_last_seen(api: TestClient, migrated_engine: Engine) -> None:
    _register(api)
    old = datetime.now(UTC) - timedelta(days=3)
    with Session(migrated_engine) as s:
        user = s.scalar(select(User))
        assert user is not None
        user.last_seen_at = old
        s.commit()
    assert _login(api).status_code == 200
    seen = _db_user(migrated_engine).last_seen_at
    assert seen is not None
    assert seen > old + timedelta(days=2)


def test_authenticated_request_updates_stale_last_seen(
    api: TestClient, migrated_engine: Engine
) -> None:
    token = _register(api).json()["access_token"]
    old = datetime.now(UTC) - timedelta(hours=1)
    with Session(migrated_engine) as s:
        user = s.scalar(select(User))
        assert user is not None
        user.last_seen_at = old
        s.commit()
    assert api.get("/api/v1/me", headers=_auth(token)).status_code == 200
    seen = _db_user(migrated_engine).last_seen_at
    assert seen is not None
    assert seen > old


def test_plaintext_password_and_email_never_logged(
    api: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    _register(api)
    _login(api)
    _login(api, password="wrong-password")  # pragma: allowlist secret
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "user_registered" in messages
    assert "login_failed" in messages
    for record in caplog.records:
        text = record.getMessage()
        assert PASSWORD not in text
        assert "wrong-password" not in text
        assert "priya@example.com" not in text.lower()


# --- /me auth ---


def test_me_requires_token(api: TestClient) -> None:
    resp = api.get("/api/v1/me")
    assert resp.status_code == 401
    assert resp.headers["content-type"] == PROBLEM
    assert resp.headers["www-authenticate"] == "Bearer"


def test_expired_access_token_401(api: TestClient, settings: Settings) -> None:
    user_id = uuid.UUID(_register(api).json()["user"]["id"])
    token = create_token(
        settings, user_id, TokenType.ACCESS, now=datetime.now(UTC) - timedelta(minutes=31)
    )
    resp = api.get("/api/v1/me", headers=_auth(token))
    assert resp.status_code == 401
    assert resp.headers["content-type"] == PROBLEM


def test_invalid_tokens_401(api: TestClient, settings: Settings) -> None:
    body = _register(api).json()
    user_id = uuid.UUID(body["user"]["id"])
    forged = jwt.encode(
        {"sub": str(user_id), "exp": 2**33, "iat": 0, "jti": "x", "type": "access"},
        "attacker-key-attacker-key-attacker-key",
        algorithm="HS256",
        headers=jwt.get_unverified_header(body["access_token"]),
    )
    unknown_user = create_token(settings, uuid.uuid4(), TokenType.ACCESS)
    for token in ["garbage", body["refresh_token"], forged, unknown_user]:
        resp = api.get("/api/v1/me", headers=_auth(token))
        assert resp.status_code == 401, token
    resp = api.get("/api/v1/me", headers={"Authorization": f"Basic {body['access_token']}"})
    assert resp.status_code == 401


# --- PATCH /me ---


def test_patch_me_updates_profile(api: TestClient) -> None:
    token = _register(api).json()["access_token"]
    resp = api.patch(
        "/api/v1/me",
        headers=_auth(token),
        json={"display_name": "  Priya S  ", "country": "kr", "timezone": "Asia/Seoul"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert (body["display_name"], body["country"], body["timezone"]) == (
        "Priya S",
        "KR",
        "Asia/Seoul",
    )
    assert body.pop("removed_platforms") == []  # P1-07: no platforms were set
    assert api.get("/api/v1/me", headers=_auth(token)).json() == body


def test_patch_me_partial_keeps_other_fields(api: TestClient) -> None:
    token = _register(api).json()["access_token"]
    resp = api.patch("/api/v1/me", headers=_auth(token), json={"country": "US"})
    assert resp.status_code == 200
    assert resp.json()["country"] == "US"
    assert resp.json()["timezone"] == "Asia/Kolkata"  # timezone not changed implicitly
    assert resp.json()["display_name"] == "Priya"


@pytest.mark.parametrize(
    "body",
    [
        {"country": "ZZ"},
        {"timezone": "Not/AZone"},
        {"display_name": ""},
        {"display_name": None},
        {"email": "new@example.com"},
        {"is_admin": True},
    ],
)
def test_patch_me_validation_422(api: TestClient, body: dict[str, Any]) -> None:
    token = _register(api).json()["access_token"]
    resp = api.patch("/api/v1/me", headers=_auth(token), json=body)
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM


def test_patch_me_requires_auth(api: TestClient) -> None:
    assert api.patch("/api/v1/me", json={"country": "US"}).status_code == 401


# --- require_admin ---


def test_require_admin(app: FastAPI, api: TestClient, migrated_engine: Engine) -> None:
    @app.get("/api/v1/_admin-probe")
    def probe(user: AdminUser) -> dict[str, str]:
        return {"id": str(user.id)}

    token = _register(api).json()["access_token"]
    resp = api.get("/api/v1/_admin-probe", headers=_auth(token))
    assert resp.status_code == 403
    assert resp.headers["content-type"] == PROBLEM
    with Session(migrated_engine) as s:
        user = s.scalar(select(User))
        assert user is not None
        user.is_admin = True
        s.commit()
    assert api.get("/api/v1/_admin-probe", headers=_auth(token)).status_code == 200
    assert api.get("/api/v1/_admin-probe").status_code == 401
