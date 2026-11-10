"""P1-14 `make seed-demo`: persona demo users created through the real API.

The API runs in-process over the P1-11 fixture catalog (TMDB is never called: the fixture
catalog caches the US provider list and availability, and every seed is in the catalog).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import AppEnv, Settings
from app.db.models import Interaction, OnboardingStep, User, UserPreferences
from app.db.models.enums import Era, TitleType
from app.schemas.auth import RegisterRequest
from app.schemas.feedback import SeedsIn
from app.schemas.preferences import PreferencesIn
from scripts import seed_demo
from scripts.seed_demo import (
    DEMO_PASSWORD,
    PERSONAS,
    Persona,
    SeedDemoError,
    describe,
    match_platforms,
)
from tests.recsys_catalog import NETFLIX, PRIME, FixtureCatalog, build_fixture_catalog

M, T = TitleType.MOVIE, TitleType.TV

# Embedded fixture-catalog movies (ids 1000 + i; i % 3 != 0 → movie, i % 10 != 9 → embedded).
DEMO = Persona(
    key="demo-ana",
    display_name="Ana",
    country="US",
    timezone="America/Chicago",
    languages=("en", "ko", "hi", "ja"),
    platforms=(("netflix",), ("Prime Video", "Amazon Prime Video"), ("Max", "HBO Max")),
    seeds=((1001, M), (1002, M), (1004, M)),
)


@pytest.fixture
def catalog(db_session: Session, settings: Settings) -> FixtureCatalog:
    return build_fixture_catalog(db_session, settings.embeddings_dir)


@pytest.fixture
def api(app: FastAPI, catalog: FixtureCatalog) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _seed_ratings(db: Session, email: str) -> int:
    user = db.scalars(select(User).where(User.email == email)).one()
    return db.scalar(select(func.count()).where(Interaction.user_id == user.id)) or 0


# --- persona definitions (§1) ---


@pytest.mark.parametrize("persona", PERSONAS, ids=lambda p: p.key)
def test_default_personas_pass_api_validation(persona: Persona) -> None:
    RegisterRequest(
        email=persona.email,
        password=DEMO_PASSWORD,
        display_name=persona.display_name,
        country=persona.country,
        timezone=persona.timezone,
        age_confirmed=True,
    )
    PreferencesIn.model_validate(persona.preferences([8]))
    SeedsIn.model_validate({"seeds": [{"tmdb_id": i, "type": t.value} for i, t in persona.seeds]})
    assert len(set(persona.seeds)) == len(persona.seeds)


def test_default_personas_follow_the_spec() -> None:
    by_key = {p.key: p for p in PERSONAS}
    assert set(by_key) == {"priya", "marcus", "ji-woo"}
    priya, marcus, jiwoo = by_key["priya"], by_key["marcus"], by_key["ji-woo"]
    assert (priya.country, priya.languages) == ("IN", ("hi", "en", "ko"))
    assert [a[0] for a in priya.platforms] == ["Netflix", "Amazon Prime Video", "JioCinema"]
    assert (marcus.country, marcus.content_types, marcus.max_runtime) == ("US", (M,), 120)
    assert [a[0] for a in marcus.platforms] == ["Netflix", "Max"]
    assert jiwoo.country == "KR"
    assert jiwoo.era is Era.ANY
    # Classics + anime: Totoro and Akira (1988) next to Spirited Away / Your Name.
    assert {(8392, M), (149, M), (129, M), (372058, M)} <= set(jiwoo.seeds)


def test_match_platforms_uses_aliases_case_insensitively() -> None:
    providers = [
        {"provider_id": 8, "name": "Netflix"},
        {"provider_id": 1899, "name": "HBO Max"},
        {"provider_id": 9, "name": "Amazon Prime Video"},
    ]
    matched, missing = match_platforms(DEMO, providers)
    assert matched == {"Netflix": 8, "Amazon Prime Video": 9, "HBO Max": 1899}
    assert missing == []
    matched, missing = match_platforms(DEMO, providers[:1])
    assert matched == {"Netflix": 8}
    assert missing == ["Prime Video", "Max"]


# --- end to end through the API ---


def test_seed_demo_creates_onboarded_user_with_recommendations(
    api: TestClient, db_session: Session
) -> None:
    [result] = seed_demo.seed_demo(api, [DEMO])

    assert result.created
    assert result.seeded
    assert result.platforms == {"Netflix": NETFLIX, "Amazon Prime Video": PRIME}
    assert result.missing_platforms == ["Max"]
    assert result.unembedded_seeds == []
    assert result.recommendations
    assert len(result.recommendations) == 10
    assert result.recommendations_error is None

    user = db_session.scalars(select(User).where(User.email == DEMO.email)).one()
    assert (user.display_name, user.country, user.timezone) == ("Ana", "US", "America/Chicago")
    prefs = db_session.get(UserPreferences, user.id)
    assert prefs is not None
    assert prefs.onboarding_step is OnboardingStep.DONE
    assert prefs.languages == ["en", "ko", "hi", "ja"]
    assert _seed_ratings(db_session, DEMO.email) == 3

    # The demo user can log in with the printed password.
    login = api.post("/api/v1/auth/login", json={"email": DEMO.email, "password": DEMO_PASSWORD})
    assert login.status_code == 200


def test_seed_demo_is_idempotent(api: TestClient, db_session: Session) -> None:
    seed_demo.seed_demo(api, [DEMO], recs=False)
    [again] = seed_demo.seed_demo(api, [DEMO], recs=False)

    assert not again.created
    assert not again.seeded
    assert again.recommendations is None
    assert _seed_ratings(db_session, DEMO.email) == 3
    assert db_session.scalar(select(func.count()).select_from(User)) == 1


def test_seed_demo_reapplies_preferences_on_rerun(api: TestClient, db_session: Session) -> None:
    seed_demo.seed_demo(api, [DEMO], recs=False)
    narrower = Persona(**{**DEMO.__dict__, "content_types": (M,), "max_runtime": 100})
    seed_demo.seed_demo(api, [narrower], recs=False)

    user = db_session.scalars(select(User).where(User.email == DEMO.email)).one()
    prefs = db_session.get(UserPreferences, user.id)
    assert prefs is not None
    db_session.refresh(prefs)
    assert prefs.max_runtime == 100
    assert prefs.content_types == [M]


def test_existing_account_with_other_password_fails_clearly(api: TestClient) -> None:
    seed_demo.seed_demo(api, [DEMO], password="another password", recs=False)
    with pytest.raises(SeedDemoError, match="exists with a different password"):
        seed_demo.seed_demo(api, [DEMO], recs=False)


def test_api_errors_name_the_persona_and_step(api: TestClient) -> None:
    # 999999 is not in the catalog and the fixture TMDB client can't fetch it → 4xx/5xx.
    unknown_seed = Persona(**{**DEMO.__dict__, "seeds": ((1001, M), (1002, M), (999_999, M))})
    with pytest.raises(SeedDemoError, match=r"Ana: submit seeds failed \(HTTP"):
        seed_demo.seed_demo(api, [unknown_seed], recs=False)


def test_recommendation_errors_are_reported_not_raised(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get = api.get

    def get(url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/recommendations"):
            return httpx.Response(503, json={"status": 503, "detail": "Index not loaded."})
        response: httpx.Response = real_get(url, **kwargs)
        return response

    monkeypatch.setattr(api, "get", get)
    [result] = seed_demo.seed_demo(api, [DEMO])
    assert result.seeded
    assert result.recommendations is None
    assert result.recommendations_error == "HTTP 503: Index not loaded."


def test_describe_summarizes_the_result(api: TestClient) -> None:
    [result] = seed_demo.seed_demo(api, [DEMO])
    text = describe(result)
    assert "Ana (US, en/ko/hi/ja): demo-ana@demo.example.com [created]" in text
    assert "services: Netflix, Amazon Prime Video (not offered in US: Max)" in text
    assert "seeds: submitted" in text
    assert "recommendations (10): " in text
    assert "(+5)" in text

    result.recommendations, result.recommendations_error = None, "HTTP 503: not ready"
    assert "recommendations: unavailable (HTTP 503: not ready)" in describe(result)
    result.recommendations_error = None
    assert "recommendations" not in describe(result)


# --- CLI ---


def test_main_refuses_prod(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    prod = settings.model_copy(update={"app_env": AppEnv.PROD})
    assert seed_demo.main([], settings=prod) == 2
    assert "APP_ENV=prod" in capsys.readouterr().err


def test_main_seeds_selected_personas(
    api: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    other = Persona(**{**DEMO.__dict__, "key": "demo-ben", "display_name": "Ben"})
    monkeypatch.setattr(seed_demo, "PERSONAS", (DEMO, other))

    @contextmanager
    def in_process(_settings: Settings, api_url: str | None) -> Iterator[httpx.Client]:
        assert api_url is None
        yield api

    monkeypatch.setattr(seed_demo, "api_client", in_process)
    assert seed_demo.main(["--only", "demo-ben", "--no-recs"], settings=settings) == 0
    out = capsys.readouterr().out
    assert "Ben (US" in out
    assert "Ana (US" not in out
    assert f"password: {DEMO_PASSWORD}" in out


def test_main_continues_after_a_failing_persona(
    api: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = Persona(**{**DEMO.__dict__, "key": "demo-zed", "display_name": "Zed", "seeds": ()})
    monkeypatch.setattr(seed_demo, "PERSONAS", (broken, DEMO))

    @contextmanager
    def in_process(_settings: Settings, api_url: str | None) -> Iterator[httpx.Client]:
        yield api

    monkeypatch.setattr(seed_demo, "api_client", in_process)
    assert seed_demo.main(["--no-recs"], settings=settings) == 1
    captured = capsys.readouterr()
    assert "seed-demo failed: Zed: submit seeds failed" in captured.err
    assert "Ana (US" in captured.out


def test_main_reports_failures(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    @contextmanager
    def unreachable(_settings: Settings, api_url: str | None) -> Iterator[httpx.Client]:
        raise httpx.ConnectError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(seed_demo, "api_client", unreachable)
    assert seed_demo.main(["--api-url", "http://127.0.0.1:1"], settings=settings) == 1
    assert "seed-demo failed: connection refused" in capsys.readouterr().err


def test_api_client_targets_a_running_api(settings: Settings) -> None:
    with seed_demo.api_client(settings, "http://127.0.0.1:8000/") as client:
        assert str(client.base_url) == "http://127.0.0.1:8000"


def test_api_client_in_process_loads_the_app(settings: Settings, catalog: FixtureCatalog) -> None:
    with seed_demo.api_client(settings, None) as client:
        assert client.get("/api/v1/ready").status_code == 200
