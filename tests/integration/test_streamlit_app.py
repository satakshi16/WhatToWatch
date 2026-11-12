"""P1-13 Streamlit UI end to end: AppTest drives `ui/streamlit_app.py`; its HTTP calls are
routed (respx) to the real FastAPI app over the P1-11 fixture catalog, with TMDB faked.

Covers Journey A (sign up → preferences → seeds → recommendations → rate / not interested),
the TMDB footer on every screen, the country-change notice and session expiry.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from app.api.deps import get_tmdb_factory
from app.core.config import Settings
from app.db.models import Interaction, Provider, Title, TitleType
from app.ingest.tmdb_models import Page, WatchProviders
from tests.recsys_catalog import NETFLIX, PRIME, FixtureCatalog, build_fixture_catalog
from ui.attribution import FOOTER_TEST_ID, JUSTWATCH_NOTICE, TMDB_NOTICE, tmdb_logo_data_uri

APP_PATH = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"
BASE = "http://wtw-api.test"
PASSWORD = "correct horse battery"


class FakeTMDB:
    """Search over the fixture catalog's titles; every title streams on Netflix (US)."""

    def __init__(self, titles: list[Title]) -> None:
        self.titles = titles

    def __call__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def search_multi(
        self, query: str, language: str = "en-US", *, page: int = 1, include_adult: bool = False
    ) -> Page:
        hits = [t for t in self.titles if query.casefold() in t.title.casefold()][:20]
        results = [
            {
                "id": t.tmdb_id,
                "media_type": t.type.value,
                ("title" if t.type is TitleType.MOVIE else "name"): t.title,
                ("original_title" if t.type is TitleType.MOVIE else "original_name"): (
                    t.original_title
                ),
                "original_language": t.original_language,
                "poster_path": "/p.jpg",
                "adult": t.adult,
            }
            for t in hits
        ]
        return Page.model_validate({"page": 1, "results": results, "total_results": len(hits)})

    async def watch_providers(self, type_: TitleType, tmdb_id: int) -> WatchProviders:
        offer = {"provider_id": NETFLIX, "provider_name": "Netflix", "display_priority": 1}
        link = f"https://www.themoviedb.org/{type_.value}/{tmdb_id}/watch?locale=US"
        return WatchProviders.model_validate(
            {"results": {"US": {"link": link, "flatrate": [offer]}}}
        )


@pytest.fixture
def catalog(db_session: Session, settings: Settings) -> FixtureCatalog:
    fixture = build_fixture_catalog(db_session, settings.embeddings_dir)
    db_session.execute(update(Title).values(poster_path="/poster.jpg"))
    # A second country with Netflix only, for the country-change notice.
    db_session.add(Provider(provider_id=NETFLIX, country="KR", name="Netflix", logo_path="/8.png"))
    db_session.commit()
    return fixture


@pytest.fixture
def api(app: FastAPI, catalog: FixtureCatalog, db_session: Session) -> Iterator[TestClient]:
    titles = list(db_session.scalars(select(Title).order_by(Title.tmdb_id)))
    app.dependency_overrides[get_tmdb_factory] = lambda: FakeTMDB(titles)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def ui(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[respx.MockRouter]:
    """Route the UI's HTTP calls (to `BASE`) into the in-process API."""
    monkeypatch.setenv("API_BASE_URL", BASE)

    def forward(request: httpx.Request) -> httpx.Response:
        resp = api.request(
            request.method,
            request.url.raw_path.decode(),
            content=request.content,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
        return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content)

    with respx.mock(assert_all_called=False) as router:
        router.route(url__startswith=BASE).mock(side_effect=forward)
        yield router


def new_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def run(target: Any) -> AppTest:
    """Run the app (`target` is the AppTest or a widget just interacted with)."""
    at: AppTest = target.run()
    assert not at.exception, at.exception
    return at


def button(at: AppTest, label: str) -> Any:
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


def texts(at: AppTest) -> list[str]:
    parts = [m.value for m in (*at.markdown, *at.caption, *at.header, *at.subheader)]
    parts += [m.value for m in (*at.success, *at.info, *at.warning, *at.error)]
    return parts


def assert_footer(at: AppTest) -> None:
    footers = [m.value for m in at.markdown if FOOTER_TEST_ID in m.value]
    assert len(footers) == 1, "footer missing or repeated"
    assert TMDB_NOTICE in footers[0]
    assert tmdb_logo_data_uri() in footers[0]
    # The app name is the page title (st.title), the TMDB logo only sits in the footer.
    assert [t.value for t in at.title] == ["🎬 WhatToWatch"]
    assert not any(tmdb_logo_data_uri() in t.value for t in at.title)


def sign_up(at: AppTest, email: str = "priya@example.com") -> AppTest:
    at.text_input(key="signup_name").input("Priya")
    at.text_input(key="signup_email").input(email)
    at.text_input(key="signup_password").input(PASSWORD)
    at.checkbox(key="signup_age").check()
    return run(button(at, "Create account").click())


def save_preferences(at: AppTest) -> AppTest:
    at.multiselect(key="pref_platforms_US").select(NETFLIX).select(PRIME)
    at.multiselect(key="pref_languages").select("hi").select("ko")
    return run(at.button(key="pref_save").click())


def pick_seeds(at: AppTest, count: int = 3) -> AppTest:
    at.text_input(key="seed_query").input("Title 1")
    run(button(at, "Search").click())
    adds = [b for b in at.button if b.key and b.key.startswith("add-")]
    assert len(adds) >= count
    keys = [b.key for b in adds[:count]]
    for key in keys:
        run(at.button(key=key).click())
    return at


def onboard(at: AppTest) -> AppTest:
    sign_up(at)
    save_preferences(at)
    pick_seeds(at)
    return run(at.button(key="seeds_submit").click())


def rec_keys(at: AppTest) -> list[str]:
    return [f.key.removeprefix("rate-") for f in at.feedback]


# --- Journey A ------------------------------------------------------------------


def test_logged_out_screen_shows_footer(ui: respx.MockRouter) -> None:
    at = new_app()
    assert {"Log in", "Create account"} <= {b.label for b in at.button}
    assert_footer(at)


def test_journey_a_reaches_recommendations_with_footer_on_every_screen(
    ui: respx.MockRouter, db_session: Session
) -> None:
    at = new_app()
    assert_footer(at)

    sign_up(at)
    assert any("Step 1 of 2" in h.value for h in at.header)
    assert_footer(at)

    save_preferences(at)
    assert any("Step 2 of 2" in h.value for h in at.header)
    assert any("Preferences saved." in s.value for s in at.success)
    assert_footer(at)

    pick_seeds(at)
    assert any("Your picks (3/10)" in s.value for s in at.subheader)
    run(at.button(key="seeds_submit").click())

    assert any(h.value == "Your picks" for h in at.header)
    assert_footer(at)
    assert len(at.feedback) == 10  # one star-rating widget per card
    assert len(at.image) >= 10  # posters
    joined = "\n".join(texts(at))
    assert JUSTWATCH_NOTICE in joined  # near the provider logos
    assert "Because you liked" in joined
    assert "https://www.themoviedb.org/" in joined  # provider logos link to the watch page

    seeds = db_session.scalars(
        select(Interaction).where(Interaction.source == "onboarding_seed")
    ).all()
    assert len(seeds) == 3


def test_rating_a_card_refreshes_the_list(ui: respx.MockRouter) -> None:
    at = onboard(new_app())
    before = rec_keys(at)
    first = before[0]

    run(at.feedback(key=f"rate-{first}").set_value(3))  # 4 stars

    assert any("★★★★" in s.value and "refreshed" in s.value for s in at.success)
    after = rec_keys(at)
    assert first not in after
    assert len(after) == 10
    assert_footer(at)


def test_not_interested_refreshes_the_list(ui: respx.MockRouter, db_session: Session) -> None:
    at = onboard(new_app())
    first = rec_keys(at)[0]

    run(at.button(key=f"skip-{first}").click())

    assert first not in rec_keys(at)
    kinds = db_session.scalars(select(Interaction.kind)).all()
    assert "not_interested" in kinds


def test_refresh_button_reloads(ui: respx.MockRouter) -> None:
    at = onboard(new_app())
    calls = len([c for c in ui.calls if c.request.url.path.endswith("/recommendations")])
    run(at.button(key="recs_refresh").click())
    after = len([c for c in ui.calls if c.request.url.path.endswith("/recommendations")])
    assert after == calls + 1
    assert len(at.feedback) == 10


def test_refresh_shows_the_next_best_titles(ui: respx.MockRouter) -> None:
    at = onboard(new_app())
    first = rec_keys(at)

    run(at.button(key="recs_refresh").click())
    second = rec_keys(at)
    assert len(second) == 10
    assert not set(first) & set(second)

    # Rating a card keeps the refreshed page (only the rated title drops out).
    run(at.feedback(key=f"rate-{second[0]}").set_value(4))
    third = rec_keys(at)
    assert second[0] not in third
    assert not set(first) & set(third)


def test_refresh_starts_over_when_every_match_was_shown(ui: respx.MockRouter) -> None:
    at = onboard(new_app())
    first = rec_keys(at)
    seen: set[str] = set(first)
    for _ in range(30):
        run(at.button(key="recs_refresh").click())
        keys = rec_keys(at)
        if seen & set(keys):
            break
        seen |= set(keys)
    else:
        pytest.fail("Refresh never wrapped around")
    assert any("starting from the top" in i.value for i in at.info)
    assert keys


def test_seed_page_shows_current_seeds_and_saving_replaces_them(
    ui: respx.MockRouter, db_session: Session
) -> None:
    at = onboard(new_app())
    run(at.sidebar.radio(key="view").set_value("Seed titles"))
    assert any("Your picks (3/10)" in s.value for s in at.subheader)
    removes = [b for b in at.button if b.key and b.key.startswith("remove-")]
    assert len(removes) == 3

    # Swap one seed for a new one and save.
    run(removes[0].click())
    at.text_input(key="seed_query").input("Title 2")
    run(button(at, "Search").click())
    adds = [b for b in at.button if b.key and b.key.startswith("add-") and not b.disabled]
    run(adds[0].click())
    assert any("Your picks (3/10)" in s.value for s in at.subheader)
    run(button(at, "Save and update recommendations →").click())

    assert len(at.feedback) > 0
    run(at.sidebar.radio(key="view").set_value("Seed titles"))
    chips = {b.key for b in at.button if b.key and b.key.startswith("remove-")}
    assert len(chips) == 3
    assert removes[0].key not in chips


def test_seed_submit_needs_three_titles(ui: respx.MockRouter) -> None:
    at = new_app()
    sign_up(at)
    save_preferences(at)
    pick_seeds(at, count=2)
    assert at.button(key="seeds_submit").disabled
    assert any("Add 1 more" in c.value for c in at.caption)
    # Removing a chip works too.
    remove = [b for b in at.button if b.key and b.key.startswith("remove-")]
    run(remove[0].click())
    assert any("Your picks (1/10)" in s.value for s in at.subheader)


# --- errors / edge cases ----------------------------------------------------------


def test_bad_login_shows_generic_error(ui: respx.MockRouter) -> None:
    at = new_app()
    at.text_input(key="login_email").input("nobody@example.com")
    at.text_input(key="login_password").input("wrong-password")
    run(button(at, "Log in").click())
    assert any("Incorrect email or password" in e.value for e in at.error)
    assert_footer(at)


def test_validation_errors_are_shown(ui: respx.MockRouter) -> None:
    at = new_app()
    at.text_input(key="signup_name").input("Priya")
    at.text_input(key="signup_email").input("not-an-email")
    at.text_input(key="signup_password").input("short")
    at.checkbox(key="signup_age").check()
    run(button(at, "Create account").click())
    assert at.error
    assert "token" not in at.session_state


def test_expired_session_signs_out(ui: respx.MockRouter) -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.session_state["token"] = "not-a-valid-token"
    at.session_state["user"] = {"display_name": "X", "email": "x@example.com", "country": "US"}
    run(at)
    assert any("session has expired" in w.value for w in at.warning)
    assert {"Log in", "Create account"} <= {b.label for b in at.button}
    assert_footer(at)


def test_country_change_reports_removed_platforms(ui: respx.MockRouter) -> None:
    at = onboard(new_app())
    at.sidebar.radio[0].set_value("Preferences")
    run(at)
    assert_footer(at)

    at.selectbox(key="pref_country").set_value("KR")
    run(at)
    assert at.multiselect(key="pref_platforms_KR").value == [NETFLIX]
    run(at.button(key="pref_save").click())

    warnings = " ".join(w.value for w in at.warning)
    assert "Not available in KR" in warnings
    assert "Amazon Prime Video" in warnings
    assert at.session_state["user"]["country"] == "KR"


def test_api_unreachable_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", BASE)
    with respx.mock() as router:
        router.route(url__startswith=BASE).mock(side_effect=httpx.ConnectError("refused"))
        at = new_app()
        at.text_input(key="login_email").input("a@example.com")
        at.text_input(key="login_password").input("whatever1")
        run(button(at, "Log in").click())
    assert any("make api" in e.value for e in at.error)
    assert_footer(at)
