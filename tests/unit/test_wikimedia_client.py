"""Wikimedia client: User-Agent policy, SPARQL / summary parsing, retries (P1-04b)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app import __version__
from app.core.config import Settings
from app.ingest.wikimedia_client import (
    WIKIDATA_SPARQL_URL,
    WikimediaClient,
    WikimediaConfigError,
    WikimediaHTTPError,
    WikimediaResponseError,
    WikimediaUnavailableError,
    article_url,
    wiki_language,
    wiki_site,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "wikimedia"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
CONTACT = "ops@whattowatch.invalid"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


@pytest.fixture
async def wm(sleeps: Sleeps) -> AsyncIterator[WikimediaClient]:
    async with WikimediaClient(contact=CONTACT, max_attempts=3, sleep=sleeps) as client:
        yield client


@pytest.mark.parametrize("contact", [None, "", "   "])
def test_missing_contact_is_refused(contact: str | None) -> None:
    with pytest.raises(WikimediaConfigError, match="WIKIMEDIA_CONTACT"):
        WikimediaClient(contact=contact)


def test_multiline_contact_is_refused() -> None:
    with pytest.raises(WikimediaConfigError, match="single line"):
        WikimediaClient(contact="a@b.c\r\nX-Evil: 1")


def test_from_settings_requires_contact() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(WikimediaConfigError):
        WikimediaClient.from_settings(settings)


def test_settings_reject_contact_with_parentheses() -> None:
    with pytest.raises(ValueError, match="WIKIMEDIA_CONTACT"):
        Settings(_env_file=None, wikimedia_contact="x (y)")  # type: ignore[call-arg]


async def test_user_agent_follows_wikimedia_policy(wm: WikimediaClient) -> None:
    expected = f"WhatToWatch/{__version__} (non-commercial; contact: {CONTACT})"
    assert wm.user_agent == expected
    with respx.mock(assert_all_called=True) as router:
        route = router.post(WIKIDATA_SPARQL_URL).mock(
            return_value=httpx.Response(200, json=fixture("sparql_results.json"))
        )
        await wm.sparql("SELECT * WHERE {}")
    request = route.calls.last.request
    assert request.headers["User-Agent"] == expected
    assert request.headers["Api-User-Agent"] == expected
    assert request.headers["Accept"] == "application/sparql-results+json"
    assert b"query=SELECT" in request.content


async def test_sparql_flattens_bindings(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.post(WIKIDATA_SPARQL_URL).mock(
            return_value=httpx.Response(200, json=fixture("sparql_results.json"))
        )
        rows = await wm.sparql("SELECT")
    assert len(rows) == 3
    assert rows[0] == {
        "tmdb": "900001",
        "item": "http://www.wikidata.org/entity/Q5001",
        "itemLabel": "The Glass Orchard",
        "kind": "item",
    }
    assert rows[2]["text"] == "The Glass Orchard (film)"


async def test_sparql_bad_shape_raises(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.post(WIKIDATA_SPARQL_URL).mock(return_value=httpx.Response(200, json={"x": 1}))
        with pytest.raises(WikimediaResponseError):
            await wm.sparql("SELECT")


async def test_page_summary_parses_standard_page(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        route = router.get(url__startswith=SUMMARY_URL).mock(
            return_value=httpx.Response(200, json=fixture("summary_standard.json"))
        )
        summary = await wm.page_summary("enwiki", "The Glass Orchard (film)")
    assert route.calls.last.request.url.raw_path.endswith(
        b"/page/summary/The_Glass_Orchard_%28film%29"
    )
    assert summary is not None
    assert summary.title == "The Glass Orchard (film)"
    assert summary.revision_id == 1180000001
    assert summary.extract.startswith("The Glass Orchard is a 2019 drama film")
    assert summary.usable


async def test_page_summary_disambiguation_is_not_usable(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.get(url__startswith=SUMMARY_URL).mock(
            return_value=httpx.Response(200, json=fixture("summary_disambiguation.json"))
        )
        summary = await wm.page_summary("enwiki", "Glass Orchard")
    assert summary is not None
    assert not summary.usable


async def test_page_summary_404_is_none(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.get(url__startswith=SUMMARY_URL).mock(return_value=httpx.Response(404))
        assert await wm.page_summary("enwiki", "Nope") is None


async def test_page_summary_follows_redirects(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.get(SUMMARY_URL + "Old_Name").mock(
            return_value=httpx.Response(302, headers={"Location": "New_Name"})
        )
        router.get(SUMMARY_URL + "New_Name").mock(
            return_value=httpx.Response(200, json=fixture("summary_standard.json"))
        )
        summary = await wm.page_summary("enwiki", "Old Name")
    assert summary is not None
    assert summary.title == "The Glass Orchard (film)"


async def test_page_summary_uses_language_wiki(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        route = router.get(
            url__startswith="https://ko.wikipedia.org/api/rest_v1/page/summary/"
        ).mock(return_value=httpx.Response(200, json=fixture("summary_standard.json")))
        await wm.page_summary("kowiki", "유리 과수원")
    assert route.called


async def test_retries_5xx_then_succeeds(wm: WikimediaClient, sleeps: Sleeps) -> None:
    with respx.mock() as router:
        router.post(WIKIDATA_SPARQL_URL).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json=fixture("sparql_results.json")),
            ]
        )
        rows = await wm.sparql("SELECT")
    assert len(rows) == 3
    assert len(sleeps.calls) == 1


async def test_429_honours_retry_after(wm: WikimediaClient, sleeps: Sleeps) -> None:
    with respx.mock() as router:
        router.get(url__startswith=SUMMARY_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json=fixture("summary_standard.json")),
            ]
        )
        await wm.page_summary("enwiki", "X")
    assert sleeps.calls[0] >= 7


async def test_retry_after_is_clamped(wm: WikimediaClient, sleeps: Sleeps) -> None:
    with respx.mock() as router:
        router.get(url__startswith=SUMMARY_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "3600"}),
                httpx.Response(404),
            ]
        )
        await wm.page_summary("enwiki", "X")
    assert sleeps.calls == [60.0]


async def test_gives_up_after_max_attempts(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.post(WIKIDATA_SPARQL_URL).mock(return_value=httpx.Response(500))
        with pytest.raises(WikimediaUnavailableError) as info:
            await wm.sparql("SELECT")
    assert info.value.attempts == 3
    assert info.value.last_status == 500


async def test_network_errors_are_retried(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.post(WIKIDATA_SPARQL_URL).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(WikimediaUnavailableError) as info:
            await wm.sparql("SELECT")
    assert info.value.last_status is None


async def test_other_4xx_fails_fast(wm: WikimediaClient, sleeps: Sleeps) -> None:
    with respx.mock() as router:
        router.post(WIKIDATA_SPARQL_URL).mock(return_value=httpx.Response(400))
        with pytest.raises(WikimediaHTTPError) as info:
            await wm.sparql("SELECT")
    assert info.value.status_code == 400
    assert sleeps.calls == []


async def test_invalid_json_raises(wm: WikimediaClient) -> None:
    with respx.mock() as router:
        router.get(url__startswith=SUMMARY_URL).mock(
            return_value=httpx.Response(200, text="<html>")
        )
        with pytest.raises(WikimediaResponseError):
            await wm.page_summary("enwiki", "X")


def test_site_helpers() -> None:
    assert wiki_site("ml") == "mlwiki"
    assert wiki_language("tawiki") == "ta"
    with pytest.raises(ValueError, match="Wikipedia"):
        wiki_language("wiki")
    assert article_url("enwiki", "The Glass Orchard (film)") == (
        "https://en.wikipedia.org/wiki/The_Glass_Orchard_%28film%29"
    )
