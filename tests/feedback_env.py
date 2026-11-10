"""Shared setup for P1-09 tests: small embedded catalog, fake TMDB details, fake Wikimedia.

Catalog: movies 1-5 have open content (so they are in the embedding index), movie 6 is
in the catalog but was never checked for open content. Titles fetched on demand use the
recorded TMDB details for Inception (movie 27205) and Attack on Titan (tv 1429).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.deps import get_tmdb_factory
from app.db.models import TitleType
from app.ingest.build_embeddings import EmbedOptions, build_embeddings
from app.ingest.tmdb_client import TMDBClient
from app.ingest.wikimedia_client import WikimediaClient
from app.recsys.embeddings import EmbeddingStore, OnDemandEmbedder
from tests.catalog_factory import add_catalog, open_content, tmdb_title
from tests.fakes import FakeEncoder
from tests.wikimedia_fakes import Entity, FakeWikimedia

M, T = TitleType.MOVIE, TitleType.TV
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tmdb"
PASSWORD = "s3cret-passw0rd"  # pragma: allowlist secret
PROBLEM = "application/problem+json"
API = r"https://api\.themoviedb\.org/3"

INCEPTION = 27205
ATTACK_ON_TITAN = 1429
CATALOG_MOVIES = (1, 2, 3, 4, 5)
UNCHECKED_MOVIE = 6
WIKI_CONTACT = "ops@whattowatch.invalid"


def _fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


DETAILS = {
    (INCEPTION, "movie"): _fixture("movie_27205.json"),
    (ATTACK_ON_TITAN, "tv"): _fixture("tv_1429.json"),
}


class FakeTMDBDetails:
    """`/movie/{id}` and `/tv/{id}` from recorded fixtures; unknown IDs → 404."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.calls: list[tuple[str, int]] = []
        self.status: int | None = None
        router.get(url__regex=rf"{API}/(?P<kind>movie|tv)/(?P<tmdb_id>\d+)(?:\?|$)").mock(
            side_effect=self._handle
        )

    def _handle(self, request: httpx.Request, kind: str, tmdb_id: str) -> httpx.Response:
        self.calls.append((kind, int(tmdb_id)))
        if self.status is not None:
            return httpx.Response(self.status, json={"status_message": "down"})
        data = DETAILS.get((int(tmdb_id), kind))
        if data is None:
            return httpx.Response(404, json=_fixture("error_404.json"))
        return httpx.Response(200, json=copy.deepcopy(data))


async def no_sleep(_: float) -> None:
    return None


def use_fake_tmdb(app: FastAPI) -> None:
    app.dependency_overrides[get_tmdb_factory] = lambda: (
        lambda: TMDBClient(api_key="test-key", max_attempts=2, sleep=no_sleep)
    )


def build_catalog(engine: Engine, embeddings_dir: Path, encoder: FakeEncoder) -> None:
    """Catalog + open content + an index built with `encoder` in `embeddings_dir`."""
    with Session(engine) as session:
        add_catalog(
            session,
            *(tmdb_title(i, M) for i in CATALOG_MOVIES),
            tmdb_title(UNCHECKED_MOVIE, M),
        )
        add_catalog(session, *(open_content(i, M) for i in CATALOG_MOVIES))
    build_embeddings(engine, encoder, EmbedOptions(batch_size=4), embeddings_dir)


def make_embedder(
    embeddings_dir: Path,
    *,
    encoder: FakeEncoder | None = None,
    wikimedia: bool = True,
    encoder_error: bool = False,
) -> OnDemandEmbedder:
    def encoder_factory() -> FakeEncoder:
        if encoder_error:
            raise ImportError("sentence_transformers")
        return encoder or FakeEncoder()

    def wikimedia_factory() -> WikimediaClient:
        return WikimediaClient(contact=WIKI_CONTACT, sleep=no_sleep, max_attempts=1)

    return OnDemandEmbedder(
        EmbeddingStore(embeddings_dir),
        encoder_factory=encoder_factory,
        wikimedia_factory=wikimedia_factory if wikimedia else None,
        seed_languages=("en",),
    )


def add_inception_open_content(fake: FakeWikimedia) -> None:
    fake.add(
        INCEPTION,
        M,
        Entity(qid="Q25188", label="Inception", sitelinks={"en": "Inception"}),
        summary="Inception is a 2010 science fiction action film.",
    )


def register(client: TestClient, email: str = "priya@example.com") -> str:
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "display_name": "Priya",
            "country": "US",
            "age_confirmed": True,
        },
    )
    assert resp.status_code == 201, resp.text
    token: str = resp.json()["access_token"]
    return token


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def save_preferences(client: TestClient, token: str) -> None:
    resp = client.put("/api/v1/me/preferences", headers=auth(token), json={})
    assert resp.status_code == 200, resp.text


def seed(tmdb_id: int, type_: str = "movie") -> dict[str, Any]:
    return {"tmdb_id": tmdb_id, "type": type_}
