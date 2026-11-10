"""Mapping TMDB details → `titles` / `title_availability` rows (P1-04), on recorded fixtures."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.db.models import NO_PROVIDER_ID, OfferType, TitleType
from app.ingest.ingest_catalog import (
    IngestOptions,
    availability_rows,
    build_parser,
    discover_queries,
    options_from_args,
    title_values,
)
from app.ingest.tmdb_models import MovieDetails, TVDetails

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tmdb"
NOW = datetime(2026, 11, 3, tzinfo=UTC)


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def inception() -> MovieDetails:
    return MovieDetails.model_validate(fixture("movie_27205.json"))


@pytest.fixture
def aot() -> TVDetails:
    return TVDetails.model_validate(fixture("tv_1429.json"))


def test_movie_title_values(inception: MovieDetails) -> None:
    values = title_values(inception, ["US", "IN"])

    assert values["tmdb_id"] == 27205
    assert values["type"] is TitleType.MOVIE
    assert values["title"] == "Inception"
    assert values["original_language"] == "en"
    assert values["genres"] == ["action", "science-fiction", "crime"]
    assert values["directors"] == ["Christopher Nolan"]
    assert values["top_cast"] == ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Ken Watanabe"]
    assert values["release_date"] == date(2010, 7, 15)
    assert values["runtime"] == 148
    assert values["seasons"] is None
    assert values["certification"] == {"US": "PG-13", "IN": "UA"}
    assert "dreams" in values["keywords"]  # type: ignore[operator]
    assert values["localized_titles"] == {
        "ja-JP": "インセプション",
        "ko-KR": "인셉션",
        "hi-IN": "चक्रव्युह",
    }
    assert values["overview"]


def test_tv_title_values_use_fallbacks(aot: TVDetails) -> None:
    values = title_values(aot, ["US", "KR"])

    assert values["type"] is TitleType.TV
    assert values["title"] == "Attack on Titan"
    assert values["genres"] == [
        "animation",
        "anime",
        "science-fiction",
        "fantasy",
        "action",
        "adventure",
    ]
    # `created_by` is empty → crew "Series Director".
    assert values["directors"] == ["Yuichiro Hayashi", "Jun Shishido"]
    # `episode_run_time` is empty → `last_episode_to_air.runtime`.
    assert values["runtime"] == 23
    assert values["seasons"] == aot.number_of_seasons
    assert values["certification"] == {"US": "TV-MA", "KR": "19"}
    # Keyword whitespace (incl. NBSP) stripped.
    assert "political intrigue" in values["keywords"]  # type: ignore[operator]
    assert all(k == k.strip() for k in values["keywords"])  # type: ignore[attr-defined]
    # Empty Japanese translation is dropped.
    assert "ja-JP" not in values["localized_titles"]  # type: ignore[operator]


def test_tv_directors_prefer_created_by(aot: TVDetails) -> None:
    data = fixture("tv_1429.json")
    data["created_by"] = [{"id": 1, "name": "Hajime Isayama"}]
    values = title_values(TVDetails.model_validate(data), ["US"])
    assert values["directors"] == ["Hajime Isayama"]


def test_tv_runtime_averages_episode_run_time() -> None:
    data = fixture("tv_1429.json")
    data["episode_run_time"] = [24, 25, 0]
    assert title_values(TVDetails.model_validate(data), ["US"])["runtime"] == 24


def test_movie_certification_prefers_theatrical() -> None:
    data = fixture("movie_27205.json")
    data["release_dates"] = {
        "results": [
            {
                "iso_3166_1": "US",
                "release_dates": [
                    {"certification": "", "type": 3},
                    {"certification": "R", "type": 4},
                    {"certification": "PG-13", "type": 3},
                ],
            },
            {"iso_3166_1": "JP", "release_dates": [{"certification": "G", "type": 2}]},
        ]
    }
    values = title_values(MovieDetails.model_validate(data), ["US", "JP"])
    assert values["certification"] == {"US": "PG-13", "JP": "G"}


def test_overview_falls_back_to_original_language_translation() -> None:
    data = fixture("tv_1429.json")
    data["overview"] = ""
    data["translations"]["translations"] = [
        {"iso_3166_1": "JP", "iso_639_1": "ja", "data": {"name": "", "overview": "日本語の概要"}}
    ]
    assert title_values(TVDetails.model_validate(data), ["US"])["overview"] == "日本語の概要"


def test_availability_rows(aot: TVDetails) -> None:
    rows = availability_rows(aot, ["US", "FR"], NOW)

    us = {(r.provider_id, r.offer_type) for r in rows if r.country == "US"}
    assert (15, OfferType.FLATRATE) in us  # Hulu
    assert (10, OfferType.BUY) in us
    assert all(
        r.link == "https://www.themoviedb.org/tv/1429/watch?locale=US"
        for r in rows
        if r.country == "US"
    )
    # No offers in FR (not in the fixture) → marker row.
    fr = [r for r in rows if r.country == "FR"]
    assert [(r.provider_id, r.offer_type) for r in fr] == [(NO_PROVIDER_ID, OfferType.NONE)]
    assert all(r.type is TitleType.TV and r.tmdb_id == 1429 and r.fetched_at == NOW for r in rows)
    assert len({(r.country, r.provider_id, r.offer_type) for r in rows}) == len(rows)


def test_availability_rows_without_appended_providers() -> None:
    data = fixture("movie_27205.json")
    del data["watch/providers"]
    assert availability_rows(MovieDetails.model_validate(data), ["US"], NOW) == []


def test_discover_queries_cover_strategies() -> None:
    options = IngestOptions(
        languages=("en", "ko"), types=tuple(TitleType), countries=("US",), pages=1
    )
    queries = discover_queries(options)
    assert len(queries) == 2 * 2 * 3
    assert {(q.sort_by, q.recent) for q in queries} == {
        ("popularity.desc", False),
        ("vote_count.desc", False),
        ("popularity.desc", True),
    }


def test_options_from_args_defaults_and_overrides() -> None:
    settings = Settings(_env_file=None, seed_languages="en,hi", ingest_pages_per_language=7)  # type: ignore[call-arg]
    parser = build_parser()

    defaults = options_from_args(parser.parse_args([]), settings)
    assert defaults.languages == ("en", "hi")
    assert defaults.types == (TitleType.MOVIE, TitleType.TV)
    assert defaults.countries == ("US",)
    assert defaults.pages == 7
    assert defaults.limit is None

    custom = options_from_args(
        parser.parse_args(
            [
                "--languages",
                "ko,ja",
                "--countries",
                "us",
                "in",
                "--types",
                "tv",
                "--pages",
                "2",
                "--limit",
                "200",
            ]
        ),
        settings,
    )
    assert custom.languages == ("ko", "ja")
    assert custom.countries == ("US", "IN")
    assert custom.types == (TitleType.TV,)
    assert (custom.pages, custom.limit) == (2, 200)


@pytest.mark.parametrize("argv", [["--limit", "0"], ["--pages", "0"], ["--countries", "USA"]])
def test_options_from_args_rejects_bad_values(argv: list[str]) -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(SystemExit):
        options_from_args(build_parser().parse_args(argv), settings)
