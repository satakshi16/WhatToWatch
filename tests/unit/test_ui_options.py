"""P1-13 `ui/options.py`: UI choices stay in step with the backend; display helpers."""

from __future__ import annotations

import pytest

from app.core.geo import country_codes
from app.db.models.enums import Era, TitleType
from app.ingest.genres import CANONICAL_GENRES
from ui import options


def test_genres_match_canonical_slugs() -> None:
    assert set(options.GENRES) == CANONICAL_GENRES


def test_eras_match_backend_enum() -> None:
    assert set(options.ERAS) == {e.value for e in Era}


def test_content_types_match_backend_enum() -> None:
    assert set(options.CONTENT_TYPES) == {t.value for t in TitleType}
    assert set(options.TYPE_LABELS) == {t.value for t in TitleType}


def test_countries_match_backend_validation() -> None:
    countries = options.countries()
    assert set(countries) == country_codes()
    assert countries["US"] == "United States"
    names = list(countries.values())
    assert names == sorted(names)


def test_languages_are_two_letter_codes() -> None:
    assert all(len(c) == 2 and c.islower() for c in options.LANGUAGES)
    assert next(iter(options.LANGUAGES)) == "en"


@pytest.mark.parametrize(
    ("minutes", "label"),
    [(None, None), (0, None), (45, "45m"), (60, "1h"), (148, "2h 28m"), (125, "2h 05m")],
)
def test_runtime_label(minutes: int | None, label: str | None) -> None:
    assert options.runtime_label(minutes) == label


@pytest.mark.parametrize(("n", "label"), [(None, None), (1, "1 season"), (3, "3 seasons")])
def test_seasons_label(n: int | None, label: str | None) -> None:
    assert options.seasons_label(n) == label


def test_language_and_genre_labels_fall_back() -> None:
    assert options.language_label("ko") == "Korean"
    assert options.language_label("xx") == "XX"
    assert options.language_label(None) == "Unknown language"
    assert options.genre_label("science-fiction") == "Science fiction"
    assert options.genre_label("new-genre") == "New genre"


def test_meta_line_movie() -> None:
    item = {
        "type": "movie",
        "year": 2010,
        "language": "en",
        "runtime_min": 148,
        "seasons": None,
        "genres": ["science-fiction", "action"],
    }
    assert options.meta_line(item) == "2010 · Movie · English · 2h 28m · Science fiction, Action"


def test_meta_line_tv_shows_seasons_and_skips_unknowns() -> None:
    item = {
        "type": "tv",
        "year": None,
        "language": "ko",
        "runtime_min": 55,
        "seasons": 2,
        "genres": [],
    }
    assert options.meta_line(item) == "Series · Korean · 2 seasons"
