from __future__ import annotations

import pytest

from app.core.geo import default_timezone, is_valid_country, is_valid_timezone


@pytest.mark.parametrize("code", ["US", "IN", "KR", "JP", "GB"])
def test_valid_countries(code: str) -> None:
    assert is_valid_country(code)


@pytest.mark.parametrize("code", ["XX", "us", "USA", "UK", ""])
def test_invalid_countries(code: str) -> None:
    assert not is_valid_country(code)


@pytest.mark.parametrize(
    ("country", "tz"),
    [
        ("US", "America/New_York"),
        ("IN", "Asia/Kolkata"),
        ("KR", "Asia/Seoul"),
        ("jp", "Asia/Tokyo"),
    ],
)
def test_default_timezone(country: str, tz: str) -> None:
    assert default_timezone(country) == tz


def test_default_timezone_falls_back_to_utc() -> None:
    assert default_timezone("XX") == "UTC"


def test_timezones() -> None:
    assert is_valid_timezone("Europe/Paris")
    assert is_valid_timezone("UTC")
    assert not is_valid_timezone("Mars/Olympus_Mons")
    assert not is_valid_timezone("europe/paris")
