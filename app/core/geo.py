"""ISO-3166-1 country codes and IANA timezones, read from the `tzdata` package.

`iso3166.tab` lists the alpha-2 codes; `zone.tab` maps each country to its zones,
with the most populous region first (e.g. `US` → `America/New_York`), which we use
as the default timezone for a country (§6 `users.timezone`).
"""

from __future__ import annotations

import importlib.resources
import zoneinfo
from functools import lru_cache

DEFAULT_TIMEZONE = "UTC"


def _read_tab(name: str) -> list[list[str]]:
    text = (importlib.resources.files("tzdata") / "zoneinfo" / name).read_text("utf-8")
    return [
        line.split("\t") for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]


@lru_cache
def country_codes() -> frozenset[str]:
    """All ISO-3166-1 alpha-2 codes (upper case)."""
    return frozenset(row[0].upper() for row in _read_tab("iso3166.tab"))


@lru_cache
def _country_zones() -> dict[str, list[str]]:
    zones: dict[str, list[str]] = {}
    for row in _read_tab("zone.tab"):
        zones.setdefault(row[0].upper(), []).append(row[2])
    return zones


@lru_cache
def _timezones() -> frozenset[str]:
    return frozenset(zoneinfo.available_timezones())


def is_valid_country(code: str) -> bool:
    return code in country_codes()


def is_valid_timezone(name: str) -> bool:
    return name in _timezones()


def default_timezone(country: str) -> str:
    """The country's primary IANA zone, or `UTC` if it has none listed."""
    zones = _country_zones().get(country.upper())
    return zones[0] if zones else DEFAULT_TIMEZONE
