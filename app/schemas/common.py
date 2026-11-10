"""Shared validated field types."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BeforeValidator, StringConstraints

from app.core.geo import is_valid_country, is_valid_timezone


def _strip(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


def _country(value: str) -> str:
    code = value.upper()
    if not is_valid_country(code):
        raise ValueError("must be an ISO-3166-1 alpha-2 country code")
    return code


def _timezone(value: str) -> str:
    if not is_valid_timezone(value):
        raise ValueError("must be an IANA timezone name, e.g. Europe/Paris")
    return value


CountryCode = Annotated[
    str,
    BeforeValidator(_strip),
    StringConstraints(min_length=2, max_length=2),
    AfterValidator(_country),
]
Timezone = Annotated[
    str,
    BeforeValidator(_strip),
    StringConstraints(min_length=1, max_length=64),
    AfterValidator(_timezone),
]
DisplayName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
