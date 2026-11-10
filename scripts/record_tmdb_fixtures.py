"""Record live TMDB responses into `tests/fixtures/tmdb/` (dev tool, needs TMDB credentials).

    uv run python -m scripts.record_tmdb_fixtures                 # everything
    uv run python -m scripts.record_tmdb_fixtures search_multi_squid.json   # only these files

Only response bodies are written, never request URLs or headers, so no credentials
end up in fixtures. Large responses are trimmed by dropping list entries and
countries we do not test (see `_trim_*`); every kept object is exactly as TMDB sent it.
Tests assert on these files, so re-recording may require updating
`tests/unit/test_tmdb_client.py` (availability and trending change over time).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.ingest.tmdb_client import MOVIE_APPEND, TMDB_API_BASE_URL, TV_APPEND

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tmdb"
KEEP_COUNTRIES = {"US", "IN", "KR", "JP"}
KEEP_LANGUAGES = {"en", "hi", "ko", "ja"}
KEEP_CAST = 10
KEEP_CREW_JOBS = {"Director", "Series Director", "Screenplay", "Writer", "Original Music Composer"}

# file name -> (path, params)
REQUESTS: dict[str, tuple[str, dict[str, str]]] = {
    "movie_27205.json": (
        "/movie/27205",
        {"language": "en-US", "append_to_response": ",".join(MOVIE_APPEND)},
    ),
    "tv_1429.json": (
        "/tv/1429",
        {"language": "en-US", "append_to_response": ",".join(TV_APPEND)},
    ),
    "search_multi_inception.json": (
        "/search/multi",
        {"query": "inception", "language": "en-US", "page": "1", "include_adult": "false"},
    ),
    # Mixed movie / tv / person results.
    "search_multi_nolan.json": (
        "/search/multi",
        {"query": "nolan", "language": "en-US", "page": "1", "include_adult": "false"},
    ),
    # Title search acceptance (P1-08): the 2019 film and the TV series.
    "search_multi_parasite.json": (
        "/search/multi",
        {"query": "parasite", "language": "en-US", "page": "1", "include_adult": "false"},
    ),
    "search_multi_squid.json": (
        "/search/multi",
        {"query": "squid", "language": "en-US", "page": "1", "include_adult": "false"},
    ),
    # Same search with a Korean UI language (localized titles).
    "search_multi_squid_ko.json": (
        "/search/multi",
        {"query": "squid", "language": "ko", "page": "1", "include_adult": "false"},
    ),
    "discover_movie_hi.json": (
        "/discover/movie",
        {
            "with_original_language": "hi",
            "sort_by": "popularity.desc",
            "page": "1",
            "include_adult": "false",
            "language": "en-US",
        },
    ),
    "discover_tv_ko.json": (
        "/discover/tv",
        {
            "with_original_language": "ko",
            "sort_by": "vote_count.desc",
            "page": "1",
            "include_adult": "false",
            "language": "en-US",
        },
    ),
    "trending_all_week.json": ("/trending/all/week", {}),
    "trending_tv_day.json": ("/trending/tv/day", {}),
    "watch_providers_movie_27205.json": ("/movie/27205/watch/providers", {}),
    "provider_list_movie_US.json": ("/watch/providers/movie", {"watch_region": "US"}),
    "error_404.json": ("/movie/999999999", {}),
}


def _trim_country_map(results: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in results.items() if k in KEEP_COUNTRIES}


def _trim_details(data: dict[str, Any]) -> dict[str, Any]:
    if credits := data.get("credits"):
        credits["cast"] = credits.get("cast", [])[:KEEP_CAST]
        credits["crew"] = [c for c in credits.get("crew", []) if c.get("job") in KEEP_CREW_JOBS]
    if wp := data.get("watch/providers"):
        wp["results"] = _trim_country_map(wp.get("results", {}))
    for key in ("release_dates", "content_ratings"):
        if block := data.get(key):
            block["results"] = [
                r for r in block.get("results", []) if r.get("iso_3166_1") in KEEP_COUNTRIES
            ]
    if tr := data.get("translations"):
        tr["translations"] = [
            t for t in tr.get("translations", []) if t.get("iso_639_1") in KEEP_LANGUAGES
        ]
    for key in ("seasons", "production_companies", "spoken_languages"):
        if isinstance(data.get(key), list):
            data[key] = data[key][:3]
    return data


def _trim(name: str, data: Any) -> Any:
    if name.startswith(("movie_", "tv_")):
        return _trim_details(data)
    if name.startswith("watch_providers_"):
        data["results"] = _trim_country_map(data.get("results", {}))
    if name.startswith("provider_list_"):
        for p in data.get("results", []):
            p["display_priorities"] = _trim_country_map(p.get("display_priorities", {}))
    return data


def _client(**auth: str) -> httpx.Client:
    settings = get_settings()
    headers = {"Accept": "application/json"}
    params: dict[str, str] = {}
    if "bearer" in auth:
        headers["Authorization"] = f"Bearer {auth['bearer']}"
    elif "api_key" in auth:
        params["api_key"] = auth["api_key"]
    elif settings.tmdb_read_token:
        headers["Authorization"] = f"Bearer {settings.tmdb_read_token.get_secret_value()}"
    elif settings.tmdb_api_key:
        params["api_key"] = settings.tmdb_api_key.get_secret_value()
    else:
        sys.exit("Set TMDB_READ_TOKEN or TMDB_API_KEY in .env")
    return httpx.Client(
        base_url=TMDB_API_BASE_URL,
        headers=headers,
        params=params,
        timeout=settings.tmdb_timeout_seconds,
    )


def _write(name: str, data: Any) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{name:40} {path.stat().st_size / 1024:7.1f} KiB")


def main(argv: list[str] | None = None) -> None:
    only = set(sys.argv[1:] if argv is None else argv)
    if unknown := only - set(REQUESTS) - {"error_401.json"}:
        sys.exit(f"Unknown fixture(s): {', '.join(sorted(unknown))}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        for name, (path, params) in REQUESTS.items():
            if only and name not in only:
                continue
            response = client.get(path, params=params)
            expected_error = name.startswith("error_")
            if response.is_error != expected_error:
                sys.exit(f"{name}: unexpected HTTP {response.status_code}")
            _write(name, _trim(name, response.json()))
    if only and "error_401.json" not in only:
        return
    # A deliberately invalid key, to record TMDB's real 401 body.
    with _client(api_key="invalid") as client:
        response = client.get("/movie/27205")
        if response.status_code != 401:
            sys.exit(f"error_401.json: unexpected HTTP {response.status_code}")
        _write("error_401.json", response.json())


if __name__ == "__main__":
    main()
