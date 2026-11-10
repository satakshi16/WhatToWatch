"""Demo users for the three spec personas (P1-14, §1): Priya, Marcus and Ji-woo.

Usage:
    make seed-demo                                   # in-process API over DATABASE_URL
    make seed-demo ARGS="--api-url http://127.0.0.1:8000"   # against a running `make api`

Each persona is created through the public API, exactly as the UI would do it: register
(or log in if the account exists), pick streaming services by name from the country's
provider list, save preferences, submit seed titles, then fetch the first recommendations.
Going through the API keeps every validation rule (provider IDs per country, seed counts,
on-demand fetch + embedding of unknown seeds) in one place.

Re-running is safe: existing accounts are logged into, preferences are replaced, and seeds
are only submitted while onboarding is unfinished (no duplicate ratings).

Needs a migrated database, TMDB credentials (provider lists for IN / KR and titles missing
from the catalog are fetched from TMDB) and, for recommendations, the embedding index
(`make embed`). Refuses to run with `APP_ENV=prod`: the demo password is public.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import AppEnv, Settings, get_settings
from app.db.models.enums import Era, TitleType

API = "/api/v1"
DEMO_PASSWORD = "whattowatch-demo"  # noqa: S105 - local demo accounts, printed on purpose
DEMO_EMAIL_DOMAIN = "demo.example.com"
M, T = TitleType.MOVIE, TitleType.TV


class SeedDemoError(RuntimeError):
    """A persona could not be set up; the message says which step failed and why."""


@dataclass(frozen=True)
class Persona:
    key: str
    display_name: str
    country: str
    timezone: str
    languages: tuple[str, ...]
    # One entry per wanted service: name aliases tried in order (TMDB renames services,
    # e.g. JioCinema → JioHotstar, Max → HBO Max). Matching is case-insensitive.
    platforms: tuple[tuple[str, ...], ...]
    seeds: tuple[tuple[int, TitleType], ...]
    content_types: tuple[TitleType, ...] = (M, T)
    era: Era = Era.ANY
    max_runtime: int | None = None

    @property
    def email(self) -> str:
        return f"{self.key}@{DEMO_EMAIL_DOMAIN}"

    def preferences(self, platform_ids: Sequence[int]) -> dict[str, Any]:
        return {
            "languages": list(self.languages),
            "platforms": list(platform_ids),
            "content_types": [t.value for t in self.content_types],
            "era": self.era.value,
            "max_runtime": self.max_runtime,
        }


# §1 personas. Seeds are TMDB IDs (all in the default seed catalog).
PERSONAS: tuple[Persona, ...] = (
    Persona(
        key="priya",
        display_name="Priya",
        country="IN",
        timezone="Asia/Kolkata",
        languages=("hi", "en", "ko"),
        platforms=(
            ("Netflix",),
            ("Amazon Prime Video",),
            ("JioCinema", "JioHotstar", "Jio Cinema", "Disney+ Hotstar", "Hotstar"),
        ),
        # 3 Idiots, Parasite, Interstellar
        seeds=((20453, M), (496243, M), (157336, M)),
    ),
    Persona(
        key="marcus",
        display_name="Marcus",
        country="US",
        timezone="America/New_York",
        languages=("en",),
        platforms=(("Netflix",), ("Max", "HBO Max")),
        content_types=(M,),
        max_runtime=120,
        # The Dark Knight, Inception, Interstellar
        seeds=((155, M), (27205, M), (157336, M)),
    ),
    Persona(
        key="ji-woo",
        display_name="Ji-woo",
        country="KR",
        timezone="Asia/Seoul",
        languages=("ja", "ko"),
        # Free tiers + one paid service (Netflix).
        platforms=(("Netflix",), ("Crunchyroll",), ("Laftel",), ("TVING",)),
        # Spirited Away, Your Name., Squid Game, My Neighbor Totoro, Akira
        seeds=((129, M), (372058, M), (93405, T), (8392, M), (149, M)),
    ),
)


@dataclass
class PersonaResult:
    persona: Persona
    created: bool
    platforms: dict[str, int] = field(default_factory=dict)  # matched name → provider ID
    missing_platforms: list[str] = field(default_factory=list)
    seeded: bool = False
    unembedded_seeds: list[str] = field(default_factory=list)
    recommendations: list[str] | None = None  # None: not fetched (`--no-recs`) or failed
    recommendations_error: str | None = None


def _problem(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else None
    errors = body.get("errors") if isinstance(body, dict) else None
    text = f"HTTP {resp.status_code}: {detail or body}"
    if errors:
        text += " " + "; ".join(str(e.get("msg", e)) for e in errors if isinstance(e, dict))
    return text


def _check(resp: httpx.Response, step: str, persona: Persona) -> Any:
    if resp.is_error:
        raise SeedDemoError(f"{persona.display_name}: {step} failed ({_problem(resp)})")
    return resp.json()


def _sign_in(client: httpx.Client, persona: Persona, password: str) -> tuple[str, bool]:
    """Access token for the persona; registers the account if it does not exist yet."""
    resp = client.post(
        f"{API}/auth/register",
        json={
            "email": persona.email,
            "password": password,
            "display_name": persona.display_name,
            "country": persona.country,
            "timezone": persona.timezone,
            "age_confirmed": True,
        },
    )
    if resp.status_code == httpx.codes.CONFLICT:
        resp = client.post(f"{API}/auth/login", json={"email": persona.email, "password": password})
        if resp.status_code == httpx.codes.UNAUTHORIZED:
            raise SeedDemoError(
                f"{persona.display_name}: {persona.email} exists with a different password"
            )
        return str(_check(resp, "log in", persona)["access_token"]), False
    return str(_check(resp, "register", persona)["access_token"]), True


def match_platforms(
    persona: Persona, providers: Sequence[dict[str, Any]]
) -> tuple[dict[str, int], list[str]]:
    """Provider IDs for the persona's wanted services, by name; plus the ones not offered."""
    by_name = {
        str(p["name"]).casefold(): (str(p["name"]), int(p["provider_id"])) for p in providers
    }
    matched: dict[str, int] = {}
    missing: list[str] = []
    for aliases in persona.platforms:
        hit = next((by_name[a.casefold()] for a in aliases if a.casefold() in by_name), None)
        if hit is None:
            missing.append(aliases[0])
        else:
            matched[hit[0]] = hit[1]
    return matched, missing


def seed_persona(
    client: httpx.Client, persona: Persona, *, password: str = DEMO_PASSWORD, recs: bool = True
) -> PersonaResult:
    token, created = _sign_in(client, persona, password)
    auth = {"Authorization": f"Bearer {token}"}
    result = PersonaResult(persona=persona, created=created)

    # The account's country is the persona's even if an earlier run changed it.
    _check(
        client.patch(f"{API}/me", json={"country": persona.country}, headers=auth),
        "set country",
        persona,
    )
    providers = _check(
        client.get(f"{API}/providers", params={"country": persona.country}, headers=auth),
        f"load {persona.country} providers",
        persona,
    )["providers"]
    result.platforms, result.missing_platforms = match_platforms(persona, providers)

    prefs = _check(
        client.put(
            f"{API}/me/preferences",
            json=persona.preferences(list(result.platforms.values())),
            headers=auth,
        ),
        "save preferences",
        persona,
    )
    if prefs["onboarding_step"] != "done":
        seeds = [{"tmdb_id": tmdb_id, "type": type_.value} for tmdb_id, type_ in persona.seeds]
        out = _check(
            client.post(f"{API}/onboarding/seeds", json={"seeds": seeds}, headers=auth),
            "submit seeds",
            persona,
        )
        result.seeded = True
        result.unembedded_seeds = [s["title"] for s in out["seeds"] if not s["embedded"]]

    if recs:
        resp = client.get(f"{API}/recommendations", params={"n": 10}, headers=auth)
        if resp.is_error:
            result.recommendations_error = _problem(resp)
        else:
            result.recommendations = [item["title"] for item in resp.json()["items"]]
    return result


def seed_demo(
    client: httpx.Client,
    personas: Sequence[Persona] = PERSONAS,
    *,
    password: str = DEMO_PASSWORD,
    recs: bool = True,
) -> list[PersonaResult]:
    return [seed_persona(client, p, password=password, recs=recs) for p in personas]


def describe(result: PersonaResult) -> str:
    p = result.persona
    lines = [
        f"{p.display_name} ({p.country}, {'/'.join(p.languages)}): {p.email} "
        f"[{'created' if result.created else 'existing account'}]",
        f"  services: {', '.join(result.platforms) or 'none'}"
        + (
            f" (not offered in {p.country}: {', '.join(result.missing_platforms)})"
            if result.missing_platforms
            else ""
        ),
        f"  seeds: {'submitted' if result.seeded else 'already onboarded, kept'}"
        + (
            f" (not embedded yet: {', '.join(result.unembedded_seeds)})"
            if result.unembedded_seeds
            else ""
        ),
    ]
    if result.recommendations_error:
        lines.append(f"  recommendations: unavailable ({result.recommendations_error})")
    elif result.recommendations is not None:
        shown = ", ".join(result.recommendations[:5])
        more = len(result.recommendations) - 5
        lines.append(
            f"  recommendations ({len(result.recommendations)}): {shown or '—'}"
            + (f", … (+{more})" if more > 0 else "")
        )
    return "\n".join(lines)


@contextmanager
def api_client(settings: Settings, api_url: str | None) -> Iterator[httpx.Client]:
    """HTTP client for a running API, or the app in-process (startup loads the index)."""
    if api_url:
        with httpx.Client(base_url=api_url.rstrip("/"), timeout=120) as client:
            yield client
        return
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(settings), raise_server_exceptions=True) as client:
        yield client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--api-url", help="Use a running API (e.g. http://127.0.0.1:8000) instead of in-process."
    )
    parser.add_argument("--password", default=DEMO_PASSWORD, help="Password for the demo users.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[p.key for p in PERSONAS],
        help="Seed only these personas.",
    )
    parser.add_argument(
        "--no-recs", action="store_true", help="Skip fetching the first recommendations."
    )
    return parser


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    if settings.app_env is AppEnv.PROD:
        print("Refusing to create demo users with APP_ENV=prod.", file=sys.stderr)
        return 2
    personas = [p for p in PERSONAS if not args.only or p.key in args.only]
    failed = 0
    try:
        with api_client(settings, args.api_url) as client:
            # One persona failing (e.g. a provider list TMDB can't serve) doesn't stop the rest.
            for persona in personas:
                try:
                    result = seed_persona(
                        client, persona, password=args.password, recs=not args.no_recs
                    )
                except SeedDemoError as exc:
                    failed += 1
                    print(f"seed-demo failed: {exc}", file=sys.stderr)
                    continue
                print(describe(result))
    except httpx.TransportError as exc:
        print(f"seed-demo failed: {exc}", file=sys.stderr)
        return 1
    if failed < len(personas):
        print(f"\nLog in at the UI (`make ui`) with the emails above, password: {args.password}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
