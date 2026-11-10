"""Latency benchmark for `RecommendationService.recommend` (P1-11, §8.9, NFR-1).

Usage:
    uv run python -m scripts.bench_recommender [--titles 15000] [--runs 50] [--dim 384]

Builds a synthetic catalog in a temporary SQLite DB (migrated with Alembic): `--titles`
titles over the 9 seed languages, random L2-normalized vectors (90% of titles embedded),
warm US availability for every title (so no TMDB call is made). Then times `--runs`
requests over several users with different preferences (3 seeds each, plus a cold-start
user) and prints p50 / p95 / max. The first request (catalog snapshot build) is reported
separately; later requests reuse the shared `CatalogCache`, as the API does.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import insert
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.base import utcnow
from app.db.models import (
    NO_PROVIDER_ID,
    Era,
    Interaction,
    InteractionKind,
    InteractionSource,
    OfferType,
    OnboardingStep,
    Provider,
    Title,
    TitleAvailability,
    TitleType,
    User,
    UserPlatform,
    UserPreferences,
)
from app.db.session import create_db_engine
from app.ingest.build_embeddings import EmbeddingIndex, key_str, save_index
from app.recsys.availability import AvailabilityService
from app.recsys.embeddings import EmbeddingStore
from app.recsys.service import CatalogCache, RecommendationService

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("en", "hi", "ko", "ja", "es", "fr", "ta", "te", "ml")
GENRES = ("drama", "comedy", "action", "thriller", "romance", "science-fiction", "horror")
PROVIDERS = {8: "Netflix", 9: "Amazon Prime Video", 1899: "Max", 15: "Hulu"}


def _no_tmdb() -> Any:
    raise RuntimeError("benchmark expects a warm availability cache")


def build(db_url: str, emb_dir: Path, n_titles: int, dim: int, seed: int) -> None:
    cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["skip_logging_config"] = True
    command.upgrade(cfg, "head")

    rng = np.random.default_rng(seed)
    now = utcnow()
    titles: list[dict[str, Any]] = []
    avail: list[dict[str, Any]] = []
    ids: list[str] = []
    for i in range(n_titles):
        type_ = TitleType.MOVIE if i % 3 else TitleType.TV
        tmdb_id = 100_000 + i
        titles.append(
            {
                "tmdb_id": tmdb_id,
                "type": type_,
                "title": f"Title {i}",
                "original_title": f"Title {i}",
                "original_language": LANGUAGES[(i // 3) % len(LANGUAGES)],
                "genres": [GENRES[int(g)] for g in rng.choice(len(GENRES), 2, replace=False)],
                "keywords": [],
                "directors": [],
                "top_cast": [],
                "release_date": date(1970, 1, 1) + timedelta(days=int(rng.integers(0, 20_000))),
                "runtime": int(rng.integers(20, 200)),
                "vote_avg": float(rng.uniform(3, 9)),
                "vote_count": int(rng.integers(0, 20_000)),
                "popularity": float(rng.uniform(0, 1000)),
                "adult": False,
                "certification": {"US": "PG-13"},
                "localized_titles": {},
                "updated_at": now,
            }
        )
        if i % 10:
            ids.append(key_str((tmdb_id, type_)))
        offers = [pid for pid in PROVIDERS if rng.random() < 0.3] or [NO_PROVIDER_ID]
        avail.extend(
            {
                "tmdb_id": tmdb_id,
                "type": type_,
                "country": "US",
                "provider_id": pid,
                "offer_type": OfferType.NONE if pid == NO_PROVIDER_ID else OfferType.FLATRATE,
                "link": None,
                "fetched_at": now,
            }
            for pid in offers
        )
    engine = create_db_engine(db_url)
    with Session(engine) as session:
        session.execute(insert(Title), titles)
        session.execute(insert(TitleAvailability), avail)
        session.add_all(
            Provider(provider_id=p, country="US", name=name, display_priority=1)
            for p, name in PROVIDERS.items()
        )
        session.commit()
    engine.dispose()
    vectors = rng.normal(size=(len(ids), dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    save_index(emb_dir, EmbeddingIndex(ids=ids, vectors=vectors, meta={"model": "bench"}))


def make_users(session: Session, seed: int) -> list[User]:
    rng = random.Random(seed)  # noqa: S311 - benchmark data
    profiles: list[dict[str, Any]] = [
        {"languages": ["en", "hi", "ko"]},
        {"languages": ["en"], "content_types": ["movie"], "max_runtime": 120},
        {"languages": ["ko", "ja"], "era": Era.CLASSICS_PRE_2000},
        {"languages": list(LANGUAGES), "avoid_genres": ["horror"]},
        {"languages": ["ta", "te", "ml"], "era": Era.LAST_10Y},
        {"languages": ["es", "fr"], "cold_start": True},
    ]
    # Embedded movies (see `build`: i % 3 != 0 → movie, i % 10 != 0 → embedded).
    seedable = [i for i in range(1, 3000) if i % 3 and i % 10]
    users = []
    for i, prof in enumerate(profiles):
        user = User(
            email=f"bench{i}@example.com",
            password_hash="x",  # noqa: S106 - never used to log in
            display_name="B",
            country="US",
        )
        session.add(user)
        session.flush()
        session.add(
            UserPreferences(
                user_id=user.id,
                languages=prof["languages"],
                content_types=prof.get("content_types", ["movie", "tv"]),
                era=prof.get("era", Era.ANY),
                avoid_genres=prof.get("avoid_genres", []),
                max_runtime=prof.get("max_runtime"),
                onboarding_step=OnboardingStep.DONE,
            )
        )
        session.add_all(UserPlatform(user_id=user.id, provider_id=p, country="US") for p in (8, 9))
        if not prof.get("cold_start"):
            for _ in range(3):
                session.add(
                    Interaction(
                        user_id=user.id,
                        tmdb_id=100_000 + rng.choice(seedable),
                        type=TitleType.MOVIE,
                        kind=InteractionKind.RATING,
                        value=5,
                        source=InteractionSource.ONBOARDING_SEED,
                    )
                )
        users.append(user)
    session.commit()
    return users


async def run(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_url = f"sqlite:///{Path(tmp) / 'bench.db'}"
        emb_dir = Path(tmp) / "emb"
        t0 = time.perf_counter()
        build(db_url, emb_dir, args.titles, args.dim, args.seed)
        print(f"built {args.titles} titles ({args.dim}-d) in {time.perf_counter() - t0:.1f}s")

        settings = Settings(_env_file=None, database_url=db_url)  # type: ignore[call-arg]
        engine = create_db_engine(db_url)
        store = EmbeddingStore(emb_dir)
        store.load()
        cache = CatalogCache()
        with Session(engine, expire_on_commit=False) as session:
            users = make_users(session, args.seed)
        timings: list[float] = []
        first: float | None = None
        returned: list[int] = []
        for i in range(args.runs + 1):
            with Session(engine, expire_on_commit=False) as session:
                svc = RecommendationService(
                    session,
                    store=store,
                    availability=AvailabilityService(session, _no_tmdb, ttl=timedelta(days=7)),
                    settings=settings,
                    catalog_cache=cache,
                )
                start = time.perf_counter()
                recs = await svc.recommend(users[i % len(users)].id, 10)
                elapsed = (time.perf_counter() - start) * 1000
            returned.append(len(recs.items))
            if i == 0:
                first = elapsed
            else:
                timings.append(elapsed)
        engine.dispose()

    timings.sort()
    p95 = timings[max(0, round(0.95 * len(timings)) - 1)]
    print(f"first request (catalog snapshot build): {first:.0f} ms")
    print(
        f"warm requests: n={len(timings)} p50={statistics.median(timings):.0f} ms "
        f"p95={p95:.0f} ms max={timings[-1]:.0f} ms; items returned min={min(returned)}"
    )
    return 0 if p95 < args.target_ms else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--titles", type=int, default=15_000)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-ms", type=float, default=2000.0)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
