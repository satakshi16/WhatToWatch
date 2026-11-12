"""`POST /onboarding/seeds` (P1-09, FR-13): 3-10 titles the user likes; `GET` lists them.

Each seed is stored as `interactions(kind=rating, value=5, source=onboarding_seed)`, all
rows of one submission sharing `payload.seed_set`. A new submission **replaces** the seed
set: earlier seed rows stay in the append-only log but stop shaping the profile.
Seeds not in the catalog are fetched from TMDB, inserted and embedded on the fly (§8.1),
so they shape recommendations immediately. Saving seeds finishes onboarding
(`onboarding_step = done`; P2-07 inserts `quick_rate` before `done`).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, tuple_

from app.api.deps import AppSettings, CurrentUser, DbSession, Embedder, TMDBClientFactory
from app.api.interactions import TitleKey, ensure_titles_or_http
from app.core.logging import get_logger
from app.db.models import InteractionKind, InteractionSource, OnboardingStep, Title, UserPreferences
from app.recsys.feedback import SEED_SET_KEY, add_interactions, current_seed_keys
from app.schemas.feedback import CurrentSeedOut, CurrentSeedsOut, SeedOut, SeedsIn, SeedsOut

router = APIRouter(prefix="/onboarding", tags=["onboarding"])
log = get_logger(__name__)

SEED_RATING = 5.0


@router.get("/seeds", response_model=CurrentSeedsOut)
def list_seeds(user: CurrentUser, db: DbSession) -> CurrentSeedsOut:
    """The current seed set, in the order it was submitted."""
    keys = current_seed_keys(db, user.id)
    titles: dict[TitleKey, Title] = {}
    if keys:
        rows = db.scalars(select(Title).where(tuple_(Title.tmdb_id, Title.type).in_(keys)))
        titles = {(t.tmdb_id, t.type): t for t in rows}
    return CurrentSeedsOut(
        seeds=[
            CurrentSeedOut(
                tmdb_id=t.tmdb_id,
                type=t.type,
                title=t.title,
                original_title=t.original_title,
                year=t.release_date.year if t.release_date else None,
            )
            for key in keys
            if (t := titles.get(key)) is not None
        ]
    )


@router.post("/seeds", response_model=SeedsOut, status_code=status.HTTP_201_CREATED)
async def submit_seeds(
    body: SeedsIn,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    tmdb_factory: TMDBClientFactory,
    embedder: Embedder,
) -> SeedsOut:
    """Save 3-10 distinct seed titles (duplicates are dropped) and finish onboarding.

    Preferences must be saved first (409 otherwise). Submitting again replaces the seed
    set: seeds left out stop shaping recommendations (their rows stay in the log, and
    they are still never recommended, FR-23). Card ratings are not affected.
    """
    prefs = db.get(UserPreferences, user.id)
    if prefs is None or prefs.onboarding_step is OnboardingStep.PREFERENCES:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="Save your preferences before choosing seed titles."
        )
    keys: list[TitleKey] = [(s.tmdb_id, s.type) for s in body.seeds]
    position = {key: i for i, key in enumerate(keys)}
    ensured = await ensure_titles_or_http(
        db,
        keys,
        user=user,
        settings=settings,
        tmdb_factory=tmdb_factory,
        embedder=embedder,
        loc_of=lambda key: ("body", "seeds", position[key]),
    )
    add_interactions(
        db,
        user.id,
        keys,
        kind=InteractionKind.RATING,
        source=InteractionSource.ONBOARDING_SEED,
        value=SEED_RATING,
        payload={SEED_SET_KEY: uuid.uuid4().hex},
    )
    prefs.onboarding_step = OnboardingStep.DONE
    db.commit()

    seeds = []
    for key in keys:
        title = db.get(Title, key)
        assert title is not None  # noqa: S101 - ensure_titles inserted it
        seeds.append(
            SeedOut(
                tmdb_id=key[0],
                type=key[1],
                title=title.title,
                original_title=title.original_title,
                added_to_catalog=key in ensured.inserted,
                embedded=ensured.embedded.get(key, False),
            )
        )
    log.info(
        "onboarding_seeds_saved",
        user_id=str(user.id),
        seeds=len(keys),
        added_to_catalog=len(ensured.inserted),
        embedded=sum(ensured.embedded.values()),
    )
    return SeedsOut(onboarding_step=prefs.onboarding_step, seeds=seeds)
