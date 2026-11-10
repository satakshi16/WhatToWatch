"""`GET /me/preferences`, `PUT /me/preferences` (P1-07, FR-10, FR-11)."""

from __future__ import annotations

from collections.abc import Collection

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.deps import AppSettings, CurrentUser, DbSession, TMDBClientFactory
from app.api.providers import load_country_providers
from app.core.logging import get_logger
from app.db.models import Era, OnboardingStep, TitleType, User, UserPlatform, UserPreferences
from app.db.models.user import DEFAULT_LANGUAGES
from app.schemas.preferences import PreferencesIn, PreferencesOut

router = APIRouter(prefix="/me/preferences", tags=["preferences"])
log = get_logger(__name__)


def platform_ids(db: Session, user: User, country: str | None = None) -> list[int]:
    """The user's provider IDs in `country` (default: the user's country), ascending."""
    return list(
        db.scalars(
            select(UserPlatform.provider_id)
            .where(
                UserPlatform.user_id == user.id, UserPlatform.country == (country or user.country)
            )
            .order_by(UserPlatform.provider_id)
        )
    )


def set_platforms(db: Session, user: User, country: str, provider_ids: Collection[int]) -> None:
    """Replace the user's platforms with `provider_ids` in `country` (caller commits).

    Rows for other countries are removed too: platforms only apply in the user's country.
    """
    db.execute(delete(UserPlatform).where(UserPlatform.user_id == user.id))
    db.add_all(UserPlatform(user_id=user.id, provider_id=p, country=country) for p in provider_ids)


def _to_out(db: Session, user: User, prefs: UserPreferences | None) -> PreferencesOut:
    platforms = platform_ids(db, user)
    if prefs is None:
        # Nothing saved yet: report the defaults without creating a row.
        return PreferencesOut(
            languages=list(DEFAULT_LANGUAGES),
            platforms=platforms,
            content_types=[TitleType.MOVIE, TitleType.TV],
            era=Era.ANY,
            avoid_genres=[],
            max_runtime=None,
            include_rent_buy=False,
            allow_adult=False,
            onboarding_step=OnboardingStep.PREFERENCES,
            updated_at=None,
        )
    return PreferencesOut(
        languages=prefs.languages,
        platforms=platforms,
        content_types=[TitleType(t) for t in prefs.content_types],
        era=prefs.era,
        avoid_genres=prefs.avoid_genres,
        max_runtime=prefs.max_runtime,
        include_rent_buy=prefs.include_rent_buy,
        allow_adult=prefs.allow_adult,
        onboarding_step=prefs.onboarding_step,
        updated_at=prefs.updated_at,
    )


@router.get("", response_model=PreferencesOut)
def get_preferences(user: CurrentUser, db: DbSession) -> PreferencesOut:
    return _to_out(db, user, db.get(UserPreferences, user.id))


@router.put("", response_model=PreferencesOut)
async def put_preferences(
    body: PreferencesIn,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    tmdb_factory: TMDBClientFactory,
) -> PreferencesOut:
    """Replace all preferences. Platforms must be TMDB provider IDs in the user's country."""
    if body.platforms:
        available = {
            p.provider_id
            for p in await load_country_providers(db, user.country, tmdb_factory, settings)
        }
        errors = [
            {
                "loc": ("body", "platforms", i),
                "msg": f"provider {pid} is not available in {user.country}",
                "type": "value_error.provider_not_in_country",
            }
            for i, pid in enumerate(body.platforms)
            if pid not in available
        ]
        if errors:
            raise RequestValidationError(errors)

    prefs = db.get(UserPreferences, user.id)
    first_save = prefs is None
    if prefs is None:
        prefs = UserPreferences(user_id=user.id)
        db.add(prefs)
    prefs.languages = list(body.languages)
    prefs.content_types = [t.value for t in body.content_types]
    prefs.era = body.era
    prefs.avoid_genres = list(body.avoid_genres)
    prefs.max_runtime = body.max_runtime
    prefs.include_rent_buy = body.include_rent_buy
    prefs.allow_adult = body.allow_adult
    if first_save or prefs.onboarding_step is OnboardingStep.PREFERENCES:
        prefs.onboarding_step = OnboardingStep.SEEDS
    set_platforms(db, user, user.country, body.platforms)
    db.commit()
    db.refresh(prefs)
    log.info(
        "preferences_saved",
        user_id=str(user.id),
        first_save=first_save,
        platforms=len(body.platforms),
    )
    return _to_out(db, user, prefs)
