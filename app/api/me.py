"""`GET /me`, `PATCH /me` (P1-06, FR-5; platform cleanup on country change P1-07)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import AppSettings, CurrentUser, DbSession, TMDBClientFactory
from app.api.preferences import platform_ids, set_platforms
from app.api.providers import load_country_providers
from app.core.logging import get_logger
from app.schemas.user import UserOut, UserUpdate, UserUpdateOut

router = APIRouter(prefix="/me", tags=["me"])
log = get_logger(__name__)


@router.get("", response_model=UserOut)
def get_me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)


@router.patch("", response_model=UserUpdateOut)
async def update_me(
    body: UserUpdate,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    tmdb_factory: TMDBClientFactory,
) -> UserUpdateOut:
    """Update the fields sent. Changing country does not change the timezone.

    Changing country keeps the user's platforms that also exist in the new country and
    removes the rest; the removed provider IDs are returned as `removed_platforms`.
    If the new country's provider list can't be loaded, nothing is changed (503).
    """
    changes = body.model_dump(exclude_unset=True)
    removed: list[int] = []
    new_country = changes.get("country")
    if new_country is not None and new_country != user.country:
        current = platform_ids(db, user)
        if current:
            available = {
                p.provider_id
                for p in await load_country_providers(db, new_country, tmdb_factory, settings)
            }
            removed = [pid for pid in current if pid not in available]
            set_platforms(db, user, new_country, [pid for pid in current if pid in available])
    for field, value in changes.items():
        setattr(user, field, value)
    db.commit()
    log.info(
        "profile_updated",
        user_id=str(user.id),
        fields=sorted(changes),
        removed_platforms=len(removed),
    )
    return UserUpdateOut(**UserOut.model_validate(user).model_dump(), removed_platforms=removed)
