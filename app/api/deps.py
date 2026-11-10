"""Shared FastAPI dependencies: DB session, TMDB client, current user, admin guard (§5.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.security import InvalidTokenError, TokenType, decode_token
from app.db.models import User, UserStatus
from app.db.session import get_db
from app.ingest.providers import TMDBFactory
from app.ingest.tmdb_client import TMDBClient
from app.recsys.embeddings import OnDemandEmbedder

log = get_logger(__name__)

# `last_seen_at` is written at most this often per user, not on every request.
LAST_SEEN_RESOLUTION = timedelta(minutes=5)

_bearer = HTTPBearer(auto_error=False, description="Access token from /auth/login")

DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def get_tmdb_factory(settings: AppSettings) -> TMDBFactory:
    """Factory for a TMDB client; endpoints open one only when they need TMDB.

    Use as `async with factory() as tmdb:`. Tests override this dependency.
    """
    return lambda: TMDBClient.from_settings(settings)


TMDBClientFactory = Annotated[TMDBFactory, Depends(get_tmdb_factory)]


def get_embedder(request: Request) -> OnDemandEmbedder:
    """The app-wide on-demand embedder (in-memory index), created in `create_app`."""
    embedder: OnDemandEmbedder = request.app.state.embedder
    return embedder


Embedder = Annotated[OnDemandEmbedder, Depends(get_embedder)]


def tmdb_unavailable() -> HTTPException:
    """503 for endpoints that need TMDB and couldn't reach it (or have no credentials)."""
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="The title catalog is temporarily unavailable. Try again later.",
        headers={"Retry-After": "60"},
    )


def _unauthorized(detail: str = "Invalid or expired access token.") -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"}
    )


def touch_last_seen(user: User, now: datetime | None = None) -> bool:
    """Set `last_seen_at` if unset or stale; return whether it changed (caller commits)."""
    now = now or datetime.now(UTC)
    if user.last_seen_at is None or now - user.last_seen_at >= LAST_SEEN_RESOLUTION:
        user.last_seen_at = now
        return True
    return False


def get_current_user(
    db: DbSession,
    settings: AppSettings,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Missing bearer token.")
    try:
        claims = decode_token(settings, credentials.credentials, TokenType.ACCESS)
    except InvalidTokenError as exc:
        log.info("auth_token_rejected", reason=str(exc))
        raise _unauthorized() from exc
    user = db.get(User, claims.sub)
    if user is None or user.status is UserStatus.DELETED:
        raise _unauthorized()
    if touch_last_seen(user):
        db.commit()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Admin privileges required.")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
