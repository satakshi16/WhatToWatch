"""`POST /auth/register`, `POST /auth/login` (P1-06, FR-1, FR-2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.api.deps import AppSettings, DbSession, touch_last_seen
from app.core.config import Settings
from app.core.geo import default_timezone
from app.core.logging import get_logger
from app.core.security import (
    TokenType,
    create_token,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from app.db.models import User, UserStatus
from app.schemas.auth import LoginRequest, RegisterRequest, RegisterResponse, TokenResponse
from app.schemas.user import UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)

_PROBLEM: dict[str, Any] = {"content": {"application/problem+json": {}}}
BAD_CREDENTIALS = "Incorrect email or password."


def _tokens(settings: Settings, user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_token(settings, user.id, TokenType.ACCESS),
        refresh_token=create_token(settings, user.id, TokenType.REFRESH),
        expires_in=settings.access_token_ttl_min * 60,
    )


def _email_taken() -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, detail="An account with this email exists.")


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=RegisterResponse,
    responses={409: {"description": "Email already registered", **_PROBLEM}},
)
def register(body: RegisterRequest, db: DbSession, settings: AppSettings) -> RegisterResponse:
    # Emails are stored lower-cased (DB CHECK), so equality is case-insensitive.
    if db.scalar(select(func.count()).select_from(User).where(User.email == body.email)):
        raise _email_taken()
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        country=body.country,
        timezone=body.timezone or default_timezone(body.country),
        last_seen_at=datetime.now(UTC),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:  # concurrent registration with the same email
        db.rollback()
        raise _email_taken() from exc
    log.info("user_registered", user_id=str(user.id), country=user.country)
    tokens = _tokens(settings, user)
    return RegisterResponse(user=UserOut.model_validate(user), **tokens.model_dump())


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={401: {"description": "Bad credentials", **_PROBLEM}},
)
def login(body: LoginRequest, db: DbSession, settings: AppSettings) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    ok = verify_password(body.password, user.password_hash if user else None)
    if user is None or not ok or user.status is UserStatus.DELETED:
        log.info("login_failed")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail=BAD_CREDENTIALS,
            headers={"WWW-Authenticate": "Bearer"},
        )
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
    touch_last_seen(user)
    db.commit()
    log.info("login_succeeded", user_id=str(user.id))
    return _tokens(settings, user)
