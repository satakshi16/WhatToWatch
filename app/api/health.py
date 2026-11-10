"""Liveness / readiness endpoints (§7, P1-01)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.errors import problem_response
from app.core.logging import get_logger
from app.db.session import check_database, get_engine

router = APIRouter(tags=["health"])
log = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: always 200 while the process is serving requests."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"description": "Not ready", "content": {"application/problem+json": {}}}},
)
def ready(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> ReadyResponse | JSONResponse:
    """Readiness: database reachable and the embedding index loaded (§8.9)."""
    checks: dict[str, str] = {}
    try:
        check_database(get_engine(settings.database_url))
        checks["database"] = "ok"
    except Exception as exc:
        log.warning("readiness_check_failed", check="database", exc_type=type(exc).__name__)
        checks["database"] = "unavailable"

    store = request.app.state.embedder.store
    checks["embeddings"] = "ok" if store.loaded else "unavailable"

    if all(v == "ok" for v in checks.values()):
        return ReadyResponse(status="ready", checks=checks)
    return problem_response(
        request, 503, detail="One or more dependencies are unavailable.", checks=checks
    )
