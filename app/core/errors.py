"""RFC 9457 `application/problem+json` exception handlers (§5.3)."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import REQUEST_ID_HEADER, get_logger

PROBLEM_JSON = "application/problem+json"

log = get_logger(__name__)


def _phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def problem_response(
    request: Request,
    status: int,
    *,
    detail: str | None = None,
    title: str | None = None,
    type_: str = "about:blank",
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": type_,
        "title": title or _phrase(status),
        "status": status,
        "instance": request.url.path,
    }
    if detail is not None:
        body["detail"] = detail
    body.update(extra)
    return JSONResponse(body, status_code=status, headers=headers, media_type=PROBLEM_JSON)


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover - registration bug
        raise exc
    detail: str | None = exc.detail if isinstance(exc.detail, str) else None
    if detail == _phrase(exc.status_code):
        detail = None
    return problem_response(
        request, exc.status_code, detail=detail, headers=getattr(exc, "headers", None)
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - registration bug
        raise exc
    # Deliberately omit the offending `input` — it may contain secrets (e.g. passwords).
    errors = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return problem_response(
        request,
        422,
        detail="Request validation failed.",
        errors=errors,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_exception", exc_type=type(exc).__name__)
    # This handler runs outside RequestIdMiddleware, so attach the ID explicitly.
    request_id = request.scope.get("state", {}).get("request_id")
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return problem_response(request, 500, detail="An unexpected error occurred.", headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
