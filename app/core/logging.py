"""structlog JSON logging + X-Request-ID middleware (§5.4, NFR-16)."""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from typing import Any

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")

_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def configure_logging(level: str = "INFO") -> None:
    """Route both structlog and stdlib (incl. uvicorn) logs to stdout as JSON lines."""
    log_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = _StdoutHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    # Make uvicorn log through the root JSON handler; our middleware emits access logs.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    logging.getLogger("uvicorn.access").disabled = True

    # httpx/httpcore log full request URLs at INFO/DEBUG, which can include the
    # TMDB `api_key` query param. Keep them at WARNING so credentials never reach logs.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


class _StdoutHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """StreamHandler that resolves `sys.stdout` at emit time (plays well with test capture)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def get_logger(name: str | None = None) -> Any:
    return structlog.stdlib.get_logger(name)


def _sanitize_request_id(raw: str | None) -> str:
    if raw and _REQUEST_ID_RE.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


class RequestIdMiddleware:
    """Ensure every request has a request ID: accept a safe incoming `X-Request-ID`
    or generate one, bind it to the log context, echo it on the response, and emit
    one JSON access-log line per request.

    The ID is also stored in `scope["state"]["request_id"]` so exception handlers that
    run outside this middleware (unhandled 500s) can attach it too.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log = get_logger("app.access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for key, value in scope.get("headers", []):
            if key == b"x-request-id":
                incoming = value.decode("latin-1")
                break
        request_id = _sanitize_request_id(incoming)
        scope.setdefault("state", {})["request_id"] = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status_code = 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.log.info(
                "request",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
