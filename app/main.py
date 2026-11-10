"""FastAPI app factory: config, logging, middleware, error handlers, routers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import (
    auth,
    health,
    interactions,
    me,
    onboarding,
    preferences,
    providers,
    recommendations,
    titles,
)
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import REQUEST_ID_HEADER, RequestIdMiddleware, configure_logging
from app.recsys.embeddings import OnDemandEmbedder
from app.recsys.service import CatalogCache

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load the embedding index once at startup (§8.9); `/ready` fails until it is in.
        await asyncio.to_thread(app.state.embedder.store.load)
        yield

    app = FastAPI(title="WhatToWatch API", version=__version__, lifespan=lifespan)
    # Endpoints depending on get_settings see the instance this app was built with.
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.title_search_cache = titles.new_search_cache(settings)
    # In-memory embedding index (loaded on first use) + on-demand embedding (§8.1).
    app.state.embedder = OnDemandEmbedder.from_settings(settings)
    # Recommender's `titles` snapshot, shared by requests, rebuilt when the table changes.
    app.state.catalog_cache = CatalogCache()

    # Middleware added last runs first: RequestId wraps CORS so preflights get an ID too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    app.add_middleware(RequestIdMiddleware)

    register_exception_handlers(app)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(me.router, prefix=API_PREFIX)
    app.include_router(preferences.router, prefix=API_PREFIX)
    app.include_router(providers.router, prefix=API_PREFIX)
    app.include_router(titles.router, prefix=API_PREFIX)
    app.include_router(onboarding.router, prefix=API_PREFIX)
    app.include_router(interactions.router, prefix=API_PREFIX)
    app.include_router(recommendations.router, prefix=API_PREFIX)
    return app


app = create_app()
