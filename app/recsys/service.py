"""`RecommendationService.recommend(user_id, n)`: the §8 pipeline (P1-11).

    profile → hard-filter mask → candidates → scoring → availability → MMR re-rank → reasons

This is the only recsys module that touches the database; the steps themselves are pure
functions in `profile`, `filters`, `candidates`, `scorer`, `rerank` and `reasons`.

Shared, process-wide state lives in `CatalogCache` (the `titles` snapshot, rebuilt only
when the table changes) and the app's `EmbeddingStore` (loaded at startup, plus vectors
appended at runtime by `OnDemandEmbedder`). Create one service per request/session.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.base import utcnow
from app.db.models import Interaction, Title, TitleType, User
from app.ingest.build_embeddings import TitleKey, Vectors
from app.ingest.providers import TMDBFactory
from app.recsys.availability import (
    Availability,
    AvailabilityService,
    Offer,
    streamable_offers,
)
from app.recsys.candidates import (
    item_similarity,
    similar_candidates,
    top_k,
    trending_candidates,
    union,
)
from app.recsys.catalog import NO_ROW, Catalog, FloatArray, IntArray, TitleRecord
from app.recsys.embeddings import EmbeddingStore
from app.recsys.feedback import excluded_title_keys
from app.recsys.filters import UserFilters, filter_mask
from app.recsys.profile import EXPLICIT_KINDS, Profile, Signal, build_profile
from app.recsys.reasons import REASON_SOURCE, template_reason
from app.recsys.rerank import mmr_rerank
from app.recsys.scorer import (
    Components,
    Weights,
    bayesian_quality,
    cold_start_components,
    cold_start_scores,
    popularity_score,
    recency_score,
    score,
)

log = get_logger(__name__)

# Bump when the pipeline changes behaviour in a way the tunables don't capture.
ALGORITHM_VERSION = "p1-weighted-1"
SLOT_CORE = "core"


class RecommendationError(Exception):
    """The user cannot get recommendations yet."""


class UserNotFoundError(RecommendationError):
    pass


class PreferencesMissingError(RecommendationError):
    """The user has not saved preferences (onboarding incomplete)."""


# --- Config hash (NFR-18) ------------------------------------------------------


def recommender_config(settings: Settings) -> dict[str, Any]:
    """Every tunable that changes the output (§5.2); hashed into `config_hash`."""
    return {
        "algorithm": ALGORITHM_VERSION,
        "candidates_similar": settings.candidates_similar,
        "candidates_trending": settings.candidates_trending,
        "w_sim": settings.w_sim,
        "w_qual": settings.w_qual,
        "w_pop": settings.w_pop,
        "w_rec": settings.w_rec,
        "w_neg": settings.w_neg,
        "bayes_m": settings.bayes_m,
        "recency_half_life_days": settings.recency_half_life_days,
        "profile_half_life_days": settings.profile_half_life_days,
        "mmr_lambda": settings.mmr_lambda,
        "cap_per_language": settings.cap_per_language,
        "cap_per_genre": settings.cap_per_genre,
        "max_runtime_applies_to_tv": settings.max_runtime_applies_to_tv,
        "availability_buffer": settings.availability_buffer,
        "availability_max_fetches_per_request": settings.availability_max_fetches_per_request,
        "adult_certifications": {
            k: sorted(v) for k, v in sorted(settings.adult_certifications.items())
        },
    }


def config_hash(settings: Settings) -> str:
    payload = json.dumps(recommender_config(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- Shared catalog snapshot ---------------------------------------------------


@dataclass(eq=False)
class _Snapshot:
    fingerprint: tuple[Any, ...]
    catalog: Catalog
    popularity: FloatArray
    quality: dict[float, FloatArray] = field(default_factory=dict)
    # (store identity, store length) → (vector row per catalog row, matrix)
    vectors_for: tuple[int, int] | None = None
    vector_rows: IntArray | None = None
    matrix: Vectors | None = None


class CatalogCache:
    """Process-wide `titles` snapshot; rebuilt when the row count or `max(updated_at)` changes.

    The check is one aggregate query per request. Vector alignment with the embedding
    store is redone only when the store grows (runtime appends) or is replaced.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: _Snapshot | None = None
        self.builds = 0

    def get(self, session: Session) -> _Snapshot:
        count, latest = session.execute(select(func.count(), func.max(Title.updated_at))).one()
        fingerprint = (int(count), latest)
        with self._lock:
            if self._snapshot is None or self._snapshot.fingerprint != fingerprint:
                catalog = Catalog.from_rows(_load_records(session))
                self._snapshot = _Snapshot(fingerprint, catalog, popularity_score(catalog))
                self.builds += 1
                log.info("recsys_catalog_loaded", titles=len(catalog))
            return self._snapshot

    def quality(self, snap: _Snapshot, m: float) -> FloatArray:
        with self._lock:
            q = snap.quality.get(m)
            if q is None:
                q = snap.quality[m] = bayesian_quality(snap.catalog, m)
            return q

    def vectors(self, snap: _Snapshot, store: EmbeddingStore) -> tuple[IntArray, Vectors | None]:
        """Embedding row per catalog row (NO_ROW if none) and the matrix they index."""
        size = len(store)
        tag = (id(store), size)
        with self._lock:
            if snap.vectors_for != tag or snap.vector_rows is None:
                shot = store.snapshot()
                if shot is None:
                    snap.vector_rows = np.full(len(snap.catalog), NO_ROW, dtype=np.int64)
                    snap.matrix = None
                else:
                    ids, matrix = shot
                    snap.vector_rows, snap.matrix = snap.catalog.vector_rows(ids), matrix
                snap.vectors_for = tag
            return snap.vector_rows, snap.matrix


def _model_version(meta: dict[str, Any]) -> str | None:
    model, version = meta.get("model"), meta.get("model_version")
    if model and version:
        return f"{model}@{version}"
    return str(model) if model else None


def _load_records(session: Session) -> list[TitleRecord]:
    stmt = select(
        Title.tmdb_id,
        Title.type,
        Title.title,
        Title.original_language,
        Title.genres,
        Title.release_date,
        Title.runtime,
        Title.vote_avg,
        Title.vote_count,
        Title.popularity,
        Title.adult,
        Title.certification,
    ).order_by(Title.type, Title.tmdb_id)
    return [
        TitleRecord(
            tmdb_id=r.tmdb_id,
            type=r.type,
            title=r.title,
            original_language=r.original_language,
            genres=tuple(r.genres or ()),
            release_date=r.release_date,
            runtime=r.runtime,
            vote_avg=r.vote_avg,
            vote_count=r.vote_count or 0,
            popularity=r.popularity or 0.0,
            adult=bool(r.adult),
            certification=dict(r.certification or {}),
        )
        for r in session.execute(stmt)
    ]


# --- Results -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecommendedItem:
    rank: int
    key: TitleKey
    score: float
    components: dict[str, float]
    reason: str
    reason_source: str
    slot: str
    # Which candidate pool the title came from: "similar", "trending" or "cold_start".
    pool: str
    availability: Availability
    offers: tuple[Offer, ...]  # FR-22 offers on the user's platforms, display order

    @property
    def tmdb_id(self) -> int:
        return self.key[0]

    @property
    def type(self) -> TitleType:
        return self.key[1]


@dataclass(frozen=True, slots=True)
class Recommendations:
    user_id: uuid.UUID
    n: int
    items: list[RecommendedItem]
    cold_start: bool
    config_hash: str
    model_version: str | None
    generated_at: datetime
    debug: dict[str, Any]


# --- Service -------------------------------------------------------------------


class RecommendationService:
    def __init__(
        self,
        session: Session,
        *,
        store: EmbeddingStore,
        availability: AvailabilityService,
        settings: Settings,
        catalog_cache: CatalogCache | None = None,
        clock: Any = utcnow,
    ) -> None:
        self._session = session
        self._store = store
        self._availability = availability
        self._settings = settings
        self._cache = catalog_cache or CatalogCache()
        self._clock = clock

    @classmethod
    def from_settings(
        cls,
        session: Session,
        *,
        store: EmbeddingStore,
        tmdb_factory: TMDBFactory,
        settings: Settings,
        catalog_cache: CatalogCache | None = None,
        clock: Any = utcnow,
    ) -> RecommendationService:
        availability = AvailabilityService.from_settings(session, tmdb_factory, settings)
        return cls(
            session,
            store=store,
            availability=availability,
            settings=settings,
            catalog_cache=catalog_cache,
            clock=clock,
        )

    @property
    def weights(self) -> Weights:
        s = self._settings
        return Weights(sim=s.w_sim, qual=s.w_qual, pop=s.w_pop, rec=s.w_rec, neg=s.w_neg)

    async def recommend(self, user_id: uuid.UUID, n: int | None = None) -> Recommendations:
        s = self._settings
        n = n or s.rec_n_default
        if n < 1:
            raise ValueError("n must be >= 1")
        timings: dict[str, float] = {}
        t0 = mark = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal mark
            now_ = time.perf_counter()
            timings[name] = round((now_ - mark) * 1000, 2)
            mark = now_

        now: datetime = self._clock()
        today = now.date()
        user = self._session.get(User, user_id)
        if user is None:
            raise UserNotFoundError(str(user_id))
        prefs = user.preferences
        if prefs is None:
            raise PreferencesMissingError(str(user_id))
        country = user.country
        platforms = sorted({p.provider_id for p in user.platforms if p.country == country})

        snap = self._cache.get(self._session)
        catalog = snap.catalog
        quality = self._cache.quality(snap, float(s.bayes_m))
        vector_rows, matrix = self._cache.vectors(snap, self._store)
        recency = recency_score(catalog, today, s.recency_half_life_days)
        lap("catalog")

        def vector_of(key: TitleKey) -> Vectors | None:
            row = catalog.index.get(key)
            if row is not None and matrix is not None and vector_rows[row] != NO_ROW:
                v: Vectors = matrix[vector_rows[row]]
                return v
            return self._store.vector(key)  # e.g. a title not in the catalog snapshot

        profile = build_profile(
            self._signals(user_id), vector_of, now=now, half_life_days=s.profile_half_life_days
        )
        lap("profile")

        excluded = excluded_title_keys(self._session, user_id)
        filters = UserFilters(
            languages=list(prefs.languages),
            content_types=[TitleType(t) for t in prefs.content_types],
            era=prefs.era,
            avoid_genres=list(prefs.avoid_genres),
            max_runtime=prefs.max_runtime,
            allow_adult=prefs.allow_adult,
            adult_certifications=s.adult_certifications.get(country, []),
            country=country,
            excluded=frozenset(excluded),
            max_runtime_applies_to_tv=s.max_runtime_applies_to_tv,
        )
        mask = filter_mask(catalog, filters, today)
        lap("filters")

        pools: dict[int, str] = {}
        if profile.vector is not None and matrix is not None:
            sims = item_similarity(matrix, vector_rows, profile.vector)
            similar = similar_candidates(sims, mask, s.candidates_similar)
            trending = trending_candidates(catalog.popularity, mask, s.candidates_trending)
            rows = union(similar, trending)
            pools = {int(r): "similar" for r in similar.tolist()}
            for r in trending.tolist():
                pools.setdefault(int(r), "trending")
            neg_sims = (
                item_similarity(matrix, vector_rows, profile.neg_vector)
                if profile.neg_vector is not None and s.w_neg > 0
                else None
            )
            comps = score(
                rows,
                sims=sims,
                neg_sims=neg_sims,
                quality=quality,
                pop=snap.popularity,
                recency=recency,
                weights=self.weights,
            )
            counts = {
                "candidates_similar": int(similar.size),
                "candidates_trending": int(trending.size),
            }
        else:
            cold = cold_start_scores(quality, snap.popularity, recency)
            rows = top_k(cold, mask, s.candidates_similar + s.candidates_trending)
            pools = {int(r): "cold_start" for r in rows.tolist()}
            comps = cold_start_components(
                rows, quality=quality, pop=snap.popularity, recency=recency
            )
            counts = {"candidates_similar": 0, "candidates_trending": 0}
        lap("candidates_scoring")

        order = np.lexsort((comps.rows, -comps.score)) if rows.size else np.zeros(0, np.int64)
        survivors, checked = await self._available(
            catalog, comps, order, country, platforms, prefs.include_rent_buy, n
        )
        lap("availability")

        cand_idx = [i for i, _, _ in survivors]
        sel_rows = comps.rows[cand_idx] if cand_idx else np.zeros(0, dtype=np.int64)
        dim = matrix.shape[1] if matrix is not None else 1
        sel_vectors = np.zeros((len(cand_idx), dim), dtype=np.float32)
        for j, row in enumerate(sel_rows.tolist()):
            if matrix is not None and vector_rows[row] != NO_ROW:
                sel_vectors[j] = matrix[vector_rows[row]]
        reranked = mmr_rerank(
            comps.score[cand_idx] if cand_idx else np.zeros(0),
            sel_vectors,
            [catalog.languages[r] for r in sel_rows.tolist()],
            [catalog.primary_genre(r) for r in sel_rows.tolist()],
            n=n,
            lam=s.mmr_lambda,
            cap_per_language=s.cap_per_language,
            cap_per_genre=s.cap_per_genre,
            single_language=len(set(prefs.languages)) == 1,
        )
        lap("rerank")

        liked_names, liked_vectors = self._liked(profile, catalog, vector_of, dim)
        items: list[RecommendedItem] = []
        for rank, j in enumerate(reranked.order, start=1):
            i, avail, offers = survivors[j]
            row = int(comps.rows[i])
            platform = next((o.provider_name for o in offers if o.provider_name), None)
            reason = template_reason(
                vector=sel_vectors[j] if vector_rows[row] != NO_ROW else None,
                liked_vectors=liked_vectors,
                liked_names=liked_names,
                language=catalog.languages[row],
                platform=platform,
                cold_start=profile.cold_start,
            )
            items.append(
                RecommendedItem(
                    rank=rank,
                    key=catalog.keys[row],
                    score=float(comps.score[i]),
                    components=comps.of(i),
                    reason=reason,
                    reason_source=REASON_SOURCE,
                    slot=SLOT_CORE,
                    pool=pools.get(row, "similar"),
                    availability=avail,
                    offers=tuple(offers),
                )
            )
        lap("reasons")
        timings["total"] = round((time.perf_counter() - t0) * 1000, 2)

        meta = self._store.meta or {}
        debug: dict[str, Any] = {
            "cold_start": profile.cold_start,
            "profile": {
                "signals_used": profile.signals_used,
                "signals_unembedded": profile.signals_unembedded,
                "liked": len(profile.liked),
            },
            "counts": {
                "catalog": len(catalog),
                "embedded": int((vector_rows != NO_ROW).sum()),
                "excluded": len(excluded),
                "eligible": int(mask.sum()),
                **counts,
                "candidates": int(rows.size),
                "availability_checked": checked,
                "available": len(survivors),
                "availability_fetches": self._availability.fetch_attempts,
                "platforms": len(platforms),
            },
            "caps": {"language": reranked.language_cap, "genre": reranked.genre_cap},
            "relaxations": reranked.relaxations,
            "weights": recommender_config(s),
            "timings_ms": timings,
        }
        result = Recommendations(
            user_id=user_id,
            n=n,
            items=items,
            cold_start=profile.cold_start,
            config_hash=config_hash(s),
            model_version=_model_version(meta),
            generated_at=now,
            debug=debug,
        )
        log.info(
            "recommendations_generated",
            user_id=str(user_id),
            n=n,
            returned=len(items),
            cold_start=profile.cold_start,
            eligible=debug["counts"]["eligible"],
            candidates=debug["counts"]["candidates"],
            available=len(survivors),
            relaxations=len(reranked.relaxations),
            ms=timings["total"],
        )
        return result

    # --- Steps that read the DB / network ------------------------------------------

    def _signals(self, user_id: uuid.UUID) -> list[Signal]:
        stmt = select(
            Interaction.id,
            Interaction.tmdb_id,
            Interaction.type,
            Interaction.kind,
            Interaction.value,
            Interaction.created_at,
        ).where(Interaction.user_id == user_id, Interaction.kind.in_(EXPLICIT_KINDS))
        return [
            Signal(
                key=(r.tmdb_id, r.type),
                kind=r.kind,
                value=r.value,
                created_at=r.created_at,
                seq=r.id,
            )
            for r in self._session.execute(stmt)
        ]

    async def _available(
        self,
        catalog: Catalog,
        comps: Components,
        order: IntArray,
        country: str,
        platforms: list[int],
        include_rent_buy: bool,
        n: int,
    ) -> tuple[list[tuple[int, Availability, list[Offer]]], int]:
        """§8.6: walk the scored list in order, keep FR-22 titles until N x buffer survive.

        Returns (candidate index, availability, streamable offers) per survivor and how
        many titles were looked up. One TMDB fetch budget covers the whole walk.
        """
        if not platforms or order.size == 0:
            return [], 0
        s = self._settings
        target = n * max(s.availability_buffer, 1)
        chunk = max(target, 2 * n, 30)
        budget = s.availability_max_fetches_per_request
        survivors: list[tuple[int, Availability, list[Offer]]] = []
        checked = 0
        start_attempts = self._availability.fetch_attempts
        for start in range(0, order.size, chunk):
            idx = [int(i) for i in order[start : start + chunk].tolist()]
            keys = [catalog.keys[int(comps.rows[i])] for i in idx]
            used = self._availability.fetch_attempts - start_attempts
            found = await self._availability.bulk_get(
                keys, country, max_fetches=max(budget - used, 0)
            )
            checked += len(keys)
            for i, key in zip(idx, keys, strict=True):
                offers = streamable_offers(found[key], platforms, include_rent_buy)
                if offers:
                    survivors.append((i, found[key], offers))
                    if len(survivors) >= target:
                        return survivors, checked
        return survivors, checked

    def _liked(
        self, profile: Profile, catalog: Catalog, vector_of: Any, dim: int
    ) -> tuple[list[str], Vectors]:
        names: list[str] = []
        vectors: list[Vectors] = []
        for key in profile.liked:
            row = catalog.index.get(key)
            vector = vector_of(key)
            if row is None or vector is None:
                continue
            names.append(catalog.titles[row])
            vectors.append(vector)
        if not vectors:
            return [], np.zeros((0, dim), dtype=np.float32)
        return names, np.stack(vectors).astype(np.float32)
