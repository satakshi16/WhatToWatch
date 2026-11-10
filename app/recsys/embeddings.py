"""In-memory embedding index + on-demand embedding of new titles (§8.1, P1-09).

`EmbeddingStore` holds the index written by `make embed` (`embeddings.npy`, `id_map.json`,
`embeddings_meta.json`), loaded lazily on first use, and accepts vectors appended at
runtime. Appended vectors live in memory only; their open content is stored in
`title_open_content`, so the next `make embed --incremental` persists them.

`OnDemandEmbedder` makes titles usable for similarity right away (no re-embed): for keys
missing from the index it fetches open content (Wikidata + Wikipedia, D-011) for titles
never checked, embeds those with a summary via `embed_one`, and appends the vectors.
Embedding is best effort: if the index, the encoder or Wikimedia is unavailable the
caller still gets an answer, and the titles are reported as not embedded.

P1-11 loads the index at app startup (`create_app` lifespan), reports it in `/ready`
(`EmbeddingStore.loaded`) and reads `snapshot()` (ids + matrix, appended rows included)
for scoring.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models import TitleOpenContent
from app.ingest.build_embeddings import (
    EmbeddingError,
    EmbeddingIndex,
    Encoder,
    SentenceTransformerEncoder,
    TitleKey,
    Vectors,
    embed_one,
    key_str,
    load_index,
)
from app.ingest.open_content import (
    fetch_open_content,
    label_languages_for,
    upsert_open_content,
    wiki_languages_for,
)
from app.ingest.wikimedia_client import WikimediaClient, WikimediaError

log = get_logger(__name__)

EncoderFactory = Callable[[], Encoder]
WikimediaFactory = Callable[[], WikimediaClient]


class EmbeddingStore:
    """Thread-safe in-memory index: vectors by title key, loaded from disk once.

    Appended rows are buffered and folded into the matrix when `matrix()` is read, so
    an append never copies the whole matrix.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._lock = threading.Lock()
        self._attempted = False
        self._index: EmbeddingIndex | None = None
        self._rows: dict[str, int] = {}
        self._extra: list[Vectors] = []
        self._loaded_count = 0

    def load(self) -> bool:
        """Load the index from disk on first call; return whether one is available.

        A missing or broken index is not retried until `reload()`.
        """
        with self._lock:
            if not self._attempted:
                self._attempted = True
                try:
                    self._index = load_index(self._directory)
                except (EmbeddingError, OSError, ValueError) as exc:
                    log.warning("embeddings_index_unreadable", error=str(exc))
                    self._index = None
                if self._index is None:
                    log.warning("embeddings_index_missing", directory=str(self._directory))
                else:
                    self._rows = {k: i for i, k in enumerate(self._index.ids)}
                    self._loaded_count = len(self._rows)
                    log.info("embeddings_index_loaded", count=self._loaded_count)
            return self._index is not None

    def reload(self) -> bool:
        with self._lock:
            self._attempted = False
            self._index, self._rows, self._extra, self._loaded_count = None, {}, [], 0
        return self.load()

    @property
    def loaded(self) -> bool:
        """Whether an index is in memory (does not trigger a load; used by `/ready`)."""
        return self._index is not None

    @property
    def appended(self) -> int:
        """Vectors added at runtime since the index was loaded."""
        return len(self._rows) - self._loaded_count

    @property
    def meta(self) -> dict[str, Any] | None:
        return self._index.meta if self.load() and self._index else None

    @property
    def dim(self) -> int | None:
        return int(self._index.vectors.shape[1]) if self.load() and self._index else None

    def ids(self) -> list[str]:
        """Row order of `matrix()` (`"<type>:<tmdb_id>"`)."""
        if not self.load():
            return []
        with self._lock:
            assert self._index is not None  # noqa: S101 - load() returned True
            return list(self._index.ids)

    def matrix(self) -> Vectors | None:
        """All vectors (loaded + appended), rows in `ids()` order."""
        if not self.load():
            return None
        with self._lock:
            assert self._index is not None  # noqa: S101 - load() returned True
            if self._extra:
                self._index.vectors = np.concatenate([self._index.vectors, *self._extra])
                self._extra = []
            return self._index.vectors

    def snapshot(self) -> tuple[list[str], Vectors] | None:
        """`(ids, matrix)` read under one lock, so rows and ids always line up.

        Appends after the call do not change the returned matrix (they build a new one).
        """
        if not self.load():
            return None
        with self._lock:
            assert self._index is not None  # noqa: S101 - load() returned True
            if self._extra:
                self._index.vectors = np.concatenate([self._index.vectors, *self._extra])
                self._extra = []
            return list(self._index.ids), self._index.vectors

    def __len__(self) -> int:
        self.load()
        return len(self._rows)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple):
            return False
        self.load()
        return key_str(key) in self._rows

    def vector(self, key: TitleKey) -> Vectors | None:
        if not self.load():
            return None
        with self._lock:
            assert self._index is not None  # noqa: S101 - load() returned True
            row = self._rows.get(key_str(key))
            if row is None:
                return None
            base = self._index.vectors.shape[0]
            found: Vectors = self._index.vectors[row] if row < base else self._extra[row - base][0]
            return found

    def append(self, key: TitleKey, vector: Vectors) -> bool:
        """Add one vector (no-op if the key is present). Returns whether it was added."""
        if not self.load():
            raise EmbeddingError("no embedding index loaded")
        with self._lock:
            assert self._index is not None  # noqa: S101 - load() returned True
            name = key_str(key)
            if name in self._rows:
                return False
            dim = self._index.vectors.shape[1]
            if vector.shape != (dim,):
                raise EmbeddingError(f"vector shape {vector.shape} does not match index dim {dim}")
            self._extra.append(vector.astype(np.float32).reshape(1, dim))
            self._index.ids.append(name)
            self._rows[name] = len(self._index.ids) - 1
            return True


class OnDemandEmbedder:
    """Embeds titles missing from an `EmbeddingStore` and appends them (§8.1)."""

    def __init__(
        self,
        store: EmbeddingStore,
        *,
        encoder_factory: EncoderFactory,
        wikimedia_factory: WikimediaFactory | None,
        seed_languages: Sequence[str] = ("en",),
    ) -> None:
        self.store = store
        self._encoder_factory = encoder_factory
        self._wikimedia_factory = wikimedia_factory
        self._encoder: Encoder | None = None
        self._encoder_lock = asyncio.Lock()
        self._wiki_languages = wiki_languages_for(seed_languages)
        self._label_languages = label_languages_for(seed_languages)

    @classmethod
    def from_settings(cls, settings: Settings) -> OnDemandEmbedder:
        def wikimedia() -> WikimediaClient:
            return WikimediaClient.from_settings(settings)

        return cls(
            EmbeddingStore(settings.embeddings_dir),
            encoder_factory=lambda: SentenceTransformerEncoder(settings.embedding_model),
            wikimedia_factory=wikimedia if settings.wikimedia_contact else None,
            seed_languages=settings.seed_languages,
        )

    async def _get_encoder(self) -> Encoder | None:
        """The encoder, created once (model load runs in a worker thread)."""
        async with self._encoder_lock:
            if self._encoder is None:
                try:
                    self._encoder = await asyncio.to_thread(self._encoder_factory)
                except Exception as exc:  # e.g. the `ml` dependency group is not installed
                    log.warning("embed_encoder_unavailable", error=type(exc).__name__)
                    return None
            return self._encoder

    async def _prefetch_open_content(self, session: Session, keys: Sequence[TitleKey]) -> None:
        """Fetch + store open content for keys that were never checked (one batch)."""
        if self._wikimedia_factory is None or not keys:
            return
        checked = {
            (tmdb_id, type_)
            for tmdb_id, type_ in session.execute(
                select(TitleOpenContent.tmdb_id, TitleOpenContent.type).where(
                    tuple_(TitleOpenContent.tmdb_id, TitleOpenContent.type).in_(list(keys))
                )
            )
        }
        unchecked = [k for k in keys if k not in checked]
        if not unchecked:
            return
        try:
            async with self._wikimedia_factory() as wm:
                result = await fetch_open_content(
                    wm,
                    unchecked,
                    wiki_languages=self._wiki_languages,
                    label_languages=self._label_languages,
                )
        except WikimediaError as exc:
            log.warning("embed_open_content_failed", titles=len(unchecked), error=str(exc))
            return
        if result.rows:
            upsert_open_content(session, result.rows)
        if result.failed:
            log.warning("embed_open_content_failed", titles=len(result.failed))

    async def ensure(self, session: Session, keys: Sequence[TitleKey]) -> dict[TitleKey, bool]:
        """Make `keys` (already in `titles`) available in the store where possible.

        Returns, per key, whether a vector is now in the store. Never raises for
        missing index / encoder / Wikimedia problems; those keys map to `False`.
        """
        keys = list(dict.fromkeys(keys))
        if not self.store.load():
            return dict.fromkeys(keys, False)
        missing = [k for k in keys if k not in self.store]
        if missing:
            await self._prefetch_open_content(session, missing)
            meta = self.store.meta
            encoder = await self._get_encoder()
            if encoder is not None and meta is not None:
                for key in missing:
                    await self._embed(session, encoder, meta, key)
        return {k: k in self.store for k in keys}

    async def _embed(
        self, session: Session, encoder: Encoder, meta: dict[str, Any], key: TitleKey
    ) -> None:
        try:
            vector = await embed_one(key, session=session, encoder=encoder, meta=meta)
            if vector is None:
                log.info("embed_on_demand_skipped", tmdb_id=key[0], type=key[1].value)
                return
            self.store.append(key, vector)
            log.info("embed_on_demand", tmdb_id=key[0], type=key[1].value)
        except EmbeddingError as exc:
            log.warning("embed_on_demand_failed", tmdb_id=key[0], type=key[1].value, error=str(exc))
