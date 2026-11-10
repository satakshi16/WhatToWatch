"""Embedding build CLI: `title_open_content` → `data/embeddings.npy` (P1-05, §8.1, D-011).

Usage:
    python -m app.ingest.build_embeddings [--incremental] [--batch-size N] [--out-dir DIR]

Writes three files to `EMBEDDINGS_DIR`:
- `embeddings.npy`: float32 matrix (N x dim), rows L2-normalized;
- `id_map.json`: row i → `"<type>:<tmdb_id>"` (e.g. `"movie:27205"`);
- `embeddings_meta.json`: model + version, dim, timestamps, counts, coverage, licences,
  config hash and the one-hot spec, so incremental runs and `embed_one` stay consistent.

Provenance (D-011): documents are built **only** from `title_open_content` (Wikidata +
Wikipedia). `OpenDocument` is the builder's sole input and has no TMDB fields; TMDB IDs
appear only as join keys in `id_map.json`. Titles without a Wikipedia summary are not
embedded.

Incremental mode embeds only eligible titles missing from the id map, keeps existing
vectors, and drops titles that are no longer eligible. It refuses to run if the model or
embedding config changed since the last full build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sqlalchemy import Engine, func, inspect, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.base import utcnow
from app.db.models import WIKIPEDIA_LICENSE, Title, TitleOpenContent, TitleType
from app.db.session import create_db_engine
from app.ingest.open_content import fetch_open_content, upsert_open_content
from app.ingest.wikimedia_client import WikimediaClient

log = get_logger(__name__)

Vectors = npt.NDArray[np.float32]
TitleKey = tuple[int, TitleType]

EMBEDDINGS_FILE = "embeddings.npy"
ID_MAP_FILE = "id_map.json"
META_FILE = "embeddings_meta.json"
META_FORMAT_VERSION = 1

CONTENT_SOURCE = "wikidata+wikipedia"
LICENSES = {"wikidata": "CC0 1.0", "wikipedia": WIKIPEDIA_LICENSE}
PASSAGE_PREFIX = "passage: "  # e5 models expect this prefix on documents
DOCUMENT_TEMPLATE = (
    "passage: {label}. {genres}. Directed by {directors}. Starring {top_cast}. {summary}"
)
# Bump when `build_document` changes output, so incremental runs refuse stale vectors.
DOCUMENT_TEMPLATE_VERSION = 1
FIRST_DECADE, LAST_DECADE = 1900, 2020
ENCODE_CHUNK = 512  # texts per encoder call (progress logging granularity)


class EmbeddingError(Exception):
    """Embedding build cannot proceed (e.g. incompatible index for incremental mode)."""


# --- Keys ----------------------------------------------------------------------


def key_str(key: TitleKey) -> str:
    return f"{key[1].value}:{key[0]}"


def parse_key(value: str) -> TitleKey:
    type_, _, tmdb_id = value.partition(":")
    return int(tmdb_id), TitleType(type_)


# --- Documents (open content only) ---------------------------------------------


@dataclass(frozen=True)
class OpenDocument:
    """Everything the document builder may see. Built only from `title_open_content`."""

    key: TitleKey
    label: str | None
    genres: tuple[str, ...]
    directors: tuple[str, ...]
    top_cast: tuple[str, ...]
    summary: str | None
    original_language: str | None
    publication_date: date | None

    @classmethod
    def from_row(cls, row: TitleOpenContent) -> OpenDocument:
        return cls(
            key=(row.tmdb_id, row.type),
            label=row.label,
            genres=tuple(row.genres or ()),
            directors=tuple(row.directors or ()),
            top_cast=tuple(row.top_cast or ()),
            summary=row.summary,
            original_language=row.original_language,
            publication_date=row.publication_date,
        )

    @property
    def embeddable(self) -> bool:
        return bool(self.summary and self.summary.strip())


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _names(values: Iterable[str]) -> str:
    return ", ".join(v for v in (_clean(x) for x in values) if v)


def build_document(doc: OpenDocument) -> str | None:
    """§8.1 passage text, empty parts left out; `None` if the title has no open content."""
    if not doc.embeddable:
        return None
    genres, directors, cast = (_names(v) for v in (doc.genres, doc.directors, doc.top_cast))
    parts = [
        _clean(doc.label),
        genres,
        f"Directed by {directors}" if directors else "",
        f"Starring {cast}" if cast else "",
    ]
    head = ". ".join(p.rstrip(".") for p in parts if p)
    summary = _clean(doc.summary)
    return f"{PASSAGE_PREFIX}{head}. {summary}" if head else f"{PASSAGE_PREFIX}{summary}"


def load_documents(session: Session) -> list[OpenDocument]:
    """All open-content rows as documents, sorted by key (deterministic order)."""
    rows = session.scalars(select(TitleOpenContent)).all()
    docs = [OpenDocument.from_row(r) for r in rows]
    return sorted(docs, key=lambda d: (d.key[1].value, d.key[0]))


# --- Encoder -------------------------------------------------------------------


class Encoder(Protocol):
    """Text → raw vectors (normalization happens in `finalize`)."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str], batch_size: int) -> Vectors: ...


class SentenceTransformerEncoder:
    """sentence-transformers model, loaded lazily on CPU (needs the `ml` dependency group)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EmbeddingError(
                "sentence-transformers is not installed: use `make embed` "
                "(runs with the `ml` dependency group)"
            ) from exc
        self._name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        if not dim:  # pragma: no cover - all supported models report it
            raise EmbeddingError(f"{model_name} does not report an embedding dimension")
        self._dim = int(dim)
        self._version = _model_revision(self._model)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str], batch_size: int) -> Vectors:
        out = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)


def _model_revision(model: Any) -> str:  # pragma: no cover - needs a real model
    """Hugging Face commit hash of the loaded model if known, plus the library version."""
    import sentence_transformers

    commit = None
    try:
        commit = getattr(model[0].auto_model.config, "_commit_hash", None)
    except (AttributeError, IndexError, TypeError):
        commit = None
    return f"{commit or 'unknown'} (sentence-transformers {sentence_transformers.__version__})"


# --- One-hot block (§8.1, optional) --------------------------------------------


@dataclass(frozen=True)
class OneHotSpec:
    """Language / genre / decade blocks, each scaled to norm `weight`. Off by default."""

    enabled: bool = False
    weight: float = 0.15
    languages: tuple[str, ...] = ()
    genres: tuple[str, ...] = ()
    decades: tuple[int, ...] = tuple(range(FIRST_DECADE, LAST_DECADE + 1, 10))

    @property
    def dim(self) -> int:
        if not self.enabled:
            return 0
        # +1 slots: "other / unknown" language and decade.
        return len(self.languages) + 1 + len(self.genres) + len(self.decades) + 1

    def to_meta(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> OneHotSpec:
        return cls(
            enabled=bool(meta["enabled"]),
            weight=float(meta["weight"]),
            languages=tuple(meta["languages"]),
            genres=tuple(meta["genres"]),
            decades=tuple(int(d) for d in meta["decades"]),
        )


def genre_vocabulary(docs: Iterable[OpenDocument], size: int) -> tuple[str, ...]:
    """The `size` most frequent Wikidata genre labels (ties alphabetical)."""
    counts = Counter(g for d in docs for g in dict.fromkeys(d.genres))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(g for g, _ in ranked[:size])


def onehot_block(doc: OpenDocument, spec: OneHotSpec) -> Vectors:
    if not spec.enabled:
        return np.zeros(0, dtype=np.float32)

    def scaled(block: Vectors) -> Vectors:
        norm = float(np.linalg.norm(block))
        return block * (spec.weight / norm) if norm > 0 else block

    lang = np.zeros(len(spec.languages) + 1, dtype=np.float32)
    code = (doc.original_language or "").lower()
    lang[spec.languages.index(code) if code in spec.languages else len(spec.languages)] = 1.0

    genres = np.zeros(len(spec.genres), dtype=np.float32)
    for g in doc.genres:
        if g in spec.genres:
            genres[spec.genres.index(g)] = 1.0

    decades = np.zeros(len(spec.decades) + 1, dtype=np.float32)
    year = doc.publication_date.year if doc.publication_date else None
    decade = (year // 10) * 10 if year is not None else None
    decades[spec.decades.index(decade) if decade in spec.decades else len(spec.decades)] = 1.0

    return np.concatenate([scaled(lang), scaled(genres), scaled(decades)])


def l2_normalize(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    out: Vectors = (vectors / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    return out


def finalize(raw: Vectors, docs: Sequence[OpenDocument], spec: OneHotSpec) -> Vectors:
    """L2-normalize text vectors, append the optional one-hot block, re-normalize."""
    text = l2_normalize(np.asarray(raw, dtype=np.float32))
    if not spec.enabled:
        return text
    blocks = np.stack([onehot_block(d, spec) for d in docs]) if docs else np.zeros((0, spec.dim))
    return l2_normalize(np.concatenate([text, blocks.astype(np.float32)], axis=1))


# --- Index files ---------------------------------------------------------------


@dataclass
class EmbeddingIndex:
    ids: list[str]
    vectors: Vectors
    meta: dict[str, Any]

    def row_of(self, key: TitleKey) -> int | None:
        try:
            return self.ids.index(key_str(key))
        except ValueError:
            return None


def load_index(directory: Path) -> EmbeddingIndex | None:
    paths = [directory / f for f in (EMBEDDINGS_FILE, ID_MAP_FILE, META_FILE)]
    if not all(p.exists() for p in paths):
        return None
    vectors = np.load(paths[0], allow_pickle=False).astype(np.float32)
    ids = json.loads(paths[1].read_text(encoding="utf-8"))
    meta = json.loads(paths[2].read_text(encoding="utf-8"))
    if len(ids) != vectors.shape[0]:
        raise EmbeddingError(f"{ID_MAP_FILE} has {len(ids)} ids but {vectors.shape[0]} vectors")
    return EmbeddingIndex(ids=list(ids), vectors=vectors, meta=meta)


def _atomic_write(path: Path, write: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as fh:
        write(fh)
    os.replace(tmp, path)


def save_index(directory: Path, index: EmbeddingIndex) -> None:
    """Write vectors and id map first, meta last (meta marks a complete index)."""
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / EMBEDDINGS_FILE, lambda fh: np.save(fh, index.vectors))
    _atomic_write(
        directory / ID_MAP_FILE,
        lambda fh: fh.write(json.dumps(index.ids).encode("utf-8")),
    )
    _atomic_write(
        directory / META_FILE,
        lambda fh: fh.write(
            json.dumps(index.meta, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ),
    )


# --- Build ---------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedOptions:
    batch_size: int = 32
    incremental: bool = False
    onehot_enabled: bool = False
    onehot_weight: float = 0.15
    onehot_genre_vocab: int = 40
    languages: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> EmbedOptions:
        values: dict[str, Any] = {
            "batch_size": settings.embed_batch_size,
            "onehot_enabled": settings.onehot_enabled,
            "onehot_weight": settings.onehot_weight,
            "onehot_genre_vocab": settings.onehot_genre_vocab,
            "languages": tuple(lang.lower() for lang in settings.seed_languages),
        }
        values.update(overrides)
        return cls(**values)


def config_hash(model: str, options: EmbedOptions) -> str:
    """Hash of everything that changes vector values (not the data-derived genre vocab)."""
    payload = {
        "model": model,
        "document_template": DOCUMENT_TEMPLATE,
        "document_template_version": DOCUMENT_TEMPLATE_VERSION,
        "content_source": CONTENT_SOURCE,
        "onehot_enabled": options.onehot_enabled,
        "onehot_weight": options.onehot_weight if options.onehot_enabled else None,
        "onehot_genre_vocab": options.onehot_genre_vocab if options.onehot_enabled else None,
        "onehot_languages": list(options.languages) if options.onehot_enabled else None,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class EmbedStats:
    mode: str = "full"
    catalog: int = 0
    open_content_rows: int = 0
    eligible: int = 0
    embedded: int = 0
    reused: int = 0
    removed: int = 0
    total: int = 0
    skipped: int = 0
    dim: int = 0
    duration_s: float = 0.0
    coverage: dict[str, int] = field(default_factory=dict)


def _coverage(docs: Iterable[OpenDocument]) -> dict[str, int]:
    """Embedded titles by Wikidata original language x type (no TMDB fields)."""
    counts = Counter(f"{d.original_language or 'und'}/{d.key[1].value}" for d in docs)
    return dict(sorted(counts.items()))


def _encode(encoder: Encoder, texts: Sequence[str], batch_size: int) -> Vectors:
    chunks: list[Vectors] = []
    for start in range(0, len(texts), ENCODE_CHUNK):
        chunk = texts[start : start + ENCODE_CHUNK]
        chunks.append(np.asarray(encoder.encode(chunk, batch_size), dtype=np.float32))
        log.info("embed_progress", done=start + len(chunk), total=len(texts))
    if not chunks:
        return np.zeros((0, encoder.dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def build_embeddings(
    engine: Engine,
    encoder: Encoder,
    options: EmbedOptions,
    out_dir: Path,
    *,
    now: datetime | None = None,
) -> EmbedStats:
    """Full or incremental build. Writes the three index files to `out_dir`."""
    started = time.monotonic()
    now = now or utcnow()
    stats = EmbedStats(mode="incremental" if options.incremental else "full")
    with Session(engine) as session:
        stats.catalog = session.scalar(select(func.count()).select_from(Title)) or 0
        docs = load_documents(session)
    stats.open_content_rows = len(docs)
    eligible = [d for d in docs if d.embeddable]
    stats.eligible = len(eligible)
    hash_ = config_hash(encoder.name, options)

    previous = load_index(out_dir) if options.incremental else None
    if previous is not None:
        _check_compatible(previous.meta, encoder, hash_)
        spec = OneHotSpec.from_meta(previous.meta["onehot"])
        created_at = str(previous.meta["created_at"])
    else:
        if options.incremental:
            log.info("embed_incremental_no_index", out_dir=str(out_dir))
            stats.mode = "full"
        spec = OneHotSpec(
            enabled=options.onehot_enabled,
            weight=options.onehot_weight,
            languages=options.languages,
            genres=genre_vocabulary(eligible, options.onehot_genre_vocab),
        )
        created_at = now.isoformat()

    eligible_ids = {key_str(d.key) for d in eligible}
    kept_rows: list[int] = []
    kept_ids: list[str] = []
    if previous is not None:
        for row, id_ in enumerate(previous.ids):
            if id_ in eligible_ids:
                kept_rows.append(row)
                kept_ids.append(id_)
        stats.removed = len(previous.ids) - len(kept_ids)
    known = set(kept_ids)
    new_docs = [d for d in eligible if key_str(d.key) not in known]

    texts = [build_document(d) or "" for d in new_docs]  # all eligible, so never empty
    new_vectors = finalize(_encode(encoder, texts, options.batch_size), new_docs, spec)
    dim = encoder.dim + spec.dim
    if previous is not None:
        kept = previous.vectors[kept_rows].astype(np.float32)
        vectors = np.concatenate([kept.reshape(-1, dim), new_vectors.reshape(-1, dim)], axis=0)
    else:
        vectors = new_vectors.reshape(-1, dim)
    ids = kept_ids + [key_str(d.key) for d in new_docs]

    stats.embedded = len(new_docs)
    stats.reused = len(kept_ids)
    stats.total = len(ids)
    stats.skipped = stats.catalog - stats.total
    stats.dim = dim
    stats.coverage = _coverage(eligible)

    meta = {
        "format_version": META_FORMAT_VERSION,
        "model": encoder.name,
        "model_version": encoder.version,
        "dim": dim,
        "text_dim": encoder.dim,
        "normalized": True,
        "created_at": created_at,
        "updated_at": now.isoformat(),
        "catalog_count": stats.catalog,
        "count": stats.total,
        "skipped": stats.skipped,
        "skipped_breakdown": {
            "no_open_content_row": stats.catalog - stats.open_content_rows,
            "no_wikipedia_summary": stats.open_content_rows - stats.eligible,
        },
        "content_source": CONTENT_SOURCE,
        "licenses": LICENSES,
        "coverage": stats.coverage,
        "config_hash": hash_,
        "document_template": DOCUMENT_TEMPLATE,
        "document_template_version": DOCUMENT_TEMPLATE_VERSION,
        "passage_prefix": PASSAGE_PREFIX,
        "onehot": spec.to_meta(),
        "last_run": {"mode": stats.mode, "embedded": stats.embedded, "removed": stats.removed},
    }
    save_index(out_dir, EmbeddingIndex(ids=ids, vectors=vectors, meta=meta))
    stats.duration_s = round(time.monotonic() - started, 2)
    log.info("embed_done", **asdict(stats))
    return stats


def _check_compatible(meta: dict[str, Any], encoder: Encoder, hash_: str) -> None:
    problems = []
    if meta.get("format_version") != META_FORMAT_VERSION:
        problems.append("index format version")
    if meta.get("model") != encoder.name:
        problems.append(f"model ({meta.get('model')} → {encoder.name})")
    if meta.get("model_version") != encoder.version:
        problems.append("model version")
    if meta.get("config_hash") != hash_:
        problems.append("embedding config")
    if problems:
        raise EmbeddingError(
            "Existing index is incompatible (" + ", ".join(problems) + "): run a full build "
            "(`make embed` without --incremental)"
        )


# --- On-demand embedding -------------------------------------------------------


async def embed_one(
    key: TitleKey,
    *,
    session: Session,
    encoder: Encoder,
    meta: dict[str, Any],
    wikimedia: WikimediaClient | None = None,
    wiki_languages: Sequence[str] = ("en",),
    label_languages: Sequence[str] = ("en", "mul"),
) -> Vectors | None:
    """Vector for one title, consistent with the index described by `meta`.

    The title must already be in `titles` (§8.1: new titles are inserted from TMDB
    first). If it has no open content yet and `wikimedia` is given, its open content is
    fetched and stored. Returns `None` if the title has no open content: the caller
    then treats it as not embeddable. The vector is not appended to any index here.
    """
    if meta.get("model") != encoder.name:
        raise EmbeddingError(f"Encoder {encoder.name} does not match index {meta.get('model')}")
    row = session.get(TitleOpenContent, key)
    if row is None and wikimedia is not None:
        result = await fetch_open_content(
            wikimedia, [key], wiki_languages=wiki_languages, label_languages=label_languages
        )
        if result.rows:
            upsert_open_content(session, result.rows)
            row = session.get(TitleOpenContent, key)
    if row is None:
        return None
    doc = OpenDocument.from_row(row)
    text = build_document(doc)
    if text is None:
        return None
    spec = OneHotSpec.from_meta(meta["onehot"])
    vector: Vectors = finalize(encoder.encode([text], 1), [doc], spec)[0]
    return vector


# --- CLI -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest.build_embeddings",
        description="Embed catalog titles from open content (Wikidata + Wikipedia).",
    )
    parser.add_argument(
        "--incremental", action="store_true", help="embed only titles missing from the id map"
    )
    parser.add_argument("--batch-size", type=int, help="encoder batch size (EMBED_BATCH_SIZE)")
    parser.add_argument("--out-dir", type=Path, help="output directory (EMBEDDINGS_DIR)")
    return parser


def main(
    argv: Sequence[str] | None = None,
    settings: Settings | None = None,
    encoder: Encoder | None = None,
) -> int:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    overrides: dict[str, Any] = {"incremental": args.incremental}
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    options = EmbedOptions.from_settings(settings, **overrides)
    out_dir: Path = args.out_dir or settings.embeddings_dir

    engine = create_db_engine(settings.database_url)
    try:
        if not inspect(engine).has_table(TitleOpenContent.__tablename__):
            print("Database schema missing: run `make migrate` first.", file=sys.stderr)
            return 2
        try:
            encoder = encoder or SentenceTransformerEncoder(settings.embedding_model)
            stats = build_embeddings(engine, encoder, options, out_dir)
        except EmbeddingError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        engine.dispose()

    print(
        f"{stats.mode.capitalize()} build: {stats.total} vectors (dim {stats.dim}) in "
        f"{out_dir}; {stats.embedded} embedded this run, {stats.reused} reused, "
        f"{stats.removed} removed; {stats.skipped} of {stats.catalog} catalog titles "
        f"skipped (no open content) in {stats.duration_s}s."
    )
    print("Embedded titles by Wikidata language/type: " + json.dumps(stats.coverage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
