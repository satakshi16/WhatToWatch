"""Sanity check for `make embed`: print nearest neighbours for a few known titles (P1-05).

Usage:
    uv run python -m scripts.embedding_neighbours [--title "Inception" ...] [--key movie:27205]
        [-k 5]

Titles are looked up by their Wikidata label in `title_open_content` (case-insensitive,
exact match first, then prefix). With no arguments, uses Inception, Attack on Titan and
Parasite. Cosine similarity = dot product, because index rows are L2-normalized.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Title, TitleOpenContent
from app.db.session import create_db_engine
from app.ingest.build_embeddings import EmbeddingIndex, TitleKey, key_str, load_index, parse_key

DEFAULT_TITLES = ("Inception", "Attack on Titan", "Parasite")


def find_key(session: Session, index: EmbeddingIndex, label: str) -> TitleKey | None:
    """Embedded title whose Wikidata label matches `label` (exact, then prefix)."""
    embedded = set(index.ids)
    for pattern in (label.lower(), f"{label.lower()}%"):
        stmt = (
            select(TitleOpenContent.tmdb_id, TitleOpenContent.type)
            .join(
                Title,
                (Title.tmdb_id == TitleOpenContent.tmdb_id) & (Title.type == TitleOpenContent.type),
            )
            .where(func.lower(TitleOpenContent.label).like(pattern))
            .order_by(Title.popularity.desc())
        )
        for tmdb_id, type_ in session.execute(stmt):
            if key_str((tmdb_id, type_)) in embedded:
                return tmdb_id, type_
    return None


def describe(session: Session, key: TitleKey) -> str:
    row = session.get(TitleOpenContent, key)
    label = row.label if row and row.label else "?"
    year = row.publication_date.year if row and row.publication_date else "?"
    lang = row.original_language if row and row.original_language else "?"
    return f"{label} ({year}, {lang}, {key_str(key)})"


def neighbours(index: EmbeddingIndex, key: TitleKey, k: int) -> list[tuple[str, float]]:
    row = index.row_of(key)
    if row is None:
        return []
    scores = index.vectors @ index.vectors[row]
    order = np.argsort(-scores)
    return [(index.ids[i], float(scores[i])) for i in order if i != row][:k]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.embedding_neighbours")
    parser.add_argument("--title", action="append", help="Wikidata label to look up")
    parser.add_argument("--key", action="append", help="title key, e.g. movie:27205")
    parser.add_argument("-k", type=int, default=5, help="neighbours per title")
    args = parser.parse_args(argv)

    settings = get_settings()
    index = load_index(settings.embeddings_dir)
    if index is None:
        print(f"No index in {settings.embeddings_dir}: run `make embed` first.", file=sys.stderr)
        return 2
    print(
        f"Index: {len(index.ids)} vectors, dim {index.vectors.shape[1]}, "
        f"model {index.meta.get('model')}, source {index.meta.get('content_source')}"
    )
    norms = np.linalg.norm(index.vectors, axis=1)
    print(f"Row norms: min {norms.min():.4f}, max {norms.max():.4f}\n")

    engine = create_db_engine(settings.database_url)
    try:
        with Session(engine) as session:
            keys: list[TitleKey] = [parse_key(k) for k in args.key or []]
            for label in args.title or ([] if args.key else DEFAULT_TITLES):
                found = find_key(session, index, label)
                if found is None:
                    print(f"'{label}': not in the index (not in the catalog or no open content)\n")
                else:
                    keys.append(found)
            for key in keys:
                print(describe(session, key))
                results = neighbours(index, key, args.k)
                if not results:
                    print("  not in the index\n")
                    continue
                for id_, score in results:
                    print(f"  {score:.3f}  {describe(session, parse_key(id_))}")
                print()
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
