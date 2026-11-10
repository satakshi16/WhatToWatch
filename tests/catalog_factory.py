"""Catalog rows for P1-04b / P1-05 tests. TMDB text columns hold sentinel strings so tests
can prove none of them leak into open content or embedding documents (D-011)."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from app.db.models import Title, TitleOpenContent, TitleType

SENTINEL = "TMDBSENTINEL"


def tmdb_title(
    tmdb_id: int, type_: TitleType, language: str = "en", popularity: float = 1.0
) -> Title:
    tag = f"{SENTINEL}_{type_.value}_{tmdb_id}"
    return Title(
        tmdb_id=tmdb_id,
        type=type_,
        title=f"{tag}_TITLE",
        original_title=f"{tag}_ORIGINAL_TITLE",
        original_language=language,
        overview=f"{tag}_OVERVIEW",
        genres=[f"{tag.lower()}-genre"],
        keywords=[f"{tag}_KEYWORD"],
        directors=[f"{tag}_DIRECTOR"],
        top_cast=[f"{tag}_CAST"],
        release_date=date(2020, 1, 1),
        popularity=popularity,
        localized_titles={"en-US": f"{tag}_LOCALIZED"},
    )


def open_content(
    tmdb_id: int,
    type_: TitleType,
    *,
    summary: str | None = "An open summary.",
    label: str | None = None,
    language: str | None = "en",
    genres: list[str] | None = None,
    qid: str | None = "Q1",
) -> TitleOpenContent:
    return TitleOpenContent(
        tmdb_id=tmdb_id,
        type=type_,
        wikidata_qid=qid,
        label=label or f"Open label {type_.value} {tmdb_id}",
        original_language=language,
        genres=genres if genres is not None else ["drama film"],
        directors=["Open Director"],
        top_cast=["Open Actor"],
        publication_date=date(2015, 6, 1),
        wiki_site="enwiki" if summary else None,
        wiki_title=f"Open label {tmdb_id}" if summary else None,
        wiki_revision_id=1 if summary else None,
        summary=summary,
        summary_license="CC BY-SA 4.0" if summary else None,
    )


def add_catalog(session: Session, *titles: Title) -> None:
    session.add_all(titles)
    session.commit()
