from __future__ import annotations

from app.ingest.genres import CANONICAL_GENRES, TMDB_GENRE_MAP, canonical_genres


def test_movie_genres_keep_tmdb_order() -> None:
    assert canonical_genres([28, 878, 12], "en") == ["action", "science-fiction", "adventure"]


def test_compound_tv_genres_split() -> None:
    assert canonical_genres([10765]) == ["science-fiction", "fantasy"]
    assert canonical_genres([10759]) == ["action", "adventure"]
    assert canonical_genres([10768]) == ["war", "politics"]


def test_movie_and_tv_ids_merge_to_same_slugs() -> None:
    movie = canonical_genres([878, 14])  # Science Fiction, Fantasy (movie list)
    tv = canonical_genres([10765])  # Sci-Fi & Fantasy (TV list)
    assert movie == tv == ["science-fiction", "fantasy"]


def test_duplicates_removed_first_wins() -> None:
    assert canonical_genres([10759, 28, 12]) == ["action", "adventure"]


def test_japanese_animation_is_also_anime() -> None:
    assert canonical_genres([16, 10765], "ja") == [
        "animation",
        "anime",
        "science-fiction",
        "fantasy",
    ]
    assert canonical_genres([16], "JA") == ["animation", "anime"]


def test_non_japanese_animation_is_not_anime() -> None:
    assert canonical_genres([16], "en") == ["animation"]
    assert canonical_genres([16]) == ["animation"]
    assert canonical_genres([18], "ja") == ["drama"]


def test_unknown_ids_skipped() -> None:
    assert canonical_genres([999999, 18]) == ["drama"]
    assert canonical_genres([]) == []


def test_canonical_set_covers_map_and_slugs_are_kebab_case() -> None:
    mapped = {slug for slugs in TMDB_GENRE_MAP.values() for slug in slugs}
    assert mapped | {"anime"} == CANONICAL_GENRES
    assert all(slug == slug.lower() and " " not in slug for slug in CANONICAL_GENRES)
