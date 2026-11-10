"""P1-08 helpers: TTL cache, query normalization, language choice, result merging."""

from __future__ import annotations

import pytest

from app.api.titles import merge_results, normalize_query, search_languages
from app.core.cache import TTLCache
from app.ingest.tmdb_models import MediaListItem


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cache_hit_then_expiry() -> None:
    clock = Clock()
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=10, max_entries=5, clock=clock)
    cache.set("a", 1)
    clock.now = 9.9
    assert cache.get("a") == 1
    clock.now = 10.0
    assert cache.get("a") is None
    assert len(cache) == 0


def test_cache_evicts_oldest_when_full() -> None:
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=10, max_entries=2, clock=Clock())
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("a", 3)  # re-set moves "a" to newest
    cache.set("c", 4)
    assert cache.get("b") is None
    assert (cache.get("a"), cache.get("c")) == (3, 4)


def test_cache_zero_ttl_stores_nothing() -> None:
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=0, max_entries=2)
    cache.set("a", 1)
    assert cache.get("a") is None


def test_cache_clear_and_bad_size() -> None:
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=5, max_entries=2)
    cache.set("a", 1)
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(ValueError, match="max_entries"):
        TTLCache(ttl_seconds=5, max_entries=0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Squid", "squid"), ("  squid \t game ", "squid game"), ("STRASSE", "strasse")],
)
def test_normalize_query(raw: str, expected: str) -> None:
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        (["en"], ["en-US"]),
        (["en", "ko"], ["en-US"]),
        (["ko", "en"], ["ko", "en-US"]),
        (["hi"], ["hi", "en-US"]),
        ([], ["en-US"]),
    ],
)
def test_search_languages(languages: list[str], expected: list[str]) -> None:
    assert search_languages(languages) == expected


def _item(
    type_: str, id_: int, title: str, original: str | None = None, language: str = "en"
) -> MediaListItem:
    key, orig_key = ("title", "original_title") if type_ == "movie" else ("name", "original_name")
    return MediaListItem.model_validate(
        {
            "id": id_,
            "media_type": type_,
            key: title,
            orig_key: original or f"{title} (original)",
            "original_language": language,
        }
    )


def test_merge_interleaves_dedupes_and_prefers_first_list() -> None:
    ko = [_item("tv", 1, "KO one"), _item("movie", 2, "KO two")]
    en = [_item("tv", 3, "EN three"), _item("tv", 1, "EN one"), _item("movie", 4, "EN four")]
    merged = merge_results([("ko", ko), ("en-US", en)])
    assert [(i.media_type, i.id, i.display_title) for i in merged] == [
        ("tv", 1, "KO one"),
        ("tv", 3, "EN three"),
        ("movie", 2, "KO two"),
        ("movie", 4, "EN four"),
    ]


def test_merge_untranslated_title_falls_back_to_english() -> None:
    # No Hindi title on TMDB: the Hindi search returns the original Korean title.
    hi = [_item("movie", 1, "기생충", original="기생충", language="ko")]
    en = [_item("movie", 1, "Parasite", original="기생충", language="ko")]
    assert merge_results([("hi", hi), ("en-US", en)])[0].display_title == "Parasite"


def test_merge_keeps_original_title_in_the_users_language() -> None:
    ko = [_item("tv", 1, "오징어 게임", original="오징어 게임", language="ko")]
    en = [_item("tv", 1, "Squid Game", original="오징어 게임", language="ko")]
    assert merge_results([("ko", ko), ("en-US", en)])[0].display_title == "오징어 게임"


def test_merge_uses_first_record_when_none_is_localized() -> None:
    hi = [_item("movie", 1, "Amélie", original="Amélie", language="fr")]
    ja = [_item("movie", 1, "Amélie", original="Amélie", language="fr")]
    assert merge_results([("hi", hi), ("ja", ja)])[0] is hi[0]


def test_merge_keeps_same_id_different_types_apart() -> None:
    merged = merge_results([("hi", [_item("movie", 1, "M")]), ("en-US", [_item("tv", 1, "T")])])
    assert [(i.media_type, i.id) for i in merged] == [("movie", 1), ("tv", 1)]


def test_merge_single_and_empty() -> None:
    one = [_item("movie", 1, "M")]
    assert merge_results([("en-US", one)]) == one
    assert merge_results([]) == []
    assert merge_results([("ko", []), ("en-US", one)]) == one
