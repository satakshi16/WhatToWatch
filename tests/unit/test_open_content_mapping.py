"""Wikidata/Wikipedia → `title_open_content` mapping (P1-04b). Synthetic data only."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from app.db.models import WIKIPEDIA_LICENSE, TitleType
from app.ingest.open_content import (
    MAX_SPARQL_IDS,
    WikidataRecord,
    choose_article,
    label_languages_for,
    open_content_values,
    parse_bindings,
    row_status,
    sparql_query,
    wiki_languages_for,
)
from app.ingest.wikimedia_client import WikiSummary
from tests.wikimedia_fakes import Entity, entity_rows

SEED = ["en", "hi", "ko", "ja", "es", "fr", "ta", "te", "ml"]
WIKIS = wiki_languages_for(SEED)
NOW = datetime(2026, 11, 3, tzinfo=UTC)


def flatten(rows: list[dict[str, dict[str, str]]]) -> list[dict[str, str]]:
    return [{k: v["value"] for k, v in row.items()} for row in rows]


def parse(tmdb_id: int, *entities: Entity) -> dict[int, WikidataRecord]:
    return parse_bindings(flatten([r for e in entities for r in entity_rows(tmdb_id, e)]))


def test_language_lists() -> None:
    assert WIKIS == ("en", "hi", "ko", "ja", "es", "fr", "ta", "te", "ml")
    assert label_languages_for(["ko", "EN"]) == ("en", "mul", "ko")


@pytest.mark.parametrize(("type_", "prop"), [(TitleType.MOVIE, "P4947"), (TitleType.TV, "P4983")])
def test_sparql_query_uses_tmdb_property_and_values(type_: TitleType, prop: str) -> None:
    query = sparql_query(type_, [27205, 1429], wiki_languages=WIKIS, label_languages=("en", "mul"))
    assert f"?item wdt:{prop} ?tmdb ." in query
    assert 'VALUES ?tmdb { "27205" "1429" }' in query
    assert "<https://en.wikipedia.org/> <https://hi.wikipedia.org/>" in query
    assert "<https://ml.wikipedia.org/>" in query
    assert 'wikibase:language "en,mul"' in query
    for needle in (
        "wdt:P136",
        "wdt:P57",
        "ps:P161",
        "pq:P1545",
        "wdt:P364",
        "wdt:P218",
        "wdt:P577",
    ):
        assert needle in query


def test_sparql_query_batch_limits() -> None:
    with pytest.raises(ValueError, match="TMDB IDs"):
        sparql_query(TitleType.MOVIE, [], wiki_languages=WIKIS, label_languages=("en",))
    ids = list(range(MAX_SPARQL_IDS + 1))
    with pytest.raises(ValueError, match="TMDB IDs"):
        sparql_query(TitleType.MOVIE, ids, wiki_languages=WIKIS, label_languages=("en",))


def test_parse_full_record() -> None:
    entity = Entity(
        qid="Q100",
        label="The Glass Orchard",
        genres=[("Q30", "drama film"), ("Q20", "romance film"), ("Q31", "drama film")],
        directors=[("Q900", "Ana Director")],
        cast=[
            ("Q7", "Seventh Actor", None),
            ("Q500", "Lead Actor", "1"),
            ("Q400", "Second Lead", "2"),
            ("Q3", "Third Actor", None),
        ],
        languages=[("Q1860", "en"), ("Q1568", "hi")],
        dates=["2019-05-01T00:00:00Z", "2018-09-10T00:00:00Z"],
        sitelinks={"en": "The Glass Orchard (film)", "hi": "ग्लास ऑर्चर्ड"},
    )
    record = parse(900001, entity)[900001]
    assert record.qid == "Q100"
    assert record.label == "The Glass Orchard"
    assert record.genre_labels == ["romance film", "drama film"]  # QID order, deduped labels
    assert record.director_labels == ["Ana Director"]
    # Ordinals first (ascending), then QID order; at most three.
    assert record.top_cast == ["Lead Actor", "Second Lead", "Third Actor"]
    assert record.original_language == "hi"  # lowest language QID (Q1568 < Q1860)
    assert record.publication_date == date(2018, 9, 10)
    assert record.sitelinks == {"enwiki": "The Glass Orchard (film)", "hiwiki": "ग्लास ऑर्चर्ड"}


def test_parse_drops_qid_only_labels_and_bad_values() -> None:
    entity = Entity(
        qid="Q100",
        label=None,
        genres=[("Q30", "Q30")],
        cast=[("Q8", "Named", "x")],
        dates=["not-a-date"],
    )
    record = parse(5, entity)[5]
    assert record.label is None
    assert record.genre_labels == []
    assert record.top_cast == ["Named"]
    assert record.publication_date is None
    assert record.original_language is None


def test_parse_prefers_lowest_qid_when_several_items_match() -> None:
    records = parse(5, Entity(qid="Q900", label="Later"), Entity(qid="Q12", label="Earlier"))
    assert records[5].qid == "Q12"
    assert records[5].label == "Earlier"


def test_parse_ignores_rows_without_item_or_tmdb() -> None:
    assert parse_bindings([{"tmdb": "x", "item": "http://www.wikidata.org/entity/Q1"}]) == {}
    assert parse_bindings([{"tmdb": "1", "item": "https://example.org/Q1"}]) == {}


def test_cast_keeps_lowest_ordinal_for_repeated_actor() -> None:
    entity = Entity(
        qid="Q1", cast=[("Q5", "Actor", "3"), ("Q5", "Actor", "1"), ("Q6", "Other", "2")]
    )
    assert parse(1, entity)[1].top_cast == ["Actor", "Other"]


def test_choose_article_prefers_enwiki() -> None:
    record = WikidataRecord(
        1, "Q1", sitelinks={"enwiki": "E", "kowiki": "K"}, languages={"Q9176": "ko"}
    )
    assert choose_article(record, WIKIS) == ("enwiki", "E")


def test_choose_article_falls_back_to_original_language_wiki() -> None:
    record = WikidataRecord(
        1, "Q1", sitelinks={"kowiki": "K", "jawiki": "J"}, languages={"Q9176": "ko"}
    )
    assert choose_article(record, WIKIS) == ("kowiki", "K")


def test_choose_article_none_without_matching_wiki() -> None:
    # Japanese article exists but the film's language is Korean → not used.
    record = WikidataRecord(1, "Q1", sitelinks={"jawiki": "J"}, languages={"Q9176": "ko"})
    assert choose_article(record, WIKIS) is None
    # Language wiki outside the configured list → not used.
    record = WikidataRecord(1, "Q1", sitelinks={"dewiki": "D"}, languages={"Q188": "de"})
    assert choose_article(record, WIKIS) is None


def test_values_without_wikidata_match() -> None:
    values = open_content_values((7, TitleType.TV), None, None, NOW)
    assert values["wikidata_qid"] is None
    assert values["summary"] is None
    assert values["genres"] == []
    assert row_status(values) == "no_wikidata"


def test_values_with_usable_summary() -> None:
    record = parse(1, Entity(qid="Q1", label="L", genres=[("Q2", "g")]))[1]
    summary = WikiSummary("enwiki", "L (film)", 42, "Summary text.", "standard")
    values = open_content_values((1, TitleType.MOVIE), record, summary, NOW)
    assert values["wikidata_qid"] == "Q1"
    assert values["wiki_site"] == "enwiki"
    assert values["wiki_title"] == "L (film)"
    assert values["wiki_revision_id"] == 42
    assert values["summary"] == "Summary text."
    assert values["summary_license"] == WIKIPEDIA_LICENSE
    assert values["fetched_at"] == NOW
    assert row_status(values) == "summary"


def test_values_with_disambiguation_page_have_no_summary() -> None:
    record = parse(1, Entity(qid="Q1", label="L"))[1]
    summary = WikiSummary("enwiki", "L", 42, "L may refer to:", "disambiguation")
    values = open_content_values((1, TitleType.MOVIE), record, summary, NOW)
    assert values["summary"] is None
    assert values["summary_license"] is None
    assert row_status(values) == "no_wikipedia"
    assert row_status(None) == "unchecked"
