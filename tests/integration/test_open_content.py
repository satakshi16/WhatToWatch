"""Open content ingest against fake Wikidata + Wikipedia endpoints (P1-04b)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta

import pytest
import respx
from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import WIKIPEDIA_LICENSE, TitleOpenContent, TitleType
from app.ingest import open_content as oc
from app.ingest.open_content import (
    OpenContentOptions,
    coverage_report,
    format_coverage,
    ingest_open_content,
    label_languages_for,
    wiki_languages_for,
)
from app.ingest.wikimedia_client import WikimediaClient
from tests.catalog_factory import SENTINEL, add_catalog, tmdb_title
from tests.wikimedia_fakes import Entity, FakeWikimedia

SEED = ["en", "hi", "ko", "ja", "es", "fr", "ta", "te", "ml"]
WIKIS = wiki_languages_for(SEED)
LABELS = label_languages_for(SEED)
M, T = TitleType.MOVIE, TitleType.TV


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fake() -> Iterator[FakeWikimedia]:
    fake = FakeWikimedia()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


@pytest.fixture
async def wm(fake: FakeWikimedia) -> AsyncIterator[WikimediaClient]:
    async with WikimediaClient(
        contact="ops@whattowatch.invalid", max_attempts=2, sleep=_no_sleep
    ) as c:
        yield c


@pytest.fixture
def catalog(migrated_engine: Engine, fake: FakeWikimedia) -> Engine:
    """Five titles: English movie (enwiki), Korean TV (kowiki only), Tamil movie (Wikidata
    only), Hindi movie (not on Wikidata), and a TV show whose ID collides with a movie ID."""
    with Session(migrated_engine) as session:
        add_catalog(
            session,
            tmdb_title(101, M, "en", popularity=50),
            tmdb_title(202, T, "ko", popularity=40),
            tmdb_title(303, M, "ta", popularity=30),
            tmdb_title(404, M, "hi", popularity=20),
            tmdb_title(101, T, "en", popularity=10),
        )
    fake.add(
        101,
        M,
        Entity(
            qid="Q101",
            label="Open Orchard",
            genres=[("Q1", "drama film")],
            directors=[("Q9", "Open Director")],
            cast=[("Q7", "Actor A", "1"), ("Q8", "Actor B", None)],
            languages=[("Q1860", "en")],
            dates=["2019-01-01T00:00:00Z"],
            sitelinks={"en": "Open Orchard"},
        ),
        summary="Open Orchard is a drama film.",
    )
    fake.add(
        202,
        T,
        Entity(
            qid="Q202",
            label="Open Seoul Nights",
            languages=[("Q9176", "ko")],
            sitelinks={"ko": "서울의 밤"},
        ),
        summary="서울의 밤은 한국 드라마이다.",
        summary_lang="ko",
    )
    fake.add(303, M, Entity(qid="Q303", label="Open Tamil Film", languages=[("Q5885", "ta")]))
    fake.add(
        101,
        T,
        Entity(qid="Q111", label="Open Same-ID Show", sitelinks={"en": "Open Show"}),
        summary="Open Show is a series.",
    )
    return migrated_engine


async def run(wm: WikimediaClient, engine: Engine, **options: object) -> oc.OpenContentStats:
    return await ingest_open_content(
        wm,
        engine,
        OpenContentOptions(**options),  # type: ignore[arg-type]
        wiki_languages=WIKIS,
        label_languages=LABELS,
    )


def rows(engine: Engine) -> dict[tuple[int, TitleType], TitleOpenContent]:
    with Session(engine) as session:
        return {(r.tmdb_id, r.type): r for r in session.scalars(select(TitleOpenContent))}


async def test_full_run_fills_open_content(
    wm: WikimediaClient, catalog: Engine, fake: FakeWikimedia
) -> None:
    stats = await run(wm, catalog)

    assert (stats.selected, stats.checked, stats.failed) == (5, 5, 0)
    assert (stats.with_summary, stats.no_wikipedia, stats.no_wikidata) == (3, 1, 1)
    assert stats.matched == 4
    assert stats.by_language_type["ko/tv"] == {"summary": 1}
    assert stats.by_language_type["hi/movie"] == {"no_wikidata": 1}

    got = rows(catalog)
    assert len(got) == 5
    movie = got[(101, M)]
    assert movie.wikidata_qid == "Q101"
    assert movie.label == "Open Orchard"
    assert movie.genres == ["drama film"]
    assert movie.directors == ["Open Director"]
    assert movie.top_cast == ["Actor A", "Actor B"]
    assert movie.original_language == "en"
    assert movie.wiki_site == "enwiki"
    assert movie.summary == "Open Orchard is a drama film."
    assert movie.summary_license == WIKIPEDIA_LICENSE
    assert movie.wiki_revision_id == 1000

    assert got[(202, T)].wiki_site == "kowiki"
    assert got[(202, T)].summary == "서울의 밤은 한국 드라마이다."
    assert got[(303, M)].wikidata_qid == "Q303"
    assert got[(303, M)].summary is None
    assert got[(404, M)].wikidata_qid is None
    # Same TMDB ID, different type → separate lookups via P4947 vs P4983.
    assert got[(101, T)].label == "Open Same-ID Show"
    assert {t for t, _ in fake.sparql_calls} == {M, T}
    assert all("non-commercial; contact: ops@whattowatch.invalid" in ua for ua in fake.user_agents)


async def test_no_tmdb_content_is_written(wm: WikimediaClient, catalog: Engine) -> None:
    await run(wm, catalog)
    for row in rows(catalog).values():
        for column in TitleOpenContent.__table__.columns:
            assert SENTINEL not in str(getattr(row, column.key)), column.key


async def test_rerun_updates_without_duplicates(
    wm: WikimediaClient, catalog: Engine, fake: FakeWikimedia
) -> None:
    await run(wm, catalog)
    first = {k: r.fetched_at for k, r in rows(catalog).items()}
    fake.summaries[("en", "Open Orchard")]["extract"] = "Updated summary."  # type: ignore[index]

    stats = await run(wm, catalog)

    got = rows(catalog)
    assert stats.checked == 5
    assert len(got) == 5
    assert got[(101, M)].summary == "Updated summary."
    assert all(got[k].fetched_at >= first[k] for k in got)


async def test_resume_skips_fresh_rows(
    wm: WikimediaClient, catalog: Engine, fake: FakeWikimedia
) -> None:
    await run(wm, catalog)
    stale = datetime.now(UTC) - timedelta(days=3)
    with Session(catalog) as session:
        session.execute(
            update(TitleOpenContent).where(TitleOpenContent.tmdb_id == 404).values(fetched_at=stale)
        )
        session.commit()
    fake.sparql_calls.clear()

    stats = await run(wm, catalog, resume=True, fresh_hours=24)

    assert stats.skipped_fresh == 4
    assert stats.checked == 1
    assert fake.sparql_calls == [(M, [404])]


async def test_limit_interleaves_languages(wm: WikimediaClient, catalog: Engine) -> None:
    stats = await run(wm, catalog, limit=2)
    assert stats.checked == 2
    # Most popular title of the two most popular buckets: en/movie and ko/tv.
    assert set(rows(catalog)) == {(101, M), (202, T)}


async def test_batches_respect_batch_size(
    wm: WikimediaClient, catalog: Engine, fake: FakeWikimedia
) -> None:
    await run(wm, catalog, batch_size=2)
    assert all(len(ids) <= 2 for _, ids in fake.sparql_calls)
    assert len(rows(catalog)) == 5


async def test_sparql_failure_writes_nothing_for_batch(
    wm: WikimediaClient, catalog: Engine, fake: FakeWikimedia
) -> None:
    fake.sparql_status = 503
    stats = await run(wm, catalog)
    assert stats.failed == 5
    assert stats.checked == 0
    assert rows(catalog) == {}


async def test_summary_failure_skips_only_that_title(
    wm: WikimediaClient, catalog: Engine, fake: FakeWikimedia
) -> None:
    fake.summaries[("ko", "서울의 밤")] = 500
    stats = await run(wm, catalog)
    assert stats.failed == 1
    assert stats.checked == 4
    assert (202, T) not in rows(catalog)


async def test_coverage_report(wm: WikimediaClient, catalog: Engine) -> None:
    with Session(catalog) as session:
        before = coverage_report(session)
    assert sum(r.unchecked for r in before) == 5

    await run(wm, catalog)
    with Session(catalog) as session:
        report = {(r.language, r.type): r for r in coverage_report(session)}
    assert report[("en", "movie")].summary == 1
    assert report[("ta", "movie")].wikidata == 1
    assert report[("ta", "movie")].summary == 0
    assert report[("hi", "movie")].coverage == 0.0
    assert report[("en", "tv")].coverage == 1.0
    table = format_coverage(list(report.values()))
    assert table.splitlines()[0].split() == [
        "lang",
        "type",
        "titles",
        "wikidata",
        "summary",
        "unchecked",
        "coverage",
    ]
    assert table.splitlines()[-1].split() == ["all", "5", "4", "3", "0", "60%"]


async def test_title_delete_cascades(wm: WikimediaClient, catalog: Engine) -> None:
    await run(wm, catalog)
    from app.db.models import Title

    with Session(catalog) as session:
        session.delete(session.get(Title, (101, M)))
        session.commit()
        assert session.scalar(select(func.count()).select_from(TitleOpenContent)) == 4


def test_cli_refuses_without_contact(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert oc.main([], settings=settings) == 2
    assert "WIKIMEDIA_CONTACT" in capsys.readouterr().err


def test_cli_requires_schema(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    configured = settings.model_copy(update={"wikimedia_contact": "ops@whattowatch.invalid"})
    assert oc.main([], settings=configured) == 2
    assert "make migrate" in capsys.readouterr().err


def test_cli_runs_and_prints_coverage(
    settings: Settings, catalog: Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    configured = settings.model_copy(update={"wikimedia_contact": "ops@whattowatch.invalid"})
    assert oc.main(["--limit", "5"], settings=configured) == 0
    out = capsys.readouterr().out
    assert "Checked 5 titles (3 with a Wikipedia summary" in out
    assert "coverage" in out


@pytest.mark.parametrize(
    "argv", [["--limit", "0"], ["--batch-size", "201"], ["--fresh-hours", "-1"]]
)
def test_cli_validates_args(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        oc.options_from_args(oc.build_parser().parse_args(argv))
