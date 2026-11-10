"""Fake Wikidata SPARQL + Wikipedia REST endpoints for P1-04b / P1-05 tests (respx).

Entities are synthetic. The fake parses the TMDB IDs and property out of the SPARQL
`VALUES` clause and answers with bindings shaped like the real endpoint's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote

import httpx
import respx

from app.db.models import TitleType

ENTITY = "http://www.wikidata.org/entity/"
SPARQL_URL = "https://query.wikidata.org/sparql"
SUMMARY_RE = r"https://(?P<lang>[a-z]+)\.wikipedia\.org/api/rest_v1/page/summary/(?P<title>.+)"
PROPERTY_TYPE = {"P4947": TitleType.MOVIE, "P4983": TitleType.TV}


@dataclass
class Entity:
    qid: str
    label: str | None = None
    genres: list[tuple[str, str]] = field(default_factory=list)
    directors: list[tuple[str, str]] = field(default_factory=list)
    cast: list[tuple[str, str, str | None]] = field(default_factory=list)  # qid, label, ordinal
    languages: list[tuple[str, str]] = field(default_factory=list)  # qid, ISO 639-1
    dates: list[str] = field(default_factory=list)
    sitelinks: dict[str, str] = field(default_factory=dict)  # "en" → article title


def _cell(value: str, kind: str = "literal") -> dict[str, str]:
    return {"type": kind, "value": value}


def entity_rows(tmdb_id: int, e: Entity) -> list[dict[str, Any]]:
    base = {
        "tmdb": _cell(str(tmdb_id)),
        "item": _cell(ENTITY + e.qid, "uri"),
        # The label service falls back to the QID when there is no label.
        "itemLabel": _cell(e.label or e.qid),
    }
    rows: list[dict[str, Any]] = [{**base, "kind": _cell("item")}]
    for kind, entries in (("genre", e.genres), ("director", e.directors)):
        rows += [
            {
                **base,
                "kind": _cell(kind),
                "value": _cell(ENTITY + q, "uri"),
                "valueLabel": _cell(lbl),
            }
            for q, lbl in entries
        ]
    for q, lbl, ordinal in e.cast:
        row = {
            **base,
            "kind": _cell("cast"),
            "value": _cell(ENTITY + q, "uri"),
            "valueLabel": _cell(lbl),
        }
        if ordinal is not None:
            row["ord"] = _cell(ordinal)
        rows.append(row)
    rows += [
        {**base, "kind": _cell("language"), "value": _cell(ENTITY + q, "uri"), "text": _cell(code)}
        for q, code in e.languages
    ]
    rows += [{**base, "kind": _cell("date"), "text": _cell(d)} for d in e.dates]
    rows += [
        {
            **base,
            "kind": _cell("sitelink"),
            "value": _cell(f"https://{lang}.wikipedia.org/", "uri"),
            "text": _cell(title),
        }
        for lang, title in e.sitelinks.items()
    ]
    return rows


def summary_json(
    title: str, extract: str, *, revision: int = 1000, page_type: str = "standard"
) -> dict[str, Any]:
    return {
        "type": page_type,
        "title": title,
        "titles": {"canonical": title.replace(" ", "_"), "normalized": title, "display": title},
        "revision": str(revision),
        "extract": extract,
    }


class FakeWikimedia:
    def __init__(self) -> None:
        self.entities: dict[tuple[int, TitleType], list[Entity]] = {}
        # (lang, title) → summary JSON, or an int HTTP status.
        self.summaries: dict[tuple[str, str], dict[str, Any] | int] = {}
        self.sparql_status: int | None = None
        self.sparql_calls: list[tuple[TitleType, list[int]]] = []
        self.summary_calls: list[tuple[str, str]] = []
        self.user_agents: set[str] = set()

    def add(
        self,
        tmdb_id: int,
        type_: TitleType,
        entity: Entity,
        *,
        summary: str | None = None,
        summary_lang: str = "en",
    ) -> None:
        self.entities.setdefault((tmdb_id, type_), []).append(entity)
        if summary is not None:
            title = entity.sitelinks[summary_lang]
            self.summaries[(summary_lang, title)] = summary_json(title, summary)

    def sparql(self, request: httpx.Request) -> httpx.Response:
        self.user_agents.add(request.headers["User-Agent"])
        query = parse_qs(request.content.decode())["query"][0]
        prop = re.search(r"wdt:(P4947|P4983) \?tmdb", query)
        values = re.search(r"VALUES \?tmdb \{([^}]*)\}", query)
        assert prop, query
        assert values, query
        type_ = PROPERTY_TYPE[prop.group(1)]
        ids = [int(v) for v in re.findall(r'"(\d+)"', values.group(1))]
        self.sparql_calls.append((type_, ids))
        if self.sparql_status is not None:
            return httpx.Response(self.sparql_status, text="error")
        rows = [
            row
            for tmdb_id in ids
            for entity in self.entities.get((tmdb_id, type_), [])
            for row in entity_rows(tmdb_id, entity)
        ]
        return httpx.Response(200, json={"head": {"vars": []}, "results": {"bindings": rows}})

    def summary(self, request: httpx.Request, lang: str, title: str) -> httpx.Response:
        self.user_agents.add(request.headers["User-Agent"])
        name = unquote(title).replace("_", " ")
        self.summary_calls.append((lang, name))
        found = self.summaries.get((lang, name))
        if found is None:
            return httpx.Response(404, json={"type": "not_found"})
        if isinstance(found, int):
            return httpx.Response(found, text="error")
        return httpx.Response(200, json=found)

    def install(self, router: respx.MockRouter) -> None:
        router.post(SPARQL_URL).mock(side_effect=self.sparql)
        router.get(url__regex=SUMMARY_RE).mock(side_effect=self.summary)
