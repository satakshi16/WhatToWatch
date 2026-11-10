"""P1-09: seed list and interaction request validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.db.models import TitleType
from app.schemas.feedback import MAX_SEED_ITEMS, InteractionIn, SeedsIn, TitleRef


def _seeds(*ids: int, type_: str = "movie") -> list[dict[str, Any]]:
    return [{"tmdb_id": i, "type": type_} for i in ids]


def test_seeds_keep_first_occurrence_order() -> None:
    body = SeedsIn.model_validate({"seeds": _seeds(3, 1, 3, 2, 1)})
    assert body.seeds == [
        TitleRef(tmdb_id=3, type=TitleType.MOVIE),
        TitleRef(tmdb_id=1, type=TitleType.MOVIE),
        TitleRef(tmdb_id=2, type=TitleType.MOVIE),
    ]


@pytest.mark.parametrize("ids", [(), (1,), (1, 2), (1, 1, 2, 2), tuple(range(1, 12))])
def test_seed_count_out_of_range(ids: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError, match="distinct titles"):
        SeedsIn.model_validate({"seeds": _seeds(*ids)})


@pytest.mark.parametrize("count", [3, 10])
def test_seed_count_bounds_accepted(count: int) -> None:
    assert len(SeedsIn.model_validate({"seeds": _seeds(*range(1, count + 1))}).seeds) == count


def test_raw_list_is_capped_even_if_it_dedupes_to_a_valid_count() -> None:
    with pytest.raises(ValidationError):
        SeedsIn.model_validate({"seeds": _seeds(*([1, 2, 3] * MAX_SEED_ITEMS))})


def test_same_id_movie_and_tv_are_distinct() -> None:
    body = SeedsIn.model_validate({"seeds": [*_seeds(1, 2), *_seeds(1, type_="tv")]})
    assert len(body.seeds) == 3


@pytest.mark.parametrize(
    ("data", "ok"),
    [
        ({"kind": "rating", "value": 1}, True),
        ({"kind": "rating", "value": 5}, True),
        ({"kind": "rating", "value": "3"}, True),  # lax int parsing, as elsewhere in the API
        ({"kind": "not_interested"}, True),
        ({"kind": "rating"}, False),
        ({"kind": "rating", "value": 0}, False),
        ({"kind": "rating", "value": 2.5}, False),
        ({"kind": "not_interested", "value": 3}, False),
        ({"kind": "seen"}, False),
    ],
)
def test_interaction_value_rules(data: dict[str, Any], ok: bool) -> None:
    payload = {"tmdb_id": 10, "type": "tv", **data}
    if ok:
        InteractionIn.model_validate(payload)
    else:
        with pytest.raises(ValidationError):
            InteractionIn.model_validate(payload)
