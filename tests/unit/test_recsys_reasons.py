"""§8.8 template reasons."""

from __future__ import annotations

import numpy as np

from app.recsys.reasons import language_name, most_similar_liked, template_reason

LIKED = np.array([[1, 0, 0], [0, 1, 0], [0.7, 0.7, 0]], dtype=np.float32)
NAMES = ["Interstellar", "Parasite", "The Prestige"]


def test_two_most_similar_liked_titles() -> None:
    pick = np.array([1, 0.2, 0], dtype=np.float32)
    reason = template_reason(
        vector=pick,
        liked_vectors=LIKED,
        liked_names=NAMES,
        language="en",
        platform="Netflix",
        cold_start=False,
    )
    assert reason == "Because you liked Interstellar and The Prestige"


def test_one_liked_title() -> None:
    reason = template_reason(
        vector=np.array([0, 1, 0], dtype=np.float32),
        liked_vectors=LIKED[1:2],
        liked_names=NAMES[1:2],
        language="ko",
        platform=None,
        cold_start=False,
    )
    assert reason == "Because you liked Parasite"


def test_cold_start_and_fallbacks_use_popular_template() -> None:
    kwargs = {"liked_vectors": LIKED, "liked_names": NAMES}
    v = np.array([1, 0, 0], dtype=np.float32)
    assert (
        template_reason(vector=v, language="hi", platform="Netflix", cold_start=True, **kwargs)
        == "Popular in Hindi on Netflix"
    )
    # Pick without a vector, or no liked titles → popular template.
    assert (
        template_reason(vector=None, language="ko", platform=None, cold_start=False, **kwargs)
        == "Popular in Korean"
    )
    assert (
        template_reason(
            vector=v,
            liked_vectors=np.zeros((0, 3), dtype=np.float32),
            liked_names=[],
            language="xx",
            platform="Max",
            cold_start=False,
        )
        == "Popular in XX on Max"
    )
    assert most_similar_liked(np.zeros(3, dtype=np.float32), LIKED, NAMES) == []


def test_language_names() -> None:
    assert language_name("ML") == "Malayalam"
    assert language_name("cn") == "Cantonese"
    assert language_name("zz") == "ZZ"
