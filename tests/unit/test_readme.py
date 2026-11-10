"""README stays in sync with the Makefile and setup docs (P1-14)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _documented_make_targets() -> set[str]:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([a-z-]+):.*## ", makefile, re.MULTILINE)) - {"help"}


def test_every_make_target_is_in_the_readme_table() -> None:
    rows = set(re.findall(r"^\| `([a-z-]+)`", README, re.MULTILINE))
    rows |= {"fmt"} if "`lint` / `fmt`" in README else set()
    rows |= {"lint"} if "`lint` / `fmt`" in README else set()
    assert _documented_make_targets() <= rows


def test_readme_covers_fresh_clone_setup() -> None:
    for needle in (
        "TMDB_READ_TOKEN",
        "TMDB_API_KEY",
        "WIKIMEDIA_CONTACT",
        "make install",
        "make ingest",
        "make open-content",
        "make embed",
        "make api",
        "make ui",
        "make seed-demo",
        "## Architecture",
    ):
        assert needle in README, needle


def test_readme_screenshots_exist() -> None:
    images = re.findall(r"\]\((docs/screenshots/[^)]+)\)", README)
    assert images
    for image in images:
        assert (ROOT / image).is_file(), image
