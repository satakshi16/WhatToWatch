"""P1-13 / §5.7: TMDB notice (exact wording, one shared constant) and the official logo."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from ui.attribution import (
    FOOTER_TEST_ID,
    JUSTWATCH_NOTICE,
    TMDB_LOGO_ALT,
    TMDB_LOGO_HEIGHT_PX,
    TMDB_LOGO_PATH,
    TMDB_NOTICE,
    footer_html,
    tmdb_logo_data_uri,
)

README = Path(__file__).resolve().parents[2] / "README.md"
# Hash in the file name TMDB serves the logo under (blue_short-<sha256>.svg).
OFFICIAL_LOGO_SHA256 = "8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c"


def test_notice_is_the_exact_tmdb_wording() -> None:
    assert TMDB_NOTICE == (
        "This application uses TMDB and the TMDB APIs but is not endorsed, certified, "
        "or otherwise approved by TMDB."
    )


def test_readme_carries_the_same_notice() -> None:
    assert TMDB_NOTICE in README.read_text("utf-8")


def test_readme_carries_the_justwatch_notice() -> None:
    assert JUSTWATCH_NOTICE in README.read_text("utf-8")


def test_logo_is_the_unmodified_official_asset() -> None:
    data = TMDB_LOGO_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == OFFICIAL_LOGO_SHA256
    assert data.startswith(b"<svg")


def test_logo_data_uri_embeds_the_vendored_file() -> None:
    uri = tmdb_logo_data_uri()
    prefix = "data:image/svg+xml;base64,"
    assert uri.startswith(prefix)
    assert base64.b64decode(uri.removeprefix(prefix)) == TMDB_LOGO_PATH.read_bytes()


def test_footer_shows_logo_together_with_the_notice() -> None:
    html = footer_html()
    assert FOOTER_TEST_ID in html
    assert TMDB_NOTICE in html
    assert tmdb_logo_data_uri() in html
    assert f'alt="{TMDB_LOGO_ALT}"' in html
    assert "position: fixed" in html  # always visible
    assert "bottom: 0" in html  # a footer, never the header


def test_footer_logo_is_small() -> None:
    # Far smaller than the app title (st.title, ~44 px), so never reads as co-branding.
    assert TMDB_LOGO_HEIGHT_PX <= 16
    heights = {int(h) for h in re.findall(r"(?<![-\w])height[=:]\s*\"?(\d+)", footer_html())}
    assert heights == {TMDB_LOGO_HEIGHT_PX}
