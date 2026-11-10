"""Data-provider attribution shown on every UI screen (§5.7, FR-29, D-010).

`TMDB_NOTICE` is the single source of the TMDB notice text: the Streamlit footer
renders it and a test checks the README carries the same string.
"""

from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path

APP_NAME = "WhatToWatch"

# Exact wording required by the TMDB API terms. Do not edit (only the noun
# "application" may be swapped for another one the terms allow).
TMDB_NOTICE = (
    "This application uses TMDB and the TMDB APIs but is not endorsed, certified, "
    "or otherwise approved by TMDB."
)
JUSTWATCH_NOTICE = "Availability data by JustWatch."

# Official, unmodified logo from https://www.themoviedb.org/about/logos-attribution
# (file name hash = SHA-256 of the file). Vendored, never hotlinked or redrawn.
TMDB_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "tmdb_logo_blue_short.svg"
TMDB_LOGO_ALT = "TMDB logo"
TMDB_URL = "https://www.themoviedb.org/"

# The logo must be smaller and less prominent than the app name (`st.title`, ~44 px).
TMDB_LOGO_HEIGHT_PX = 12
FOOTER_TEST_ID = "wtw-footer"


@lru_cache
def tmdb_logo_data_uri() -> str:
    data = base64.b64encode(TMDB_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{data}"


def footer_html() -> str:
    """Always-visible footer: TMDB logo + exact notice (logo and notice always together)."""
    notice = html.escape(TMDB_NOTICE)
    return f"""
<style>
  .wtw-footer {{
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 999990;
    display: flex; align-items: center; justify-content: center; gap: 0.6rem;
    padding: 0.35rem 1rem; font-size: 0.75rem; line-height: 1.3;
    background: rgba(246, 246, 248, 0.97); color: #555;
    border-top: 1px solid rgba(0, 0, 0, 0.08);
  }}
  .wtw-footer img {{ height: {TMDB_LOGO_HEIGHT_PX}px; width: auto; flex: none; }}
  @media (prefers-color-scheme: dark) {{
    .wtw-footer {{ background: rgba(20, 22, 28, 0.97); color: #aaa;
                   border-top-color: rgba(255, 255, 255, 0.1); }}
  }}
</style>
<div class="wtw-footer" data-testid="{FOOTER_TEST_ID}">
  <a href="{TMDB_URL}" target="_blank" rel="noopener noreferrer"><img
     src="{tmdb_logo_data_uri()}" alt="{TMDB_LOGO_ALT}"
     height="{TMDB_LOGO_HEIGHT_PX}"></a>
  <span>{notice}</span>
</div>
"""
