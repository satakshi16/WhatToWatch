"""WhatToWatch Streamlit UI (P1-13): sign up → preferences → seeds → recommendations.

Run with `make ui` (needs `make api`). Talks to the API only through `ui.api_client`;
never imports `app.*`. The access token lives in `st.session_state`.
Every screen ends with the always-visible TMDB attribution footer (§5.7).
"""

from __future__ import annotations

import html
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# `streamlit run ui/streamlit_app.py` puts `ui/` (not the repo root) on sys.path.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from ui import options  # noqa: E402
from ui.api_client import JSON, ApiClient, ApiError, api_base_url, connect  # noqa: E402
from ui.attribution import APP_NAME, JUSTWATCH_NOTICE, footer_html  # noqa: E402

MIN_SEEDS = 3
MAX_SEEDS = 10
REC_COUNT = 10
SEARCH_LIMIT = 10
MAX_RUNTIME = 1000

VIEW_RECS = "Recommendations"
VIEW_PREFS = "Preferences"
VIEW_SEEDS = "Seed titles"
VIEWS = (VIEW_RECS, VIEW_PREFS, VIEW_SEEDS)

state = st.session_state


# --------------------------------------------------------------------------- plumbing


@st.cache_resource(show_spinner=False)
def _http(base_url: str) -> httpx.Client:
    return connect(base_url)


def client() -> ApiClient:
    return ApiClient(_http(api_base_url()), state.get("token"))


def flash(level: str, message: str) -> None:
    """Message shown at the top of the next run (survives `st.rerun`)."""
    state.setdefault("flash", []).append((level, message))


def show_flash() -> None:
    for level, message in state.pop("flash", []):
        {"success": st.success, "warning": st.warning, "info": st.info}.get(level, st.error)(
            message
        )


def sign_out(message: str | None = None) -> None:
    pending = state.get("flash", [])
    state.clear()
    state.flash = pending
    if message:
        flash("warning", message)


def call[T](fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """Run an API call; show the problem+json error (or sign out on 401) and return None."""
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        if exc.unauthorized and state.get("token"):
            sign_out("Your session has expired. Please log in again.")
            st.rerun()
        st.error(exc.message)
        return None


def callback_call[T](fn: Callable[..., T], *args: Any) -> T | None:
    """Like `call`, for widget callbacks (they run before the script, so errors are flashed)."""
    try:
        return fn(*args)
    except ApiError as exc:
        if exc.unauthorized and state.get("token"):
            sign_out("Your session has expired. Please log in again.")
        else:
            flash("error", exc.message)
        return None


def navigate(view: str) -> None:
    """Switch view on the next run (the sidebar radio owns `state.view` once rendered)."""
    state.pending_view = view


def start_session(token: str, user: JSON) -> None:
    state.token = token
    state.user = user
    state.prefs = None
    state.recs = None


# --------------------------------------------------------------------------- auth


def country_select(label: str, current: str | None, key: str) -> str:
    countries = options.countries()
    codes = list(countries)
    index = codes.index(current) if current in countries else codes.index("US")
    return st.selectbox(
        label, codes, index=index, format_func=lambda c: f"{countries[c]} ({c})", key=key
    )


def render_auth() -> None:
    st.subheader("Weekly picks you can actually stream")
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
    with login_tab, st.form("login"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Log in", type="primary"):
            tokens = call(client().login, email=email.strip(), password=password)
            if tokens:
                api = ApiClient(_http(api_base_url()), tokens["access_token"])
                user = call(api.me)
                if user:
                    start_session(tokens["access_token"], user)
                    st.rerun()
    with signup_tab, st.form("signup"):
        display_name = st.text_input("Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input(
            "Password", type="password", help="At least 8 characters.", key="signup_password"
        )
        country = country_select("Country", "US", key="signup_country")
        age_ok = st.checkbox("I am 16 or older", key="signup_age")
        if st.form_submit_button("Create account", type="primary"):
            if not age_ok:
                st.error("You must be 16 or older to sign up.")
                return
            out = call(
                client().register,
                email=email.strip(),
                password=password,
                display_name=display_name.strip(),
                country=country,
                age_confirmed=age_ok,
            )
            if out:
                start_session(out["access_token"], out["user"])
                flash("success", f"Welcome, {out['user']['display_name']}! Let's set you up.")
                st.rerun()


# --------------------------------------------------------------------------- preferences


def load_providers(country: str) -> list[JSON] | None:
    cache: dict[str, list[JSON]] = state.setdefault("providers", {})
    if country not in cache:
        with st.spinner("Loading streaming services…"):
            out = call(client().providers, country)
        if out is None:
            return None
        cache[country] = out["providers"]
    return cache[country]


def logo_row(providers: list[JSON], link: str | None = None, size: int = 32) -> str:
    """Provider logos as HTML (optionally each linking to the TMDB watch page)."""
    tags = []
    for p in providers:
        name = html.escape(p.get("name") or f"Provider {p['provider_id']}")
        if p.get("logo_url"):
            inner = (
                f'<img src="{html.escape(p["logo_url"])}" alt="{name}" title="{name}" '
                f'height="{size}" style="border-radius:6px;margin:0 6px 4px 0">'
            )
        else:
            inner = f'<span style="margin-right:8px">{name}</span>'
        if link:
            inner = (
                f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer">'
                f"{inner}</a>"
            )
        tags.append(inner)
    return "".join(tags)


def render_preferences(user: JSON, prefs: JSON) -> None:
    first_time = prefs["onboarding_step"] == "preferences"
    st.header("Step 1 of 2 · Your preferences" if first_time else "Preferences")

    country = country_select("Country", user["country"], key="pref_country")
    providers = load_providers(country)
    if providers is None:
        return
    by_id = {p["provider_id"]: p for p in providers}

    saved_platforms = [pid for pid in prefs["platforms"] if pid in by_id]
    platforms = st.multiselect(
        "Streaming services you have",
        list(by_id),
        default=saved_platforms,
        format_func=lambda pid: by_id[pid]["name"],
        help="Only titles available on these services are recommended.",
        key=f"pref_platforms_{country}",
    )
    if platforms:
        st.markdown(logo_row([by_id[p] for p in platforms]), unsafe_allow_html=True)
    else:
        st.caption("Pick at least one service to get recommendations.")

    language_choices = list(options.LANGUAGES) + [
        code for code in prefs["languages"] if code not in options.LANGUAGES
    ]
    languages = st.multiselect(
        "Languages you enjoy (the first one is your preferred language)",
        language_choices,
        default=prefs["languages"],
        format_func=options.language_label,
        max_selections=20,
        key="pref_languages",
    )
    content_types = st.multiselect(
        "Show me",
        list(options.CONTENT_TYPES),
        default=prefs["content_types"],
        format_func=options.CONTENT_TYPES.__getitem__,
        key="pref_types",
    )
    era = st.selectbox(
        "Release years",
        list(options.ERAS),
        index=list(options.ERAS).index(prefs["era"]),
        format_func=options.ERAS.__getitem__,
        key="pref_era",
    )
    avoid_genres = st.multiselect(
        "Genres to avoid",
        list(options.GENRES),
        default=[g for g in prefs["avoid_genres"] if g in options.GENRES],
        format_func=options.genre_label,
        key="pref_avoid",
    )
    max_runtime = st.number_input(
        "Maximum movie runtime in minutes (0 = no limit)",
        min_value=0,
        max_value=MAX_RUNTIME,
        value=prefs["max_runtime"] or 0,
        step=10,
        key="pref_runtime",
    )
    include_rent_buy = st.checkbox(
        "Include titles I'd have to rent or buy",
        value=prefs["include_rent_buy"],
        key="pref_rent_buy",
    )
    allow_adult = st.checkbox("Allow adult content", value=prefs["allow_adult"], key="pref_adult")

    label = "Save and pick titles →" if first_time else "Save preferences"
    if not st.button(label, type="primary", disabled=not content_types, key="pref_save"):
        if not content_types:
            st.caption("Choose movies, series or both.")
        return

    if country != user["country"]:
        old_names = {p["provider_id"]: p["name"] for p in state.providers.get(user["country"], [])}
        updated = call(client().update_me, country=country)
        if updated is None:
            return
        removed = updated.pop("removed_platforms", [])
        state.user = updated
        if removed:
            names = ", ".join(old_names.get(pid, f"#{pid}") for pid in removed)
            flash("warning", f"Not available in {country}, removed from your services: {names}.")
    saved = call(
        client().save_preferences,
        {
            "languages": languages,
            "platforms": platforms,
            "content_types": content_types,
            "era": era,
            "avoid_genres": avoid_genres,
            "max_runtime": int(max_runtime) or None,
            "include_rent_buy": include_rent_buy,
            "allow_adult": allow_adult,
        },
    )
    if saved is None:
        return
    state.prefs = saved
    state.recs = None
    flash("success", "Preferences saved.")
    if first_time:
        navigate(VIEW_SEEDS)
    st.rerun()


# --------------------------------------------------------------------------- seeds


def add_seed(result: JSON) -> None:
    seeds: list[JSON] = state.setdefault("seeds", [])
    if len(seeds) < MAX_SEEDS and not any(_same(s, result) for s in seeds):
        seeds.append(result)


def remove_seed(result: JSON) -> None:
    state.seeds = [s for s in state.get("seeds", []) if not _same(s, result)]


def _same(a: JSON, b: JSON) -> bool:
    return bool(a["tmdb_id"] == b["tmdb_id"] and a["type"] == b["type"])


def _seed_label(seed: JSON) -> str:
    year = f" ({seed['year']})" if seed.get("year") else ""
    return f"{seed['title']}{year} · {options.TYPE_LABELS.get(seed['type'], seed['type'])}"


def run_search(query: str, type_: str) -> list[JSON] | None:
    """Search on submit only (one TMDB call per query, D-015), cached for the session."""
    cache: dict[str, list[JSON]] = state.setdefault("search_cache", {})
    key = f"{type_}:{' '.join(query.split()).casefold()}"
    if key not in cache:
        out = call(client().search_titles, query, type_, SEARCH_LIMIT)
        if out is None:
            return None
        cache[key] = out["results"]
    return cache[key]


def render_seeds(prefs: JSON) -> None:
    first_time = prefs["onboarding_step"] != "done"
    st.header("Step 2 of 2 · Titles you love" if first_time else "Add titles you love")
    st.write(
        f"Pick {MIN_SEEDS} to {MAX_SEEDS} movies or series you enjoyed. "
        "They shape your first recommendations."
    )
    seeds: list[JSON] = state.setdefault("seeds", [])

    with st.form("seed_search"):
        query_col, type_col = st.columns([3, 1])
        query = query_col.text_input(
            "Search titles", placeholder="e.g. Parasite, Squid Game", key="seed_query"
        )
        type_ = type_col.selectbox(
            "Type",
            list(options.SEARCH_TYPES),
            format_func=options.SEARCH_TYPES.__getitem__,
            key="seed_type",
        )
        if st.form_submit_button("Search"):
            if len(query.strip()) < 2:
                st.warning("Type at least 2 characters.")
            else:
                state.search = {"query": query.strip(), "type": type_}

    search = state.get("search")
    results = run_search(search["query"], search["type"]) if search else None
    if search and results is not None:
        if not results:
            st.info(f"No titles found for “{search['query']}”.")
        for result in results:
            chosen = any(_same(s, result) for s in seeds)
            poster_col, text_col, button_col = st.columns([1, 6, 2], vertical_alignment="center")
            if result.get("poster_url"):
                poster_col.image(result["poster_url"], width=46)
            text_col.markdown(f"**{html.escape(_seed_label(result))}**")
            if result["original_title"] != result["title"]:
                text_col.caption(result["original_title"])
            button_col.button(
                "Added ✓" if chosen else "Add",
                key=f"add-{result['type']}-{result['tmdb_id']}",
                on_click=add_seed,
                args=(result,),
                disabled=chosen or len(seeds) >= MAX_SEEDS,
            )

    st.subheader(f"Your picks ({len(seeds)}/{MAX_SEEDS})")
    if not seeds:
        st.caption(f"Nothing yet: search above and add at least {MIN_SEEDS} titles.")
    chip_cols = st.columns(2)
    for i, seed in enumerate(seeds):
        chip_cols[i % 2].button(
            f"✕  {_seed_label(seed)}",
            key=f"remove-{seed['type']}-{seed['tmdb_id']}",
            on_click=remove_seed,
            args=(seed,),
            help="Remove",
        )

    ready = MIN_SEEDS <= len(seeds) <= MAX_SEEDS
    if not ready and seeds:
        st.caption(f"Add {MIN_SEEDS - len(seeds)} more to continue.")
    if st.button(
        "Get my recommendations →", type="primary", disabled=not ready, key="seeds_submit"
    ):
        with st.spinner("Saving your titles…"):
            out = call(client().submit_seeds, seeds)
        if out is None:
            return
        state.prefs = {**prefs, "onboarding_step": out["onboarding_step"]}
        state.seed_report = out["seeds"]
        state.seeds = []
        state.search = None
        state.recs = None
        navigate(VIEW_RECS)
        st.rerun()


# --------------------------------------------------------------------------- recommendations


def _item_key(item: JSON) -> str:
    return f"{item['type']}-{item['tmdb_id']}"


def on_rate(item: JSON, widget_key: str) -> None:
    index = state.get(widget_key)
    if index is None:
        return
    stars = int(index) + 1
    if callback_call(client().rate, item["tmdb_id"], item["type"], stars) is not None:
        flash("success", f"Rated “{item['title']}” {'★' * stars}. Your list was refreshed.")
        state.recs = None


def on_not_interested(item: JSON) -> None:
    if callback_call(client().not_interested, item["tmdb_id"], item["type"]) is not None:
        flash("info", f"Got it, no more “{item['title']}”. Your list was refreshed.")
        state.recs = None


def render_card(item: JSON) -> None:
    with st.container(border=True):
        poster_col, body_col = st.columns([1, 4])
        if item.get("poster_url"):
            poster_col.image(item["poster_url"], width=150)
        else:
            poster_col.caption("No poster")
        with body_col:
            st.markdown(f"### {item['rank']}. {html.escape(item['title'])}")
            if item["original_title"] != item["title"]:
                st.caption(f"Original title: {item['original_title']}")
            st.write(options.meta_line(item))
            st.markdown(f"💡 *{html.escape(item['reason'])}*")
            providers = item.get("providers") or []
            links = logo_row(providers, link=item["watch_link"], size=36)
            st.markdown(
                f"{links}<br><a href='{html.escape(item['watch_link'])}' target='_blank' "
                f"rel='noopener noreferrer'>Where to watch</a>"
                f" · <small>{html.escape(JUSTWATCH_NOTICE)}</small>",
                unsafe_allow_html=True,
            )
            rate_col, skip_col = st.columns([3, 2], vertical_alignment="center")
            with rate_col:
                widget_key = f"rate-{_item_key(item)}"
                st.caption("Seen it? Rate it")
                st.feedback("stars", key=widget_key, on_change=on_rate, args=(item, widget_key))
            skip_col.button(
                "Not interested",
                key=f"skip-{_item_key(item)}",
                on_click=on_not_interested,
                args=(item,),
            )


def render_recommendations() -> None:
    title_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
    title_col.header("Your picks")
    if button_col.button("↻ Refresh", key="recs_refresh"):
        state.recs = None

    for seed in state.pop("seed_report", None) or []:
        if not seed["embedded"]:
            st.info(
                f"“{seed['title']}” was saved but doesn't shape your recommendations yet "
                "(we have no open description for it)."
            )

    if state.get("recs") is None:
        with st.spinner("Finding titles you can stream…"):
            state.recs = call(client().recommendations, REC_COUNT)
    recs = state.get("recs")
    if recs is None:
        return
    if recs["cold_start"]:
        st.info("Your picks are mostly popular titles for now. Rate a few to personalise them.")
    if not recs["items"]:
        st.warning(
            "Nothing matched right now. Add more streaming services or relax your preferences."
        )
        if st.button("Edit preferences", key="recs_to_prefs"):
            navigate(VIEW_PREFS)
            st.rerun()
        return
    for item in recs["items"]:
        render_card(item)


# --------------------------------------------------------------------------- layout


def default_view(prefs: JSON) -> str:
    step = prefs["onboarding_step"]
    if step == "preferences":
        return VIEW_PREFS
    return VIEW_SEEDS if step != "done" else VIEW_RECS


def render_signed_in() -> None:
    if state.get("prefs") is None:
        state.prefs = call(client().preferences)
        if state.prefs is None:
            return
    user: JSON = state.user
    prefs: JSON = state.prefs
    step = prefs["onboarding_step"]
    if "pending_view" in state:
        state.view = state.pop("pending_view")
    elif "view" not in state:
        state.view = default_view(prefs)

    with st.sidebar:
        st.markdown(f"Signed in as **{html.escape(user['display_name'])}**")
        st.caption(f"{user['email']} · {user['country']}")
        if step == "done":
            st.radio("Go to", VIEWS, key="view")
        if st.button("Log out", key="logout"):
            sign_out()
            st.rerun()

    # Onboarding is a straight path: preferences first, then seeds.
    view = state.view
    if step == "preferences":
        view = VIEW_PREFS
    elif step != "done" and view == VIEW_RECS:
        view = VIEW_SEEDS

    if view == VIEW_PREFS:
        render_preferences(user, prefs)
    elif view == VIEW_SEEDS:
        render_seeds(prefs)
    else:
        render_recommendations()


def render_footer() -> None:
    st.markdown("<div style='height:3.5rem'></div>", unsafe_allow_html=True)
    st.markdown(footer_html(), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="centered")
    st.title(f"🎬 {APP_NAME}")
    show_flash()
    try:
        if state.get("token"):
            render_signed_in()
        else:
            render_auth()
    finally:
        render_footer()


main()
