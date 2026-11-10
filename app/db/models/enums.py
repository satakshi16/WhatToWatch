"""String enums persisted as VARCHAR + CHECK constraint (§6)."""

from __future__ import annotations

from enum import StrEnum


class TitleType(StrEnum):
    MOVIE = "movie"
    TV = "tv"


class UserStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    DELETED = "deleted"


class Era(StrEnum):
    LAST_2Y = "last_2y"
    LAST_10Y = "last_10y"
    ANY = "any"
    CLASSICS_PRE_2000 = "classics_pre_2000"


class OnboardingStep(StrEnum):
    PREFERENCES = "preferences"
    SEEDS = "seeds"
    QUICK_RATE = "quick_rate"
    DONE = "done"


class OfferType(StrEnum):
    FLATRATE = "flatrate"
    FREE = "free"
    ADS = "ads"
    RENT = "rent"
    BUY = "buy"
    # Only for the "checked, none available" marker row (provider_id = 0).
    NONE = "none"


class InteractionKind(StrEnum):
    RATING = "rating"
    THUMB = "thumb"
    SEEN = "seen"
    NOT_INTERESTED = "not_interested"
    WATCHLIST = "watchlist"
    CLICK = "click"
    REFINE = "refine"


class InteractionSource(StrEnum):
    ONBOARDING_SEED = "onboarding_seed"
    QUICK_RATE = "quick_rate"
    WEEKLY = "weekly"
    BROWSE = "browse"
    AGENT = "agent"
