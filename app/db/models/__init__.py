"""ORM models. Importing this package registers every table on `Base.metadata`."""

from app.db.models.enums import (
    Era,
    InteractionKind,
    InteractionSource,
    OfferType,
    OnboardingStep,
    TitleType,
    UserStatus,
)
from app.db.models.interaction import Interaction, Rating
from app.db.models.open_content import WIKIPEDIA_LICENSE, TitleOpenContent
from app.db.models.title import NO_PROVIDER_ID, Provider, Title, TitleAvailability
from app.db.models.user import User, UserPlatform, UserPreferences

__all__ = [
    "NO_PROVIDER_ID",
    "WIKIPEDIA_LICENSE",
    "Era",
    "Interaction",
    "InteractionKind",
    "InteractionSource",
    "OfferType",
    "OnboardingStep",
    "Provider",
    "Rating",
    "Title",
    "TitleAvailability",
    "TitleOpenContent",
    "TitleType",
    "User",
    "UserPlatform",
    "UserPreferences",
    "UserStatus",
]
