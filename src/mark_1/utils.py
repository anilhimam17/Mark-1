from dataclasses import dataclass

from fastf1.core import Session


@dataclass
class EventSessions:
    """This data class outlines the structured response that is returned on loading
    data using the FastF1 API."""

    # Conventional Grandprix Weekend
    quali_session: Session | None = None
    race_session: Session | None = None

    # Sprint Grandprix Weekend
    sprint_quali_session: Session | None = None
    sprint_session: Session | None = None