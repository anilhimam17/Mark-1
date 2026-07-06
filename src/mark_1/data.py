from fastf1.events import Event
from fastf1.core import Session


class DataUtils:
    """This class is responsible for managing the loading and storing operations
    performed on the raw FastF1 dataframes."""

    def __init__(self, race_event: Event) -> None:
        self.race_event = race_event

    def load_data(self) -> tuple[Session | None, Session | None]:
        """Loads all the information available through FastF1 for the Qualifying and
        Race sessions for a Race Event instance provided and returns them."""

        # Loading the Qualifying Session
        quali_session = None
        try:
            quali_session = self.race_event.get_qualifying()
            quali_session.load(
                laps=True,
                telemetry=True,
                weather=True,
                messages=True
            )
        except Exception as e:
            print(f"Exeception {str(e)} incurred while retrieveing the data for Qualifying.")

        # Loading the Race Session
        race_session = None
        try:
            race_session = self.race_event.get_race()
            race_session.load(
                laps=True,
                telemetry=True,
                weather=True,
                messages=True
            )
        except Exception as e:
            print(f"Exeception {str(e)} incurred while retrieveing the data for the Race.")

        return quali_session, race_session

