from fastf1.events import Event

from .utils import EventSessions


class DataLoader:
    """This class is responsible for managing the loading and storing operations
    performed on the raw FastF1 dataframes."""

    def __init__(self, race_event: Event) -> None:
        self.race_event = race_event

    def load_data(self, is_sprint: bool = False, load_telemetry: bool = False) -> EventSessions:
        """Loads all the information available through FastF1 for the Qualifying and
        Race sessions for a Race Event instance provided and returns them."""

        gp_weekend = EventSessions()

        # Loading the Qualifying Session
        try:
            quali_session = self.race_event.get_qualifying()
            quali_session.load(
                laps=True,
                telemetry=load_telemetry,
                weather=True,
                messages=True
            )
            gp_weekend.quali_session = quali_session
        except Exception as e:
            print(f"Exeception {str(e)} incurred while retrieving the data for Qualifying.")

        # Loading the Race Session
        try:
            race_session = self.race_event.get_race()
            race_session.load(
                laps=True,
                telemetry=load_telemetry,
                weather=True,
                messages=True
            )
            gp_weekend.race_session = race_session
        except Exception as e:
            print(f"Exeception {str(e)} incurred while retrieving the data for the Race.")

        # Loading the Sprint Data
        if is_sprint:
            
            # Loading the Sprint Qualifying
            try:
                sprint_quali_session = self.race_event.get_sprint_qualifying()
                sprint_quali_session.load(
                    laps=True,
                    telemetry=load_telemetry,
                    weather=True,
                    messages=True
                )
                gp_weekend.sprint_quali_session = sprint_quali_session
            except Exception as e:
                print(f"Exception {str(e)} incurred while retrieving the data for Sprint Qualifying.")

            # Loading the Sprint Race
            try:
                sprint_race_session = self.race_event.get_sprint()
                sprint_race_session.load(
                    laps=True,
                    telemetry=load_telemetry,
                    weather=True,
                    messages=True
                )
                gp_weekend.sprint_session = sprint_race_session
            except Exception as e:
                print(f"Exception {str(e)} incurred while retrieving the data for the Sprint Race.")

        return gp_weekend

