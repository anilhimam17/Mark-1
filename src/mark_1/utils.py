# FastF1 Deps
from fastf1.core import Session

# Data Deps
from dataclasses import dataclass
from pandas import Series

# Core Deps
from json import load
from importlib import resources


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


# ============ Utility Functions ============


def load_circuit_config(circuit: str) -> dict[str, str]:
    """Helper function to load the config file from data."""

    # Accessing the Data Path
    data_dir = resources.files("mark_1") / "data"
    json_transversable = data_dir / f"{circuit}.json"

    try:
        with resources.as_file(json_transversable) as file_path:
            with open(file_path, encoding="utf-8") as file:
                return load(file)
    except Exception as e:
        raise AssertionError(f"Error incurred when loading the circuit specification. Failed with {str(e)}")

      
def display_sector_stat(driver_lap: Series, sector: int) -> None:
    """Helper function to display the driver stats."""
    
    print(f"Sector - {sector}")
    print(f"Driver: {driver_lap['Driver']}")

    if sector == 1:
        print(f"Sector Time: {driver_lap['Sector1Time']}")
        print(f"Speed I1: {driver_lap['SpeedI1']}")
    elif sector == 2:
        print(f"Sector Time: {driver_lap['Sector2Time']}")
        print(f"Speed I2: {driver_lap['SpeedI2']}")
    else:
        print(f"Sector Time: {driver_lap['Sector3Time']}")
        print(f"Speed I3: {driver_lap['SpeedFL']}")

    print(f"Longest Straight Speed: {driver_lap['SpeedST']}\n")