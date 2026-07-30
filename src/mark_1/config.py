"""This scripts holds all the configurations used in mark-1.

The below configurations contribute to various 
downstream data analysis workflows.
"""


from dataclasses import dataclass, field

# ======================= Generic Mapping =======================

SECTOR_MAPS: dict[str, tuple[str, str, str]] = {
    "Sector1": ("SpeedI1", "Sector1Time", "KineticEnergyS1_KJ"),
    "Sector2": ("SpeedI2", "Sector2Time", "KineticEnergyS2_KJ"),
    "Sector3": ("SpeedFL", "Sector3Time", "KineticEnergyS3_KJ")
}

# ======================= Feature / Data Configurations =======================


@dataclass(frozen=True)
class FeatureConfig:
    """This class holds all the feature specific configurations necessary for downstream workflows."""

    # ======================= Feature Categorisation =======================
    # Overlapping Features, especially for this regulations
    COMMON_CATEGORIES: list[str] = field(default_factory=lambda: [
        "AccelerationTime", "ERS_Clipping"
    ])

    # Pace Specific Categories
    PACE_CATEGORIES: list[str] = field(default_factory=lambda: [
        "FuelAwareLapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "AccelerationTime", "ERS_Clipping"
    ])

    # Speed Specific Categories
    SPEED_CATEGORIES: list[str] = field(default_factory=lambda: [
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "FrontAEI", "BalancedAEI", "RearAEI", 
        "AccelerationTime", "ERS_Clipping", "FuelAwareLapTime"
    ])

    # Energy Specific Categories
    ENERGY_CATEGORIES: list[str] = field(default_factory=lambda: [
        "KineticEnergyS1_KJ", "KineticEnergyS2_KJ", "KineticEnergyS3_KJ",
        "AccelerationTime", "ERS_Clipping", "FuelAwareLapTime"
    ])

    # ======================= Feature Scaling Properties =======================
    DIRECT_PROPORTION: list[str] = field(default_factory=lambda: [
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", 
        "FrontAEI", "BalancedAEI", "RearAEI"
    ])

    INVERSE_PROPORTION: list[str] = field(default_factory=lambda: [
        "Sector1Time", "Sector2Time", "Sector3Time", 
        "KineticEnergyS1_KJ", "KineticEnergyS2_KJ", "KineticEnergyS3_KJ",
        "AccelerationTime", "ERS_Clipping", "FuelAwareLapTime"
    ])


@dataclass(frozen=True)
class DataEngineeringConfig:
    """This class provides all the configurations used in Data Engineering Operations."""

    DROP_COLS: list[str] = field(default_factory=lambda: [
        "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime", 
        "DeletedReason", "FastF1Generated", "IsAccurate",
        "LapStartDate", "LapStartTime"
    ])

    TIME_COLS: list[str] = field(default_factory=lambda: [
        "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"
    ])


# ======================= Assumed Constant for Pace Grounding =======================


@dataclass(frozen=True)
class CarSpecifications:
    """This class is the container for all the car specifications."""

    # Fuel Load
    MAX_FUEL_LOAD_IN_KG: float = 70

    # Car Weight without Fuel with Driver (assumed optimal)
    CAR_WEIGHT_IN_KG: float = 772
    
    # Fuel Flow
    MAX_FUEL_FLOW_IN_KGH: float = 100
    
    # Considering Hybrid as a 48% split
    HYBRID_POWER: float = 0.48


@dataclass(frozen=True)
class ConversionConstants:
    """This class contains all the constants used for car-based calculations."""

    # Fuel Flow in g/sec
    FUEL_FLOW_CONV_CONST = 1000 / 3600

    # Weight to Time Conversion => 0.3s every 10kg
    WEIGHT_TIME_CONV_CONST = 0.3 / 10

    # KM-HR to M-S Time Conversion
    MS_CONV_CONST = 5 / 18
    
    
@dataclass(frozen=True)
class RaceStrategyConfig:
    """This class is the container for all the race specific configurations."""

    # Fuel Save for Race Trim pace
    FUEL_STRAT = 5

    # Fuel Sample Limit
    FUEL_SAMPLE_LIMIT = 3


# ======================= Circuit Specific Configurations =======================


@dataclass
class SectorConfig:
    """This class standardises the data structure expected for each sector in circuit json files."""

    title: str
    x_var: str
    y_var: str


@dataclass
class AccelerationConfig:
    """This class standardises the data structure specification for Acceleration data."""

    acceleration_dist: float
    v1_var: str
    v2_var: str


@dataclass
class CircuitConfig:
    """This class standardises the structure of data expected from each circuit json file."""

    # Meta Data
    circuit_name: str

    # Configurations
    aero_config: dict[str, SectorConfig]
    ke_config: dict[str, SectorConfig]
    power_config: dict[str, SectorConfig]
    acceleration_config: AccelerationConfig

    @classmethod
    def from_dict(cls, json_dict: dict) -> 'CircuitConfig':
        """Reconstructs the CircuitConfig object using the loaded Circuit JSON as dict."""
        # Parsing all the Aero Plot Configurations
        aero_parsed = {
            k: SectorConfig(**v) for k, v in 
            json_dict["aero_config"].items()
        }

        # Parsing all the Kinetic Energy Plot Configurations
        ke_parsed = {
            k: SectorConfig(**v) for k, v in
            json_dict["ke_config"].items()
        }

        # Parsing all the Power Plot Configurations
        power_parsed = {
            k: SectorConfig(**v) for k, v in
            json_dict["power_config"].items()
        }

        # Parsing all the Acceleration Configurations
        acc_parsed = AccelerationConfig(**json_dict["acceleration_config"])

        return cls(
            circuit_name=json_dict["circuit_name"],
            acceleration_config=acc_parsed,
            aero_config=aero_parsed,
            ke_config=ke_parsed,
            power_config=power_parsed
        )


# ======================= Data Visualisation Configurations =======================


@dataclass(frozen=True)
class VisualisationConfig:
    """This class if the container for all the Visualisation based configurations."""

    ERS_VIS_CONFIG: list[tuple[str, str, str]] = field(default_factory=lambda: [
        ("AccelerationTime", "ERS_Clipping", "")
    ])

    POLAR_CONFIG: dict = field(default_factory=lambda: 
        dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        )
    )
