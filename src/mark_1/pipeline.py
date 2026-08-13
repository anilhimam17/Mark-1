"""This script houses the custom class DataPipeline.

The purpose of this script is to abstract the Data Engineering operations
under a single class.
"""

# FastF1 Deps
from fastf1.core import Laps

# Data Deps
from pandas import Series, DataFrame, concat

# Source Deps
from .config import (
    SECTOR_MAPS,
    CircuitConfig,
    CarSpecifications,
    ConversionConstants,
    FeatureConfig,
    RaceStrategyConfig
)
from .utils import load_circuit_config


class DataPipeline:
    """This class handles all the Data Engineering API.

    It house multiple method for all the transformations made to the raw FastF1 data.
    This includes: Feature Engineering, Feature Scaling and Point Stats.
    """

    def __init__(self, circuit: str) -> None:
        """Class Constructor."""
        circuit_json = load_circuit_config(circuit=circuit)
        
        # Instances of all the configurations used by the Pipeline
        self.circuit_spec = CircuitConfig.from_dict(json_dict=circuit_json)
        self.car_spec = CarSpecifications()
        self.conversion_spec = ConversionConstants()
        self.feature_spec = FeatureConfig()
        self.race_spec = RaceStrategyConfig()

    # ==================== Filtering Methods ====================
    
    def get_filtered_quali_laps(
            self, 
            laps_frame: Laps | DataFrame, 
        ) -> DataFrame:
        """Filters the fastest Qualifying Laps for each driver during the session."""
        fastest_quali_laps = (
            laps_frame
            .groupby("Driver")["FuelAwareLapTime"]
            .idxmin()
        )

        filtered_fastest_quali_laps = laps_frame.loc[fastest_quali_laps]
        return filtered_fastest_quali_laps
    
    def get_mean_race_laps(
        self,
        laps_frame: Laps | DataFrame,
    ) -> DataFrame:
        """Filters the mean Race Lap (performance) for each of driver during the session."""
        # Aggregation functions for mean performance
        agg_functions = {
            "Sector1Time": "mean",
            "Sector2Time": "mean",
            "Sector3Time": "mean",
            "LapTime": "mean",
            "SpeedI1": "mean",
            "SpeedI2": "mean",
            "SpeedFL": "mean",
            "SpeedST": "mean"
        }

        filtered_mean_race_laps = (
            laps_frame
            .groupby("Driver")
            .agg(agg_functions)
            .reset_index()
        )

        return filtered_mean_race_laps
    
    def get_mean_race_performance(
            self,
            laps_frame: Laps | DataFrame
    ) -> DataFrame:
        """Filters the mean Race Performance (with all Engineered Features) for each driver."""
        # Aggregation Functions
        agg_functions = {
            "Sector1Time": "mean",
            "Sector2Time": "mean",
            "Sector3Time": "mean",
            "FuelAwareLapTime": "mean",
            "SpeedI1": "mean",
            "SpeedI2": "mean",
            "SpeedFL": "mean",
            "SpeedST": "mean",
            "FrontAEI": "mean",
            "BalancedAEI": "mean",
            "RearAEI": "mean",
            "KineticEnergyS1_KJ": "mean",
            "KineticEnergyS2_KJ": "mean",
            "KineticEnergyS3_KJ": "mean",
            "PowerS1_KW": "mean",
            "PowerS2_KW": "mean",
            "PowerS3_KW": "mean",
            "AccelerationTime": "mean",
            "ERS_Clipping": "mean"
        }

        mean_race_performance = (
            laps_frame
            .groupby("Driver")
            .agg(agg_functions)
            .reset_index()
        )

        return mean_race_performance
    
    # ==================== Feature Engineering Methods ====================
    
    def get_aero_efficiency(
            self,
            sector: str,
            laps_frame: Laps
    ) -> Series:
        """Orchestrates the calculation of Aerodynamic Efficiency based on the Circuit Characteristics."""
        # Accessing the Respective Keys from Config
        speed_key, time_key, _ = SECTOR_MAPS[sector]

        # Purple Sector Time for Reference
        purple_sector_time = laps_frame[time_key].min()

        # Full AEI series
        sector_aei = self._calc_aero_efficiency(
            v_sector=laps_frame[speed_key],
            v_st=laps_frame["SpeedST"],
            sector_time=laps_frame[time_key],
            purple_sector_time=purple_sector_time
        )

        return sector_aei
    
    def _calc_aero_efficiency(
            self, 
            v_sector: Series, 
            v_st: Series, 
            sector_time: Series,
            purple_sector_time: float
        ) -> Series:
        """Estimates the Aerodynamic Efficiency using the appropriate velocity parameters."""
        # Raw Speed Retention
        speed_ratio = v_sector / v_st

        # Time Weighting
        time_ratio = purple_sector_time / sector_time

        # Sector Time Weighting for better Pace Capture
        aei = speed_ratio * time_ratio

        assert aei is not None, "The Sector AEI was None"
        return aei
    
    def get_delta_kinetic_energy(
            self,
            sector: str,
            laps_frame: Laps
    ) -> Series:
        """Orchestrates the calculation of Kinetic Energy Retention based on Circuit Characteristics."""
        # Accessing the Respective Keys from Config
        speed_key, _, _ = SECTOR_MAPS[sector]

        # Full KE series
        sector_ke = self._calc_delta_kinetic_energy(
            v1=laps_frame[speed_key],
            v2=laps_frame["SpeedST"]
        )

        assert sector_ke is not None, "There Sector KE was None"
        return sector_ke
    
    def _calc_delta_kinetic_energy(
        self,
        v1: Series,
        v2: Series
    ) -> Series:
        """Estimates the Difference in Kinetic Energy using the appropriate velocity parameters."""
        # Convert velocities to m/s before squaring to preserve physical scaling
        v1_ms = v1 * self.conversion_spec.MS_CONV_CONST ** 2
        v2_ms = v2 * self.conversion_spec.MS_CONV_CONST ** 2

        delta_kinetic_energy = (
            (1 / 2) * self.car_spec.CAR_WEIGHT_IN_KG * 
            (v2_ms - v1_ms)
        )

        return delta_kinetic_energy / 1e3
    
    def get_power_expenditure(
        self,
        sector: str,
        laps_frame: Laps
    ) -> Series:
        """Estimates the Power Deployment based on Circuit Characteristics and Kinetic Energy Retention."""
        # Accessing the Respective Keys from Config
        _, time_key, energy_key = SECTOR_MAPS[sector]

        # Full Power series
        sector_power = laps_frame[energy_key] / laps_frame[time_key]

        assert sector_power is not None, "There Sector Power was None"
        return sector_power
    
    def get_delta_acceleration_time(
        self,
        v1: Series,
        v2: Series
    ) -> Series:
        """Estimates the Average Acceleration using the appropriate velocity parameters."""
        distance_straight = self.circuit_spec.acceleration_config.acceleration_dist
        delta_acceleration_time = (2 * distance_straight) / (v1 + v2)

        return delta_acceleration_time * 3600
    
    def get_ers_clipping(
        self,
        v1: Series,
        v2: Series
    ) -> Series:
        """Estimates the ERS clipping factor as a ratio of speeds.
        
        It utilises the top speed through the speed trap and the exit speed from
        a prior corner to generate the ratio.
        """
        return v2 / v1

    # ==================== Traffic and Delta related methods ====================
    
    def get_traffic_delta(self, laps_frame: Laps) -> Laps:
        """Orchestrates the calculation of the traffic window that each driver experiences during the race.
        
        It is based on the Time Synchronised Positions of each driver at the start and end of the lap.
        """
        # Sorting all the Laps wrt Session Time for Traffic
        laps_frame_traffic = laps_frame.sort_values(
            by="Time", 
            ascending=True, 
            axis=0
        )

        # Shifting the LapTimes by 1 period for delta
        shifted_laptimes_traffic = laps_frame_traffic.groupby("LapNumber")["Time"].shift(1)

        # Adding the New Delta's
        laps_frame["TrafficDelta"] = self._calculate_traffic_delta(
            current_driver_time=laps_frame["Time"],
            driver_infront_time=shifted_laptimes_traffic
        )

        return laps_frame

    def _calculate_traffic_delta(
            self, 
            current_driver_time: Series, 
            driver_infront_time: Series
        ) -> Series:
        """Estimates the traffic window that each driver experiences during the race."""
        # Driver Delta wrt Session Time
        driver_deltas = current_driver_time - driver_infront_time
        
        return (
            driver_deltas
            .dt.total_seconds()
            .fillna(0.0)
        )

    # ==================== Fuel and Pace related methods ====================
    
    def get_effective_fuel_load(
            self, 
            max_fuel_load_in_kg: float = CarSpecifications.MAX_FUEL_LOAD_IN_KG,
            fuel_strat: float = RaceStrategyConfig.FUEL_STRAT,
            fuel_sample_limit: float = RaceStrategyConfig.FUEL_SAMPLE_LIMIT
        ) -> float:
        """Estimates the Effective Fuel Load uniformly carried by each driver during the race."""
        return (
            max_fuel_load_in_kg     # in kg
            - fuel_strat            # in kg
            - fuel_sample_limit     # in kg
        )
    
    def get_effective_fuel_flow(
            self,
            effective_fuel_load: float,
            max_race_laps: int,
            avg_laptime: float
        ) -> float:
        """Estimates a linear decay of the fuel based on each drivers mean laptime."""
        # Average fuel burn per lap
        avg_fuel_burn = effective_fuel_load / max_race_laps
        target_fuel_flow = (avg_fuel_burn / avg_laptime) * 1000

        return target_fuel_flow
    
    def get_lap_fuel_burn(
            self, 
            laptimes: Series,
            effective_fuel_flow: float
        ) -> Series:
        """Helper function to estimate the linear fuel burn in kg."""
        return (laptimes * effective_fuel_flow) / 1000

    def get_lap_fuel_penality(
            self, 
            cumulative_fuel_burn: Series,
            effective_fuel_load: float
        ) -> Series:
        """Helper function to estimate the time penality to negate for Zero-Fuel pace."""
        delta_fuel_load = effective_fuel_load - cumulative_fuel_burn
        remaining_fuel_load = delta_fuel_load.apply(lambda x: max(x, 0.0))

        return remaining_fuel_load * self.conversion_spec.WEIGHT_TIME_CONV_CONST

    def get_fuel_aware_laptime(
            self, 
            laptimes: Series,
            fuel_penality: Series
        ) -> Series:
        """Helper function to estimate the fuel-aware (Zero-Fuel pace) laptime."""
        return laptimes - fuel_penality
    
    # ==================== Rescaling Functions ====================
    
    def get_rescaled_direct_features(self, laps_frame: DataFrame) -> DataFrame:
        """Rescales the features which are directly proportional."""
        for feature in self.feature_spec.DIRECT_PROPORTION:
            laps_frame.loc[:, feature] = self._scale_direct(
                    x=laps_frame[feature],
                    min_x=laps_frame[feature].min(),
                    max_x=laps_frame[feature].max(),
                )
        
        return laps_frame
    
    def get_rescaled_inverse_features(self, laps_frame: DataFrame) -> DataFrame:
        """Rescales the features which are inversely proportional."""
        for feature in self.feature_spec.INVERSE_PROPORTION:
            laps_frame.loc[:, feature] = self._scale_inverse(
                    x=laps_frame[feature],
                    min_x=laps_frame[feature].min(),
                    max_x=laps_frame[feature].max(),
                )
        
        return laps_frame

    def _scale_direct(self, x: Series, min_x: float, max_x: float) -> Series:
        """This function rescales the values of each feature which scales directly."""
        numerator = x - min_x
        denominator = max_x - min_x
        
        return (numerator / (denominator + 1e-7)) * 100

    def _scale_inverse(self, x: Series, min_x: float, max_x: float) -> Series:
        """This funciton rescales the values of each feature which scales inversely."""
        numerator = max_x - x
        denominator = max_x - min_x
        
        return (numerator / (denominator + 1e-7)) * 100