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
    """This class is responsible for all the transformation operations performed
    on the FastF1 Laps DataFrames. It can generate all the new features that are
    added to the raw FastF1 Laps DataFrame."""

    def __init__(self, circuit: str) -> None:

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
            laps_frame: Laps, 
            drivers: list
        ) -> DataFrame:
        """This function filters all the laps from Q3 in qualifying to only the top runners
        from the race and retrieves their respective fastest laptimes."""

        # Aggregations functions for each of the cols for the best lap
        agg_functions = {
            "Sector1Time": "min",
            "Sector2Time": "min",
            "Sector3Time": "min",
            "LapTime": "min",
            "SpeedI1": "max",
            "SpeedI2": "max",
            "SpeedFL": "max",
            "SpeedST": "max"
        }

        filtered_fastest_quali_laps = (
            laps_frame
            .pick_drivers(drivers)
            .groupby("Driver")
            .agg(agg_functions)
            .reset_index()
        )

        return filtered_fastest_quali_laps
    
    def get_mean_race_laps(
        self,
        laps_frame: Laps,
        drivers: list
    ) -> DataFrame:
        """This function filters all the race laps for the mean performance of 
        each provided drivers and returns the filtered frame."""

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
            .pick_drivers(drivers)
            .groupby("Driver")
            .agg(agg_functions)
            .reset_index()
        )

        return filtered_mean_race_laps
    
    # ==================== Feature Engineering Methods ====================
    def get_aero_efficiency(
            self,
            sector: str,
            laps_frame: Laps,
            drivers: list
    ) -> Series:
        """The function orchestrates the calculation of Aero Efficiency for each driver
        and returns the combined series for all the drivers."""

        # Accessing the Respective Keys from Config
        speed_key, time_key, _ = SECTOR_MAPS[sector]

        # Purple Sector Time for Reference
        purple_sector_time = laps_frame[time_key].min()

        # Full AEI series
        sector_aei = None
        for driver in drivers:
            
            # Filtering the laps for the current driver
            driver_laps = laps_frame.pick_drivers(driver)
            
            # Calculating the AEI for the driver
            driver_aei = driver_laps.apply(
                lambda x: self._calc_aero_efficiency(
                    v_sector=x[speed_key],
                    v_st=x["SpeedST"],
                    sector_time=x[time_key],
                    purple_sector_time=purple_sector_time
                ),
                axis=1
            )

            if sector_aei is None:
                sector_aei = driver_aei
            else:
                sector_aei = concat([sector_aei, driver_aei], axis=0)

        assert sector_aei is not None, "There sector aei was None"
        return sector_aei
    
    def _calc_aero_efficiency(
            self, 
            v_sector: float, 
            v_st: float, 
            sector_time: float,
            purple_sector_time: float
        ) -> float:
        """This function estimates the Aero Efficiency of the Front and Rear Axles based on
        the velocity params provided and the corresponding sector time."""

        # Raw Speed Retention
        speed_ratio = v_sector / v_st

        # Time Weighting
        time_ratio = purple_sector_time / sector_time

        # Sector Time Weighting for better Pace Capture
        aei = speed_ratio * time_ratio * self.conversion_spec.MS_CONV_CONST

        return aei
    
    def get_delta_kinetic_energy(
            self,
            sector: str,
            laps_frame: Laps,
            drivers: list
    ) -> Series:
        """The function orchestrates the calculation of Kinetic Energy Retention for 
        each driver and returns the combined series for all the drivers."""

        # Accessing the Respective Keys from Config
        speed_key, _, _ = SECTOR_MAPS[sector]

        # Full KE series
        sector_ke = None
        for driver in drivers:
            
            # Filtering the laps for the current driver
            driver_laps = laps_frame.pick_drivers(driver)
            
            # Calculating the AEI for the driver
            driver_ke = driver_laps.apply(
                lambda x: self._calc_delta_kinetic_energy(
                    v1=x[speed_key],
                    v2=x["SpeedST"]
                ),
                axis=1
            )

            if sector_ke is None:
                sector_ke = driver_ke
            else:
                sector_ke = concat([sector_ke, driver_ke], axis=0)

        assert sector_ke is not None, "There sector ke was None"
        return sector_ke
    
    def _calc_delta_kinetic_energy(
        self,
        v1: float,
        v2: float
    ) -> float:
        """This function calculates the change in Kinetic Energy given velocity params and
        returns the result in Kilo Joules."""

        # Convert velocities to m/s before squaring to preserve physical scaling
        v1_ms = v1 * self.conversion_spec.MS_CONV_CONST
        v2_ms = v2 * self.conversion_spec.MS_CONV_CONST
        delta_kinetic_energy = (
            (1 / 2) * self.car_spec.CAR_WEIGHT_IN_KG * 
            (v2_ms ** 2 - v1_ms ** 2)
        )

        return delta_kinetic_energy / 1e3
    
    def get_power_expenditure(
        self,
        sector: str,
        laps_frame: Laps,
        drivers: list
    ) -> Series:
        """The function orchestrates the calculation of Power Expenditure of
        each driver and returns the combined series for all the drivers."""

        # Accessing the Respective Keys from Config
        _, time_key, energy_key = SECTOR_MAPS[sector]

        # Full Power series
        sector_power = None
        for driver in drivers:
            
            # Filtering the laps for the current driver
            driver_laps = laps_frame.pick_drivers(driver)
            
            # Calculating the AEI for the driver
            driver_power = driver_laps[energy_key] / driver_laps[time_key]

            if sector_power is None:
                sector_power = driver_power
            else:
                sector_power = concat([sector_power, driver_power], axis=0)

        assert sector_power is not None, "There sector power was None"
        return sector_power
    
    def get_delta_acceleration_time(
        self,
        v1: float,
        v2: float
    ) -> float:
        """This function calculates the acceleration time on the longest straight 
        given velocity params and returns the result in seconds."""

        distance_straight = self.circuit_spec.acceleration_dist
        delta_acceleration_time = (2 * distance_straight) / (v1 + v2)

        return delta_acceleration_time * 3600

    # ==================== Traffic and Delta related methods ====================
    def get_traffic_delta(self, laps_frame: Laps) -> Laps:
        """This function generates the effective traffic delta that each driver
        tackles during the race. It especially plays a major role in pace and deg calculations."""

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
        """This function calculates the actual interval between two drivers for each lap."""

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
            max_fuel_load_in_kg: float,
            fuel_strat: float,
            fuel_sample_limit: float
        ) -> float:
        """This function estimates the effective fuel available based on 
        the provided parameters."""

        return (
            max_fuel_load_in_kg     # in kg
            - fuel_strat            # in kg
            - fuel_sample_limit     # in kg
        )
    
    def get_effective_fuel_flow(
            self,
            effective_fuel_load: float,
            race_laps: int,
            avg_laptime: float,
        ) -> float:
        """This function estimates the effective fuel flow used by the team
        based on the provided parameters."""

        # Average fuel burn per lap
        avg_fuel_burn = effective_fuel_load / race_laps
        target_fuel_flow = (avg_fuel_burn / avg_laptime) * 1000

        return target_fuel_flow
    
    def get_lap_fuel_burn(
            self, 
            laptime: float,
            effective_fuel_flow: float
        ) -> float:
        """Helper function to estimate the linear fuel burn in kg."""
        
        return (laptime * effective_fuel_flow) / 1000

    def get_lap_fuel_penality(
            self, 
            cumulative_fuel_burn: float,
            effective_fuel_load: float
        ) -> float:
        """Helper function to estimate the time penality to negate for Zero-Fuel pace."""

        delta_fuel_load = effective_fuel_load - cumulative_fuel_burn
        remaining_fuel_load = max(delta_fuel_load, 0.0)

        return remaining_fuel_load * self.conversion_spec.WEIGHT_TIME_CONV_CONST

    def get_fuel_aware_laptime(
            self, 
            laptime: float,
            fuel_penality: float
        ) -> float:
        """Helper function to estimate the fuel-aware (Zero-Fuel pace) laptime."""
        
        return laptime - fuel_penality
    
    # ==================== Rescaling Functions ====================
    def get_rescaled_direct_features(self, laps_frame: DataFrame) -> DataFrame:
        """This function orchestrates the rescaling of all the Directly Proportional
        features from Mean / Best performance frames."""

        for feature in self.feature_spec.DIRECT_PROPORTION:
            laps_frame.loc[:, feature] = laps_frame.apply(
                lambda x: self._scale_direct(
                    x=x[feature],
                    min_x=laps_frame[feature].min(),
                    max_x=laps_frame[feature].max(),
                ),
                axis=1
            )
        
        return laps_frame
    
    def get_rescaled_inverse_features(self, laps_frame: DataFrame) -> DataFrame:
        """This function orchestrates the rescaling of all the Inversely Proportional
        features from Mean / Best performance frames."""

        for feature in self.feature_spec.INVERSE_PROPORTION:
            laps_frame.loc[:, feature] = laps_frame.apply(
                lambda x: self._scale_inverse(
                    x=x[feature],
                    min_x=laps_frame[feature].min(),
                    max_x=laps_frame[feature].max(),
                ),
                axis=1
            )
        
        return laps_frame

    def _scale_direct(self, x: float, min_x: float, max_x: float) -> float:
        """This function rescales the values of each feature which scales directly."""

        numerator = x - min_x
        denominator = max_x - min_x
        return (numerator / denominator + 1e-7) * 100

    def _scale_inverse(self, x: float, min_x: float, max_x: float) -> float:
        """This funciton rescales the values of each feature which scales inversely."""

        numerator = max_x - x
        denominator = max_x - min_x
        return (numerator / denominator + 1e-7) * 100