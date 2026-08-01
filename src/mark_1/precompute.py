"""This script houses the custom class PreCompute.

The purpose of this script is to unify all of the Data Engineering,
Data Transformation API and output precomputed, rich dataframes in their
final form. This is essential for easier integration into Steffi-1 
and Racing-all-Along workflows downstream.
"""

# FastF1 Deps
from fastf1.events import Event
from fastf1.core import Laps
from fastf1 import get_event

# Data Deps
from pandas import DataFrame, Series, concat

# Source Deps
from .config import DataEngineeringConfig, CircuitConfig
from .data import DataLoader
from .pipeline import DataPipeline
from .utils import EventSessions, load_circuit_config


class PreCompute:
    """This class provides an unified API of Data Ops.
    
    It integrates all the steps taken in DataPipeline, DataLoader to precompute
    and store the final DataFrames that are useful for Visualisation.
    """

    def __init__(
            self, 
            race_name: str, 
            load_telemetry: bool = False, 
            load_drivers: int = 5
        ) -> None:
        """Class Constructor."""
        assert race_name, "Invalid Grand Prix name was provided."
        self.race_name = race_name.lower()

        # Accessing the Race Event from FastF1
        self.race_event: Event = get_event(year=2026, gp=self.race_name)
        self.is_sprint: bool = True if self.race_event.get("EventFormat") == "sprint_qualifying" else False

        # Presetting the No of Drivers to Analyse (Max - 10)
        if load_drivers > 10 or load_drivers <= 0:
            raise ValueError("The max number of drivers being analysed exceed, please set a number upto 10")
        self.load_drivers: int = load_drivers

        # Loading the Data
        self.data_loader: DataLoader = DataLoader(race_event=self.race_event)
        self.gp_weekend: EventSessions = self.data_loader.load_data(
            is_sprint=self.is_sprint, 
            load_telemetry=load_telemetry
        )

        # Placeholders for all the Sessions
        self.max_race_laps: int = 0
        self.precomputed_frames: dict[str, Laps] = {}
        self.point_frames: dict[str, DataFrame] = {}
        self.results_frames: dict[str, list] = {}

        # Data Pipeline and Configs
        self.data_pipe = DataPipeline(circuit=self.race_name)
        self.circuit_config = CircuitConfig.from_dict(
            json_dict=load_circuit_config(circuit=self.race_name)
        )

    def _get_cleaned_data(self, session_laps: Laps, session_type: str) -> Laps:
        """Applies preliminary data cleaning operations on the Raw FastF1 Dataframe."""
        if session_type == "race":
            # Dropping Unnecessary Columns 
            cleaned_dataframe = session_laps.drop(DataEngineeringConfig().DROP_COLS, axis=1)

            # Converting all the timedelta objects
            for time_col in DataEngineeringConfig().TIME_COLS:
                cleaned_dataframe[time_col] = cleaned_dataframe[time_col].dt.total_seconds()
                
            # Picking all the green flag laps without boxing
            return (
                cleaned_dataframe
                .pick_wo_box()
                .pick_track_status("1", how="equals")
                .copy()
            )
        elif session_type == "quali":
            _, _, q3_frame = session_laps.split_qualifying_sessions()
            assert q3_frame is not None, "Failed loading laps from Q3"

            # Dropping Unnecessary Columns 
            cleaned_dataframe = q3_frame.drop(DataEngineeringConfig().DROP_COLS, axis=1)

            # Converting all the timedelta objects
            for time_col in DataEngineeringConfig().TIME_COLS:
                cleaned_dataframe[time_col] = cleaned_dataframe[time_col].dt.total_seconds()

            return cleaned_dataframe
        else:
            raise ValueError("The provided session type is invalid")

    def _data_engineering_wrapper(self, session_laps: Laps, session_type: str) -> Laps:
        """Applies all the data engineering operations in order."""
        # ==================== Preliminary Cleaning ====================
        
        cleaned_dataframe = self._get_cleaned_data(
            session_laps=session_laps,
            session_type=session_type
        )
        if session_type == "race":
            self.max_race_laps = cleaned_dataframe["LapNumber"].max()

        # ==================== Aero Efficiency Features ====================
        
        # Sector - 1
        s1_aero_config = self.circuit_config.aero_config.get("Sector1")
        assert s1_aero_config, "Failed loading S1 Aero configs"
        s1_aei = s1_aero_config.y_var
        cleaned_dataframe.loc[:, s1_aei] = self.data_pipe.get_aero_efficiency(
            sector="Sector1",
            laps_frame=cleaned_dataframe
        )

        # Sector - 2
        s2_aero_config = self.circuit_config.aero_config.get("Sector2")
        assert s2_aero_config, "Failed loading S2 Aero configs"
        s2_aei = s2_aero_config.y_var
        cleaned_dataframe.loc[:, s2_aei] = self.data_pipe.get_aero_efficiency(
            sector="Sector2",
            laps_frame=cleaned_dataframe
        )

        # Sector - 3
        s3_aero_config = self.circuit_config.aero_config.get("Sector3")
        assert s3_aero_config, "Failed loading S3 Aero configs"
        s3_aei = s3_aero_config.y_var
        cleaned_dataframe.loc[:, s3_aei] = self.data_pipe.get_aero_efficiency(
            sector="Sector3",
            laps_frame=cleaned_dataframe
        )

        # ==================== Kinetic Energy Features ====================
        
        # Sector - 1
        s1_ke_config = self.circuit_config.ke_config.get("Sector1")
        assert s1_ke_config, "Failed loading S1 KE configs"
        s1_ke = s1_ke_config.y_var
        cleaned_dataframe.loc[:, s1_ke] = self.data_pipe.get_delta_kinetic_energy(
            sector="Sector1",
            laps_frame=cleaned_dataframe
        )

        # Sector - 2
        s2_ke_config = self.circuit_config.ke_config.get("Sector2")
        assert s2_ke_config, "Failed loading S2 KE configs"
        s2_ke = s2_ke_config.y_var
        cleaned_dataframe.loc[:, s2_ke] = self.data_pipe.get_delta_kinetic_energy(
            sector="Sector2",
            laps_frame=cleaned_dataframe
        )

        # Sector - 3
        s3_ke_config = self.circuit_config.ke_config.get("Sector3")
        assert s3_ke_config, "Failed loading S3 KE configs"
        s3_ke = s3_ke_config.y_var
        cleaned_dataframe.loc[:, s3_ke] = self.data_pipe.get_delta_kinetic_energy(
            sector="Sector3",
            laps_frame=cleaned_dataframe
        )

        # ==================== Power Features ====================
        
        # Sector - 1
        s1_power_config = self.circuit_config.power_config.get("Sector1")
        assert s1_power_config, "Failed loading S1 Power configs"
        s1_power = s1_power_config.y_var
        cleaned_dataframe.loc[:, s1_power] = self.data_pipe.get_power_expenditure(
            sector="Sector1",
            laps_frame=cleaned_dataframe
        )

        # Sector - 2
        s2_power_config = self.circuit_config.power_config.get("Sector2")
        assert s2_power_config, "Failed loading S2 Power configs"
        s2_power = s2_power_config.y_var
        cleaned_dataframe.loc[:, s2_power] = self.data_pipe.get_power_expenditure(
            sector="Sector2",
            laps_frame=cleaned_dataframe
        )

        # Sector - 3
        s3_kpower_config = self.circuit_config.power_config.get("Sector3")
        assert s3_kpower_config, "Failed loading S3 Power configs"
        s3_power = s3_kpower_config.y_var
        cleaned_dataframe.loc[:, s3_power] = self.data_pipe.get_power_expenditure(
            sector="Sector3",
            laps_frame=cleaned_dataframe
        )

        # ==================== Acceleration and ERS ====================
        
        cleaned_dataframe.loc[:, "AccelerationTime"] = self.data_pipe.get_delta_acceleration_time(
            v1=cleaned_dataframe[self.circuit_config.acceleration_config.v1_var],
            v2=cleaned_dataframe[self.circuit_config.acceleration_config.v2_var]
        )

        cleaned_dataframe.loc[:, "ERS_Clipping"] = self.data_pipe.get_ers_clipping(
            v1=cleaned_dataframe[self.circuit_config.acceleration_config.v1_var],
            v2=cleaned_dataframe[self.circuit_config.acceleration_config.v2_var]
        )

        # ==================== Fuel-based Features ====================

        if session_type == "race":
            # Mean Race Laptimes for Fuel Estimations (will be better with Telemetry)
            init_mean_race_performance = self.data_pipe.get_mean_race_laps(
                laps_frame=cleaned_dataframe
            )
            
            # Sample Fuel Load (same for all drivers, for now)
            race_fuel_load = self.data_pipe.get_effective_fuel_load()

            # Fuel Flow based on Driver Laptimes
            driver_fuel_flows = {}
            for _, row in init_mean_race_performance.iterrows():
                driver_fuel_flows[row["Driver"]] = self.data_pipe.get_effective_fuel_flow(
                    effective_fuel_load=race_fuel_load,
                    max_race_laps=self.max_race_laps,
                    avg_laptime=row["LapTime"]
                )
            self.point_frames["driver_fuel_flows"] = DataFrame({
                "Driver": list(driver_fuel_flows.keys()),
                "MeanFuelFlow": list(driver_fuel_flows.values())
            })

            # Lap-wise fuel burn based on laptime and avg fuel flow
            lap_fuel_burn = Series([], dtype=float)
            for _, row in self.point_frames["driver_fuel_flows"].iterrows():
                driver_fuel_burn = self.data_pipe.get_lap_fuel_burn(
                    laptimes=cleaned_dataframe.pick_drivers(row["Driver"])["LapTime"],
                    effective_fuel_flow=row["MeanFuelFlow"]
                )
                if lap_fuel_burn is None:
                    lap_fuel_burn = driver_fuel_burn
                else:
                    lap_fuel_burn = concat([lap_fuel_burn, driver_fuel_burn], axis=0)
            cleaned_dataframe.loc[:, "LapFuelBurn"] = lap_fuel_burn

            # Lap-wise cumulative fuel burn over the race distance
            cleaned_dataframe.loc[:, "CumulativeLapFuelBurn"] = (
                cleaned_dataframe
                .groupby("Driver")["LapFuelBurn"]
                .cumsum()
            )

            # Lap-wise fuel penality to be negated (for true pace)
            cleaned_dataframe.loc[:, "LapFuelPenality"] = self.data_pipe.get_lap_fuel_penality(
                cumulative_fuel_burn=cleaned_dataframe["CumulativeLapFuelBurn"],
                effective_fuel_load=race_fuel_load
            )

            # Fuel-Aware Laptimes
            cleaned_dataframe.loc[:, "FuelAwareLapTime"] = self.data_pipe.get_fuel_aware_laptime(
                laptimes=cleaned_dataframe["LapTime"],
                fuel_penality=cleaned_dataframe["LapFuelPenality"]
            )

            # Max Fuel burned by each driver over the race distance
            self.point_frames["max_fuel_burn"] = (
                cleaned_dataframe
                .groupby("Driver")["CumulativeLapFuelBurn"]
                .last()
                .reset_index()
            )

        # ==================== Best Performance Frames ====================

        if session_type == "quali":
            # Renaming the LapTime feature
            new_cols = {k: k if k != "LapTime" else "FuelAwareLapTime" for k in cleaned_dataframe.columns}
            cleaned_dataframe.rename(new_cols, axis=1, inplace=True)

            # Fastest Lap Point Estimates
            self.point_frames["best_quali_performance"] = self.data_pipe.get_filtered_quali_laps(
                laps_frame=cleaned_dataframe
            )
            
            # Frames for Radar Plots
            scaled_quali_performance = self.data_pipe.get_rescaled_direct_features(
                laps_frame=self.point_frames["best_quali_performance"].copy()
            )
            scaled_quali_performance = self.data_pipe.get_rescaled_inverse_features(
                laps_frame=scaled_quali_performance,
                session_type="quali"
            )

            # Fastest Lap Scaled Performance
            self.point_frames["scaled_quali_performance"] = scaled_quali_performance

        elif session_type == "race":
            # Mean Lap Peformance Estimates
            self.point_frames["mean_race_performance"] = self.data_pipe.get_mean_race_performance(
                laps_frame=cleaned_dataframe
            )

            # Frames for Radar Plots
            scaled_race_performance = self.data_pipe.get_rescaled_direct_features(
                laps_frame=self.point_frames["mean_race_performance"].copy()
            )
            scaled_race_performance = self.data_pipe.get_rescaled_inverse_features(
                laps_frame=scaled_race_performance,
                session_type="race"
            )

            # Fastest Lap Scaled Performance
            self.point_frames["scaled_race_performance"] = scaled_race_performance

        return cleaned_dataframe


    def _run_all(self) -> None:
        """Orchestrates all the Data Engineering Operations on each of the Laps Frames."""
        # Standard Sessions
        if self.gp_weekend.quali_session:
            # Cacheing the Results for the Session
            self.results_frames["quali"] = (
                self.gp_weekend
                .quali_session
                .results
                .iloc[:self.load_drivers]["Abbreviation"]
                .tolist()
            )
            
            # Picking only the Top 10 Drivers from the Session
            session_laps = (
                self.gp_weekend
                .quali_session
                .laps
                .pick_drivers(self.results_frames["quali"])    
            )
            self.precomputed_frames["quali_laps"] = self._data_engineering_wrapper(
                session_laps=session_laps,
                session_type="quali"
            )
        
        if self.gp_weekend.race_session:
            # Cacheing the Results for the Session
            self.results_frames["race"] = (
                self.gp_weekend
                .race_session
                .results
                .iloc[:self.load_drivers]["Abbreviation"]
                .tolist()
            )

            # Picking only the Top 10 Drivers from the Session
            session_laps = (
                self.gp_weekend
                .race_session
                .laps
                .pick_drivers(self.results_frames["race"])
            )
            self.precomputed_frames["race_laps"] = self._data_engineering_wrapper(
                session_laps=session_laps,
                session_type="race"
            )
        
        # Optional Sessions if applicable
        if self.gp_weekend.sprint_quali_session:
            # Cacheing the Results for the Session
            self.results_frames["sprint_quali"] = (
                self.gp_weekend
                .sprint_quali_session
                .results
                .iloc[:self.load_drivers]["Abbreviation"]
                .tolist()
            )

            # Picking only the Top 10 Drivers from the Session
            session_laps = (
                self.gp_weekend
                .sprint_quali_session
                .laps
                .pick_drivers(self.results_frames["sprint_quali"])
            )
            self.precomputed_frames["sprint_quali_laps"] = self._data_engineering_wrapper(
                session_laps=session_laps,
                session_type="quali"
            )

        if self.gp_weekend.sprint_session:
            # Cacheing the Results for the Session
            self.results_frames["sprint_race"] = (
                self.gp_weekend
                .sprint_session
                .results
                .iloc[:self.load_drivers]["Abbreviation"]
                .tolist()
            )

            # Picking only the Top 10 Drivers from the Session
            session_laps = (
                self.gp_weekend
                .sprint_session
                .laps
                .pick_drivers(self.results_frames["sprint_race"])
            )
            self.precomputed_frames["sprint_race_laps"] = self._data_engineering_wrapper(
                session_laps=session_laps,
                session_type="race"
            )
