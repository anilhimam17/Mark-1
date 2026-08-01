"""This script houses the custom class DataVisualisation.

The purpose of this script is to manage the generation of 
all the charts used for analysis.
"""

# FastF1 Deps
from fastf1.core import Session

# Visualisation Deps
from fastf1.plotting import get_driver_style

import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.colors import to_rgba
import seaborn as sns

# Data Deps
from numpy import append
from pandas import DataFrame

# Source Deps
from .config import (
    CircuitConfig,
    FeatureConfig,
    VisualisationConfig,
)
from .utils import load_circuit_config


class DataVisualisation:
    """This class handles all the Visualisation API.

    It houses multiple methods that aid visualising Aero Efficiency,
    Energy Retention and Power Deployment among others.
    """

    def __init__(
        self, 
        session: Session,
        driver_names: list,
        circuit: str
    ) -> None:
        """Class Constructor."""
        # Cacheing the Driver Colors and Markers
        self.driver_colors_hex = {}
        self.driver_colors_rgba = {}
        self.driver_markers = {}
        
        # Accessing the Styles for each Driver
        for driver in driver_names:
            driver_style = get_driver_style(driver, style=["marker", "color"], session=session)
            self.driver_colors_hex[driver] = driver_style["color"]
            self.driver_markers[driver] = driver_style["marker"].replace("x", "X")
            
            # Converting the Hexcode to RGBA
            rgba = to_rgba(self.driver_colors_hex[driver], alpha=0.5)
            rgba_str = f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"
            self.driver_colors_rgba[driver] = rgba_str

        # Instance of all the necessary configurations
        self.vis_config = VisualisationConfig()
        self.feature_config = FeatureConfig()

        # Loading the circuit specific configurations
        circuit_spec = load_circuit_config(circuit=circuit)
        self.circuit_config = CircuitConfig.from_dict(json_dict=circuit_spec)

    # ======================= Member Methods =======================
    def create_scatter_plots(
        self,
        nrows: int,
        ncols: int,
        figsize: tuple[int, int],
        data: DataFrame,
        hue: str,
        style: str,
        size: int,
        plot_kind: str,
    ) -> figure.Figure:
        """Generates a Matplotlib subplot with Seaborn Scatterplots.

        It can generate plots that match different parameters based on the plot_kind parameter.
        This includes: Aero Efficiency and ERS Clipping.
        """
        # Matplotlib Canvas
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=False,
            sharey=False
        )

        # Flatten the axes only when multiple subplots are being used
        if nrows > 1 or ncols > 1:
            axes = axes.flatten()

        # Accessing the Scatter Plot Configurations
        plot_configs = None
        if plot_kind == "aero":
            plot_configs = self.circuit_config.aero_config
        elif plot_kind == "ers_clip":
            plot_configs = self.vis_config.ERS_VIS_CONFIG

        # Sanity Check for Plot Configs
        assert plot_configs, "The plot configuration wasn't satisfied."

        # If there is only one subplot
        if len(plot_configs) == 1:
            assert isinstance(plot_configs, list), "ERS plotting config failed"
            x, y, title = plot_configs[0]

            # Subplot for the Axes.
            sns.scatterplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                palette=self.driver_colors_hex,
                style=style,
                markers=self.driver_markers,
                ax=axes,
                s=size
            )
            axes.set_title(title, pad=25)
            axes.grid()
        # If there are multiple subplots
        else:
            assert isinstance(plot_configs, dict), "Aero plotting config failed"
            for ax_idx, ax_config in enumerate(plot_configs.items()):
                _, sector_config = ax_config

                # Subplot for the Axes.
                sns.scatterplot(
                    data=data,
                    x=sector_config.x_var,
                    y=sector_config.y_var,
                    hue=hue,
                    palette=self.driver_colors_hex,
                    style=style,
                    markers=self.driver_markers,
                    ax=axes[ax_idx],
                    s=size
                )
                axes[ax_idx].set_title(sector_config.title, pad=25)
                axes[ax_idx].grid()
            
        return fig
    
    def create_bar_plots(
            self,
            nrows: int,
            ncols: int,
            figsize: tuple[int, int],
            data: DataFrame,
            hue: str,
            plot_kind: str
        ) -> figure.Figure:
        """Generates a Matplotlib subplot with Seaborn Barplots.
        
        It can generate plots that match different parameters based on the plot_kind parameter.
        This includes: Kinetic Energy Retention and Power Deployment.
        """
        # Matplotlib Canvas
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=False,
            sharey=False,
            figsize=figsize
        )
        axes = axes.flatten()

        # Accessing the Scatter Plot Configurations
        plot_configs = None
        if plot_kind == "ke":
            plot_configs = self.circuit_config.ke_config
        elif plot_kind == "power":
            plot_configs = self.circuit_config.power_config

        # Sanity Check for Plot Configs
        assert plot_configs, "The plot configuration wasn't satisfied."
        for ax_idx, ax_config in enumerate(plot_configs.items()):
            _, sector_config = ax_config

            # Subplot for the Axes.
            sns.barplot(
                data=data,
                x=sector_config.x_var,
                y=sector_config.y_var,
                hue=hue,
                palette=self.driver_colors_hex,
                ax=axes[ax_idx],
            )
            
            axes[ax_idx].set_title(sector_config.title, pad=25)
            axes[ax_idx].grid()

        return fig
    
    def create_degradation_plot(
        self,
        laps_frame: DataFrame,
        x: str,
        y: str,
        order: int,
        hue: str,
        height: int,
        aspect: float,
        row: str | None = None,
        col: str | None = None
    ) -> sns.FacetGrid:
        """Generates a Seaborn FacetGrid of Regression Plots.
        
        It can generate multiple regression plots illustrating the Tyre Degradation
        of the Drivers through each of their respective stint for a given race / 
        sector over the race.
        """
        # Outlined Grid for Facet Plots
        pace_grid = sns.FacetGrid(
            data=laps_frame,
            sharex=False,
            sharey=False,
            hue=hue,
            row=row,
            col=col,
            height=height,
            aspect=aspect,
            palette=self.driver_colors_hex
        )

        # Plotting the Facets with Regplots
        pace_grid.map_dataframe(
            sns.regplot,
            x=x,
            y=y,
            order=order,
            scatter_kws={"s": 60},
        )

        # Annotating the Facets
        for ax in pace_grid.axes.flatten():
            ax.grid()
            ax.legend()

        return pace_grid
    
    def create_pace_plot(
        self,
        laps_frame: DataFrame,
        x: str,
        y: str,
        hue: str,
        figsize: tuple[int, int]
    ) -> figure.Figure:
        """Generates a Seaborn Boxplot.
        
        It can generate a boxplot illustrating the pace of each driver through a stint
        over the race distance for the full lap and each individual sector.
        """
        pace_grid, axes = plt.subplots(
            nrows=1, 
            ncols=1, 
            figsize=figsize
        )

        # Plotting the Facets with Regplots
        sns.boxplot(
            data=laps_frame,
            x=x,
            y=y,
            ax=axes,
            hue=hue,
            palette=self.driver_colors_hex,
            gap=0.2
        )

        # Annotating the Facets
        axes.legend()
        axes.grid()

        return pace_grid
