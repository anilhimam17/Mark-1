# FastF1 Deps
from fastf1.core import Session

# Data Deps
from pandas import DataFrame
from numpy import append

# Visualisation Deps
from fastf1.plotting import get_driver_style

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import figure

import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Source Deps
from .config import VisualisationConfig, FeatureConfig


class DataVisualisation:
    """This class is responsible for all the Visualisation generated on the data
    that was processed using the downstream API."""

    def __init__(
        self, 
        session: Session,
        driver_names: list,
    ) -> None:
        
        # Cacheing the Driver Colors and Markers
        self.driver_colors_hex = {}
        self.driver_colors_rgba = {}
        self.driver_markers = {}
        
        # Accessing the Styles for each Driver
        for driver in driver_names:
            driver_style = get_driver_style(driver, style=["marker", "color"], session=session)
            self.driver_colors_hex[driver] = driver_style["color"]
            self.driver_markers[driver] = driver_style["marker"]
            
            # Converting the Hexcode to RGBA
            rgba = to_rgba(self.driver_colors_hex[driver], alpha=0.5)
            rgba_str = f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"
            self.driver_colors_rgba[driver] = rgba_str

        # Instance of all the necessary configurations
        self.vis_config = VisualisationConfig()
        self.feature_config = FeatureConfig()

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
        """This function creates a Matplotlib canvas of subplots and creates a Seaborn
        Scatterplot for each of the subplots. It then returns the completed figure."""

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
            plot_configs = self.vis_config.AERO_VIS_CONFIG
        elif plot_kind == "ers_clip":
            plot_configs = self.vis_config.ERS_VIS_CONFIG

        # Sanity Check for Plot Configs
        assert plot_configs, "The plot configuration wasn't satisfied."

        # If there is only one subplot
        if len(plot_configs) == 1:
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
            for ax_idx, ax_config in enumerate(plot_configs):
                x, y, title = ax_config

                # Subplot for the Axes.
                sns.scatterplot(
                    data=data,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=self.driver_colors_hex,
                    style=style,
                    markers=self.driver_markers,
                    ax=axes[ax_idx],
                    s=size
                )
                axes[ax_idx].set_title(title, pad=25)
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
        """This function creates a Matplotlib canvas of subplots and creates a Seaborn
        Barplot for each of the subplots. It then returns the completed figure."""

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
            plot_configs = self.vis_config.KE_VIS_CONFIG
        elif plot_kind == "power":
            plot_configs = self.vis_config.POWER_VIS_CONFIG

        # Sanity Check for Plot Configs
        assert plot_configs, "The plot configuration wasn't satisfied."
        for ax_idx, ax_config in enumerate(plot_configs):
            x, y, title = ax_config

            # Subplot for the Axes.
            sns.barplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                palette=self.driver_colors_hex,
                ax=axes[ax_idx],
            )
            
            axes[ax_idx].set_title(title, pad=25)
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
        """This function generates a seaborn Facet Grid that visualises Tyre Degradation
        by Driver and Stint along with other customisations."""

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
        """This function generates a Seaborn Boxplot that visualises the Race Pace
        by Driver and Stint along with other customisations."""

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
    
    def create_comparison_radar(
            self,
            quali_performance: DataFrame,
            race_performance: DataFrame,
            drivers: list,
            show_pace_categories: bool = True,
            show_speed_categories: bool = True,
            show_energy_categories: bool = True
        ) -> go.Figure:
        """This function generates a Plotly subplot that visualises multiple
        Radar Plots (go.Scatterpolar objects) which compare the performances of 
        the drivers in quali and race trim."""

        # Subplot Spec
        ncols = 2
        nrows = [
            show_pace_categories, 
            show_energy_categories, 
            show_speed_categories
        ].count(True)
        row_idx = [i for i in range(1, nrows + 1)]

        energy_categories_closed = None
        pace_categories_closed = None 
        speed_categories_closed = None
        subplot_titles = []
        row_track = 0
        energy_row_index = None
        pace_row_index = None
        speed_row_index = None
        
        if show_pace_categories:
            subplot_titles.extend(["Pace: Quali", "Pace: Race"])
            pace_categories_closed = (
                self.feature_config.PACE_CATEGORIES + 
                [self.feature_config.PACE_CATEGORIES[0]]
            )
            pace_row_index = row_idx[row_track]
            row_track += 1
        if show_speed_categories:
            subplot_titles.extend(["Speed: Quali", "Speed: Race"])
            speed_categories_closed = (
                self.feature_config.SPEED_CATEGORIES +
                [self.feature_config.SPEED_CATEGORIES[0]]
            )
            speed_row_index = row_idx[row_track]
            row_track += 1
        if show_energy_categories:
            subplot_titles.extend(["Energy: Quali", "Energy: Race"])
            energy_categories_closed = (
                self.feature_config.ENERGY_CATEGORIES + 
                [self.feature_config.ENERGY_CATEGORIES[0]]
            )
            energy_row_index = row_idx[row_track]
            row_track += 1

        # Plotly Subplots Canvas
        fig = make_subplots(
            rows=nrows, 
            cols=ncols, 
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=subplot_titles,
            specs=[[{"type": "polar"}, {"type": "polar"}] for _ in range(nrows)],
            vertical_spacing=0.1,
            horizontal_spacing=0.3,
        )

        # Adding the Traces
        for driver in drivers:
            
            # Accessing the Driver Data
            driver_quali_data = quali_performance[quali_performance["Driver"] == driver]
            driver_race_data = race_performance[race_performance["Driver"] == driver]

            # ================ Pace Categories ================
            if show_pace_categories:
                r_quali_pace = driver_quali_data[self.feature_config.PACE_CATEGORIES].to_numpy().flatten()
                r_quali_pace_closed = append(r_quali_pace, r_quali_pace[0])

                r_race_pace = driver_race_data[self.feature_config.PACE_CATEGORIES].to_numpy().flatten()
                r_race_pace_closed = append(r_race_pace, r_race_pace[0])

                # Quali Trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_quali_pace_closed,
                        theta=pace_categories_closed,
                        fill="toself",
                        name=driver,
                        fillcolor=self.driver_colors_rgba[driver],
                        legendgrouptitle_text=driver,
                        legendgroup=driver,
                        subplot="polar",
                        showlegend=True
                    ),
                    row=pace_row_index,
                    col=1,
                )
                # Race Trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_race_pace_closed,
                        theta=pace_categories_closed,
                        fill="toself",
                        name=driver,
                        fillcolor=self.driver_colors_rgba[driver],
                        legendgrouptitle_text=driver,
                        legendgroup=driver,
                        subplot="polar2",
                        showlegend=False
                    ),
                    row=pace_row_index,
                    col=2
                )
            
            # ================ Speed Categories ================
            if show_speed_categories:
                r_quali_speed = driver_quali_data[self.feature_config.SPEED_CATEGORIES].to_numpy().flatten()
                r_quali_speed_closed = append(r_quali_speed, r_quali_speed[0])

                r_race_speed = driver_race_data[self.feature_config.SPEED_CATEGORIES].to_numpy().flatten()
                r_race_speed_closed = append(r_race_speed, r_race_speed[0])

                # Quali Trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_quali_speed_closed,
                        theta=speed_categories_closed,
                        fill="toself",
                        name=driver,
                        fillcolor=self.driver_colors_rgba[driver],
                        legendgrouptitle_text=driver,
                        legendgroup=driver,
                        subplot="polar3",
                        showlegend=False
                    ),
                    row=speed_row_index,
                    col=1,
                )
                # Race Trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_race_speed_closed,
                        theta=speed_categories_closed,
                        fill="toself",
                        name=driver,
                        fillcolor=self.driver_colors_rgba[driver],
                        legendgrouptitle_text=driver,
                        legendgroup=driver,
                        subplot="polar4",
                        showlegend=False
                    ),
                    row=speed_row_index,
                    col=2
                )
            
            # ================ Energy Categories ================
            if show_energy_categories:
                r_quali_energy = driver_quali_data[self.feature_config.ENERGY_CATEGORIES].to_numpy().flatten()
                r_quali_energy_closed = append(r_quali_energy, r_quali_energy[0])

                r_race_energy = driver_race_data[self.feature_config.ENERGY_CATEGORIES].to_numpy().flatten()
                r_race_energy_closed = append(r_race_energy, r_race_energy[0])

                # Quali Trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_quali_energy_closed,
                        theta=energy_categories_closed,
                        fill="toself",
                        name=driver,
                        fillcolor=self.driver_colors_rgba[driver],
                        legendgrouptitle_text=driver,
                        legendgroup=driver,
                        subplot="polar5",
                        showlegend=False
                    ),
                    row=energy_row_index,
                    col=1,
                )
                # Race Trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_race_energy_closed,
                        theta=energy_categories_closed,
                        fill="toself",
                        name=driver,
                        fillcolor=self.driver_colors_rgba[driver],
                        legendgrouptitle_text=driver,
                        legendgroup=driver,
                        subplot="polar6",
                        showlegend=False
                    ),
                    row=energy_row_index,
                    col=2
                )

        polar_configs = {}
        for i in range(1, nrows * ncols):
            if i == 1:
                polar_configs["polar"] = self.vis_config.POLAR_CONFIG
            else:
                polar_configs[f"polar{i}"] = self.vis_config.POLAR_CONFIG

        fig.update_layout(
            showlegend=True,
            height=int(nrows * 600),
            width=1280,
            title_text="Driver-Wise Performance Quali V Race",
            title_x=0.5,
            title_y=0.98,
            legend=dict(x=1.1, y=0.5, xanchor="left", yanchor="middle"),
            margin=dict(l=150, r=150, t=150, b=50),
            **polar_configs
        )

        return fig
