"""
Visualization Module

This module provides comprehensive visualization capabilities for the marine
ecosystem simulation, including population dynamics, environmental conditions,
and spatial distributions.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .simulation import MarineEcosystemSimulation


class SimulationVisualizer:
    """
    Comprehensive visualization class for marine ecosystem simulation results.

    Provides methods for creating static plots, interactive visualizations,
    and animations of simulation data.
    """

    def __init__(self, simulation: MarineEcosystemSimulation):
        """
        Initialize the visualizer with a simulation instance.

        Args:
            simulation: MarineEcosystemSimulation instance with data to visualize
        """
        self.simulation = simulation
        self.fig_size = (12, 8)
        self.colors = {
            "phytoplankton": "#2E8B57",  # Sea green
            "zooplankton": "#4169E1",  # Royal blue
            "fish": "#DC143C",  # Crimson
            "temperature": "#FF6347",  # Tomato
            "ph": "#9370DB",  # Medium purple
            "oxygen": "#00CED1",  # Dark turquoise
        }

    def plot_population_dynamics(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive population dynamics plot.

        Args:
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        results_df = self.simulation.get_results_dataframe()

        if results_df.empty:
            print("No simulation data available. Run simulation first.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Marine Ecosystem Population Dynamics", fontsize=16, fontweight="bold"
        )

        # Population counts over time
        ax1 = axes[0, 0]
        ax1.plot(
            results_df["step"],
            results_df["phytoplankton_count"],
            label="Phytoplankton",
            color=self.colors["phytoplankton"],
            linewidth=2,
        )
        ax1.plot(
            results_df["step"],
            results_df["zooplankton_count"],
            label="Zooplankton",
            color=self.colors["zooplankton"],
            linewidth=2,
        )
        ax1.plot(
            results_df["step"],
            results_df["fish_count"],
            label="Fish",
            color=self.colors["fish"],
            linewidth=2,
        )
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Population Count")
        ax1.set_title("Population Trends")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Energy levels over time
        ax2 = axes[0, 1]
        ax2.plot(
            results_df["step"],
            results_df["phyto_avg_energy"],
            label="Phytoplankton",
            color=self.colors["phytoplankton"],
            linewidth=2,
        )
        ax2.plot(
            results_df["step"],
            results_df["zoo_avg_energy"],
            label="Zooplankton",
            color=self.colors["zooplankton"],
            linewidth=2,
        )
        ax2.plot(
            results_df["step"],
            results_df["fish_avg_energy"],
            label="Fish",
            color=self.colors["fish"],
            linewidth=2,
        )
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Average Energy")
        ax2.set_title("Average Energy Levels")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Stress levels over time
        ax3 = axes[1, 0]
        ax3.plot(
            results_df["step"],
            results_df["phyto_avg_stress"],
            label="Phytoplankton",
            color=self.colors["phytoplankton"],
            linewidth=2,
        )
        ax3.plot(
            results_df["step"],
            results_df["zoo_avg_stress"],
            label="Zooplankton",
            color=self.colors["zooplankton"],
            linewidth=2,
        )
        ax3.plot(
            results_df["step"],
            results_df["fish_avg_stress"],
            label="Fish",
            color=self.colors["fish"],
            linewidth=2,
        )
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("Average Stress Level")
        ax3.set_title("Environmental Stress Levels")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Biodiversity metrics
        ax4 = axes[1, 1]
        ax4.plot(
            results_df["step"],
            results_df["shannon_diversity"],
            color="purple",
            linewidth=2,
            label="Shannon Diversity",
        )
        ax4_twin = ax4.twinx()
        ax4_twin.plot(
            results_df["step"],
            results_df["total_biomass"],
            color="orange",
            linewidth=2,
            label="Total Biomass",
        )
        ax4.set_xlabel("Time Steps")
        ax4.set_ylabel("Shannon Diversity Index", color="purple")
        ax4_twin.set_ylabel("Total Biomass (Energy)", color="orange")
        ax4.set_title("Ecosystem Health Metrics")
        ax4.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_environmental_conditions(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot environmental parameter changes over time.

        Args:
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        env_df = self.simulation.get_environmental_dataframe()

        if env_df.empty:
            print("No environmental data available. Run simulation first.")
            return None

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(
            "Environmental Conditions Over Time", fontsize=16, fontweight="bold"
        )

        # Temperature
        axes[0].plot(
            env_df["time"],
            env_df["avg_temperature"],
            color=self.colors["temperature"],
            linewidth=2,
        )
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title("Average Temperature")
        axes[0].grid(True, alpha=0.3)

        # pH
        axes[1].plot(
            env_df["time"], env_df["avg_ph"], color=self.colors["ph"], linewidth=2
        )
        axes[1].set_ylabel("pH")
        axes[1].set_title("Average pH")
        axes[1].grid(True, alpha=0.3)

        # Oxygen
        axes[2].plot(
            env_df["time"],
            env_df["avg_oxygen"],
            color=self.colors["oxygen"],
            linewidth=2,
        )
        axes[2].set_xlabel("Time Steps")
        axes[2].set_ylabel("Oxygen (mg/L)")
        axes[2].set_title("Average Oxygen Level")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_spatial_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a spatial distribution plot of current agent positions.

        Args:
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        distribution = self.simulation.get_agent_spatial_distribution()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Spatial Distribution of Marine Organisms", fontsize=16, fontweight="bold"
        )

        # Combined distribution
        ax1 = axes[0, 0]

        if distribution["phytoplankton"]:
            phyto_x, phyto_y = zip(*distribution["phytoplankton"])
            ax1.scatter(
                phyto_x,
                phyto_y,
                c=self.colors["phytoplankton"],
                alpha=0.6,
                s=20,
                label="Phytoplankton",
            )

        if distribution["zooplankton"]:
            zoo_x, zoo_y = zip(*distribution["zooplankton"])
            ax1.scatter(
                zoo_x,
                zoo_y,
                c=self.colors["zooplankton"],
                alpha=0.7,
                s=30,
                label="Zooplankton",
            )

        if distribution["fish"]:
            fish_x, fish_y = zip(*distribution["fish"])
            ax1.scatter(
                fish_x, fish_y, c=self.colors["fish"], alpha=0.8, s=50, label="Fish"
            )

        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position (Depth)")
        ax1.set_title("All Organisms")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Invert Y-axis so surface is at top

        # Individual distributions
        for i, (agent_type, positions) in enumerate(distribution.items()):
            ax = axes[0, 1] if i == 0 else axes[1, 0] if i == 1 else axes[1, 1]

            if positions:
                x_coords, y_coords = zip(*positions)
                ax.scatter(
                    x_coords, y_coords, c=self.colors[agent_type], alpha=0.7, s=40
                )

            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position (Depth)")
            ax.set_title(f"{agent_type.capitalize()} Distribution")
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_scenario_comparison(
        self, scenario_results: Dict[str, pd.DataFrame], save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare results from different climate scenarios.

        Args:
            scenario_results: Dictionary mapping scenario names to result DataFrames
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Climate Scenario Comparison", fontsize=16, fontweight="bold")

        # Population comparison for each species
        species = ["phytoplankton_count", "zooplankton_count", "fish_count"]
        species_names = ["Phytoplankton", "Zooplankton", "Fish"]

        for i, (species_col, species_name) in enumerate(zip(species, species_names)):
            ax = axes[i // 2, i % 2] if i < 3 else axes[1, 1]

            for scenario, df in scenario_results.items():
                ax.plot(df["step"], df[species_col], label=scenario, linewidth=2)

            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Population Count")
            ax.set_title(f"{species_name} Population")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Overall ecosystem health comparison
        ax = axes[1, 1]
        for scenario, df in scenario_results.items():
            ax.plot(df["step"], df["shannon_diversity"], label=scenario, linewidth=2)

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Shannon Diversity Index")
        ax.set_title("Ecosystem Diversity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create an interactive Plotly dashboard for exploring simulation results.

        Returns:
            Plotly Figure object
        """
        results_df = self.simulation.get_results_dataframe()
        env_df = self.simulation.get_environmental_dataframe()

        if results_df.empty:
            print("No simulation data available. Run simulation first.")
            return None

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Population Dynamics",
                "Energy Levels",
                "Environmental Stress",
                "Environmental Conditions",
                "Biodiversity Metrics",
                "Ecosystem Health",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}],
            ],
        )

        # Population dynamics
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["phytoplankton_count"],
                name="Phytoplankton",
                line=dict(color=self.colors["phytoplankton"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["zooplankton_count"],
                name="Zooplankton",
                line=dict(color=self.colors["zooplankton"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["fish_count"],
                name="Fish",
                line=dict(color=self.colors["fish"]),
            ),
            row=1,
            col=1,
        )

        # Energy levels
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["phyto_avg_energy"],
                name="Phyto Energy",
                line=dict(color=self.colors["phytoplankton"], dash="dot"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["zoo_avg_energy"],
                name="Zoo Energy",
                line=dict(color=self.colors["zooplankton"], dash="dot"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["fish_avg_energy"],
                name="Fish Energy",
                line=dict(color=self.colors["fish"], dash="dot"),
            ),
            row=1,
            col=2,
        )

        # Stress levels
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["phyto_avg_stress"],
                name="Phyto Stress",
                line=dict(color=self.colors["phytoplankton"], dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["zoo_avg_stress"],
                name="Zoo Stress",
                line=dict(color=self.colors["zooplankton"], dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["fish_avg_stress"],
                name="Fish Stress",
                line=dict(color=self.colors["fish"], dash="dash"),
            ),
            row=2,
            col=1,
        )

        # Environmental conditions
        if not env_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=env_df["time"],
                    y=env_df["avg_temperature"],
                    name="Temperature",
                    line=dict(color=self.colors["temperature"]),
                ),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=env_df["time"],
                    y=env_df["avg_ph"],
                    name="pH",
                    line=dict(color=self.colors["ph"]),
                ),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=env_df["time"],
                    y=env_df["avg_oxygen"],
                    name="Oxygen",
                    line=dict(color=self.colors["oxygen"]),
                ),
                row=2,
                col=2,
            )

        # Biodiversity metrics (with secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["shannon_diversity"],
                name="Shannon Diversity",
                line=dict(color="purple"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["total_biomass"],
                name="Total Biomass",
                line=dict(color="orange"),
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Marine Ecosystem Simulation Dashboard",
            title_x=0.5,
        )

        return fig

    def create_environment_heatmap(self, parameter: str = "temperature") -> plt.Figure:
        """
        Create a heatmap of environmental conditions.

        Args:
            parameter: Environmental parameter to visualize ('temperature', 'ph', 'oxygen')

        Returns:
            matplotlib Figure object
        """
        env = self.simulation.environment

        if parameter == "temperature":
            data = env.temperature
            title = "Temperature Distribution (°C)"
            cmap = "coolwarm"
        elif parameter == "ph":
            data = env.ph
            title = "pH Distribution"
            cmap = "viridis"
        elif parameter == "oxygen":
            data = env.oxygen
            title = "Oxygen Distribution (mg/L)"
            cmap = "Blues"
        else:
            raise ValueError("Parameter must be 'temperature', 'ph', or 'oxygen'")

        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(data.T, cmap=cmap, origin="lower", aspect="auto")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position (Depth)")
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(parameter.capitalize())

        # Invert y-axis so surface is at top
        ax.invert_yaxis()

        return fig

    def create_animation(
        self, steps: int = 100, interval: int = 200, save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """
        Create an animation of the simulation.

        Args:
            steps: Number of animation frames
            interval: Milliseconds between frames
            save_path: Optional path to save animation

        Returns:
            matplotlib FuncAnimation object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Initialize empty plots
        phyto_scat = ax1.scatter(
            [],
            [],
            c=self.colors["phytoplankton"],
            alpha=0.6,
            s=20,
            label="Phytoplankton",
        )
        zoo_scat = ax1.scatter(
            [], [], c=self.colors["zooplankton"], alpha=0.7, s=30, label="Zooplankton"
        )
        fish_scat = ax1.scatter(
            [], [], c=self.colors["fish"], alpha=0.8, s=50, label="Fish"
        )

        # Set up spatial plot
        ax1.set_xlim(0, self.simulation.environment.width)
        ax1.set_ylim(0, self.simulation.environment.height)
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position (Depth)")
        ax1.set_title("Organism Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        # Population plot
        time_data = []
        phyto_data = []
        zoo_data = []
        fish_data = []

        (line_phyto,) = ax2.plot(
            [],
            [],
            label="Phytoplankton",
            color=self.colors["phytoplankton"],
            linewidth=2,
        )
        (line_zoo,) = ax2.plot(
            [], [], label="Zooplankton", color=self.colors["zooplankton"], linewidth=2
        )
        (line_fish,) = ax2.plot(
            [], [], label="Fish", color=self.colors["fish"], linewidth=2
        )

        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Population Count")
        ax2.set_title("Population Dynamics")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        def animate(frame):
            # Run one simulation step
            self.simulation.run_step()

            # Update spatial distribution
            distribution = self.simulation.get_agent_spatial_distribution()

            # Clear previous points
            ax1.clear()
            ax1.set_xlim(0, self.simulation.environment.width)
            ax1.set_ylim(0, self.simulation.environment.height)
            ax1.set_xlabel("X Position")
            ax1.set_ylabel("Y Position (Depth)")
            ax1.set_title(f"Organism Distribution - Step {frame}")
            ax1.grid(True, alpha=0.3)
            ax1.invert_yaxis()

            # Plot current positions
            if distribution["phytoplankton"]:
                phyto_x, phyto_y = zip(*distribution["phytoplankton"])
                ax1.scatter(
                    phyto_x,
                    phyto_y,
                    c=self.colors["phytoplankton"],
                    alpha=0.6,
                    s=20,
                    label="Phytoplankton",
                )

            if distribution["zooplankton"]:
                zoo_x, zoo_y = zip(*distribution["zooplankton"])
                ax1.scatter(
                    zoo_x,
                    zoo_y,
                    c=self.colors["zooplankton"],
                    alpha=0.7,
                    s=30,
                    label="Zooplankton",
                )

            if distribution["fish"]:
                fish_x, fish_y = zip(*distribution["fish"])
                ax1.scatter(
                    fish_x, fish_y, c=self.colors["fish"], alpha=0.8, s=50, label="Fish"
                )

            ax1.legend()

            # Update population plot
            time_data.append(frame)
            phyto_data.append(len(distribution["phytoplankton"]))
            zoo_data.append(len(distribution["zooplankton"]))
            fish_data.append(len(distribution["fish"]))

            line_phyto.set_data(time_data, phyto_data)
            line_zoo.set_data(time_data, zoo_data)
            line_fish.set_data(time_data, fish_data)

            # Adjust plot limits
            if time_data:
                ax2.set_xlim(0, max(time_data) + 10)
                max_pop = max(max(phyto_data), max(zoo_data), max(fish_data))
                ax2.set_ylim(0, max_pop + 50)

            return [line_phyto, line_zoo, line_fish]

        anim = animation.FuncAnimation(
            fig, animate, frames=steps, interval=interval, blit=False, repeat=False
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=1000 // interval)

        return anim

    def generate_summary_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive text summary of the simulation results.

        Args:
            save_path: Optional path to save the report

        Returns:
            Report text as string
        """
        results_df = self.simulation.get_results_dataframe()
        health_metrics = self.simulation.get_ecosystem_health_metrics()

        if results_df.empty:
            return "No simulation data available. Run simulation first."

        # Calculate summary statistics
        initial_stats = results_df.iloc[0]
        final_stats = results_df.iloc[-1]

        report = f"""
MARINE ECOSYSTEM SIMULATION REPORT
=====================================

Simulation Configuration:
- Environment Size: {self.simulation.environment.width} x {self.simulation.environment.height}
- Climate Scenario: {self.simulation.config.climate_scenario}
- Simulation Duration: {len(results_df)} time steps

Initial Population:
- Phytoplankton: {initial_stats['phytoplankton_count']}
- Zooplankton: {initial_stats['zooplankton_count']}
- Fish: {initial_stats['fish_count']}
- Total: {initial_stats['total_agents']}

Final Population:
- Phytoplankton: {final_stats['phytoplankton_count']}
- Zooplankton: {final_stats['zooplankton_count']}
- Fish: {final_stats['fish_count']}
- Total: {final_stats['total_agents']}

Population Change:
- Phytoplankton: {((final_stats['phytoplankton_count'] - initial_stats['phytoplankton_count']) / initial_stats['phytoplankton_count'] * 100):+.1f}%
- Zooplankton: {((final_stats['zooplankton_count'] - initial_stats['zooplankton_count']) / initial_stats['zooplankton_count'] * 100):+.1f}%
- Fish: {((final_stats['fish_count'] - initial_stats['fish_count']) / initial_stats['fish_count'] * 100):+.1f}%

Ecosystem Health Metrics:
- Final Diversity Index: {health_metrics.get('current_diversity', 0):.3f}
- Final Biomass: {health_metrics.get('current_biomass', 0):.1f}
- Average Stress Level: {health_metrics.get('average_stress_level', 0):.3f}
- Ecosystem Resilience Score: {health_metrics.get('ecosystem_resilience', 0):.3f}

Environmental Conditions (Final):
- Average Temperature: {final_stats['avg_temperature']:.1f}°C
- Average pH: {final_stats['avg_ph']:.2f}
- Average Oxygen: {final_stats['avg_oxygen']:.1f} mg/L

Key Observations:
"""

        # Add key observations based on data
        if final_stats["fish_count"] == 0:
            report += "- CRITICAL: Fish population went extinct during simulation\n"
        elif final_stats["fish_count"] < initial_stats["fish_count"] * 0.1:
            report += "- WARNING: Fish population severely depleted\n"

        if health_metrics.get("ecosystem_resilience", 0) < 0.3:
            report += "- WARNING: Low ecosystem resilience indicates vulnerability\n"
        elif health_metrics.get("ecosystem_resilience", 0) > 0.7:
            report += "- POSITIVE: High ecosystem resilience indicates stability\n"

        if health_metrics.get("average_stress_level", 0) > 0.7:
            report += "- CONCERN: High average stress levels across all species\n"

        report += f"""
Generated by Marine Ecosystem Simulation v{self.simulation.__class__.__module__.split('.')[0]}
"""

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)

        return report
