"""
Streamlit Interface Module

This module provides a user-friendly web interface for running and visualizing
the marine ecosystem simulation using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import tempfile
from .simulation import MarineEcosystemSimulation, SimulationConfig
from .visualization import SimulationVisualizer
import matplotlib

matplotlib.use("Agg")


class MarineEcosystemInterface:
    """
    Streamlit-based web interface for the marine ecosystem simulation.

    Provides interactive controls for configuring and running simulations,
    real-time visualization of results, and comprehensive analysis tools.
    """

    def __init__(self):
        """Initialize the interface with session state management."""
        self.setup_page_config()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables to persist data across interactions."""
        # Initialize simulation in session state if not exists
        if "simulation" not in st.session_state:
            st.session_state.simulation = None

        # Initialize visualizer in session state if not exists
        if "visualizer" not in st.session_state:
            st.session_state.visualizer = None

        # Set instance variables to session state values
        self.simulation = st.session_state.simulation
        self.visualizer = st.session_state.visualizer

    def setup_page_config(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Marine Ecosystem Simulation",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS for better styling
        st.markdown(
            """
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stAlert {
            margin: 1rem 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def run(self):
        """Main interface runner."""
        self.show_header()
        self.show_sidebar()
        self.show_main_content()

    def show_header(self):
        """Display the main header."""
        st.markdown(
            '<h1 class="main-header">🌊 Marine Ecosystem Simulation</h1>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        This interactive simulation models marine ecosystem dynamics under various environmental conditions.
        Use the sidebar to configure simulation parameters and run experiments.
        """
        )

    def show_sidebar(self):
        """Display the sidebar with controls."""
        st.sidebar.title("Simulation Controls")

        # Simulation Configuration
        st.sidebar.subheader("🔧 Configuration")

        # Environment parameters
        st.sidebar.write("**Environment Settings**")
        width = st.sidebar.slider("Environment Width", 20, 100, 50)
        height = st.sidebar.slider("Environment Height (Depth)", 15, 50, 30)

        # Initial conditions
        st.sidebar.write("**Initial Conditions**")
        initial_temp = st.sidebar.slider("Initial Temperature (°C)", 5, 30, 20)
        initial_ph = st.sidebar.slider("Initial pH", 6.5, 9.0, 8.1, 0.1)
        initial_oxygen = st.sidebar.slider("Initial Oxygen (mg/L)", 2, 12, 7)

        # Population settings
        st.sidebar.write("**Initial Population**")
        phyto_count = st.sidebar.slider(
            "Phytoplankton", 100, 2000, 1500
        )  # Higher default
        zoo_count = st.sidebar.slider("Zooplankton", 50, 1000, 300)  # Lower default
        fish_count = st.sidebar.slider("Fish", 10, 500, 40)  # Much lower default

        # Simulation parameters
        st.sidebar.write("**Simulation Settings**")
        max_steps = st.sidebar.slider("Maximum Steps", 100, 2000, 500)
        climate_scenario = st.sidebar.selectbox(
            "Climate Scenario",
            ["stable", "warming", "extreme"],
            help="Select the climate change scenario to simulate",
        )

        # Random seed for reproducibility
        use_seed = st.sidebar.checkbox("Use Random Seed (Reproducible)")
        random_seed = None
        if use_seed:
            random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

        # Create configuration
        config = SimulationConfig(
            width=width,
            height=height,
            initial_temperature=initial_temp,
            initial_ph=initial_ph,
            initial_oxygen=initial_oxygen,
            initial_phytoplankton=phyto_count,
            initial_zooplankton=zoo_count,
            initial_fish=fish_count,
            max_steps=max_steps,
            climate_scenario=climate_scenario,
            random_seed=random_seed,
        )

        # Run simulation button
        st.sidebar.subheader("🚀 Simulation Control")

        if st.sidebar.button(
            "🆕 New Simulation", type="primary", use_container_width=True
        ):
            self.create_new_simulation(config)

        if st.sidebar.button("▶️ Run Simulation", use_container_width=True):
            if st.session_state.simulation is None:
                self.create_new_simulation(config)
            self.run_simulation()

        # Reset button
        if st.sidebar.button("🔄 Reset", use_container_width=True):
            self.reset_simulation()

        # Scenario comparison
        st.sidebar.subheader("📊 Analysis Tools")
        if st.sidebar.button("Compare Scenarios", use_container_width=True):
            self.run_scenario_comparison()

        if st.sidebar.button("🎯 Spatial Animation", use_container_width=True):
            self._create_spatial_animation(st.session_state.simulation)

    def show_main_content(self):
        """Display the main content area."""

        if st.session_state.simulation is None:
            self.show_welcome_screen()
        elif (
            hasattr(st.session_state.simulation, "detailed_stats")
            and len(st.session_state.simulation.detailed_stats) > 0
        ):
            # Simulation has run - show results
            self.show_simulation_dashboard()
        else:
            # Simulation configured but not run
            self.show_simulation_dashboard()

    def show_welcome_screen(self):
        """Display welcome screen when no simulation is loaded."""
        st.info(
            "👈 Configure your simulation parameters in the sidebar and click 'New Simulation' to begin!"
        )

        # Show example images or information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🦠 Phytoplankton")
            st.write(
                "Primary producers that form the base of the marine food web. Sensitive to light, temperature, and pH."
            )

        with col2:
            st.subheader("🦐 Zooplankton")
            st.write(
                "Primary consumers that feed on phytoplankton. Mobile organisms affected by temperature and oxygen."
            )

        with col3:
            st.subheader("🐟 Fish")
            st.write(
                "Top predators that feed on zooplankton. Most sensitive to environmental changes."
            )

        st.subheader("🌡️ Climate Scenarios")

        scenario_info = {
            "Stable": "No climate change - baseline conditions remain constant",
            "Warming": "Gradual warming, acidification, and deoxygenation",
            "Extreme": "Rapid environmental changes representing worst-case scenarios",
        }

        for scenario, description in scenario_info.items():
            st.write(f"**{scenario}**: {description}")

    def show_simulation_dashboard(self):
        """Display the main simulation dashboard."""

        # Current simulation info
        config = st.session_state.simulation.config
        st.subheader(
            f"🔬 Current Simulation: {config.climate_scenario.title()} Scenario"
        )

        # Check if simulation has been run
        has_results = (
            hasattr(st.session_state.simulation, "detailed_stats")
            and len(st.session_state.simulation.detailed_stats) > 0
        )

        if has_results:
            # Show results
            final_stats = st.session_state.simulation.detailed_stats[-1]
            st.success(
                f"✅ Simulation completed! {len(st.session_state.simulation.detailed_stats)} steps run. "
                f"Final population: {final_stats['total_agents']} organisms"
            )

            # Always show current status and visualizations when we have results
            self.show_current_status()
            self.show_visualizations()
        else:
            st.info(
                "Simulation configured but not yet run. Click 'Run Simulation' to start!"
            )
            self.show_configuration_summary()

    def show_configuration_summary(self):
        """Show summary of current configuration."""
        config = st.session_state.simulation.config

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Environment Size", f"{config.width} × {config.height}")
            st.metric("Climate Scenario", config.climate_scenario.title())

        with col2:
            st.metric("Initial Temperature", f"{config.initial_temperature}°C")
            st.metric("Initial pH", f"{config.initial_ph}")
            st.metric("Initial Oxygen", f"{config.initial_oxygen} mg/L")

        with col3:
            st.metric("Phytoplankton", config.initial_phytoplankton)
            st.metric("Zooplankton", config.initial_zooplankton)
            st.metric("Fish", config.initial_fish)

    def show_current_status(self):
        """Display current simulation status and key metrics."""
        latest_stats = st.session_state.simulation.detailed_stats[-1]

        # Key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Current Step",
                latest_stats["step"],
                help="Current simulation time step",
            )

        with col2:
            st.metric(
                "Total Organisms",
                latest_stats["total_agents"],
                help="Total number of living organisms",
            )

        with col3:
            st.metric(
                "Diversity Index",
                f"{latest_stats['shannon_diversity']:.3f}",
                help="Shannon diversity index (higher = more diverse)",
            )

        with col4:
            st.metric(
                "Total Biomass",
                f"{latest_stats['total_biomass']:.0f}",
                help="Total energy across all organisms",
            )

        with col5:
            avg_stress = latest_stats.get("average_stress_level", 0)
            st.metric(
                "Avg Stress",
                f"{avg_stress:.3f}",
                help="Average environmental stress (0-1, lower is better)",
            )

        # Population breakdown
        st.subheader("🔢 Population Breakdown")

        col1, col2, col3 = st.columns(3)

        with col1:
            delta_phyto = self.calculate_population_change("phytoplankton_count")
            st.metric(
                "🦠 Phytoplankton",
                latest_stats["phytoplankton_count"],
                delta=delta_phyto if delta_phyto is not None else None,
                help="Current population (arrow shows total change from start)",
            )

        with col2:
            delta_zoo = self.calculate_population_change("zooplankton_count")
            st.metric(
                "🦐 Zooplankton",
                latest_stats["zooplankton_count"],
                delta=delta_zoo if delta_zoo is not None else None,
                help="Current population (arrow shows total change from start)",
            )

        with col3:
            delta_fish = self.calculate_population_change("fish_count")
            st.metric(
                "🐟 Fish",
                latest_stats["fish_count"],
                delta=delta_fish if delta_fish is not None else None,
                help="Current population (arrow shows total change from start)",
            )

        # Warnings for critical conditions
        self.show_warnings(latest_stats)

    def calculate_population_change(self, species: str) -> int:
        """Calculate population change from first step to final step."""
        if len(st.session_state.simulation.detailed_stats) < 2:
            return None  # Return None for insufficient data

        initial = st.session_state.simulation.detailed_stats[0][species]  # First step
        current = st.session_state.simulation.detailed_stats[-1][species]  # Final step
        return current - initial

    def show_warnings(self, stats: Dict[str, Any]):
        """Display warnings for critical conditions."""
        warnings = []

        if stats["fish_count"] == 0:
            warnings.append("🚨 **CRITICAL**: Fish population extinct!")
        elif stats["fish_count"] < 10:
            warnings.append("⚠️ **WARNING**: Fish population critically low!")

        if stats["total_agents"] < 100:
            warnings.append("⚠️ **WARNING**: Total ecosystem population very low!")

        # Use the calculated average_stress_level from the stats
        avg_stress = stats.get("average_stress_level", 0)
        if avg_stress > 0.8:
            warnings.append("⚠️ **WARNING**: High environmental stress detected!")

        for warning in warnings:
            st.warning(warning)

    def show_visualizations(self):
        """Display simulation visualizations."""

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Population Dynamics",
                "🌡️ Environment",
                "🗺️ Spatial Distribution",
                "📊 Analysis",
            ]
        )

        with tab1:
            self.show_population_dynamics()

        with tab2:
            self.show_environmental_conditions()

        with tab3:
            self.show_spatial_distribution()

        with tab4:
            self.show_detailed_analysis()

    def show_population_dynamics(self):
        """Display population dynamics visualizations."""
        results_df = st.session_state.simulation.get_results_dataframe()

        # Interactive population plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["phytoplankton_count"],
                name="Phytoplankton",
                line=dict(color="#2E8B57", width=3),
                hovertemplate="Step: %{x}<br>Count: %{y}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["zooplankton_count"],
                name="Zooplankton",
                line=dict(color="#4169E1", width=3),
                hovertemplate="Step: %{x}<br>Count: %{y}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["fish_count"],
                name="Fish",
                line=dict(color="#DC143C", width=3),
                hovertemplate="Step: %{x}<br>Count: %{y}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Population Dynamics Over Time",
            xaxis_title="Time Steps",
            yaxis_title="Population Count",
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True, key="population_dynamics_chart")

        # Energy levels plot
        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["phyto_avg_energy"],
                name="Phytoplankton",
                line=dict(color="#2E8B57", dash="dot", width=2),
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["zoo_avg_energy"],
                name="Zooplankton",
                line=dict(color="#4169E1", dash="dot", width=2),
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df["fish_avg_energy"],
                name="Fish",
                line=dict(color="#DC143C", dash="dot", width=2),
            )
        )

        fig2.update_layout(
            title="Average Energy Levels",
            xaxis_title="Time Steps",
            yaxis_title="Average Energy",
            height=400,
        )

        st.plotly_chart(fig2, use_container_width=True, key="energy_levels_chart")

    def show_environmental_conditions(self):
        """Display environmental condition visualizations."""
        env_df = st.session_state.simulation.get_environmental_dataframe()

        if env_df.empty:
            st.warning("No environmental data available yet.")
            return

        # Environmental parameters plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=env_df["time"],
                y=env_df["avg_temperature"],
                name="Temperature (°C)",
                line=dict(color="#FF6347", width=3),
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=env_df["time"],
                y=env_df["avg_ph"],
                name="pH",
                line=dict(color="#9370DB", width=3),
                yaxis="y2",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=env_df["time"],
                y=env_df["avg_oxygen"],
                name="Oxygen (mg/L)",
                line=dict(color="#00CED1", width=3),
                yaxis="y3",
            )
        )

        fig.update_layout(
            title="Environmental Conditions Over Time",
            xaxis_title="Time Steps",
            height=500,
            yaxis=dict(title="Temperature (°C)", side="left"),
            yaxis2=dict(title="pH", side="right", overlaying="y"),
            yaxis3=dict(
                title="Oxygen (mg/L)", side="right", overlaying="y", position=0.85
            ),
        )

        st.plotly_chart(
            fig, use_container_width=True, key="environmental_conditions_chart"
        )

        # Environmental heatmaps
        st.subheader("🗺️ Environmental Distribution")

        col1, col2, col3 = st.columns(3)

        env = st.session_state.simulation.environment

        with col1:
            fig_temp = go.Figure(
                data=go.Heatmap(
                    z=env.temperature.T, colorscale="RdYlBu_r", showscale=True
                )
            )
            fig_temp.update_layout(title="Temperature Distribution", height=300)
            st.plotly_chart(
                fig_temp, use_container_width=True, key="temperature_heatmap"
            )

        with col2:
            fig_ph = go.Figure(
                data=go.Heatmap(z=env.ph.T, colorscale="viridis", showscale=True)
            )
            fig_ph.update_layout(title="pH Distribution", height=300)
            st.plotly_chart(fig_ph, use_container_width=True, key="ph_heatmap")

        with col3:
            fig_oxygen = go.Figure(
                data=go.Heatmap(z=env.oxygen.T, colorscale="Blues", showscale=True)
            )
            fig_oxygen.update_layout(title="Oxygen Distribution", height=300)
            st.plotly_chart(fig_oxygen, use_container_width=True, key="oxygen_heatmap")

    def show_spatial_distribution(self):
        """Display spatial distribution of organisms."""
        distribution = st.session_state.simulation.get_agent_spatial_distribution()

        # Create spatial plot
        fig = go.Figure()

        # Add each species
        if distribution["phytoplankton"]:
            phyto_x, phyto_y = zip(*distribution["phytoplankton"])
            fig.add_trace(
                go.Scatter(
                    x=phyto_x,
                    y=phyto_y,
                    mode="markers",
                    name="Phytoplankton",
                    marker=dict(color="#2E8B57", size=6, opacity=0.7),
                    hovertemplate="Phytoplankton<br>X: %{x}<br>Y: %{y}<extra></extra>",
                )
            )

        if distribution["zooplankton"]:
            zoo_x, zoo_y = zip(*distribution["zooplankton"])
            fig.add_trace(
                go.Scatter(
                    x=zoo_x,
                    y=zoo_y,
                    mode="markers",
                    name="Zooplankton",
                    marker=dict(color="#4169E1", size=8, opacity=0.8),
                    hovertemplate="Zooplankton<br>X: %{x}<br>Y: %{y}<extra></extra>",
                )
            )

        if distribution["fish"]:
            fish_x, fish_y = zip(*distribution["fish"])
            fig.add_trace(
                go.Scatter(
                    x=fish_x,
                    y=fish_y,
                    mode="markers",
                    name="Fish",
                    marker=dict(color="#DC143C", size=12, opacity=0.9),
                    hovertemplate="Fish<br>X: %{x}<br>Y: %{y}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Current Spatial Distribution of Organisms",
            xaxis_title="X Position",
            yaxis_title="Y Position (Depth)",
            height=600,
            yaxis=dict(autorange="reversed"),  # Invert y-axis so surface is at top
        )

        st.plotly_chart(fig, use_container_width=True, key="spatial_distribution_chart")

        # Distribution statistics
        st.subheader("📊 Distribution Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Phytoplankton Locations", len(distribution["phytoplankton"]))
            if distribution["phytoplankton"]:
                avg_depth = np.mean([pos[1] for pos in distribution["phytoplankton"]])
                st.metric("Average Depth", f"{avg_depth:.1f}")

        with col2:
            st.metric("Zooplankton Locations", len(distribution["zooplankton"]))
            if distribution["zooplankton"]:
                avg_depth = np.mean([pos[1] for pos in distribution["zooplankton"]])
                st.metric("Average Depth", f"{avg_depth:.1f}")

        with col3:
            st.metric("Fish Locations", len(distribution["fish"]))
            if distribution["fish"]:
                avg_depth = np.mean([pos[1] for pos in distribution["fish"]])
                st.metric("Average Depth", f"{avg_depth:.1f}")

    def show_detailed_analysis(self):
        """Display detailed analysis and insights."""

        # Ecosystem health metrics
        health_metrics = st.session_state.simulation.get_ecosystem_health_metrics()

        st.subheader("🏥 Ecosystem Health Assessment")

        col1, col2 = st.columns(2)

        with col1:
            # Resilience score with color coding
            resilience = health_metrics.get("ecosystem_resilience", 0)
            if resilience > 0.7:
                color = "green"
                status = "Excellent"
            elif resilience > 0.5:
                color = "orange"
                status = "Good"
            elif resilience > 0.3:
                color = "orange"
                status = "Moderate"
            else:
                color = "red"
                status = "Poor"

            st.markdown(
                f"**Ecosystem Resilience**: :{color}[{resilience:.3f}] ({status})"
            )

            st.metric(
                "Current Diversity", f"{health_metrics.get('current_diversity', 0):.3f}"
            )
            st.metric(
                "Population Stability (Fish)",
                f"{health_metrics.get('fish_population_stability', 0):.3f}",
            )

        with col2:
            st.metric(
                "Current Biomass", f"{health_metrics.get('current_biomass', 0):.0f}"
            )
            st.metric(
                "Average Stress Level",
                f"{health_metrics.get('average_stress_level', 0):.3f}",
            )
            st.metric(
                "Population Stability (Zoo)",
                f"{health_metrics.get('zoo_population_stability', 0):.3f}",
            )

        # Trend analysis
        st.subheader("📈 Trend Analysis")

        results_df = st.session_state.simulation.get_results_dataframe()

        if len(results_df) > 10:
            # Calculate trends for last 20% of simulation
            recent_steps = int(len(results_df) * 0.2)
            recent_data = results_df.tail(recent_steps)

            trends = {}
            for column in [
                "phytoplankton_count",
                "zooplankton_count",
                "fish_count",
                "shannon_diversity",
            ]:
                if len(recent_data) > 1:
                    # Simple linear trend calculation
                    x = np.arange(len(recent_data))
                    y = recent_data[column].values
                    trend = np.polyfit(x, y, 1)[0]  # Slope of linear fit
                    trends[column] = trend

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Population Trends (Recent)**")
                for species in [
                    "phytoplankton_count",
                    "zooplankton_count",
                    "fish_count",
                ]:
                    trend = trends.get(species, 0)
                    species_name = species.replace("_count", "").title()

                    if trend > 0.1:
                        st.success(f"{species_name}: ↗️ Increasing ({trend:+.2f}/step)")
                    elif trend < -0.1:
                        st.error(f"{species_name}: ↘️ Decreasing ({trend:+.2f}/step)")
                    else:
                        st.info(f"{species_name}: ➡️ Stable ({trend:+.2f}/step)")

            with col2:
                st.write("**Ecosystem Trends**")
                diversity_trend = trends.get("shannon_diversity", 0)

                if diversity_trend > 0.001:
                    st.success(f"Diversity: ↗️ Increasing ({diversity_trend:+.4f}/step)")
                elif diversity_trend < -0.001:
                    st.error(f"Diversity: ↘️ Decreasing ({diversity_trend:+.4f}/step)")
                else:
                    st.info(f"Diversity: ➡️ Stable ({diversity_trend:+.4f}/step)")

        # Export options
        st.subheader("💾 Export Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            results_df = st.session_state.simulation.get_results_dataframe()
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📊 Download Results CSV",
                csv_data,
                "simulation_results.csv",
                "text/csv",
                key="download_results_csv",
            )

        with col2:
            env_df = st.session_state.simulation.get_environmental_dataframe()
            env_csv_data = env_df.to_csv(index=False)
            st.download_button(
                "🌡️ Download Environment CSV",
                env_csv_data,
                "environmental_data.csv",
                "text/csv",
                key="download_env_csv",
            )

        with col3:
            if st.session_state.visualizer is None:
                st.session_state.visualizer = SimulationVisualizer(
                    st.session_state.simulation
                )

            report = st.session_state.visualizer.generate_summary_report()
            st.download_button(
                "📋 Generate Report",
                report,
                "simulation_report.txt",
                "text/plain",
                key="download_report",
            )

    def create_new_simulation(self, config: SimulationConfig):
        """Create a new simulation with the given configuration."""
        # Create simulation and store in session state
        st.session_state.simulation = MarineEcosystemSimulation(config)
        st.session_state.visualizer = None

        # Update instance variables
        self.simulation = st.session_state.simulation
        self.visualizer = st.session_state.visualizer

        st.success("✅ New simulation created!")

    def run_simulation(self):
        """Run the current simulation."""
        if st.session_state.simulation is None:
            st.error("No simulation configured. Please create a new simulation first.")
            return

        # Use simulation from session state
        simulation = st.session_state.simulation

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        try:
            # Run simulation with progress updates
            total_steps = simulation.config.max_steps

            for step in range(total_steps):
                simulation.run_step()

                # Update progress
                progress = (step + 1) / total_steps
                progress_bar.progress(progress)

                # Update status every 50 steps
                if (step + 1) % 50 == 0:
                    current_stats = simulation.detailed_stats[-1]
                    status_text.text(
                        f"Step {step + 1}/{total_steps} - "
                        f"Agents: {current_stats['total_agents']} - "
                        f"Diversity: {current_stats['shannon_diversity']:.3f}"
                    )

                # Check for extinction
                if simulation.detailed_stats[-1]["total_agents"] == 0:
                    status_text.text(f"Ecosystem collapse at step {step + 1}")
                    break

            # Simulation completed
            end_time = time.time()
            duration = end_time - start_time

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Create visualizer and store in session state
            st.session_state.visualizer = SimulationVisualizer(simulation)
            self.visualizer = st.session_state.visualizer

            st.success("🎉 Simulation completed successfully!")
            st.balloons()

            final_stats = simulation.detailed_stats[-1]
            st.info(
                f"⏱️ Completed in {duration:.1f} seconds | "
                f"👥 Final population: {final_stats['total_agents']} | "
                f"🌊 Diversity: {final_stats['shannon_diversity']:.3f}"
            )

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")

    def reset_simulation(self):
        """Reset the current simulation."""
        if st.session_state.simulation is not None:
            st.session_state.simulation.reset_simulation()
            st.session_state.visualizer = None

            # Update instance variables
            self.simulation = st.session_state.simulation
            self.visualizer = st.session_state.visualizer

            st.success("🔄 Simulation reset!")

    def _create_spatial_animation(self, simulation):
        """Create spatial animation showing agent movement over time."""
        st.write("**🌍 Creating Spatial Movement Animation...**")

        try:
            df = simulation.get_agent_spatial_dataframe()

            if df is None or df.empty:
                st.error("No spatial data available.")
                return

            required_cols = {"step", "type", "x", "y"}
            if not required_cols.issubset(df.columns):
                st.error("Data must contain columns: step, type, x, y")
                return

            # Progress bar
            progress = st.progress(0.0)
            progress.progress(0.1)

            steps = sorted(df["step"].unique())
            max_x = simulation.environment.width
            max_y = simulation.environment.height
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, max_x)
            ax.set_ylim(0, max_y)
            ax.invert_yaxis()  # Invert y-axis so surface is at top
            ax.set_title("Marine Ecosystem Spatial Animation")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position (Depth)")

            # Set up initial plot styling
            ax.grid(True, alpha=0.3)

            def update(step):
                ax.clear()  # Clear the previous frame

                # Reset the plot properties
                ax.set_xlim(0, max_x)
                ax.set_ylim(0, max_y)
                ax.invert_yaxis()  # Re-invert after clearing

                # Custom Y-axis labels: show 0 at bottom, max_y at top (but keep visual top as surface)
                # Force the full range regardless of data
                full_range_ticks = np.arange(
                    0, max_y + 1, 5
                )  # Every 5 units: 0, 5, 10, 15, 20, 25, 30
                full_range_labels = [
                    f"{int(max_y - tick)}" for tick in full_range_ticks
                ]  # Reverse: 30, 25, 20, 15, 10, 5, 0
                ax.set_yticks(full_range_ticks)
                ax.set_yticklabels(full_range_labels)

                ax.set_title(f"Marine Ecosystem - Step {step}")
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position (Depth)")
                ax.grid(True, alpha=0.3)

                # Get data for current step
                step_data = df[df["step"] == step]

                # Plot each organism type with proper styling
                phyto_data = step_data[step_data["type"] == "phytoplankton"]
                if not phyto_data.empty:
                    ax.scatter(
                        phyto_data["x"],
                        phyto_data["y"],
                        c="#2E8B57",
                        s=30,
                        alpha=0.7,
                        label="🦠 Phytoplankton",
                    )

                zoo_data = step_data[step_data["type"] == "zooplankton"]
                if not zoo_data.empty:
                    ax.scatter(
                        zoo_data["x"],
                        zoo_data["y"],
                        c="#4169E1",
                        s=40,
                        alpha=0.8,
                        label="🦐 Zooplankton",
                    )

                fish_data = step_data[step_data["type"] == "fish"]
                if not fish_data.empty:
                    ax.scatter(
                        fish_data["x"],
                        fish_data["y"],
                        c="#DC143C",
                        s=60,
                        alpha=0.9,
                        label="🐟 Fish",
                    )

                ax.legend()

                return []  # Return empty list since we're using ax.clear()

            anim = FuncAnimation(fig, update, frames=steps, interval=300, blit=False)

            progress.progress(0.7)

            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                anim.save(tmpfile.name, writer="pillow", fps=3)
                gif_path = tmpfile.name

            with open(gif_path, "rb") as f:
                gif_data = f.read()

            os.remove(gif_path)
            plt.close(fig)

            progress.progress(1.0)

            st.image(
                gif_data, caption="Spatial Agent Movement", use_container_width=True
            )
            st.download_button(
                label="📥 Download Spatial Animation (GIF)",
                data=gif_data,
                file_name="spatial_movement.gif",
                mime="image/gif",
            )

            st.success("✅ Spatial animation created!")

        except Exception as e:
            st.error(f"Failed to create spatial animation: {e}")

    def run_scenario_comparison(self):
        """Run and compare multiple climate scenarios."""
        if st.session_state.simulation is None:
            st.error("No simulation configured. Please create a new simulation first.")
            return

        st.subheader("🔬 Climate Scenario Comparison")

        scenarios = ["stable", "warming", "extreme"]
        comparison_steps = st.slider("Steps per scenario", 100, 1000, 300)

        if st.button("🚀 Run Comparison"):
            results = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, scenario in enumerate(scenarios):
                status_text.text(f"Running {scenario} scenario...")

                # Reset and configure for scenario
                st.session_state.simulation.reset_simulation()
                st.session_state.simulation.config.climate_scenario = scenario
                st.session_state.simulation.config.max_steps = comparison_steps

                # Run simulation
                st.session_state.simulation.run_simulation(
                    steps=comparison_steps, verbose=False
                )
                results[scenario] = st.session_state.simulation.get_results_dataframe()

                progress_bar.progress((i + 1) / len(scenarios))

            status_text.text("Generating comparison plots...")

            # Create comparison visualization
            fig = go.Figure()

            colors = {"stable": "green", "warming": "orange", "extreme": "red"}

            for scenario, df in results.items():
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df["fish_count"],
                        name=f"Fish - {scenario.title()}",
                        line=dict(color=colors[scenario], width=3),
                    )
                )

            fig.update_layout(
                title="Fish Population Under Different Climate Scenarios",
                xaxis_title="Time Steps",
                yaxis_title="Fish Count",
                height=500,
            )

            st.plotly_chart(
                fig, use_container_width=True, key="scenario_comparison_chart"
            )

            # Summary statistics
            st.subheader("📊 Scenario Summary")

            summary_data = []
            for scenario, df in results.items():
                final_stats = df.iloc[-1]
                summary_data.append(
                    {
                        "Scenario": scenario.title(),
                        "Final Fish Count": final_stats["fish_count"],
                        "Final Total Population": final_stats["total_agents"],
                        "Final Diversity": f"{final_stats['shannon_diversity']:.3f}",
                        "Final Biomass": f"{final_stats['total_biomass']:.0f}",
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)

            status_text.text("Comparison completed!")
            progress_bar.progress(1.0)


def main():
    """Main function to run the Streamlit interface."""
    interface = MarineEcosystemInterface()
    interface.run()


if __name__ == "__main__":
    main()
