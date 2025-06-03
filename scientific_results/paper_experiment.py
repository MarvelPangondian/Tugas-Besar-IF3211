#!/usr/bin/env python3
"""
Comprehensive Experiments for IEEE Paper Results
Runs all necessary experiments to generate Results and Discussion section.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulation import MarineEcosystemSimulation, SimulationConfig
from src.visualization import SimulationVisualizer


def experiment_1_climate_scenarios():
    """
    Experiment 1: Compare population dynamics under different climate scenarios
    This replicates your stress gradient test but with longer simulations
    """
    print("Experiment 1: Climate Scenario Comparison")
    print("=" * 60)

    scenarios = ["stable", "warming", "extreme"]
    results = {}

    base_config = {
        "width": 50,
        "height": 30,
        "initial_phytoplankton": 800,
        "initial_zooplankton": 400,
        "initial_fish": 100,
        "max_steps": 500,  # Longer for detailed analysis
        "random_seed": 42,
    }

    for scenario in scenarios:
        print(f"\nRunning {scenario} scenario...")

        config = SimulationConfig(**base_config, climate_scenario=scenario)

        simulation = MarineEcosystemSimulation(config)
        df = simulation.run_simulation(verbose=False)

        results[scenario] = {
            "dataframe": df,
            "simulation": simulation,
            "final_stats": df.iloc[-1] if not df.empty else None,
        }

        if not df.empty:
            final = df.iloc[-1]
            print(
                f"  Final populations: Phyto={final['phytoplankton_count']}, "
                f"Zoo={final['zooplankton_count']}, Fish={final['fish_count']}"
            )
            print(f"  Diversity: {final['shannon_diversity']:.3f}")

    return results


def experiment_2_stress_thresholds():
    """
    Experiment 2: Identify critical environmental thresholds
    """
    print("\n🧪 Experiment 2: Critical Threshold Identification")
    print("=" * 60)

    base_config = {
        "width": 40,
        "height": 25,
        "initial_phytoplankton": 600,
        "initial_zooplankton": 300,
        "initial_fish": 80,
        "max_steps": 200,
        "climate_scenario": "stable",
        "random_seed": 42,
    }

    # Test temperature thresholds
    print("\Testing Temperature Thresholds...")
    temp_results = []
    temperatures = np.arange(18, 32, 1)  # 18C to 32C

    for temp in temperatures:
        config = SimulationConfig(**base_config, initial_temperature=temp)
        simulation = MarineEcosystemSimulation(config)
        df = simulation.run_simulation(verbose=False)

        if not df.empty:
            final = df.iloc[-1]
            initial = df.iloc[0]

            # Calculate survival rates
            fish_survival = (
                final["fish_count"] / initial["fish_count"]
                if initial["fish_count"] > 0
                else 0
            )
            zoo_survival = (
                final["zooplankton_count"] / initial["zooplankton_count"]
                if initial["zooplankton_count"] > 0
                else 0
            )
            phyto_survival = (
                final["phytoplankton_count"] / initial["phytoplankton_count"]
                if initial["phytoplankton_count"] > 0
                else 0
            )

            temp_results.append(
                {
                    "temperature": temp,
                    "fish_survival": fish_survival,
                    "zoo_survival": zoo_survival,
                    "phyto_survival": phyto_survival,
                    "total_survival": final["total_agents"] / initial["total_agents"],
                    "diversity": final["shannon_diversity"],
                }
            )

    # Test pH thresholds
    print("\nTesting pH Thresholds...")
    ph_results = []
    ph_values = np.arange(6.8, 8.4, 0.1)

    for ph in ph_values:
        config = SimulationConfig(**base_config, initial_ph=ph)
        simulation = MarineEcosystemSimulation(config)
        df = simulation.run_simulation(verbose=False)

        if not df.empty:
            final = df.iloc[-1]
            initial = df.iloc[0]

            fish_survival = (
                final["fish_count"] / initial["fish_count"]
                if initial["fish_count"] > 0
                else 0
            )
            zoo_survival = (
                final["zooplankton_count"] / initial["zooplankton_count"]
                if initial["zooplankton_count"] > 0
                else 0
            )
            phyto_survival = (
                final["phytoplankton_count"] / initial["phytoplankton_count"]
                if initial["phytoplankton_count"] > 0
                else 0
            )

            ph_results.append(
                {
                    "ph": ph,
                    "fish_survival": fish_survival,
                    "zoo_survival": zoo_survival,
                    "phyto_survival": phyto_survival,
                    "total_survival": final["total_agents"] / initial["total_agents"],
                    "diversity": final["shannon_diversity"],
                }
            )

    # Test oxygen thresholds
    print("\nTesting Oxygen Thresholds...")
    oxygen_results = []
    oxygen_values = np.arange(2.0, 9.0, 0.5)

    for oxygen in oxygen_values:
        config = SimulationConfig(**base_config, initial_oxygen=oxygen)
        simulation = MarineEcosystemSimulation(config)
        df = simulation.run_simulation(verbose=False)

        if not df.empty:
            final = df.iloc[-1]
            initial = df.iloc[0]

            fish_survival = (
                final["fish_count"] / initial["fish_count"]
                if initial["fish_count"] > 0
                else 0
            )
            zoo_survival = (
                final["zooplankton_count"] / initial["zooplankton_count"]
                if initial["zooplankton_count"] > 0
                else 0
            )
            phyto_survival = (
                final["phytoplankton_count"] / initial["phytoplankton_count"]
                if initial["phytoplankton_count"] > 0
                else 0
            )

            oxygen_results.append(
                {
                    "oxygen": oxygen,
                    "fish_survival": fish_survival,
                    "zoo_survival": zoo_survival,
                    "phyto_survival": phyto_survival,
                    "total_survival": final["total_agents"] / initial["total_agents"],
                    "diversity": final["shannon_diversity"],
                }
            )

    return {
        "temperature": pd.DataFrame(temp_results),
        "ph": pd.DataFrame(ph_results),
        "oxygen": pd.DataFrame(oxygen_results),
    }


def experiment_3_spatial_analysis():
    """
    Experiment 3: Spatial distribution analysis under stress
    """
    print("\n🧪 Experiment 3: Spatial Distribution Analysis")
    print("=" * 60)

    scenarios = [
        {"name": "control", "temp": 20, "ph": 8.1, "oxygen": 7.0},
        {"name": "warming", "temp": 25, "ph": 8.1, "oxygen": 7.0},
        {"name": "acidic", "temp": 20, "ph": 7.5, "oxygen": 7.0},
        {"name": "hypoxic", "temp": 20, "ph": 8.1, "oxygen": 4.0},
    ]

    spatial_results = {}

    base_config = {
        "width": 50,
        "height": 30,
        "initial_phytoplankton": 800,
        "initial_zooplankton": 400,
        "initial_fish": 100,
        "max_steps": 150,
        "climate_scenario": "stable",
        "random_seed": 42,
    }

    for scenario in scenarios:
        print(f"\nAnalyzing {scenario['name']} spatial distribution...")

        config = SimulationConfig(
            **base_config,
            initial_temperature=scenario["temp"],
            initial_ph=scenario["ph"],
            initial_oxygen=scenario["oxygen"],
        )

        simulation = MarineEcosystemSimulation(config)
        df = simulation.run_simulation(verbose=False)

        if not df.empty:
            # Get final spatial distribution
            final_distribution = simulation.get_agent_spatial_distribution()

            # Calculate depth statistics
            depth_stats = {}
            for species, positions in final_distribution.items():
                if positions:
                    depths = [pos[1] for pos in positions]
                    depth_stats[species] = {
                        "count": len(positions),
                        "avg_depth": np.mean(depths),
                        "depth_std": np.std(depths),
                        "min_depth": min(depths),
                        "max_depth": max(depths),
                    }
                else:
                    depth_stats[species] = {
                        "count": 0,
                        "avg_depth": 0,
                        "depth_std": 0,
                        "min_depth": 0,
                        "max_depth": 0,
                    }

            spatial_results[scenario["name"]] = {
                "distribution": final_distribution,
                "depth_stats": depth_stats,
                "dataframe": df,
            }

    return spatial_results


def experiment_4_vulnerability_ranking():
    """
    Experiment 4: Systematic vulnerability ranking (your isolated test)
    """
    print("\nExperiment 4: Vulnerability Ranking Analysis")
    print("=" * 60)

    # This is your existing isolated stress test - run it to get the data
    from scientific_results.stress_test import isolated_trophic_stress_test, analyze_direct_vulnerability

    results = isolated_trophic_stress_test()
    vulnerability_analysis = analyze_direct_vulnerability(results)

    return {"raw_results": results, "analysis": vulnerability_analysis}


def generate_paper_plots(experiment_results):
    """
    Generate all plots needed for the paper
    """
    print("\nGenerating Paper Figures...")
    print("=" * 60)

    # Figure 1: Climate scenario comparison
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(
        "Population Dynamics Under Different Climate Scenarios",
        fontsize=14,
        fontweight="bold",
    )

    colors = {
        "phytoplankton_count": "#2E8B57",
        "zooplankton_count": "#4169E1",
        "fish_count": "#DC143C",
    }

    # Plot population dynamics for each scenario
    for i, (scenario, data) in enumerate(
        experiment_results["climate_scenarios"].items()
    ):
        df = data["dataframe"]
        if df.empty:
            continue

        ax = axes[i // 2, i % 2] if i < 3 else axes[1, 1]

        ax.plot(
            df["step"],
            df["phytoplankton_count"],
            color=colors["phytoplankton_count"],
            linewidth=2,
            label="Phytoplankton",
        )
        ax.plot(
            df["step"],
            df["zooplankton_count"],
            color=colors["zooplankton_count"],
            linewidth=2,
            label="Zooplankton",
        )
        ax.plot(
            df["step"],
            df["fish_count"],
            color=colors["fish_count"],
            linewidth=2,
            label="Fish",
        )

        ax.set_title(f"{scenario.title()} Scenario")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Population Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    if len(experiment_results["climate_scenarios"]) < 4:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig("paper_figure1_scenarios.png", dpi=300, bbox_inches="tight")
    print("Saved: paper_figure1_scenarios.png")

    # Figure 2: Critical thresholds
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle("Environmental Stress Thresholds", fontsize=14, fontweight="bold")

    threshold_data = experiment_results["thresholds"]

    # Temperature thresholds
    temp_df = threshold_data["temperature"]
    axes[0].plot(
        temp_df["temperature"],
        temp_df["fish_survival"],
        "r-",
        linewidth=3,
        label="Fish",
    )
    axes[0].plot(
        temp_df["temperature"],
        temp_df["zoo_survival"],
        "b-",
        linewidth=3,
        label="Zooplankton",
    )
    axes[0].plot(
        temp_df["temperature"],
        temp_df["phyto_survival"],
        "g-",
        linewidth=3,
        label="Phytoplankton",
    )
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Survival Rate")
    axes[0].set_title("Temperature Thresholds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # pH thresholds
    ph_df = threshold_data["ph"]
    axes[1].plot(ph_df["ph"], ph_df["fish_survival"], "r-", linewidth=3, label="Fish")
    axes[1].plot(
        ph_df["ph"], ph_df["zoo_survival"], "b-", linewidth=3, label="Zooplankton"
    )
    axes[1].plot(
        ph_df["ph"], ph_df["phyto_survival"], "g-", linewidth=3, label="Phytoplankton"
    )
    axes[1].set_xlabel("pH")
    axes[1].set_ylabel("Survival Rate")
    axes[1].set_title("pH Thresholds")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Oxygen thresholds
    oxygen_df = threshold_data["oxygen"]
    axes[2].plot(
        oxygen_df["oxygen"], oxygen_df["fish_survival"], "r-", linewidth=3, label="Fish"
    )
    axes[2].plot(
        oxygen_df["oxygen"],
        oxygen_df["zoo_survival"],
        "b-",
        linewidth=3,
        label="Zooplankton",
    )
    axes[2].plot(
        oxygen_df["oxygen"],
        oxygen_df["phyto_survival"],
        "g-",
        linewidth=3,
        label="Phytoplankton",
    )
    axes[2].set_xlabel("Oxygen (mg/L)")
    axes[2].set_ylabel("Survival Rate")
    axes[2].set_title("Oxygen Thresholds")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper_figure2_thresholds.png", dpi=300, bbox_inches="tight")
    print("Saved: paper_figure2_thresholds.png")

    # Figure 3: Spatial distributions
    spatial_data = experiment_results["spatial"]
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle(
        "Spatial Distribution Under Environmental Stress",
        fontsize=14,
        fontweight="bold",
    )

    scenario_list = ["control", "warming", "acidic", "hypoxic"]
    for i, scenario in enumerate(scenario_list):
        if scenario not in spatial_data:
            continue

        ax = axes[i // 2, i % 2]
        distribution = spatial_data[scenario]["distribution"]

        # Plot each species
        if distribution["phytoplankton"]:
            phyto_x, phyto_y = zip(*distribution["phytoplankton"])
            ax.scatter(
                phyto_x, phyto_y, c="green", alpha=0.6, s=20, label="Phytoplankton"
            )

        if distribution["zooplankton"]:
            zoo_x, zoo_y = zip(*distribution["zooplankton"])
            ax.scatter(zoo_x, zoo_y, c="blue", alpha=0.7, s=30, label="Zooplankton")

        if distribution["fish"]:
            fish_x, fish_y = zip(*distribution["fish"])
            ax.scatter(fish_x, fish_y, c="red", alpha=0.8, s=50, label="Fish")

        ax.set_title(f"{scenario.title()} Conditions")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position (Depth)")
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper_figure3_spatial.png", dpi=300, bbox_inches="tight")
    print("📊 Saved: paper_figure3_spatial.png")

    plt.show()


def calculate_paper_statistics(experiment_results):
    """
    Calculate key statistics for the Results section
    """
    print("\n📈 Calculating Key Statistics...")
    print("=" * 60)

    stats = {}

    # Climate scenario statistics
    climate_data = experiment_results["climate_scenarios"]
    stats["climate"] = {}

    for scenario, data in climate_data.items():
        if data["final_stats"] is not None:
            final = data["final_stats"]
            initial = data["dataframe"].iloc[0]

            stats["climate"][scenario] = {
                "fish_decline": (initial["fish_count"] - final["fish_count"])
                / initial["fish_count"],
                "zoo_decline": (
                    initial["zooplankton_count"] - final["zooplankton_count"]
                )
                / initial["zooplankton_count"],
                "phyto_decline": (
                    initial["phytoplankton_count"] - final["phytoplankton_count"]
                )
                / initial["phytoplankton_count"],
                "diversity_final": final["shannon_diversity"],
                "total_survival": final["total_agents"] / initial["total_agents"],
            }

    # Critical threshold statistics
    threshold_data = experiment_results["thresholds"]
    stats["thresholds"] = {}

    # Find critical temperature (50% survival threshold)
    temp_df = threshold_data["temperature"]
    for species in ["fish", "zoo", "phyto"]:
        survival_col = f"{species}_survival"
        critical_temp = temp_df[temp_df[survival_col] <= 0.5]
        if not critical_temp.empty:
            stats["thresholds"][f"{species}_critical_temp"] = critical_temp[
                "temperature"
            ].iloc[0]
        else:
            stats["thresholds"][f"{species}_critical_temp"] = None

    # Similar for pH and oxygen
    ph_df = threshold_data["ph"]
    for species in ["fish", "zoo", "phyto"]:
        survival_col = f"{species}_survival"
        critical_ph = ph_df[ph_df[survival_col] <= 0.5]
        if not critical_ph.empty:
            stats["thresholds"][f"{species}_critical_ph"] = critical_ph["ph"].iloc[0]
        else:
            stats["thresholds"][f"{species}_critical_ph"] = None

    oxygen_df = threshold_data["oxygen"]
    for species in ["fish", "zoo", "phyto"]:
        survival_col = f"{species}_survival"
        critical_oxygen = oxygen_df[oxygen_df[survival_col] <= 0.5]
        if not critical_oxygen.empty:
            stats["thresholds"][f"{species}_critical_oxygen"] = critical_oxygen[
                "oxygen"
            ].iloc[0]
        else:
            stats["thresholds"][f"{species}_critical_oxygen"] = None

    return stats


def main():
    """
    Run all experiments needed for the IEEE paper
    """
    print("🔬 Marine Ecosystem Simulation - Paper Experiments")
    print("=" * 70)

    experiment_results = {}

    # Run all experiments
    experiment_results["climate_scenarios"] = experiment_1_climate_scenarios()
    experiment_results["thresholds"] = experiment_2_stress_thresholds()
    experiment_results["spatial"] = experiment_3_spatial_analysis()
    experiment_results["vulnerability"] = experiment_4_vulnerability_ranking()

    # Calculate statistics
    paper_stats = calculate_paper_statistics(experiment_results)

    # Generate plots
    generate_paper_plots(experiment_results)

    # Save results for analysis
    with open("paper_statistics.json", "w") as f:
        json.dump(paper_stats, f, indent=2, default=str)

    print("\nAll experiments completed!")
    print("Figures saved as paper_figure*.png")
    print("Statistics saved as paper_statistics.json")

    return experiment_results, paper_stats


if __name__ == "__main__":
    results, stats = main()
