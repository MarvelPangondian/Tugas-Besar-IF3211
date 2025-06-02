#!/usr/bin/env python3
"""
Marine Ecosystem Simulation - Main Entry Point

This script provides multiple ways to run the marine ecosystem simulation:
1. Command-line interface for batch simulations
2. Interactive Streamlit web interface
3. Programmatic API for research use

Usage:
    python main.py --help                    # Show help
    python main.py --web                     # Launch web interface
    python main.py --run                     # Run default simulation
    python main.py --compare                 # Compare climate scenarios
    python main.py --config config.json     # Use custom configuration
"""

import argparse
import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulation import MarineEcosystemSimulation, SimulationConfig
from src.visualization import SimulationVisualizer
from src.interface import main as run_web_interface


def run_default_simulation():
    """Run a default simulation with standard parameters."""
    print("🌊 Marine Ecosystem Simulation")
    print("=" * 50)

    # Create default configuration
    config = SimulationConfig(
        width=50,
        height=30,
        initial_phytoplankton=800,
        initial_zooplankton=400,
        initial_fish=100,
        max_steps=500,
        climate_scenario="stable",
        random_seed=42,
    )

    print(f"Configuration:")
    print(f"  Environment: {config.width}x{config.height}")
    print(f"  Climate: {config.climate_scenario}")
    print(
        f"  Initial populations: {config.initial_phytoplankton} phyto, {config.initial_zooplankton} zoo, {config.initial_fish} fish"
    )
    print(f"  Steps: {config.max_steps}")
    print()

    # Create and run simulation
    simulation = MarineEcosystemSimulation(config)
    print("Running simulation...")

    results_df = simulation.run_simulation(verbose=True)

    # Create visualizations
    print("\nGenerating visualizations...")
    visualizer = SimulationVisualizer(simulation)

    # Population dynamics plot
    fig1 = visualizer.plot_population_dynamics()
    fig1.savefig("population_dynamics.png", dpi=300, bbox_inches="tight")
    print("📊 Saved: population_dynamics.png")

    # Environmental conditions plot
    fig2 = visualizer.plot_environmental_conditions()
    fig2.savefig("environmental_conditions.png", dpi=300, bbox_inches="tight")
    print("🌡️ Saved: environmental_conditions.png")

    # Spatial distribution plot
    fig3 = visualizer.plot_spatial_distribution()
    fig3.savefig("spatial_distribution.png", dpi=300, bbox_inches="tight")
    print("🗺️ Saved: spatial_distribution.png")

    # Generate summary report
    report = visualizer.generate_summary_report()
    with open("simulation_report.txt", "w") as f:
        f.write(report)
    print("📋 Saved: simulation_report.txt")

    # Save data
    results_df.to_csv("simulation_results.csv", index=False)
    print("💾 Saved: simulation_results.csv")

    env_df = simulation.get_environmental_dataframe()
    env_df.to_csv("environmental_data.csv", index=False)
    print("💾 Saved: environmental_data.csv")

    print("\n✅ Simulation completed successfully!")
    print(f"Final population: {results_df.iloc[-1]['total_agents']} organisms")
    print(f"Final diversity: {results_df.iloc[-1]['shannon_diversity']:.3f}")

    plt.show()


def compare_climate_scenarios():
    """Compare different climate scenarios."""
    print("🔬 Climate Scenario Comparison")
    print("=" * 50)

    scenarios = ["stable", "warming", "extreme"]
    results = {}

    base_config = SimulationConfig(
        width=50,
        height=30,
        initial_phytoplankton=800,
        initial_zooplankton=400,
        initial_fish=100,
        max_steps=300,
        random_seed=42,
    )

    print(
        f"Running {len(scenarios)} scenarios with {base_config.max_steps} steps each..."
    )
    print()

    for scenario in scenarios:
        print(f"Running {scenario} scenario...")

        # Configure scenario
        config = SimulationConfig(
            width=base_config.width,
            height=base_config.height,
            initial_phytoplankton=base_config.initial_phytoplankton,
            initial_zooplankton=base_config.initial_zooplankton,
            initial_fish=base_config.initial_fish,
            max_steps=base_config.max_steps,
            climate_scenario=scenario,
            random_seed=base_config.random_seed,
        )

        # Run simulation
        simulation = MarineEcosystemSimulation(config)
        results[scenario] = simulation.run_simulation(verbose=False)

        final_stats = results[scenario].iloc[-1]
        print(f"  Final fish count: {final_stats['fish_count']}")
        print(f"  Final diversity: {final_stats['shannon_diversity']:.3f}")
        print()

    # Create comparison visualization
    print("Generating comparison plots...")

    # Use the last simulation's visualizer for the comparison plot
    visualizer = SimulationVisualizer(simulation)
    fig = visualizer.plot_scenario_comparison(results)
    fig.savefig("scenario_comparison.png", dpi=300, bbox_inches="tight")
    print("📊 Saved: scenario_comparison.png")

    # Create summary table
    print("\n📊 Scenario Comparison Summary:")
    print("-" * 80)
    print(
        f"{'Scenario':<12} {'Final Fish':<12} {'Total Pop':<12} {'Diversity':<12} {'Biomass':<12}"
    )
    print("-" * 80)

    for scenario, df in results.items():
        final = df.iloc[-1]
        print(
            f"{scenario.title():<12} {final['fish_count']:<12} {final['total_agents']:<12} "
            f"{final['shannon_diversity']:<12.3f} {final['total_biomass']:<12.0f}"
        )

    # Save comparison data
    for scenario, df in results.items():
        df.to_csv(f"results_{scenario}.csv", index=False)
        print(f"💾 Saved: results_{scenario}.csv")

    print("\n✅ Scenario comparison completed!")
    plt.show()


def load_config_from_file(config_path: str) -> SimulationConfig:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return SimulationConfig(**config_dict)


def save_default_config():
    """Save a default configuration file as example."""
    config = SimulationConfig()
    config_dict = {
        "width": config.width,
        "height": config.height,
        "initial_temperature": config.initial_temperature,
        "initial_ph": config.initial_ph,
        "initial_oxygen": config.initial_oxygen,
        "initial_phytoplankton": config.initial_phytoplankton,
        "initial_zooplankton": config.initial_zooplankton,
        "initial_fish": config.initial_fish,
        "max_steps": config.max_steps,
        "climate_scenario": config.climate_scenario,
        "random_seed": config.random_seed,
    }

    with open("default_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    print("📁 Saved: default_config.json")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Marine Ecosystem Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --web                    # Launch web interface
  python main.py --run                    # Run default simulation
  python main.py --compare                # Compare climate scenarios
  python main.py --config my_config.json # Use custom configuration
  python main.py --save-config            # Save default config file
        """,
    )

    parser.add_argument(
        "--web", action="store_true", help="Launch the Streamlit web interface"
    )

    parser.add_argument("--run", action="store_true", help="Run a default simulation")

    parser.add_argument(
        "--compare", action="store_true", help="Compare different climate scenarios"
    )

    parser.add_argument("--config", type=str, help="Path to JSON configuration file")

    parser.add_argument(
        "--save-config", action="store_true", help="Save a default configuration file"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps (default: 500)",
    )

    parser.add_argument(
        "--scenario",
        choices=["stable", "warming", "extreme"],
        default="stable",
        help="Climate scenario to run (default: stable)",
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    try:
        if args.save_config:
            save_default_config()

        elif args.web:
            print("🌐 Launching web interface...")
            print("Open your browser to http://localhost:8501")
            run_web_interface()

        elif args.compare:
            compare_climate_scenarios()

        elif args.run or args.config:
            if args.config:
                print(f"📁 Loading configuration from: {args.config}")
                config = load_config_from_file(args.config)
                # Override with command line arguments if provided
                if args.steps != 500:  # If steps was explicitly set
                    config.max_steps = args.steps
                if args.scenario != "stable":  # If scenario was explicitly set
                    config.climate_scenario = args.scenario
            else:
                config = SimulationConfig(
                    max_steps=args.steps, climate_scenario=args.scenario
                )

            print("🌊 Running single simulation...")

            # Create and run simulation
            simulation = MarineEcosystemSimulation(config)
            results_df = simulation.run_simulation(verbose=True)

            # Generate outputs
            visualizer = SimulationVisualizer(simulation)

            # Save plots
            fig1 = visualizer.plot_population_dynamics()
            fig1.savefig("population_dynamics.png", dpi=300, bbox_inches="tight")
            plt.close(fig1)

            fig2 = visualizer.plot_environmental_conditions()
            fig2.savefig("environmental_conditions.png", dpi=300, bbox_inches="tight")
            plt.close(fig2)

            fig3 = visualizer.plot_spatial_distribution()
            fig3.savefig("spatial_distribution.png", dpi=300, bbox_inches="tight")
            plt.close(fig3)

            # Save data and report
            results_df.to_csv("simulation_results.csv", index=False)
            env_df = simulation.get_environmental_dataframe()
            env_df.to_csv("environmental_data.csv", index=False)

            report = visualizer.generate_summary_report()
            with open("simulation_report.txt", "w") as f:
                f.write(report)

            print("\n✅ Simulation completed!")
            print("📊 Generated: population_dynamics.png")
            print("🌡️ Generated: environmental_conditions.png")
            print("🗺️ Generated: spatial_distribution.png")
            print("💾 Generated: simulation_results.csv")
            print("💾 Generated: environmental_data.csv")
            print("📋 Generated: simulation_report.txt")

            # Print summary
            final_stats = results_df.iloc[-1]
            print(f"\n📈 Final Results:")
            print(f"   Total organisms: {final_stats['total_agents']}")
            print(f"   Fish: {final_stats['fish_count']}")
            print(f"   Zooplankton: {final_stats['zooplankton_count']}")
            print(f"   Phytoplankton: {final_stats['phytoplankton_count']}")
            print(f"   Diversity index: {final_stats['shannon_diversity']:.3f}")
            print(f"   Total biomass: {final_stats['total_biomass']:.0f}")

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON configuration - {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise  # Re-raise for debugging


if __name__ == "__main__":
    main()
