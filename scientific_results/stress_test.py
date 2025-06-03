#!/usr/bin/env python3
"""
Isolated Stress Test - Test each trophic level separately
This removes food web effects to test direct physiological vulnerability.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulation import MarineEcosystemSimulation, SimulationConfig


def isolated_trophic_stress_test():
    """
    Test each trophic level in isolation to measure direct physiological vulnerability.
    """
    print("Isolated Trophic Level Stress Test")
    print("Testing direct physiological vulnerability without food web effects")
    print("=" * 80)

    # Test configurations for each trophic level in isolation
    base_config = {
        "width": 40,
        "height": 25,
        "max_steps": 100,
        "climate_scenario": "stable",
        "random_seed": 42,
    }

    # Stress levels to test
    stress_levels = [
        {"name": "mild", "temp": 22, "ph": 7.9, "oxygen": 6.0},
        {"name": "moderate", "temp": 24, "ph": 7.7, "oxygen": 5.0},
        {"name": "severe", "temp": 26, "ph": 7.5, "oxygen": 4.0},
        {"name": "extreme", "temp": 28, "ph": 7.3, "oxygen": 3.0},
    ]

    results = {}

    for stress in stress_levels:
        print(f"\Testing {stress['name'].upper()} stress level:")
        print(
            f"   Temperature: {stress['temp']}°C, pH: {stress['ph']}, Oxygen: {stress['oxygen']} mg/L"
        )

        stress_results = {}

        # Test 1: Only Phytoplankton
        print(f"\nTesting Phytoplankton alone...")
        phyto_config = SimulationConfig(
            **base_config,
            initial_temperature=stress["temp"],
            initial_ph=stress["ph"],
            initial_oxygen=stress["oxygen"],
            initial_phytoplankton=500,
            initial_zooplankton=0,  # No zooplankton
            initial_fish=0,  # No fish
        )

        phyto_sim = MarineEcosystemSimulation(phyto_config)
        phyto_df = phyto_sim.run_simulation(verbose=False)

        if not phyto_df.empty:
            phyto_survival = (
                phyto_df.iloc[-1]["phytoplankton_count"]
                / phyto_df.iloc[0]["phytoplankton_count"]
            )
            phyto_avg_stress = phyto_df["phyto_avg_stress"].mean()
            print(f"      Survival: {phyto_survival:.3f} ({phyto_survival*100:.1f}%)")
            print(f"      Avg Stress: {phyto_avg_stress:.3f}")
        else:
            phyto_survival = 0
            phyto_avg_stress = 1.0

        stress_results["phytoplankton"] = {
            "survival": phyto_survival,
            "avg_stress": phyto_avg_stress,
            "data": phyto_df,
        }

        # Test 2: Only Zooplankton (with enough phytoplankton food)
        print(f"\nTesting Zooplankton (with abundant food)...")
        zoo_config = SimulationConfig(
            **base_config,
            initial_temperature=stress["temp"],
            initial_ph=stress["ph"],
            initial_oxygen=stress["oxygen"],
            initial_phytoplankton=2000,  # Abundant food
            initial_zooplankton=300,
            initial_fish=0,  # No fish predation
        )

        zoo_sim = MarineEcosystemSimulation(zoo_config)
        zoo_df = zoo_sim.run_simulation(verbose=False)

        if not zoo_df.empty:
            zoo_survival = (
                zoo_df.iloc[-1]["zooplankton_count"]
                / zoo_df.iloc[0]["zooplankton_count"]
            )
            zoo_avg_stress = zoo_df["zoo_avg_stress"].mean()
            print(f"      Survival: {zoo_survival:.3f} ({zoo_survival*100:.1f}%)")
            print(f"      Avg Stress: {zoo_avg_stress:.3f}")
        else:
            zoo_survival = 0
            zoo_avg_stress = 1.0

        stress_results["zooplankton"] = {
            "survival": zoo_survival,
            "avg_stress": zoo_avg_stress,
            "data": zoo_df,
        }

        # Test 3: Only Fish (with abundant food)
        print(f"\nTesting Fish (with abundant food)...")
        fish_config = SimulationConfig(
            **base_config,
            initial_temperature=stress["temp"],
            initial_ph=stress["ph"],
            initial_oxygen=stress["oxygen"],
            initial_phytoplankton=1500,  # Food for zooplankton
            initial_zooplankton=800,  # Abundant fish food
            initial_fish=60,
        )

        fish_sim = MarineEcosystemSimulation(fish_config)
        fish_df = fish_sim.run_simulation(verbose=False)

        if not fish_df.empty:
            fish_survival = (
                fish_df.iloc[-1]["fish_count"] / fish_df.iloc[0]["fish_count"]
            )
            fish_avg_stress = fish_df["fish_avg_stress"].mean()
            print(f"      Survival: {fish_survival:.3f} ({fish_survival*100:.1f}%)")
            print(f"      Avg Stress: {fish_avg_stress:.3f}")
        else:
            fish_survival = 0
            fish_avg_stress = 1.0

        stress_results["fish"] = {
            "survival": fish_survival,
            "avg_stress": fish_avg_stress,
            "data": fish_df,
        }

        results[stress["name"]] = stress_results

    return results


def analyze_direct_vulnerability(results):
    """
    Analyze direct physiological vulnerability without food web effects.
    """
    print(f"\nDIRECT PHYSIOLOGICAL VULNERABILITY ANALYSIS")
    print("=" * 70)

    # Create vulnerability summary
    vulnerability_data = []

    for stress_level, stress_results in results.items():
        print(f"\n{stress_level.upper()} Stress Level:")
        print("-" * 40)

        level_data = {"stress_level": stress_level}

        for species in ["phytoplankton", "zooplankton", "fish"]:
            survival = stress_results[species]["survival"]
            avg_stress = stress_results[species]["avg_stress"]

            level_data[f"{species}_survival"] = survival
            level_data[f"{species}_stress"] = avg_stress

            print(
                f"{species.title():>15}: {survival:.3f} survival, {avg_stress:.3f} stress"
            )

        vulnerability_data.append(level_data)

        # Rank vulnerability for this stress level
        survivals = {
            "Phytoplankton": stress_results["phytoplankton"]["survival"],
            "Zooplankton": stress_results["zooplankton"]["survival"],
            "Fish": stress_results["fish"]["survival"],
        }

        most_vulnerable = min(survivals.keys(), key=lambda k: survivals[k])
        least_vulnerable = max(survivals.keys(), key=lambda k: survivals[k])

        print(
            f"   Most vulnerable: {most_vulnerable} ({survivals[most_vulnerable]:.3f})"
        )
        print(
            f"   Least vulnerable: {least_vulnerable} ({survivals[least_vulnerable]:.3f})"
        )

    return pd.DataFrame(vulnerability_data)


def plot_isolated_vulnerability(results):
    """
    Plot direct vulnerability curves for each trophic level.
    """
    print(f"\nGenerating isolated vulnerability plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Direct Physiological Vulnerability (Isolated Trophic Levels)",
        fontsize=16,
        fontweight="bold",
    )

    stress_order = ["mild", "moderate", "severe", "extreme"]
    stress_numeric = [1, 2, 3, 4]  # For plotting

    colors = {"phytoplankton": "#2E8B57", "zooplankton": "#4169E1", "fish": "#DC143C"}

    # Plot 1: Survival rates
    ax1 = axes[0, 0]
    for species in ["phytoplankton", "zooplankton", "fish"]:
        survivals = [results[stress][species]["survival"] for stress in stress_order]
        ax1.plot(
            stress_numeric,
            survivals,
            "o-",
            color=colors[species],
            linewidth=3,
            markersize=8,
            label=species.title(),
        )

    ax1.set_xlabel("Stress Level")
    ax1.set_ylabel("Survival Rate")
    ax1.set_title("Survival vs Stress Level")
    ax1.set_xticks(stress_numeric)
    ax1.set_xticklabels([s.title() for s in stress_order])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Plot 2: Average stress levels
    ax2 = axes[0, 1]
    for species in ["phytoplankton", "zooplankton", "fish"]:
        stress_levels = [
            results[stress][species]["avg_stress"] for stress in stress_order
        ]
        ax2.plot(
            stress_numeric,
            stress_levels,
            "s-",
            color=colors[species],
            linewidth=3,
            markersize=8,
            label=species.title(),
        )

    ax2.set_xlabel("Environmental Stress Level")
    ax2.set_ylabel("Physiological Stress")
    ax2.set_title("Physiological Stress Response")
    ax2.set_xticks(stress_numeric)
    ax2.set_xticklabels([s.title() for s in stress_order])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Vulnerability ranking by stress level
    ax3 = axes[1, 0]
    vulnerability_scores = {"Phytoplankton": [], "Zooplankton": [], "Fish": []}

    for stress in stress_order:
        survivals = {
            "Phytoplankton": results[stress]["phytoplankton"]["survival"],
            "Zooplankton": results[stress]["zooplankton"]["survival"],
            "Fish": results[stress]["fish"]["survival"],
        }

        # Rank from most vulnerable (1) to least vulnerable (3)
        sorted_species = sorted(survivals.keys(), key=lambda k: survivals[k])

        for i, species in enumerate(sorted_species):
            vulnerability_scores[species].append(
                len(sorted_species) - i
            )  # Invert ranking

    for species, scores in vulnerability_scores.items():
        ax3.plot(
            stress_numeric,
            scores,
            "o-",
            color=colors[species.lower()],
            linewidth=3,
            markersize=8,
            label=species,
        )

    ax3.set_xlabel("Stress Level")
    ax3.set_ylabel("Vulnerability Rank (Higher = More Vulnerable)")
    ax3.set_title("Vulnerability Ranking")
    ax3.set_xticks(stress_numeric)
    ax3.set_xticklabels([s.title() for s in stress_order])
    ax3.set_yticks([1, 2, 3])
    ax3.set_yticklabels(["Least\nVulnerable", "Intermediate", "Most\nVulnerable"])
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Summary bar chart for extreme stress
    ax4 = axes[1, 1]
    extreme_survivals = [
        results["extreme"]["phytoplankton"]["survival"],
        results["extreme"]["zooplankton"]["survival"],
        results["extreme"]["fish"]["survival"],
    ]

    bars = ax4.bar(
        ["Phytoplankton", "Zooplankton", "Fish"],
        extreme_survivals,
        color=[colors["phytoplankton"], colors["zooplankton"], colors["fish"]],
    )

    ax4.set_ylabel("Survival Rate")
    ax4.set_title("Survival Under Extreme Stress")
    ax4.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, extreme_survivals):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("./scientific_results/all_results/isolated_vulnerability.png", dpi=300, bbox_inches="tight")
    print("Saved: isolated_vulnerability.png")
    return fig


def print_final_assessment(results):
    """
    Print final assessment of the vulnerability hypothesis.
    """
    print(f"\nFINAL ASSESSMENT: Direct Physiological Vulnerability")
    print("=" * 70)

    # Count how often each species is most vulnerable across stress levels
    vulnerability_counts = {"Phytoplankton": 0, "Zooplankton": 0, "Fish": 0}

    for stress_level, stress_results in results.items():
        survivals = {
            "Phytoplankton": stress_results["phytoplankton"]["survival"],
            "Zooplankton": stress_results["zooplankton"]["survival"],
            "Fish": stress_results["fish"]["survival"],
        }

        most_vulnerable = min(survivals.keys(), key=lambda k: survivals[k])
        vulnerability_counts[most_vulnerable] += 1

        print(f"{stress_level.title()} stress - Most vulnerable: {most_vulnerable}")

    print(f"\nOverall vulnerability ranking (direct physiological effects):")
    sorted_vuln = sorted(vulnerability_counts.items(), key=lambda x: x[1], reverse=True)

    for i, (species, count) in enumerate(sorted_vuln, 1):
        print(f"  {i}. {species} (most vulnerable in {count}/4 scenarios)")

    # Test hypothesis
    print(f"\nHYPOTHESIS TEST RESULTS:")
    print("-" * 40)

    fish_rank = next(
        i for i, (species, _) in enumerate(sorted_vuln, 1) if species == "Fish"
    )

	# TODO : Research more about other hypothesis...
    if fish_rank == 1:
        print(
            "✅ HYPOTHESIS SUPPORTED: Fish show highest direct physiological vulnerability"
        )
        print("   Fish are most sensitive when food web effects are removed")
    elif fish_rank == 2:
        print(
            "⚠️  HYPOTHESIS PARTIALLY SUPPORTED: Fish show moderate direct vulnerability"
        )
        print("   Fish are not the most vulnerable, but still highly sensitive")
    else:
        print(
            "❌ HYPOTHESIS NOT SUPPORTED: Fish show lowest direct physiological vulnerability"
        )
        print("   Fish appear to be most resilient even to direct environmental stress")

    print(f"\n💡 ECOLOGICAL INSIGHTS:")
    print("-" * 25)
    if sorted_vuln[0][0] == "Fish":
        print("• Fish physiology is most sensitive to environmental changes")
        print("• Fish behavioral adaptations help them survive in ecosystem context")
        print("• Direct vulnerability ≠ ecological vulnerability")
    elif sorted_vuln[0][0] == "Phytoplankton":
        print("• Primary producers bear the brunt of environmental stress")
        print("• Phytoplankton have narrow tolerance ranges")
        print(
            "• Bottom-up ecosystem effects dominate over direct top-predator vulnerability"
        )
    else:
        print("• Zooplankton show intermediate vulnerability as expected")
        print("• Trophic position may correlate with environmental sensitivity")


def main1():
    """Run the isolated trophic stress test."""

    print("Marine Ecosystem - Isolated Trophic Level Stress Test")
    print("Testing direct physiological vulnerability without food web interactions")
    print("=" * 90)

    # Run isolated stress tests
    results = isolated_trophic_stress_test()

    # Analyze results
    vulnerability_df = analyze_direct_vulnerability(results)

    # Create plots
    fig = plot_isolated_vulnerability(results)

    # Final assessment
    print_final_assessment(results)

    # Save data
    print(f"\nSaving results...")
    vulnerability_df.to_csv("./scientific_results/all_results/isolated_vulnerability_results.csv", index=False)
    print(f"Saved: isolated_vulnerability_results.csv")

    print(f"\nIsolated vulnerability test complete!")
    print(f"Check isolated_vulnerability.png for visual results")

    plt.show()


if __name__ == "__main__":
    main1()
