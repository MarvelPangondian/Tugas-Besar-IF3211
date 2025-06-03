"""
Marine Ecosystem Simulation Module

This module contains the main simulation class that orchestrates all components
of the marine ecosystem model, including environment, agents, and data collection.
"""

import random
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .environment import Environment
from .agents import Phytoplankton, Zooplankton, Fish


@dataclass
class SimulationConfig:
    width: int = 50
    height: int = 30
    initial_temperature: float = 20.0
    initial_ph: float = 8.1
    initial_oxygen: float = 7.0

    initial_phytoplankton: int = 800
    initial_zooplankton: int = 400
    initial_fish: int = 100

    max_steps: int = 1000
    climate_scenario: str = "stable"

    phytoplankton_depth_limit: float = 0.7
    random_seed: Optional[int] = None


class MarineEcosystemSimulation:
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self.environment = Environment(
            width=self.config.width,
            height=self.config.height,
            initial_temperature=self.config.initial_temperature,
            initial_ph=self.config.initial_ph,
            initial_oxygen=self.config.initial_oxygen,
        )

        self.agents: List[Any] = []
        self.initialize_agents()

        self.current_step = 0
        self.is_running = False
        self.detailed_stats = []
        self.environmental_history = []
        self.spatial_history = []

    def initialize_agents(self) -> None:
        self.agents.clear()
        max_phyto_depth = max(5, int(self.environment.height * 0.2))

        for _ in range(self.config.initial_phytoplankton):
            x = random.randint(0, self.environment.width - 1)
            if random.random() < 0.8:
                y = random.randint(0, min(2, max_phyto_depth))
            else:
                y = random.randint(0, max_phyto_depth)
            energy = random.uniform(4, 8)
            initial_age = random.randint(0, 150)
            phyto = Phytoplankton(self.environment, x, y, energy)
            phyto.age = initial_age
            self.agents.append(phyto)

        phyto_depths = [
            agent.y for agent in self.agents if isinstance(agent, Phytoplankton)
        ]
        print(
            f"Phytoplankton depths: min={min(phyto_depths)}, max={max(phyto_depths)}, avg={np.mean(phyto_depths):.1f}"
        )

        max_zoo_depth = int(self.environment.height * 0.4)
        for _ in range(self.config.initial_zooplankton):
            x = random.randint(0, self.environment.width - 1)
            if random.random() < 0.7:
                y = random.randint(0, min(8, max_zoo_depth))
            else:
                y = random.randint(0, max_zoo_depth)
            energy = random.uniform(8, 15)
            self.agents.append(Zooplankton(self.environment, x, y, energy))

        for _ in range(self.config.initial_fish):
            x = random.randint(0, self.environment.width - 1)
            if random.random() < 0.6:
                y = random.randint(0, int(self.environment.height * 0.4))
            else:
                y = random.randint(
                    int(self.environment.height * 0.2),
                    int(self.environment.height * 0.7),
                )
            energy = random.uniform(25, 40)
            self.agents.append(Fish(self.environment, x, y, energy))

    def run_step(self) -> Dict[str, Any]:
        self.environment.update_environment(self.config.climate_scenario)
        random.shuffle(self.agents)

        agents_to_remove = []
        for agent in self.agents:
            if agent.alive:
                offspring = agent.update()
                if offspring is not None:
                    self.agents.append(offspring)
                if not agent.alive:
                    agents_to_remove.append(agent)
            else:
                agents_to_remove.append(agent)

        for agent in agents_to_remove:
            if agent in self.agents:
                self.agents.remove(agent)

        self.spatial_history.append(
            [
                {"type": type(agent).__name__.lower(), "x": agent.x, "y": agent.y}
                for agent in self.agents
                if agent.alive
            ]
        )

        self.environment.update_statistics(self.agents)
        step_stats = self.collect_step_statistics()
        self.detailed_stats.append(step_stats)
        self.environmental_history.append(self.environment.get_environmental_summary())
        self.current_step += 1
        return step_stats

    def collect_step_statistics(self) -> Dict[str, Any]:
        phyto_count = sum(1 for a in self.agents if isinstance(a, Phytoplankton))
        zoo_count = sum(1 for a in self.agents if isinstance(a, Zooplankton))
        fish_count = sum(1 for a in self.agents if isinstance(a, Fish))

        phyto_energies = [a.energy for a in self.agents if isinstance(a, Phytoplankton)]
        zoo_energies = [a.energy for a in self.agents if isinstance(a, Zooplankton)]
        fish_energies = [a.energy for a in self.agents if isinstance(a, Fish)]

        phyto_stress = [
            a.calculate_environmental_stress()
            for a in self.agents
            if isinstance(a, Phytoplankton)
        ]
        zoo_stress = [
            a.calculate_environmental_stress()
            for a in self.agents
            if isinstance(a, Zooplankton)
        ]
        fish_stress = [
            a.calculate_environmental_stress()
            for a in self.agents
            if isinstance(a, Fish)
        ]

        return {
            "step": self.current_step,
            "total_agents": len(self.agents),
            "phytoplankton_count": phyto_count,
            "zooplankton_count": zoo_count,
            "fish_count": fish_count,
            "phyto_avg_energy": np.mean(phyto_energies) if phyto_energies else 0,
            "zoo_avg_energy": np.mean(zoo_energies) if zoo_energies else 0,
            "fish_avg_energy": np.mean(fish_energies) if fish_energies else 0,
            "phyto_energy_std": np.std(phyto_energies) if phyto_energies else 0,
            "zoo_energy_std": np.std(zoo_energies) if zoo_energies else 0,
            "fish_energy_std": np.std(fish_energies) if fish_energies else 0,
            "phyto_avg_stress": np.mean(phyto_stress) if phyto_stress else 0,
            "zoo_avg_stress": np.mean(zoo_stress) if zoo_stress else 0,
            "fish_avg_stress": np.mean(fish_stress) if fish_stress else 0,
            "average_stress_level": np.mean(
                [
                    np.mean(phyto_stress) if phyto_stress else 0,
                    np.mean(zoo_stress) if zoo_stress else 0,
                    np.mean(fish_stress) if fish_stress else 0,
                ]
            ),
            "shannon_diversity": self.calculate_shannon_diversity(),
            "total_biomass": self.calculate_total_biomass(),
            "avg_temperature": np.mean(self.environment.temperature),
            "avg_ph": np.mean(self.environment.ph),
            "avg_oxygen": np.mean(self.environment.oxygen),
        }

    def calculate_shannon_diversity(self) -> float:
        counts = [
            sum(1 for a in self.agents if isinstance(a, Phytoplankton)),
            sum(1 for a in self.agents if isinstance(a, Zooplankton)),
            sum(1 for a in self.agents if isinstance(a, Fish)),
        ]
        total = sum(counts)
        if total == 0:
            return 0
        return -sum((p := c / total) * np.log(p) for c in counts if c > 0)

    def calculate_total_biomass(self) -> float:
        return sum(agent.energy for agent in self.agents)

    def run_simulation(
        self, steps: Optional[int] = None, verbose: bool = True
    ) -> pd.DataFrame:
        steps = steps or self.config.max_steps
        self.is_running = True

        if verbose:
            print(f"Starting simulation with {len(self.agents)} initial agents")
            print(f"Climate scenario: {self.config.climate_scenario}")
            print(f"Running for {steps} steps...")

        try:
            for step in range(steps):
                stats = self.run_step()
                if verbose and (step + 1) % 50 == 0:
                    print(
                        f"Step {step + 1}/{steps}: Agents: {stats['total_agents']}, Diversity: {stats['shannon_diversity']:.3f}"
                    )
                if stats["total_agents"] == 0:
                    print(f"Ecosystem collapse at step {step + 1}")
                    break
                if stats["fish_count"] == 0 and step > 100:
                    print(f"Fish extinction at step {step + 1}")
        except KeyboardInterrupt:
            print(f"Simulation interrupted at step {self.current_step}")
        finally:
            self.is_running = False

        if verbose:
            print("Simulation completed!")
            final_stats = self.detailed_stats[-1] if self.detailed_stats else {}
            print(f"Final population: {final_stats.get('total_agents', 0)} agents")

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.detailed_stats)

    def get_environmental_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.environmental_history)

    def reset_simulation(self) -> None:
        self.current_step = 0
        self.is_running = False
        self.detailed_stats.clear()
        self.environmental_history.clear()
        self.environment.reset()
        self.initialize_agents()

    def get_agent_spatial_distribution(self) -> Dict[str, List[Tuple[int, int]]]:
        distribution = {"phytoplankton": [], "zooplankton": [], "fish": []}
        for agent in self.agents:
            key = type(agent).__name__.lower()
            if key in distribution:
                distribution[key].append((agent.x, agent.y))
        return distribution

    def get_ecosystem_health_metrics(self) -> Dict[str, float]:
        if not self.detailed_stats:
            return {}

        latest = self.detailed_stats[-1]
        history_subset = int(len(self.detailed_stats) * 0.8)
        recent = (
            self.detailed_stats[history_subset:]
            if history_subset < len(self.detailed_stats)
            else self.detailed_stats
        )

        if len(recent) < 2:
            return latest

        def cv(vals):
            m = np.mean(vals)
            return np.std(vals) / m if m else float("inf")

        metrics = {
            "current_diversity": latest["shannon_diversity"],
            "current_biomass": latest["total_biomass"],
            "fish_population_stability": 1
            / (1 + cv([s["fish_count"] for s in recent])),
            "zoo_population_stability": 1
            / (1 + cv([s["zooplankton_count"] for s in recent])),
            "phyto_population_stability": 1
            / (1 + cv([s["phytoplankton_count"] for s in recent])),
            "average_stress_level": (
                latest["phyto_avg_stress"]
                + latest["zoo_avg_stress"]
                + latest["fish_avg_stress"]
            )
            / 3,
            "ecosystem_resilience": self._calculate_resilience_score(),
        }
        return metrics

    def _calculate_resilience_score(self) -> float:
        if len(self.detailed_stats) < 10:
            return 0.5

        latest = self.detailed_stats[-1]
        trophic_score = (
            1.0
            if all(
                [
                    latest["phytoplankton_count"] > 0,
                    latest["zooplankton_count"] > 0,
                    latest["fish_count"] > 0,
                ]
            )
            else 0.3
        )

        stress_score = 1 - latest["average_stress_level"]
        max_diversity = np.log(3)
        diversity_score = latest["shannon_diversity"] / max_diversity
        return max(
            0, min(1, trophic_score * 0.4 + stress_score * 0.3 + diversity_score * 0.3)
        )

    def compare_scenarios(
        self, scenarios: List[str], steps: int = 500
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        original = self.config.climate_scenario

        for scenario in scenarios:
            print(f"\nRunning scenario: {scenario}")
            self.reset_simulation()
            self.config.climate_scenario = scenario
            results[scenario] = self.run_simulation(steps, verbose=True)

        self.config.climate_scenario = original
        return results

    def get_agent_spatial_dataframe(self):
        records = []
        for step_index, agent_list in enumerate(self.spatial_history):
            for agent in agent_list:
                records.append({"step": step_index, **agent})
        return pd.DataFrame(records)
