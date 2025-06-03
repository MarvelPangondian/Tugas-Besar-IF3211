"""
Marine Environment Module

This module contains the Environment class that manages the marine ecosystem's
environmental conditions including temperature, pH, oxygen levels, and spatial
agent tracking.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


class Environment:
    def __init__(
        self,
        width: int,
        height: int,
        initial_temperature: float = 20,
        initial_ph: float = 8.1,
        initial_oxygen: float = 7,
    ):
        self.width = width
        self.height = height

        self.temperature = np.ones((width, height)) * initial_temperature
        self.ph = np.ones((width, height)) * initial_ph
        self.oxygen = np.ones((width, height)) * initial_oxygen
        self.light = np.ones((width, height))

        self.create_environmental_gradients()

        self.agents_by_position = defaultdict(list)
        self.time = 0

        self.statistics = {
            "time": [],
            "phytoplankton": [],
            "zooplankton": [],
            "fish": [],
            "avg_temperature": [],
            "avg_ph": [],
            "avg_oxygen": [],
        }

    def create_environmental_gradients(self) -> None:
        for y in range(self.height):
            depth_factor = y / self.height
            self.temperature[:, y] *= 1 - 0.3 * depth_factor
            self.oxygen[:, y] *= 1 - 0.2 * depth_factor
            kd = 0.25
            self.light[:, y] = np.exp(-kd * depth_factor)
            self.ph[:, y] -= 0.1 * depth_factor

        self._add_spatial_variation()

    def _add_spatial_variation(self) -> None:
        temp_noise = np.random.normal(0, 0.5, (self.width, self.height))
        ph_noise = np.random.normal(0, 0.05, (self.width, self.height))
        oxygen_noise = np.random.normal(0, 0.3, (self.width, self.height))

        self.temperature += temp_noise
        self.ph += ph_noise
        self.oxygen += oxygen_noise

        self.temperature = np.clip(self.temperature, 0, 35)
        self.ph = np.clip(self.ph, 6.5, 9.0)
        self.oxygen = np.clip(self.oxygen, 0, 12)

    def update_environment(self, climate_scenario: str = "stable") -> None:
        """Enhanced environment update with realistic climate change effects."""
        self.time += 1

        if climate_scenario == "warming":
            self._apply_warming_scenario()
        elif climate_scenario == "extreme":
            self._apply_extreme_scenario()

        self._apply_synergistic_effects(climate_scenario)

        # Apply tipping point effects (NEW)
        self._check_tipping_points()

        self._enforce_environmental_bounds()

    def _apply_warming_scenario(self) -> None:
        """Realistic IPCC warming scenario - moderate but significant change."""

        base_temp_increase = 0.08
        base_ph_decrease = 0.008
        base_oxygen_decrease = 0.04

        # Add spatial variation - some areas warm faster
        for x in range(self.width):
            for y in range(self.height):
                # Surface waters warm faster
                depth_factor = 1.0 + (y / self.height) * 0.5  # 1.0 to 1.5x

                # Add regional variation (some areas more affected)
                regional_factor = 1.0 + 0.3 * np.sin(x / self.width * np.pi)

                temp_increase = base_temp_increase * depth_factor * regional_factor
                ph_decrease = base_ph_decrease * depth_factor
                oxygen_decrease = base_oxygen_decrease * depth_factor

                self.temperature[x, y] += temp_increase
                self.ph[x, y] -= ph_decrease
                self.oxygen[x, y] -= oxygen_decrease

    def _apply_extreme_scenario(self) -> None:
        """Extreme climate change - worst case scenario."""
        # Even more dramatic changes representing climate breakdown

        base_temp_increase = 0.15
        base_ph_decrease = 0.015
        base_oxygen_decrease = 0.08

        # extreme spatial variation
        for x in range(self.width):
            for y in range(self.height):
                # Extreme surface warming
                depth_factor = 1.0 + (y / self.height) * 1.0  # 1.0 to 2.0x

                # Extreme regional variation
                regional_factor = 1.0 + 0.6 * np.sin(x / self.width * np.pi)

                # Add periodic heat waves (NEW)
                if self.time % 20 < 5:
                    heat_wave_factor = 2.0
                else:
                    heat_wave_factor = 1.0

                temp_increase = (
                    base_temp_increase
                    * depth_factor
                    * regional_factor
                    * heat_wave_factor
                )
                ph_decrease = base_ph_decrease * depth_factor
                oxygen_decrease = base_oxygen_decrease * depth_factor

                self.temperature[x, y] += temp_increase
                self.ph[x, y] -= ph_decrease
                self.oxygen[x, y] -= oxygen_decrease

    def _apply_synergistic_effects(self, climate_scenario: str) -> None:
        """Apply synergistic effects where multiple stressors amplify each other."""
        if climate_scenario == "stable":
            return

        for x in range(self.width):
            for y in range(self.height):
                temp = self.temperature[x, y]
                ph = self.ph[x, y]
                oxygen = self.oxygen[x, y]

                # High temperature reduces oxygen solubility (real physical effect)
                if temp > 25:
                    oxygen_loss = (temp - 25) * 0.02
                    self.oxygen[x, y] -= oxygen_loss

                # Low pH + high temp = deadly combination for marine life
                if ph < 7.5 and temp > 23:
                    self.temperature[x, y] += 0.02

                # Hypoxia + acidification creates dead zones
                if oxygen < 3.0 and ph < 7.8:
                    self.oxygen[x, y] -= 0.05

    def _check_tipping_points(self) -> None:
        """Check for environmental tipping points and cascade effects."""

        avg_temp = np.mean(self.temperature)
        avg_ph = np.mean(self.ph)
        avg_oxygen = np.mean(self.oxygen)

        # Temperature tipping point - marine heatwave
        if avg_temp > 28:
            self.temperature += 0.05

        # Acidification tipping point - chemistry cascade
        if avg_ph < 7.0:
            self.ph -= 0.01

        # Hypoxia tipping point - dead zone expansion
        if avg_oxygen < 2.0:
            # Hypoxia begets more hypoxia due to ecosystem collapse
            self.oxygen -= 0.02

    def calculate_climate_vulnerability_multiplier(
        self, stress_level: float, climate_scenario: str
    ) -> float:
        """Calculate how climate change amplifies stress effects."""

        if climate_scenario == "stable":
            return 1.0
        elif climate_scenario == "warming":
            # Moderate amplification
            return 1.0 + stress_level * 0.5  # Up to 1.5x mortality
        elif climate_scenario == "extreme":
            # Severe amplification - climate refugees
            return 1.0 + stress_level * 1.2  # Up to 2.2x mortality

        return 1.0

    def _enforce_environmental_bounds(self) -> None:
        self.temperature = np.clip(self.temperature, -2, 40)
        self.ph = np.clip(self.ph, 6.0, 9.5)
        self.oxygen = np.clip(self.oxygen, 0, 15)

    def get_conditions(self, x: int, y: int) -> Dict[str, float]:
        return {
            "temperature": self.temperature[x, y],
            "ph": self.ph[x, y],
            "oxygen": self.oxygen[x, y],
            "light": self.light[x, y],
        }

    def register_agent(self, agent: Any) -> None:
        self.agents_by_position[(agent.x, agent.y)].append(agent)

    def unregister_agent(self, agent: Any) -> None:
        if agent in self.agents_by_position[(agent.x, agent.y)]:
            self.agents_by_position[(agent.x, agent.y)].remove(agent)

    def update_agent_position(self, agent: Any, old_x: int, old_y: int) -> None:
        if agent in self.agents_by_position[(old_x, old_y)]:
            self.agents_by_position[(old_x, old_y)].remove(agent)
        self.agents_by_position[(agent.x, agent.y)].append(agent)

    def get_agents_at(
        self, x: int, y: int, agent_type: Optional[type] = None
    ) -> List[Any]:
        agents = self.agents_by_position[(x, y)]
        if agent_type:
            return [a for a in agents if isinstance(a, agent_type)]
        return agents

    def get_agent_count_in_radius(
        self, x: int, y: int, radius: int, agent_type: Optional[type] = None
    ) -> int:
        count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    agents = self.get_agents_at(nx, ny, agent_type)
                    count += len(agents)
        return count

    def find_best_location_in_radius(
        self,
        x: int,
        y: int,
        radius: int,
        criteria: str = "temperature",
        target_value: Optional[float] = None,
    ) -> Tuple[int, int]:
        best_pos = (x, y)
        best_score = float("inf") if target_value else float("-inf")

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    conditions = self.get_conditions(nx, ny)
                    value = conditions.get(criteria, 0)
                    score = (
                        -abs(value - target_value)
                        if target_value is not None
                        else value
                    )

                    if (target_value and score > best_score) or (
                        not target_value and score > best_score
                    ):
                        best_score = score
                        best_pos = (nx, ny)

        return best_pos

    def update_statistics(self, agents: List[Any]) -> None:
        phyto_count = sum(1 for a in agents if a.__class__.__name__ == "Phytoplankton")
        zoo_count = sum(1 for a in agents if a.__class__.__name__ == "Zooplankton")
        fish_count = sum(1 for a in agents if a.__class__.__name__ == "Fish")

        avg_temp = np.mean(self.temperature)
        avg_ph = np.mean(self.ph)
        avg_oxygen = np.mean(self.oxygen)

        self.statistics["time"].append(self.time)
        self.statistics["phytoplankton"].append(phyto_count)
        self.statistics["zooplankton"].append(zoo_count)
        self.statistics["fish"].append(fish_count)
        self.statistics["avg_temperature"].append(avg_temp)
        self.statistics["avg_ph"].append(avg_ph)
        self.statistics["avg_oxygen"].append(avg_oxygen)

    def get_environmental_summary(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "temperature_range": (np.min(self.temperature), np.max(self.temperature)),
            "ph_range": (np.min(self.ph), np.max(self.ph)),
            "oxygen_range": (np.min(self.oxygen), np.max(self.oxygen)),
            "avg_temperature": np.mean(self.temperature),
            "avg_ph": np.mean(self.ph),
            "avg_oxygen": np.mean(self.oxygen),
        }

    def reset(self) -> None:
        self.__init__(self.width, self.height)
