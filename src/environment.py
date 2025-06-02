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
            self.light[:, y] = np.exp(-2.0 * depth_factor)
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
        self.time += 1

        if climate_scenario == "warming":
            self._apply_warming_scenario()
        elif climate_scenario == "extreme":
            self._apply_extreme_scenario()

        self._enforce_environmental_bounds()

    def _apply_warming_scenario(self) -> None:
        self.temperature += 0.01
        self.ph -= 0.001
        self.oxygen -= 0.005

    def _apply_extreme_scenario(self) -> None:
        self.temperature += 0.03
        self.ph -= 0.003
        self.oxygen -= 0.01

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
