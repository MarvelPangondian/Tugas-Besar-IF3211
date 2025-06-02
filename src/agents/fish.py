"""
Fish Agent Module

This module contains the Fish class, representing the top predators in the
marine ecosystem that feed on zooplankton and are most sensitive to
environmental changes.
"""

import random
from typing import Optional, List, Tuple
from .base_agent import Agent
from .zooplankton import Zooplankton


class Fish(Agent):
    """
    Fish agent - top predator that feeds on zooplankton.
    """

    def __init__(self, environment, x: int, y: int, energy: float = 30):
        super().__init__(environment, x, y, energy)

        self.optimal_temperature = 18.0
        self.temp_tolerance = 5.0
        self.optimal_oxygen = 7.0
        self.oxygen_tolerance = 2.0
        self.optimal_ph = 8.1
        self.ph_tolerance = 0.3

        self.feeding_efficiency = 0.9
        self.energy_per_zooplankton = 12
        self.hunting_range = 4
        self.max_feeding_per_step = 3
        self.hunt_success_rate = 0.8

        self.reproduction_threshold = 60
        self.reproduction_cost = 25
        self.reproduction_age_min = 25
        self.reproduction_age_max = 200
        self.offspring_energy = 15

        self.base_metabolism = 0.12
        self.movement_cost = 0.08
        self.max_movement_distance = 4
        self.migration_threshold = 0.7

        self.hunger_threshold = 25
        self.territory_radius = 3
        self.schooling_tendency = 0.3
        self.max_lifespan = 500

    def find_prey(self) -> List[Zooplankton]:
        prey_list = []
        for radius in range(1, self.hunting_range + 1):
            circle_prey = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius or radius == 1:
                        nx = self.x + dx
                        ny = self.y + dy
                        if (
                            0 <= nx < self.environment.width
                            and 0 <= ny < self.environment.height
                        ):
                            zooplankton = self.environment.get_agents_at(
                                nx, ny, Zooplankton
                            )
                            circle_prey.extend(zooplankton)
            if circle_prey:
                prey_list.extend(circle_prey)
                if radius <= 2:
                    break
        return [prey for prey in prey_list if prey.alive]

    def hunt(self) -> float:
        total_energy_gained = 0
        meals_consumed = 0
        local_prey = self.environment.get_agents_at(self.x, self.y, Zooplankton)
        for prey in local_prey:
            if meals_consumed >= self.max_feeding_per_step:
                break
            if prey.alive and random.random() < self.hunt_success_rate:
                energy_gained = self.energy_per_zooplankton * self.feeding_efficiency
                self.energy = min(self.energy + energy_gained, self.max_energy)
                total_energy_gained += energy_gained
                meals_consumed += 1
                prey.die()
        return total_energy_gained

    def seek_prey(self) -> bool:
        prey_list = self.find_prey()
        if not prey_list:
            return False
        prey_density = {}
        for prey in prey_list:
            pos = (prey.x, prey.y)
            prey_density[pos] = prey_density.get(pos, 0) + 1
        if not prey_density:
            return False
        best_pos = max(prey_density.keys(), key=lambda pos: prey_density[pos])
        return self.move_towards(best_pos[0], best_pos[1], self.max_movement_distance)

    def migrate_to_better_conditions(self) -> bool:
        if self.calculate_environmental_stress() < self.migration_threshold:
            return False
        best_pos = self.find_optimal_location(search_radius=6)
        if best_pos != (self.x, self.y):
            return self.move_towards(
                best_pos[0], best_pos[1], self.max_movement_distance
            )
        return False

    def school_behavior(self) -> bool:
        if random.random() > self.schooling_tendency:
            return False
        nearby_fish = []
        search_radius = 4
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = self.x + dx
                ny = self.y + dy
                if (
                    0 <= nx < self.environment.width
                    and 0 <= ny < self.environment.height
                ):
                    agents = self.environment.get_agents_at(nx, ny)
                    for agent in agents:
                        if (
                            agent.__class__.__name__ == "Fish"
                            and agent != self
                            and agent.alive
                        ):
                            nearby_fish.append(agent)
        if len(nearby_fish) < 2:
            return False
        center_x = sum(fish.x for fish in nearby_fish) / len(nearby_fish)
        center_y = sum(fish.y for fish in nearby_fish) / len(nearby_fish)
        if abs(center_x - self.x) > 1 or abs(center_y - self.y) > 1:
            return self.move_towards(int(center_x), int(center_y), max_distance=2)
        return False

    def calculate_environmental_stress(self) -> float:
        conditions = self.get_environmental_conditions()
        temp_stress = min(
            abs(conditions["temperature"] - self.optimal_temperature)
            / self.temp_tolerance,
            1,
        )
        oxygen_stress = min(
            abs(conditions["oxygen"] - self.optimal_oxygen) / self.oxygen_tolerance, 1
        )
        ph_stress = min(abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance, 1)
        total_stress = temp_stress * 0.3 + oxygen_stress * 0.4 + ph_stress * 0.3
        return min(total_stress, 1.0)

    def can_reproduce(self) -> bool:
        return (
            self.energy > self.reproduction_threshold
            and self.reproduction_age_min <= self.age <= self.reproduction_age_max
            and self.calculate_environmental_stress() < 0.5
        )

    def reproduce(self) -> Optional["Fish"]:
        if not self.can_reproduce():
            return None
        suitable_positions = []
        search_radius = 3
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                new_x = max(0, min(self.x + dx, self.environment.width - 1))
                new_y = max(0, min(self.y + dy, self.environment.height - 1))
                conditions = self.environment.get_conditions(new_x, new_y)
                site_stress = (
                    min(
                        abs(conditions["temperature"] - self.optimal_temperature)
                        / self.temp_tolerance,
                        1,
                    )
                    * 0.3
                    + min(
                        abs(conditions["oxygen"] - self.optimal_oxygen)
                        / self.oxygen_tolerance,
                        1,
                    )
                    * 0.4
                    + min(
                        abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance, 1
                    )
                    * 0.3
                )
                if site_stress < 0.4:
                    if len(self.environment.get_agents_at(new_x, new_y)) < 3:
                        suitable_positions.append((new_x, new_y, site_stress))
        if not suitable_positions:
            return None
        best_pos = min(suitable_positions, key=lambda x: x[2])
        self.energy -= self.reproduction_cost
        offspring = Fish(
            self.environment, best_pos[0], best_pos[1], self.offspring_energy
        )
        return offspring

    def find_optimal_location(self, search_radius: int = 4) -> Tuple[int, int]:
        best_pos = (self.x, self.y)
        best_score = float("-inf")
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = max(0, min(self.x + dx, self.environment.width - 1))
                ny = max(0, min(self.y + dy, self.environment.height - 1))
                conditions = self.environment.get_conditions(nx, ny)
                temp_score = 1 - min(
                    abs(conditions["temperature"] - self.optimal_temperature)
                    / self.temp_tolerance,
                    1,
                )
                oxygen_score = 1 - min(
                    abs(conditions["oxygen"] - self.optimal_oxygen)
                    / self.oxygen_tolerance,
                    1,
                )
                ph_score = 1 - min(
                    abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance, 1
                )
                env_score = temp_score * 0.3 + oxygen_score * 0.4 + ph_score * 0.3
                prey_count = self.environment.get_agent_count_in_radius(
                    nx, ny, 2, Zooplankton
                )
                prey_score = min(prey_count / 15.0, 1.0)
                fish_count = self.environment.get_agent_count_in_radius(nx, ny, 2) - 1
                competition_score = 1 / (1 + fish_count * 0.3)
                total_score = (
                    env_score * 0.5 + prey_score * 0.3 + competition_score * 0.2
                )
                if total_score > best_score:
                    best_score = total_score
                    best_pos = (nx, ny)
        return best_pos

    def update(self) -> None:
        self.basic_update()
        if not self.alive:
            return
        if self.age > self.max_lifespan:
            self.die()
            return
        stress_level = self.calculate_environmental_stress()
        is_hungry = self.energy < self.hunger_threshold
        needs_migration = stress_level > self.migration_threshold

        if needs_migration:
            if self.migrate_to_better_conditions():
                return
        if is_hungry:
            energy_gained = self.hunt()
            if energy_gained == 0:
                if not self.seek_prey():
                    optimal_pos = self.find_optimal_location(search_radius=3)
                    if optimal_pos != (self.x, self.y):
                        self.move_towards(
                            optimal_pos[0], optimal_pos[1], self.max_movement_distance
                        )
                else:
                    self.hunt()
        else:
            behavior_choice = random.random()
            if behavior_choice < 0.4:
                self.hunt()
                if random.random() < 0.3:
                    self.seek_prey()
            elif behavior_choice < 0.6:
                self.school_behavior()
            elif behavior_choice < 0.8:
                optimal_pos = self.find_optimal_location(search_radius=3)
                if optimal_pos != (self.x, self.y):
                    self.move_towards(
                        optimal_pos[0], optimal_pos[1], self.max_movement_distance
                    )
            else:
                self.random_move(distance=2)
        offspring = self.reproduce()
        if stress_level > 0.9 and random.random() < 0.1:
            self.die()
        return offspring
