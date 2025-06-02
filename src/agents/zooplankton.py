"""
Zooplankton Agent Module

This module contains the Zooplankton class, representing the primary consumers
in the marine ecosystem that feed on phytoplankton and serve as food for fish.
"""

import random
from typing import Optional, List
from .base_agent import Agent
from .phytoplankton import Phytoplankton


class Zooplankton(Agent):
    def __init__(self, environment, x: int, y: int, energy: float = 10):
        super().__init__(environment, x, y, energy)

        self.optimal_temperature = 20.0
        self.temp_tolerance = 6.0
        self.optimal_oxygen = 6.0
        self.oxygen_tolerance = 2.0

        self.feeding_efficiency = 0.8
        self.energy_per_phytoplankton = 8
        self.hunting_range = 2
        self.max_feeding_per_step = 2

        self.reproduction_threshold = 25
        self.reproduction_cost = 15
        self.reproduction_age_min = 10
        self.reproduction_age_max = 150
        self.offspring_energy = 8

        self.base_metabolism = 0.08
        self.movement_cost = 0.03
        self.max_movement_distance = 3

        self.hunger_threshold = 15
        self.fear_threshold = 0.6

    def find_food(self) -> List[Phytoplankton]:
        prey_list = []
        for radius in range(1, self.hunting_range + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        nx = self.x + dx
                        ny = self.y + dy
                        if (
                            0 <= nx < self.environment.width
                            and 0 <= ny < self.environment.height
                        ):
                            phytoplankton = self.environment.get_agents_at(
                                nx, ny, Phytoplankton
                            )
                            prey_list.extend(phytoplankton)
            if prey_list:
                break
        return prey_list

    def feed(self) -> float:
        total_energy_gained = 0
        meals_consumed = 0
        local_prey = self.environment.get_agents_at(self.x, self.y, Phytoplankton)

        for prey in local_prey:
            if meals_consumed >= self.max_feeding_per_step:
                break
            if prey.alive:
                energy_gained = self.energy_per_phytoplankton * self.feeding_efficiency
                self.energy = min(self.energy + energy_gained, self.max_energy)
                total_energy_gained += energy_gained
                meals_consumed += 1
                prey.die()

        return total_energy_gained

    def seek_food(self) -> bool:
        prey_list = self.find_food()
        if not prey_list:
            return False

        closest_prey = None
        min_distance = float("inf")
        for prey in prey_list:
            if prey.alive:
                distance = abs(prey.x - self.x) + abs(prey.y - self.y)
                if distance < min_distance:
                    min_distance = distance
                    closest_prey = prey

        if closest_prey:
            return self.move_towards(
                closest_prey.x, closest_prey.y, self.max_movement_distance
            )
        return False

    def avoid_predators(self) -> bool:
        predator_positions = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx = self.x + dx
                ny = self.y + dy
                if (
                    0 <= nx < self.environment.width
                    and 0 <= ny < self.environment.height
                ):
                    agents = self.environment.get_agents_at(nx, ny)
                    for agent in agents:
                        if agent.__class__.__name__ == "Fish":
                            predator_positions.append((nx, ny))

        if not predator_positions:
            return False

        escape_x = self.x
        escape_y = self.y
        for pred_x, pred_y in predator_positions:
            if pred_x < self.x:
                escape_x += 1
            elif pred_x > self.x:
                escape_x -= 1
            if pred_y < self.y:
                escape_y += 1
            elif pred_y > self.y:
                escape_y -= 1

        escape_x = max(0, min(escape_x, self.environment.width - 1))
        escape_y = max(0, min(escape_y, self.environment.height - 1))
        return self.move(escape_x, escape_y)

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
        return min(temp_stress * 0.4 + oxygen_stress * 0.6, 1.0)

    def can_reproduce(self) -> bool:
        return (
            self.energy > self.reproduction_threshold
            and self.reproduction_age_min <= self.age <= self.reproduction_age_max
            and self.calculate_environmental_stress() < 0.6
        )

    def reproduce(self) -> Optional["Zooplankton"]:
        if not self.can_reproduce():
            return None

        offspring_positions = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                new_x = max(0, min(self.x + dx, self.environment.width - 1))
                new_y = max(0, min(self.y + dy, self.environment.height - 1))
                if (new_x, new_y) != (self.x, self.y):
                    offspring_positions.append((new_x, new_y))

        if not offspring_positions:
            return None

        best_pos = None
        min_crowding = float("inf")
        for pos in offspring_positions:
            crowding = len(self.environment.get_agents_at(pos[0], pos[1]))
            if crowding < min_crowding:
                min_crowding = crowding
                best_pos = pos

        if best_pos is None:
            best_pos = random.choice(offspring_positions)

        self.energy -= self.reproduction_cost
        return Zooplankton(
            self.environment, best_pos[0], best_pos[1], self.offspring_energy
        )

    def find_optimal_location(self, search_radius: int = 3) -> tuple:
        best_pos = (self.x, self.y)
        best_score = float("-inf")
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = max(0, min(self.x + dx, self.environment.width - 1))
                ny = max(0, min(self.y + dy, self.environment.height - 1))
                conditions = self.environment.get_conditions(nx, ny)
                env_score = 1 - self.calculate_environmental_stress()

                food_count = len(self.environment.get_agents_at(nx, ny, Phytoplankton))
                nearby_food = self.environment.get_agent_count_in_radius(
                    nx, ny, 1, Phytoplankton
                )
                food_score = min((food_count + nearby_food) / 10.0, 1.0)

                predator_count = 0
                for px in range(max(0, nx - 2), min(self.environment.width, nx + 3)):
                    for py in range(
                        max(0, ny - 2), min(self.environment.height, ny + 3)
                    ):
                        agents = self.environment.get_agents_at(px, py)
                        predator_count += sum(
                            1 for a in agents if a.__class__.__name__ == "Fish"
                        )

                safety_score = 1 / (1 + predator_count * 0.5)
                total_score = env_score * 0.3 + food_score * 0.5 + safety_score * 0.2

                if total_score > best_score:
                    best_score = total_score
                    best_pos = (nx, ny)
        return best_pos

    def update(self) -> None:
        self.basic_update()
        if not self.alive:
            return

        stress_level = self.calculate_environmental_stress()
        is_hungry = self.energy < self.hunger_threshold
        is_scared = stress_level > self.fear_threshold

        if is_scared:
            if not self.avoid_predators():
                optimal_pos = self.find_optimal_location(search_radius=2)
                if optimal_pos != (self.x, self.y):
                    self.move_towards(
                        optimal_pos[0], optimal_pos[1], self.max_movement_distance
                    )

        elif is_hungry:
            energy_gained = self.feed()
            if energy_gained == 0:
                self.seek_food()
                self.feed()

        else:
            if random.random() < 0.6:
                self.feed()
                if random.random() < 0.3:
                    self.seek_food()
            else:
                if random.random() < 0.7:
                    optimal_pos = self.find_optimal_location(search_radius=2)
                    if optimal_pos != (self.x, self.y):
                        self.move_towards(
                            optimal_pos[0], optimal_pos[1], self.max_movement_distance
                        )
                else:
                    self.random_move(distance=2)

        offspring = self.reproduce()
        if self.age > 300 or stress_level > 0.95:
            self.die()
        return offspring
