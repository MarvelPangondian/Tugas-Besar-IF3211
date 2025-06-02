"""
Phytoplankton Agent Module

This module contains the Phytoplankton class, representing the primary producers
in the marine ecosystem that use photosynthesis to convert light and nutrients
into energy.

Updated to use scientifically accurate photosynthesis models based on:
- Jassby, A.D. & Platt, T. (1976). Mathematical formulation of the relationship between
photosynthesis and light for phytoplankton. Limnology and Oceanography, 21(4), 540-547.
- Platt, T., Gallegos, C.L., & Harrison, W.G. (1980). Photoinhibition of photosynthesis
in natural assemblages of marine phytoplankton. Journal of Marine Research, 38, 687-701.
"""

import random
import numpy as np
from typing import Dict
from .base_agent import Agent


class Phytoplankton(Agent):
    def __init__(self, environment, x: int, y: int, energy: float = 5):
        super().__init__(environment, x, y, energy)

        self.optimal_temperature = 20.0
        self.temp_tolerance = 6.0
        self.optimal_ph = 8.1
        self.ph_tolerance = 0.4

        self.alpha = 0.8
        self.P_max_ref = 15.0
        self.beta = 0.02
        self.Q10 = 2.0
        self.photosynthesis_efficiency = 0.85

        self.reproduction_threshold = 8
        self.division_energy_cost = 4
        self.reproduction_age_min = 2
        self.reproduction_age_max = 150

        self.base_metabolism = 0.05
        self.max_photosynthesis_rate = 4.0

    def photosynthesize(self) -> float:
        conditions = self.get_environmental_conditions()
        irradiance = conditions["light"] * 2000
        temp_factor = self._calculate_temperature_effect(conditions["temperature"])
        P_max_adjusted = self.P_max_ref * temp_factor

        if irradiance <= 0 or P_max_adjusted <= 0:
            photosynthetic_rate = 0
        else:
            photosynthetic_rate = P_max_adjusted * np.tanh(
                (self.alpha * irradiance) / P_max_adjusted
            )

        ph_factor = self._calculate_ph_effect(conditions["ph"])
        nutrient_factor = 0.9
        final_rate = photosynthetic_rate * ph_factor * nutrient_factor

        max_possible_rate = 20.0
        energy_gain = (
            (final_rate / max_possible_rate)
            * self.max_photosynthesis_rate
            * self.photosynthesis_efficiency
        )
        self.energy = min(self.energy + energy_gain, self.max_energy)

        return energy_gain

    def _calculate_temperature_effect(self, temperature: float) -> float:
        temp_diff = (temperature - self.optimal_temperature) / 10.0
        temp_diff = np.clip(temp_diff, -3, 3)
        temp_factor = self.Q10**temp_diff
        if temperature < 0 or temperature > 35:
            temp_factor *= 0.1
        return max(0, temp_factor)

    def _calculate_ph_effect(self, ph: float) -> float:
        ph_variance = self.ph_tolerance
        ph_stress = ((ph - self.optimal_ph) / ph_variance) ** 2
        ph_factor = np.exp(-0.5 * ph_stress)
        if ph < self.optimal_ph and ph > 7.5:
            co2_enhancement = 0.15 * (self.optimal_ph - ph)
            ph_factor += co2_enhancement
        return np.clip(ph_factor, 0, 1.2)

    def _photoinhibition_model(self, irradiance: float, P_max: float) -> float:
        if irradiance <= 0 or P_max <= 0:
            return 0
        normalized_irradiance = self.alpha * irradiance / P_max
        if normalized_irradiance > 50:
            return 0
        light_response = normalized_irradiance / np.sqrt(1 + normalized_irradiance**2)
        inhibition = np.exp(-self.beta * irradiance / P_max)
        return P_max * light_response * inhibition

    def calculate_environmental_stress(self) -> float:
        conditions = self.get_environmental_conditions()
        temp_stress = min(
            abs(conditions["temperature"] - self.optimal_temperature)
            / self.temp_tolerance,
            1,
        )
        ph_stress = min(abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance, 1)
        optimal_light = 0.7
        light_tolerance = 0.4
        light_stress = min(
            abs(conditions["light"] - optimal_light) / light_tolerance, 1
        )
        total_stress = temp_stress * 0.4 + ph_stress * 0.4 + light_stress * 0.2
        return min(total_stress, 1.0)

    def can_reproduce(self) -> bool:
        return (
            self.energy > self.reproduction_threshold
            and self.reproduction_age_min <= self.age <= self.reproduction_age_max
            and self.calculate_environmental_stress() < 0.8
        )

    def reproduce(self) -> "Phytoplankton":
        if not self.can_reproduce():
            return None
        offspring_positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                new_x = max(0, min(self.x + dx, self.environment.width - 1))
                new_y = max(0, min(self.y + dy, self.environment.height - 1))
                if (new_x, new_y) != (self.x, self.y):
                    offspring_positions.append((new_x, new_y))
        if not offspring_positions:
            return None
        new_x, new_y = random.choice(offspring_positions)
        self.energy -= self.division_energy_cost
        offspring_energy = self.division_energy_cost * 0.6
        offspring = Phytoplankton(self.environment, new_x, new_y, offspring_energy)
        return offspring

    def find_optimal_location(self, search_radius: int = 2) -> tuple:
        best_pos = (self.x, self.y)
        best_score = float("-inf")
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = max(0, min(self.x + dx, self.environment.width - 1))
                ny = max(0, min(self.y + dy, self.environment.height - 1))
                conditions = self.environment.get_conditions(nx, ny)
                irradiance = conditions["light"] * 2000
                temp_factor = self._calculate_temperature_effect(
                    conditions["temperature"]
                )
                P_max_adjusted = self.P_max_ref * temp_factor
                if irradiance > 0 and P_max_adjusted > 0:
                    photo_rate = P_max_adjusted * np.tanh(
                        (self.alpha * irradiance) / P_max_adjusted
                    )
                    ph_factor = self._calculate_ph_effect(conditions["ph"])
                    total_score = photo_rate * ph_factor
                else:
                    total_score = 0
                if total_score > best_score:
                    best_score = total_score
                    best_pos = (nx, ny)
        return best_pos

    def get_photosynthetic_capacity(self) -> Dict[str, float]:
        conditions = self.get_environmental_conditions()
        irradiance = conditions["light"] * 2000
        temp_factor = self._calculate_temperature_effect(conditions["temperature"])
        ph_factor = self._calculate_ph_effect(conditions["ph"])
        P_max_current = self.P_max_ref * temp_factor
        if irradiance > 0 and P_max_current > 0:
            current_rate = (
                P_max_current
                * np.tanh((self.alpha * irradiance) / P_max_current)
                * ph_factor
            )
        else:
            current_rate = 0
        return {
            "alpha": self.alpha,
            "P_max_reference": self.P_max_ref,
            "P_max_current": P_max_current,
            "current_photosynthetic_rate": current_rate,
            "temperature_factor": temp_factor,
            "ph_factor": ph_factor,
            "irradiance": irradiance,
            "efficiency": current_rate / P_max_current if P_max_current > 0 else 0,
        }

    def update(self) -> None:
        self.basic_update()
        if not self.alive:
            return None
        self.photosynthesize()
        
        if random.random() < 0.2:
            if random.random() < 0.7:
                optimal_pos = self.find_optimal_location(search_radius=1)
                if optimal_pos != (self.x, self.y):
                    self.move_towards(optimal_pos[0], optimal_pos[1], max_distance=1)
            else:
                self.random_move(distance=1)
        if self.age > 200 or self.calculate_environmental_stress() > 0.9:
            self.die()
        return self.reproduce()
        