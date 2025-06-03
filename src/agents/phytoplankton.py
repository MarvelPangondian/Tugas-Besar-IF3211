import random
import math
import numpy as np
from typing import Dict, Optional, Tuple
from .base_agent import Agent


class Phytoplankton(Agent):
    """
    Phytoplankton Version 2.0
    Simulates phytoplankton as primary producers in the marine ecosystem.

    This agent photosynthesizes, responds to environmental stress (temperature, pH, light),
    reproduces via binary fission, and can vertically migrate to optimize conditions.
    """

    def __init__(self, environment, x: int, y: int, energy: float = 5):
        super().__init__(environment, x, y, energy)

        self.max_energy_storage = energy * 6

        # Assign one of several phytoplankton types, which affect physiology
        self.phyto_type = random.choice(
            ["diatom", "dinoflagellate", "coccolithophore", "flagellate"]
        )

        # Thermal preferences and lethal thresholds based on empirical observations
        self.optimal_temperature = random.uniform(18, 23)
        self.temp_tolerance = random.uniform(8, 12)
        self.thermal_death_temp = self.optimal_temperature + self.temp_tolerance + 5
        self.thermal_minimum = max(
            0, self.optimal_temperature - self.temp_tolerance - 2
        )

        # pH sensitivity (most phytoplankton optimized for 8.0–8.2 seawater)
        self.optimal_ph = 8.1
        self.ph_tolerance = 0.4

        # Photosynthesis parameters based on Jassby-Platt P-I (photosynthesis-irradiance) model
        self.alpha = random.uniform(0.05, 0.15)  # Initial slope of P-I curve
        self.P_max_ref = random.uniform(8, 25)  # Max rate at reference temperature
        self.beta = random.uniform(0.01, 0.05)  # Controls photoinhibition at high light
        self.I_sat = self.P_max_ref / self.alpha  # Light saturation intensity
        self.I_comp = 5.0  # Light compensation point (threshold for net gain)

        # Temperature sensitivity using Q10 scaling (rate doubles per +10°C)
        self.q10_photosynthesis = random.uniform(1.8, 2.5)
        self.q10_respiration = random.uniform(2.0, 3.0)

        # Energy conversion and capacity
        self.photosynthetic_efficiency = random.uniform(0.08, 0.15)
        self.max_energy_storage = energy * 4
        self.maintenance_respiration = 0.01  # maintenance v2

        # Reproduction thresholds and energy cost
        self.division_energy_threshold = 3
        self.division_energy_cost = 3.5
        self.max_division_rate = 1.5  # Max 1.5 divisions per day
        self.base_division_probability = 0.40

        # Cell size affects carbon content and sinking
        self.cell_size = random.uniform(5, 50)  # μm diameter
        self.carbon_content = (self.cell_size**2.85) * 0.003

        # Motility assignment and vertical swimming speed
        self.is_motile = (
            random.random() < 0.6
            if self.phyto_type != "diatom"
            else random.random() < 0.2
        )
        self.swimming_speed = random.uniform(0.5, 5.0) if self.is_motile else 0
        self.sinking_rate = self.calculate_sinking_rate()

        # Vertical migration preference
        self.migration_amplitude = random.randint(3, 8) if self.is_motile else 0
        self.optimal_light_level = random.uniform(50, 200)
        self.migration_energy_cost = 0.01

        # Mortality characteristics - v2
        self.max_lifespan = random.randint(60, 150)  # Increase from 30-100
        self.viral_lysis_rate = 0.02  # Reduce from 0.14 to 2%
        self.senescence_age = self.max_lifespan * 0.8  # Increase from 0.7

        # Nutrient limitation threshold for growth suppression
        self.nutrient_limitation_threshold = 0.3

        # Track light exposure, growth, and division history
        self.last_division_time = 0
        self.accumulated_light_damage = 0
        self.current_growth_rate = 0

    def calculate_sinking_rate(self) -> float:
        """
        Estimate sinking rate based on size and structure.

        Diatoms sink faster due to silica shells. Some species have buoyancy adaptations
        that reduce their sinking velocity.
        """
        if self.phyto_type == "diatom":
            base_sinking = (self.cell_size**2) * 0.0005
        else:
            base_sinking = (self.cell_size**2) * 0.0002

        if random.random() < 0.3:  # 30% chance of buoyancy regulation
            base_sinking *= 0.3

        return min(base_sinking, 10.0)

    def calculate_light_irradiance(self) -> float:
        """
        Use Beer's Law to compute light availability at the agent's depth.

        Light declines exponentially with depth using a fixed attenuation coefficient.
        """
        surface_light = self.environment.get_conditions(self.x, 0)["light"]
        k_d = 0.2  # Typical for coastal waters
        depth_fraction = self.y / self.environment.height
        actual_depth = depth_fraction * 50  # Scale to 50m total ocean depth
        return max(surface_light * 2000 * math.exp(-k_d * actual_depth), 0.1)

    def photosynthesis_jassby_platt(
        self, irradiance: float, temperature: float
    ) -> float:
        """
        Calculate photosynthetic energy gain using the Jassby-Platt model with Q10 temperature correction.

        - Applies temperature scaling (Q10).
        - Incorporates light saturation and photoinhibition.
        - Applies light compensation threshold.
        """
        temp_diff = (temperature - self.optimal_temperature) / 10.0
        temp_factor = self.q10_photosynthesis**temp_diff

        if temperature < self.thermal_minimum or temperature > self.thermal_death_temp:
            temp_factor *= 0.1  # Severe inhibition

        P_max_temp = self.P_max_ref * temp_factor

        if irradiance <= 0 or P_max_temp <= 0:
            return 0

        I_norm = self.alpha * irradiance / P_max_temp
        light_response = math.tanh(I_norm)
        photoinhibition = math.exp(-self.beta * irradiance / P_max_temp)
        P_rate = P_max_temp * light_response * photoinhibition

        energy_gain = P_rate * self.photosynthetic_efficiency * 0.1
        return max(0, energy_gain) if irradiance >= self.I_comp else 0

    def calculate_respiration(self, temperature: float) -> float:
        """
        Calculate respiratory energy loss with temperature dependence.
        """
        temp_diff = (temperature - self.optimal_temperature) / 10.0
        resp_temp_factor = self.q10_respiration**temp_diff

        base_respiration = self.maintenance_respiration * resp_temp_factor

        # Add stress-related respiration
        stress = self.calculate_environmental_stress()
        stress_respiration = stress * 0.02

        return base_respiration + stress_respiration

    def perform_vertical_migration(self) -> bool:
        """
        Vertical migration behavior for motile phytoplankton.

        Based on Wirtz & Smith (2020) - many phytoplankton actively migrate to optimize
        light exposure and nutrient acquisition.
        """
        if not self.is_motile or self.migration_amplitude == 0:
            return False

        current_irradiance = self.calculate_light_irradiance()

        # Simple light-seeking behavior
        moved = False

        if current_irradiance < self.optimal_light_level * 0.5:
            # Too dark - move toward surface (decreasing y)
            target_y = max(
                0, self.y - random.randint(1, min(2, self.migration_amplitude))
            )
            energy_cost = abs(self.y - target_y) * self.migration_energy_cost

            if self.energy > energy_cost:
                if self.move(self.x, target_y):
                    self.energy -= energy_cost
                    moved = True

        elif current_irradiance > self.optimal_light_level * 2:
            # Too bright - move deeper (increasing y) to avoid photoinhibition
            max_depth = min(self.environment.height - 1, self.y + random.randint(1, 2))
            target_y = max(self.y, max_depth)
            energy_cost = abs(self.y - target_y) * self.migration_energy_cost

            if self.energy > energy_cost:
                if self.move(self.x, target_y):
                    self.energy -= energy_cost
                    moved = True

        return moved

    def passive_sinking(self) -> bool:
        """
        Passive sinking for non-motile or low-energy phytoplankton.
        """
        if self.is_motile and self.energy > 2:
            return False  # Can actively swim

        # Stochastic sinking based on sinking rate
        sinking_probability = self.sinking_rate / 10.0  # Convert m/day to probability

        if random.random() < sinking_probability:
            new_y = min(self.environment.height - 1, self.y + 1)
            return self.move(self.x, new_y)

        return False

    def can_divide(self) -> bool:
        """
        Determine if phytoplankton can reproduce based on realistic criteria.
        """
        # Energy requirement
        if self.energy < self.division_energy_threshold:
            return False

        # Environmental stress check
        stress = self.calculate_environmental_stress()
        if stress > 0.7:
            return False

        # Light requirement (must have had recent photosynthesis)
        current_irradiance = self.calculate_light_irradiance()
        if current_irradiance < self.I_comp:
            return False

        # Age check (young cells divide more readily)
        age_factor = max(0.1, 1 - (self.age / self.senescence_age))

        # Division probability based on conditions and maximum rate
        division_prob = self.base_division_probability * age_factor * (1 - stress)

        return random.random() < division_prob

    def divide(self) -> Optional["Phytoplankton"]:
        """
        Cell division (binary fission) - primary reproduction method.

        Based on realistic phytoplankton reproduction biology.
        """
        if not self.can_divide():
            return None

        # Find suitable nearby location
        offspring_positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                new_x = max(0, min(self.x + dx, self.environment.width - 1))
                new_y = max(0, min(self.y + dy, self.environment.height - 1))

                if (new_x, new_y) != (self.x, self.y):
                    # Check light availability at this position
                    test_light = self.environment.get_conditions(new_x, new_y)["light"]
                    if test_light > 0.1:  # Minimum light for survival
                        offspring_positions.append((new_x, new_y))

        if not offspring_positions:
            return None

        # Choose best position (highest light, closest to optimal)
        best_pos = offspring_positions[0]
        best_score = 0

        for pos in offspring_positions:
            conditions = self.environment.get_conditions(pos[0], pos[1])
            light_score = min(1.0, conditions["light"])
            temp_score = 1 - min(
                1.0,
                abs(conditions["temperature"] - self.optimal_temperature)
                / self.temp_tolerance,
            )
            total_score = light_score * 0.7 + temp_score * 0.3

            if total_score > best_score:
                best_score = total_score
                best_pos = pos

        # Pay division cost (energy is split between parent and offspring)
        division_cost = self.division_energy_cost
        self.energy -= division_cost

        # Create offspring with realistic energy allocation
        offspring_energy = division_cost * 0.8  # Offspring gets most of the energy
        offspring = Phytoplankton(
            self.environment, best_pos[0], best_pos[1], offspring_energy
        )

        # Inherit some parental traits with mutation
        offspring.optimal_temperature = self.optimal_temperature + random.gauss(0, 0.5)
        offspring.temp_tolerance = max(3.0, self.temp_tolerance + random.gauss(0, 0.3))
        offspring.phyto_type = self.phyto_type
        offspring.P_max_ref = max(1.0, self.P_max_ref + random.gauss(0, 1.0))
        offspring.alpha = max(0.01, self.alpha + random.gauss(0, 0.01))

        # Update division timing
        self.last_division_time = 0

        return offspring

    def calculate_environmental_stress(self) -> float:
        """
        Calculate environmental stress from multiple factors.
        """
        conditions = self.get_environmental_conditions()

        # Temperature stress
        temp_stress = min(
            1.0,
            abs(conditions["temperature"] - self.optimal_temperature)
            / self.temp_tolerance,
        )

        # pH stress
        ph_stress = min(
            1.0, abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance
        )

        # Light stress (both too little and too much)
        current_irradiance = self.calculate_light_irradiance()
        if current_irradiance < self.I_comp:
            light_stress = 1.0  # Severe stress below compensation point
        elif current_irradiance > self.I_sat * 3:
            light_stress = 0.5  # Moderate stress from photoinhibition
        else:
            light_stress = 0.0

        # Combined stress with appropriate weightings
        total_stress = temp_stress * 0.3 + ph_stress * 0.2 + light_stress * 0.5

        return min(total_stress, 1.0)

    def calculate_growth_rate(self) -> float:
        """
        Calculate current growth rate based on environmental conditions.
        Used for monitoring population dynamics.
        """
        conditions = self.get_environmental_conditions()
        irradiance = self.calculate_light_irradiance()

        # Photosynthetic energy gain
        photo_rate = self.photosynthesis_jassby_platt(
            irradiance, conditions["temperature"]
        )

        # Respiratory energy loss
        resp_rate = self.calculate_respiration(conditions["temperature"])

        # Net growth rate
        net_rate = photo_rate - resp_rate

        # Environmental stress reduces growth
        stress = self.calculate_environmental_stress()
        growth_rate = net_rate * (1 - stress)

        return max(0, growth_rate)

    def update(self) -> Optional["Phytoplankton"]:
        """
        Main update method with realistic phytoplankton behavior.
        """
        self.basic_update()
        if not self.alive:
            return None

        conditions = self.get_environmental_conditions()

        # Age-related mortality (most phytoplankton live days to weeks)
        if self.age > self.max_lifespan:
            self.die()
            return None

        # Extreme temperature mortality
        if (
            conditions["temperature"] > self.thermal_death_temp
            or conditions["temperature"] < self.thermal_minimum
        ):
            if random.random() < 0.3:  # 30% chance of death at lethal temperatures
                self.die()
                return None

        # Calculate current light and photosynthesis
        irradiance = self.calculate_light_irradiance()
        photo_energy = self.photosynthesis_jassby_platt(
            irradiance, conditions["temperature"]
        )

        # Respiration (always occurring)
        resp_loss = self.calculate_respiration(conditions["temperature"])

        # Net energy change
        net_energy = photo_energy - resp_loss
        self.energy = max(0, min(self.energy + net_energy, self.max_energy_storage))

        # Update growth rate for monitoring
        self.current_growth_rate = self.calculate_growth_rate()

        # Behavioral decisions based on current state
        offspring = None

        # Priority 1: Vertical migration (if motile)
        if self.is_motile and random.random() < 0.4:  # 40% chance to attempt migration
            self.perform_vertical_migration()

        # Priority 2: Passive sinking (if not motile or low energy)
        elif random.random() < 0.3:
            self.passive_sinking()

        # Priority 3: Random horizontal movement (rare)
        elif self.is_motile and random.random() < 0.1:
            self.random_move(distance=1)

        # Reproduction attempt
        offspring = self.divide()

        # Mortality processes

        # 1. Viral lysis (major cause of phytoplankton mortality) - v2
        if random.random() < self.viral_lysis_rate / 500:  # Change from /100 to /500
            self.die()
            return offspring

        # 2. Senescence (age-related decline)
        if self.age > self.senescence_age:
            senescence_mortality = (
                (self.age - self.senescence_age)
                / (self.max_lifespan - self.senescence_age)
            ) * 0.05  # Reduce from 0.1 to 0.05
            if random.random() < senescence_mortality:
                self.die()
                return offspring

        # 3. Stress-induced mortality - v2
        stress = self.calculate_environmental_stress()
        if stress > 0.9:  # Increase threshold from 0.8 to 0.9
            stress_mortality = (stress - 0.9) * 0.1  # Reduce from 0.25 to 0.1
            if random.random() < stress_mortality:
                self.die()
                return offspring

        # 4. Energy depletion mortality - v2
        if self.energy < 0.2:  # Reduce threshold from 0.5 to 0.2
            if random.random() < 0.05:  # Reduce from 0.1 to 0.05
                self.die()
                return offspring

        # 5. Light limitation mortality (prolonged darkness) - v2
        if irradiance < self.I_comp:
            if random.random() < 0.01:  # Reduce from 0.02 to 0.01
                self.die()
                return offspring

        return offspring

    def get_phytoplankton_info(self) -> Dict:
        """
        Get detailed information about this phytoplankton for analysis.
        """
        base_info = self.get_agent_info()

        phyto_info = {
            **base_info,
            "phyto_type": self.phyto_type,
            "cell_size": self.cell_size,
            "carbon_content": self.carbon_content,
            "is_motile": self.is_motile,
            "current_growth_rate": self.current_growth_rate,
            "irradiance": self.calculate_light_irradiance(),
            "sinking_rate": self.sinking_rate,
            "optimal_temperature": self.optimal_temperature,
            "P_max": self.P_max_ref,
            "alpha": self.alpha,
            "can_divide": self.can_divide(),
        }

        return phyto_info
