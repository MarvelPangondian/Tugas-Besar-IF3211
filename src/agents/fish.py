import random
from typing import Optional, List, Tuple
from .base_agent import Agent
from .zooplankton import Zooplankton


class Fish(Agent):
    """
    Scientifically accurate fish agent based on marine fish physiology research - V2
    """

    def __init__(self, environment, x: int, y: int, energy: float = 25):
        super().__init__(environment, x, y, energy)

        # Species-specific characteristics (simulate different marine fish species)
        self.species_type = random.choice(
            ["polar", "temperate", "tropical", "eurythermal"]
        )

        if self.species_type == "polar":
            self.optimal_temperature = random.uniform(0.0, 5.0)  # Cold-water
            self.temp_tolerance = random.uniform(12.0, 17.0)
        elif self.species_type == "temperate":
            self.optimal_temperature = random.uniform(6.0, 18.0)  # Mid-latitude species
            self.temp_tolerance = random.uniform(8.0, 13.0)
        elif self.species_type == "tropical":
            self.optimal_temperature = random.uniform(23.0, 30.0)  # Warm-water
            self.temp_tolerance = random.uniform(4.0, 8.0)
        else:
            self.optimal_temperature = random.uniform(12.0, 24.0)
            self.temp_tolerance = random.uniform(10.0, 18.0)

        # Critical thermal
        self.thermal_death_temp = (
            self.optimal_temperature + self.temp_tolerance + random.uniform(3, 6)
        )
        self.thermal_minimum = max(
            0, self.optimal_temperature - self.temp_tolerance - random.uniform(2, 4)
        )

        # Oxygen requirements
        self.optimal_oxygen = random.uniform(6.0, 8.5)
        self.oxygen_tolerance = random.uniform(1.5, 3.0)

        # Critical oxygen levels (Pcrit)
        self.critical_oxygen = random.uniform(2.0, 4.5)
        self.lethal_oxygen = random.uniform(0.5, 1.5)

        # pH tolerance
        self.optimal_ph = random.uniform(7.9, 8.2)
        self.ph_tolerance = random.uniform(0.2, 0.4)

        # Feeding parameters
        self.feeding_efficiency = random.uniform(0.65, 0.85)
        self.energy_per_zooplankton = random.uniform(8, 15)
        self.hunting_range = random.randint(2, 4)
        self.max_feeding_per_step = random.randint(2, 5)
        self.hunt_success_rate = random.uniform(0.3, 0.7)

        # Handling time and search efficiency
        self.handling_time_per_prey = 0.1
        self.search_efficiency = random.uniform(0.4, 0.8)

        # Reproduction parameters based on marine fish reproductive biology
        # Size-dependent reproduction (hyperallometric scaling)
        self.reproduction_threshold = random.uniform(45, 65)  # Higher energy threshold
        self.reproduction_cost = random.uniform(15, 25)

        # Maturation parameters based on life history research
        self.reproduction_age_min = random.randint(15, 35)  # Delayed maturation
        self.reproduction_age_max = random.randint(150, 300)
        self.offspring_energy = random.uniform(8, 15)

        # Spawning frequency and timing
        self.spawning_season_length = random.randint(30, 90)
        self.batch_spawning = random.random() < 0.7
        self.spawning_interval = random.randint(3, 14) if self.batch_spawning else 365
        self.last_spawning_time = -self.spawning_interval

        # Metabolic parameters with Q10 temperature scaling
        self.base_metabolism = random.uniform(0.08, 0.15)
        self.q10_metabolism = random.uniform(2.0, 2.8)  # Q10 for fish metabolism
        self.movement_cost = random.uniform(0.04, 0.08)
        self.max_movement_distance = random.randint(2, 5)

        # Behavioral parameters based on fish behavioral ecology
        self.hunger_threshold = random.uniform(20, 35)  # When to prioritize feeding
        self.predator_detection_range = random.randint(
            3, 6
        )  # Predator avoidance distance
        self.schooling_tendency = random.uniform(
            0.2, 0.6
        )  # Species variation in schooling
        self.territory_radius = random.randint(2, 5)  # Territorial behavior range

        # Stress and mortality parameters
        self.stress_mortality_threshold = 0.4
        self.maximum_lifespan = random.randint(200, 600)  # Variable lifespan by species
        self.senescence_age = self.maximum_lifespan * random.uniform(0.6, 0.8)

        # Hypoxia tolerance mechanisms
        self.hypoxia_tolerance = random.uniform(0.3, 0.8)
        self.oxygen_debt_capacity = random.uniform(5, 15)

        # Migration and habitat selection
        self.migration_threshold = random.uniform(0.6, 0.8)
        self.optimal_depth_preference = random.randint(5, 20)

        # Add feeding status tracking for post-feeding thermotaxis
        self.recently_fed = False
        self.last_feeding_time = 0

    def calculate_temperature_metabolic_effect(self, temperature: float) -> float:
        """
        Calculate metabolic rate adjustment due to temperature using Q10 scaling
        """
        temp_diff = (temperature - self.optimal_temperature) / 10.0

        # Q10 scaling - metabolic rate changes with temperature
        if temperature > 0:
            q10_factor = self.q10_metabolism**temp_diff
        else:
            q10_factor = 0.1  # Severely reduced at freezing

        # Additional stress at extreme temperatures
        if temperature > self.thermal_death_temp * 0.9:
            q10_factor *= 0.5  # Heat stress penalty
        elif temperature < self.thermal_minimum * 1.1:
            q10_factor *= 0.3  # Cold stress penalty

        return max(0.1, min(q10_factor, 4.0))

    def calculate_oxygen_stress(self, oxygen_level: float) -> float:
        """
        Calculate oxygen stress based on critical oxygen thresholds.
        """
        if oxygen_level <= self.lethal_oxygen:
            return 1.0  # Lethal hypoxia
        elif oxygen_level <= self.critical_oxygen:
            deficit = (self.critical_oxygen - oxygen_level) / self.critical_oxygen
            return 0.7 + 0.3 * deficit
        elif oxygen_level < self.optimal_oxygen - self.oxygen_tolerance:
            deficit = (self.optimal_oxygen - oxygen_level) / self.oxygen_tolerance
            return 0.3 * deficit
        else:
            return 0.0  # No oxygen stress

    def find_prey_realistic(self) -> List[Zooplankton]:
        """
        Realistic prey detection based on marine fish foraging ecology, V2.
        """
        prey_list = []
        search_success = random.random() < self.search_efficiency

        if not search_success:
            return prey_list

        # Search in realistic pattern (expanding circles)
        for radius in range(1, self.hunting_range + 1):
            circle_prey = []
            search_positions = []

            # Generate search positions in current radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius or radius == 1:
                        nx = self.x + dx
                        ny = self.y + dy
                        if (
                            0 <= nx < self.environment.width
                            and 0 <= ny < self.environment.height
                        ):
                            search_positions.append((nx, ny))

            # Search positions in random order (realistic foraging)
            random.shuffle(search_positions)

            for nx, ny in search_positions:
                zooplankton = self.environment.get_agents_at(nx, ny, Zooplankton)
                for prey in zooplankton:
                    if prey.alive:
                        detection_prob = 0.8
                        conditions = self.environment.get_conditions(nx, ny)
                        if conditions.get("light", 1.0) < 0.3:
                            detection_prob *= 0.6
                        if random.random() < detection_prob:
                            circle_prey.append(prey)
            if circle_prey:
                prey_list.extend(circle_prey)
                if len(prey_list) >= self.max_feeding_per_step * 2:
                    break
        return prey_list

    def hunt_realistic(self) -> float:
        """
        Realistic hunting behavior based on marine fish predation studies, V2.
        """
        total_energy_gained = 0
        meals_consumed = 0
        energy_spent_hunting = 0

        # Check for prey at current location first
        local_prey = self.environment.get_agents_at(self.x, self.y, Zooplankton)

        for prey in local_prey:
            if meals_consumed >= self.max_feeding_per_step:
                break

            if prey.alive:
                base_success = self.hunt_success_rate

                # Prey condition affects capture success
                prey_stress = prey.calculate_environmental_stress()
                prey_vulnerability = 0.5 + 0.5 * prey_stress

                # Environmental factors
                conditions = self.get_environmental_conditions()
                temp_factor = 1.0
                if (
                    conditions["temperature"]
                    < self.optimal_temperature - self.temp_tolerance
                ):
                    temp_factor = 0.7  # Reduced performance in cold
                elif (
                    conditions["temperature"]
                    > self.optimal_temperature + self.temp_tolerance
                ):
                    temp_factor = 0.6  # Reduced performance in heat

                # Final success probability
                success_prob = base_success * prey_vulnerability * temp_factor

                # Hunting attempt
                hunt_energy_cost = self.movement_cost * 0.5  # Energy cost per attempt
                energy_spent_hunting += hunt_energy_cost

                if random.random() < success_prob:
                    # Successful capture
                    handling_cost = self.handling_time_per_prey * self.base_metabolism

                    prey_energy_content = self.energy_per_zooplankton
                    if hasattr(prey, "energy"):
                        prey_condition = min(1.0, prey.energy / 10.0)
                        prey_energy_content *= 0.5 + 0.5 * prey_condition

                    net_energy_gain = (
                        prey_energy_content * self.feeding_efficiency - handling_cost
                    )

                    if net_energy_gain > 0:
                        self.energy = min(
                            self.energy + net_energy_gain, self.max_energy
                        )
                        total_energy_gained += net_energy_gain
                        meals_consumed += 1
                        prey.die()

        self.energy -= energy_spent_hunting

        if total_energy_gained > 0:
            self.recently_fed = True
            self.last_feeding_time = self.age

        if self.age - self.last_feeding_time > 5:
            self.recently_fed = False

        return total_energy_gained

    def calculate_environmental_stress(self) -> float:
        """
        Comprehensive environmental stress calculation based on multiple factors.
        """
        conditions = self.get_environmental_conditions()

        temp_stress = min(
            1.0,
            abs(conditions["temperature"] - self.optimal_temperature)
            / self.temp_tolerance,
        )

        # Oxygen stress
        oxygen_stress = self.calculate_oxygen_stress(conditions["oxygen"])

        # pH stress
        ph_stress = min(
            1.0, abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance
        )

        total_stress = (
            temp_stress * 0.35  # Temperature most critical
            + oxygen_stress * 0.45  # Oxygen second most critical
            + ph_stress * 0.20
        )  # pH important but less immediate

        return min(total_stress, 1.0)

    def can_reproduce_realistic(self) -> bool:
        """
        Reproduction criteria based on marine fish reproductive biology
        """
        # Age requirements
        if not (self.reproduction_age_min <= self.age <= self.reproduction_age_max):
            return False

        energy_threshold = self.reproduction_threshold
        if self.age > self.senescence_age:
            energy_threshold *= 1.2

        if self.energy < energy_threshold:
            return False

        # Environmental stress check
        stress_level = self.calculate_environmental_stress()
        if stress_level > 0.6:  # Cannot reproduce under high stress
            return False

        # Spawning timing (seasonal reproduction)
        time_since_last_spawn = self.age - self.last_spawning_time
        if time_since_last_spawn < self.spawning_interval:
            return False

        # Temperature-dependent spawning (many fish have thermal cues)
        conditions = self.get_environmental_conditions()
        temp_suitable = (
            self.optimal_temperature - self.temp_tolerance * 0.5
            <= conditions["temperature"]
            <= self.optimal_temperature + self.temp_tolerance * 0.5
        )

        if not temp_suitable:
            return False

        # Probabilistic spawning (not deterministic)
        spawning_probability = 0.05
        if self.batch_spawning:
            spawning_probability *= 2

        return random.random() < spawning_probability

    def reproduce_realistic(self) -> Optional["Fish"]:
        """
        Realistic reproduction based on marine fish spawning biology.
        """
        if not self.can_reproduce_realistic():
            return None

        # Find suitable spawning habitat
        suitable_positions = []
        search_radius = min(4, self.territory_radius)

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                new_x = max(0, min(self.x + dx, self.environment.width - 1))
                new_y = max(0, min(self.y + dy, self.environment.height - 1))

                # Check habitat suitability
                conditions = self.environment.get_conditions(new_x, new_y)

                # Oxygen requirement for eggs/larvae
                if conditions["oxygen"] < self.optimal_oxygen * 0.8:
                    continue

                # Temperature suitability
                temp_score = 1 - min(
                    1.0,
                    abs(conditions["temperature"] - self.optimal_temperature)
                    / self.temp_tolerance,
                )
                if temp_score < 0.6:
                    continue

                # Avoid overcrowded areas
                local_fish = self.environment.get_agents_at(new_x, new_y, Fish)
                if len(local_fish) > 2:
                    continue

                suitable_positions.append((new_x, new_y, temp_score))

        if not suitable_positions:
            return None

        best_site = max(suitable_positions, key=lambda x: x[2])

        # Pay reproduction cost
        reproduction_cost = self.reproduction_cost

        if self.batch_spawning:
            reproduction_cost *= 1.5

        self.energy -= reproduction_cost
        self.last_spawning_time = self.age

        offspring_energy = self.offspring_energy
        offspring = Fish(self.environment, best_site[0], best_site[1], offspring_energy)

        offspring.optimal_temperature = self.optimal_temperature + random.gauss(0, 0.5)
        offspring.temp_tolerance = max(3.0, self.temp_tolerance + random.gauss(0, 0.3))
        offspring.species_type = self.species_type
        offspring.optimal_oxygen = max(4.0, self.optimal_oxygen + random.gauss(0, 0.2))
        offspring.hunt_success_rate = max(
            0.1, min(0.9, self.hunt_success_rate + random.gauss(0, 0.05))
        )

        return offspring

    def metabolize_realistic(self) -> None:
        """
        Realistic metabolic calculations with temperature effects.
        """
        conditions = self.get_environmental_conditions()

        # Base metabolic cost
        base_cost = self.base_metabolism

        # Temperature effect on metabolism
        temp_factor = self.calculate_temperature_metabolic_effect(
            conditions["temperature"]
        )
        metabolic_cost = base_cost * temp_factor

        # Stress increases metabolic cost
        stress = self.calculate_environmental_stress()
        stress_cost = stress * 0.03  # Stress metabolism

        # Oxygen limitation effects
        if conditions["oxygen"] < self.critical_oxygen:
            # Below Pcrit - must use anaerobic metabolism (less efficient)
            metabolic_cost *= 1.5
            stress_cost += 0.02  # Additional stress from hypoxia

        # Total metabolic cost
        total_cost = metabolic_cost + stress_cost
        self.energy -= total_cost

    def update(self) -> Optional["Fish"]:
        """
        Main update method with scientifically accurate fish behavior.
        """
        self.basic_update()
        if not self.alive:
            return None

        # Use realistic metabolism
        self.metabolize_realistic()

        # Age-related mortality
        if self.age > self.maximum_lifespan:
            self.die()
            return None

        # Environmental stress effects
        conditions = self.get_environmental_conditions()
        stress_level = self.calculate_environmental_stress()

        # Lethal conditions
        if (
            conditions["temperature"] > self.thermal_death_temp
            or conditions["temperature"] < self.thermal_minimum
            or conditions["oxygen"] <= self.lethal_oxygen
        ):
            if random.random() < 0.4:  # 40% chance of death in lethal conditions
                self.die()
                return None

        # Stress-induced mortality
        if stress_level > self.stress_mortality_threshold:
            mortality_risk = (stress_level - self.stress_mortality_threshold) * 0.3
            if random.random() < mortality_risk:
                self.die()
                return None

        # Behavioral priorities
        is_hungry = self.energy < self.hunger_threshold
        high_stress = stress_level > self.migration_threshold
        needs_thermoregulation = self.calculate_thermal_stress() > 0.3

        offspring = None

        # Priority 1: Escape lethal conditions
        if high_stress or conditions["oxygen"] < self.critical_oxygen:
            self.migrate_to_better_conditions()

        # Priority 2: Behavioral thermoregulation (scientifically observed behavior)
        elif (
            needs_thermoregulation and random.random() < 0.4
        ):  # 40% chance to thermoregulate
            self.behavioral_thermoregulation()

        # Priority 3: Feeding behavior (most important for survival)
        elif is_hungry or random.random() < 0.6:  # 60% chance to hunt when not stressed
            energy_gained = self.hunt_realistic()
            if energy_gained == 0 and is_hungry:
                # No local food, search actively
                self.seek_food_realistic()

        # Priority 4: Reproduction (when conditions are good)
        if not high_stress and not is_hungry:
            offspring = self.reproduce_realistic()

        # Priority 5: Other behaviors
        if not high_stress and random.random() < 0.3:
            behavior_choice = random.random()
            if behavior_choice < 0.4:
                self.schooling_behavior()
            elif behavior_choice < 0.7:
                self.territorial_behavior()
            else:
                self.random_move(distance=random.randint(1, 2))

        return offspring

    def calculate_thermal_stress(self) -> float:
        """Calculate just the thermal component of stress for thermoregulation decisions."""
        conditions = self.get_environmental_conditions()
        temp_deviation = abs(conditions["temperature"] - self.optimal_temperature)
        return min(1.0, temp_deviation / self.temp_tolerance)

    def seek_food_realistic(self) -> bool:
        """Realistic food seeking with energy cost considerations."""
        prey_list = self.find_prey_realistic()
        if not prey_list:
            return False

        # Calculate prey density at different locations
        prey_patches = {}
        for prey in prey_list:
            if prey.alive:
                pos = (prey.x, prey.y)
                prey_patches[pos] = prey_patches.get(pos, 0) + 1

        if not prey_patches:
            return False

        # Choose location with highest prey density
        best_location = max(prey_patches.keys(), key=lambda pos: prey_patches[pos])

        # Only move if the energy cost is justified
        distance = abs(best_location[0] - self.x) + abs(best_location[1] - self.y)
        movement_cost = distance * self.movement_cost
        expected_gain = prey_patches[best_location] * self.energy_per_zooplankton * 0.5

        if expected_gain > movement_cost:
            return self.move_towards(
                best_location[0], best_location[1], self.max_movement_distance
            )

        return False

    def migrate_to_better_conditions(self) -> bool:
        """Migration to escape poor environmental conditions."""
        best_pos = self.find_optimal_habitat(search_radius=6)
        if best_pos != (self.x, self.y):
            return self.move_towards(
                best_pos[0], best_pos[1], self.max_movement_distance
            )
        return False

    def behavioral_thermoregulation(self) -> bool:
        """
        Fish actively seek optimal temperatures through vertical and horizontal movement
        """
        current_conditions = self.get_environmental_conditions()
        current_temp = current_conditions["temperature"]

        # Calculate temperature deviation from optimal
        temp_deviation = current_temp - self.optimal_temperature
        temp_stress = abs(temp_deviation) / self.temp_tolerance

        # Only thermoregulate if temperature stress is significant
        if temp_stress < 0.3:
            return False

        # Determine thermoregulatory strategy based on species and situation
        target_temp_adjustment = 0

        # Post-feeding thermotaxis - seek warmer water after feeding to aid digestion
        if hasattr(self, "recently_fed") and self.recently_fed and temp_deviation < 0:
            target_temp_adjustment = +2.0  # Seek 2°C warmer for digestion

        # Cold-water species avoiding heat stress
        elif (
            self.species_type == "temperate"
            and temp_deviation > self.temp_tolerance * 0.5
        ):
            target_temp_adjustment = -3.0  # Seek cooler water

        # General thermal preference toward optimal
        elif temp_deviation > 0:
            target_temp_adjustment = -1.0  # Seek cooler
        else:
            target_temp_adjustment = +1.0  # Seek warmer

        target_temp = self.optimal_temperature + target_temp_adjustment

        # Search for better thermal habitat (prioritize vertical movement)
        best_pos = self.find_thermal_habitat(target_temp, search_radius=5)

        if best_pos != (self.x, self.y):
            return self.move_towards(
                best_pos[0], best_pos[1], self.max_movement_distance
            )

        return False

    def find_thermal_habitat(
        self, target_temp: float, search_radius: int = 5
    ) -> Tuple[int, int]:
        """
        Find location with temperature closest to target, prioritizing vertical movement.
        """
        best_pos = (self.x, self.y)
        best_temp_score = float("-inf")

        # Prioritize vertical movement (depth changes) for thermoregulation
        search_positions = []

        # Add vertical positions first (higher priority)
        for dy in range(-search_radius, search_radius + 1):
            if dy != 0:  # Skip current position
                ny = max(0, min(self.y + dy, self.environment.height - 1))
                search_positions.append((self.x, ny, "vertical"))

        # Add horizontal positions second (lower priority)
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx != 0 or dy != 0:  # Skip current position
                    nx = max(0, min(self.x + dx, self.environment.width - 1))
                    ny = max(0, min(self.y + dy, self.environment.height - 1))
                    if (nx, ny) not in [(pos[0], pos[1]) for pos in search_positions]:
                        search_positions.append((nx, ny, "horizontal"))

        for nx, ny, movement_type in search_positions:
            conditions = self.environment.get_conditions(nx, ny)

            # Temperature score (primary factor for thermoregulation)
            temp_diff = abs(conditions["temperature"] - target_temp)
            temp_score = 1 / (1 + temp_diff)  # Higher score for closer temperature

            if movement_type == "vertical":
                temp_score *= 1.5

            if (
                conditions["temperature"] < self.thermal_minimum
                or conditions["temperature"] > self.thermal_death_temp
                or conditions["oxygen"] < self.lethal_oxygen
            ):
                continue

            # Additional factors
            oxygen_score = 1 - self.calculate_oxygen_stress(conditions["oxygen"])

            total_score = temp_score * 0.8 + oxygen_score * 0.2

            if total_score > best_temp_score:
                best_temp_score = total_score
                best_pos = (nx, ny)

        return best_pos

    def find_optimal_habitat(self, search_radius: int = 4) -> Tuple[int, int]:
        """Find optimal habitat considering multiple environmental factors."""
        best_pos = (self.x, self.y)
        best_score = float("-inf")

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = max(0, min(self.x + dx, self.environment.width - 1))
                ny = max(0, min(self.y + dy, self.environment.height - 1))

                conditions = self.environment.get_conditions(nx, ny)

                # Environmental quality scores
                temp_score = 1 - min(
                    1.0,
                    abs(conditions["temperature"] - self.optimal_temperature)
                    / self.temp_tolerance,
                )
                oxygen_score = 1 - self.calculate_oxygen_stress(conditions["oxygen"])
                ph_score = 1 - min(
                    1.0, abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance
                )

                env_score = temp_score * 0.4 + oxygen_score * 0.5 + ph_score * 0.1

                # Prey availability
                prey_count = self.environment.get_agent_count_in_radius(
                    nx, ny, 2, Zooplankton
                )
                food_score = min(prey_count / 10.0, 1.0)

                fish_count = (
                    self.environment.get_agent_count_in_radius(nx, ny, 2, Fish) - 1
                )
                competition_score = 1 / (1 + fish_count * 0.2)

                total_score = (
                    env_score * 0.6 + food_score * 0.25 + competition_score * 0.15
                )

                if total_score > best_score:
                    best_score = total_score
                    best_pos = (nx, ny)

        return best_pos

    def schooling_behavior(self) -> bool:
        """Realistic schooling behavior with benefits and costs."""
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
                    fish = self.environment.get_agents_at(nx, ny, Fish)
                    for other_fish in fish:
                        if (
                            other_fish != self
                            and other_fish.alive
                            and other_fish.species_type == self.species_type
                        ):
                            nearby_fish.append(other_fish)

        if len(nearby_fish) < 1:
            return False

        # Move toward school center with some randomness
        center_x = sum(fish.x for fish in nearby_fish) / len(nearby_fish)
        center_y = sum(fish.y for fish in nearby_fish) / len(nearby_fish)

        # Add noise to avoid perfect clustering
        target_x = int(center_x + random.uniform(-1, 1))
        target_y = int(center_y + random.uniform(-1, 1))

        return self.move_towards(target_x, target_y, max_distance=2)

    def territorial_behavior(self) -> bool:
        """Simple territorial behavior for dominant individuals."""
        if self.energy < self.reproduction_threshold * 0.8:
            return False  # Only territorial when in good condition

        # Check for intruders in territory
        intruders = []
        for dx in range(-self.territory_radius, self.territory_radius + 1):
            for dy in range(-self.territory_radius, self.territory_radius + 1):
                nx = self.x + dx
                ny = self.y + dy
                if (
                    0 <= nx < self.environment.width
                    and 0 <= ny < self.environment.height
                ):
                    fish = self.environment.get_agents_at(nx, ny, Fish)
                    for other_fish in fish:
                        if (
                            other_fish != self
                            and other_fish.alive
                            and other_fish.energy < self.energy * 0.8
                        ):  # Only chase smaller/weaker fish
                            intruders.append(other_fish)

        if intruders:
            # Chase away closest intruder
            closest = min(
                intruders, key=lambda f: abs(f.x - self.x) + abs(f.y - self.y)
            )
            # Move toward intruder (they will likely move away)
            return self.move_towards(closest.x, closest.y, max_distance=1)

        return False

    def get_fish_info(self) -> dict:
        """
        Get detailed information about this fish for analysis.
        """
        base_info = self.get_agent_info()

        fish_info = {
            **base_info,
            "species_type": self.species_type,
            "optimal_temperature": self.optimal_temperature,
            "thermal_death_temp": self.thermal_death_temp,
            "critical_oxygen": self.critical_oxygen,
            "hunt_success_rate": self.hunt_success_rate,
            "reproduction_threshold": self.reproduction_threshold,
            "can_reproduce": self.can_reproduce_realistic(),
            "spawning_interval": self.spawning_interval,
            "last_spawning_time": self.last_spawning_time,
            "hypoxia_tolerance": self.hypoxia_tolerance,
            "senescence_age": self.senescence_age,
            "territory_radius": self.territory_radius,
        }

        return fish_info
