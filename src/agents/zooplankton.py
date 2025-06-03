import random
import math
from typing import Optional, List, Tuple
from .base_agent import Agent
from .phytoplankton import Phytoplankton


class Zooplankton(Agent):
    """
    Zooplankton Agent based on marine copepod biology.
    """

    def __init__(self, environment, x: int, y: int, energy: float = 12):
        super().__init__(environment, x, y, energy)

        # Environmental tolerances based on marine copepod literature
        self.optimal_temperature = 18.0  # Slightly cooler optimal
        self.temp_tolerance = 8.0
        self.thermal_death_temp = 32.0

        # Oxygen requirements - copepods are sensitive to hypoxia
        self.optimal_oxygen = 6.5
        self.oxygen_tolerance = 2.5
        self.critical_oxygen = 2.0

        # pH tolerance - less sensitive than fish but still affected
        self.optimal_ph = 8.0
        self.ph_tolerance = 0.5

        # Feeding parameters based on copepod literature
        self.feeding_efficiency = 0.75
        self.energy_per_phytoplankton = 6
        self.hunting_range = 2
        self.max_feeding_per_step = 3
        self.clearance_rate = 2.5

        # Reproduction parameters based on copepod biology
        self.reproduction_threshold = 18
        self.reproduction_cost = 8
        self.reproduction_age_min = 5
        self.reproduction_age_max = 120
        self.offspring_energy = 6
        self.egg_production_rate = 0.05

        # Metabolic parameters with temperature scaling
        self.base_metabolism = 0.06
        self.q10_metabolism = 2.5
        self.movement_cost = 0.04
        self.max_movement_distance = 3

        # Diel vertical migration parameters
        self.dvm_amplitude = 8
        self.surface_feeding_preference = 0.8
        self.deep_refuge_depth = None
        self.migration_energy_cost = 0.02

        # Behavioral parameters
        self.hunger_threshold = 12
        self.predator_detection_range = 3
        self.schooling_tendency = 0.4
        self.optimal_density = 2

        # Stress and mortality
        self.stress_mortality_threshold = 0.4
        self.maximum_lifespan = 200

        # Initialize DVM state
        self.current_migration_phase = "neutral"  # "up", "down", "neutral"
        self.target_depth = y

    def calculate_temperature_effect_on_metabolism(self, temperature: float) -> float:
        """
        Calculate temperature effect on metabolism using Q10 scaling.
        """
        temp_diff = (temperature - self.optimal_temperature) / 10.0

        # Q10 scaling
        if temperature > 0:
            temp_factor = self.q10_metabolism**temp_diff
        else:
            temp_factor = 0.1

        # Additional stress at extreme temperatures
        if temperature > 28 or temperature < 2:
            temp_factor *= 0.5  # Additional stress penalty

        return max(0.1, temp_factor)

    def get_current_light_phase(self) -> str:
        """
        Determine current light phase for DVM behavior.
        """
        # Simple day/night cycle - 24 step cycle
        time_of_day = self.environment.time % 24

        if 6 <= time_of_day <= 18:  # Day
            return "day"
        elif time_of_day <= 6 or time_of_day >= 18:  # Night
            return "night"
        else:
            return "twilight"

    def perform_diel_vertical_migration(self) -> bool:
        """
        Implement diel vertical migration (DVM) behavior.
        """
        light_phase = self.get_current_light_phase()
        environment_height = self.environment.height

        # Define target depths based on light phase
        surface_layer = max(1, int(environment_height * 0.1))  # Top 10%
        deep_layer = min(
            environment_height - 1, int(environment_height * 0.6)
        )  # 60% depth

        moved = False

        if light_phase == "night":
            # Migrate upward for feeding
            if self.y > surface_layer:
                target_y = max(surface_layer, self.y - random.randint(1, 3))
                energy_cost = abs(self.y - target_y) * self.migration_energy_cost

                if self.energy > energy_cost:
                    if self.move(self.x, target_y):
                        self.energy -= energy_cost
                        moved = True

        elif light_phase == "day":
            # Migrate downward to avoid predators
            if self.y < deep_layer:
                target_y = min(deep_layer, self.y + random.randint(1, 2))
                energy_cost = abs(self.y - target_y) * self.migration_energy_cost

                if self.energy > energy_cost:
                    if self.move(self.x, target_y):
                        self.energy -= energy_cost
                        moved = True

        return moved

    def find_food(self) -> List[Phytoplankton]:
        """
        Find phytoplankton prey using realistic search behavior.
        Copepods use both active and passive feeding strategies.
        """
        prey_list = []

        # Search in expanding circles (copepod feeding behavior)
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
                            phytoplankton = self.environment.get_agents_at(
                                nx, ny, Phytoplankton
                            )
                            circle_prey.extend(phytoplankton)

            if circle_prey:
                prey_list.extend(circle_prey)
                break  # Found food in this radius, don't search further

        return [prey for prey in prey_list if prey.alive]

    def feed(self) -> float:
        """
        Feeding behavior based on copepod clearance rates and selectivity.
        """
        total_energy_gained = 0
        meals_consumed = 0

        # Local feeding - copepods can feed on what's immediately available
        local_prey = self.environment.get_agents_at(self.x, self.y, Phytoplankton)

        for prey in local_prey:
            if meals_consumed >= self.max_feeding_per_step:
                break

            if prey.alive:
                # Size selectivity - copepods prefer certain phytoplankton sizes
                feeding_success = 0.8  # Base success rate

                # Reduce feeding efficiency if prey is stressed (lower quality)
                prey_stress = prey.calculate_environmental_stress()
                feeding_success *= 1 - prey_stress * 0.3

                if random.random() < feeding_success:
                    energy_gained = (
                        self.energy_per_phytoplankton * self.feeding_efficiency
                    )
                    self.energy = min(self.energy + energy_gained, self.max_energy)
                    total_energy_gained += energy_gained
                    meals_consumed += 1
                    prey.die()

        return total_energy_gained

    def seek_food(self) -> bool:
        """
        Active food seeking behavior with realistic movement patterns.
        """
        prey_list = self.find_food()
        if not prey_list:
            return False

        # Calculate food density at different locations
        food_patches = {}
        for prey in prey_list:
            if prey.alive:
                pos = (prey.x, prey.y)
                food_patches[pos] = food_patches.get(pos, 0) + 1

        if not food_patches:
            return False

        # Choose location with highest food density
        best_location = max(food_patches.keys(), key=lambda pos: food_patches[pos])

        # Move toward best food patch
        return self.move_towards(
            best_location[0], best_location[1], self.max_movement_distance
        )

    def avoid_predators(self) -> bool:
        """
        Predator avoidance behavior - escape from fish.
        """
        predator_positions = []
        threat_level = 0

        # Detect nearby fish (predators)
        for dx in range(
            -self.predator_detection_range, self.predator_detection_range + 1
        ):
            for dy in range(
                -self.predator_detection_range, self.predator_detection_range + 1
            ):
                nx = self.x + dx
                ny = self.y + dy
                if (
                    0 <= nx < self.environment.width
                    and 0 <= ny < self.environment.height
                ):
                    agents = self.environment.get_agents_at(nx, ny)
                    for agent in agents:
                        if agent.__class__.__name__ == "Fish" and agent.alive:
                            distance = abs(dx) + abs(dy)
                            threat_level += 1.0 / max(
                                1, distance
                            )  # Closer = more threatening
                            predator_positions.append((nx, ny, distance))

        if not predator_positions:
            return False

        # Calculate escape direction (away from predators)
        escape_x = self.x
        escape_y = self.y

        for pred_x, pred_y, distance in predator_positions:
            # Weight by distance - closer predators have more influence
            weight = 1.0 / max(1, distance)

            if pred_x < self.x:
                escape_x += int(weight * 2)
            elif pred_x > self.x:
                escape_x -= int(weight * 2)

            if pred_y < self.y:
                escape_y += int(weight * 1)  # Prefer moving down to escape
            elif pred_y > self.y:
                escape_y -= int(weight * 1)

        # Constrain to environment bounds
        escape_x = max(0, min(escape_x, self.environment.width - 1))
        escape_y = max(0, min(escape_y, self.environment.height - 1))

        return self.move(escape_x, escape_y)

    def schooling_behavior(self) -> bool:
        """
        Schooling/aggregation behavior - stay near other zooplankton.
        """
        if random.random() > self.schooling_tendency:
            return False

        nearby_zooplankton = []
        search_radius = 3

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
                            agent.__class__.__name__ == "Zooplankton"
                            and agent != self
                            and agent.alive
                        ):
                            nearby_zooplankton.append(agent)

        if len(nearby_zooplankton) < 1:
            return False

        # Move toward center of nearby zooplankton
        center_x = sum(zoo.x for zoo in nearby_zooplankton) / len(nearby_zooplankton)
        center_y = sum(zoo.y for zoo in nearby_zooplankton) / len(nearby_zooplankton)

        target_x = int(center_x + random.uniform(-1, 1))  # Add some randomness
        target_y = int(center_y + random.uniform(-1, 1))

        return self.move_towards(target_x, target_y, max_distance=2)

    def calculate_environmental_stress(self) -> float:
        """
        Calculate environmental stress based on multiple factors.
        """
        conditions = self.get_environmental_conditions()

        # Temperature stress
        temp_stress = min(
            abs(conditions["temperature"] - self.optimal_temperature)
            / self.temp_tolerance,
            1.0,
        )

        # Oxygen stress
        oxygen_stress = min(
            abs(conditions["oxygen"] - self.optimal_oxygen) / self.oxygen_tolerance, 1.0
        )

        # Critical oxygen threshold
        if conditions["oxygen"] < self.critical_oxygen:
            oxygen_stress = min(1.0, oxygen_stress + 0.5)

        # pH stress
        ph_stress = min(
            abs(conditions["ph"] - self.optimal_ph) / self.ph_tolerance, 1.0
        )

        # Combined stress with different weightings based on literature
        total_stress = temp_stress * 0.4 + oxygen_stress * 0.45 + ph_stress * 0.15

        return min(total_stress, 1.0)

    def can_reproduce(self) -> bool:
        """
        Determine if zooplankton can reproduce based on realistic criteria.
        """
        stress_level = self.calculate_environmental_stress()

        return (
            self.energy > self.reproduction_threshold
            and self.reproduction_age_min <= self.age <= self.reproduction_age_max
            and stress_level < 0.7
            and random.random() < self.egg_production_rate
        )

    def reproduce(self) -> Optional["Zooplankton"]:
        """
        Reproduction based on copepod biology - broadcast spawning.
        """
        if not self.can_reproduce():
            return None

        offspring_positions = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                new_x = max(0, min(self.x + dx, self.environment.width - 1))
                new_y = max(0, min(self.y + dy, self.environment.height - 1))

                if (new_x, new_y) != (self.x, self.y):
                    local_agents = self.environment.get_agents_at(new_x, new_y)
                    local_zooplankton = [
                        a for a in local_agents if isinstance(a, Zooplankton)
                    ]

                    if len(local_zooplankton) < 4:
                        offspring_positions.append((new_x, new_y))

        if not offspring_positions:
            return None

        # Choose position with best environmental conditions
        best_pos = None
        best_score = -1

        for pos in offspring_positions:
            conditions = self.environment.get_conditions(pos[0], pos[1])

            # Score based on temperature and oxygen
            temp_score = 1 - min(
                1,
                abs(conditions["temperature"] - self.optimal_temperature)
                / self.temp_tolerance,
            )
            oxygen_score = 1 - min(
                1,
                abs(conditions["oxygen"] - self.optimal_oxygen) / self.oxygen_tolerance,
            )

            total_score = temp_score * 0.6 + oxygen_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_pos = pos

        if best_pos is None:
            best_pos = random.choice(offspring_positions)

        # Pay reproduction cost
        self.energy -= self.reproduction_cost

        # Create offspring (nauplius larva)
        offspring = Zooplankton(
            self.environment, best_pos[0], best_pos[1], self.offspring_energy
        )

        # Add some genetic variation
        offspring.optimal_temperature = self.optimal_temperature + random.gauss(0, 0.5)
        offspring.temp_tolerance = max(4.0, self.temp_tolerance + random.gauss(0, 0.3))

        return offspring

    def find_optimal_location(self, search_radius: int = 3) -> Tuple[int, int]:
        """
        Find optimal location considering multiple factors.
        """
        best_pos = (self.x, self.y)
        best_score = float("-inf")

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = max(0, min(self.x + dx, self.environment.width - 1))
                ny = max(0, min(self.y + dy, self.environment.height - 1))

                conditions = self.environment.get_conditions(nx, ny)

                # Environmental quality score
                temp_score = 1 - min(
                    1,
                    abs(conditions["temperature"] - self.optimal_temperature)
                    / self.temp_tolerance,
                )
                oxygen_score = 1 - min(
                    1,
                    abs(conditions["oxygen"] - self.optimal_oxygen)
                    / self.oxygen_tolerance,
                )
                env_score = temp_score * 0.6 + oxygen_score * 0.4

                # Food availability score
                food_count = self.environment.get_agent_count_in_radius(
                    nx, ny, 1, Phytoplankton
                )
                food_score = min(food_count / 8.0, 1.0)  # Normalize to 0-1

                # Predator avoidance score
                predator_count = 0
                for px in range(max(0, nx - 2), min(self.environment.width, nx + 3)):
                    for py in range(
                        max(0, ny - 2), min(self.environment.height, ny + 3)
                    ):
                        agents = self.environment.get_agents_at(px, py)
                        predator_count += sum(
                            1 for a in agents if a.__class__.__name__ == "Fish"
                        )

                safety_score = 1 / (1 + predator_count * 0.4)

                # Combined score
                total_score = env_score * 0.4 + food_score * 0.4 + safety_score * 0.2

                if total_score > best_score:
                    best_score = total_score
                    best_pos = (nx, ny)

        return best_pos

    def metabolize(self) -> None:
        """
        Enhanced metabolism with temperature dependence.
        """
        conditions = self.get_environmental_conditions()

        # Base metabolic cost
        base_cost = self.base_metabolism

        # Temperature effect on metabolism
        temp_factor = self.calculate_temperature_effect_on_metabolism(
            conditions["temperature"]
        )
        metabolic_cost = base_cost * temp_factor

        # Stress increases metabolic cost
        stress = self.calculate_environmental_stress()
        stress_cost = stress * 0.04  # Additional cost from stress

        # Total metabolic cost
        total_cost = metabolic_cost + stress_cost
        self.energy -= total_cost

    def update(self) -> Optional["Zooplankton"]:
        """
        Main update method with realistic behavioral priorities.
        """
        self.basic_update()
        if not self.alive:
            return None

        # Age-related mortality
        if self.age > self.maximum_lifespan:
            self.die()
            return None

        self.metabolize()

        # Environmental stress mortality
        stress_level = self.calculate_environmental_stress()
        if stress_level > self.stress_mortality_threshold:
            if random.random() < (stress_level - 0.7) * 0.3:
                self.die()
                return None

        # Behavioral priorities based on current conditions
        is_hungry = self.energy < self.hunger_threshold
        high_stress = stress_level > 0.6
        light_phase = self.get_current_light_phase()

        # Priority 1: Diel vertical migration (if not under immediate threat)
        if not high_stress and random.random() < 0.3:
            self.perform_diel_vertical_migration()

        # Priority 2: Predator avoidance (highest priority when threatened)
        if high_stress or random.random() < 0.2:
            if self.avoid_predators():
                return self.reproduce()

        # Priority 3: Feeding behavior
        if is_hungry or (light_phase == "night" and random.random() < 0.7):
            energy_gained = self.feed()
            if energy_gained == 0 and is_hungry:
                if not self.seek_food():
                    optimal_pos = self.find_optimal_location(search_radius=2)
                    if optimal_pos != (self.x, self.y):
                        self.move_towards(
                            optimal_pos[0], optimal_pos[1], self.max_movement_distance
                        )

        # Priority 4: Social behavior and random movement
        else:
            behavior_choice = random.random()
            if behavior_choice < 0.3:
                self.feed()
            elif behavior_choice < 0.5:
                self.schooling_behavior()
            elif behavior_choice < 0.7:
                optimal_pos = self.find_optimal_location(search_radius=2)
                if optimal_pos != (self.x, self.y):
                    self.move_towards(
                        optimal_pos[0], optimal_pos[1], self.max_movement_distance
                    )
            else:
                # Random exploration
                self.random_move(distance=1)

        offspring = self.reproduce()

        # Additional mortality from extreme conditions
        conditions = self.get_environmental_conditions()
        if conditions["temperature"] > self.thermal_death_temp:
            if random.random() < 0.4:
                self.die()
                return None

        if conditions["oxygen"] < self.critical_oxygen:
            if random.random() < 0.2:
                self.die()
                return None

        return offspring
