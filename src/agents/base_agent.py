import random
from typing import TYPE_CHECKING, Dict, Any
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..environment import Environment


class Agent(ABC):
    """
    Abstract base class for all agents in the marine ecosystem simulation.
    Defines common attributes and methods that all marine organisms share.
    """

    def __init__(self, environment: "Environment", x: int, y: int, energy: float = 10):
        self.environment = environment
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True
        self.age = 0
        self.max_energy = energy * 3

        self.base_metabolism = 0.1
        self.movement_cost = 0.05

        environment.register_agent(self)

    def move(self, new_x: int, new_y: int) -> bool:
        new_x = max(0, min(new_x, self.environment.width - 1))
        new_y = max(0, min(new_y, self.environment.height - 1))

        if new_x == self.x and new_y == self.y:
            return False

        distance = abs(new_x - self.x) + abs(new_y - self.y)
        cost = self.movement_cost * distance

        if self.energy >= cost:
            old_x, old_y = self.x, self.y
            self.x, self.y = new_x, new_y
            self.energy -= cost
            self.environment.update_agent_position(self, old_x, old_y)
            return True

        return False

    def random_move(self, distance: int = 1) -> bool:
        dx = random.randint(-distance, distance)
        dy = random.randint(-distance, distance)
        return self.move(self.x + dx, self.y + dy)

    def move_towards(self, target_x: int, target_y: int, max_distance: int = 1) -> bool:
        dx = target_x - self.x
        dy = target_y - self.y

        if dx != 0:
            dx = min(max_distance, abs(dx)) * (1 if dx > 0 else -1)
        if dy != 0:
            dy = min(max_distance, abs(dy)) * (1 if dy > 0 else -1)

        return self.move(self.x + dx, self.y + dy)

    def die(self) -> None:
        if self.alive:
            self.alive = False
            self.environment.unregister_agent(self)

    def get_environmental_conditions(self) -> Dict[str, float]:
        return self.environment.get_conditions(self.x, self.y)

    def calculate_environmental_stress(self) -> float:
        return 0.0

    def age_one_step(self) -> None:
        self.age += 1

    def metabolize(self) -> None:
        metabolic_cost = self.base_metabolism
        stress = self.calculate_environmental_stress()
        stress_cost = stress * 0.05
        total_cost = metabolic_cost + stress_cost
        self.energy -= total_cost

    def can_reproduce(self) -> bool:
        return False

    def find_optimal_location(self, search_radius: int = 3) -> tuple:
        return (self.x, self.y)

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "position": (self.x, self.y),
            "energy": self.energy,
            "age": self.age,
            "alive": self.alive,
            "stress_level": self.calculate_environmental_stress(),
            "environmental_conditions": self.get_environmental_conditions(),
        }

    @abstractmethod
    def update(self) -> None:
        pass

    def basic_update(self) -> None:
        if not self.alive:
            return
        self.age_one_step()
        self.metabolize()
        if self.energy <= 0:
            self.die()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pos=({self.x},{self.y}), energy={self.energy:.1f}, age={self.age})"
