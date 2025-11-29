"""
Core simulation classes for Protozoa.
"""
import random
import math
from typing import List, Dict

# Configure hyperparameters
PARAMS = {
    "target": 0.8,
    "sensor_dist": 2.0,
    "sensor_angle": 0.5,
    "learning_rate": 0.15,
    "max_speed": 1.5,
}

class PetriDish:
    """
    The Environment (Fields).
    The domain is a continuous 2D plane.
    """

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.sources: List[Dict[str, float]] = []
        self._init_sources()

    def _init_sources(self):
        """Create 5-10 random sources on init."""
        num_sources = random.randint(5, 10)
        for _ in range(num_sources):
            self.sources.append(self._create_random_source())

    def _create_random_source(self) -> Dict[str, float]:
        """Generates a random nutrient source."""
        margin = 10.0
        return {
            "x": random.uniform(margin, self.width - margin),
            "y": random.uniform(margin, self.height - margin),
            "radius": random.uniform(5.0, 15.0),
            "intensity": random.uniform(0.5, 1.0),
        }

    def get_concentration(self, x: float, y: float) -> float:
        """
        Calculates Nutrient Concentration C(x,y).
        Sum of Gaussian blobs.
        """
        # Proposal 1: Toxic Void
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return -1.0

        concentration = 0.0
        for source in self.sources:
            d_x = x - source["x"]
            d_y = y - source["y"]
            dist_sq = d_x * d_x + d_y * d_y
            sigma_sq = source["radius"] ** 2
            # Gaussian: I * exp(-dist^2 / (2*sigma^2))
            val = source["intensity"] * math.exp(
                -dist_sq / (2 * sigma_sq)
            )
            concentration += val

        return min(1.0, max(0.0, concentration))

    def update(self):
        """
        Dynamics update: Entropy, Brownian Motion, Regrowth.
        """
        decay_factor = 0.995
        brownian_step = 0.5
        respawn_threshold = 0.05

        for i, source in enumerate(self.sources):
            # Entropy
            source["intensity"] *= decay_factor

            # Brownian Motion
            source["x"] += random.uniform(-brownian_step, brownian_step)
            source["y"] += random.uniform(-brownian_step, brownian_step)

            # Clamp position (optional, but keeps them on screen mostly)
            source["x"] = max(0, min(self.width, source["x"]))
            source["y"] = max(0, min(self.height, source["y"]))

            # Regrowth
            if source["intensity"] < respawn_threshold:
                self.sources[i] = self._create_random_source()


class Protozoa:
    """
    The FEP Agent.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0

        # Sensors (will be updated in sense)
        self.val_l = 0.0
        self.val_r = 0.0

        # Internal State
        self.energy = 1.0

    def sense(self, dish: PetriDish):
        """
        Read chemical gradients from the dish.
        """
        # Left Sensor
        theta_l = self.angle + PARAMS["sensor_angle"]
        x_l = self.x + PARAMS["sensor_dist"] * math.cos(theta_l)
        y_l = self.y + PARAMS["sensor_dist"] * math.sin(theta_l)
        self.val_l = dish.get_concentration(x_l, y_l)

        # Right Sensor
        theta_r = self.angle - PARAMS["sensor_angle"]
        x_r = self.x + PARAMS["sensor_dist"] * math.cos(theta_r)
        y_r = self.y + PARAMS["sensor_dist"] * math.sin(theta_r)
        self.val_r = dish.get_concentration(x_r, y_r)

    def update_state(self, dish: PetriDish):
        """
        The Active Inference Core.
        Minimizes Free Energy (Error).
        """
        # 1. Sensation
        mean_sense = (self.val_l + self.val_r) / 2.0

        # 2. Error (Target - Actual) or (Actual - Target)?
        # Specification says: Error (E) = mu - rho (Actual - Target)
        error = mean_sense - PARAMS["target"]

        # 3. Gradient
        gradient = self.val_l - self.val_r

        # 4. Dynamics
        # Heading Update: d_theta = -lr * E * G
        # Add noise (Brownian tumbling) proportional to error magnitude
        # This allows escaping local minima (corners) where gradient is zero.
        noise = random.uniform(-0.5, 0.5) * abs(error)
        d_theta = (-PARAMS["learning_rate"] * error * gradient) + noise
        self.angle += d_theta

        # Normalize angle
        self.angle %= 2 * math.pi

        # Speed Update: v = max_speed * |E|
        self.speed = PARAMS["max_speed"] * abs(error)

        # Metabolic Update (Energy)
        # Consumes energy to move, gains energy from nutrients.
        # Tuning (Proposal A): reduced cost, increased intake.
        metabolic_cost = 0.0005 + (0.0025 * (self.speed / PARAMS["max_speed"]))
        intake = 0.03 * mean_sense
        self.energy = self.energy - metabolic_cost + intake
        self.energy = max(0.0, min(1.0, self.energy))

        # If energy is critical, agent is exhausted but can still move slowly
        if self.energy <= 0.01:
            self.speed *= 0.5

        # Position Update
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        # Boundary Check
        self.x = max(0, min(dish.width, self.x))
        self.y = max(0, min(dish.height, self.y))
