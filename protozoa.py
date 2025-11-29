"""
Protozoa - Continuous Active Inference Simulation.

A real-time simulation of a single-cell organism (the Agent) living in a petri dish.
"""

import math
import random
import time
import curses
from typing import List, Dict

# Configure hyperparameters
PARAMS = {
    'target': 0.8,
    'sensor_dist': 2.0,
    'sensor_angle': 0.5,
    'learning_rate': 0.15,
    'max_speed': 1.5
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
        return {
            'x': random.uniform(0, self.width),
            'y': random.uniform(0, self.height),
            'radius': random.uniform(5.0, 15.0),
            'intensity': random.uniform(0.5, 1.0)
        }

    def get_concentration(self, x: float, y: float) -> float:
        """
        Calculates Nutrient Concentration C(x,y).
        Sum of Gaussian blobs.
        """
        concentration = 0.0
        for source in self.sources:
            d_x = x - source['x']
            d_y = y - source['y']
            dist_sq = d_x*d_x + d_y*d_y
            sigma_sq = source['radius']**2
            # Gaussian: I * exp(-dist^2 / (2*sigma^2))
            val = source['intensity'] * math.exp(-dist_sq / (2 * sigma_sq))
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
            source['intensity'] *= decay_factor

            # Brownian Motion
            source['x'] += random.uniform(-brownian_step, brownian_step)
            source['y'] += random.uniform(-brownian_step, brownian_step)

            # Clamp position (optional, but keeps them on screen mostly)
            source['x'] = max(0, min(self.width, source['x']))
            source['y'] = max(0, min(self.height, source['y']))

            # Regrowth
            if source['intensity'] < respawn_threshold:
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

    def sense(self, dish: PetriDish):
        """
        Read chemical gradients from the dish.
        """
        # Left Sensor
        theta_l = self.angle + PARAMS['sensor_angle']
        x_l = self.x + PARAMS['sensor_dist'] * math.cos(theta_l)
        y_l = self.y + PARAMS['sensor_dist'] * math.sin(theta_l)
        self.val_l = dish.get_concentration(x_l, y_l)

        # Right Sensor
        theta_r = self.angle - PARAMS['sensor_angle']
        x_r = self.x + PARAMS['sensor_dist'] * math.cos(theta_r)
        y_r = self.y + PARAMS['sensor_dist'] * math.sin(theta_r)
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
        error = mean_sense - PARAMS['target']

        # 3. Gradient
        gradient = self.val_l - self.val_r

        # 4. Dynamics
        # Heading Update: d_theta = -lr * E * G
        d_theta = -PARAMS['learning_rate'] * error * gradient
        self.angle += d_theta

        # Normalize angle
        self.angle %= (2 * math.pi)

        # Speed Update: v = max_speed * |E|
        self.speed = PARAMS['max_speed'] * abs(error)

        # Position Update
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        # Boundary Check
        self.x = max(0, min(dish.width, self.x))
        self.y = max(0, min(dish.height, self.y))

class Simulation:
    """
    Handles the visual simulation using curses.
    """
    def __init__(self):
        self.stdscr = None
        self.dish = None
        self.agent = None
        self.running = True
        self.chars = " .:-=+*#%@"

    def run(self):
        """Entry point for the simulation."""
        try:
            # Initialize curses
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)
            self.stdscr.nodelay(True)

            # Get screen dimensions
            rows, cols = self.stdscr.getmaxyx()

            self.dish = PetriDish(100.0, 100.0)
            self.agent = Protozoa(50.0, 50.0)

            while self.running:
                self.handle_input()
                self.update()
                self.render(rows, cols)
                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            if self.stdscr:
                curses.nocbreak()
                self.stdscr.keypad(False)
                curses.echo()
                curses.endwin()

    def handle_input(self):
        """Process user input."""
        try:
            key = self.stdscr.getch()
            if key == ord('q'):
                self.running = False
        except curses.error:
            pass

    def update(self):
        """Update simulation state."""
        self.dish.update()
        self.agent.sense(self.dish)
        self.agent.update_state(self.dish)

    def render(self, rows, cols):
        """Render the current state."""
        self.stdscr.erase()
        self._render_field(rows, cols)
        self._render_agent(rows, cols)
        self._render_hud()
        self.stdscr.refresh()

    def _render_field(self, rows, cols):
        """Render the nutrient field."""
        # Scaling factors
        scale_y = self.dish.height / rows
        scale_x = self.dish.width / cols

        step_y = 1
        step_x = 1

        for r in range(0, rows - 1, step_y):
            for c in range(0, cols, step_x):
                world_y = r * scale_y
                world_x = c * scale_x

                val = self.dish.get_concentration(world_x, world_y)
                char_idx = int(val * (len(self.chars) - 1))
                try:
                    self.stdscr.addch(r, c, self.chars[char_idx])
                except curses.error:
                    pass

    def _render_agent(self, rows, cols):
        """Render the agent."""
        agent_r = int(self.agent.y / self.dish.height * rows)
        agent_c = int(self.agent.x / self.dish.width * cols)

        # Clip to screen
        agent_r = max(0, min(rows - 1, agent_r))
        agent_c = max(0, min(cols - 1, agent_c))

        try:
            self.stdscr.addch(agent_r, agent_c, 'O', curses.A_BOLD)
            # Draw sensor indicators
            theta = self.agent.angle
            p_r = int(agent_r + math.sin(theta) * 2)
            p_c = int(agent_c + math.cos(theta) * 2)
            if 0 <= p_r < rows and 0 <= p_c < cols:
                self.stdscr.addch(p_r, p_c, '.')
        except curses.error:
            pass

    def _render_hud(self):
        """Render the HUD."""
        hud = (
            f"Sens: {(self.agent.val_l + self.agent.val_r)/2:.2f} | "
            f"Tgt: {PARAMS['target']:.2f} | "
            f"Err: {((self.agent.val_l + self.agent.val_r)/2 - PARAMS['target']):.2f} | "
            f"Spd: {self.agent.speed:.2f}"
        )
        try:
            self.stdscr.addstr(0, 0, hud, curses.A_REVERSE)
        except curses.error:
            pass

if __name__ == '__main__':
    sim = Simulation()
    sim.run()
