"""
Protozoa - Continuous Active Inference Simulation.

A real-time simulation of a single-cell organism (the Agent)
living in a petri dish.
"""

import math
import time
import curses
from simulation_core import PetriDish, Protozoa, PARAMS

class Simulation:
    """
    Handles the visual simulation using curses.
    """

    def __init__(self):
        self.stdscr = None  # type: curses.window | None
        self.dish = None    # type: PetriDish | None
        self.agent = None   # type: Protozoa | None
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
                if self.dish and self.agent and self.stdscr:
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
        if not self.stdscr:
            return

        try:
            key = self.stdscr.getch()
            if key == ord("q"):
                self.running = False
        except curses.error:
            pass

    def update(self):
        """Update simulation state."""
        if not self.dish or not self.agent:
            return

        self.dish.update()
        self.agent.sense(self.dish)
        self.agent.update_state(self.dish)

    def render(self, rows, cols):
        """Render the current state."""
        if not self.stdscr:
            return

        self.stdscr.erase()
        self._render_field(rows, cols)
        self._render_agent(rows, cols)
        self._render_hud()
        self.stdscr.refresh()

    def _render_field(self, rows, cols):
        """Render the nutrient field."""
        if not self.stdscr or not self.dish:
            return

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
        if not self.stdscr or not self.agent or not self.dish:
            return

        agent_r = int(self.agent.y / self.dish.height * rows)
        agent_c = int(self.agent.x / self.dish.width * cols)

        # Clip to screen
        agent_r = max(0, min(rows - 1, agent_r))
        agent_c = max(0, min(cols - 1, agent_c))

        try:
            self.stdscr.addch(agent_r, agent_c, "O", curses.A_BOLD)
            # Draw sensor indicators
            theta = self.agent.angle
            p_r = int(agent_r + math.sin(theta) * 2)
            p_c = int(agent_c + math.cos(theta) * 2)
            if 0 <= p_r < rows and 0 <= p_c < cols:
                self.stdscr.addch(p_r, p_c, ".")
        except curses.error:
            pass

    def _render_hud(self):
        """Render the HUD."""
        if not self.stdscr or not self.agent:
            return

        mean_sense = (self.agent.val_l + self.agent.val_r) / 2
        error = mean_sense - PARAMS["target"]
        hud = (
            f"Sens: {mean_sense:.2f} | "
            f"Tgt: {PARAMS['target']:.2f} | "
            f"Err: {error:.2f} | "
            f"Spd: {self.agent.speed:.2f} | "
            f"Egy: {self.agent.energy:.2f}"
        )
        try:
            self.stdscr.addstr(0, 0, hud, curses.A_REVERSE)
        except curses.error:
            pass


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
