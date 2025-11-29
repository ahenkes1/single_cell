# Protozoa
**Continuous Active Inference Simulation**

Protozoa is a zero-player biological simulation where a single-cell agent navigates a nutrient-rich petri dish using the **Free Energy Principle (FEP)**. Unlike traditional AI that follows hard-coded rules, this agent operates by minimizing the difference between its genetic expectation (Homeostasis) and its actual sensory input.

![Platform](https://img.shields.io/badge/Platform-Linux%20Terminal-black)
![Language](https://img.shields.io/badge/Language-Python-blue)
![License](https://img.shields.io/badge/License-AGPLv3-green)

## ✨ Features
*   **Active Inference Engine:** The agent survives by minimizing "Free Energy" (Prediction Error).
*   **Stereo Vision:** Two chemical sensors detect continuous gradients.
*   **Dynamic Environment:** Food sources decay, move (Brownian motion), and regrow.
*   **Metabolic System:** Managing energy (ATP) is crucial; exhaustion leads to death spirals.
*   **Emergent Behavior:** Watch the agent panic, tumble, sprint, and graze without explicit instructions.
*   **Optimized:** Built with `curses` for efficient terminal visualization.

## 🚀 Getting Started

### Prerequisites
This project uses **uv** for ultra-fast dependency management.

1.  **Install `uv`** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    *(Or use `pip install uv`)*

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/protozoa.git
    cd protozoa
    ```

### Running the Simulation
Simply run the entry script. `uv` will automatically create the virtual environment and install dependencies (`curses` is part of the standard library on Linux/Mac).

```bash
uv run protozoa.py
```

## 🎮 Controls
This is a **zero-player game**, meaning you watch life unfold.
*   **`q`**: Quit the simulation.

## 🛠️ Development

### Project Structure
*   `protozoa.py`: Entry point and Visualization loop (`curses`).
*   `simulation_core.py`: Core logic (Math, Physics, and Agent brain).
*   `tests/`: Unit tests for TDD.

### Running Tests
```bash
uv run pytest
```

### Code Quality
To ensure the simulation logic is robust, we enforce strict quality standards:
```bash
# Format code
uv run black .

# Linting
uv run flake8 .
uv run pylint simulation_core.py protozoa.py

# Type Checking
uv run mypy simulation_core.py protozoa.py
```

## 🧠 How it Works
The agent follows the equation:
$$ \dot{\theta} \propto - \text{Error} \times \text{Gradient} $$

1.  **Error:** The difference between current sensing and target (0.8 concentration).
2.  **Gradient:** The difference between Left and Right sensors.
3.  **Panic:** If the agent senses conditions getting worse over time (Temporal Gradient), it initiates a random tumble to escape local minima.
