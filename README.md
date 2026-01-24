# Protozoa
**Continuous Active Inference Simulation**

Protozoa is a zero-player biological simulation where a single-cell agent navigates a nutrient-rich petri dish using the **Free Energy Principle (FEP)**. Unlike traditional AI that follows hard-coded rules, this agent operates by minimizing the difference between its genetic expectation (Homeostasis) and its actual sensory input.

![Platform](https://img.shields.io/badge/Platform-Linux%20Terminal-black)
![Language](https://img.shields.io/badge/Language-Rust-orange)
![License](https://img.shields.io/badge/License-AGPLv3-green)

## ✨ Features
*   **Active Inference Engine:** The agent survives by minimizing "Free Energy" (Prediction Error).
*   **Stereo Vision:** Two chemical sensors detect continuous gradients.
*   **Multi-Layer Memory:**
    *   **Short-term:** Ring buffer of 32 recent experiences
    *   **Long-term:** 20×10 spatial grid learning nutrient expectations (Welford's algorithm)
    *   **Episodic:** Up to 8 remembered landmarks with reliability decay
*   **MCTS Planning:** Monte Carlo Tree Search with Expected Free Energy (pragmatic + epistemic value).
*   **Goal-Directed Navigation:** Returns to remembered food sources when energy is low.
*   **High Performance:** Parallelized field rendering using `rayon`.
*   **Static Binary:** Ship a single executable with no external dependencies.
*   **Dynamic Environment:** Food sources decay, move (Brownian motion), and regrow.
*   **Metabolic System:** Managing energy (ATP) is crucial; exhaustion leads to death spirals.
*   **Emergent Behavior:** Watch the agent panic, tumble, sprint, and graze without explicit instructions.

## 🚀 Getting Started

### Prerequisites
*   **Rust Toolchain:** Install via [rustup.rs](https://rustup.rs/).

### Running the Simulation
Clone the repository and run using `cargo`:

```bash
git clone https://github.com/ahenkes1/protozoa.git
cd protozoa/protozoa_rust
cargo run --release
```

*(Note: Use `--release` for optimal frame rates)*

### Static Compilation (Linux)
To build a dependency-free static binary (MUSL):

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```
The binary will be located at `target/x86_64-unknown-linux-musl/release/protozoa_rust`.

## 🎮 Controls
This is a **zero-player game**, meaning you watch life unfold.
*   **`q`**: Quit the simulation.

## 🛠️ Development

### Project Structure
*   `src/main.rs`: Entry point and visualization loop (`ratatui` + `crossterm`).
*   `src/simulation/`: Core logic module.
    *   `agent.rs`: Active Inference (FEP) logic with memory systems and MCTS planning.
    *   `environment.rs`: Petri Dish and Nutrient physics.
    *   `params.rs`: All configurable hyperparameters.
    *   `memory/`: Memory systems (ring buffer, spatial grid, episodic landmarks).
    *   `planning/`: MCTS planner with Expected Free Energy evaluation.
*   `src/ui/`: Rendering module.
    *   `field.rs`: Parallelized grid computation (`rayon`).
    *   `render.rs`: TUI rendering and coordinate transformation.

### Configuration
All simulation parameters are defined in `src/simulation/params.rs`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_CONCENTRATION` | 0.8 | Homeostatic set-point |
| `LEARNING_RATE` | 0.15 | Gradient descent step size |
| `MAX_SPEED` | 1.5 | Maximum movement speed |
| `PANIC_THRESHOLD` | -0.01 | Temporal gradient trigger |
| `EXHAUSTION_THRESHOLD` | 0.01 | Energy level for exhaustion |
| `EXPLORATION_SCALE` | 0.3 | Bonus for exploring uncertain regions |
| `MAX_LANDMARKS` | 8 | Max remembered food locations |
| `LANDMARK_THRESHOLD` | 0.7 | Min nutrient to store landmark |
| `MCTS_ROLLOUTS` | 50 | Trajectories per planning step |
| `MCTS_DEPTH` | 10 | Lookahead depth for planning |
| `PLANNING_WEIGHT` | 0.3 | Blend of planned vs reactive control |

### Running Tests
```bash
cargo test  # Runs 94 tests across 8 test files
```

### Code Quality
We enforce strict linting and formatting (also in CI):
```bash
cargo fmt --check
cargo clippy -- -D warnings
```

## 🧠 How it Works

### Core Control Loop
The agent blends reactive control with MCTS planning:

$$
\dot{\theta} = (1-w_p) \cdot \dot{\theta}_{reactive} + w_p \cdot \dot{\theta}_{planned} + \text{exploration} + \text{goal}
$$

1.  **Error:** Precision-weighted difference between sensing and target (0.8).
2.  **Gradient:** The difference between Left and Right sensors.
3.  **Planning:** Every 20 ticks, MCTS evaluates 50 trajectories using learned spatial priors.
4.  **Exploration:** Bonus for visiting uncertain regions (inverse precision).
5.  **Goal Navigation:** When energy < 30%, navigate toward remembered landmarks.
6.  **Panic:** Random tumble if conditions worsen rapidly (temporal gradient).

### Memory Systems
- **Short-term:** 32-element ring buffer of recent experiences
- **Long-term:** 20×10 grid learns nutrient expectations via Welford's algorithm
- **Episodic:** Stores up to 8 high-nutrient landmarks with reliability decay

### MCTS Expected Free Energy
For each trajectory τ:
```
G(τ) = pragmatic + exploration_scale × epistemic
pragmatic = Σ prior.mean × state.energy    (prefer nutrients + survival)
epistemic = Σ 1/precision                   (prefer uncertainty reduction)
```
