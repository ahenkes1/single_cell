# Protozoa
**Continuous Active Inference Simulation**

Protozoa is a zero-player biological simulation where a single-cell agent navigates a nutrient-rich petri dish using the **Free Energy Principle (FEP)**. Unlike traditional AI that follows hard-coded rules, this agent operates by minimizing the difference between its genetic expectation (Homeostasis) and its actual sensory input.

![Platform](https://img.shields.io/badge/Platform-Linux%20Terminal-black)
![Language](https://img.shields.io/badge/Language-Rust-orange)
![License](https://img.shields.io/badge/License-AGPLv3-green)

## ✨ Features
*   **Active Inference Engine:** The agent survives by minimizing "Free Energy" (Prediction Error).
*   **Stereo Vision:** Two chemical sensors detect continuous gradients.
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
git clone https://github.com/yourusername/protozoa.git
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
    *   `agent.rs`: Active Inference (FEP) logic.
    *   `environment.rs`: Petri Dish and Nutrient physics.
*   `src/ui/`: Rendering module.
    *   `field.rs`: Parallelized grid computation (`rayon`).

### Running Tests
```bash
cargo test
```

### Code Quality
We enforce strict linting and formatting:
```bash
cargo fmt
cargo clippy -- -D warnings
```

## 🧠 How it Works
The agent follows the equation:
$$
\dot{\theta} \propto - \text{Error} \times \text{Gradient}
$$

1.  **Error:** The difference between current sensing and target (0.8 concentration).
2.  **Gradient:** The difference between Left and Right sensors.
3.  **Panic:** If the agent senses conditions getting worse over time (Temporal Gradient), it initiates a random tumble to escape local minima.
