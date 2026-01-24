# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

All commands run from `protozoa_rust/` directory:

```bash
cargo run --release      # Run simulation (use --release for optimal frame rates)
cargo test               # Run all tests
cargo fmt                # Format code
cargo clippy -- -D warnings  # Lint (strict, warnings as errors)
```

Static binary build (Linux MUSL):
```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## Architecture Overview

This is an Active Inference biological simulation where a single-cell agent (Protozoa) navigates a nutrient-rich petri dish using the Free Energy Principle. The agent minimizes prediction error between its sensory input and homeostatic target rather than following hard-coded rules.

### Core Modules

**`simulation/`** - Domain logic
- `agent.rs`: Protozoa struct implementing Active Inference. Key algorithm: `update_state()` calculates prediction error (sensed concentration vs TARGET_CONCENTRATION=0.8), spatial gradient (left-right sensor difference), and temporal gradient to drive heading/speed updates
- `environment.rs`: PetriDish with multiple NutrientSource Gaussian blobs. Concentration at (x,y) is sum of Gaussians. Sources decay, drift via Brownian motion, and respawn when depleted
- `params.rs`: Hyperparameters (sensor distance, angles, learning rate, max speed, dish dimensions)

**`ui/`** - Rendering
- `field.rs`: Parallel grid computation using `rayon`. Maps concentration values to ASCII density characters
- `render.rs`: `ratatui` draw logic

**`main.rs`** - Event loop: terminal setup (crossterm), tick-based update cycle (sense -> update_state -> render), input handling ('q' to quit)

### Key Mathematical Concepts

The agent uses stereo chemical sensors (left/right at configurable angle offset). Each tick:
1. Error = mean_sense - target (0.8)
2. Gradient = left_sensor - right_sensor
3. Heading change = -learning_rate * error * gradient + noise + panic_turn
4. Speed = max_speed * |error|

Boundary sensing returns -1.0 (toxic void) to create repulsion.

### Code Style

- Strict clippy linting enabled (`#![warn(clippy::all, clippy::pedantic)]`)
- Files target <200 LOC
- Rust 2024 edition
