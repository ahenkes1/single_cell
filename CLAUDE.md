# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

All commands run from `protozoa_rust/` directory:

```bash
cargo run --release      # Run simulation (use --release for optimal frame rates)
cargo test               # Run all tests (22 tests across 4 test files)
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
- `agent.rs`: Protozoa struct implementing Active Inference. Key algorithm: `update_state()` calculates prediction error, spatial gradient (left-right sensor difference), and temporal gradient to drive heading/speed updates. Includes NaN propagation guards via `assert_finite()` helper function.
- `environment.rs`: PetriDish with multiple NutrientSource Gaussian blobs. Concentration at (x,y) is sum of Gaussians. Sources decay, drift via Brownian motion, and respawn when depleted. Includes epsilon guard for near-zero radius.
- `params.rs`: All simulation hyperparameters organized into sections:
  - **Sensing**: `TARGET_CONCENTRATION` (0.8), `SENSOR_DIST`, `SENSOR_ANGLE`, `LEARNING_RATE`, `MAX_SPEED`
  - **Behavior**: `PANIC_THRESHOLD`, `PANIC_TURN_RANGE`, `NOISE_SCALE`, `EXHAUSTION_THRESHOLD`, `EXHAUSTION_SPEED_FACTOR`
  - **Metabolism**: `BASE_METABOLIC_COST`, `SPEED_METABOLIC_COST`, `INTAKE_RATE`
  - **Environment**: `DISH_WIDTH/HEIGHT`, `SOURCE_MARGIN`, `SOURCE_RADIUS_MIN/MAX`, `SOURCE_INTENSITY_MIN/MAX`, `SOURCE_DECAY_MIN/MAX`, `BROWNIAN_STEP`, `RESPAWN_THRESHOLD`, `SOURCE_COUNT_MIN/MAX`

**`ui/`** - Rendering
- `field.rs`: Parallel grid computation using `rayon`. Maps concentration values to ASCII density characters
- `render.rs`: `ratatui` draw logic with `world_to_grid_coords()` for coordinate transformation

**`main.rs`** - Event loop: terminal setup (crossterm), tick-based update cycle (sense -> update_state -> render), input handling ('q' to quit). Uses saturating arithmetic for overflow safety.

### Key Mathematical Concepts

The agent uses stereo chemical sensors (left/right at configurable angle offset). Each tick:
1. Error = mean_sense - TARGET_CONCENTRATION (0.8)
2. Gradient = left_sensor - right_sensor
3. Heading change = -LEARNING_RATE * error * gradient + noise + panic_turn
4. Speed = MAX_SPEED * |error|
5. Angle normalized using `rem_euclid(2π)` for numerical stability

Boundary sensing returns -1.0 (toxic void) to create repulsion.

### Numerical Safety

- `assert_finite()` guards on critical calculations (mean_sense, error, gradient, d_theta, energy)
- Epsilon guard on Gaussian sigma_sq to prevent division by near-zero
- `rem_euclid()` instead of `%` for angle normalization
- Saturating arithmetic for sensor coordinate calculations

### Test Coverage

22 tests across 4 files covering:
- Agent: initialization, sensing, movement, energy, exhaustion, boundary clamping, angle normalization, temporal gradient, speed-error correlation
- Environment: initialization, concentration bounds, boundaries, Gaussian properties, source decay/respawn, Brownian motion bounds
- Rendering: grid computation, coordinate transformation

### Code Style

- Strict clippy linting enabled (`#![warn(clippy::all, clippy::pedantic)]`)
- Files target <200 LOC
- Rust 2024 edition
- CI pipeline runs: fmt check, clippy, build, tests

### See Also

- [AGENTS.md](AGENTS.md) - Detailed project specification with mathematical formulas and algorithmic derivations
