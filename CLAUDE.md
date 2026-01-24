# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

All commands run from `protozoa_rust/` directory:

```bash
cargo run --release      # Run simulation (use --release for optimal frame rates)
cargo test               # Run all tests (94 tests across 8 test files)
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
- `agent.rs`: Protozoa struct implementing Active Inference with memory systems and MCTS planning. Key algorithm: `update_state()` calculates precision-weighted prediction error, spatial gradient, temporal gradient, MCTS planning, and goal-directed navigation. Includes NaN propagation guards via `assert_finite()` helper function.
- `environment.rs`: PetriDish with multiple NutrientSource Gaussian blobs. Concentration at (x,y) is sum of Gaussians. Sources decay, drift via Brownian motion, and respawn when depleted. Includes epsilon guard for near-zero radius.
- `params.rs`: All simulation hyperparameters organized into sections:
  - **Sensing**: `TARGET_CONCENTRATION` (0.8), `SENSOR_DIST`, `SENSOR_ANGLE`, `LEARNING_RATE`, `MAX_SPEED`
  - **Behavior**: `PANIC_THRESHOLD`, `PANIC_TURN_RANGE`, `NOISE_SCALE`, `EXHAUSTION_THRESHOLD`, `EXHAUSTION_SPEED_FACTOR`
  - **Metabolism**: `BASE_METABOLIC_COST`, `SPEED_METABOLIC_COST`, `INTAKE_RATE`
  - **Environment**: `DISH_WIDTH/HEIGHT`, `SOURCE_MARGIN`, `SOURCE_RADIUS_MIN/MAX`, `SOURCE_INTENSITY_MIN/MAX`, `SOURCE_DECAY_MIN/MAX`, `BROWNIAN_STEP`, `RESPAWN_THRESHOLD`, `SOURCE_COUNT_MIN/MAX`
  - **Memory**: `HISTORY_SIZE` (32), `GRID_WIDTH` (20), `GRID_HEIGHT` (10)
  - **Learning**: `PRIOR_LEARNING_RATE`, `EXPLORATION_SCALE`, `MIN_PRECISION`, `MAX_PRECISION`
  - **Episodic**: `MAX_LANDMARKS` (8), `LANDMARK_THRESHOLD`, `LANDMARK_DECAY`, `LANDMARK_ATTRACTION_SCALE`, `LANDMARK_VISIT_RADIUS`
  - **Planning**: `MCTS_ROLLOUTS` (50), `MCTS_DEPTH` (10), `MCTS_REPLAN_INTERVAL` (20), `MCTS_URGENT_ENERGY`, `PLANNING_WEIGHT`

**`simulation/memory/`** - Memory systems
- `ring_buffer.rs`: Generic fixed-size circular buffer for short-term memory
- `spatial_grid.rs`: 2D grid with Welford's online variance algorithm for spatial priors
- `episodic.rs`: Landmark storage with reliability decay for goal-directed navigation

**`simulation/planning/`** - Planning systems
- `mcts.rs`: Monte Carlo Tree Search with Expected Free Energy (pragmatic + epistemic value)

**`ui/`** - Rendering
- `field.rs`: Parallel grid computation using `rayon`. Maps concentration values to ASCII density characters
- `render.rs`: `ratatui` draw logic with `world_to_grid_coords()` for coordinate transformation

**`main.rs`** - Event loop: terminal setup (crossterm), tick-based update cycle (sense -> update_state -> render), input handling ('q' to quit). Uses saturating arithmetic for overflow safety.

### Key Mathematical Concepts

The agent uses stereo chemical sensors (left/right at configurable angle offset). Each tick:
1. Error = mean_sense - TARGET_CONCENTRATION (0.8)
2. Precision = learned confidence from spatial prior grid
3. Precision-weighted error = error × precision
4. Gradient = left_sensor - right_sensor
5. MCTS plans best action using learned priors as world model
6. Heading change = blend(reactive, planned) + exploration + noise + panic_turn + goal_attraction
7. Speed = MAX_SPEED * |error|
8. Update spatial priors with observation (Welford's algorithm)
9. Update episodic memory (landmark detection, decay, visit updates)
10. Angle normalized using `rem_euclid(2π)` for numerical stability

Boundary sensing returns -1.0 (toxic void) to create repulsion.

**MCTS Expected Free Energy**: For each trajectory τ:
- Pragmatic value: Σ prior.mean × state.energy (prefer high nutrients + survival)
- Epistemic value: Σ 1/precision (prefer uncertainty reduction)
- G(τ) = pragmatic + EXPLORATION_SCALE × epistemic

### Numerical Safety

- `assert_finite()` guards on critical calculations (mean_sense, error, gradient, d_theta, energy)
- Epsilon guard on Gaussian sigma_sq to prevent division by near-zero
- `rem_euclid()` instead of `%` for angle normalization
- Saturating arithmetic for sensor coordinate calculations

### Test Coverage

94 tests across 8 files covering:
- Agent: initialization, sensing, movement, energy, exhaustion, boundary clamping, angle normalization, temporal gradient, speed-error correlation
- Environment: initialization, concentration bounds, boundaries, Gaussian properties, source decay/respawn, Brownian motion bounds
- Memory: ring buffer operations, spatial grid updates, Welford's variance, precision calculation
- Episodic: landmark creation, decay, refresh, storage replacement, goal navigation
- Planning: MCTS rollouts, Expected Free Energy, action selection, trajectory validity
- Integration: cognitive stack integration, performance benchmarks, numerical stability
- Rendering: grid computation, coordinate transformation

### Code Style

- Strict clippy linting enabled (`#![warn(clippy::all, clippy::pedantic)]`)
- Files target <200 LOC
- Rust 2024 edition
- CI pipeline runs: fmt check, clippy, build, tests

### See Also

- [AGENTS.md](AGENTS.md) - Detailed project specification with mathematical formulas and algorithmic derivations
