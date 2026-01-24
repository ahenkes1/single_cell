# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

All commands run from `protozoa_rust/` directory:

```bash
cargo run --release      # Run simulation (use --release for optimal frame rates)
cargo test               # Run all tests (136 tests across 9 test files)
cargo fmt                # Format code
cargo clippy -- -D warnings  # Lint (strict, warnings as errors)
```

Static binary build (Linux MUSL):
```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## Mandatory Verification After Every Implementation

**CRITICAL: After EVERY code change, run all three checks in order:**

```bash
cargo fmt                     # Format code (no arguments = auto-fix)
cargo clippy -- -D warnings   # Lint with strictest settings (warnings = errors)
cargo test                    # Run full test suite
```

**Requirements:**
- All three must pass before committing
- Zero tolerance for warnings - every clippy warning must be addressed
- Every test must pass - no skipping or ignoring failures
- Format must be applied - no manual formatting exceptions

**If any check fails:**
1. Fix the issue immediately
2. Re-run all three checks
3. Only commit when all pass

## Mandatory Documentation Updates

**CRITICAL: After EVERY implementation, update all relevant documentation:**

- **CLAUDE.md** - Architecture overview, module descriptions, test counts
- **AGENTS.md** - Mathematical formulas, algorithmic specifications
- **README.md** - Features, configuration, user-facing documentation

**Requirements:**
- Documentation must accurately reflect the current implementation
- Test counts must be updated when tests are added
- New modules/features must be documented in all relevant files
- Mathematical formulas must match the actual code

## Architecture Overview

This is a genuine Active Inference biological simulation where a single-cell agent (Protozoa) navigates a nutrient-rich petri dish using the Free Energy Principle. The agent maintains Gaussian beliefs over hidden states and minimizes Variational Free Energy through gradient descent, selecting actions by minimizing Expected Free Energy.

### Core Modules

**`simulation/`** - Domain logic
- `agent.rs`: Protozoa struct implementing Continuous Active Inference with Gaussian beliefs, memory systems, and MCTS planning. Key algorithm: `update_state()` performs VFE gradient descent on beliefs, updates precision estimates, selects actions via EFE, and executes movement. Includes NaN propagation guards via `assert_finite()` helper function.
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
  - **Active Inference**: `BELIEF_LEARNING_RATE` (0.15), `MAX_VFE` (5.0), `INITIAL_SENSORY_PRECISION` (5.0), `NUTRIENT_PRIOR_PRECISION` (2.0), `MIN/MAX_SENSORY_PRECISION`, `UNCERTAINTY_GROWTH/REDUCTION`

**`simulation/inference/`** - Active Inference engine
- `beliefs.rs`: Gaussian belief state q(s) = N(μ, Σ) with `BeliefState`, `BeliefMean`, `BeliefCovariance`. Methods for gradient descent updates and uncertainty management.
- `generative_model.rs`: Generative model p(o,s) = p(o|s)×p(s) with `PriorMean`, `PriorPrecision`, `SensoryPrecision`. Observation function g(s) and Jacobian ∂g/∂s.
- `free_energy.rs`: Variational Free Energy F, VFE gradient ∂F/∂μ, Expected Free Energy G(π), and prediction error computation.
- `precision.rs`: Online precision estimation from prediction errors using exponential moving average.

**`simulation/memory/`** - Memory systems
- `ring_buffer.rs`: Generic fixed-size circular buffer for short-term memory
- `spatial_grid.rs`: 2D grid with Welford's online variance algorithm for spatial priors
- `episodic.rs`: Landmark storage with reliability decay for goal-directed navigation

**`simulation/planning/`** - Planning systems
- `mcts.rs`: Monte Carlo Tree Search with Expected Free Energy (pragmatic + epistemic value)

**`ui/`** - Rendering
- `field.rs`: Parallel grid computation using `rayon`. Maps concentration values to ASCII density characters
- `render.rs`: `ratatui` draw logic with sidebar layout. Key functions:
  - `compute_sidebar_layout()`: 70%/30% horizontal split (main + sidebar)
  - `draw_dashboard()`: Orchestrates all panels
  - `draw_petri_dish_panel()`: ASCII environment visualization (left, full height)
  - `draw_metrics_panel()`: Agent stats - energy, mode, sensors (sidebar top)
  - `draw_mcts_panel()`: Planning info - best action, EFE breakdown (sidebar)
  - `draw_landmarks_panel()`: Episodic memory table (sidebar)
  - `draw_spatial_grid_panel()`: Spatial priors heatmap with compression (sidebar bottom)
  - `compress_spatial_grid()`: Dynamic grid compression for narrow panels

**`main.rs`** - Event loop: terminal setup (crossterm), tick-based update cycle (sense -> update_state -> render), input handling ('q' to quit). Uses saturating arithmetic for overflow safety.

### Key Mathematical Concepts

**Variational Free Energy (VFE):**
```
F = ½(o - g(μ))ᵀ Πₒ (o - g(μ)) + ½(μ - η)ᵀ Πη (μ - η)
```
Where: o = observations, g(μ) = predicted observations, Πₒ = sensory precision, μ = belief mean, η = prior mean, Πη = prior precision.

**Belief Update (Gradient Descent on VFE):**
```
dμ/dt = -∂F/∂μ = Πₒ × J × (o - g(μ)) - Πη × (μ - η)
```
Where J = observation Jacobian ∂g/∂μ.

**Expected Free Energy (EFE) for Action Selection:**
```
G(π) = Risk + Ambiguity - Epistemic
Risk = deviation from preferred nutrient concentration
Ambiguity = uncertainty in predicted observations
Epistemic = information gain (uncertainty reduction)
```

The agent uses stereo chemical sensors (left/right at configurable angle offset). Each tick:
1. **Infer**: Compute VFE gradient and update Gaussian beliefs via gradient descent
2. **Learn**: Update sensory precision from prediction errors (EMA)
3. **Plan**: Evaluate actions by Expected Free Energy, select minimum
4. **Act**: Blend reactive gradient + planned action + exploration + panic + goal attraction
5. **Update**: Spatial priors (Welford), episodic memory (landmarks), position
6. Speed = MAX_SPEED × (VFE / MAX_VFE), clamped to [0, 1]
7. Angle normalized using `rem_euclid(2π)` for numerical stability

Boundary sensing returns -1.0 (toxic void) to create repulsion.

### Numerical Safety

- `assert_finite()` guards on critical calculations (mean_sense, error, gradient, d_theta, energy)
- Epsilon guard on Gaussian sigma_sq to prevent division by near-zero
- `rem_euclid()` instead of `%` for angle normalization
- Saturating arithmetic for sensor coordinate calculations

### Test Coverage

136 tests across 9 files covering:
- Agent: initialization, sensing, movement, energy, exhaustion, boundary clamping, angle normalization, temporal gradient, speed-error correlation
- Inference: belief state operations, VFE computation, VFE gradient descent, EFE evaluation, prediction errors, precision estimation
- Environment: initialization, concentration bounds, boundaries, Gaussian properties, source decay/respawn, Brownian motion bounds
- Memory: ring buffer operations, spatial grid updates, Welford's variance, precision calculation
- Episodic: landmark creation, decay, refresh, storage replacement, goal navigation
- Planning: MCTS rollouts, Expected Free Energy, action selection, trajectory validity
- Integration: cognitive stack integration, performance benchmarks, numerical stability
- Rendering: grid computation, coordinate transformation, sidebar layout, panel rendering, grid compression

### Code Style

- Strict clippy linting enabled (`#![warn(clippy::all, clippy::pedantic)]`)
- Files target <200 LOC
- Rust 2024 edition
- CI pipeline runs: fmt check, clippy, build, tests

### See Also

- [AGENTS.md](AGENTS.md) - Detailed project specification with mathematical formulas and algorithmic derivations
