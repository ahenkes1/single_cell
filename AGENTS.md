# Project Specification: Protozoa - Continuous Active Inference Simulation

## 1. Project Overview
**Title:** Protozoa
**Platform:** Linux Terminal (Rust + `ratatui`)
**Genre:** Biological Simulation / Zero-Player Game

**Concept:** A real-time simulation of a single-cell organism (the Agent) living in a petri dish. The environment consists of continuous chemical gradients (Nutrients). The Agent navigates this world not through hard-coded rules (e.g., "If food is close, move to it"), but using the **Free Energy Principle (FEP)**.

The Agent minimizes the difference between its *genetic expectation* of the world (Homeostasis) and its *actual sensory input*. This results in emergent survival behaviors: seeking food when hungry, resting when satiated, and avoiding extremes.

---

## 2. Mathematical & Algorithmic Framework

### A. The Environment (Fields)
The domain is a continuous 2D plane $D \in \mathbb{R}^2$ with width $W$ and height $H$.
At any coordinate $(x, y)$, the **Nutrient Concentration** $C(x,y)$ is determined by the sum of Gaussian blobs:

$$C(x, y) = \sum_{i} I_i \cdot \exp\left( -\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma_i^2} \right)$$

* $I_i$: Intensity of food source $i$.
* $\sigma_i$: Radius/Spread of food source $i$.

### B. The Agent (Sensors & Actuators)
The agent has a position $(x, y)$ and a heading $\theta$ (radians).
It has **Stereo Vision** (two chemical receptors) to detect local gradients.
* **Sensor Distance ($d$):** Distance from body center to sensor.
* **Sensor Angle ($\delta$):** Offset angle.
* **Left Sensor ($s_L$):** Located at $\theta + \delta$.
* **Right Sensor ($s_R$):** Located at $\theta - \delta$.
* **Energy (ATP):** Internal energy store (0.0 to 1.0). Depletes with movement, refills with nutrient intake.

### C. The Active Inference Engine (Behavior)
The Agent operates by minimizing **Variational Free Energy ($F$)**.
We define $F$ based on the **Prediction Error** ($E$) relative to a **Target Set-Point** ($\rho$).

1.  **Sensation ($\mu$):** The average input.
    $$\mu = \frac{s_L + s_R}{2}$$
    *Boundary Logic:* If a sensor is outside the dish, it returns `-1.0` (Toxic Void), creating a strong repulsion gradient.
2.  **Target ($\rho$):** The homeostatic goal (e.g., 0.8 concentration).
3.  **Error ($E$):** $$E = \mu - \rho$$
4.  **Spatial Gradient ($G$):**
    $$G = s_L - s_R$$
5.  **Temporal Gradient ($G_{temp}$):**
    $$G_{temp} = \mu_t - \mu_{t-1}$$
    Used to detect if conditions are worsening over time, triggering a "panic turn" even if spatial gradient is zero.

### D. Cognitive Architecture

The agent has a multi-layer cognitive architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 1: SHORT-TERM MEMORY (Ring Buffer)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Last 32 experiences: (val_l, val_r, x, y, energy, tick)             ││
│  │ Used for: temporal gradient, pattern detection                       ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 2: LONG-TERM MEMORY (Spatial Prior Grid)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Grid cells: { mean: f64, m2: f64, visits: u32 }                     ││
│  │ Size: 20×10 cells (5×5 world units per cell)                        ││
│  │ Updated via Welford's algorithm for online mean/variance            ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 3: EPISODIC MEMORY (Landmark Store)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Landmarks: [Option<Landmark>; 8] (max 8 remembered locations)       ││
│  │ Landmark: { x, y, peak_nutrient, last_visit_tick, reliability }     ││
│  │ Triggers: Store when mean_sense > 0.7 (high-nutrient discovery)     ││
│  │ Used for: Goal-directed navigation when energy low                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 4: PLANNING (Monte Carlo Tree Search)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Rollouts: Simulate N=50 trajectories using learned spatial priors   ││
│  │ Objective: Maximize expected free energy (exploit + explore)        ││
│  │ Actions: Discrete heading changes (-45°, 0°, +45°)                  ││
│  │ Depth: 10 ticks lookahead                                           ││
│  │ Triggers: Every 20 ticks OR when energy < 0.3 (urgent replanning)   ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  CONTROL INTEGRATION                                                     │
│  base_error = sensation - TARGET_CONCENTRATION                          │
│  precision_weighted_error = base_error × prior_precision                │
│  exploration_bonus = EXPLORATION_SCALE / prior_precision                │
│  goal_attraction = direction_to_best_landmark (if energy < 0.3)         │
│  planned_heading = MCTS best action (updated every 20 ticks)            │
│  d_theta = blend(reactive_control, planned_heading) + goal_attraction   │
└─────────────────────────────────────────────────────────────────────────┘
```

### E. Mathematical Formulations

#### Spatial Prior Learning (Welford's Algorithm)
```
delta = observation - mean
mean += delta / visits
delta2 = observation - mean
m2 += delta * delta2

variance = m2 / (visits - 1)  # if visits >= 2
precision = visits / (1 + variance)
```

#### Precision-Weighted Control
```
base_error = mean_sense - TARGET_CONCENTRATION
precision = prior.precision().clamp(MIN_PRECISION, MAX_PRECISION)
precision_weighted_error = base_error × precision

reactive_d_theta = -LEARNING_RATE × precision_weighted_error × gradient
```

#### MCTS Expected Free Energy
For each trajectory τ = [state₁, state₂, ..., stateₙ]:
```
G(τ) = pragmatic + EXPLORATION_SCALE × epistemic

pragmatic = Σ prior.mean × state.energy    # prefer high nutrients + survival
epistemic = Σ 1 / prior.precision          # prefer uncertain regions (info gain)
```
Higher values are better (we maximize EFE).

#### Landmark Reliability Decay
```
reliability *= LANDMARK_DECAY  # 0.995 per tick when not visited
reliability = 1.0              # reset on revisit
```

#### Goal-Directed Navigation
When energy < MCTS_URGENT_ENERGY (0.3):
```
target_angle = atan2(landmark.y - y, landmark.x - x)
angle_diff = normalize_angle(target_angle - current_angle)
goal_attraction = LANDMARK_ATTRACTION_SCALE × angle_diff × landmark.reliability
```

### F. The Dynamics (Action Update)
The agent updates its heading ($\theta$) and speed ($v$) to minimize the error over time.

**Heading Update:**
The turning rate blends reactive control with planned actions:
$$\dot{\theta} = (1 - w_p) \cdot \dot{\theta}_{reactive} + w_p \cdot \dot{\theta}_{planned} + \text{Exploration} + \text{Noise} + \text{Panic} + \text{Goal}$$

Where:
- $\dot{\theta}_{reactive} = - \text{LEARNING\_RATE} \cdot E_{precision} \cdot G$
- $\dot{\theta}_{planned}$ = MCTS best action angle delta
- $w_p$ = PLANNING_WEIGHT (0.3)
- *Exploration* = random direction scaled by inverse precision
- *Noise* is scaled by `NOISE_SCALE` (0.5) and proportional to Error
- *Panic* is a large random turn (±`PANIC_TURN_RANGE` radians) if $G_{temp} <$ `PANIC_THRESHOLD` (-0.01)
- *Goal* = attraction toward remembered landmarks when energy < 0.3

**Speed Update:**
The agent conserves energy. It only moves when "anxious" (high error).
$$v = \text{MAX\_SPEED} \cdot |E|$$
*Modulation:* Speed is reduced by `EXHAUSTION_SPEED_FACTOR` (50%) if Energy ≤ `EXHAUSTION_THRESHOLD` (1%).

**Metabolism:**
*   **Cost:** `BASE_METABOLIC_COST` + (`SPEED_METABOLIC_COST` × speed_ratio) = 0.0005 + (0.0025 × speed_ratio)
*   **Intake:** `INTAKE_RATE` × mean_sense = 0.03 × mean_sense

**Numerical Safety:**
*   All critical calculations are guarded by `assert_finite()` to prevent NaN propagation
*   Angle normalization uses `rem_euclid(2π)` for numerical stability
*   Gaussian sigma uses epsilon guard: `sigma_sq.max(f64::EPSILON)`
*   Spatial priors ignore non-finite observations
*   M2 values clamped to non-negative

---

## 3. Rust Implementation Plan & Checklist

### Architecture (Modules)
The project structure is strictly modularized to ensure files remain under 200 LOC.

*   `src/main.rs`: Entry point and event loop.
*   `src/simulation/`:
    *   `params.rs`: All hyperparameters organized into sections (Sensing, Behavior, Metabolism, Environment, Memory, Learning, Episodic, Planning).
    *   `environment.rs`: `PetriDish` and `NutrientSource` logic with epsilon guards.
    *   `agent.rs`: `Protozoa` FEP logic with NaN propagation guards, memory systems, and MCTS integration.
    *   `memory/`:
        *   `mod.rs`: Memory module exports and `SensorSnapshot` type.
        *   `ring_buffer.rs`: Generic fixed-size ring buffer for short-term memory.
        *   `spatial_grid.rs`: 2D grid with Welford's online variance for spatial priors.
        *   `episodic.rs`: Landmark storage and goal-directed navigation support.
    *   `planning/`:
        *   `mod.rs`: Planning module exports.
        *   `mcts.rs`: Monte Carlo Tree Search with Expected Free Energy evaluation.
*   `src/ui/`:
    *   `field.rs`: Parallelized field calculation (`rayon`).
    *   `render.rs`: `ratatui` draw logic with coordinate transformation.

### Checklist

#### Step 1: Domain Logic (TDD)
- [x] **Parameters:** Define `PARAMS` struct/constants.
- [x] **Environment (`PetriDish`):**
    - TDD: Unit tests for `get_concentration` and random init.
    - Implement: Gaussian sum, decay, regrowth.
- [x] **Agent (`Protozoa`):**
    - TDD: Unit tests for movement, sensing, and energy.
    - Implement: FEP core (Error, Gradient, Panic).

#### Step 2: Parallel Rendering Engine
- [x] **Field Buffer:**
    - TDD: Test parallel iterator logic.
    - Implement: `compute_field_grid` using `rayon` to pre-calculate characters.
- [x] **TUI Components:**
    - Implement: `draw_ui` using `ratatui`.

#### Step 3: Application Loop
- [x] **Main Loop:**
    - Setup `crossterm` backend.
    - Integrate `update` -> `compute` -> `draw` loop.
    - Handle input.

#### Step 4: Memory & Learning Systems
- [x] **Short-Term Memory:**
    - Ring buffer implementation for sensor history.
    - 32-element buffer with O(1) push/access.
- [x] **Long-Term Memory (Spatial Priors):**
    - 20×10 grid covering the dish.
    - Welford's algorithm for online mean/variance.
    - Precision weighting for confidence.
- [x] **Episodic Memory:**
    - Landmark detection (threshold > 0.7).
    - Reliability decay (0.995 per tick).
    - Goal-directed navigation when energy low.

#### Step 5: Planning System
- [x] **MCTS Planner:**
    - 50 rollouts per action.
    - 10-step trajectory simulation.
    - Expected Free Energy evaluation (pragmatic + epistemic).
- [x] **World Model:**
    - Uses learned spatial priors (not actual environment).
    - Discrete actions: TurnLeft, Straight, TurnRight.
- [x] **Control Integration:**
    - Blend reactive + planned (30% planning weight).
    - Replan every 20 ticks or when energy < 0.3.

#### Step 6: Quality Assurance
- [x] **Linting:** `cargo clippy` (strict).
- [x] **Formatting:** `cargo fmt`.
- [x] **Tests:** `cargo test` passes (94 tests across 8 test files).

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Developer guidance with build commands, code style, and quick reference for working with the codebase
