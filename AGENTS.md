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
The Agent operates by minimizing **Variational Free Energy ($F$)** through gradient descent on Gaussian beliefs, and selects actions by minimizing **Expected Free Energy ($G$)**.

#### Generative Model
The agent maintains an internal model $p(o, s) = p(o|s) \cdot p(s)$:
- **Likelihood** $p(o|s)$: Observation function $g(s)$ with sensory precision $\Pi_o$
- **Prior** $p(s)$: Preferences encoded as prior mean $\eta$ with precision $\Pi_\eta$

#### Gaussian Beliefs
The agent maintains approximate posterior beliefs $q(s) = \mathcal{N}(\mu, \Sigma)$:
- **Belief Mean** $\mu$: (nutrient, x, y, angle) - the agent's best estimate of hidden states
- **Belief Covariance** $\Sigma$: Uncertainty over each hidden state (diagonal for efficiency)

#### Variational Free Energy
$$F = \frac{1}{2}(o - g(\mu))^T \Pi_o (o - g(\mu)) + \frac{1}{2}(\mu - \eta)^T \Pi_\eta (\mu - \eta)$$

Where:
- $o = (s_L, s_R)$: Observations (left/right sensor readings)
- $g(\mu)$: Predicted observations from beliefs
- $\Pi_o$: Sensory precision (inverse observation variance, learned online)
- $\eta$: Prior mean (nutrient target = 0.8, encodes preferences)
- $\Pi_\eta$: Prior precision (strength of preferences)

#### Belief Update (Gradient Descent)
$$\frac{d\mu}{dt} = -\frac{\partial F}{\partial \mu} = \Pi_o \cdot J \cdot (o - g(\mu)) - \Pi_\eta \cdot (\mu - \eta)$$

Where $J = \frac{\partial g}{\partial \mu}$ is the observation Jacobian.

#### Sensory Precision Learning
Precision is estimated online from prediction errors using exponential moving average:
$$\sigma^2 \leftarrow (1-\alpha)\sigma^2 + \alpha \cdot \epsilon^2$$
$$\Pi_o = 1/\sigma^2 \text{ (clamped to valid range)}$$

#### Supporting Signals
1.  **Spatial Gradient ($G$):** $$G = s_L - s_R$$
2.  **Temporal Gradient ($G_{temp}$):** $$G_{temp} = \mu_t - \mu_{t-1}$$
    Used to detect if conditions are worsening over time, triggering a "panic turn".
3.  *Boundary Logic:* If a sensor is outside the dish, it returns `-1.0` (Toxic Void).

### D. Cognitive Architecture

The agent has a multi-layer cognitive architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 0: ACTIVE INFERENCE ENGINE                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Beliefs: q(s) = N(μ, Σ) over (nutrient, x, y, angle)                ││
│  │ Generative Model: p(o,s) = p(o|s) × p(s)                            ││
│  │ Inference: Gradient descent on VFE: dμ/dt = -∂F/∂μ                  ││
│  │ Precision: Learned online from prediction errors (EMA)              ││
│  │ Action: Select by minimizing Expected Free Energy G(π)              ││
│  └─────────────────────────────────────────────────────────────────────┘│
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
│  │ Objective: Minimize Expected Free Energy (EFE)                      ││
│  │ Actions: Discrete heading changes (-45°, 0°, +45°)                  ││
│  │ Depth: 10 ticks lookahead                                           ││
│  │ Triggers: Every 20 ticks OR when energy < 0.3 (urgent replanning)   ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  CONTROL INTEGRATION                                                     │
│  VFE = variational_free_energy(observations, beliefs, model)            │
│  gradient = vfe_gradient(observations, beliefs, model)                  │
│  beliefs.update(gradient, learning_rate)  # Perception                  │
│  best_action = argmin_π G(π)              # Action selection            │
│  d_theta = blend(reactive, planned) + goal_attraction + panic           │
└─────────────────────────────────────────────────────────────────────────┘
```

### E. Mathematical Formulations

#### Variational Free Energy (VFE)
```
F = ½ × Σᵢ πₒᵢ × (oᵢ - gᵢ(μ))² + ½ × Σⱼ πηⱼ × (μⱼ - ηⱼ)²

Where:
- πₒ = sensory precision (learned from prediction errors)
- πη = prior precision (strength of preferences)
- g(μ) = observation function (predicts sensors from beliefs)
```

#### VFE Gradient (for Belief Updates)
```
∂F/∂μ_nutrient = -πₒ_L × (o_L - pred_L) - πₒ_R × (o_R - pred_R) + πη × (μ_nutrient - η)

Gradient includes Jacobian terms from observation function:
∂g_L/∂nutrient = 1, ∂g_R/∂nutrient = 1
∂g_L/∂angle = gradient_factor × cos(angle)
∂g_R/∂angle = -gradient_factor × cos(angle)
```

#### Expected Free Energy (EFE) for Action Selection
```
G(π) = Risk + Ambiguity - Epistemic

Risk = |μ_nutrient - η_nutrient|  # deviation from preferred state
Ambiguity = σ²_nutrient           # uncertainty in predictions
Epistemic = -log(σ²_nutrient)     # information gain (negative = good)
```
Lower EFE is better (we minimize G).

#### Precision Learning (Exponential Moving Average)
```
σ² ← (1-α)σ² + α × ε²           # update variance estimate
π = 1/σ²                         # precision = inverse variance
π = clamp(π, MIN_PRECISION, MAX_PRECISION)
```

#### Spatial Prior Learning (Welford's Algorithm)
```
delta = observation - mean
mean += delta / visits
delta2 = observation - mean
m2 += delta * delta2

variance = m2 / (visits - 1)  # if visits >= 2
```

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
    *   `params.rs`: All hyperparameters organized into sections (Sensing, Behavior, Metabolism, Environment, Memory, Learning, Episodic, Planning, Active Inference).
    *   `environment.rs`: `PetriDish` and `NutrientSource` logic with epsilon guards.
    *   `agent.rs`: `Protozoa` implementing Continuous Active Inference with Gaussian beliefs, VFE minimization, EFE action selection, memory systems, and MCTS integration.
    *   `inference/`:
        *   `mod.rs`: Inference module exports.
        *   `beliefs.rs`: Gaussian belief state q(s) = N(μ, Σ) with update methods.
        *   `generative_model.rs`: Generative model p(o,s) with observation function and Jacobian.
        *   `free_energy.rs`: VFE computation, VFE gradient, EFE evaluation, prediction errors.
        *   `precision.rs`: Online precision estimation from prediction errors.
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
    *   `render.rs`: `ratatui` draw logic with sidebar layout:
        *   `compute_sidebar_layout()`: 70%/30% horizontal split
        *   `draw_dashboard()`: Orchestrates panel rendering
        *   Left panel (70%): Petri Dish visualization (full height)
        *   Right sidebar (30%): Agent metrics, MCTS planning, Landmarks, Spatial Memory
        *   `compress_spatial_grid()`: Dynamic grid compression for narrow panels

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
- [x] **Tests:** `cargo test` passes (136 tests across 9 test files).

### Mandatory Documentation Updates

**CRITICAL: After EVERY implementation, update all relevant documentation:**

- **CLAUDE.md** - Architecture overview, module descriptions, test counts
- **AGENTS.md** - Mathematical formulas, algorithmic specifications
- **README.md** - Features, configuration, user-facing documentation

**Requirements:**
- Documentation must accurately reflect the current implementation
- Test counts must be updated when tests are added
- New modules/features must be documented in all relevant files
- Mathematical formulas must match the actual code

### Mandatory Verification After Every Implementation

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

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Developer guidance with build commands, code style, and quick reference for working with the codebase
