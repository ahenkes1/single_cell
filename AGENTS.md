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

### D. The Dynamics (Action Update)
The agent updates its heading ($\theta$) and speed ($v$) to minimize the error over time.

**Heading Update:**
The turning rate is proportional to the Error times the Gradient.
$$\dot{\theta} = - \text{LEARNING\_RATE} \cdot E \cdot G + \text{Noise} + \text{Panic}$$
*Noise* is scaled by `NOISE_SCALE` (0.5) and proportional to Error.
*Panic* is a large random turn (±`PANIC_TURN_RANGE` radians) added if $G_{temp} <$ `PANIC_THRESHOLD` (-0.01).

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

---

## 3. Rust Implementation Plan & Checklist

### Architecture (Modules)
The project structure is strictly modularized to ensure files remain under 200 LOC.

*   `src/main.rs`: Entry point and event loop.
*   `src/simulation/`:
    *   `params.rs`: All hyperparameters organized into sections (Sensing, Behavior, Metabolism, Environment).
    *   `environment.rs`: `PetriDish` and `NutrientSource` logic with epsilon guards.
    *   `agent.rs`: `Protozoa` FEP logic with NaN propagation guards.
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

#### Step 4: Quality Assurance
- [x] **Linting:** `cargo clippy` (strict).
- [x] **Formatting:** `cargo fmt`.
- [x] **Tests:** `cargo test` passes.

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Developer guidance with build commands, code style, and quick reference for working with the codebase
