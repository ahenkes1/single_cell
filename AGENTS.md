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
$$\dot{\theta} = - \text{learning\_rate} \cdot E \cdot G + \text{Noise} + \text{Panic}$$
*Noise* is added proportional to Error.
*Panic* is a large random turn added if $G_{temp} < -0.01$ (conditions getting worse).

**Speed Update:**
The agent conserves energy. It only moves when "anxious" (high error).
$$v = \text{max\_speed} \cdot |E|$$
*Modulation:* Speed is reduced (50%) if Energy is depleted (< 1%).

**Metabolism:**
*   **Cost:** 0.0005 + (0.0025 * speed_ratio)
*   **Intake:** 0.03 * mean_sense

---

## 3. Rust Implementation Plan & Checklist

### Architecture (Modules)
The project structure is strictly modularized to ensure files remain under 200 LOC.

*   `src/main.rs`: Entry point and event loop.
*   `src/simulation/`:
    *   `params.rs`: Hyperparameters.
    *   `environment.rs`: `PetriDish` and `NutrientSource` logic.
    *   `agent.rs`: `Protozoa` FEP logic.
*   `src/ui/`:
    *   `field.rs`: Parallelized field calculation (`rayon`).
    *   `render.rs`: `ratatui` draw logic.

### Checklist

#### Step 1: Domain Logic (TDD)
- [ ] **Parameters:** Define `PARAMS` struct/constants.
- [ ] **Environment (`PetriDish`):**
    - TDD: Unit tests for `get_concentration` and random init.
    - Implement: Gaussian sum, decay, regrowth.
- [ ] **Agent (`Protozoa`):**
    - TDD: Unit tests for movement, sensing, and energy.
    - Implement: FEP core (Error, Gradient, Panic).

#### Step 2: Parallel Rendering Engine
- [ ] **Field Buffer:**
    - TDD: Test parallel iterator logic.
    - Implement: `compute_field_grid` using `rayon` to pre-calculate characters.
- [ ] **TUI Components:**
    - Implement: `draw_ui` using `ratatui`.

#### Step 3: Application Loop
- [ ] **Main Loop:**
    - Setup `crossterm` backend.
    - Integrate `update` -> `compute` -> `draw` loop.
    - Handle input.

#### Step 4: Quality Assurance
- [ ] **Linting:** `cargo clippy` (strict).
- [ ] **Formatting:** `cargo fmt`.
- [ ] **Tests:** `cargo test` passes.
