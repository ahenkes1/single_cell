# Project Specification: Protozoa - Continuous Active Inference Simulation

## 1. Project Overview
**Title:** Protozoa
**Platform:** Linux Terminal (Python + `curses`)
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
2.  **Target ($\rho$):** The homeostatic goal (e.g., 0.8 concentration).
3.  **Error ($E$):** $$E = \mu - \rho$$
4.  **Spatial Gradient ($G$):**
    $$G = s_L - s_R$$

### D. The Dynamics (Action Update)
The agent updates its heading ($\theta$) and speed ($v$) to minimize the error over time.

**Heading Update:**
The turning rate is proportional to the Error times the Gradient.
$$\dot{\theta} = - \text{learning\_rate} \cdot E \cdot G + \text{Noise}$$
*Noise* is added proportional to Error to allow escaping local minima (zero-gradient zones).

**Speed Update:**
The agent conserves energy. It only moves when "anxious" (high error).
$$v = \text{max\_speed} \cdot |E|$$
*Modulation:* Speed is reduced if Energy is depleted.

---

## 3. Implementation Plan & Checklist

### Class Structure
The code should be organized into three classes: `PetriDish`, `Protozoa`, and `Simulation`.

### Checklist for Coding Agent

#### Step 1: The Environment (`PetriDish` Class)
- [x] **Initialization:**
    - Accept `width` and `height` (floats, e.g., 100.0).
    - Initialize a list of nutrient sources (`sources`). Each source is a dict: `{x, y, radius, intensity}`.
    - Create 5-10 random sources on init.
- [x] **Math Helper `get_concentration(x, y)`:**
    - Input: float x, float y.
    - Logic: Sum the Gaussian contributions of all sources at this point.
    - Output: Float clipped between 0.0 and 1.0.
- [x] **Dynamics `update()`:**
    - Entropy: Reduce the `intensity` of all sources by a small decay factor (e.g., 0.995) every frame.
    - Brownian Motion: Randomly jiggle the `x, y` of sources slightly.
    - Regrowth: If a source fades below a threshold, respawn it at a new random location to keep the simulation running.

#### Step 2: The FEP Agent (`Protozoa` Class)
- [x] **Initialization:**
    - `x, y`: Center of world.
    - `angle`: Random 0 to $2\pi$.
    - `speed`: 0.0.
    - **Hyperparameters:**
        - `target` ($\rho$): 0.8
        - `sensor_dist`: 2.0
        - `sensor_angle`: 0.5 radians
        - `learning_rate`: 0.15 (Turn speed)
        - `max_speed`: 1.5
- [x] **Method `sense(dish)`:**
    - Calculate world coordinates for Left Sensor:
        $$x_L = x + d \cdot \cos(\theta + \delta)$$
        $$y_L = y + d \cdot \sin(\theta + \delta)$$
    - Calculate world coordinates for Right Sensor (use $\theta - \delta$).
    - Query `dish.get_concentration` for both points.
    - Store `val_L` and `val_R`.
- [x] **Method `update_state()` (The FEP Core):**
    - Compute `mean_sense` ($\mu$).
    - Compute `error` ($E = \mu - \rho$).
    - Compute `gradient` ($G = s_L - s_R$).
    - Update Angle: `angle += -learning_rate * error * gradient`.
    - Update Speed: `speed = abs(error) * max_speed`.
    - Update Position: 
        * `x += speed * cos(angle)`
        * `y += speed * sin(angle)`
    - **Boundary Check:** Clamp `x, y` to be within `0` and `dish.width/height`.

#### Step 3: Visualization (`Simulation` Class)
- [x] **Libraries:** Import `curses`, `time`, `math`, `random`.
- [x] **Mapping:** Create a function to map continuous world coordinates to integer terminal row/col.
- [x] **Render Logic:**
    - **Field:** Iterate over terminal rows/cols. Convert back to world float coordinates. Query `dish.get_concentration`. Select ASCII char from map: ` .:-=+*#%@` (Empty to Dense).
    - **Optimization:** Step by 2 pixels if rendering is slow.
    - **Agent:** Draw the agent character (e.g., `O` or `Q`) at its integer position. Draw sensor markers (`'` or `.`) to indicate heading.
    - **HUD:** Draw a text bar at the top: `Sens: {:.2f} | Tgt: {:.2f} | Err: {:.2f} | Spd: {:.2f}`.
- [x] **Main Loop:**
    - Initialize `curses` (noecho, nodelay, curs_set 0).
    - Loop forever:
        1.  `dish.update()`
        2.  `agent.sense(dish)`
        3.  `agent.update_state()`
        4.  `render()`
        5.  `time.sleep(0.05)`
    - **Safety:** Use `try...finally` to ensure `curses.endwin()` is called even if the user crashes the script or hits Ctrl+C.
