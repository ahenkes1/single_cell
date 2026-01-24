# Implementation Plan: Continuous State-Space Active Inference

**Date:** 2026-01-24
**Goal:** Transform the Protozoa simulation into a genuine continuous Active Inference agent
**Formulation:** Gaussian beliefs with gradient-based variational inference

---

## Executive Summary

This plan transforms the current homeostatic gradient-following agent into a genuine continuous Active Inference system. The key architectural changes:

1. **Generative Model**: Explicit probabilistic model with Gaussian likelihood and priors
2. **Belief States**: Gaussian distributions (μ, Σ) over hidden states, not point estimates
3. **Variational Inference**: Gradient descent on Free Energy to update beliefs
4. **Expected Free Energy**: Proper EFE with Risk (KL from preferences) and Ambiguity (expected entropy)
5. **Precision**: True sensory precision as inverse variance of observation noise

---

## Part 1: Mathematical Foundation

### 1.1 Generative Model Specification

For continuous Active Inference, we define a generative model:

```
p(o, s) = p(o | s) × p(s)
```

**Hidden States (s):**
```
s = [s_nutrient, s_x, s_y, s_θ]ᵀ

Where:
- s_nutrient ∈ [0, 1]: Believed local nutrient concentration
- s_x, s_y: Believed position
- s_θ: Believed heading
```

**Observations (o):**
```
o = [o_L, o_R]ᵀ

Where:
- o_L: Left sensor reading
- o_R: Right sensor reading
```

**Likelihood p(o | s):**
```
p(o | s) = N(o; g(s), Σ_o)

Where:
- g(s) = [g_L(s), g_R(s)]ᵀ is the observation function
- g_L(s) = s_nutrient + gradient_offset(s_θ, s_x, s_y)
- Σ_o = diag(1/ω_L, 1/ω_R) is sensory covariance (inverse precision)
```

**Prior p(s):**
```
p(s) = N(s; η, Σ_η)

Where:
- η = [η_nutrient, η_x, η_y, η_θ]ᵀ are prior means
- η_nutrient = 0.8 (homeostatic target - this encodes preferences!)
- Σ_η encodes prior uncertainty
```

### 1.2 Variational Free Energy

For Gaussian approximations (Laplace):

```
F = -ln p(o, s) + ln q(s)
  ≈ (1/2)(o - g(μ))ᵀ Π_o (o - g(μ))     [Sensory prediction error]
  + (1/2)(μ - η)ᵀ Π_η (μ - η)            [Prior prediction error]
  + (1/2)ln|Σ| + const                   [Entropy term]
```

Where:
- μ = E_q[s] is the posterior mean (beliefs)
- Σ = Cov_q[s] is the posterior covariance
- Π_o = Σ_o⁻¹ is sensory precision
- Π_η = Σ_η⁻¹ is prior precision

### 1.3 Belief Update (Gradient Descent on F)

```
dμ/dt = -∂F/∂μ = Π_o × ∂g/∂s × (o - g(μ)) + Π_η × (η - μ)
```

This is the **prediction error dynamics** - beliefs are updated by precision-weighted prediction errors.

### 1.4 Expected Free Energy (for Planning)

```
G(π) = E_Q(o,s|π)[ln Q(s|π) - ln P(o,s)]

For Gaussian:
G(π) ≈ (1/2)(μ_π - η)ᵀ Π_η (μ_π - η)           [Risk: divergence from preferences]
      + (1/2)tr(Π_o × Σ_π)                      [Ambiguity: expected sensory uncertainty]
      - (1/2)ln|Σ_π|                            [Epistemic value: belief uncertainty]
```

### 1.5 Action Selection

Actions are selected to minimize EFE:

```
a* = argmin_a G(π_a)
```

Where π_a is the policy starting with action a.

---

## Part 2: Architecture Overview

### 2.1 New Module Structure

```
src/simulation/
├── mod.rs
├── params.rs                    # Extended with new parameters
├── environment.rs               # Unchanged
├── agent.rs                     # Refactored to use new inference
├── memory/
│   ├── mod.rs
│   ├── ring_buffer.rs           # Unchanged
│   ├── spatial_grid.rs          # Unchanged (used for world model learning)
│   └── episodic.rs              # Unchanged
├── planning/
│   ├── mod.rs
│   ├── mcts.rs                  # Refactored with proper EFE
│   └── policy.rs                # NEW: Policy representation
└── inference/                   # NEW MODULE
    ├── mod.rs
    ├── generative_model.rs      # Likelihood, prior, observation function
    ├── beliefs.rs               # Gaussian belief state
    ├── free_energy.rs           # VFE computation
    └── precision.rs             # Precision estimation
```

### 2.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE LOOP                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SENSE                                                           │
│     o = [o_L, o_R] ← environment.get_concentration(sensor_pos)      │
│                                                                      │
│  2. INFER (minimize VFE via gradient descent)                       │
│     prediction_error = o - g(μ)                                     │
│     prior_error = μ - η                                             │
│     dμ/dt = Π_o × ∂g/∂μ × pred_error - Π_η × prior_error           │
│     μ ← μ + learning_rate × dμ/dt                                   │
│                                                                      │
│  3. PLAN (minimize EFE)                                             │
│     For each candidate action a:                                    │
│       Predict: μ_a, Σ_a = forward_model(μ, Σ, a)                   │
│       Compute: G(a) = risk(μ_a) + ambiguity(Σ_a) - epistemic(Σ_a)  │
│     Select: a* = argmin G(a)                                        │
│                                                                      │
│  4. ACT                                                             │
│     Execute a* (change heading, move)                               │
│                                                                      │
│  5. LEARN (update world model)                                      │
│     spatial_priors.update(x, y, o_mean)                             │
│     Update precision estimates from prediction errors               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Implementation Steps

### Step 1: Create Inference Module Foundation

**File: `src/simulation/inference/mod.rs`**
```rust
//! Continuous Active Inference engine.
//!
//! Implements variational inference with Gaussian beliefs
//! for perception and action under the Free Energy Principle.

mod beliefs;
mod free_energy;
mod generative_model;
mod precision;

pub use beliefs::BeliefState;
pub use free_energy::{variational_free_energy, expected_free_energy};
pub use generative_model::GenerativeModel;
pub use precision::PrecisionEstimator;
```

### Step 2: Implement Belief State

**File: `src/simulation/inference/beliefs.rs`**
```rust
//! Gaussian belief state over hidden variables.

/// Represents Gaussian beliefs: q(s) = N(μ, Σ)
#[derive(Clone, Debug)]
pub struct BeliefState {
    /// Posterior mean (believed state)
    pub mean: BeliefMean,
    /// Posterior covariance (uncertainty)
    pub covariance: BeliefCovariance,
}

/// Mean of beliefs over hidden states
#[derive(Clone, Copy, Debug)]
pub struct BeliefMean {
    /// Believed nutrient concentration at current location
    pub nutrient: f64,
    /// Believed x position
    pub x: f64,
    /// Believed y position
    pub y: f64,
    /// Believed heading
    pub angle: f64,
}

/// Diagonal covariance (assumes independence for simplicity)
#[derive(Clone, Copy, Debug)]
pub struct BeliefCovariance {
    pub nutrient_var: f64,
    pub x_var: f64,
    pub y_var: f64,
    pub angle_var: f64,
}

impl BeliefState {
    /// Create initial beliefs with high uncertainty
    pub fn new(x: f64, y: f64, angle: f64) -> Self {
        Self {
            mean: BeliefMean {
                nutrient: 0.5,  // Neutral prior
                x,
                y,
                angle,
            },
            covariance: BeliefCovariance {
                nutrient_var: 0.25,  // High initial uncertainty
                x_var: 1.0,
                y_var: 1.0,
                angle_var: 0.5,
            },
        }
    }

    /// Update beliefs via gradient descent on VFE
    pub fn update(&mut self, gradient: &BeliefMean, learning_rate: f64) {
        self.mean.nutrient += learning_rate * gradient.nutrient;
        self.mean.x += learning_rate * gradient.x;
        self.mean.y += learning_rate * gradient.y;
        self.mean.angle += learning_rate * gradient.angle;

        // Clamp to valid ranges
        self.mean.nutrient = self.mean.nutrient.clamp(0.0, 1.0);
        self.mean.angle = self.mean.angle.rem_euclid(2.0 * std::f64::consts::PI);
    }

    /// Total uncertainty (trace of covariance)
    pub fn total_uncertainty(&self) -> f64 {
        self.covariance.nutrient_var
            + self.covariance.x_var
            + self.covariance.y_var
            + self.covariance.angle_var
    }

    /// Log determinant of covariance (for entropy)
    pub fn log_det_covariance(&self) -> f64 {
        (self.covariance.nutrient_var
            * self.covariance.x_var
            * self.covariance.y_var
            * self.covariance.angle_var)
            .ln()
    }
}
```

### Step 3: Implement Generative Model

**File: `src/simulation/inference/generative_model.rs`**
```rust
//! Generative model: p(o, s) = p(o|s) × p(s)

use crate::simulation::params::{SENSOR_ANGLE, SENSOR_DIST, TARGET_CONCENTRATION};
use super::beliefs::{BeliefMean, BeliefCovariance};

/// The agent's generative model of the world
#[derive(Clone, Debug)]
pub struct GenerativeModel {
    /// Prior mean (homeostatic target encodes preferences)
    pub prior_mean: PriorMean,
    /// Prior precision (inverse covariance)
    pub prior_precision: PriorPrecision,
    /// Sensory precision (inverse observation noise)
    pub sensory_precision: SensoryPrecision,
}

#[derive(Clone, Copy, Debug)]
pub struct PriorMean {
    pub nutrient: f64,  // TARGET_CONCENTRATION - this is the preference!
    pub x: f64,
    pub y: f64,
    pub angle: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PriorPrecision {
    pub nutrient: f64,  // How strongly to prefer target concentration
    pub x: f64,
    pub y: f64,
    pub angle: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SensoryPrecision {
    pub left: f64,   // Precision of left sensor
    pub right: f64,  // Precision of right sensor
}

impl GenerativeModel {
    pub fn new() -> Self {
        Self {
            prior_mean: PriorMean {
                nutrient: TARGET_CONCENTRATION,  // Preference encoded as prior!
                x: 50.0,  // Center of dish
                y: 25.0,
                angle: 0.0,
            },
            prior_precision: PriorPrecision {
                nutrient: 2.0,  // Strong preference for target
                x: 0.01,        // Weak position prior
                y: 0.01,
                angle: 0.01,
            },
            sensory_precision: SensoryPrecision {
                left: 5.0,   // Reasonably reliable sensors
                right: 5.0,
            },
        }
    }

    /// Observation function: g(s) - predicts observations from hidden states
    /// Returns (predicted_left, predicted_right)
    pub fn observation_function(&self, beliefs: &BeliefMean) -> (f64, f64) {
        // Base prediction is believed nutrient concentration
        let base = beliefs.nutrient;

        // Gradient creates differential between sensors
        // This is a simplification - in full model would use spatial model
        let gradient_factor = 0.1;  // How much gradient affects sensors

        // Left sensor slightly ahead-left, right sensor ahead-right
        let predicted_left = base + gradient_factor * beliefs.angle.sin();
        let predicted_right = base - gradient_factor * beliefs.angle.sin();

        (predicted_left.clamp(0.0, 1.0), predicted_right.clamp(0.0, 1.0))
    }

    /// Jacobian of observation function: ∂g/∂s
    pub fn observation_jacobian(&self, beliefs: &BeliefMean) -> ObservationJacobian {
        let gradient_factor = 0.1;

        ObservationJacobian {
            // ∂g_L/∂nutrient, ∂g_R/∂nutrient
            d_obs_d_nutrient: (1.0, 1.0),
            // ∂g_L/∂angle, ∂g_R/∂angle
            d_obs_d_angle: (
                gradient_factor * beliefs.angle.cos(),
                -gradient_factor * beliefs.angle.cos(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ObservationJacobian {
    pub d_obs_d_nutrient: (f64, f64),  // (left, right)
    pub d_obs_d_angle: (f64, f64),
}
```

### Step 4: Implement Free Energy Computation

**File: `src/simulation/inference/free_energy.rs`**
```rust
//! Variational and Expected Free Energy computation.

use super::beliefs::BeliefState;
use super::generative_model::GenerativeModel;

/// Compute Variational Free Energy (for perception)
///
/// F = (1/2)(o - g(μ))ᵀ Π_o (o - g(μ)) + (1/2)(μ - η)ᵀ Π_η (μ - η)
pub fn variational_free_energy(
    observations: (f64, f64),
    beliefs: &BeliefState,
    model: &GenerativeModel,
) -> f64 {
    let (obs_l, obs_r) = observations;
    let (pred_l, pred_r) = model.observation_function(&beliefs.mean);

    // Sensory prediction error (precision-weighted)
    let sensory_error_l = obs_l - pred_l;
    let sensory_error_r = obs_r - pred_r;
    let sensory_term = 0.5 * model.sensory_precision.left * sensory_error_l.powi(2)
        + 0.5 * model.sensory_precision.right * sensory_error_r.powi(2);

    // Prior prediction error (precision-weighted)
    let prior_error_nutrient = beliefs.mean.nutrient - model.prior_mean.nutrient;
    let prior_term = 0.5 * model.prior_precision.nutrient * prior_error_nutrient.powi(2);

    sensory_term + prior_term
}

/// Compute gradient of VFE w.r.t. beliefs (for belief update)
///
/// ∂F/∂μ = -Π_o × ∂g/∂μ × (o - g(μ)) + Π_η × (μ - η)
pub fn vfe_gradient(
    observations: (f64, f64),
    beliefs: &BeliefState,
    model: &GenerativeModel,
) -> super::beliefs::BeliefMean {
    let (obs_l, obs_r) = observations;
    let (pred_l, pred_r) = model.observation_function(&beliefs.mean);
    let jacobian = model.observation_jacobian(&beliefs.mean);

    // Sensory prediction errors
    let error_l = obs_l - pred_l;
    let error_r = obs_r - pred_r;

    // Gradient w.r.t. nutrient belief
    let d_nutrient = -model.sensory_precision.left * jacobian.d_obs_d_nutrient.0 * error_l
        - model.sensory_precision.right * jacobian.d_obs_d_nutrient.1 * error_r
        + model.prior_precision.nutrient * (beliefs.mean.nutrient - model.prior_mean.nutrient);

    // Gradient w.r.t. angle belief
    let d_angle = -model.sensory_precision.left * jacobian.d_obs_d_angle.0 * error_l
        - model.sensory_precision.right * jacobian.d_obs_d_angle.1 * error_r;

    super::beliefs::BeliefMean {
        nutrient: -d_nutrient,  // Negative because we descend gradient
        x: 0.0,  // Position updated from proprioception, not inference
        y: 0.0,
        angle: -d_angle,
    }
}

/// Compute Expected Free Energy for a predicted future state (for planning)
///
/// G(π) = Risk + Ambiguity - Epistemic
///      = (1/2)(μ_π - η)ᵀ Π_η (μ_π - η) + (1/2)tr(Π_o Σ_π) - (1/2)ln|Σ_π|
pub fn expected_free_energy(
    predicted_beliefs: &BeliefState,
    model: &GenerativeModel,
) -> f64 {
    // Risk: KL divergence from preferred states (encoded in prior)
    let risk = 0.5 * model.prior_precision.nutrient
        * (predicted_beliefs.mean.nutrient - model.prior_mean.nutrient).powi(2);

    // Ambiguity: expected sensory uncertainty
    let ambiguity = 0.5 * (
        model.sensory_precision.left * predicted_beliefs.covariance.nutrient_var
        + model.sensory_precision.right * predicted_beliefs.covariance.nutrient_var
    );

    // Epistemic value: negative log determinant (prefer uncertainty reduction)
    let epistemic = -0.5 * predicted_beliefs.log_det_covariance();

    risk + ambiguity + epistemic
}
```

### Step 5: Implement Precision Estimation

**File: `src/simulation/inference/precision.rs`**
```rust
//! Online precision estimation from prediction errors.

/// Estimates sensory precision from accumulated prediction errors
#[derive(Clone, Debug)]
pub struct PrecisionEstimator {
    /// Running sum of squared errors (left sensor)
    sum_sq_error_l: f64,
    /// Running sum of squared errors (right sensor)
    sum_sq_error_r: f64,
    /// Number of observations
    count: u32,
    /// Minimum precision (prevents division by zero)
    min_precision: f64,
    /// Maximum precision (prevents over-confidence)
    max_precision: f64,
}

impl PrecisionEstimator {
    pub fn new(min_precision: f64, max_precision: f64) -> Self {
        Self {
            sum_sq_error_l: 0.0,
            sum_sq_error_r: 0.0,
            count: 0,
            min_precision,
            max_precision,
        }
    }

    /// Update with new prediction errors
    pub fn update(&mut self, error_l: f64, error_r: f64) {
        self.count = self.count.saturating_add(1);

        // Exponential moving average for adaptivity
        let alpha = 0.1;
        self.sum_sq_error_l = (1.0 - alpha) * self.sum_sq_error_l + alpha * error_l.powi(2);
        self.sum_sq_error_r = (1.0 - alpha) * self.sum_sq_error_r + alpha * error_r.powi(2);
    }

    /// Get estimated precision (inverse variance)
    pub fn precision_left(&self) -> f64 {
        if self.count < 2 {
            return self.min_precision;
        }
        let variance = self.sum_sq_error_l.max(0.01);
        (1.0 / variance).clamp(self.min_precision, self.max_precision)
    }

    pub fn precision_right(&self) -> f64 {
        if self.count < 2 {
            return self.min_precision;
        }
        let variance = self.sum_sq_error_r.max(0.01);
        (1.0 / variance).clamp(self.min_precision, self.max_precision)
    }
}
```

### Step 6: Refactor Agent to Use Inference

**File: `src/simulation/agent.rs` (modified)**

Key changes:
1. Add `BeliefState` as agent field
2. Add `GenerativeModel` as agent field
3. Replace error computation with VFE gradient
4. Replace speed heuristic with policy from EFE minimization

```rust
// New fields in Protozoa struct:
pub beliefs: BeliefState,
pub generative_model: GenerativeModel,
pub precision_estimator: PrecisionEstimator,

// New update_state logic:
pub fn update_state(&mut self, dish: &PetriDish) {
    let mut rng = rand::rng();

    // 1. Get observations
    let observations = (self.val_l, self.val_r);

    // 2. INFER: Update beliefs by gradient descent on VFE
    let gradient = vfe_gradient(observations, &self.beliefs, &self.generative_model);
    self.beliefs.update(&gradient, BELIEF_LEARNING_RATE);

    // 3. Update precision estimates from prediction errors
    let (pred_l, pred_r) = self.generative_model.observation_function(&self.beliefs.mean);
    self.precision_estimator.update(self.val_l - pred_l, self.val_r - pred_r);

    // Update model with learned precisions
    self.generative_model.sensory_precision.left = self.precision_estimator.precision_left();
    self.generative_model.sensory_precision.right = self.precision_estimator.precision_right();

    // 4. PLAN: Select action minimizing EFE
    let best_action = self.select_action_efe(&self.beliefs);

    // 5. ACT: Apply action
    let d_theta = best_action.angle_delta();
    self.angle += d_theta;
    self.angle = self.angle.rem_euclid(2.0 * PI);

    // Speed from "active inference" - move to reduce prediction error
    let vfe = variational_free_energy(observations, &self.beliefs, &self.generative_model);
    self.speed = MAX_SPEED * (vfe / MAX_VFE).clamp(0.0, 1.0);

    // ... rest of update (metabolism, position, etc.)
}

fn select_action_efe(&self, beliefs: &BeliefState) -> Action {
    let mut best_action = Action::Straight;
    let mut best_efe = f64::INFINITY;

    for action in Action::all() {
        // Predict beliefs after action
        let predicted = self.predict_beliefs(beliefs, action);
        let efe = expected_free_energy(&predicted, &self.generative_model);

        if efe < best_efe {
            best_efe = efe;
            best_action = action;
        }
    }

    best_action
}

fn predict_beliefs(&self, current: &BeliefState, action: Action) -> BeliefState {
    let mut predicted = current.clone();

    // Predict state change from action
    predicted.mean.angle += action.angle_delta();
    predicted.mean.angle = predicted.mean.angle.rem_euclid(2.0 * PI);

    // Predict position change
    let speed_estimate = 0.5;  // Expected speed
    predicted.mean.x += speed_estimate * predicted.mean.angle.cos();
    predicted.mean.y += speed_estimate * predicted.mean.angle.sin();

    // Uncertainty increases with prediction
    predicted.covariance.nutrient_var *= 1.1;
    predicted.covariance.angle_var *= 1.05;

    predicted
}
```

### Step 7: Update Parameters

**File: `src/simulation/params.rs` (additions)**

```rust
// === Inference Parameters ===
/// Learning rate for belief updates
pub const BELIEF_LEARNING_RATE: f64 = 0.1;
/// Maximum VFE for speed scaling
pub const MAX_VFE: f64 = 10.0;
/// Initial sensory precision
pub const INITIAL_SENSORY_PRECISION: f64 = 5.0;
/// Prior precision on nutrient (strength of preference)
pub const NUTRIENT_PRIOR_PRECISION: f64 = 2.0;
/// Minimum sensory precision
pub const MIN_SENSORY_PRECISION: f64 = 0.1;
/// Maximum sensory precision
pub const MAX_SENSORY_PRECISION: f64 = 20.0;
```

---

## Part 4: Testing Strategy

### 4.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vfe_minimized_at_target() {
        let model = GenerativeModel::new();
        let mut beliefs = BeliefState::new(50.0, 25.0, 0.0);

        // Beliefs at target should have low VFE
        beliefs.mean.nutrient = TARGET_CONCENTRATION;
        let vfe_at_target = variational_free_energy(
            (TARGET_CONCENTRATION, TARGET_CONCENTRATION),
            &beliefs,
            &model
        );

        // Beliefs far from target should have high VFE
        beliefs.mean.nutrient = 0.1;
        let vfe_far = variational_free_energy(
            (0.1, 0.1),
            &beliefs,
            &model
        );

        assert!(vfe_at_target < vfe_far);
    }

    #[test]
    fn test_gradient_descent_reduces_vfe() {
        let model = GenerativeModel::new();
        let mut beliefs = BeliefState::new(50.0, 25.0, 0.0);
        beliefs.mean.nutrient = 0.3;  // Start far from target

        let observations = (0.7, 0.7);  // Observe high nutrients

        let initial_vfe = variational_free_energy(observations, &beliefs, &model);

        // One gradient step
        let gradient = vfe_gradient(observations, &beliefs, &model);
        beliefs.update(&gradient, 0.1);

        let final_vfe = variational_free_energy(observations, &beliefs, &model);

        assert!(final_vfe < initial_vfe, "VFE should decrease after gradient step");
    }

    #[test]
    fn test_efe_prefers_preferred_states() {
        let model = GenerativeModel::new();

        let mut good_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        good_beliefs.mean.nutrient = TARGET_CONCENTRATION;

        let mut bad_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        bad_beliefs.mean.nutrient = 0.1;

        let efe_good = expected_free_energy(&good_beliefs, &model);
        let efe_bad = expected_free_energy(&bad_beliefs, &model);

        assert!(efe_good < efe_bad, "EFE should prefer states near prior (preference)");
    }

    #[test]
    fn test_precision_estimation() {
        let mut estimator = PrecisionEstimator::new(0.1, 20.0);

        // Low errors → high precision
        for _ in 0..100 {
            estimator.update(0.01, 0.01);
        }
        let high_precision = estimator.precision_left();

        // Reset and use high errors → low precision
        let mut estimator2 = PrecisionEstimator::new(0.1, 20.0);
        for _ in 0..100 {
            estimator2.update(0.5, 0.5);
        }
        let low_precision = estimator2.precision_left();

        assert!(high_precision > low_precision);
    }
}
```

### 4.2 Integration Tests

```rust
#[test]
fn test_agent_survives_with_active_inference() {
    let mut dish = PetriDish::new();
    let mut agent = Protozoa::new(50.0, 25.0);

    for _ in 0..1000 {
        agent.sense(&dish);
        agent.update_state(&dish);
        dish.update();
    }

    assert!(agent.energy > 0.0, "Agent should survive with Active Inference");
}

#[test]
fn test_beliefs_converge_to_observations() {
    let dish = PetriDish::new();
    let mut agent = Protozoa::new(50.0, 25.0);

    // Run for many steps in same location
    for _ in 0..100 {
        agent.sense(&dish);
        agent.update_state(&dish);
    }

    let mean_obs = (agent.val_l + agent.val_r) / 2.0;
    let belief_diff = (agent.beliefs.mean.nutrient - mean_obs).abs();

    assert!(belief_diff < 0.2, "Beliefs should converge to observations");
}
```

---

## Part 5: Migration Path

### Phase 1: Foundation (Non-Breaking)
1. Create `inference/` module with all new types
2. Add new fields to `Protozoa` struct
3. Keep old logic working alongside new

### Phase 2: Parallel Running
1. Compute VFE alongside old error
2. Log both for comparison
3. Verify behavior similarity

### Phase 3: Switch Over
1. Replace old error with VFE gradient
2. Replace old "precision" with true sensory precision
3. Replace action selection with EFE minimization

### Phase 4: Cleanup
1. Remove deprecated fields
2. Update documentation
3. Update tests

---

## Part 6: Documentation Updates

### AGENTS.md Changes

Replace:
```
The Agent operates by minimizing **Variational Free Energy ($F$)**.
We define $F$ based on the **Prediction Error** ($E$) relative to a **Target Set-Point** ($\rho$).
```

With:
```
The Agent operates by minimizing **Variational Free Energy ($F$)**.

$$F = \frac{1}{2}(o - g(\mu))^\top \Pi_o (o - g(\mu)) + \frac{1}{2}(\mu - \eta)^\top \Pi_\eta (\mu - \eta)$$

Where:
- $\mu$ = posterior beliefs over hidden states
- $g(\mu)$ = observation function (predicted sensory input)
- $\Pi_o$ = sensory precision (inverse observation variance)
- $\eta$ = prior means (homeostatic preferences)
- $\Pi_\eta$ = prior precision (strength of preferences)

Beliefs are updated via gradient descent on $F$:
$$\dot{\mu} = -\frac{\partial F}{\partial \mu}$$
```

---

## Verification Checklist

After implementation:

```bash
# All must pass
cargo fmt
cargo clippy -- -D warnings
cargo test

# Manual verification
cargo run --release
# Check:
# - Agent still navigates to nutrients
# - Energy stabilizes (0.3-0.9 range)
# - Beliefs converge to observations
# - VFE decreases over time
# - EFE correctly selects actions toward nutrients
```

---

## Summary

This plan transforms the agent from a reactive homeostatic regulator to a genuine continuous Active Inference system:

| Component | Before | After |
|-----------|--------|-------|
| State representation | Point estimates | Gaussian beliefs (μ, Σ) |
| Error signal | μ - target | VFE gradient |
| Precision | visits/(1+var) | 1/σ² (sensory noise) |
| Action selection | Heuristic | EFE minimization |
| Preferences | TARGET constant | Prior distribution |

The implementation maintains the biological simulation aesthetics while achieving theoretical rigor.
