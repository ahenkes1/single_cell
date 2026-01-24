# Comprehensive Codebase Review: Protozoa Active Inference Simulation

**Review Date:** 2026-01-24
**Reviewer:** Claude Opus 4.5
**Scope:** Theoretical Accuracy, Code Quality, Documentation
**Methodology:** Evidence-based critique grounded in academic literature (2019-2025 focus)

---

## Executive Summary

This review examines the Protozoa simulation's claims against canonical Active Inference literature. The findings reveal **significant theoretical gaps** between the documentation's claims and the actual implementation:

| Claim | Status | Severity |
|-------|--------|----------|
| "Minimizes Variational Free Energy" | **Incorrect** | Critical |
| "Precision-weighted error" | **Misinterpreted** | High |
| "MCTS with Expected Free Energy" | **Partially Correct** | Medium |
| "Welford's algorithm" | **Correct** | N/A |

**Recommendation:** The implementation is a well-engineered **homeostatic gradient-following agent with learned spatial priors**. To claim Active Inference, fundamental architectural changes are required. Alternatively, reframe the documentation to accurately describe the system.

---

## Part 1: Theoretical Foundations Analysis

### 1.1 Canonical Variational Free Energy (VFE)

#### Academic Definition

Per Friston et al. (2015) and the canonical formulation in Parr, Pezzulo & Friston (2022), Variational Free Energy is defined as:

```
F = E_q[ln q(s) - ln p(o,s)]
  = E_q[ln q(s)] - E_q[ln p(o|s)] - E_q[ln p(s)]
  = KL[q(s) || p(s)] - E_q[ln p(o|s)]
  = Complexity - Accuracy
```

Where:
- `q(s)` = approximate posterior over hidden states (agent's beliefs)
- `p(o,s)` = generative model (joint distribution over observations and states)
- `p(o|s)` = likelihood (how states generate observations)
- `p(s)` = prior over hidden states

**Key requirements for VFE minimization:**
1. An explicit **generative model** `p(o,s)` with defined likelihood and prior
2. An **approximate posterior** `q(s)` representing beliefs about hidden states
3. A **belief update mechanism** that minimizes the divergence `KL[q(s) || p(s|o)]`

#### Implementation Analysis

The current implementation ([agent.rs:139-148](protozoa_rust/src/simulation/agent.rs#L139-L148)):

```rust
// 2. Homeostatic error: difference from target (what agent WANTS)
let homeostatic_error = assert_finite(mean_sense - TARGET_CONCENTRATION, "error");

// 3. Get learned prior for precision weighting (confidence in this location)
let prior = self.spatial_priors.get_cell(self.x, self.y);
let precision = prior.precision().clamp(MIN_PRECISION, MAX_PRECISION);

// 4. Precision-weighted error: more confident = stronger response
let precision_weighted_error = assert_finite(homeostatic_error * precision, "prec_error");
```

**Derivation showing divergence from VFE:**

The implementation computes:
```
error = μ - ρ
```
Where:
- `μ = (s_L + s_R) / 2` (mean sensory input)
- `ρ = 0.8` (fixed target concentration)

This is a **setpoint tracking error**, not variational free energy. To see why:

1. **No generative model**: There is no `p(o|s)` likelihood distribution. The agent directly observes concentration values without modeling how hidden states generate observations.

2. **No hidden states**: The agent doesn't infer hidden states. It directly uses sensor readings.

3. **No approximate posterior**: The "spatial priors" are point estimates of mean concentration per grid cell, not probability distributions over hidden states.

4. **No KL divergence**: The error `μ - ρ` is not a KL divergence. In VFE:
   ```
   F ≈ (1/2σ²)(o - μ)² + (1/2)(μ - η)²/σ_η²
   ```
   Where `σ²` is sensory precision and `σ_η²` is prior precision. The implementation lacks this structure.

**Mathematical proof of the gap:**

Canonical VFE for Gaussian generative models (Laplace approximation):
```
F = (1/2) × (o - g(s))ᵀ × Σ_o⁻¹ × (o - g(s)) + (1/2) × (s - μ_s)ᵀ × Σ_s⁻¹ × (s - μ_s) + const
```

Implementation's "free energy":
```
error = observation_mean - target_setpoint
```

These are fundamentally different mathematical objects. The implementation has **no covariance matrices**, **no hidden state inference**, and **no model evidence**.

#### Verdict: CRITICAL MISALIGNMENT

The implementation does not minimize Variational Free Energy. It implements **reactive setpoint regulation**, which Pezzulo et al. (2015) explicitly distinguish from Active Inference:

> "The regulation of homeostatic states…has long been described in terms of control-theoretic and cybernetic mechanisms of error cancellation and feedback control."

This is what the implementation does. True Active Inference is "anticipatory prediction-based regulation rather than reactive error correction."

---

### 1.2 Precision Analysis

#### Academic Definition

In Active Inference, precision has a specific mathematical meaning (Pezzulo et al., 2024, "The Many Roles of Precision in Action"):

```
Precision ω = 1/σ² (inverse variance of the likelihood)
```

Precision-weighted prediction error:
```
ε_weighted = ω × (o - μ)
```

Where:
- `ω` = precision of the likelihood `p(o|s)`
- `o` = observation
- `μ` = predicted observation (from generative model)

**Role of precision:**
- High precision → trust observations more (sensory reliability)
- Low precision → trust prior beliefs more (uncertain sensing)

#### Implementation Analysis

The current implementation ([spatial_grid.rs:55-57](protozoa_rust/src/simulation/memory/spatial_grid.rs#L55-L57)):

```rust
pub fn precision(&self) -> f64 {
    f64::from(self.visits) / (1.0 + self.variance())
}
```

This computes:
```
precision = visits / (1 + variance)
```

**Critical issues:**

1. **Conflation of concepts**: This is **spatial familiarity**, not sensory precision. It measures "how often have I been here" and "how variable were observations here."

2. **Wrong denominator**: Academic precision = `1/variance`. The implementation uses `visits / (1 + variance)`, which scales with visit count.

3. **Not sensory precision**: True sensory precision would be about the reliability of the sensors themselves (noise in the chemoreceptors), not about spatial statistics.

**Mathematical comparison:**

| Concept | Academic Definition | Implementation |
|---------|---------------------|----------------|
| Precision | ω = 1/σ² (inverse variance of likelihood) | visits / (1 + variance) |
| Meaning | Reliability of sensory data | Familiarity with location |
| Scaling | Constant (sensor property) | Increases with visits |

#### Verdict: TERMINOLOGY MISMATCH

The implementation uses "precision" to mean something different from the academic definition. It should be called **spatial confidence** or **location familiarity**.

---

### 1.3 Expected Free Energy (EFE) Analysis

#### Academic Definition

Per Parr & Friston (2019) "Generalised free energy and active inference" (PMC6848054):

```
G^π_τ = -E_{Q̃(o_τ,s_τ|π)}[ln P(o_τ,s_τ) - ln Q(s_τ|π)]
```

Decomposition into Risk and Ambiguity:
```
G^π_τ = D_KL[Q(o_τ|π) || P(o_τ)]  +  E_{Q(s_τ|π)}[H[P(o_τ|s_τ)]]
        └──────── Risk ────────┘     └───────── Ambiguity ──────────┘
```

Where:
- **Risk**: KL divergence between expected observations and preferred observations `P(o_τ)` (encoded in the C vector)
- **Ambiguity**: Expected entropy of the likelihood (uncertainty about state-observation mapping)

Alternative decomposition with epistemic value:
```
G^π = E_Q[ln Q(s|π) - ln Q(s|o,π)]  +  D_KL[Q(o|π) || P(o)]
      └───── Epistemic Value ────┘     └───── Pragmatic Value ─────┘
```

**Critical requirements:**
1. A **preference distribution** `P(o)` or `C` vector over desired observations
2. A **generative model** with likelihood `P(o|s)`
3. **Posterior predictive distributions** `Q(o|π)` and `Q(s|π)`

#### Implementation Analysis

The current implementation ([mcts.rs:239-257](protozoa_rust/src/simulation/planning/mcts.rs#L239-L257)):

```rust
fn efe_components(&self, trajectory: &[AgentState], priors: &SpatialGrid<20, 10>) -> (f64, f64) {
    let mut pragmatic = 0.0;
    let mut epistemic = 0.0;

    for state in trajectory {
        let prior = priors.get_cell(state.x, state.y);
        pragmatic += prior.mean * state.energy;
        let precision = prior.precision().max(MIN_PRECISION);
        epistemic += 1.0 / precision;
    }

    (pragmatic, epistemic)
}
```

**Implementation's EFE formula:**
```
G(τ) = Σ(prior.mean × energy) + EXPLORATION_SCALE × Σ(1/precision)
```

**Derivation showing divergence from canonical EFE:**

1. **Pragmatic value mismatch:**
   - Academic: `D_KL[Q(o|π) || P(o)]` = divergence from preferred outcomes
   - Implementation: `prior.mean × energy` = product of expected nutrient and remaining energy

   The implementation has **no preference distribution**. The term `prior.mean × energy` is a heuristic reward, not a KL divergence. There is no log-probability, no distribution comparison.

2. **Epistemic value mismatch:**
   - Academic: `E_Q[ln Q(s|π) - ln Q(s|o,π)]` = information gain about hidden states
   - Implementation: `1/precision = (1 + variance) / visits`

   The implementation assumes inverse precision equals information gain. This is only approximately true if:
   - Precision represents posterior uncertainty about states
   - The formula accounts for expected posterior update

   Neither condition holds. The implementation's "precision" is spatial familiarity, and there's no expectation over future posterior updates.

3. **No preference encoding:**

   Active Inference requires explicit preferences via a C vector:
   ```python
   # pymdp example
   C = np.array([
       [0.0, 0.0],  # First observation modality preferences
       [1.0, 0.0]   # Second observation modality preferences
   ])
   ```

   The implementation has only `TARGET_CONCENTRATION = 0.8`, which is a setpoint, not a probability distribution over observations.

**Mathematical proof:**

Canonical EFE requires computing:
```
Risk = Σ_o Q(o|π) × [ln Q(o|π) - ln P(o)]
```

Implementation computes:
```
pragmatic = Σ_t μ_t × E_t
```

These are not mathematically equivalent. The implementation lacks:
- Log-probabilities
- Probability distributions Q(o|π)
- Preference distribution P(o)

#### Verdict: PSEUDOSCIENTIFIC ADAPTATION

The EFE formula borrows Active Inference terminology but lacks mathematical rigor. It's a custom reward function: `reward = nutrient × energy + exploration_bonus`.

---

### 1.4 MCTS Algorithm Analysis

#### Academic Definition

Classical Monte Carlo Tree Search (Kocsis & Szepesvári, 2006; Silver et al., 2016) consists of four phases:

1. **Selection**: Traverse tree using UCB1 (Upper Confidence Bound):
   ```
   UCB1(s,a) = Q(s,a) + c × √(ln N(s) / N(s,a))
   ```

2. **Expansion**: Add new node to tree

3. **Simulation**: Random rollout from new node

4. **Backpropagation**: Update Q-values up the tree

**Key characteristics:**
- Maintains persistent tree structure
- UCB balances exploitation vs exploration
- Q-values are updated through backpropagation
- Tree reused across planning cycles

#### Implementation Analysis

The current implementation ([mcts.rs:162-208](protozoa_rust/src/simulation/planning/mcts.rs#L162-L208)):

```rust
pub fn plan(&mut self, state: &AgentState, priors: &SpatialGrid<20, 10>) -> Action {
    // ...
    for action in Action::all() {
        for i in 0..MCTS_ROLLOUTS {
            let trajectory = self.rollout(*state, action, priors, &mut rng);
            let (pragmatic, epistemic) = self.efe_components(&trajectory, priors);
            // ... accumulate values
        }
        // ... average and compare
    }
}
```

And the rollout ([mcts.rs:214-237](protozoa_rust/src/simulation/planning/mcts.rs#L214-L237)):

```rust
fn rollout(&self, initial_state: AgentState, initial_action: Action, priors: &SpatialGrid<20, 10>, rng: &mut impl Rng) -> Vec<AgentState> {
    // Take initial action
    let mut current_state = initial_state.step(initial_action, priors);
    trajectory.push(current_state);

    // Continue with random actions
    for _ in 1..MCTS_DEPTH {
        let random_action = actions[rng.random_range(0..3)];
        current_state = current_state.step(random_action, priors);
        trajectory.push(current_state);
    }
    trajectory
}
```

**Critical issues:**

1. **No tree structure**: The implementation runs 50 independent rollouts, not a tree search. There's no tree node, no parent-child relationships.

2. **No UCB selection**: All actions evaluated equally with random rollouts, no exploration-exploitation balancing via UCB.

3. **No backpropagation**: Values are averaged, not backpropagated through a tree.

4. **No tree persistence**: Planning starts fresh each cycle. Classical MCTS reuses the tree.

5. **Random-only rollouts**: After the first action, all subsequent actions are random. True MCTS uses policy for rollouts (or UCB for selection).

**What this actually is:**

This is **random shooting** or **random sampling planning**:
```
For each candidate action a:
    For i in 1..N:
        Rollout: a → random → random → ... → random
        Evaluate trajectory
    Average values
Select best action
```

This is a valid planning method, but it's not Monte Carlo Tree Search.

#### Verdict: MISNAMED ALGORITHM

The implementation should be called **Random Rollout Planning** or **Monte Carlo Policy Evaluation**, not MCTS.

---

## Part 2: Code Quality Analysis

### 2.1 Welford's Algorithm Implementation

**File:** [spatial_grid.rs:59-87](protozoa_rust/src/simulation/memory/spatial_grid.rs#L59-L87)

**Academic reference:** Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products."

**Implementation:**
```rust
pub fn update(&mut self, observed: f64) {
    if !observed.is_finite() { return; }

    self.visits = self.visits.saturating_add(1);
    let delta = observed - self.mean;
    self.mean += delta / f64::from(self.visits);
    let delta2 = observed - self.mean;
    self.m2 += delta * delta2;

    // Guards...
}
```

**Verification against canonical algorithm:**

| Step | Canonical | Implementation | Match |
|------|-----------|----------------|-------|
| δ₁ = x - μₙ₋₁ | Yes | `delta = observed - self.mean` | ✓ |
| μₙ = μₙ₋₁ + δ₁/n | Yes | `self.mean += delta / f64::from(self.visits)` | ✓ |
| δ₂ = x - μₙ | Yes | `delta2 = observed - self.mean` | ✓ |
| M₂ += δ₁ × δ₂ | Yes | `self.m2 += delta * delta2` | ✓ |
| σ² = M₂/(n-1) | Yes | `self.m2 / (f64::from(self.visits) - 1.0)` | ✓ |

**Verdict: CORRECT IMPLEMENTATION**

The Welford's algorithm is correctly implemented with appropriate numerical guards.

**Minor concern:** M2 clamping to zero on negative values may mask floating-point errors. Consider logging when this occurs.

---

### 2.2 Numerical Stability Audit

**Files reviewed:**
- [agent.rs](protozoa_rust/src/simulation/agent.rs)
- [environment.rs](protozoa_rust/src/simulation/environment.rs)
- [mcts.rs](protozoa_rust/src/simulation/planning/mcts.rs)
- [spatial_grid.rs](protozoa_rust/src/simulation/memory/spatial_grid.rs)

| Check | Location | Status |
|-------|----------|--------|
| Division by zero guards | spatial_grid.rs:56 | ✓ `1.0 + variance()` prevents divide-by-zero |
| NaN propagation guards | agent.rs:33-36 | ✓ `assert_finite()` helper |
| Gaussian sigma guard | environment.rs (inferred) | ✓ `sigma_sq.max(f64::EPSILON)` |
| Angle normalization | agent.rs:220 | ✓ `rem_euclid(2π)` handles negatives |
| Energy clamping | agent.rs:259 | ✓ `clamp(0.0, 1.0)` |
| Position bounds | agent.rs:271-272 | ✓ `clamp(0.0, dish.width/height)` |
| M2 non-negative guard | spatial_grid.rs:79-81 | ✓ Clamps to 0.0 |
| Mean range guard | spatial_grid.rs:76 | ✓ `clamp(-0.5, 1.5)` |
| Non-finite observation filter | spatial_grid.rs:65-67 | ✓ Early return |
| Saturating arithmetic | agent.rs:199, spatial_grid.rs:69 | ✓ `saturating_add/sub` |

**Verdict: ROBUST NUMERICAL HANDLING**

The implementation has comprehensive guards against numerical issues. Well done.

---

### 2.3 Behavioral Bug Analysis

#### Bug 2.3.1: Bang-Bang Speed Control

**Location:** [agent.rs:223](protozoa_rust/src/simulation/agent.rs#L223)

```rust
self.speed = MAX_SPEED * homeostatic_error.abs();
```

**Issue:** When `mean_sense = TARGET_CONCENTRATION = 0.8`:
- `error = 0.8 - 0.8 = 0`
- `speed = MAX_SPEED × 0 = 0`

The agent **stops completely** when at target concentration.

**Consequences:**
1. Agent can get stuck on concentration plateaus < 0.8
2. No proactive exploration when satiated
3. Requires panic turns or exploration bonus to escape

**Severity:** Medium

**Recommendation:** Use proportional-integral (PI) control or add minimum speed:
```rust
self.speed = MAX_SPEED * (MIN_SPEED_FACTOR + homeostatic_error.abs());
```

---

#### Bug 2.3.2: Panic Threshold Sensitivity

**Location:** [agent.rs:167-169](protozoa_rust/src/simulation/agent.rs#L167-L169)

```rust
if self.temp_gradient < PANIC_THRESHOLD {  // PANIC_THRESHOLD = -0.01
    panic_turn = rng.random_range(-PANIC_TURN_RANGE..PANIC_TURN_RANGE);
}
```

**Issue:** A drop of 0.01 in one tick triggers panic.

Given:
- Environment has Brownian motion (sources move randomly)
- Gaussian blur can shift concentration
- A 0.01 drop is within natural noise

**Consequences:**
- False positive panic triggers
- Unnecessary random turns
- Jittery behavior

**Severity:** Low-Medium

**Recommendation:** Use smoothed temporal gradient over multiple ticks:
```rust
let smoothed_gradient = self.sensor_history.recent_gradient(5); // Last 5 ticks
if smoothed_gradient < PANIC_THRESHOLD {
    // panic
}
```

---

#### Bug 2.3.3: Random Panic Direction

**Location:** [agent.rs:168](protozoa_rust/src/simulation/agent.rs#L168)

```rust
panic_turn = rng.random_range(-PANIC_TURN_RANGE..PANIC_TURN_RANGE);
```

**Issue:** Panic turn is 50-50 left/right, ignoring gradient information.

**Consequences:**
- 50% chance of turning into worse conditions
- Not adaptive to spatial information

**Severity:** Low

**Recommendation:** Bias panic toward better gradient:
```rust
let bias = if self.val_l > self.val_r { 1.0 } else { -1.0 };
panic_turn = bias * rng.random_range(PANIC_TURN_RANGE/2..PANIC_TURN_RANGE);
```

---

#### Bug 2.3.4: Landmark Replacement Ignores Recency

**Location:** (episodic.rs - inferred from exploration)

**Issue:** Landmarks replaced by lowest `value = nutrient × reliability`. Recently discovered landmarks may have low reliability decay and get replaced.

**Severity:** Low

**Recommendation:** Add recency bias:
```rust
value = nutrient × reliability × recency_factor(last_visit_tick)
```

---

## Part 3: Documentation Accuracy

### 3.1 AGENTS.md Corrections

| Line | Current Claim | Correction |
|------|---------------|------------|
| 35 | "minimizes Variational Free Energy (F)" | "minimizes homeostatic error via gradient descent" |
| 36 | "Prediction Error (E) relative to Target Set-Point (ρ)" | Accurate as stated |
| 88-94 | "precision_weighted_error = base_error × precision" | "confidence_weighted_error = base_error × spatial_familiarity" |
| 107 | "precision = visits / (1 + variance)" | "spatial_confidence = visits / (1 + variance)" |
| 119-126 | "MCTS Expected Free Energy" | "Random rollout trajectory evaluation" |
| 122 | "G(τ) = pragmatic + EXPLORATION_SCALE × epistemic" | "utility(τ) = nutrient_sum × energy_sum + exploration_bonus" |

### 3.2 Terminology Mapping

| Current Term | Accurate Term |
|--------------|---------------|
| Variational Free Energy | Homeostatic error |
| Precision | Spatial confidence / Location familiarity |
| Expected Free Energy | Trajectory utility |
| MCTS | Random rollout planning |
| Prior | Spatial mean estimate |
| Generative model | Spatial prior grid |

---

## Part 4: Proposed Code Changes for Genuine Active Inference

If the goal is to implement genuine Active Inference, the following architectural changes are required:

### 4.1 Generative Model Specification

**Required components (per pymdp framework):**

```rust
/// A matrix: Likelihood P(o|s)
/// Maps hidden states to observations
struct Likelihood {
    /// P(concentration | nutrient_state, position_state)
    /// Dimensions: [n_observations, n_nutrient_states, n_position_states]
    a_matrix: Array3<f64>,
}

/// B matrix: Transition model P(s'|s,a)
/// How actions change hidden states
struct Transitions {
    /// P(next_state | current_state, action)
    /// One matrix per action
    b_matrices: Vec<Array2<f64>>,
}

/// C vector: Preferences over observations
/// What observations the agent prefers
struct Preferences {
    /// Log-preferences over observation levels
    /// E.g., [0.0, 0.5, 1.0, 0.8, 0.3] for concentrations [0, 0.25, 0.5, 0.75, 1.0]
    c_vector: Array1<f64>,
}

/// D vector: Prior over initial states
struct InitialPrior {
    d_vector: Array1<f64>,
}
```

### 4.2 Posterior Inference Implementation

**Required:** Approximate posterior `Q(s)` over hidden states

```rust
/// Variational posterior over hidden states
struct Beliefs {
    /// Q(s) - probability distribution over discrete hidden states
    posterior: Array1<f64>,
}

impl Beliefs {
    /// Update beliefs given observation using variational inference
    /// q(s) = softmax(ln A[o,:] + ln B × q_prev)
    fn update(&mut self, observation: usize, likelihood: &Likelihood, transition: &Transitions, action: Action) {
        let log_likelihood = likelihood.a_matrix.slice(s![observation, ..]).mapv(f64::ln);
        let log_prior = transition.b_matrices[action as usize].dot(&self.posterior).mapv(f64::ln);
        self.posterior = softmax(&(log_likelihood + log_prior));
    }
}
```

### 4.3 Proper EFE Implementation

```rust
/// Compute Expected Free Energy for a policy
fn expected_free_energy(
    policy: &[Action],
    beliefs: &Beliefs,
    likelihood: &Likelihood,
    transition: &Transitions,
    preferences: &Preferences,
) -> f64 {
    let mut efe = 0.0;
    let mut current_beliefs = beliefs.posterior.clone();

    for &action in policy {
        // Predict next state: Q(s') = B × Q(s)
        let predicted_state = transition.b_matrices[action as usize].dot(&current_beliefs);

        // Predict observation: Q(o) = A × Q(s')
        let predicted_obs = likelihood.a_matrix.t().dot(&predicted_state);

        // Risk: KL[Q(o) || P(o)]
        let risk = kl_divergence(&predicted_obs, &softmax(&preferences.c_vector));

        // Ambiguity: E_Q(s')[H[P(o|s')]]
        let ambiguity = expected_entropy(&likelihood.a_matrix, &predicted_state);

        efe += risk + ambiguity;
        current_beliefs = predicted_state;
    }

    efe
}
```

### 4.4 Proper Precision Implementation

```rust
/// Sensory precision: inverse variance of observation noise
struct SensoryPrecision {
    /// ω = 1/σ² for each observation modality
    precision: f64,
}

/// Precision-weighted prediction error
fn prediction_error(observation: f64, prediction: f64, precision: f64) -> f64 {
    precision * (observation - prediction)
}
```

### 4.5 MCTS Upgrade Path

For genuine MCTS:

```rust
struct MCTSNode {
    state: AgentState,
    action: Option<Action>,
    parent: Option<usize>,
    children: Vec<usize>,
    visits: u32,
    value: f64,
}

struct MCTSTree {
    nodes: Vec<MCTSNode>,
    root: usize,
}

impl MCTSTree {
    fn select(&self, node: usize) -> usize {
        // UCB1 selection
        let n = &self.nodes[node];
        if n.children.is_empty() {
            return node;
        }

        let best_child = n.children.iter()
            .max_by(|&&a, &&b| {
                let ucb_a = self.ucb1(a, n.visits);
                let ucb_b = self.ucb1(b, n.visits);
                ucb_a.partial_cmp(&ucb_b).unwrap()
            })
            .copied()
            .unwrap();

        self.select(best_child)
    }

    fn ucb1(&self, node: usize, parent_visits: u32) -> f64 {
        let n = &self.nodes[node];
        if n.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = n.value / n.visits as f64;
        let exploration = C * ((parent_visits as f64).ln() / n.visits as f64).sqrt();
        exploitation + exploration
    }

    fn backpropagate(&mut self, node: usize, value: f64) {
        let mut current = Some(node);
        while let Some(idx) = current {
            self.nodes[idx].visits += 1;
            self.nodes[idx].value += value;
            current = self.nodes[idx].parent;
        }
    }
}
```

---

## Part 5: Summary and Recommendations

### 5.1 Findings Summary

| Component | Academic Standard | Implementation | Gap |
|-----------|-------------------|----------------|-----|
| Free Energy | VFE with generative model | Setpoint error | **Critical** |
| Precision | Inverse sensory variance | Spatial familiarity | **High** |
| EFE | KL divergence + entropy | Heuristic reward | **High** |
| MCTS | Tree search with UCB | Random rollouts | **Medium** |
| Welford | Online variance | Correctly implemented | **None** |

### 5.2 Recommended Path Forward

**Option A: Genuine Active Inference (Major Effort)**

1. Implement discrete state space model with A, B, C, D matrices
2. Add variational belief update
3. Implement proper EFE with Risk and Ambiguity
4. Either upgrade to real MCTS or use policy enumeration (simpler)
5. Add proper precision modeling for sensory noise

**Option B: Honest Reframing (Documentation Only)**

Rename/reframe:
- "Active Inference" → "Biologically-Inspired Homeostatic Agent"
- "Variational Free Energy" → "Homeostatic Error"
- "Precision" → "Spatial Confidence"
- "Expected Free Energy" → "Trajectory Utility"
- "MCTS" → "Random Rollout Planning"

Acknowledge inspiration from Active Inference without claiming full implementation.

### 5.3 Verification Plan

After any changes:

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
cargo run --release  # Manual verification
```

Behavioral benchmarks:
- Agent survives for 10,000+ ticks
- Landmarks discovered within 500 ticks
- Energy remains stable (0.3-0.9 typical range)
- Mode transitions observed (Exploring ↔ Exploiting)

---

## References

1. Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

2. Friston, K., et al. (2015). "Active inference and free energy." *Neuroscience & Biobehavioral Reviews*, 47, 486-505.

3. Parr, T. & Friston, K. (2019). "Generalised free energy and active inference." *Biological Cybernetics*, 113(5-6), 495-513.

4. Pezzulo, G., et al. (2015). "Active Inference, homeostatic regulation and adaptive behavioural control." *Progress in Neurobiology*, 134, 17-35.

5. Pezzulo, G., et al. (2024). "The Many Roles of Precision in Action." *Entropy*, 26(9), 790.

6. Da Costa, L., et al. (2022). "pymdp: A Python library for active inference in discrete state spaces." *Journal of Open Source Software*.

7. Kocsis, L., & Szepesvári, C. (2006). "Bandit based Monte-Carlo planning." *ICML*, 81-88.

8. Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search." *Nature*, 529(7587), 484-489.

9. Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 419-420.

---

*Review completed 2026-01-24*
