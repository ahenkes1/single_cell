//! Agent implementation using Continuous Active Inference.
//!
//! The agent minimizes Variational Free Energy through gradient descent on beliefs,
//! and selects actions by minimizing Expected Free Energy over predicted futures.

use crate::simulation::environment::PetriDish;
use crate::simulation::inference::{
    BeliefState, GenerativeModel, PrecisionEstimator, expected_free_energy, prediction_errors,
    variational_free_energy, vfe_gradient,
};
use crate::simulation::memory::{EpisodicMemory, SensorHistory, SensorSnapshot, SpatialGrid};
use crate::simulation::params::{
    BASE_METABOLIC_COST, BELIEF_LEARNING_RATE, DISH_HEIGHT, DISH_WIDTH, EXHAUSTION_SPEED_FACTOR,
    EXHAUSTION_THRESHOLD, EXPLORATION_SCALE, INTAKE_RATE, LANDMARK_ATTRACTION_SCALE,
    LANDMARK_THRESHOLD, LANDMARK_VISIT_RADIUS, MAX_PRECISION, MAX_SPEED, MAX_VFE,
    MCTS_REPLAN_INTERVAL, MCTS_URGENT_ENERGY, MIN_PRECISION, NOISE_SCALE, PANIC_THRESHOLD,
    PANIC_TURN_RANGE, SENSOR_ANGLE, SENSOR_DIST, SPEED_METABOLIC_COST, TARGET_CONCENTRATION,
    UNCERTAINTY_GROWTH, UNCERTAINTY_REDUCTION,
};
use crate::simulation::planning::{Action, AgentState, MCTSPlanner};
use rand::Rng;
use std::f64::consts::PI;

/// Behavioral mode of the agent, derived from internal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Used by tests and future UI components
pub enum AgentMode {
    /// Normal gradient following with exploration bonus
    Exploring,
    /// In high-nutrient area with high precision
    Exploiting,
    /// Temporal gradient below panic threshold
    Panicking,
    /// Energy below exhaustion threshold
    Exhausted,
    /// Actively navigating toward a landmark
    GoalNav,
}

/// Validates that a value is finite (not NaN or infinite).
/// Returns a safe fallback (0.0) in release mode if the value is non-finite.
#[inline]
fn assert_finite(value: f64, context: &str) -> f64 {
    debug_assert!(value.is_finite(), "Non-finite value in {context}: {value}");
    if value.is_finite() { value } else { 0.0 }
}

/// Represents the single-cell organism (Agent) using Continuous Active Inference.
///
/// The agent minimizes Variational Free Energy by updating Gaussian beliefs
/// about hidden states, and selects actions to minimize Expected Free Energy.
///
/// # Active Inference Components
/// - **Beliefs**: Gaussian posterior q(s) = N(μ, Σ) over hidden states
/// - **Generative Model**: Likelihood p(o|s) and prior p(s) with preferences
/// - **Precision**: Learned sensory precision (inverse observation variance)
///
/// # Cognitive Architecture
/// - **Short-term memory**: Ring buffer of recent sensor experiences
/// - **Long-term memory**: Spatial grid of learned nutrient expectations
/// - **Episodic memory**: Landmarks for goal-directed navigation
#[derive(Debug, Clone)]
pub struct Protozoa {
    // === Position and Movement ===
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub speed: f64,

    // === Internal State ===
    pub energy: f64,
    pub last_mean_sense: f64,
    pub temp_gradient: f64,
    pub val_l: f64,
    pub val_r: f64,

    // === Active Inference Components ===
    /// Gaussian beliefs about hidden states: q(s) = N(μ, Σ)
    pub beliefs: BeliefState,
    /// The agent's generative model: p(o,s) = p(o|s)p(s)
    pub generative_model: GenerativeModel,
    /// Online precision estimator from prediction errors
    pub precision_estimator: PrecisionEstimator,
    /// Current Variational Free Energy (for monitoring/visualization)
    pub current_vfe: f64,

    // === Memory Systems ===
    /// Spatial prior grid: learned expectations about nutrient concentration
    pub spatial_priors: SpatialGrid<20, 10>,
    /// Short-term memory: recent sensor experiences
    pub sensor_history: SensorHistory,
    /// Episodic memory: remembered high-nutrient landmarks
    pub episodic_memory: EpisodicMemory,
    /// Current simulation tick
    pub tick_count: u64,

    // === Planning System ===
    /// MCTS planner for trajectory optimization
    pub planner: MCTSPlanner,
    /// Tick when last planning occurred
    pub last_plan_tick: u64,
    /// Best action from last planning cycle
    pub planned_action: Action,
}

impl Protozoa {
    /// Creates a new Protozoa agent at the given position.
    ///
    /// Initializes Active Inference components with neutral priors.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        let mut rng = rand::rng();
        let initial_angle = rng.random_range(0.0..2.0 * PI);

        Self {
            x,
            y,
            angle: initial_angle,
            speed: 0.0,
            energy: 1.0,
            last_mean_sense: 0.0,
            temp_gradient: 0.0,
            val_l: 0.0,
            val_r: 0.0,
            // Active Inference components
            beliefs: BeliefState::new(x, y, initial_angle),
            generative_model: GenerativeModel::new(),
            precision_estimator: PrecisionEstimator::new(),
            current_vfe: 0.0,
            // Memory systems
            spatial_priors: SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT),
            sensor_history: SensorHistory::new(),
            episodic_memory: EpisodicMemory::new(),
            tick_count: 0,
            // Planning
            planner: MCTSPlanner::new(),
            last_plan_tick: 0,
            planned_action: Action::Straight,
        }
    }

    /// Updates the agent's sensory inputs based on the current environment.
    ///
    /// Detects concentration at two points (left and right sensors).
    pub fn sense(&mut self, dish: &PetriDish) {
        // Left Sensor
        let theta_l = self.angle + SENSOR_ANGLE;
        let x_l = self.x + SENSOR_DIST * theta_l.cos();
        let y_l = self.y + SENSOR_DIST * theta_l.sin();
        self.val_l = dish.get_concentration(x_l, y_l);

        // Right Sensor
        let theta_r = self.angle - SENSOR_ANGLE;
        let x_r = self.x + SENSOR_DIST * theta_r.cos();
        let y_r = self.y + SENSOR_DIST * theta_r.sin();
        self.val_r = dish.get_concentration(x_r, y_r);
    }

    /// Updates the agent's internal state using Active Inference.
    ///
    /// # Active Inference Loop
    /// 1. **Infer**: Update beliefs via gradient descent on Variational Free Energy
    /// 2. **Learn**: Update precision estimates from prediction errors
    /// 3. **Plan**: Select action minimizing Expected Free Energy
    /// 4. **Act**: Execute action and update position
    #[allow(clippy::too_many_lines)]
    pub fn update_state(&mut self, dish: &PetriDish) {
        let mut rng = rand::rng();

        // Get observations
        let observations = (self.val_l, self.val_r);
        let mean_sense = assert_finite(f64::midpoint(self.val_l, self.val_r), "mean_sense");

        // === PHASE 1: INFERENCE (Minimize VFE) ===

        // Synchronize position beliefs with actual position (proprioception)
        self.beliefs.sync_position(self.x, self.y, self.angle);

        // Compute VFE gradient and update beliefs
        let gradient = vfe_gradient(observations, &self.beliefs, &self.generative_model);
        self.beliefs.update(&gradient, BELIEF_LEARNING_RATE);

        // Reduce uncertainty after incorporating observation
        self.beliefs.decrease_uncertainty(UNCERTAINTY_REDUCTION);

        // Compute and store current VFE for monitoring
        self.current_vfe =
            variational_free_energy(observations, &self.beliefs, &self.generative_model);

        // === PHASE 2: PRECISION LEARNING ===

        // Update precision estimates from prediction errors
        let (err_l, err_r) = prediction_errors(observations, &self.beliefs, &self.generative_model);
        self.precision_estimator.update(err_l, err_r);

        // Update generative model with learned precisions
        self.generative_model.update_sensory_precision(
            self.precision_estimator.precision_left(),
            self.precision_estimator.precision_right(),
        );

        // === PHASE 3: PLANNING (Minimize EFE) ===

        // Compute temporal gradient (for panic detection)
        self.temp_gradient = mean_sense - self.last_mean_sense;
        self.last_mean_sense = mean_sense;

        // Select action using EFE-based planning
        let efe_action = self.select_action_efe();

        // MCTS Planning: replan periodically or when urgent
        let should_replan = self.tick_count == 0
            || self.tick_count.saturating_sub(self.last_plan_tick) >= MCTS_REPLAN_INTERVAL
            || self.energy < MCTS_URGENT_ENERGY;

        if should_replan {
            let state = AgentState::new(self.x, self.y, self.angle, self.speed, self.energy);
            self.planned_action = self.planner.plan(&state, &self.spatial_priors);
            self.last_plan_tick = self.tick_count;
        }

        // === PHASE 4: ACTION EXECUTION ===

        // Blend EFE-selected action with MCTS and reactive components
        let efe_delta = efe_action.angle_delta();
        let mcts_delta = self.planned_action.angle_delta();

        // Reactive gradient following (legacy, weighted lower now)
        let prior = self.spatial_priors.get_cell(self.x, self.y);
        let spatial_precision = prior.precision().clamp(MIN_PRECISION, MAX_PRECISION);
        let homeostatic_error = mean_sense - TARGET_CONCENTRATION;
        let gradient = self.val_l - self.val_r;
        let reactive_d_theta = -0.1 * homeostatic_error * spatial_precision * gradient;

        // Exploration bonus for uncertain regions
        let exploration_bonus = EXPLORATION_SCALE / spatial_precision;
        let explore_direction = rng.random_range(-1.0..1.0) * exploration_bonus;

        // Noise proportional to VFE (high uncertainty = more exploration)
        let noise = rng.random_range(-NOISE_SCALE..NOISE_SCALE)
            * (self.current_vfe / MAX_VFE).clamp(0.0, 1.0);

        // Panic Turn (if conditions worsening rapidly)
        let mut panic_turn = 0.0;
        if self.temp_gradient < PANIC_THRESHOLD {
            panic_turn = rng.random_range(-PANIC_TURN_RANGE..PANIC_TURN_RANGE);
        }

        // Goal-directed navigation toward remembered landmarks when energy is low
        let goal_attraction = if self.energy < MCTS_URGENT_ENERGY {
            if let Some(landmark) =
                self.episodic_memory
                    .best_distant_landmark(self.x, self.y, LANDMARK_VISIT_RADIUS)
            {
                let dx = landmark.x - self.x;
                let dy = landmark.y - self.y;
                let target_angle = dy.atan2(dx);
                let angle_diff = (target_angle - self.angle).rem_euclid(2.0 * PI);
                let normalized_diff = if angle_diff > PI {
                    angle_diff - 2.0 * PI
                } else {
                    angle_diff
                };
                LANDMARK_ATTRACTION_SCALE * normalized_diff * landmark.reliability
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Blend all heading contributions
        // EFE action gets highest weight as it's the principled Active Inference component
        let d_theta = assert_finite(
            0.4 * efe_delta
                + 0.2 * mcts_delta
                + 0.2 * reactive_d_theta
                + explore_direction
                + noise
                + panic_turn
                + goal_attraction,
            "d_theta",
        );

        self.angle += d_theta;
        self.angle = self.angle.rem_euclid(2.0 * PI);

        // Speed Update: Move to reduce VFE (proportional to free energy)
        // Higher VFE = more "anxious" = move faster to find preferred states
        self.speed = MAX_SPEED * (self.current_vfe / MAX_VFE).clamp(0.1, 1.0);

        // === PHASE 5: MEMORY & LEARNING ===

        // Update spatial prior with observation (world model learning)
        self.spatial_priors.update(self.x, self.y, mean_sense);

        // Record experience in short-term memory
        self.sensor_history.push(SensorSnapshot {
            val_l: self.val_l,
            val_r: self.val_r,
            x: self.x,
            y: self.y,
            energy: self.energy,
            tick: self.tick_count,
        });
        self.tick_count += 1;

        // Episodic memory: landmark detection and maintenance
        self.episodic_memory.decay_all();

        if mean_sense > LANDMARK_THRESHOLD {
            self.episodic_memory
                .maybe_store(self.x, self.y, mean_sense, self.tick_count);
        }

        self.episodic_memory
            .update_on_visit(self.x, self.y, mean_sense, self.tick_count);

        // === PHASE 6: METABOLISM ===

        let metabolic_cost =
            BASE_METABOLIC_COST + (SPEED_METABOLIC_COST * (self.speed / MAX_SPEED));
        let intake = INTAKE_RATE * mean_sense;

        self.energy = assert_finite(self.energy - metabolic_cost + intake, "energy");
        self.energy = self.energy.clamp(0.0, 1.0);

        // Exhaustion check
        if self.energy <= EXHAUSTION_THRESHOLD {
            self.speed *= EXHAUSTION_SPEED_FACTOR;
        }

        // === PHASE 7: POSITION UPDATE ===

        self.x += self.speed * self.angle.cos();
        self.y += self.speed * self.angle.sin();

        // Boundary Check
        self.x = self.x.clamp(0.0, dish.width);
        self.y = self.y.clamp(0.0, dish.height);
    }

    /// Select action by minimizing Expected Free Energy.
    ///
    /// Evaluates each candidate action and returns the one with lowest EFE.
    fn select_action_efe(&self) -> Action {
        let mut best_action = Action::Straight;
        let mut best_efe = f64::INFINITY;

        for action in Action::all() {
            // Predict beliefs after taking this action
            let predicted = self.predict_beliefs_after_action(action);
            let efe = expected_free_energy(&predicted, &self.generative_model);

            if efe < best_efe {
                best_efe = efe;
                best_action = action;
            }
        }

        best_action
    }

    /// Predict beliefs after taking an action.
    ///
    /// Uses the generative model's transition dynamics to predict future beliefs.
    fn predict_beliefs_after_action(&self, action: Action) -> BeliefState {
        let mut predicted = self.beliefs.clone();

        // Predict state change from action
        predicted.mean.angle += action.angle_delta();
        predicted.mean.angle = predicted.mean.angle.rem_euclid(2.0 * PI);

        // Predict position change (assuming current speed)
        let speed_estimate = self.speed.max(0.5); // Minimum expected speed
        predicted.mean.x += speed_estimate * predicted.mean.angle.cos();
        predicted.mean.y += speed_estimate * predicted.mean.angle.sin();

        // Clamp predicted position to dish
        predicted.mean.x = predicted.mean.x.clamp(0.0, DISH_WIDTH);
        predicted.mean.y = predicted.mean.y.clamp(0.0, DISH_HEIGHT);

        // Predict nutrient belief from spatial priors
        let expected_nutrient = self
            .spatial_priors
            .get_cell(predicted.mean.x, predicted.mean.y);
        // Blend current belief with expected from spatial prior
        predicted.mean.nutrient =
            0.5 * predicted.mean.nutrient + 0.5 * expected_nutrient.mean.clamp(0.0, 1.0);

        // Uncertainty increases with prediction (future is uncertain)
        predicted.increase_uncertainty(UNCERTAINTY_GROWTH);

        predicted
    }

    /// Returns the current behavioral mode derived from internal state.
    #[must_use]
    #[allow(dead_code)] // Used by tests and future UI components
    pub fn current_mode(&self, _dish: &PetriDish) -> AgentMode {
        // Check exhausted first (most critical)
        if self.energy <= EXHAUSTION_THRESHOLD {
            return AgentMode::Exhausted;
        }

        // Check if panicking (temporal gradient)
        if self.temp_gradient < PANIC_THRESHOLD {
            return AgentMode::Panicking;
        }

        // Check goal navigation (low energy, has landmark)
        if self.energy < MCTS_URGENT_ENERGY
            && self
                .episodic_memory
                .best_distant_landmark(self.x, self.y, LANDMARK_VISIT_RADIUS)
                .is_some()
        {
            return AgentMode::GoalNav;
        }

        // Check exploiting (high precision at current location and low VFE)
        let mean_sense = f64::midpoint(self.val_l, self.val_r);
        let precision = self.spatial_priors.get_cell(self.x, self.y).precision();
        if precision > 5.0 && mean_sense > 0.6 && self.current_vfe < 1.0 {
            return AgentMode::Exploiting;
        }

        AgentMode::Exploring
    }

    /// Returns ticks until next MCTS replan.
    #[must_use]
    #[allow(dead_code)] // Used by tests and future UI components
    pub fn ticks_until_replan(&self) -> u64 {
        let elapsed = self.tick_count.saturating_sub(self.last_plan_tick);
        MCTS_REPLAN_INTERVAL.saturating_sub(elapsed)
    }

    /// Returns the current Variational Free Energy.
    #[must_use]
    #[allow(dead_code)]
    pub fn free_energy(&self) -> f64 {
        self.current_vfe
    }

    /// Returns the agent's current beliefs about nutrient concentration.
    #[must_use]
    #[allow(dead_code)]
    pub fn believed_nutrient(&self) -> f64 {
        self.beliefs.mean.nutrient
    }

    /// Returns the agent's belief uncertainty (total variance).
    #[must_use]
    #[allow(dead_code)]
    pub fn belief_uncertainty(&self) -> f64 {
        self.beliefs.total_uncertainty()
    }
}
