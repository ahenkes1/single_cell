use crate::simulation::environment::PetriDish;
use crate::simulation::memory::{EpisodicMemory, SensorHistory, SensorSnapshot, SpatialGrid};
use crate::simulation::params::{
    BASE_METABOLIC_COST, DISH_HEIGHT, DISH_WIDTH, EXHAUSTION_SPEED_FACTOR, EXHAUSTION_THRESHOLD,
    EXPLORATION_SCALE, INTAKE_RATE, LANDMARK_ATTRACTION_SCALE, LANDMARK_THRESHOLD,
    LANDMARK_VISIT_RADIUS, LEARNING_RATE, MAX_PRECISION, MAX_SPEED, MCTS_REPLAN_INTERVAL,
    MCTS_URGENT_ENERGY, MIN_PRECISION, NOISE_SCALE, PANIC_THRESHOLD, PANIC_TURN_RANGE,
    PLANNING_WEIGHT, SENSOR_ANGLE, SENSOR_DIST, SPEED_METABOLIC_COST, TARGET_CONCENTRATION,
};
use crate::simulation::planning::{Action, AgentState, MCTSPlanner};
use rand::Rng;
use std::f64::consts::PI;

/// Validates that a value is finite (not NaN or infinite).
/// Returns a safe fallback (0.0) in release mode if the value is non-finite.
#[inline]
fn assert_finite(value: f64, context: &str) -> f64 {
    debug_assert!(value.is_finite(), "Non-finite value in {context}: {value}");
    if value.is_finite() { value } else { 0.0 }
}

/// Represents the single-cell organism (Agent) using Active Inference.
///
/// The agent minimizes Variational Free Energy by minimizing the difference (error)
/// between its sensed nutrient concentration and its learned spatial priors.
///
/// # Cognitive Architecture
/// - **Short-term memory**: Ring buffer of recent sensor experiences
/// - **Long-term memory**: Spatial grid of learned nutrient expectations
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
    pub val_l: f64,
    pub val_r: f64,

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
    /// Initializes memory systems with neutral priors (no prior knowledge).
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        let mut rng = rand::rng();
        Self {
            x,
            y,
            angle: rng.random_range(0.0..2.0 * PI),
            speed: 0.0,
            energy: 1.0,
            last_mean_sense: 0.0,
            val_l: 0.0,
            val_r: 0.0,
            spatial_priors: SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT),
            sensor_history: SensorHistory::new(),
            episodic_memory: EpisodicMemory::new(),
            tick_count: 0,
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

    /// Updates the agent's internal state, heading, speed, and position.
    ///
    /// This implements the Active Inference loop with learned priors:
    /// 1. Calculates Prediction Error using learned spatial priors.
    /// 2. Calculates Spatial and Temporal Gradients.
    /// 3. Updates Heading with precision-weighted error and exploration bonus.
    /// 4. Updates Speed based on "anxiety" (Magnitude of Error).
    /// 5. Updates spatial priors with observation (Hebbian learning).
    /// 6. Applies metabolic costs and intake.
    pub fn update_state(&mut self, dish: &PetriDish) {
        let mut rng = rand::rng();

        // 1. Sensation
        let mean_sense = assert_finite(f64::midpoint(self.val_l, self.val_r), "mean_sense");

        // 2. Homeostatic error: difference from target (what agent WANTS)
        // The target remains fixed - this is the agent's goal, not its prediction
        let homeostatic_error = assert_finite(mean_sense - TARGET_CONCENTRATION, "error");

        // 3. Get learned prior for precision weighting (confidence in this location)
        let prior = self.spatial_priors.get_cell(self.x, self.y);
        let precision = prior.precision().clamp(MIN_PRECISION, MAX_PRECISION);

        // 4. Precision-weighted error: more confident = stronger response
        let precision_weighted_error = assert_finite(homeostatic_error * precision, "prec_error");

        // 4. Spatial Gradient (G = sL - sR)
        let gradient = assert_finite(self.val_l - self.val_r, "gradient");

        // 5. Temporal Gradient
        let temp_gradient = mean_sense - self.last_mean_sense;
        self.last_mean_sense = mean_sense;

        // 6. Exploration bonus for uncertain regions (inverse precision)
        let exploration_bonus = EXPLORATION_SCALE / precision;
        let explore_direction = rng.random_range(-1.0..1.0) * exploration_bonus;

        // 7. Dynamics
        // Noise proportional to error
        let noise = rng.random_range(-NOISE_SCALE..NOISE_SCALE) * homeostatic_error.abs();

        // Panic Turn
        let mut panic_turn = 0.0;
        if temp_gradient < PANIC_THRESHOLD {
            panic_turn = rng.random_range(-PANIC_TURN_RANGE..PANIC_TURN_RANGE);
        }

        // 8. Goal-directed navigation toward remembered landmarks when energy is low
        let goal_attraction = if self.energy < MCTS_URGENT_ENERGY {
            // Find best distant landmark (not the one we're currently at)
            if let Some(landmark) =
                self.episodic_memory
                    .best_distant_landmark(self.x, self.y, LANDMARK_VISIT_RADIUS)
            {
                let dx = landmark.x - self.x;
                let dy = landmark.y - self.y;
                let target_angle = dy.atan2(dx);
                // Calculate shortest angular distance to target
                let angle_diff = (target_angle - self.angle).rem_euclid(2.0 * PI);
                let normalized_diff = if angle_diff > PI {
                    angle_diff - 2.0 * PI
                } else {
                    angle_diff
                };
                // Scale by landmark reliability and attraction constant
                LANDMARK_ATTRACTION_SCALE * normalized_diff * landmark.reliability
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 9. MCTS Planning: replan periodically or when urgent
        let should_replan = self.tick_count == 0
            || self.tick_count.saturating_sub(self.last_plan_tick) >= MCTS_REPLAN_INTERVAL
            || self.energy < MCTS_URGENT_ENERGY;

        if should_replan {
            let state = AgentState::new(self.x, self.y, self.angle, self.speed, self.energy);
            self.planned_action = self.planner.plan(&state, &self.spatial_priors);
            self.last_plan_tick = self.tick_count;
        }

        // Get heading delta from planned action
        let planned_delta = self.planned_action.angle_delta();

        // 10. Heading Update: blend reactive control with planned action
        let reactive_d_theta = -LEARNING_RATE * precision_weighted_error * gradient;
        let blended_d_theta =
            (1.0 - PLANNING_WEIGHT) * reactive_d_theta + PLANNING_WEIGHT * planned_delta;
        let d_theta = assert_finite(
            blended_d_theta + explore_direction + noise + panic_turn + goal_attraction,
            "d_theta",
        );
        self.angle += d_theta;
        self.angle = self.angle.rem_euclid(2.0 * PI);

        // 11. Speed Update (based on homeostatic error)
        self.speed = MAX_SPEED * homeostatic_error.abs();

        // 12. Update spatial prior with observation (Hebbian learning)
        self.spatial_priors.update(self.x, self.y, mean_sense);

        // 13. Record experience in short-term memory
        self.sensor_history.push(SensorSnapshot {
            val_l: self.val_l,
            val_r: self.val_r,
            x: self.x,
            y: self.y,
            energy: self.energy,
            tick: self.tick_count,
        });
        self.tick_count += 1;

        // 14. Episodic memory: landmark detection and maintenance
        // Decay reliability of all remembered landmarks
        self.episodic_memory.decay_all();

        // Store new landmark if high-nutrient area discovered
        if mean_sense > LANDMARK_THRESHOLD {
            self.episodic_memory
                .maybe_store(self.x, self.y, mean_sense, self.tick_count);
        }

        // Update landmark reliability if revisiting a known location
        self.episodic_memory
            .update_on_visit(self.x, self.y, mean_sense, self.tick_count);

        // 15. Metabolism
        let metabolic_cost =
            BASE_METABOLIC_COST + (SPEED_METABOLIC_COST * (self.speed / MAX_SPEED));
        let intake = INTAKE_RATE * mean_sense;

        self.energy = assert_finite(self.energy - metabolic_cost + intake, "energy");
        self.energy = self.energy.clamp(0.0, 1.0);

        // 16. Exhaustion check
        if self.energy <= EXHAUSTION_THRESHOLD {
            self.speed *= EXHAUSTION_SPEED_FACTOR;
        }

        // 17. Position Update
        self.x += self.speed * self.angle.cos();
        self.y += self.speed * self.angle.sin();

        // 18. Boundary Check
        self.x = self.x.clamp(0.0, dish.width);
        self.y = self.y.clamp(0.0, dish.height);
    }
}
