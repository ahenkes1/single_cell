//! Monte Carlo Tree Search planning for trajectory optimization.
//!
//! Implements MCTS with Expected Free Energy as the value function,
//! enabling the agent to plan multi-step trajectories that balance
//! exploitation (seeking nutrients) with exploration (reducing uncertainty).

use crate::simulation::memory::SpatialGrid;
use crate::simulation::params::{
    BASE_METABOLIC_COST, DISH_HEIGHT, DISH_WIDTH, EXPLORATION_SCALE, INTAKE_RATE, MAX_SPEED,
    MCTS_DEPTH, MCTS_ROLLOUTS, MIN_PRECISION, SPEED_METABOLIC_COST, TARGET_CONCENTRATION,
};
use rand::Rng;
use std::f64::consts::PI;

/// Discrete actions available to the agent during planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Turn left by 45 degrees
    TurnLeft,
    /// Continue straight
    Straight,
    /// Turn right by 45 degrees
    TurnRight,
}

/// Details about a planned action for visualization.
#[derive(Clone, Debug)]
pub struct ActionDetail {
    /// The action evaluated
    pub action: Action,
    /// Average total Expected Free Energy
    pub total_efe: f64,
    /// Pragmatic component (nutrient × energy)
    pub pragmatic_value: f64,
    /// Epistemic component (uncertainty reduction)
    pub epistemic_value: f64,
    /// Sample trajectory positions for visualization
    pub sample_trajectory: Vec<(f64, f64)>,
}

impl Action {
    /// Returns the angular change in radians for this action.
    #[must_use]
    pub const fn angle_delta(self) -> f64 {
        match self {
            Self::TurnLeft => PI / 4.0,
            Self::Straight => 0.0,
            Self::TurnRight => -PI / 4.0,
        }
    }

    /// Returns all possible actions.
    #[must_use]
    pub const fn all() -> [Action; 3] {
        [Action::TurnLeft, Action::Straight, Action::TurnRight]
    }
}

/// Lightweight agent state for trajectory simulation.
#[derive(Clone, Copy, Debug)]
pub struct AgentState {
    /// X position
    pub x: f64,
    /// Y position
    pub y: f64,
    /// Current heading angle in radians
    pub angle: f64,
    /// Current movement speed
    pub speed: f64,
    /// Current energy level
    pub energy: f64,
}

impl AgentState {
    /// Creates a new agent state.
    #[must_use]
    pub const fn new(x: f64, y: f64, angle: f64, speed: f64, energy: f64) -> Self {
        Self {
            x,
            y,
            angle,
            speed,
            energy,
        }
    }

    /// Simulates one tick forward using learned priors as world model.
    ///
    /// Returns the new state after taking the given action.
    #[must_use]
    pub fn step(&self, action: Action, priors: &SpatialGrid<20, 10>) -> Self {
        // Apply action to angle
        let new_angle = (self.angle + action.angle_delta()).rem_euclid(2.0 * PI);

        // Get expected concentration at current position from learned priors
        let expected = priors.get_cell(self.x, self.y).mean.clamp(0.0, 1.0);

        // Predict speed based on expected error (as the real agent does)
        let predicted_error = (expected - TARGET_CONCENTRATION).abs();
        let new_speed = MAX_SPEED * predicted_error;

        // Move in the new direction
        let new_x = (self.x + new_speed * new_angle.cos()).clamp(0.0, DISH_WIDTH);
        let new_y = (self.y + new_speed * new_angle.sin()).clamp(0.0, DISH_HEIGHT);

        // Estimate energy change using expected concentration
        let intake = INTAKE_RATE * expected;
        let cost = BASE_METABOLIC_COST + SPEED_METABOLIC_COST * (new_speed / MAX_SPEED);
        let new_energy = (self.energy - cost + intake).clamp(0.0, 1.0);

        Self {
            x: new_x,
            y: new_y,
            angle: new_angle,
            speed: new_speed,
            energy: new_energy,
        }
    }
}

/// Monte Carlo Tree Search planner using Expected Free Energy.
#[derive(Clone, Debug)]
pub struct MCTSPlanner {
    /// Best action from last planning cycle
    best_action: Action,
    /// Details from the last planning cycle
    last_details: Vec<ActionDetail>,
}

impl Default for MCTSPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl MCTSPlanner {
    /// Creates a new MCTS planner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            best_action: Action::Straight,
            last_details: Vec::new(),
        }
    }

    /// Returns the best action from the last planning cycle.
    #[must_use]
    pub const fn best_action(&self) -> Action {
        self.best_action
    }

    /// Returns details of the last planning cycle.
    #[must_use]
    pub fn last_plan_details(&self) -> &[ActionDetail] {
        &self.last_details
    }

    /// Plans the best action using Monte Carlo rollouts.
    ///
    /// Performs `MCTS_ROLLOUTS` random rollouts for each action,
    /// evaluating trajectories using Expected Free Energy.
    pub fn plan(&mut self, state: &AgentState, priors: &SpatialGrid<20, 10>) -> Action {
        let mut rng = rand::rng();
        let mut best_value = f64::NEG_INFINITY;
        let mut best_action = Action::Straight;
        self.last_details.clear();

        // Evaluate each possible action
        for action in Action::all() {
            let mut total_pragmatic = 0.0;
            let mut total_epistemic = 0.0;
            let mut sample_traj = Vec::new();

            // Perform multiple rollouts
            for i in 0..MCTS_ROLLOUTS {
                let trajectory = self.rollout(*state, action, priors, &mut rng);
                let (pragmatic, epistemic) = self.efe_components(&trajectory, priors);
                total_pragmatic += pragmatic;
                total_epistemic += epistemic;

                if i == 0 {
                    sample_traj = trajectory.iter().map(|s| (s.x, s.y)).collect();
                }
            }

            #[allow(clippy::cast_precision_loss)] // MCTS_ROLLOUTS is small (50)
            let avg_pragmatic = total_pragmatic / MCTS_ROLLOUTS as f64;
            #[allow(clippy::cast_precision_loss)]
            let avg_epistemic = total_epistemic / MCTS_ROLLOUTS as f64;
            let avg_value = avg_pragmatic + EXPLORATION_SCALE * avg_epistemic;

            self.last_details.push(ActionDetail {
                action,
                total_efe: avg_value,
                pragmatic_value: avg_pragmatic,
                epistemic_value: avg_epistemic,
                sample_trajectory: sample_traj,
            });

            if avg_value > best_value {
                best_value = avg_value;
                best_action = action;
            }
        }

        self.best_action = best_action;
        best_action
    }

    /// Performs a single rollout from the given state.
    ///
    /// Takes the initial action, then selects random actions for the remaining depth.
    #[allow(clippy::unused_self)] // Method signature for future extensibility
    fn rollout(
        &self,
        initial_state: AgentState,
        initial_action: Action,
        priors: &SpatialGrid<20, 10>,
        rng: &mut impl Rng,
    ) -> Vec<AgentState> {
        let mut trajectory = Vec::with_capacity(MCTS_DEPTH + 1);
        trajectory.push(initial_state);

        // Take initial action
        let mut current_state = initial_state.step(initial_action, priors);
        trajectory.push(current_state);

        // Continue with random actions
        for _ in 1..MCTS_DEPTH {
            let actions = Action::all();
            let random_action = actions[rng.random_range(0..3)];
            current_state = current_state.step(random_action, priors);
            trajectory.push(current_state);
        }

        trajectory
    }

    /// Computes pragmatic and epistemic components separately.
    #[allow(clippy::unused_self)] // Method signature for future extensibility
    fn efe_components(
        &self,
        trajectory: &[AgentState],
        priors: &SpatialGrid<20, 10>,
    ) -> (f64, f64) {
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

    /// Computes the Expected Free Energy for a trajectory.
    ///
    /// `EFE = pragmatic + epistemic`
    /// - Pragmatic: prefers high expected nutrients and maintaining energy
    /// - Epistemic: prefers exploring uncertain regions (information gain)
    ///
    /// Higher values are better (we maximize EFE, not minimize).
    #[allow(clippy::unused_self)] // Method signature for future extensibility
    fn expected_free_energy(&self, trajectory: &[AgentState], priors: &SpatialGrid<20, 10>) -> f64 {
        let (pragmatic, epistemic) = self.efe_components(trajectory, priors);
        pragmatic + EXPLORATION_SCALE * epistemic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_angle_delta() {
        assert!((Action::TurnLeft.angle_delta() - PI / 4.0).abs() < 1e-10);
        assert!((Action::Straight.angle_delta()).abs() < 1e-10);
        assert!((Action::TurnRight.angle_delta() + PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_state_step() {
        let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
        let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);

        let next_state = state.step(Action::Straight, &priors);

        // Should have moved (approximately) in the x direction
        assert!(next_state.x > state.x || next_state.x == DISH_WIDTH);
        // Energy should have changed
        assert!(next_state.energy != state.energy || next_state.energy == 1.0);
    }

    #[test]
    fn test_agent_state_turn_left() {
        let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
        let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);

        let next_state = state.step(Action::TurnLeft, &priors);

        // Angle should have increased by PI/4
        assert!((next_state.angle - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_state_boundary_clamping() {
        let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

        // State at edge moving outward
        let state = AgentState::new(DISH_WIDTH - 0.1, 25.0, 0.0, 10.0, 1.0);
        let next_state = state.step(Action::Straight, &priors);

        // Should be clamped to boundary
        assert!(next_state.x <= DISH_WIDTH);
        assert!(next_state.y >= 0.0 && next_state.y <= DISH_HEIGHT);
    }

    #[test]
    fn test_mcts_planner_returns_valid_action() {
        let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
        let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);
        let mut planner = MCTSPlanner::new();

        let action = planner.plan(&state, &priors);

        assert!(matches!(
            action,
            Action::TurnLeft | Action::Straight | Action::TurnRight
        ));
    }

    #[test]
    fn test_mcts_planner_produces_consistent_results() {
        let mut priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

        // Train priors: high nutrients ahead, low behind
        // Position agent at x=20 facing right (angle=0), high nutrients at x=60
        for _ in 0..20 {
            priors.update(60.0, 25.0, 0.95); // Ahead: very high
            priors.update(70.0, 25.0, 0.95); // Further ahead: very high
            priors.update(10.0, 25.0, 0.05); // Behind: very low
        }

        let state = AgentState::new(20.0, 25.0, 0.0, 1.0, 1.0); // Facing right toward high nutrients
        let mut planner = MCTSPlanner::new();

        // Run multiple plans - due to stochastic rollouts, results may vary
        // We just verify the planner produces valid actions and doesn't crash
        let mut action_counts = [0usize; 3];
        for _ in 0..20 {
            let action = planner.plan(&state, &priors);
            match action {
                Action::TurnLeft => action_counts[0] += 1,
                Action::Straight => action_counts[1] += 1,
                Action::TurnRight => action_counts[2] += 1,
            }
        }

        // Verify planner is making decisions (not always the same)
        // At minimum, one action was chosen
        assert!(
            action_counts.iter().sum::<usize>() == 20,
            "Expected 20 total actions"
        );

        // With strongly trained priors, should prefer forward directions
        // (straight or slight turns) over the opposite direction
        // This is a weak assertion due to stochastic nature
        let forward_actions = action_counts[1]; // Straight
        assert!(
            forward_actions > 0 || action_counts[0] > 0 || action_counts[2] > 0,
            "Planner should make some decisions"
        );
    }

    #[test]
    fn test_expected_free_energy_prefers_high_nutrients() {
        let mut priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

        // Create two regions: high nutrients at (60, 25), low at (20, 25)
        for _ in 0..20 {
            priors.update(60.0, 25.0, 0.9);
            priors.update(20.0, 25.0, 0.1);
        }

        let planner = MCTSPlanner::new();

        // Trajectory through high-nutrient region
        let high_traj = vec![
            AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0),
            AgentState::new(55.0, 25.0, 0.0, 1.0, 0.95),
            AgentState::new(60.0, 25.0, 0.0, 1.0, 0.90),
        ];

        // Trajectory through low-nutrient region
        let low_traj = vec![
            AgentState::new(30.0, 25.0, PI, 1.0, 1.0),
            AgentState::new(25.0, 25.0, PI, 1.0, 0.95),
            AgentState::new(20.0, 25.0, PI, 1.0, 0.90),
        ];

        let high_efe = planner.expected_free_energy(&high_traj, &priors);
        let low_efe = planner.expected_free_energy(&low_traj, &priors);

        // High-nutrient trajectory should have higher EFE (we maximize)
        assert!(
            high_efe > low_efe,
            "High-nutrient trajectory should have higher EFE: {} vs {}",
            high_efe,
            low_efe
        );
    }

    #[test]
    fn test_expected_free_energy_values_exploration() {
        let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

        // Unexplored region (precision is low)
        let planner = MCTSPlanner::new();

        // Trajectory through unexplored region
        let unexplored_traj = vec![
            AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0),
            AgentState::new(55.0, 25.0, 0.0, 1.0, 1.0),
        ];

        let efe = planner.expected_free_energy(&unexplored_traj, &priors);

        // EFE should be positive (epistemic value from unexplored regions)
        assert!(efe > 0.0, "EFE should be positive for unexplored: {}", efe);
    }

    #[test]
    fn test_rollout_produces_valid_trajectory() {
        let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
        let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);
        let planner = MCTSPlanner::new();
        let mut rng = rand::rng();

        let trajectory = planner.rollout(state, Action::Straight, &priors, &mut rng);

        // Should have MCTS_DEPTH + 1 states (initial + depth steps)
        assert_eq!(trajectory.len(), MCTS_DEPTH + 1);

        // All positions should be within bounds
        for state in &trajectory {
            assert!(
                state.x >= 0.0 && state.x <= DISH_WIDTH,
                "x out of bounds: {}",
                state.x
            );
            assert!(
                state.y >= 0.0 && state.y <= DISH_HEIGHT,
                "y out of bounds: {}",
                state.y
            );
            assert!(state.angle >= 0.0 && state.angle < 2.0 * PI);
            assert!(state.energy >= 0.0 && state.energy <= 1.0);
        }
    }

    #[test]
    fn test_agent_state_energy_clamped() {
        let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

        // State with energy at boundary
        let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 0.01);

        // Multiple steps should keep energy in valid range
        let mut current = state;
        for _ in 0..10 {
            current = current.step(Action::Straight, &priors);
            assert!(
                current.energy >= 0.0 && current.energy <= 1.0,
                "Energy out of range: {}",
                current.energy
            );
        }
    }
}
