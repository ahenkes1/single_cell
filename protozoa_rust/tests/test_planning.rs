//! Tests for planning module components.

use protozoa_rust::simulation::memory::SpatialGrid;
use protozoa_rust::simulation::planning::{Action, AgentState, MCTSPlanner};

const DISH_WIDTH: f64 = 100.0;
const DISH_HEIGHT: f64 = 50.0;

#[test]
fn test_action_all_returns_three_actions() {
    let actions = Action::all();
    assert_eq!(actions.len(), 3);
    assert!(actions.contains(&Action::TurnLeft));
    assert!(actions.contains(&Action::Straight));
    assert!(actions.contains(&Action::TurnRight));
}

#[test]
fn test_agent_state_new() {
    let state = AgentState::new(50.0, 25.0, 1.0, 0.5, 0.8);
    assert_eq!(state.x, 50.0);
    assert_eq!(state.y, 25.0);
    assert_eq!(state.angle, 1.0);
    assert_eq!(state.speed, 0.5);
    assert_eq!(state.energy, 0.8);
}

#[test]
fn test_agent_state_step_changes_state() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
    let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);

    let next = state.step(Action::Straight, &priors);

    // State should change
    assert_ne!(state.x, next.x);
}

#[test]
fn test_planner_default() {
    let planner = MCTSPlanner::default();
    // Default should start with Straight action
    assert!(matches!(planner.best_action(), Action::Straight));
}

#[test]
fn test_planner_best_action_accessor() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
    let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);
    let mut planner = MCTSPlanner::new();

    // Plan once
    let action = planner.plan(&state, &priors);

    // best_action() should return the same as the last plan
    assert_eq!(planner.best_action(), action);
}

#[test]
fn test_multiple_plans_with_different_states() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
    let mut planner = MCTSPlanner::new();

    // Plan from different positions - should not crash
    for i in 0..5 {
        #[allow(clippy::cast_precision_loss)]
        let state = AgentState::new(10.0 + i as f64 * 15.0, 25.0, 0.0, 1.0, 1.0);
        let action = planner.plan(&state, &priors);

        assert!(matches!(
            action,
            Action::TurnLeft | Action::Straight | Action::TurnRight
        ));
    }
}

#[test]
fn test_planner_with_trained_priors() {
    let mut priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

    // Heavily train a region
    for _ in 0..50 {
        priors.update(50.0, 25.0, 0.95);
    }

    let state = AgentState::new(40.0, 25.0, 0.0, 1.0, 1.0);
    let mut planner = MCTSPlanner::new();
    let action = planner.plan(&state, &priors);

    // Should return a valid action
    assert!(matches!(
        action,
        Action::TurnLeft | Action::Straight | Action::TurnRight
    ));
}

#[test]
fn test_angle_wrapping_in_step() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
    let state = AgentState::new(50.0, 25.0, std::f64::consts::PI * 1.9, 1.0, 1.0);

    // Turn left should wrap angle correctly
    let next = state.step(Action::TurnLeft, &priors);
    assert!(next.angle >= 0.0);
    assert!(next.angle < 2.0 * std::f64::consts::PI);
}

#[test]
fn test_step_near_boundary() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

    // Test each corner
    let corners = [
        (0.1, 0.1, 0.0),                               // Bottom-left, facing right
        (DISH_WIDTH - 0.1, 0.1, std::f64::consts::PI), // Bottom-right, facing left
        (0.1, DISH_HEIGHT - 0.1, 0.0),                 // Top-left, facing right
        (DISH_WIDTH - 0.1, DISH_HEIGHT - 0.1, std::f64::consts::PI), // Top-right, facing left
    ];

    for (x, y, angle) in corners {
        let state = AgentState::new(x, y, angle, 10.0, 1.0);
        let next = state.step(Action::Straight, &priors);

        // Should stay in bounds
        assert!(
            next.x >= 0.0 && next.x <= DISH_WIDTH,
            "x out of bounds: {}",
            next.x
        );
        assert!(
            next.y >= 0.0 && next.y <= DISH_HEIGHT,
            "y out of bounds: {}",
            next.y
        );
    }
}

#[test]
fn test_step_energy_never_exceeds_one() {
    let mut priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

    // Create very high nutrient area
    for _ in 0..100 {
        priors.update(50.0, 25.0, 1.0);
    }

    let state = AgentState::new(50.0, 25.0, 0.0, 0.1, 0.99);
    let next = state.step(Action::Straight, &priors);

    assert!(next.energy <= 1.0, "Energy exceeds 1.0: {}", next.energy);
}

#[test]
fn test_step_energy_never_below_zero() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);

    // Start with very low energy
    let state = AgentState::new(50.0, 25.0, 0.0, MAX_SPEED, 0.001);

    // Multiple steps should not go below zero
    let mut current = state;
    for _ in 0..5 {
        current = current.step(Action::Straight, &priors);
        assert!(
            current.energy >= 0.0,
            "Energy went below 0: {}",
            current.energy
        );
    }
}

const MAX_SPEED: f64 = 1.5;
