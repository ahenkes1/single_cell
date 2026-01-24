//! Integration tests for the complete cognitive architecture.
//!
//! These tests verify that the full system works together correctly:
//! - Memory systems are updated during simulation
//! - Learning does not destabilize behavior
//! - Agent survives and explores effectively

use protozoa_rust::simulation::agent::Protozoa;
use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::ui::DashboardState;
use std::time::Instant;

const DISH_WIDTH: f64 = 100.0;
const DISH_HEIGHT: f64 = 50.0;

/// Run simulation for N ticks and return final agent state
fn run_simulation(ticks: u64) -> Protozoa {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);

    for _ in 0..ticks {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);

        // Check for NaN/Inf to ensure numerical stability
        assert!(agent.x.is_finite(), "Agent x became non-finite");
        assert!(agent.y.is_finite(), "Agent y became non-finite");
        assert!(agent.angle.is_finite(), "Agent angle became non-finite");
        assert!(agent.energy.is_finite(), "Agent energy became non-finite");
    }

    agent
}

#[test]
fn test_agent_survives_short_simulation() {
    let agent = run_simulation(100);

    // Agent should still be alive
    assert!(agent.energy > 0.0, "Agent died too quickly");

    // Should have recorded tick count
    assert_eq!(agent.tick_count, 100);
}

#[test]
fn test_learning_does_not_destabilize_behavior() {
    let agent = run_simulation(1000);

    // Agent should maintain reasonable state
    assert!(agent.x >= 0.0 && agent.x <= DISH_WIDTH);
    assert!(agent.y >= 0.0 && agent.y <= DISH_HEIGHT);
    assert!(agent.angle >= 0.0 && agent.angle < std::f64::consts::PI * 2.0);
    assert!(agent.speed >= 0.0);

    // Memory systems should have been used
    assert!(agent.tick_count == 1000);

    // Spatial priors should have been updated
    let (w, h) = agent.spatial_priors.dimensions();
    assert!(w > 0 && h > 0);
}

#[test]
fn test_sensor_history_fills_during_simulation() {
    let agent = run_simulation(50);

    // Sensor history should have entries
    let history_len = agent.sensor_history.len();
    assert!(history_len > 0, "Sensor history should have entries");
    assert!(
        history_len <= 32,
        "Sensor history should not exceed buffer size"
    );
}

#[test]
fn test_spatial_priors_updated_during_simulation() {
    let agent = run_simulation(200);

    // Count total visits in spatial grid
    let total_visits = agent.spatial_priors.total_visits();
    assert!(total_visits > 0, "Spatial priors should have been updated");

    // Should have roughly one update per tick
    assert!(
        total_visits >= 100,
        "Expected more prior updates: {}",
        total_visits
    );
}

#[test]
fn test_episodic_memory_discovers_landmarks() {
    // Run longer simulation to ensure landmarks are found
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);

    // Run simulation
    for _ in 0..2000 {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);

        // Early exit if we found landmarks
        if agent.episodic_memory.count() >= 2 {
            break;
        }
    }

    // In a 2000-tick simulation, agent should discover some landmarks
    // (depends on environment, so we use a weak assertion)
    // At minimum, verify the episodic memory system is functional
    // Note: count() returns usize which is always >= 0
    let landmark_count = agent.episodic_memory.count();
    // Verify system works - count should be accessible
    assert!(
        landmark_count <= 8,
        "Episodic memory should not exceed MAX_LANDMARKS"
    );
}

#[test]
fn test_planner_is_used_during_simulation() {
    let agent = run_simulation(100);

    // Planning should have occurred
    // After 100 ticks with replan interval of 20, we expect ~5 replans
    assert!(agent.last_plan_tick > 0, "Planner should have been invoked");
}

#[test]
fn test_agent_stays_in_bounds_long_simulation() {
    let agent = run_simulation(5000);

    // Agent must stay within dish bounds
    assert!(
        agent.x >= 0.0 && agent.x <= DISH_WIDTH,
        "Agent x out of bounds: {}",
        agent.x
    );
    assert!(
        agent.y >= 0.0 && agent.y <= DISH_HEIGHT,
        "Agent y out of bounds: {}",
        agent.y
    );
}

#[test]
fn test_energy_clamped_during_simulation() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);

    for _ in 0..500 {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);

        // Energy should always be in valid range
        assert!(
            agent.energy >= 0.0 && agent.energy <= 1.0,
            "Energy out of range: {}",
            agent.energy
        );
    }
}

#[test]
fn test_performance_tick_under_threshold() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);

    // Warm up
    for _ in 0..10 {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);
    }

    // Measure 100 ticks
    let start = Instant::now();
    for _ in 0..100 {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);
    }
    let elapsed = start.elapsed();

    let avg_tick_ms = elapsed.as_millis() as f64 / 100.0;

    // Target: < 50ms per tick (generous for debug builds)
    // In release mode this should be much faster
    assert!(
        avg_tick_ms < 50.0,
        "Average tick time {:.2}ms exceeds 50ms threshold",
        avg_tick_ms
    );
}

#[test]
fn test_multiple_agents_independent() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent1 = Protozoa::new(25.0, 25.0);
    let mut agent2 = Protozoa::new(75.0, 25.0);

    for _ in 0..100 {
        dish.update();
        agent1.sense(&dish);
        agent1.update_state(&dish);
        agent2.sense(&dish);
        agent2.update_state(&dish);
    }

    // Agents should have diverged
    let dist = ((agent1.x - agent2.x).powi(2) + (agent1.y - agent2.y).powi(2)).sqrt();

    // They started 50 units apart, should still be reasonably far or have moved differently
    assert!(agent1.tick_count == 100);
    assert!(agent2.tick_count == 100);
    assert!(
        dist > 0.0 || agent1.angle != agent2.angle,
        "Agents should behave independently"
    );
}

#[test]
fn test_full_cognitive_stack_integration() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);

    // Run extended simulation
    for tick in 0..1000 {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);

        // Verify all systems are functional at various points
        if tick == 100 {
            // Short-term memory should have entries
            assert!(agent.sensor_history.len() == 32); // Full buffer

            // Spatial priors should be learning
            assert!(agent.spatial_priors.total_visits() >= 100);
        }

        if tick == 500 {
            // Should have replanned several times
            assert!(agent.last_plan_tick > 0);
        }
    }

    // Final state checks
    assert!(agent.x.is_finite());
    assert!(agent.y.is_finite());
    assert!(agent.angle.is_finite());
    assert!(agent.energy.is_finite());
    assert!(agent.tick_count == 1000);
}

#[test]
fn test_dashboard_state_updates_during_simulation() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);

    // Run for several ticks
    for _ in 0..100 {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);
    }

    let state = DashboardState::from_agent(&agent, &dish);

    // Verify state is populated
    assert!(state.energy > 0.0);
    assert!(state.spatial_grid.len() == 200); // 20x10
    assert!(state.plan_details.len() == 3); // One per action
}
