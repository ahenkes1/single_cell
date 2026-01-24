use protozoa_rust::simulation::agent::Protozoa;
use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::simulation::params::{
    DISH_HEIGHT, DISH_WIDTH, EXHAUSTION_SPEED_FACTOR, EXHAUSTION_THRESHOLD, MAX_SPEED,
};
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

fn assert_float_eq(a: f64, b: f64, msg: &str) {
    assert!((a - b).abs() < EPSILON, "{msg}: expected {b}, got {a}");
}

#[test]
fn test_agent_initialization() {
    let agent = Protozoa::new(50.0, 50.0);
    assert_float_eq(agent.x, 50.0, "x position");
    assert_float_eq(agent.y, 50.0, "y position");
    assert_float_eq(agent.speed, 0.0, "speed");
    assert_float_eq(agent.energy, 1.0, "energy");
}

#[test]
fn test_sense() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    agent.sense(&dish);
    // Values should be between -1.0 (void) and 1.0 (max nutrient)
    assert!(agent.val_l >= -1.0 && agent.val_l <= 1.0);
    assert!(agent.val_r >= -1.0 && agent.val_r <= 1.0);
}

#[test]
fn test_update_state_movement() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Set high error to force movement
    agent.val_l = 0.0;
    agent.val_r = 0.0;
    // target is 0.8, so error = 0.0 - 0.8 = -0.8. |Error| = 0.8
    // Speed should be max_speed * 0.8

    agent.update_state(&dish);

    assert!(agent.speed > 0.0);
    assert!(agent.speed <= MAX_SPEED);

    // Position should have changed (unless speed is 0, which it shouldn't be)
    assert!(agent.x != 50.0 || agent.y != 50.0);
}

#[test]
fn test_energy_consumption() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Force movement
    agent.val_l = 0.0;
    agent.val_r = 0.0;

    agent.update_state(&dish);

    // Energy should decrease because intake (0.03 * 0) is 0, but cost is > 0
    assert!(agent.energy < 1.0);
}

#[test]
fn test_exhaustion_state() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Force low energy state
    agent.energy = EXHAUSTION_THRESHOLD / 2.0; // Below threshold
    agent.val_l = 0.0;
    agent.val_r = 0.0;

    agent.update_state(&dish);

    // Speed should be reduced by exhaustion factor
    // Base speed would be MAX_SPEED * 0.8 (error = -0.8)
    // After exhaustion: speed *= EXHAUSTION_SPEED_FACTOR
    let expected_max = MAX_SPEED * 0.8 * EXHAUSTION_SPEED_FACTOR;
    assert!(
        agent.speed <= expected_max + EPSILON,
        "Speed {} should be <= {} when exhausted",
        agent.speed,
        expected_max
    );
}

#[test]
fn test_boundary_clamping() {
    let mut agent = Protozoa::new(DISH_WIDTH - 0.1, DISH_HEIGHT / 2.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Set angle to push agent past right boundary
    agent.angle = 0.0; // Moving right
    agent.val_l = 0.0;
    agent.val_r = 0.0;

    // Run multiple updates to ensure agent would go past boundary
    for _ in 0..100 {
        agent.update_state(&dish);
    }

    // Agent should be clamped to dish bounds
    assert!(
        agent.x >= 0.0 && agent.x <= DISH_WIDTH,
        "x={} should be in [0, {}]",
        agent.x,
        DISH_WIDTH
    );
    assert!(
        agent.y >= 0.0 && agent.y <= DISH_HEIGHT,
        "y={} should be in [0, {}]",
        agent.y,
        DISH_HEIGHT
    );
}

#[test]
fn test_angle_normalization() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Set angle to extreme negative value
    agent.angle = -10.0 * PI;
    agent.val_l = 0.0;
    agent.val_r = 0.0;

    agent.update_state(&dish);

    // Angle should be normalized to [0, 2*PI)
    assert!(agent.angle >= 0.0, "Angle {} should be >= 0", agent.angle);
    assert!(
        agent.angle < 2.0 * PI,
        "Angle {} should be < 2*PI",
        agent.angle
    );
}

#[test]
fn test_angle_normalization_positive() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Set angle to extreme positive value
    agent.angle = 100.0 * PI;
    agent.val_l = 0.0;
    agent.val_r = 0.0;

    agent.update_state(&dish);

    // Angle should be normalized to [0, 2*PI)
    assert!(agent.angle >= 0.0, "Angle {} should be >= 0", agent.angle);
    assert!(
        agent.angle < 2.0 * PI,
        "Angle {} should be < 2*PI",
        agent.angle
    );
}

#[test]
fn test_temporal_gradient_tracking() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    agent.val_l = 0.6;
    agent.val_r = 0.4;
    agent.update_state(&dish);

    // last_mean_sense should be updated to midpoint of val_l and val_r
    let expected = (0.6 + 0.4) / 2.0; // 0.5
    assert!(
        (agent.last_mean_sense - expected).abs() < EPSILON,
        "last_mean_sense {} should be {}",
        agent.last_mean_sense,
        expected
    );
}

#[test]
fn test_speed_proportional_to_error() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Target is 0.8, so if mean_sense = 0.0, error = -0.8
    agent.val_l = 0.0;
    agent.val_r = 0.0;
    agent.update_state(&dish);
    let speed_high_error = agent.speed;

    // Reset and test with lower error
    let mut agent2 = Protozoa::new(50.0, 50.0);
    // If mean_sense = 0.7, error = 0.7 - 0.8 = -0.1
    agent2.val_l = 0.7;
    agent2.val_r = 0.7;
    agent2.update_state(&dish);
    let speed_low_error = agent2.speed;

    // Higher error should result in higher speed
    assert!(
        speed_high_error > speed_low_error,
        "Speed with high error ({}) should be > speed with low error ({})",
        speed_high_error,
        speed_low_error
    );
}

#[test]
fn test_energy_clamped_to_valid_range() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Force very low energy
    agent.energy = 0.0001;
    agent.val_l = 0.0;
    agent.val_r = 0.0;

    // Run many updates to deplete energy
    for _ in 0..1000 {
        agent.update_state(&dish);
    }

    // Energy should never go below 0
    assert!(
        agent.energy >= 0.0,
        "Energy {} should be >= 0",
        agent.energy
    );
    assert!(
        agent.energy <= 1.0,
        "Energy {} should be <= 1",
        agent.energy
    );
}

#[test]
fn test_energy_increases_near_nutrients() {
    let mut agent = Protozoa::new(50.0, 50.0);
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Simulate being in a high-nutrient area (mean_sense close to target)
    // At target (0.8), error = 0, speed = 0, cost is minimal, intake is positive
    agent.val_l = 0.8;
    agent.val_r = 0.8;
    agent.energy = 0.5; // Start at half energy

    agent.update_state(&dish);

    // With high nutrient and low speed, energy should increase
    // Intake = 0.03 * 0.8 = 0.024
    // Cost = 0.0005 + 0 (speed is ~0 at target)
    // Net = 0.024 - 0.0005 = 0.0235 (positive)
    assert!(
        agent.energy > 0.5,
        "Energy {} should increase when near nutrients (started at 0.5)",
        agent.energy
    );
}
