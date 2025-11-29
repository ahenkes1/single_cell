use protozoa_rust::simulation::agent::Protozoa;
use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::simulation::params::{DISH_HEIGHT, DISH_WIDTH, MAX_SPEED};

#[test]
fn test_agent_initialization() {
    let agent = Protozoa::new(50.0, 50.0);
    assert_eq!(agent.x, 50.0);
    assert_eq!(agent.y, 50.0);
    assert_eq!(agent.speed, 0.0);
    assert_eq!(agent.energy, 1.0);
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

