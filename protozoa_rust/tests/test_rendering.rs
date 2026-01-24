use protozoa_rust::simulation::agent::{AgentMode, Protozoa};
use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::simulation::params::{DISH_HEIGHT, DISH_WIDTH};
use protozoa_rust::ui::DashboardState;
use protozoa_rust::ui::field::compute_field_grid;

#[test]
fn test_dashboard_state_from_agent() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let agent = Protozoa::new(50.0, 25.0);

    let state = DashboardState::from_agent(&agent, &dish);

    assert!((state.energy - 1.0).abs() < 0.01);
    assert!(matches!(state.mode, AgentMode::Exploring));
    assert_eq!(state.landmark_count, 0);
}

#[test]
fn test_field_grid_computation() {
    let dish = PetriDish::new(100.0, 50.0);
    let rows = 10;
    let cols = 20;

    let grid = compute_field_grid(&dish, rows, cols);

    assert_eq!(grid.len(), rows);
    assert_eq!(grid[0].len(), cols);

    // Check that characters are valid ASCII
    for row in grid {
        for c in row.chars() {
            assert!(" .:-=+*#%@".contains(c));
        }
    }
}
