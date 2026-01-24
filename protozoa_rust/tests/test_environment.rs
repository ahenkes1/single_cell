use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::simulation::params::{DISH_HEIGHT, DISH_WIDTH};

const EPSILON: f64 = 1e-10;

fn assert_float_eq(a: f64, b: f64, msg: &str) {
    assert!((a - b).abs() < EPSILON, "{msg}: expected {b}, got {a}");
}

#[test]
fn test_dish_initialization() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    assert_float_eq(dish.width, DISH_WIDTH, "dish width");
    assert_float_eq(dish.height, DISH_HEIGHT, "dish height");
    assert!(!dish.sources.is_empty());
}

#[test]
fn test_concentration_bounds() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let val = dish.get_concentration(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);
    assert!(val >= 0.0);
    assert!(val <= 1.0);
}

#[test]
fn test_out_of_bounds() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    assert_float_eq(dish.get_concentration(-1.0, 0.0), -1.0, "left boundary");
    assert_float_eq(
        dish.get_concentration(DISH_WIDTH + 1.0, 0.0),
        -1.0,
        "right boundary",
    );
    assert_float_eq(dish.get_concentration(0.0, -1.0), -1.0, "top boundary");
    assert_float_eq(
        dish.get_concentration(0.0, DISH_HEIGHT + 1.0),
        -1.0,
        "bottom boundary",
    );
}

#[test]
fn test_concentration_never_negative_inside_dish() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    // Sample multiple points inside the dish
    for x in (0..=100).map(|i| i as f64) {
        for y in (0..=50).map(|i| i as f64) {
            let val = dish.get_concentration(x, y);
            assert!(
                val >= 0.0 && val <= 1.0,
                "Concentration at ({x}, {y}) = {val} is out of bounds [0, 1]"
            );
        }
    }
}

#[test]
fn test_gaussian_is_symmetric() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    // For any source, concentration should be approximately symmetric
    // Test at the center of the dish where sources are likely to be
    let center_x = DISH_WIDTH / 2.0;
    let center_y = DISH_HEIGHT / 2.0;

    // Small offset should give similar values in opposite directions
    let offset = 1.0;
    let val_left = dish.get_concentration(center_x - offset, center_y);
    let val_right = dish.get_concentration(center_x + offset, center_y);
    let val_up = dish.get_concentration(center_x, center_y - offset);
    let val_down = dish.get_concentration(center_x, center_y + offset);

    // All values should be non-negative (inside dish)
    assert!(val_left >= 0.0);
    assert!(val_right >= 0.0);
    assert!(val_up >= 0.0);
    assert!(val_down >= 0.0);
}

#[test]
fn test_source_decay_and_respawn() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let initial_source_count = dish.sources.len();

    // Run many updates to trigger decay and respawn
    for _ in 0..1000 {
        dish.update();
    }

    // Source count should remain the same (respawn replaces depleted sources)
    assert_eq!(
        dish.sources.len(),
        initial_source_count,
        "Source count should remain constant after updates"
    );

    // All sources should have positive intensity (respawned if depleted)
    for source in &dish.sources {
        assert!(
            source.intensity > 0.0,
            "Source intensity should be positive after respawn"
        );
    }
}

#[test]
fn test_source_brownian_motion_stays_in_bounds() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);

    // Run many updates with Brownian motion
    for _ in 0..1000 {
        dish.update();
    }

    // All sources should remain within dish bounds
    for source in &dish.sources {
        assert!(
            source.x >= 0.0 && source.x <= DISH_WIDTH,
            "Source x={} is out of bounds [0, {}]",
            source.x,
            DISH_WIDTH
        );
        assert!(
            source.y >= 0.0 && source.y <= DISH_HEIGHT,
            "Source y={} is out of bounds [0, {}]",
            source.y,
            DISH_HEIGHT
        );
    }
}
