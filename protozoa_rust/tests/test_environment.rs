use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::simulation::params::{DISH_HEIGHT, DISH_WIDTH};

#[test]
fn test_dish_initialization() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    assert_eq!(dish.width, DISH_WIDTH);
    assert_eq!(dish.height, DISH_HEIGHT);
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
    assert_eq!(dish.get_concentration(-1.0, 0.0), -1.0);
    assert_eq!(dish.get_concentration(DISH_WIDTH + 1.0, 0.0), -1.0);
}

