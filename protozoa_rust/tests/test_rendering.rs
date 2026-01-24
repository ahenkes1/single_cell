use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::ui::field::compute_field_grid;

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
