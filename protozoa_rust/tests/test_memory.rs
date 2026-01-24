//! Tests for memory module components.

use protozoa_rust::simulation::memory::{
    CellPrior, RingBuffer, SensorHistory, SensorSnapshot, SpatialGrid,
};

// ============== Ring Buffer Tests ==============

#[test]
fn test_ring_buffer_basic() {
    let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
    assert!(buf.is_empty());

    buf.push(1);
    buf.push(2);
    buf.push(3);

    assert_eq!(buf.len(), 3);
    assert_eq!(*buf.get(0).unwrap(), 1);
    assert_eq!(*buf.get(2).unwrap(), 3);
}

#[test]
fn test_ring_buffer_overflow_ordering() {
    let mut buf: RingBuffer<i32, 3> = RingBuffer::new();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf.push(4);
    buf.push(5);

    // After overflow, oldest should be 3
    assert_eq!(buf.len(), 3);
    let items: Vec<_> = buf.iter().copied().collect();
    assert_eq!(items, vec![3, 4, 5]);
}

#[test]
fn test_sensor_history() {
    let mut history: SensorHistory = SensorHistory::new();

    history.push(SensorSnapshot {
        val_l: 0.5,
        val_r: 0.6,
        x: 10.0,
        y: 20.0,
        energy: 0.8,
        tick: 0,
    });

    history.push(SensorSnapshot {
        val_l: 0.7,
        val_r: 0.8,
        x: 15.0,
        y: 25.0,
        energy: 0.75,
        tick: 1,
    });

    assert_eq!(history.len(), 2);

    let last = history.last().unwrap();
    assert_eq!(last.tick, 1);
    assert!((last.val_l - 0.7).abs() < 1e-10);
}

// ============== Spatial Grid Tests ==============

#[test]
fn test_spatial_grid_cell_prior() {
    let mut cell = CellPrior::default();
    assert_eq!(cell.visits, 0);
    assert!((cell.mean - 0.5).abs() < 1e-10);

    // First observation
    cell.update(0.8);
    assert_eq!(cell.visits, 1);

    // Mean should move toward observation
    cell.update(0.8);
    cell.update(0.8);
    assert!(cell.mean > 0.7);
}

#[test]
fn test_spatial_grid_welford_convergence() {
    let mut cell = CellPrior::default();

    // Add observations centered around 0.6
    let observations = [0.55, 0.65, 0.58, 0.62, 0.60, 0.59, 0.61];
    for obs in observations {
        cell.update(obs);
    }

    // Mean should be close to average of observations
    let expected_mean: f64 = observations.iter().sum::<f64>() / observations.len() as f64;
    assert!((cell.mean - expected_mean).abs() < 0.1);
}

#[test]
fn test_spatial_grid_precision_with_consistent_data() {
    let mut grid: SpatialGrid<20, 10> = SpatialGrid::default();

    // Consistent observations should increase precision
    let initial_precision = grid.precision(50.0, 25.0);

    for _ in 0..20 {
        grid.update(50.0, 25.0, 0.7);
    }

    let final_precision = grid.precision(50.0, 25.0);
    assert!(final_precision > initial_precision * 2.0);
}

#[test]
fn test_spatial_grid_different_cells() {
    let mut grid: SpatialGrid<20, 10> = SpatialGrid::default();

    // Update two different locations
    grid.update(10.0, 10.0, 0.9);
    grid.update(90.0, 40.0, 0.1);

    // They should have different means
    let cell1 = grid.get_cell(10.0, 10.0);
    let cell2 = grid.get_cell(90.0, 40.0);

    assert!(cell1.mean > 0.7);
    assert!(cell2.mean < 0.3);
}

#[test]
fn test_spatial_grid_boundary_conditions() {
    let grid: SpatialGrid<20, 10> = SpatialGrid::default();

    // These should not panic
    let _ = grid.get_cell(0.0, 0.0);
    let _ = grid.get_cell(99.999, 49.999);
    let _ = grid.get_cell(-0.001, -0.001); // Clamped to 0
    let _ = grid.get_cell(100.1, 50.1); // Clamped to max
}

#[test]
fn test_spatial_grid_expected_value() {
    let mut grid: SpatialGrid<20, 10> = SpatialGrid::default();

    grid.update(50.0, 25.0, 0.8);
    grid.update(50.0, 25.0, 0.8);
    grid.update(50.0, 25.0, 0.8);

    let expected = grid.expected(50.0, 25.0);
    assert!((expected - 0.8).abs() < 0.1);
}
