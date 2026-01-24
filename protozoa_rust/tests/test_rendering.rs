use protozoa_rust::simulation::agent::{AgentMode, Protozoa};
use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::simulation::memory::CellPrior;
use protozoa_rust::simulation::params::{DISH_HEIGHT, DISH_WIDTH};
use protozoa_rust::simulation::planning::{Action, ActionDetail};
use protozoa_rust::ui::DashboardState;
use protozoa_rust::ui::LandmarkSnapshot;
use protozoa_rust::ui::field::compute_field_grid;
use protozoa_rust::ui::render::{
    compute_quadrant_layout, compute_sidebar_layout, format_landmarks_list, format_mcts_summary,
    format_metrics_overlay, petri_dish_grid_size, render_spatial_grid_lines,
};
use ratatui::layout::Rect;
use ratatui::widgets::{Block, Borders};

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
fn test_dashboard_temporal_gradient_not_zero() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(50.0, 25.0);

    // First tick: establish baseline
    agent.val_l = 0.3;
    agent.val_r = 0.3;
    agent.update_state(&dish);

    // Second tick: create a temporal gradient
    agent.val_l = 0.7;
    agent.val_r = 0.7;
    agent.update_state(&dish);

    let state = DashboardState::from_agent(&agent, &dish);

    // Temporal gradient should be 0.7 - 0.3 = 0.4
    assert!(
        (state.temporal_gradient - 0.4).abs() < 0.01,
        "temporal_gradient {} should be approximately 0.4",
        state.temporal_gradient
    );
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

#[test]
fn test_quadrant_layout_dimensions() {
    let area = Rect::new(0, 0, 120, 40);
    let quadrants = compute_quadrant_layout(area);

    // Should have 4 quadrants
    assert_eq!(quadrants.len(), 4);

    // Each quadrant should be roughly half the area
    for q in &quadrants {
        assert!(q.width >= 50);
        assert!(q.height >= 15);
    }

    // Top-left should start at origin
    assert_eq!(quadrants[0].x, 0);
    assert_eq!(quadrants[0].y, 0);
}

#[test]
fn test_petri_dish_grid_size_uses_main_panel() {
    let area = Rect::new(0, 0, 120, 40);
    let (main, _) = compute_sidebar_layout(area);
    let inner = Block::default().borders(Borders::ALL).inner(main);

    let (rows, cols) = petri_dish_grid_size(area);

    assert_eq!(rows, inner.height as usize);
    assert_eq!(cols, inner.width as usize);
}

#[test]
fn test_metrics_overlay_content() {
    let lines = format_metrics_overlay(
        0.82, // energy
        AgentMode::Exploring,
        -0.12, // prediction_error
        0.85,  // precision
        1.3,   // speed
        127.0, // angle in degrees
        0.74,  // sensor_left
        0.68,  // sensor_right
        -0.02, // temporal_gradient
    );

    // Should have 6 lines
    assert_eq!(lines.len(), 6);

    // First line should contain energy bar
    assert!(lines[0].contains("E:"));
    assert!(lines[0].contains("82%"));

    // Second line should contain mode
    assert!(lines[1].contains("EXPLORING"));
}

#[test]
fn test_spatial_grid_ascii_mapping() {
    // Create a simple 4x2 grid
    let mut cells = vec![CellPrior::default(); 8];

    // Set different mean values
    cells[0].mean = 0.0; // Should be ' '
    cells[1].mean = 0.3; // Should be around ':'
    cells[2].mean = 0.6; // Should be around '+'
    cells[3].mean = 0.9; // Should be around '@'

    let lines = render_spatial_grid_lines(&cells, 4, 2, None);

    assert_eq!(lines.len(), 2);
    // First row contains cells 0-3
    assert!(lines[0].contains(' ')); // Low value
    assert!(lines[1].len() >= 4);
}

#[test]
fn test_mcts_summary_format() {
    let details = vec![
        ActionDetail {
            action: Action::TurnLeft,
            total_efe: -1.5,
            pragmatic_value: -1.2,
            epistemic_value: -1.0,
            sample_trajectory: vec![(50.0, 25.0), (52.0, 27.0)],
        },
        ActionDetail {
            action: Action::Straight,
            total_efe: -2.3,
            pragmatic_value: -1.8,
            epistemic_value: -1.67,
            sample_trajectory: vec![(50.0, 25.0), (55.0, 25.0)],
        },
    ];

    let lines = format_mcts_summary(&details, 7);

    // Should have lines for best action, G, Prag, Epis, Rolls, Depth, Replan
    assert!(lines.len() >= 5);
    assert!(lines[0].contains("Best"));
}

#[test]
fn test_mcts_summary_turn_right_arrow() {
    let details = vec![
        ActionDetail {
            action: Action::TurnRight,
            total_efe: 1.0,
            pragmatic_value: 0.5,
            epistemic_value: 0.5,
            sample_trajectory: vec![(50.0, 25.0)],
        },
        ActionDetail {
            action: Action::TurnLeft,
            total_efe: 0.2,
            pragmatic_value: 0.1,
            epistemic_value: 0.1,
            sample_trajectory: vec![(50.0, 25.0)],
        },
    ];

    let lines = format_mcts_summary(&details, 4);

    assert!(lines[0].contains("↘"));
}

#[test]
fn test_landmarks_list_format() {
    let landmarks = vec![
        LandmarkSnapshot {
            x: 12.0,
            y: 8.0,
            reliability: 0.92,
            visit_count: 4,
        },
        LandmarkSnapshot {
            x: 3.0,
            y: 18.0,
            reliability: 0.67,
            visit_count: 2,
        },
    ];

    let lines = format_landmarks_list(&landmarks, Some(0));

    // Should have header + landmarks
    assert!(lines.len() >= 3);
    // First landmark should have nav arrow
    assert!(lines[2].starts_with('→'));
}

#[test]
fn test_landmarks_list_reliability_full() {
    let landmarks = vec![LandmarkSnapshot {
        x: 12.0,
        y: 8.0,
        reliability: 1.0,
        visit_count: 1,
    }];

    let lines = format_landmarks_list(&landmarks, None);

    assert!(lines[2].contains("1.00"));
}

#[test]
fn test_dashboard_state_uses_visit_count() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(10.0, 10.0);

    agent.episodic_memory.maybe_store(10.0, 10.0, 0.9, 10);
    agent.episodic_memory.update_on_visit(10.0, 10.0, 0.9, 42);

    let state = DashboardState::from_agent(&agent, &dish);

    assert_eq!(state.landmarks.len(), 1);
    assert_eq!(state.landmarks[0].visit_count, 2);
}
