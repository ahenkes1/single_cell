use crate::simulation::agent::AgentMode;
use crate::simulation::memory::CellPrior;
use crate::simulation::params::{MCTS_DEPTH, MCTS_ROLLOUTS};
use crate::simulation::planning::{Action, ActionDetail};
use crate::ui::{DashboardState, LandmarkSnapshot};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

/// Computes the main + sidebar layout for the dashboard.
/// Returns (`main_area`, `sidebar_panels`) where `sidebar_panels` is [Metrics, MCTS, Landmarks, Spatial].
#[must_use]
pub fn compute_sidebar_layout(area: Rect) -> (Rect, Vec<Rect>) {
    // Horizontal split: 70% main, 30% sidebar
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);

    let main = horizontal[0];

    // Sidebar vertical split: fixed heights for top 3, remaining for Spatial
    let sidebar_panels = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Metrics
            Constraint::Length(9),  // MCTS
            Constraint::Length(12), // Landmarks
            Constraint::Min(0),     // Spatial (remaining)
        ])
        .split(horizontal[1]);

    (main, sidebar_panels.to_vec())
}

#[must_use]
pub fn petri_dish_grid_size(area: Rect) -> (usize, usize) {
    let (main_area, _) = compute_sidebar_layout(area);
    let inner = Block::default().borders(Borders::ALL).inner(main_area);
    (inner.height as usize, inner.width as usize)
}

/// Computes the four quadrant areas for the dashboard layout.
#[must_use]
#[allow(dead_code)] // Legacy layout, kept as fallback
pub fn compute_quadrant_layout(area: Rect) -> Vec<Rect> {
    // Split vertically into top and bottom
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Split each row horizontally
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(vertical[0]);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(vertical[1]);

    vec![top[0], top[1], bottom[0], bottom[1]]
}

/// Draws the full cognitive dashboard with sidebar layout.
pub fn draw_dashboard(f: &mut Frame, grid_lines: Vec<String>, state: &DashboardState) {
    let (main_area, sidebar) = compute_sidebar_layout(f.area());

    // === Left: Petri Dish (full height) ===
    draw_petri_dish_panel(f, main_area, grid_lines);

    // === Right Sidebar ===
    // [0] Metrics (top)
    draw_metrics_panel(f, sidebar[0], state);

    // [1] MCTS Planning
    draw_mcts_panel(f, sidebar[1], state);

    // [2] Landmarks
    draw_landmarks_panel(f, sidebar[2], state);

    // [3] Spatial Memory (bottom, takes remaining space)
    draw_spatial_grid_panel(f, sidebar[3], state);
}

fn draw_petri_dish_panel(f: &mut Frame, area: Rect, grid_lines: Vec<String>) {
    let block = Block::default().title(" Petri Dish ").borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Render field only (no overlay - metrics moved to sidebar)
    let text: Vec<Line> = grid_lines
        .into_iter()
        .map(|s| Line::from(Span::raw(s)))
        .collect();
    let field = Paragraph::new(text);
    f.render_widget(field, inner);
}

fn draw_metrics_panel(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default().title(" Agent ").borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let angle_deg = state.angle.to_degrees();
    let lines = format_metrics_overlay(
        state.energy,
        state.mode,
        state.prediction_error,
        state.precision,
        state.speed,
        angle_deg,
        state.sensor_left,
        state.sensor_right,
        state.temporal_gradient,
    );

    let text: Vec<Line> = lines
        .into_iter()
        .map(|s| {
            Line::from(Span::styled(
                s,
                Style::default().add_modifier(Modifier::BOLD),
            ))
        })
        .collect();
    let paragraph = Paragraph::new(text);
    f.render_widget(paragraph, inner);
}

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn draw_spatial_grid_panel(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(" Spatial Memory ")
        .borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Determine target width based on available space
    let target_width = (inner.width as usize).min(state.grid_width);

    // Compress grid if needed
    let display_cells = if target_width < state.grid_width {
        compress_spatial_grid(
            &state.spatial_grid,
            state.grid_width,
            state.grid_height,
            target_width,
        )
    } else {
        state.spatial_grid.clone()
    };

    let display_width = target_width.min(state.grid_width);

    // Calculate agent's grid cell (in compressed coordinates)
    let compression_ratio = state.grid_width as f64 / display_width as f64;
    let agent_col =
        ((state.x / 100.0) * state.grid_width as f64 / compression_ratio).floor() as usize;
    let agent_row = ((state.y / 50.0) * state.grid_height as f64).floor() as usize;
    let agent_cell = Some((
        agent_row.min(state.grid_height.saturating_sub(1)),
        agent_col.min(display_width.saturating_sub(1)),
    ));

    let lines =
        render_spatial_grid_lines(&display_cells, display_width, state.grid_height, agent_cell);
    let text: Vec<Line> = lines
        .into_iter()
        .map(|s| Line::from(Span::raw(s)))
        .collect();
    let grid = Paragraph::new(text);
    f.render_widget(grid, inner);
}

fn draw_mcts_panel(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(" MCTS Planning ")
        .borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines = format_mcts_summary(&state.plan_details, state.ticks_until_replan);
    let text: Vec<Line> = lines
        .into_iter()
        .map(|s| Line::from(Span::raw(s)))
        .collect();
    let summary = Paragraph::new(text);
    f.render_widget(summary, inner);
}

fn draw_landmarks_panel(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default().title(" Landmarks ").borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines = format_landmarks_list(&state.landmarks, state.nav_target_index);
    let text: Vec<Line> = lines
        .into_iter()
        .map(|s| Line::from(Span::raw(s)))
        .collect();
    let list = Paragraph::new(text);
    f.render_widget(list, inner);
}

/// Formats the metrics overlay lines for the petri dish panel.
#[must_use]
#[allow(dead_code)] // Used by tests and will be used by dashboard renderer
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn format_metrics_overlay(
    energy: f64,
    mode: AgentMode,
    prediction_error: f64,
    precision: f64,
    speed: f64,
    angle_deg: f64,
    sensor_left: f64,
    sensor_right: f64,
    temporal_gradient: f64,
) -> Vec<String> {
    // Energy bar (10 chars)
    let filled = (energy * 10.0).round() as usize;
    let empty = 10 - filled.min(10);
    let bar: String = "\u{2588}".repeat(filled.min(10)) + &"\u{2591}".repeat(empty);
    let pct = (energy * 100.0).round() as i32;

    let mode_str = match mode {
        AgentMode::Exploring => "EXPLORING",
        AgentMode::Exploiting => "EXPLOITING",
        AgentMode::Panicking => "PANICKING",
        AgentMode::Exhausted => "EXHAUSTED",
        AgentMode::GoalNav => "GOAL-NAV",
    };

    vec![
        format!("E:[{bar}] {pct:>3}%"),
        format!("Mode: {mode_str}"),
        format!("PE:{prediction_error:>6.2}  \u{03C1}:{precision:.2}"),
        format!("v:{speed:>4.1}  \u{03B8}:{angle_deg:>4.0}\u{00B0}"),
        format!("L:{sensor_left:.2}  R:{sensor_right:.2}"),
        format!("\u{2202}t:{temporal_gradient:>6.2}"),
    ]
}

/// ASCII density characters for heat map visualization (low to high).
#[allow(dead_code)] // Used by tests and will be used by dashboard renderer
const DENSITY_CHARS: [char; 9] = [' ', '.', ',', ':', ';', '+', '*', '#', '@'];

/// Converts a mean value (0.0-1.0) to an ASCII density character.
#[allow(dead_code)] // Used by render_spatial_grid_lines
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn mean_to_char(mean: f64) -> char {
    let idx = ((mean.clamp(0.0, 1.0)) * 8.0).round() as usize;
    DENSITY_CHARS[idx.min(8)]
}

/// Compresses spatial grid horizontally by averaging adjacent cells.
/// If `target_width` >= `orig_width`, returns a copy unchanged.
#[must_use]
#[allow(dead_code)] // Will be used when sidebar layout needs compression
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn compress_spatial_grid(
    cells: &[CellPrior],
    orig_width: usize,
    orig_height: usize,
    target_width: usize,
) -> Vec<CellPrior> {
    if target_width >= orig_width {
        return cells.to_vec();
    }

    let mut result = Vec::with_capacity(target_width * orig_height);
    let ratio = orig_width as f64 / target_width as f64;

    for row in 0..orig_height {
        for target_col in 0..target_width {
            let start_col = (target_col as f64 * ratio).floor() as usize;
            let end_col = (((target_col + 1) as f64) * ratio).floor() as usize;
            let end_col = end_col.min(orig_width);

            let mut sum_mean = 0.0;
            let mut count = 0;

            for col in start_col..end_col {
                let idx = row * orig_width + col;
                if let Some(cell) = cells.get(idx) {
                    sum_mean += cell.mean;
                    count += 1;
                }
            }

            let mut compressed = CellPrior::default();
            if count > 0 {
                compressed.mean = sum_mean / f64::from(count);
            }
            result.push(compressed);
        }
    }

    result
}

/// Renders spatial grid as ASCII lines.
/// `agent_cell` is (row, col) of agent's current grid cell, if known.
#[must_use]
#[allow(dead_code)] // Used by tests and will be used by dashboard renderer
pub fn render_spatial_grid_lines(
    cells: &[CellPrior],
    width: usize,
    height: usize,
    agent_cell: Option<(usize, usize)>,
) -> Vec<String> {
    let mut lines = Vec::with_capacity(height);

    for row in 0..height {
        let mut line = String::with_capacity(width);
        for col in 0..width {
            let idx = row * width + col;
            if let Some(cell) = cells.get(idx) {
                if agent_cell == Some((row, col)) {
                    line.push('○');
                } else {
                    line.push(mean_to_char(cell.mean));
                }
            } else {
                line.push(' ');
            }
        }
        lines.push(line);
    }

    lines
}

/// Direction arrow for an action based on base angle.
#[allow(dead_code)] // Used by format_mcts_summary
#[allow(clippy::cast_possible_truncation)]
fn action_to_arrow(action: Action, base_angle: f64) -> &'static str {
    let angle = base_angle + action.angle_delta();
    let octant =
        ((angle + std::f64::consts::PI / 8.0) / (std::f64::consts::PI / 4.0)).floor() as i32;
    match octant.rem_euclid(8) {
        0 | 8.. => "→",
        1 => "↗",
        2 => "↑",
        3 => "↖",
        4 => "←",
        5 => "↙",
        6 => "↓",
        7 => "↘",
        // rem_euclid(8) guarantees 0-7, but match must be exhaustive
        _ => unreachable!(),
    }
}

/// Direction name for an action.
#[allow(dead_code)] // Used by format_mcts_summary
fn action_to_name(action: Action) -> &'static str {
    match action {
        Action::TurnLeft => "L",
        Action::Straight => "S",
        Action::TurnRight => "R",
    }
}

/// Formats MCTS planning summary text.
#[must_use]
#[allow(dead_code)] // Used by tests and will be used by dashboard renderer
pub fn format_mcts_summary(details: &[ActionDetail], ticks_until_replan: u64) -> Vec<String> {
    // Find best action (highest EFE)
    let best = details
        .iter()
        .max_by(|a, b| a.total_efe.total_cmp(&b.total_efe));

    if let Some(best) = best {
        vec![
            format!(
                "Best: {} ({})",
                action_to_arrow(best.action, 0.0),
                action_to_name(best.action)
            ),
            format!("G: {:.2}", best.total_efe),
            format!("├─Prag: {:.2}", best.pragmatic_value),
            format!("└─Epis: {:.2}", best.epistemic_value),
            format!("Rolls: {}", MCTS_ROLLOUTS),
            format!("Depth: {}", MCTS_DEPTH),
            format!("Replan: {}", ticks_until_replan),
        ]
    } else {
        vec!["No plan data".to_string()]
    }
}

/// Formats landmarks as a list table.
#[must_use]
#[allow(dead_code)] // Used by tests and will be used by dashboard renderer
#[allow(clippy::cast_possible_truncation)]
pub fn format_landmarks_list(
    landmarks: &[LandmarkSnapshot],
    nav_target: Option<usize>,
) -> Vec<String> {
    let mut lines = vec![
        " # │ Pos     │Rel │Vis".to_string(),
        "───┼─────────┼────┼───".to_string(),
    ];

    for (i, lm) in landmarks.iter().enumerate() {
        let prefix = if nav_target == Some(i) { "→" } else { " " };
        let reliability = format!("{:>4.2}", lm.reliability.clamp(0.0, 1.0));
        lines.push(format!(
            "{}{} │({:>3},{:>3})│{}│ {}",
            prefix,
            i + 1,
            lm.x as i32,
            lm.y as i32,
            reliability,
            lm.visit_count
        ));
    }

    // Pad with empty slots up to 8
    for i in landmarks.len()..8 {
        lines.push(format!(" {} │   --    │ -- │ -", i + 1));
    }

    lines
}

#[allow(dead_code)] // Legacy single-panel view, kept as fallback
pub fn draw_ui(f: &mut Frame, grid_lines: Vec<String>, hud_info: &str) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // HUD
            Constraint::Min(0),    // Field
        ])
        .split(f.area());

    // HUD
    let hud = Paragraph::new(Span::styled(
        hud_info,
        Style::default().add_modifier(Modifier::REVERSED),
    ));
    f.render_widget(hud, chunks[0]);

    // Field
    // We convert Vec<String> to Vec<Line> for Paragraph
    let text: Vec<Line> = grid_lines
        .into_iter()
        .map(|s| Line::from(Span::raw(s)))
        .collect();

    let field = Paragraph::new(text)
        .block(Block::default().borders(Borders::NONE))
        .style(Style::default().fg(Color::White).bg(Color::Reset));

    f.render_widget(field, chunks[1]);
}

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[must_use]
pub fn world_to_grid_coords(
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    rows: usize,
    cols: usize,
) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let scale_y = height / rows as f64;
    let scale_x = width / cols as f64;

    let r = ((y / scale_y).floor() as usize).min(rows - 1);
    let c = ((x / scale_x).floor() as usize).min(cols - 1);

    (r, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_metrics_panel_renders_without_panic() {
        use crate::simulation::agent::AgentMode;
        use crate::simulation::memory::CellPrior;
        use crate::ui::DashboardState;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(30, 10);
        let mut terminal = Terminal::new(backend).unwrap();

        let state = DashboardState {
            x: 50.0,
            y: 25.0,
            angle: 1.0,
            speed: 0.5,
            energy: 0.8,
            mode: AgentMode::Exploring,
            prediction_error: -0.2,
            precision: 5.0,
            sensor_left: 0.6,
            sensor_right: 0.5,
            temporal_gradient: 0.03,
            spatial_grid: vec![CellPrior::default(); 200],
            grid_width: 20,
            grid_height: 10,
            plan_details: vec![],
            ticks_until_replan: 15,
            landmarks: vec![],
            landmark_count: 0,
            nav_target_index: None,
        };

        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 25, 8);
                draw_metrics_panel(f, area, &state);
            })
            .unwrap();

        // If we get here without panic, the test passes
    }

    #[test]
    fn test_compute_sidebar_layout() {
        use ratatui::layout::Rect;

        let area = Rect::new(0, 0, 100, 40);
        let (main, sidebar) = compute_sidebar_layout(area);

        // Main panel should be ~70% width
        assert!(
            main.width >= 68 && main.width <= 72,
            "main width: {}",
            main.width
        );
        assert_eq!(main.height, 40);
        assert_eq!(main.x, 0);

        // Sidebar should be ~30% width
        assert!(sidebar.len() == 4, "should have 4 sidebar panels");
        assert!(
            sidebar[0].width >= 28 && sidebar[0].width <= 32,
            "sidebar width: {}",
            sidebar[0].width
        );

        // Sidebar panels should stack vertically
        assert_eq!(sidebar[0].y, 0); // Metrics at top
        assert!(sidebar[1].y > sidebar[0].y); // MCTS below Metrics
        assert!(sidebar[2].y > sidebar[1].y); // Landmarks below MCTS
        assert!(sidebar[3].y > sidebar[2].y); // Spatial below Landmarks
    }

    #[test]
    fn test_boundary_coordinates() {
        let width = 100.0;
        let height = 50.0;
        let rows = 10;
        let cols = 20;

        // Case 1: Middle
        let (r, c) = world_to_grid_coords(50.0, 25.0, width, height, rows, cols);
        assert_eq!(r, 5);
        assert_eq!(c, 10);

        // Case 2: Exact boundary (Right/Bottom edge)
        // This is where it fails currently. If x = 100.0, scale_x = 5.0. 100/5 = 20.
        // Valid indices are 0..19. So 20 is out of bounds.
        let (r_edge, c_edge) = world_to_grid_coords(width, height, width, height, rows, cols);
        assert_eq!(
            r_edge,
            rows - 1,
            "Row index should be clamped to max valid index"
        );
        assert_eq!(
            c_edge,
            cols - 1,
            "Col index should be clamped to max valid index"
        );
    }

    #[test]
    fn test_compress_spatial_grid_no_compression_needed() {
        use crate::simulation::memory::CellPrior;

        // 4x2 grid, target width 4 (no compression)
        let cells: Vec<CellPrior> = (0..8)
            .map(|i| {
                let mut c = CellPrior::default();
                c.mean = i as f64 * 0.1;
                c
            })
            .collect();

        let result = compress_spatial_grid(&cells, 4, 2, 4);
        assert_eq!(result.len(), 8);
        assert!((result[0].mean - 0.0).abs() < 0.001);
        assert!((result[3].mean - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_compress_spatial_grid_halves_width() {
        use crate::simulation::memory::CellPrior;

        // 4x2 grid, compress to width 2
        // Row 0: [0.0, 0.2, 0.4, 0.6] -> [0.1, 0.5]
        // Row 1: [0.4, 0.6, 0.8, 1.0] -> [0.5, 0.9]
        let mut cells = Vec::new();
        for row in 0..2 {
            for col in 0..4 {
                let mut c = CellPrior::default();
                c.mean = (row * 4 + col) as f64 * 0.2;
                cells.push(c);
            }
        }

        let result = compress_spatial_grid(&cells, 4, 2, 2);
        assert_eq!(result.len(), 4); // 2x2 grid

        // Check averaged values
        assert!(
            (result[0].mean - 0.1).abs() < 0.001,
            "got {}",
            result[0].mean
        ); // avg(0.0, 0.2)
        assert!(
            (result[1].mean - 0.5).abs() < 0.001,
            "got {}",
            result[1].mean
        ); // avg(0.4, 0.6)
    }

    #[test]
    fn test_spatial_grid_panel_handles_narrow_width() {
        use crate::simulation::agent::AgentMode;
        use crate::simulation::memory::CellPrior;
        use crate::ui::DashboardState;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(15, 15); // Narrow terminal
        let mut terminal = Terminal::new(backend).unwrap();

        let state = DashboardState {
            x: 50.0,
            y: 25.0,
            angle: 1.0,
            speed: 0.5,
            energy: 0.8,
            mode: AgentMode::Exploring,
            prediction_error: -0.2,
            precision: 5.0,
            sensor_left: 0.6,
            sensor_right: 0.5,
            temporal_gradient: 0.03,
            spatial_grid: vec![CellPrior::default(); 200], // 20x10 grid
            grid_width: 20,
            grid_height: 10,
            plan_details: vec![],
            ticks_until_replan: 15,
            landmarks: vec![],
            landmark_count: 0,
            nav_target_index: None,
        };

        // Should not panic even with narrow width
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 15, 12);
                draw_spatial_grid_panel(f, area, &state);
            })
            .unwrap();
    }

    #[test]
    fn test_draw_dashboard_uses_sidebar_layout() {
        use crate::simulation::agent::AgentMode;
        use crate::simulation::memory::CellPrior;
        use crate::ui::DashboardState;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let state = DashboardState {
            x: 50.0,
            y: 25.0,
            angle: 1.0,
            speed: 0.5,
            energy: 0.8,
            mode: AgentMode::Exploring,
            prediction_error: -0.2,
            precision: 5.0,
            sensor_left: 0.6,
            sensor_right: 0.5,
            temporal_gradient: 0.03,
            spatial_grid: vec![CellPrior::default(); 200],
            grid_width: 20,
            grid_height: 10,
            plan_details: vec![],
            ticks_until_replan: 15,
            landmarks: vec![],
            landmark_count: 0,
            nav_target_index: None,
        };

        let grid_lines: Vec<String> = (0..30).map(|_| ".".repeat(60)).collect();

        terminal
            .draw(|f| {
                draw_dashboard(f, grid_lines.clone(), &state);
            })
            .unwrap();

        // Verify buffer has content in expected regions
        let buffer = terminal.backend().buffer();

        // Check "Petri Dish" title is in top-left area
        let petri_title_found =
            (0..20).any(|x| buffer.cell((x, 0)).map(|c| c.symbol()).unwrap_or("") == "P");
        assert!(petri_title_found, "Petri Dish title should be on left side");

        // Check "Agent" title is in right sidebar area (x > 60)
        let agent_title_found =
            (60..100).any(|x| buffer.cell((x, 0)).map(|c| c.symbol()).unwrap_or("") == "A");
        assert!(
            agent_title_found,
            "Agent panel title should be on right side"
        );
    }
}
