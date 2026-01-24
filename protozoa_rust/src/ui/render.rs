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

/// Computes the four quadrant areas for the dashboard layout.
#[must_use]
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

/// Draws the full cognitive dashboard.
pub fn draw_dashboard(f: &mut Frame, grid_lines: Vec<String>, state: &DashboardState) {
    let quadrants = compute_quadrant_layout(f.area());

    // === Top-Left: Petri Dish with Overlay ===
    draw_petri_dish_panel(f, quadrants[0], grid_lines, state);

    // === Top-Right: Spatial Memory Grid ===
    draw_spatial_grid_panel(f, quadrants[1], state);

    // === Bottom-Left: MCTS Planning ===
    draw_mcts_panel(f, quadrants[2], state);

    // === Bottom-Right: Landmarks ===
    draw_landmarks_panel(f, quadrants[3], state);
}

#[allow(clippy::cast_possible_truncation)]
fn draw_petri_dish_panel(
    f: &mut Frame,
    area: Rect,
    grid_lines: Vec<String>,
    state: &DashboardState,
) {
    let block = Block::default().title(" Petri Dish ").borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Render field
    let text: Vec<Line> = grid_lines
        .into_iter()
        .map(|s| Line::from(Span::raw(s)))
        .collect();
    let field = Paragraph::new(text);
    f.render_widget(field, inner);

    // Metrics overlay (bottom-left of inner area)
    let angle_deg = state.angle.to_degrees();
    let overlay_lines = format_metrics_overlay(
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

    let overlay_height = overlay_lines.len() as u16 + 2;
    let overlay_width = 23;
    if inner.height > overlay_height && inner.width > overlay_width {
        let overlay_area = Rect::new(
            inner.x,
            inner.y + inner.height - overlay_height,
            overlay_width,
            overlay_height,
        );
        let overlay_text: Vec<Line> = overlay_lines
            .into_iter()
            .map(|s| {
                Line::from(Span::styled(
                    s,
                    Style::default().add_modifier(Modifier::BOLD),
                ))
            })
            .collect();
        let overlay = Paragraph::new(overlay_text)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().bg(Color::Black));
        f.render_widget(overlay, overlay_area);
    }
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

    // Calculate agent's grid cell
    let agent_col = ((state.x / 100.0) * state.grid_width as f64).floor() as usize;
    let agent_row = ((state.y / 50.0) * state.grid_height as f64).floor() as usize;
    let agent_cell = Some((
        agent_row.min(state.grid_height.saturating_sub(1)),
        agent_col.min(state.grid_width.saturating_sub(1)),
    ));

    let lines = render_spatial_grid_lines(
        &state.spatial_grid,
        state.grid_width,
        state.grid_height,
        agent_cell,
    );
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
    let octant = ((angle + std::f64::consts::PI / 8.0) / (std::f64::consts::PI / 4.0)) as i32 % 8;
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
        lines.push(format!(
            "{}{} │({:>3},{:>3})│.{:02}│ {}",
            prefix,
            i + 1,
            lm.x as i32,
            lm.y as i32,
            (lm.reliability * 100.0) as i32 % 100,
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
}
