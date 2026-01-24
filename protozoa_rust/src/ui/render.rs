use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

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
