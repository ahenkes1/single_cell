use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Span, Line},
    widgets::{Block, Borders, Paragraph},
    Frame,
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
        .style(Style::default().fg(Color::White).bg(Color::Black));
    
    f.render_widget(field, chunks[1]);
}
