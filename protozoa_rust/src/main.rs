#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::collapsible_if)]

mod simulation;
mod ui;

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::simulation::{
    agent::Protozoa,
    environment::PetriDish,
    params::{DISH_HEIGHT, DISH_WIDTH},
};
use crate::ui::{
    DashboardState,
    field::compute_field_grid,
    render::{draw_dashboard, petri_dish_grid_size, world_to_grid_coords},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup Terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Check terminal size
    let size = terminal.size()?;
    if size.width < 80 || size.height < 24 {
        eprintln!(
            "Warning: Terminal size {}x{} is smaller than recommended 80x24. Dashboard may not display correctly.",
            size.width, size.height
        );
    }

    // App State
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);
    let tick_rate = Duration::from_millis(50);

    let res = run_app(&mut terminal, &mut dish, &mut agent, tick_rate);

    // Restore Terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    dish: &mut PetriDish,
    agent: &mut Protozoa,
    tick_rate: Duration,
) -> io::Result<()> {
    let mut last_tick = Instant::now();
    loop {
        // 1. Update
        if last_tick.elapsed() >= tick_rate {
            dish.update();
            agent.sense(dish);
            agent.update_state(dish);
            last_tick = Instant::now();
        }

        // 2. Render
        terminal.draw(|f| {
            let area = f.area();

            // Use the petri dish panel inner size for field computation
            let (field_rows, field_cols) = petri_dish_grid_size(area);

            // Compute background in parallel
            let mut grid = compute_field_grid(dish, field_rows, field_cols);

            // Overlay Agent on field
            if field_rows > 0 && field_cols > 0 {
                let (r, c) = world_to_grid_coords(
                    agent.x,
                    agent.y,
                    dish.width,
                    dish.height,
                    field_rows,
                    field_cols,
                );

                if r < field_rows && c < field_cols {
                    if let Some(line) = grid.get_mut(r) {
                        if c < line.len() {
                            line.replace_range(c..=c, "O");
                        }
                    }
                }
            }

            // Create dashboard state
            let dashboard_state = DashboardState::from_agent(agent, dish);

            // Draw the full dashboard
            draw_dashboard(f, grid, &dashboard_state);
        })?;

        // 3. Input
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    return Ok(());
                }
            }
        }
    }
}
