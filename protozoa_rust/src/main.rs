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
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use crate::simulation::{
    agent::Protozoa,
    environment::PetriDish,
    params::{DISH_HEIGHT, DISH_WIDTH, TARGET_CONCENTRATION},
};
use crate::ui::{field::compute_field_grid, render::{draw_ui, world_to_grid_coords}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup Terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

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
            let rows = area.height as usize - 1; // -1 for HUD
            let cols = area.width as usize;

            // Compute background in parallel
            let mut grid = compute_field_grid(dish, rows, cols);

            // Overlay Agent
            if rows > 0 && cols > 0 {
                let scale_y = dish.height / rows as f64;
                let scale_x = dish.width / cols as f64;
                
                let (r, c) = world_to_grid_coords(agent.x, agent.y, dish.width, dish.height, rows, cols);
                
                if r < rows && c < cols {
                     // Ensure we don't panic if row is missing (shouldn't happen)
                     if let Some(line) = grid.get_mut(r) {
                         if c < line.len() {
                             line.replace_range(c..=c, "O");
                             
                             // Sensors (simple visualization)
                             let sensor_r = (agent.angle.sin() * 2.0 / scale_y) as isize + r as isize;
                             let sensor_c = (agent.angle.cos() * 2.0 / scale_x) as isize + c as isize;
                             
                             if sensor_r >= 0 && sensor_r < rows as isize && sensor_c >= 0 && sensor_c < cols as isize {
                                 if let Some(s_line) = grid.get_mut(sensor_r as usize) {
                                     let sc = sensor_c as usize;
                                     if sc < s_line.len() {
                                          s_line.replace_range(sc..=sc, ".");
                                     }
                                 }
                             }
                         }
                     }
                }
            }

            // HUD Info
            let mean_sense = f64::midpoint(agent.val_l, agent.val_r);
            let error = mean_sense - TARGET_CONCENTRATION;
            let hud = format!(
                "Sens: {:.2} | Tgt: {:.2} | Err: {:.2} | Spd: {:.2} | Egy: {:.2}",
                mean_sense, TARGET_CONCENTRATION, error, agent.speed, agent.energy
            );

            draw_ui(f, grid, &hud);
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
