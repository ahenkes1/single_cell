use crate::simulation::environment::PetriDish;
use rayon::prelude::*;

const CHARS: [char; 10] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[must_use]
pub fn compute_field_grid(dish: &PetriDish, rows: usize, cols: usize) -> Vec<String> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }

    let scale_y = dish.height / rows as f64;
    let scale_x = dish.width / cols as f64;

    // Use rayon to compute rows in parallel
    (0..rows)
        .into_par_iter()
        .map(|r| {
            let mut line = String::with_capacity(cols);
            for c in 0..cols {
                let world_y = r as f64 * scale_y;
                let world_x = c as f64 * scale_x;

                let val = dish.get_concentration(world_x, world_y);
                
                // Map 0.0..1.0 to index 0..9
                let idx = (val * (CHARS.len() - 1) as f64).round() as usize;
                let idx = idx.min(CHARS.len() - 1); // Safety clamp
                
                line.push(CHARS[idx]);
            }
            line
        })
        .collect()
}
