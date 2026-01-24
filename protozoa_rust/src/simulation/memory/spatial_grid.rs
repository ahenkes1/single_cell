//! Spatial grid for long-term memory of environmental priors.
//!
//! Implements a discretized map of learned nutrient expectations using
//! Welford's online algorithm for numerically stable variance computation.

use crate::simulation::params::{DISH_HEIGHT, DISH_WIDTH};

/// Prior beliefs about nutrient concentration at a grid cell.
///
/// Uses Welford's online algorithm for numerically stable
/// incremental mean and variance computation.
#[derive(Clone, Copy, Debug)]
pub struct CellPrior {
    /// Running mean of observed concentrations
    pub mean: f64,
    /// M2 accumulator for Welford's variance (sum of squared deviations)
    pub m2: f64,
    /// Number of observations at this cell
    pub visits: u32,
}

impl Default for CellPrior {
    fn default() -> Self {
        Self {
            mean: 0.5, // Neutral prior (middle of [0, 1] range)
            m2: 0.0,
            visits: 0,
        }
    }
}

impl CellPrior {
    /// Creates a new cell with default uninformative prior.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the sample variance of observations.
    ///
    /// Returns 1.0 (high uncertainty) if fewer than 2 observations.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.visits < 2 {
            1.0 // High uncertainty with few observations
        } else {
            self.m2 / (f64::from(self.visits) - 1.0)
        }
    }

    /// Returns the precision (inverse variance, scaled by visits).
    ///
    /// Higher precision means more confidence in the mean estimate.
    #[must_use]
    pub fn precision(&self) -> f64 {
        f64::from(self.visits) / (1.0 + self.variance())
    }

    /// Updates the prior with a new observation using Welford's algorithm.
    ///
    /// This provides numerically stable incremental mean/variance updates.
    /// Non-finite observations are ignored to maintain numerical stability.
    pub fn update(&mut self, observed: f64) {
        // Guard against non-finite observations
        if !observed.is_finite() {
            return;
        }

        self.visits = self.visits.saturating_add(1);
        let delta = observed - self.mean;
        self.mean += delta / f64::from(self.visits);
        let delta2 = observed - self.mean;
        self.m2 += delta * delta2;

        // Clamp mean to valid range with some margin for numerical stability
        self.mean = self.mean.clamp(-0.5, 1.5);

        // Ensure m2 doesn't go negative due to floating point errors
        if self.m2 < 0.0 || !self.m2.is_finite() {
            self.m2 = 0.0;
        }

        // Final sanity check on mean
        if !self.mean.is_finite() {
            self.mean = 0.5; // Reset to neutral prior
        }
    }

    /// Returns true if the prior is in a valid numerical state.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.mean.is_finite() && self.m2.is_finite() && self.m2 >= 0.0
    }
}

/// A 2D grid storing learned spatial priors about the environment.
///
/// Each cell tracks the mean and variance of nutrient concentrations
/// observed at that location, enabling precision-weighted prediction.
#[derive(Clone, Debug)]
pub struct SpatialGrid<const W: usize, const H: usize> {
    cells: [[CellPrior; W]; H],
    cell_width: f64,
    cell_height: f64,
    world_width: f64,
    world_height: f64,
}

impl<const W: usize, const H: usize> Default for SpatialGrid<W, H> {
    fn default() -> Self {
        Self::new(DISH_WIDTH, DISH_HEIGHT)
    }
}

impl<const W: usize, const H: usize> SpatialGrid<W, H> {
    /// Creates a new spatial grid covering the given world dimensions.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Grid dimensions are small, precision loss is negligible
    pub fn new(world_width: f64, world_height: f64) -> Self {
        Self {
            cells: [[CellPrior::default(); W]; H],
            cell_width: world_width / W as f64,
            cell_height: world_height / H as f64,
            world_width,
            world_height,
        }
    }

    /// Converts world coordinates to grid indices.
    #[allow(
        clippy::cast_precision_loss,  // Grid dimensions are small
        clippy::cast_possible_truncation,  // Values are clamped to valid range
        clippy::cast_sign_loss  // Values are clamped to non-negative
    )]
    fn world_to_grid(&self, x: f64, y: f64) -> (usize, usize) {
        let col = ((x / self.world_width) * W as f64)
            .floor()
            .clamp(0.0, (W - 1) as f64) as usize;
        let row = ((y / self.world_height) * H as f64)
            .floor()
            .clamp(0.0, (H - 1) as f64) as usize;
        (row, col)
    }

    /// Returns a reference to the cell prior at the given world position.
    #[must_use]
    pub fn get_cell(&self, x: f64, y: f64) -> &CellPrior {
        let (row, col) = self.world_to_grid(x, y);
        &self.cells[row][col]
    }

    /// Returns a mutable reference to the cell prior at the given world position.
    pub fn get_cell_mut(&mut self, x: f64, y: f64) -> &mut CellPrior {
        let (row, col) = self.world_to_grid(x, y);
        &mut self.cells[row][col]
    }

    /// Updates the cell at the given position with a new observation.
    pub fn update(&mut self, x: f64, y: f64, observed: f64) {
        let (row, col) = self.world_to_grid(x, y);
        self.cells[row][col].update(observed);
    }

    /// Returns the precision at the given world position.
    #[must_use]
    pub fn precision(&self, x: f64, y: f64) -> f64 {
        self.get_cell(x, y).precision()
    }

    /// Returns the expected (mean) concentration at the given position.
    #[must_use]
    pub fn expected(&self, x: f64, y: f64) -> f64 {
        self.get_cell(x, y).mean
    }

    /// Returns grid dimensions.
    #[must_use]
    #[allow(clippy::unused_self)] // Self needed for consistent API
    pub const fn dimensions(&self) -> (usize, usize) {
        (W, H)
    }

    /// Returns cell dimensions in world units.
    #[must_use]
    pub fn cell_dimensions(&self) -> (f64, f64) {
        (self.cell_width, self.cell_height)
    }

    /// Returns total number of visits across all cells.
    #[must_use]
    pub fn total_visits(&self) -> u64 {
        self.cells
            .iter()
            .flat_map(|row| row.iter())
            .map(|cell| u64::from(cell.visits))
            .sum()
    }

    /// Resets all cells to default priors.
    pub fn reset(&mut self) {
        for row in &mut self.cells {
            for cell in row {
                *cell = CellPrior::default();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_prior_default() {
        let cell = CellPrior::default();
        assert_eq!(cell.mean, 0.5);
        assert_eq!(cell.visits, 0);
        assert_eq!(cell.variance(), 1.0); // High uncertainty
    }

    #[test]
    fn test_cell_prior_update() {
        let mut cell = CellPrior::new();
        cell.update(0.8);
        assert_eq!(cell.visits, 1);
        assert!((cell.mean - 0.8).abs() < 1e-10);

        cell.update(0.6);
        assert_eq!(cell.visits, 2);
        assert!((cell.mean - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_precision_increases_with_visits() {
        let mut cell = CellPrior::new();
        let initial_precision = cell.precision();

        // Add consistent observations
        for _ in 0..10 {
            cell.update(0.7);
        }

        assert!(cell.precision() > initial_precision);
    }

    #[test]
    fn test_variance_with_consistent_observations() {
        let mut cell = CellPrior::new();
        for _ in 0..5 {
            cell.update(0.5);
        }
        // Variance should be very low with identical observations
        assert!(cell.variance() < 0.01);
    }

    #[test]
    fn test_variance_with_varying_observations() {
        let mut cell = CellPrior::new();
        cell.update(0.0);
        cell.update(1.0);
        cell.update(0.0);
        cell.update(1.0);
        // Variance should be high with alternating observations
        assert!(cell.variance() > 0.3);
    }

    #[test]
    fn test_spatial_grid_coordinates() {
        let grid: SpatialGrid<10, 5> = SpatialGrid::new(100.0, 50.0);

        // Corner cases
        let cell_00 = grid.get_cell(0.0, 0.0);
        let cell_max = grid.get_cell(99.9, 49.9);
        assert!(cell_00.mean == 0.5);
        assert!(cell_max.mean == 0.5);
    }

    #[test]
    fn test_spatial_grid_update() {
        let mut grid: SpatialGrid<10, 5> = SpatialGrid::new(100.0, 50.0);

        grid.update(50.0, 25.0, 0.9);
        let cell = grid.get_cell(50.0, 25.0);
        assert_eq!(cell.visits, 1);
        assert!((cell.mean - 0.9).abs() < 1e-10);

        // Nearby position should be in same cell
        grid.update(51.0, 26.0, 0.7);
        let cell = grid.get_cell(50.0, 25.0);
        assert_eq!(cell.visits, 2);
    }

    #[test]
    fn test_spatial_grid_precision() {
        let mut grid: SpatialGrid<10, 5> = SpatialGrid::new(100.0, 50.0);

        let initial = grid.precision(50.0, 25.0);

        for _ in 0..10 {
            grid.update(50.0, 25.0, 0.8);
        }

        assert!(grid.precision(50.0, 25.0) > initial);
    }

    #[test]
    fn test_total_visits() {
        let mut grid: SpatialGrid<10, 5> = SpatialGrid::new(100.0, 50.0);

        grid.update(10.0, 10.0, 0.5);
        grid.update(50.0, 25.0, 0.5);
        grid.update(90.0, 40.0, 0.5);

        assert_eq!(grid.total_visits(), 3);
    }

    #[test]
    fn test_reset() {
        let mut grid: SpatialGrid<10, 5> = SpatialGrid::new(100.0, 50.0);

        grid.update(50.0, 25.0, 0.9);
        grid.update(50.0, 25.0, 0.8);

        grid.reset();

        assert_eq!(grid.total_visits(), 0);
        assert_eq!(grid.get_cell(50.0, 25.0).mean, 0.5);
    }
}
