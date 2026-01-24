use crate::simulation::params::{
    BROWNIAN_STEP, RESPAWN_THRESHOLD, SOURCE_COUNT_MAX, SOURCE_COUNT_MIN, SOURCE_DECAY_MAX,
    SOURCE_DECAY_MIN, SOURCE_INTENSITY_MAX, SOURCE_INTENSITY_MIN, SOURCE_MARGIN, SOURCE_RADIUS_MAX,
    SOURCE_RADIUS_MIN,
};
use rand::Rng;

/// Represents a single Gaussian source of nutrients in the petri dish.
///
/// The source has a position, radius (spread), and intensity (concentration).
/// It decays over time and moves slightly via Brownian motion.
#[derive(Debug, Clone)]
pub struct NutrientSource {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub intensity: f64,
    pub decay_rate: f64,
}

impl NutrientSource {
    /// Creates a new random nutrient source within the given bounds.
    fn random(width: f64, height: f64) -> Self {
        let mut rng = rand::rng();
        Self {
            x: rng.random_range(SOURCE_MARGIN..width - SOURCE_MARGIN),
            y: rng.random_range(SOURCE_MARGIN..height - SOURCE_MARGIN),
            radius: rng.random_range(SOURCE_RADIUS_MIN..SOURCE_RADIUS_MAX),
            intensity: rng.random_range(SOURCE_INTENSITY_MIN..SOURCE_INTENSITY_MAX),
            decay_rate: rng.random_range(SOURCE_DECAY_MIN..SOURCE_DECAY_MAX),
        }
    }
}

/// Represents the simulation environment (the "dish").
///
/// Contains multiple `NutrientSource`s and handles their dynamics (decay, movement, respawn).
/// It calculates the aggregate nutrient concentration at any point.
pub struct PetriDish {
    pub width: f64,
    pub height: f64,
    pub sources: Vec<NutrientSource>,
}

impl PetriDish {
    /// Creates a new Petri dish with the specified dimensions and random nutrient sources.
    #[must_use]
    pub fn new(width: f64, height: f64) -> Self {
        let mut rng = rand::rng();
        let num_sources = rng.random_range(SOURCE_COUNT_MIN..=SOURCE_COUNT_MAX);
        let sources = (0..num_sources)
            .map(|_| NutrientSource::random(width, height))
            .collect();

        Self {
            width,
            height,
            sources,
        }
    }

    /// Calculates the nutrient concentration at a specific coordinate (x, y).
    ///
    /// Returns the sum of Gaussian contributions from all sources.
    /// If the coordinate is outside the bounds, returns -1.0 (Toxic Void).
    #[must_use]
    pub fn get_concentration(&self, x: f64, y: f64) -> f64 {
        if x < 0.0 || x > self.width || y < 0.0 || y > self.height {
            return -1.0;
        }

        let mut concentration = 0.0;
        for source in &self.sources {
            let d_x = x - source.x;
            let d_y = y - source.y;
            let dist_sq = d_x.powi(2) + d_y.powi(2);
            let sigma_sq = source.radius.powi(2).max(f64::EPSILON);

            // Gaussian: I * exp(-dist^2 / (2*sigma^2))
            concentration += source.intensity * (-dist_sq / (2.0 * sigma_sq)).exp();
        }

        concentration.clamp(0.0, 1.0)
    }

    /// Updates the state of the environment (nutrient decay, brownian motion, regrowth).
    pub fn update(&mut self) {
        let mut rng = rand::rng();

        for i in 0..self.sources.len() {
            // Entropy
            self.sources[i].intensity *= self.sources[i].decay_rate;

            // Brownian Motion
            self.sources[i].x += rng.random_range(-BROWNIAN_STEP..BROWNIAN_STEP);
            self.sources[i].y += rng.random_range(-BROWNIAN_STEP..BROWNIAN_STEP);

            // Clamp
            self.sources[i].x = self.sources[i].x.clamp(0.0, self.width);
            self.sources[i].y = self.sources[i].y.clamp(0.0, self.height);

            // Regrowth
            if self.sources[i].intensity < RESPAWN_THRESHOLD {
                self.sources[i] = NutrientSource::random(self.width, self.height);
            }
        }
    }
}
