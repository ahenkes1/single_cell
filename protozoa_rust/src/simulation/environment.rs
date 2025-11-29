use rand::Rng;

#[derive(Debug, Clone)]
pub struct NutrientSource {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub intensity: f64,
    pub decay_rate: f64,
}

impl NutrientSource {
    fn random(width: f64, height: f64) -> Self {
        let mut rng = rand::rng();
        let margin = 10.0;
        Self {
            x: rng.random_range(margin..width - margin),
            y: rng.random_range(margin..height - margin),
            radius: rng.random_range(2.5..8.0),
            intensity: rng.random_range(0.5..1.0),
            decay_rate: rng.random_range(0.990..0.998),
        }
    }
}

pub struct PetriDish {
    pub width: f64,
    pub height: f64,
    pub sources: Vec<NutrientSource>,
}

impl PetriDish {
    #[must_use]
    pub fn new(width: f64, height: f64) -> Self {
        let mut rng = rand::rng();
        let num_sources = rng.random_range(5..=10);
        let sources = (0..num_sources)
            .map(|_| NutrientSource::random(width, height))
            .collect();

        Self {
            width,
            height,
            sources,
        }
    }

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
            let sigma_sq = source.radius.powi(2);
            
            // Gaussian: I * exp(-dist^2 / (2*sigma^2))
            concentration += source.intensity * (-dist_sq / (2.0 * sigma_sq)).exp();
        }

        concentration.clamp(0.0, 1.0)
    }

    pub fn update(&mut self) {
        let brownian_step = 0.5;
        let respawn_threshold = 0.05;
        let mut rng = rand::rng();

        for i in 0..self.sources.len() {
            // Entropy
            self.sources[i].intensity *= self.sources[i].decay_rate;

            // Brownian Motion
            self.sources[i].x += rng.random_range(-brownian_step..brownian_step);
            self.sources[i].y += rng.random_range(-brownian_step..brownian_step);

            // Clamp
            self.sources[i].x = self.sources[i].x.clamp(0.0, self.width);
            self.sources[i].y = self.sources[i].y.clamp(0.0, self.height);

            // Regrowth
            if self.sources[i].intensity < respawn_threshold {
                self.sources[i] = NutrientSource::random(self.width, self.height);
            }
        }
    }
}

