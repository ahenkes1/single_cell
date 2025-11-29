use crate::simulation::environment::PetriDish;
use crate::simulation::params::{
    LEARNING_RATE, MAX_SPEED, SENSOR_ANGLE, SENSOR_DIST, TARGET_CONCENTRATION,
};
use rand::Rng;
use std::f64::consts::PI;

/// Represents the single-cell organism (Agent) using Active Inference.
///
/// The agent minimizes Variational Free Energy by minimizing the difference (error)
/// between its sensed nutrient concentration and its homeostatic set-point.
#[derive(Debug, Clone)]
pub struct Protozoa {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub speed: f64,
    pub energy: f64,
    pub last_mean_sense: f64,
    pub val_l: f64,
    pub val_r: f64,
}

impl Protozoa {
    /// Creates a new Protozoa agent at the given position.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        let mut rng = rand::rng();
        Self {
            x,
            y,
            angle: rng.random_range(0.0..2.0 * PI),
            speed: 0.0,
            energy: 1.0,
            last_mean_sense: 0.0,
            val_l: 0.0,
            val_r: 0.0,
        }
    }

    /// Updates the agent's sensory inputs based on the current environment.
    ///
    /// Detects concentration at two points (left and right sensors).
    pub fn sense(&mut self, dish: &PetriDish) {
        // Left Sensor
        let theta_l = self.angle + SENSOR_ANGLE;
        let x_l = self.x + SENSOR_DIST * theta_l.cos();
        let y_l = self.y + SENSOR_DIST * theta_l.sin();
        self.val_l = dish.get_concentration(x_l, y_l);

        // Right Sensor
        let theta_r = self.angle - SENSOR_ANGLE;
        let x_r = self.x + SENSOR_DIST * theta_r.cos();
        let y_r = self.y + SENSOR_DIST * theta_r.sin();
        self.val_r = dish.get_concentration(x_r, y_r);
    }

    /// Updates the agent's internal state, heading, speed, and position.
    ///
    /// This implements the Active Inference loop:
    /// 1. Calculates Prediction Error (Sense - Target).
    /// 2. Calculates Spatial and Temporal Gradients.
    /// 3. Updates Heading to minimize error (Gradient Descent on F).
    /// 4. Updates Speed based on "anxiety" (Magnitude of Error).
    /// 5. Applies metabolic costs and intake.
    pub fn update_state(&mut self, dish: &PetriDish) {
        let mut rng = rand::rng();

        // 1. Sensation
        let mean_sense = f64::midpoint(self.val_l, self.val_r);

        // 2. Error (E = mu - rho)
        let error = mean_sense - TARGET_CONCENTRATION;

        // 3. Gradient (G = sL - sR)
        let gradient = self.val_l - self.val_r;

        // 4. Temporal Gradient
        let temp_gradient = mean_sense - self.last_mean_sense;
        self.last_mean_sense = mean_sense;

        // 5. Dynamics
        // Noise proportional to error
        let noise = rng.random_range(-0.5..0.5) * error.abs();

        // Panic Turn
        let mut panic_turn = 0.0;
        if temp_gradient < -0.01 {
            panic_turn = rng.random_range(-2.0..2.0);
        }

        // Heading Update
        let d_theta = (-LEARNING_RATE * error * gradient) + noise + panic_turn;
        self.angle += d_theta;
        self.angle %= 2.0 * PI;

        // Speed Update
        self.speed = MAX_SPEED * error.abs();

        // Metabolism
        // Cost: 0.0005 + (0.0025 * speed_ratio)
        let metabolic_cost = 0.0005 + (0.0025 * (self.speed / MAX_SPEED));
        // Intake: 0.03 * mean_sense
        let intake = 0.03 * mean_sense;

        self.energy = self.energy - metabolic_cost + intake;
        self.energy = self.energy.clamp(0.0, 1.0);

        // Exhaustion check
        if self.energy <= 0.01 {
            self.speed *= 0.5;
        }

        // Position Update
        self.x += self.speed * self.angle.cos();
        self.y += self.speed * self.angle.sin();

        // Boundary Check
        self.x = self.x.clamp(0.0, dish.width);
        self.y = self.y.clamp(0.0, dish.height);
    }
}
