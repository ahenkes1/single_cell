use crate::simulation::environment::PetriDish;
use crate::simulation::params::{
    BASE_METABOLIC_COST, EXHAUSTION_SPEED_FACTOR, EXHAUSTION_THRESHOLD, INTAKE_RATE, LEARNING_RATE,
    MAX_SPEED, NOISE_SCALE, PANIC_THRESHOLD, PANIC_TURN_RANGE, SENSOR_ANGLE, SENSOR_DIST,
    SPEED_METABOLIC_COST, TARGET_CONCENTRATION,
};
use rand::Rng;
use std::f64::consts::PI;

/// Validates that a value is finite (not NaN or infinite).
/// Returns a safe fallback (0.0) in release mode if the value is non-finite.
#[inline]
fn assert_finite(value: f64, context: &str) -> f64 {
    debug_assert!(value.is_finite(), "Non-finite value in {context}: {value}");
    if value.is_finite() { value } else { 0.0 }
}

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
        let mean_sense = assert_finite(f64::midpoint(self.val_l, self.val_r), "mean_sense");

        // 2. Error (E = mu - rho)
        let error = assert_finite(mean_sense - TARGET_CONCENTRATION, "error");

        // 3. Gradient (G = sL - sR)
        let gradient = assert_finite(self.val_l - self.val_r, "gradient");

        // 4. Temporal Gradient
        let temp_gradient = mean_sense - self.last_mean_sense;
        self.last_mean_sense = mean_sense;

        // 5. Dynamics
        // Noise proportional to error
        let noise = rng.random_range(-NOISE_SCALE..NOISE_SCALE) * error.abs();

        // Panic Turn
        let mut panic_turn = 0.0;
        if temp_gradient < PANIC_THRESHOLD {
            panic_turn = rng.random_range(-PANIC_TURN_RANGE..PANIC_TURN_RANGE);
        }

        // Heading Update
        let d_theta = assert_finite(
            (-LEARNING_RATE * error * gradient) + noise + panic_turn,
            "d_theta",
        );
        self.angle += d_theta;
        self.angle = self.angle.rem_euclid(2.0 * PI);

        // Speed Update
        self.speed = MAX_SPEED * error.abs();

        // Metabolism
        let metabolic_cost =
            BASE_METABOLIC_COST + (SPEED_METABOLIC_COST * (self.speed / MAX_SPEED));
        let intake = INTAKE_RATE * mean_sense;

        self.energy = assert_finite(self.energy - metabolic_cost + intake, "energy");
        self.energy = self.energy.clamp(0.0, 1.0);

        // Exhaustion check
        if self.energy <= EXHAUSTION_THRESHOLD {
            self.speed *= EXHAUSTION_SPEED_FACTOR;
        }

        // Position Update
        self.x += self.speed * self.angle.cos();
        self.y += self.speed * self.angle.sin();

        // Boundary Check
        self.x = self.x.clamp(0.0, dish.width);
        self.y = self.y.clamp(0.0, dish.height);
    }
}
