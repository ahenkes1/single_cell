//! Generative model: p(o, s) = p(o|s) × p(s)
//!
//! Defines the agent's internal model of how observations arise from hidden states.

use super::beliefs::BeliefMean;
use crate::simulation::params::{
    INITIAL_SENSORY_PRECISION, NUTRIENT_PRIOR_PRECISION, SENSOR_ANGLE, TARGET_CONCENTRATION,
};

/// The agent's generative model of the world.
///
/// Contains the likelihood p(o|s) and prior p(s) that define the agent's
/// expectations about the environment.
#[derive(Clone, Debug)]
pub struct GenerativeModel {
    /// Prior mean (homeostatic target encodes preferences!)
    pub prior_mean: PriorMean,
    /// Prior precision (inverse covariance) - strength of preferences
    pub prior_precision: PriorPrecision,
    /// Sensory precision (inverse observation noise)
    pub sensory_precision: SensoryPrecision,
}

/// Prior means over hidden states.
///
/// The prior mean for nutrient encodes the agent's *preference* - this is
/// the key insight of Active Inference: preferences are priors.
#[derive(Clone, Copy, Debug)]
pub struct PriorMean {
    /// Target nutrient concentration (preference!)
    pub nutrient: f64,
    /// Prior mean for x position (center of dish)
    pub x: f64,
    /// Prior mean for y position (center of dish)
    pub y: f64,
    /// Prior mean for heading (no preferred direction)
    #[allow(dead_code)] // Reserved for future heading preference
    pub angle: f64,
}

/// Prior precision (inverse variance) for each hidden state.
///
/// Higher precision = stronger preference/belief.
#[derive(Clone, Copy, Debug)]
pub struct PriorPrecision {
    /// How strongly to prefer target nutrient concentration
    pub nutrient: f64,
    /// Precision on x position (weak = explore freely)
    pub x: f64,
    /// Precision on y position (weak = explore freely)
    pub y: f64,
    /// Precision on heading (weak = any direction OK)
    #[allow(dead_code)] // Reserved for future heading precision
    pub angle: f64,
}

/// Sensory precision (inverse observation variance).
///
/// This is the *true* precision in the Active Inference sense:
/// how reliable are the sensors? High precision = trust observations.
#[derive(Clone, Copy, Debug)]
pub struct SensoryPrecision {
    /// Precision of left chemoreceptor
    pub left: f64,
    /// Precision of right chemoreceptor
    pub right: f64,
}

impl Default for GenerativeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerativeModel {
    /// Create a new generative model with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prior_mean: PriorMean {
                nutrient: TARGET_CONCENTRATION, // Preference encoded as prior!
                x: 50.0,                        // Center of dish
                y: 25.0,
                angle: 0.0,
            },
            prior_precision: PriorPrecision {
                nutrient: NUTRIENT_PRIOR_PRECISION, // Strong preference for target
                x: 0.001,                           // Very weak position prior (free to roam)
                y: 0.001,
                angle: 0.001,
            },
            sensory_precision: SensoryPrecision {
                left: INITIAL_SENSORY_PRECISION,
                right: INITIAL_SENSORY_PRECISION,
            },
        }
    }

    /// Observation function: g(s) - predicts observations from hidden states.
    ///
    /// Returns `(predicted_left, predicted_right)` sensor readings.
    #[must_use]
    #[allow(clippy::unused_self)] // Self reserved for future model parameters
    pub fn observation_function(&self, beliefs: &BeliefMean) -> (f64, f64) {
        // Base prediction is believed nutrient concentration
        let base = beliefs.nutrient;

        // Sensor angle offset creates differential between left/right
        // This models how sensors at different angles sample different parts
        // of the gradient field
        let gradient_factor = SENSOR_ANGLE.sin() * 0.2;

        // Left sensor is offset by +SENSOR_ANGLE from heading
        // Right sensor is offset by -SENSOR_ANGLE from heading
        // In a gradient field, this creates a differential
        let predicted_left = base + gradient_factor * beliefs.angle.sin();
        let predicted_right = base - gradient_factor * beliefs.angle.sin();

        (
            predicted_left.clamp(0.0, 1.0),
            predicted_right.clamp(0.0, 1.0),
        )
    }

    /// Jacobian of observation function: ∂g/∂s
    ///
    /// Used for computing gradients of free energy w.r.t. beliefs.
    #[must_use]
    #[allow(clippy::unused_self)] // Self reserved for future model parameters
    pub fn observation_jacobian(&self, beliefs: &BeliefMean) -> ObservationJacobian {
        let gradient_factor = SENSOR_ANGLE.sin() * 0.2;

        ObservationJacobian {
            // ∂g_L/∂nutrient = 1, ∂g_R/∂nutrient = 1
            d_obs_d_nutrient: (1.0, 1.0),
            // ∂g_L/∂angle = gradient_factor × cos(angle)
            // ∂g_R/∂angle = -gradient_factor × cos(angle)
            d_obs_d_angle: (
                gradient_factor * beliefs.angle.cos(),
                -gradient_factor * beliefs.angle.cos(),
            ),
        }
    }

    /// Update sensory precision based on learned estimates.
    pub fn update_sensory_precision(&mut self, left: f64, right: f64) {
        self.sensory_precision.left = left;
        self.sensory_precision.right = right;
    }
}

/// Jacobian of the observation function.
///
/// Contains partial derivatives of each observation w.r.t. each belief.
#[derive(Clone, Copy, Debug)]
pub struct ObservationJacobian {
    /// `(∂g_L/∂nutrient, ∂g_R/∂nutrient)`
    pub d_obs_d_nutrient: (f64, f64),
    /// `(∂g_L/∂angle, ∂g_R/∂angle)`
    pub d_obs_d_angle: (f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generative_model_default() {
        let model = GenerativeModel::new();
        assert!((model.prior_mean.nutrient - TARGET_CONCENTRATION).abs() < 1e-10);
    }

    #[test]
    fn test_observation_function_bounds() {
        let model = GenerativeModel::new();
        let beliefs = BeliefMean {
            nutrient: 0.5,
            x: 50.0,
            y: 25.0,
            angle: 1.0,
        };

        let (pred_l, pred_r) = model.observation_function(&beliefs);

        assert!(pred_l >= 0.0 && pred_l <= 1.0);
        assert!(pred_r >= 0.0 && pred_r <= 1.0);
    }

    #[test]
    fn test_observation_function_symmetric_at_zero_angle() {
        let model = GenerativeModel::new();
        let beliefs = BeliefMean {
            nutrient: 0.5,
            x: 50.0,
            y: 25.0,
            angle: 0.0,
        };

        let (pred_l, pred_r) = model.observation_function(&beliefs);

        // At angle=0, predictions should be equal
        assert!((pred_l - pred_r).abs() < 1e-10);
    }

    #[test]
    fn test_observation_jacobian() {
        let model = GenerativeModel::new();
        let beliefs = BeliefMean {
            nutrient: 0.5,
            x: 50.0,
            y: 25.0,
            angle: 0.5,
        };

        let jacobian = model.observation_jacobian(&beliefs);

        // Nutrient derivatives should be 1.0 (direct mapping)
        assert!((jacobian.d_obs_d_nutrient.0 - 1.0).abs() < 1e-10);
        assert!((jacobian.d_obs_d_nutrient.1 - 1.0).abs() < 1e-10);

        // Angle derivatives should be opposite signs
        assert!(jacobian.d_obs_d_angle.0 * jacobian.d_obs_d_angle.1 <= 0.0);
    }
}
