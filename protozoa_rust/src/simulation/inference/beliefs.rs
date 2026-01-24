//! Gaussian belief state over hidden variables.
//!
//! Represents the agent's approximate posterior q(s) = N(μ, Σ) over hidden states.

use std::f64::consts::PI;

/// Represents Gaussian beliefs: q(s) = N(μ, Σ)
#[derive(Clone, Debug)]
pub struct BeliefState {
    /// Posterior mean (believed state)
    pub mean: BeliefMean,
    /// Posterior covariance (uncertainty)
    pub covariance: BeliefCovariance,
}

/// Mean of beliefs over hidden states.
#[derive(Clone, Copy, Debug)]
pub struct BeliefMean {
    /// Believed nutrient concentration at current location
    pub nutrient: f64,
    /// Believed x position
    pub x: f64,
    /// Believed y position
    pub y: f64,
    /// Believed heading angle (radians)
    pub angle: f64,
}

/// Diagonal covariance matrix (assumes independence for computational efficiency).
#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_field_names)]
pub struct BeliefCovariance {
    /// Variance in nutrient belief
    pub nutrient_var: f64,
    /// Variance in x position belief
    pub x_var: f64,
    /// Variance in y position belief
    pub y_var: f64,
    /// Variance in angle belief
    pub angle_var: f64,
}

impl Default for BeliefCovariance {
    fn default() -> Self {
        Self {
            nutrient_var: 0.25, // High initial uncertainty
            x_var: 1.0,
            y_var: 1.0,
            angle_var: 0.5,
        }
    }
}

impl BeliefState {
    /// Create initial beliefs with specified position and high uncertainty.
    #[must_use]
    pub fn new(x: f64, y: f64, angle: f64) -> Self {
        Self {
            mean: BeliefMean {
                nutrient: 0.5, // Neutral prior
                x,
                y,
                angle,
            },
            covariance: BeliefCovariance::default(),
        }
    }

    /// Update beliefs via gradient descent on VFE.
    ///
    /// `μ ← μ + learning_rate × gradient`
    pub fn update(&mut self, gradient: &BeliefMean, learning_rate: f64) {
        self.mean.nutrient += learning_rate * gradient.nutrient;
        self.mean.x += learning_rate * gradient.x;
        self.mean.y += learning_rate * gradient.y;
        self.mean.angle += learning_rate * gradient.angle;

        // Clamp to valid ranges
        self.mean.nutrient = self.mean.nutrient.clamp(0.0, 1.0);
        self.mean.angle = self.mean.angle.rem_euclid(2.0 * PI);
    }

    /// Synchronize position beliefs with actual position (proprioception).
    ///
    /// Position is directly observable, so beliefs should track actual position.
    pub fn sync_position(&mut self, x: f64, y: f64, angle: f64) {
        self.mean.x = x;
        self.mean.y = y;
        self.mean.angle = angle;
        // Reduce position uncertainty after proprioceptive update
        self.covariance.x_var = 0.01;
        self.covariance.y_var = 0.01;
        self.covariance.angle_var = 0.01;
    }

    /// Total uncertainty (trace of covariance matrix).
    #[must_use]
    pub fn total_uncertainty(&self) -> f64 {
        self.covariance.nutrient_var
            + self.covariance.x_var
            + self.covariance.y_var
            + self.covariance.angle_var
    }

    /// Log determinant of covariance (for entropy computation).
    ///
    /// For diagonal covariance: ln|Σ| = Σ ln(σᵢ²)
    #[must_use]
    pub fn log_det_covariance(&self) -> f64 {
        // Guard against zero/negative variances
        let safe_nutrient = self.covariance.nutrient_var.max(1e-10);
        let safe_x = self.covariance.x_var.max(1e-10);
        let safe_y = self.covariance.y_var.max(1e-10);
        let safe_angle = self.covariance.angle_var.max(1e-10);

        safe_nutrient.ln() + safe_x.ln() + safe_y.ln() + safe_angle.ln()
    }

    /// Increase uncertainty (used for prediction into the future).
    pub fn increase_uncertainty(&mut self, factor: f64) {
        self.covariance.nutrient_var *= factor;
        self.covariance.x_var *= factor;
        self.covariance.y_var *= factor;
        self.covariance.angle_var *= factor;

        // Cap maximum uncertainty
        self.covariance.nutrient_var = self.covariance.nutrient_var.min(1.0);
        self.covariance.x_var = self.covariance.x_var.min(10.0);
        self.covariance.y_var = self.covariance.y_var.min(10.0);
        self.covariance.angle_var = self.covariance.angle_var.min(1.0);
    }

    /// Decrease uncertainty after observation (used after belief update).
    pub fn decrease_uncertainty(&mut self, factor: f64) {
        self.covariance.nutrient_var *= factor;
        // Keep minimum uncertainty
        self.covariance.nutrient_var = self.covariance.nutrient_var.max(0.001);
    }
}

impl BeliefMean {
    /// Create a zero gradient (no change to beliefs).
    #[must_use]
    #[allow(dead_code)] // Reserved for future use in belief initialization
    pub const fn zero() -> Self {
        Self {
            nutrient: 0.0,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_state_new() {
        let beliefs = BeliefState::new(50.0, 25.0, 1.0);
        assert!((beliefs.mean.x - 50.0).abs() < 1e-10);
        assert!((beliefs.mean.y - 25.0).abs() < 1e-10);
        assert!((beliefs.mean.nutrient - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_belief_update_clamps_nutrient() {
        let mut beliefs = BeliefState::new(50.0, 25.0, 0.0);

        // Try to push nutrient above 1.0
        let gradient = BeliefMean {
            nutrient: 100.0,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        };
        beliefs.update(&gradient, 0.1);

        assert!(beliefs.mean.nutrient <= 1.0);
    }

    #[test]
    fn test_belief_update_wraps_angle() {
        let mut beliefs = BeliefState::new(50.0, 25.0, 0.0);

        // Push angle beyond 2π
        let gradient = BeliefMean {
            nutrient: 0.0,
            x: 0.0,
            y: 0.0,
            angle: 100.0,
        };
        beliefs.update(&gradient, 0.1);

        assert!(beliefs.mean.angle >= 0.0);
        assert!(beliefs.mean.angle < 2.0 * PI);
    }

    #[test]
    fn test_log_det_covariance() {
        let beliefs = BeliefState::new(50.0, 25.0, 0.0);
        let log_det = beliefs.log_det_covariance();
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_total_uncertainty() {
        let beliefs = BeliefState::new(50.0, 25.0, 0.0);
        let uncertainty = beliefs.total_uncertainty();
        assert!(uncertainty > 0.0);
    }
}
