//! Online precision estimation from prediction errors.
//!
//! Precision (inverse variance) is learned from the running statistics of
//! prediction errors. This allows the agent to adapt to sensor noise levels.

use crate::simulation::params::{MAX_SENSORY_PRECISION, MIN_SENSORY_PRECISION};

/// Estimates sensory precision from accumulated prediction errors.
///
/// Uses exponential moving average for adaptivity to changing conditions.
#[derive(Clone, Debug)]
pub struct PrecisionEstimator {
    /// Running estimate of error variance (left sensor)
    variance_l: f64,
    /// Running estimate of error variance (right sensor)
    variance_r: f64,
    /// Number of observations processed
    count: u32,
    /// Exponential moving average decay rate
    alpha: f64,
}

impl Default for PrecisionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl PrecisionEstimator {
    /// Create a new precision estimator with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variance_l: 0.1, // Initial moderate uncertainty
            variance_r: 0.1,
            count: 0,
            alpha: 0.05, // Slow adaptation for stability
        }
    }

    /// Update variance estimates with new prediction errors.
    ///
    /// Uses exponential moving average: σ² ← (1-α)σ² + α×e²
    pub fn update(&mut self, error_l: f64, error_r: f64) {
        self.count = self.count.saturating_add(1);

        // Exponential moving average of squared errors
        self.variance_l = (1.0 - self.alpha) * self.variance_l + self.alpha * error_l.powi(2);
        self.variance_r = (1.0 - self.alpha) * self.variance_r + self.alpha * error_r.powi(2);

        // Ensure minimum variance (numerical stability)
        self.variance_l = self.variance_l.max(0.001);
        self.variance_r = self.variance_r.max(0.001);
    }

    /// Get estimated precision for left sensor.
    ///
    /// Precision = 1/variance, clamped to valid range.
    #[must_use]
    pub fn precision_left(&self) -> f64 {
        (1.0 / self.variance_l).clamp(MIN_SENSORY_PRECISION, MAX_SENSORY_PRECISION)
    }

    /// Get estimated precision for right sensor.
    #[must_use]
    pub fn precision_right(&self) -> f64 {
        (1.0 / self.variance_r).clamp(MIN_SENSORY_PRECISION, MAX_SENSORY_PRECISION)
    }

    /// Get the number of observations processed.
    #[must_use]
    #[allow(dead_code)] // Used by tests and future diagnostics
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// Reset the estimator to initial state.
    #[allow(dead_code)] // Used by tests and future episode boundaries
    pub fn reset(&mut self) {
        self.variance_l = 0.1;
        self.variance_r = 0.1;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_estimator_new() {
        let estimator = PrecisionEstimator::new();
        assert_eq!(estimator.count(), 0);

        // Initial precision should be moderate
        let prec = estimator.precision_left();
        assert!(prec > MIN_SENSORY_PRECISION);
        assert!(prec < MAX_SENSORY_PRECISION);
    }

    #[test]
    fn test_precision_increases_with_low_errors() {
        let mut estimator = PrecisionEstimator::new();
        let initial_precision = estimator.precision_left();

        // Feed consistently small errors
        for _ in 0..100 {
            estimator.update(0.01, 0.01);
        }

        let final_precision = estimator.precision_left();
        assert!(
            final_precision > initial_precision,
            "Precision should increase with low errors: {} -> {}",
            initial_precision,
            final_precision
        );
    }

    #[test]
    fn test_precision_decreases_with_high_errors() {
        let mut estimator = PrecisionEstimator::new();

        // Start with low errors to get high precision
        for _ in 0..50 {
            estimator.update(0.01, 0.01);
        }
        let high_precision = estimator.precision_left();

        // Now feed high errors
        for _ in 0..100 {
            estimator.update(0.5, 0.5);
        }
        let low_precision = estimator.precision_left();

        assert!(
            low_precision < high_precision,
            "Precision should decrease with high errors"
        );
    }

    #[test]
    fn test_precision_clamped() {
        let mut estimator = PrecisionEstimator::new();

        // Very small errors (should hit max precision)
        for _ in 0..1000 {
            estimator.update(0.001, 0.001);
        }
        assert!(estimator.precision_left() <= MAX_SENSORY_PRECISION);

        // Very large errors (should hit min precision)
        estimator.reset();
        for _ in 0..1000 {
            estimator.update(10.0, 10.0);
        }
        assert!(estimator.precision_left() >= MIN_SENSORY_PRECISION);
    }

    #[test]
    fn test_reset() {
        let mut estimator = PrecisionEstimator::new();

        for _ in 0..100 {
            estimator.update(0.5, 0.5);
        }
        assert!(estimator.count() > 0);

        estimator.reset();
        assert_eq!(estimator.count(), 0);
    }
}
