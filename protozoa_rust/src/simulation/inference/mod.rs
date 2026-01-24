//! Continuous Active Inference engine.
//!
//! Implements variational inference with Gaussian beliefs for perception
//! and action under the Free Energy Principle.
//!
//! # Mathematical Foundation
//!
//! Variational Free Energy:
//! ```text
//! F = (1/2)(o - g(μ))ᵀ Πₒ (o - g(μ)) + (1/2)(μ - η)ᵀ Πη (μ - η)
//! ```
//!
//! Belief update via gradient descent:
//! ```text
//! dμ/dt = -∂F/∂μ
//! ```
//!
//! Expected Free Energy for planning:
//! ```text
//! G(π) = Risk + Ambiguity - Epistemic
//! ```

mod beliefs;
mod free_energy;
mod generative_model;
mod precision;

#[allow(unused_imports)] // Types exported for future use and API completeness
pub use beliefs::{BeliefCovariance, BeliefMean, BeliefState};
pub use free_energy::{
    expected_free_energy, prediction_errors, variational_free_energy, vfe_gradient,
};
#[allow(unused_imports)] // Types exported for future use and API completeness
pub use generative_model::{GenerativeModel, ObservationJacobian, PriorMean, SensoryPrecision};
pub use precision::PrecisionEstimator;
