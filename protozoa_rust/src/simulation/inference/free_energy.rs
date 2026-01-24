//! Variational and Expected Free Energy computation.
//!
//! # Variational Free Energy (VFE)
//!
//! For perception - measures surprise given current beliefs:
//! ```text
//! F = (1/2)(o - g(μ))ᵀ Πₒ (o - g(μ)) + (1/2)(μ - η)ᵀ Πη (μ - η)
//! ```
//!
//! # Expected Free Energy (EFE)
//!
//! For planning - measures expected surprise under a policy:
//! ```text
//! G(π) = Risk + Ambiguity - Epistemic
//! ```

use super::beliefs::{BeliefMean, BeliefState};
use super::generative_model::GenerativeModel;

/// Compute Variational Free Energy (for perception).
///
/// F = Sensory prediction error + Prior prediction error
///
/// Lower VFE means beliefs are more consistent with observations and priors.
#[must_use]
pub fn variational_free_energy(
    observations: (f64, f64),
    beliefs: &BeliefState,
    model: &GenerativeModel,
) -> f64 {
    let (obs_l, obs_r) = observations;
    let (pred_l, pred_r) = model.observation_function(&beliefs.mean);

    // Sensory prediction error (precision-weighted squared error)
    // (1/2) × Πₒ × (o - g(μ))²
    let sensory_error_l = obs_l - pred_l;
    let sensory_error_r = obs_r - pred_r;
    let sensory_term = 0.5 * model.sensory_precision.left * sensory_error_l.powi(2)
        + 0.5 * model.sensory_precision.right * sensory_error_r.powi(2);

    // Prior prediction error (precision-weighted squared error)
    // (1/2) × Πη × (μ - η)²
    let prior_error_nutrient = beliefs.mean.nutrient - model.prior_mean.nutrient;
    let prior_term = 0.5 * model.prior_precision.nutrient * prior_error_nutrient.powi(2);

    // Position prior errors (weak priors, so small contribution)
    let prior_x = 0.5 * model.prior_precision.x * (beliefs.mean.x - model.prior_mean.x).powi(2);
    let prior_y = 0.5 * model.prior_precision.y * (beliefs.mean.y - model.prior_mean.y).powi(2);

    sensory_term + prior_term + prior_x + prior_y
}

/// Compute gradient of VFE w.r.t. beliefs (for belief update).
///
/// ∂F/∂μ = -Πₒ × ∂g/∂μ × (o - g(μ)) + Πη × (μ - η)
///
/// The negative gradient points toward lower free energy.
#[must_use]
pub fn vfe_gradient(
    observations: (f64, f64),
    beliefs: &BeliefState,
    model: &GenerativeModel,
) -> BeliefMean {
    let (obs_l, obs_r) = observations;
    let (pred_l, pred_r) = model.observation_function(&beliefs.mean);
    let jacobian = model.observation_jacobian(&beliefs.mean);

    // Sensory prediction errors
    let error_l = obs_l - pred_l;
    let error_r = obs_r - pred_r;

    // Gradient w.r.t. nutrient belief
    // ∂F/∂μ_nutrient = -Πₒ,L × (∂g_L/∂μ_nutrient) × error_L
    //                 - Πₒ,R × (∂g_R/∂μ_nutrient) × error_R
    //                 + Πη,nutrient × (μ_nutrient - η_nutrient)
    let d_nutrient_sensory = model.sensory_precision.left * jacobian.d_obs_d_nutrient.0 * error_l
        + model.sensory_precision.right * jacobian.d_obs_d_nutrient.1 * error_r;
    let d_nutrient_prior =
        model.prior_precision.nutrient * (beliefs.mean.nutrient - model.prior_mean.nutrient);

    // Gradient w.r.t. angle belief
    let d_angle_sensory = model.sensory_precision.left * jacobian.d_obs_d_angle.0 * error_l
        + model.sensory_precision.right * jacobian.d_obs_d_angle.1 * error_r;

    // Return negative gradient (descent direction)
    BeliefMean {
        nutrient: d_nutrient_sensory - d_nutrient_prior,
        x: 0.0, // Position updated from proprioception
        y: 0.0,
        angle: d_angle_sensory,
    }
}

/// Compute Expected Free Energy for a predicted future state (for planning).
///
/// G(π) = Risk + Ambiguity - Epistemic Value
///
/// - Risk: KL divergence from preferred states (encoded in prior)
/// - Ambiguity: Expected sensory uncertainty
/// - Epistemic: Preference for reducing uncertainty
///
/// Lower EFE is better (we minimize EFE for action selection).
#[must_use]
pub fn expected_free_energy(predicted_beliefs: &BeliefState, model: &GenerativeModel) -> f64 {
    // Risk: squared distance from preferred nutrient (scaled by prior precision)
    // This encodes "pragmatic value" - prefer states where I expect to be satisfied
    let risk = 0.5
        * model.prior_precision.nutrient
        * (predicted_beliefs.mean.nutrient - model.prior_mean.nutrient).powi(2);

    // Ambiguity: expected sensory prediction error variance
    // Higher nutrient variance → more uncertain about what I'll observe
    let ambiguity = 0.5
        * (model.sensory_precision.left + model.sensory_precision.right)
        * predicted_beliefs.covariance.nutrient_var;

    // Epistemic value: negative entropy of beliefs (prefer certainty)
    // -H[q(s)] ∝ -(1/2)ln|Σ|
    // We want to REDUCE uncertainty, so high uncertainty = high EFE (bad)
    let epistemic = -0.5 * predicted_beliefs.log_det_covariance();

    // Epistemic value is negative (we want to reduce uncertainty)
    // so we ADD it to make high uncertainty costly
    risk + ambiguity - epistemic
}

/// Compute prediction errors for precision learning.
///
/// Returns `(error_left, error_right)`.
#[must_use]
pub fn prediction_errors(
    observations: (f64, f64),
    beliefs: &BeliefState,
    model: &GenerativeModel,
) -> (f64, f64) {
    let (obs_l, obs_r) = observations;
    let (pred_l, pred_r) = model.observation_function(&beliefs.mean);
    (obs_l - pred_l, obs_r - pred_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::params::TARGET_CONCENTRATION;

    #[test]
    fn test_vfe_minimized_at_correct_beliefs() {
        let model = GenerativeModel::new();

        // Beliefs matching observations and prior should have low VFE
        let mut good_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        good_beliefs.mean.nutrient = TARGET_CONCENTRATION;

        // Observations matching beliefs
        let good_obs = (TARGET_CONCENTRATION, TARGET_CONCENTRATION);
        let vfe_good = variational_free_energy(good_obs, &good_beliefs, &model);

        // Beliefs NOT matching observations
        let mut bad_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        bad_beliefs.mean.nutrient = 0.1; // Believes low nutrient

        // But observes high nutrient
        let bad_obs = (0.9, 0.9);
        let vfe_bad = variational_free_energy(bad_obs, &bad_beliefs, &model);

        assert!(
            vfe_good < vfe_bad,
            "VFE should be lower when beliefs match observations"
        );
    }

    #[test]
    fn test_gradient_descent_reduces_vfe() {
        let model = GenerativeModel::new();
        let mut beliefs = BeliefState::new(50.0, 25.0, 0.0);
        beliefs.mean.nutrient = 0.3; // Start far from actual

        let observations = (0.7, 0.7); // Observe high nutrients

        let initial_vfe = variational_free_energy(observations, &beliefs, &model);

        // Take gradient step
        let gradient = vfe_gradient(observations, &beliefs, &model);
        beliefs.update(&gradient, 0.1);

        let final_vfe = variational_free_energy(observations, &beliefs, &model);

        assert!(
            final_vfe < initial_vfe,
            "VFE should decrease after gradient step: {} -> {}",
            initial_vfe,
            final_vfe
        );
    }

    #[test]
    fn test_efe_prefers_preferred_states() {
        let model = GenerativeModel::new();

        // Beliefs at preferred state (target concentration)
        let mut good_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        good_beliefs.mean.nutrient = TARGET_CONCENTRATION;

        // Beliefs far from preferred state
        let mut bad_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        bad_beliefs.mean.nutrient = 0.1;

        let efe_good = expected_free_energy(&good_beliefs, &model);
        let efe_bad = expected_free_energy(&bad_beliefs, &model);

        assert!(
            efe_good < efe_bad,
            "EFE should prefer states near prior mean (preference)"
        );
    }

    #[test]
    fn test_efe_penalizes_uncertainty() {
        let model = GenerativeModel::new();

        // Low uncertainty beliefs
        let mut certain_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        certain_beliefs.mean.nutrient = 0.5;
        certain_beliefs.covariance.nutrient_var = 0.01;

        // High uncertainty beliefs (same mean)
        let mut uncertain_beliefs = BeliefState::new(50.0, 25.0, 0.0);
        uncertain_beliefs.mean.nutrient = 0.5;
        uncertain_beliefs.covariance.nutrient_var = 0.5;

        let efe_certain = expected_free_energy(&certain_beliefs, &model);
        let efe_uncertain = expected_free_energy(&uncertain_beliefs, &model);

        assert!(
            efe_certain < efe_uncertain,
            "EFE should penalize uncertainty (epistemic drive)"
        );
    }

    #[test]
    fn test_prediction_errors() {
        let model = GenerativeModel::new();
        let mut beliefs = BeliefState::new(50.0, 25.0, 0.0);
        beliefs.mean.nutrient = 0.5;

        let observations = (0.7, 0.3);
        let (err_l, err_r) = prediction_errors(observations, &beliefs, &model);

        // Errors should be non-zero for mismatched beliefs
        assert!(err_l.abs() > 0.0 || err_r.abs() > 0.0);
    }
}
