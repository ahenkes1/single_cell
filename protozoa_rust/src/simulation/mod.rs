pub mod agent;
pub mod environment;
pub mod inference;
pub mod memory;
pub mod params;
pub mod planning;

#[allow(unused_imports)] // Used by tests and future UI components
pub use agent::AgentMode;
#[allow(unused_imports)] // Used by tests and future dashboard
pub use planning::ActionDetail;

// Re-export inference types for convenience
#[allow(unused_imports)]
pub use inference::{BeliefState, GenerativeModel, PrecisionEstimator};
