//! Simulation hyperparameters.

// Allow unused constants - these will be used in future tasks (MCTS, goal-directed navigation)
#![allow(dead_code)]

// === Agent Sensing Parameters ===
pub const TARGET_CONCENTRATION: f64 = 0.8;
pub const SENSOR_DIST: f64 = 2.0;
/// Sensor stereo spread in radians (~28.6 degrees)
pub const SENSOR_ANGLE: f64 = 0.5;
pub const LEARNING_RATE: f64 = 0.15;
pub const MAX_SPEED: f64 = 1.5;

// === Agent Behavior Parameters ===
/// Temporal gradient threshold below which a panic turn is triggered
pub const PANIC_THRESHOLD: f64 = -0.01;
/// Maximum panic turn magnitude in radians (~115 degrees each direction)
pub const PANIC_TURN_RANGE: f64 = 2.0;
/// Scale factor for random noise on heading updates
pub const NOISE_SCALE: f64 = 0.5;
/// Energy level at or below which the agent enters exhaustion state
pub const EXHAUSTION_THRESHOLD: f64 = 0.01;
/// Speed multiplier applied when agent is exhausted
pub const EXHAUSTION_SPEED_FACTOR: f64 = 0.5;

// === Agent Metabolism Parameters ===
/// Base metabolic energy cost per tick (independent of movement)
pub const BASE_METABOLIC_COST: f64 = 0.0005;
/// Additional metabolic cost per unit of normalized speed
pub const SPEED_METABOLIC_COST: f64 = 0.0025;
/// Energy intake rate per unit of sensed concentration
pub const INTAKE_RATE: f64 = 0.03;

// === Environment Parameters ===
pub const DISH_WIDTH: f64 = 100.0;
/// Adjusted for terminal aspect ratio
pub const DISH_HEIGHT: f64 = 50.0;
/// Margin from dish edges for source placement
pub const SOURCE_MARGIN: f64 = 10.0;
/// Minimum radius for nutrient sources
pub const SOURCE_RADIUS_MIN: f64 = 2.5;
/// Maximum radius for nutrient sources
pub const SOURCE_RADIUS_MAX: f64 = 8.0;
/// Minimum initial intensity for nutrient sources
pub const SOURCE_INTENSITY_MIN: f64 = 0.5;
/// Maximum initial intensity for nutrient sources
pub const SOURCE_INTENSITY_MAX: f64 = 1.0;
/// Minimum decay rate for nutrient sources (per tick multiplier)
pub const SOURCE_DECAY_MIN: f64 = 0.990;
/// Maximum decay rate for nutrient sources (per tick multiplier)
pub const SOURCE_DECAY_MAX: f64 = 0.998;
/// Brownian motion step size for source drift
pub const BROWNIAN_STEP: f64 = 0.5;
/// Intensity threshold below which a source respawns
pub const RESPAWN_THRESHOLD: f64 = 0.05;
/// Minimum number of nutrient sources in dish
pub const SOURCE_COUNT_MIN: usize = 5;
/// Maximum number of nutrient sources in dish
pub const SOURCE_COUNT_MAX: usize = 10;

// === Memory Parameters ===
/// Size of sensor history ring buffer
pub const HISTORY_SIZE: usize = 32;
/// Width of spatial prior grid (cells)
pub const GRID_WIDTH: usize = 20;
/// Height of spatial prior grid (cells)
pub const GRID_HEIGHT: usize = 10;

// === Learning Parameters ===
/// Learning rate for spatial prior updates (Hebbian-like)
pub const PRIOR_LEARNING_RATE: f64 = 0.1;
/// Scale factor for exploration bonus in uncertain regions
pub const EXPLORATION_SCALE: f64 = 0.3;
/// Minimum precision value (prevents division by zero)
pub const MIN_PRECISION: f64 = 0.1;
/// Maximum precision value (prevents over-confidence)
pub const MAX_PRECISION: f64 = 10.0;

// === Episodic Memory Parameters ===
/// Maximum number of landmarks to remember
pub const MAX_LANDMARKS: usize = 8;
/// Minimum nutrient concentration to trigger landmark storage
pub const LANDMARK_THRESHOLD: f64 = 0.7;
/// Reliability decay rate per tick (when not visited)
pub const LANDMARK_DECAY: f64 = 0.995;
/// Scale factor for goal-directed navigation toward landmarks
pub const LANDMARK_ATTRACTION_SCALE: f64 = 0.5;
/// Distance threshold for considering a landmark "visited"
pub const LANDMARK_VISIT_RADIUS: f64 = 5.0;

// === Planning Parameters ===
/// Number of MCTS rollouts per planning step
pub const MCTS_ROLLOUTS: usize = 50;
/// Maximum depth for MCTS trajectory simulation
pub const MCTS_DEPTH: usize = 10;
/// Ticks between replanning (unless urgent)
pub const MCTS_REPLAN_INTERVAL: u64 = 20;
/// Energy threshold below which replanning becomes urgent
pub const MCTS_URGENT_ENERGY: f64 = 0.3;
/// Weight for blending planned action with reactive control
pub const PLANNING_WEIGHT: f64 = 0.3;

// === Active Inference Parameters ===
/// Learning rate for belief updates via VFE gradient descent
pub const BELIEF_LEARNING_RATE: f64 = 0.15;
/// Maximum VFE value for speed scaling normalization
pub const MAX_VFE: f64 = 5.0;
/// Initial sensory precision (inverse observation variance)
pub const INITIAL_SENSORY_PRECISION: f64 = 5.0;
/// Prior precision on nutrient belief (strength of homeostatic preference)
pub const NUTRIENT_PRIOR_PRECISION: f64 = 2.0;
/// Minimum sensory precision (prevents over-trust of noisy sensors)
pub const MIN_SENSORY_PRECISION: f64 = 0.5;
/// Maximum sensory precision (prevents over-confidence)
pub const MAX_SENSORY_PRECISION: f64 = 20.0;
/// Uncertainty growth factor for predictive beliefs
pub const UNCERTAINTY_GROWTH: f64 = 1.1;
/// Uncertainty reduction factor after observation
pub const UNCERTAINTY_REDUCTION: f64 = 0.95;
