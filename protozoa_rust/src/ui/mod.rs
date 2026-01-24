pub mod field;
pub mod render;

use crate::simulation::agent::{AgentMode, Protozoa};
use crate::simulation::environment::PetriDish;
use crate::simulation::memory::CellPrior;
use crate::simulation::params::{LANDMARK_VISIT_RADIUS, TARGET_CONCENTRATION};
use crate::simulation::planning::ActionDetail;

/// Snapshot of agent state for dashboard rendering.
#[derive(Clone, Debug)]
#[allow(dead_code)] // Used by tests and future UI components
pub struct DashboardState {
    // Position
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub speed: f64,

    // Metrics
    pub energy: f64,
    pub mode: AgentMode,
    pub prediction_error: f64,
    pub precision: f64,
    pub sensor_left: f64,
    pub sensor_right: f64,
    pub temporal_gradient: f64,

    // Spatial memory (flattened 20x10 grid)
    pub spatial_grid: Vec<CellPrior>,
    pub grid_width: usize,
    pub grid_height: usize,

    // MCTS planning
    pub plan_details: Vec<ActionDetail>,
    pub ticks_until_replan: u64,

    // Episodic memory
    pub landmarks: Vec<LandmarkSnapshot>,
    pub landmark_count: usize,
    pub nav_target_index: Option<usize>,
}

/// Snapshot of a landmark for rendering.
#[derive(Clone, Debug)]
#[allow(dead_code)] // Used by tests and future UI components
pub struct LandmarkSnapshot {
    pub x: f64,
    pub y: f64,
    pub reliability: f64,
    pub visit_count: u64,
}

impl DashboardState {
    /// Creates a dashboard state snapshot from agent and environment.
    #[must_use]
    #[allow(dead_code)] // Used by tests and future UI components
    pub fn from_agent(agent: &Protozoa, dish: &PetriDish) -> Self {
        let mean_sense = f64::midpoint(agent.val_l, agent.val_r);
        let prediction_error = mean_sense - TARGET_CONCENTRATION;
        let precision = agent.spatial_priors.get_cell(agent.x, agent.y).precision();
        let temporal_gradient = mean_sense - agent.last_mean_sense;

        // Flatten spatial grid
        let (gw, gh) = agent.spatial_priors.dimensions();
        let mut spatial_grid = Vec::with_capacity(gw * gh);
        for row in 0..gh {
            for col in 0..gw {
                #[allow(clippy::cast_precision_loss)]
                let x = (col as f64 + 0.5) * dish.width / gw as f64;
                #[allow(clippy::cast_precision_loss)]
                let y = (row as f64 + 0.5) * dish.height / gh as f64;
                spatial_grid.push(*agent.spatial_priors.get_cell(x, y));
            }
        }

        // Collect landmarks
        let landmarks: Vec<LandmarkSnapshot> = agent
            .episodic_memory
            .iter()
            .map(|lm| LandmarkSnapshot {
                x: lm.x,
                y: lm.y,
                reliability: lm.reliability,
                visit_count: lm.last_visit_tick,
            })
            .collect();

        // Find nav target (if in GoalNav mode)
        let nav_target_index = if agent.current_mode(dish) == AgentMode::GoalNav {
            agent
                .episodic_memory
                .best_distant_landmark(agent.x, agent.y, LANDMARK_VISIT_RADIUS)
                .and_then(|target| {
                    landmarks.iter().position(|lm| {
                        (lm.x - target.x).abs() < 0.1 && (lm.y - target.y).abs() < 0.1
                    })
                })
        } else {
            None
        };

        Self {
            x: agent.x,
            y: agent.y,
            angle: agent.angle,
            speed: agent.speed,
            energy: agent.energy,
            mode: agent.current_mode(dish),
            prediction_error,
            precision,
            sensor_left: agent.val_l,
            sensor_right: agent.val_r,
            temporal_gradient,
            spatial_grid,
            grid_width: gw,
            grid_height: gh,
            plan_details: agent.planner.last_plan_details().to_vec(),
            ticks_until_replan: agent.ticks_until_replan(),
            landmarks,
            landmark_count: agent.episodic_memory.count(),
            nav_target_index,
        }
    }
}
