# Cognitive Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a real-time TUI dashboard exposing the agent's internal cognitive state (spatial memory, MCTS planning, episodic landmarks, and detailed metrics) in a quadrant layout.

**Architecture:** Replace single-panel rendering with a four-quadrant layout using ratatui's Layout system. Add AgentMode enum and expose MCTS internals for rendering. Each panel has its own render function.

**Tech Stack:** Rust, ratatui (Layout, Block, Paragraph), crossterm

---

## Task 1: Add AgentMode Enum

**Files:**
- Modify: `protozoa_rust/src/simulation/agent.rs`
- Test: `protozoa_rust/tests/test_agent.rs`

**Step 1: Write the failing test**

Add to `tests/test_agent.rs`:

```rust
#[test]
fn test_agent_mode_exploring() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let agent = Protozoa::new(50.0, 25.0);
    // New agent with full energy should be exploring
    assert!(matches!(agent.current_mode(&dish), AgentMode::Exploring));
}

#[test]
fn test_agent_mode_exhausted() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(50.0, 25.0);
    agent.energy = 0.005; // Below EXHAUSTION_THRESHOLD (0.01)
    assert!(matches!(agent.current_mode(&dish), AgentMode::Exhausted));
}
```

Also add import at top of test file:
```rust
use protozoa_rust::simulation::agent::AgentMode;
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_agent_mode --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find type `AgentMode`"

**Step 3: Write minimal implementation**

Add to `src/simulation/agent.rs` after the imports:

```rust
/// Behavioral mode of the agent, derived from internal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentMode {
    /// Normal gradient following with exploration bonus
    Exploring,
    /// In high-nutrient area with high precision
    Exploiting,
    /// Temporal gradient below panic threshold
    Panicking,
    /// Energy below exhaustion threshold
    Exhausted,
    /// Actively navigating toward a landmark
    GoalNav,
}
```

Add method to `impl Protozoa`:

```rust
/// Returns the current behavioral mode derived from internal state.
#[must_use]
pub fn current_mode(&self, dish: &PetriDish) -> AgentMode {
    use crate::simulation::params::{
        EXHAUSTION_THRESHOLD, MCTS_URGENT_ENERGY, PANIC_THRESHOLD,
    };

    // Check exhausted first (most critical)
    if self.energy <= EXHAUSTION_THRESHOLD {
        return AgentMode::Exhausted;
    }

    // Check if panicking (temporal gradient)
    let mean_sense = f64::midpoint(self.val_l, self.val_r);
    let temp_gradient = mean_sense - self.last_mean_sense;
    if temp_gradient < PANIC_THRESHOLD {
        return AgentMode::Panicking;
    }

    // Check goal navigation (low energy, has landmark)
    if self.energy < MCTS_URGENT_ENERGY {
        if self.episodic_memory
            .best_distant_landmark(self.x, self.y, crate::simulation::params::LANDMARK_VISIT_RADIUS)
            .is_some()
        {
            return AgentMode::GoalNav;
        }
    }

    // Check exploiting (high precision at current location)
    let precision = self.spatial_priors.get_cell(self.x, self.y).precision();
    if precision > 5.0 && mean_sense > 0.6 {
        return AgentMode::Exploiting;
    }

    AgentMode::Exploring
}
```

Export AgentMode in `src/simulation/mod.rs`:
```rust
pub use agent::AgentMode;
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_agent_mode -v`
Expected: 2 tests pass

**Step 5: Commit**

```bash
git add src/simulation/agent.rs src/simulation/mod.rs tests/test_agent.rs
git commit -m "feat: add AgentMode enum with current_mode() method"
```

---

## Task 2: Expose MCTS Planning Details

**Files:**
- Modify: `protozoa_rust/src/simulation/planning/mcts.rs`
- Test: `protozoa_rust/tests/test_planning.rs`

**Step 1: Write the failing test**

Add to `tests/test_planning.rs`:

```rust
#[test]
fn test_planner_exposes_top_trajectories() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
    let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);
    let mut planner = MCTSPlanner::new();

    planner.plan(&state, &priors);

    let details = planner.last_plan_details();
    assert_eq!(details.len(), 3); // One per action
    for detail in &details {
        assert!(detail.action == Action::TurnLeft
            || detail.action == Action::Straight
            || detail.action == Action::TurnRight);
    }
}

#[test]
fn test_planner_exposes_efe_breakdown() {
    let priors: SpatialGrid<20, 10> = SpatialGrid::new(DISH_WIDTH, DISH_HEIGHT);
    let state = AgentState::new(50.0, 25.0, 0.0, 1.0, 1.0);
    let mut planner = MCTSPlanner::new();

    planner.plan(&state, &priors);

    let details = planner.last_plan_details();
    for detail in &details {
        // Total should equal pragmatic + epistemic (scaled)
        let expected = detail.pragmatic_value + 0.3 * detail.epistemic_value;
        assert!((detail.total_efe - expected).abs() < 0.01);
    }
}
```

Also add import:
```rust
use protozoa_rust::simulation::planning::ActionDetail;
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_planner_exposes --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find value `last_plan_details`"

**Step 3: Write minimal implementation**

Add to `src/simulation/planning/mcts.rs` after `Action` enum:

```rust
/// Details about a planned action for visualization.
#[derive(Clone, Debug)]
pub struct ActionDetail {
    /// The action evaluated
    pub action: Action,
    /// Average total Expected Free Energy
    pub total_efe: f64,
    /// Pragmatic component (nutrient × energy)
    pub pragmatic_value: f64,
    /// Epistemic component (uncertainty reduction)
    pub epistemic_value: f64,
    /// Sample trajectory positions for visualization
    pub sample_trajectory: Vec<(f64, f64)>,
}
```

Modify `MCTSPlanner` struct:

```rust
#[derive(Clone, Debug)]
pub struct MCTSPlanner {
    /// Best action from last planning cycle
    best_action: Action,
    /// Details of last planning cycle for each action
    last_details: Vec<ActionDetail>,
}
```

Update `MCTSPlanner::new()`:

```rust
pub const fn new() -> Self {
    Self {
        best_action: Action::Straight,
        last_details: Vec::new(),
    }
}
```

Note: Change `const fn` to `fn` since Vec::new() is not const.

Add method to `impl MCTSPlanner`:

```rust
/// Returns details of the last planning cycle.
#[must_use]
pub fn last_plan_details(&self) -> &[ActionDetail] {
    &self.last_details
}
```

Modify the `plan` method to store details:

```rust
pub fn plan(&mut self, state: &AgentState, priors: &SpatialGrid<20, 10>) -> Action {
    let mut rng = rand::rng();
    let mut best_value = f64::NEG_INFINITY;
    let mut best_action = Action::Straight;
    self.last_details.clear();

    // Evaluate each possible action
    for action in Action::all() {
        let mut total_pragmatic = 0.0;
        let mut total_epistemic = 0.0;
        let mut sample_traj = Vec::new();

        // Perform multiple rollouts
        for i in 0..MCTS_ROLLOUTS {
            let trajectory = self.rollout(*state, action, priors, &mut rng);
            let (pragmatic, epistemic) = self.efe_components(&trajectory, priors);
            total_pragmatic += pragmatic;
            total_epistemic += epistemic;

            // Store first trajectory as sample
            if i == 0 {
                sample_traj = trajectory.iter().map(|s| (s.x, s.y)).collect();
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let avg_pragmatic = total_pragmatic / MCTS_ROLLOUTS as f64;
        #[allow(clippy::cast_precision_loss)]
        let avg_epistemic = total_epistemic / MCTS_ROLLOUTS as f64;
        let avg_value = avg_pragmatic + EXPLORATION_SCALE * avg_epistemic;

        self.last_details.push(ActionDetail {
            action,
            total_efe: avg_value,
            pragmatic_value: avg_pragmatic,
            epistemic_value: avg_epistemic,
            sample_trajectory: sample_traj,
        });

        if avg_value > best_value {
            best_value = avg_value;
            best_action = action;
        }
    }

    self.best_action = best_action;
    best_action
}
```

Add helper method:

```rust
/// Computes pragmatic and epistemic components separately.
fn efe_components(&self, trajectory: &[AgentState], priors: &SpatialGrid<20, 10>) -> (f64, f64) {
    let mut pragmatic = 0.0;
    let mut epistemic = 0.0;

    for state in trajectory {
        let prior = priors.get_cell(state.x, state.y);
        pragmatic += prior.mean * state.energy;
        let precision = prior.precision().max(MIN_PRECISION);
        epistemic += 1.0 / precision;
    }

    (pragmatic, epistemic)
}
```

Update `expected_free_energy` to use the new helper:

```rust
fn expected_free_energy(&self, trajectory: &[AgentState], priors: &SpatialGrid<20, 10>) -> f64 {
    let (pragmatic, epistemic) = self.efe_components(trajectory, priors);
    pragmatic + EXPLORATION_SCALE * epistemic
}
```

Export in `src/simulation/planning/mod.rs`:
```rust
pub use mcts::ActionDetail;
```

And in `src/simulation/mod.rs`:
```rust
pub use planning::ActionDetail;
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_planner_exposes -v`
Expected: 2 tests pass

**Step 5: Commit**

```bash
git add src/simulation/planning/mcts.rs src/simulation/planning/mod.rs src/simulation/mod.rs tests/test_planning.rs
git commit -m "feat: expose MCTS planning details with EFE breakdown"
```

---

## Task 3: Add Replan Countdown to Agent

**Files:**
- Modify: `protozoa_rust/src/simulation/agent.rs`
- Test: `protozoa_rust/tests/test_agent.rs`

**Step 1: Write the failing test**

Add to `tests/test_agent.rs`:

```rust
#[test]
fn test_agent_ticks_until_replan() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(50.0, 25.0);

    // Initial tick should trigger planning
    agent.sense(&dish);
    agent.update_state(&dish);

    // Should be MCTS_REPLAN_INTERVAL - 1 ticks until next replan
    assert!(agent.ticks_until_replan() > 0);
    assert!(agent.ticks_until_replan() <= 20); // MCTS_REPLAN_INTERVAL
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_agent_ticks_until_replan --no-run 2>&1 | head -20`
Expected: Compilation error "no method named `ticks_until_replan`"

**Step 3: Write minimal implementation**

Add method to `impl Protozoa` in `src/simulation/agent.rs`:

```rust
/// Returns ticks until next MCTS replan.
#[must_use]
pub fn ticks_until_replan(&self) -> u64 {
    let elapsed = self.tick_count.saturating_sub(self.last_plan_tick);
    MCTS_REPLAN_INTERVAL.saturating_sub(elapsed)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_agent_ticks_until_replan -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/simulation/agent.rs tests/test_agent.rs
git commit -m "feat: add ticks_until_replan() method to agent"
```

---

## Task 4: Create DashboardState Struct

**Files:**
- Modify: `protozoa_rust/src/ui/mod.rs`
- Test: `protozoa_rust/tests/test_rendering.rs`

**Step 1: Write the failing test**

Add to `tests/test_rendering.rs`:

```rust
use protozoa_rust::ui::DashboardState;
use protozoa_rust::simulation::agent::{AgentMode, Protozoa};
use protozoa_rust::simulation::environment::PetriDish;
use protozoa_rust::simulation::params::{DISH_WIDTH, DISH_HEIGHT};

#[test]
fn test_dashboard_state_from_agent() {
    let dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let agent = Protozoa::new(50.0, 25.0);

    let state = DashboardState::from_agent(&agent, &dish);

    assert!((state.energy - 1.0).abs() < 0.01);
    assert!(matches!(state.mode, AgentMode::Exploring));
    assert_eq!(state.landmark_count, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_dashboard_state --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find value `DashboardState`"

**Step 3: Write minimal implementation**

Create new file content for `src/ui/mod.rs`:

```rust
pub mod field;
pub mod render;

use crate::simulation::agent::{AgentMode, Protozoa};
use crate::simulation::environment::PetriDish;
use crate::simulation::memory::CellPrior;
use crate::simulation::planning::ActionDetail;
use crate::simulation::params::TARGET_CONCENTRATION;

/// Snapshot of agent state for dashboard rendering.
#[derive(Clone, Debug)]
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
pub struct LandmarkSnapshot {
    pub x: f64,
    pub y: f64,
    pub reliability: f64,
    pub visit_count: u64,
}

impl DashboardState {
    /// Creates a dashboard state snapshot from agent and environment.
    #[must_use]
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
                let x = (col as f64 + 0.5) * dish.width / gw as f64;
                let y = (row as f64 + 0.5) * dish.height / gh as f64;
                spatial_grid.push(*agent.spatial_priors.get_cell(x, y));
            }
        }

        // Collect landmarks
        let landmarks: Vec<LandmarkSnapshot> = agent.episodic_memory
            .iter()
            .map(|lm| LandmarkSnapshot {
                x: lm.x,
                y: lm.y,
                reliability: lm.reliability,
                visit_count: lm.last_visit_tick, // Using tick as proxy
            })
            .collect();

        // Find nav target (if in GoalNav mode)
        let nav_target_index = if agent.current_mode(dish) == AgentMode::GoalNav {
            agent.episodic_memory
                .best_distant_landmark(agent.x, agent.y, crate::simulation::params::LANDMARK_VISIT_RADIUS)
                .and_then(|target| {
                    landmarks.iter().position(|lm|
                        (lm.x - target.x).abs() < 0.1 && (lm.y - target.y).abs() < 0.1
                    )
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
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_dashboard_state -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ui/mod.rs tests/test_rendering.rs
git commit -m "feat: add DashboardState struct for dashboard rendering"
```

---

## Task 5: Implement Quadrant Layout

**Files:**
- Modify: `protozoa_rust/src/ui/render.rs`
- Test: `protozoa_rust/tests/test_rendering.rs`

**Step 1: Write the failing test**

Add to `tests/test_rendering.rs`:

```rust
use ratatui::layout::Rect;
use protozoa_rust::ui::render::compute_quadrant_layout;

#[test]
fn test_quadrant_layout_dimensions() {
    let area = Rect::new(0, 0, 120, 40);
    let quadrants = compute_quadrant_layout(area);

    // Should have 4 quadrants
    assert_eq!(quadrants.len(), 4);

    // Each quadrant should be roughly half the area
    for q in &quadrants {
        assert!(q.width >= 50);
        assert!(q.height >= 15);
    }

    // Top-left should start at origin
    assert_eq!(quadrants[0].x, 0);
    assert_eq!(quadrants[0].y, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_quadrant_layout --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find function `compute_quadrant_layout`"

**Step 3: Write minimal implementation**

Add to `src/ui/render.rs`:

```rust
use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Computes the four quadrant areas for the dashboard layout.
#[must_use]
pub fn compute_quadrant_layout(area: Rect) -> Vec<Rect> {
    // Split vertically into top and bottom
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Split each row horizontally
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(vertical[0]);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(vertical[1]);

    vec![top[0], top[1], bottom[0], bottom[1]]
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_quadrant_layout -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ui/render.rs tests/test_rendering.rs
git commit -m "feat: add compute_quadrant_layout for dashboard"
```

---

## Task 6: Implement Metrics Overlay Rendering

**Files:**
- Modify: `protozoa_rust/src/ui/render.rs`
- Test: `protozoa_rust/tests/test_rendering.rs`

**Step 1: Write the failing test**

Add to `tests/test_rendering.rs`:

```rust
use protozoa_rust::ui::render::format_metrics_overlay;
use protozoa_rust::simulation::agent::AgentMode;

#[test]
fn test_metrics_overlay_content() {
    let lines = format_metrics_overlay(
        0.82,             // energy
        AgentMode::Exploring,
        -0.12,            // prediction_error
        0.85,             // precision
        1.3,              // speed
        127.0,            // angle in degrees
        0.74,             // sensor_left
        0.68,             // sensor_right
        -0.02,            // temporal_gradient
    );

    // Should have 6 lines
    assert_eq!(lines.len(), 6);

    // First line should contain energy bar
    assert!(lines[0].contains("E:"));
    assert!(lines[0].contains("82%"));

    // Second line should contain mode
    assert!(lines[1].contains("EXPLORING"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_metrics_overlay_content --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find function `format_metrics_overlay`"

**Step 3: Write minimal implementation**

Add to `src/ui/render.rs`:

```rust
use crate::simulation::agent::AgentMode;

/// Formats the metrics overlay lines for the petri dish panel.
#[must_use]
pub fn format_metrics_overlay(
    energy: f64,
    mode: AgentMode,
    prediction_error: f64,
    precision: f64,
    speed: f64,
    angle_deg: f64,
    sensor_left: f64,
    sensor_right: f64,
    temporal_gradient: f64,
) -> Vec<String> {
    // Energy bar (10 chars)
    let filled = (energy * 10.0).round() as usize;
    let empty = 10 - filled.min(10);
    let bar: String = "█".repeat(filled.min(10)) + &"░".repeat(empty);
    let pct = (energy * 100.0).round() as i32;

    let mode_str = match mode {
        AgentMode::Exploring => "EXPLORING",
        AgentMode::Exploiting => "EXPLOITING",
        AgentMode::Panicking => "PANICKING",
        AgentMode::Exhausted => "EXHAUSTED",
        AgentMode::GoalNav => "GOAL-NAV",
    };

    vec![
        format!("E:[{bar}] {pct:>3}%"),
        format!("Mode: {mode_str}"),
        format!("PE:{prediction_error:>6.2}  ρ:{precision:.2}"),
        format!("v:{speed:>4.1}  θ:{angle_deg:>4.0}°"),
        format!("L:{sensor_left:.2}  R:{sensor_right:.2}"),
        format!("∂t:{temporal_gradient:>6.2}"),
    ]
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_metrics_overlay -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ui/render.rs tests/test_rendering.rs
git commit -m "feat: add format_metrics_overlay for dashboard"
```

---

## Task 7: Implement Spatial Grid Rendering

**Files:**
- Modify: `protozoa_rust/src/ui/render.rs`
- Test: `protozoa_rust/tests/test_rendering.rs`

**Step 1: Write the failing test**

Add to `tests/test_rendering.rs`:

```rust
use protozoa_rust::ui::render::render_spatial_grid_lines;
use protozoa_rust::simulation::memory::CellPrior;

#[test]
fn test_spatial_grid_ascii_mapping() {
    // Create a simple 4x2 grid
    let mut cells = vec![CellPrior::default(); 8];

    // Set different mean values
    cells[0].mean = 0.0;  // Should be ' '
    cells[1].mean = 0.3;  // Should be around ':'
    cells[2].mean = 0.6;  // Should be around '+'
    cells[3].mean = 0.9;  // Should be around '@'

    let lines = render_spatial_grid_lines(&cells, 4, 2, None);

    assert_eq!(lines.len(), 2);
    // First row contains cells 0-3
    assert!(lines[0].contains(' ')); // Low value
    assert!(lines[1].len() >= 4);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_spatial_grid_ascii --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find function `render_spatial_grid_lines`"

**Step 3: Write minimal implementation**

Add to `src/ui/render.rs`:

```rust
use crate::simulation::memory::CellPrior;

/// ASCII density characters for heat map visualization (low to high).
const DENSITY_CHARS: [char; 9] = [' ', '.', ',', ':', ';', '+', '*', '#', '@'];

/// Converts a mean value (0.0-1.0) to an ASCII density character.
fn mean_to_char(mean: f64) -> char {
    let idx = ((mean.clamp(0.0, 1.0)) * 8.0).round() as usize;
    DENSITY_CHARS[idx.min(8)]
}

/// Renders spatial grid as ASCII lines.
/// agent_cell is (row, col) of agent's current grid cell, if known.
#[must_use]
pub fn render_spatial_grid_lines(
    cells: &[CellPrior],
    width: usize,
    height: usize,
    agent_cell: Option<(usize, usize)>,
) -> Vec<String> {
    let mut lines = Vec::with_capacity(height);

    for row in 0..height {
        let mut line = String::with_capacity(width);
        for col in 0..width {
            let idx = row * width + col;
            if let Some(cell) = cells.get(idx) {
                if agent_cell == Some((row, col)) {
                    line.push('○');
                } else {
                    line.push(mean_to_char(cell.mean));
                }
            } else {
                line.push(' ');
            }
        }
        lines.push(line);
    }

    lines
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_spatial_grid_ascii -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ui/render.rs tests/test_rendering.rs
git commit -m "feat: add render_spatial_grid_lines for spatial memory panel"
```

---

## Task 8: Implement MCTS Panel Rendering

**Files:**
- Modify: `protozoa_rust/src/ui/render.rs`
- Test: `protozoa_rust/tests/test_rendering.rs`

**Step 1: Write the failing test**

Add to `tests/test_rendering.rs`:

```rust
use protozoa_rust::ui::render::format_mcts_summary;
use protozoa_rust::simulation::planning::{Action, ActionDetail};

#[test]
fn test_mcts_summary_format() {
    let details = vec![
        ActionDetail {
            action: Action::TurnLeft,
            total_efe: -1.5,
            pragmatic_value: -1.2,
            epistemic_value: -1.0,
            sample_trajectory: vec![(50.0, 25.0), (52.0, 27.0)],
        },
        ActionDetail {
            action: Action::Straight,
            total_efe: -2.3,
            pragmatic_value: -1.8,
            epistemic_value: -1.67,
            sample_trajectory: vec![(50.0, 25.0), (55.0, 25.0)],
        },
    ];

    let lines = format_mcts_summary(&details, 7);

    // Should have lines for best action, G, Prag, Epis, Rolls, Depth, Replan
    assert!(lines.len() >= 5);
    assert!(lines[0].contains("Best"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_mcts_summary --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find function `format_mcts_summary`"

**Step 3: Write minimal implementation**

Add to `src/ui/render.rs`:

```rust
use crate::simulation::planning::{Action, ActionDetail};
use crate::simulation::params::{MCTS_ROLLOUTS, MCTS_DEPTH};

/// Direction arrow for an action.
fn action_to_arrow(action: Action, base_angle: f64) -> &'static str {
    let angle = base_angle + action.angle_delta();
    let octant = ((angle + std::f64::consts::PI / 8.0) / (std::f64::consts::PI / 4.0)) as i32 % 8;
    match octant.rem_euclid(8) {
        0 => "→",
        1 => "↗",
        2 => "↑",
        3 => "↖",
        4 => "←",
        5 => "↙",
        6 => "↓",
        7 => "↘",
        _ => "→",
    }
}

/// Direction name for an action.
fn action_to_name(action: Action) -> &'static str {
    match action {
        Action::TurnLeft => "L",
        Action::Straight => "S",
        Action::TurnRight => "R",
    }
}

/// Formats MCTS planning summary text.
#[must_use]
pub fn format_mcts_summary(details: &[ActionDetail], ticks_until_replan: u64) -> Vec<String> {
    // Find best action (highest EFE)
    let best = details.iter()
        .max_by(|a, b| a.total_efe.total_cmp(&b.total_efe));

    if let Some(best) = best {
        vec![
            format!("Best: {} ({})", action_to_arrow(best.action, 0.0), action_to_name(best.action)),
            format!("G: {:.2}", best.total_efe),
            format!("├─Prag: {:.2}", best.pragmatic_value),
            format!("└─Epis: {:.2}", best.epistemic_value),
            format!("Rolls: {}", MCTS_ROLLOUTS),
            format!("Depth: {}", MCTS_DEPTH),
            format!("Replan: {}", ticks_until_replan),
        ]
    } else {
        vec!["No plan data".to_string()]
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_mcts_summary -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ui/render.rs tests/test_rendering.rs
git commit -m "feat: add format_mcts_summary for MCTS panel"
```

---

## Task 9: Implement Landmarks Panel Rendering

**Files:**
- Modify: `protozoa_rust/src/ui/render.rs`
- Test: `protozoa_rust/tests/test_rendering.rs`

**Step 1: Write the failing test**

Add to `tests/test_rendering.rs`:

```rust
use protozoa_rust::ui::{LandmarkSnapshot, render::format_landmarks_list};

#[test]
fn test_landmarks_list_format() {
    let landmarks = vec![
        LandmarkSnapshot { x: 12.0, y: 8.0, reliability: 0.92, visit_count: 4 },
        LandmarkSnapshot { x: 3.0, y: 18.0, reliability: 0.67, visit_count: 2 },
    ];

    let lines = format_landmarks_list(&landmarks, Some(0));

    // Should have header + landmarks
    assert!(lines.len() >= 3);
    // First landmark should have nav arrow
    assert!(lines[2].starts_with('→'));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_landmarks_list --no-run 2>&1 | head -20`
Expected: Compilation error "cannot find function `format_landmarks_list`"

**Step 3: Write minimal implementation**

Add to `src/ui/render.rs`:

```rust
use crate::ui::LandmarkSnapshot;

/// Formats landmarks as a list table.
#[must_use]
pub fn format_landmarks_list(landmarks: &[LandmarkSnapshot], nav_target: Option<usize>) -> Vec<String> {
    let mut lines = vec![
        " # │ Pos     │Rel │Vis".to_string(),
        "───┼─────────┼────┼───".to_string(),
    ];

    for (i, lm) in landmarks.iter().enumerate() {
        let prefix = if nav_target == Some(i) { "→" } else { " " };
        lines.push(format!(
            "{}{} │({:>3},{:>3})│.{:02}│ {}",
            prefix,
            i + 1,
            lm.x as i32,
            lm.y as i32,
            (lm.reliability * 100.0) as i32 % 100,
            lm.visit_count
        ));
    }

    // Pad with empty slots up to 8
    for i in landmarks.len()..8 {
        lines.push(format!(" {} │   --    │ -- │ -", i + 1));
    }

    lines
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_landmarks_list -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ui/render.rs tests/test_rendering.rs
git commit -m "feat: add format_landmarks_list for landmarks panel"
```

---

## Task 10: Integrate Dashboard into Main Render Loop

**Files:**
- Modify: `protozoa_rust/src/ui/render.rs`
- Modify: `protozoa_rust/src/main.rs`

**Step 1: Write the new draw_dashboard function**

Add to `src/ui/render.rs`:

```rust
use ratatui::{
    Frame,
    style::{Color, Modifier, Style},
    text::{Line as TuiLine, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use crate::ui::DashboardState;

/// Draws the full cognitive dashboard.
pub fn draw_dashboard(f: &mut Frame, grid_lines: Vec<String>, state: &DashboardState) {
    let quadrants = compute_quadrant_layout(f.area());

    // === Top-Left: Petri Dish with Overlay ===
    draw_petri_dish_panel(f, quadrants[0], grid_lines, state);

    // === Top-Right: Spatial Memory Grid ===
    draw_spatial_grid_panel(f, quadrants[1], state);

    // === Bottom-Left: MCTS Planning ===
    draw_mcts_panel(f, quadrants[2], state);

    // === Bottom-Right: Landmarks ===
    draw_landmarks_panel(f, quadrants[3], state);
}

fn draw_petri_dish_panel(f: &mut Frame, area: Rect, grid_lines: Vec<String>, state: &DashboardState) {
    let block = Block::default()
        .title(" Petri Dish ")
        .borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Render field
    let text: Vec<TuiLine> = grid_lines
        .into_iter()
        .map(|s| TuiLine::from(Span::raw(s)))
        .collect();
    let field = Paragraph::new(text);
    f.render_widget(field, inner);

    // Metrics overlay (bottom-left of inner area)
    let angle_deg = state.angle.to_degrees();
    let overlay_lines = format_metrics_overlay(
        state.energy,
        state.mode,
        state.prediction_error,
        state.precision,
        state.speed,
        angle_deg,
        state.sensor_left,
        state.sensor_right,
        state.temporal_gradient,
    );

    let overlay_height = overlay_lines.len() as u16 + 2;
    let overlay_width = 23;
    if inner.height > overlay_height && inner.width > overlay_width {
        let overlay_area = Rect::new(
            inner.x,
            inner.y + inner.height - overlay_height,
            overlay_width,
            overlay_height,
        );
        let overlay_text: Vec<TuiLine> = overlay_lines
            .into_iter()
            .map(|s| TuiLine::from(Span::styled(s, Style::default().add_modifier(Modifier::BOLD))))
            .collect();
        let overlay = Paragraph::new(overlay_text)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().bg(Color::Black));
        f.render_widget(overlay, overlay_area);
    }
}

fn draw_spatial_grid_panel(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(" Spatial Memory ")
        .borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Calculate agent's grid cell
    let agent_col = ((state.x / 100.0) * state.grid_width as f64).floor() as usize;
    let agent_row = ((state.y / 50.0) * state.grid_height as f64).floor() as usize;
    let agent_cell = Some((agent_row.min(state.grid_height - 1), agent_col.min(state.grid_width - 1)));

    let lines = render_spatial_grid_lines(&state.spatial_grid, state.grid_width, state.grid_height, agent_cell);
    let text: Vec<TuiLine> = lines
        .into_iter()
        .map(|s| TuiLine::from(Span::raw(s)))
        .collect();
    let grid = Paragraph::new(text);
    f.render_widget(grid, inner);
}

fn draw_mcts_panel(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(" MCTS Planning ")
        .borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Just the text summary for now (mini-map is optional enhancement)
    let lines = format_mcts_summary(&state.plan_details, state.ticks_until_replan);
    let text: Vec<TuiLine> = lines
        .into_iter()
        .map(|s| TuiLine::from(Span::raw(s)))
        .collect();
    let summary = Paragraph::new(text);
    f.render_widget(summary, inner);
}

fn draw_landmarks_panel(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(" Landmarks ")
        .borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines = format_landmarks_list(&state.landmarks, state.nav_target_index);
    let text: Vec<TuiLine> = lines
        .into_iter()
        .map(|s| TuiLine::from(Span::raw(s)))
        .collect();
    let list = Paragraph::new(text);
    f.render_widget(list, inner);
}
```

**Step 2: Update main.rs to use new dashboard**

Replace the render section in `run_app` in `src/main.rs`:

```rust
use crate::ui::{
    field::compute_field_grid,
    render::{draw_dashboard, world_to_grid_coords},
    DashboardState,
};
```

Update the terminal.draw closure:

```rust
terminal.draw(|f| {
    let area = f.area();

    // Use top-left quadrant size for field computation
    let field_rows = (area.height / 2).saturating_sub(2) as usize;
    let field_cols = (area.width / 2).saturating_sub(2) as usize;

    // Compute background in parallel
    let mut grid = compute_field_grid(dish, field_rows, field_cols);

    // Overlay Agent on field
    if field_rows > 0 && field_cols > 0 {
        let (r, c) = world_to_grid_coords(
            agent.x, agent.y,
            dish.width, dish.height,
            field_rows, field_cols
        );

        if r < field_rows && c < field_cols {
            if let Some(line) = grid.get_mut(r) {
                if c < line.len() {
                    line.replace_range(c..=c, "O");
                }
            }
        }
    }

    // Create dashboard state
    let dashboard_state = DashboardState::from_agent(agent, dish);

    // Draw the full dashboard
    draw_dashboard(f, grid, &dashboard_state);
})?;
```

**Step 3: Run to verify it compiles**

Run: `cargo build --release`
Expected: Compiles successfully

**Step 4: Run tests**

Run: `cargo test`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/ui/render.rs src/main.rs
git commit -m "feat: integrate cognitive dashboard into main render loop"
```

---

## Task 11: Add Terminal Size Check

**Files:**
- Modify: `protozoa_rust/src/main.rs`

**Step 1: Add size check before run_app**

Add after terminal creation in `main()`:

```rust
// Check terminal size
let size = terminal.size()?;
if size.width < 80 || size.height < 24 {
    eprintln!(
        "Warning: Terminal size {}x{} is smaller than recommended 80x24. Dashboard may not display correctly.",
        size.width, size.height
    );
}
```

**Step 2: Run to verify**

Run: `cargo run --release`
Expected: Dashboard renders (press 'q' to quit)

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add terminal size warning for small terminals"
```

---

## Task 12: Final Integration Test

**Files:**
- Modify: `protozoa_rust/tests/test_integration.rs`

**Step 1: Add dashboard integration test**

Add to `tests/test_integration.rs`:

```rust
use protozoa_rust::ui::DashboardState;

#[test]
fn test_dashboard_state_updates_during_simulation() {
    let mut dish = PetriDish::new(DISH_WIDTH, DISH_HEIGHT);
    let mut agent = Protozoa::new(DISH_WIDTH / 2.0, DISH_HEIGHT / 2.0);

    // Run for several ticks
    for _ in 0..100 {
        dish.update();
        agent.sense(&dish);
        agent.update_state(&dish);
    }

    let state = DashboardState::from_agent(&agent, &dish);

    // Verify state is populated
    assert!(state.energy > 0.0);
    assert!(state.spatial_grid.len() == 200); // 20x10
    assert!(state.plan_details.len() == 3); // One per action
}
```

**Step 2: Run all tests**

Run: `cargo test`
Expected: All tests pass

**Step 3: Final commit**

```bash
git add tests/test_integration.rs
git commit -m "test: add dashboard integration test"
```

---

## Summary

This plan implements the cognitive dashboard in 12 incremental tasks:

1. **AgentMode enum** — Derive behavioral mode from state
2. **MCTS details** — Expose planning internals
3. **Replan countdown** — Track ticks until next plan
4. **DashboardState** — Bundle all data for rendering
5. **Quadrant layout** — Split terminal into 4 panels
6. **Metrics overlay** — Energy, mode, sensors, gradients
7. **Spatial grid** — ASCII heat map of learned priors
8. **MCTS panel** — Planning summary with EFE breakdown
9. **Landmarks panel** — List with navigation target
10. **Integration** — Wire everything into main loop
11. **Size check** — Warn on small terminals
12. **Integration test** — Verify end-to-end

Each task follows TDD: failing test → implementation → passing test → commit.
