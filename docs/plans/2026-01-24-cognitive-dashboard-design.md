# Cognitive Dashboard Design

**Date:** 2026-01-24
**Status:** Approved

## Overview

Add a real-time TUI dashboard exposing the agent's internal cognitive state: spatial memory, MCTS planning, episodic landmarks, and detailed metrics.

## Layout

Quadrant layout with four panels:

```
┌── Petri Dish ─────────┬── Spatial Memory ─────┐
│                       │                       │
│  (with metrics        │  (20×10 learned       │
│   overlay)            │   priors heat map)    │
│                       │                       │
├── MCTS Planning ──────┼── Landmarks ──────────┤
│                       │                       │
│  (mini-map +          │  (mini-map +          │
│   text summary)       │   list view)          │
│                       │                       │
└───────────────────────┴───────────────────────┘
```

## Panel Specifications

### 1. Petri Dish (Top-Left) with Metrics Overlay

The existing petri dish visualization with a metrics overlay box in the corner.

**Overlay contents:**
```
┌─────────────────────┐
│ E:[████████░░] 82%  │  ← Energy bar (10 chars) + percentage
│ Mode: EXPLOITING    │  ← Current behavioral mode
│ PE:-0.12  ρ:0.85    │  ← Prediction Error, Precision (rho)
│ v:1.3  θ:127°       │  ← Speed, Heading in degrees
│ L:0.74  R:0.68      │  ← Left/Right sensor readings
│ ∂t:-0.02            │  ← Temporal gradient
└─────────────────────┘
```

**Behavioral modes** (derived from agent state):
- `EXPLORING` — Normal gradient following with exploration bonus
- `EXPLOITING` — In high-nutrient area, precision is high
- `PANICKING` — Temporal gradient below `PANIC_THRESHOLD` (-0.01)
- `EXHAUSTED` — Energy below `EXHAUSTION_THRESHOLD` (0.01)
- `GOAL-NAV` — Actively navigating toward a landmark (energy < 30%)

**Implementation notes:**
- Mode derived from existing state variables each tick (not stored)
- Energy bar: `█` filled, `░` empty
- Precision shown as `ρ` to save space
- Floats formatted to 2 decimal places
- Overlay width fixed at ~21 characters

### 2. Spatial Memory Grid (Top-Right)

Displays the 20×10 learned nutrient expectations using ASCII density.

**Rendering:**
- Each cell maps to a character based on `mean` value
- Character scale: ` ` (0.0) → `.` → `,` → `:` → `;` → `+` → `*` → `#` → `@` (1.0)
- Agent's current grid cell marked with `○`

```
── Spatial Memory (20×10) ─────────────
  · · · · · · · · · · · · · · · · · · · ·
  · · · · · , : ; ; : , · · · · · · · · ·
  · · · · : ; + * * + ; : · · · · · · · ·
  · · · · ; * # @ @ # * ; · · · · · · · ·
  · · · · : ; + ○ * + ; : · · · · · · · ·
  · · · · · , : ; ; : , · · · · · · · · ·
  · · · · · · · · · · · · · · · ; + ; · ·
  · · · · · · · · · · · · · · : + * + : ·
  · · · · · · · · · · · · · · · ; + ; · ·
  · · · · · · · · · · · · · · · · · · · ·
```

**Data source:**
- `SpatialGrid::cells[y][x].mean`
- Grid dimensions from `params::GRID_WIDTH` (20) and `GRID_HEIGHT` (10)
- Coordinate mapping: `grid_x = (agent.x / DISH_WIDTH * GRID_WIDTH).floor()`

### 3. MCTS Planning (Bottom-Left)

Mini-map and text summary side by side.

**Mini-map (left half, ~10×6 chars):**
- Scaled-down petri dish
- Agent position: `@`
- Top 3 trajectories:
  - Best: solid arrows (`→` `↗` `↑` `↖` `←` `↙` `↓` `↘`)
  - 2nd/3rd: dots along path
- Endpoints marked with intensity based on EFE score

```
╭──────────╮
│·  · →→↗  │
│ @↗↗  · ↗ │
│  ·· ·  · │
│ ·  ·  ·  │
╰──────────╯
```

**Text summary (right half):**
```
Best: ↗ (NE)
G: -2.31
├─Prag: -1.82
└─Epis: -0.49
Rolls: 50
Depth: 10
Replan: 7
```

- `Best`: Direction of chosen action
- `G`: Total Expected Free Energy (lower is better)
- `Prag`: Pragmatic value (nutrient × energy)
- `Epis`: Epistemic value (uncertainty reduction)
- `Rolls`: Number of MCTS rollouts
- `Replan`: Ticks until next planning cycle

### 4. Landmarks (Bottom-Right)

Mini-map and list view stacked vertically.

**Mini-map (top, ~10×5 chars):**
- Scaled-down petri dish
- Landmarks marked `1`-`8`, intensity based on reliability
- Navigation target highlighted: `[3]`
- Agent position: `@`

```
╭────────────────╮
│  1  ·  ·  · 3  │
│  ·  ·  @  ·  · │
│  ·  ·  ·  · [5]│
│  2  ·  ·  4  · │
╰────────────────╯
```

**List view (bottom):**
```
 # │ Pos    │Rel │Vis
───┼────────┼────┼───
 1 │ (12,8) │.92 │ 4
 2 │ (3,18) │.67 │ 2
→3 │ (45,5) │.81 │ 3
 4 │ (38,20)│.34 │ 1
 5 │ (52,12)│.88 │ 5
```

- `Pos`: Landmark coordinates
- `Rel`: Reliability score (0-1)
- `Vis`: Visit count
- `→` prefix: current navigation target
- Empty slots: `- │   --   │ -- │ -`

## Implementation Architecture

### File Changes

1. **`src/ui/render.rs`** — Major refactor
   - Replace single-widget render with quadrant layout using `ratatui::layout::Layout`
   - Create four sub-render functions: `render_petri_dish()`, `render_spatial_grid()`, `render_mcts()`, `render_landmarks()`
   - Add metrics overlay rendering within petri dish

2. **`src/simulation/agent.rs`** — Minor additions
   - Add `pub fn current_mode(&self) -> AgentMode` method deriving mode from state
   - Expose MCTS internal state via new getters
   - Add `AgentMode` enum: `Exploring`, `Exploiting`, `Panicking`, `Exhausted`, `GoalNav`

3. **`src/simulation/planning/mcts.rs`** — Minor additions
   - Store and expose top 3 trajectories (not just the best)
   - Add getters for pragmatic/epistemic value breakdown

4. **`src/ui/mod.rs`** — New types
   - Add `DashboardState` struct bundling all data needed for rendering

### No Changes Needed
- `environment.rs` — Already exposes what we need
- `memory/` modules — Already have public getters
- `params.rs` — No new parameters required

## Testing

### New Tests

1. **`tests/test_rendering.rs`** — Extend existing
   - `test_quadrant_layout_dimensions()` — Verify layout splits correctly
   - `test_metrics_overlay_content()` — Check all fields rendered
   - `test_spatial_grid_ascii_mapping()` — Verify density character selection
   - `test_landmark_minimap_scaling()` — Coordinate transform accuracy

2. **`tests/test_agent.rs`** — Extend existing
   - `test_current_mode_derivation()` — Each mode correctly identified from state

## Constraints

### Terminal Size
- Minimum: 80×24 (standard) — panels tight but functional
- Recommended: 120×40 for comfortable viewing
- Add terminal size check on startup; warn if too small

### Performance
- Four panels instead of one — minimal impact (field already computed)
- MCTS trajectory storage (3 extra) — negligible memory
- Target: maintain 60fps on `--release` builds
