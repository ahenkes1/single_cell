//! Planning systems for the Active Inference agent.
//!
//! This module provides:
//! - Monte Carlo Tree Search for trajectory planning
//! - Expected Free Energy computation for action evaluation

// Allow unused items - will be used when integrated with agent
#![allow(dead_code, unused_imports)]

mod mcts;

pub use mcts::{Action, AgentState, MCTSPlanner};
