//! Memory systems for the Active Inference agent.
//!
//! This module provides:
//! - Short-term memory via ring buffers
//! - Long-term memory via spatial prior grids
//! - Episodic memory for landmark recall

// Allow unused items - these will be used in future tasks (MCTS, goal-directed navigation)
#![allow(dead_code, unused_imports)]

pub mod episodic;
mod ring_buffer;
pub mod spatial_grid;

pub use episodic::{EpisodicMemory, Landmark};
pub use ring_buffer::RingBuffer;
pub use spatial_grid::{CellPrior, SpatialGrid};

/// A snapshot of sensory experience at a single tick.
#[derive(Clone, Copy, Default, Debug)]
pub struct SensorSnapshot {
    /// Left sensor concentration value
    pub val_l: f64,
    /// Right sensor concentration value
    pub val_r: f64,
    /// Agent x position when snapshot was taken
    pub x: f64,
    /// Agent y position when snapshot was taken
    pub y: f64,
    /// Agent energy level when snapshot was taken
    pub energy: f64,
    /// Tick number when snapshot was taken
    pub tick: u64,
}

/// Short-term memory buffer holding recent sensor experiences.
pub type SensorHistory = RingBuffer<SensorSnapshot, 32>;
