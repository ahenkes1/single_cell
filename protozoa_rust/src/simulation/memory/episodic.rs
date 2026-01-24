//! Episodic memory for landmark storage and recall.
//!
//! The agent remembers high-nutrient locations (landmarks) and can
//! navigate back to them when energy is low.

use crate::simulation::params::{LANDMARK_DECAY, LANDMARK_VISIT_RADIUS, MAX_LANDMARKS};

/// A remembered high-nutrient location.
#[derive(Clone, Copy, Debug)]
pub struct Landmark {
    /// X position of the landmark
    pub x: f64,
    /// Y position of the landmark
    pub y: f64,
    /// Peak nutrient concentration observed at this location
    pub peak_nutrient: f64,
    /// Tick when this landmark was last visited
    pub last_visit_tick: u64,
    /// Reliability score (decays over time when not visited)
    pub reliability: f64,
}

impl Landmark {
    /// Creates a new landmark at the given position.
    #[must_use]
    pub fn new(x: f64, y: f64, nutrient: f64, tick: u64) -> Self {
        Self {
            x,
            y,
            peak_nutrient: nutrient,
            last_visit_tick: tick,
            reliability: 1.0,
        }
    }

    /// Returns the distance from this landmark to a given position.
    #[must_use]
    pub fn distance_to(&self, x: f64, y: f64) -> f64 {
        let dx = self.x - x;
        let dy = self.y - y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns the weighted value of this landmark (nutrient * reliability).
    #[must_use]
    pub fn value(&self) -> f64 {
        self.peak_nutrient * self.reliability
    }

    /// Decays the reliability of this landmark.
    pub fn decay(&mut self) {
        self.reliability *= LANDMARK_DECAY;
    }

    /// Refreshes the landmark on revisit.
    pub fn refresh(&mut self, nutrient: f64, tick: u64) {
        self.peak_nutrient = self.peak_nutrient.max(nutrient);
        self.last_visit_tick = tick;
        self.reliability = 1.0;
    }
}

/// Episodic memory storing remembered landmarks.
#[derive(Clone, Debug)]
pub struct EpisodicMemory {
    landmarks: [Option<Landmark>; MAX_LANDMARKS],
}

impl Default for EpisodicMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl EpisodicMemory {
    /// Creates a new empty episodic memory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            landmarks: [None; MAX_LANDMARKS],
        }
    }

    /// Returns the number of stored landmarks.
    #[must_use]
    pub fn count(&self) -> usize {
        self.landmarks.iter().filter(|l| l.is_some()).count()
    }

    /// Attempts to store a new landmark if it's valuable enough.
    ///
    /// If memory is full, replaces the least valuable landmark.
    /// If the position is near an existing landmark, updates that one instead.
    pub fn maybe_store(&mut self, x: f64, y: f64, nutrient: f64, tick: u64) {
        // Check if near an existing landmark
        for landmark in self.landmarks.iter_mut().flatten() {
            if landmark.distance_to(x, y) < LANDMARK_VISIT_RADIUS {
                // Update existing landmark
                landmark.refresh(nutrient, tick);
                return;
            }
        }

        // Find an empty slot or the least valuable landmark
        let mut target_index = None;
        let mut min_value = f64::MAX;

        for (i, slot) in self.landmarks.iter().enumerate() {
            match slot {
                None => {
                    target_index = Some(i);
                    break; // Empty slot found, use it
                }
                Some(landmark) => {
                    let value = landmark.value();
                    if value < min_value {
                        min_value = value;
                        target_index = Some(i);
                    }
                }
            }
        }

        // Store if we found a slot and the new landmark is more valuable
        if let Some(i) = target_index {
            let new_value = nutrient; // New landmarks have reliability 1.0
            if self.landmarks[i].is_none() || new_value > min_value {
                self.landmarks[i] = Some(Landmark::new(x, y, nutrient, tick));
            }
        }
    }

    /// Decays the reliability of all landmarks.
    pub fn decay_all(&mut self) {
        for slot in &mut self.landmarks {
            if let Some(landmark) = slot {
                landmark.decay();
                // Remove landmarks with very low reliability
                if landmark.reliability < 0.01 {
                    *slot = None;
                }
            }
        }
    }

    /// Updates a landmark if the agent is visiting it.
    pub fn update_on_visit(&mut self, x: f64, y: f64, nutrient: f64, tick: u64) {
        for landmark in self.landmarks.iter_mut().flatten() {
            if landmark.distance_to(x, y) < LANDMARK_VISIT_RADIUS {
                landmark.refresh(nutrient, tick);
            }
        }
    }

    /// Returns the best landmark to navigate toward.
    ///
    /// "Best" is defined as highest value (nutrient * reliability).
    #[must_use]
    pub fn best_landmark(&self) -> Option<&Landmark> {
        self.landmarks
            .iter()
            .filter_map(|slot| slot.as_ref())
            .max_by(|a, b| a.value().total_cmp(&b.value()))
    }

    /// Returns the best landmark excluding a given radius from current position.
    ///
    /// Useful for finding a landmark to navigate TO (not the one we're at).
    #[must_use]
    pub fn best_distant_landmark(&self, x: f64, y: f64, min_distance: f64) -> Option<&Landmark> {
        self.landmarks
            .iter()
            .filter_map(|slot| slot.as_ref())
            .filter(|l| l.distance_to(x, y) >= min_distance)
            .max_by(|a, b| a.value().total_cmp(&b.value()))
    }

    /// Returns an iterator over all stored landmarks.
    pub fn iter(&self) -> impl Iterator<Item = &Landmark> {
        self.landmarks.iter().filter_map(|slot| slot.as_ref())
    }

    /// Clears all landmarks.
    pub fn clear(&mut self) {
        self.landmarks = [None; MAX_LANDMARKS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landmark_creation() {
        let lm = Landmark::new(50.0, 25.0, 0.9, 100);
        assert_eq!(lm.x, 50.0);
        assert_eq!(lm.y, 25.0);
        assert_eq!(lm.peak_nutrient, 0.9);
        assert_eq!(lm.reliability, 1.0);
    }

    #[test]
    fn test_landmark_distance() {
        let lm = Landmark::new(0.0, 0.0, 0.9, 0);
        assert!((lm.distance_to(3.0, 4.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_landmark_decay() {
        let mut lm = Landmark::new(50.0, 25.0, 0.9, 0);
        let initial = lm.reliability;
        lm.decay();
        assert!(lm.reliability < initial);
        assert!(lm.reliability > 0.99); // LANDMARK_DECAY = 0.995
    }

    #[test]
    fn test_episodic_memory_storage() {
        let mut mem = EpisodicMemory::new();
        assert_eq!(mem.count(), 0);

        mem.maybe_store(10.0, 10.0, 0.8, 0);
        assert_eq!(mem.count(), 1);

        mem.maybe_store(50.0, 25.0, 0.9, 1);
        assert_eq!(mem.count(), 2);
    }

    #[test]
    fn test_episodic_memory_nearby_update() {
        let mut mem = EpisodicMemory::new();
        mem.maybe_store(10.0, 10.0, 0.7, 0);

        // Store near the same location - should update, not add
        mem.maybe_store(11.0, 11.0, 0.9, 1);
        assert_eq!(mem.count(), 1);

        // Peak nutrient should be updated to higher value
        let best = mem.best_landmark().unwrap();
        assert!((best.peak_nutrient - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_episodic_memory_best_landmark() {
        let mut mem = EpisodicMemory::new();
        mem.maybe_store(10.0, 10.0, 0.6, 0);
        mem.maybe_store(50.0, 25.0, 0.9, 1);
        mem.maybe_store(80.0, 40.0, 0.7, 2);

        let best = mem.best_landmark().unwrap();
        assert!((best.peak_nutrient - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_decay_removes_stale_landmarks() {
        let mut mem = EpisodicMemory::new();
        mem.maybe_store(10.0, 10.0, 0.8, 0);

        // Decay many times until reliability < 0.01
        for _ in 0..1000 {
            mem.decay_all();
        }

        assert_eq!(mem.count(), 0);
    }

    #[test]
    fn test_best_distant_landmark() {
        let mut mem = EpisodicMemory::new();
        mem.maybe_store(10.0, 10.0, 0.9, 0);
        mem.maybe_store(50.0, 25.0, 0.8, 1);

        // From position near first landmark, best distant should be second
        let best = mem.best_distant_landmark(11.0, 11.0, 10.0).unwrap();
        assert!((best.x - 50.0).abs() < 1e-10);
    }
}
