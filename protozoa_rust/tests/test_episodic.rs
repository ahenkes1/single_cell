//! Tests for episodic memory module.

use protozoa_rust::simulation::memory::{EpisodicMemory, Landmark};

#[test]
fn test_landmark_value_decay() {
    let mut lm = Landmark::new(50.0, 25.0, 0.9, 0);

    // Initial value is peak_nutrient * reliability (1.0)
    assert!((lm.value() - 0.9).abs() < 1e-10);

    // After decay, value should decrease
    lm.decay();
    assert!(lm.value() < 0.9);
    assert!(lm.value() > 0.89); // 0.9 * 0.995 = 0.8955
}

#[test]
fn test_landmark_refresh_restores_reliability() {
    let mut lm = Landmark::new(50.0, 25.0, 0.9, 0);

    // Decay several times
    for _ in 0..100 {
        lm.decay();
    }

    // Reliability should be reduced
    assert!(lm.reliability < 1.0);

    // Refresh should restore reliability
    lm.refresh(0.85, 100);
    assert_eq!(lm.reliability, 1.0);

    // Peak should update if new is higher
    lm.refresh(0.95, 101);
    assert!((lm.peak_nutrient - 0.95).abs() < 1e-10);

    // Peak should NOT update if new is lower
    lm.refresh(0.80, 102);
    assert!((lm.peak_nutrient - 0.95).abs() < 1e-10);
}

#[test]
fn test_episodic_memory_replaces_low_value_landmarks() {
    let mut mem = EpisodicMemory::new();

    // Fill memory with landmarks
    for i in 0..8 {
        #[allow(clippy::cast_precision_loss)]
        mem.maybe_store(i as f64 * 10.0, 25.0, 0.5, i as u64);
    }

    assert_eq!(mem.count(), 8);

    // Decay all to reduce reliability
    for _ in 0..200 {
        mem.decay_all();
    }

    // Now add a higher-value landmark - should replace lowest
    mem.maybe_store(85.0, 25.0, 0.95, 300);

    // The new landmark should be the best
    let best = mem.best_landmark().unwrap();
    assert!((best.peak_nutrient - 0.95).abs() < 1e-10);
    assert!((best.x - 85.0).abs() < 1e-10);
}

#[test]
fn test_episodic_memory_clear() {
    let mut mem = EpisodicMemory::new();

    mem.maybe_store(10.0, 10.0, 0.8, 0);
    mem.maybe_store(50.0, 25.0, 0.9, 1);
    assert_eq!(mem.count(), 2);

    mem.clear();
    assert_eq!(mem.count(), 0);
    assert!(mem.best_landmark().is_none());
}

#[test]
fn test_episodic_memory_iter() {
    let mut mem = EpisodicMemory::new();

    mem.maybe_store(10.0, 10.0, 0.8, 0);
    mem.maybe_store(50.0, 25.0, 0.9, 1);
    mem.maybe_store(80.0, 40.0, 0.7, 2);

    let landmarks: Vec<_> = mem.iter().collect();
    assert_eq!(landmarks.len(), 3);
}

#[test]
fn test_landmark_nearby_update_vs_new_storage() {
    let mut mem = EpisodicMemory::new();

    // Store initial landmark
    mem.maybe_store(50.0, 25.0, 0.7, 0);
    assert_eq!(mem.count(), 1);

    // Store nearby - should update existing, not add new
    mem.maybe_store(52.0, 27.0, 0.8, 1); // within LANDMARK_VISIT_RADIUS (5.0)
    assert_eq!(mem.count(), 1);

    // Verify peak was updated
    let best = mem.best_landmark().unwrap();
    assert!((best.peak_nutrient - 0.8).abs() < 1e-10);

    // Store far away - should add new
    mem.maybe_store(90.0, 40.0, 0.6, 2);
    assert_eq!(mem.count(), 2);
}

#[test]
fn test_best_distant_landmark_excludes_current() {
    let mut mem = EpisodicMemory::new();

    // High-value landmark nearby
    mem.maybe_store(10.0, 10.0, 0.95, 0);
    // Lower-value landmark far away
    mem.maybe_store(80.0, 40.0, 0.7, 1);

    // From position near first landmark
    let best = mem.best_distant_landmark(12.0, 12.0, 10.0);
    assert!(best.is_some());

    // Should return the distant one, not the nearby high-value one
    let lm = best.unwrap();
    assert!((lm.x - 80.0).abs() < 1e-10);
}
