import math
import pytest
from protozoa import Protozoa, PetriDish, PARAMS

def test_protozoa_initialization():
    p = Protozoa(x=50.0, y=50.0)
    assert p.x == 50.0
    assert p.y == 50.0
    assert 0 <= p.angle <= 2 * math.pi
    assert p.speed == 0.0
    
    # Hyperparameters are now global constants, checking them via PARAMS
    assert PARAMS['target'] == 0.8
    assert PARAMS['sensor_dist'] == 2.0

def test_sense():
    p = Protozoa(x=50.0, y=50.0)
    p.angle = 0.0 # Facing East
    
    # We need to control params for the test logic
    # Since PARAMS is a dict, we can modify it temporarily
    original_dist = PARAMS['sensor_dist']
    original_angle = PARAMS['sensor_angle']
    
    PARAMS['sensor_dist'] = 2.0
    PARAMS['sensor_angle'] = math.pi / 2 # 90 degrees
    
    try:
        # Left sensor should be at (50, 50 + 2) = (50, 52)
        # Right sensor should be at (50, 50 - 2) = (50, 48)
        
        dish = PetriDish(width=100.0, height=100.0)
        
        def mock_get_concentration(x, y):
            if y > 50: return 0.9
            return 0.1
            
        dish.get_concentration = mock_get_concentration
        
        p.sense(dish)
        
        assert math.isclose(p.val_l, 0.9)
        assert math.isclose(p.val_r, 0.1)
    finally:
        # Restore params
        PARAMS['sensor_dist'] = original_dist
        PARAMS['sensor_angle'] = original_angle

def test_update_state_active_inference():
    p = Protozoa(x=50.0, y=50.0)
    p.angle = 0.0
    p.speed = 0.0
    
    # Mock sensor values
    # Case 1: High Error, Gradient present
    # Target = 0.8.
    # Let L=0.2, R=0.0. Mean=0.1. Error = 0.1 - 0.8 = -0.7 (Hungry).
    # Gradient = 0.2 - 0.0 = 0.2 (Food on Left).
    # d_theta = -lr * E * G = -0.15 * (-0.7) * (0.2) = +positive.
    # Should turn Left (positive angle change).
    
    p.val_l = 0.2
    p.val_r = 0.0
    
    # We need to set dish width/height for boundary checks
    dish_mock = type('obj', (object,), {'width': 100.0, 'height': 100.0})
    
    p.update_state(dish_mock)
    
    # Check angle increased
    assert p.angle > 0.0
    
    # Check speed increased (because |Error| > 0)
    assert p.speed > 0.0

def test_boundary_check():
    p = Protozoa(x=105.0, y=-5.0)
    p.val_l = 0.8
    p.val_r = 0.8
    
    dish_mock = type('obj', (object,), {'width': 100.0, 'height': 100.0})
    p.update_state(dish_mock)
    
    assert p.x <= 100.0
    assert p.y >= 0.0
