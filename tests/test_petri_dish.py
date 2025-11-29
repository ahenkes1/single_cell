import math
import pytest
from protozoa import PetriDish

def test_petri_dish_initialization():
    dish = PetriDish(width=100.0, height=100.0)
    assert dish.width == 100.0
    assert dish.height == 100.0
    assert isinstance(dish.sources, list)
    assert 5 <= len(dish.sources) <= 10
    
    for source in dish.sources:
        assert 'x' in source
        assert 'y' in source
        assert 'radius' in source
        assert 'intensity' in source

def test_get_concentration():
    dish = PetriDish(width=100.0, height=100.0)
    # Clear sources to test specific scenarios
    dish.sources = []
    
    # Add a known source
    dish.sources.append({'x': 50.0, 'y': 50.0, 'radius': 10.0, 'intensity': 1.0})
    
    # Test center
    c_center = dish.get_concentration(50.0, 50.0)
    assert math.isclose(c_center, 1.0, abs_tol=0.01)
    
    # Test far away
    c_far = dish.get_concentration(0.0, 0.0)
    assert c_far < 0.1
    
    # Test multiple sources summing
    dish.sources.append({'x': 50.0, 'y': 50.0, 'radius': 10.0, 'intensity': 0.5})
    # Since it's summed and clipped to 1.0
    c_sum = dish.get_concentration(50.0, 50.0)
    assert math.isclose(c_sum, 1.0, abs_tol=0.01) # clipped

def test_update_dynamics():
    dish = PetriDish(width=100.0, height=100.0)
    # Set fixed sources
    dish.sources = [{'x': 50.0, 'y': 50.0, 'radius': 10.0, 'intensity': 1.0}]
    
    initial_intensity = dish.sources[0]['intensity']
    initial_x = dish.sources[0]['x']
    
    dish.update()
    
    # Check decay
    assert dish.sources[0]['intensity'] < initial_intensity
    
    # Check brownian motion (might stay same if random is 0, but unlikely with float)
    # We can't strictly assert changed without mocking random, but we can check it's still valid
    assert 0 <= dish.sources[0]['x'] <= 100.0

def test_regrowth():
    dish = PetriDish(width=100.0, height=100.0)
    # Source about to die
    dish.sources = [{'x': 50.0, 'y': 50.0, 'radius': 10.0, 'intensity': 0.001}]
    
    dish.update()
    
    # Should be replaced or intensity reset (implementation detail: respawn at new location with high intensity)
    # If it respawned, intensity should be high
    assert dish.sources[0]['intensity'] > 0.1
