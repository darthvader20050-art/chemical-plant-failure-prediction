import pytest
from src.features import FEATURES, validate_input

GOOD_INPUT = {
    "temperature_c": 80,
    "pressure_bar": 8,
    "vibration_mm_s": 2.0,
    "operating_hours": 5000,
    "days_since_maintenance": 60,
    "load_percent": 70,
}

def test_valid_input_passes():
    validate_input(GOOD_INPUT)

def test_missing_feature_raises():
    bad = GOOD_INPUT.copy()
    del bad["pressure_bar"]
    with pytest.raises(ValueError, match="Missing required features"):
        validate_input(bad)

def test_out_of_range_value_raises():
    bad = GOOD_INPUT.copy()
    bad["temperature_c"] = 999
    with pytest.raises(ValueError, match="temperature_c"):
        validate_input(bad)

def test_non_numeric_value_raises():
    bad = GOOD_INPUT.copy()
    bad["pressure_bar"] = "high"
    with pytest.raises(ValueError, match="must be numeric"):
        validate_input(bad)
