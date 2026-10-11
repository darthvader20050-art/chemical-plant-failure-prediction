import pytest
from predict import predict, MODEL_PATH

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train model first")
def test_prediction_schema():
    values = {
        "temperature_c": 80,
        "pressure_bar": 8,
        "vibration_mm_s": 2.0,
        "operating_hours": 5000,
        "days_since_maintenance": 60,
        "load_percent": 70,
    }
    result = predict(values)
    assert 0 <= result["risk_score"] <= 1
    assert result["prediction"] in {"potential_failure_risk", "lower_predicted_risk"}
    assert "disclaimer" in result
