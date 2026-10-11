FEATURES = [
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "operating_hours",
    "days_since_maintenance",
    "load_percent",
]
TARGET = "failure_risk"

def validate_input(values: dict) -> None:
    missing = [name for name in FEATURES if name not in values]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    for name in FEATURES:
        value = values[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
    bounds = {
        "temperature_c": (0, 250),
        "pressure_bar": (0, 100),
        "vibration_mm_s": (0, 100),
        "operating_hours": (0, 200000),
        "days_since_maintenance": (0, 5000),
        "load_percent": (0, 200),
    }
    for name, (low, high) in bounds.items():
        if not low <= values[name] <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
