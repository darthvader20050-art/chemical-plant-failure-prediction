from pathlib import Path
import argparse
import joblib
import pandas as pd
from src.features import FEATURES, validate_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "failure_model.joblib"

def predict(values: dict) -> dict:
    validate_input(values)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    score = float(model.predict_proba(row)[0, 1])
    return {
        "risk_score": round(score, 4),
        "prediction": "potential_failure_risk" if score >= 0.5 else "lower_predicted_risk",
        "threshold": 0.5,
        "disclaimer": "Educational demo using synthetic data; not for real plant decisions."
    }

def main():
    parser = argparse.ArgumentParser(description="Estimate equipment failure risk.")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--vibration", type=float, required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--maintenance-days", type=float, required=True)
    parser.add_argument("--load", type=float, required=True)
    args = parser.parse_args()
    values = {
        "temperature_c": args.temperature, "pressure_bar": args.pressure,
        "vibration_mm_s": args.vibration, "operating_hours": args.hours,
        "days_since_maintenance": args.maintenance_days, "load_percent": args.load,
    }
    result = predict(values)
    print(f"Prediction: {result['prediction']}")
    print(f"Estimated risk score: {result['risk_score']:.1%}")
    print(result["disclaimer"])

if __name__ == "__main__":
    main()
