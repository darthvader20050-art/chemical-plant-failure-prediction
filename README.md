# Chemical Plant Equipment Failure Prediction

A portfolio project applying supervised machine learning to an industrial-maintenance use case. It compares a Logistic Regression baseline with a Random Forest model, uses stratified cross-validation, reports classification metrics, validates prediction inputs, and exposes predictions through a small FastAPI service.

> **Data limitation:** The included 1,200-row dataset is synthetic and generated for learning. Metrics measure performance on this synthetic dataset only; they do not demonstrate real-world industrial reliability. Never use this prototype for operational or safety decisions.

## Capabilities
- Data preparation and stratified train/test split
- Model comparison: Logistic Regression vs Random Forest
- 5-fold cross-validation using ROC-AUC on the training split
- Holdout metrics: accuracy, precision, recall, F1, ROC-AUC and confusion matrix
- Feature importance export
- Input validation and command-line prediction
- FastAPI endpoints: `/health` and `/predict`
- Pytest checks for validation and prediction output

## Windows setup
Use Python 3.10 or 3.11 if available:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv` with a supported Python version.

## Train and evaluate
```powershell
python train_model.py
```

This creates:
- `models/failure_model.joblib`
- `reports/metrics.json`
- `reports/feature_importance.csv`

## Predict from the command line
```powershell
python predict.py --temperature 95 --pressure 11 --vibration 5.5 --hours 9000 --maintenance-days 180 --load 92
```

## Run tests
```powershell
pytest -q
```

## Start the API
Train the model first, then run:

```powershell
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API interactively.

Example JSON for `POST /predict`:
```json
{
  "temperature_c": 95,
  "pressure_bar": 11,
  "vibration_mm_s": 5.5,
  "operating_hours": 9000,
  "days_since_maintenance": 180,
  "load_percent": 92
}
```

## Repository layout
```text
chemical_plant_failure_prediction_pro/
├── data/equipment_data.csv
├── src/features.py
├── tests/test_features.py
├── tests/test_prediction.py
├── train_model.py
├── predict.py
├── api.py
├── requirements.txt
└── README.md
```


