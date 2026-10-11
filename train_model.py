from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.features import FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "equipment_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

def build_logistic_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])

def build_random_forest():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'")
    X, y = df[FEATURES], df[TARGET].astype(int)
    if y.nunique() != 2:
        raise ValueError("Target must contain both classes (0 and 1)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": build_logistic_pipeline(),
        "random_forest": build_random_forest(),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=None)
        cv_results[name] = {
            "mean_roc_auc": round(float(scores.mean()), 4),
            "std_roc_auc": round(float(scores.std()), 4),
        }

    best_name = max(cv_results, key=lambda name: cv_results[name]["mean_roc_auc"])
    model = candidates[best_name]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "selected_model": best_name,
        "dataset_note": "Synthetic demonstration data; not real industrial measurements.",
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "cross_validation_roc_auc": cv_results,
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "confusion_matrix_labels": ["actual_0", "actual_1"],
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        },
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    joblib.dump(model, MODEL_DIR / "failure_model.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save feature importance for the selected model when supported.
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = abs(estimator.coef_[0])
    else:
        importances = [0] * len(FEATURES)
    importance_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    print(f"Selected model: {best_name}")
    print(f"5-fold CV ROC-AUC: {cv_results[best_name]['mean_roc_auc']} ± {cv_results[best_name]['std_roc_auc']}")
    print(json.dumps(metrics["test_metrics"], indent=2))
    print(f"Saved model to {MODEL_DIR / 'failure_model.joblib'}")
    print(f"Saved reports to {REPORT_DIR}")

if __name__ == "__main__":
    main()
