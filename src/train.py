"""
Model training and evaluation pipeline for SmartShip.
Trains Baseline and RandomForestRegressor models, computes MAE and RMSE,
and serializes artifacts to models/eta_model.joblib and models/evaluation_metrics.json.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.clean_data import clean_dataset
from src.features import (
    build_preprocessor,
    prepare_training_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

def train_and_evaluate(
    data_path: str = "data/processed_shipments.csv",
    model_output_path: str = "models/eta_model.joblib",
    metrics_output_path: str = "models/evaluation_metrics.json",
    test_predictions_path: str = "models/test_predictions.csv",
    test_size: float = 0.20,
    random_state: int = 42
) -> dict:
    csv_file = Path(data_path)
    if not csv_file.exists():
        print(f"Processed dataset not found at {data_path}. Running cleaning pipeline first...")
        df = clean_dataset()
    else:
        df = pd.read_csv(csv_file)

    print("SmartShip: Training and evaluating regression models...")

    X, y = prepare_training_data(df)
    total_valid = len(X)
    print(f"Total valid training samples: {total_valid}")

    # Train / Test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # 1. Simple Baseline Predictor (Training-set Median delivery duration)
    baseline_median_days = float(np.median(y_train))
    baseline_median_hours = baseline_median_days * 24.0
    baseline_preds = np.full(shape=len(y_test), fill_value=baseline_median_days)

    baseline_mae_days = float(mean_absolute_error(y_test, baseline_preds))
    baseline_rmse_days = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
    baseline_mae_hours = baseline_mae_days * 24.0
    baseline_rmse_hours = baseline_rmse_days * 24.0
    baseline_r2 = float(r2_score(y_test, baseline_preds))

    print(f"\n--- 1. Baseline Predictor (Median Delivery Duration) ---")
    print(f"  Training Set Median : {baseline_median_days:.2f} days ({baseline_median_hours:.1f} hours)")
    print(f"  Test Set MAE        : {baseline_mae_days:.3f} days ({baseline_mae_hours:.2f} hours)")
    print(f"  Test Set RMSE       : {baseline_rmse_days:.3f} days ({baseline_rmse_hours:.2f} hours)")
    print(f"  Test Set R²         : {baseline_r2:.4f}")

    # 2. Primary Model: RandomForestRegressor in a Scikit-learn Pipeline
    preprocessor = build_preprocessor()
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf_model)
    ])

    print(f"\n--- 2. Fitting Primary RandomForestRegressor Pipeline ---")
    pipeline.fit(X_train, y_train)
    rf_preds_days = pipeline.predict(X_test)
    rf_preds_hours = rf_preds_days * 24.0

    rf_mae_days = float(mean_absolute_error(y_test, rf_preds_days))
    rf_rmse_days = float(np.sqrt(mean_squared_error(y_test, rf_preds_days)))
    rf_mae_hours = rf_mae_days * 24.0
    rf_rmse_hours = rf_rmse_days * 24.0
    rf_r2 = float(r2_score(y_test, rf_preds_days))

    print(f"  Test Set MAE        : {rf_mae_days:.3f} days ({rf_mae_hours:.2f} hours)")
    print(f"  Test Set RMSE       : {rf_rmse_days:.3f} days ({rf_rmse_hours:.2f} hours)")
    print(f"  Test Set R²         : {rf_r2:.4f}")

    # Comparison
    mae_diff_hours = baseline_mae_hours - rf_mae_hours
    rmse_diff_hours = baseline_rmse_hours - rf_rmse_hours
    print(f"\n--- 3. Performance Comparison vs Baseline ---")
    print(f"  MAE Improvement     : {mae_diff_hours:+.2f} hours (RF is {'better' if mae_diff_hours > 0 else 'worse'})")
    print(f"  RMSE Improvement    : {rmse_diff_hours:+.2f} hours (RF is {'better' if rmse_diff_hours > 0 else 'worse'})")

    # Serialize Model Pipeline
    out_model = Path(model_output_path)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_model)
    print(f"\nSaved trained pipeline to: {out_model.resolve()}")

    # Save Test Predictions for Dashboard Visualization
    # Signed error definition: prediction_error = (predicted - actual)
    # Negative values mean underprediction; positive values mean overprediction.
    prediction_error_days = rf_preds_days - y_test.values
    prediction_error_hours = prediction_error_days * 24.0

    test_eval_df = X_test.copy()
    test_eval_df["actual_days"] = y_test.values
    test_eval_df["actual_hours"] = y_test.values * 24.0
    test_eval_df["predicted_days"] = rf_preds_days
    test_eval_df["predicted_hours"] = rf_preds_hours
    test_eval_df["baseline_days"] = baseline_preds
    test_eval_df["baseline_hours"] = baseline_preds * 24.0
    test_eval_df["prediction_error_days"] = prediction_error_days
    test_eval_df["prediction_error_hours"] = prediction_error_hours
    test_eval_df["residual_days"] = prediction_error_days
    test_eval_df["residual_hours"] = prediction_error_hours

    out_preds = Path(test_predictions_path)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    test_eval_df.to_csv(out_preds, index=False)
    print(f"Saved test evaluation predictions to: {out_preds.resolve()}")

    # Build Metrics JSON
    metrics_data = {
        "dataset": {
            "total_samples": int(len(df)),
            "valid_training_samples": int(total_valid),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_used": FEATURE_COLUMNS,
            "target": TARGET_COLUMN
        },
        "baseline": {
            "name": "Median Duration Baseline",
            "median_duration_days": round(baseline_median_days, 3),
            "median_duration_hours": round(baseline_median_hours, 2),
            "mae_days": round(baseline_mae_days, 4),
            "mae_hours": round(baseline_mae_hours, 2),
            "rmse_days": round(baseline_rmse_days, 4),
            "rmse_hours": round(baseline_rmse_hours, 2),
            "r2": round(baseline_r2, 4)
        },
        "random_forest": {
            "name": "RandomForestRegressor Pipeline",
            "n_estimators": 200,
            "max_depth": 12,
            "mae_days": round(rf_mae_days, 4),
            "mae_hours": round(rf_mae_hours, 2),
            "rmse_days": round(rf_rmse_days, 4),
            "rmse_hours": round(rf_rmse_hours, 2),
            "r2": round(rf_r2, 4)
        },
        "comparison": {
            "mae_improvement_hours": round(mae_diff_hours, 2),
            "rmse_improvement_hours": round(rmse_diff_hours, 2),
            "rf_beats_baseline": bool(mae_diff_hours > 0 and rmse_diff_hours > 0)
        }
    }

    out_metrics = Path(metrics_output_path)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics JSON to: {out_metrics.resolve()}")
    print("=" * 65)

    return metrics_data

if __name__ == "__main__":
    train_and_evaluate()
