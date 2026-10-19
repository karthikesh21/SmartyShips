"""
Unit tests for ETA prediction, schedule-relative delay, and risk categorization.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.predict import (
    load_model,
    predict_delivery_duration,
    predict_eta_and_delay,
    normalize_timestamp
)
from src.explain import (
    categorize_delay_risk,
    get_feature_importances,
    DEFAULT_THRESHOLDS
)

MODEL_PATH = "models/eta_model.joblib"

def test_model_artifact_exists():
    """Verify trained model exists at models/eta_model.joblib."""
    assert Path(MODEL_PATH).exists(), f"Model artifact missing at {MODEL_PATH}"

def test_prediction_output_finite_and_positive():
    """Verify prediction outputs are finite, positive, and realistic."""
    res = predict_delivery_duration(
        origin="Warehouse_MIA",
        destination="San Francisco",
        carrier="UPS",
        distance_miles=2600.0,
        weight_kg=25.0,
        departure_timestamp="2024-04-01 08:00:00"
    )
    assert np.isfinite(res["predicted_duration_days"])
    assert np.isfinite(res["predicted_duration_hours"])
    assert res["predicted_duration_days"] > 0
    assert res["predicted_duration_hours"] > 0
    # Sanity check: interstate shipment duration between 0.5 and 30 days
    assert 0.5 <= res["predicted_duration_days"] <= 30.0

def test_eta_timestamp_arithmetic():
    """Verify predicted ETA equals departure timestamp plus predicted duration."""
    departure_str = "2024-06-15T09:00:00Z"
    dep_dt = normalize_timestamp(departure_str)

    res = predict_eta_and_delay(
        origin="Warehouse_CHI",
        destination="New York",
        carrier="FedEx",
        distance_miles=800.0,
        weight_kg=12.0,
        departure_timestamp=dep_dt
    )

    pred_hours = res["predicted_duration_hours"]
    expected_eta = dep_dt + timedelta(hours=pred_hours)
    actual_eta = normalize_timestamp(res["predicted_eta_utc"])

    # Difference in seconds should be zero
    assert abs((expected_eta - actual_eta).total_seconds()) < 1.0

def test_schedule_delay_calculation():
    """Verify delay is positive when late, negative when early, and zero when on time."""
    departure = datetime(2024, 5, 1, 10, 0, tzinfo=timezone.utc)
    
    # Run prediction first to get ETA
    base_res = predict_eta_and_delay(
        origin="Warehouse_LA",
        destination="Dallas",
        carrier="DHL",
        distance_miles=1400.0,
        weight_kg=30.0,
        departure_timestamp=departure
    )
    predicted_eta = normalize_timestamp(base_res["predicted_eta_utc"])

    # Case A: User expects arrival exactly at predicted ETA
    res_exact = predict_eta_and_delay(
        origin="Warehouse_LA",
        destination="Dallas",
        carrier="DHL",
        distance_miles=1400.0,
        weight_kg=30.0,
        departure_timestamp=departure,
        expected_arrival_timestamp=predicted_eta
    )
    assert abs(res_exact["estimated_delay_hours"]) < 0.01
    assert res_exact["is_delayed"] is False

    # Case B: User expects arrival 10 hours after predicted ETA (Early delivery)
    exp_early = predicted_eta + timedelta(hours=10)
    res_early = predict_eta_and_delay(
        origin="Warehouse_LA",
        destination="Dallas",
        carrier="DHL",
        distance_miles=1400.0,
        weight_kg=30.0,
        departure_timestamp=departure,
        expected_arrival_timestamp=exp_early
    )
    assert res_early["estimated_delay_hours"] < 0
    assert pytest.approx(res_early["estimated_delay_hours"], 0.1) == -10.0
    assert res_early["is_delayed"] is False

    # Case C: User expects arrival 15 hours before predicted ETA (Late delivery)
    exp_late = predicted_eta - timedelta(hours=15)
    res_late = predict_eta_and_delay(
        origin="Warehouse_LA",
        destination="Dallas",
        carrier="DHL",
        distance_miles=1400.0,
        weight_kg=30.0,
        departure_timestamp=departure,
        expected_arrival_timestamp=exp_late
    )
    assert res_late["estimated_delay_hours"] > 0
    assert pytest.approx(res_late["estimated_delay_hours"], 0.1) == 15.0
    assert res_late["is_delayed"] is True

def test_risk_categorization_thresholds():
    """Verify delay risk categorization according to documented thresholds."""
    assert categorize_delay_risk(-2.0)["category"] == "On Time / Early"
    assert categorize_delay_risk(0.0)["category"] == "On Time / Early"
    assert categorize_delay_risk(6.0)["category"] == "Low Risk"
    assert categorize_delay_risk(12.0)["category"] == "Low Risk"
    assert categorize_delay_risk(18.0)["category"] == "Moderate Risk"
    assert categorize_delay_risk(24.0)["category"] == "Moderate Risk"
    assert categorize_delay_risk(36.0)["category"] == "High Risk"
    assert categorize_delay_risk(None)["category"] == "Schedule Not Set"

def test_missing_model_file_raises_clear_error():
    """Verify loading from a nonexistent file raises FileNotFoundError with actionable message."""
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        load_model("models/nonexistent_model_file.joblib")

def test_feature_importance_extraction():
    """Verify feature importances are extracted from the pipeline cleanly."""
    model = load_model(MODEL_PATH)
    imp_df = get_feature_importances(model)
    assert isinstance(imp_df, pd.DataFrame)
    assert "Feature" in imp_df.columns
    assert "Importance_Pct" in imp_df.columns
    assert len(imp_df) > 0
    assert pytest.approx(imp_df["Importance_Pct"].sum(), 0.1) == 100.0
