"""
ETA and Delivery Duration Prediction module for SmartShip.
Computes predicted delivery duration, estimated arrival timestamp (ETA),
and schedule-relative delay against target deadlines.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Union
import joblib
import pandas as pd
import numpy as np

from src.features import build_inference_dataframe

_MODEL_CACHE = None

def load_model(model_path: str = "models/eta_model.joblib"):
    """
    Loads and caches the trained Scikit-learn ETA pipeline.
    """
    global _MODEL_CACHE
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'python -m src.train' to train and save the pipeline."
        )
    if _MODEL_CACHE is None:
        _MODEL_CACHE = joblib.load(path)
    return _MODEL_CACHE

def normalize_timestamp(
    ts: Union[str, datetime, pd.Timestamp],
    default_tz=timezone.utc
) -> datetime:
    """
    Normalizes any datetime input to a timezone-aware UTC datetime.
    Convention: All timestamps are processed in UTC to avoid daylight saving ambiguities.
    """
    if isinstance(ts, str):
        # Support common formats
        dt = pd.to_datetime(ts).to_pydatetime()
    elif isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        raise ValueError(f"Unsupported timestamp format: {type(ts)}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    else:
        dt = dt.astimezone(default_tz)
    return dt

def predict_delivery_duration(
    origin: str,
    destination: str,
    carrier: str,
    distance_miles: float,
    weight_kg: float,
    departure_timestamp: Union[str, datetime, pd.Timestamp],
    model=None,
    model_path: str = "models/eta_model.joblib"
) -> Dict[str, float]:
    """
    Predicts delivery duration in days and hours using genuine pre-shipment features.
    """
    if model is None:
        model = load_model(model_path)

    dep_dt = normalize_timestamp(departure_timestamp)
    dep_pandas_ts = pd.Timestamp(dep_dt)

    input_df = build_inference_dataframe(
        origin=origin,
        destination=destination,
        carrier=carrier,
        distance_miles=distance_miles,
        weight_kg=weight_kg,
        departure_timestamp=dep_pandas_ts
    )

    pred_days = float(model.predict(input_df)[0])
    # Enforce physical non-negativity constraint
    pred_days = max(0.5, pred_days)
    pred_hours = pred_days * 24.0

    return {
        "predicted_duration_days": round(pred_days, 2),
        "predicted_duration_hours": round(pred_hours, 2),
    }

def predict_eta_and_delay(
    origin: str,
    destination: str,
    carrier: str,
    distance_miles: float,
    weight_kg: float,
    departure_timestamp: Union[str, datetime, pd.Timestamp],
    expected_arrival_timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None,
    model=None,
    model_path: str = "models/eta_model.joblib"
) -> Dict[str, Any]:
    """
    Full prediction pipeline:
    1. Computes predicted delivery duration.
    2. Computes predicted ETA: departure_timestamp + predicted_duration.
    3. If expected_arrival is provided, computes schedule-relative delay:
       estimated_delay_hours = predicted_eta - expected_arrival.
       (Positive = later than expectation, Negative = earlier).
    """
    dep_dt = normalize_timestamp(departure_timestamp)
    durations = predict_delivery_duration(
        origin=origin,
        destination=destination,
        carrier=carrier,
        distance_miles=distance_miles,
        weight_kg=weight_kg,
        departure_timestamp=dep_dt,
        model=model,
        model_path=model_path
    )

    pred_duration_hours = durations["predicted_duration_hours"]
    predicted_eta = dep_dt + timedelta(hours=pred_duration_hours)

    result = {
        "origin": origin,
        "destination": destination,
        "carrier": carrier,
        "distance_miles": distance_miles,
        "weight_kg": weight_kg,
        "departure_timestamp_utc": dep_dt.isoformat(),
        "predicted_duration_days": durations["predicted_duration_days"],
        "predicted_duration_hours": pred_duration_hours,
        "predicted_eta_utc": predicted_eta.isoformat(),
        "timezone": "UTC"
    }

    if expected_arrival_timestamp is not None:
        exp_dt = normalize_timestamp(expected_arrival_timestamp)
        delay_seconds = (predicted_eta - exp_dt).total_seconds()
        delay_hours = delay_seconds / 3600.0

        result["expected_arrival_utc"] = exp_dt.isoformat()
        result["estimated_delay_hours"] = round(delay_hours, 2)
        result["is_delayed"] = bool(delay_hours > 0)
    else:
        result["expected_arrival_utc"] = None
        result["estimated_delay_hours"] = None
        result["is_delayed"] = None

    return result
