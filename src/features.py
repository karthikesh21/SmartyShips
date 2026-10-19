"""
Feature engineering and preprocessing pipeline for Aiqon SmartShip AI.
Ensures zero target leakage and produces clean feature matrices for training and inference.
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explicit list of pre-shipment operational features
CATEGORICAL_FEATURES = ["Origin_Warehouse", "Destination", "Carrier"]
NUMERICAL_FEATURES = ["Distance_miles", "Weight_kg", "Shipment_Month", "Shipment_DayOfWeek"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Target variable definition
TARGET_COLUMN = "delivery_duration_days"
TARGET_COLUMN_HOURS = "delivery_duration_hours"

# Explicit list of columns excluded to prevent target leakage or ID memorization
EXCLUDED_COLUMNS = [
    "Shipment_ID",           # Non-predictive unique tracking identifier
    "Delivery_Date",         # Post-hoc outcome; causes direct target leakage
    "Delivery_Date_Parsed",  # Post-hoc outcome
    "Status",                # Recorded after delivery (e.g. Delivered, Delayed, Lost)
    "Transit_Days",          # In dataset, identical to actual duration for delivered shipments
    "Cost",                  # Billing metric often settled post-delivery, contains missing values
    "is_valid_training_sample"
]

def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer for pre-shipment features.
    Uses OneHotEncoder with handle_unknown='ignore' for robust production inference.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts features and target from the processed dataframe, filtering for valid samples.
    """
    if "is_valid_training_sample" in df.columns:
        valid_df = df[df["is_valid_training_sample"] == True].copy()
    else:
        valid_df = df.copy()

    # Ensure temporal features exist
    if "Shipment_Month" not in valid_df.columns:
        s_date = pd.to_datetime(valid_df["Shipment_Date"], errors="coerce")
        valid_df["Shipment_Month"] = s_date.dt.month
        valid_df["Shipment_DayOfWeek"] = s_date.dt.dayofweek

    X = valid_df[FEATURE_COLUMNS].copy()
    y = valid_df[TARGET_COLUMN].copy()
    return X, y

def build_inference_dataframe(
    origin: str,
    destination: str,
    carrier: str,
    distance_miles: float,
    weight_kg: float,
    departure_timestamp: pd.Timestamp
) -> pd.DataFrame:
    """
    Creates a single-row DataFrame suitable for the trained pipeline from input parameters.
    """
    record = {
        "Origin_Warehouse": str(origin).strip(),
        "Destination": str(destination).strip(),
        "Carrier": str(carrier).strip(),
        "Distance_miles": float(distance_miles),
        "Weight_kg": float(weight_kg),
        "Shipment_Month": int(departure_timestamp.month),
        "Shipment_DayOfWeek": int(departure_timestamp.dayofweek),
    }
    return pd.DataFrame([record])[FEATURE_COLUMNS]
