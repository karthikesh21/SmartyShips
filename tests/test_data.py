"""
Unit and integration tests for data inspection, cleaning, and preprocessing.
Verifies data integrity, target calculation, and zero target leakage.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.inspect_data import inspect_dataset
from src.clean_data import clean_dataset
from src.features import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    EXCLUDED_COLUMNS,
    prepare_training_data
)

RAW_CSV = "data/logistics_shipments_dataset.csv"
PROCESSED_CSV = "data/processed_shipments.csv"

def test_raw_dataset_exists_and_inspectable():
    """Verify that the Kaggle dataset exists and inspect_dataset returns expected schema."""
    assert Path(RAW_CSV).exists(), f"Raw dataset missing at {RAW_CSV}"
    info = inspect_dataset(RAW_CSV)
    assert info["num_rows"] == 2000
    assert info["num_cols"] == 11
    assert "Shipment_ID" in info["columns"]
    assert "Shipment_Date" in info["columns"]
    assert "Delivery_Date" in info["columns"]

def test_cleaning_produces_processed_csv():
    """Verify clean_dataset generates processed_shipments.csv with required columns."""
    df = clean_dataset(input_path=RAW_CSV, output_path=PROCESSED_CSV)
    assert Path(PROCESSED_CSV).exists()
    assert len(df) == 2000
    assert "delivery_duration_days" in df.columns
    assert "delivery_duration_hours" in df.columns
    assert "is_valid_training_sample" in df.columns

def test_no_negative_delivery_durations_in_training_data():
    """Verify that no negative delivery durations exist in samples flagged for training."""
    df = pd.read_csv(PROCESSED_CSV)
    train_df = df[df["is_valid_training_sample"] == True]
    assert len(train_df) > 0
    assert (train_df["delivery_duration_days"] <= 0).sum() == 0, "Found non-positive durations in training samples!"
    assert (train_df["delivery_duration_hours"] <= 0).sum() == 0

def test_preprocessing_reproducibility():
    """Verify running cleaning multiple times produces identical output."""
    df1 = clean_dataset(RAW_CSV, PROCESSED_CSV)
    df2 = clean_dataset(RAW_CSV, PROCESSED_CSV)
    pd.testing.assert_frame_equal(df1, df2)

def test_zero_target_leakage_in_feature_columns():
    """
    CRITICAL TEST: Ensure post-hoc columns (Delivery_Date, Status, Transit_Days)
    are strictly excluded from predictive features.
    """
    for forbidden in ["Delivery_Date", "Delivery_Date_Parsed", "Status", "Transit_Days", "Cost", "Shipment_ID"]:
        assert forbidden not in FEATURE_COLUMNS, f"TARGET LEAKAGE DETECTED: {forbidden} is in FEATURE_COLUMNS!"

def test_training_features_preparation():
    """Verify prepare_training_data correctly extracts valid matrices."""
    df = pd.read_csv(PROCESSED_CSV)
    X, y = prepare_training_data(df)
    assert len(X) == len(y)
    assert len(X) == df["is_valid_training_sample"].sum()
    assert list(X.columns) == FEATURE_COLUMNS
    assert not y.isna().any()
    assert (y > 0).all()
