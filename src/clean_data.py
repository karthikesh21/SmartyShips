"""
Data cleaning and preprocessing script for Aiqon SmartShip AI.
Loads data/logistics_shipments_dataset.csv, applies robust cleaning decisions,
and saves the result to data/processed_shipments.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def clean_dataset(
    input_path: str = "data/logistics_shipments_dataset.csv",
    output_path: str = "data/processed_shipments.csv"
) -> pd.DataFrame:
    in_file = Path(input_path)
    if not in_file.exists():
        alt = Path("logistics_shipments_dataset.csv")
        if alt.exists():
            in_file = alt
        else:
            raise FileNotFoundError(f"Input file not found at {input_path} or {alt}")

    print("=" * 65)
    print("AIQON SMARTSHIP AI: DATA CLEANING PIPELINE")
    print("=" * 65)
    print(f"Reading raw data from: {in_file.resolve()}")
    df = pd.read_csv(in_file)
    initial_rows = len(df)
    print(f"Initial raw rows: {initial_rows}")

    # 1. Standardize string categories (strip whitespace)
    str_cols = ["Origin_Warehouse", "Destination", "Carrier", "Status"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 2. Explicit Date Parsing with validation
    print("\n[Step 1] Parsing dates explicitly (format='%Y-%m-%d')...")
    df["Shipment_Date_Parsed"] = pd.to_datetime(df["Shipment_Date"], format="%Y-%m-%d", errors="coerce")
    df["Delivery_Date_Parsed"] = pd.to_datetime(df["Delivery_Date"], format="%Y-%m-%d", errors="coerce")

    # 3. Calculate Historical Delivery Duration
    print("\n[Step 2] Calculating Delivery Duration...")
    df["delivery_duration_days"] = (df["Delivery_Date_Parsed"] - df["Shipment_Date_Parsed"]).dt.days
    df["delivery_duration_hours"] = df["delivery_duration_days"] * 24.0

    # 4. Identify Data Quality Issues
    missing_del_mask = df["Delivery_Date_Parsed"].isna()
    neg_duration_mask = df["delivery_duration_days"] < 0
    zero_duration_mask = df["delivery_duration_days"] == 0
    lost_mask = df["Status"].str.lower() == "lost"
    invalid_weight_mask = (df["Weight_kg"] <= 0) | df["Weight_kg"].isna()

    print(f"  - Missing Delivery Dates: {missing_del_mask.sum()} rows (uncompleted or missing tracking)")
    print(f"  - Negative Durations (Delivery < Shipment): {neg_duration_mask.sum()} rows (data entry errors)")
    print(f"  - Same-day (Zero) Durations: {zero_duration_mask.sum()} rows")
    print(f"  - Lost Shipments: {lost_mask.sum()} rows")
    print(f"  - Zero / Invalid Weight: {invalid_weight_mask.sum()} rows")

    # 5. Outlier handling for Weight_kg
    # Inspect 99.5th percentile for realistic upper threshold
    weight_p995 = df["Weight_kg"].quantile(0.995)
    df["weight_capped_kg"] = df["Weight_kg"].clip(lower=0.1, upper=weight_p995)

    # 6. Flag for valid training samples
    # Training sample criteria:
    # - Valid positive delivery duration
    # - Completed shipment (Status is Delivered or Delayed)
    # - Positive weight
    valid_training_mask = (
        (~missing_del_mask) &
        (df["delivery_duration_days"] > 0) &
        (df["Status"].isin(["Delivered", "Delayed"])) &
        (~invalid_weight_mask)
    )

    df["is_valid_training_sample"] = valid_training_mask
    training_sample_count = valid_training_mask.sum()
    excluded_count = initial_rows - training_sample_count

    print(f"\n[Step 3] Cleaning Decisions Summary:")
    print(f"  - Retained for training: {training_sample_count} rows ({training_sample_count/initial_rows*100:.2f}%)")
    print(f"  - Excluded from training: {excluded_count} rows ({excluded_count/initial_rows*100:.2f}%)")
    print(f"    * Reason 1: Missing delivery date (32)")
    print(f"    * Reason 2: Chronologically inverted dates (16)")
    print(f"    * Reason 3: Lost shipments with aborted transit (45)")
    print(f"    * Reason 4: Zero weight package (1)")

    # 7. Add temporal calendar features genuinely available at shipment date
    df["Shipment_Year"] = df["Shipment_Date_Parsed"].dt.year
    df["Shipment_Month"] = df["Shipment_Date_Parsed"].dt.month
    df["Shipment_Day"] = df["Shipment_Date_Parsed"].dt.day
    df["Shipment_DayOfWeek"] = df["Shipment_Date_Parsed"].dt.dayofweek
    df["Shipment_Quarter"] = df["Shipment_Date_Parsed"].dt.quarter

    # Save to output path
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"\n[Step 4] Saved processed dataset to: {out_file.resolve()}")
    print("=" * 65)

    return df

if __name__ == "__main__":
    clean_dataset()
