"""
Data inspection script for SmartShip.
Audits data/logistics_shipments_dataset.csv and reports dataset statistics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def inspect_dataset(csv_path: str = "data/logistics_shipments_dataset.csv") -> dict:
    path = Path(csv_path)
    if not path.exists():
        # Check parent folder if running from different working directory
        alt_path = Path("logistics_shipments_dataset.csv")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path} or {alt_path}")

    print("=" * 60)
    print(f"INSPECTING DATASET: {path.resolve()}")
    print("=" * 60)

    df = pd.read_csv(path)
    num_rows, num_cols = df.shape
    print(f"\n1. Dimensions: {num_rows} rows, {num_cols} columns")

    print("\n2. Columns and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:20s}: {dtype}")

    print("\n3. First 5 Rows:")
    print(df.head())

    print("\n4. Missing Values per Column:")
    nulls = df.isnull().sum()
    for col, count in nulls.items():
        pct = (count / num_rows) * 100
        print(f"   - {col:20s}: {count} ({pct:.2f}%)")

    num_duplicates = df.duplicated().sum()
    print(f"\n5. Duplicate Rows: {num_duplicates}")

    id_duplicates = df["Shipment_ID"].duplicated().sum() if "Shipment_ID" in df.columns else 0
    print(f"   Duplicate Shipment_IDs: {id_duplicates}")

    print("\n6. Shipment Status Distribution:")
    if "Status" in df.columns:
        status_counts = df["Status"].value_counts(dropna=False)
        for status, count in status_counts.items():
            pct = (count / num_rows) * 100
            print(f"   - {str(status):20s}: {count:5d} ({pct:.2f}%)")

    print("\n7. Date Field Analysis:")
    for date_col in ["Shipment_Date", "Delivery_Date"]:
        if date_col in df.columns:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            invalid_dates = parsed.isna().sum() - df[date_col].isna().sum()
            min_date = parsed.min()
            max_date = parsed.max()
            print(f"   - {date_col:15s}: Min={min_date}, Max={max_date}, Unparseable={invalid_dates}")

    # Inspect date order and transit days
    if "Shipment_Date" in df.columns and "Delivery_Date" in df.columns:
        s_date = pd.to_datetime(df["Shipment_Date"], errors="coerce")
        d_date = pd.to_datetime(df["Delivery_Date"], errors="coerce")
        diff_days = (d_date - s_date).dt.days
        neg_count = (diff_days < 0).sum()
        zero_count = (diff_days == 0).sum()
        valid_pos = (diff_days > 0).sum()
        print(f"\n8. Date Difference (Delivery_Date - Shipment_Date):")
        print(f"   - Negative durations (Delivery < Shipment): {neg_count}")
        print(f"   - Zero duration (Same day): {zero_count}")
        print(f"   - Positive duration: {valid_pos}")
        print(f"   - Summary statistics of (Delivery - Shipment) in days:")
        print(diff_days.describe())

    if "Transit_Days" in df.columns:
        print("\n9. Dataset 'Transit_Days' column summary:")
        print(df["Transit_Days"].describe())
        if "Shipment_Date" in df.columns and "Delivery_Date" in df.columns:
            matches = (df["Transit_Days"] == diff_days).sum()
            mismatches = (df["Transit_Days"] != diff_days).sum()
            print(f"   - Matches (Delivery_Date - Shipment_Date): {matches}")
            print(f"   - Mismatches: {mismatches}")

    print("\n10. Numerical Features Summary:")
    num_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[num_cols_list].describe())

    print("\n11. Categorical Feature Unique Values:")
    for cat_col in ["Origin_Warehouse", "Destination", "Carrier"]:
        if cat_col in df.columns:
            uniques = df[cat_col].unique()
            print(f"   - {cat_col:18s} ({len(uniques)} unique): {list(uniques)[:10]}")

    results = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": list(df.columns),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(num_duplicates),
        "status_distribution": df["Status"].value_counts().to_dict() if "Status" in df.columns else {},
        "negative_durations": int((diff_days < 0).sum()) if "Shipment_Date" in df.columns and "Delivery_Date" in df.columns else 0
    }
    return results

if __name__ == "__main__":
    inspect_dataset()
