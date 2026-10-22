"""
Explainability and Delay Risk Categorization module for SmartShip.
Categorizes schedule-relative delay into operational risk tiers and provides
concise dispatch summaries of contributing route factors.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Default operational risk thresholds in hours relative to target delivery
DEFAULT_THRESHOLDS = {
    "on_time_max": 0.0,
    "low_risk_max": 12.0,
    "moderate_risk_max": 24.0
}

RISK_COLORS = {
    "On Time / Early": "#10b981",    # Emerald Green
    "Low Risk": "#f59e0b",           # Amber
    "Moderate Risk": "#f97316",      # Bright Orange
    "High Risk": "#ef4444",          # Crimson Red
    "Schedule Not Set": "#64748b"     # Slate Gray
}

def categorize_delay_risk(
    estimated_delay_hours: Optional[float],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Categorizes estimated delay into operational risk tiers based on configured thresholds.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if estimated_delay_hours is None:
        return {
            "category": "Schedule Not Set",
            "badge_color": RISK_COLORS["Schedule Not Set"],
            "description": "No expected arrival time provided. Showing raw duration estimate.",
            "threshold_applied": "N/A"
        }

    delay = float(estimated_delay_hours)
    t_on_time = thresholds.get("on_time_max", 0.0)
    t_low = thresholds.get("low_risk_max", 12.0)
    t_mod = thresholds.get("moderate_risk_max", 24.0)

    if delay <= t_on_time:
        category = "On Time / Early"
        desc = f"Arrival is {abs(delay):.1f} hours ahead of or on the promised schedule."
        rule = f"delay <= {t_on_time} hrs"
    elif delay <= t_low:
        category = "Low Risk"
        desc = f"Minor expected delay of {delay:.1f} hours, within standard operational buffer."
        rule = f"{t_on_time} < delay <= {t_low} hrs"
    elif delay <= t_mod:
        category = "Moderate Risk"
        desc = f"Expected delay of {delay:.1f} hours exceeds routine buffer tolerances."
        rule = f"{t_low} < delay <= {t_mod} hrs"
    else:
        category = "High Risk"
        desc = f"Critical delay of {delay:.1f} hours (> {t_mod}h). High probability of SLA breach."
        rule = f"delay > {t_mod} hrs"

    return {
        "category": category,
        "badge_color": RISK_COLORS.get(category, "#64748b"),
        "description": desc,
        "threshold_applied": rule,
        "delay_hours": round(delay, 2)
    }

def get_feature_importances(pipeline) -> pd.DataFrame:
    """
    Extracts and aggregates feature importance from the fitted Scikit-learn Pipeline.
    Rolls up one-hot encoded dummy columns into parent operational dimensions.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["regressor"]

    # Extract transformed feature names
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(["Origin_Warehouse", "Destination", "Carrier"])
    num_names = ["Distance_miles", "Weight_kg", "Shipment_Month", "Shipment_DayOfWeek"]
    all_feature_names = list(cat_names) + list(num_names)

    raw_importances = model.feature_importances_

    # Aggregate back to parent features
    parent_map = {}
    for name, imp in zip(all_feature_names, raw_importances):
        if name.startswith("Origin_Warehouse"):
            parent = "Origin Warehouse"
        elif name.startswith("Destination"):
            parent = "Destination City"
        elif name.startswith("Carrier"):
            parent = "Carrier"
        elif name == "Distance_miles":
            parent = "Transit Distance (miles)"
        elif name == "Weight_kg":
            parent = "Shipment Weight (kg)"
        elif name == "Shipment_Month":
            parent = "Departure Month"
        elif name == "Shipment_DayOfWeek":
            parent = "Day of Week"
        else:
            parent = name
        parent_map[parent] = parent_map.get(parent, 0.0) + imp

    imp_df = pd.DataFrame([
        {"Feature": k, "Importance": v} for k, v in parent_map.items()
    ]).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Normalize percentage
    total = imp_df["Importance"].sum()
    if total > 0:
        imp_df["Importance_Pct"] = (imp_df["Importance"] / total) * 100
    else:
        imp_df["Importance_Pct"] = 0.0

    return imp_df

def generate_plain_english_explanation(
    prediction_result: Dict[str, Any],
    risk_info: Dict[str, Any],
    importance_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generates a concise operational dispatch summary of the ETA and risk prediction.
    """
    origin = prediction_result.get("origin", "Origin")
    dest = prediction_result.get("destination", "Destination")
    carrier = prediction_result.get("carrier", "Carrier")
    dist = prediction_result.get("distance_miles", 0)
    weight = prediction_result.get("weight_kg", 0)
    duration_days = prediction_result.get("predicted_duration_days", 0)
    duration_hours = prediction_result.get("predicted_duration_hours", 0)
    delay_hours = prediction_result.get("estimated_delay_hours")
    category = risk_info.get("category", "Unknown")

    top_feature = importance_df.iloc[0]["Feature"] if importance_df is not None and not importance_df.empty else "Transit Distance"

    lines = []
    lines.append(f"**Route**: {origin} to {dest} ({dist:,.0f} miles via {carrier}, cargo weight {weight:.1f} kg).")
    lines.append(f"**Estimated Transit**: **{duration_days:.1f} days** ({duration_hours:.1f} hours).")

    if delay_hours is not None:
        if delay_hours <= 0:
            lines.append(f"**Schedule Status**: **{category}** — delivery expected with a **{abs(delay_hours):.1f}h buffer** before the target SLA.")
        else:
            lines.append(f"**Schedule Status**: **{category}** — delivery projected **{delay_hours:.1f} hours past** the target SLA.")
    else:
        lines.append("**Schedule Status**: No delivery deadline specified; estimate reflects nominal transit time.")

    lines.append(f"**Primary Driver**: **{top_feature}** has the strongest statistical weighting on this corridor.")

    return "\n\n".join(lines)
