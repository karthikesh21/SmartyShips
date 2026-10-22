"""
Delay Propagation Simulator for SmartShip.
Simulates multi-stage supply chain timelines, propagates upstream disruptions
downstream without double counting.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import pandas as pd

# Documented nominal stage duration distribution proportions
# Sums to 1.0 (100% of total predicted delivery duration)
STAGE_PROPORTIONS = {
    "Origin Dispatch": 0.15,
    "Main Transit": 0.55,
    "Sorting / Transfer Hub": 0.20,
    "Final Delivery": 0.10
}

STAGE_DESCRIPTIONS = {
    "Origin Dispatch": "Warehouse picking, palletizing, carrier scanning, and departure queue.",
    "Main Transit": "Long-haul interstate road freight, rail, or air trunkline movement.",
    "Sorting / Transfer Hub": "Regional break-bulk sorting, cross-docking, and destination routing.",
    "Final Delivery": "Last-mile courier dispatch, local transit, and customer drop-off."
}

def simulate_stage_propagation(
    predicted_duration_hours: float,
    departure_timestamp: datetime,
    delay_stage_name: Optional[str] = None,
    introduced_delay_hours: float = 0.0,
    recovery_buffer_hours: float = 0.0
) -> Dict[str, Any]:
    """
    Simulates stage-level timeline and propagates disruptions downstream.
    
    Data Provenance Guarantee:
    - Departure timestamp: [Observed / User Input]
    - Total predicted duration: [Model Prediction]
    - Stage breakdown & delay propagation: [SIMULATED - For Demonstration]
    """
    if predicted_duration_hours <= 0:
        predicted_duration_hours = 24.0

    if departure_timestamp.tzinfo is None:
        departure_timestamp = departure_timestamp.replace(tzinfo=timezone.utc)

    # 1. Calculate nominal baseline timeline
    stages_order = list(STAGE_PROPORTIONS.keys())
    nominal_timeline = []
    current_nominal_time = departure_timestamp

    for stage in stages_order:
        stage_hrs = predicted_duration_hours * STAGE_PROPORTIONS[stage]
        end_nominal_time = current_nominal_time + timedelta(hours=stage_hrs)
        nominal_timeline.append({
            "stage": stage,
            "duration_hours": round(stage_hrs, 2),
            "start_time": current_nominal_time,
            "end_time": end_nominal_time,
            "provenance": "[SIMULATED Stage Breakdown]"
        })
        current_nominal_time = end_nominal_time

    nominal_final_eta = current_nominal_time

    # 2. Propagate introduced delay downstream without double counting
    simulated_stages = []
    cumulative_delay_hours = 0.0
    current_actual_time = departure_timestamp

    for idx, stage in enumerate(stages_order):
        nominal_stage_duration = predicted_duration_hours * STAGE_PROPORTIONS[stage]
        
        # Check if delay is injected at this stage
        injected_here = 0.0
        if delay_stage_name == stage and introduced_delay_hours > 0:
            injected_here = float(introduced_delay_hours)
            cumulative_delay_hours += injected_here

        # Optional recovery buffer in downstream stages (cannot reduce delay below 0)
        recovered_here = 0.0
        if delay_stage_name != stage and cumulative_delay_hours > 0 and recovery_buffer_hours > 0:
            # Downstream stage can absorb buffer up to available recovery
            recovered_here = min(cumulative_delay_hours, recovery_buffer_hours / len(stages_order))
            cumulative_delay_hours -= recovered_here

        # Duration of this stage with any local injection
        actual_stage_duration = nominal_stage_duration + injected_here - recovered_here
        start_time = current_actual_time
        end_time = start_time + timedelta(hours=actual_stage_duration)
        current_actual_time = end_time

        simulated_stages.append({
            "Stage": stage,
            "Description": STAGE_DESCRIPTIONS[stage],
            "Nominal_Duration_Hours": round(nominal_stage_duration, 2),
            "Injected_Delay_Hours": round(injected_here, 2),
            "Recovered_Buffer_Hours": round(recovered_here, 2),
            "Actual_Duration_Hours": round(actual_stage_duration, 2),
            "Cumulative_Delay_Hours": round(cumulative_delay_hours, 2),
            "Nominal_Start": nominal_timeline[idx]["start_time"],
            "Nominal_End": nominal_timeline[idx]["end_time"],
            "Simulated_Start": start_time,
            "Simulated_End": end_time,
            "Provenance": "[SIMULATED - Downstream Propagation]"
        })

    propagated_final_eta = current_actual_time
    total_net_delay = (propagated_final_eta - nominal_final_eta).total_seconds() / 3600.0

    return {
        "nominal_eta_utc": nominal_final_eta.isoformat(),
        "propagated_eta_utc": propagated_final_eta.isoformat(),
        "total_nominal_hours": round(predicted_duration_hours, 2),
        "total_simulated_hours": round(predicted_duration_hours + total_net_delay, 2),
        "net_propagated_delay_hours": round(total_net_delay, 2),
        "disrupted_stage": delay_stage_name if introduced_delay_hours > 0 else "None",
        "stages": simulated_stages,
        "notes": (
            "Disclaimer: Intermediate stage timestamps and delay propagation are simulated "
            "for demonstration purposes. The Kaggle dataset contains only shipment and delivery dates; "
            "no checkpoint telemetry exists in the raw historical records."
        )
    }

def get_real_deployment_roadmap() -> List[Dict[str, str]]:
    """
    Returns documentation of how a production deployment would replace simulated
    stages with genuine enterprise logistics telemetry.
    """
    return [
        {
            "Data Source": "Telematics & GPS / ELD",
            "Integration": "Fleet ELD (Samsara / Geotab API)",
            "Stage Impact": "Real-time truck coordinates, road speed, traffic congestion slowdowns."
        },
        {
            "Data Source": "EDI 214 Shipment Status Messages",
            "Integration": "Carrier EDI / API Feeds (FedEx, UPS, Freight brokers)",
            "Stage Impact": "Actual scan timestamps at origin dock, break-bulk hubs, and out-for-delivery."
        },
        {
            "Data Source": "Customs & Port Telematics",
            "Integration": "Automated Commercial Environment (CBP ACE) / AIS Vessel Feeds",
            "Stage Impact": "Actual customs hold durations, berth wait times, and terminal dwell."
        },
        {
            "Data Source": "Live Weather & Road Hazards",
            "Integration": "NOAA Weather API & DOT 511 Road Advisories",
            "Stage Impact": "Dynamic transit velocity adjustments based on blizzards, floods, or closures."
        }
    ]
