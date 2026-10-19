"""
Unit tests for the Delay Propagation Simulation module.
Verifies stage duration proportions, no double counting, downstream carryover,
and safe handling of missing or optional inputs.
"""

from datetime import datetime, timezone, timedelta
import pytest

from src.propagation import (
    simulate_stage_propagation,
    STAGE_PROPORTIONS,
    STAGE_DESCRIPTIONS,
    get_real_deployment_roadmap
)

def test_nominal_stage_proportions_sum_to_one():
    """Verify that stage nominal distribution weights sum to exactly 1.0 (100%)."""
    total = sum(STAGE_PROPORTIONS.values())
    assert pytest.approx(total, 0.0001) == 1.0
    assert len(STAGE_PROPORTIONS) == 4

def test_no_delay_injected_preserves_nominal_timeline():
    """Verify simulation with zero injected delay produces zero net propagated delay."""
    dep = datetime(2024, 7, 1, 8, 0, tzinfo=timezone.utc)
    res = simulate_stage_propagation(
        predicted_duration_hours=100.0,
        departure_timestamp=dep,
        delay_stage_name=None,
        introduced_delay_hours=0.0
    )
    assert res["net_propagated_delay_hours"] == 0.0
    assert res["total_simulated_hours"] == 100.0
    assert res["disrupted_stage"] == "None"

def test_no_double_counting_in_propagation():
    """
    CRITICAL TEST: Ensure single cumulative delay tracking prevents double counting.
    Injecting 10 hours at 'Main Transit' should increase total duration by exactly 10 hours,
    not 10 hours per subsequent stage.
    """
    dep = datetime(2024, 7, 1, 8, 0, tzinfo=timezone.utc)
    base_hours = 80.0
    injected_hours = 12.0

    res = simulate_stage_propagation(
        predicted_duration_hours=base_hours,
        departure_timestamp=dep,
        delay_stage_name="Main Transit",
        introduced_delay_hours=injected_hours
    )

    # Total net delay must be exactly 12.0 hours, NOT 24.0 or 36.0 hours!
    assert pytest.approx(res["net_propagated_delay_hours"], 0.01) == injected_hours
    assert pytest.approx(res["total_simulated_hours"], 0.01) == (base_hours + injected_hours)

    # Check stage breakdown
    stages = res["stages"]
    assert len(stages) == 4
    
    # Stage 0: Origin Dispatch (upstream of injection, unchanged)
    assert stages[0]["Stage"] == "Origin Dispatch"
    assert stages[0]["Injected_Delay_Hours"] == 0.0
    assert stages[0]["Cumulative_Delay_Hours"] == 0.0

    # Stage 1: Main Transit (injection site)
    assert stages[1]["Stage"] == "Main Transit"
    assert stages[1]["Injected_Delay_Hours"] == injected_hours
    assert stages[1]["Cumulative_Delay_Hours"] == injected_hours

    # Stage 2: Sorting Hub (downstream, shifted by cumulative delay)
    assert stages[2]["Stage"] == "Sorting / Transfer Hub"
    assert stages[2]["Injected_Delay_Hours"] == 0.0
    assert stages[2]["Cumulative_Delay_Hours"] == injected_hours

    # Stage 3: Final Delivery (downstream, shifted by cumulative delay)
    assert stages[3]["Stage"] == "Final Delivery"
    assert stages[3]["Injected_Delay_Hours"] == 0.0
    assert stages[3]["Cumulative_Delay_Hours"] == injected_hours

def test_missing_optional_inputs_handled_safely():
    """Verify simulation gracefully defaults when optional parameters are omitted."""
    dep = datetime(2024, 8, 1, 12, 0)  # Naive datetime
    res = simulate_stage_propagation(
        predicted_duration_hours=0.0,  # Zero duration fallback
        departure_timestamp=dep
    )
    assert res["total_nominal_hours"] > 0
    assert res["net_propagated_delay_hours"] == 0.0
    assert len(res["stages"]) == 4

def test_real_deployment_roadmap_contains_required_sources():
    """Verify deployment roadmap documents real-world telemetry enhancements."""
    roadmap = get_real_deployment_roadmap()
    assert len(roadmap) >= 4
    sources = [r["Data Source"] for r in roadmap]
    assert any("Telematics" in s for s in sources)
    assert any("EDI" in s for s in sources)
    assert any("Customs" in s for s in sources)
    assert any("Weather" in s for s in sources)
