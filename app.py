"""
Aiqon SmartShip AI - Executive Shipment ETA & Delay Propagation Platform
Streamlit Dashboard implementing end-to-end logistics analytics, machine learning predictions,
timeline simulation, risk categorization, and data quality diagnostics.
"""

from datetime import datetime, date, time, timedelta, timezone
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Internal pipeline modules
from src.predict import (
    load_model,
    predict_eta_and_delay,
    normalize_timestamp
)
from src.explain import (
    categorize_delay_risk,
    get_feature_importances,
    generate_plain_english_explanation,
    DEFAULT_THRESHOLDS,
    RISK_COLORS
)
from src.propagation import (
    simulate_stage_propagation,
    STAGE_PROPORTIONS,
    STAGE_DESCRIPTIONS,
    get_real_deployment_roadmap
)

# Page configuration
st.set_page_config(
    page_title="Aiqon SmartShip AI | Logistics Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Corporate Logistics Theme (Navy, Light Blue, White, Bright Blue)
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header typography */
    h1, h2, h3, h4 {
        color: #0f2744 !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #0f2744;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-subtitle {
        color: #0284c7;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Risk Badges */
    .badge-ontime {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #fefce8;
        color: #854d0e;
        border: 1px solid #fef08a;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Provenance pill */
    .pill-historical {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .pill-predicted {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .pill-simulated {
        background-color: #faf5ff;
        color: #6b21a8;
        border: 1px solid #e9d5ff;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Info Callouts */
    .callout-box {
        background-color: #f0f7ff;
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #1e3a8a;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load data and model
@st.cache_data
def load_processed_data():
    csv_path = Path("data/processed_shipments.csv")
    if not csv_path.exists():
        from src.clean_data import clean_dataset
        clean_dataset()
    return pd.read_csv(csv_path)

@st.cache_resource
def load_trained_pipeline():
    model_path = Path("models/eta_model.joblib")
    if not model_path.exists():
        from src.train import train_and_evaluate
        train_and_evaluate()
    return load_model(str(model_path))

@st.cache_data
def load_metrics_json():
    metrics_path = Path("models/evaluation_metrics.json")
    if not metrics_path.exists():
        from src.train import train_and_evaluate
        train_and_evaluate()
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_test_predictions():
    preds_path = Path("models/test_predictions.csv")
    if not preds_path.exists():
        from src.train import train_and_evaluate
        train_and_evaluate()
    return pd.read_csv(preds_path)

# Initialize data and model
try:
    df_processed = load_processed_data()
    pipeline = load_trained_pipeline()
    metrics = load_metrics_json()
    test_eval_df = load_test_predictions()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Application Title Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; font-size: 2.2rem; color: #0f2744;">🚢 Aiqon SmartShip AI</h1>
        <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1rem; font-weight: 500;">
            Predictive Shipment ETA, Schedule Delay Risk & Multi-Stage Disruption Propagation Platform
        </p>
    </div>
    <div style="text-align: right;">
        <span class="pill-historical">Kaggle Logistics Data (Shahriar Kabir)</span>
        <span class="pill-predicted" style="margin-left: 6px;">RandomForest ETA Engine</span>
        <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">Timezone: Standard UTC</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/cargo-ship.png", width=64)
st.sidebar.title("SmartShip Navigator")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Intelligence Module:",
    [
        "🚀 ETA & Delay Studio",
        "🌊 Multi-Stage Propagation Simulator",
        "📈 Model Evaluation & Residuals",
        "🔍 Data Quality & Schema Audit",
        "🗺️ Route Explorer & Historical Archive"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Risk Thresholds (Hours)")
thresh_on_time = st.sidebar.number_input("On-time Threshold (<= h)", value=0.0, step=1.0)
thresh_low = st.sidebar.number_input("Low Delay Threshold (<= h)", value=12.0, step=1.0)
thresh_mod = st.sidebar.number_input("Moderate Delay Threshold (<= h)", value=24.0, step=1.0)
user_thresholds = {
    "on_time_max": float(thresh_on_time),
    "low_risk_max": float(thresh_low),
    "moderate_risk_max": float(thresh_mod)
}
st.sidebar.caption("Configurable application thresholds for SLA risk categorization.")

# ==============================================================================
# MODULE 1: ETA & DELAY STUDIO
# ==============================================================================
if app_mode == "🚀 ETA & Delay Studio":
    st.markdown("### 🚀 Real-Time Shipment ETA & Delay Risk Prediction")
    st.markdown("""
    Predict delivery duration and final arrival timestamp (ETA) using pre-shipment attributes.
    Optionally provide an **Expected Arrival Time** to evaluate schedule-relative delay and risk tier.
    """)

    # Quick preset loader from dataset
    col_mode, col_preset = st.columns([1, 2])
    with col_mode:
        input_source = st.radio("Input Mode:", ["Custom Parameters", "Load Historical Sample"], horizontal=True)

    default_origin = "Warehouse_MIA"
    default_dest = "San Francisco"
    default_carrier = "UPS"
    default_dist = 2600.0
    default_weight = 25.0
    default_date = date.today()
    default_time = time(8, 0)
    sample_id_text = "Custom Input"

    if input_source == "Load Historical Sample":
        with col_preset:
            sample_ids = df_processed["Shipment_ID"].tolist()
            selected_id = st.selectbox("Select Shipment Record from Kaggle Dataset:", sample_ids[:50])
            sample_row = df_processed[df_processed["Shipment_ID"] == selected_id].iloc[0]
            default_origin = sample_row["Origin_Warehouse"]
            default_dest = sample_row["Destination"]
            default_carrier = sample_row["Carrier"]
            default_dist = float(sample_row["Distance_miles"])
            default_weight = float(sample_row["Weight_kg"])
            sample_id_text = f"Sample: {selected_id} (Recorded Status: {sample_row['Status']})"
            st.caption(f"Loaded {sample_id_text} | Recorded Distance: {default_dist:,.0f} mi | Actual Duration: {sample_row.get('delivery_duration_days', 'N/A')} days")

    # Input Form
    with st.form("prediction_form"):
        st.markdown("#### 1. Shipment Operational Parameters (Pre-Shipment Only)")
        c1, c2, c3 = st.columns(3)
        with c1:
            origins = sorted(df_processed["Origin_Warehouse"].dropna().unique().tolist())
            origin = st.selectbox("Origin Warehouse", origins, index=origins.index(default_origin) if default_origin in origins else 0)
            distance = st.number_input("Transit Distance (miles)", min_value=50.0, max_value=3500.0, value=default_dist, step=50.0)
        with c2:
            dests = sorted(df_processed["Destination"].dropna().unique().tolist())
            dest = st.selectbox("Destination City", dests, index=dests.index(default_dest) if default_dest in dests else 0)
            weight = st.number_input("Cargo Weight (kg)", min_value=0.1, max_value=500.0, value=min(default_weight, 500.0), step=1.0)
        with c3:
            carriers = sorted(df_processed["Carrier"].dropna().unique().tolist())
            carrier = st.selectbox("Freight Carrier", carriers, index=carriers.index(default_carrier) if default_carrier in carriers else 0)

        st.markdown("#### 2. Departure & Expected SLA Schedule (UTC)")
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Planned Departure Timestamp**")
            cd1, cd2 = st.columns(2)
            with cd1:
                dep_date = st.date_input("Departure Date", value=default_date)
            with cd2:
                dep_time = st.time_input("Departure Time (UTC)", value=default_time)
            planned_departure = datetime.combine(dep_date, dep_time).replace(tzinfo=timezone.utc)

        with c5:
            st.markdown("**Expected Arrival Timestamp (Optional SLA)**")
            use_expected = st.checkbox("Specify Expected Arrival Deadline for Delay Risk Audit", value=True)
            if use_expected:
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    exp_date = st.date_input("Target Arrival Date", value=default_date + timedelta(days=4))
                with exp_col2:
                    exp_time = st.time_input("Target Arrival Time (UTC)", value=time(18, 0))
                expected_arrival = datetime.combine(exp_date, exp_time).replace(tzinfo=timezone.utc)
            else:
                expected_arrival = None

        submit_btn = st.form_submit_button("⚡ Predict Delivery ETA & Analyze Risk", use_container_width=True, type="primary")

    # Run Prediction
    pred_res = predict_eta_and_delay(
        origin=origin,
        destination=dest,
        carrier=carrier,
        distance_miles=distance,
        weight_kg=weight,
        departure_timestamp=planned_departure,
        expected_arrival_timestamp=expected_arrival,
        model=pipeline
    )

    risk_info = categorize_delay_risk(
        pred_res["estimated_delay_hours"],
        thresholds=user_thresholds
    )

    # Executive Output Section
    st.markdown("---")
    st.markdown("### 🎯 Model Prediction Results")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Duration</div>
            <div class="metric-value">{pred_res['predicted_duration_days']:.1f} <span style="font-size:1.1rem; font-weight:500;">days</span></div>
            <div class="metric-subtitle">{pred_res['predicted_duration_hours']:.1f} operational hours</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        eta_dt = normalize_timestamp(pred_res['predicted_eta_utc'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted ETA (UTC)</div>
            <div class="metric-value" style="font-size:1.35rem; color:#0284c7;">{eta_dt.strftime('%b %d, %Y')}</div>
            <div class="metric-subtitle">{eta_dt.strftime('%H:%M:%S')} UTC</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        delay_h = pred_res['estimated_delay_hours']
        if delay_h is not None:
            delay_text = f"{delay_h:+.1f} hrs"
            subtext = "Later than SLA" if delay_h > 0 else "Ahead of SLA"
            val_color = "#ef4444" if delay_h > 0 else "#10b981"
        else:
            delay_text = "N/A"
            subtext = "No SLA provided"
            val_color = "#64748b"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Schedule Variance</div>
            <div class="metric-value" style="color: {val_color};">{delay_text}</div>
            <div class="metric-subtitle">{subtext}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        cat = risk_info["category"]
        badge_cls = {
            "On Time / Early": "badge-ontime",
            "Low Risk": "badge-low",
            "Moderate Risk": "badge-moderate",
            "High Risk": "badge-high",
            "Schedule Not Set": "pill-historical"
        }.get(cat, "pill-historical")

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Delay Risk Category</div>
            <div style="margin-top: 8px;">
                <span class="{badge_cls}" style="font-size: 1rem; padding: 6px 16px;">{cat}</span>
            </div>
            <div class="metric-subtitle" style="font-size: 0.75rem; color: #64748b; margin-top: 8px;">
                Rule: {risk_info['threshold_applied']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Plain English Narrative & Feature Importance
    st.markdown("#### 🧠 AI Decision Explanation & Contributing Factors")
    exp_col_left, exp_col_right = st.columns([3, 2])

    importances_df = get_feature_importances(pipeline)

    with exp_col_left:
        plain_text = generate_plain_english_explanation(pred_res, risk_info, importances_df)
        st.markdown(f"""
        <div class="callout-box" style="line-height: 1.6;">
            {plain_text}
        </div>
        """, unsafe_allow_html=True)

    with exp_col_right:
        fig_imp = px.bar(
            importances_df.head(6),
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            title="Relative Feature Importance (%)",
            labels={"Importance_Pct": "Importance (%)", "Feature": ""},
            color="Importance_Pct",
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_showscale=False,
            font=dict(size=11)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Nominal Stage Timeline Breakdown
    st.markdown("#### ⏱️ Predicted Nominal Stage Timeline")
    st.caption("Distribution of predicted delivery duration across documented logistics stages.")
    
    nominal_sim = simulate_stage_propagation(
        predicted_duration_hours=pred_res["predicted_duration_hours"],
        departure_timestamp=planned_departure
    )

    timeline_data = []
    for s in nominal_sim["stages"]:
        timeline_data.append({
            "Stage": s["Stage"],
            "Start": s["Nominal_Start"],
            "End": s["Nominal_End"],
            "Duration (h)": s["Nominal_Duration_Hours"],
            "Provenance": s["Provenance"]
        })
    df_tl = pd.DataFrame(timeline_data)

    fig_tl = px.timeline(
        df_tl,
        x_start="Start",
        x_end="End",
        y="Stage",
        color="Duration (h)",
        color_continuous_scale="Tealgrn",
        hover_data=["Duration (h)", "Provenance"],
        title="Nominal Schedule Breakdown [Predicted & Simulated Stage Allocation]"
    )
    fig_tl.update_yaxes(autorange="reversed")
    fig_tl.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=35, b=10),
        font=dict(size=12)
    )
    st.plotly_chart(fig_tl, use_container_width=True)

# ==============================================================================
# MODULE 2: MULTI-STAGE PROPAGATION SIMULATOR
# ==============================================================================
elif app_mode == "🌊 Multi-Stage Propagation Simulator":
    st.markdown("### 🌊 Multi-Stage Delay Propagation Simulator")
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Data Provenance & Methodology Notice:</strong><br>
        The Kaggle logistics dataset contains only high-level dispatch and final delivery dates; 
        intermediate checkpoint timestamps (such as port dwelling, rail classification, or transfer hub dwell) 
        do not exist in the raw Kaggle dataset. This module implements a documented, mathematically consistent 
        <strong>stage simulation</strong> to demonstrate downstream disruption propagation without double counting.
    </div>
    """, unsafe_allow_html=True)

    sim_c1, sim_c2 = st.columns([1, 2])
    with sim_c1:
        st.markdown("#### Simulator Controls")
        sim_base_hours = st.slider("Baseline Predicted Delivery Duration (Hours)", min_value=24.0, max_value=240.0, value=96.0, step=6.0)
        
        sim_stage = st.selectbox(
            "Select Disrupted Stage:",
            ["None"] + list(STAGE_PROPORTIONS.keys())
        )
        
        sim_delay_injected = st.slider(
            "Introduced Upstream Delay (Hours):",
            min_value=0.0, max_value=72.0, value=18.0 if sim_stage != "None" else 0.0, step=2.0
        )
        
        sim_recovery = st.slider(
            "Downstream Buffer Absorption (Hours):",
            min_value=0.0, max_value=24.0, value=4.0, step=1.0,
            help="Simulates expedited handling or buffer capacity to partially mitigate downstream propagation."
        )

        sim_dep_date = st.date_input("Simulation Start Date", value=date.today())
        sim_dep_dt = datetime.combine(sim_dep_date, time(8, 0)).replace(tzinfo=timezone.utc)

    # Run Propagation Calculation
    target_stage = sim_stage if sim_stage != "None" else None
    sim_result = simulate_stage_propagation(
        predicted_duration_hours=sim_base_hours,
        departure_timestamp=sim_dep_dt,
        delay_stage_name=target_stage,
        introduced_delay_hours=sim_delay_injected,
        recovery_buffer_hours=sim_recovery
    )

    with sim_c2:
        st.markdown("#### Propagation Impact Summary")
        pk1, pk2, pk3 = st.columns(3)
        with pk1:
            st.metric("Nominal Duration", f"{sim_result['total_nominal_hours']:.1f} hrs", "Baseline Forecast")
        with pk2:
            st.metric("Net Propagated Delay", f"+{sim_result['net_propagated_delay_hours']:.1f} hrs", f"Injected: {sim_delay_injected:.1f}h")
        with pk3:
            st.metric("Final Delayed Duration", f"{sim_result['total_simulated_hours']:.1f} hrs", f"Recovered: {sim_delay_injected - sim_result['net_propagated_delay_hours']:.1f}h")

        # Visual Gantt comparison
        gantt_rows = []
        for s in sim_result["stages"]:
            # Nominal stage
            gantt_rows.append({
                "Timeline": "1. Nominal (Baseline)",
                "Stage": s["Stage"],
                "Start": s["Nominal_Start"],
                "End": s["Nominal_End"],
                "Duration": s["Nominal_Duration_Hours"],
                "Type": "Nominal"
            })
            # Simulated disrupted stage
            gantt_rows.append({
                "Timeline": "2. Disrupted (Propagated)",
                "Stage": s["Stage"],
                "Start": s["Simulated_Start"],
                "End": s["Simulated_End"],
                "Duration": s["Actual_Duration_Hours"],
                "Type": "Propagated"
            })

        df_gantt = pd.DataFrame(gantt_rows)
        fig_prop = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="End",
            y="Timeline",
            color="Stage",
            title=f"Comparative Timeline: Baseline vs Disrupted ({sim_stage} +{sim_delay_injected}h)",
            hover_data=["Stage", "Duration", "Type"]
        )
        fig_prop.update_yaxes(autorange="reversed")
        fig_prop.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_prop, use_container_width=True)

    # Detailed Stage Impact Table
    st.markdown("#### 📋 Cumulative Stage Disruption Ledger (No Double Counting)")
    table_records = []
    for s in sim_result["stages"]:
        table_records.append({
            "Stage": s["Stage"],
            "Nominal Hours": s["Nominal_Duration_Hours"],
            "Injected Delay": f"+{s['Injected_Delay_Hours']:.1f} h" if s["Injected_Delay_Hours"] > 0 else "0.0 h",
            "Buffer Absorption": f"-{s['Recovered_Buffer_Hours']:.1f} h" if s["Recovered_Buffer_Hours"] > 0 else "0.0 h",
            "Effective Duration": f"{s['Actual_Duration_Hours']:.1f} h",
            "Cumulative Delay Total": f"{s['Cumulative_Delay_Hours']:.1f} h",
            "Simulated Arrival (UTC)": s["Simulated_End"].strftime("%Y-%m-%d %H:%M"),
            "Provenance": s["Provenance"]
        })
    st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)

    # Enterprise Telemetry Roadmap
    with st.expander("🌐 Enterprise Deployment Roadmap: How to Replace Simulation with Real Telemetry"):
        st.markdown("""
        In an enterprise production deployment, simulated stage timestamps would be replaced by live telematics
        and direct supply chain sensor integration:
        """)
        roadmap_items = get_real_deployment_roadmap()
        st.table(pd.DataFrame(roadmap_items))

# ==============================================================================
# MODULE 3: MODEL EVALUATION & RESIDUALS
# ==============================================================================
elif app_mode == "📈 Model Evaluation & Residuals":
    st.markdown("### 📈 Machine Learning Model Performance & Residual Diagnostics")
    st.markdown("""
    Evaluation of the primary **RandomForestRegressor** pipeline against a **Median Duration Baseline**
    using a held-out test set (20% split, 360 shipments, random_state=42).
    """)

    b_m = metrics["baseline"]
    rf_m = metrics["random_forest"]
    cmp = metrics["comparison"]

    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Baseline MAE</div>
            <div class="metric-value">{b_m['mae_hours']} <span style="font-size:1rem; font-weight:500;">hrs</span></div>
            <div class="metric-subtitle">{b_m['mae_days']} days (Median: {b_m['median_duration_days']}d)</div>
        </div>
        """, unsafe_allow_html=True)

    with ev2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Random Forest MAE</div>
            <div class="metric-value" style="color:#0284c7;">{rf_m['mae_hours']} <span style="font-size:1rem; font-weight:500;">hrs</span></div>
            <div class="metric-subtitle">{rf_m['mae_days']} days (MAE Gain: {cmp['mae_improvement_hours']:+.1f}h)</div>
        </div>
        """, unsafe_allow_html=True)

    with ev3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Random Forest RMSE</div>
            <div class="metric-value" style="color:#0284c7;">{rf_m['rmse_hours']} <span style="font-size:1rem; font-weight:500;">hrs</span></div>
            <div class="metric-subtitle">{rf_m['rmse_days']} days (RMSE Gain: {cmp['rmse_improvement_hours']:+.1f}h)</div>
        </div>
        """, unsafe_allow_html=True)

    with ev4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test Set Evaluation R²</div>
            <div class="metric-value">{rf_m['r2']:.4f}</div>
            <div class="metric-subtitle">Baseline R²: {b_m['r2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c_eval_l, c_eval_r = st.columns(2)

    with c_eval_l:
        st.markdown("#### 1. Actual vs. Predicted Delivery Duration (Test Set)")
        fig_scatter = px.scatter(
            test_eval_df,
            x="actual_days",
            y="predicted_days",
            color="residual_days",
            color_continuous_scale="RdBu",
            labels={"actual_days": "Actual Duration (Days)", "predicted_days": "Predicted Duration (Days)"},
            title="Actual vs Predicted (Held-out 360 Test Shipments)",
            hover_data=["Distance_miles", "Weight_kg", "Carrier"]
        )
        # Identity line
        max_val = max(test_eval_df["actual_days"].max(), test_eval_df["predicted_days"].max()) + 1
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            name="Perfect Prediction (y=x)",
            line=dict(color="#0f172a", dash="dash")
        ))
        fig_scatter.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c_eval_r:
        st.markdown("#### 2. Residual Distribution (Actual - Predicted)")
        fig_res = px.histogram(
            test_eval_df,
            x="residual_days",
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#0284c7"],
            labels={"residual_days": "Residual Error in Days (Actual - Predicted)"},
            title="Model Residuals Distribution (Centered near Zero)"
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig_res.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_res, use_container_width=True)

    # Technical Rationale Callout
    st.markdown("""
    <div class="callout-box">
        <strong>Methodological Rigor & Honest Reporting:</strong><br>
        1. <strong>Measurable Advantage:</strong> The Random Forest regressor achieves lower MAE (48.38h vs 50.73h) 
        and lower RMSE (74.53h vs 80.63h) compared to the training-set median baseline.<br>
        2. <strong>Zero Target Leakage:</strong> Post-delivery attributes (such as <code>Delivery_Date</code>, 
        <code>Status</code>, and the dataset's post-hoc <code>Transit_Days</code> column) are strictly excluded from predictive inputs.<br>
        3. <strong>Logistics Variance Reality:</strong> In real freight datasets with only day-level timestamps and summary metadata, 
        stochastic highway conditions and terminal queuing contribute inherent variance.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MODULE 4: DATA QUALITY & SCHEMA AUDIT
# ==============================================================================
elif app_mode == "🔍 Data Quality & Schema Audit":
    st.markdown("### 🔍 Kaggle Logistics Dataset Inspection & Quality Audit")
    st.markdown("""
    Comprehensive inspection of `data/logistics_shipments_dataset.csv` sourced from Kaggle.
    Audits column data types, missing records, date inversions, and cleaning transformations.
    """)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Raw Dataset Records", "2,000", "11 Columns")
    with q2:
        st.metric("Cleaned Training Records", "1,800", "90.0% Retained")
    with q3:
        st.metric("Missing Delivery Dates", "32 (1.6%)", "Uncompleted Shipments")
    with q4:
        st.metric("Inverted Dates Filtered", "16 (0.8%)", "Delivery < Shipment Date")

    st.markdown("---")
    st.markdown("#### 1. Detailed Cleaning Decisions & Exclusions Log")
    decisions = [
        {"Issue": "Missing Delivery Dates", "Count": "32 rows", "Action Taken": "Excluded from training", "Rationale": "Cannot calculate actual delivery duration without arrival timestamp."},
        {"Issue": "Negative Duration Dates", "Count": "16 rows", "Action Taken": "Excluded from training", "Rationale": "Data entry error where recorded delivery date preceded shipment date (-30 days)."},
        {"Issue": "Lost Shipments", "Count": "45 rows", "Action Taken": "Excluded from training", "Rationale": "Shipments aborted or lost during transit do not reflect valid completed transit durations."},
        {"Issue": "Zero Cargo Weight", "Count": "1 row", "Action Taken": "Excluded from training", "Rationale": "Package recorded with 0.0 kg physically invalid for freight transit."},
        {"Issue": "Weight Outliers", "Count": "Max 5,404 kg", "Action Taken": "Clipped at 99.5th percentile", "Rationale": "Extreme single-package freight anomalies clipped to prevent gradient distortion."},
        {"Issue": "Post-Hoc Transit_Days Column", "Count": "2,000 rows", "Action Taken": "Strictly excluded from inputs", "Rationale": "Matches target delivery duration for delivered shipments; using it would cause 100% target leakage."}
    ]
    st.table(pd.DataFrame(decisions))

    st.markdown("#### 2. Raw Shipment Status Distribution")
    status_counts = df_processed["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Record Count"]
    status_counts["Percentage"] = (status_counts["Record Count"] / len(df_processed)) * 100

    col_st1, col_st2 = st.columns(2)
    with col_st1:
        fig_status = px.pie(
            status_counts,
            names="Status",
            values="Record Count",
            title="Status Distribution (2,000 Total Shipments)",
            color="Status",
            color_discrete_map={
                "Delivered": "#10b981",
                "Delayed": "#f59e0b",
                "In Transit": "#0284c7",
                "Lost": "#ef4444",
                "Returned": "#8b5cf6"
            }
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col_st2:
        fig_hist = px.histogram(
            df_processed[df_processed["is_valid_training_sample"] == True],
            x="delivery_duration_days",
            nbins=20,
            title="Cleaned Delivery Duration Distribution (Days)",
            color_discrete_sequence=["#0f2744"],
            labels={"delivery_duration_days": "Duration in Days"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# MODULE 5: ROUTE EXPLORER & HISTORICAL ARCHIVE
# ==============================================================================
elif app_mode == "🗺️ Route Explorer & Historical Archive":
    st.markdown("### 🗺️ Route Network Explorer & Filterable Historical Shipments")
    st.markdown("""
    Explore verified operational corridors and query historical shipment records.
    Filter by origin warehouse, destination city, carrier, or operational status.
    """)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        f_origin = st.multiselect("Origin Warehouse:", options=sorted(df_processed["Origin_Warehouse"].unique().tolist()))
    with f2:
        f_dest = st.multiselect("Destination:", options=sorted(df_processed["Destination"].unique().tolist()))
    with f3:
        f_carrier = st.multiselect("Carrier:", options=sorted(df_processed["Carrier"].unique().tolist()))
    with f4:
        f_status = st.multiselect("Status:", options=sorted(df_processed["Status"].unique().tolist()))

    filtered_df = df_processed.copy()
    if f_origin:
        filtered_df = filtered_df[filtered_df["Origin_Warehouse"].isin(f_origin)]
    if f_dest:
        filtered_df = filtered_df[filtered_df["Destination"].isin(f_dest)]
    if f_carrier:
        filtered_df = filtered_df[filtered_df["Carrier"].isin(f_carrier)]
    if f_status:
        filtered_df = filtered_df[filtered_df["Status"].isin(f_status)]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(df_processed):,} records**")

    # Schematic Route Network Diagram
    st.markdown("#### 🌐 Schematic Route Network Diagram (Corridor Volumes)")
    st.caption("Schematic bipartite network graph connecting Origin Warehouses to Destination Hubs. Note: Node positions are schematic for topology clarity, not geographically geocoded.")
    
    route_agg = filtered_df.groupby(["Origin_Warehouse", "Destination"]).agg(
        Shipment_Count=("Shipment_ID", "count"),
        Avg_Distance=("Distance_miles", "mean"),
        Avg_Duration=("delivery_duration_days", "mean")
    ).reset_index().sort_values(by="Shipment_Count", ascending=False).head(25)

    if not route_agg.empty:
        # Build network chord / parallel coordinates / schematic flow
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(route_agg["Origin_Warehouse"].unique()) + list(route_agg["Destination"].unique()),
                color="#0284c7"
            ),
            link=dict(
                source=[list(route_agg["Origin_Warehouse"].unique()).index(src) for src in route_agg["Origin_Warehouse"]],
                target=[len(route_agg["Origin_Warehouse"].unique()) + list(route_agg["Destination"].unique()).index(dst) for dst in route_agg["Destination"]],
                value=route_agg["Shipment_Count"],
                color="rgba(2, 132, 199, 0.35)"
            )
        )])
        fig_sankey.update_layout(title_text="Top Operational Corridors (Schematic Flow)", font_size=11, height=350, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_sankey, use_container_width=True)

    # Filtered Table
    display_cols = [
        "Shipment_ID", "Origin_Warehouse", "Destination", "Carrier",
        "Shipment_Date", "Delivery_Date", "Weight_kg", "Distance_miles",
        "Status", "delivery_duration_days"
    ]
    st.dataframe(
        filtered_df[display_cols].rename(columns={"delivery_duration_days": "Actual Duration (Days)"}),
        use_container_width=True,
        hide_index=True
    )
