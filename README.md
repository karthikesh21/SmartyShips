# SmartyShips: Intelligent Shipment Analytics & Delay Propagation Platform

A machine learning pipeline and interactive operations dashboard for estimating freight transit durations, detecting contract SLA breach risks, and modeling multi-stage delay propagation across supply chain corridors.

Built on the **US Logistics Performance Dataset** (2,000 shipment records).

---

## Overview

Accurate transit time forecasting is critical for modern logistics operations. Delays at early freight milestones often cascade into severe downstream cross-dock congestion and customer SLA penalties.

**SmartyShips** provides:
- **Transit Duration & ETA Modeling**: Supervised machine learning (`RandomForestRegressor` pipeline) predicting delivery duration from pre-shipment attributes without target leakage.
- **SLA Breach Risk Classification**: Compares predicted arrival against user-defined contract deadlines into operational risk tiers (`On Time`, `Low Risk`, `Moderate Risk`, `High Risk`).
- **Multi-Stage Disruption Simulation**: Simulates transit echelons (*Origin Dispatch*, *Main Line-Haul Transit*, *Transfer Hub*, *Final Delivery*) to model how upstream bottlenecks propagate downstream without double-counting.
- **Data Quality & Audit Pipeline**: Automated detection of inverted dates, missing tracking records, and post-hoc leakage columns.
- **Operations Dashboard**: Five-section Streamlit interface for dispatchers and fleet planners to evaluate routes, stress-test disruptions, and review model residuals in both Dark and Light themes.

---

## Key Features

1. **Pre-Shipment Predictive Modeling**: Estimates transit duration in both days and hours using strictly pre-departure features.
2. **UTC-Standardized ETA Arithmetic**: Calculates estimated arrival times in timezone-aware UTC:
   $$\text{ETA}_{\text{UTC}} = T_{\text{departure}} + \Delta t_{\text{predicted\_hours}}$$
3. **Contract SLA Delay Variance**: Evaluates schedule variance against customer deadlines:
   $$\text{Delay}_{\text{hours}} = \frac{\text{ETA}_{\text{UTC}} - T_{\text{expected}}}{3600}$$
4. **Single-Ledger Disruption Simulator**: Propagates delays downstream across distinct legs while guaranteeing zero double-counting.
5. **Interactive Operations Dashboard**: High-contrast, dynamic dashboard featuring 5 operational views, corridor volume flow Sankey diagrams, and residual diagnostic plots.

---

## Technology Stack

- **Language**: Python 3.10+ (tested on Python 3.11)
- **Interactive UI**: Streamlit 1.30+
- **Machine Learning**: Scikit-Learn (Pipelines, `RandomForestRegressor`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`)
- **Data Processing**: Pandas, NumPy
- **Interactive Visualizations**: Plotly Express & Plotly Graph Objects
- **Model Persistence**: Joblib
- **Automated Testing**: Pytest

---

## Project Structure

```
SmartShip/
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration (Dark base palette)
├── app.py                        # Streamlit operations dashboard (5 functional views)
├── requirements.txt              # Core Python dependencies
├── LICENSE                       # MIT License with dataset attribution notice
├── README.md                     # Project documentation & setup instructions
│
├── data/
│   ├── logistics_shipments_dataset.csv   # Raw Kaggle shipment records (2,000 rows)
│   └── processed_shipments.csv           # Cleaned dataset (1,800 validated rows)
│
├── src/
│   ├── __init__.py               # Package root
│   ├── inspect_data.py           # Data audit and schema verification script
│   ├── clean_data.py             # Data cleaning and sanity checks
│   ├── features.py               # Feature transformation pipeline (OneHot + Scaler)
│   ├── train.py                  # Model training & benchmarking (Baseline vs Random Forest)
│   ├── predict.py                # UTC-standardized ETA and delay calculation engine
│   ├── explain.py                # Feature importance and SLA risk categorization
│   ├── propagation.py            # Multi-stage disruption propagation logic
│   └── chart_theme.py            # Unified Plotly theme and dark/light mode palette
│
├── models/
│   ├── eta_model.joblib          # Serialized Scikit-learn pipeline artifact (~6.2 MB)
│   ├── evaluation_metrics.json   # Test-set evaluation metrics
│   └── test_predictions.csv      # Holdout test set predictions and signed errors
│
├── tests/
│   ├── test_chart_theme.py       # Theme palette and chart styling tests
│   ├── test_data.py              # Data integrity, schema, and leakage tests
│   ├── test_predictions.py       # Inference arithmetic, delay sign, and risk threshold tests
│   └── test_propagation.py       # Stage duration balance and delay ledger tests
│
└── report/
    └── technical_report.md       # In-depth engineering report and validation results
```

---

## Dataset Source & Attribution

- **Dataset**: US Logistics Performance Dataset
- **Platform**: [Kaggle Logistics Datasets](https://www.kaggle.com/)
- **Volume**: 2,000 raw shipment records across 11 features (`Shipment_ID`, `Origin_Warehouse`, `Destination`, `Carrier`, `Shipment_Date`, `Delivery_Date`, `Weight_kg`, `Cost`, `Status`, `Distance_miles`, `Transit_Days`).
- **Attribution**: Sourced from public Kaggle contributors for supply chain benchmarking. Authors of SmartyShips claim no ownership of the raw shipment records.

---

## Data Pipeline & Leakage Prevention

### Cleaning Decisions
The raw dataset contains 2,000 records. The cleaning pipeline handles:
1. **Chronological Inversions (16 records / 0.8%)**: Records where `Delivery_Date` occurs prior to `Shipment_Date` (yielding negative transit durations down to -30 days) are removed as corrupt data.
2. **Missing Delivery Dates (32 records / 1.6%)**: Incomplete shipments marked delivered but lacking arrival timestamps are excluded from duration modeling.
3. **Lost Shipments (45 records / 2.25%)**: Shipments marked `Lost` where transit was aborted are filtered out.
4. **Weight Anomalies**: Zero-weight records are filtered and extreme outliers capped at the 99.5th percentile (152.7 kg) to prevent excessive leverage on tree splits.

**Final Cleaned Dataset**: **1,800 validated records (90.0% retention)**.

### Target Leakage Prevention
To guarantee realistic operational inference, only features available *prior to dispatch* are used:
- **Included Features**: `Origin_Warehouse`, `Destination`, `Carrier`, `Distance_miles`, `Weight_kg`, `Shipment_Month`, `Shipment_DayOfWeek`.
- **Excluded Features**: `Delivery_Date`, `Status`, `Cost` (finalized post-delivery).
- **Strictly Excluded Target Leakage**: The dataset column `Transit_Days` identically matches `(Delivery_Date - Shipment_Date)` for delivered cargo. Passing it as a predictor would constitute 100% target leakage, so it is strictly excluded.

---

## Model Evaluation & Verified Metrics

Models were evaluated on a held-out test split of **360 shipments (20% test partition, fixed seed `random_state=42`)**:

| Metric | Baseline Predictor (Median) | RandomForestRegressor Pipeline | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (Days)** | 2.114 | **2.016** | **-0.098 days** |
| **Mean Absolute Error (Hours)** | 50.73 | **48.38** | **-2.35 hours** |
| **Root Mean Squared Error (Days)** | 3.359 | **3.105** | **-0.254 days** |
| **Root Mean Squared Error (Hours)** | 80.63 | **74.53** | **-6.10 hours** |
| **$R^2$ Score** | -0.0886 | **+0.0699** | **+0.1585** |

### Signed Error Definition
Model residuals in [`models/test_predictions.csv`](models/test_predictions.csv) follow the consistent signed definition:
$$\text{prediction\_error\_hours} = (\text{predicted\_duration\_days} - \text{actual\_duration\_days}) \times 24.0$$
- **Negative value**: Underprediction (model predicted a shorter delivery duration than actual).
- **Positive value**: Overprediction (model predicted a longer delivery duration than actual).

The Random Forest model captures non-linear carrier-route interactions and outperforms the central-tendency median baseline across all error metrics while producing physically realistic, non-negative duration outputs.

---

## Disruption Propagation Engine

Real supply chain networks consist of connected operational legs:
1. **Origin Dispatch (15%)**: Staging, picking, and carrier outbound scanning.
2. **Main Line-Haul Transit (55%)**: Interstate long-haul highway movement.
3. **Sorting / Transfer Hub (20%)**: Break-bulk cross-dock sorting.
4. **Final Delivery (10%)**: Courier dispatch and customer drop-off.

> [!NOTE]
> **Simulation Provenance**: The raw dataset contains total origin dispatch and delivery timestamps, but does not record intermediate GPS telemetry. Stage breakdowns and disruption cascades are engineering simulations designed to stress-test supply chain bottlenecks, clearly distinguished from historical observations.

### Guaranteed Single-Ledger Arithmetic
Delays are injected at a specific stage and carried forward downstream through a single cumulative delay ledger:
- An injected **+6.0 hour delay** at Main Transit shifts the final ETA by **exactly +6.0 hours** when no buffer is active.
- Delays are applied once and cascaded without recursive double-counting across subsequent transit legs.

---

## Installation & Windows Quickstart

### 1. Environment Setup

Create and activate a Python virtual environment (PowerShell):

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Inspect & Clean Dataset

Run data inspection:
```powershell
.\.venv\Scripts\python.exe -m src.inspect_data
```

Execute the cleaning pipeline to produce `data/processed_shipments.csv`:
```powershell
.\.venv\Scripts\python.exe -m src.clean_data
```

### 3. Train Model

Train the median baseline and the Random Forest regression pipeline:
```powershell
.\.venv\Scripts\python.exe -m src.train
```
Artifacts are saved to `models/eta_model.joblib`, `models/evaluation_metrics.json`, and `models/test_predictions.csv`.

### 4. Run Automated Test Suite

Execute the 22 automated unit and integration tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### 5. Launch the Dashboard

Launch the interactive Streamlit dashboard:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Dashboard Sections

The application provides 5 operational tabs:
1. **ETA & Delay Estimator**: Select origin warehouse, destination, carrier, distance, and weight to calculate predicted duration, UTC arrival timestamp, and SLA risk tier.
2. **Disruption Simulator**: Interactive waterfall timeline modeling injected bottlenecks and buffer recovery across four shipment stages.
3. **Model Evaluation & Residuals**: Parity charts, signed error distributions, and benchmark tables comparing the Random Forest pipeline against the median baseline.
4. **Data Quality & Audit**: Cleaning audit scorecard showing filtered inversions, handled missing values, and leakage control audits.
5. **Route Network & Records**: Multiselect filterable table and national corridor volume flow Sankey diagrams.

---

## Known Limitations & Future Improvements

1. **Calendar-Day Granularity**: The source dataset records timestamps at day-level resolution rather than hour/minute timestamps, introducing unavoidable discrete rounding variance.
2. **Absence of Real-Time Telematics**: Highway conditions, weather storms, and terminal dwell times are unobserved in the historical dataset; integrating ELD/GPS feeds (e.g. Samsara, Geotab) would improve $R^2$.
3. **Feature Correlation vs. Causation**: Feature importances indicate empirical association in decision tree splits, not direct causal drivers.


