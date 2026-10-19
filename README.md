# 🚢 Aiqon SmartShip AI: Predictive Shipment ETA & Delay Propagation Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-18%20Passed-brightgreen.svg)]()

> **Aiqon AI Internship Take-Home Assignment: Predicting Shipment ETA and Delay Propagation Using AI**  
> Developed as a production-grade, reproducible logistics intelligence platform built strictly on the genuine Kaggle US Logistics Performance Dataset.

---

## 📌 Executive Overview

**Aiqon SmartShip AI** is an intelligent supply chain analytics application designed to predict shipment transit duration, calculate precise estimated arrival timestamps (ETA), assess schedule-relative delay risk, and simulate multi-stage disruption cascades across national logistics corridors.

### Core Capabilities
- **Supervised ML Transit Duration & ETA Forecasting**: Predicts transit days/hours using genuine pre-shipment attributes without target leakage.
- **Contract Schedule Delay & Risk Categorization**: Audits predicted arrival against user-defined SLA target dates with documented, configurable risk tiers (`On Time / Early`, `Low Risk`, `Moderate Risk`, `High Risk`).
- **Multi-Stage Disruption Propagation Simulator**: Decomposes routes into documented operational echelons (Origin Dispatch, Main Transit, Transfer Hub, Final Delivery) and tracks cumulative downstream delay without double-counting.
- **Interpretable AI Analytics**: Quantifies feature importance and produces plain-English executive decision narratives with statistical association disclaimers.
- **Data Quality & Leakage Audit**: Identifies date order inversions, missing tracking records, and the Kaggle dataset's post-hoc `Transit_Days` column to prevent data leakage.
- **Interactive Streamlit Intelligence Dashboard**: A responsive executive cockpit styled in a modern corporate logistics theme (Navy, Light Blue, White, Emerald, Amber, Crimson).

---

## 🏗️ Project Architecture

```
SmartShip/
├── app.py                     # Interactive Streamlit dashboard
├── requirements.txt           # Python dependencies (Streamlit, Scikit-learn, Plotly, etc.)
├── README.md                  # Comprehensive documentation & Windows setup guide
│
├── data/
│   ├── logistics_shipments_dataset.csv   # Primary raw Kaggle dataset (Shahriar Kabir)
│   └── processed_shipments.csv           # Cleaned dataset with quality flags & audit trails
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── inspect_data.py        # Dataset exploratory audit and schema verification
│   ├── clean_data.py          # Cleaning pipeline, date order checks, and outlier handling
│   ├── features.py            # Preprocessing pipeline, OneHotEncoding, zero-leakage guarantee
│   ├── train.py               # Model training (Baseline Median vs RandomForestRegressor)
│   ├── predict.py             # UTC timestamp normalization, ETA & schedule delay engine
│   ├── explain.py             # Risk categorization, feature importance & plain-English narratives
│   └── propagation.py         # Multi-stage cascade simulation and single cumulative delay ledger
│
├── models/
│   ├── eta_model.joblib       # Fitted Scikit-learn pipeline artifact
│   ├── evaluation_metrics.json# Machine-readable performance metrics
│   └── test_predictions.csv  # Held-out test set predictions & residuals for analytics
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py           # Tests for schema, cleaning, negative durations, and zero leakage
│   ├── test_predictions.py    # Tests for model inference, ETA arithmetic, and risk rules
│   └── test_propagation.py    # Tests for stage proportions, no double counting, and safe fallbacks
│
└── report/
    └── technical_report.md    # 3-page comprehensive technical evaluation report
```

---

## 💻 Windows Setup & Execution Workflow

Follow this step-by-step workflow in PowerShell or Windows Terminal:

### 1. Clone or Open the Repository
```powershell
cd d:\projects\SmartShip
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify Raw Dataset
Ensure the raw Kaggle dataset is located at:
```
data/logistics_shipments_dataset.csv
```
*(If missing, download from [Kaggle US Logistics Performance Dataset](https://www.kaggle.com/datasets/shahriarkabir/us-logistics-performance-dataset) and place in `data/`)*.

### 5. Inspect the Raw Dataset
```powershell
python -m src.inspect_data
```
*Outputs row/column counts, data types, missing records, date distributions, and outlier metrics.*

### 6. Clean Dataset & Generate Processed Artifacts
```powershell
python -m src.clean_data
```
*Generates `data/processed_shipments.csv` after filtering inverted dates, missing tracking records, and lost shipments.*

### 7. Train and Evaluate Machine Learning Models
```powershell
python -m src.train
```
*Trains the Baseline Median model and the RandomForestRegressor pipeline, calculates MAE and RMSE in days and hours, and writes artifacts to `models/`.*

### 8. Run Automated Test Suite
```powershell
python -m pytest -v
```
*Executes all 18 unit and integration tests across data integrity, model inference, and propagation logic.*

### 9. Launch the Streamlit Interactive Dashboard
```powershell
streamlit run app.py
```
*Opens the executive web dashboard in your default browser at `http://localhost:8501`.*

---

## 📊 Measured Machine Learning Performance

Evaluated strictly on a held-out test set of 360 shipments (20% split, `random_state=42`):

| Evaluation Metric | Baseline Predictor (Median) | RandomForestRegressor Pipeline | Performance Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 2.114 days (50.73 hrs) | **2.016 days (48.38 hrs)** | **-2.35 hrs (RF is superior)** |
| **Root Mean Squared Error (RMSE)** | 3.359 days (80.63 hrs) | **3.105 days (74.53 hrs)** | **-6.10 hrs (RF is superior)** |
| **Test Set $R^2$ Score** | -0.0886 | **+0.0699** | **+0.1585 gain** |

> **Zero Target Leakage Guarantee**: The model strictly excludes post-event attributes (`Delivery_Date`, `Status`, `Cost`) and the dataset's post-hoc `Transit_Days` column.

---

## ⏱️ Five-Minute Demonstration Outline

Use this outline during technical evaluation interviews to demonstrate the platform:

| Timestamp | Phase | Screen / Feature | Key Points to Highlight |
| :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | **Problem & Data Audit** | `Data Quality & Schema Audit` | Show 2,000 raw vs 1,800 clean records. Highlight discovery of 16 date inversions (-30 days) and explain why Kaggle's `Transit_Days` column was excluded to avoid 100% target leakage. |
| **1:00 - 2:15** | **Live ETA & Risk Prediction** | `ETA & Delay Studio` | Load a historical sample or input a route (e.g. Miami to San Francisco). Predict ETA in UTC, demonstrate how user-specified target SLA calculates delay, and show the plain-English explanation card with causality disclaimers. |
| **2:15 - 3:30** | **Delay Propagation Simulator** | `Multi-Stage Propagation Simulator` | Emphasize data provenance (simulated stages clearly labeled). Inject an 18-hour disruption into *Main Transit*, demonstrate how downstream stages shift via single cumulative delay ledger, and prove zero double-counting. |
| **3:30 - 4:15** | **ML Model Validation** | `Model Evaluation & Residuals` | Review measured MAE/RMSE comparisons against the median baseline. Review actual vs predicted scatter plot and residual normality. |
| **4:15 - 5:00** | **Route Network & Summary** | `Route Explorer & Historical Archive` | Show interactive Sankey flow diagram of top corridors, query historical shipments by carrier/status, and conclude with the enterprise telematics deployment roadmap. |

---

## 🛡️ License & Academic Integrity
Developed as part of the Aiqon AI Internship technical assessment. Primary data source courtesy of Shahriar Kabir via Kaggle under Open Database License. All simulations and machine learning pipelines are fully reproducible.
