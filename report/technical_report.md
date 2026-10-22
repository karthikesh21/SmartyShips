# SmartyShips: Technical Evaluation Report
**Shipment Transit Duration Forecasting, SLA Risk Assessment, and Delay Propagation**

---

## 1. Problem Formulation & Scope

Freight logistics operations face compounding transit uncertainties across line-haul corridors, cross-dock sortation hubs, and regional carrier handoffs. Inaccurate transit time estimates lead to misallocated warehouse dock labor, missed delivery service level agreements (SLAs), and costly downstream expediting.

SmartyShips is an engineering prototype designed to:
1. Predict total shipment transit duration from pre-departure attributes.
2. Calculate estimated arrival timestamps (ETA) standardized in UTC.
3. Compare forecasted ETAs against contract delivery targets to classify SLA breach risks.
4. Model stage-by-stage disruption cascades across distinct shipment legs without double-counting.
5. Provide interpretable operational summaries explaining primary drivers of transit variance.

---

## 2. Dataset & Quality Audit

### 2.1 Source Data
The analysis uses the US Logistics Performance Dataset:
- **Volume**: 2,000 raw shipment records across 11 features.
- **Attributes**: `Shipment_ID`, `Origin_Warehouse`, `Destination`, `Carrier`, `Shipment_Date`, `Delivery_Date`, `Weight_kg`, `Cost`, `Status`, `Distance_miles`, `Transit_Days`.

### 2.2 Preprocessing & Data Cleaning
An exploratory audit identified several anomalies requiring remediation before modeling:

1. **Chronological Inversions (16 records / 0.8%)**:
   - Records with `Delivery_Date` earlier than `Shipment_Date` (e.g., dispatched August 10, recorded delivered July 11, resulting in `-30` transit days).
   - *Resolution*: Removed as data entry errors.
2. **Missing Delivery Dates (32 records / 1.6%)**:
   - Records marked `Delivered` or `Delayed` with `NaN` values in `Delivery_Date`.
   - *Resolution*: Excluded from duration training because ground truth arrival is absent.
3. **Lost Cargo (45 records / 2.25%)**:
   - Shipments marked `Lost` that never completed delivery.
   - *Resolution*: Excluded from duration modeling.
4. **Weight Anomalies**:
   - Zero-weight record (`SH10744`) filtered. Extreme outliers capped at the 99.5th percentile (152.7 kg) to reduce leverage on tree splits.

**Final Cleaned Dataset**: **1,800 records** (90.0% retention).

---

## 3. Target Formulation & Feature Space

### 3.1 Target Definition
Because timestamps in the source dataset are provided at calendar-day granularity, the historical transit duration is defined as:
$$\text{duration\_days} = \text{Delivery\_Date} - \text{Shipment\_Date}$$
$$\text{duration\_hours} = \text{duration\_days} \times 24.0$$

### 3.2 Pre-Shipment Features & Leakage Control
To prevent data leakage, only attributes available *prior to dispatch* are used:

| Feature | Type | Preprocessing | Role |
| :--- | :--- | :--- | :--- |
| `Distance_miles` | Numeric | Standard scaling | Spatial corridor distance |
| `Weight_kg` | Numeric | Capping + standard scaling | Cargo mass |
| `Shipment_Month` | Numeric | Standard scaling | Seasonal freight volume proxy |
| `Shipment_DayOfWeek` | Numeric | Standard scaling | Weekend dispatch cycle proxy |
| `Origin_Warehouse` | Categorical | One-hot encoding (unknowns handled) | Staging facility (10 hubs) |
| `Destination` | Categorical | One-hot encoding (unknowns handled) | Target metropolitan city (15 cities) |
| `Carrier` | Categorical | One-hot encoding (unknowns handled) | Operating freight carrier (7 carriers) |

**Excluded Features**:
- `Delivery_Date`, `Status`: Recorded post-delivery.
- `Cost`: Invoiced post-transit, incomplete records.
- `Transit_Days`: Dataset column identically equal to `(Delivery_Date - Shipment_Date)`. Passing this into the model would constitute 100% target leakage.

---

## 4. Modeling & Experimental Results

### 4.1 Methodology
- **Split**: 80% train (1,440 records) and 20% test (360 records), fixed seed (`random_state=42`).
- **Baseline**: Median transit duration of the training set ($4.0\text{ days} / 96.0\text{ hours}$).
- **Production Pipeline**: `RandomForestRegressor` (200 estimators, max depth 12, min samples split 4) combined with a Scikit-Learn `ColumnTransformer`.

### 4.2 Test Set Performance

| Metric | Baseline (Median) | Random Forest Pipeline | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (Days)** | 2.114 | **2.016** | **-0.098 days** |
| **Mean Absolute Error (Hours)** | 50.73 | **48.38** | **-2.35 hours** |
| **Root Mean Squared Error (Days)** | 3.359 | **3.105** | **-0.254 days** |
| **Root Mean Squared Error (Hours)** | 80.63 | **74.53** | **-6.10 hours** |
| **$R^2$ Score** | -0.0886 | **+0.0699** | **+0.1585** |

### 4.3 Signed Error Definition & Residual Analysis
Evaluation errors and residuals are defined using the consistent signed convention:
$$\text{prediction\_error\_days} = \text{predicted\_days} - \text{actual\_days}$$
$$\text{prediction\_error\_hours} = (\text{predicted\_days} - \text{actual\_days}) \times 24.0$$
- **Negative value**: Underprediction (predicted duration is shorter than actual delivery).
- **Positive value**: Overprediction (predicted duration is longer than actual delivery).

Given that the source dataset contains day-level dates without real-time GPS telematics or weather telemetry, residual variance naturally reflects unobserved roadway conditions and local distribution queue times. The Random Forest pipeline captures non-linear carrier-route interactions and reduces overall MAE by over 2.3 hours compared to the central tendency baseline.

---

## 5. ETA Arithmetic & Risk Categorization

### 5.1 ETA Formulation
For a departure timestamp $T_{\text{dep}}$:
$$\text{ETA}_{\text{UTC}} = T_{\text{dep}} + \Delta t_{\text{predicted\_hours}}$$
All timestamps are standardized to UTC.

### 5.2 Contract SLA Delay Calculation
When a contractual arrival deadline $T_{\text{target}}$ is specified:
$$\text{Delay}_{\text{hours}} = \frac{\text{ETA}_{\text{UTC}} - T_{\text{target}}}{3600}$$
- **Negative delay ($\le 0$ hours)**: On track / early delivery.
- **Positive delay ($> 0$ hours)**: Potential breach of contract deadline.

### 5.3 Risk Classification Tiers
Operational risk tiers are configured as:
- **On Time / Early**: $\text{Delay} \le 0\text{ h}$ (SLA buffer preserved).
- **Low Risk**: $0 < \text{Delay} \le 12\text{ h}$ (Within routine operational variance).
- **Moderate Risk**: $12 < \text{Delay} \le 24\text{ h}$ (Buffer exhausted, priority tracking advised).
- **High Risk**: $\text{Delay} > 24\text{ h}$ (Imminent SLA penalty; re-routing or expediting required).

---

## 6. Multi-Stage Disruption Propagation

> [!NOTE]
> **Data Provenance**: The raw Kaggle logistics dataset records total origin dispatch and final delivery dates, but does not provide intermediate GPS or EDI milestone scans. Consequently, stage breakdowns (15% Origin, 55% Line-haul, 20% Hub, 10% Last-mile) and disruption propagation timelines are engineering simulations constructed to model real-world supply chain mechanics, clearly distinguished from historical observations.

### 6.1 Operational Stages
Routes are partitioned across four standard logistics echelons:
1. **Origin Dispatch (15%)**: Sorting, palletization, outbound scanning.
2. **Main Transit (55%)**: Interstate long-haul line-haul.
3. **Sorting / Transfer Hub (20%)**: Break-bulk cross-dock sorting.
4. **Final Delivery (10%)**: Courier dispatch and delivery.

### 6.2 Single Cumulative Delay Ledger
To avoid double-counting injected delays, subsequent stages inherit upstream delays via cumulative summation:
$$T_{\text{start}, i} = T_{\text{end}, i-1}$$
$$T_{\text{end}, i} = T_{\text{start}, i} + \text{NominalDuration}_i + \text{InjectedDelay}_i - \text{BufferRecovered}_i$$

Injecting a 6-hour delay during Main Transit shifts the final ETA by exactly 6 hours (when no recovery buffer is configured). The delay is applied once at the disruption point and carried forward across downstream legs without recursive re-injection.

---

## 7. Operational Drivers & Feature Impact

Aggregating one-hot encoded tree split importances back to root operational variables indicates their relative predictive association (importances reflect empirical correlation in tree splits, not direct causal proof):
1. **Transit Distance (~40-45%)**: Primary structural baseline for line-haul transit times.
2. **Carrier Selection (~20-25%)**: Differences between express networks (FedEx, UPS) and ground freight services.
3. **Cargo Weight (~15-18%)**: Heavier consignments require freight pallet handling and LTL consolidation.
4. **Origin-Destination Corridor (~10-15%)**: Regional infrastructure density and cross-dock hub frequency.
5. **Dispatch Day / Month (~5%)**: Weekend depot schedules and seasonal shipping volume shifts.


---

## 8. Enterprise Roadmap & Production Integration

In an enterprise production deployment, this platform can be integrated with live supply chain systems:
1. **Telematics & ELD Feeds (Samsara, Geotab)**: Live truck GPS positions, driver hours-of-service, and average speed.
2. **Carrier EDI Feeds (EDI 214 / API Webhooks)**: Real-time milestone scans (departure scan, arrival at terminal, out for delivery).
3. **Weather & Traffic Feeds (NOAA, DOT 511)**: Dynamic impedance modeling for severe weather or highway closures.
4. **Dynamic Rescheduling Engine**: Automated alerts to destination cross-dock managers when moderate or high delay risks are detected.
