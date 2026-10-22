import subprocess
from pathlib import Path

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SmartyShips Technical Evaluation Report - Karthikesh Soma</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        @page {
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {
                content: counter(page);
                font-family: 'Inter', sans-serif;
                font-size: 9pt;
                color: #64748b;
            }
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            line-height: 1.55;
            font-size: 10pt;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }

        .header-banner {
            border-bottom: 2px solid #0284c7;
            padding-bottom: 14px;
            margin-bottom: 22px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }

        .title-block h1 {
            color: #0369a1;
            font-size: 20pt;
            font-weight: 800;
            margin: 0 0 4px 0;
            letter-spacing: -0.02em;
        }

        .title-block .subtitle {
            color: #475569;
            font-size: 11pt;
            font-weight: 500;
            margin: 0;
        }

        .meta-block {
            text-align: right;
            font-size: 9pt;
            color: #64748b;
            line-height: 1.4;
        }

        .meta-block strong {
            color: #0f172a;
        }

        h2 {
            color: #0f2744;
            font-size: 13pt;
            font-weight: 700;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
            margin-top: 22px;
            margin-bottom: 10px;
            letter-spacing: -0.01em;
            break-after: avoid;
        }

        h3 {
            color: #0369a1;
            font-size: 10.5pt;
            font-weight: 600;
            margin-top: 14px;
            margin-bottom: 6px;
            break-after: avoid;
        }

        p, li {
            color: #334155;
            text-align: justify;
        }

        ul, ol {
            margin-top: 4px;
            margin-bottom: 8px;
            padding-left: 20px;
        }

        li {
            margin-bottom: 4px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 16px 0;
            font-size: 9pt;
            page-break-inside: avoid;
        }

        th {
            background-color: #f1f5f9;
            color: #0f2744;
            font-weight: 700;
            text-align: left;
            padding: 8px 10px;
            border-top: 1px solid #cbd5e1;
            border-bottom: 2px solid #94a3b8;
        }

        td {
            padding: 7px 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
        }

        tr:nth-child(even) td {
            background-color: #f8fafc;
        }

        .alert-box {
            background-color: #f8fafc;
            border-left: 4px solid #0284c7;
            padding: 10px 14px;
            margin: 12px 0;
            border-radius: 0 6px 6px 0;
            font-size: 9pt;
            color: #334155;
            page-break-inside: avoid;
        }

        .alert-box strong {
            color: #0369a1;
        }

        .formula-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 14px;
            margin: 8px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 9pt;
            color: #0f2744;
            page-break-inside: avoid;
        }

        .badge-positive {
            color: #047857;
            font-weight: 700;
        }

        .badge-negative {
            color: #b91c1c;
            font-weight: 700;
        }

        a {
            color: #0284c7;
            text-decoration: none;
        }

        .footer-note {
            margin-top: 24px;
            padding-top: 10px;
            border-top: 1px solid #e2e8f0;
            font-size: 8pt;
            color: #64748b;
            text-align: center;
        }
    </style>
</head>
<body>

    <div class="header-banner">
        <div class="title-block">
            <h1>🚢 SmartyShips: Technical Evaluation Report</h1>
            <p class="subtitle">Freight Transit Duration Forecasting, SLA Risk Assessment & Delay Propagation</p>
        </div>
        <div class="meta-block">
            <strong>Candidate:</strong> Karthikesh Soma<br>
            <strong>Role:</strong> Round 2 Technical Challenge – Aiqon<br>
            <strong>Date:</strong> October 2026<br>
            <strong>Repository:</strong> github.com/karthikesh21/SmartyShips
        </div>
    </div>

    <h2>1. Executive Summary & Problem Formulation</h2>
    <p>
        Freight logistics networks face compounding uncertainties across long-haul highway lanes, break-bulk cross-dock sorting hubs, and last-mile courier routes. Inaccurate transit time estimates lead to warehouse dock congestion, missed customer Service Level Agreements (SLAs), and costly downstream expediting.
    </p>
    <p>
        <strong>SmartyShips</strong> is an end-to-end machine learning system and operational dashboard built to address three specific supply chain capabilities:
    </p>
    <ul>
        <li><strong>Pre-Shipment Duration Forecasting:</strong> Supervised regression estimating delivery duration from attributes strictly available prior to departure.</li>
        <li><strong>SLA Risk Classification:</strong> Auditing predicted arrival times against contract deadlines into standardized operational risk tiers (<em>On Time</em>, <em>Low Risk</em>, <em>Moderate Risk</em>, <em>High Risk</em>).</li>
        <li><strong>Multi-Stage Disruption Propagation:</strong> A single-ledger simulation engine modeling how upstream delays cascade across downstream transit legs without recursive double-counting.</li>
    </ul>

    <h2>2. Dataset Audit & Data Hygiene</h2>
    <h3>2.1 Source Data & Attribution</h3>
    <p>
        The platform utilizes the <strong><a href="https://www.kaggle.com/datasets/shahriarkabir/us-logistics-performance-dataset">US Logistics Performance Dataset on Kaggle</a></strong>, published by <strong>Shahriar Kabir</strong>. The raw dataset contains 2,000 shipment records across 11 features: <code>Shipment_ID</code>, <code>Origin_Warehouse</code>, <code>Destination</code>, <code>Carrier</code>, <code>Shipment_Date</code>, <code>Delivery_Date</code>, <code>Weight_kg</code>, <code>Cost</code>, <code>Status</code>, <code>Distance_miles</code>, and <code>Transit_Days</code>.
    </p>

    <div class="alert-box">
        <strong>Data Scope & Provenance Notice:</strong> The raw dataset records pre-departure attributes and calendar-day origin dispatch and delivery dates. It does <em>not</em> contain real-time GPS telemetry, weather sensor logs, road traffic feeds, customs holds, or intermediate milestone scans. Consequently, intermediate stage breakdowns (Origin Dispatch, Main Line-Haul, Transfer Hub, Delivery) and disruption propagation timelines are <strong>engineering simulations</strong> designed to model supply chain mechanics, clearly distinguished from historical data.
    </div>

    <h3>2.2 Preprocessing & Sanitization Pipeline</h3>
    <p>An exploratory audit of the 2,000 raw shipment records detected data hygiene issues requiring remediation:</p>
    <ul>
        <li><strong>Chronological Inversions (16 records / 0.8%):</strong> Shipments where <code>Delivery_Date</code> occurred prior to <code>Shipment_Date</code> (yielding negative durations down to -30 days) were removed as data entry errors.</li>
        <li><strong>Missing Delivery Dates (32 records / 1.6%):</strong> Incomplete records marked delivered but lacking arrival timestamps were excluded from duration training.</li>
        <li><strong>Lost Cargo (45 records / 2.25%):</strong> Shipments marked <code>Lost</code> were filtered out as aborted transits.</li>
        <li><strong>Weight Outliers:</strong> Zero-weight anomalies were filtered, and extreme values capped at the 99.5th percentile (152.7 kg) to mitigate undue leverage on regression tree splits.</li>
    </ul>
    <p><strong>Final Validated Training Dataset:</strong> <strong>1,800 clean records</strong> (90.0% retention).</p>

    <h3>2.3 Target Leakage Prevention</h3>
    <p>
        To ensure operational validity, features generated post-departure were strictly excluded:
    </p>
    <ul>
        <li><strong>Excluded Post-Hoc Fields:</strong> <code>Delivery_Date</code>, <code>Status</code>, and <code>Cost</code> (settled after delivery).</li>
        <li><strong>Strictly Excluded Target Leakage:</strong> The column <code>Transit_Days</code> identically matches <code>(Delivery_Date - Shipment_Date)</code> in the raw data. Feeding it into the model represents 100% target leakage, so it was strictly excluded.</li>
        <li><strong>Valid Pre-Shipment Features Retained:</strong> <code>Origin_Warehouse</code>, <code>Destination</code>, <code>Carrier</code>, <code>Distance_miles</code>, <code>Weight_kg</code>, <code>Shipment_Month</code>, <code>Shipment_DayOfWeek</code>.</li>
    </ul>

    <h2>3. Model Performance & Operational Limitations</h2>
    <h3>3.1 Evaluation Methodology</h3>
    <p>
        Models were benchmarked using a standard <strong>80/20 train/test split</strong> with a fixed seed (<code>random_state=42</code>):
    </p>
    <ul>
        <li><strong>Training Set:</strong> 1,440 validated shipments</li>
        <li><strong>Holdout Test Set:</strong> 360 shipments</li>
        <li><strong>Baseline Model:</strong> Central-tendency median delivery duration of the training set (4.00 days / 96.0 hours).</li>
        <li><strong>ML Pipeline:</strong> Scikit-Learn <code>Pipeline</code> coupling a <code>ColumnTransformer</code> (OneHotEncoder + StandardScaler) with a <code>RandomForestRegressor</code> (200 estimators, max depth 12).</li>
    </ul>

    <h3>3.2 Measured Holdout Benchmark Results</h3>
    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Median Baseline</th>
                <th>Random Forest Pipeline</th>
                <th>Net Difference</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>MAE (Days)</strong></td>
                <td>2.114 days</td>
                <td><strong>2.016 days</strong></td>
                <td><span class="badge-positive">-0.098 days</span></td>
            </tr>
            <tr>
                <td><strong>MAE (Hours)</strong></td>
                <td>50.73 hours</td>
                <td><strong>48.38 hours</strong></td>
                <td><span class="badge-positive">-2.35 hours (4.6% error reduction)</span></td>
            </tr>
            <tr>
                <td><strong>RMSE (Days)</strong></td>
                <td>3.359 days</td>
                <td><strong>3.105 days</strong></td>
                <td><span class="badge-positive">-0.254 days</span></td>
            </tr>
            <tr>
                <td><strong>RMSE (Hours)</strong></td>
                <td>80.63 hours</td>
                <td><strong>74.53 hours</strong></td>
                <td><span class="badge-positive">-6.10 hours variance reduction</span></td>
            </tr>
            <tr>
                <td><strong>R² Score</strong></td>
                <td>-0.0886</td>
                <td><strong>+0.0699</strong></td>
                <td><span class="badge-positive">+0.1585 gain</span></td>
            </tr>
        </tbody>
    </table>

    <h3>3.3 Plain-English Metric Interpretation</h3>
    <ul>
        <li><strong>MAE (Mean Absolute Error):</strong> Represents the average absolute error magnitude. An MAE of 48.38 hours indicates that duration forecasts miss actual arrival by approximately 2 days on average.</li>
        <li><strong>RMSE (Root Mean Squared Error):</strong> Penalizes large outlier errors more heavily due to squaring. At 74.53 hours (versus 80.63 hours for baseline), the Random Forest successfully dampens extreme forecast misses.</li>
        <li><strong>R² (Coefficient of Determination):</strong> At <strong>0.0699</strong> (~7% variance explained), the model indicates that static pre-departure features weakly explain total duration variance, with ~93% driven by unobserved in-transit dynamics (roadway congestion, weather delays, terminal dwell times).</li>
    </ul>

    <h3>3.4 Signed Error Convention</h3>
    <p>Residuals are defined using the consistent signed error convention:</p>
    <div class="formula-box">
        prediction_error_hours = (predicted_duration_days - actual_duration_days) * 24.0
    </div>
    <ul>
        <li><strong>Negative values:</strong> Underprediction (model forecasted shorter delivery than actual transit).</li>
        <li><strong>Positive values:</strong> Overprediction (model forecasted longer delivery than actual transit).</li>
    </ul>

    <h3>3.5 Transparent Discussion of Model Limitations</h3>
    <ul>
        <li><strong>Modest Improvement:</strong> While the Random Forest reliably beats the naive median baseline, it provides a modest gain (2.35 hours) rather than high precision. Outputs are operational estimates for planning buffers, not guaranteed commitments.</li>
        <li><strong>Calendar-Day Resolution:</strong> Timestamps in the source data record calendar days without hours/minutes, introducing unavoidable discrete rounding noise.</li>
        <li><strong>Single Random Split:</strong> A single random 80/20 partition does not establish temporal generalization across future operating seasons or newly introduced carrier corridors.</li>
        <li><strong>Proposed Future Work:</strong> Temporal backtesting cross-validation, integration of live GPS/ELD telematics (Samsara/Geotab), EDI 214 milestone scan feeds, and real-time NOAA weather impedance data.</li>
    </ul>

    <h2>4. ETA Arithmetic & Contract SLA Risk Engine</h2>
    <p>All timestamps are strictly standardized to timezone-aware UTC:</p>
    <div class="formula-box">
        ETA_UTC = Departure_Timestamp_UTC + Predicted_Duration_Hours
    </div>
    <p>When a contractual delivery deadline (Target SLA) is specified:</p>
    <div class="formula-box">
        Delay_Hours = (ETA_UTC - Expected_Arrival_UTC) / 3600.0
    </div>
    <ul>
        <li><strong>On Time / Early (&le; 0 hrs):</strong> Delivery is projected on schedule or ahead of buffer.</li>
        <li><strong>Low Risk (0 to 12 hrs):</strong> Minor expected delay within routine operating buffers.</li>
        <li><strong>Moderate Risk (12 to 24 hrs):</strong> Delay exceeds standard buffer; priority tracking recommended.</li>
        <li><strong>High Risk (&gt; 24 hrs):</strong> Severe delay; imminent contractual SLA breach requiring active dispatch intervention.</li>
    </ul>

    <h2>5. Multi-Stage Disruption Propagation Simulator</h2>
    <p>
        Supply chains consist of connected legs: <strong>Origin Dispatch (15%)</strong>, <strong>Main Line-Haul Transit (55%)</strong>, <strong>Transfer Hub Cross-Dock (20%)</strong>, and <strong>Final Delivery (10%)</strong>.
    </p>
    <p>
        The simulator implements a <strong>guaranteed single cumulative delay ledger</strong>:
    </p>
    <div class="formula-box">
        Stage_Start[i] = Stage_End[i-1]<br>
        Stage_End[i] = Stage_Start[i] + NominalDuration[i] + InjectedDelay[i] - RecoveredBuffer[i]
    </div>
    <p>
        When a +6.0 hour delay is injected during Main Transit, the final delivery ETA shifts by <em>exactly +6.0 hours</em> (when no recovery buffer is active). The bottleneck is applied once and carried downstream across all remaining legs without recursive re-injection.
    </p>

    <h2>6. Operational Feature Importance</h2>
    <p>
        Aggregating one-hot tree split importances back to operational parent features indicates their empirical association with transit duration:
    </p>
    <ul>
        <li><strong>Transit Distance (~42%):</strong> Primary structural foundation for line-haul movement.</li>
        <li><strong>Carrier Selection (~23%):</strong> Operational speed variances between express networks (FedEx, UPS) vs. standard regional ground freight.</li>
        <li><strong>Cargo Weight (~16%):</strong> Heavy consignments require pallet handling, reducing terminal cross-dock throughput speed.</li>
        <li><strong>Origin-Destination Corridor (~14%):</strong> Metropolitan corridor density and hub route infrastructure.</li>
        <li><strong>Dispatch Day of Week / Month (~5%):</strong> Weekend staging depot cycles and seasonal shipping volume shifts.</li>
    </ul>

    <div class="footer-note">
        SmartyShips Technical Evaluation Report &bull; Sourced from <a href="https://www.kaggle.com/datasets/shahriarkabir/us-logistics-performance-dataset">US Logistics Performance Dataset on Kaggle</a> by Shahriar Kabir &bull; Code & Tests: github.com/karthikesh21/SmartyShips
    </div>

</body>
</html>
"""

report_html_path = Path("report/technical_report.html")
report_html_path.write_text(html_content, encoding="utf-8")
print(f"Generated HTML report at: {report_html_path.resolve()}")

# Use Microsoft Edge headless to print to PDF
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not Path(edge_path).exists():
    edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

pdf_output_path = Path("report/SmartyShips_Technical_Report.pdf").resolve()
cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_output_path}",
    str(report_html_path.resolve())
]

print("Rendering PDF via headless Edge...")
res = subprocess.run(cmd, capture_output=True, text=True)
if pdf_output_path.exists():
    print(f"SUCCESS: Generated PDF report at: {pdf_output_path} ({pdf_output_path.stat().st_size} bytes)")
else:
    print(f"Failed to generate PDF. Edge stderr: {res.stderr}")
