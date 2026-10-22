"""
SmartyShips - Intelligent Shipment Analytics & Delay Propagation Platform
Streamlit operations dashboard for predictive freight transit duration,
schedule-relative SLA risk categorization, and multi-stage disruption simulation.
Branded for SmartyShips with dynamic light and dark theme synchronization.
"""

from datetime import datetime, date, time, timedelta, timezone
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Clean table renderer without pyarrow dependency (avoiding Windows SAC DLL restrictions)
def _safe_render_table(data, *args, **kwargs):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if len(data) > 100:
        st.caption(f"Displaying first 100 of {len(data)} records:")
        data = data.head(100)
    html = data.to_html(index=False)
    st.markdown(f'<div style="overflow-x:auto; width:100%; margin:12px 0;">{html}</div>', unsafe_allow_html=True)

st.dataframe = _safe_render_table
st.table = _safe_render_table

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
from src.chart_theme import get_theme_palette, apply_smarty_theme

# Page configuration
st.set_page_config(
    page_title="SmartyShips | Intelligent Shipment Analytics",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Theme State Initialization
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

# Sidebar: Controls
st.sidebar.markdown("### Interface Mode")
is_dark = st.sidebar.toggle("Dark Theme", value=(st.session_state["theme_mode"] == "dark"))
st.session_state["theme_mode"] = "dark" if is_dark else "light"
dark_mode = (st.session_state["theme_mode"] == "dark")
palette = get_theme_palette(dark_mode)

# 2. Dynamic High-Contrast, Colorful CSS
if dark_mode:
    theme_css = """
    <style>
        /* Base Dark Canvas */
        .stApp {
            background-color: #0b0f19 !important;
            color: #f8fafc !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        header[data-testid="stHeader"], [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] {
            background-color: #030712 !important;
            border-right: 1px solid #1f2937 !important;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        p, span, div {
            color: #cbd5e1;
        }

        /* Top Brand Navigation Bar */
        .brand-header-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 12px;
            border-bottom: 1px solid #1f2937;
            margin-bottom: 16px;
        }
        .brand-title {
            font-size: clamp(1.5rem, 2.6vw, 2.1rem);
            font-weight: 800;
            color: #38bdf8 !important;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-sub {
            color: #94a3b8 !important;
            font-size: clamp(0.85rem, 1vw, 0.95rem);
            margin: 2px 0 0 0;
            font-weight: 500;
        }
        .brand-pill {
            background-color: #111827;
            border: 1px solid #1f2937;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            color: #38bdf8 !important;
            font-weight: 600;
        }

        /* Horizontal Tab Bar */
        div[data-testid="stRadio"] div[role="radiogroup"] {
            background-color: #111827;
            padding: 5px;
            border-radius: 10px;
            border: 1px solid #1f2937;
            gap: 6px;
            margin-bottom: 18px;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background-color: transparent;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            border: none !important;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background-color: #1f2937 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background-color: #0284c7 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #f1f5f9 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        /* Widget Labels */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            color: #e2e8f0 !important;
            font-weight: 600 !important;
            font-size: clamp(0.82rem, 1vw, 0.92rem) !important;
        }

        /* Input Controls */
        input,
        input[type="number"],
        input[type="text"],
        div[data-baseweb="input"],
        div[data-baseweb="input"] > div,
        div[data-baseweb="input"] input,
        .stDateInput > div,
        .stDateInput input,
        .stTimeInput > div,
        .stTimeInput input {
            color: #ffffff !important;
            background-color: #111827 !important;
            font-weight: 600 !important;
            font-size: clamp(0.9rem, 1.2vw, 1.05rem) !important;
            border-color: #374151 !important;
        }
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {
            background-color: #1f2937 !important;
            color: #f8fafc !important;
            border-color: #374151 !important;
        }
        button[data-testid="stNumberInputStepDown"] svg,
        button[data-testid="stNumberInputStepUp"] svg {
            fill: #f8fafc !important;
        }

        /* Dropdowns & Multiselect Popovers (Dark Mode) */
        div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] > div,
        div[data-testid="stSelectbox"] > div {
            background-color: #111827 !important;
            border-color: #374151 !important;
            color: #ffffff !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-testid="stMultiSelect"] span,
        div[data-testid="stMultiSelect"] div {
            color: #ffffff !important;
        }

        /* Multiselect Tags/Chips */
        [data-testid="stMultiSelectTagsContainer"] span,
        [data-tag] {
            background-color: #1e293b !important;
            color: #38bdf8 !important;
            border: 1px solid #334155 !important;
        }

        /* Dropdown Popover Menus (Both Modern Floating UI and Legacy BaseWeb) */
        [data-testid="stMultiSelectDropdown"],
        [data-testid="stMultiSelectDropdown"] *,
        [data-testid="stSelectboxVirtualDropdown"],
        [data-testid="stSelectboxVirtualDropdown"] *,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] *,
        ul[role="listbox"],
        ul[role="listbox"] *,
        div[role="listbox"],
        div[role="listbox"] *,
        li[role="option"],
        li[role="option"] *,
        div[role="option"],
        div[role="option"] * {
            background-color: #111827 !important;
            color: #f8fafc !important;
            border-color: #374151 !important;
        }

        /* Hover & Selected states */
        [data-testid="stMultiSelectDropdown"] [role="option"]:hover,
        [data-testid="stMultiSelectDropdown"] [role="option"][aria-selected="true"],
        [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        div[role="option"]:hover,
        div[role="option"][aria-selected="true"],
        [data-hovered="true"],
        [data-focused="true"],
        [data-item-hl="true"] {
            background-color: #1f2937 !important;
            color: #38bdf8 !important;
        }

        /* Select All button / header item */
        [data-testid="stMultiSelectDropdown"] button,
        [data-testid="stMultiSelectDropdown"] > div:first-child {
            background-color: #1f2937 !important;
            color: #38bdf8 !important;
            border-bottom: 1px solid #374151 !important;
        }

        /* KPI Colorful Cards */
        .kpi-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 16px 18px;
            position: relative;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }
        .kpi-card.cyan-accent { border-top: 4px solid #06b6d4; }
        .kpi-card.blue-accent { border-top: 4px solid #3b82f6; }
        .kpi-card.green-accent { border-top: 4px solid #10b981; }
        .kpi-card.amber-accent { border-top: 4px solid #f59e0b; }
        .kpi-card.red-accent { border-top: 4px solid #ef4444; }
        .kpi-card.purple-accent { border-top: 4px solid #8b5cf6; }

        .kpi-title {
            color: #94a3b8;
            font-size: clamp(0.72rem, 0.85vw, 0.82rem);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .kpi-value {
            color: #f8fafc;
            font-size: clamp(1.4rem, 2.4vw, 2.0rem);
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
        }
        .kpi-subtitle {
            color: #38bdf8;
            font-size: clamp(0.75rem, 0.9vw, 0.85rem);
            font-weight: 500;
            margin-top: 4px;
        }

        /* Risk Badges */
        .badge-ontime {
            background-color: #064e3b;
            color: #6ee7b7;
            border: 1px solid #047857;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-low {
            background-color: #78350f;
            color: #fde68a;
            border: 1px solid #b45309;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-moderate {
            background-color: #7c2d12;
            color: #fed7aa;
            border: 1px solid #c2410c;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-high {
            background-color: #7f1d1d;
            color: #fca5a5;
            border: 1px solid #b91c1c;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }

        /* Callout Box */
        .callout-box {
            background-color: #111827;
            border-left: 4px solid #38bdf8;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin: 10px 0;
            font-size: clamp(0.85rem, 1vw, 0.95rem);
            color: #e2e8f0;
            line-height: 1.6;
            border-top: 1px solid #1f2937;
            border-right: 1px solid #1f2937;
            border-bottom: 1px solid #1f2937;
        }
        .notice-box {
            background-color: #111827;
            border: 1px solid #374151;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: clamp(0.82rem, 0.95vw, 0.9rem);
            color: #cbd5e1;
        }

        /* Buttons */
        button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            box-shadow: 0 2px 4px rgba(2, 132, 199, 0.3) !important;
        }
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"] *,
        table, th, td {
            color: #f8fafc !important;
            background-color: #111827 !important;
            border-color: #374151 !important;
        }
        th {
            background-color: #1f2937 !important;
            font-weight: 700 !important;
        }
    </style>
    """
else:
    theme_css = """
    <style>
        /* Base Light Canvas */
        .stApp {
            background-color: #f8fafc !important;
            color: #0f172a !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        header[data-testid="stHeader"], [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] * {
            color: #0f172a;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #0f2744 !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        p, span, div {
            color: #334155;
        }

        /* Top Brand Navigation Bar */
        .brand-header-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 12px;
            border-bottom: 1px solid #cbd5e1;
            margin-bottom: 16px;
        }
        .brand-title {
            font-size: clamp(1.5rem, 2.6vw, 2.1rem);
            font-weight: 800;
            color: #0284c7 !important;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-sub {
            color: #475569 !important;
            font-size: clamp(0.85rem, 1vw, 0.95rem);
            margin: 2px 0 0 0;
            font-weight: 500;
        }
        .brand-pill {
            background-color: #f0f9ff;
            border: 1px solid #bae6fd;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            color: #0284c7 !important;
            font-weight: 600;
        }

        /* Horizontal Tab Bar */
        div[data-testid="stRadio"] div[role="radiogroup"] {
            background-color: #f1f5f9;
            padding: 5px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            gap: 6px;
            margin-bottom: 18px;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background-color: transparent;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            border: none !important;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background-color: #e2e8f0 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background-color: #0284c7 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p,
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #0f2744 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        /* Widget Labels */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            color: #0f2744 !important;
            font-weight: 700 !important;
            font-size: clamp(0.82rem, 1vw, 0.92rem) !important;
        }

        /* Input Controls */
        input,
        input[type="number"],
        input[type="text"],
        div[data-baseweb="input"],
        div[data-baseweb="input"] > div,
        div[data-baseweb="input"] input,
        .stDateInput > div,
        .stDateInput input,
        .stTimeInput > div,
        .stTimeInput input {
            color: #0f172a !important;
            background-color: #ffffff !important;
            font-weight: 700 !important;
            font-size: clamp(0.9rem, 1.2vw, 1.05rem) !important;
            border-color: #cbd5e1 !important;
        }
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }
        button[data-testid="stNumberInputStepDown"] svg,
        button[data-testid="stNumberInputStepUp"] svg {
            fill: #0f172a !important;
        }

        /* Dropdowns & Multiselect Popovers (Light Mode) */
        div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] > div,
        div[data-testid="stSelectbox"] > div {
            background-color: #ffffff !important;
            border-color: #cbd5e1 !important;
            color: #0f172a !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-testid="stMultiSelect"] span,
        div[data-testid="stMultiSelect"] div {
            color: #0f172a !important;
        }

        /* Multiselect Tags/Chips */
        [data-testid="stMultiSelectTagsContainer"] span,
        [data-tag] {
            background-color: #f1f5f9 !important;
            color: #0284c7 !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Dropdown Popover Menus */
        [data-testid="stMultiSelectDropdown"],
        [data-testid="stMultiSelectDropdown"] *,
        [data-testid="stSelectboxVirtualDropdown"],
        [data-testid="stSelectboxVirtualDropdown"] *,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] *,
        ul[role="listbox"],
        ul[role="listbox"] *,
        div[role="listbox"],
        div[role="listbox"] *,
        li[role="option"],
        li[role="option"] *,
        div[role="option"],
        div[role="option"] * {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }

        /* Hover & Selected states */
        [data-testid="stMultiSelectDropdown"] [role="option"]:hover,
        [data-testid="stMultiSelectDropdown"] [role="option"][aria-selected="true"],
        [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        div[role="option"]:hover,
        div[role="option"][aria-selected="true"],
        [data-hovered="true"],
        [data-focused="true"],
        [data-item-hl="true"] {
            background-color: #f1f5f9 !important;
            color: #0284c7 !important;
        }

        /* Select All button / header item */
        [data-testid="stMultiSelectDropdown"] button,
        [data-testid="stMultiSelectDropdown"] > div:first-child {
            background-color: #f8fafc !important;
            color: #0284c7 !important;
            border-bottom: 1px solid #e2e8f0 !important;
        }

        /* KPI Colorful Cards */
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }
        .kpi-card.cyan-accent { border-top: 4px solid #0891b2; }
        .kpi-card.blue-accent { border-top: 4px solid #0284c7; }
        .kpi-card.green-accent { border-top: 4px solid #059669; }
        .kpi-card.amber-accent { border-top: 4px solid #d97706; }
        .kpi-card.red-accent { border-top: 4px solid #dc2626; }
        .kpi-card.purple-accent { border-top: 4px solid #7c3aed; }

        .kpi-title {
            color: #64748b;
            font-size: clamp(0.72rem, 0.85vw, 0.82rem);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .kpi-value {
            color: #0f2744;
            font-size: clamp(1.4rem, 2.4vw, 2.0rem);
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
        }
        .kpi-subtitle {
            color: #0284c7;
            font-size: clamp(0.75rem, 0.9vw, 0.85rem);
            font-weight: 500;
            margin-top: 4px;
        }

        /* Risk Badges */
        .badge-ontime {
            background-color: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-low {
            background-color: #fefce8;
            color: #854d0e;
            border: 1px solid #fef08a;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-moderate {
            background-color: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-high {
            background-color: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }

        /* Callout Box */
        .callout-box {
            background-color: #f1f5f9;
            border-left: 4px solid #0284c7;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin: 10px 0;
            font-size: clamp(0.85rem, 1vw, 0.95rem);
            color: #1e293b;
            line-height: 1.6;
            border-top: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }
        .notice-box {
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: clamp(0.82rem, 0.95vw, 0.9rem);
            color: #334155;
        }

        /* Buttons */
        button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2) !important;
        }
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"] *,
        table, th, td {
            color: #0f172a !important;
            background-color: #ffffff !important;
            border-color: #e2e8f0 !important;
        }
        th {
            background-color: #f8fafc !important;
            font-weight: 700 !important;
        }
    </style>
    """

# Inject theme CSS immediately
st.markdown(theme_css, unsafe_allow_html=True)

# Sidebar: Risk Thresholds
st.sidebar.markdown("---")
st.sidebar.markdown("### Risk Thresholds (Hours)")
thresh_on_time = st.sidebar.number_input("On-Time / Early (<= hrs)", value=0.0, step=1.0)
thresh_low = st.sidebar.number_input("Low Delay (<= hrs)", value=12.0, step=1.0)
thresh_mod = st.sidebar.number_input("Moderate Delay (<= hrs)", value=24.0, step=1.0)
user_thresholds = {
    "on_time_max": float(thresh_on_time),
    "low_risk_max": float(thresh_low),
    "moderate_risk_max": float(thresh_mod)
}
st.sidebar.caption("Configurable thresholds for schedule delay risk categorization.")

# Data and Model loaders
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

try:
    df_processed = load_processed_data()
    pipeline = load_trained_pipeline()
    metrics = load_metrics_json()
    test_eval_df = load_test_predictions()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# ==============================================================================
# MAIN BRAND HEADER
# ==============================================================================
st.markdown("""
<div class="brand-header-box">
    <div>
        <div class="brand-title">
            <span>🚢</span> SmartyShips
        </div>
        <p class="brand-sub">Intelligent Shipment Analytics & Delay Propagation</p>
    </div>
    <div style="display: flex; align-items: center; font-size: 1.6rem;">
        <span>🚢</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Horizontal Navigation Bar
NAV_OPTIONS = [
    "ETA & Delay Estimator",
    "Disruption Simulator",
    "Model Evaluation & Residuals",
    "Data Quality & Audit",
    "Route Network & Records"
]

if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = NAV_OPTIONS[0]

active_view = st.radio(
    "Navigation Tabs",
    NAV_OPTIONS,
    horizontal=True,
    index=NAV_OPTIONS.index(st.session_state["active_nav"]),
    label_visibility="collapsed"
)
st.session_state["active_nav"] = active_view

# ==============================================================================
# SECTION 1: ETA & DELAY ESTIMATOR
# ==============================================================================
if active_view == "ETA & Delay Estimator":
    st.markdown("### Transit Duration & Schedule Delay Estimation")
    st.markdown("""
    Estimate transit duration and arrival timestamps (ETA) using pre-departure operational parameters.
    Specify a target delivery SLA to assess schedule variance and risk tiers.
    """)

    col_mode, col_preset = st.columns([1, 2])
    with col_mode:
        input_source = st.radio("Input Method:", ["Custom Route", "Historical Shipment Sample"], horizontal=True)

    default_origin = "Warehouse_MIA"
    default_dest = "San Francisco"
    default_carrier = "UPS"
    default_dist = 2600.0
    default_weight = 25.0
    default_date = date.today()
    default_time = time(8, 0)
    sample_id_text = "Custom Input"

    if input_source == "Historical Shipment Sample":
        with col_preset:
            sample_ids = df_processed["Shipment_ID"].tolist()
            selected_id = st.selectbox("Select Historical Sample ID:", sample_ids[:50])
            sample_row = df_processed[df_processed["Shipment_ID"] == selected_id].iloc[0]
            default_origin = sample_row["Origin_Warehouse"]
            default_dest = sample_row["Destination"]
            default_carrier = sample_row["Carrier"]
            default_dist = float(sample_row["Distance_miles"])
            default_weight = float(sample_row["Weight_kg"])
            sample_id_text = f"Sample {selected_id} ({sample_row['Status']})"
            st.caption(f"Loaded {sample_id_text} | Corridor: {default_origin} -> {default_dest} ({default_dist:,.0f} mi) | Historical Duration: {sample_row.get('delivery_duration_days', 'N/A')} days")

    with st.form("prediction_form"):
        st.markdown("#### 1. Shipment Attributes")
        c1, c2, c3 = st.columns(3)
        with c1:
            origins = sorted(df_processed["Origin_Warehouse"].dropna().unique().tolist())
            origin = st.selectbox("Origin Warehouse", origins, index=origins.index(default_origin) if default_origin in origins else 0)
            distance = st.number_input("Distance (miles)", min_value=50.0, max_value=3500.0, value=default_dist, step=50.0)
        with c2:
            dests = sorted(df_processed["Destination"].dropna().unique().tolist())
            dest = st.selectbox("Destination City", dests, index=dests.index(default_dest) if default_dest in dests else 0)
            weight = st.number_input("Cargo Weight (kg)", min_value=0.1, max_value=500.0, value=min(default_weight, 500.0), step=1.0)
        with c3:
            carriers = sorted(df_processed["Carrier"].dropna().unique().tolist())
            carrier = st.selectbox("Carrier", carriers, index=carriers.index(default_carrier) if default_carrier in carriers else 0)

        st.markdown("#### 2. Schedule & Target Delivery")
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Planned Departure (UTC)**")
            cd1, cd2 = st.columns(2)
            with cd1:
                dep_date = st.date_input("Departure Date", value=default_date)
            with cd2:
                dep_time = st.time_input("Departure Time", value=default_time)
            planned_departure = datetime.combine(dep_date, dep_time).replace(tzinfo=timezone.utc)

        with c5:
            st.markdown("**Target Delivery SLA (UTC)**")
            use_expected = st.checkbox("Compare against Target Delivery Deadline", value=True)
            if use_expected:
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    exp_date = st.date_input("Target Date", value=default_date + timedelta(days=4))
                with exp_col2:
                    exp_time = st.time_input("Target Time", value=time(18, 0))
                expected_arrival = datetime.combine(exp_date, exp_time).replace(tzinfo=timezone.utc)
            else:
                expected_arrival = None

        submit_btn = st.form_submit_button("Calculate Transit & Schedule Risk", type="primary")

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

    # 4 Colorful Primary KPI Cards
    st.markdown("---")
    st.markdown("### Prediction Summary")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-card cyan-accent">
            <div class="kpi-title">Predicted Duration</div>
            <div class="kpi-value">{pred_res['predicted_duration_days']:.1f} <span style="font-size:1.1rem; font-weight:500;">days</span></div>
            <div class="kpi-subtitle">{pred_res['predicted_duration_hours']:.1f} operational hours</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        eta_dt = normalize_timestamp(pred_res['predicted_eta_utc'])
        st.markdown(f"""
        <div class="kpi-card blue-accent">
            <div class="kpi-title">Estimated Arrival (ETA)</div>
            <div class="kpi-value" style="font-size:clamp(1.2rem, 1.8vw, 1.6rem); color:{palette['primary_blue']};">{eta_dt.strftime('%b %d, %Y')}</div>
            <div class="kpi-subtitle">{eta_dt.strftime('%H:%M')} UTC</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        delay_h = pred_res['estimated_delay_hours']
        if delay_h is not None:
            delay_text = f"{delay_h:+.1f} hrs"
            if delay_h > 0:
                card_accent = "red-accent" if delay_h > user_thresholds["moderate_risk_max"] else "amber-accent"
                val_color = palette["crimson_red"] if delay_h > user_thresholds["moderate_risk_max"] else palette["amber_orange"]
                subtext = f"Late by {delay_h:.1f} hrs past SLA"
            else:
                card_accent = "green-accent"
                val_color = palette["emerald_green"]
                subtext = f"Early by {abs(delay_h):.1f} hrs buffer"
        else:
            card_accent = "blue-accent"
            delay_text = "N/A"
            subtext = "No SLA target provided"
            val_color = palette["text_muted"]

        st.markdown(f"""
        <div class="kpi-card {card_accent}">
            <div class="kpi-title">Schedule Variance</div>
            <div class="kpi-value" style="color: {val_color};">{delay_text}</div>
            <div class="kpi-subtitle" style="color: {val_color};">{subtext}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        cat = risk_info["category"]
        badge_cls = {
            "On Time / Early": "badge-ontime",
            "Low Risk": "badge-low",
            "Moderate Risk": "badge-moderate",
            "High Risk": "badge-high",
            "Schedule Not Set": "badge-low"
        }.get(cat, "badge-low")

        accent_map = {
            "On Time / Early": "green-accent",
            "Low Risk": "amber-accent",
            "Moderate Risk": "amber-accent",
            "High Risk": "red-accent",
            "Schedule Not Set": "blue-accent"
        }

        st.markdown(f"""
        <div class="kpi-card {accent_map.get(cat, 'blue-accent')}">
            <div class="kpi-title">Risk Assessment</div>
            <div style="margin-top: 8px;">
                <span class="{badge_cls}">{cat}</span>
            </div>
            <div class="kpi-subtitle" style="color: {palette['text_muted']}; margin-top: 8px;">
                Rule: {risk_info['threshold_applied']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Route Summary & Feature Attribution
    st.markdown("#### Route Characteristics & Feature Drivers")
    exp_col_left, exp_col_right = st.columns([3, 2])

    importances_df = get_feature_importances(pipeline)

    with exp_col_left:
        plain_text = generate_plain_english_explanation(pred_res, risk_info, importances_df)
        st.markdown(f"""
        <div class="callout-box">
            {plain_text}
        </div>
        """, unsafe_allow_html=True)

    with exp_col_right:
        fig_imp = px.bar(
            importances_df.head(6),
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            labels={"Importance_Pct": "Weight (%)", "Feature": ""},
            color="Importance_Pct",
            color_continuous_scale="Tealgrn" if dark_mode else "Blues"
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            height=260,
            margin=dict(l=10, r=10, t=35, b=10),
            coloraxis_showscale=False
        )
        apply_smarty_theme(fig_imp, dark_mode, title="Feature Attribution Weights (%)")
        st.plotly_chart(fig_imp, width="stretch")

    # Nominal Stage Timeline
    st.markdown("#### Projected Milestone Breakdown")
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
            "Duration (hrs)": s["Nominal_Duration_Hours"]
        })
    df_tl = pd.DataFrame(timeline_data)

    fig_tl = px.timeline(
        df_tl,
        x_start="Start",
        x_end="End",
        y="Stage",
        color="Stage",
        color_discrete_sequence=["#06b6d4", "#3b82f6", "#8b5cf6", "#10b981"],
        hover_data=["Duration (hrs)"]
    )
    fig_tl.update_yaxes(autorange="reversed")
    fig_tl.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False
    )
    apply_smarty_theme(fig_tl, dark_mode, title="Corridor Leg Schedule (Planned Milestone Progress)")
    st.plotly_chart(fig_tl, width="stretch")

# ==============================================================================
# SECTION 2: DISRUPTION SIMULATOR
# ==============================================================================
elif active_view == "Disruption Simulator":
    st.markdown("### Multi-Stage Disruption Propagation Simulator")
    st.markdown("""
    <div class="notice-box">
        <strong>Simulation Mechanism:</strong>
        Freight corridors are divided into operational legs: Origin Dispatch (15%), Main Transit (55%), Sorting Hub (20%), and Final Delivery (10%).
        Disruptions injected upstream propagate downstream using a cumulative delay ledger, ensuring delays are not double-counted.
    </div>
    """, unsafe_allow_html=True)

    sim_c1, sim_c2 = st.columns([1, 2])
    with sim_c1:
        st.markdown("#### Disruption Settings")
        sim_base_hours = st.slider("Baseline Nominal Duration (Hours)", min_value=24.0, max_value=240.0, value=96.0, step=6.0)
        
        sim_stage = st.selectbox(
            "Disrupted Leg:",
            ["None"] + list(STAGE_PROPORTIONS.keys())
        )
        
        sim_delay_injected = st.slider(
            "Injected Delay (Hours):",
            min_value=0.0, max_value=72.0, value=18.0 if sim_stage != "None" else 0.0, step=2.0
        )
        
        sim_recovery = st.slider(
            "Downstream Buffer Absorption (Hours):",
            min_value=0.0, max_value=24.0, value=4.0 if sim_stage != "None" else 0.0, step=1.0,
            help="Simulates expedited line-haul or dock priority to mitigate downstream cascading delay."
        )

        sim_dep_date = st.date_input("Simulation Dispatch Date", value=date.today())
        sim_dep_dt = datetime.combine(sim_dep_date, time(8, 0)).replace(tzinfo=timezone.utc)

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
            st.markdown(f"""
            <div class="kpi-card blue-accent">
                <div class="kpi-title">Nominal Duration</div>
                <div class="kpi-value">{sim_result['total_nominal_hours']:.1f} <span style="font-size:1.0rem;">hrs</span></div>
                <div class="kpi-subtitle">Planned Corridor Schedule</div>
            </div>
            """, unsafe_allow_html=True)
        with pk2:
            delay_val_color = palette["crimson_red"] if sim_result['net_propagated_delay_hours'] > 0 else palette["emerald_green"]
            st.markdown(f"""
            <div class="kpi-card red-accent" style="border-top-color: {delay_val_color};">
                <div class="kpi-title">Net Delay Shift</div>
                <div class="kpi-value" style="color: {delay_val_color};">+{sim_result['net_propagated_delay_hours']:.1f} <span style="font-size:1.0rem;">hrs</span></div>
                <div class="kpi-subtitle">Injected: +{sim_delay_injected:.1f}h</div>
            </div>
            """, unsafe_allow_html=True)
        with pk3:
            st.markdown(f"""
            <div class="kpi-card amber-accent">
                <div class="kpi-title">Final Delayed Duration</div>
                <div class="kpi-value">{sim_result['total_simulated_hours']:.1f} <span style="font-size:1.0rem;">hrs</span></div>
                <div class="kpi-subtitle">Buffer Absorbed: -{sim_delay_injected - sim_result['net_propagated_delay_hours']:.1f}h</div>
            </div>
            """, unsafe_allow_html=True)

        # Timeline comparison
        gantt_rows = []
        for s in sim_result["stages"]:
            gantt_rows.append({
                "Timeline": "Baseline (Nominal)",
                "Stage": s["Stage"],
                "Start": s["Nominal_Start"],
                "End": s["Nominal_End"],
                "Duration": s["Nominal_Duration_Hours"]
            })
            gantt_rows.append({
                "Timeline": "Disrupted (Propagated)",
                "Stage": s["Stage"],
                "Start": s["Simulated_Start"],
                "End": s["Simulated_End"],
                "Duration": s["Actual_Duration_Hours"]
            })

        df_gantt = pd.DataFrame(gantt_rows)
        fig_prop = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="End",
            y="Timeline",
            color="Stage",
            color_discrete_sequence=["#06b6d4", "#3b82f6", "#8b5cf6", "#ef4444"],
            hover_data=["Stage", "Duration"]
        )
        fig_prop.update_yaxes(autorange="reversed")
        fig_prop.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
        apply_smarty_theme(fig_prop, dark_mode, title=f"Comparative Timeline: Baseline vs Disrupted ({sim_stage} +{sim_delay_injected:.1f}h)")
        st.plotly_chart(fig_prop, width="stretch")

    st.markdown("#### Cumulative Stage Disruption Ledger (Guaranteed Single Ledger)")
    table_records = []
    for s in sim_result["stages"]:
        table_records.append({
            "Stage": s["Stage"],
            "Nominal Duration": f"{s['Nominal_Duration_Hours']:.1f} hrs",
            "Injected Delay": f"+{s['Injected_Delay_Hours']:.1f} hrs" if s["Injected_Delay_Hours"] > 0 else "0.0 hrs",
            "Buffer Absorption": f"-{s['Recovered_Buffer_Hours']:.1f} hrs" if s["Recovered_Buffer_Hours"] > 0 else "0.0 hrs",
            "Effective Duration": f"{s['Actual_Duration_Hours']:.1f} hrs",
            "Cumulative Delay Total": f"{s['Cumulative_Delay_Hours']:.1f} hrs",
            "Simulated Arrival (UTC)": s["Simulated_End"].strftime("%Y-%m-%d %H:%M")
        })
    st.dataframe(pd.DataFrame(table_records), width="stretch", hide_index=True)

    with st.expander("Enterprise Roadmap: Live Telematics Integration Plan"):
        roadmap_items = get_real_deployment_roadmap()
        st.table(pd.DataFrame(roadmap_items))

# ==============================================================================
# SECTION 3: MODEL EVALUATION & RESIDUALS
# ==============================================================================
elif active_view == "Model Evaluation & Residuals":
    st.markdown("### Machine Learning Model Performance & Residual Diagnostics")
    st.markdown("""
    Holdout evaluation on 360 shipments (20% test partition, seed=42) comparing the
    production **Random Forest Pipeline** against the **Median Baseline**.
    """)

    b_m = metrics["baseline"]
    rf_m = metrics["random_forest"]
    cmp = metrics["comparison"]

    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1:
        st.markdown(f"""
        <div class="kpi-card amber-accent">
            <div class="kpi-title">Baseline MAE</div>
            <div class="kpi-value">{b_m['mae_hours']} <span style="font-size:1.0rem;">hrs</span></div>
            <div class="kpi-subtitle">{b_m['mae_days']} days (Median: {b_m['median_duration_days']}d)</div>
        </div>
        """, unsafe_allow_html=True)

    with ev2:
        st.markdown(f"""
        <div class="kpi-card blue-accent">
            <div class="kpi-title">Random Forest MAE</div>
            <div class="kpi-value" style="color: {palette['primary_blue']};">{rf_m['mae_hours']} <span style="font-size:1.0rem;">hrs</span></div>
            <div class="kpi-subtitle">{rf_m['mae_days']} days (Gain: {cmp['mae_improvement_hours']:+.1f}h)</div>
        </div>
        """, unsafe_allow_html=True)

    with ev3:
        st.markdown(f"""
        <div class="kpi-card purple-accent">
            <div class="kpi-title">Random Forest RMSE</div>
            <div class="kpi-value" style="color: {palette['purple_violet']};">{rf_m['rmse_hours']} <span style="font-size:1.0rem;">hrs</span></div>
            <div class="kpi-subtitle">{rf_m['rmse_days']} days (Gain: {cmp['rmse_improvement_hours']:+.1f}h)</div>
        </div>
        """, unsafe_allow_html=True)

    with ev4:
        st.markdown(f"""
        <div class="kpi-card green-accent">
            <div class="kpi-title">Test R² Score</div>
            <div class="kpi-value" style="color: {palette['emerald_green']};">{rf_m['r2']:.4f}</div>
            <div class="kpi-subtitle">Baseline R²: {b_m['r2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c_eval_l, c_eval_r = st.columns(2)

    with c_eval_l:
        fig_scatter = px.scatter(
            test_eval_df,
            x="actual_days",
            y="predicted_days",
            color="residual_days",
            color_continuous_scale="RdBu",
            labels={"actual_days": "Actual Duration (Days)", "predicted_days": "Predicted Duration (Days)", "residual_days": "Signed Error (d)"},
            hover_data=["Distance_miles", "Weight_kg", "Carrier"]
        )
        max_val = max(test_eval_df["actual_days"].max(), test_eval_df["predicted_days"].max()) + 1
        line_col = palette["text_muted"]
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            name="Perfect Parity (y=x)",
            line=dict(color=line_col, dash="dash")
        ))
        fig_scatter.update_layout(height=370, margin=dict(l=10, r=10, t=35, b=10))
        apply_smarty_theme(fig_scatter, dark_mode, title="Actual vs. Predicted Duration (360 Held-out Test Samples)")
        st.plotly_chart(fig_scatter, width="stretch")

    with c_eval_r:
        fig_res = px.histogram(
            test_eval_df,
            x="residual_days",
            nbins=30,
            marginal="box",
            color_discrete_sequence=[palette["accent_cyan"]],
            labels={"residual_days": "Prediction Error in Days (Predicted - Actual: - Underprediction, + Overprediction)"}
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color=palette["crimson_red"])
        fig_res.update_layout(height=370, margin=dict(l=10, r=10, t=35, b=10))
        apply_smarty_theme(fig_res, dark_mode, title="Signed Prediction Error Distribution (Mean Centered Around Zero)")
        st.plotly_chart(fig_res, width="stretch")

    st.markdown("""
    <div class="notice-box">
        <strong>Evaluation Findings:</strong>
        The Random Forest pipeline achieves an MAE of 48.38 hours versus the baseline median of 50.73 hours (a modest error reduction of 2.35 hours).
        Without GPS telematics or weather telemetry in the raw dataset, residual variance reflects highway congestion and terminal queuing.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 4: DATA QUALITY & AUDIT
# ==============================================================================
elif active_view == "Data Quality & Audit":
    st.markdown("### Logistics Dataset Quality & Preprocessing Audit")
    st.markdown("""
    Systematic audit of `data/logistics_shipments_dataset.csv` verifying data hygiene,
    identifying inverted dates, and preventing target leakage.
    """)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(f"""
        <div class="kpi-card blue-accent">
            <div class="kpi-title">Raw Records</div>
            <div class="kpi-value">2,000</div>
            <div class="kpi-subtitle">11 Source Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown(f"""
        <div class="kpi-card green-accent">
            <div class="kpi-title">Validated Training Records</div>
            <div class="kpi-value">1,800</div>
            <div class="kpi-subtitle">90.0% Retained for Modeling</div>
        </div>
        """, unsafe_allow_html=True)
    with q3:
        st.markdown(f"""
        <div class="kpi-card amber-accent">
            <div class="kpi-title">Missing Delivery Dates</div>
            <div class="kpi-value">32</div>
            <div class="kpi-subtitle">1.6% Excluded (No Ground Truth)</div>
        </div>
        """, unsafe_allow_html=True)
    with q4:
        st.markdown(f"""
        <div class="kpi-card red-accent">
            <div class="kpi-title">Inverted Dates Filtered</div>
            <div class="kpi-value">16</div>
            <div class="kpi-subtitle">0.8% Corrupt (Delivery < Dispatch)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Quality Audit Decisions")
    decisions = [
        {"Condition": "Missing Delivery Dates", "Impacted Records": "32 rows (1.6%)", "Handling Decision": "Excluded from training", "Engineering Rationale": "Ground truth arrival timestamp is required to compute actual transit time."},
        {"Condition": "Chronological Inversions", "Impacted Records": "16 rows (0.8%)", "Handling Decision": "Excluded from training", "Engineering Rationale": "Delivery date precedes shipment date (-30 days); corrupt data entry."},
        {"Condition": "Lost Freight", "Impacted Records": "45 rows (2.25%)", "Handling Decision": "Excluded from training", "Engineering Rationale": "Aborted shipments do not represent completed transit duration."},
        {"Condition": "Zero Cargo Weight", "Impacted Records": "1 row (0.05%)", "Handling Decision": "Excluded from training", "Engineering Rationale": "0.0 kg is physically invalid for commercial freight."},
        {"Condition": "Weight Outliers", "Impacted Records": "Max 5,404 kg", "Handling Decision": "Capped at 99.5th percentile", "Engineering Rationale": "Reduces leverage on decision tree splits."},
        {"Condition": "Post-Hoc Transit_Days Column", "Impacted Records": "2,000 rows", "Handling Decision": "Strictly excluded from features", "Engineering Rationale": "Identical to target duration for delivered cargo; would cause 100% target leakage."}
    ]
    st.table(pd.DataFrame(decisions))

    col_st1, col_st2 = st.columns(2)
    with col_st1:
        status_counts = df_processed["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Records"]
        fig_status = px.pie(
            status_counts,
            names="Status",
            values="Records",
            color="Status",
            color_discrete_map={
                "Delivered": "#10b981",
                "Delayed": "#f59e0b",
                "In Transit": "#0284c7",
                "Lost": "#ef4444",
                "Returned": "#8b5cf6"
            },
            hole=0.4
        )
        fig_status.update_layout(height=340, margin=dict(l=10, r=10, t=35, b=10))
        apply_smarty_theme(fig_status, dark_mode, title="Raw Shipment Status Distribution (2,000 Records)")
        st.plotly_chart(fig_status, width="stretch")

    with col_st2:
        fig_hist = px.histogram(
            df_processed[df_processed["is_valid_training_sample"] == True],
            x="delivery_duration_days",
            nbins=20,
            color_discrete_sequence=[palette["primary_blue"]],
            labels={"delivery_duration_days": "Duration (Days)"}
        )
        fig_hist.update_layout(height=340, margin=dict(l=10, r=10, t=35, b=10))
        apply_smarty_theme(fig_hist, dark_mode, title="Validated Transit Duration Distribution (1,800 Clean Records)")
        st.plotly_chart(fig_hist, width="stretch")

# ==============================================================================
# SECTION 5: ROUTE NETWORK & RECORDS
# ==============================================================================
elif active_view == "Route Network & Records":
    st.markdown("### Corridor Network & Historical Records")
    st.markdown("Filter and inspect verified route volume flows and searchable shipment records.")

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

    # Corridor Flow Sankey Diagram
    route_agg = filtered_df.groupby(["Origin_Warehouse", "Destination"]).agg(
        Shipment_Count=("Shipment_ID", "count"),
        Avg_Distance=("Distance_miles", "mean"),
        Avg_Duration=("delivery_duration_days", "mean")
    ).reset_index().sort_values(by="Shipment_Count", ascending=False).head(25)

    if not route_agg.empty:
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=18,
                line=dict(color=palette["border"], width=0.5),
                label=list(route_agg["Origin_Warehouse"].unique()) + list(route_agg["Destination"].unique()),
                color=palette["accent_cyan"]
            ),
            link=dict(
                source=[list(route_agg["Origin_Warehouse"].unique()).index(src) for src in route_agg["Origin_Warehouse"]],
                target=[len(route_agg["Origin_Warehouse"].unique()) + list(route_agg["Destination"].unique()).index(dst) for dst in route_agg["Destination"]],
                value=route_agg["Shipment_Count"],
                color="rgba(6, 182, 212, 0.35)" if dark_mode else "rgba(8, 145, 178, 0.3)"
            )
        )])
        fig_sankey.update_layout(font_size=11, height=340, margin=dict(l=10, r=10, t=35, b=10))
        apply_smarty_theme(fig_sankey, dark_mode, title="Top Freight Corridors (Origin Warehouse to Destination Hub)")
        st.plotly_chart(fig_sankey, width="stretch")
    else:
        st.info("No shipments match the selected filter combination.")

    # Filtered Table
    display_cols = [
        "Shipment_ID", "Origin_Warehouse", "Destination", "Carrier",
        "Shipment_Date", "Delivery_Date", "Weight_kg", "Distance_miles",
        "Status", "delivery_duration_days"
    ]
    st.dataframe(
        filtered_df[display_cols].rename(columns={"delivery_duration_days": "Actual Duration (Days)"}),
        width="stretch",
        hide_index=True
    )
