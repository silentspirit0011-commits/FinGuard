"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: Streamlit Institutional Wealth Management Dashboard
Frontend platform designed for compliance surveillance, risk analysis, and transaction auditing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

from database import (
    get_connection,
    get_complete_monitoring_records,
    query_dataframe,
    insert_anomaly_results,
    DB_PATH
)
from analytics import QUERIES, execute_analytics_query, execute_custom_sql
from anomaly_detector import AnomalyDetectionEngine, MODEL_PATH, MODEL_VERSION
from preprocessing import preprocess_pipeline
from pipeline_setup import run_full_pipeline

# ---------------------------------------------------------
# Page Configuration & Global Theme Settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="FinGuard | Financial Transaction Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS styling (Light Theme)
CUSTOM_CSS = """
<style>
    /* Clean Light Institutional Theme */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Metric / KPI Card Styling */
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-subtitle {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 6px;
    }
    .kpi-badge-danger {
        display: inline-block;
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .kpi-badge-success {
        display: inline-block;
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .kpi-badge-info {
        display: inline-block;
        background-color: #F0F9FF;
        color: #0284C7;
        border: 1px solid #BAE6FD;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    
    /* Institutional Alert & Cards */
    .institution-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        color: #1E293B;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        background-color: #FFFFFF;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0284C7;
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0369A1;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Caching & Model Loader
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def load_surveillance_data():
    """
    Loads unified transactions, customer metadata, and anomaly scoring from SQLite.
    """
    if not os.path.exists(DB_PATH):
        # Auto-initialize pipeline if database file doesn't exist yet
        run_full_pipeline()
    df = get_complete_monitoring_records(DB_PATH)
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df

@st.cache_resource
def load_model_engine():
    """
    Loads trained Isolation Forest instance from disk.
    """
    if os.path.exists(MODEL_PATH):
        return AnomalyDetectionEngine.load(MODEL_PATH)
    return None

# Load primary datasets
df_data = load_surveillance_data()
model_engine = load_model_engine()

# ---------------------------------------------------------
# Sidebar Navigation & System Telemetry
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 15px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 2rem;">🛡️</span>
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">FinGuard</div>
                <div style="font-size: 0.72rem; color: #0284C7; font-weight: 600; letter-spacing: 0.05em;">WEALTH SURVEILLANCE SUITE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION MODULES",
        [
            "Executive Overview",
            "Transaction Monitoring",
            "Anomaly Detection",
            "Customer Analytics",
            "Model Performance",
            "SQL Insights & Reports"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #475569; text-transform: uppercase;'>Surveillance Scope</div>", unsafe_allow_html=True)
    
    # Global Date Range Selector
    min_date = df_data["transaction_date"].min().date()
    max_date = df_data["transaction_date"].max().date()
    selected_date_range = st.date_input(
        "Audit Date Horizon",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Customer Segment Filter
    all_segments = ["All Segments"] + sorted(list(df_data["customer_segment"].dropna().unique()))
    selected_segment = st.selectbox("Customer Segment", all_segments)
    
    # Global Risk Filter
    selected_risk_filter = st.selectbox(
        "Risk Stratification Filter",
        ["All Risk Tiers", "High Risk Only", "Medium & High Risk", "Low Risk Only"]
    )

    st.markdown("---")
    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #475569; text-transform: uppercase;'>System Health & Governance</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="font-size: 0.75rem; color: #475569; line-height: 1.6; margin-top: 5px;">
        <div><b>Database:</b> <span style="color:#059669;">SQLite Connected (FK Enabled)</span></div>
        <div><b>Engine:</b> <span style="color:#0284C7;">Isolation Forest {MODEL_VERSION}</span></div>
        <div><b>Total Ingested:</b> {len(df_data):,} Records</div>
        <div><b>Active Accounts:</b> {df_data['customer_id'].nunique()} Portfolios</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Retrain / Re-index Pipeline"):
        with st.spinner("Retraining Isolation Forest and re-indexing SQLite database..."):
            run_full_pipeline(force_regenerate=True)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Surveillance pipeline successfully updated!")
            st.rerun()

# ---------------------------------------------------------
# Apply Global Filters to Working Data
# ---------------------------------------------------------
filtered_df = df_data.copy()

if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
    start_d, end_d = selected_date_range
    filtered_df = filtered_df[
        (filtered_df["transaction_date"].dt.date >= start_d) &
        (filtered_df["transaction_date"].dt.date <= end_d)
    ]

if selected_segment != "All Segments":
    filtered_df = filtered_df[filtered_df["customer_segment"] == selected_segment]

if selected_risk_filter == "High Risk Only":
    filtered_df = filtered_df[filtered_df["risk_category"] == "High Risk"]
elif selected_risk_filter == "Medium & High Risk":
    filtered_df = filtered_df[filtered_df["risk_category"].isin(["High Risk", "Medium Risk"])]
elif selected_risk_filter == "Low Risk Only":
    filtered_df = filtered_df[filtered_df["risk_category"] == "Low Risk"]


# Plotly theme helper (Light Institutional)
def get_plotly_layout(**kwargs):
    base = dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1E293B", family="Segoe UI, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    )
    base.update(kwargs)
    return base

PLOTLY_LAYOUT = get_plotly_layout()

# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":
    st.markdown("## Executive Overview & Risk Horizon")
    st.markdown("<p style='color: #64748B; margin-top: -10px;'>High-level surveillance metrics, liquidity velocity, and anomalous capital exposure.</p>", unsafe_allow_html=True)

    # Core Metrics Calculations
    total_txns = len(filtered_df)
    total_val = filtered_df["transaction_amount"].sum()
    flagged_txns = (filtered_df["anomaly_flag"] == 1).sum()
    anomaly_rate = (flagged_txns / total_txns * 100) if total_txns > 0 else 0.0
    total_customers = filtered_df["customer_id"].nunique()
    flagged_volume = filtered_df[filtered_df["anomaly_flag"] == 1]["transaction_amount"].sum()

    # Row 1: KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-value">{total_txns:,}</div>
            <div class="kpi-subtitle"><span class="kpi-badge-info">Active Scope</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Total Transaction Value</div>
            <div class="kpi-value">${total_val / 1e6:,.2f}M</div>
            <div class="kpi-subtitle">Mean ticket: ${filtered_df['transaction_amount'].mean():,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Flagged Incidents</div>
            <div class="kpi-value">{flagged_txns:,}</div>
            <div class="kpi-subtitle"><span class="kpi-badge-danger">Surveillance Alerts</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Anomaly Rate</div>
            <div class="kpi-value">{anomaly_rate:.2f}%</div>
            <div class="kpi-subtitle">Target baseline: ~5.00%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">At-Risk Capital Exposure</div>
            <div class="kpi-value">${flagged_volume / 1e6:,.2f}M</div>
            <div class="kpi-subtitle">{(flagged_volume / total_val * 100 if total_val else 0):.1f}% of total transacted flow</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Charts (Time Series & Anomaly Ratio)
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.markdown("#### Transaction Volume & Value Over Time")
        daily_df = filtered_df.copy()
        daily_df["day"] = daily_df["transaction_date"].dt.date
        daily_agg = daily_df.groupby("day").agg(
            txn_count=("transaction_id", "count"),
            txn_volume=("transaction_amount", "sum"),
            anomalies=("anomaly_flag", "sum")
        ).reset_index()
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=daily_agg["day"],
            y=daily_agg["txn_count"],
            name="Daily Transactions",
            marker_color="#0284C7",
            opacity=0.7
        ))
        fig_time.add_trace(go.Scatter(
            x=daily_agg["day"],
            y=daily_agg["anomalies"],
            name="Flagged Anomalies",
            mode="lines+markers",
            marker=dict(color="#EF4444", size=6),
            line=dict(color="#EF4444", width=2),
            yaxis="y2"
        ))
        fig_time.update_layout(
            get_plotly_layout(
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Transaction Count", gridcolor="#F1F5F9"),
                yaxis2=dict(title="Flagged Count", overlaying="y", side="right", showgrid=False, color="#EF4444")
            )
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with chart_col2:
        st.markdown("#### Normal vs Flagged Ratio")
        counts = filtered_df["risk_category"].value_counts().reset_index()
        counts.columns = ["Risk Category", "Count"]
        color_map = {
            "Low Risk": "#10B981",
            "Medium Risk": "#F59E0B",
            "High Risk": "#EF4444"
        }
        fig_donut = px.pie(
            counts,
            names="Risk Category",
            values="Count",
            hole=0.55,
            color="Risk Category",
            color_discrete_map=color_map
        )
        fig_donut.update_layout(get_plotly_layout(height=340, showlegend=True))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Row 3: Category Breakdown & Risk Heatmap
    c3_col1, c3_col2 = st.columns(2)
    
    with c3_col1:
        st.markdown("#### Transaction Distribution by Merchant Category")
        cat_agg = filtered_df.groupby("merchant_category").agg(
            total_amount=("transaction_amount", "sum"),
            anomalies=("anomaly_flag", "sum"),
            count=("transaction_id", "count")
        ).reset_index().sort_values("total_amount", ascending=True)
        
        fig_cat = px.bar(
            cat_agg,
            x="total_amount",
            y="merchant_category",
            orientation="h",
            color="anomalies",
            color_continuous_scale=["#0284C7", "#EF4444"],
            labels={"total_amount": "Total Value ($)", "merchant_category": "Category", "anomalies": "Anomalies"}
        )
        fig_cat.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_cat, use_container_width=True)

    with c3_col2:
        st.markdown("#### Anomaly Rate by Payment Channel")
        pm_agg = filtered_df.groupby("payment_method").agg(
            total_txns=("transaction_id", "count"),
            anomalies=("anomaly_flag", "sum")
        ).reset_index()
        pm_agg["anomaly_rate"] = (pm_agg["anomalies"] / pm_agg["total_txns"] * 100).round(2)
        pm_agg = pm_agg.sort_values("anomaly_rate", ascending=False)
        
        fig_pm = px.bar(
            pm_agg,
            x="payment_method",
            y="anomaly_rate",
            color="anomaly_rate",
            color_continuous_scale="Reds",
            labels={"anomaly_rate": "Anomaly Rate (%)", "payment_method": "Channel"}
        )
        fig_pm.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_pm, use_container_width=True)

# =========================================================
# PAGE 2: TRANSACTION MONITORING
# =========================================================
elif page == "Transaction Monitoring":
    st.markdown("## Transaction Monitoring & Audit Ledger")
    st.markdown("<p style='color: #64748B; margin-top: -10px;'>Search, filter, and inspect granular banking transactions with contextual behavioral metadata.</p>", unsafe_allow_html=True)

    # Filtering Controls
    with st.expander("🔍 Advanced Audit Filters & Search", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            search_txn = st.text_input("Search Transaction ID / Customer ID", placeholder="e.g. TXN-100234 or CUST-1045")
        with f_col2:
            categories = ["All Categories"] + sorted(list(df_data["merchant_category"].dropna().unique()))
            selected_cat = st.selectbox("Merchant Category", categories)
        with f_col3:
            payment_methods = ["All Methods"] + sorted(list(df_data["payment_method"].dropna().unique()))
            selected_pm = st.selectbox("Payment Channel", payment_methods)
        with f_col4:
            sort_option = st.selectbox("Sort Order", [
                "Highest Amount First",
                "Lowest Amount First",
                "Highest Anomaly Score First",
                "Most Recent First"
            ])

    # Apply search and filters
    view_df = filtered_df.copy()
    if search_txn.strip():
        term = search_txn.strip().upper()
        view_df = view_df[
            view_df["transaction_id"].str.upper().str.contains(term) |
            view_df["customer_id"].str.upper().str.contains(term) |
            view_df["customer_name"].str.upper().str.contains(term)
        ]
    if selected_cat != "All Categories":
        view_df = view_df[view_df["merchant_category"] == selected_cat]
    if selected_pm != "All Methods":
        view_df = view_df[view_df["payment_method"] == selected_pm]

    # Sort
    if sort_option == "Highest Amount First":
        view_df = view_df.sort_values("transaction_amount", ascending=False)
    elif sort_option == "Lowest Amount First":
        view_df = view_df.sort_values("transaction_amount", ascending=True)
    elif sort_option == "Highest Anomaly Score First":
        view_df = view_df.sort_values("anomaly_score", ascending=False)
    else:
        view_df = view_df.sort_values("transaction_date", ascending=False)

    st.markdown(f"**Showing {len(view_df):,} matching transaction records** (Audit scope filtered)")

    # Display Table with risk badge simulation
    display_cols = [
        "transaction_id", "transaction_date", "customer_name", "customer_segment",
        "transaction_amount", "transaction_type", "merchant_category",
        "payment_method", "location", "risk_category", "anomaly_score"
    ]
    
    st.dataframe(
        view_df[display_cols].style.format({
            "transaction_amount": "${:,.2f}",
            "anomaly_score": "{:.1f}",
            "transaction_date": lambda t: t.strftime("%Y-%m-%d %H:%M")
        }),
        height=420,
        use_container_width=True
    )

    # Detailed Transaction Inspector
    st.markdown("### 🔎 Individual Transaction Inspector")
    inspect_id = st.selectbox(
        "Select Transaction to Inspect in Depth",
        options=view_df["transaction_id"].head(100).tolist() if len(view_df) > 0 else []
    )

    if inspect_id:
        record = view_df[view_df["transaction_id"] == inspect_id].iloc[0]
        
        ins_col1, ins_col2, ins_col3 = st.columns(3)
        with ins_col1:
            st.markdown(f"""
            <div class="institution-card">
                <div style="font-size: 0.85rem; font-weight: 700; color: #0284C7; margin-bottom: 8px;">TRANSACTION METRICS</div>
                <div><b>ID:</b> {record['transaction_id']}</div>
                <div><b>Timestamp:</b> {record['transaction_date']}</div>
                <div><b>Amount:</b> <span style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">${record['transaction_amount']:,.2f}</span></div>
                <div><b>Type:</b> {record['transaction_type']}</div>
                <div><b>Category:</b> {record['merchant_category']}</div>
                <div><b>Channel:</b> {record['payment_method']}</div>
                <div><b>Origin:</b> {record['location']}</div>
            </div>
            """, unsafe_allow_html=True)

        with ins_col2:
            st.markdown(f"""
            <div class="institution-card">
                <div style="font-size: 0.85rem; font-weight: 700; color: #0284C7; margin-bottom: 8px;">PORTFOLIO PROFILE</div>
                <div><b>Client:</b> {record['customer_name']} ({record['customer_id']})</div>
                <div><b>Segment:</b> {record['customer_segment']}</div>
                <div><b>Account Type:</b> {record['account_type']}</div>
                <div><b>Liquid Balance:</b> ${record['account_balance']:,.2f}</div>
                <div><b>Balance Ratio:</b> {(record['transaction_amount'] / (record['account_balance'] + 1) * 100):.2f}% of balance</div>
            </div>
            """, unsafe_allow_html=True)

        with ins_col3:
            badge_class = "kpi-badge-danger" if record['risk_category'] == "High Risk" else ("kpi-badge-info" if record['risk_category'] == "Medium Risk" else "kpi-badge-success")
            st.markdown(f"""
            <div class="institution-card">
                <div style="font-size: 0.85rem; font-weight: 700; color: #0284C7; margin-bottom: 8px;">SURVEILLANCE ASSESSMENT</div>
                <div><b>Risk Tier:</b> <span class="{badge_class}">{record['risk_category']}</span></div>
                <div><b>Anomaly Risk Score:</b> <span style="font-size: 1.25rem; font-weight: 700; color: #DC2626;">{record['anomaly_score']:.1f} / 100</span></div>
                <div><b>Model Flag:</b> {'⚠️ FLAGGED AS OUTLIER' if record['anomaly_flag'] == 1 else '✅ NORMAL INLIER'}</div>
                <div><b>Engine:</b> {record['model_version']}</div>
                <div><b>Ground Truth Label:</b> {'Simulated Anomaly' if record['is_injected_anomaly'] == 1 else 'Simulated Normal'}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# PAGE 3: ANOMALY DETECTION
# =========================================================
elif page == "Anomaly Detection":
    st.markdown("## Anomaly Detection & Risk Stratification")
    st.markdown("<p style='color: #64748B; margin-top: -10px;'>Statistical isolation metrics, anomaly scoring distribution, and multi-dimensional behavioral divergence.</p>", unsafe_allow_html=True)

    # Risk Metrics
    high_risk_count = (filtered_df["risk_category"] == "High Risk").sum()
    med_risk_count = (filtered_df["risk_category"] == "Medium Risk").sum()
    low_risk_count = (filtered_df["risk_category"] == "Low Risk").sum()
    mean_score = filtered_df["anomaly_score"].mean()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">High Risk Incidents</div>
            <div class="kpi-value" style="color: #DC2626;">{high_risk_count:,}</div>
            <div class="kpi-subtitle">Priority Level 1 Review</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Medium Risk Alerts</div>
            <div class="kpi-value" style="color: #D97706;">{med_risk_count:,}</div>
            <div class="kpi-subtitle">Secondary Queue</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Low Risk (Compliant)</div>
            <div class="kpi-value" style="color: #059669;">{low_risk_count:,}</div>
            <div class="kpi-subtitle">Normal operational activity</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Mean Anomaly Score</div>
            <div class="kpi-value">{mean_score:.1f}</div>
            <div class="kpi-subtitle">Range: 0.0 - 100.0 index</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts: Score Distribution & Behavioral Divergence
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        st.markdown("#### Anomaly Risk Score Distribution")
        fig_hist = px.histogram(
            filtered_df,
            x="anomaly_score",
            nbins=40,
            color="risk_category",
            color_discrete_map={"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"},
            labels={"anomaly_score": "Calibrated Anomaly Score (0 - 100)", "count": "Transaction Count"}
        )
        fig_hist.add_vline(x=50.0, line_dash="dash", line_color="#D97706", annotation_text="Medium (50.0)")
        fig_hist.add_vline(x=75.0, line_dash="dash", line_color="#DC2626", annotation_text="High (75.0)")
        fig_hist.update_layout(get_plotly_layout(height=360, legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_col2:
        st.markdown("#### Transaction Amount vs Account Balance")
        fig_scatter = px.scatter(
            filtered_df,
            x="account_balance",
            y="transaction_amount",
            color="risk_category",
            size="anomaly_score",
            hover_data=["transaction_id", "customer_name", "merchant_category"],
            color_discrete_map={"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"},
            labels={"account_balance": "Account Balance ($)", "transaction_amount": "Transaction Amount ($)"}
        )
        fig_scatter.update_layout(get_plotly_layout(height=360, legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Flagged High-Risk Transactions Table
    st.markdown("### 🚨 Prioritized High-Risk Alert Queue")
    high_risk_df = filtered_df[filtered_df["risk_category"] == "High Risk"].sort_values("anomaly_score", ascending=False)
    
    st.dataframe(
        high_risk_df[[
            "transaction_id", "transaction_date", "customer_name", "customer_segment",
            "transaction_amount", "account_balance", "merchant_category",
            "payment_method", "location", "anomaly_score"
        ]].style.format({
            "transaction_amount": "${:,.2f}",
            "account_balance": "${:,.2f}",
            "anomaly_score": "{:.1f}",
            "transaction_date": lambda t: t.strftime("%Y-%m-%d %H:%M")
        }),
        height=300,
        use_container_width=True
    )

# =========================================================
# PAGE 4: CUSTOMER ANALYTICS
# =========================================================
elif page == "Customer Analytics":
    st.markdown("## Customer & Wealth Portfolio Analytics")
    st.markdown("<p style='color: #64748B; margin-top: -10px;'>Customer segmentation profiling, behavioral spend distributions, and risk concentration.</p>", unsafe_allow_html=True)

    # Aggregate customer level data
    cust_agg = filtered_df.groupby(["customer_id", "customer_name", "customer_segment", "account_type"]).agg(
        txn_count=("transaction_id", "count"),
        total_spent=("transaction_amount", "sum"),
        avg_spent=("transaction_amount", "mean"),
        max_spent=("transaction_amount", "max"),
        anomaly_count=("anomaly_flag", "sum"),
        avg_risk_score=("anomaly_score", "mean")
    ).reset_index()

    # Top Customers Charts
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("#### Top 10 Portfolios by Transaction Count")
        top_vol = cust_agg.sort_values("txn_count", ascending=False).head(10)
        fig_top_vol = px.bar(
            top_vol,
            x="txn_count",
            y="customer_name",
            orientation="h",
            color="anomaly_count",
            color_continuous_scale=["#0284C7", "#EF4444"],
            labels={"txn_count": "Transactions", "customer_name": "Customer", "anomaly_count": "Anomalies"}
        )
        fig_top_vol.update_layout(get_plotly_layout(height=360))
        st.plotly_chart(fig_top_vol, use_container_width=True)

    with col_c2:
        st.markdown("#### Top 10 Portfolios by Total Transacted Value")
        top_val = cust_agg.sort_values("total_spent", ascending=False).head(10)
        fig_top_val = px.bar(
            top_val,
            x="total_spent",
            y="customer_name",
            orientation="h",
            color="avg_risk_score",
            color_continuous_scale="Viridis",
            labels={"total_spent": "Total Value ($)", "customer_name": "Customer", "avg_risk_score": "Risk Index"}
        )
        fig_top_val.update_layout(get_plotly_layout(height=360))
        st.plotly_chart(fig_top_val, use_container_width=True)

    # Segment Comparison
    st.markdown("#### Customer Segment Spend & Risk Exposure")
    seg_summary = cust_agg.groupby("customer_segment").agg(
        client_count=("customer_id", "count"),
        total_volume=("total_spent", "sum"),
        mean_ticket=("avg_spent", "mean"),
        total_anomalies=("anomaly_count", "sum")
    ).reset_index()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_seg_pie = px.pie(
            seg_summary,
            names="customer_segment",
            values="total_volume",
            title="Capital Flow Distribution by Client Segment",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_seg_pie.update_layout(get_plotly_layout(height=320))
        st.plotly_chart(fig_seg_pie, use_container_width=True)

    with col_s2:
        fig_seg_anom = px.bar(
            seg_summary,
            x="customer_segment",
            y="total_anomalies",
            color="total_anomalies",
            color_continuous_scale="Reds",
            title="Flagged Anomalies by Customer Segment",
            labels={"total_anomalies": "Anomaly Count", "customer_segment": "Segment"}
        )
        fig_seg_anom.update_layout(get_plotly_layout(height=320))
        st.plotly_chart(fig_seg_anom, use_container_width=True)

    # Customer Deep-Dive Selection
    st.markdown("### 👤 Portfolio Drill-Down Analysis")
    selected_client_id = st.selectbox(
        "Select Client to Audit Historical Ledger",
        options=sorted(cust_agg["customer_id"].unique())
    )

    if selected_client_id:
        c_meta = cust_agg[cust_agg["customer_id"] == selected_client_id].iloc[0]
        c_txns = filtered_df[filtered_df["customer_id"] == selected_client_id].sort_values("transaction_date", ascending=False)
        
        info_c1, info_c2, info_c3, info_c4 = st.columns(4)
        info_c1.metric("Client Name", c_meta['customer_name'])
        info_c2.metric("Segment", c_meta['customer_segment'])
        info_c3.metric("Total Transacted", f"${c_meta['total_spent']:,.2f}")
        info_c4.metric("Flagged Incidents", f"{int(c_meta['anomaly_count'])} alerts")

        st.dataframe(
            c_txns[[
                "transaction_id", "transaction_date", "transaction_amount",
                "transaction_type", "merchant_category", "payment_method",
                "risk_category", "anomaly_score"
            ]].style.format({
                "transaction_amount": "${:,.2f}",
                "anomaly_score": "{:.1f}",
                "transaction_date": lambda t: t.strftime("%Y-%m-%d %H:%M")
            }),
            height=260,
            use_container_width=True
        )

# =========================================================
# PAGE 5: MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    st.markdown("## Isolation Forest Anomaly Detection Engine")
    st.markdown("<p style='color: #64748B; margin-top: -10px;'>Algorithm architecture, calibration metrics against injected ground-truth, and institutional governance.</p>", unsafe_allow_html=True)

    # Evaluate metrics against ground truth
    y_true = filtered_df["is_injected_anomaly"].values
    y_pred = filtered_df["anomaly_flag"].values
    
    if model_engine and len(filtered_df) > 0:
        metrics = model_engine.evaluate(y_true, y_pred)
    else:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0, "true_positives": 0}

    # Row 1: KPI Cards for Model Performance
    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Accuracy</div>
            <div class="kpi-value">{metrics['accuracy'] * 100:.2f}%</div>
            <div class="kpi-subtitle">Overall classification consistency</div>
        </div>
        """, unsafe_allow_html=True)
    with ev2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Precision</div>
            <div class="kpi-value" style="color: #0284C7;">{metrics['precision'] * 100:.2f}%</div>
            <div class="kpi-subtitle">True anomalies / total flagged</div>
        </div>
        """, unsafe_allow_html=True)
    with ev3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Recall (Sensitivity)</div>
            <div class="kpi-value" style="color: #0284C7;">{metrics['recall'] * 100:.2f}%</div>
            <div class="kpi-subtitle">Detected anomalies / ground truth</div>
        </div>
        """, unsafe_allow_html=True)
    with ev4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">F1-Score</div>
            <div class="kpi-value" style="color: #059669;">{metrics['f1_score']:.4f}</div>
            <div class="kpi-subtitle">Harmonic mean of precision & recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Confusion Matrix & Algorithm Architecture
    cm_col, doc_col = st.columns([1, 1])

    with cm_col:
        st.markdown("#### Evaluation Confusion Matrix")
        cm_matrix = [
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ]
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Observations"),
            x=["Predicted Normal (0)", "Predicted Anomaly (1)"],
            y=["Actual Normal (0)", "Actual Anomaly (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(get_plotly_layout(height=340))
        st.plotly_chart(fig_cm, use_container_width=True)

    with doc_col:
        st.markdown("#### Model Architecture & Parameters")
        st.markdown(f"""
        <div class="institution-card">
            <div><b>Model Family:</b> Scikit-Learn IsolationForest</div>
            <div><b>Active Model Version:</b> <code>{MODEL_VERSION}</code></div>
            <div><b>Contamination Parameter:</b> <code>0.05</code> (5.0% expected baseline)</div>
            <div><b>Ensemble Estimators:</b> <code>150 Isolation Trees</code></div>
            <div><b>Sub-sample Size:</b> <code>256</code> (Optimized tree isolation)</div>
            <div><b>Engineered Feature Count:</b> <code>11 Numerical Features</code></div>
            <hr style="border-color: #E2E8F0; margin: 10px 0;">
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">
                <b>Why Contamination = 0.05?</b><br>
                In Tier-1 wealth management surveillance desks, alert fatigue is a primary operational bottleneck. Setting contamination to 5% strikes an optimal tradeoff between capturing statistically isolated behaviors and ensuring investigator queues remain actionable.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Algorithm Walkthrough & Governance
    st.markdown("---")
    st.markdown("### 📘 Algorithmic Mechanics: How Isolation Forest Operates")
    st.markdown("""
    1. **Recursive Partitioning:** Isolation Forest builds an ensemble of isolation trees (iTrees) by randomly selecting a feature and randomly choosing a split value between that feature's minimum and maximum.
    2. **Short Path Length for Anomalies:** Because anomalies have extreme values, unusual burst frequencies, or atypical spend-to-balance ratios, they require significantly fewer splits to isolate in terminal leaf nodes compared to dense clusters of normal transactions.
    3. **Path Length Score:** Anomaly score $s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}}$, where $E(h(x))$ is the average path length across all isolation trees and $c(n)$ is the average path length of unsuccessful searches in Binary Search Trees.
    4. **Calibrated Risk Index:** In FinGuard, raw decision function scores are normalized to a standardized $0 - 100$ risk scale for intuitive analyst triage.
    """)

    st.markdown("### ⚠️ Limitations of Evaluating on Synthetic Data")
    st.markdown("""
    > [!IMPORTANT]
    > **Educational Prototype Governance Disclosure:**
    > While FinGuard achieves high precision and recall on the reproducible synthetic dataset, real-world banking fraud is inherently **adversarial and non-stationary**. Fraud syndicates intentionally obfuscate transaction amounts, mimic benign customer schedules, and exploit multi-hop intermediary accounts. Real production deployments at institutions like Morgan Stanley combine unsupervised anomaly detection with graph analytics, KYC knowledge graphs, behavioral biometrics, and active learning human feedback loops.
    """)

# =========================================================
# PAGE 6: SQL INSIGHTS & REPORTS
# =========================================================
elif page == "SQL Insights & Reports":
    st.markdown("## SQL Analytics Engine & Standard Reports")
    st.markdown("<p style='color: #64748B; margin-top: -10px;'>Execute pre-compiled relational SQL queries, run ad-hoc queries, and export audit files.</p>", unsafe_allow_html=True)

    # Query Selector
    selected_query_name = st.selectbox(
        "Select Analytical SQL Procedure",
        options=list(QUERIES.keys())
    )

    if selected_query_name:
        query_result_df, raw_sql = execute_analytics_query(selected_query_name)
        
        st.markdown("#### Formatted SQL Query")
        st.code(raw_sql.strip(), language="sql")

        st.markdown(f"#### Query Results ({len(query_result_df)} rows returned)")
        st.dataframe(query_result_df, height=320, use_container_width=True)

        # CSV Download Button
        csv_data = query_result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Query Results as CSV",
            data=csv_data,
            file_name=f"finguard_report_{selected_query_name[:2].strip()}.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("### 💻 Custom SQL Query Console (Read-Only)")
    st.markdown("<p style='color: #64748B; font-size: 0.85rem;'>Execute ad-hoc <code>SELECT</code> statements directly against the SQLite database schema.</p>", unsafe_allow_html=True)

    default_custom = """SELECT 
    t.merchant_category,
    t.payment_method,
    COUNT(*) as total_txns,
    ROUND(SUM(t.transaction_amount), 2) as total_amount,
    ROUND(AVG(t.transaction_amount), 2) as avg_amount
FROM transactions t
GROUP BY t.merchant_category, t.payment_method
ORDER BY total_amount DESC
LIMIT 10;"""

    user_sql = st.text_area("Enter SQL Statement", value=default_custom, height=140)
    
    if st.button("▶️ Execute Custom SQL"):
        try:
            custom_res = execute_custom_sql(user_sql)
            st.success(f"Executed successfully! {len(custom_res)} rows returned.")
            st.dataframe(custom_res, height=300, use_container_width=True)
            
            custom_csv = custom_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Custom Result to CSV",
                data=custom_csv,
                file_name="finguard_custom_query_export.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"SQL Execution Error: {str(e)}")

    st.markdown("---")
    st.markdown("### 📦 Bulk Audit Export Center")
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("##### Export All Active Filtered Transactions")
        st.markdown(f"Current filtered scope: **{len(filtered_df):,} records**")
        full_csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Filtered Dataset (CSV)",
            data=full_csv,
            file_name="finguard_filtered_transactions.csv",
            mime="text/csv"
        )
        
    with exp_col2:
        st.markdown("##### Export Flagged High & Medium Risk Incidents")
        anomalies_only = filtered_df[filtered_df["anomaly_flag"] == 1]
        st.markdown(f"Flagged incidents: **{len(anomalies_only):,} records**")
        anom_csv = anomalies_only.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Anomalies Incident Log (CSV)",
            data=anom_csv,
            file_name="finguard_flagged_anomalies.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# Footer / Legal Disclaimer
# ---------------------------------------------------------
st.markdown("<br><hr style='border-color: #E2E8F0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 0.75rem; color: #64748B; padding-bottom: 20px;">
    <b>FinGuard Surveillance Suite</b> | Morgan Stanley Wealth Management Apprenticeship Technical Portfolio<br>
    Educational Analytical Prototype. Built with Python 3, Streamlit, Scikit-Learn Isolation Forest, SQLite & Plotly.
</div>
""", unsafe_allow_html=True)

