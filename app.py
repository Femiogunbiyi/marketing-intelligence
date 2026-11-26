from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_loader import load_marketing_data
from src.cleaning import clean_marketing_data
from src.metrics import add_core_marketing_metrics, summarize_overall_kpis
from src.campaign_analysis import analyze_campaign_performance
from src.customer_analysis import compute_rfm_and_ltv, summarize_customer_retention
from src.segmentation import segment_customers_kmeans, summarize_segments
from src.alerts import check_marketing_alerts
from src.reporting import export_marketing_report


st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Marketing Intelligence & ML Dashboard")
st.caption("Automated KPI tracking, campaign diagnostics, RFM customer segmentation, and real-time alerts.")

# --- Sidebar: Data Source, Currency & Alert Controls ---
st.sidebar.header("⚙️ Pipeline Settings")
uploaded_file = st.sidebar.file_uploader("Upload Raw Marketing CSV", type=["csv"])
default_csv_path = Path("data/raw/sample_lagos_ecommerce.csv")
currency = st.sidebar.selectbox("Display Currency", options=["₦", "$", "£", "€"], index=0)

st.sidebar.subheader("🚨 Alert Thresholds")
min_roas = st.sidebar.slider("Minimum Target ROAS (x)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
min_ctr = st.sidebar.slider("Minimum Target CTR (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
n_clusters = st.sidebar.slider("ML Customer Segments (K)", min_value=2, max_value=6, value=4, step=1)


@st.cache_data
def run_cleaning_and_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data and compute row-level KPIs (cached for fast UI interactions)."""
    cleaned = clean_marketing_data(raw_df)
    return add_core_marketing_metrics(cleaned)


# --- Load Data (Uploaded CSV or Default Local File) ---
raw_df: pd.DataFrame | None = None
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
elif default_csv_path.is_file():
    raw_df = load_marketing_data(default_csv_path)

if raw_df is None or raw_df.empty:
    st.info(
        "👋 Upload a raw marketing CSV in the sidebar—or place one at "
        "`data/raw/marketing_data.csv`—to launch your automated dashboard."
    )
    st.stop()

enriched_df = run_cleaning_and_metrics(raw_df)
overall_kpis = summarize_overall_kpis(enriched_df)

# --- Optional Channel Filters ---
if "channel" in enriched_df.columns:
    channels = sorted(enriched_df["channel"].dropna().unique().tolist())
    selected_channels = st.sidebar.multiselect("Filter by Channel", options=channels, default=channels)
    if selected_channels:
        enriched_df = enriched_df[enriched_df["channel"].isin(selected_channels)]
        overall_kpis = summarize_overall_kpis(enriched_df)

# --- Section 1: Top-Level KPI Cards ---
st.subheader("1. Overall Account Performance")
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Spend", f"{currency}{overall_kpis.get('total_spend', 0):,.2f}")
kpi_cols[1].metric("Total Revenue", f"{currency}{overall_kpis.get('total_revenue', 0):,.2f}")
kpi_cols[2].metric("Net Profit", f"{currency}{overall_kpis.get('overall_net_profit', 0):,.2f}")
kpi_cols[3].metric("Overall ROAS", f"{overall_kpis.get('overall_roas', 0):.2f}x")
kpi_cols[4].metric("Overall CTR", f"{overall_kpis.get('overall_ctr', 0):.2f}%")


# --- Section 2: Campaign Performance & Charts ---
campaign_df: pd.DataFrame | None = None
group_candidates = [c for c in ("campaign", "campaign_name", "channel") if c in enriched_df.columns]

if group_candidates:
    st.divider()
    st.subheader("2. Campaign & Channel Breakdown")
    group_col = st.selectbox("Group Performance By", options=group_candidates, index=0)
    campaign_df = analyze_campaign_performance(enriched_df, group_col=group_col, sort_by="spend")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown(f"**Spend vs. Revenue ({currency})**")
        plot_cols = [c for c in ("spend", "revenue") if c in campaign_df.columns]
        if plot_cols:
            st.bar_chart(campaign_df, x=group_col, y=plot_cols)  # type: ignore

    with chart_col2:
        if "roas" in campaign_df.columns:
            st.markdown("**Return on Ad Spend (ROAS)**")
            st.bar_chart(campaign_df, x=group_col, y="roas")  # type: ignore

    st.dataframe(campaign_df, width="stretch")  # type: ignore

# --- Section 3: Customer RFM, LTV & ML Segmentation ---
segment_summary: pd.DataFrame | None = None
if {"customer_id", "revenue"}.issubset(enriched_df.columns):
    st.divider()
    st.subheader("3. Customer Retention & ML Segmentation (K-Means)")

    customer_df = compute_rfm_and_ltv(enriched_df)
    retention = summarize_customer_retention(customer_df)

    ret_cols = st.columns(4)
    ret_cols[0].metric("Total Customers", f"{int(retention['total_customers']):,}")
    ret_cols[1].metric("Repeat Purchase Rate", f"{retention['repeat_purchase_rate']:.1f}%")
    ret_cols[2].metric("Average LTV", f"{currency}{retention['average_ltv']:,.2f}")
    ret_cols[3].metric("Avg Order Frequency", f"{retention['average_purchase_frequency']:.2f}")

    if len(customer_df) >= n_clusters:
        segmented_df = segment_customers_kmeans(customer_df, n_clusters=n_clusters)
        segment_summary = summarize_segments(segmented_df)

        seg_col1, seg_col2 = st.columns([1, 1.5])
        with seg_col1:
            st.markdown(f"**Revenue Share by ML Segment ({currency})**")
            st.bar_chart(segment_summary, x="segment_label", y="total_segment_revenue")  # type: ignore
        with seg_col2:
            st.markdown("**Segment Profile Summary**")
            st.dataframe(segment_summary, width="stretch")  # type: ignore

# --- Section 4: Live Alerts & Automated Report Export ---
st.divider()
st.subheader("4. Automated Performance Alerts & Export")

alerts = check_marketing_alerts(
    overall_kpis=overall_kpis,
    campaign_df=campaign_df,
    min_roas=min_roas,
    min_ctr=min_ctr,
)

if not alerts:
    st.success("✅ All marketing KPIs are within your target thresholds.")
else:
    for alert in alerts:
        if alert.severity == "CRITICAL":
            st.error(f"🚨 **[{alert.scope}]** {alert.message}")
        else:
            st.warning(f"⚠️ **[{alert.scope}]** {alert.message}")

if st.button("📥 Export Snapshot to reports/ Folder"):
    saved = export_marketing_report(
        overall_kpis=overall_kpis,
        alerts=alerts,
        campaign_df=campaign_df,
        segment_summary=segment_summary,
    )
    st.success("Reports saved successfully:")
    for label, path in saved.items():
        st.write(f"- **{label}**: `{path}`")
