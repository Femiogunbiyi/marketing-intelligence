from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_marketing_data
from src.cleaning import clean_marketing_data
from src.metrics import add_core_marketing_metrics, summarize_overall_kpis
from src.campaign_analysis import analyze_campaign_performance
from src.customer_analysis import compute_rfm_and_ltv, summarize_customer_retention
from src.segmentation import segment_customers_kmeans, summarize_segments
from src.alerts import check_marketing_alerts
from src.reporting import export_marketing_report


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Create a realistic messy marketing DataFrame for testing."""
    return pd.DataFrame(
        {
            "Customer ID": ["C001", "C001", "C002", "C003", "C004", "C005"],
            "Date": ["2026-10-01", "2026-10-15", "2026-10-05", "2026-09-20", "2026-10-18", "2026-10-20"],
            "Campaign": ["Meta  Retargeting", "Meta  Retargeting", "Google Search", "Google Search", "Email VIP", "TikTok Awareness"],
            "Impressions": [10000, 12000, 8000, 6000, 5000, 2000],
            "Clicks": [500, 600, 400, 120, 250, 20],
            "Conversions": [50, 60, 40, 0, 100, 5],
            "Spend ($)": ["$500.00", "$600.00", "$400.00", "$300.00", "$200.00", "$500.00"],
            "Revenue ($)": ["$2,500.00", "$3,000.00", "$1,200.00", "$0.00", "$4,000.00", "$100.00"],
        }
    )


def test_data_loader_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_marketing_data("non_existent_folder/missing.csv")


def test_cleaning_pipeline(sample_raw_df: pd.DataFrame) -> None:
    clean_df = clean_marketing_data(sample_raw_df)
    assert "spend" in clean_df.columns
    assert "revenue" in clean_df.columns
    assert clean_df["spend"].dtype in ("float64", "int64")
    assert clean_df["campaign"].iloc[0] == "Meta Retargeting"


def test_metrics_and_zero_division(sample_raw_df: pd.DataFrame) -> None:
    clean_df = clean_marketing_data(sample_raw_df)
    enriched = add_core_marketing_metrics(clean_df)

    # Row index 3 has 0 conversions -> CPA should safely be 0.0, not inf/NaN
    assert enriched.loc[3, "cpa"] == 0.0
    assert enriched.loc[0, "roas"] == pytest.approx(5.0)  # 2500 / 500 = 5.0 ROAS

    kpis = summarize_overall_kpis(enriched)
    assert kpis["total_spend"] == 2500.0
    assert kpis["total_revenue"] == 10800.0


def test_campaign_analysis(sample_raw_df: pd.DataFrame) -> None:
    clean_df = clean_marketing_data(sample_raw_df)
    camp_df = analyze_campaign_performance(clean_df, group_col="campaign", sort_by="spend")
    assert len(camp_df) == 4
    assert camp_df.iloc[0]["campaign"] == "Meta Retargeting"


def test_customer_rfm_and_ml_segmentation(sample_raw_df: pd.DataFrame, tmp_path: Path) -> None:
    clean_df = clean_marketing_data(sample_raw_df)
    customer_df = compute_rfm_and_ltv(clean_df)
    retention = summarize_customer_retention(customer_df)

    assert retention["total_customers"] == 5.0
    assert retention["repeat_customers"] == 1.0

    model_file = tmp_path / "test_kmeans.joblib"
    segmented = segment_customers_kmeans(customer_df, n_clusters=3, model_save_path=model_file)
    summary = summarize_segments(segmented)

    assert model_file.is_file()
    assert len(summary) == 3


def test_alerts_and_reporting(sample_raw_df: pd.DataFrame, tmp_path: Path) -> None:
    clean_df = clean_marketing_data(sample_raw_df)
    enriched = add_core_marketing_metrics(clean_df)
    kpis = summarize_overall_kpis(enriched)
    camp_df = analyze_campaign_performance(clean_df)

    alerts = check_marketing_alerts(kpis, campaign_df=camp_df, min_roas=1.5, min_ctr=1.0)
    assert any(a.scope == "TikTok Awareness" for a in alerts)

    saved = export_marketing_report(kpis, alerts, campaign_df=camp_df, output_dir=tmp_path)
    assert saved["json_summary"].is_file()
    assert saved["markdown_report"].is_file()
