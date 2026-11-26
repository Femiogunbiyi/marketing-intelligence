import pandas as pd
from src.metrics import add_core_marketing_metrics

def analyze_campaign_performance(
    df: pd.DataFrame,
    group_col: str = "campaign",
    sort_by: str = "spend",
) -> pd.DataFrame:
    """Aggregate performance totals and compute weighted KPIs per campaign or channel.

    Args:
        df: Cleaned marketing DataFrame(from `cleaning.py`).
        group_col: Column to group by (e.g., 'campaign', 'campaign_name', or 'channel').
        sort_by: Metric column used to rank results from highest to lowest.

    Returns:
        A summary DataFrame ranked by `sort_by` with aggregated totals and KPIs.

    Raises:
        ValueError: If `group_col` is not found in the DataFrame.
    """ 

    if group_col not in df.columns:
        raise ValueError(f"Grouping column '{group_col}' not found in DataFrame.")

    metric_candidates = ["impressions", "clicks", "conversions", "spend", "revenue"]
    available_metrics = [col for col in metric_candidates if col in df.columns]

    if not available_metrics:
        raise ValueError("No numeric marketing metric columns found to aggregate.")

    # Sum raw totals per campaign first so ratios (CTR, ROAS, CPA) are properly weighted
    grouped = (
        df.groupby(group_col, as_index=False)[available_metrics]
        .sum()
    )

    # Reuse metrics.py to compute CTR, CPC, CPM, Conversion Rate, CPA, ROAS, ROI, Net Profit, AOV
    summary = add_core_marketing_metrics(grouped)

    if sort_by in summary.columns:
        summary = summary.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    return summary

def get_top_and_bottom_campaigns(
    campaign_summary: pd.DataFrame,
    metric: str = "roas",
    n: int = 5,
) -> dict[str, pd.DataFrame]:
    """Identify the top N best-performing and bottom N underperforming campaigns."""
    if metric not in campaign_summary.columns:
        raise ValueError(f"Metric '{metric}' not found in campaign summary.")

    ranked = campaign_summary.sort_values(by=metric, ascending=False)
    return{
        "top": ranked.head(n).reset_index(drop=True),
        "bottom": ranked.tail(n).reset_index(drop=True),
    }