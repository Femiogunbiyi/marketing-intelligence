from dataclasses import dataclass
import pandas as pd


@dataclass
class MarketingAlert:
    """Structured representation of a triggered marketing performance alert."""
    severity: str       # "CRITICAL" or "WARNING"
    scope: str          # "OVERALL" or specific campaign name
    metric: str         # e.g., "roas", "cpa", "ctr", "net_profit"
    actual_value: float
    threshold: float
    message: str


def check_marketing_alerts(
    overall_kpis: dict[str, float],
    campaign_df: pd.DataFrame | None = None,
    min_roas: float = 1.5,
    max_cpa: float | None = None,
    min_ctr: float = 1.0,
) -> list[MarketingAlert]:
    """Evaluate overall and campaign-level KPIs against target thresholds.

    Args:
        overall_kpis: Summary dictionary returned by `summarize_overall_kpis()`.
        campaign_df: Optional campaign summary DataFrame from `analyze_campaign_performance()`.
        min_roas: Minimum acceptable Return on Ad Spend multiplier (default: 1.5x).
        max_cpa: Optional maximum acceptable Cost Per Acquisition.
        min_ctr: Minimum acceptable Click-Through Rate percentage (default: 1.0%).

    Returns:
        A list of `MarketingAlert` objects for any metric breaching thresholds.
    """
    alerts: list[MarketingAlert] = []

    # 1. Check Overall Account Health
    if "overall_net_profit" in overall_kpis and overall_kpis["overall_net_profit"] < 0:
        alerts.append(
            MarketingAlert(
                severity="CRITICAL",
                scope="OVERALL",
                metric="net_profit",
                actual_value=overall_kpis["overall_net_profit"],
                threshold=0.0,
                message=f"Overall net profit is negative ({overall_kpis['overall_net_profit']:.2f}).",
            )
        )

    if "overall_roas" in overall_kpis and overall_kpis["overall_roas"] < min_roas:
        alerts.append(
            MarketingAlert(
                severity="WARNING",
                scope="OVERALL",
                metric="roas",
                actual_value=overall_kpis["overall_roas"],
                threshold=min_roas,
                message=(
                    f"Overall ROAS ({overall_kpis['overall_roas']:.2f}x) is below "
                    f"target ({min_roas:.2f}x)."
                ),
            )
        )

    if max_cpa is not None and "overall_cpa" in overall_kpis and overall_kpis["overall_cpa"] > max_cpa:
        alerts.append(
            MarketingAlert(
                severity="WARNING",
                scope="OVERALL",
                metric="cpa",
                actual_value=overall_kpis["overall_cpa"],
                threshold=max_cpa,
                message=(
                    f"Overall CPA ({overall_kpis['overall_cpa']:.2f}) exceeds "
                    f"maximum limit ({max_cpa:.2f})."
                ),
            )
        )

    # 2. Check Individual Campaigns (if provided)
    if campaign_df is not None and not campaign_df.empty:
        name_col = campaign_df.columns[0]

        for _, row in campaign_df.iterrows():
            camp_name = str(row[name_col])

            if "roas" in campaign_df.columns and "spend" in campaign_df.columns:
                if row["spend"] > 0 and row["roas"] < min_roas:
                    alerts.append(
                        MarketingAlert(
                            severity="WARNING",
                            scope=camp_name,
                            metric="roas",
                            actual_value=round(float(row["roas"]), 2),
                            threshold=min_roas,
                            message=(
                                f"Campaign '{camp_name}' ROAS is {row['roas']:.2f}x "
                                f"(target >= {min_roas:.2f}x)."
                            ),
                        )
                    )

            if "ctr" in campaign_df.columns and "impressions" in campaign_df.columns:
                if row["impressions"] > 0 and row["ctr"] < min_ctr:
                    alerts.append(
                        MarketingAlert(
                            severity="WARNING",
                            scope=camp_name,
                            metric="ctr",
                            actual_value=round(float(row["ctr"]), 2),
                            threshold=min_ctr,
                            message=(
                                f"Campaign '{camp_name}' CTR is {row['ctr']:.2f}% "
                                f"(target >= {min_ctr:.2f}%)."
                            ),
                        )
                    )

    return alerts
