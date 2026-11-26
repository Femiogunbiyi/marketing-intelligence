import numpy as np
import pandas as pd

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two pandas Series, returning 0.0 where denominator is 0 or null."""
    valid = (denominator != 0) & denominator.notna()
    values = np.where(valid, numerator / denominator.where(valid), 0.0)
    return pd.Series(values, index=numerator.index, dtype="float64")

def add_core_marketing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard marketing KPIs based on available columns in the DataFrame."""
    enriched = df.copy()
    cols = set(enriched.columns)

    if {"clicks", "impressions"}.issubset(cols):
        enriched["ctr"] = _safe_divide(enriched["clicks"], enriched["impressions"]) * 100

    if {"spend", "clicks"}.issubset(cols):
        enriched["cpc"] = _safe_divide(enriched["spend"], enriched["clicks"])

    if {"spend", "impressions"}.issubset(cols):
        enriched["cpm"] = _safe_divide(enriched["spend"], enriched["impressions"]) * 1000

    if {"conversions", "clicks"}.issubset(cols):
        enriched["conversion_rate"] = _safe_divide(enriched["conversions"], enriched["clicks"]) * 100

    if {"spend", "conversions"}.issubset(cols):
        enriched["cpa"] = _safe_divide(enriched["spend"], enriched["conversions"])

    if {"revenue", "spend"}.issubset(cols):
        enriched["roas"] = _safe_divide(enriched["revenue"], enriched["spend"])
        enriched["roi"] = _safe_divide(enriched["revenue"] - enriched["spend"], enriched["spend"]) * 100
        enriched["net_profit"] = enriched["revenue"] - enriched["spend"]

    if {"revenue", "conversions"}.issubset(cols):
        enriched["aov"] = _safe_divide(enriched["revenue"], enriched["conversions"])

    return enriched
    
def summarize_overall_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Aggregate total and weighted KPIs across the entire dataset."""
    summary: dict[str, float] = {}
    cols = set(df.columns)

    totals = {
        col: float(df[col].sum())
        for col in ("impressions", "clicks", "conversions", "spend", "revenue")
        if col in cols
    }
    summary.update({f"total_{k}": round(v, 2) for k, v in totals.items()})

    if "impressions" in totals and totals["impressions"] > 0 and "clicks" in totals:
        summary["overall_ctr"] = round((totals["clicks"] / totals["impressions"]) * 100, 2)

    if "clicks" in totals and totals["clicks"] > 0 and "spend" in totals:
        summary["overall_cpc"] = round((totals["spend"] / totals["clicks"]) , 2)

    if "clicks" in totals and totals["clicks"] > 0 and "conversions" in totals:
        summary["overall_conversion_rate"] = round((totals["conversions"] / totals["clicks"]) * 100, 2)

    if "conversions" in totals and totals["conversions"] > 0 and "spend" in totals:
        summary["overall_cpa"] = round((totals["spend"] / totals["conversions"]) , 2)

    if "spend" in totals and totals["spend"] > 0 and "revenue" in totals:
        summary["overall_roas"] = round((totals["revenue"] / totals["spend"]) , 2)
        summary["overall_net_profit"] = round((totals["revenue"] - totals["spend"]) , 2)

    return summary
    

        