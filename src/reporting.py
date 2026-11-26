from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from src.alerts import MarketingAlert
from typing import Any


def export_marketing_report(
    overall_kpis: dict[str, float],
    alerts: list[MarketingAlert],
    campaign_df: pd.DataFrame | None = None,
    segment_summary: pd.DataFrame | None = None,
    output_dir: str | Path = "reports",
) -> dict[str, Path]:
    """Export structured CSV tables, JSON metrics, and a Markdown executive summary."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files: dict[str, Path] = {}

    # 1. Save JSON snapshot of overall KPIs & alerts
    json_path = report_dir / f"kpi_summary_{timestamp}.json"
    payload: dict[str, Any] = {
        "generated_at": timestamp,
        "overall_kpis": overall_kpis,
        "alerts": [asdict(a) for a in alerts],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    saved_files["json_summary"] = json_path

    # 2. Save CSV tables if available
    if campaign_df is not None and not campaign_df.empty:
        camp_path = report_dir / f"campaign_performance_{timestamp}.csv"
        campaign_df.to_csv(camp_path, index=False)
        saved_files["campaign_csv"] = camp_path

    if segment_summary is not None and not segment_summary.empty:
        seg_path = report_dir / f"customer_segments_{timestamp}.csv"
        segment_summary.to_csv(seg_path, index=False)
        saved_files["segments_csv"] = seg_path

    # 3. Build human-readable Markdown Executive Report
    md_path = report_dir / f"executive_report_{timestamp}.md"
    lines = [
        f"# Marketing Intelligence Executive Report ({timestamp})",
        "",
        "## 1. Overall KPIs",
    ]
    for kpi, val in overall_kpis.items():
        lines.append(f"- **{kpi}**: {val}")

    lines.extend(["", "## 2. Active Performance Alerts"])
    if not alerts:
        lines.append("- No alerts triggered. All metrics within target thresholds.")
    else:
        for a in alerts:
            lines.append(f"- **[{a.severity}] {a.scope}**: {a.message}")

    if campaign_df is not None and not campaign_df.empty:
        lines.extend(["", "## 3. Top Campaigns", campaign_df.head(5).to_markdown(index=False)])

    if segment_summary is not None and not segment_summary.empty:
        lines.extend(["", "## 4. Customer Segments", segment_summary.to_markdown(index=False)])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    saved_files["markdown_report"] = md_path

    return saved_files
