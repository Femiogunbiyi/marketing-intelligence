from src.data_loader import load_marketing_data
from src.cleaning import clean_marketing_data
from src.metrics import add_core_marketing_metrics, summarize_overall_kpis
from src.campaign_analysis import analyze_campaign_performance, get_top_and_bottom_campaigns
from src.customer_analysis import compute_rfm_and_ltv, summarize_customer_retention
from src.segmentation import segment_customers_kmeans, summarize_segments

if __name__=="__main__":
    # 1. Load raw data
    raw_df = load_marketing_data("data/raw/campaigns.csv")

    # 2. Clean and save processed copy
    clean_df = clean_marketing_data(
        raw_df,
        save_path="data/processed/cleaned_marketing_data.csv",
    )

    # 3. Add row-level KPIs & print overall summary
    enriched_df = add_core_marketing_metrics(clean_df)
    overall_kpis = summarize_overall_kpis(enriched_df)
    print("--- Overall Marketing KPIs ---")
    for kpi, value in overall_kpis.items():
        print(f"{kpi}: {value}")

    # 4. Campaign-level breakdown (change 'campaign' to 'channel' if needed)
    if "campaign" in clean_df.columns:
        campaign_df = analyze_campaign_performance(clean_df, group_col="campaign", sort_by="spend")
        print("\n--- Campaign Performance Summary ---")
        print(campaign_df.head())

        highlights = get_top_and_bottom_campaigns(campaign_df, metric="roas", n=3)
        print("\n--- Top 3 Campaigns by ROAS ---")
        print(highlights["top"])

    # 5. Customer analysis  RFM (Recency, Frequency, Monetary) and LTV (Lifetime Value)
    if "customer_id" in clean_df.columns and "revenue" in clean_df.columns:
        customer_df = compute_rfm_and_ltv(clean_df)
        retention_kpis = summarize_customer_retention(customer_df)
        print("--- Customer Retention & LTV Summary ---")
        for kpi, value in retention_kpis.items():
            print(f"{kpi}: {value}")

        # Run ML segmentation right here where customer_df is guaranteed to exist
        if len(customer_df) >= 4:
            segmented_df = segment_customers_kmeans(customer_df, n_clusters=4)
            segment_summary = summarize_segments(segmented_df)
            print("\n--- ML Customer Segments ---")
            print(segment_summary)


        
        


    