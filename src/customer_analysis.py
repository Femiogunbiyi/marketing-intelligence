import pandas as pd

def compute_rfm_and_ltv(
    df: pd.DataFrame, 
    customer_col: str = "customer_id",
    date_col: str = "date",
    revenue_col: str = "revenue",
) -> pd.DataFrame:
    """ Calculate Recency, Frequency, Monetary (RFM) metrics and Historical LTV per customer.
    
    Args:
        df: Cleaned marketing or transaction DataFrame.
        customer_col: Column uniquely identifying each customer.
        date_col: Transaction or interaction date column.
        revenue_col: Revenue or purchase value column.

    Returns:
        Customer-level DataFrame containing recency_days, frequency, monetary_ltv,
        avg_order_value, and 1-5 RFM scores.
    """
    required = {customer_col, revenue_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns for customer analysis: {required - set(df.columns)}")

    data = df.copy()
    has_dates = date_col in data.columns

    if has_dates:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        snapshot_date = data[date_col].max() + pd.Timedelta(days=1)

        customer_df = (
            data.groupby(customer_col, as_index=False)
            .agg(
                last_purchase_date=(date_col, "max"),
                frequency=(revenue_col, "count"),
                monetary_ltv=(revenue_col, "sum"),
            )
        )
        customer_df["recency_days"] = (snapshot_date - customer_df["last_purchase_date"]).dt.days
    else:
        customer_df = (
            data.groupby(customer_col, as_index=False)
            .agg(
                frequency=(revenue_col, "count"),
                monetary_ltv=(revenue_col, "sum"),
            )
        )
        customer_df["recency_days"] = 0

    customer_df["avg_order_value"] = customer_df["monetary_ltv"] / customer_df["frequency"].replace(0, 1)

    # Score Frequency and Monetary from 1 to 5 (5 = highest) using rank percentiles
    customer_df["f_score"] = pd.qcut(
        customer_df["frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    customer_df["m_score"] = pd.qcut(
        customer_df["monetary_ltv"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # Score Recency from 1 to 5 (5 = most recent / lowest recency_days)
    if has_dates:
        customer_df["r_score"] = pd.qcut(
            customer_df["recency_days"].rank(method="first"),
            q=5,
            labels=[5, 4, 3, 2, 1],
        ).astype(int)
    else:
        customer_df["r_score"] = 3

    customer_df["rfm_total_score"] = (
        customer_df["r_score"] + customer_df["f_score"] + customer_df["m_score"]
    )

    return customer_df.sort_values(by="monetary_ltv", ascending=False).reset_index(drop=True)

def summarize_customer_retention(customer_df: pd.DataFrame) -> dict[str, float]:
    """Compute high-level customer retention, repeat purchase rate, and average LTV"""
    total_customers = len(customer_df)
    if total_customers == 0:
        return {
            "total_customers": 0.0,
            "repeat_customers": 0.0,
            "repeat_purchase_rate": 0.0,
            "average_ltv": 0.0,
            "average_purchase_frequency": 0.0,
        }

    repeat_customers = int((customer_df["frequency"] > 1).sum())
    repeat_rate = (repeat_customers / total_customers) * 100

    return {
        "total_customers": float(total_customers),
        "repeat_customers": float(repeat_customers),
        "repeat_purchase_rate": round(repeat_rate, 2),
        "average_ltv": round(float(customer_df["monetary_ltv"].mean()), 2),
        "average_purchase_frequency": round(float(customer_df["frequency"].mean()), 2),
    }