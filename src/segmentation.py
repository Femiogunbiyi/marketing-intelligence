from pathlib import Path
import joblib # type: ignore
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["recency_days", "frequency", "monetary_ltv"]

def segment_customers_kmeans(
    customer_df : pd.DataFrame,
    n_clusters: int = 4,
    features: list[str] | None = None,
    model_save_path: str | Path | None =  "models/customer_segmentation_kmeans.joblib",
    random_state: int = 42,
) -> pd.DataFrame:
    """Segment customers into behavioural clusters using StandardScaler + KMeans.
    
    Args:
        customer_df: Customer-level DataFrame produced by `customer_analysis.py`.
        n_clusters: Number of customer segments to created (default: 4).
        features: Numeric columns for clustering (defaults to RFM metrics).
        model_save_path: Optional file path to save the trained pipeline artifact.
        random_state: Seed for reproducible clustering results.
        
    Returns:
        A copy of `customer_df` with `cluster_id` and readable `segment_label` columns.
    """
    features_cols = features or DEFAULT_FEATURES
    missing = set(features_cols) - set(customer_df.columns)
    if missing:
        raise ValueError(f"Missing required features for segmentation: {missing}")

    if len(customer_df) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} customers to form {n_clusters} clusters, "
            f"got {len(customer_df)}."
        )

    segmented = customer_df.copy()

    # Log-transform skewed monetary/frequency features before scaling so outliers don't distort clusters
    X = segmented[features_cols].copy()
    for col in X.columns:
        X[col] = np.log1p(X[col].clip(lower=0))

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
        ]
    )

    segmented["cluster_id"] = pipeline.fit_predict(X.to_numpy()) # type: ignore
    segmented["segment_label"] = _assign_segment_names(segmented)

    if model_save_path is not None:
        artifact_path = Path(model_save_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifact_path) # type: ignore

    return segmented

def _assign_segment_names(segmented_df: pd.DataFrame) -> pd.Series:
    """Map raw numeric cluster IDs (0, 1, 2...) to business-friendly marketing labels
    by ranking each cluster's average monetary value and recency.
    """
    cluster_stats = (
        segmented_df.groupby("cluster_id")
        .agg(
            avg_monetary=("monetary_ltv", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_recency=("recency_days", "mean"),
        )
        .sort_values(by=["avg_monetary", "avg_frequency"], ascending=False)
    )

    ranked_ids = list(cluster_stats.index)
    label_templates = [
        "Champions (High Value)",
        "Loyal Customers",
        "Potential / Recent",
        "At-Risk / Dormat",
    ]

    mapping: dict[int, str] ={}
    for rank, cluster_id in enumerate(ranked_ids):
        if rank < len(label_templates):
            mapping[int(cluster_id)] = label_templates[rank]
        else:
            mapping[int(cluster_id)] = f"Segment {rank + 1}"

    return segmented_df["cluster_id"].map(mapping)

def summarize_segments(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary profile table showing size and average RFM metrics per segment."""
    if "segment_label" not in segmented_df.columns:
        raise ValueError("DataFrame must contain 'segment_label'. Run `segment_customers_kmeans` first.")

    summary = (
        segmented_df.groupby("segment_label", as_index=False)
        .agg(
            customer_count=("segment_label", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_ltv=("monetary_ltv", "mean"),
            total_segment_revenue=("monetary_ltv", "sum"),
        )
        .sort_values(by="total_segment_revenue", ascending=False)
        .reset_index(drop=True)
    )

    total_customers = summary["customer_count"].sum()
    summary["customer_share_pct"] = ((summary["customer_count"] / total_customers) * 100).round(2)
    return summary.round(2)