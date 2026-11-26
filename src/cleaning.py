from pathlib import Path
import pandas as pd
from src.column_mapper import map_platform_columns

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to clean snake_case (e.g., 'Total Spend ($)' -> 'total_spend')."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
    )
    return cleaned


def clean_currency_and_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip currency symbols/commas from financial or metric columns and convert to numeric."""
    cleaned = df.copy()
    numeric_keywords = ("spend", "cost", "revenue", "sales", "budget", "price", "cpc", "cpa", "roas", "clicks", "impressions", "conversions")

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        if any(keyword in col for keyword in numeric_keywords):
            # Remove currency symbols ($, €, £, ₦), commas, and whitespace
            series = (
                cleaned[col]
                .astype(str)
                .str.replace(r"[$€£₦,\s]", "", regex=True)
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-": pd.NA})
            )
            converted = pd.to_numeric(series, errors="coerce")
            # Only overwrite if conversion succeeded for at least some non-null values
            if converted.notna().any():
                cleaned[col] = converted

    return cleaned

def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically detects and convert date/time columns into pandas datetime objects."""
    cleaned = df.copy()
    date_keywords = ("date", "time", "created", "updated", "timestamp", "day", "month")

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        if any(keyword in col for keyword in date_keywords):
            converted = pd.to_datetime(cleaned[col], errors="coerce")
            if converted.notna().any():
                cleaned[col] = converted

    return cleaned

def clean_text_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize text columns (channel, campaign, segment) by trimming whitespace and casing."""
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = (
            cleaned[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    return cleaned

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric metrics with 0 and missing categorical labels with 'Unknown'."""
    cleaned = df.copy()

    # Fill numeric columns with 0 (standard for missing spend, clicks, or conversions)
    num_cols = cleaned.select_dtypes(include=["number"]).columns
    cleaned[num_cols] = cleaned[num_cols].fillna(0)

    # Fill text/categorical columns with 'Unknown'
    cat_cols = cleaned.select_dtypes(include=["object", "string", "category"]).columns
    cleaned[cat_cols] = cleaned[cat_cols].fillna("Unknown")

    return cleaned

def clean_marketing_data(
        df: pd.DataFrame,
        save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full marketing data cleaning pipeline.
    
    Args:
        df: Raw DataFrame loaded from `data_loader.py`.
        save_path: Optional path to export the cleaned CSV (e.g., 'data/processed/clean_marketing.csv').
    
    Returns:
        A cleaned, standardize DataFrame ready for analysis and modeling.
    """
    cleaned = (
        df.pipe(standardize_column_names)
        .pipe(map_platform_columns)
        .drop_duplicates()
        .pipe(parse_date_columns)
        .pipe(clean_currency_and_numeric_columns)
        .pipe(clean_text_categories)
        .pipe(handle_missing_values)
        .reset_index(drop=True)
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_path, index=False)

    return cleaned