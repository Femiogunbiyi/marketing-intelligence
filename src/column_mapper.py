import pandas as pd


# Maps standardized (snake_case) platform export headers -> internal pipeline names
COMMON_COLUMN_ALIASES: dict[str, str] = {
    # Spend aliases (Meta, Google, TikTok)
    "amount_spent": "spend",
    "amount_spent_ngn": "spend",
    "amount_spent_usd": "spend",
    "cost": "spend",
    "ad_spend": "spend",
    "total_spend": "spend",
    # Revenue aliases (Meta, Shopify, Paystack, Flutterwave)
    "purchase_conversion_value": "revenue",
    "purchases_conversion_value": "revenue",
    "conversion_value": "revenue",
    "total_sales": "revenue",
    "gross_sales": "revenue",
    "amount": "revenue",
    "amount_paid": "revenue",
    "settled_amount": "revenue",
    # Conversions aliases
    "purchases": "conversions",
    "results": "conversions",
    "orders": "conversions",
    "total_orders": "conversions",
    "transactions": "conversions",
    # Clicks & Impressions aliases
    "link_clicks": "clicks",
    "outbound_clicks": "clicks",
    "impr": "impressions",
    # Campaign & Channel aliases
    "campaign_name": "campaign",
    "ad_set_name": "campaign",
    "platform": "channel",
    "source": "channel",
    "traffic_source": "channel",
    # Customer ID aliases (Paystack, Shopify, CRM)
    "customer_email": "customer_id",
    "email": "customer_id",
    "user_id": "customer_id",
    "client_id": "customer_id",
    "phone_number": "customer_id",
    # Date aliases
    "reporting_starts": "date",
    "day": "date",
    "created_at": "date",
    "transaction_date": "date",
    "paid_at": "date",
}


def map_platform_columns(
    df: pd.DataFrame,
    custom_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rename common ad-platform and payment-gateway columns to internal standard names.

    Only renames a column if the target standard name does not already exist in `df`.
    """
    mapped = df.copy()
    aliases = {**COMMON_COLUMN_ALIASES, **(custom_mapping or {})}

    rename_dict: dict[str, str] = {}
    existing_cols = set(mapped.columns)

    for col in mapped.columns:
        target = aliases.get(col)
        if target and target not in existing_cols and target not in rename_dict.values():
            rename_dict[col] = target

    if rename_dict:
        mapped = mapped.rename(columns=rename_dict)

    return mapped
